"""
services/llm_service.py
------------------------
LLM answer generation for the Local Employee Knowledge Assistant.

Responsibilities:
  - Build a grounded prompt from retrieved chunks + user question
  - Call Ollama LLM locally
  - Return the generated answer as a string

Swapping the LLM model:
  Change LLM_MODEL in config.py — nothing here needs to change.

  Supported local Ollama models:
    qwen2.5:3b    (default — good quality, ~2GB RAM)
    phi3:mini     (faster, slightly lower quality)
    llama3.2:3b   (alternative)

Design:
  - One public function: generate_answer(question, chunks) -> str
  - Prompt template is defined in config.py — not hardcoded here
  - No streaming for POC — returns complete response
  - No chat history — stateless single-turn Q&A
"""

import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import LLM_MODEL, OLLAMA_BASE_URL, PROMPT_TEMPLATE

logger = logging.getLogger(__name__)


def _call_ollama(prompt: str) -> str:
    """
    Call the Ollama REST API directly using requests.

    Model name, temperature, and num_predict are read live from
    services.settings_service on every call — this is what lets the
    admin settings panel swap the LLM model or generation parameters
    without restarting the server. config.LLM_MODEL is only the
    fallback default the very first time the app runs.

    Why not LangChain OllamaLLM:
      LangChain does not forward the `think` parameter to Ollama, so Qwen3
      thinking models return 0-char responses (they emit only thinking tokens
      and no answer text). Calling the API directly lets us pass think=false
      explicitly, which tells Ollama to suppress the thinking phase entirely.

    The think=false option is:
      - Required for Qwen3 models (qwen3.5:0.8b, qwen3:*, etc.)
      - Silently ignored by non-thinking models (qwen2.5, phi3, llama3, etc.)

    Raises:
        RuntimeError: If Ollama is unreachable or returns a non-200 status.
    """
    import requests
    import json
    from services.settings_service import get_setting

    model       = get_setting("llm_model")
    temperature = get_setting("temperature")
    num_predict = get_setting("num_predict")

    payload = {
        "model":  model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
        },
        "think": False,   # suppresses <think> blocks for Qwen3 models
    }

    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=120,   # CPU inference can be slow
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "")

    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            f"Cannot reach Ollama at {OLLAMA_BASE_URL}.\n"
            "Run:  ollama serve"
        )
    except requests.exceptions.Timeout:
        raise RuntimeError(
            "Ollama request timed out (>120s). "
            "The model may be too large for available RAM."
        )
    except Exception as exc:
        raise RuntimeError(
            f"Ollama API error: {exc}\n"
            f"Model: {model}  |  URL: {OLLAMA_BASE_URL}"
        )


def _strip_thinking(text: str) -> str:
    """
    Remove <think>...</think> blocks from model output.

    Qwen3 thinking models emit internal chain-of-thought wrapped in these tags.
    We strip them so only the final answer is returned to the user.
    Works even if think=False fails to suppress them (safety net).
    """
    import re
    # Remove all <think>...</think> blocks including multiline
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned.strip()


def build_prompt(question: str, chunks: list, history: list = None, use_rag: bool = True) -> str:
    """
    Construct the grounded prompt from retrieved chunks, optional conversation
    history, and the user question.

    Args:
        question : The user's natural language question.
        chunks   : List of chunk dicts from retrieval_service.retrieve().
        history  : Optional list of previous messages for buffer memory.
                   Format: [{"role": "user"/"assistant", "content": str}, ...]
        use_rag  : Boolean indicating whether to use context/RAG format.

    Returns:
        Formatted prompt string ready to send to the LLM.
    """
    if not use_rag:
        # Build standard LLM prompt without RAG context
        history_block = ""
        if history:
            lines = []
            for msg in history:
                role_label = "User" if msg["role"] == "user" else "Assistant"
                lines.append(f"{role_label}: {msg['content']}")
            history_block = (
                "\nConversation so far:\n"
                + "\n".join(lines)
                + "\n"
            )

        prompt = (
            "You are a helpful and intelligent AI assistant.\n\n"
            + history_block
            + "\nQuestion:\n"
            + question
            + "\n\nAnswer:"
        )
        return prompt

    if not chunks:
        context = "No relevant context found in the uploaded documents."
    else:
        # Sort by reranker score descending — most relevant chunk first.
        # Small models suffer from "lost in the middle": they attend well to
        # the first and last chunks but poorly to the middle. Putting the
        # best chunk first maximises the chance of a correct answer.
        sorted_chunks = sorted(
            chunks,
            key=lambda c: c.get("reranker_score", c.get("similarity", 0)),
            reverse=True,
        )
        context_parts = []
        for i, chunk in enumerate(sorted_chunks, start=1):
            context_parts.append(
                f"[Source {i}: {chunk['filename']}, chunk {chunk['chunk_number']}]\n"
                f"{chunk['chunk_text']}"
            )
        context = "\n\n---\n\n".join(context_parts)

    # Build conversation history block
    history_block = ""
    if history:
        lines = []
        for msg in history:
            role_label = "User" if msg["role"] == "user" else "Assistant"
            lines.append(f"{role_label}: {msg['content']}")
        history_block = (
            "\nConversation so far:\n"
            + "\n".join(lines)
            + "\n"
        )

    # Build prompt using config template, injecting history before context
    from config import PROMPT_TEMPLATE

    # Inject history block into the prompt before context
    history_prefix = ""
    if history_block:
        history_prefix = history_block + "\n"

    prompt = (
        "You are a helpful assistant that answers questions using the provided context.\n\n"
        "Guidelines:\n"
        "- Use the context to answer even if information appears as a list or fragments.\n"
        "- Synthesise and present the answer clearly in your own words.\n"
        "- If context has partial information, use what is available.\n"
        "- Only say you cannot find the answer if context has NO relevant information.\n"
        + history_prefix
        + "\nContext:\n"
        + context
        + "\n\nQuestion:\n"
        + question
        + "\n\nAnswer:"
    )

    return prompt


def generate_answer(question: str, chunks: list, history: list = None) -> str:
    """
    Generate a grounded answer to the user's question from retrieved chunks.

    Args:
        question : The user's natural language question.
        chunks   : Retrieved chunks from retrieval_service.retrieve().
        history  : Optional conversation history for buffer memory context.
                   Format: [{"role": "user"/"assistant", "content": str}, ...]

    Returns:
        The LLM answer as a plain string.

    Raises:
        RuntimeError: If Ollama is unreachable or the model is not loaded.
    """
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    prompt = build_prompt(question, chunks, history=history)

    from services.settings_service import get_setting
    current_model = get_setting("llm_model")

    logger.info(
        "Generating answer with model '%s' — prompt length: %d chars.",
        current_model, len(prompt),
    )

    try:
        raw = _call_ollama(prompt)
        answer = _strip_thinking(raw)

        if not answer:
            logger.warning(
                "LLM returned empty answer after stripping thinking tags. "
                "Raw response length: %d chars.", len(raw)
            )
            answer = "I could not generate an answer. Please try rephrasing your question."

        logger.info("Answer generated successfully (%d chars).", len(answer))
        return answer

    except Exception as exc:
        raise RuntimeError(
            f"LLM generation failed.\n"
            f"  Model      : {current_model}\n"
            f"  Ollama URL : {OLLAMA_BASE_URL}\n"
            f"  Error      : {exc}\n\n"
            f"  Is Ollama running?  →  ollama serve\n"
            f"  Is model pulled?    →  ollama pull {current_model}"
        ) from exc


def generate_followups(question: str, answer: str, max_suggestions: int = 3) -> list:
    """
    Generate short follow-up questions based on the last Q&A pair.
    Powers the "Smart Follow-up Suggestions" engagement feature —
    shown as clickable chips below every assistant answer.

    Uses a small, fast prompt (not the main retrieval prompt) so this
    adds minimal latency. Failures are non-fatal — an empty list means
    the frontend simply doesn't render the suggestion chips.

    Args:
        question : The user's original question.
        answer   : The assistant's answer that was just generated.
        max_suggestions : How many follow-ups to request.

    Returns:
        List of short question strings, e.g.
        ["What floor is it on?", "What is the contact number?", ...]
    """
    prompt = f"""Based on this question and answer, suggest {max_suggestions} short, \
natural follow-up questions a professional might ask next. Keep each under 12 words.

Question: {question}
Answer: {answer[:500]}

Return ONLY a JSON array of strings, nothing else. Example:
["What floor is it on?", "Is there parking available?", "What are the office hours?"]
"""

    try:
        raw = _call_ollama(prompt).strip()
        raw = _strip_thinking(raw)

        # Extract JSON array even if the model added stray text around it
        import re, json
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if not match:
            return []

        suggestions = json.loads(match.group(0))
        if not isinstance(suggestions, list):
            return []

        # Clean and cap
        cleaned = [str(s).strip().strip('"') for s in suggestions if str(s).strip()]
        return cleaned[:max_suggestions]

    except Exception as exc:
        logger.warning("Follow-up suggestion generation failed (non-fatal): %s", exc)
        return []


def check_llm_connection() -> dict:
    """
    Verify Ollama is reachable and the configured LLM model is available.

    Returns a dict with:
        reachable   : bool
        model_ready : bool
        model_name  : str
        error       : str or None
    """
    import requests
    from services.settings_service import get_setting

    current_model = get_setting("llm_model")

    result = {
        "reachable":   False,
        "model_ready": False,
        "model_name":  current_model,
        "error":       None,
    }

    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        resp.raise_for_status()
        result["reachable"] = True

        models = [m["name"] for m in resp.json().get("models", [])]
        result["model_ready"] = any(current_model in m for m in models)

        if not result["model_ready"]:
            result["error"] = (
                f"Model '{current_model}' not found in Ollama.\n"
                f"Run:  ollama pull {current_model}"
            )

    except requests.exceptions.ConnectionError:
        result["error"] = (
            f"Cannot reach Ollama at {OLLAMA_BASE_URL}.\n"
            "Run:  ollama serve"
        )
    except Exception as exc:
        result["error"] = str(exc)

    return result