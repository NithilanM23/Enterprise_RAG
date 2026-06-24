"""
services/embedding_service.py
------------------------------
Embedding generation for the Local Employee Knowledge Assistant.

Responsibilities:
  - Generate embeddings for a single text string (used for query embedding)
  - Generate embeddings for a list of texts (used for document chunks)
  - Return raw float vectors ready for pgvector storage

Uses Ollama running locally — no internet, no API keys.

Swapping the embedding model:
  The model name is read live from services.settings_service on every
  call — never a module-level constant. To swap models safely, use
  services.settings_service.update_embedding_model(), NOT a direct
  setting write. Changing the embedding model makes every existing
  embedding meaningless (different vector space) — that function
  handles nulling old embeddings and migrating the pgvector column
  dimension if needed. See that module's docstring for details.

  Supported Ollama embedding models:
    mxbai-embed-large   -> dimension 1024  (default)
    nomic-embed-text    -> dimension 768
    bge-small           -> dimension 384
    all-MiniLM-L6-v2    -> dimension 384

IMPORTANT:
  The same model must always be used for both document and query embeddings.
  Mixing models produces meaningless similarity scores.
"""

import logging
import time

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import OLLAMA_BASE_URL   # endpoint is structural, not swappable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ollama client initialisation
# ---------------------------------------------------------------------------

def _get_ollama_client(model: str = None):
    """
    Return a LangChain OllamaEmbeddings instance.
    Reads the live 'embedding_model' setting if no model is explicitly
    passed. Instantiated fresh per call — lightweight, stateless.
    """
    from langchain_ollama import OllamaEmbeddings

    if model is None:
        from services.settings_service import get_setting
        model = get_setting("embedding_model")

    return OllamaEmbeddings(
        model=model,
        base_url=OLLAMA_BASE_URL,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def embed_text(text: str, model: str = None) -> list:
    """
    Generate an embedding vector for a single text string.

    Used for:
      - Query embedding at retrieval time

    Args:
        text: The text to embed. Should be non-empty.
        model: Optional model override. Defaults to the current active setting.

    Returns:
        A list of floats representing the embedding vector.

    Raises:
        ValueError : If text is empty.
        RuntimeError: If Ollama is unreachable or the model is not loaded.
    """
    if not text or not text.strip():
        raise ValueError("Cannot embed empty text.")

    if model is None:
        from services.settings_service import get_setting
        model = get_setting("embedding_model")

    logger.debug("Embedding single text (%d characters) with model '%s'.", len(text), model)

    try:
        client = _get_ollama_client(model)
        vector = client.embed_query(text)
        logger.debug("Embedding generated: dimension=%d.", len(vector))
        return vector
    except Exception as exc:
        raise RuntimeError(
            f"Embedding generation failed.\n"
            f"  Model      : {model}\n"
            f"  Ollama URL : {OLLAMA_BASE_URL}\n"
            f"  Error      : {exc}\n\n"
            f"  Is Ollama running?  →  ollama serve\n"
            f"  Is model pulled?    →  ollama pull {model}"
        ) from exc


def embed_chunks(chunks: list, batch_size: int = 10) -> list:
    """
    Generate embeddings for a list of chunk dicts (in-place update).

    Processes chunks in batches to avoid overwhelming Ollama on CPU.
    Logs progress so the user can see it working.

    Args:
        chunks     : List of dicts with at least a 'chunk_text' key.
                     Each dict is updated in-place with its 'embedding'.
        batch_size : Number of chunks to send to Ollama per request.
                     Lower = less memory pressure. Default 10 is safe for CPU.

    Returns:
        The same list of chunk dicts, each now with 'embedding' populated
        as a list of floats.

    Raises:
        RuntimeError: If Ollama is unreachable or embedding fails.
    """
    if not chunks:
        logger.warning("embed_chunks called with empty list.")
        return chunks

    from services.settings_service import get_setting
    model = get_setting("embedding_model")
    dimension = get_setting("embedding_dimension")

    total = len(chunks)
    logger.info("Generating embeddings for %d chunks using model '%s'.", total, model)

    client = _get_ollama_client(model)
    embedded_count = 0

    for batch_start in range(0, total, batch_size):
        batch = chunks[batch_start : batch_start + batch_size]
        texts = [c["chunk_text"] for c in batch]

        try:
            t0 = time.time()
            vectors = client.embed_documents(texts)
            elapsed = time.time() - t0

            for chunk, vector in zip(batch, vectors):
                chunk["embedding"] = vector
                chunk["embedding_model"] = model
                chunk["embedding_dimension"] = dimension

            embedded_count += len(batch)
            logger.info(
                "Embedded %d/%d chunks (batch took %.1fs).",
                embedded_count, total, elapsed,
            )

        except Exception as exc:
            raise RuntimeError(
                f"Embedding failed at chunk {batch_start}–{batch_start + len(batch)}.\n"
                f"  Model      : {model}\n"
                f"  Ollama URL : {OLLAMA_BASE_URL}\n"
                f"  Error      : {exc}\n\n"
                f"  Is Ollama running?  →  ollama serve\n"
                f"  Is model pulled?    →  ollama pull {model}"
            ) from exc

    logger.info("All %d embeddings generated successfully.", total)
    return chunks


def check_ollama_connection() -> dict:
    """
    Verify Ollama is reachable and the configured embedding model is available.

    Returns a dict with:
        reachable    : bool
        model_ready  : bool
        model_name   : str
        error        : str or None
    """
    import requests
    from services.settings_service import get_setting

    model = get_setting("embedding_model")

    result = {
        "reachable":   False,
        "model_ready": False,
        "model_name":  model,
        "error":       None,
    }

    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        resp.raise_for_status()
        result["reachable"] = True

        models = [m["name"] for m in resp.json().get("models", [])]
        result["model_ready"] = any(model in m for m in models)

        if not result["model_ready"]:
            result["error"] = (
                f"Model '{model}' not found in Ollama.\n"
                f"Run:  ollama pull {model}"
            )

    except requests.exceptions.ConnectionError:
        result["error"] = (
            f"Cannot reach Ollama at {OLLAMA_BASE_URL}.\n"
            "Run:  ollama serve"
        )
    except Exception as exc:
        result["error"] = str(exc)

    return result