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
  Change EMBEDDING_MODEL and EMBEDDING_DIMENSION in config.py.
  That is the ONLY change needed — nothing in this file needs to be touched.

  Supported Ollama embedding models:
    mxbai-embed-large   -> dimension 1024  (default)
    nomic-embed-text    -> dimension 768
    bge-small           -> dimension 384
    all-MiniLM-L6-v2   -> dimension 384

IMPORTANT:
  The same model must always be used for both document and query embeddings.
  Mixing models produces meaningless similarity scores.
"""

import logging
import time

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import EMBEDDING_MODEL, OLLAMA_BASE_URL

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ollama client initialisation
# ---------------------------------------------------------------------------

def _get_ollama_client():
    """
    Return a LangChain OllamaEmbeddings instance configured from config.py.
    Instantiated fresh per call — lightweight, stateless.
    """
    from langchain_ollama import OllamaEmbeddings

    return OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def embed_text(text: str) -> list:
    """
    Generate an embedding vector for a single text string.

    Used for:
      - Query embedding at retrieval time

    Args:
        text: The text to embed. Should be non-empty.

    Returns:
        A list of floats representing the embedding vector.
        Length matches EMBEDDING_DIMENSION in config.py.

    Raises:
        ValueError : If text is empty.
        RuntimeError: If Ollama is unreachable or the model is not loaded.
    """
    if not text or not text.strip():
        raise ValueError("Cannot embed empty text.")

    logger.debug("Embedding single text (%d characters).", len(text))

    try:
        client = _get_ollama_client()
        vector = client.embed_query(text)
        logger.debug("Embedding generated: dimension=%d.", len(vector))
        return vector
    except Exception as exc:
        raise RuntimeError(
            f"Embedding generation failed.\n"
            f"  Model      : {EMBEDDING_MODEL}\n"
            f"  Ollama URL : {OLLAMA_BASE_URL}\n"
            f"  Error      : {exc}\n\n"
            f"  Is Ollama running?  →  ollama serve\n"
            f"  Is model pulled?    →  ollama pull {EMBEDDING_MODEL}"
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

    total = len(chunks)
    logger.info(
        "Generating embeddings for %d chunks using model '%s'.",
        total, EMBEDDING_MODEL,
    )

    client = _get_ollama_client()
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

            embedded_count += len(batch)
            logger.info(
                "Embedded %d/%d chunks (batch took %.1fs).",
                embedded_count, total, elapsed,
            )

        except Exception as exc:
            raise RuntimeError(
                f"Embedding failed at chunk {batch_start}–{batch_start + len(batch)}.\n"
                f"  Model      : {EMBEDDING_MODEL}\n"
                f"  Ollama URL : {OLLAMA_BASE_URL}\n"
                f"  Error      : {exc}\n\n"
                f"  Is Ollama running?  →  ollama serve\n"
                f"  Is model pulled?    →  ollama pull {EMBEDDING_MODEL}"
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

    result = {
        "reachable":   False,
        "model_ready": False,
        "model_name":  EMBEDDING_MODEL,
        "error":       None,
    }

    try:
        # Check if Ollama server is up
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        resp.raise_for_status()
        result["reachable"] = True

        # Check if our model is in the list
        models = [m["name"] for m in resp.json().get("models", [])]
        # Ollama model names may include tags like "mxbai-embed-large:latest"
        result["model_ready"] = any(
            EMBEDDING_MODEL in m for m in models
        )

        if not result["model_ready"]:
            result["error"] = (
                f"Model '{EMBEDDING_MODEL}' not found in Ollama.\n"
                f"Run:  ollama pull {EMBEDDING_MODEL}"
            )

    except requests.exceptions.ConnectionError:
        result["error"] = (
            f"Cannot reach Ollama at {OLLAMA_BASE_URL}.\n"
            "Run:  ollama serve"
        )
    except Exception as exc:
        result["error"] = str(exc)

    return result
