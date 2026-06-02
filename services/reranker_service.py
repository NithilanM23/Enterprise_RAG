"""
services/reranker_service.py
-----------------------------
Cross-encoder reranking for the Local Employee Knowledge Assistant.

After hybrid search produces a pool of candidate chunks (from both semantic
and BM25 search), the reranker re-scores each (query, chunk) pair using a
cross-encoder model that reads both together — producing much more accurate
relevance scores than the vector similarity or BM25 score alone.

Why reranking matters:
  - Semantic similarity tells you "this chunk is about a similar topic"
  - BM25 tells you "this chunk contains these keywords"
  - The reranker tells you "this chunk actually answers this specific question"

Model used:
  cross-encoder/ms-marco-MiniLM-L-6-v2
    - ~80MB download (one time, cached by HuggingFace)
    - Runs entirely on CPU
    - Fast: ~50-200ms per (query, chunk) pair on i5 12th Gen
    - Trained on MS MARCO passage ranking — excellent for Q&A tasks

Swapping the reranker model:
  Change RERANKER_MODEL below. Any HuggingFace cross-encoder works.
  Other good options:
    cross-encoder/ms-marco-MiniLM-L-12-v2   (better quality, ~2x slower)
    cross-encoder/ms-marco-TinyBERT-L-2     (faster, slightly lower quality)

Dependencies:
  sentence-transformers  (pip install sentence-transformers)
"""

import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

# Change this to swap the reranker model — nothing else needs to change
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


# ---------------------------------------------------------------------------
# Model loading — cached after first load
# ---------------------------------------------------------------------------

_reranker = None


def _get_reranker():
    """
    Load and cache the cross-encoder model.
    Model is downloaded once and cached by HuggingFace in ~/.cache/huggingface.
    Subsequent loads are instant (from disk cache).
    """
    global _reranker

    if _reranker is None:
        from sentence_transformers import CrossEncoder

        logger.info(
            "Loading reranker model '%s' (first load may download ~80MB)...",
            RERANKER_MODEL,
        )
        _reranker = CrossEncoder(
            RERANKER_MODEL,
            max_length=512,    # truncate long chunks to fit model context
        )
        logger.info("Reranker model loaded.")

    return _reranker


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rerank(query: str, chunks: list, top_k: int = 5) -> list:
    """
    Re-score and reorder candidate chunks using a cross-encoder model.

    The cross-encoder reads (query, chunk_text) pairs together and produces
    a relevance score — more accurate than dot-product similarity because
    it can model the interaction between query and passage directly.

    Args:
        query  : The user's natural language question.
        chunks : List of candidate chunk dicts from hybrid search.
                 Each must have at least 'chunk_text' and 'id' keys.
        top_k  : Number of top chunks to return after reranking.

    Returns:
        The top_k most relevant chunks, sorted by reranker score descending.
        Each dict gets a new 'reranker_score' key added (float).

    Raises:
        RuntimeError: If the reranker model fails to load.
    """
    if not chunks:
        return []

    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    try:
        reranker = _get_reranker()
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load reranker model '{RERANKER_MODEL}': {exc}\n"
            "Run:  pip install sentence-transformers"
        ) from exc

    # Build (query, chunk_text) pairs for the cross-encoder
    pairs = [(query, chunk["chunk_text"]) for chunk in chunks]

    logger.info(
        "Reranking %d candidate chunks with model '%s'...",
        len(chunks), RERANKER_MODEL,
    )

    # Score all pairs — returns a numpy array of floats
    scores = reranker.predict(pairs)

    # Attach reranker score to each chunk
    for chunk, score in zip(chunks, scores):
        chunk["reranker_score"] = float(score)

    # Sort by reranker score descending and return top_k
    reranked = sorted(chunks, key=lambda x: x["reranker_score"], reverse=True)

    logger.info(
        "Reranking complete. Top score: %.4f  |  Returning top %d chunks.",
        reranked[0]["reranker_score"] if reranked else 0.0,
        min(top_k, len(reranked)),
    )

    return reranked[:top_k]


def is_model_cached() -> bool:
    """
    Check if the reranker model is already cached locally.
    Avoids triggering a download just to check availability.
    """
    from pathlib import Path

    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    model_slug = RERANKER_MODEL.replace("/", "--")

    return any(
        d.name.startswith(f"models--{model_slug}")
        for d in cache_dir.iterdir()
        if cache_dir.exists() and d.is_dir()
    )
