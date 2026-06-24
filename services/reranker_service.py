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

Default model:
  cross-encoder/ms-marco-MiniLM-L-6-v2
    - ~80MB download (one time, cached by HuggingFace)
    - Runs entirely on CPU
    - Fast: ~50-200ms per (query, chunk) pair on i5 12th Gen
    - Trained on MS MARCO passage ranking — excellent for Q&A tasks

Swapping the reranker model:
  Read live from services.settings_service ('reranker_model' key) — never
  a module-level constant. Use settings_service.update_reranker_model()
  to swap it; that function also calls reset_reranker_cache() here so the
  next rerank() call loads the new model instead of reusing a stale one.

  Other good HuggingFace cross-encoder options:
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


# ---------------------------------------------------------------------------
# Model loading — cached after first load, keyed by model name so a model
# swap via the admin panel correctly triggers a fresh load.
# ---------------------------------------------------------------------------

_reranker       = None
_reranker_model_loaded = None   # tracks which model name is currently cached


def _get_reranker(model: str = None):
    """
    Load and cache the cross-encoder model. If the requested model differs
    from whatever is currently cached (e.g. admin just swapped it), the old
    one is discarded and the new one loads fresh.

    Model is downloaded once and cached by HuggingFace in ~/.cache/huggingface.
    Subsequent loads of the same model are instant (from disk cache).
    """
    global _reranker, _reranker_model_loaded

    if model is None:
        from services.settings_service import get_setting
        model = get_setting("reranker_model")

    if _reranker is None or _reranker_model_loaded != model:
        from sentence_transformers import CrossEncoder

        logger.info(
            "Loading reranker model '%s' (first load may download ~80MB)...", model
        )
        _reranker = CrossEncoder(model, max_length=512, device="cpu")
        _reranker_model_loaded = model
        logger.info("Reranker model loaded: '%s'.", model)

    return _reranker


def reset_reranker_cache() -> None:
    """
    Discard the cached reranker model. Called by
    settings_service.update_reranker_model() right after the setting
    changes, so the NEXT rerank() call loads the newly selected model
    instead of silently continuing to use the old one.
    """
    global _reranker, _reranker_model_loaded
    _reranker = None
    _reranker_model_loaded = None
    logger.info("Reranker cache cleared — next call will load the current model setting.")


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

    from services.settings_service import get_setting
    model_name = get_setting("reranker_model")

    try:
        reranker = _get_reranker(model_name)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load reranker model '{model_name}': {exc}\n"
            "Run:  pip install sentence-transformers"
        ) from exc

    pairs = [(query, chunk["chunk_text"]) for chunk in chunks]

    logger.info("Reranking %d candidate chunks with model '%s'...", len(chunks), model_name)

    scores = reranker.predict(pairs)

    for chunk, score in zip(chunks, scores):
        chunk["reranker_score"] = float(score)

    reranked = sorted(chunks, key=lambda x: x["reranker_score"], reverse=True)

    logger.info(
        "Reranking complete. Top score: %.4f  |  Returning top %d chunks.",
        reranked[0]["reranker_score"] if reranked else 0.0,
        min(top_k, len(reranked)),
    )

    return reranked[:top_k]


def is_model_cached(model: str = None) -> bool:
    """
    Check if a reranker model is already cached locally on disk.
    Avoids triggering a download just to check availability. Defaults
    to checking the currently configured model if none is specified —
    useful for the admin UI to show "✓ already downloaded" vs
    "will download ~80MB" before confirming a model swap.
    """
    from pathlib import Path

    if model is None:
        from services.settings_service import get_setting
        model = get_setting("reranker_model")

    cache_dir  = Path.home() / ".cache" / "huggingface" / "hub"
    model_slug = model.replace("/", "--")

    return any(
        d.name.startswith(f"models--{model_slug}")
        for d in cache_dir.iterdir()
        if cache_dir.exists() and d.is_dir()
    )