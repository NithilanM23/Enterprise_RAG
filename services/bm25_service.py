"""
services/bm25_service.py
-------------------------
BM25 keyword search for the Local Employee Knowledge Assistant.

BM25 (Best Match 25) is a classical keyword ranking algorithm that scores
chunks based on term frequency and inverse document frequency. It excels at
exact keyword matches — employee IDs, product codes, proper nouns, dates,
acronyms — where semantic search often fails.

How it works here:
  1. On every document ingestion, rebuild the BM25 index from all chunks in DB
  2. Serialize the index to disk (bm25_index.pkl) for fast reloading
  3. At query time, tokenize the query and score all chunks
  4. Return top-K chunk IDs and scores

Why not store BM25 in PostgreSQL:
  BM25 needs the entire corpus in memory to compute IDF scores correctly.
  A serialized pickle file is simple, fast, and sufficient for POC scale.
  At enterprise scale this would move to Elasticsearch or PostgreSQL FTS.

Dependencies:
  rank_bm25  (pip install rank-bm25)
"""

import logging
import os
import pickle
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import BASE_DIR

logger = logging.getLogger(__name__)

# Path where the BM25 index is persisted between runs
BM25_INDEX_PATH = BASE_DIR / "bm25_index.pkl"


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list:
    """
    Tokenize text for BM25 indexing and querying.

    Steps:
      1. Lowercase
      2. Split on non-alphanumeric characters
      3. Remove tokens shorter than 2 characters
      4. No stemming — keeps proper nouns, IDs, codes intact

    The same tokenizer must be used for both indexing and querying.
    """
    text = text.lower()
    tokens = re.split(r"[^a-z0-9]+", text)
    return [t for t in tokens if len(t) >= 2]


# ---------------------------------------------------------------------------
# Index management
# ---------------------------------------------------------------------------

def build_index() -> dict:
    """
    Build a fresh BM25 index from ALL chunks currently in the database.

    Fetches every chunk's text and ID, builds the BM25 corpus, fits the
    model, and saves the index to disk.

    Returns:
        index_info dict:
            chunk_count : int   — number of chunks indexed
            index_path  : str   — path to saved index file

    Raises:
        RuntimeError: If no chunks are found in the database (nothing to index).
    """
    from rank_bm25 import BM25Okapi
    import psycopg2.extras
    from services.database_service import _get_connection

    logger.info("Building BM25 index from database chunks...")

    # Fetch all chunks with their IDs
    query = """
        SELECT e.id, e.chunk_text, e.document_id, d.filename, e.chunk_number
        FROM embeddings e
        JOIN documents d ON d.id = e.document_id
        ORDER BY e.document_id, e.chunk_number;
    """

    with _get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(query)
            rows = [dict(r) for r in cur.fetchall()]

    if not rows:
        raise RuntimeError(
            "No chunks found in database. "
            "Ingest at least one document before building the BM25 index."
        )

    # Build tokenized corpus — one list of tokens per chunk
    chunk_ids   = [r["id"]          for r in rows]
    chunk_texts = [r["chunk_text"]  for r in rows]
    filenames   = [r["filename"]    for r in rows]
    chunk_nums  = [r["chunk_number"] for r in rows]
    doc_ids     = [r["document_id"] for r in rows]

    tokenized_corpus = [_tokenize(text) for text in chunk_texts]

    # Fit BM25 model
    bm25 = BM25Okapi(tokenized_corpus)

    # Package everything needed for search into the index payload
    index_payload = {
        "bm25":         bm25,
        "chunk_ids":    chunk_ids,
        "chunk_texts":  chunk_texts,
        "filenames":    filenames,
        "chunk_nums":   chunk_nums,
        "doc_ids":      doc_ids,
        "total_chunks": len(rows),
    }

    # Persist to disk
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(index_payload, f)

    logger.info(
        "BM25 index built: %d chunks indexed → saved to %s",
        len(rows), BM25_INDEX_PATH,
    )

    return {
        "chunk_count": len(rows),
        "index_path":  str(BM25_INDEX_PATH),
    }


def load_index() -> dict:
    """
    Load the BM25 index from disk.

    Returns:
        The index payload dict (same structure as built by build_index).

    Raises:
        FileNotFoundError: If no index file exists yet.
    """
    if not BM25_INDEX_PATH.exists():
        raise FileNotFoundError(
            "BM25 index not found. "
            "Run:  python main.py --build-index  to create it."
        )

    with open(BM25_INDEX_PATH, "rb") as f:
        payload = pickle.load(f)

    logger.debug(
        "BM25 index loaded: %d chunks from %s",
        payload["total_chunks"], BM25_INDEX_PATH,
    )
    return payload


def index_exists() -> bool:
    """Return True if a BM25 index file exists on disk."""
    return BM25_INDEX_PATH.exists()


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def search(query: str, top_k: int = 10, document_ids: list = None) -> list:
    """
    Search the BM25 index for chunks matching the query keywords.

    Args:
        query        : The user's natural language question.
        top_k        : Number of top results to return.
        document_ids : Optional list of document IDs to restrict results to.
                       Filters AFTER scoring — BM25 still scores all chunks
                       but only returns chunks belonging to selected documents.

    Returns:
        List of dicts sorted by BM25 score descending.
        Empty list if index does not exist or query has no matching tokens.
    """
    if not index_exists():
        logger.warning("BM25 index not found — skipping keyword search.")
        return []

    try:
        payload = load_index()
    except Exception as exc:
        logger.error("Failed to load BM25 index: %s", exc)
        return []

    bm25         = payload["bm25"]
    chunk_ids    = payload["chunk_ids"]
    chunk_texts  = payload["chunk_texts"]
    filenames    = payload["filenames"]
    chunk_nums   = payload["chunk_nums"]
    doc_ids      = payload["doc_ids"]

    # Tokenize the query with the same tokenizer used for indexing
    query_tokens = _tokenize(query)

    if not query_tokens:
        logger.warning("Query produced no tokens after tokenization: '%s'", query)
        return []

    # Get BM25 scores for every chunk
    scores = bm25.get_scores(query_tokens)

    # Pair scores with chunk metadata and sort descending
    scored = [
        {
            "id":           chunk_ids[i],
            "chunk_text":   chunk_texts[i],
            "filename":     filenames[i],
            "chunk_number": chunk_nums[i],
            "document_id":  doc_ids[i],
            "bm25_score":   float(scores[i]),
        }
        for i in range(len(chunk_ids))
        if scores[i] > 0     # skip chunks with zero relevance
    ]

    scored.sort(key=lambda x: x["bm25_score"], reverse=True)

    # Apply document scope filter if specified
    if document_ids:
        doc_id_set = set(document_ids)
        scored = [c for c in scored if c["document_id"] in doc_id_set]

    logger.info(
        "BM25 search: %d chunks with score > 0, returning top %d.",
        len(scored), min(top_k, len(scored)),
    )

    return scored[:top_k]