"""
services/mmr_service.py
------------------------
Maximal Marginal Relevance (MMR) diversification.

Problem MMR solves:
    Standard semantic search returns the top-K most similar chunks.
    If one document dominates the corpus (e.g. a 500-page ML book),
    all K slots may be filled by chunks from that single document,
    burying relevant chunks from smaller documents.

How MMR works:
    Instead of picking the K most similar chunks all at once,
    MMR picks them one at a time using a greedy algorithm:

    Step 1: Pick the chunk most similar to the query. Add to Selected.

    Step 2: For each remaining chunk, compute:
        MMR_score = λ × Sim(chunk, query)
                  - (1-λ) × max(Sim(chunk, selected_chunk) for selected_chunk in Selected)

        First term  → rewards relevance to the query
        Second term → penalises redundancy with already-selected chunks

    Step 3: Pick the chunk with the highest MMR_score. Add to Selected.

    Repeat Steps 2-3 until K chunks are selected.

The λ parameter controls the tradeoff:
    λ = 1.0 → pure relevance (same as standard top-K, no diversity)
    λ = 0.5 → equal weight on relevance and diversity
    λ = 0.7 → recommended default (prioritise relevance, but ensure diversity)
    λ = 0.0 → pure diversity (ignores query relevance entirely)

Why we apply MMR before the reranker:
    MMR works on semantic similarity scores (fast, no model inference).
    It produces a diverse candidate pool.
    The reranker then selects the most relevant from that diverse pool.
    Result: diverse AND precise.
"""

import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

# MMR lambda — tradeoff between relevance and diversity
# 0.7 = prioritise relevance but ensure no single document dominates
MMR_LAMBDA = 0.7


# ---------------------------------------------------------------------------
# Cosine similarity helper
# ---------------------------------------------------------------------------

def _cosine_similarity(vec_a: list, vec_b: list) -> float:
    """
    Compute cosine similarity between two vectors.
    Both must be the same length.
    Returns a float in [-1, 1]. Higher = more similar.
    """
    dot   = sum(a * b for a, b in zip(vec_a, vec_b))
    mag_a = sum(a * a for a in vec_a) ** 0.5
    mag_b = sum(b * b for b in vec_b) ** 0.5

    if mag_a == 0 or mag_b == 0:
        return 0.0

    return dot / (mag_a * mag_b)


# ---------------------------------------------------------------------------
# Fetch embeddings from DB
# ---------------------------------------------------------------------------

def _fetch_embeddings(chunk_ids: list) -> dict:
    """
    Fetch raw embedding vectors for a list of chunk IDs from the database.

    Returns:
        Dict mapping chunk_id → embedding vector (list of floats).
        Chunks with NULL embeddings are excluded.
    """
    import psycopg2.extras
    from services.database_service import _get_connection

    if not chunk_ids:
        return {}

    with _get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, embedding
                FROM embeddings
                WHERE id = ANY(%s) AND embedding IS NOT NULL;
                """,
                (chunk_ids,),
            )
            rows = cur.fetchall()

    return {
        row["id"]: list(row["embedding"])
        for row in rows
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply_mmr(
    query_embedding: list,
    chunks: list,
    top_k: int,
    lambda_val: float = MMR_LAMBDA,
) -> list:
    """
    Apply MMR to a list of candidate chunks to ensure diversity.

    Args:
        query_embedding : The query vector (same model as chunk embeddings).
        chunks          : Candidate chunk dicts from retrieval.
                          Each must have an 'id' key.
        top_k           : Number of diverse chunks to return.
        lambda_val      : MMR lambda (0.7 default — relevance-biased diversity).

    Returns:
        Top_k chunks selected by MMR, ordered by selection order
        (first = most relevant, subsequent = most relevant + diverse).

        Each chunk gets an 'mmr_score' key added.
        Returns original list unchanged if embeddings cannot be fetched
        (graceful fallback).
    """
    if not chunks:
        return chunks

    if len(chunks) <= top_k:
        logger.debug("MMR: fewer candidates (%d) than top_k (%d) — skipping.", len(chunks), top_k)
        return chunks

    # Fetch embeddings for all candidate chunks
    chunk_ids = [c["id"] for c in chunks]
    embeddings = _fetch_embeddings(chunk_ids)

    if not embeddings:
        logger.warning("MMR: could not fetch embeddings — returning original ranking.")
        return chunks[:top_k]

    # Filter to chunks that have embeddings
    valid_chunks = [c for c in chunks if c["id"] in embeddings]

    if len(valid_chunks) < 2:
        return valid_chunks[:top_k]

    logger.info(
        "MMR: selecting %d from %d candidates (lambda=%.1f).",
        top_k, len(valid_chunks), lambda_val,
    )

    # Pre-compute query similarities for all candidates
    query_sims = {
        c["id"]: _cosine_similarity(query_embedding, embeddings[c["id"]])
        for c in valid_chunks
    }

    selected      = []    # selected chunk dicts
    selected_ids  = []    # selected chunk IDs (for fast lookup)
    remaining     = list(valid_chunks)   # candidates not yet selected

    while len(selected) < top_k and remaining:

        if not selected:
            # First selection: pick chunk most similar to query
            best = max(remaining, key=lambda c: query_sims[c["id"]])
            best["mmr_score"] = query_sims[best["id"]]

        else:
            # Subsequent selections: MMR score balances relevance vs redundancy
            best       = None
            best_score = float("-inf")

            for chunk in remaining:
                cid = chunk["id"]

                # Relevance term
                relevance = query_sims[cid]

                # Redundancy term — max similarity to any already-selected chunk
                redundancy = max(
                    _cosine_similarity(embeddings[cid], embeddings[sel_id])
                    for sel_id in selected_ids
                )

                mmr_score = lambda_val * relevance - (1 - lambda_val) * redundancy

                if mmr_score > best_score:
                    best_score = mmr_score
                    best       = chunk

            best["mmr_score"] = best_score

        selected.append(best)
        selected_ids.append(best["id"])
        remaining = [c for c in remaining if c["id"] != best["id"]]

    logger.info(
        "MMR complete: %d chunks selected across %d unique documents.",
        len(selected),
        len({c["document_id"] for c in selected}),
    )

    return selected
