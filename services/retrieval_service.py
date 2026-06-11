"""
services/retrieval_service.py
------------------------------
Full hybrid retrieval pipeline:

    Query
      |
      ├── [1] SMART ROUTING
      |     classify_query() → category → document_ids to search
      |
      ├── [2] SEMANTIC SEARCH (pgvector, scoped to category)
      |         → top SEMANTIC_K chunks
      |
      ├── [3] BM25 KEYWORD SEARCH (scoped to category)
      |         → top BM25_K chunks
      |
      ├── [4] RRF FUSION
      |         → merge both lists → top RERANK_POOL candidates
      |
      ├── [5] MMR DIVERSIFICATION
      |         → ensure no single document monopolises results
      |         → diverse MMR_POOL candidates
      |
      └── [6] CROSS-ENCODER RERANKER
                → re-score for true relevance
                → final top TOP_K chunks → LLM
"""

import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import TOP_K

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Retrieval hyperparameters
# ---------------------------------------------------------------------------

SEMANTIC_K    = 20    # chunks from semantic search
BM25_K        = 20    # chunks from BM25 search
RRF_K         = 60    # RRF dampening constant
RRF_POOL      = 30    # candidates after RRF → into MMR
MMR_POOL      = 20    # candidates after MMR → into reranker
# Raised from 0.7 — strongly prefer relevance, only diversify when chunks are nearly equal
MMR_LAMBDA    = 0.85  # MMR relevance/diversity tradeoff (0=diverse, 1=relevant)


# ---------------------------------------------------------------------------
# RRF merger
# ---------------------------------------------------------------------------

def _reciprocal_rank_fusion(
    semantic_results: list,
    bm25_results: list,
    k: int = RRF_K,
) -> list:
    """
    Merge semantic and BM25 results using Reciprocal Rank Fusion.
    Chunks appearing in both lists get additive score boost.
    """
    rrf_scores = {}
    chunk_map  = {}

    for rank, chunk in enumerate(semantic_results, start=1):
        cid = chunk["id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank)
        chunk_map[cid]  = chunk

    for rank, chunk in enumerate(bm25_results, start=1):
        cid = chunk["id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank)
        if cid not in chunk_map:
            chunk_map[cid] = chunk

    fused = []
    for cid, score in rrf_scores.items():
        chunk = chunk_map[cid].copy()
        chunk["rrf_score"] = score
        fused.append(chunk)

    fused.sort(key=lambda x: x["rrf_score"], reverse=True)

    logger.info(
        "RRF: %d semantic + %d BM25 → %d unique candidates.",
        len(semantic_results), len(bm25_results), len(fused),
    )
    return fused


# ---------------------------------------------------------------------------
# Helper: fetch full chunk metadata for BM25-only results
# ---------------------------------------------------------------------------

def _fetch_chunks_by_ids(chunk_ids: list) -> list:
    """Fetch full chunk metadata from DB for chunks only found by BM25."""
    import psycopg2.extras
    from services.database_service import _get_connection

    if not chunk_ids:
        return []

    with _get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT e.id, e.chunk_text, e.chunk_number, e.document_id,
                       d.filename
                FROM embeddings e
                JOIN documents d ON d.id = e.document_id
                WHERE e.id = ANY(%s);
            """, (chunk_ids,))
            rows = [dict(r) for r in cur.fetchall()]

    for r in rows:
        r["similarity"] = 0.0
    return rows


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def retrieve(
    query: str,
    top_k: int = None,
    document_ids: list = None,
) -> list:
    """
    Full retrieval pipeline: routing → semantic + BM25 → RRF → MMR → rerank.

    Args:
        query        : The user's natural language question.
        top_k        : Final chunks to return. Defaults to TOP_K in config.
        document_ids : Optional manual override for document scoping.
                       If None, smart routing determines the scope automatically.

    Returns:
        List of top_k chunk dicts ordered by reranker score, each containing:
            id              : int
            chunk_text      : str
            filename        : str
            chunk_number    : int
            document_id     : int
            similarity      : float   (semantic score)
            bm25_score      : float
            rrf_score       : float
            mmr_score       : float
            reranker_score  : float
            routing         : dict    (routing decision metadata)
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    k = top_k if top_k is not None else TOP_K
    logger.info("Retrieval started for query: '%s'", query[:80])

    # ------------------------------------------------------------------
    # Step 1 — Smart Routing (auto document scoping)
    # ------------------------------------------------------------------
    routing_info = {"category": "general", "routed": False, "confidence": 0.0}

    if document_ids is None:
        from services.router_service import (
            classify_query, get_document_ids_for_category, ROUTING_CONFIDENCE_THRESHOLD
        )
        routing_info = classify_query(query)

        if routing_info["routed"]:
            scoped_ids = get_document_ids_for_category(routing_info["category"])

            if scoped_ids:
                confidence = routing_info["confidence"]

                if confidence >= 2.0:
                    # High confidence → hard scope to category only
                    document_ids = scoped_ids
                    logger.info(
                        "Hard routing: category='%s' confidence=%.1f → %d docs.",
                        routing_info["category"], confidence, len(scoped_ids),
                    )
                else:
                    # Low confidence → soft scope: prefer category but don't exclude others.
                    # Search globally; category docs will naturally rank higher because
                    # their content matches the routing keyword that triggered routing.
                    document_ids = None
                    routing_info["routed"] = False
                    routing_info["soft_scope"] = scoped_ids
                    logger.info(
                        "Soft routing: category='%s' confidence=%.1f → global search "
                        "(category docs preferred by relevance).",
                        routing_info["category"], confidence,
                    )
            else:
                document_ids = None
                routing_info["routed"] = False
        else:
            logger.info(
                "Routing confidence %.1f below threshold — global search.",
                routing_info["confidence"],
            )

    # ------------------------------------------------------------------
    # Step 2 — Semantic search
    # ------------------------------------------------------------------
    from services.embedding_service import embed_text
    from services.database_service import search_similar_chunks

    query_vector    = embed_text(query)
    semantic_chunks = search_similar_chunks(
        query_embedding=query_vector,
        top_k=SEMANTIC_K,
        document_ids=document_ids,
    )

    for c in semantic_chunks:
        c.setdefault("bm25_score", 0.0)

    logger.info("Semantic: %d chunks retrieved.", len(semantic_chunks))

    # ------------------------------------------------------------------
    # Step 3 — BM25 keyword search
    # ------------------------------------------------------------------
    from services.bm25_service import search as bm25_search, index_exists

    bm25_chunks = []
    if index_exists():
        raw_bm25 = bm25_search(query, top_k=BM25_K, document_ids=document_ids)

        semantic_ids     = {c["id"] for c in semantic_chunks}
        bm25_id_to_score = {c["id"]: c["bm25_score"] for c in raw_bm25}

        # Add BM25 scores to chunks already in semantic results
        for c in semantic_chunks:
            if c["id"] in bm25_id_to_score:
                c["bm25_score"] = bm25_id_to_score[c["id"]]

        # Fetch full metadata for BM25-only chunks
        missing_ids = [c["id"] for c in raw_bm25 if c["id"] not in semantic_ids]
        if missing_ids:
            bm25_only = _fetch_chunks_by_ids(missing_ids)
            for c in bm25_only:
                c["bm25_score"] = bm25_id_to_score.get(c["id"], 0.0)
                c["similarity"] = 0.0
            bm25_chunks = bm25_only

        logger.info(
            "BM25: %d chunks retrieved (%d new).", len(raw_bm25), len(bm25_chunks)
        )
    else:
        logger.warning("BM25 index not found — semantic-only retrieval.")

    # ------------------------------------------------------------------
    # Step 4 — RRF Fusion
    # ------------------------------------------------------------------
    fused = _reciprocal_rank_fusion(semantic_chunks, bm25_chunks)

    if not fused:
        logger.warning("No chunks after fusion.")
        return []

    rrf_candidates = fused[:RRF_POOL]

    # ------------------------------------------------------------------
    # Step 5 — MMR Diversification
    # ------------------------------------------------------------------
    try:
        from services.mmr_service import apply_mmr
        mmr_candidates = apply_mmr(
            query_embedding=query_vector,
            chunks=rrf_candidates,
            top_k=MMR_POOL,
            lambda_val=MMR_LAMBDA,
        )
        logger.info(
            "MMR: %d candidates across %d unique docs.",
            len(mmr_candidates),
            len({c["document_id"] for c in mmr_candidates}),
        )
    except Exception as exc:
        logger.warning("MMR failed (%s) — using RRF order.", exc)
        mmr_candidates = rrf_candidates[:MMR_POOL]
        for c in mmr_candidates:
            c["mmr_score"] = c.get("rrf_score", 0.0)

    # ------------------------------------------------------------------
    # Step 6 — Cross-encoder reranker
    # ------------------------------------------------------------------
    try:
        from services.reranker_service import rerank
        reranked = rerank(query, mmr_candidates, top_k=len(mmr_candidates))

        # Use top-K only — no hard score threshold.
        # The reranker already orders correctly; a threshold adds no value
        # and removes valid internal document chunks that score low on this
        # web-trained model due to writing style differences.
        final_chunks = reranked[:k]
        logger.info(
            "Reranker: returning top %d of %d candidates.",
            len(final_chunks), len(reranked),
        )

    except Exception as exc:
        logger.warning("Reranker unavailable (%s) — using MMR order.", exc)
        for c in mmr_candidates:
            c["reranker_score"] = c.get("rrf_score", 0.0)
        final_chunks = mmr_candidates[:k]

    # Attach routing info to each chunk for UI display
    for c in final_chunks:
        c["routing"] = routing_info

    return final_chunks