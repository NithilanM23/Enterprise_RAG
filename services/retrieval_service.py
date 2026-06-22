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
      |         → top semantic_k chunks
      |
      ├── [3] BM25 KEYWORD SEARCH (scoped to category)
      |         → top bm25_k chunks
      |
      ├── [4] RRF FUSION
      |         → merge both lists → top RRF_POOL candidates
      |
      ├── [5] MMR DIVERSIFICATION
      |         → ensure no single document monopolises results
      |         → diverse mmr_pool candidates
      |
      └── [6] CROSS-ENCODER RERANKER
                → re-score for true relevance
                → final top top_k chunks → LLM

All hyperparameters below (semantic_k, bm25_k, rrf_k, mmr_pool, mmr_lambda,
top_k, routing threshold) are read from services.settings_service at
CALL TIME inside retrieve() — never captured as module-level constants.
This is what makes the admin settings panel actually take effect without
a server restart. See services/settings_service.py for the single source
of truth and services/category_service.py for category keyword profiles.
"""

import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

# Fixed structural constant — not admin-tunable. RRF_POOL determines how
# many fused candidates flow into MMR; it should always be >= mmr_pool.
# Kept generous so MMR has enough candidates to diversify from.
RRF_POOL = 30


# ---------------------------------------------------------------------------
# RRF merger
# ---------------------------------------------------------------------------

def _reciprocal_rank_fusion(semantic_results: list, bm25_results: list, k: int) -> list:
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
    user_id: int,
    top_k: int = None,
    document_ids: list = None,
) -> list:
    """
    Full retrieval pipeline: routing → semantic + BM25 → RRF → MMR → rerank.

    Args:
        query        : The user's natural language question.
        top_k        : Final chunks to return. Defaults to the live
                       'top_k' setting (admin-tunable).
        document_ids : Optional manual override for document scoping.
                       If None, smart routing determines the scope automatically.

    Returns:
        List of top_k chunk dicts ordered by reranker score, each containing:
            id, chunk_text, filename, chunk_number, document_id,
    Execute the full retrieval pipeline:
      Routing → Semantic (pgvector) → BM25 → RRF Fusion → MMR → Cross-Encoder
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    from services.settings_service import get_setting

    k             = top_k if top_k is not None else get_setting("top_k", user_id)
    semantic_k    = get_setting("semantic_k", user_id)
    bm25_k        = get_setting("bm25_k", user_id)
    rrf_k         = get_setting("rrf_k", user_id)
    mmr_pool      = get_setting("mmr_pool", user_id)
    mmr_lambda    = get_setting("mmr_lambda", user_id)

    logger.info("Retrieval started for query: '%s'", query[:80])

    from services.database_service import get_all_documents
    user_docs = [d["id"] for d in get_all_documents(user_id)]
    
    if document_ids is None:
        document_ids = user_docs
    else:
        document_ids = [d for d in document_ids if d in user_docs]

    if not document_ids:
        logger.info("No documents available for retrieval for this user.")
        return []

    # ------------------------------------------------------------------
    # Step 1 — Smart Routing (auto document scoping)
    # ------------------------------------------------------------------
    routing_info = {"category": "general", "routed": False, "confidence": 0.0}

    if True: # Always run routing but scope within document_ids
        from services.router_service import classify_query, get_document_ids_for_category
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
                    document_ids = None
                    routing_info["routed"] = False
                    routing_info["soft_scope"] = scoped_ids
                    logger.info(
                        "Soft routing: category='%s' confidence=%.1f → global search.",
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
        top_k=semantic_k,
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
        raw_bm25 = bm25_search(query, top_k=bm25_k, document_ids=document_ids)

        semantic_ids     = {c["id"] for c in semantic_chunks}
        bm25_id_to_score = {c["id"]: c["bm25_score"] for c in raw_bm25}

        for c in semantic_chunks:
            if c["id"] in bm25_id_to_score:
                c["bm25_score"] = bm25_id_to_score[c["id"]]

        missing_ids = [c["id"] for c in raw_bm25 if c["id"] not in semantic_ids]
        if missing_ids:
            bm25_only = _fetch_chunks_by_ids(missing_ids)
            for c in bm25_only:
                c["bm25_score"] = bm25_id_to_score.get(c["id"], 0.0)
                c["similarity"] = 0.0
            bm25_chunks = bm25_only

        logger.info("BM25: %d chunks retrieved (%d new).", len(raw_bm25), len(bm25_chunks))
    else:
        logger.warning("BM25 index not found — semantic-only retrieval.")

    # ------------------------------------------------------------------
    # Step 4 — RRF Fusion
    # ------------------------------------------------------------------
    fused = _reciprocal_rank_fusion(semantic_chunks, bm25_chunks, k=rrf_k)

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
            top_k=mmr_pool,
            lambda_val=mmr_lambda,
        )
        logger.info(
            "MMR: %d candidates across %d unique docs.",
            len(mmr_candidates),
            len({c["document_id"] for c in mmr_candidates}),
        )
    except Exception as exc:
        logger.warning("MMR failed (%s) — using RRF order.", exc)
        mmr_candidates = rrf_candidates[:mmr_pool]
        for c in mmr_candidates:
            c["mmr_score"] = c.get("rrf_score", 0.0)

    # ------------------------------------------------------------------
    # Step 6 — Cross-encoder reranker
    # ------------------------------------------------------------------
    try:
        from services.reranker_service import rerank
        reranked = rerank(query, mmr_candidates, top_k=len(mmr_candidates))

        final_chunks = reranked[:k]
        logger.info("Reranker: returning top %d of %d candidates.", len(final_chunks), len(reranked))

    except Exception as exc:
        logger.warning("Reranker unavailable (%s) — using MMR order.", exc)
        for c in mmr_candidates:
            c["reranker_score"] = c.get("rrf_score", 0.0)
        final_chunks = mmr_candidates[:k]

    for c in final_chunks:
        c["routing"] = routing_info

    return final_chunks