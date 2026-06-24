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

def _reciprocal_rank_fusion(result_lists: list, k: int) -> list:
    """
    Merge multiple ranked results using Reciprocal Rank Fusion.
    Chunks appearing in multiple lists get additive score boost.
    """
    rrf_scores = {}
    chunk_map  = {}

    for results in result_lists:
        for rank, chunk in enumerate(results, start=1):
            cid = chunk["id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank)
            if cid not in chunk_map:
                chunk_map[cid]  = chunk

    fused = []
    for cid, score in rrf_scores.items():
        chunk = chunk_map[cid].copy()
        chunk["rrf_score"] = score
        fused.append(chunk)

    fused.sort(key=lambda x: x["rrf_score"], reverse=True)

    logger.info(
        "RRF: Merged %d lists → %d unique candidates.",
        len(result_lists), len(fused),
    )
    return fused



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
    from services.database_service import search_similar_chunks, get_active_embedding_models

    active_models = get_active_embedding_models(document_ids)
    if not active_models:
        from services.settings_service import get_setting
        active_models = [get_setting("embedding_model")]
        logger.info("No active embedding models found. Falling back to default %s", active_models[0])
        
    query_embeddings = {}
    for model in active_models:
        try:
            query_embeddings[model] = embed_text(query, model=model)
        except Exception as exc:
            logger.warning("Failed to generate query embedding for model %s: %s", model, exc)

    if query_embeddings:
        semantic_chunks_dict = search_similar_chunks(
            query_embeddings=query_embeddings,
            top_k=semantic_k,
            document_ids=document_ids,
        )
    else:
        semantic_chunks_dict = {}

    all_semantic_chunks = []
    for model, chunks in semantic_chunks_dict.items():
        for c in chunks:
            c.setdefault("bm25_score", 0.0)
        all_semantic_chunks.extend(chunks)

    logger.info("Semantic: retrieved %d chunks across %d models.", len(all_semantic_chunks), len(semantic_chunks_dict))

    # ------------------------------------------------------------------
    # Step 3 — BM25 keyword search
    # ------------------------------------------------------------------
    from services.bm25_service import search as bm25_search, index_exists

    raw_bm25 = []
    if index_exists():
        raw_bm25 = bm25_search(query, top_k=bm25_k, document_ids=document_ids)
        logger.info("BM25: %d chunks retrieved.", len(raw_bm25))
    else:
        logger.warning("BM25 index not found — semantic-only retrieval.")

    # ------------------------------------------------------------------
    # Step 4 — RRF Fusion
    # ------------------------------------------------------------------
    lists_to_fuse = list(semantic_chunks_dict.values())
    if raw_bm25:
        lists_to_fuse.append(raw_bm25)
        
    fused = _reciprocal_rank_fusion(lists_to_fuse, k=rrf_k)

    if not fused:
        logger.warning("No chunks after fusion.")
        return []

    rrf_candidates = fused[:RRF_POOL]

    # ------------------------------------------------------------------
    # Step 5 — MMR Diversification
    # ------------------------------------------------------------------
    if len(active_models) == 1 and query_embeddings.get(active_models[0]):
        try:
            from services.mmr_service import apply_mmr
            mmr_candidates = apply_mmr(
                query_embedding=query_embeddings[active_models[0]],
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
    else:
        logger.info("MMR skipped (multi-model or missing query vector) — using RRF order.")
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