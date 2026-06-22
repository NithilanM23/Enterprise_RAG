"""
services/router_service.py
---------------------------
Query routing — classifies a question into the most relevant document
category so retrieval can be scoped automatically.

How it works:
  1. Each document is tagged with a category when uploaded
     (predefined: hr, engineering, company_info, finance, reference,
      or any custom category created via category_service).

  2. When a query comes in, classify_query() scores it against each
     category's keyword profile (read live from the database via
     category_service — NOT a hardcoded dict anymore, so categories
     created through the admin UI or upload flow participate in
     routing immediately, no restart needed).

  3. If confidence is above threshold → scope search to that category.
     If confidence is low → search all documents (safe fallback).

  4. The user never sees any of this — it's fully automatic.

Why keyword-based routing (not LLM-based):
  - Zero latency — no extra LLM call before retrieval
  - Deterministic — same query always routes the same way
  - Transparent — you can see exactly why a query was routed

Category keyword profiles now live in the `categories` table
(see services/category_service.py) instead of a hardcoded dict here.
This is what makes custom categories possible without code changes.
"""

import logging
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> list:
    text = text.lower()
    tokens = re.split(r"[^a-z0-9]+", text)
    return [t for t in tokens if len(t) >= 2]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_query(query: str) -> dict:
    """
    Classify a query into the most relevant document category.

    Scoring:
        For each category, count how many of its keywords appear in the
        query, multiplied by the category's weight. Highest score wins.
        Categories with no keywords (new custom categories, or "general")
        always score 0 and never win — they rely on global/soft search
        until an admin tunes their keyword profile.

    Returns:
        dict with:
            category   : str   — winning category name
            confidence : float — score of the winning category
            scores     : dict  — all category scores (for debugging/display)
            routed     : bool  — True if confidence was high enough to scope search
    """
    from services.category_service import get_all_category_profiles
    from services.settings_service import get_setting

    threshold = get_setting("routing_confidence_threshold")
    profiles  = get_all_category_profiles()

    query_tokens = set(_tokenize(query))

    scores = {}
    for category, profile in profiles.items():
        keywords = profile.get("keywords") or []
        if not keywords:
            scores[category] = 0.0
            continue
        keyword_set = set(keywords)
        overlap = len(query_tokens & keyword_set)
        scores[category] = overlap * profile.get("weight", 1.0)

    searchable = {k: v for k, v in scores.items() if k != "general"}
    if not searchable or max(searchable.values()) == 0:
        return {
            "category":   "general",
            "confidence": 0.0,
            "scores":     scores,
            "routed":     False,
        }

    best_category = max(searchable, key=searchable.get)
    best_score    = searchable[best_category]
    routed        = best_score >= threshold

    logger.info(
        "Query routing: category='%s' confidence=%.1f routed=%s | query='%s'",
        best_category, best_score, routed, query[:60],
    )

    return {
        "category":   best_category if routed else "general",
        "confidence": best_score,
        "scores":     scores,
        "routed":     routed,
    }


def get_document_ids_for_category(category: str) -> list:
    """
    Return all document IDs tagged with the given category.
    "general" returns None (meaning: search all documents).
    """
    if category == "general":
        return None

    import psycopg2.extras
    from services.database_service import _get_raw_connection

    with _get_raw_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT id FROM documents WHERE category = %s;", (category,))
            rows = cur.fetchall()

    ids = [r["id"] for r in rows]

    if not ids:
        logger.warning(
            "No documents found for category '%s' — falling back to global search.",
            category,
        )
        return None

    logger.info("Routing to category '%s': %d document(s) in scope.", category, len(ids))
    return ids


def get_all_categories() -> list:
    """Return list of all category names (predefined + custom), excluding 'general'."""
    from services.category_service import list_categories
    return [c["name"] for c in list_categories(include_doc_counts=False) if c["name"] != "general"]


def get_category_description(category: str) -> str:
    """Human-readable label for a category."""
    from services.category_service import get_category_description as _get_desc
    return _get_desc(category)