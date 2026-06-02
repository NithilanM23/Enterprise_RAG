"""
services/router_service.py
---------------------------
Query routing and document category management.

How it works:
  1. Each document is tagged with a category when uploaded
     (e.g. "hr", "engineering", "company_info", "reference")

  2. When a query comes in, classify_query() scores it against
     each category's keyword profile using TF-IDF-like overlap.

  3. If confidence is above threshold → scope search to that category.
     If confidence is low → search all documents (safe fallback).

  4. The user never sees any of this — it's fully automatic.

Why keyword-based routing (not LLM-based):
  - Zero latency — no extra LLM call before retrieval
  - No extra model download
  - Deterministic — same query always routes the same way
  - Transparent — you can see exactly why a query was routed
  - LLM-based routing can be added later as an upgrade

Categories are defined in CATEGORY_PROFILES below.
Add a new category by adding one entry to that dict — nothing else changes.
"""

import logging
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Category definitions
# ---------------------------------------------------------------------------
# Each category has:
#   keywords  : terms strongly associated with this category
#   weight    : importance multiplier (higher = stronger signal)
#
# The router scores a query by counting weighted keyword matches.
# Add new categories here — no other code needs to change.
# ---------------------------------------------------------------------------

CATEGORY_PROFILES = {
    "company_info": {
        "keywords": [
            # Identity & location
            "company", "organisation", "organization", "founded", "located",
            "address", "headquarters", "office", "history", "about", "profile",
            "contact", "phone", "email", "website", "overview",
            # Leadership & people
            "ceo", "management", "team", "employees", "staff", "director",
            # Commercial
            "clients", "customers", "revenue", "partners", "vendor",
            # Products & services — critical additions
            "products", "services", "solutions", "offerings", "portfolio",
            "software", "platform", "tool", "application", "system", "product",
            "provide", "offer", "build", "develop", "makes", "delivers",
            # Mission & values
            "mission", "vision", "values", "culture", "goal",
        ],
        "weight": 1.0,
    },
    "hr": {
        "keywords": [
            "leave", "holiday", "vacation", "salary", "payroll", "appraisal",
            "performance", "policy", "employee", "onboarding", "offboarding",
            "resignation", "termination", "benefits", "insurance", "pf",
            "provident", "gratuity", "attendance", "wfh", "remote", "hybrid",
            "dress", "code", "conduct", "grievance", "complaint", "hr",
            "human", "resources", "recruitment", "interview", "joining",
            "training", "probation", "notice", "period", "increment",
        ],
        "weight": 1.0,
    },
    "engineering": {
        "keywords": [
            "specification", "torque", "pressure", "temperature", "voltage",
            "current", "resistance", "circuit", "component", "assembly",
            "installation", "maintenance", "calibration", "tolerance",
            "dimension", "material", "process", "procedure", "sop",
            "machine", "equipment", "tool", "sensor", "actuator", "motor",
            "hydraulic", "pneumatic", "electrical", "mechanical", "software",
            "firmware", "protocol", "interface", "api", "system", "design",
            "architecture", "diagram", "schematic", "drawing", "cad",
        ],
        "weight": 1.0,
    },
    "finance": {
        "keywords": [
            "budget", "cost", "expense", "revenue", "profit", "loss",
            "invoice", "payment", "tax", "gst", "audit", "balance",
            "sheet", "income", "statement", "cashflow", "forecast",
            "quarter", "annual", "report", "financial", "fund", "account",
            "vendor", "purchase", "procurement", "order", "contract",
        ],
        "weight": 1.0,
    },
    "reference": {
        "keywords": [
            "research", "paper", "study", "algorithm", "model", "dataset",
            "training", "neural", "network", "deep", "learning", "machine",
            "accuracy", "benchmark", "experiment", "results", "hypothesis",
            "theory", "equation", "formula", "proof", "theorem", "chapter",
            "section", "appendix", "bibliography", "reference", "citation",
        ],
        "weight": 0.8,   # lower weight — reference docs are often background material
    },
    "general": {
        "keywords": [],   # catch-all — always scores 0, used as fallback label
        "weight": 1.0,
    },
}

# Minimum confidence score to route to a specific category.
# Below this → search all documents.
# Minimum keyword matches needed to route to a specific category.
# 1.0 = route on single keyword match (more aggressive routing)
# 2.0 = require two matches (was too strict — missed "products" queries)
ROUTING_CONFIDENCE_THRESHOLD = 1.0

# ---------------------------------------------------------------------------
# Tokeniser (same as BM25 for consistency)
# ---------------------------------------------------------------------------

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
        For each category, count how many category keywords appear in the query.
        Multiply by the category weight.
        The category with the highest score wins.
        If top score < ROUTING_CONFIDENCE_THRESHOLD → return "general" (no scoping).

    Args:
        query : The user's natural language question.

    Returns:
        dict with:
            category   : str   — winning category name
            confidence : float — score of the winning category
            scores     : dict  — all category scores (for debugging/display)
            routed     : bool  — True if confidence was high enough to scope search
    """
    query_tokens = set(_tokenize(query))

    scores = {}
    for category, profile in CATEGORY_PROFILES.items():
        if not profile["keywords"]:
            scores[category] = 0.0
            continue
        keyword_set = set(profile["keywords"])
        overlap = len(query_tokens & keyword_set)
        scores[category] = overlap * profile["weight"]

    # Find winning category (excluding "general")
    searchable = {k: v for k, v in scores.items() if k != "general"}
    if not searchable or max(searchable.values()) == 0:
        return {
            "category":   "general",
            "confidence": 0.0,
            "scores":     scores,
            "routed":     False,
        }

    best_category  = max(searchable, key=searchable.get)
    best_score     = searchable[best_category]
    routed         = best_score >= ROUTING_CONFIDENCE_THRESHOLD

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

    Args:
        category : Category name (e.g. "hr", "engineering").
                   "general" returns None (meaning: search all).

    Returns:
        List of document IDs, or None if category is "general".
    """
    if category == "general":
        return None

    import psycopg2.extras
    from services.database_service import _get_raw_connection

    with _get_raw_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT id FROM documents WHERE category = %s;",
                (category,),
            )
            rows = cur.fetchall()

    ids = [r["id"] for r in rows]

    if not ids:
        logger.warning(
            "No documents found for category '%s' — falling back to global search.",
            category,
        )
        return None   # fallback: search everything

    logger.info(
        "Routing to category '%s': %d document(s) in scope.", category, len(ids)
    )
    return ids


def get_all_categories() -> list:
    """Return list of all defined category names (excluding 'general')."""
    return [c for c in CATEGORY_PROFILES.keys() if c != "general"]


def get_category_description(category: str) -> str:
    """Return a human-readable label for a category."""
    labels = {
        "company_info": "Company Information",
        "hr":           "HR & People",
        "engineering":  "Engineering & Technical",
        "finance":      "Finance & Accounts",
        "reference":    "Reference Materials",
        "general":      "General (all documents)",
    }
    return labels.get(category, category.replace("_", " ").title())