"""
services/category_service.py
------------------------------
Database-backed document categories — replaces the hardcoded
CATEGORY_PROFILES dict that used to live in router_service.py.

Why this exists:
  Categories used to be fixed at code-time (company_info, hr, engineering,
  finance, reference, general). Admins/users can now create their own
  category on the fly while uploading a document — e.g. "Legal", "Quality
  Control" — without touching code or restarting the server.

Two kinds of categories:
  Predefined — seeded once on first startup, ship with a curated keyword
               profile so smart routing works well out of the box.
  Custom     — created by a user/admin at upload time. Starts with NO
               keywords (so it never wins the routing competition and
               falls back to global "soft" search) until an admin later
               adds keywords via the admin panel to make routing for it
               more precise.

Schema:
  categories
    id, name (slug, unique), label, keywords (JSONB array), weight,
    is_custom (bool), created_at
"""

import logging
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Seed data — same keyword profiles that used to be hardcoded.
# Only inserted if the categories table is empty (first run).
# ---------------------------------------------------------------------------

_SEED_CATEGORIES = [
    {
        "name": "company_info", "label": "Company Information", "weight": 1.0,
        "keywords": [
            "company", "organisation", "organization", "founded", "located",
            "address", "headquarters", "office", "history", "about", "profile",
            "contact", "phone", "email", "website", "overview",
            "ceo", "management", "team", "employees", "staff", "director",
            "clients", "customers", "revenue", "partners", "vendor",
            "products", "services", "solutions", "offerings", "portfolio",
            "software", "platform", "tool", "application", "system", "product",
            "provide", "offer", "build", "develop", "makes", "delivers",
            "mission", "vision", "values", "culture", "goal",
        ],
    },
    {
        "name": "hr", "label": "HR & People", "weight": 1.0,
        "keywords": [
            "leave", "holiday", "vacation", "salary", "payroll", "appraisal",
            "performance", "policy", "employee", "onboarding", "offboarding",
            "resignation", "termination", "benefits", "insurance", "pf",
            "provident", "gratuity", "attendance", "wfh", "remote", "hybrid",
            "dress", "code", "conduct", "grievance", "complaint", "hr",
            "human", "resources", "recruitment", "interview", "joining",
            "training", "probation", "notice", "period", "increment",
        ],
    },
    {
        "name": "engineering", "label": "Engineering & Technical", "weight": 1.0,
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
    },
    {
        "name": "finance", "label": "Finance & Accounts", "weight": 1.0,
        "keywords": [
            "budget", "cost", "expense", "revenue", "profit", "loss",
            "invoice", "payment", "tax", "gst", "audit", "balance",
            "sheet", "income", "statement", "cashflow", "forecast",
            "quarter", "annual", "report", "financial", "fund", "account",
            "vendor", "purchase", "procurement", "order", "contract",
        ],
    },
    {
        "name": "reference", "label": "Reference Materials", "weight": 0.8,
        "keywords": [
            "research", "paper", "study", "algorithm", "model", "dataset",
            "training", "neural", "network", "deep", "learning", "machine",
            "accuracy", "benchmark", "experiment", "results", "hypothesis",
            "theory", "equation", "formula", "proof", "theorem", "chapter",
            "section", "appendix", "bibliography", "reference", "citation",
        ],
    },
    {
        "name": "general", "label": "General (all documents)", "weight": 1.0,
        "keywords": [],   # catch-all, always scores 0, used as fallback
    },
]


# ---------------------------------------------------------------------------
# Table creation + seeding
# ---------------------------------------------------------------------------

def ensure_category_table() -> None:
    """Create categories table and seed predefined categories. Idempotent."""
    import json
    from services.database_service import _get_raw_connection

    with _get_raw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS categories (
                    id         SERIAL PRIMARY KEY,
                    name       TEXT UNIQUE NOT NULL,
                    label      TEXT NOT NULL,
                    keywords   JSONB NOT NULL DEFAULT '[]',
                    weight     REAL NOT NULL DEFAULT 1.0,
                    is_custom  BOOLEAN NOT NULL DEFAULT FALSE,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
            """)
            cur.execute("SELECT COUNT(*) FROM categories;")
            count = cur.fetchone()[0]

            if count == 0:
                for cat in _SEED_CATEGORIES:
                    cur.execute("""
                        INSERT INTO categories (name, label, keywords, weight, is_custom)
                        VALUES (%s, %s, %s, %s, FALSE)
                        ON CONFLICT (name) DO NOTHING;
                    """, (cat["name"], cat["label"], json.dumps(cat["keywords"]), cat["weight"]))
                logger.info("Seeded %d predefined categories.", len(_SEED_CATEGORIES))

    logger.debug("categories table ensured.")


def _slugify(name: str) -> str:
    """Convert a display name into a safe slug for the 'name' column."""
    slug = name.strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "_", slug).strip("_")
    return slug or "custom"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_all_category_profiles() -> dict:
    """
    Return every category as {name: {keywords, weight}} — the same shape
    router_service.classify_query() used to read from the hardcoded
    CATEGORY_PROFILES dict, so routing logic is unchanged, only the
    source of truth moved to the database.
    """
    import json
    import psycopg2.extras
    from services.database_service import _get_connection

    # Safety net — ensures table exists even if init_system() was cached
    ensure_category_table()

    with _get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT name, keywords, weight FROM categories;")
            rows = cur.fetchall()

    profiles = {}
    for r in rows:
        kw = r["keywords"]
        if isinstance(kw, str):
            kw = json.loads(kw)
        profiles[r["name"]] = {"keywords": kw, "weight": r["weight"]}
    return profiles


def list_categories(user_id: int, include_doc_counts: bool = True) -> list:
    """
    Return all categories with metadata, optionally including how many
    documents are tagged with each for a specific user — used by the upload page dropdown
    and the admin category manager.
    """
    import json
    import psycopg2.extras
    from services.database_service import _get_connection

    # Safety net — create table if Streamlit's @st.cache_resource skipped
    # initialize_database (same pattern as ensure_excel_table in ingest_excel)
    ensure_category_table()

    sql = """
        SELECT
            c.id, c.name, c.label, c.keywords, c.weight, c.is_custom, c.created_at
            {doc_count_select}
        FROM categories c
        {doc_count_join}
        {group_by}
        ORDER BY c.is_custom ASC, c.label ASC;
    """
    if include_doc_counts:
        sql = sql.format(
            doc_count_select=", COUNT(d.id) AS document_count",
            doc_count_join="LEFT JOIN documents d ON d.category = c.name AND d.user_id = %s",
            group_by="GROUP BY c.id",
        )
    else:
        sql = sql.format(doc_count_select="", doc_count_join="", group_by="")

    with _get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            if include_doc_counts:
                cur.execute(sql, (user_id,))
            else:
                cur.execute(sql)
            rows = [dict(r) for r in cur.fetchall()]

    for r in rows:
        if isinstance(r.get("keywords"), str):
            r["keywords"] = json.loads(r["keywords"])
        if hasattr(r.get("created_at"), "isoformat"):
            r["created_at"] = r["created_at"].isoformat()

    return rows


def create_category(label: str, keywords: list = None, weight: float = 1.0) -> dict:
    """
    Create a new custom category. Used both by the dedicated
    POST /api/categories endpoint and inline during document upload
    when the user types a brand-new category name.

    New categories start with an empty keyword list by default — they
    will never win the routing competition (score 0) until an admin
    adds keywords later. This is the safe default: a custom category's
    documents are still fully searchable via global/soft search, they
    just don't get hard-scoped routing until tuned.

    Returns the created category dict. If a category with the same
    slug already exists, returns the EXISTING one instead of erroring —
    this makes "create category if it doesn't exist yet" idempotent,
    which is exactly what the upload flow needs.
    """
    import json
    import psycopg2.extras
    from services.database_service import _get_connection

    name = _slugify(label)
    keywords = keywords or []

    with _get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT * FROM categories WHERE name = %s;", (name,))
            existing = cur.fetchone()
            if existing:
                logger.info("Category '%s' already exists — reusing it.", name)
                result = dict(existing)
                if isinstance(result.get("keywords"), str):
                    result["keywords"] = json.loads(result["keywords"])
                return result

            cur.execute("""
                INSERT INTO categories (name, label, keywords, weight, is_custom)
                VALUES (%s, %s, %s, %s, TRUE)
                RETURNING id, name, label, keywords, weight, is_custom, created_at;
            """, (name, label.strip(), json.dumps(keywords), weight))
            row = dict(cur.fetchone())

    if isinstance(row.get("keywords"), str):
        row["keywords"] = json.loads(row["keywords"])
    if hasattr(row.get("created_at"), "isoformat"):
        row["created_at"] = row["created_at"].isoformat()

    logger.info("Created custom category '%s' (label='%s').", name, label)
    return row


def update_category_keywords(name: str, keywords: list, weight: float = None) -> dict:
    """
    Admin tuning — add keywords to a custom category so it starts
    participating in smart routing, or adjust an existing category's
    weight/keywords.
    """
    import json
    import psycopg2.extras
    from services.database_service import _get_connection

    with _get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            if weight is not None:
                cur.execute("""
                    UPDATE categories SET keywords = %s, weight = %s
                    WHERE name = %s
                    RETURNING id, name, label, keywords, weight, is_custom;
                """, (json.dumps(keywords), weight, name))
            else:
                cur.execute("""
                    UPDATE categories SET keywords = %s
                    WHERE name = %s
                    RETURNING id, name, label, keywords, weight, is_custom;
                """, (json.dumps(keywords), name))
            row = cur.fetchone()

    if not row:
        raise ValueError(f"Category '{name}' not found.")

    row = dict(row)
    if isinstance(row.get("keywords"), str):
        row["keywords"] = json.loads(row["keywords"])
    return row


def delete_category(name: str) -> bool:
    """
    Delete a custom category. Predefined categories cannot be deleted
    (only edited) to avoid breaking routing for documents that still
    reference them.
    """
    from services.database_service import _get_connection

    with _get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT is_custom FROM categories WHERE name = %s;", (name,))
            row = cur.fetchone()
            if not row:
                return False
            if not row[0]:
                raise ValueError(f"'{name}' is a predefined category and cannot be deleted.")

            cur.execute(
                "UPDATE documents SET category = 'general' WHERE category = %s;", (name,)
            )
            cur.execute("DELETE FROM categories WHERE name = %s RETURNING id;", (name,))
            return cur.fetchone() is not None


def get_category_description(name: str) -> str:
    """Human-readable label for a category name. Falls back to a titleised slug."""
    import psycopg2.extras
    from services.database_service import _get_connection

    with _get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT label FROM categories WHERE name = %s;", (name,))
            row = cur.fetchone()

    if row:
        return row["label"]
    return name.replace("_", " ").title()