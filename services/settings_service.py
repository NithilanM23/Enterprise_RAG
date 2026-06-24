"""
services/settings_service.py
------------------------------
Single source of truth for every swappable runtime parameter — LLM model,
embedding model, reranker model, chunk size/overlap, retrieval pool sizes,
MMR lambda, history window, etc.

Why this exists:
  Previously these were hardcoded module-level constants (CHUNK_SIZE in
  config.py, MMR_LAMBDA in retrieval_service.py, RERANKER_MODEL in
  reranker_service.py...). Changing any of them meant editing code and
  restarting the server. That's not "admin-swappable."

How it works:
  - app_settings table stores key/value overrides in PostgreSQL.
  - get_setting(key) returns the DB value if present, else the hardcoded
    default from config.py / each service's own DEFAULT.
  - Every service reads settings INSIDE its functions (call-time), never
    as a module-level constant captured once at import time. This is the
    one rule that makes live admin swapping actually work — no restart
    needed for non-structural settings.
  - Values are stored as plain text and cast to the right type on read.

Two categories of settings:
  SAFE   — take effect immediately on next request (chunk_size, top_k,
           mmr_lambda, history_window, llm_model, num_predict, ...)
  UNSAFE — embedding_model / embedding_dimension. Changing these makes
           every existing embedding meaningless (different vector space).
           Handled by a dedicated function that nulls old embeddings and
           migrates the pgvector column type — never silently swapped.
"""

import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults — used when no DB override exists yet (first run, or after a
# settings table wipe). Pulled from the same values config.py originally had.
# ---------------------------------------------------------------------------

_DEFAULTS = {
    "llm_model":           ("llama3.2:latest",                         str),
    "embedding_model":     ("mxbai-embed-large:latest",                  str),
    "embedding_dimension": (1024,                                 int),
    "reranker_model":      ("cross-encoder/ms-marco-MiniLM-L-6-v2", str),
    "chunk_size":           (1000,                                int),
    "chunk_overlap":        (200,                                 int),
    "top_k":                (3,                                   int),
    "semantic_k":           (20,                                  int),
    "bm25_k":               (20,                                  int),
    "mmr_pool":             (20,                                  int),
    "mmr_lambda":           (0.85,                                float),
    "rrf_k":                (60,                                  int),
    "history_window":       (3,                                   int),
    "num_predict":          (1024,                                int),
    "temperature":          (0.1,                                 float),
    "routing_confidence_threshold": (1.0,                         float),
}

# Settings that require a special migration path, not a plain value swap.
UNSAFE_KEYS = {"embedding_model", "embedding_dimension"}


# ---------------------------------------------------------------------------
# Table creation
# ---------------------------------------------------------------------------

def ensure_settings_table() -> None:
    """Create app_settings table if it does not exist. Idempotent."""
    from services.database_service import _get_raw_connection

    with _get_raw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS app_settings (
                    user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    key        TEXT NOT NULL,
                    value      TEXT NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    PRIMARY KEY (user_id, key)
                );
            """)
    logger.debug("app_settings table ensured.")


# ---------------------------------------------------------------------------
# Core get/set
# ---------------------------------------------------------------------------

def get_setting(key: str, user_id: int = None):
    """
    Return the effective value for a setting — DB override if present for the user,
    otherwise the hardcoded default. Always returns the correctly typed
    value (int/float/str), never a raw string.
    """
    if key not in _DEFAULTS:
        raise KeyError(f"Unknown setting: '{key}'")

    default_value, cast_fn = _DEFAULTS[key]
    
    if user_id is None:
        user_id = 1

    from services.database_service import _get_raw_connection
    try:
        with _get_raw_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT value FROM app_settings WHERE key = %s AND user_id = %s;", (key, user_id))
                row = cur.fetchone()
        if row is not None:
            return cast_fn(row[0])
    except Exception as exc:
        logger.warning("Settings lookup failed for '%s', using default: %s", key, exc)

    return default_value


def get_all_settings(user_id: int) -> dict:
    """Return every setting's effective value as a flat dict."""
    return {key: get_setting(key, user_id) for key in _DEFAULTS}


def set_setting(user_id: int, key: str, value) -> dict:
    """
    Persist a setting override for a user. Refuses UNSAFE_KEYS.

    Returns the stored {key, value} on success.
    """
    if key not in _DEFAULTS:
        raise KeyError(f"Unknown setting: '{key}'")
    if key in UNSAFE_KEYS:
        raise ValueError(
            f"'{key}' cannot be set directly — use update_embedding_model() "
            f"which safely migrates existing embeddings."
        )

    from services.database_service import _get_raw_connection

    with _get_raw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO app_settings (user_id, key, value, updated_at)
                VALUES (%s, %s, %s, NOW())
                ON CONFLICT (user_id, key) DO UPDATE SET value = EXCLUDED.value, updated_at = NOW();
            """, (user_id, key, str(value)))

    logger.info("Setting updated: %s = %s", key, value)
    return {"key": key, "value": value}


def reset_setting(user_id: int, key: str) -> None:
    """Remove a DB override, reverting to the hardcoded default."""
    from services.database_service import _get_raw_connection
    with _get_raw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM app_settings WHERE key = %s AND user_id = %s;", (key, user_id))


# ---------------------------------------------------------------------------
# LLM model swap — safe, but should verify the model is pulled in Ollama
# ---------------------------------------------------------------------------

def update_llm_model(user_id: int, model_name: str) -> dict:
    """
    Swap the LLM model. Safe to apply immediately — no stored data depends
    on the LLM model, only generation quality/speed changes.

    Validates the model exists in Ollama's local model list first so the
    admin gets an immediate clear error instead of every chat request
    failing afterward.
    """
    available = list_ollama_models()
    if model_name not in available:
        raise ValueError(
            f"Model '{model_name}' is not pulled in Ollama. "
            f"Run: ollama pull {model_name}\n"
            f"Available models: {', '.join(available) if available else '(none found)'}"
        )
    return set_setting(user_id, "llm_model", model_name)


def update_reranker_model(user_id: int, model_name: str) -> dict:
    """
    Swap the reranker model. Safe in the sense that nothing in the DB
    depends on it, but the new HuggingFace model must download on first
    use — done by reranker_service when it's next called, not here.

    Resets the in-memory cached model so the new one loads on next request.
    """
    result = set_setting(user_id, "reranker_model", model_name)
    from services.reranker_service import reset_reranker_cache
    reset_reranker_cache()
    return result


# ---------------------------------------------------------------------------
# Embedding model swap — UNSAFE, requires migration
# ---------------------------------------------------------------------------

def preview_embedding_model_change(new_model: str, new_dimension: int) -> dict:
    """
    Dry-run — shows what changing the embedding model WOULD do, without
    applying anything. The admin UI calls this first, shows the warning,
    and only calls update_embedding_model(confirm=True) if the admin
    explicitly confirms.
    """
    from services.database_service import _get_raw_connection

    current_model     = get_setting("embedding_model")
    current_dimension = get_setting("embedding_dimension")

    with _get_raw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM embeddings WHERE embedding IS NOT NULL;")
            embedded_count = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM excel_rows;")
            excel_row_count = cur.fetchone()[0]

    return {
        "current_model":      current_model,
        "current_dimension":  current_dimension,
        "new_model":           new_model,
        "new_dimension":       new_dimension,
        "dimension_changing":  new_dimension != current_dimension,
        "chunks_to_reembed":   embedded_count,
        "excel_rows_unaffected": excel_row_count,
        "warning": (
            f"This will change the default embedding model for all NEW documents. "
            f"Your existing {embedded_count} chunks will remain searchable using '{current_model}'. "
            f"During search, the system will seamlessly query both models and merge the results."
        ),
    }


def update_embedding_model(new_model: str, new_dimension: int, confirm: bool = False) -> dict:
    """
    Apply an embedding model swap.
    This now preserves old embeddings and merely updates the settings so that
    new documents will be ingested using the new model.
    """
    if not confirm:
        raise ValueError(
            "Call preview_embedding_model_change() first and pass confirm=True "
            "after the admin has seen the warning."
        )

    current_dimension = get_setting("embedding_dimension")
    dimension_changing = new_dimension != current_dimension

    # Step 3 — persist new settings (bypass set_setting's UNSAFE_KEYS guard)
    from services.database_service import _get_raw_connection as _raw
    with _raw() as conn:
        with conn.cursor() as cur:
            for key, value in [("embedding_model", new_model), ("embedding_dimension", new_dimension)]:
                cur.execute("""
                    INSERT INTO app_settings (user_id, key, value, updated_at)
                    VALUES (%s, %s, %s, NOW())
                    ON CONFLICT (user_id, key) DO UPDATE SET value = EXCLUDED.value, updated_at = NOW();
                """, (1, key, str(value)))  # Hardcode user_id=1 for structural swaps for now

    logger.info(
        "Embedding model changed: model='%s' dimension=%d. Old embeddings retained.", 
        new_model, new_dimension
    )

    return {
        "applied":            True,
        "new_model":          new_model,
        "new_dimension":      new_dimension,
        "dimension_changed":  dimension_changing,
        "embeddings_nulled":  0,
        "next_step":          "New documents will now use the new embedding model.",
    }


# ---------------------------------------------------------------------------
# Ollama model discovery
# ---------------------------------------------------------------------------

def list_ollama_models() -> list:
    """
    Query Ollama's local model list (GET /api/tags). Used to validate
    LLM/embedding model swaps and to populate the admin UI's dropdown.
    Returns an empty list (never raises) if Ollama is unreachable.
    """
    import requests
    from config import OLLAMA_BASE_URL

    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        resp.raise_for_status()
        data = resp.json()
        return [m["name"] for m in data.get("models", [])]
    except Exception as exc:
        logger.warning("Could not list Ollama models: %s", exc)
        return []
