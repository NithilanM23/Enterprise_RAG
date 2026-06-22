"""
services/chat_service.py
-------------------------
Persistent chat session management for the Local Employee Knowledge Assistant.

Responsibilities:
  - Create, list, rename, delete chat sessions
  - Store and retrieve messages per session
  - Build conversation history buffer for LLM context
  - Auto-generate session titles from first user message

Schema:
  chat_sessions  — one row per conversation session
  chat_messages  — one row per message (user or assistant)

Design:
  - Sessions and messages persist in PostgreSQL
  - Survives app restarts, browser refreshes, machine reboots
  - CASCADE delete: deleting a session removes all its messages
  - Buffer memory: last HISTORY_WINDOW messages injected into LLM prompt
"""

import logging
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

# History window is now read live from services.settings_service
# ('history_window' key) inside get_history_buffer() — no module constant.


# ---------------------------------------------------------------------------
# Table creation (called from database_service.initialize_database)
# ---------------------------------------------------------------------------

def ensure_chat_tables() -> None:
    """
    Create chat_sessions and chat_messages tables if they do not exist.
    Idempotent — safe to call on every startup.
    """
    from services.database_service import _get_raw_connection

    create_sessions = """
        CREATE TABLE IF NOT EXISTS chat_sessions (
            id         SERIAL PRIMARY KEY,
            title      TEXT        NOT NULL DEFAULT 'New Chat',
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
    """

    create_messages = """
        CREATE TABLE IF NOT EXISTS chat_messages (
            id         SERIAL PRIMARY KEY,
            session_id INTEGER     NOT NULL
                           REFERENCES chat_sessions(id) ON DELETE CASCADE,
            role       TEXT        NOT NULL CHECK (role IN ('user', 'assistant')),
            content    TEXT        NOT NULL,
            sources    JSONB,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
    """

    with _get_raw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(create_sessions)
            cur.execute(create_messages)

    logger.debug("Chat tables ensured: chat_sessions, chat_messages.")


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------

def create_session(user_id: int, title: str = "New Chat") -> dict:
    """
    Create a new chat session and return it.

    Returns:
        dict with id, user_id, title, created_at, updated_at
    """
    import psycopg2.extras
    from services.database_service import _get_connection

    query = """
        INSERT INTO chat_sessions (user_id, title, created_at, updated_at)
        VALUES (%s, %s, NOW(), NOW())
        RETURNING id, user_id, title, created_at, updated_at;
    """
    with _get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(query, (user_id, title))
            row = dict(cur.fetchone())

    logger.info("Created chat session id=%d title='%s' user_id=%d", row["id"], row["title"], row["user_id"])
    return row


def get_all_sessions(user_id: int) -> list:
    """
    Return all sessions ordered by most recently updated first.

    Returns:
        List of dicts: id, user_id, title, created_at, updated_at, message_count
    """
    import psycopg2.extras
    from services.database_service import _get_connection

    query = """
        SELECT
            s.id,
            s.title,
            s.created_at,
            s.updated_at,
            COUNT(m.id) AS message_count
        FROM chat_sessions s
        LEFT JOIN chat_messages m ON m.session_id = s.id
        WHERE s.user_id = %s
        GROUP BY s.id
        ORDER BY s.updated_at DESC;
    """
    with _get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(query, (user_id,))
            return [dict(r) for r in cur.fetchall()]


def get_session(user_id: int, session_id: int) -> dict:
    """Return a single session dict or None if not found."""
    import psycopg2.extras
    from services.database_service import _get_connection

    with _get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT id, user_id, title, created_at, updated_at FROM chat_sessions WHERE id = %s AND user_id = %s;",
                (session_id, user_id)
            )
            row = cur.fetchone()
            return dict(row) if row else None


def rename_session(user_id: int, session_id: int, new_title: str) -> bool:
    """Rename a session. Returns True if found and updated."""
    from services.database_service import _get_connection

    with _get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE chat_sessions SET title = %s, updated_at = NOW() WHERE id = %s AND user_id = %s RETURNING id;",
                (new_title.strip()[:120], session_id, user_id)
            )
            return cur.fetchone() is not None


def delete_session(user_id: int, session_id: int) -> bool:
    """
    Delete a session and all its messages (CASCADE).
    Returns True if a session was deleted.
    """
    from services.database_service import _get_connection

    with _get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM chat_sessions WHERE id = %s AND user_id = %s RETURNING id;",
                (session_id, user_id)
            )
            deleted = cur.fetchone() is not None

    if deleted:
        logger.info("Deleted chat session id=%d and all its messages.", session_id)
    return deleted


def auto_title_session(user_id: int, session_id: int, first_question: str) -> None:
    """
    Set the session title based on the first user question.
    Truncates to 60 characters for clean sidebar display.
    """
    title = first_question.strip()
    if len(title) > 60:
        title = title[:57] + "..."
    rename_session(user_id, session_id, title)


# ---------------------------------------------------------------------------
# Message management
# ---------------------------------------------------------------------------

def add_message(
    session_id: int,
    role: str,
    content: str,
    sources: list = None,
) -> dict:
    """
    Add a message to a session and return the saved message dict.

    Args:
        session_id : Session this message belongs to.
        role       : "user" or "assistant".
        content    : Message text.
        sources    : List of source chunk dicts (for assistant messages only).
                     Stored as JSONB — filename, chunk_number, reranker_score.

    Returns:
        dict with id, session_id, role, content, sources, created_at
    """
    import psycopg2.extras
    from services.database_service import _get_connection

    # Serialise sources to JSON — store only what's needed for display
    sources_json = None
    if sources:
        sources_json = json.dumps([
            {
                "filename":       s.get("filename", ""),
                "chunk_number":   s.get("chunk_number", 0),
                "chunk_text":     s.get("chunk_text", "")[:500],  # cap at 500 chars
                "reranker_score": round(float(s.get("reranker_score", 0)), 4),
                "similarity":     round(float(s.get("similarity", 0)), 4),
            }
            for s in sources
        ])

    insert_query = """
        INSERT INTO chat_messages (session_id, role, content, sources, created_at)
        VALUES (%s, %s, %s, %s, NOW())
        RETURNING id, session_id, role, content, sources, created_at;
    """

    # Touch session updated_at so it floats to top of sidebar list
    touch_query = "UPDATE chat_sessions SET updated_at = NOW() WHERE id = %s;"

    with _get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(insert_query, (session_id, role, content, sources_json))
            msg = dict(cur.fetchone())
            cur.execute(touch_query, (session_id,))

    # Parse sources back from JSONB
    if msg["sources"]:
        msg["sources"] = json.loads(msg["sources"]) if isinstance(msg["sources"], str) else msg["sources"]

    return msg


def get_messages(user_id: int, session_id: int) -> list:
    """
    Return all messages in a session ordered by creation time.

    Returns:
        List of dicts: id, session_id, role, content, sources, created_at
    """
    import psycopg2.extras
    from services.database_service import _get_connection

    with _get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT m.id, m.session_id, m.role, m.content, m.sources, m.created_at
                FROM chat_messages m
                JOIN chat_sessions s ON s.id = m.session_id
                WHERE m.session_id = %s AND s.user_id = %s
                ORDER BY m.created_at ASC;
                """,
                (session_id, user_id)
            )
            rows = cur.fetchall()

    messages = []
    for row in rows:
        msg = dict(row)
        if msg["sources"]:
            msg["sources"] = (
                json.loads(msg["sources"])
                if isinstance(msg["sources"], str)
                else msg["sources"]
            )
        messages.append(msg)

    return messages


def get_history_buffer(session_id: int) -> list:
    """
    Return the last N messages formatted for LLM context injection.
    N comes from the live 'history_window' setting (admin-tunable,
    no restart needed) — never a module-level constant.

    Returns:
        List of dicts: [{"role": "user"/"assistant", "content": str}, ...]
        Most recent N messages, oldest first.
    """
    import psycopg2.extras
    from services.database_service import _get_connection
    from services.settings_service import get_setting

    window = get_setting("history_window")

    with _get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT role, content
                FROM chat_messages
                WHERE session_id = %s
                ORDER BY created_at DESC
                LIMIT %s;
                """,
                (session_id, window)
            )
            rows = cur.fetchall()

    # Reverse to get chronological order (oldest first)
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]


def search_sessions(user_id: int, query: str, limit: int = 20) -> list:
    """
    Full-text search across session titles AND message content.
    Powers the Ctrl+K command palette "jump to conversation" feature.

    Returns sessions ranked by relevance, with a short snippet of the
    matching message content so the frontend can show context in the
    search results dropdown.

    Returns:
        List of dicts: {
            session_id, title, snippet, matched_in ("title"|"message"),
            updated_at
        }
    """
    import psycopg2.extras
    from services.database_service import _get_connection

    if not query or not query.strip():
        return []

    like_pattern = f"%{query.strip()}%"

    sql = """
        WITH title_matches AS (
            SELECT
                s.id AS session_id, s.title, s.updated_at,
                'title' AS matched_in,
                s.title AS snippet
            FROM chat_sessions s
            WHERE s.title ILIKE %(pattern)s AND s.user_id = %(user_id)s
        ),
        message_matches AS (
            SELECT
                s.id AS session_id, s.title, s.updated_at,
                'message' AS matched_in,
                m.content  AS snippet
            FROM chat_messages m
            JOIN chat_sessions s ON s.id = m.session_id
            WHERE m.content ILIKE %(pattern)s AND s.user_id = %(user_id)s
        )
        SELECT DISTINCT ON (session_id)
            session_id, title, updated_at, matched_in, snippet
        FROM (
            SELECT * FROM title_matches
            UNION ALL
            SELECT * FROM message_matches
        ) combined
        ORDER BY session_id, updated_at DESC
        LIMIT %(limit)s;
    """

    with _get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, {"pattern": like_pattern, "limit": limit, "user_id": user_id})
            rows = [dict(r) for r in cur.fetchall()]

    # Trim snippet to a short preview around the match
    for r in rows:
        snippet = r.get("snippet") or ""
        if len(snippet) > 140:
            idx = snippet.lower().find(query.lower())
            start = max(0, idx - 40)
            snippet = ("..." if start > 0 else "") + snippet[start:start + 140] + "..."
        r["snippet"] = snippet

    # Sort overall by recency
    rows.sort(key=lambda r: r["updated_at"], reverse=True)
    return rows[:limit]


def clear_session_messages(session_id: int) -> None:
    """Clear all messages in a session without deleting the session itself."""
    from services.database_service import _get_connection

    with _get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM chat_messages WHERE session_id = %s;",
                (session_id,)
            )
            cur.execute(
                "UPDATE chat_sessions SET title = 'New Chat', updated_at = NOW() WHERE id = %s;",
                (session_id,)
            )

    logger.info("Cleared messages for session id=%d.", session_id)