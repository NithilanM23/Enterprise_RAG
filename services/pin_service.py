"""
services/pin_service.py
------------------------
Pinned / saved answers — a personal reference library for each user's
best answers. Directly supports the "Answer Pinning" retention feature
from the frontend design: users save answers they want to find again
without re-asking the question.

Schema:
  pinned_answers
    id, message_id, note, created_at

Design notes:
  - message_id references chat_messages.id — the saved answer is always
    tied back to the original Q&A pair and its sources.
  - note is an optional personal annotation the user can add when pinning.
  - No user_id column yet (single-tenant POC). When auth is added, add
    a user_id column and filter all queries by it.
"""

import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)


def ensure_pin_table() -> None:
    """Create pinned_answers table if it does not exist. Idempotent."""
    from services.database_service import _get_raw_connection

    with _get_raw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS pinned_answers (
                    id          SERIAL PRIMARY KEY,
                    message_id  INTEGER NOT NULL
                                    REFERENCES chat_messages(id) ON DELETE CASCADE,
                    note        TEXT,
                    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    CONSTRAINT pinned_answers_message_unique UNIQUE (message_id)
                );
            """)
    logger.debug("pinned_answers table ensured.")


def pin_answer(message_id: int, note: str = None) -> dict:
    """
    Pin a message (must be an assistant message) for later reference.
    Returns the pinned record, or raises if message_id doesn't exist
    or is already pinned.
    """
    import psycopg2.extras
    from services.database_service import _get_connection

    with _get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT id, role FROM chat_messages WHERE id = %s;",
                (message_id,)
            )
            msg = cur.fetchone()
            if not msg:
                raise ValueError(f"Message id={message_id} not found.")

            cur.execute("""
                INSERT INTO pinned_answers (message_id, note, created_at)
                VALUES (%s, %s, NOW())
                ON CONFLICT (message_id) DO UPDATE SET note = EXCLUDED.note
                RETURNING id, message_id, note, created_at;
            """, (message_id, note))
            return dict(cur.fetchone())


def unpin_answer(message_id: int) -> bool:
    """Remove a pin. Returns True if a pin was removed."""
    from services.database_service import _get_connection

    with _get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM pinned_answers WHERE message_id = %s RETURNING id;",
                (message_id,)
            )
            return cur.fetchone() is not None


def get_pinned_answers(user_id: int) -> list:
    """
    Return all pinned answers joined with their original message,
    sources, and parent session — newest pin first.
    """
    import json
    import psycopg2.extras
    from services.database_service import _get_connection

    with _get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT
                    p.id            AS pin_id,
                    p.note,
                    p.created_at    AS pinned_at,
                    m.id            AS message_id,
                    m.content,
                    m.sources,
                    m.created_at    AS message_created_at,
                    s.id            AS session_id,
                    s.title         AS session_title
                FROM pinned_answers p
                JOIN chat_messages  m ON m.id = p.message_id
                JOIN chat_sessions  s ON s.id = m.session_id
                WHERE s.user_id = %s
                ORDER BY p.created_at DESC;
            """, (user_id,))
            rows = [dict(r) for r in cur.fetchall()]

    for r in rows:
        if isinstance(r.get("sources"), str):
            r["sources"] = json.loads(r["sources"])

    return rows


def is_pinned(message_id: int) -> bool:
    """Quick check used when rendering chat history."""
    from services.database_service import _get_connection

    with _get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM pinned_answers WHERE message_id = %s;",
                (message_id,)
            )
            return cur.fetchone() is not None
