"""
services/database_service.py
-----------------------------
All PostgreSQL + pgvector interactions for the Local Employee Knowledge Assistant.

Responsibilities:
  - Database initialization (create DB, enable pgvector, apply schema)
  - Document CRUD (insert, list, delete)
  - Embedding storage (bulk insert per document)
  - Semantic retrieval (cosine similarity search via pgvector)
  - Health check

Design principles:
  - Every public method opens and closes its own connection (stateless).
  - No ORM — plain psycopg2 for transparency and minimal dependencies.
  - Embedding dimension is read from config; no hardcoded values here.
  - All SQL uses parameterised queries — no f-string SQL.
"""

import logging
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Optional

import psycopg2
import psycopg2.extras
from psycopg2 import sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DB_CONFIG, EMBEDDING_DIMENSION

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------

def _build_dsn(database: Optional[str] = None) -> str:
    """Build a psycopg2 DSN string from DB_CONFIG."""
    cfg = DB_CONFIG.copy()
    if database is not None:
        cfg["database"] = database
    return (
        f"host={cfg['host']} "
        f"port={cfg['port']} "
        f"dbname={cfg['database']} "
        f"user={cfg['user']} "
        f"password={cfg['password']}"
    )


@contextmanager
def _get_raw_connection(database: Optional[str] = None):
    """
    Plain psycopg2 connection — NO register_vector.

    Used exclusively by initialisation functions (_ensure_pgvector_extension,
    _ensure_tables) that run BEFORE the vector extension exists in the DB.
    Calling register_vector before the extension is installed raises
    "vector type not found in the database", so init functions must use
    this helper instead of _get_connection.
    """
    conn = psycopg2.connect(_build_dsn(database))
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


@contextmanager
def _get_connection(database: Optional[str] = None):
    """
    psycopg2 connection with pgvector type adapter registered.

    Use this for ALL data operations (insert embeddings, similarity search,
    document CRUD) — anywhere a vector column is read or written.

    Registers the pgvector adapter so that:
      - Python list[float]  ->  PostgreSQL vector literal  (on INSERT)
      - PostgreSQL vector   ->  Python list[float]         (on SELECT)

    Never use this during initialisation — the vector extension must already
    exist in the DB before register_vector can succeed.
    """
    from pgvector.psycopg2 import register_vector

    conn = psycopg2.connect(_build_dsn(database))
    try:
        register_vector(conn)   # requires vector extension to already be installed
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def initialize_database() -> None:
    """
    Full one-time setup:
      1. Create the target database if it does not exist.
      2. Enable the pgvector extension.
      3. Create tables (documents + embeddings).

    Safe to call on every application start — all operations are idempotent.
    """
    _ensure_database_exists()
    _ensure_pgvector_extension()
    _ensure_tables()
    _migrate_add_category_column()

    # Chat session tables
    from services.chat_service import ensure_chat_tables
    ensure_chat_tables()

    # Metadata and Excel tables
    from services.metadata_service import ensure_metadata_table
    from services.excel_service import ensure_excel_table
    ensure_metadata_table()
    ensure_excel_table()

    # Settings, Categories, and Pins tables
    from services.settings_service import ensure_settings_table
    from services.category_service import ensure_category_table
    from services.pin_service import ensure_pin_table
    ensure_settings_table()
    ensure_category_table()
    ensure_pin_table()

    logger.info("Database initialisation complete.")


def _ensure_database_exists() -> None:
    """
    Connect to the default 'postgres' database and create our target DB if absent.

    Why we manage the connection manually here (not via _get_connection):
      CREATE DATABASE is one of the few PostgreSQL commands that cannot run
      inside any transaction block. psycopg2's `with conn:` context manager
      always opens a transaction, so we must set AUTOCOMMIT *before* the
      connection is used — which means no context manager for this call.
    """
    target_db = DB_CONFIG["database"]

    conn = psycopg2.connect(_build_dsn("postgres"))
    try:
        # Must be set before any command is issued — disables implicit transactions
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s",
                (target_db,),
            )
            exists = cur.fetchone()
            if not exists:
                cur.execute(
                    sql.SQL("CREATE DATABASE {}").format(
                        sql.Identifier(target_db)
                    )
                )
                logger.info("Created database: %s", target_db)
            else:
                logger.debug("Database already exists: %s", target_db)
    finally:
        conn.close()


def _migrate_add_category_column() -> None:
    """
    Idempotent migration: add category column to documents table if absent.
    Safe to run on existing databases that were created before this column existed.
    """
    with _get_raw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                ALTER TABLE documents
                ADD COLUMN IF NOT EXISTS category TEXT NOT NULL DEFAULT 'general';
            """)
    logger.debug("Migration: category column ensured on documents table.")


def _ensure_pgvector_extension() -> None:
    """
    Enable the pgvector extension in our target database.
    Uses _get_raw_connection because register_vector cannot run before
    the extension is installed — that is exactly what we are doing here.
    """
    with _get_raw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    logger.debug("pgvector extension enabled.")


def _ensure_tables() -> None:
    """Create documents and embeddings tables if they do not exist."""
    create_users = """
        CREATE TABLE IF NOT EXISTS users (
            id            SERIAL PRIMARY KEY,
            username      TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
    """

    create_documents = """
        CREATE TABLE IF NOT EXISTS documents (
            id          SERIAL PRIMARY KEY,
            user_id     INTEGER     NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            filename    TEXT        NOT NULL,
            filepath    TEXT        NOT NULL,
            upload_time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            category    TEXT        NOT NULL DEFAULT 'general',
            CONSTRAINT documents_user_filename_unique UNIQUE (user_id, filename)
        );
    """

    # The vector dimension is read from config — no hardcoding here.
    create_embeddings = f"""
        CREATE TABLE IF NOT EXISTS embeddings (
            id           SERIAL PRIMARY KEY,
            document_id  INTEGER  NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            chunk_number INTEGER  NOT NULL,
            chunk_text   TEXT     NOT NULL,
            embedding    vector({EMBEDDING_DIMENSION}),
            CONSTRAINT embeddings_doc_chunk_unique UNIQUE (document_id, chunk_number)
        );
    """

    # Use _get_raw_connection: tables may include a vector column but we are only
    # running DDL here; register_vector is not needed for CREATE TABLE.
    with _get_raw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(create_users)
            cur.execute(create_documents)
            cur.execute(create_embeddings)
    logger.debug("Tables ensured: users, documents, embeddings.")


# ---------------------------------------------------------------------------
# Document operations
# ---------------------------------------------------------------------------

def insert_document(user_id: int, filename: str, filepath: str, category: str = "general") -> int:
    """
    Insert a document record and return its new ID.

    Args:
        filename : Original filename.
        filepath : Path on disk.
        category : Document category for smart routing.
                   One of: company_info, hr, engineering, finance, reference, general.

    Raises:
        ValueError: if a document with the same filename already exists.
    """
    query = """
        INSERT INTO documents (user_id, filename, filepath, upload_time, category)
        VALUES (%s, %s, %s, NOW(), %s)
        RETURNING id;
    """
    with _get_connection() as conn:
        with conn.cursor() as cur:
            try:
                cur.execute(query, (user_id, filename, filepath, category))
                row = cur.fetchone()
                doc_id = row[0]
                logger.info("Inserted document '%s' with id=%d", filename, doc_id)
                return doc_id
            except psycopg2.errors.UniqueViolation:
                raise ValueError(
                    f"A document named '{filename}' already exists in the database. "
                    "Delete it first before re-uploading."
                )


def get_all_documents(user_id: int) -> list[dict]:
    """
    Return a list of all documents ordered by upload time (newest first).

    Each dict contains: id, filename, filepath, upload_time.
    """
    query = """
        SELECT id, filename, filepath, upload_time, category
        FROM documents
        WHERE user_id = %s
        ORDER BY upload_time DESC;
    """
    with _get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(query, (user_id,))
            rows = cur.fetchall()
            return [dict(r) for r in rows]


def get_document(user_id: int, doc_id: int) -> Optional[dict]:
    """Return a single document dict if it belongs to user."""
    query = "SELECT id, filename, filepath, upload_time, category FROM documents WHERE id = %s AND user_id = %s;"
    with _get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(query, (doc_id, user_id))
            row = cur.fetchone()
            return dict(row) if row else None


def delete_document(user_id: int, doc_id: int) -> bool:
    """
    Delete a document and all its embeddings (CASCADE handles embeddings).
    Returns True if a row was deleted, False if no document with that ID existed.
    """
    with _get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM documents WHERE id = %s AND user_id = %s RETURNING id;", (doc_id, user_id))
            deleted = cur.fetchone()
            if deleted:
                logger.info("Deleted document id=%d and its embeddings.", doc_id)
                rebuild_bm25_index_if_possible()
                return True
            logger.warning("Delete attempted on non-existent document id=%d.", doc_id)
            return False


# ---------------------------------------------------------------------------
# Embedding operations
# ---------------------------------------------------------------------------

def insert_embeddings(document_id: int, chunks: list[dict]) -> None:
    """
    Bulk-insert all chunks for a document.

    Expected chunk format:
        {
            "chunk_number": int,              # 0-based index
            "chunk_text":  str,               # raw text of the chunk
            "embedding":   list[float] | None # None is allowed in Phase 2;
                                              # filled in during Phase 3
        }

    Uses execute_values for efficient bulk insert.
    HNSW index is built only when at least one non-NULL embedding exists.
    """
    if not chunks:
        logger.warning("insert_embeddings called with empty chunks list.")
        return

    query = """
        INSERT INTO embeddings (document_id, chunk_number, chunk_text, embedding)
        VALUES %s
        ON CONFLICT (document_id, chunk_number) DO NOTHING;
    """

    values = [
        (
            document_id,
            c["chunk_number"],
            c["chunk_text"],
            c.get("embedding"),   # None is fine — embedding column is nullable
        )
        for c in chunks
    ]

    # When embedding is None we must use _get_raw_connection to avoid
    # register_vector trying to cast NULL through the vector adapter.
    has_vectors = any(c.get("embedding") is not None for c in chunks)
    ctx = _get_connection if has_vectors else _get_raw_connection

    with ctx() as conn:
        with conn.cursor() as cur:
            psycopg2.extras.execute_values(cur, query, values)
            logger.info(
                "Inserted %d chunks for document_id=%d (embeddings: %s).",
                len(chunks), document_id,
                "yes" if has_vectors else "deferred to Phase 3",
            )

    if has_vectors:
        _ensure_hnsw_index()
    # BM25 rebuild is NOT called here — it is called by the app layer
    # (do_ingest / generate_embeddings) after all operations complete.
    # This keeps insert_embeddings focused and avoids masking insert errors.

def _ensure_hnsw_index() -> None:
    """
    Create an HNSW index on the embedding column if it does not already exist.

    Why HNSW over IVFFlat (pgvector 0.8.2+):
      - Works at ANY dataset size — no minimum row count.
      - No 'lists' tuning parameter required.
      - Better recall at equivalent query speed.
      - Index is maintained incrementally on INSERT — no rebuild needed.

    HNSW parameters used:
      m              = 16   (connections per layer; 16 is the recommended default)
      ef_construction = 64  (build-time accuracy; higher = better index, slower build)

    For CPU-only use these defaults are well-balanced.
    Increase ef_construction to 128 for higher recall if build time is acceptable.
    """
    with _get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE INDEX IF NOT EXISTS embeddings_hnsw_idx
                ON embeddings
                USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64);
            """)
    logger.debug("HNSW index ensured on embeddings.embedding.")


def get_chunks_for_document(doc_id: int) -> list[dict]:
    """Return all chunks (without embeddings) for a given document, ordered by chunk_number."""
    query = """
        SELECT chunk_number, chunk_text
        FROM embeddings
        WHERE document_id = %s
        ORDER BY chunk_number;
    """
    # _get_raw_connection is sufficient — we are not reading the vector column.
    with _get_raw_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(query, (doc_id,))
            return [dict(r) for r in cur.fetchall()]


# ---------------------------------------------------------------------------
# Semantic retrieval
# ---------------------------------------------------------------------------

def search_similar_chunks(
    query_embedding: list,
    top_k: int = 5,
    document_ids: list = None,
) -> list:
    """
    Find the top_k most semantically similar chunks using cosine similarity.

    Args:
        query_embedding : The query vector.
        top_k           : Number of results to return.
        document_ids    : Optional list of document IDs to scope the search.
                          If None, searches across ALL documents.
                          Pass a list of IDs to restrict to specific documents.
    """
    if document_ids:
        query = """
            SELECT e.id, e.chunk_text, e.chunk_number, e.document_id,
                   d.filename,
                   1 - (e.embedding <=> %s::vector) AS similarity
            FROM embeddings e
            JOIN documents d ON d.id = e.document_id
            WHERE e.document_id = ANY(%s)
            ORDER BY e.embedding <=> %s::vector
            LIMIT %s;
        """
        params = (query_embedding, document_ids, query_embedding, top_k)
    else:
        query = """
            SELECT e.id, e.chunk_text, e.chunk_number, e.document_id,
                   d.filename,
                   1 - (e.embedding <=> %s::vector) AS similarity
            FROM embeddings e
            JOIN documents d ON d.id = e.document_id
            ORDER BY e.embedding <=> %s::vector
            LIMIT %s;
        """
        params = (query_embedding, query_embedding, top_k)

    with _get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(query, params)
            rows = cur.fetchall()
            return [dict(r) for r in rows]

# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

def health_check() -> dict:
    """
    Verify database connectivity and return status details.

    Returns a dict with keys:
        connected      : bool
        pgvector_ready : bool
        document_count : int  (or None on failure)
        error          : str  (or None on success)
    """
    result = {
        "connected":      False,
        "pgvector_ready": False,
        "document_count": None,
        "error":          None,
    }
    try:
        with _get_connection() as conn:
            with conn.cursor() as cur:
                # Basic connectivity
                cur.execute("SELECT 1;")
                result["connected"] = True

                # Check pgvector is installed
                cur.execute(
                    "SELECT COUNT(*) FROM pg_extension WHERE extname = 'vector';"
                )
                ext_count = cur.fetchone()[0]
                result["pgvector_ready"] = ext_count > 0

                # Document count (also verifies tables exist)
                cur.execute("SELECT COUNT(*) FROM documents;")
                result["document_count"] = cur.fetchone()[0]

    except Exception as exc:
        result["error"] = str(exc)
        logger.error("Health check failed: %s", exc)

    return result


# ---------------------------------------------------------------------------
# BM25 index auto-rebuild helper
# ---------------------------------------------------------------------------

def rebuild_bm25_index_if_possible() -> None:
    """
    Silently rebuild the BM25 index after any document set change.

    Called automatically after:
      - insert_embeddings()  (new document added)
      - delete_document()    (document removed)

    Failures are logged as warnings but never raised — BM25 is an
    enhancement. If it fails, semantic search still works fine.
    """
    try:
        from services.bm25_service import build_index, index_exists
        import psycopg2
        from services.database_service import _get_connection

        # Only rebuild if there are embedded chunks to index
        with _get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) FROM embeddings WHERE embedding IS NOT NULL;"
                )
                count = cur.fetchone()[0]

        if count == 0:
            logger.debug(
                "BM25 rebuild skipped — no embedded chunks in DB yet."
            )
            return

        info = build_index()
        logger.info(
            "BM25 index auto-rebuilt: %d chunks indexed.", info["chunk_count"]
        )

    except Exception as exc:
        logger.warning(
            "BM25 auto-rebuild failed (non-fatal — semantic search unaffected): %s",
            exc,
        )