"""
api.py
------
FastAPI backend for the Local Employee Knowledge Assistant.

Designed for 30+ concurrent users:
  - All endpoints are async — non-blocking I/O throughout
  - LLM answers stream via Server-Sent Events (SSE)
  - Embedding and retrieval run in thread pools (CPU-bound work)
  - Ollama parallelism handled server-side (OLLAMA_NUM_PARALLEL=3)

Run with:
    uvicorn api:app --host 0.0.0.0 --port 8000 --workers 4

Swagger UI:
    http://localhost:8000/docs
"""

import asyncio
import json
import logging
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional

from fastapi import (
    FastAPI, File, Form, HTTPException, UploadFile,
    BackgroundTasks, Depends
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from fastapi.security import OAuth2PasswordRequestForm
from services.auth_service import get_current_user, create_access_token, verify_password, get_password_hash, get_user_by_username, create_user

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Thread pool for CPU-bound work (embedding, reranking, BM25)
# 4 workers = 4 parallel CPU tasks without blocking the async loop
_thread_pool = ThreadPoolExecutor(max_workers=4)

# Concurrency guard for Ollama LLM calls. This mirrors OLLAMA_NUM_PARALLEL
# on the API side so FastAPI itself is aware of how many generations are
# in flight — used to expose live queue depth via /api/admin/queue and to
# avoid piling up more concurrent httpx streams than Ollama can actually
# service in parallel.
OLLAMA_MAX_PARALLEL = int(os.getenv("OLLAMA_MAX_PARALLEL", "4"))
_ollama_semaphore = asyncio.Semaphore(OLLAMA_MAX_PARALLEL)
_queue_state = {"active": 0, "queued": 0}

# ---------------------------------------------------------------------------
# App initialisation
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Local Employee Knowledge Assistant",
    description="Private RAG system for internal document Q&A",
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# Auth Routes
# ---------------------------------------------------------------------------

class UserCreate(BaseModel):
    username: str
    password: str

@app.post("/api/auth/register", tags=["Auth"])
async def register(user: UserCreate):
    db_user = await run_sync(lambda: get_user_by_username(user.username))
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    hashed_password = get_password_hash(user.password)
    new_user = await run_sync(lambda: create_user(user.username, hashed_password))
    return {"id": new_user["id"], "username": new_user["username"]}

@app.post("/api/auth/login", tags=["Auth"])
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = await run_sync(lambda: get_user_by_username(form_data.username))
    if not user or not verify_password(form_data.password, user["password_hash"]):
        from fastapi import status
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(data={"sub": str(user["id"])})
    return {"access_token": access_token, "token_type": "bearer", "user_id": user["id"], "username": user["username"]}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten to your Next.js origin in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    """Initialise DB tables on startup."""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        _thread_pool,
        _init_db
    )
    logger.info("Database initialised.")


def _init_db():
    from services.database_service import initialize_database
    initialize_database()


# ---------------------------------------------------------------------------
# Helper: run sync code in thread pool without blocking async loop
# ---------------------------------------------------------------------------

async def run_sync(fn, *args):
    """Execute a blocking function in the thread pool."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_thread_pool, fn, *args)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    session_id:   int
    question:     str
    document_ids: Optional[List[int]] = None


class SessionCreate(BaseModel):
    title: Optional[str] = "New Chat"


class SessionRename(BaseModel):
    title: str


class CategoryUpdate(BaseModel):
    category: str


class ExplorerQuery(BaseModel):
    query:    str
    filename: str      # identifies which uploaded file to query


class PinRequest(BaseModel):
    note: Optional[str] = None


class CategoryCreate(BaseModel):
    label:    str
    keywords: Optional[List[str]] = None
    weight:   Optional[float] = 1.0


class CategoryKeywordsUpdate(BaseModel):
    keywords: List[str]
    weight:   Optional[float] = None


class SettingsUpdate(BaseModel):
    """
    Bulk update for SAFE settings only (chunk_size, top_k, mmr_lambda, etc).
    Any subset of fields may be provided — only those are changed.
    Model swaps (llm_model, reranker_model, embedding_model) go through
    their own dedicated endpoints because they need validation/migration.
    """
    chunk_size:                    Optional[int]   = None
    chunk_overlap:                 Optional[int]   = None
    top_k:                         Optional[int]   = None
    semantic_k:                    Optional[int]   = None
    bm25_k:                        Optional[int]   = None
    mmr_pool:                      Optional[int]   = None
    mmr_lambda:                    Optional[float] = None
    rrf_k:                         Optional[int]   = None
    history_window:                Optional[int]   = None
    num_predict:                   Optional[int]   = None
    temperature:                   Optional[float] = None
    routing_confidence_threshold:  Optional[float] = None


class ModelSwapRequest(BaseModel):
    model: str


class EmbeddingModelSwapRequest(BaseModel):
    model:     str
    dimension: int
    confirm:   bool = False


# ---------------------------------------------------------------------------
# 1. Health & Status
# ---------------------------------------------------------------------------

@app.get("/api/health", tags=["Status"])
async def health():
    """
    Returns DB connectivity, pgvector status, Ollama status,
    and embedding readiness. Used by the frontend status bar.
    """
    def _check():
        from services.database_service import health_check
        from services.embedding_service import check_ollama_connection
        db     = health_check()
        ollama = check_ollama_connection()
        return {"db": db, "ollama": ollama}

    result = await run_sync(_check)
    return result


@app.get("/api/status/embeddings", tags=["Status"])
async def embedding_status(user: dict = Depends(get_current_user)):
    """Per-document embedding completion status."""
    import psycopg2.extras
    from services.database_service import _get_connection

    def _status():
        with _get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("""
                    SELECT d.id, d.filename, d.category,
                           COUNT(e.id)            AS total_chunks,
                           COUNT(e.embedding)     AS embedded,
                           COUNT(e.id) - COUNT(e.embedding) AS pending
                    FROM documents d
                    LEFT JOIN embeddings e ON e.document_id = d.id
                    WHERE d.user_id = %s
                    GROUP BY d.id
                    ORDER BY d.id;
                """, (user["id"],))
                return [dict(r) for r in cur.fetchall()]

    return await run_sync(_status)


@app.get("/api/status/knowledge-health", tags=["Status"])
async def knowledge_health(user: dict = Depends(get_current_user)):
    """
    Per-category embedding completeness — powers the dashboard's
    'Knowledge Health' widget (progress bar per category).

    Response: [
        { category: "hr", total_chunks: 340, embedded: 340, percent: 100 },
        { category: "finance", total_chunks: 120, embedded: 48, percent: 40 },
        ...
    ]
    Excel-only documents (no chunks) are reported separately with row counts.
    """
    import psycopg2.extras
    from services.database_service import _get_connection

    def _health():
        with _get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("""
                    SELECT
                        d.category,
                        COUNT(DISTINCT d.id)              AS document_count,
                        COUNT(e.id)                        AS total_chunks,
                        COUNT(e.embedding)                 AS embedded
                    FROM documents d
                    LEFT JOIN embeddings e ON e.document_id = d.id
                    WHERE d.user_id = %s
                    GROUP BY d.category
                    ORDER BY d.category;
                """, (user["id"],))
                rows = [dict(r) for r in cur.fetchall()]

        for r in rows:
            total = r["total_chunks"] or 0
            r["percent"] = round((r["embedded"] / total) * 100, 1) if total else 100.0
        return rows

    return await run_sync(_health)


# ---------------------------------------------------------------------------
# 2. Documents & Ingestion
# ---------------------------------------------------------------------------

@app.get("/api/documents", tags=["Documents"])
async def list_documents(user: dict = Depends(get_current_user)):
    """List all ingested documents with metadata."""
    def _list():
        from services.database_service import get_all_documents
        return get_all_documents(user["id"])

    docs = await run_sync(_list)
    # Serialise datetime
    for d in docs:
        if hasattr(d.get("upload_time"), "isoformat"):
            d["upload_time"] = d["upload_time"].isoformat()
    return docs


@app.post("/api/documents/upload", tags=["Documents"])
async def upload_document(
    background_tasks: BackgroundTasks,
    file:               UploadFile = File(...),
    category:           str        = Form("general"),
    new_category_label: Optional[str] = Form(None),
    user: dict = Depends(get_current_user),
):
    """
    Upload and ingest a document.
    File is saved immediately; chunking + embedding runs as a background task
    so the endpoint returns quickly (~200ms) and the user sees progress via
    GET /api/documents/{doc_id}/status.

    Category handling — two ways to call this:
      1. category="hr"  → use an existing category as-is.
      2. new_category_label="Quality Control" → creates a brand-new
         custom category on the fly (slugified to "quality_control") and
         uses it for this upload. This is what the frontend's upload
         page "+ Create new category" option calls. The new category
         starts with no keyword profile, so it's immediately usable for
         document scoping/filtering but won't participate in automatic
         smart-routing until an admin tunes its keywords later via
         PATCH /api/categories/{name}/keywords.
    """
    from config import UPLOAD_DIR, SUPPORTED_EXTENSIONS, EXCEL_EXTENSIONS

    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS and ext not in EXCEL_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}"
        )

    user_dir = UPLOAD_DIR / str(user["id"])
    user_dir.mkdir(parents=True, exist_ok=True)
    dest = user_dir / file.filename
    if dest.exists():
        raise HTTPException(
            status_code=409,
            detail=f"'{file.filename}' already exists in your account. Delete it first."
        )

    # Resolve the category to use — create a new custom one if requested
    resolved_category = category
    if new_category_label and new_category_label.strip():
        def _create_cat():
            from services.category_service import create_category
            return create_category(new_category_label.strip(), user_id=user["id"])
        new_cat = await run_sync(_create_cat)
        resolved_category = new_cat["name"]

    # Save file
    raw = await file.read()
    dest.write_bytes(raw)

    # Kick off ingestion as background task — returns immediately
    background_tasks.add_task(_ingest_background, raw, dest, file.filename, resolved_category, ext, user["id"])

    return {
        "status":   "ingesting",
        "filename": file.filename,
        "category": resolved_category,
        "message":  "File received. Ingestion running in background.",
    }


def _ingest_background(raw: bytes, dest: Path, filename: str, category: str, ext: str, user_id: int):
    """Background ingestion — runs in thread pool, not blocking event loop."""
    import io as _io
    import gc, time
    from services.database_service import insert_document, insert_embeddings, delete_document
    from services.metadata_service  import extract_metadata
    from config import EXCEL_EXTENSIONS

    doc_id = None
    try:
        doc_id = insert_document(
            filename=filename,
            filepath=str(dest.resolve()),
            category=category,
            user_id=user_id,
        )

        file_bytes = _io.BytesIO(raw)

        if ext in EXCEL_EXTENSIONS:
            from services.excel_service import ingest_excel
            try:
                extract_metadata(str(dest.resolve()), doc_id)
            except Exception:
                pass
            file_bytes.seek(0)
            ingest_excel(file_bytes, doc_id, filename=filename)

        else:
            try:
                extract_metadata(str(dest.resolve()), doc_id)
            except Exception:
                pass
            from services.loader   import load_document
            from services.chunker  import chunk_text
            text   = load_document(str(dest.resolve()))
            chunks = chunk_text(text)
            insert_embeddings(document_id=doc_id, chunks=chunks)
            try:
                from services.bm25_service import build_index
                build_index()
            except Exception:
                pass

            _embed_background(user_id)

        logger.info("Background ingestion complete: %s", filename)

    except Exception as exc:
        logger.error("Background ingestion failed for %s: %s", filename, exc)
        try:
            gc.collect()
            time.sleep(0.3)
            dest.unlink(missing_ok=True)
        except Exception:
            pass
        if doc_id:
            try:
                delete_document(doc_id)
            except Exception:
                pass


@app.get("/api/documents/{doc_id}", tags=["Documents"])
async def get_document(doc_id: int, user: dict = Depends(get_current_user)):
    def _get():
        from services.database_service import get_document_by_id
        return get_document_by_id(doc_id, user["id"])

    doc = await run_sync(_get)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    if hasattr(doc.get("upload_time"), "isoformat"):
        doc["upload_time"] = doc["upload_time"].isoformat()
    return doc


@app.get("/api/documents/{doc_id}/status", tags=["Documents"])
async def document_status(doc_id: int, user: dict = Depends(get_current_user)):
    """
    Ingestion progress for a single document. The frontend polls this
    every ~3 seconds right after upload to drive the progress card
    ("Saving → Chunking → Embedding → Ready").

    Status is computed from existing tables — no new schema needed:
      - No chunk rows yet           → "chunking"
      - Chunk rows exist, some/all
        embeddings still NULL       → "embedding"
      - All chunk rows embedded     → "ready"
      - Excel rows exist instead    → "ready" (Excel skips embedding)
      - Document row missing        → 404
    """
    import psycopg2.extras
    from services.database_service import _get_connection

    def _status():
        with _get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("SELECT id, filename FROM documents WHERE id = %s AND user_id = %s;", (doc_id, user["id"]))
                doc = cur.fetchone()
                if not doc:
                    return None

                cur.execute(
                    "SELECT COUNT(*) AS total, COUNT(embedding) AS embedded "
                    "FROM embeddings WHERE document_id = %s;",
                    (doc_id,)
                )
                chunk_row = cur.fetchone()

                cur.execute(
                    "SELECT COUNT(*) AS rows FROM excel_rows WHERE document_id = %s;",
                    (doc_id,)
                )
                excel_row = cur.fetchone()

        total      = chunk_row["total"]
        embedded   = chunk_row["embedded"]
        excel_rows = excel_row["rows"]

        if excel_rows > 0:
            stage, percent = "ready", 100.0
        elif total == 0:
            stage, percent = "chunking", 0.0
        elif embedded < total:
            stage, percent = "embedding", round((embedded / total) * 100, 1)
        else:
            stage, percent = "ready", 100.0

        return {
            "doc_id":           doc_id,
            "filename":         doc["filename"],
            "stage":            stage,
            "chunks_total":     total,
            "chunks_embedded":  embedded,
            "excel_rows":       excel_rows,
            "percent_complete": percent,
            "error":            None,
        }

    result = await run_sync(_status)
    if result is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return result


@app.delete("/api/documents/{doc_id}", tags=["Documents"])
async def delete_document(doc_id: int, user: dict = Depends(get_current_user)):
    def _delete():
        from services.database_service import delete_document, get_document_by_id
        from config import UPLOAD_DIR
        doc = get_document_by_id(doc_id, user["id"])
        if not doc:
            return None
        filepath = doc.get("filepath")
        ok = delete_document(doc_id, user["id"])
        if ok and filepath:
            Path(filepath).unlink(missing_ok=True)
        return ok

    result = await run_sync(_delete)
    if result is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"deleted": result, "doc_id": doc_id}


@app.patch("/api/documents/{doc_id}/category", tags=["Documents"])
async def update_category(doc_id: int, body: CategoryUpdate, user: dict = Depends(get_current_user)):
    def _update():
        from services.database_service import _get_raw_connection
        with _get_raw_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE documents SET category=%s WHERE id=%s AND user_id=%s RETURNING id;",
                    (body.category, doc_id, user["id"])
                )
                return cur.fetchone()

    row = await run_sync(_update)
    if not row:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"doc_id": doc_id, "category": body.category}


@app.post("/api/documents/embed", tags=["Documents"])
async def generate_embeddings(background_tasks: BackgroundTasks, user: dict = Depends(get_current_user)):
    """Generate embeddings for all pending chunks. Runs in background."""
    background_tasks.add_task(_embed_background, user["id"])
    return {"status": "started", "message": "Embedding generation running in background."}


def _embed_background(user_id: int):
    import psycopg2.extras
    from services.embedding_service  import embed_chunks
    from services.database_service   import _get_connection

    with _get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT e.id, e.chunk_text FROM embeddings e
                JOIN documents d ON d.id = e.document_id
                WHERE e.embedding IS NULL AND d.user_id = %s
                ORDER BY e.document_id, e.chunk_number;
            """, (user_id,))
            rows = [dict(r) for r in cur.fetchall()]

    if not rows:
        return

    chunks = [{"id": r["id"], "chunk_text": r["chunk_text"], "embedding": None} for r in rows]
    embed_chunks(chunks)

    with _get_connection() as conn:
        with conn.cursor() as cur:
            for c in chunks:
                cur.execute(
                    "UPDATE embeddings SET embedding=%s WHERE id=%s;",
                    (c["embedding"], c["id"])
                )

    try:
        from services.bm25_service import build_index
        build_index()
    except Exception:
        pass

    logger.info("Background embedding complete: %d chunks.", len(chunks))


# ---------------------------------------------------------------------------
# 3. Categories
# ---------------------------------------------------------------------------

@app.get("/api/categories", tags=["Categories"])
async def list_categories(user: dict = Depends(get_current_user)):
    """
    All categories (predefined + custom) with document counts.
    Powers the category dropdown on the upload page and the admin
    category manager.
    """
    def _list():
        from services.category_service import list_categories
        return list_categories(user["id"], include_doc_counts=True)
    return await run_sync(_list)


@app.post("/api/categories", tags=["Categories"])
async def create_category(body: CategoryCreate, user: dict = Depends(get_current_user)):
    """
    Create a new custom category explicitly (separate from the inline
    creation available via POST /api/documents/upload's
    new_category_label field). Useful for setting up categories with
    a tuned keyword profile up front via the admin panel, before any
    document uses them.
    """
    def _create():
        from services.category_service import create_category
        return create_category(body.label, keywords=body.keywords, weight=body.weight)
    return await run_sync(_create)


@app.patch("/api/categories/{name}/keywords", tags=["Categories"])
async def update_category_keywords(name: str, body: CategoryKeywordsUpdate, user: dict = Depends(get_current_user)):
    """
    Admin tuning — add/replace keywords for a category so it starts
    (or stops) participating in automatic smart-routing. New custom
    categories start with zero keywords; this is how an admin makes
    one routable.
    """
    def _update():
        from services.category_service import update_category_keywords
        return update_category_keywords(name, body.keywords, weight=body.weight)

    try:
        return await run_sync(_update)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.delete("/api/categories/{name}", tags=["Categories"])
async def delete_category(name: str, user: dict = Depends(get_current_user)):
    """
    Delete a custom category. Predefined categories (hr, engineering,
    company_info, finance, reference, general) cannot be deleted —
    only their keywords can be edited. Documents using a deleted
    category are reassigned to 'general' automatically.
    """
    def _delete():
        from services.category_service import delete_category
        return delete_category(name)

    try:
        ok = await run_sync(_delete)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    if not ok:
        raise HTTPException(status_code=404, detail="Category not found")
    return {"deleted": True, "name": name}


# ---------------------------------------------------------------------------
# 4. Chat Sessions
# ---------------------------------------------------------------------------

@app.get("/api/sessions/search", tags=["Chat"])
async def search_sessions(q: str, limit: int = 20, user: dict = Depends(get_current_user)):
    """
    Powers the Ctrl+K command palette. Searches session titles AND
    message content, returns a ranked list with a short snippet so
    the dropdown can show context, not just a title.
    """
    def _search():
        from services.chat_service import search_sessions
        results = search_sessions(user["id"], q, limit=limit)
        for r in results:
            if hasattr(r.get("updated_at"), "isoformat"):
                r["updated_at"] = r["updated_at"].isoformat()
        return results
    return await run_sync(_search)


@app.get("/api/sessions", tags=["Chat"])
async def list_sessions(user: dict = Depends(get_current_user)):
    def _list():
        from services.chat_service import get_all_sessions
        sessions = get_all_sessions(user["id"])
        for s in sessions:
            for k in ("created_at", "updated_at"):
                if hasattr(s.get(k), "isoformat"):
                    s[k] = s[k].isoformat()
        return sessions
    return await run_sync(_list)


@app.post("/api/sessions", tags=["Chat"])
async def create_session(body: SessionCreate, user: dict = Depends(get_current_user)):
    def _create():
        from services.chat_service import create_session
        s = create_session(user["id"], body.title)
        for k in ("created_at", "updated_at"):
            if hasattr(s.get(k), "isoformat"):
                s[k] = s[k].isoformat()
        return s
    return await run_sync(_create)


@app.get("/api/sessions/{session_id}/messages", tags=["Chat"])
async def get_messages(session_id: int, user: dict = Depends(get_current_user)):
    def _get():
        from services.chat_service import get_messages
        from services.pin_service import is_pinned
        msgs = get_messages(user["id"], session_id)
        for m in msgs:
            if hasattr(m.get("created_at"), "isoformat"):
                m["created_at"] = m["created_at"].isoformat()
            m["is_pinned"] = is_pinned(m["id"]) if m["role"] == "assistant" else False
        return msgs
    return await run_sync(_get)


@app.delete("/api/sessions/{session_id}", tags=["Chat"])
async def delete_session(session_id: int, user: dict = Depends(get_current_user)):
    def _delete():
        from services.chat_service import delete_session
        return delete_session(user["id"], session_id)
    ok = await run_sync(_delete)
    if not ok:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"deleted": True, "session_id": session_id}


@app.patch("/api/sessions/{session_id}", tags=["Chat"])
async def rename_session(session_id: int, body: SessionRename, user: dict = Depends(get_current_user)):
    def _rename():
        from services.chat_service import rename_session
        return rename_session(user["id"], session_id, body.title)
    ok = await run_sync(_rename)
    if not ok:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"session_id": session_id, "title": body.title}


# ---------------------------------------------------------------------------
# 5. RAG — Ask (Streaming SSE)
# ---------------------------------------------------------------------------

@app.post("/api/chat/ask", tags=["Chat"])
async def ask(body: AskRequest, user: dict = Depends(get_current_user)):
    """
    Core RAG endpoint. Streams the answer token by token using Server-Sent Events.

    Why SSE instead of WebSocket:
      - SSE is one-directional (server → client) — perfect for streaming text
      - Works through proxies and firewalls better than WebSocket
      - Built into the browser's EventSource API — no library needed

    Response format (each SSE event):
      data: {"type": "routing",  "category": "company_info", "confidence": 2.0}
      data: {"type": "sources",  "chunks": [...]}
      data: {"type": "token",    "text": "The company"}
      data: {"type": "token",    "text": " is located"}
      data: {"type": "done",     "full_answer": "...", "session_id": 1}
      data: {"type": "error",    "message": "..."}

    Concurrent users:
      Each request runs independently in the async loop.
      Retrieval (CPU) runs in thread pool — 4 parallel.
      Ollama streaming runs async via httpx — non-blocking.
      30 users stream simultaneously without blocking each other.
    """
    async def event_stream():
        try:
            # --- Step 1: Retrieve chunks (CPU-bound → thread pool) ---
            def _retrieve():
                from services.chat_service    import get_history_buffer
                from services.retrieval_service import retrieve
                history = get_history_buffer(body.session_id)
                chunks  = retrieve(body.question, user["id"], document_ids=body.document_ids)
                return history, chunks

            history, chunks = await run_sync(_retrieve)

            # Send routing info
            if chunks:
                routing = chunks[0].get("routing", {})
                yield f"data: {json.dumps({'type': 'routing', 'category': routing.get('category','general'), 'confidence': routing.get('confidence', 0)})}\n\n"

            # Send source chunks to frontend
            sources = []
            for c in chunks:
                sources.append({
                    "filename":       c.get("filename", ""),
                    "chunk_number":   c.get("chunk_number", 0),
                    "chunk_text":     c.get("chunk_text", "")[:500],
                    "reranker_score": round(float(c.get("reranker_score", 0)), 4),
                    "similarity":     round(float(c.get("similarity", 0)), 4),
                })
            yield f"data: {json.dumps({'type': 'sources', 'chunks': sources})}\n\n"

            if not chunks:
                no_ctx = "I could not find this information in the uploaded documents."
                yield f"data: {json.dumps({'type': 'token', 'text': no_ctx})}\n\n"
                message_id = await run_sync(_save_messages, user["id"], body.session_id, body.question, no_ctx, [], history)
                yield f"data: {json.dumps({'type': 'done', 'full_answer': no_ctx, 'session_id': body.session_id, 'message_id': message_id})}\n\n"
                return

            # --- Step 2: Build prompt (sync, fast) ---
            def _build_prompt():
                from services.llm_service import build_prompt
                return build_prompt(body.question, chunks, history=history)

            prompt = await run_sync(_build_prompt)

            # --- Step 3: Stream from Ollama (async HTTP — non-blocking) ---
            # Acquire the concurrency semaphore BEFORE calling Ollama. If
            # OLLAMA_MAX_PARALLEL slots are all taken, this request waits
            # here — which is exactly the queued state /api/admin/queue
            # reports to the frontend so it can show "server busy" instead
            # of a silent hang.
            import httpx
            from config import OLLAMA_BASE_URL
            from services.settings_service import get_setting

            model       = get_setting("llm_model")
            temperature = get_setting("temperature")
            num_predict = get_setting("num_predict")

            payload = {
                "model":  model,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "temperature": temperature,
                    "num_predict": num_predict,
                },
                "think": False,
            }

            full_answer = ""

            _queue_state["queued"] += 1
            async with _ollama_semaphore:
                _queue_state["queued"] -= 1
                _queue_state["active"] += 1
                try:
                    async with httpx.AsyncClient(timeout=120.0) as client:
                        async with client.stream(
                            "POST",
                            f"{OLLAMA_BASE_URL}/api/generate",
                            json=payload,
                        ) as response:
                            async for line in response.aiter_lines():
                                if not line.strip():
                                    continue
                                try:
                                    chunk_data = json.loads(line)
                                    token = chunk_data.get("response", "")
                                    if token:
                                        full_answer += token
                                        yield f"data: {json.dumps({'type': 'token', 'text': token})}\n\n"
                                    if chunk_data.get("done"):
                                        break
                                except json.JSONDecodeError:
                                    continue
                finally:
                    _queue_state["active"] -= 1

            # Clean any thinking tags from full answer
            import re
            full_answer = re.sub(r"<think>.*?</think>", "", full_answer, flags=re.DOTALL).strip()

            # --- Step 4: Save to chat history ---
            message_id = await run_sync(
                _save_messages,
                user["id"], body.session_id, body.question, full_answer, chunks, history
            )

            yield f"data: {json.dumps({'type': 'done', 'full_answer': full_answer, 'session_id': body.session_id, 'message_id': message_id})}\n\n"

            # --- Step 5: Follow-up suggestions (best-effort, non-blocking the 'done' event) ---
            # Sent as a separate event AFTER 'done' so the answer renders
            # immediately; suggestion chips pop in a moment later.
            def _suggestions():
                from services.llm_service import generate_followups
                return generate_followups(body.question, full_answer)

            try:
                followups = await run_sync(_suggestions)
                if followups:
                    yield f"data: {json.dumps({'type': 'followups', 'suggestions': followups})}\n\n"
            except Exception:
                pass   # follow-ups are purely additive — never fail the response over this

        except Exception as exc:
            logger.error("Ask endpoint error: %s", exc)
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering": "no",      # important for nginx proxies
        },
    )


def _save_messages(user_id, session_id, question, answer, chunks, history):
    from services.chat_service import add_message, auto_title_session, get_messages
    existing = get_messages(user_id, session_id)
    if not existing:
        auto_title_session(user_id, session_id, question)
    add_message(session_id, "user", question)
    assistant_msg = add_message(session_id, "assistant", answer, sources=chunks)
    return assistant_msg["id"]


# ---------------------------------------------------------------------------
# 6. Pinned / Saved Answers
# ---------------------------------------------------------------------------

@app.post("/api/messages/{message_id}/pin", tags=["Saved"])
async def pin_message(message_id: int, body: PinRequest):
    """Pin an assistant answer for later reference (personal saved library)."""
    def _pin():
        from services.pin_service import pin_answer
        return pin_answer(message_id, note=body.note)

    try:
        result = await run_sync(_pin)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    if hasattr(result.get("created_at"), "isoformat"):
        result["created_at"] = result["created_at"].isoformat()
    return result


@app.delete("/api/messages/{message_id}/pin", tags=["Saved"])
async def unpin_message(message_id: int):
    """Remove a pin."""
    def _unpin():
        from services.pin_service import unpin_answer
        return unpin_answer(message_id)

    ok = await run_sync(_unpin)
    if not ok:
        raise HTTPException(status_code=404, detail="Pin not found")
    return {"unpinned": True, "message_id": message_id}


@app.get("/api/saved", tags=["Saved"])
async def list_saved_answers():
    """
    Personal saved-answers library. Shown in the sidebar 'Saved' section.
    Each entry includes the original question context (session) and
    the sources that backed the answer, so a pinned answer is fully
    self-contained and verifiable without reopening the original chat.
    """
    def _list():
        from services.pin_service import get_pinned_answers
        rows = get_pinned_answers()
        for r in rows:
            for k in ("pinned_at", "message_created_at"):
                if hasattr(r.get(k), "isoformat"):
                    r[k] = r[k].isoformat()
        return rows
    return await run_sync(_list)


# ---------------------------------------------------------------------------
# 7. Data Explorer (Excel / CSV — session scoped)
# ---------------------------------------------------------------------------

# In-memory store for explorer DataFrames keyed by session token
# In production replace with Redis or temp file store
_explorer_store: dict = {}


@app.post("/api/explorer/upload", tags=["Explorer"])
async def explorer_upload(file: UploadFile = File(...), user: dict = Depends(get_current_user)):
    """
    Upload a CSV or Excel file for ad-hoc exploration.
    Returns a token the frontend uses for subsequent queries.
    """
    import io, hashlib
    from services.data_explorer_service import load_file, build_schema_context

    raw      = await file.read()
    token    = hashlib.md5(raw + file.filename.encode()).hexdigest()[:12]

    def _load():
        df, err = load_file(io.BytesIO(raw), file.filename)
        if err:
            return None, err
        schema = build_schema_context(df, file.filename)
        return df, schema

    df, result = await run_sync(_load)
    if df is None:
        raise HTTPException(status_code=400, detail=result)

    _explorer_store[token] = {
        "df":       df,
        "filename": file.filename,
        "schema":   result,
        "user_id":  user["id"],
    }

    return {
        "token":    token,
        "filename": file.filename,
        "rows":     int(df.shape[0]),
        "cols":     int(df.shape[1]),
        "columns":  list(df.columns),
    }


@app.post("/api/explorer/query", tags=["Explorer"])
async def explorer_query(body: ExplorerQuery, user: dict = Depends(get_current_user)):
    """
    Natural language query against an uploaded explorer file.
    Returns result as JSON (table, text, or chart config).
    """
    # Find the right explorer session by filename and user_id
    store_entry = next(
        (v for v in _explorer_store.values() if v["filename"] == body.filename and v.get("user_id") == user["id"]),
        None
    )
    if not store_entry:
        raise HTTPException(
            status_code=404,
            detail="File not found in explorer. Upload it first via /api/explorer/upload"
        )

    def _query():
        from services.data_explorer_service import answer_query
        result = answer_query(
            body.query,
            store_entry["df"],
            store_entry["schema"],
            body.filename,
        )
        if result.get("result_type") == "dataframe":
            import pandas as pd
            import numpy as np
            df_result = result["result"]
            if isinstance(df_result, pd.DataFrame):
                df_clean = df_result.head(200).copy()
                # Handle Inf and NaN -> None for JSON compliance
                df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
                df_clean = df_clean.astype(object).where(pd.notnull(df_clean), None)
                result["result"] = df_clean.to_dict(orient="records")
                result["columns"] = list(df_clean.columns)
        elif result.get("result_type") == "series":
            result["result"] = str(result["result"])
        elif result.get("image_bytes"):
            import base64
            result["image_b64"] = base64.b64encode(
                result["image_bytes"].getvalue()
            ).decode()
            result.pop("image_bytes")
        return result

    result = await run_sync(_query)
    return result


# ---------------------------------------------------------------------------
# 8. Metadata
# ---------------------------------------------------------------------------

@app.get("/api/documents/{doc_id}/metadata", tags=["Documents"])
async def get_metadata(doc_id: int):
    def _get():
        from services.metadata_service import get_metadata
        return get_metadata(doc_id)
    return await run_sync(_get)


# ---------------------------------------------------------------------------
# 9. Admin
# ---------------------------------------------------------------------------

@app.get("/api/admin/settings", tags=["Admin"])
async def get_settings(user: dict = Depends(get_current_user)):
    """
    Return every current effective setting — the admin panel's main
    config view. Includes both SAFE settings (live-tunable, no restart)
    and the current model selections (llm_model, embedding_model,
    reranker_model), plus whether each reranker/LLM model is actually
    available in Ollama / cached locally.
    """
    def _get():
        from services.settings_service import get_all_settings, list_ollama_models
        from services.reranker_service import is_model_cached

        settings = get_all_settings(user["id"])
        ollama_models = list_ollama_models()

        return {
            "settings": settings,
            "llm_model_available":       settings["llm_model"] in ollama_models,
            "embedding_model_available": settings["embedding_model"] in ollama_models,
            "reranker_model_cached":     is_model_cached(settings["reranker_model"]),
            "available_ollama_models":   ollama_models,
        }
    return await run_sync(_get)


@app.patch("/api/admin/settings", tags=["Admin"])
async def update_settings(body: SettingsUpdate, user: dict = Depends(get_current_user)):
    """
    Bulk-update SAFE settings (chunk size/overlap, retrieval pool sizes,
    mmr_lambda, top_k, history_window, num_predict, temperature, routing
    threshold). Every service already reads these live on each call —
    changes apply to the very next request, no restart.

    Only fields explicitly provided in the request body are changed.
    """
    def _update():
        from services.settings_service import set_setting
        updated = {}
        for key, value in body.dict(exclude_unset=True, exclude_none=True).items():
            updated[key] = set_setting(user["id"], key, value)
        return updated
    return await run_sync(_update)


@app.post("/api/admin/settings/reset/{key}", tags=["Admin"])
async def reset_setting_endpoint(key: str, user: dict = Depends(get_current_user)):
    """Revert one setting back to its hardcoded default value."""
    def _reset():
        from services.settings_service import reset_setting
        reset_setting(user["id"], key)
    try:
        await run_sync(_reset)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown setting: '{key}'")
    return {"reset": True, "key": key}


@app.get("/api/admin/models", tags=["Admin"])
async def list_models():
    """
    List every model currently pulled in Ollama. Populates the LLM /
    embedding model dropdowns in the admin panel so the admin can only
    pick models that actually exist — no typo'd model name errors.
    """
    def _list():
        from services.settings_service import list_ollama_models
        return {"models": list_ollama_models()}
    return await run_sync(_list)


@app.post("/api/admin/settings/llm-model", tags=["Admin"])
async def swap_llm_model(body: ModelSwapRequest, user: dict = Depends(get_current_user)):
    """
    Swap the LLM model. Safe — applies immediately, no migration needed.
    Validates the model is actually pulled in Ollama first.
    """
    def _swap():
        from services.settings_service import update_llm_model
        return update_llm_model(user["id"], body.model)
    try:
        return await run_sync(_swap)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/admin/settings/reranker-model", tags=["Admin"])
async def swap_reranker_model(body: ModelSwapRequest, user: dict = Depends(get_current_user)):
    """
    Swap the cross-encoder reranker model. Safe — applies immediately.
    If the model isn't cached locally yet, the FIRST rerank call after
    this swap will download it (~80MB typical) which adds latency to
    that one request only.
    """
    def _swap():
        from services.settings_service import update_reranker_model
        from services.reranker_service import is_model_cached
        result = update_reranker_model(user["id"], body.model)
        result["already_cached"] = is_model_cached(body.model)
        return result
    return await run_sync(_swap)


@app.post("/api/admin/settings/embedding-model/preview", tags=["Admin"])
async def preview_embedding_model_swap(body: EmbeddingModelSwapRequest):
    """
    DRY RUN — shows what swapping the embedding model would do, without
    applying anything. The admin UI must call this first and show the
    returned warning to the admin before allowing them to confirm via
    POST /api/admin/settings/embedding-model/apply.

    This endpoint is intentionally non-destructive no matter what body
    is sent — 'confirm' is ignored here.
    """
    def _preview():
        from services.settings_service import preview_embedding_model_change
        return preview_embedding_model_change(body.model, body.dimension)
    return await run_sync(_preview)


@app.post("/api/admin/settings/embedding-model/apply", tags=["Admin"])
async def apply_embedding_model_swap(body: EmbeddingModelSwapRequest):
    """
    APPLIES the embedding model swap. Destructive to search quality:
    every existing embedding is nulled (they're from a different vector
    space) and must be regenerated via POST /api/documents/embed
    afterward. Requires confirm=true in the request body — the admin
    UI should only send that after the admin has seen the
    /preview response and explicitly clicked through a warning dialog.
    """
    def _apply():
        from services.settings_service import update_embedding_model
        return update_embedding_model(body.model, body.dimension, confirm=body.confirm)

    try:
        result = await run_sync(_apply)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return result


@app.post("/api/admin/rebuild-index", tags=["Admin"])
async def rebuild_index(background_tasks: BackgroundTasks):
    """Rebuild BM25 index. Runs in background."""
    def _rebuild():
        from services.bm25_service import build_index
        return build_index()
    background_tasks.add_task(lambda: asyncio.get_event_loop().run_in_executor(_thread_pool, _rebuild))
    return {"status": "rebuilding"}


@app.get("/api/admin/queue", tags=["Admin"])
async def queue_status():
    """
    Live Ollama queue depth — how many LLM generations are currently
    running vs waiting for a slot. The frontend uses this to show a
    'Server is busy, estimated wait: ~Ns' banner instead of letting
    a request silently hang with no explanation.

    active : requests currently inside Ollama generating tokens
    queued : requests waiting for a free slot (OLLAMA_MAX_PARALLEL)
    """
    avg_generation_seconds = 12  # rough estimate, tune from real logs
    est_wait = _queue_state["queued"] * avg_generation_seconds / max(OLLAMA_MAX_PARALLEL, 1)
    return {
        "active":               _queue_state["active"],
        "queued":                _queue_state["queued"],
        "max_parallel":          OLLAMA_MAX_PARALLEL,
        "estimated_wait_seconds": round(est_wait, 1),
    }


@app.post("/api/admin/clear-explorer-cache", tags=["Admin"])
async def clear_explorer_cache():
    """Clear all in-memory Data Explorer DataFrames to free RAM."""
    cleared = len(_explorer_store)
    _explorer_store.clear()
    return {"cleared": cleared}