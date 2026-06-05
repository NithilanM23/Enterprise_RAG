"""
main.py
-------
Entry point for the Local Employee Knowledge Assistant.

Phase 1 : Database init + health check
Phase 2 : Document ingestion  (load -> chunk -> store)
Phase 3 : Embedding generation (embed chunks -> update DB)
Phase 4 : Ask questions        (retrieve -> prompt -> LLM answer)

Usage:
    python main.py                          # health check
    python main.py --ingest <filepath>      # ingest a document
    python main.py --embed                  # generate embeddings for pending chunks
    python main.py --ask "your question"    # ask a question
    python main.py --list                   # list all documents
    python main.py --status                 # embedding completion status
"""

import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Phase 1 — Startup
# ---------------------------------------------------------------------------

def run_startup() -> bool:
    from services.database_service import initialize_database, health_check

    print("\n" + "=" * 60)
    print("  Local Employee Knowledge Assistant")
    print("=" * 60)

    print("\n[1/2] Initializing database...")
    try:
        initialize_database()
        print("      ✓ Database initialized successfully.")
    except Exception as exc:
        print(f"      ✗ Database initialization failed: {exc}")
        return False

    print("\n[2/2] Running health check...")
    status = health_check()

    print(f"\n  PostgreSQL connected : {'✓' if status['connected']      else '✗'}")
    print(f"  pgvector ready       : {'✓' if status['pgvector_ready'] else '✗'}")
    print(f"  Documents in DB      : {status['document_count']}")

    if status["error"] or not status["connected"] or not status["pgvector_ready"]:
        if status["error"]:
            print(f"\n  Error: {status['error']}")
        return False

    print("\n" + "=" * 60)
    print("  System ready.")
    print("=" * 60 + "\n")
    return True


# ---------------------------------------------------------------------------
# Phase 2 — Document ingestion
# ---------------------------------------------------------------------------

def ingest_document(filepath: str, category: str = "general") -> None:
    import shutil
    from pathlib import Path
    from services.loader import load_document
    from services.chunker import chunk_text
    from services.database_service import insert_document, insert_embeddings
    from config import UPLOAD_DIR

    filepath = Path(filepath)

    if not filepath.exists():
        print(f"\n  ✗ File not found: {filepath}")
        return

    print(f"\n{'=' * 60}")
    print(f"  Ingesting: {filepath.name}")
    print(f"{'=' * 60}")

    dest_path = UPLOAD_DIR / filepath.name
    if dest_path.exists():
        print(f"\n  ✗ '{filepath.name}' already exists in uploads.")
        print("    Delete it first if you want to re-ingest.")
        return

    shutil.copy2(filepath, dest_path)
    print(f"\n  [1/4] Copied to uploads folder.")

    from config import EXCEL_EXTENSIONS
    from services.metadata_service import extract_metadata

    # Insert document row first
    doc_id = None
    try:
        doc_id = insert_document(
            filename=filepath.name,
            filepath=str(dest_path.resolve()),
            category=category,
        )
    except ValueError as exc:
        print(f"        \u2717 {exc}")
        dest_path.unlink(missing_ok=True)
        return

    # Always extract metadata
    print(f"  [2/4] Extracting metadata...")
    try:
        meta = extract_metadata(str(dest_path.resolve()), doc_id)
        print(f"        \u2713 {len(meta)} metadata fields stored.")
    except Exception as exc:
        print(f"        \u26a0  Metadata extraction failed (non-fatal): {exc}")

    # Excel files -> row storage
    if dest_path.suffix.lower() in EXCEL_EXTENSIONS:
        from services.excel_service import ingest_excel
        print(f"  [3/4] Ingesting Excel rows...")
        try:
            info = ingest_excel(str(dest_path.resolve()), doc_id)
            print(f"        \u2713 {info['sheet_count']} sheet(s), {info['total_rows']} rows stored.")
            print(f"\n  \u2713 '{filepath.name}' ingested (Excel).")
            print(f"    Rows     : {info['total_rows']}")
            print(f"    Category : {category}\n")
        except Exception as exc:
            print(f"        \u2717 Excel ingestion failed: {exc}")
            dest_path.unlink(missing_ok=True)
            from services.database_service import delete_document
            delete_document(doc_id)
        return

    # All other formats -> RAG pipeline
    print(f"  [3/4] Extracting text + chunking...")
    try:
        text   = load_document(str(dest_path))
        chunks = chunk_text(text)
        print(f"        \u2713 Created {len(chunks)} chunks.")
    except Exception as exc:
        print(f"        \u2717 Extraction/chunking failed: {exc}")
        dest_path.unlink(missing_ok=True)
        from services.database_service import delete_document
        delete_document(doc_id)
        return

    print(f"  [4/4] Storing chunks in database...")
    try:
        insert_embeddings(document_id=doc_id, chunks=chunks)
        print(f"        \u2713 Stored {len(chunks)} chunks.")
    except Exception as exc:
        print(f"        \u2717 Database storage failed: {exc}")
        dest_path.unlink(missing_ok=True)
        from services.database_service import delete_document
        delete_document(doc_id)
        return

    print(f"\n  \u2713 '{filepath.name}' ingested successfully.")
    print(f"    Document ID : {doc_id}")
    print(f"    Chunks      : {len(chunks)}")
    print(f"    Category    : {category}")
    print(f"    Embeddings  : run  python main.py --embed  to generate\n")


# ---------------------------------------------------------------------------
# Phase 3 — Embedding generation
# ---------------------------------------------------------------------------

def generate_embeddings() -> None:
    import psycopg2
    import psycopg2.extras
    from services.embedding_service import embed_chunks, check_ollama_connection
    from services.database_service import _get_connection
    from config import EMBEDDING_MODEL

    print(f"\n{'=' * 60}")
    print(f"  Phase 3 — Embedding Generation")
    print(f"  Model: {EMBEDDING_MODEL}")
    print(f"{'=' * 60}")

    print("\n  Checking Ollama...")
    status = check_ollama_connection()
    print(f"  Ollama reachable : {'✓' if status['reachable']   else '✗'}")
    print(f"  Model ready      : {'✓' if status['model_ready'] else '✗'}")

    if not status["reachable"] or not status["model_ready"]:
        print(f"\n  ✗ {status['error']}")
        return

    fetch_query = """
        SELECT e.id, e.chunk_text, d.filename
        FROM embeddings e
        JOIN documents d ON d.id = e.document_id
        WHERE e.embedding IS NULL
        ORDER BY e.document_id, e.chunk_number;
    """

    with _get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(fetch_query)
            rows = [dict(r) for r in cur.fetchall()]

    if not rows:
        print("\n  ✓ All chunks already have embeddings. Nothing to do.\n")
        return

    print(f"\n  Found {len(rows)} chunks pending embedding.")
    print(f"  This may take a few minutes on CPU — please wait...\n")

    chunks_to_embed = [
        {"id": row["id"], "chunk_text": row["chunk_text"], "embedding": None}
        for row in rows
    ]

    try:
        embed_chunks(chunks_to_embed)
    except RuntimeError as exc:
        print(f"\n  ✗ {exc}")
        return

    print("\n  Saving embeddings to database...")
    update_query = "UPDATE embeddings SET embedding = %s WHERE id = %s;"

    with _get_connection() as conn:
        with conn.cursor() as cur:
            for chunk in chunks_to_embed:
                cur.execute(update_query, (chunk["embedding"], chunk["id"]))

    print(f"  ✓ {len(chunks_to_embed)} embeddings saved successfully.")

    # Auto-rebuild BM25 index after embedding so it stays in sync
    print("\n  Rebuilding BM25 index...")
    try:
        from services.bm25_service import build_index
        info = build_index()
        print(f"  ✓ BM25 index rebuilt: {info['chunk_count']} chunks indexed.")
    except Exception as exc:
        print(f"  ⚠  BM25 index rebuild failed: {exc}")

    print(f"\n  Ready — you can now ask questions.\n")


# ---------------------------------------------------------------------------
# Phase 4 — Ask a question
# ---------------------------------------------------------------------------

def ask_question(question: str) -> None:
    """
    Full RAG pipeline:
        question -> embed -> retrieve top-K chunks -> build prompt -> LLM -> answer
    """
    from services.retrieval_service import retrieve
    from services.llm_service import generate_answer, check_llm_connection
    from config import LLM_MODEL, TOP_K

    print(f"\n{'=' * 60}")
    print(f"  Question: {question}")
    print(f"{'=' * 60}")

    # --- Check LLM is ready ---
    print(f"\n  Checking LLM ({LLM_MODEL})...")
    llm_status = check_llm_connection()
    if not llm_status["reachable"] or not llm_status["model_ready"]:
        print(f"  ✗ {llm_status['error']}")
        return
    print(f"  ✓ LLM ready.")

    # --- Retrieve relevant chunks ---
    print(f"\n  Retrieving top-{TOP_K} relevant chunks...")
    try:
        chunks = retrieve(question)
    except RuntimeError as exc:
        print(f"  ✗ Retrieval failed: {exc}")
        return

    if not chunks:
        print("  ✗ No relevant chunks found. Have you ingested and embedded documents?")
        return

    print(f"  ✓ Retrieved {len(chunks)} chunks.")

    # --- Generate answer ---
    print(f"\n  Generating answer (this may take 15–45s on CPU)...\n")
    try:
        answer = generate_answer(question, chunks)
    except RuntimeError as exc:
        print(f"  ✗ LLM generation failed: {exc}")
        return

    # --- Display answer ---
    print("=" * 60)
    print("  ANSWER")
    print("=" * 60)
    print(f"\n{answer}\n")

    # --- Display source chunks ---
    print("=" * 60)
    print(f"  SOURCES  (top {len(chunks)} chunks used as context)")
    print("=" * 60)

    for i, chunk in enumerate(chunks, start=1):
        print(f"\n  [{i}] {chunk['filename']}  —  chunk {chunk['chunk_number']}"
              f"  (similarity: {chunk['similarity']:.4f})")
        print(f"  {'-' * 56}")
        # Print first 300 chars of chunk to keep output readable
        preview = chunk["chunk_text"][:300].replace("\n", " ")
        if len(chunk["chunk_text"]) > 300:
            preview += "..."
        print(f"  {preview}")

    print()


# ---------------------------------------------------------------------------
# List documents
# ---------------------------------------------------------------------------

def list_documents() -> None:
    from services.database_service import get_all_documents

    docs = get_all_documents()
    print(f"\n{'=' * 60}")
    print(f"  Ingested Documents ({len(docs)} total)")
    print(f"{'=' * 60}")

    if not docs:
        print("\n  No documents ingested yet.")
        print("  Use:  python main.py --ingest <filepath>\n")
        return

    for doc in docs:
        print(f"\n  ID       : {doc['id']}")
        print(f"  Filename : {doc['filename']}")
        print(f"  Uploaded : {doc['upload_time'].strftime('%Y-%m-%d %H:%M:%S')}")
    print()


# ---------------------------------------------------------------------------
# Embedding status
# ---------------------------------------------------------------------------

def show_status() -> None:
    import psycopg2
    import psycopg2.extras
    from services.database_service import _get_connection

    query = """
        SELECT
            d.filename,
            COUNT(e.id)              AS total_chunks,
            COUNT(e.embedding)       AS embedded,
            COUNT(e.id) - COUNT(e.embedding) AS pending
        FROM documents d
        JOIN embeddings e ON e.document_id = d.id
        GROUP BY d.id, d.filename
        ORDER BY d.id;
    """

    print(f"\n{'=' * 60}")
    print(f"  Embedding Status")
    print(f"{'=' * 60}\n")

    with _get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(query)
            rows = cur.fetchall()

    if not rows:
        print("  No documents found.\n")
        return

    for row in rows:
        status = "✓ complete" if row["pending"] == 0 else f"⚠  {row['pending']} pending"
        print(f"  {row['filename']}")
        print(f"    Chunks: {row['total_chunks']}  |  Embedded: {row['embedded']}  |  {status}")
        print()



# ---------------------------------------------------------------------------
# Delete a document
# ---------------------------------------------------------------------------

def delete_document_cmd(filename: str) -> None:
    """
    Delete a document by filename — removes from DB and uploads folder.
    BM25 index is rebuilt automatically inside database_service.delete_document.
    """
    from services.database_service import (
        get_document_by_filename, delete_document
    )
    from config import UPLOAD_DIR
    from pathlib import Path

    print(f"\n{'=' * 60}")
    print(f"  Deleting: {filename}")
    print(f"{'=' * 60}\n")

    doc = get_document_by_filename(filename)
    if not doc:
        print(f"  ✗ No document named '{filename}' found in database.\n")
        print("  Use:  python main.py --list  to see available documents.")
        return

    success = delete_document(doc["id"])
    if success:
        file_path = UPLOAD_DIR / filename
        file_path.unlink(missing_ok=True)
        print(f"  ✓ '{filename}' deleted from database and uploads folder.")
        print(f"    BM25 index rebuilt automatically.\n")
    else:
        print(f"  ✗ Failed to delete '{filename}'.\n")


def set_document_category(filename: str, category: str) -> None:
    """Update the category of an existing document."""
    from services.router_service import get_all_categories, get_category_description
    from services.database_service import _get_raw_connection

    valid = get_all_categories() + ["general"]
    if category not in valid:
        print(f"\n  ✗ Unknown category '{category}'.")
        print(f"  Valid categories: {', '.join(valid)}\n")
        return

    with _get_raw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE documents SET category = %s WHERE filename = %s RETURNING id;",
                (category, filename),
            )
            row = cur.fetchone()

    if row:
        label = get_category_description(category)
        print(f"\n  ✓ '{filename}' category set to: {label}\n")
    else:
        print(f"\n  ✗ No document named '{filename}' found.")
        print("  Use:  python main.py --list  to see available documents.\n")


# ---------------------------------------------------------------------------
# BM25 index management
# ---------------------------------------------------------------------------

def build_bm25_index() -> None:
    """Build (or rebuild) the BM25 keyword search index from all DB chunks."""
    from services.bm25_service import build_index

    print(f"\n{'=' * 60}")
    print(f"  Building BM25 Index")
    print(f"{'=' * 60}\n")

    try:
        info = build_index()
        print(f"  ✓ BM25 index built successfully.")
        print(f"    Chunks indexed : {info['chunk_count']}")
        print(f"    Index saved to : {info['index_path']}\n")
    except RuntimeError as exc:
        print(f"  ✗ {exc}\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = sys.argv[1:]

    ok = run_startup()
    if not ok:
        sys.exit(1)

    if not args:
        sys.exit(0)

    if args[0] == "--ingest" and len(args) == 2:
        ingest_document(args[1])

    elif args[0] == "--ingest" and len(args) == 4 and args[2] == "--category":
        ingest_document(args[1], category=args[3])

    elif args[0] == "--set-category" and len(args) == 3:
        set_document_category(args[1], args[2])

    elif args[0] == "--embed":
        generate_embeddings()

    elif args[0] == "--ask" and len(args) >= 2:
        question = " ".join(args[1:])   # handles multi-word questions without quotes
        ask_question(question)

    elif args[0] == "--list":
        list_documents()

    elif args[0] == "--status":
        show_status()


    elif args[0] == "--delete" and len(args) == 2:
        delete_document_cmd(args[1])

    elif args[0] == "--build-index":
        build_bm25_index()
    else:
        print("Usage:")
        print("  python main.py                          # health check")
        print("  python main.py --ingest <filepath>      # ingest a document")
        print("  python main.py --embed                  # generate embeddings")
        print('  python main.py --ask "your question"    # ask a question')
        print("  python main.py --list                   # list all documents")
        print("  python main.py --delete <filename>      # delete a document")
        print("  python main.py --status                 # embedding status")
        print("  python main.py --build-index             # build BM25 keyword index")
        sys.exit(1)