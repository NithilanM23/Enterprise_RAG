"""
app.py
------
Streamlit UI for the Local Employee Knowledge Assistant.

Run with:
    streamlit run app.py

Pages:
  Chat         — persistent multi-session chat with source attribution
  My Documents — manage uploaded documents, generate embeddings
  Upload       — ingest new documents with category tagging
"""

import streamlit as st
import sys
import os
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.basicConfig(level=logging.WARNING)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Employee Knowledge Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

[data-testid="stSidebar"] { background-color: #0d1117; border-right: 1px solid #21262d; }
[data-testid="stSidebar"] * { color: #c9d1d9 !important; }

/* Chat bubbles */
.user-bubble {
    background: #1f6feb;
    color: #ffffff;
    border-radius: 18px 18px 4px 18px;
    padding: 12px 16px;
    margin: 4px 0 4px 15%;
    font-size: 14px;
    line-height: 1.6;
    word-wrap: break-word;
}
.assistant-bubble {
    background: #161b22;
    border: 1px solid #30363d;
    color: #e6edf3;
    border-radius: 18px 18px 18px 4px;
    padding: 12px 16px;
    margin: 4px 15% 4px 0;
    font-size: 14px;
    line-height: 1.6;
    word-wrap: break-word;
}
.role-label-user {
    text-align: right;
    color: #8b949e;
    font-size: 11px;
    margin: 8px 0 2px 0;
}
.role-label-assistant {
    text-align: left;
    color: #8b949e;
    font-size: 11px;
    margin: 8px 0 2px 0;
}
.source-card {
    background: #0d1117;
    border: 1px solid #21262d;
    border-left: 3px solid #1f6feb;
    border-radius: 6px;
    padding: 10px 14px;
    margin: 4px 15% 4px 0;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11.5px;
    color: #8b949e;
    line-height: 1.5;
}
.chat-divider {
    border: none;
    border-top: 1px solid #21262d;
    margin: 16px 0;
}
.session-title {
    font-size: 13px;
    color: #c9d1d9;
    padding: 6px 8px;
    border-radius: 6px;
    cursor: pointer;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.session-active {
    background: #1f6feb22;
    border-left: 3px solid #1f6feb;
}
.status-ok  { color: #3fb950; font-weight: 600; }
.status-warn { color: #d29922; font-weight: 600; }
hr { border: none; border-top: 1px solid #21262d; margin: 16px 0; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Connecting to database...")
def init_system():
    from services.database_service import initialize_database, health_check
    try:
        initialize_database()
        return health_check()
    except Exception as exc:
        return {"connected": False, "pgvector_ready": False,
                "document_count": 0, "error": str(exc)}

def get_documents():
    from services.database_service import get_all_documents
    return get_all_documents()

def get_embedding_status():
    import psycopg2.extras
    from services.database_service import _get_connection
    with _get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*), COUNT(embedding) FROM embeddings;")
            total, embedded = cur.fetchone()
    return total, embedded, total - embedded

# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------

def load_sessions():
    from services.chat_service import get_all_sessions
    return get_all_sessions()

def ensure_active_session():
    """Create a new session if none is active."""
    if "active_session_id" not in st.session_state or st.session_state.active_session_id is None:
        from services.chat_service import create_session
        session = create_session("New Chat")
        st.session_state.active_session_id = session["id"]

def switch_session(session_id: int):
    st.session_state.active_session_id = session_id

# ---------------------------------------------------------------------------
# Core RAG pipeline
# ---------------------------------------------------------------------------

def do_ask(question: str, history: list, document_ids=None):
    from services.retrieval_service import retrieve
    from services.llm_service import generate_answer
    try:
        chunks = retrieve(question, document_ids=document_ids)
        if not chunks:
            return "I could not find this information in the uploaded documents.", []
        answer = generate_answer(question, chunks, history=history)
        return answer, chunks
    except Exception as exc:
        return f"Error: {exc}", []

def do_ingest(uploaded_file, category="general"):
    from services.loader import load_document
    from services.chunker import chunk_text
    from services.database_service import insert_document, insert_embeddings, delete_document
    from services.database_service import get_document_by_filename
    from config import UPLOAD_DIR

    dest_path = UPLOAD_DIR / uploaded_file.name
    if dest_path.exists():
        return False, f"'{uploaded_file.name}' already exists. Delete it first.", 0

    # Save file to disk
    with open(dest_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    doc_id = None
    try:
        # Step 1 — extract text
        text = load_document(str(dest_path))

        # Step 2 — chunk
        chunks = chunk_text(text)

        # Step 3 — insert document row
        doc_id = insert_document(
            filename=uploaded_file.name,
            filepath=str(dest_path.resolve()),
            category=category,
        )

        # Step 4 — insert chunks (embedding=None, filled in Phase 3)
        insert_embeddings(document_id=doc_id, chunks=chunks)

        # Step 5 — rebuild BM25 index (best-effort, non-fatal)
        try:
            from services.bm25_service import build_index
            build_index()
        except Exception:
            pass

        return True, f"Ingested '{uploaded_file.name}' → {len(chunks)} chunks stored.", len(chunks)

    except Exception as exc:
        # Clean up: remove file and document row so state stays consistent
        dest_path.unlink(missing_ok=True)
        if doc_id is not None:
            try:
                delete_document(doc_id)
            except Exception:
                pass
        return False, f"Ingestion failed: {str(exc)}", 0

def do_embed_all():
    import psycopg2.extras
    from services.embedding_service import embed_chunks
    from services.database_service import _get_connection

    with _get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT e.id, e.chunk_text
                FROM embeddings e WHERE e.embedding IS NULL
                ORDER BY e.document_id, e.chunk_number;
            """)
            rows = [dict(r) for r in cur.fetchall()]

    if not rows:
        return 0, None

    chunks = [{"id": r["id"], "chunk_text": r["chunk_text"], "embedding": None} for r in rows]
    try:
        embed_chunks(chunks)
    except RuntimeError as exc:
        return 0, str(exc)

    with _get_connection() as conn:
        with conn.cursor() as cur:
            for c in chunks:
                cur.execute("UPDATE embeddings SET embedding = %s WHERE id = %s;",
                            (c["embedding"], c["id"]))
    try:
        from services.bm25_service import build_index
        build_index()
    except Exception:
        pass
    return len(chunks), None

def do_delete_doc(doc_id, filename):
    from services.database_service import delete_document
    from config import UPLOAD_DIR
    from pathlib import Path
    deleted = delete_document(doc_id)
    if deleted:
        (UPLOAD_DIR / filename).unlink(missing_ok=True)
        return True, f"'{filename}' deleted."
    return False, f"Could not delete document id={doc_id}."

# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------

health = init_system()
db_ok  = health.get("connected") and health.get("pgvector_ready")

with st.sidebar:
    st.markdown("## 🧠 Knowledge Assistant")
    st.markdown("*Local · Private · CPU-only*")

    # System status
    db_html = '<span class="status-ok">● Connected</span>' if db_ok else '<span class="status-warn">● Disconnected</span>'
    st.markdown("**System**", unsafe_allow_html=False)
    st.markdown("DB: " + db_html, unsafe_allow_html=True)

    if db_ok:
        try:
            total, embedded, pending = get_embedding_status()
            ec = "status-ok" if pending == 0 else "status-warn"
            el = "● Ready" if pending == 0 else f"● {pending} pending"
            st.markdown("Embeddings: " + f'<span class="{ec}">{el}</span>', unsafe_allow_html=True)
        except Exception:
            pass

    st.markdown("---")

    # Navigation
    st.markdown("**Navigation**")
    page = st.radio(
        "page", ["💬 Chat", "📂 My Documents", "⬆️ Upload"],
        label_visibility="collapsed"
    )

    # Chat session list (only shown on Chat page)
    if page == "💬 Chat":
        st.markdown("---")
        st.markdown("**Conversations**")

        # New chat button
        if st.button("＋ New Chat", use_container_width=True):
            from services.chat_service import create_session
            session = create_session("New Chat")
            st.session_state.active_session_id = session["id"]
            st.rerun()

        # Session list
        sessions = load_sessions()
        ensure_active_session()

        for s in sessions:
            active = s["id"] == st.session_state.get("active_session_id")
            col1, col2 = st.columns([5, 1])
            with col1:
                label = ("▶ " if active else "") + s["title"]
                if st.button(label, key=f"sess_{s['id']}", use_container_width=True,
                             help=f"{s['message_count']} messages"):
                    switch_session(s["id"])
                    st.rerun()
            with col2:
                if st.button("🗑", key=f"del_sess_{s['id']}", help="Delete session"):
                    from services.chat_service import delete_session
                    delete_session(s["id"])
                    if st.session_state.get("active_session_id") == s["id"]:
                        st.session_state.active_session_id = None
                    st.rerun()

    st.markdown("---")
    st.markdown(
        "<small style='color:#484f58'>Powered by Ollama · pgvector</small>",
        unsafe_allow_html=True
    )

# ---------------------------------------------------------------------------
# PAGE: CHAT
# ---------------------------------------------------------------------------

if page == "💬 Chat":

    ensure_active_session()
    session_id = st.session_state.active_session_id

    # Load session info
    from services.chat_service import (
        get_session, get_messages, get_history_buffer,
        add_message, auto_title_session, clear_session_messages
    )

    session     = get_session(session_id)
    messages    = get_messages(session_id)

    # Session header
    col_title, col_clear = st.columns([6, 1])
    with col_title:
        st.markdown(f"### {session['title'] if session else 'New Chat'}")
    with col_clear:
        if messages and st.button("🗑 Clear", help="Clear this chat's messages"):
            clear_session_messages(session_id)
            st.rerun()

    # Document scope selector
    docs = get_documents()
    selected_doc_ids = None
    if docs:
        with st.expander("🔍 Search scope (optional)", expanded=False):
            scope_mode = st.radio(
                "scope", ["All documents", "Selected documents only"],
                horizontal=True, label_visibility="collapsed"
            )
            if scope_mode == "Selected documents only":
                doc_options = {d["filename"]: d["id"] for d in docs}
                selected_names = st.multiselect(
                    "Choose documents",
                    options=list(doc_options.keys()),
                    default=list(doc_options.keys()),
                )
                if selected_names:
                    selected_doc_ids = [doc_options[n] for n in selected_names]

    st.markdown("---")

    # Render chat history
    for msg in messages:
        if msg["role"] == "user":
            st.markdown('<div class="role-label-user">You</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="user-bubble">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="role-label-assistant">Assistant</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="assistant-bubble">{msg["content"]}</div>', unsafe_allow_html=True)

            # Show sources if available
            if msg.get("sources"):
                with st.expander(f"📎 {len(msg['sources'])} source(s) used", expanded=False):
                    for i, src in enumerate(msg["sources"], 1):
                        score_str = f"reranker: {src.get('reranker_score', 0):.3f}"
                        preview   = src.get("chunk_text", "")[:300].replace("\n", " ")
                        if len(src.get("chunk_text", "")) > 300:
                            preview += "..."
                        st.markdown(
                            f'<div class="source-card">'
                            f'<strong>[{i}] {src["filename"]} — chunk {src["chunk_number"]} ({score_str})</strong><br>'
                            f'{preview}'
                            f'</div>',
                            unsafe_allow_html=True
                        )

    # Input area
    st.markdown("---")

    if not db_ok:
        st.error("Database not connected. Check PostgreSQL is running.")
    else:
        with st.form("chat_form", clear_on_submit=True):
            col_input, col_send = st.columns([6, 1])
            with col_input:
                question = st.text_input(
                    "Message",
                    placeholder="Ask anything about your documents...",
                    label_visibility="collapsed"
                )
            with col_send:
                send = st.form_submit_button("Send →", use_container_width=True)

        if send and question.strip():
            q = question.strip()

            # Get history buffer for context
            history = get_history_buffer(session_id)

            # Save user message
            add_message(session_id, "user", q)

            # Auto-title session on first message
            if not messages:
                auto_title_session(session_id, q)

            # Run RAG pipeline
            with st.spinner("Thinking..."):
                answer, chunks = do_ask(q, history, document_ids=selected_doc_ids)

            # Save assistant message with sources
            add_message(session_id, "assistant", answer, sources=chunks)

            # Show routing info if available
            if chunks:
                routing = chunks[0].get("routing", {})
                if routing.get("routed"):
                    from services.router_service import get_category_description
                    cat = get_category_description(routing["category"])
                    st.caption(f"Auto-routed → {cat} (confidence: {routing['confidence']:.1f})")

            st.rerun()

# ---------------------------------------------------------------------------
# PAGE: MY DOCUMENTS
# ---------------------------------------------------------------------------

elif page == "📂 My Documents":
    st.markdown("## 📂 My Documents")
    st.markdown("---")

    col1, _ = st.columns([2, 5])
    with col1:
        if st.button("⚡ Generate Embeddings", use_container_width=True):
            with st.spinner("Generating embeddings via Ollama..."):
                count, error = do_embed_all()
            if error:
                st.error(f"❌ {error}")
            elif count == 0:
                st.success("✅ All chunks already embedded.")
            else:
                st.success(f"✅ Generated embeddings for {count} chunks.")
                st.rerun()

    st.markdown("---")
    docs = get_documents()

    if not docs:
        st.info("No documents uploaded yet. Go to **Upload** to get started.")
    else:
        st.markdown(f"**{len(docs)} document(s) in knowledge base**")
        st.markdown("")

        for doc in docs:
            c1, c2, c3 = st.columns([5, 2, 1])
            with c1:
                from services.router_service import get_category_description
                cat_label = get_category_description(doc.get("category", "general"))
                st.markdown(f"**📄 {doc['filename']}**")
                st.markdown(
                    f"<small style='color:#666'>"
                    f"Uploaded {doc['upload_time'].strftime('%Y-%m-%d %H:%M')} · {cat_label}"
                    f"</small>",
                    unsafe_allow_html=True
                )
            with c2:
                st.markdown(f"<small style='color:#555'>ID: {doc['id']}</small>", unsafe_allow_html=True)
            with c3:
                if st.button("🗑", key=f"del_doc_{doc['id']}", help=f"Delete {doc['filename']}"):
                    ok, msg = do_delete_doc(doc["id"], doc["filename"])
                    if ok:
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)
            st.markdown("---")

# ---------------------------------------------------------------------------
# PAGE: UPLOAD
# ---------------------------------------------------------------------------

elif page == "⬆️ Upload":
    st.markdown("## ⬆️ Upload Documents")
    st.markdown("Upload PDF, TXT, PPTX, PPT, or DOCX files.")
    st.markdown("---")

    uploaded_files = st.file_uploader(
        "Choose files",
        type=["pdf", "txt", "pptx", "ppt", "docx"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        from services.router_service import get_all_categories, get_category_description
        cat_options = ["general"] + get_all_categories()
        cat_labels  = {c: get_category_description(c) for c in cat_options}
        selected_cat = st.selectbox(
            "Document category",
            options=cat_options,
            format_func=lambda c: cat_labels[c],
            help="Categorised documents are searched first for matching queries",
        )

        if st.button("⬆️ Ingest Selected Files", type="primary"):
            results = []
            for f in uploaded_files:
                with st.spinner(f"Processing {f.name}..."):
                    ok, msg, count = do_ingest(f, category=selected_cat)
                    results.append((f.name, ok, msg))

            for name, ok, msg in results:
                if ok:
                    st.success(f"✅ **{name}** — {msg}")
                else:
                    st.error(f"❌ **{name}** — {msg}")

            if any(r[1] for r in results):
                st.info("📌 Go to **My Documents → Generate Embeddings** to make files searchable.")

    st.markdown("---")
    st.markdown("**Supported formats**")
    st.markdown("""
| Format | Notes |
|--------|-------|
| PDF    | Text-based only (no scanned/image PDFs) |
| TXT    | UTF-8 or auto-detected encoding |
| PPTX / PPT | Text slides only |
| DOCX   | Paragraphs and table text |
""")