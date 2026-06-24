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
    return get_all_sessions(1)

def ensure_active_session():
    """Create a new session if none is active."""
    if "active_session_id" not in st.session_state or st.session_state.active_session_id is None:
        from services.chat_service import create_session
        session = create_session(1, "New Chat")
        st.session_state.active_session_id = session["id"]

def switch_session(session_id: int):
    st.session_state.active_session_id = session_id

# ---------------------------------------------------------------------------
# Core RAG pipeline
# ---------------------------------------------------------------------------

def do_ask(question: str, history: list, document_ids=None):
    """
    Full answer pipeline — routes to the correct handler:
      1. Metadata question   → metadata_service (instant, no RAG)
      2. Excel lookup        → excel_service row search
      3. Excel aggregation   → excel_service text-to-SQL
      4. Everything else     → hybrid RAG pipeline
    """
    from services.metadata_service import is_metadata_question, answer_metadata_question
    from services.excel_service import detect_excel_intent, search_rows, answer_aggregation
    from services.database_service import _get_connection, get_all_documents

    # --- Check if any Excel files are in scope ---
    try:
        import psycopg2.extras
        with _get_connection() as conn:
            with conn.cursor() as cur:
                if document_ids:
                    cur.execute(
                        "SELECT COUNT(*) FROM excel_rows WHERE document_id = ANY(%s);",
                        (document_ids,)
                    )
                else:
                    cur.execute("SELECT COUNT(*) FROM excel_rows;")
                excel_row_count = cur.fetchone()[0]
    except Exception:
        excel_row_count = 0

    # --- Route: metadata question ---
    if is_metadata_question(question):
        result = answer_metadata_question(question)
        if result["answered"]:
            return result["response"], [], {"type": "metadata", "data": result["data"]}

    # --- Route: Excel question (if Excel data exists) ---
    if excel_row_count > 0:
        intent = detect_excel_intent(question)

        if intent == "lookup":
            rows = search_rows(question, document_ids=document_ids, top_k=20)
            if rows:
                return "", rows, {"type": "excel_rows"}

        elif intent == "aggregation":
            result = answer_aggregation(question)
            if result["answered"]:
                return result["response"], result["result_rows"], {
                    "type": "excel_aggregation",
                    "sql": result.get("sql", "")
                }

    # --- Route: RAG pipeline (default) ---
    from services.retrieval_service import retrieve
    from services.llm_service import generate_answer
    try:
        chunks = retrieve(question, document_ids=document_ids)
        if not chunks:
            return "I could not find this information in the uploaded documents.", [], {}
        answer = generate_answer(question, chunks, history=history)
        return answer, chunks, {"type": "rag"}
    except Exception as exc:
        return f"Error: {exc}", [], {}

def do_ingest(uploaded_file, category="general"):
    from services.loader import load_document
    from services.chunker import chunk_text
    from services.database_service import insert_document, insert_embeddings, delete_document
    from services.metadata_service import extract_metadata
    from config import UPLOAD_DIR, EXCEL_EXTENSIONS

    dest_path = UPLOAD_DIR / uploaded_file.name
    ext       = uploaded_file.name.lower().rsplit(".", 1)[-1]

    if dest_path.exists():
        return False, f"'{uploaded_file.name}' already exists. Delete it first.", 0

    with open(dest_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    doc_id = None
    try:
        import io as _io, gc, time

        # Read file bytes ONCE into memory — prevents Windows double file-lock.
        # openpyxl on Windows holds a handle even after wb.close(); reading to
        # BytesIO releases the OS handle immediately so ingest_excel can open cleanly.
        raw_bytes  = dest_path.read_bytes()
        file_bytes = _io.BytesIO(raw_bytes)

        # Insert document row
        doc_id = insert_document(
            filename=uploaded_file.name,
            filepath=str(dest_path.resolve()),
            category=category,
        )

        # Excel → row storage (no embedding needed)
        if f".{ext}" in EXCEL_EXTENSIONS:
            from services.excel_service import ingest_excel

            # Extract metadata — uses BytesIO internally, no double file open
            try:
                extract_metadata(str(dest_path.resolve()), doc_id)
            except Exception:
                pass

            # Ingest rows from BytesIO
            file_bytes.seek(0)
            info = ingest_excel(file_bytes, doc_id)
            return (
                True,
                f"Excel ingested: {info['sheet_count']} sheet(s), "
                f"{info['total_rows']} rows stored.",
                info["total_rows"],
            )

        # All other formats → RAG pipeline
        try:
            extract_metadata(str(dest_path.resolve()), doc_id)
        except Exception:
            pass

        text   = load_document(str(dest_path))
        chunks = chunk_text(text)
        insert_embeddings(document_id=doc_id, chunks=chunks)

        try:
            from services.bm25_service import build_index
            build_index()
        except Exception:
            pass

        return True, f"Ingested '{uploaded_file.name}' → {len(chunks)} chunks stored.", len(chunks)

    except Exception as exc:
        # Best-effort cleanup — handle Windows file lock gracefully
        try:
            gc.collect()
            time.sleep(0.3)
            dest_path.unlink(missing_ok=True)
        except PermissionError:
            pass  # file still locked by OS — leave it, do not crash
        except Exception:
            pass

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
        "page", ["💬 Chat", "📊 Data Explorer", "📂 My Documents", "⬆️ Upload"],
        label_visibility="collapsed"
    )

    # Chat session list (only shown on Chat page)
    if page == "💬 Chat":
        st.markdown("---")
        st.markdown("**Conversations**")

        # New chat button
        if st.button("＋ New Chat", use_container_width=True):
            from services.chat_service import create_session
            session = create_session(1, "New Chat")
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
                    delete_session(1, s["id"])
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

    session     = get_session(1, session_id)
    messages    = get_messages(1, session_id)

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

            history = get_history_buffer(session_id)
            add_message(session_id, "user", q)

            if not messages:
                auto_title_session(1, session_id, q)

            with st.spinner("Thinking..."):
                answer, data, meta = do_ask(q, history, document_ids=selected_doc_ids)

            answer_type = meta.get("type", "rag") if meta else "rag"

            if answer_type == "metadata":
                # Metadata answer — plain text
                add_message(session_id, "assistant", answer)

            elif answer_type == "excel_rows":
                # Excel row lookup — show as table
                if data:
                    import json
                    rows_display = [
                        r["row_data"] if isinstance(r["row_data"], dict)
                        else json.loads(r["row_data"])
                        for r in data
                    ]
                    summary = (
                        f"Found {len(data)} matching row(s) across "
                        f"{len(set(r['sheet_name'] for r in data))} sheet(s)."
                    )
                    add_message(session_id, "assistant", summary)
                else:
                    add_message(session_id, "assistant",
                                "No matching rows found in the Excel files.")

            elif answer_type == "excel_aggregation":
                add_message(session_id, "assistant", answer or "Aggregation complete.")

            else:
                # RAG answer
                add_message(session_id, "assistant", answer,
                            sources=data if data else None)
                if data:
                    routing = data[0].get("routing", {})
                    if routing.get("routed"):
                        from services.category_service import get_category_description
                        cat = get_category_description(routing["category"])
                        st.caption(
                            f"Auto-routed → {cat} "
                            f"(confidence: {routing['confidence']:.1f})"
                        )

            # Show Excel table result inline (before rerun)
            if answer_type == "excel_rows" and data:
                import json
                rows_display = []
                for r in data:
                    rd = r["row_data"] if isinstance(r["row_data"], dict)                          else json.loads(r["row_data"])
                    rd["_sheet"] = r["sheet_name"]
                    rd["_file"]  = r.get("filename", "")
                    rows_display.append(rd)
                st.dataframe(rows_display, use_container_width=True)

            st.rerun()


# ---------------------------------------------------------------------------
# PAGE: DATA EXPLORER
# ---------------------------------------------------------------------------

elif page == "📊 Data Explorer":
    st.markdown("## 📊 Data Explorer")
    st.markdown("Upload any CSV or Excel file and explore it with natural language — no embeddings needed.")
    st.markdown("---")

    # File uploader
    exp_file = st.file_uploader(
        "Upload CSV or Excel for exploration",
        type=["csv", "xlsx", "xls"],
        key="explorer_upload",
        help="This file is used for this session only — not stored in the knowledge base"
    )

    if exp_file:
        from services.data_explorer_service import (
            load_file, build_schema_context, answer_query
        )

        # Load file into session state (reload only when file changes)
        if (st.session_state.get("explorer_filename") != exp_file.name
                or "explorer_df" not in st.session_state):
            df, load_err = load_file(exp_file, exp_file.name)
            if load_err:
                st.error(f"❌ {load_err}")
                st.stop()
            st.session_state["explorer_df"]       = df
            st.session_state["explorer_filename"]  = exp_file.name
            st.session_state["explorer_schema"]    = build_schema_context(df, exp_file.name)
            st.session_state["explorer_history"]   = []
            st.success(f"✅ Loaded **{exp_file.name}** — {df.shape[0]} rows × {df.shape[1]} columns")

        df            = st.session_state["explorer_df"]
        schema_ctx    = st.session_state["explorer_schema"]
        history       = st.session_state.get("explorer_history", [])

        # Schema preview
        with st.expander("📋 Data Preview & Schema", expanded=False):
            tab1, tab2, tab3 = st.tabs(["Preview", "Schema", "Statistics"])
            with tab1:
                st.dataframe(df.head(10), use_container_width=True)
            with tab2:
                import pandas as pd
                schema_df = pd.DataFrame({
                    "Column":    df.columns,
                    "Type":      [str(df[c].dtype) for c in df.columns],
                    "Non-Null":  [df[c].count() for c in df.columns],
                    "Unique":    [df[c].nunique() for c in df.columns],
                    "Null %":    [round(df[c].isnull().mean() * 100, 1) for c in df.columns],
                })
                st.dataframe(schema_df, use_container_width=True)
            with tab3:
                st.dataframe(df.describe(include="all"), use_container_width=True)

        st.markdown("---")

        # Render history
        for item in history:
            # User query
            st.markdown(f'<div class="role-label-user">You</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="user-bubble">{item["query"]}</div>', unsafe_allow_html=True)

            # Assistant result
            st.markdown(f'<div class="role-label-assistant">Explorer ({item["mode"]})</div>',
                        unsafe_allow_html=True)

            if item.get("friendly_error"):
                # Show partial result if fallback succeeded
                if item.get("fallback") and item.get("result") is not None:
                    st.warning(
                        f"⚠️ Automatic analysis used instead: {item['friendly_error']}"
                    )
                else:
                    st.error(f"❌ {item['friendly_error']}")
                    st.info("💡 Try rephrasing, or use a Quick Query chip below for instant results.")
            elif item.get("error"):
                st.error(f"❌ {item['error']}")
            if item.get("fallback") and not item.get("friendly_error"):
                st.caption("⟳ Fell back to automatic statistical analysis.")
            if item["result_type"] == "image" and item.get("image_bytes"):
                pass  # handled below
            elif False:
                pass  # placeholder to fix elif chain
            if item["result_type"] == "image" and item.get("image_bytes"):
                st.image(item["image_bytes"], use_container_width=True)
            elif item["result_type"] == "dataframe":
                from services.data_explorer_service import get_result_notice
                notice = get_result_notice(item["result"])
                if notice:
                    st.caption(f"⚠️ {notice}")
                st.dataframe(item["result"], use_container_width=True)
            elif item["result_type"] in ("text", "list", "series"):
                st.markdown(
                    f'<div class="assistant-bubble">{str(item["result"])}</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="assistant-bubble">{str(item.get("result", "Done."))}</div>',
                    unsafe_allow_html=True
                )

            # Show generated code (collapsible)
            if item.get("code") and item["mode"] != "statistical":
                with st.expander("🔍 Generated code", expanded=False):
                    st.code(item["code"], language="python")
                if item.get("retried"):
                    st.caption("⟳ Retried once after initial error.")

        st.markdown("---")

        # Query input
        col_q, col_btn, col_clr = st.columns([5, 1, 1])
        with col_q:
            exp_query = st.text_input(
                "Ask anything about the data",
                placeholder='e.g. "show top 10 by sales", "plot a bar chart", "how many nulls?"',
                key="explorer_query",
                label_visibility="collapsed",
            )
        with col_btn:
            run_btn = st.button("Run →", type="primary", use_container_width=True)
        with col_clr:
            if st.button("🗑 Clear", use_container_width=True):
                st.session_state["explorer_history"] = []
                st.rerun()

        if run_btn and exp_query.strip():
            with st.spinner("Analysing..."):
                result_dict = answer_query(
                    exp_query.strip(), df, schema_ctx, exp_file.name
                )

            # Append to history
            history.append({
                "query":       exp_query.strip(),
                "mode":        result_dict["mode"],
                "result":      result_dict["result"],
                "result_type": result_dict["result_type"],
                "code":        result_dict.get("code"),
                "image_bytes": result_dict.get("image_bytes"),
                "error":       result_dict.get("error"),
                "retried":     result_dict.get("retried", False),
            })
            st.session_state["explorer_history"] = history
            st.rerun()

        # Quick query chips
        st.markdown("**Quick queries:**")
        chips = [
            "Describe the data", "Show first 10 rows",
            "How many null values?", "Show correlation matrix",
            "Value counts for each column", "Show data types",
        ]
        chip_cols = st.columns(3)
        for i, chip in enumerate(chips):
            with chip_cols[i % 3]:
                if st.button(chip, key=f"chip_{i}", use_container_width=True):
                    with st.spinner("Analysing..."):
                        result_dict = answer_query(chip, df, schema_ctx, exp_file.name)
                    history.append({
                        "query":       chip,
                        "mode":        result_dict["mode"],
                        "result":      result_dict["result"],
                        "result_type": result_dict["result_type"],
                        "code":        result_dict.get("code"),
                        "image_bytes": result_dict.get("image_bytes"),
                        "error":       result_dict.get("error"),
                        "retried":     result_dict.get("retried", False),
                    })
                    st.session_state["explorer_history"] = history
                    st.rerun()

    else:
        # No file uploaded yet — show instructions
        st.markdown("### How to use the Data Explorer")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**📤 Upload**")
            st.markdown("Upload any CSV or Excel file above. It stays in your session only.")
        with col2:
            st.markdown("**💬 Ask**")
            st.markdown("Type questions in plain English. Describe, filter, aggregate, chart.")
        with col3:
            st.markdown("**🔍 Inspect**")
            st.markdown("See the generated pandas code for every answer — full transparency.")

        st.markdown("---")
        st.markdown("**Example queries you can ask:**")
        examples = [
            ("Statistical", "How many null values are there?", "Instant — no LLM needed"),
            ("Statistical", "Show me the correlation matrix", "Instant — no LLM needed"),
            ("Statistical", "What are the unique values in Status?", "Instant — no LLM needed"),
            ("Code", "Show me all rows where Amount > 5000", "LLM generates pandas filter"),
            ("Code", "Group by Category and show total sales", "LLM generates groupby code"),
            ("Code", "Find the top 5 customers by revenue", "LLM generates sort + head"),
            ("Visualization", "Plot a bar chart of sales by region", "LLM generates matplotlib"),
            ("Visualization", "Show a histogram of Amount column", "LLM generates histogram"),
        ]
        import pandas as pd
        ex_df = pd.DataFrame(examples, columns=["Mode", "Query", "How it works"])
        st.dataframe(ex_df, use_container_width=True, hide_index=True)

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
                from services.category_service import get_category_description
                cat_label = get_category_description(doc.get("category", "general"))
                st.markdown(f"**📄 {doc['filename']}**")
                st.markdown(
                    f"<small style='color:#666'>"
                    f"Uploaded {doc['upload_time'].strftime('%Y-%m-%d %H:%M')} · {cat_label}"
                    f"</small>",
                    unsafe_allow_html=True
                )
            with c2:
                # Show row count for Excel docs, chunk count for RAG docs
                try:
                    import psycopg2
                    from services.database_service import _get_connection
                    with _get_connection() as conn:
                        with conn.cursor() as cur:
                            cur.execute(
                                "SELECT COUNT(*) FROM excel_rows WHERE document_id = %s;",
                                (doc["id"],)
                            )
                            excel_rows = cur.fetchone()[0]
                    if excel_rows > 0:
                        st.markdown(
                            f"<small style='color:#555'>{excel_rows} rows (Excel)</small>",
                            unsafe_allow_html=True
                        )
                    else:
                        with _get_connection() as conn:
                            with conn.cursor() as cur:
                                cur.execute(
                                    "SELECT COUNT(*) FROM embeddings WHERE document_id = %s;",
                                    (doc["id"],)
                                )
                                chunk_count = cur.fetchone()[0]
                        st.markdown(
                            f"<small style='color:#555'>{chunk_count} chunks</small>",
                            unsafe_allow_html=True
                        )
                except Exception:
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
        type=["pdf", "txt", "pptx", "ppt", "docx", "xlsx", "xls"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        from services.category_service import list_categories, create_category
        cats = list_categories(include_doc_counts=False)
        cat_options = [c["name"] for c in cats]
        cat_labels  = {c["name"]: c["label"] for c in cats}

        col_cat, col_new = st.columns([3, 2])
        with col_cat:
            selected_cat = st.selectbox(
                "Document category",
                options=cat_options,
                format_func=lambda c: cat_labels.get(c, c),
                help="Categorised documents are searched first for matching queries",
            )
        with col_new:
            new_cat_label = st.text_input(
                "Or create new category",
                placeholder="e.g. Quality Control",
                help="Type a name to create a custom category and use it for this upload",
            )
        # If user typed a new category name, that takes priority
        if new_cat_label.strip():
            cat_obj    = create_category(new_cat_label.strip())
            selected_cat = cat_obj["name"]
            st.success(f"Category created: **{cat_obj['label']}**")

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
| Format | Type | Notes |
|--------|------|-------|
| PDF    | RAG  | Text-based only (no scanned/image PDFs) |
| TXT    | RAG  | UTF-8 or auto-detected encoding |
| PPTX / PPT | RAG | Text slides only |
| DOCX   | RAG  | Paragraphs and table text |
| XLSX / XLS | **Tabular** | Row-level storage — supports lookup, aggregation, metadata queries |
""")

    st.info("Excel files are stored differently — each row is indexed for lookup and aggregation. Ask: Is order 1042 in the sheet? / How many pending invoices? / What columns are in the sales file?")