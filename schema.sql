-- =============================================================================
-- schema.sql
-- Local Employee Knowledge Assistant — Complete Database Schema
-- =============================================================================
-- Reference file. All CREATE statements are idempotent (IF NOT EXISTS).
-- Tables are created automatically by initialize_database() on startup.
-- Manual execution is only needed for fresh setup or inspection.
--
-- Prerequisites:
--   PostgreSQL 18 + pgvector 0.8.2
-- =============================================================================

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- =============================================================================
-- CORE RAG TABLES
-- =============================================================================

-- documents
-- One row per uploaded file. Category drives smart query routing.
CREATE TABLE IF NOT EXISTS documents (
    id          SERIAL PRIMARY KEY,
    filename    TEXT        NOT NULL,
    filepath    TEXT        NOT NULL,
    upload_time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    category    TEXT        NOT NULL DEFAULT 'general',
    CONSTRAINT documents_filename_unique UNIQUE (filename)
);

-- embeddings
-- One row per text chunk. embedding column filled after Ollama embedding run.
-- embedding column is NULL until --embed is run.
CREATE TABLE IF NOT EXISTS embeddings (
    id           SERIAL PRIMARY KEY,
    document_id  INTEGER     NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_number INTEGER     NOT NULL,
    chunk_text   TEXT        NOT NULL,
    embedding    vector(1024),           -- dimension = EMBEDDING_DIMENSION in config.py
    CONSTRAINT embeddings_doc_chunk_unique UNIQUE (document_id, chunk_number)
);

-- HNSW index for approximate nearest-neighbour search
-- Created automatically after first bulk insert. Listed here for reference.
-- CREATE INDEX IF NOT EXISTS embeddings_hnsw_idx
--     ON embeddings
--     USING hnsw (embedding vector_cosine_ops)
--     WITH (m = 16, ef_construction = 64);

-- =============================================================================
-- CHAT TABLES
-- =============================================================================

-- chat_sessions
-- One row per conversation session. Persists across app restarts.
CREATE TABLE IF NOT EXISTS chat_sessions (
    id         SERIAL PRIMARY KEY,
    title      TEXT        NOT NULL DEFAULT 'New Chat',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- chat_messages
-- One row per message (user or assistant) within a session.
-- sources stored as JSONB: [{filename, chunk_number, chunk_text, reranker_score}]
CREATE TABLE IF NOT EXISTS chat_messages (
    id         SERIAL PRIMARY KEY,
    session_id INTEGER     NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    role       TEXT        NOT NULL CHECK (role IN ('user', 'assistant')),
    content    TEXT        NOT NULL,
    sources    JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- =============================================================================
-- METADATA TABLE
-- =============================================================================

-- document_metadata
-- Structural metadata extracted at ingestion time for all formats.
-- Key-value pairs: page_count, slide_count, row_count, column_names, etc.
-- Used to answer metadata questions without going through RAG pipeline.
CREATE TABLE IF NOT EXISTS document_metadata (
    id          SERIAL PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    key         TEXT    NOT NULL,
    value       TEXT    NOT NULL,
    CONSTRAINT document_metadata_unique UNIQUE (document_id, key)
);

-- =============================================================================
-- EXCEL TABLES
-- =============================================================================

-- excel_rows
-- One row per data row in an Excel/CSV file uploaded to the knowledge base.
-- row_data stored as JSONB: {"ColumnName": value, ...}
-- row_text is the stringified version for full-text search.
-- Supports: row lookup, aggregation queries, metadata questions.
-- Note: Excel files are NOT chunked/embedded — they use this table instead.
CREATE TABLE IF NOT EXISTS excel_rows (
    id          SERIAL PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    sheet_name  TEXT    NOT NULL,
    row_number  INTEGER NOT NULL,
    row_data    JSONB   NOT NULL,
    row_text    TEXT    NOT NULL,
    CONSTRAINT excel_rows_unique UNIQUE (document_id, sheet_name, row_number)
);

-- Full-text search index on row_text for fast keyword lookup
CREATE INDEX IF NOT EXISTS excel_rows_text_idx
    ON excel_rows USING gin(to_tsvector('english', row_text));

-- =============================================================================
-- SUMMARY
-- =============================================================================
-- Table              Purpose
-- ─────────────────  ──────────────────────────────────────────────────────
-- documents          Uploaded file registry (all formats)
-- embeddings         Text chunks + 1024-dim vectors (RAG pipeline)
-- chat_sessions      Persistent conversation sessions
-- chat_messages      Individual messages within sessions
-- document_metadata  Structural metadata (pages, rows, columns, etc.)
-- excel_rows         Excel/CSV row-level storage for tabular queries
-- =============================================================================