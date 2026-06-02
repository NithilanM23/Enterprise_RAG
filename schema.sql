-- =============================================================================
-- schema.sql
-- Local Employee Knowledge Assistant — Database Schema
-- =============================================================================
-- Run this file once to initialize the database.
-- Safe to re-run: uses IF NOT EXISTS everywhere.
--
-- Prerequisites:
--   1. PostgreSQL is installed and running
--   2. The target database already exists
--      (created by database_service.py or manually)
--   3. pgvector extension is available
--      (comes with pgvector package installation)
-- =============================================================================

-- Enable pgvector extension (idempotent)
CREATE EXTENSION IF NOT EXISTS vector;

-- =============================================================================
-- Table: documents
-- Tracks every uploaded file — one row per file.
-- =============================================================================

CREATE TABLE IF NOT EXISTS documents (
    id          SERIAL PRIMARY KEY,
    filename    TEXT        NOT NULL,
    filepath    TEXT        NOT NULL,
    upload_time TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT documents_filename_unique UNIQUE (filename)
);

-- =============================================================================
-- Table: embeddings
-- Stores every chunk derived from a document, along with its vector embedding.
-- One document → many embeddings (one per chunk).
-- =============================================================================

CREATE TABLE IF NOT EXISTS embeddings (
    id           SERIAL PRIMARY KEY,
    document_id  INTEGER     NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_number INTEGER     NOT NULL,   -- 0-based index within the document
    chunk_text   TEXT        NOT NULL,
    embedding    vector(1024),           -- dimension matches EMBEDDING_DIMENSION in config.py

    CONSTRAINT embeddings_doc_chunk_unique UNIQUE (document_id, chunk_number)
);

-- =============================================================================
-- Index: HNSW index for approximate nearest-neighbour search
-- Requires pgvector 0.6.0+ (you have 0.8.2 — fully supported)
-- =============================================================================
-- HNSW is preferred over IVFFlat for this project because:
--   - Works at any dataset size (no minimum row count)
--   - No 'lists' parameter to tune
--   - Better recall at equivalent speed
--   - Maintained incrementally on INSERT — no rebuild needed
--
-- Created automatically by database_service.py on first document upload.
-- Listed here for documentation purposes.
--
-- CREATE INDEX IF NOT EXISTS embeddings_hnsw_idx
--     ON embeddings
--     USING hnsw (embedding vector_cosine_ops)
--     WITH (m = 16, ef_construction = 64);
-- =============================================================================