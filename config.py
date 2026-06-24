"""
config.py
---------
Central configuration for the Local Employee Knowledge Assistant.
All tuneable parameters live here — no magic numbers scattered across the codebase.
To switch models, change DB credentials, or adjust retrieval behavior,
edit ONLY this file.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent.resolve()
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Database — PostgreSQL + pgvector
# ---------------------------------------------------------------------------

DB_CONFIG = {
    "host":     os.getenv("DB_HOST",     "localhost"),
    "port":     int(os.getenv("DB_PORT", "5432")),
    "database": os.getenv("DB_NAME"),
    "user":     os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
}

# ---------------------------------------------------------------------------
# Embedding Model
# ---------------------------------------------------------------------------
# Swap the model name here to change the embedding model globally.
# The SAME model is always used for both document and query embeddings.
#
# Supported local Ollama embedding models and their output dimensions:
#   mxbai-embed-large   → 1024   (default, best quality on CPU)
#   nomic-embed-text    → 768
#   bge-small           → 384
#   all-MiniLM-L6-v2   → 384
#
# IMPORTANT: If you change EMBEDDING_MODEL you MUST also update
# EMBEDDING_DIMENSION to match, and re-embed all existing documents.
# ---------------------------------------------------------------------------

EMBEDDING_MODEL     = os.getenv("EMBEDDING_MODEL",     "mxbai-embed-large:latest")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "1024"))

# ---------------------------------------------------------------------------
# LLM Model
# ---------------------------------------------------------------------------
# Swap the model name here to change the LLM globally.
#
# Recommended CPU-only models (via Ollama):
#   qwen3.5:0.8b    (default — good quality, ~2GB RAM)
#   phi3:mini     (faster, slightly lower quality)
#   llama3.2:3b   (alternative)
# ---------------------------------------------------------------------------

LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2:latest")

# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE",    "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

TOP_K = int(os.getenv("TOP_K", "3"))   # Number of chunks returned per query

# ---------------------------------------------------------------------------
# Document Upload
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".ppt", ".pptx", ".docx"}
EXCEL_EXTENSIONS     = {".xlsx", ".xls"}


# ---------------------------------------------------------------------------
# Prompt Template
# ---------------------------------------------------------------------------
# Modify this to change how context is presented to the LLM.

PROMPT_TEMPLATE = """You are a helpful assistant that answers questions using the provided context.
 
Guidelines:
- Use the context to answer the question, even if the information appears as a list, table, or fragmented text.
- Synthesise and present the answer clearly in your own words.
- If the context contains partial information, use what is available and be clear about what is partial.
- Only say you cannot find the answer if the context has genuinely no relevant information at all.
 
Context:
{context}
 
Question:
{question}
 
Answer:"""