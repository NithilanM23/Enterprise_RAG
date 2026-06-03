# Local Employee Knowledge Assistant (Enterprise RAG)

A fully local, CPU-only RAG (Retrieval-Augmented Generation) system for internal employee use. Upload company documents and ask natural language questions — everything runs on your machine with no internet, no cloud, no GPU required.

---

## What It Does

Employees upload internal documents (PDFs, Word docs, PowerPoints, text files) and ask questions in plain English. The system retrieves the most relevant content and generates grounded answers — showing exactly which document and chunk the answer came from.

```
Employee asks: "Where is the company located?"
System finds:  Chunk 25 from company.docx — similarity 0.53
Answer:        "No K5, 2nd Floor, Vijaya Road, Mandaveli, Chennai – 600028"
Sources shown: filename · chunk number · relevance score
```

---

## Key Features

### Retrieval Pipeline
- **Hybrid Search** — Semantic search (pgvector) + BM25 keyword search run in parallel. Semantic handles paraphrased questions and synonyms. BM25 handles exact terms, product codes, IDs, and proper nouns.
- **Reciprocal Rank Fusion** — Merges both ranked lists without needing score normalisation. Chunks appearing in both lists get a consensus boost.
- **MMR Diversification** — Maximal Marginal Relevance prevents a single document from monopolising all result slots. Ensures diverse context reaches the LLM.
- **Cross-Encoder Reranker** — Scores (query, chunk) pairs together for true question-answer relevance, not just topic similarity. Filters noise before the LLM.

### Smart Routing
- **Automatic Query Routing** — Queries are classified by keyword scoring into categories (HR, Engineering, Finance, Company Info, Reference). Search is automatically scoped to the relevant documents — a 500-page ML textbook never pollutes answers about company location.
- **Retrieval Drift Prevention** — Category tagging + routing + MMR work together to ensure large unrelated documents don't bury relevant chunks.

### Document Handling
- **Multi-format** — PDF (text-based), TXT, PPTX, PPT, DOCX
- **Heading-aware chunking** — Section headings detected and prepended to each chunk. The LLM always knows what section a chunk belongs to.
- **Auto BM25 rebuild** — BM25 index rebuilds automatically on every document add or delete. No manual commands needed.

### Persistent Chat
- **Multi-session chat** — Multiple conversation sessions stored in PostgreSQL. Survives app restarts.
- **Buffer memory** — Last 6 messages injected into LLM prompt. Follow-up questions like "what about their HR policy?" work correctly.
- **Session management** — Create, rename, delete sessions from the UI sidebar.

### Infrastructure
- **PostgreSQL + pgvector** — Production-grade vector storage with HNSW index (works at any dataset size, no minimum rows, incremental updates).
- **BM25 precomputed index** — Serialised to disk, millisecond query time.
- **Modular models** — Swap embedding model or LLM by changing one line in `config.py`.
- **Graceful fallbacks** — BM25 unavailable → semantic only. Reranker unavailable → RRF order. Routing miss → global search.

---

## Architecture

### Ingestion Pipeline

```
File Upload  (PDF / TXT / PPTX / DOCX)
       |
  Text Extraction  ──  per-format loader with heading marker injection
       |
  Chunking  ──  RecursiveCharacterTextSplitter (chunk=500, overlap=100)
               Section headings prepended to each chunk
       |
  PostgreSQL  ──  documents table (filename, filepath, category)
               ── embeddings table (chunk_text, embedding=NULL)
       |
  Ollama Embedding  ──  mxbai-embed-large (1024-dim)  ──  fill embedding column
       |
  BM25 Auto-Rebuild  ──  bm25_index.pkl updated
```

### Query Pipeline

```
User Question
       |
  [1]  Smart Router  ─── keyword scoring → document category → scope search
       |
  [2]  Semantic Search ── pgvector cosine similarity  (top 20)
       |
  [3]  BM25 Search ────── keyword match  (top 20)
       |
  [4]  RRF Fusion ──────── merge both ranked lists  → 30 candidates
       |
  [5]  MMR ──────────────── diversity filter  → 20 diverse candidates
       |
  [6]  Cross-Encoder Reranker  ── score (query, chunk) pairs
       |                           filter: score < -6.0 dropped
  [7]  LLM  ─── Ollama local model ── grounded answer + sources
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Vector Database | PostgreSQL 18 + pgvector 0.8.2 |
| Vector Index | HNSW (Hierarchical Navigable Small World) |
| Embedding Model | mxbai-embed-large via Ollama (1024-dim, CPU) |
| LLM | llama3.2 / qwen variants via Ollama (CPU) |
| Keyword Search | BM25 (rank_bm25 library, precomputed) |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 (~80MB, CPU) |
| UI | Streamlit |
| Language | Python 3.11+ |

---

## Hardware Requirements

- Windows laptop (tested on Intel i5 12th Gen)
- 8–16 GB RAM
- No dedicated GPU — runs entirely on CPU
- No internet required after initial model downloads

---

## Prerequisites

### 1. PostgreSQL 18
Download from [postgresql.org](https://www.postgresql.org/download/windows/)

### 2. pgvector 0.8.2
Download prebuilt zip for PostgreSQL 18 from [pgvector releases](https://github.com/pgvector/pgvector/releases/tag/v0.8.2)

Copy files to PostgreSQL 18 install folder:
```
lib\vector.dll        →  C:\Program Files\PostgreSQL\18\lib\
share\extension\*     →  C:\Program Files\PostgreSQL\18\share\extension\
```

Then enable in psql:
```sql
CREATE EXTENSION vector;
```

### 3. Ollama
Download from [ollama.com](https://ollama.com) and pull the required models:
```bash
ollama pull mxbai-embed-large
ollama pull llama3.2
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/yourname/local-knowledge-assistant.git
cd local-knowledge-assistant

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Configuration

Edit `config.py` to set your database credentials and model preferences:

```python
DB_CONFIG = {
    "host":     "localhost",
    "port":     5432,
    "database": "employee_knowledge_db",
    "user":     "postgres",
    "password": "your_password_here",   # ← set this
}

EMBEDDING_MODEL     = "mxbai-embed-large"   # change to swap embedding model
EMBEDDING_DIMENSION = 1024                   # must match model output dimension

LLM_MODEL = "llama3.2"                      # change to swap LLM
```

---

## Running the App

```bash
# Start the Streamlit UI
streamlit run app.py
```

Opens at `http://localhost:8501`

### CLI Commands (alternative to UI)

```bash
python main.py                                  # health check
python main.py --ingest path/to/file.pdf        # ingest a document
python main.py --ingest file.pdf --category hr  # ingest with category
python main.py --embed                          # generate embeddings
python main.py --ask "your question here"       # ask a question
python main.py --list                           # list all documents
python main.py --delete filename.pdf            # delete a document
python main.py --set-category file.pdf hr       # update document category
python main.py --build-index                    # manually rebuild BM25 index
python main.py --status                         # embedding completion status
```

---

## Usage Guide

### 1. Upload Documents
Go to **⬆️ Upload** in the sidebar. Select files and choose a category:

| Category | Use for |
|---|---|
| Company Information | About us, contact, overview, products |
| HR & People | Policies, leave, salary, onboarding |
| Engineering & Technical | SOPs, manuals, specifications |
| Finance & Accounts | Reports, budgets, invoices |
| Reference Materials | Textbooks, research papers, background reading |
| General | Everything else |

### 2. Generate Embeddings
Go to **📂 My Documents** → click **Generate Embeddings**. This runs the embedding model on all stored chunks. Required before asking questions.

### 3. Ask Questions
Go to **💬 Chat**. Type your question and press Send. The system:
- Routes your query to the right document category automatically
- Retrieves the most relevant chunks using hybrid search
- Reranks for precision
- Generates a grounded answer showing sources

### 4. Manage Sessions
The sidebar shows all your chat sessions. Click any session to resume it. Use **＋ New Chat** to start a fresh session. Click 🗑 to delete.

---

## Project Structure

```
local_knowledge_assistant/
├── app.py                      Streamlit UI (Chat, Documents, Upload)
├── main.py                     CLI entry point
├── config.py                   All configurable parameters
├── schema.sql                  Reference database schema
├── requirements.txt            Python dependencies
├── bm25_index.pkl              Auto-generated BM25 index
├── uploads/                    Uploaded documents stored here
│
└── services/
    ├── loader.py               Text extraction — PDF, TXT, PPTX, DOCX
    ├── chunker.py              Section-aware chunking with heading enrichment
    ├── embedding_service.py    Ollama embedding generation
    ├── database_service.py     PostgreSQL + pgvector operations
    ├── bm25_service.py         BM25 index build and keyword search
    ├── router_service.py       Query classification and category routing
    ├── retrieval_service.py    Full hybrid pipeline orchestrator
    ├── mmr_service.py          MMR diversification algorithm
    ├── reranker_service.py     Cross-encoder reranking
    ├── llm_service.py          Ollama LLM answer generation
    └── chat_service.py         Persistent session and message management
```

---

## Swapping Models

Everything is modular. To change the embedding model:

```python
# config.py
EMBEDDING_MODEL     = "nomic-embed-text"   # ollama pull nomic-embed-text
EMBEDDING_DIMENSION = 768
```

To change the LLM:
```python
# config.py
LLM_MODEL = "phi3:mini"   # ollama pull phi3:mini
```

Re-ingest all documents after changing the embedding model (dimension change requires new embeddings).

---

## Document Categories & Routing

The router classifies each query by matching tokens against category keyword profiles. If confidence meets the threshold, search is automatically scoped to that category's documents.

To tag an existing document:
```bash
python main.py --set-category "company_overview.docx" company_info
```

If routing confidence is low, the system falls back to global search across all documents automatically.

---

## Planned Improvements

- Vision extraction (Qwen3.5 + pdf2image) for diagrams and charts
- SemanticChunker for better chunk quality on large structured documents
- RAGAS evaluation framework with golden dataset
- User feedback loop (thumbs up/down per answer)
- Multi-user support with role-based document access
- Electron desktop packaging with one-click installer

---

## Why Not ChatGPT or Microsoft Copilot

| | ChatGPT / Copilot | This System |
|---|---|---|
| Data leaves your network | Yes | **No** |
| Works offline | No | **Yes** |
| GPU required | No | **No** |
| Answers from your specific docs only | No | **Yes** |
| Shows exact source chunk | No | **Yes** |
| Per-query API cost | Yes | **No** |
| Vendor lock-in | Yes | **No** |

---

## License

Internal use — not for public distribution.
