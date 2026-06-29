# Local Employee Knowledge Assistant

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
- **Modular models** — Swap embedding model or LLM dynamically via Admin UI.
- **Graceful fallbacks** — BM25 unavailable → semantic only. Reranker unavailable → RRF order. Routing miss → global search.

---

## Architecture

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
| Frontend | Next.js (React) |
| Backend | FastAPI (Python) |

---

## How to Run This Application

The setup process has been simplified into a few straightforward steps. 

### 1. Prerequisites
Ensure you have the following installed on your machine:
- **PostgreSQL 18** (with the `pgvector` extension enabled)
- **Ollama** (make sure to run `ollama pull mxbai-embed-large` and `ollama pull llama3.2`)
- **Python 3.11+**
- **Node.js** (for the frontend)

### 2. Configuration
Open `config.py` in the root folder and add your database password:
```python
DB_CONFIG = {
    "host":     "localhost",
    "port":     5432,
    "database": "employee_knowledge_db",
    "user":     "postgres",
    "password": "your_password_here",   # ← Add your PostgreSQL password
}
```

### 3. Start the Backend (API)
Open a terminal in the root folder and run:
```bash
pip install -r requirements.txt
python api.py
```
*This starts the Python FastAPI server that handles all the heavy lifting (database, embeddings, LLM).*

### 4. Start the Frontend (User Interface)
Open a **new** terminal, navigate to the `frontend` folder, and run:
```bash
cd frontend
npm install
npm run dev
```
*This starts the beautiful Next.js user interface.*

You can now open your browser and go to `http://localhost:3000` to start using the assistant!

---

## Usage Guide (For End Users)

1. **Log In:** Create an account or sign in via the UI.
2. **Upload Documents:** Go to the Upload page to drag-and-drop company policies, code guidelines, or reference material. Assign a category to them.
3. **Generate Embeddings:** Once uploaded, go to the Admin/Documents panel and click **Generate Embeddings** so the system can read and index the files.
4. **Chat:** Head to the Chat page and ask plain-English questions! The system will route your query to the correct documents, read the chunks, and generate a sourced answer.

---

## Why Not ChatGPT or Microsoft Copilot?

| Feature | ChatGPT / Public AI | Our Local Assistant |
|---|---|---|
| **Privacy & Security** | Data leaves your network | **100% Private (stays on your machine)** |
| **Offline Access** | Needs internet | **Works entirely offline** |
| **Knowledge Source** | The entire public internet | **Only your company's actual documents** |
| **Fact-Checking** | Can invent fake answers | **Shows you the exact source document** |
