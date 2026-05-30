# PaperMind AI — Research Intelligence Platform

A production-ready AI research assistant that lets you upload research papers (PDFs) and interact with them through a citation-aware RAG system.

## Features

| Feature | Description |
|---|---|
| **PDF Upload** | Upload multiple PDFs, auto-extract title, authors, abstract, year |
| **Citation-Aware QA** | Ask questions, get grounded answers with exact source citations + page numbers |
| **Paper Summary** | Structured summaries: contributions, methodology, results, limitations |
| **Paper Comparison** | Compare two papers across methodology, datasets, metrics, conclusions |
| **Semantic Search** | Natural language search with author/year filters |
| **Related Work** | Embedding-based similarity recommendations |
| **Dashboard** | Track papers, embeddings, and query history |
| **JWT Auth** | Secure multi-user authentication |

## Tech Stack

- **Backend**: FastAPI (async) + SQLAlchemy 2.0 + Alembic
- **Frontend**: Next.js 14 + TypeScript + TailwindCSS
- **Vector DB**: FAISS (IndexFlatIP, cosine similarity)
- **Database**: PostgreSQL 16
- **LLM**: OpenAI / OpenRouter compatible
- **Embeddings**: BGE-small-en-v1.5 (Sentence Transformers)
- **PDF Parsing**: PyMuPDF (fitz)
- **Auth**: JWT (python-jose + bcrypt)

## Project Structure

```
papermind-ai/
├── backend/
│   ├── app/
│   │   ├── api/routes/     # auth, papers, rag
│   │   ├── core/           # config, security, logging
│   │   ├── db/             # session, base
│   │   ├── models/         # SQLAlchemy models
│   │   ├── rag/            # pdf_parser, chunker, embedder, vector_store, llm_client
│   │   ├── schemas/        # Pydantic schemas
│   │   ├── services/       # user_service, paper_service, rag_service
│   │   └── main.py
│   ├── alembic/
│   ├── .env.example
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── app/            # Next.js App Router pages
│   │   ├── components/     # Sidebar, UI
│   │   ├── lib/            # api client, store, utils
│   │   └── types/
│   ├── .env.local.example
│   └── package.json
└── README.md
```

## Setup & Running

### Prerequisites

- Python 3.11+
- Node.js 20+
- PostgreSQL 14+ running locally

---

### 1. PostgreSQL — Create the Database

```sql
-- Run in psql or pgAdmin
CREATE USER papermind WITH PASSWORD 'papermind';
CREATE DATABASE papermind OWNER papermind;
```

---

### 2. Backend

```bash
cd backend

# Create virtual environment
python -m venv .venv

# Activate
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and set your OPENAI_API_KEY and DATABASE_URL

# Start the server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Tables are created automatically on first startup via SQLAlchemy.

For migrations (optional, after model changes):
```bash
alembic revision --autogenerate -m "describe change"
alembic upgrade head
```

---

### 3. Frontend

```bash
cd frontend

# Install dependencies
npm install

# Configure environment
cp .env.local.example .env.local

# Start dev server
npm run dev
```

---

### Access

| Service | URL |
|---|---|
| Frontend | http://localhost:3000 |
| Backend API | http://localhost:8000 |
| API Docs (Swagger) | http://localhost:8000/api/docs |
| API Docs (ReDoc) | http://localhost:8000/api/redoc |

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/v1/auth/register` | Register user |
| POST | `/api/v1/auth/login` | Login |
| GET | `/api/v1/auth/me` | Current user |
| POST | `/api/v1/papers/upload` | Upload PDF |
| GET | `/api/v1/papers` | List papers (filters: author, year) |
| DELETE | `/api/v1/papers/{id}` | Delete paper |
| GET | `/api/v1/papers/dashboard` | Dashboard stats |
| POST | `/api/v1/rag/ask` | Q&A with citations |
| POST | `/api/v1/rag/summarize` | Paper summary |
| POST | `/api/v1/rag/compare` | Compare two papers |
| POST | `/api/v1/rag/search` | Semantic search |
| POST | `/api/v1/rag/recommend` | Related papers |

---

## Environment Variables

**backend/.env**
```env
# Required
SECRET_KEY=your-secret-key
DATABASE_URL=postgresql+asyncpg://papermind:papermind@localhost:5432/papermind
DATABASE_URL_SYNC=postgresql+psycopg2://papermind:papermind@localhost:5432/papermind
OPENAI_API_KEY=sk-...

# Optional LLM overrides
LLM_MODEL=gpt-4o-mini
OPENROUTER_API_KEY=sk-or-...
OPENAI_BASE_URL=https://openrouter.ai/api/v1  # for OpenRouter

# Tuning
CHUNK_SIZE=512
CHUNK_OVERLAP=64
TOP_K_RETRIEVAL=5
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
```

**frontend/.env.local**
```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

---

## RAG Architecture

```
PDF Upload
   ↓ PyMuPDF parsing (per-page text + metadata)
   ↓ Sentence-boundary chunking (512 tokens, 64 overlap)
   ↓ BGE-small embeddings (normalized, cosine-ready)
   ↓ FAISS IndexFlatIP (inner product = cosine on normalized vecs)
   ↓ PostgreSQL (chunk content + page numbers)

Query
   ↓ Embed with instruction prefix (BGE retrieval format)
   ↓ FAISS top-k search (optional paper_id filter)
   ↓ Load chunk text + paper metadata from PostgreSQL
   ↓ LLM generation with citation-aware prompt
   ↓ Return answer + CitationSource[] with page numbers
```
