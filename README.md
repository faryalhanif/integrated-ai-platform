# AI Website Generation Platform — RAG Backend + Knowledge Base

Two services that together implement the full RAG chatbot pipeline:

```
backend/          The chat backend (FastAPI + LangGraph): onboarding,
                   guardrails, query rewriting, response generation.
knowledge_base/    The ingestion + retrieval service (FastAPI +
                   LangGraph): parses documents/URLs, chunks them,
                   embeds them locally, and answers similarity-search
                   queries from a per-project ChromaDB collection.
```

`backend/` was already built expecting a separate Knowledge Base
service reachable at `KNOWLEDGE_BASE_BASE_URL` (see
`backend/app/rag/knowledge_client.py` for the exact HTTP contract).
`knowledge_base/` is that service — it exposes `POST
/knowledge/retrieve` matching that contract exactly, plus ingestion
endpoints to actually populate the vector store (the original
`knevaflow_auto_embedder` notebook only ever wrote to Chroma; it had
no query/retrieval side, no HTTP interface, and no way to be called
from anywhere else).

The two services talk to each other only over HTTP, using `project_id`
as the shared key — nothing is hardcoded between them.

## What changed while integrating

- **`knowledge_base/` didn't exist as a service before** — it's a
  restructured, callable version of the notebook's LangGraph pipeline,
  wrapped in FastAPI with the retrieval endpoint added.
- **Removed hardcoded secrets.** The notebook had a Groq API key
  hardcoded in three different parser functions; `backend/.env` and
  `.env.example` had a live-looking Groq key checked in. Both are now
  placeholders read from `.env` — **rotate those two keys**, since they
  were exposed in the files you provided.
- **Removed blocking `input()` calls.** The notebook prompted for a
  file path and project name on the terminal. Ingestion is now a
  function (`run_ingestion_pipeline(project_id, file_path=None,
  url=None)`) called from an API endpoint instead.
- **Removed the hardcoded `127.0.0.1:8000/extract` dependency** for
  web-page ingestion. That URL is now `WEB_EXTRACTION_SERVICE_URL` in
  `knowledge_base/.env` (optional) — if it's unset or unreachable, the
  service falls back to a built-in `httpx` + `BeautifulSoup` extractor
  so URL ingestion works out of the box instead of silently failing.
- **Added the missing retrieval side.** Wrote the vector-store query
  logic, cosine-similarity scoring, and the `/knowledge/retrieve`
  endpoint from scratch to match what the backend already expected.
- Dropped an unused `img2table`/`Img2TableImage` import left over in
  the notebook (never actually called).

## Running locally

Both services need Python 3.11+.

```bash
# 1. Knowledge Base service (start first)
cd knowledge_base
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env      # fill in LLM_API_KEY if you want table summarization
uvicorn main:app --reload --port 8001

# 2. Backend (in a second terminal)
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # fill in LLM_API_KEY (required)
uvicorn main:app --reload --port 8000
```

Or with Docker:

```bash
cp knowledge_base/.env.example knowledge_base/.env
cp backend/.env.example backend/.env
# edit both .env files with real LLM_API_KEY values
docker compose up --build
```

- Backend docs: http://localhost:8000/docs
- Knowledge Base docs: http://localhost:8001/docs

## End-to-end flow

1. **Onboard a business** — `POST http://localhost:8000/business/submit`
   with business info. Returns a `project_id`.
2. **Ingest its documents** — for each file:
   `POST http://localhost:8001/ingest/file` with form fields
   `project_id` (the one from step 1) and `file`. For a web page:
   `POST http://localhost:8001/ingest/url` with `{"project_id", "url"}`.
3. **Chat** — `POST http://localhost:8000/chat` with
   `{"project_id", "session_id", "user_query"}`. The backend calls the
   Knowledge Base service internally to retrieve context and answers
   with citations.

Both services default to `*` CORS and no auth for local dev. Set
`SERVICE_API_KEY` in `knowledge_base/.env` and the matching
`KNOWLEDGE_BASE_API_KEY` in `backend/.env` to require a bearer token
between them in shared/production environments.

## Notes

- `knowledge_base` downloads the `sentence-transformers/all-MiniLM-L6-v2`
  embedding model from Hugging Face on first run (needs internet
  access once; cached afterwards). Swap `EMBEDDING_MODEL` in `.env` for
  any other `sentence-transformers`-compatible model.
- Image OCR ingestion requires the `tesseract` binary installed on the
  host (`apt install tesseract-ocr` / `brew install tesseract`); point
  `TESSERACT_CMD` at it if it's not on `PATH`.
- Both `ProjectStore` (backend) and the Chroma vector store
  (knowledge_base) are local-disk/file-based — fine for development,
  swap for a real database / managed vector DB before scaling out (see
  the comments in `backend/app/services/prompt_service.py` and
  `knowledge_base/app/pipeline/vectorstore.py`).
