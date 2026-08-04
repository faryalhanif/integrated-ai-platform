# RAG Backend — AI Website Generation Platform

Implements the two workflows from your diagram:

1. **Onboarding**: `Business Information API -> Business Profile Agent ->
   Store Business Profile + System Prompt`
2. **Runtime Chat**: `LangGraph -> Guardrails -> Load Project Context ->
   Query Rewriter -> Knowledge Client -> Prompt Builder -> Response Agent
   -> Answer + Citations`
   (Load Project Context runs first internally so Guardrails has the
   business's restricted topics to check against — see the comment in
   `app/graph/workflow.py`.)

## Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env          # then fill in real values
uvicorn main:app --reload
```

Visit `http://localhost:8000/docs` for interactive API docs.

## Project layout

```
backend/
  main.py                     # FastAPI app entrypoint
  requirements.txt
  .env.example
  app/
    core/       config.py (all settings), prompts.py (all prompt templates)
    schemas/    business.py, chat.py  (pydantic models)
    agents/     business_profile_agent, prompt_generator, query_rewriter,
                guardrail_agent, response_agent  (raw LLM-calling classes)
    rag/        knowledge_client (talks to teammate's KB API),
                prompt_builder (context + citations), guardrails (orchestration)
    graph/      state, nodes, routing, workflow  (LangGraph wiring)
    services/   llm_service, prompt_service (project persistence),
                session_service (chat history), onboarding_service, chat_service
    api/        business.py (/business/submit), chat.py (/chat), health.py
```

## Integrating with your teammate's Knowledge Base

`app/rag/knowledge_client.py` documents the exact HTTP contract this
backend expects from the ingestion/vector-search service:

```
POST {KNOWLEDGE_BASE_BASE_URL}{KNOWLEDGE_BASE_RETRIEVE_PATH}
Body: {"project_id": "...", "query": "...", "top_k": 5}
Response: {"results": [{"content": "...", "source": "...", "score": 0.8}]}
```

Point `KNOWLEDGE_BASE_BASE_URL` at wherever that service runs — nothing
else needs to change once the endpoint matches this shape. If the shape
ends up different, only `knowledge_client.py` needs updating.

## What's intentionally NOT built yet (MVP scope)

- Only `primary_goal = "customer_support"` is implemented end-to-end
  (lead generation / technical support are stubbed as future work via
  the `Literal` type in `schemas/business.py`).
- `ProjectStore` is a JSON-file store for local dev — swap
  `JSONFileProjectStore` for a real DB-backed class in
  `app/services/prompt_service.py` for production; nothing else in the
  codebase needs to change.
- `SessionStore` is in-memory (per-process) chat history — fine for a
  single backend instance, swap for Redis/DB when you scale out.
- Website generation (the "AI Website Generation" box in your bigger
  diagram) isn't part of this RAG backend.

## No hardcoding

Every business-specific value (name, tone, restricted topics, fallback
message, escalation contact) flows through `BusinessSubmission` ->
`BusinessProfileAgent` -> `BusinessProfile`, and every environment-specific
value (LLM provider/model/keys, knowledge base URL, storage location,
CORS) comes from `app/core/config.py`/`.env`. No prompt file, agent, or
node contains a hardcoded business name, URL, or API key.
