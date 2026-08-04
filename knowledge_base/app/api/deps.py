from fastapi import Header, HTTPException

from app.core.config import get_settings


async def verify_service_api_key(authorization: str = Header(default=None)):
    """
    Optional bearer-token auth. If SERVICE_API_KEY is unset (default,
    local dev), every request is allowed. If it's set, callers must
    send `Authorization: Bearer <key>` — matching what the RAG
    backend's KnowledgeClient sends when KNOWLEDGE_BASE_API_KEY is
    configured on its side.
    """

    settings = get_settings()

    if not settings.SERVICE_API_KEY:
        return

    expected = f"Bearer {settings.SERVICE_API_KEY}"

    if authorization != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")
