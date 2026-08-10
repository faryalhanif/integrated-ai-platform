import logging

from fastapi import APIRouter, HTTPException

from app.schemas.chat import ChatRequest, ChatResponse
from app.services.chat_service import ChatService
from app.utils.llm_errors import friendly_llm_error_message, is_llm_rate_limited

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/chat",
    tags=["Chat"],
)

chat_service = ChatService()


@router.post(
    "",
    response_model=ChatResponse,
)
async def chat(
    request: ChatRequest,
):
    """
    Runs the visitor's message through the RAG workflow: load project
    context -> guardrails -> query rewrite -> retrieve -> respond.
    """

    try:

        return chat_service.chat(request)

    except Exception as e:

        logger.exception("Chat failed for project_id=%s", request.project_id)
        status = 429 if is_llm_rate_limited(e) else 500
        raise HTTPException(
            status_code=status,
            detail=friendly_llm_error_message(e),
        )