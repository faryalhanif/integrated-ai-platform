from fastapi import APIRouter, HTTPException

from app.schemas.chat import ChatRequest, ChatResponse
from app.services.chat_service import ChatService

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

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )
