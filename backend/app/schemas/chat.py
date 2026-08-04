from typing import Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """
    Incoming chat request from the website chatbot widget.
    """

    project_id: str = Field(
        ...,
        description="Project associated with this chatbot.",
    )

    session_id: str = Field(
        ...,
        description="Unique conversation/session identifier (used to track chat history).",
    )

    user_query: str = Field(
        ...,
        min_length=1,
        description="Visitor's question.",
    )


class ChatResponse(BaseModel):
    """
    Final response returned to the frontend.
    """

    answer: str

    citations: list[str] = Field(default_factory=list)

    session_id: str

    allowed: bool = True

    success: bool = True

    message: Optional[str] = None
