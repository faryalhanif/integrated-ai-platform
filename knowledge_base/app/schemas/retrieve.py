from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class RetrieveRequest(BaseModel):
    """
    Matches the contract documented in the RAG backend's
    app/rag/knowledge_client.py exactly:

        POST {KNOWLEDGE_BASE_BASE_URL}{KNOWLEDGE_BASE_RETRIEVE_PATH}
        Body: {"project_id": "...", "query": "...", "top_k": 5}
    """

    project_id: str = Field(..., min_length=1)
    query: str = Field(..., min_length=1)
    top_k: Optional[int] = Field(default=None, ge=1, le=50)


class RetrievedChunk(BaseModel):
    content: str
    source: str
    score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RetrieveResponse(BaseModel):
    results: List[RetrievedChunk] = Field(default_factory=list)
