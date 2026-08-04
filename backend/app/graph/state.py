from typing import Any, Dict, List, Optional, TypedDict


class RAGState(TypedDict):
    """
    Shared state passed between LangGraph nodes.
    """

    project_id: str

    session_id: str

    user_query: str

    business_profile: Optional[Dict[str, Any]]

    system_prompt: Optional[str]

    chat_history: Optional[str]

    error: Optional[str]

    allowed: Optional[bool]

    rejection_reason: Optional[str]

    is_conversational: Optional[bool]

    rewritten_query: Optional[str]

    retrieved_documents: List[Dict[str, Any]]

    context: Optional[str]

    response: Optional[str]

    citations: List[str]

    metadata: Dict[str, Any]