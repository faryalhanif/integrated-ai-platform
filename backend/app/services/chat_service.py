from app.graph.state import RAGState
from app.graph.workflow import RAGWorkflow
from app.schemas.chat import ChatRequest, ChatResponse


class ChatService:
    """
    Handles chatbot requests by executing the RAG workflow.
    """

    def __init__(self):

        self.workflow = RAGWorkflow().build()

    def chat(
        self,
        request: ChatRequest,
    ) -> ChatResponse:

        initial_state: RAGState = {
            "project_id": request.project_id,
            "session_id": request.session_id,
            "user_query": request.user_query,

            "business_profile": None,
            "system_prompt": None,
            "chat_history": None,

            "error": None,

            "allowed": None,
            "rejection_reason": None,
            "is_conversational": None,

            "rewritten_query": None,

            "retrieved_documents": [],
            "context": None,

            "response": None,
            "citations": [],

            "metadata": {},
        }

        final_state = self.workflow.invoke(initial_state)

        return ChatResponse(
            answer=final_state.get("response")
            or "Sorry, something went wrong while generating a response.",
            citations=final_state.get("citations", []),
            session_id=request.session_id,
            allowed=final_state.get("allowed") is not False,
            success=final_state.get("error") is None,
            message=final_state.get("rejection_reason") or final_state.get("error"),
        )