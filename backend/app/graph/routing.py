from app.graph.state import RAGState


def load_project_router(state: RAGState) -> str:

    if state.get("error"):
        return "error"

    return "ok"


def guardrail_router(state: RAGState) -> str:

    if state.get("allowed"):
        return "allowed"

    return "blocked"
