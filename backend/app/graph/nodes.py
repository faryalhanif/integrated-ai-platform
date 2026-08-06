import logging

from app.agents.guardrail_agent import GuardrailAgent
from app.agents.query_rewriter import QueryRewriter
from app.agents.response_agent import ResponseAgent
from app.graph.state import RAGState
from app.rag.guardrails import run_guardrail
from app.rag.knowledge_client import KnowledgeClient, KnowledgeClientError
from app.rag.prompt_builder import build_context_and_citations
from app.services.llm_service import LLMService
from app.services.prompt_service import ProjectNotFoundError, get_project_store
from app.services.session_service import get_session_store
from app.utils.helpers import default_fallback_message, strip_markdown_emphasis

logger = logging.getLogger(__name__)

# Shared, lazily-created singletons — built once per process, reused
# across every chat request/graph invocation.
_guardrail_agent = None
_query_rewriter = None
_response_agent = None
_knowledge_client = None


def _get_guardrail_agent() -> GuardrailAgent:
    global _guardrail_agent
    if _guardrail_agent is None:
        _guardrail_agent = GuardrailAgent(LLMService.get_llm())
    return _guardrail_agent


def _get_query_rewriter() -> QueryRewriter:
    global _query_rewriter
    if _query_rewriter is None:
        _query_rewriter = QueryRewriter(LLMService.get_llm())
    return _query_rewriter


def _get_response_agent() -> ResponseAgent:
    global _response_agent
    if _response_agent is None:
        _response_agent = ResponseAgent(LLMService.get_llm())
    return _response_agent


def _get_knowledge_client() -> KnowledgeClient:
    global _knowledge_client
    if _knowledge_client is None:
        _knowledge_client = KnowledgeClient()
    return _knowledge_client


def load_project_node(state: RAGState) -> dict:
    """
    Loads the project's BusinessProfile + generated system prompt, and
    the recent chat history for this session.
    """

    store = get_project_store()

    try:
        profile = store.get_business_profile(state["project_id"])
        system_prompt = store.get_system_prompt(state["project_id"])

    except ProjectNotFoundError:
        return {
            "error": "project_not_found",
            "response": (
                "This chatbot isn't set up yet. Please contact the "
                "site owner."
            ),
            "citations": [],
        }

    chat_history = get_session_store().get_history_text(state["session_id"])

    return {
        "business_profile": profile.model_dump(mode="json"),
        "system_prompt": system_prompt,
        "chat_history": chat_history,
    }


def guardrail_node(state: RAGState) -> dict:
    """
    Decides whether the query is allowed to continue. Runs *after*
    query rewriting so it judges a context-resolved, standalone query
    (e.g. "what's the difference?" resolved against the prior turn)
    rather than a bare fragment — otherwise short follow-ups and casual
    replies get misread as unrelated and blocked. If blocked,
    immediately fills in the final response so the graph can end here.
    """

    profile = state.get("business_profile") or {}
    restricted_topics = profile.get("restricted_topics", [])

    query_for_guardrail = state.get("rewritten_query") or state["user_query"]

    result = run_guardrail(
        guardrail_agent=_get_guardrail_agent(),
        query=query_for_guardrail,
        restricted_topics=restricted_topics,
        business_name=profile.get("business_name", "the business"),
        business_description=profile.get("business_description", "(not provided)"),
        chat_history=state.get("chat_history") or "(no previous conversation)",
    )

    logger.info(
        "guardrail: query=%r allowed=%s is_conversational=%s reason=%r",
        query_for_guardrail,
        result.allowed,
        result.is_conversational,
        result.reason,
    )

    if not result.allowed:
        fallback = profile.get("fallback_message") or default_fallback_message(
            business_name=profile.get("business_name", "our team"),
            escalation_contact=profile.get("escalation_contact"),
        )

        return {
            "allowed": False,
            "rejection_reason": result.reason,
            "is_conversational": result.is_conversational,
            "response": fallback,
            "citations": [],
        }

    return {
        "allowed": True,
        "rejection_reason": None,
        "is_conversational": result.is_conversational,
    }


def rewrite_query_node(state: RAGState) -> dict:

    rewritten = _get_query_rewriter().run(
        query=state["user_query"],
        chat_history=state.get("chat_history") or "(no previous conversation)",
    )

    return {"rewritten_query": rewritten}


def retrieve_node(state: RAGState) -> dict:

    if state.get("is_conversational"):
        # Casual chit-chat (greeting, thanks, short confirmation) —
        # no knowledge-base lookup needed.
        return {"retrieved_documents": [], "context": "", "citations": []}

    profile = state.get("business_profile") or {}
    knowledge_base_id = profile.get("knowledge_base_id") or state["project_id"]

    try:
        documents = _get_knowledge_client().retrieve(
            project_id=knowledge_base_id,
            query=state.get("rewritten_query") or state["user_query"],
        )

    except KnowledgeClientError:
        # Degrade gracefully: answer with "no context found" rather than
        # failing the whole request if the knowledge base is unreachable.
        logger.exception(
            "retrieve: knowledge base unreachable for knowledge_base_id=%r",
            knowledge_base_id,
        )
        documents = []

    logger.info(
        "retrieve: knowledge_base_id=%r query=%r -> %d document(s) [scores=%s]",
        knowledge_base_id,
        state.get("rewritten_query") or state["user_query"],
        len(documents),
        [round(d.get("score", 0.0), 3) for d in documents],
    )

    context, citations = build_context_and_citations(documents)

    return {
        "retrieved_documents": documents,
        "context": context,
        "citations": citations,
    }


def response_node(state: RAGState) -> dict:

    profile = state.get("business_profile") or {}

    if state.get("is_conversational"):
        # Casual conversation doesn't need grounded context — answer
        # directly and briefly in the business's tone.
        answer = _get_response_agent().run_casual(
            business_name=profile.get("business_name", "our team"),
            brand_tone=profile.get("brand_tone", "friendly"),
            query=state.get("rewritten_query") or state["user_query"],
            chat_history=state.get("chat_history") or "(no previous conversation)",
        )
        answer = strip_markdown_emphasis(answer)

        get_session_store().add_turn(
            session_id=state["session_id"],
            user_msg=state["user_query"],
            assistant_msg=answer,
        )

        return {"response": answer}

    context = state.get("context") or ""

    if not context.strip():
        # No relevant documents were retrieved at all. Don't even ask
        # the LLM to answer — an empty context is exactly when models
        # are most likely to fall back on generic training knowledge
        # and present it as this business's actual policy. Enforcing
        # this in code (not just the prompt) removes that risk entirely.
        answer = profile.get("fallback_message") or default_fallback_message(
            business_name=profile.get("business_name", "our team"),
            escalation_contact=profile.get("escalation_contact"),
        )

        get_session_store().add_turn(
            session_id=state["session_id"],
            user_msg=state["user_query"],
            assistant_msg=answer,
        )

        return {"response": answer}

    answer = _get_response_agent().run(
        system_prompt=state["system_prompt"],
        context=context,
        query=state.get("rewritten_query") or state["user_query"],
        chat_history=state.get("chat_history") or "(no previous conversation)",
    )
    answer = strip_markdown_emphasis(answer)

    get_session_store().add_turn(
        session_id=state["session_id"],
        user_msg=state["user_query"],
        assistant_msg=answer,
    )

    return {"response": answer}