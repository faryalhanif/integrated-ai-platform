from typing import List

from app.agents.guardrail_agent import GuardrailAgent, GuardrailResult
from app.core.config import get_settings


def _matches_restricted_topic(query: str, restricted_topics: List[str]) -> str:
    """
    Cheap, deterministic pre-check: if the query directly mentions one
    of the business's restricted topic keywords/phrases, block
    immediately without spending an LLM call.

    Returns the matched topic, or "" if nothing matched.
    """

    lowered = query.lower()

    for topic in restricted_topics:
        if topic and topic.strip().lower() in lowered:
            return topic

    return ""


def run_guardrail(
    guardrail_agent: GuardrailAgent,
    query: str,
    restricted_topics: List[str],
    chat_history: str = "(no previous conversation)",
) -> GuardrailResult:
    """
    Combined guardrail check used by the LangGraph "guardrail" node:

    1. If guardrails are disabled in settings, always allow.
    2. Fast rule-based restricted-topic match -> block immediately.
    3. Otherwise defer to the LLM-based GuardrailAgent for prompt
       injection / off-topic / harmful-content detection, using recent
       conversation history to correctly read follow-ups and casual
       replies rather than judging the message in isolation.
    """

    settings = get_settings()

    if not settings.GUARDRAILS_ENABLED:
        return GuardrailResult(allowed=True, reason="Guardrails disabled.")

    matched_topic = _matches_restricted_topic(query, restricted_topics)

    if matched_topic:
        return GuardrailResult(
            allowed=False,
            reason=f"Query touches a restricted topic: {matched_topic}",
        )

    return guardrail_agent.run(
        query=query,
        restricted_topics=restricted_topics,
        chat_history=chat_history,
    )