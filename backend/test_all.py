"""
Standalone test script — no pytest required.

Place this file directly inside your `backend/` folder (next to
main.py) and run:

    pip install -r requirements.txt   # if you haven't already
    python3 test_all.py

It stubs out the LLM and the knowledge base (so it needs no real API
key and no running services) and exercises every path through the
system: onboarding, guardrails (rule-based + LLM-based), and the full
chat workflow (allowed / blocked / restricted-topic / missing-project /
knowledge-base-down).

Each check prints PASS or FAIL. Exit code is 0 if everything passed,
1 otherwise — so you can also use this in a CI step later.
"""

import os
import shutil
import sys
import tempfile
import traceback

# ---------------------------------------------------------------------
# Environment must be set BEFORE importing anything from `app`, since
# Settings are read at import time via get_settings().
# ---------------------------------------------------------------------
TMP_STORAGE_DIR = tempfile.mkdtemp(prefix="rag_backend_test_")

os.environ["LLM_PROVIDER"] = "groq"
os.environ["LLM_MODEL"] = "test-model"
os.environ["LLM_API_KEY"] = "test-key"  # never actually called, we stub the LLM
os.environ["STORAGE_DIR"] = TMP_STORAGE_DIR
os.environ["KNOWLEDGE_BASE_BASE_URL"] = "http://localhost:8001"
os.environ.setdefault("GUARDRAILS_ENABLED", "true")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

import app.graph.nodes as nodes_mod
import app.services.llm_service as llm_service_mod
from app.agents.guardrail_agent import GuardrailAgent
from app.rag.guardrails import run_guardrail
from app.rag.knowledge_client import KnowledgeClientError
from app.schemas.business import BusinessProfile, BusinessSubmission
from app.schemas.chat import ChatRequest
from app.services.chat_service import ChatService
from app.services.onboarding_service import OnboardingService
from app.services.prompt_service import get_project_store


# ---------------------------------------------------------------------
# Fakes — stand in for the real LLM and the (not-yet-built) knowledge
# base, so this script needs no API keys and no other services running.
# ---------------------------------------------------------------------
def _tail_after(text_lower: str, marker: str) -> str:
    """
    Returns everything after the last occurrence of `marker` in
    text_lower — used to isolate the actual user query/message from
    the surrounding instructional prompt text, so keyword checks below
    don't accidentally match example words used in the instructions
    themselves (e.g. the guardrail prompt's own text mentions "thanks"
    and "weather" as examples of allowed/blocked message types).
    """
    idx = text_lower.rfind(marker)
    return text_lower[idx + len(marker):] if idx != -1 else text_lower


def _fake_llm_fn(prompt_value):
    text = prompt_value.to_string() if hasattr(prompt_value, "to_string") else str(prompt_value)
    lowered = text.lower()

    # This check is on the instructional (system-prompt) text itself,
    # not the query — CASUAL_RESPONSE_SYSTEM_PROMPT is the only prompt
    # containing this exact phrase, so it's a safe way to identify
    # "this is a run_casual() call" regardless of what the query is.
    if "does not require" in lowered and "acknowledgement" in lowered:
        return AIMessage(content="You're welcome! Have a great day.")

    if "safety guard" in lowered:
        query_tail = _tail_after(lowered, "current message:")
        if "weather" in query_tail:
            return AIMessage(content='{"allowed": false, "reason": "unrelated to business", "is_conversational": false}')
        if "thanks" in query_tail or "thank you" in query_tail:
            return AIMessage(content='{"allowed": true, "reason": "casual reply", "is_conversational": true}')
        return AIMessage(content='{"allowed": true, "reason": "customer support question", "is_conversational": false}')

    if "rewrite the user" in lowered:
        query_tail = _tail_after(lowered, "user question:")
        if "weather" in query_tail:
            return AIMessage(content="What is the weather like today?")
        if "refund dispute" in query_tail:
            return AIMessage(content="I have a question about refund disputes.")
        if "thanks" in query_tail or "thank you" in query_tail:
            return AIMessage(content="Thanks!")
        return AIMessage(content="What are your bakery's opening hours?")

    if "expert ai prompt engineer" in lowered:
        return AIMessage(
            content="You are Acme Bakery's support bot. Be friendly. Only use the provided context."
        )

    if "expert business analyst" in lowered:
        return AIMessage(
            content='{"business_name": "Acme Bakery", "business_description": '
            '"We sell fresh bread and pastries daily.", "target_audience": '
            '"Local customers", "brand_tone": "Friendly & Casual", "metadata": '
            '{"industry": "Bakery"}}'
        )

    return AIMessage(content="**We are open** 8am-6pm daily. [1]\n* Weekends included")


FAKE_LLM = RunnableLambda(_fake_llm_fn)


class FakeKnowledgeClient:
    def retrieve(self, project_id, query, top_k=None):
        return [
            {
                "content": "Store hours are 8am-6pm every day, including weekends.",
                "source": "faq.pdf",
                "score": 0.91,
            }
        ]


class BrokenKnowledgeClient:
    def retrieve(self, project_id, query, top_k=None):
        raise KnowledgeClientError("service down")


def _seed_project(project_id="bakery-1", **overrides):
    kwargs = dict(
        project_id=project_id,
        business_name="Acme Bakery",
        business_description="We sell fresh bread and pastries daily.",
        target_audience="Local customers",
        brand_tone="Friendly & Casual",
        escalation_contact="support@acmebakery.com",
        restricted_topics=["refund disputes"],
        knowledge_base_id=project_id,
    )
    kwargs.update(overrides)
    profile = BusinessProfile(**kwargs)
    get_project_store().save_project(
        profile, "You are Acme Bakery's support bot. Be friendly. Only use the provided context."
    )
    return profile


# ---------------------------------------------------------------------
# Tiny test runner
# ---------------------------------------------------------------------
RESULTS = []


def check(name, fn):
    try:
        fn()
        RESULTS.append((name, True, None))
        print(f"PASS - {name}")
    except Exception as e:
        RESULTS.append((name, False, f"{e}\n{traceback.format_exc()}"))
        print(f"FAIL - {name}: {e}")


def reset_singletons():
    """Every check starts from a clean slate."""
    llm_service_mod.LLMService._llm = FAKE_LLM
    nodes_mod._guardrail_agent = None
    nodes_mod._query_rewriter = None
    nodes_mod._response_agent = None
    nodes_mod._knowledge_client = FakeKnowledgeClient()


# ---------------------------------------------------------------------
# 1. Onboarding
# ---------------------------------------------------------------------
def test_onboarding_creates_and_persists_project():
    reset_singletons()
    service = OnboardingService()

    submission = BusinessSubmission(
        business_name="Acme Bakery",
        business_description="We sell fresh bread and pastries daily out of our downtown shop.",
        target_audience="Local customers looking for fresh baked goods",
        brand_tone="Friendly & Casual",
        escalation_contact="support@acmebakery.com",
        restricted_topics=["refund disputes"],
    )

    result = service.create_project(submission)

    assert result["success"] is True
    assert result["business_profile"].business_name == "Acme Bakery"
    assert result["business_profile"].restricted_topics == ["refund disputes"]
    assert "Acme Bakery" in result["system_prompt"]

    store = get_project_store()
    loaded = store.get_business_profile(result["project_id"])
    assert loaded.business_name == "Acme Bakery"
    assert store.get_system_prompt(result["project_id"]) == result["system_prompt"]


def test_onboarding_generates_default_fallback_when_omitted():
    reset_singletons()
    service = OnboardingService()

    submission = BusinessSubmission(
        business_name="Acme Bakery",
        business_description="We sell fresh bread and pastries daily out of our downtown shop.",
        target_audience="Local customers",
        brand_tone="Friendly & Casual",
    )

    result = service.create_project(submission)

    assert result["business_profile"].fallback_message is None
    assert result["system_prompt"]


# ---------------------------------------------------------------------
# 2. Guardrails
# ---------------------------------------------------------------------
def test_rule_based_block_never_calls_llm():
    reset_singletons()
    agent = GuardrailAgent(FAKE_LLM)

    def blow_up(*a, **k):
        raise AssertionError("LLM guardrail should not run when a rule matched")

    agent.run = blow_up

    result = run_guardrail(
        guardrail_agent=agent,
        query="I want to talk about refund disputes",
        restricted_topics=["refund disputes"],
    )
    assert result.allowed is False
    assert "refund disputes" in result.reason


def test_llm_allows_on_topic_question():
    reset_singletons()
    agent = GuardrailAgent(FAKE_LLM)
    result = run_guardrail(agent, "What time do you open?", [])
    assert result.allowed is True


def test_llm_blocks_unrelated_question():
    reset_singletons()
    agent = GuardrailAgent(FAKE_LLM)
    result = run_guardrail(agent, "what is the weather today", [])
    assert result.allowed is False


def test_guardrails_disabled_flag():
    reset_singletons()
    os.environ["GUARDRAILS_ENABLED"] = "false"
    from app.core.config import get_settings
    get_settings.cache_clear()

    agent = GuardrailAgent(FAKE_LLM)
    result = run_guardrail(agent, "what is the weather today", ["anything"])
    assert result.allowed is True

    os.environ["GUARDRAILS_ENABLED"] = "true"
    get_settings.cache_clear()


# ---------------------------------------------------------------------
# 3. Full chat workflow
# ---------------------------------------------------------------------
def test_allowed_query_returns_answer_with_citations():
    reset_singletons()
    _seed_project()

    resp = ChatService().chat(
        ChatRequest(project_id="bakery-1", session_id="s1", user_query="What time do you open?")
    )
    assert resp.success is True
    assert resp.allowed is True
    assert "8am" in resp.answer
    assert "*" not in resp.answer  # markdown emphasis/bullets stripped
    assert resp.citations == ["faq.pdf"]


def test_casual_reply_skips_retrieval_and_answers_directly():
    reset_singletons()
    _seed_project()
    nodes_mod._knowledge_client = BrokenKnowledgeClient()  # would raise if ever called

    resp = ChatService().chat(
        ChatRequest(project_id="bakery-1", session_id="s6", user_query="thanks so much!")
    )
    assert resp.success is True
    assert resp.allowed is True
    assert resp.citations == []
    assert "welcome" in resp.answer.lower()


def test_followup_turn_passes_chat_history_to_guardrail():
    """
    Regression test for the original bug: guardrail_node used to judge
    each message in isolation, so short follow-ups like "what's the
    difference?" got misread as unrelated and blocked. It must now
    receive the conversation history and judge the *rewritten*
    (context-resolved) query.
    """
    reset_singletons()
    _seed_project()

    captured_history = []
    original_run = GuardrailAgent.run

    def spy_run(self, query, restricted_topics=None, chat_history="(no previous conversation)"):
        captured_history.append(chat_history)
        return original_run(self, query, restricted_topics, chat_history)

    GuardrailAgent.run = spy_run

    try:
        session_id = "followup-session"

        first = ChatService().chat(
            ChatRequest(project_id="bakery-1", session_id=session_id, user_query="What time do you open?")
        )
        assert first.allowed is True

        second = ChatService().chat(
            ChatRequest(project_id="bakery-1", session_id=session_id, user_query="what about weekends?")
        )

        # First turn had no prior history; second turn's guardrail call
        # must have received the first turn's history, not the default.
        assert captured_history[0] == "(no previous conversation)"
        assert captured_history[1] != "(no previous conversation)"
        assert "open" in captured_history[1].lower() or "8am" in captured_history[1].lower()
        assert second.allowed is True

    finally:
        GuardrailAgent.run = original_run


def test_list_projects_returns_seeded_project():
    reset_singletons()
    _seed_project()

    projects = get_project_store().list_projects()
    project_ids = [p["project_id"] for p in projects]
    assert "bakery-1" in project_ids


def test_unrelated_query_is_blocked():
    reset_singletons()
    _seed_project()

    resp = ChatService().chat(
        ChatRequest(project_id="bakery-1", session_id="s2", user_query="what is the weather today")
    )
    assert resp.allowed is False
    assert resp.citations == []
    assert "support@acmebakery.com" in resp.answer


def test_restricted_topic_blocked_without_llm():
    reset_singletons()
    _seed_project()

    resp = ChatService().chat(
        ChatRequest(
            project_id="bakery-1",
            session_id="s3",
            user_query="I have a question about refund disputes",
        )
    )
    assert resp.allowed is False
    assert "restricted topic" in resp.message


def test_missing_project_fails_gracefully():
    reset_singletons()
    resp = ChatService().chat(
        ChatRequest(project_id="does-not-exist", session_id="s4", user_query="hello")
    )
    assert resp.success is False
    assert resp.message == "project_not_found"
    assert "isn't set up yet" in resp.answer


def test_knowledge_base_down_degrades_gracefully():
    reset_singletons()
    _seed_project()
    nodes_mod._knowledge_client = BrokenKnowledgeClient()

    resp = ChatService().chat(
        ChatRequest(project_id="bakery-1", session_id="s5", user_query="What time do you open?")
    )
    assert resp.success is True
    assert resp.citations == []


# ---------------------------------------------------------------------
# Run everything
# ---------------------------------------------------------------------
if __name__ == "__main__":

    print(f"(using temp storage dir: {TMP_STORAGE_DIR})\n")

    checks = [
        ("Onboarding: creates and persists project", test_onboarding_creates_and_persists_project),
        ("Onboarding: default fallback message when omitted", test_onboarding_generates_default_fallback_when_omitted),
        ("Guardrails: rule-based block skips LLM call", test_rule_based_block_never_calls_llm),
        ("Guardrails: LLM allows on-topic question", test_llm_allows_on_topic_question),
        ("Guardrails: LLM blocks unrelated question", test_llm_blocks_unrelated_question),
        ("Guardrails: GUARDRAILS_ENABLED=false bypasses checks", test_guardrails_disabled_flag),
        ("Chat: allowed query returns answer + citations", test_allowed_query_returns_answer_with_citations),
        ("Chat: casual reply skips retrieval, answers directly", test_casual_reply_skips_retrieval_and_answers_directly),
        ("Chat: unrelated query blocked", test_unrelated_query_is_blocked),
        ("Chat: restricted topic blocked without LLM", test_restricted_topic_blocked_without_llm),
        ("Chat: missing project fails gracefully", test_missing_project_fails_gracefully),
        ("Chat: knowledge base down degrades gracefully", test_knowledge_base_down_degrades_gracefully),
        ("Chat: follow-up turn passes chat_history to guardrail", test_followup_turn_passes_chat_history_to_guardrail),
        ("Projects: list_projects returns seeded project", test_list_projects_returns_seeded_project),
    ]

    for name, fn in checks:
        check(name, fn)

    shutil.rmtree(TMP_STORAGE_DIR, ignore_errors=True)

    passed = sum(1 for _, ok, _ in RESULTS if ok)
    total = len(RESULTS)
    print(f"\n{passed}/{total} checks passed.")

    failed = [(n, err) for n, ok, err in RESULTS if not ok]
    if failed:
        print("\n--- FAILURE DETAILS ---")
        for name, err in failed:
            print(f"\n{name}:\n{err}")
        sys.exit(1)

    sys.exit(0)