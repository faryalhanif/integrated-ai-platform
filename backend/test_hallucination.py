"""
Hallucination & correctness test — uses your REAL LLM (not a stub).

Why real, unlike test_all.py: hallucination is a property of the actual
model's behavior. A fake/stubbed LLM always returns the same canned
text no matter what context it's given, so it can't tell you anything
about whether the real model invents facts. This script feeds the real
model carefully controlled fake "retrieved context" (since your
coworker's knowledge base doesn't exist yet) and checks how it behaves.

REQUIREMENTS:
    - A real LLM_API_KEY in your .env (this WILL make real API calls)
    - Run from inside backend/, same folder as main.py

    python3 test_hallucination.py

WHAT THIS CHECKS:
    1. Grounded answer   - given a fact in context, does it answer
                           correctly using that fact?
    2. Empty context     - given NO context, does it admit it doesn't
                           know, instead of inventing an answer?
    3. Irrelevant context - given context about a different topic, does
                           it avoid inventing details about what was
                           actually asked?
    4. Consistency       - asked twice, does it give the same core fact
                           both times (not one true, one made up)?

IMPORTANT: these checks use simple keyword heuristics (regex/substring
matching), not a perfect hallucination detector. Anything printed as
"FLAG - review manually" is NOT necessarily wrong — read the printed
answer yourself and judge. That's normal for hallucination testing;
there's no fully automatic way to catch every case.
"""

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.agents.response_agent import ResponseAgent
from app.services.llm_service import LLMService

# ---------------------------------------------------------------------
# Sanity check: make sure a real key is actually configured
# ---------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(SCRIPT_DIR, ".env")

print(f"Looking for .env at: {ENV_PATH}")
print(f".env exists at that path: {os.path.exists(ENV_PATH)}")

if not os.environ.get("LLM_API_KEY"):
    try:
        from dotenv import load_dotenv
        loaded = load_dotenv(dotenv_path=ENV_PATH, override=True)
        print(f"load_dotenv() loaded a file: {loaded}")
    except ImportError:
        print("ERROR: python-dotenv isn't installed. Run: pip install python-dotenv")
        sys.exit(1)

key = os.environ.get("LLM_API_KEY", "")
masked = (key[:4] + "..." + key[-2:]) if len(key) > 8 else "(empty or too short)"
print(f"LLM_API_KEY found: {masked}")

if not key or key in ("", "test-key", "YOUR_GROQ_API_KEY"):
    print("\nERROR: No real LLM_API_KEY found.")
    print("Checklist:")
    print(f"  1. Does {ENV_PATH} actually exist? (see above)")
    print("  2. On Windows, check the file isn't secretly named '.env.txt'")
    print("     (File Explorer hides known extensions by default).")
    print("  3. Inside .env, the line should look exactly like:")
    print("     LLM_API_KEY=gsk_your_real_key_here")
    print("     (no quotes needed, no spaces around the '=').")
    sys.exit(1)

SYSTEM_PROMPT = (
    "You are Acme Bakery's customer support bot. Be friendly and concise. "
    "Only answer using the provided context. If the answer isn't in the "
    "context, say you don't have that information and suggest contacting "
    "the team at support@acmebakery.com. Never invent facts, numbers, "
    "or policies that aren't in the context."
)

agent = ResponseAgent(LLMService.get_llm())

RESULTS = []


def run_case(name, context, query, checks):
    print(f"\n--- {name} ---")
    print(f"Context given: {context!r}")
    print(f"Question: {query}")

    answer = agent.run(
        system_prompt=SYSTEM_PROMPT,
        context=context,
        query=query,
        chat_history="(no previous conversation)",
    )

    print(f"AI answered: {answer}")

    all_ok = True
    for check_name, check_fn in checks:
        ok = check_fn(answer)
        print(f"  [{'PASS' if ok else 'FLAG - review manually'}] {check_name}")
        all_ok = all_ok and ok

    RESULTS.append((name, all_ok))


def contains(text, *substrings):
    lowered = text.lower()
    return any(s.lower() in lowered for s in substrings)


def no_fabricated_numbers(text, allowed_numbers=()):
    """
    Flags if the answer contains a number-like fact (money, %, days)
    that wasn't explicitly allowed — a rough sign of an invented detail.
    """
    found = re.findall(r"\$?\d+(?:\.\d+)?\s?(?:%|days?|dollars?|hours?)?", text.lower())
    suspicious = [f for f in found if f.strip() and not any(a in f for a in allowed_numbers)]
    return len(suspicious) == 0


# ---------------------------------------------------------------------
# 1. Grounded answer — fact IS in context, should be used correctly
# ---------------------------------------------------------------------
run_case(
    name="Grounded answer (fact present in context)",
    context="[1] Source: policy.pdf\nReturns are accepted within 30 days of purchase for store credit.",
    query="What is your return policy?",
    checks=[
        ("Mentions the correct number (30)", lambda a: "30" in a),
    ],
)

# ---------------------------------------------------------------------
# 2. Empty context — NO facts given. Testing the RAW agent here on
# purpose, with no pipeline guard in front of it — so a FLAG here is
# EXPECTED and fine. What actually matters is check 2b below, which
# tests the real pipeline that production chat requests go through.
# ---------------------------------------------------------------------
run_case(
    name="Empty context (no relevant documents found)",
    context="",
    query="What is your return policy?",
    checks=[
        (
            "Admits it doesn't know / points to contact",
            lambda a: contains(a, "don't have", "do not have", "not sure", "unable to", "contact", "support@acmebakery.com"),
        ),
        (
            "Does not invent a specific number/day count",
            lambda a: no_fabricated_numbers(a),
        ),
    ],
)

# ---------------------------------------------------------------------
# 2b. Pipeline-level protection — does the actual guard in nodes.py work?
# ---------------------------------------------------------------------
# NOTE: the check above calls the raw ResponseAgent directly, which has
# no protection of its own — that's expected, and is exactly why the
# guard below exists one layer up, in graph/nodes.py's response_node.
# This checks THAT guard, which is what real chat requests actually go
# through.
print("\n--- Pipeline-level guard (graph/nodes.py response_node) ---")
from app.graph.nodes import response_node

state = {
    "system_prompt": SYSTEM_PROMPT,
    "context": "",  # no documents retrieved at all
    "rewritten_query": "What is your return policy?",
    "user_query": "What is your return policy?",
    "chat_history": "(no previous conversation)",
    "session_id": "hallucination-test-session",
    "business_profile": {
        "business_name": "Acme Bakery",
        "escalation_contact": "support@acmebakery.com",
        "fallback_message": None,
    },
}

result = response_node(state)
pipeline_answer = result["response"]
print(f"Pipeline answer (should be the fallback, NOT an AI-generated guess): {pipeline_answer}")

guard_worked = no_fabricated_numbers(pipeline_answer) and contains(
    pipeline_answer, "contact", "support@acmebakery.com", "don't have"
)
print(f"  [{'PASS' if guard_worked else 'FLAG - review manually'}] Empty context never reached the AI — returned a safe fallback instead")
RESULTS.append(("Pipeline-level empty-context guard", guard_worked))


# ---------------------------------------------------------------------
# 3. Irrelevant context — context about a different topic entirely
# ---------------------------------------------------------------------
run_case(
    name="Irrelevant context (context is about hours, question is about pricing)",
    context="[1] Source: hours.pdf\nWe are open 8am-6pm every day, including weekends.",
    query="How much does a custom birthday cake cost?",
    checks=[
        (
            "Does not invent a specific price",
            lambda a: "$" not in a and not re.search(r"\b\d+\s?(dollars|usd)\b", a.lower()),
        ),
        (
            "Admits it doesn't know / points to contact",
            lambda a: contains(a, "don't have", "do not have", "not sure", "unable to", "contact", "support@acmebakery.com"),
        ),
    ],
)

# ---------------------------------------------------------------------
# 4. Consistency — same grounded question asked twice
# ---------------------------------------------------------------------
print("\n--- Consistency check (asking the grounded question again) ---")
answer_a = agent.run(
    system_prompt=SYSTEM_PROMPT,
    context="[1] Source: policy.pdf\nReturns are accepted within 30 days of purchase for store credit.",
    query="What is your return policy?",
    chat_history="(no previous conversation)",
)
answer_b = agent.run(
    system_prompt=SYSTEM_PROMPT,
    context="[1] Source: policy.pdf\nReturns are accepted within 30 days of purchase for store credit.",
    query="What is your return policy?",
    chat_history="(no previous conversation)",
)
print(f"Answer 1: {answer_a}")
print(f"Answer 2: {answer_b}")
consistent = ("30" in answer_a) and ("30" in answer_b)
print(f"  [{'PASS' if consistent else 'FLAG - review manually'}] Both answers mention the same correct number (30)")
RESULTS.append(("Consistency check", consistent))


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------
print("\n" + "=" * 60)
passed = sum(1 for _, ok in RESULTS if ok)
total = len(RESULTS)
print(f"{passed}/{total} checks passed automatically.")
print(
    "Read every 'AI answered' line above yourself too — these heuristic "
    "checks catch obvious cases but not subtle hallucinations."
)

flagged = [name for name, ok in RESULTS if not ok]
if flagged:
    print("\nFlagged for manual review:")
    for name in flagged:
        print(f"  - {name}")
    sys.exit(1)

sys.exit(0)