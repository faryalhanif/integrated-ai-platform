import json
import logging
import re
from typing import Any, Dict, List

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.core.prompts import WEBSITE_PLANNER_SYSTEM_PROMPT
from app.schemas.business import BusinessProfile
from app.schemas.website import WebsitePlan

logger = logging.getLogger(__name__)


class WebsitePlannerAgent:
    """
    Reads BusinessProfile + a knowledge-base sample + the user's prompt
    (and, on re-planning, any answers already collected) and produces the
    Website Specification: which pages are needed, and what's still
    strictly missing before generation can proceed.

    Corresponds to the "Planner Agent" box in the architecture diagram.
    """

    def __init__(self, llm: BaseChatModel):
        self.llm = llm

        self.parser = JsonOutputParser(pydantic_object=WebsitePlan)

        self.prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=WEBSITE_PLANNER_SYSTEM_PROMPT),
                (
                    "human",
                    """
Business Profile:

{business_profile}

Knowledge Base Sample (may be empty if nothing has been ingested yet):

{knowledge_base_sample}

User's Website Prompt:

{user_prompt}

Previously Collected Answers (may be empty; treat as authoritative,
never re-ask for something answered here):

{answers}

{format_instructions}
                    """,
                ),
            ]
        )

    def run(
        self,
        business_profile: BusinessProfile,
        user_prompt: str,
        knowledge_base_sample: List[Dict[str, Any]],
        answers: Dict[str, str],
    ) -> WebsitePlan:

        chain = self.prompt | self.llm | self.parser

        kb_text = (
            "\n\n".join(
                f"- {chunk.get('content', '')}" for chunk in knowledge_base_sample
            )
            or "(no knowledge base content available)"
        )

        logger.info(
            "Planner kb_text built from %d chunk(s): %s | preview: %r",
            len(knowledge_base_sample),
            [
                (c.get("source", "?"), len(c.get("content", "")))
                for c in knowledge_base_sample
            ],
            kb_text[:800],
        )

        result = chain.invoke(
            {
                "business_profile": business_profile.model_dump_json(indent=2),
                "knowledge_base_sample": kb_text,
                "user_prompt": user_prompt,
                "answers": json.dumps(answers or {}, indent=2),
                "format_instructions": self.parser.get_format_instructions(),
            }
        )

        plan = self._coerce_plan(result)

        return self._drop_missing_fields_already_covered(plan, kb_text)

    @staticmethod
    def _coerce_plan(result: Any) -> WebsitePlan:
        """
        The LLM is asked to return a single JSON object matching
        WebsitePlan, but occasionally (model-dependent, seen with
        gemini-2.5-flash) it instead returns a bare JSON list -- either
        a one-item list wrapping the real object, or a list of page
        objects with the rest of the object's fields omitted. Both were
        previously crashing generation with
        "argument after ** must be a mapping, not list". Recover in
        both cases instead of failing the whole request.
        """

        if isinstance(result, list):
            if len(result) == 1 and isinstance(result[0], dict):
                result = result[0]
            elif all(isinstance(item, dict) for item in result):
                result = {"pages": result, "missing_fields": [], "ready": True}
            else:
                raise ValueError(
                    f"Planner Agent returned an unrecognized list shape: {result!r}"
                )

        if not isinstance(result, dict):
            raise ValueError(
                f"Planner Agent returned an unexpected type: {type(result)!r}"
            )

        return WebsitePlan(**result)

    # Generic pattern for "concrete pricing already exists somewhere in the
    # retrieved knowledge base sample" -- a currency symbol next to a
    # number ("$29", "€79"), or a number paired with a billing period
    # ("29/month", "199 per year"). Deliberately generic (no business
    # names, plan names, or amounts hardcoded) so it works for any
    # project's data, not just one business's pricing table.
    _PRICING_EVIDENCE_PATTERN = re.compile(
        r"[$€£]\s?\d|\b\d{1,6}\s?(?:/|per\s)\s?(?:month|mo|year|yr|wk|week)\b",
        re.IGNORECASE,
    )

    @classmethod
    def _drop_missing_fields_already_covered(
        cls, plan: WebsitePlan, kb_text: str
    ) -> WebsitePlan:
        """
        Safety net for a specific failure mode seen with gemini-2.5-flash:
        the model asks the user for pricing via missing_fields even when
        concrete pricing is already sitting in the knowledge_base_sample
        it was just given -- it's a model-reliability gap, not a missing
        RAG connection, so fixing it here (deterministically) is more
        reliable than only tightening the prompt further. If the KB
        sample clearly already contains real pricing figures, any
        pricing-related missing_field is dropped and `ready` is
        recalculated. Every other missing_field (contact info, team
        bios, testimonials, etc.) is left untouched -- this only guards
        against re-asking for pricing that's demonstrably already there.
        """

        if not plan.missing_fields:
            return plan

        if not cls._PRICING_EVIDENCE_PATTERN.search(kb_text):
            logger.info(
                "Pricing safety net: no currency/billing-period pattern found "
                "in the %d-char knowledge base sample -- leaving missing_fields "
                "as the LLM returned them (%d field(s)).",
                len(kb_text),
                len(plan.missing_fields),
            )
            return plan

        remaining = [
            field
            for field in plan.missing_fields
            if "pric" not in field.key.lower()
            and "pric" not in field.question.lower()
        ]

        if len(remaining) == len(plan.missing_fields):
            logger.info(
                "Pricing safety net: currency/billing-period pattern WAS found "
                "in the knowledge base sample, but none of the %d missing_field(s) "
                "look pricing-related by key/question text -- nothing dropped.",
                len(plan.missing_fields),
            )
            return plan

        logger.info(
            "Pricing safety net: dropped %d pricing-related missing_field(s) "
            "because pricing was already present in the knowledge base sample "
            "(%d remaining, ready=%s).",
            len(plan.missing_fields) - len(remaining),
            len(remaining),
            len(remaining) == 0,
        )

        plan.missing_fields = remaining
        plan.ready = len(remaining) == 0

        return plan