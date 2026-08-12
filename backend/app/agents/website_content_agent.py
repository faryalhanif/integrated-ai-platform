import json
from typing import Any, Dict, List

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.core.prompts import WEBSITE_CONTENT_SYSTEM_PROMPT
from app.schemas.business import BusinessProfile
from app.schemas.website import WebsitePlan
from app.utils.helpers import normalize_llm_json_object


class WebsiteContentAgent:
    """
    Writes the actual on-page copy for every planned page, grounded in
    the BusinessProfile, the knowledge base, and any answers the user
    supplied — never inventing facts. Corresponds to the "Content Agent
    (Uses RAG)" box in the architecture diagram.
    """

    def __init__(self, llm: BaseChatModel):
        self.llm = llm

        self.parser = JsonOutputParser()

        self.prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=WEBSITE_CONTENT_SYSTEM_PROMPT),
                (
                    "human",
                    """
Business Profile:

{business_profile}

Website Specification (pages + planned sections):

{plan}

Knowledge Base Sample:

{knowledge_base_sample}

Answers Supplied By The User:

{answers}

Return the JSON object described in your instructions, keyed by page
slug for every page slug in the Website Specification above.
                    """,
                ),
            ]
        )

    def run(
        self,
        business_profile: BusinessProfile,
        plan: WebsitePlan,
        knowledge_base_sample: List[Dict[str, Any]],
        answers: Dict[str, str],
    ) -> Dict[str, Any]:

        chain = self.prompt | self.llm | self.parser

        kb_text = (
            "\n\n".join(
                f"- {chunk.get('content', '')}" for chunk in knowledge_base_sample
            )
            or "(no knowledge base content available)"
        )

        result = chain.invoke(
            {
                "business_profile": business_profile.model_dump_json(indent=2),
                "plan": plan.model_dump_json(indent=2),
                "knowledge_base_sample": kb_text,
                "answers": json.dumps(answers or {}, indent=2),
            }
        )

        try:
            return normalize_llm_json_object(result)
        except ValueError:
            # Genuinely unrecognizable shape (not the common single-item-
            # list wrap normalize_llm_json_object already handles) --
            # degrade to empty content rather than crashing the whole
            # generation; downstream agents already tolerate missing
            # per-page content.
            return {}