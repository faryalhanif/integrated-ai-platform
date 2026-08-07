import json
from typing import Any, Dict, List

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.core.prompts import WEBSITE_PLANNER_SYSTEM_PROMPT
from app.schemas.business import BusinessProfile
from app.schemas.website import WebsitePlan


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

        result = chain.invoke(
            {
                "business_profile": business_profile.model_dump_json(indent=2),
                "knowledge_base_sample": kb_text,
                "user_prompt": user_prompt,
                "answers": json.dumps(answers or {}, indent=2),
                "format_instructions": self.parser.get_format_instructions(),
            }
        )

        return WebsitePlan(**result)
