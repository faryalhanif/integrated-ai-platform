import json
from typing import Any, Dict, List

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.core.prompts import WEBSITE_CODEGEN_SYSTEM_PROMPT
from app.schemas.business import BusinessProfile
from app.schemas.website import WebsitePlan
from app.utils.helpers import assert_complete_html_document, extract_html_document


class WebsiteCodeGeneratorAgent:
    """
    Turns the finished Website Specification + Content + Design + Layout
    into one complete, self-contained HTML document (Tailwind CDN,
    vanilla JS) — the Live Preview. Corresponds to the "Code Generation
    (React/Next.js + Tailwind)" box in the architecture diagram: this MVP
    targets a dependency-free static HTML/Tailwind build so the preview
    renders instantly in an iframe with no build step, and so the same
    output can be deployed as-is to any static host (Vercel/Netlify).
    """

    def __init__(self, llm: BaseChatModel):
        self.llm = llm

        self.parser = StrOutputParser()

        self.prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=WEBSITE_CODEGEN_SYSTEM_PROMPT),
                (
                    "human",
                    """
Business Profile:

{business_profile}

Website Specification:

{plan}

Design System:

{design}

Finalized Section Order Per Page:

{layout}

Content Per Page:

{content}
                    """,
                ),
            ]
        )

    def run(
        self,
        business_profile: BusinessProfile,
        plan: WebsitePlan,
        content: Dict[str, Any],
        design: Dict[str, Any],
        layout: Dict[str, List[str]],
    ) -> str:

        chain = self.prompt | self.llm | self.parser

        raw = chain.invoke(
            {
                "business_profile": business_profile.model_dump_json(indent=2),
                "plan": plan.model_dump_json(indent=2),
                "design": json.dumps(design, indent=2),
                "layout": json.dumps(layout, indent=2),
                "content": json.dumps(content, indent=2),
            }
        )

        html = extract_html_document(raw)

        assert_complete_html_document(html)

        return html