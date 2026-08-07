from typing import Any, Dict, List

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.core.prompts import WEBSITE_LAYOUT_SYSTEM_PROMPT
from app.schemas.website import WebsitePlan


class WebsiteLayoutAgent:
    """
    Finalizes the section order for every page, using the plan's rough
    sections plus what the Content Agent actually produced (dropping any
    section that ended up empty). Corresponds to the "Layout Agent" box
    in the architecture diagram.
    """

    def __init__(self, llm: BaseChatModel):
        self.llm = llm

        self.parser = JsonOutputParser()

        self.prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=WEBSITE_LAYOUT_SYSTEM_PROMPT),
                (
                    "human",
                    """
Website Specification (planned pages + rough sections):

{plan}

Generated Content (per page slug — use this to see which sections
actually have real content and which don't):

{content}
                    """,
                ),
            ]
        )

    def run(
        self,
        plan: WebsitePlan,
        content: Dict[str, Any],
    ) -> Dict[str, List[str]]:

        chain = self.prompt | self.llm | self.parser

        try:
            result = chain.invoke(
                {
                    "plan": plan.model_dump_json(indent=2),
                    "content": _safe_json(content),
                }
            )
        except Exception:
            result = None

        if isinstance(result, dict) and result:
            return {
                slug: [str(s) for s in sections]
                for slug, sections in result.items()
                if isinstance(sections, list)
            }

        # Fallback: use the planner's own section order verbatim, minus
        # any section the content agent produced nothing for.
        fallback: Dict[str, List[str]] = {}
        for page in plan.pages:
            page_content = content.get(page.slug, {}) if isinstance(content, dict) else {}
            available_types = {
                s.get("type") for s in page_content.get("sections", []) if isinstance(s, dict)
            }
            fallback[page.slug] = [
                s for s in page.sections if s in available_types or s in ("hero", "footer")
            ] or page.sections

        return fallback


def _safe_json(content: Dict[str, Any]) -> str:
    import json

    try:
        return json.dumps(content, indent=2)
    except TypeError:
        return "{}"
