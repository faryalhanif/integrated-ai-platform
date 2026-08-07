from typing import Any, Dict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.core.prompts import WEBSITE_DESIGN_SYSTEM_PROMPT
from app.schemas.business import BusinessProfile
from app.schemas.website import WebsitePlan

_DEFAULT_DESIGN: Dict[str, Any] = {
    "primary_color": "#4f46e5",
    "secondary_color": "#0f172a",
    "accent_color": "#22d3ee",
    "background_color": "#ffffff",
    "text_color": "#0f172a",
    "font_heading": "Inter",
    "font_body": "Inter",
    "mode": "light",
    "border_radius": "soft",
    "style_keywords": ["clean", "modern", "trustworthy"],
}


class WebsiteDesignAgent:
    """
    Chooses a cohesive color palette, typography, and visual style based
    on the business's brand — filling in anything not explicitly set.
    Corresponds to the "Design Agent" box in the architecture diagram.
    """

    def __init__(self, llm: BaseChatModel):
        self.llm = llm

        self.parser = JsonOutputParser()

        self.prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=WEBSITE_DESIGN_SYSTEM_PROMPT),
                (
                    "human",
                    """
Business Profile:

{business_profile}

Website Specification:

{plan}
                    """,
                ),
            ]
        )

    def run(
        self,
        business_profile: BusinessProfile,
        plan: WebsitePlan,
    ) -> Dict[str, Any]:

        chain = self.prompt | self.llm | self.parser

        try:
            result = chain.invoke(
                {
                    "business_profile": business_profile.model_dump_json(indent=2),
                    "plan": plan.model_dump_json(indent=2),
                }
            )
        except Exception:
            # A malformed design response should never block generation —
            # fall back to a safe, neutral default palette instead.
            return dict(_DEFAULT_DESIGN)

        if not isinstance(result, dict):
            return dict(_DEFAULT_DESIGN)

        design = dict(_DEFAULT_DESIGN)
        design.update({k: v for k, v in result.items() if v})

        if business_profile.brand_colors:
            if business_profile.brand_colors.get("primary"):
                design["primary_color"] = business_profile.brand_colors["primary"]
            if business_profile.brand_colors.get("secondary"):
                design["secondary_color"] = business_profile.brand_colors["secondary"]

        return design
