from datetime import datetime, timezone

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.core.prompts import BUSINESS_PROFILE_SYSTEM_PROMPT
from app.schemas.business import BusinessProfile, BusinessSubmission


class BusinessProfileAgent:
    """
    Generates a structured BusinessProfile from client onboarding data.
    """

    def __init__(self, llm: BaseChatModel):
        self.llm = llm

        self.parser = JsonOutputParser(
            pydantic_object=BusinessProfile
        )

        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    BUSINESS_PROFILE_SYSTEM_PROMPT,
                ),
                (
                    "human",
                    """
Business Information:

{business_information}

{format_instructions}
                    """,
                ),
            ]
        )

    def run(
        self,
        submission: BusinessSubmission,
        project_id: str,
    ) -> BusinessProfile:

        chain = (
            self.prompt
            | self.llm
            | self.parser
        )

        result = chain.invoke(
            {
                "business_information": submission.model_dump_json(
                    indent=2
                ),
                "format_instructions": self.parser.get_format_instructions(),
            }
        )

        # The LLM is not the source of truth for these — carry them over
        # verbatim from the client's submission rather than trusting the
        # model to echo them back correctly.
        result["project_id"] = project_id
        result["primary_goal"] = submission.primary_goal
        result["restricted_topics"] = submission.restricted_topics or result.get(
            "restricted_topics", []
        )
        result["escalation_contact"] = submission.escalation_contact
        result["fallback_message"] = submission.fallback_message
        result["knowledge_base_id"] = (
            submission.knowledge_base_id or project_id
        )

        # created_at is server-assigned, never client/LLM-assigned. The
        # format instructions we hand the model include this field (it's
        # part of BusinessProfile), so the model routinely "fills it in"
        # with null since it has no real timestamp to give — and an
        # *explicit* null in `result` overrides BusinessProfile's
        # default_factory (a default_factory only kicks in when the key
        # is absent, not when it's present-but-None), which is exactly
        # what was blowing up BusinessProfile(**result) with:
        #   "created_at: Input should be a valid string [NoneType]"
        # Always stamp it ourselves so it can never come from the LLM.
        result["created_at"] = datetime.now(timezone.utc).isoformat()

        if submission.brand_colors:
            result["brand_colors"] = submission.brand_colors

        if submission.logo_url:
            result["logo_url"] = str(submission.logo_url)

        if submission.contact_phone:
            result["contact_phone"] = submission.contact_phone

        if submission.contact_address:
            result["contact_address"] = submission.contact_address

        if submission.social_links:
            result["social_links"] = {
                k: str(v) for k, v in submission.social_links.items()
            }

        return BusinessProfile(**result)