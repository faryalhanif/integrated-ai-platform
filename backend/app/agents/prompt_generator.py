from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.core.prompts import PROMPT_GENERATOR_SYSTEM_PROMPT
from app.schemas.business import BusinessProfile
from app.utils.helpers import default_fallback_message


class PromptGenerator:
    """
    Generates the chatbot's system prompt from a BusinessProfile.
    """

    def __init__(self, llm: BaseChatModel):

        self.llm = llm

        self.parser = StrOutputParser()

        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    PROMPT_GENERATOR_SYSTEM_PROMPT,
                ),
                (
                    "human",
                    """
Business Profile:

{business_profile}
                    """,
                ),
            ]
        )

    def run(
        self,
        profile: BusinessProfile,
    ) -> str:

        # Fill in a safe default fallback message if the client didn't
        # provide one, so the generated prompt always has something
        # concrete to work with (never invented by the LLM).
        effective_profile = profile.model_copy(deep=True)

        if not effective_profile.fallback_message:
            effective_profile.fallback_message = default_fallback_message(
                business_name=profile.business_name,
                escalation_contact=profile.escalation_contact,
            )

        chain = (
            self.prompt
            | self.llm
            | self.parser
        )

        return chain.invoke(
            {
                "business_profile": effective_profile.model_dump_json(
                    indent=2
                )
            }
        )
