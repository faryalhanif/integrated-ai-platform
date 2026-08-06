from typing import List

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from app.core.prompts import GUARDRAIL_SYSTEM_PROMPT


class GuardrailResult(BaseModel):
    """
    Output returned by the guardrail agent. Field order matches the
    two-step decision process in GUARDRAIL_SYSTEM_PROMPT
    (is_conversational first, then allowed) — JSON is generated
    field-by-field in declaration order, so deciding is_conversational
    first measurably improves how reliably casual messages ("how are
    you", "ok", etc.) get routed correctly instead of being treated as
    unanswerable business questions.
    """

    is_conversational: bool = Field(
        default=False,
        description=(
            "True if this message is casual/social conversation that "
            "needs no knowledge-base lookup (greetings, farewells, "
            "thanks, well-being check-ins like 'how are you', short "
            "acknowledgements like 'ok'/'yes', small talk). False if "
            "it's an actual question about the business that needs "
            "retrieved context to answer."
        ),
    )

    allowed: bool = Field(
        description="Whether the query is allowed to continue."
    )

    reason: str = Field(
        description="Reason for the decision."
    )


class GuardrailAgent:
    """
    LLM-based check for whether a query should continue through the RAG
    pipeline. Combined with a fast rule-based check in
    app/rag/guardrails.py before this is even called.
    """

    def __init__(self, llm: BaseChatModel):

        self.llm = llm

        self.parser = JsonOutputParser(
            pydantic_object=GuardrailResult
        )

        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    GUARDRAIL_SYSTEM_PROMPT,
                ),
                (
                    "human",
                    "Conversation History:\n{chat_history}\n\nCurrent Message:\n{query}",
                ),
            ]
        )

    def run(
        self,
        query: str,
        business_name: str = "the business",
        business_description: str = "(not provided)",
        restricted_topics: List[str] = None,
        chat_history: str = "(no previous conversation)",
    ) -> GuardrailResult:

        chain = (
            self.prompt
            | self.llm
            | self.parser
        )

        result = chain.invoke(
            {
                "query": query,
                "business_name": business_name,
                "business_description": business_description,
                "chat_history": chat_history,
                "restricted_topics": ", ".join(restricted_topics) if restricted_topics else "(none)",
                "format_instructions": self.parser.get_format_instructions(),
            }
        )

        return GuardrailResult(**result)