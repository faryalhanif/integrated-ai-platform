from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.core.prompts import CASUAL_RESPONSE_SYSTEM_PROMPT, RESPONSE_AGENT_USER_PROMPT


class ResponseAgent:
    """
    Generates the final chatbot response, grounded in retrieved context
    and the project's dynamically-generated system prompt.
    """

    def __init__(
        self,
        llm: BaseChatModel,
    ):

        self.llm = llm

        self.parser = StrOutputParser()

        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "{system_prompt}",
                ),
                (
                    "human",
                    RESPONSE_AGENT_USER_PROMPT,
                ),
            ]
        )

        self.casual_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    CASUAL_RESPONSE_SYSTEM_PROMPT,
                ),
                (
                    "human",
                    "Conversation History:\n{chat_history}\n\nCurrent Message:\n{query}",
                ),
            ]
        )

    def run(
        self,
        system_prompt: str,
        context: str,
        query: str,
        chat_history: str = "(no previous conversation)",
    ) -> str:

        chain = (
            self.prompt
            | self.llm
            | self.parser
        )

        return chain.invoke(
            {
                "system_prompt": system_prompt,
                "chat_history": chat_history,
                "context": context if context else "(no relevant context found)",
                "query": query,
            }
        )

    def run_casual(
        self,
        business_name: str,
        brand_tone: str,
        query: str,
        chat_history: str = "(no previous conversation)",
    ) -> str:
        """
        Answers casual conversation (greetings, thanks, short
        confirmations) without requiring retrieved knowledge-base
        context — the main `run()` path is grounded-answer-only by
        design and would otherwise have nothing to work with here.
        """

        chain = (
            self.casual_prompt
            | self.llm
            | self.parser
        )

        return chain.invoke(
            {
                "business_name": business_name,
                "brand_tone": brand_tone,
                "chat_history": chat_history,
                "query": query,
            }
        )