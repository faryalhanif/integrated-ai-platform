from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.core.prompts import QUERY_REWRITER_SYSTEM_PROMPT


class QueryRewriter:
    """
    Rewrites user queries to improve retrieval quality, using recent
    conversation history for context resolution.
    """

    def __init__(self, llm: BaseChatModel):

        self.llm = llm

        self.parser = StrOutputParser()

        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    QUERY_REWRITER_SYSTEM_PROMPT,
                ),
                (
                    "human",
                    """
Conversation History:
{chat_history}

User Question:
{query}
                    """,
                ),
            ]
        )

    def run(
        self,
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
                "query": query,
                "chat_history": chat_history,
            }
        )
