from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.core.prompts import WEBSITE_EDIT_SYSTEM_PROMPT
from app.utils.helpers import assert_complete_html_document, extract_html_document


class WebsiteEditAgent:
    """
    Applies a plain-English edit instruction ("make it darker", "add a
    pricing page") to an already-generated website by rewriting the full
    HTML document with just that change applied. Corresponds to the
    "Edit Agent" box in the architecture diagram — the user edits using
    chat, and the Live Preview updates in place.
    """

    def __init__(self, llm: BaseChatModel):
        self.llm = llm

        self.parser = StrOutputParser()

        self.prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=WEBSITE_EDIT_SYSTEM_PROMPT),
                (
                    "human",
                    """
Current HTML Document:

{current_html}

User's Requested Change:

{instruction}
                    """,
                ),
            ]
        )

    def run(self, current_html: str, instruction: str) -> str:

        chain = self.prompt | self.llm | self.parser

        raw = chain.invoke(
            {
                "current_html": current_html,
                "instruction": instruction,
            }
        )

        html = extract_html_document(raw)

        assert_complete_html_document(html)

        return html