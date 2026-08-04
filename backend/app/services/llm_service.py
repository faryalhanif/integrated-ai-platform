from langchain_core.language_models.chat_models import BaseChatModel
from langchain_groq import ChatGroq

from app.core.config import get_settings


class LLMService:
    """
    Centralized LLM service.

    Every agent obtains its LLM from here. This is the only place that
    knows about concrete provider SDKs, so switching providers later
    (OpenAI, Anthropic, etc.) means adding a branch here — nothing else
    in the codebase changes. Provider/model/keys always come from
    Settings, never hardcoded.
    """

    _llm: BaseChatModel = None

    @classmethod
    def get_llm(cls) -> BaseChatModel:

        if cls._llm is not None:
            return cls._llm

        settings = get_settings()

        provider = settings.LLM_PROVIDER.lower()

        if provider == "groq":
            cls._llm = ChatGroq(
                model=settings.LLM_MODEL,
                api_key=settings.LLM_API_KEY,
                temperature=settings.LLM_TEMPERATURE,
                max_tokens=settings.LLM_MAX_TOKENS,
            )
            return cls._llm

        raise ValueError(
            f"Unsupported LLM provider: {settings.LLM_PROVIDER}"
        )
