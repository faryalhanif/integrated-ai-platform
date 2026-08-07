from langchain_core.language_models.chat_models import BaseChatModel

from app.core.config import get_settings


def _build_chat_model(
    provider: str,
    model: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
) -> BaseChatModel:
    """
    Only place in the codebase that knows about concrete provider SDKs.
    Adding a new provider means adding a branch here — nothing else that
    calls LLMService changes.
    """

    provider = provider.lower()

    if provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    if provider == "gemini":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as exc:
            raise ImportError(
                "LLM_PROVIDER/WEBSITE_LLM_PROVIDER is set to 'gemini' but "
                "langchain-google-genai isn't installed. Add it to "
                "requirements.txt and `pip install langchain-google-genai`."
            ) from exc

        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

    raise ValueError(f"Unsupported LLM provider: {provider}")


class LLMService:
    """
    Centralized LLM service.

    Every agent obtains its LLM from here. Provider/model/keys always
    come from Settings, never hardcoded. Two cached instances are
    exposed:

    - get_llm(): the short, cheap chat-turn model used by the RAG chatbot
      (app/graph/*), configured from LLM_*.
    - get_website_llm(): the website-generation model (planner, content,
      design, layout, code-gen, edit agents — app/agents/website_*),
      configured from WEBSITE_LLM_* with a fallback to LLM_* for anything
      left unset. Kept separate because website code generation needs a
      much larger max_tokens budget than a chat reply, and — per the
      platform's plan of "Gemini free during testing, Claude/GPT later in
      production" — may run on an entirely different provider.
    """

    _llm: BaseChatModel = None
    _website_llm: BaseChatModel = None
    _website_edit_llm: BaseChatModel = None

    @classmethod
    def get_llm(cls) -> BaseChatModel:

        if cls._llm is not None:
            return cls._llm

        settings = get_settings()

        cls._llm = _build_chat_model(
            provider=settings.LLM_PROVIDER,
            model=settings.LLM_MODEL,
            api_key=settings.LLM_API_KEY,
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS,
        )

        return cls._llm

    @classmethod
    def get_website_llm(cls) -> BaseChatModel:

        if cls._website_llm is not None:
            return cls._website_llm

        settings = get_settings()

        cls._website_llm = _build_chat_model(
            provider=settings.WEBSITE_LLM_PROVIDER or settings.LLM_PROVIDER,
            model=settings.WEBSITE_LLM_MODEL or settings.LLM_MODEL,
            api_key=settings.WEBSITE_LLM_API_KEY or settings.LLM_API_KEY,
            temperature=settings.WEBSITE_LLM_TEMPERATURE,
            max_tokens=settings.WEBSITE_LLM_MAX_TOKENS,
        )

        return cls._website_llm

    @classmethod
    def get_website_edit_llm(cls) -> BaseChatModel:
        """
        Same provider/model as get_website_llm(), but at a much lower
        temperature — the Edit Agent's job is to apply exactly one
        described change and leave everything else untouched, which
        benefits far more from determinism than from creativity.
        """

        if cls._website_edit_llm is not None:
            return cls._website_edit_llm

        settings = get_settings()

        cls._website_edit_llm = _build_chat_model(
            provider=settings.WEBSITE_LLM_PROVIDER or settings.LLM_PROVIDER,
            model=settings.WEBSITE_LLM_MODEL or settings.LLM_MODEL,
            api_key=settings.WEBSITE_LLM_API_KEY or settings.LLM_API_KEY,
            temperature=settings.WEBSITE_EDIT_TEMPERATURE,
            max_tokens=settings.WEBSITE_LLM_MAX_TOKENS,
        )

        return cls._website_edit_llm