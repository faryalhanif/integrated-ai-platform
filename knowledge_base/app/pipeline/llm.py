"""
Thin, config-driven LLM accessor used only to turn extracted tables
into short factual summaries before embedding.

Mirrors the RAG backend's LLMService pattern: provider/model/key always
come from Settings, never hardcoded. If no key is configured (or the
provider call fails), callers fall back to a plain-text flatten of the
table instead of crashing ingestion.
"""

from typing import Optional

from app.core.config import get_settings

_llm = None
_llm_init_attempted = False


def get_table_summarizer_llm() -> Optional[object]:
    """
    Returns a cached LangChain chat model instance, or None if no LLM
    key is configured / table summarization is disabled. Ingestion
    degrades gracefully (flattened markdown instead of an LLM summary)
    when this returns None.
    """

    global _llm, _llm_init_attempted

    if _llm is not None:
        return _llm

    if _llm_init_attempted:
        return None

    _llm_init_attempted = True

    settings = get_settings()

    if not settings.TABLE_SUMMARIZATION_ENABLED or not settings.LLM_API_KEY:
        return None

    provider = settings.LLM_PROVIDER.lower()

    try:
        if provider == "groq":
            from langchain_groq import ChatGroq

            _llm = ChatGroq(
                model=settings.LLM_MODEL,
                api_key=settings.LLM_API_KEY,
                temperature=settings.LLM_TEMPERATURE,
            )
        else:
            return None

    except Exception:
        return None

    return _llm


def summarize_table_markdown(md_table: str) -> str:
    """
    Converts a markdown table into a short factual paragraph via the
    LLM. Falls back to returning the raw markdown table if no LLM is
    configured or the call fails, so ingestion never hard-fails on a
    table.
    """

    llm = get_table_summarizer_llm()

    if llm is None:
        return md_table

    prompt = (
        "You are a strict data analyst. Convert the following markdown "
        "table into a concise, factual paragraph. Do not exceed 600 "
        "characters. Do not include any introductory speeches, "
        f"greetings, or filler words.\nTable Data:\n{md_table}"
    )

    try:
        result = llm.invoke(prompt)
        content = getattr(result, "content", None)
        return content.strip() if content else md_table
    except Exception:
        return md_table
