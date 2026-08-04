import re
import unicodedata

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import get_settings
from app.pipeline.state import PipelineState


def clean_extracted_text(text: str) -> str:
    """Cleans and normalizes raw extracted text before chunking."""
    if not text:
        return ""

    # Normalize unicode (fixes ligatures like 'fi' -> 'fi', standardizes
    # quotes/spaces).
    text = unicodedata.normalize("NFKC", text)

    # Remove invisible/zero-width formatting characters.
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)

    # Collapse 3+ consecutive newlines down to 2 (preserve paragraph
    # breaks, drop huge vertical gaps).
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Collapse repeated spaces/tabs (but not newlines) to one space.
    text = re.sub(r"[^\S\r\n]{2,}", " ", text)

    return text.strip()


def chunking_node(state: PipelineState) -> PipelineState:
    documents = state.get("documents", [])
    chunked_documents = []

    settings = get_settings()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )

    for doc in documents:
        content_type = doc["metadata"].get("content_type")

        cleaned_metadata = doc["metadata"].copy()
        # layout_detected is debug-only bookkeeping; Chroma metadata
        # values must be str/int/float/bool so drop anything else here.
        cleaned_metadata.pop("layout_detected", None)

        if content_type == "text":
            cleaned_text = clean_extracted_text(doc["page_content"])
            chunks = text_splitter.split_text(cleaned_text)

            for chunk in chunks:
                chunked_documents.append(
                    {"page_content": chunk, "metadata": cleaned_metadata}
                )

        elif content_type == "table_summary":
            # Already LLM-summarized (or pre-cleaned) — keep intact to
            # preserve context rather than re-splitting it.
            chunked_documents.append(
                {"page_content": doc["page_content"], "metadata": cleaned_metadata}
            )

    return {"chunked_documents": chunked_documents}
