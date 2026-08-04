from typing import Any, Dict, List, Tuple

from app.utils.helpers import truncate_text


def build_context(
    documents: List[Dict[str, Any]],
    max_chars_per_chunk: int = 2000,
) -> str:
    """
    Formats retrieved documents into a numbered context block the
    Response Agent can cite as [1], [2], etc.
    """

    if not documents:
        return ""

    blocks = []

    for i, doc in enumerate(documents, start=1):
        source = doc.get("source", "unknown source")
        content = truncate_text(
            doc.get("content", ""),
            max_chars=max_chars_per_chunk,
        )

        blocks.append(
            f"[{i}] Source: {source}\n{content}"
        )

    return "\n\n".join(blocks)


def extract_citations(documents: List[Dict[str, Any]]) -> List[str]:
    """
    Returns the unique, ordered list of source identifiers from the
    retrieved documents, for display alongside the answer.
    """

    seen = set()
    citations: List[str] = []

    for doc in documents:
        source = doc.get("source")

        if source and source not in seen:
            seen.add(source)
            citations.append(source)

    return citations


def build_context_and_citations(
    documents: List[Dict[str, Any]],
) -> Tuple[str, List[str]]:

    return build_context(documents), extract_citations(documents)
