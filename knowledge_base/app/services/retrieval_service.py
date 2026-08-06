from typing import List
import logging

from app.core.config import get_settings
from app.pipeline.embedding import embed_texts
from app.pipeline.vectorstore import query_collection
from app.schemas.retrieve import RetrievedChunk

logger = logging.getLogger(__name__)


def retrieve(project_id: str, query: str, top_k: int = None) -> List[RetrievedChunk]:
    settings = get_settings()

    effective_top_k = top_k or settings.DEFAULT_TOP_K

    query_embedding = embed_texts([query])[0]

    raw_results = query_collection(
        project_id=project_id,
        query_embedding=query_embedding,
        top_k=effective_top_k,
    )

    results = [
        r for r in raw_results if r["score"] >= settings.MIN_RELEVANCE_SCORE
    ]

    if not results and raw_results:
        # Nothing cleared the relevance bar. Rather than returning
        # nothing at all — which forces a blanket "I don't know,
        # contact the business" even when the right chunk exists but
        # scored a bit low (e.g. OCR'd text is never as clean as typed
        # text, so it similarity-scores a little lower even when it's
        # the correct match) — pass along the single best match anyway
        # and let the response LLM judge it directly. The response
        # agent's own system prompt already instructs it to only
        # answer from what's actually in the given context and say so
        # when it isn't there, which is a more accurate judge of "is
        # this actually relevant" than one fixed global number can be.
        raw_results.sort(key=lambda r: r["score"], reverse=True)
        results = raw_results[:1]

    logger.info(
        "retrieve: project_id=%r query=%r -> %d raw result(s) %s, "
        "%d passed MIN_RELEVANCE_SCORE=%.2f, %d returned",
        project_id,
        query,
        len(raw_results),
        [r["score"] for r in raw_results],
        len([r for r in raw_results if r["score"] >= settings.MIN_RELEVANCE_SCORE]),
        settings.MIN_RELEVANCE_SCORE,
        len(results),
    )

    return [RetrievedChunk(**r) for r in results]