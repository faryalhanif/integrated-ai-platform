from typing import List

from app.core.config import get_settings
from app.pipeline.embedding import embed_texts
from app.pipeline.vectorstore import query_collection
from app.schemas.retrieve import RetrievedChunk


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
        RetrievedChunk(**r)
        for r in raw_results
        if r["score"] >= settings.MIN_RELEVANCE_SCORE
    ]

    return results
