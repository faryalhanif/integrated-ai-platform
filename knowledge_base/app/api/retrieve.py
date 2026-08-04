import logging

from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import verify_service_api_key
from app.schemas.retrieve import RetrieveRequest, RetrieveResponse
from app.services.retrieval_service import retrieve

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Knowledge"])


@router.post(
    "/knowledge/retrieve",
    response_model=RetrieveResponse,
    dependencies=[Depends(verify_service_api_key)],
)
async def retrieve_knowledge(request: RetrieveRequest):
    """
    Vector-search retrieval endpoint consumed by the RAG backend's
    KnowledgeClient. Returns an empty result list (HTTP 200) rather
    than an error when the project has no ingested documents yet, so
    the backend can degrade gracefully.
    """

    try:
        results = retrieve(
            project_id=request.project_id,
            query=request.query,
            top_k=request.top_k,
        )

    except Exception as exc:
        logger.exception("Retrieval failed for project_id=%s", request.project_id)
        raise HTTPException(status_code=500, detail=str(exc))

    return RetrieveResponse(results=results)
