import logging

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.api.deps import verify_service_api_key
from app.pipeline.graph import IngestionError
from app.schemas.ingest import IngestResponse, IngestUrlRequest
from app.services.ingestion_service import UploadTooLargeError, ingest_file, ingest_url

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ingest", tags=["Ingestion"])


@router.post(
    "/file",
    response_model=IngestResponse,
    dependencies=[Depends(verify_service_api_key)],
)
async def ingest_file_endpoint(
    project_id: str = Form(...),
    file: UploadFile = File(...),
):
    """
    Ingests a single file (pdf, docx, pptx, md, txt, png/jpg) into the
    given project's knowledge base collection. Runs the full
    parse -> chunk -> embed -> store pipeline synchronously.
    """

    try:
        return await ingest_file(project_id=project_id, file=file)

    except UploadTooLargeError as exc:
        raise HTTPException(status_code=413, detail=str(exc))

    except IngestionError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    except Exception as exc:
        logger.exception("File ingestion failed for project_id=%s", project_id)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post(
    "/url",
    response_model=IngestResponse,
    dependencies=[Depends(verify_service_api_key)],
)
async def ingest_url_endpoint(request: IngestUrlRequest):
    """
    Fetches and ingests a web page into the given project's knowledge
    base collection.
    """

    try:
        return ingest_url(project_id=request.project_id, url=str(request.url))

    except IngestionError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    except Exception as exc:
        logger.exception("URL ingestion failed for project_id=%s", request.project_id)
        raise HTTPException(status_code=500, detail=str(exc))
