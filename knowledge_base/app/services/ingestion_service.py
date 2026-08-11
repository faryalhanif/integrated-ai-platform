import uuid
from pathlib import Path

from fastapi import UploadFile

from app.core.config import get_settings
from app.pipeline.graph import run_ingestion_pipeline
from app.pipeline.vectorstore import sanitize_collection_name
from app.schemas.ingest import IngestResponse


class UploadTooLargeError(Exception):
    pass


async def ingest_file(project_id: str, file: UploadFile) -> IngestResponse:
    settings = get_settings()

    upload_dir = Path(settings.UPLOAD_TMP_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)

    suffix = Path(file.filename or "upload").suffix
    tmp_path = upload_dir / f"{uuid.uuid4().hex}{suffix}"

    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    size = 0

    try:
        with open(tmp_path, "wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > max_bytes:
                    raise UploadTooLargeError(
                        f"File exceeds the {settings.MAX_UPLOAD_SIZE_MB}MB limit."
                    )
                out.write(chunk)

        final_state = run_ingestion_pipeline(
            project_id=project_id,
            file_path=str(tmp_path),
        )

        chunks = final_state.get("chunks_ingested", 0)
        source = file.filename or str(tmp_path)

        return IngestResponse(
            success=True,
            project_id=project_id,
            collection_name=final_state.get(
                "collection_name", sanitize_collection_name(project_id)
            ),
            source=source,
            chunks_ingested=chunks,
            message=f"Ingested {chunks} chunks from '{source}'.",
        )

    finally:
        tmp_path.unlink(missing_ok=True)


def ingest_url(project_id: str, url: str) -> IngestResponse:
    final_state = run_ingestion_pipeline(project_id=project_id, url=url)

    chunks = final_state.get("chunks_ingested", 0)

    return IngestResponse(
        success=True,
        project_id=project_id,
        collection_name=final_state.get(
            "collection_name", sanitize_collection_name(project_id)
        ),
        source=url,
        chunks_ingested=chunks,
        message=f"Ingested {chunks} chunks from '{url}'.",
    )
