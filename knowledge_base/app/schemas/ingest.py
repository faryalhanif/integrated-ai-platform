from typing import Optional

from pydantic import BaseModel, Field, HttpUrl


class IngestUrlRequest(BaseModel):
    project_id: str = Field(..., min_length=1)
    url: HttpUrl


class IngestResponse(BaseModel):
    success: bool
    project_id: str
    collection_name: str
    source: str
    chunks_ingested: int
    message: Optional[str] = None


class ProjectSummary(BaseModel):
    collection_name: str
    chunk_count: int
