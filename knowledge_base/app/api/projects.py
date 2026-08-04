from fastapi import APIRouter, Depends

from app.api.deps import verify_service_api_key
from app.pipeline.vectorstore import delete_project, list_projects
from app.schemas.ingest import ProjectSummary

router = APIRouter(prefix="/projects", tags=["Projects"])


@router.get(
    "",
    response_model=list[ProjectSummary],
    dependencies=[Depends(verify_service_api_key)],
)
async def list_all_projects():
    """Lists every project (Chroma collection) with its chunk count."""
    return list_projects()


@router.delete(
    "/{project_id}",
    dependencies=[Depends(verify_service_api_key)],
)
async def delete_project_endpoint(project_id: str):
    """Deletes a project's entire knowledge base collection."""
    deleted = delete_project(project_id)
    return {"project_id": project_id, "deleted": deleted}
