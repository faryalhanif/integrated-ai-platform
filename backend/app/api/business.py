import logging

from fastapi import APIRouter, HTTPException

from app.rag.knowledge_client import KnowledgeClient
from app.schemas.business import BusinessSubmission
from app.services.onboarding_service import OnboardingService
from app.services.prompt_service import ProjectNotFoundError, get_project_store
from app.utils.llm_errors import friendly_llm_error_message, is_llm_rate_limited

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/business",
    tags=["Business"],
)

onboarding_service = OnboardingService()
knowledge_client = KnowledgeClient()


@router.post("/submit")
async def submit_business_information(
    submission: BusinessSubmission,
):
    """
    Handles business onboarding: generates the BusinessProfile and the
    chatbot's system prompt, and persists both under a new project_id.
    """

    try:

        return onboarding_service.create_project(submission)

    except Exception as e:

        logger.exception("Business submission failed for business_name=%s", submission.business_name)
        status = 429 if is_llm_rate_limited(e) else 500
        raise HTTPException(
            status_code=status,
            detail=friendly_llm_error_message(e),
        )


@router.get("/projects")
async def list_projects():
    """
    Lists every previously created chatbot project (most recent
    first) — used to let a returning user pick up an existing chatbot
    instead of only being able to create a new one.
    """

    return get_project_store().list_projects()


@router.get("/projects/{project_id}")
async def get_project(project_id: str):
    """
    Fetches a single project's full business profile, e.g. to resume
    onboarding or inspect its configuration.
    """

    try:
        profile = get_project_store().get_business_profile(project_id)
        return profile

    except ProjectNotFoundError:
        raise HTTPException(status_code=404, detail="Project not found.")


@router.delete("/projects/{project_id}")
async def delete_project(project_id: str):
    """
    Permanently deletes a chatbot project: its business profile +
    generated system prompt, and (best-effort) its knowledge base
    collection. This is irreversible — there is no undo/trash.
    """

    deleted = get_project_store().delete_project(project_id)

    if not deleted:
        raise HTTPException(status_code=404, detail="Project not found.")

    kb_deleted = False
    try:
        kb_deleted = knowledge_client.delete_project(project_id)
    except Exception:
        # Never fail the whole delete just because the knowledge base
        # call had an unexpected error — the project record itself is
        # already gone at this point.
        logger.exception(
            "Failed to delete knowledge base collection for project_id=%s",
            project_id,
        )

    return {
        "success": True,
        "project_id": project_id,
        "knowledge_base_deleted": kb_deleted,
    }