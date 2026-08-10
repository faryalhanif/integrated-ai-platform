import logging

from fastapi import APIRouter, HTTPException, Request

from app.schemas.website import (
    GeneratedWebsite,
    WebsiteEditRequest,
    WebsiteGenerateRequest,
    WebsitePlan,
    WebsitePlanRequest,
    WidgetEmbedResponse,
)
from app.services.prompt_service import ProjectNotFoundError
from app.services.website_service import WebsiteGenerationService
from app.services.website_store import WebsiteNotFoundError
from app.utils.llm_errors import friendly_llm_error_message, is_llm_rate_limited

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/website",
    tags=["Website Generation"],
)

website_service = WebsiteGenerationService()


def _with_site_url(site: GeneratedWebsite, request: Request) -> GeneratedWebsite:
    """
    Attaches the live, shareable URL (GET /sites/{slug}) for this
    response only — never persisted to storage, so it's always built
    from wherever this request actually came in on (localhost today,
    the real domain automatically once deployed behind one).
    """

    if site.slug:
        site.site_url = f"{str(request.base_url).rstrip('/')}/sites/{site.slug}"

    return site


@router.post("/plan", response_model=WebsitePlan)
async def plan_website(request: WebsitePlanRequest):
    """
    Runs the Planner Agent: decides which pages are needed for the
    prompt and returns anything strictly necessary that's still missing
    (missing_fields). Call this first; once plan.ready is true (or the
    user has answered every missing_fields question), call /generate.
    """

    try:
        return website_service.plan(
            project_id=request.project_id,
            prompt=request.prompt,
            answers=request.answers,
        )

    except ProjectNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="Project not found. Create a business project first via /business/submit.",
        )

    except Exception as e:
        logger.exception("Website planning failed for project_id=%s", request.project_id)
        status = 429 if is_llm_rate_limited(e) else 500
        raise HTTPException(status_code=status, detail=friendly_llm_error_message(e))


@router.post("/generate", response_model=GeneratedWebsite)
async def generate_website(request: WebsiteGenerateRequest, http_request: Request):
    """
    Runs the full pipeline: Planner -> Content/Design/Layout agents ->
    Code Generation, and persists the result as this project's current
    website (overwriting any previous one). Returns the live-preview-
    ready HTML plus the specification/design behind it, plus site_url —
    the shareable link to the actual hosted page.
    """

    try:
        site = website_service.generate(
            project_id=request.project_id,
            prompt=request.prompt,
            answers=request.answers,
        )
        return _with_site_url(site, http_request)

    except ProjectNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="Project not found. Create a business project first via /business/submit.",
        )

    except Exception as e:
        logger.exception("Website generation failed for project_id=%s", request.project_id)
        status = 429 if is_llm_rate_limited(e) else 500
        raise HTTPException(status_code=status, detail=friendly_llm_error_message(e))


@router.post("/edit", response_model=GeneratedWebsite)
async def edit_website(request: WebsiteEditRequest, http_request: Request):
    """
    Runs the Edit Agent: applies a plain-English instruction ("make it
    darker", "add a pricing page") to the project's current generated
    website and returns the updated site for the Live Preview.
    """

    try:
        site = website_service.edit(
            project_id=request.project_id,
            site_id=request.site_id,
            instruction=request.instruction,
        )
        return _with_site_url(site, http_request)

    except WebsiteNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    except Exception as e:
        logger.exception("Website edit failed for project_id=%s", request.project_id)
        status = 429 if is_llm_rate_limited(e) else 500
        raise HTTPException(status_code=status, detail=friendly_llm_error_message(e))


@router.get("/{project_id}/widget", response_model=WidgetEmbedResponse)
async def get_widget_embed(project_id: str, request: Request):
    """
    Zero-friction Chat Widget path: no form, no generation step — just
    an embed snippet wired to the project's existing BusinessProfile +
    RAG data. The widget itself is served by this backend as an iframe
    (GET /widget, GET /widget-loader.js), so the snippet is built from
    this request's own base URL — it works the same on localhost as it
    does once deployed, no extra config needed.
    """

    try:
        return website_service.widget_embed(project_id, str(request.base_url))

    except ProjectNotFoundError:
        raise HTTPException(status_code=404, detail="Project not found.")


@router.delete("/{project_id}")
async def delete_website(project_id: str):
    """
    Deletes this project's currently generated website. This is what
    "Start Over" should call before the user submits a new prompt --
    otherwise the next prompt finds an existing site still in storage
    and gets routed through the Edit Agent (rewrite old HTML by
    instruction) instead of a genuine fresh generation grounded in the
    Business Profile + knowledge base. Safe to call even if nothing has
    been generated yet -- returns deleted=False rather than erroring.
    """

    deleted = website_service.delete(project_id)

    return {
        "success": True,
        "project_id": project_id,
        "deleted": deleted,
    }


@router.get("/{project_id}", response_model=GeneratedWebsite)
async def get_latest_website(project_id: str, request: Request):
    """
    Fetches the project's current generated website (e.g. to resume the
    preview after a page reload).
    """

    try:
        site = website_service.get_latest(project_id)
        return _with_site_url(site, request)

    except WebsiteNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="No generated website found for this project yet.",
        )