from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse

from app.services.website_store import WebsiteNotFoundError, get_website_store

router = APIRouter(tags=["Public Website Hosting"])


@router.get("/sites/{slug}", response_class=HTMLResponse)
async def serve_generated_site(slug: str):
    """
    The URL handed to end users at the end of website generation
    (GeneratedWebsite.site_url). Serves the actual live HTML page —
    not JSON, not a download — so opening the link just shows the
    website. In dev this is http://localhost:8000/sites/{slug}; once
    deployed behind a real domain it becomes https://yourdomain.com/sites/{slug}
    automatically, since the URL is built from the request's own host
    at request time (see app/api/website.py), not hardcoded anywhere.
    """

    try:
        site = get_website_store().get_by_slug(slug)
    except WebsiteNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="No website found at this address.",
        )

    return HTMLResponse(content=site.html)