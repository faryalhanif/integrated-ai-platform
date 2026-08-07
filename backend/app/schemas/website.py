from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class WebsitePageSpec(BaseModel):
    """
    One page (or, in the single-file preview build, one in-page section
    group) that the Planner Agent decided the site actually needs.
    """

    name: str = Field(..., description="Human-readable page name, e.g. 'Pricing'.")

    slug: str = Field(..., description="URL-safe identifier, e.g. 'pricing'.")

    purpose: str = Field(
        ...,
        description="One sentence on why this page exists / what it should accomplish.",
    )

    sections: List[str] = Field(
        default_factory=list,
        description=(
            "Ordered content blocks this page should contain, e.g. "
            "['hero', 'features', 'cta']. Refined further by the Layout Agent."
        ),
    )


class MissingField(BaseModel):
    """
    A single piece of information the Planner Agent could not find in the
    BusinessProfile or the knowledge base, and that is strictly required
    to build a page the prompt asked for (e.g. a physical address for a
    'Visit us' section on the Contact page).
    """

    key: str = Field(..., description="Stable identifier, e.g. 'physical_address'.")

    question: str = Field(..., description="Plain-language question to show the user.")

    why: Optional[str] = Field(
        default=None,
        description="Why this is needed, e.g. 'Requested a Contact page with a map'.",
    )


class WebsitePlan(BaseModel):
    """
    The Planner Agent's output: the internal blueprint referenced in the
    architecture diagram as 'Website Specification'. Read by the Content,
    Design, and Layout agents.
    """

    pages: List[WebsitePageSpec] = Field(default_factory=list)

    missing_fields: List[MissingField] = Field(
        default_factory=list,
        description="Empty once every strictly-necessary detail is available.",
    )

    ready: bool = Field(
        ...,
        description="True once missing_fields is empty and generation can proceed.",
    )

    notes: Optional[str] = Field(
        default=None,
        description="Short explanation of the plan, shown to the user for transparency.",
    )


class WebsitePlanRequest(BaseModel):
    project_id: str = Field(..., min_length=1)

    prompt: str = Field(
        ...,
        min_length=5,
        description="e.g. 'Build a modern SaaS website with pricing and a blog'.",
    )

    answers: Dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Answers to a previous plan's missing_fields (keyed by MissingField.key). "
            "Re-planning with answers supplied lets the plan resolve to ready=true."
        ),
    )


class WebsiteGenerateRequest(BaseModel):
    project_id: str = Field(..., min_length=1)

    prompt: str = Field(..., min_length=5)

    answers: Dict[str, str] = Field(
        default_factory=dict,
        description="Answers to the plan's missing_fields, keyed by MissingField.key.",
    )


class WebsiteEditRequest(BaseModel):
    project_id: str = Field(..., min_length=1)

    site_id: str = Field(..., min_length=1)

    instruction: str = Field(
        ...,
        min_length=3,
        description="Plain-English edit, e.g. 'Make it darker' or 'Add a pricing page'.",
    )


class GeneratedWebsite(BaseModel):
    """
    A generated website, persisted per project. Only the latest revision
    is kept (edit_history records the instructions that produced it).
    """

    site_id: str

    project_id: str

    prompt: str

    plan: WebsitePlan

    design: Dict[str, Any] = Field(default_factory=dict)

    html: str = Field(..., description="Complete, self-contained HTML document.")

    revision: int = 1

    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    updated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    edit_history: List[Dict[str, str]] = Field(default_factory=list)


class WidgetEmbedResponse(BaseModel):
    """
    Zero-friction chat-widget embed: unlike the Full Website path there is
    no form to fill and nothing to generate — the snippet just points the
    widget loader at the project's existing BusinessProfile + RAG data.
    """

    project_id: str

    embed_snippet: str = Field(
        ...,
        description=(
            "Recommended: a <script> tag that injects a small, collapsed "
            "bubble which expands into an iframe on click. Doesn't block "
            "clicks elsewhere on the host page while collapsed."
        ),
    )

    iframe_snippet: str = Field(
        ...,
        description=(
            "Alternative: a single, self-contained <iframe> tag — no "
            "companion script. Simpler to paste, but occupies a fixed "
            "area of the page at all times (no collapse/expand)."
        ),
    )

    config: Dict[str, Any] = Field(default_factory=dict)