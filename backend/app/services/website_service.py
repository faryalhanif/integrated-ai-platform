import logging
from datetime import datetime, timezone
from typing import Any, Dict, List
from uuid import uuid4

from app.agents.website_codegen_agent import WebsiteCodeGeneratorAgent
from app.agents.website_content_agent import WebsiteContentAgent
from app.agents.website_design_agent import WebsiteDesignAgent
from app.agents.website_edit_agent import WebsiteEditAgent
from app.agents.website_layout_agent import WebsiteLayoutAgent
from app.agents.website_planner_agent import WebsitePlannerAgent
from app.rag.knowledge_client import KnowledgeClient, KnowledgeClientError
from app.schemas.website import GeneratedWebsite, WebsitePlan, WidgetEmbedResponse
from app.services.llm_service import LLMService
from app.services.prompt_service import get_project_store
from app.services.website_store import WebsiteNotFoundError, get_website_store

logger = logging.getLogger(__name__)


class WebsiteGenerationService:
    """
    Orchestrates the full website-generation workflow shown in the
    architecture diagram:

        User Prompt
            -> Planner Agent (reads Business Profile + RAG + prompt)
                -> Website Specification
                    -> Content / Design / Layout Agents (parallel-ish)
                        -> LLM API
                            -> Code Generation
                                -> Live Preview
                                    -> Edit Agent (chat-based edits, loop)

    Plus the zero-friction Chat Widget path, which skips all of the
    above and just returns an embed snippet pointed at the project's
    already-existing BusinessProfile + RAG data.
    """

    def __init__(self):
        self.llm = LLMService.get_website_llm()
        self.edit_llm = LLMService.get_website_edit_llm()

        self.planner = WebsitePlannerAgent(self.llm)
        self.content_agent = WebsiteContentAgent(self.llm)
        self.design_agent = WebsiteDesignAgent(self.llm)
        self.layout_agent = WebsiteLayoutAgent(self.llm)
        self.codegen_agent = WebsiteCodeGeneratorAgent(self.llm)
        self.edit_agent = WebsiteEditAgent(self.edit_llm)

        self.project_store = get_project_store()
        self.website_store = get_website_store()
        self.knowledge_client = KnowledgeClient()

    # ------------------------------------------------------------
    # Full Website path
    # ------------------------------------------------------------

    def _rag_sample(
        self,
        project_id: str,
        query: str,
        top_k: int = 8,
    ) -> List[Dict[str, Any]]:
        """
        Best-effort knowledge-base sample used to ground planning and
        copywriting. Never blocks the pipeline — an unreachable or empty
        knowledge base just means the Planner/Content agents see less
        context (and, correctly, may ask the user for more).
        """

        try:
            return self.knowledge_client.retrieve(
                project_id=project_id,
                query=query,
                top_k=top_k,
            )
        except KnowledgeClientError:
            logger.warning(
                "Knowledge base unreachable while planning/generating "
                "website for project_id=%s — continuing without RAG context.",
                project_id,
            )
            return []

    def plan(
        self,
        project_id: str,
        prompt: str,
        answers: Dict[str, str] = None,
    ) -> WebsitePlan:
        """
        Runs the Planner Agent and returns the Website Specification.
        If plan.missing_fields is non-empty, the caller should collect
        answers from the user and call plan() (or generate()) again with
        them — the Planner treats supplied answers as authoritative and
        won't re-ask for the same thing twice.
        """

        profile = self.project_store.get_business_profile(project_id)

        rag_sample = self._rag_sample(project_id, prompt)

        return self.planner.run(
            business_profile=profile,
            user_prompt=prompt,
            knowledge_base_sample=rag_sample,
            answers=answers or {},
        )

    def generate(
        self,
        project_id: str,
        prompt: str,
        answers: Dict[str, str] = None,
    ) -> GeneratedWebsite:
        """
        Runs the full pipeline and persists (overwriting any previous
        site for this project): Planner -> Content/Design/Layout ->
        Code Generation -> saved GeneratedWebsite.

        Proceeds even if the (re-)planned missing_fields isn't fully
        empty — the Content Agent is instructed to write honest,
        non-specific copy for anything ungrounded rather than block
        generation entirely, so the user always gets a working preview
        to iterate on with the Edit Agent.
        """

        answers = answers or {}

        profile = self.project_store.get_business_profile(project_id)

        rag_sample = self._rag_sample(project_id, prompt, top_k=12)

        plan = self.planner.run(
            business_profile=profile,
            user_prompt=prompt,
            knowledge_base_sample=rag_sample,
            answers=answers,
        )

        content = self.content_agent.run(
            business_profile=profile,
            plan=plan,
            knowledge_base_sample=rag_sample,
            answers=answers,
        )

        design = self.design_agent.run(business_profile=profile, plan=plan)

        layout = self.layout_agent.run(plan=plan, content=content)

        html = self.codegen_agent.run(
            business_profile=profile,
            plan=plan,
            content=content,
            design=design,
            layout=layout,
        )

        now = datetime.now(timezone.utc).isoformat()

        site = GeneratedWebsite(
            site_id=str(uuid4()),
            project_id=project_id,
            prompt=prompt,
            plan=plan,
            design=design,
            html=html,
            revision=1,
            created_at=now,
            updated_at=now,
            edit_history=[],
        )

        self.website_store.save(site)

        return site

    def edit(
        self,
        project_id: str,
        site_id: str,
        instruction: str,
    ) -> GeneratedWebsite:
        """
        Applies a chat-based edit to the project's current site and
        persists the new revision.
        """

        site = self.website_store.get(project_id)

        if site.site_id != site_id:
            raise WebsiteNotFoundError(
                f"site_id={site_id!r} does not match the current "
                f"generated website for project_id={project_id!r} "
                f"(current site_id={site.site_id!r})."
            )

        updated_html = self.edit_agent.run(site.html, instruction)

        site.html = updated_html
        site.revision += 1
        site.updated_at = datetime.now(timezone.utc).isoformat()
        site.edit_history.append(
            {"instruction": instruction, "at": site.updated_at}
        )

        self.website_store.save(site)

        return site

    def get_latest(self, project_id: str) -> GeneratedWebsite:
        return self.website_store.get(project_id)

    # ------------------------------------------------------------
    # Chat Widget path — zero friction, no generation step at all
    # ------------------------------------------------------------

    def widget_embed(self, project_id: str, backend_base_url: str) -> WidgetEmbedResponse:
        """
        The Chat Widget path never asks the user for anything: the
        widget is instantiated straight from the project's existing
        BusinessProfile + RAG data via the /chat endpoint.

        The widget itself is served by this same backend as an iframe
        (see app/api/widget.py) — the returned snippet only ever touches
        the host page's DOM to inject that one iframe; the chat UI's own
        CSS/JS lives inside it, isolated from the host site.
        """

        profile = self.project_store.get_business_profile(project_id)

        backend_base_url = backend_base_url.rstrip("/")

        config = {
            "project_id": project_id,
            "business_name": profile.business_name,
            "brand_tone": profile.brand_tone,
            "primary_color": (profile.brand_colors or {}).get("primary", "#4f46e5"),
            "logo_url": str(profile.logo_url) if profile.logo_url else None,
        }

        embed_snippet = (
            "<!-- {business_name} chat widget — paste before </body> -->\n"
            '<script\n'
            '  src="{backend_base_url}/widget-loader.js"\n'
            '  data-project-id="{project_id}"\n'
            '  data-backend-url="{backend_base_url}"\n'
            '  data-primary-color="{primary_color}"\n'
            "  async\n"
            "></script>"
        ).format(
            business_name=profile.business_name,
            backend_base_url=backend_base_url,
            project_id=project_id,
            primary_color=config["primary_color"],
        )

        iframe_snippet = (
            "<!-- {business_name} chat widget — direct iframe, no script.\n"
            "     Simpler to paste, but always occupies this fixed area\n"
            "     of the page (no collapse-to-bubble behavior). -->\n"
            '<iframe\n'
            '  src="{backend_base_url}/widget?project_id={project_id}&primary_color={primary_color_encoded}"\n'
            '  title="Chat widget"\n'
            '  style="position:fixed;bottom:16px;right:16px;width:370px;height:600px;'
            'max-width:calc(100vw - 24px);max-height:calc(100vh - 24px);'
            'border:none;border-radius:14px;box-shadow:0 10px 40px rgba(0,0,0,.25);z-index:999999;"\n'
            "></iframe>"
        ).format(
            business_name=profile.business_name,
            backend_base_url=backend_base_url,
            project_id=project_id,
            primary_color_encoded=config["primary_color"].replace("#", "%23"),
        )

        return WidgetEmbedResponse(
            project_id=project_id,
            embed_snippet=embed_snippet,
            iframe_snippet=iframe_snippet,
            config=config,
        )