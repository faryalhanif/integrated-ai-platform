from functools import lru_cache
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Global application configuration.

    Everything that could differ between environments (dev/staging/prod)
    or between deployments lives here and is loaded from the .env file.
    NOTHING in the rest of the codebase should hardcode URLs, model
    names, API keys, or business-specific values.
    """

    # ---------------------------------
    # Application
    # ---------------------------------
    APP_NAME: str = "AI Website Generation Platform"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"

    # Comma separated list of allowed CORS origins, e.g.
    # "https://app.example.com,https://admin.example.com"
    # Use "*" to allow all (fine for local dev only).
    CORS_ORIGINS: str = "*"

    # ---------------------------------
    # LLM
    # ---------------------------------
    LLM_PROVIDER: str
    LLM_MODEL: str
    LLM_API_KEY: str

    LLM_TEMPERATURE: float = 0.2
    LLM_MAX_TOKENS: int = 2048

    # Provider SDKs (e.g. langchain-google-genai) retry failed calls with
    # growing backoff by default (often 5-6 attempts, ~30s total) even
    # for errors like a daily quota being exhausted, which can never
    # succeed by retrying sooner. A low default here fails fast instead
    # of making the caller wait ~30s just to see the same error.
    LLM_MAX_RETRIES: int = 2

    # ---------------------------------
    # Website Generation LLM
    # ---------------------------------
    # Website generation (planning + full-page HTML code generation) has
    # very different needs from the short, cheap chat-turn LLM above: much
    # longer outputs, and — per the "free during testing, Claude/GPT later
    # in production" plan — often a different provider entirely. All three
    # fields are optional and fall back to the LLM_* values above when
    # left blank, so a single-provider setup (e.g. everything on Groq)
    # still works with zero extra config.
    WEBSITE_LLM_PROVIDER: str = ""
    WEBSITE_LLM_MODEL: str = ""
    WEBSITE_LLM_API_KEY: str = ""

    WEBSITE_LLM_TEMPERATURE: float = 0.4
    WEBSITE_LLM_MAX_TOKENS: int = 32768
    WEBSITE_LLM_MAX_RETRIES: int = 2

    # Edits should be precise, not creative — a lower temperature than
    # initial generation noticeably improves how reliably the Edit Agent
    # applies exactly the requested change and nothing else.
    WEBSITE_EDIT_TEMPERATURE: float = 0.1

    # Directory where generated websites (spec + design + HTML) are
    # persisted as JSON, one file per project — same zero-setup pattern
    # as STORAGE_DIR above.
    WEBSITE_STORAGE_DIR: str = "./data/websites"

    # ---------------------------------
    # Project / Business Profile storage
    # ---------------------------------
    # Directory where BusinessProfile + generated system prompts are
    # persisted as JSON. Only used when DATABASE_URL (below) is unset —
    # kept as the zero-setup local-dev fallback.
    STORAGE_DIR: str = "./data/projects"

    # PostgreSQL connection string for production project storage, e.g.
    #   postgresql+psycopg2://app_user:password@localhost:5432/integrated_ai_platform
    # When this is set, get_project_store() (app/services/prompt_service.py)
    # uses PostgresProjectStore instead of JSONFileProjectStore — no other
    # file needs to change. Leave empty to keep using JSON files.
    DATABASE_URL: str = ""

    # ---------------------------------
    # Knowledge Base (owned by teammate's ingestion pipeline)
    # ---------------------------------
    # Base URL of the Knowledge Base / vector-search service that exposes
    # the retrieval endpoint documented in app/rag/knowledge_client.py.
    KNOWLEDGE_BASE_BASE_URL: str
    KNOWLEDGE_BASE_API_KEY: str = ""
    KNOWLEDGE_BASE_RETRIEVE_PATH: str = "/knowledge/retrieve"
    KNOWLEDGE_BASE_TOP_K: int = 5
    KNOWLEDGE_BASE_TIMEOUT_SECONDS: float = 15.0

    # ---------------------------------
    # Guardrails
    # ---------------------------------
    GUARDRAILS_ENABLED: bool = True

    # ---------------------------------
    # Backend API access control
    # ---------------------------------
    # Protects the project-management endpoints (list/get/delete
    # projects) from being readable/deletable by anyone who can reach
    # this service. Optional bearer-token auth, same pattern as the
    # knowledge base service's SERVICE_API_KEY: if unset (default,
    # local dev), those endpoints are open. Set this in .env for any
    # environment reachable by more than just your own trusted admin
    # frontend, and send it as `Authorization: Bearer <key>`.
    BACKEND_API_KEY: str = ""

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
    )

    @property
    def cors_origins_list(self) -> List[str]:
        if self.CORS_ORIGINS.strip() == "*":
            return ["*"]
        return [
            origin.strip()
            for origin in self.CORS_ORIGINS.split(",")
            if origin.strip()
        ]


@lru_cache
def get_settings() -> Settings:
    """
    Returns a cached Settings instance.
    """
    return Settings()