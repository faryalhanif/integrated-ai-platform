from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict

# This file lives at knowledge_base/app/core/config.py -- three parents
# up is the knowledge_base package root, used below to anchor relative
# storage paths so they resolve the same way regardless of the process's
# current working directory at launch time (see chroma_persist_dir_abs /
# upload_tmp_dir_abs).
_PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """
    Global configuration for the Knowledge Base / ingestion + retrieval
    service.

    Nothing in the rest of this codebase should hardcode API keys,
    model names, file paths, or ports — everything that could differ
    between environments lives here and is loaded from `.env`.
    """

    # ---------------------------------
    # Application
    # ---------------------------------
    APP_NAME: str = "Knowledge Base Service"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"

    # Host/port this service listens on (used by `python main.py`; when
    # run via `uvicorn main:app` these are informational only).
    HOST: str = "0.0.0.0"
    PORT: int = 8001

    # Comma separated list of allowed CORS origins, or "*" for local dev.
    CORS_ORIGINS: str = "*"

    # ---------------------------------
    # LLM (used only to turn extracted tables into concise summaries
    # before embedding — not required for plain text ingestion or for
    # retrieval)
    # ---------------------------------
    LLM_PROVIDER: str = "groq"
    LLM_MODEL: str = "llama-3.1-8b-instant"
    LLM_API_KEY: str = ""
    LLM_TEMPERATURE: float = 0.0

    # If no LLM key is configured, table extraction falls back to a
    # plain markdown-to-text flatten instead of an LLM summary, rather
    # than crashing ingestion.
    TABLE_SUMMARIZATION_ENABLED: bool = True

    # ---------------------------------
    # Embeddings
    # ---------------------------------
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DEVICE: str = "cpu"

    # ---------------------------------
    # Vector store (local persistent ChromaDB)
    # ---------------------------------
    CHROMA_PERSIST_DIR: str = "./data/vector_store"
    CHROMA_DISTANCE_METRIC: str = "cosine"

    # ---------------------------------
    # Chunking
    # ---------------------------------
    CHUNK_SIZE: int = 600
    CHUNK_OVERLAP: int = 90

    # ---------------------------------
    # Retrieval
    # ---------------------------------
    DEFAULT_TOP_K: int = 5
    # Chunks scoring below this (cosine similarity, 0-1) are treated as
    # "not found" rather than handed to the LLM as context. This is the
    # main lever against hallucination on off-topic/out-of-domain
    # questions: without it, a weakly-related chunk still counts as
    # "context found" and the model may answer from it anyway instead
    # of falling back. Raise this further (e.g. 0.5) for stricter
    # grounding at the cost of occasionally missing a real answer;
    # lower it if legitimate answers are getting filtered out.
    #
    # Lowered from 0.35 -> 0.28 based on an observed real-world miss:
    # a genuinely correct chunk ("Team - Priya Raman, Co-founder & CEO...")
    # scored ~0.40 for a keyword-heavy query ("team members founders
    # leadership") but a natural conversational phrasing of the same
    # question ("who is the founder or the co founder?") scored below
    # 0.35 and got filtered out entirely -- all-MiniLM-L6-v2 is a small,
    # fast embedding model and its cosine scores for a correct match run
    # noticeably lower than a stronger model's would. 0.28 gives enough
    # headroom for that gap while still filtering genuinely unrelated
    # chunks. If off-topic questions start getting weakly-grounded
    # answers instead of a clean fallback, raise this back up; if real
    # answers are still getting missed, lower it further.
    MIN_RELEVANCE_SCORE: float = 0.28

    # ---------------------------------
    # File uploads
    # ---------------------------------
    UPLOAD_TMP_DIR: str = "./data/uploads"
    MAX_UPLOAD_SIZE_MB: int = 50

    # ---------------------------------
    # Web ingestion
    # ---------------------------------
    # Optional external extraction microservice (e.g. a headless-browser
    # / readability service your team runs separately). If unset or
    # unreachable, web_parser falls back to a built-in httpx + BeautifulSoup
    # extraction so URL ingestion still works out of the box.
    WEB_EXTRACTION_SERVICE_URL: str = ""
    WEB_EXTRACTION_SERVICE_PATH: str = "/extract"
    WEB_FETCH_TIMEOUT_SECONDS: float = 30.0
    # Many sites (Wikipedia included) return 403 Forbidden to requests
    # with no User-Agent or a generic library one (e.g. "python-httpx/x.x")
    # as basic bot protection. Send an identifiable one instead. Override
    # in .env if you want to identify your own service/company by name.
    WEB_FETCH_USER_AGENT: str = (
        "Mozilla/5.0 (compatible; IntegratedAIPlatformBot/1.0; "
        "+https://github.com/your-org/integrated-ai-platform)"
    )

    # ---------------------------------
    # OCR (for image ingestion)
    # ---------------------------------
    TESSERACT_CMD: str = ""  # leave empty to use tesseract from PATH

    # API key required from callers of this service (e.g. the RAG
    # backend). Leave empty to disable auth (local dev only). Matches
    # KNOWLEDGE_BASE_API_KEY on the backend side.
    SERVICE_API_KEY: str = ""

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

    @property
    def chroma_persist_dir_abs(self) -> str:
        """
        Resolves CHROMA_PERSIST_DIR to an absolute path anchored to this
        package's own location -- not the process's current working
        directory. A bare relative path here previously meant the vector
        store silently pointed at a *different, empty* folder any time
        this service happened to be launched from a different cwd (e.g.
        after re-extracting/re-downloading a fresh copy of the project),
        which looked exactly like "my previously ingested documents
        disappeared" even though the original data was untouched, just
        sitting at another path on disk. Already-absolute values (e.g.
        an explicit .env override) are used as-is, unchanged.
        """
        path = Path(self.CHROMA_PERSIST_DIR)
        return str(path if path.is_absolute() else (_PACKAGE_ROOT / path).resolve())

    @property
    def upload_tmp_dir_abs(self) -> str:
        """Same fix as chroma_persist_dir_abs, for uploaded source files."""
        path = Path(self.UPLOAD_TMP_DIR)
        return str(path if path.is_absolute() else (_PACKAGE_ROOT / path).resolve())


@lru_cache
def get_settings() -> Settings:
    """Returns a cached Settings instance."""
    return Settings()