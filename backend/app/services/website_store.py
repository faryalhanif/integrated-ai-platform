import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from threading import Lock
from typing import Optional

from app.core.config import get_settings
from app.schemas.website import GeneratedWebsite


class WebsiteNotFoundError(Exception):
    """Raised when a project has no generated website yet."""


class WebsiteStore(ABC):
    """
    Storage abstraction for a project's generated website.

    Swap JSONFileWebsiteStore for PostgresWebsiteStore (or any other
    implementation) without touching WebsiteGenerationService or the
    API layer — same pattern as ProjectStore in prompt_service.py.
    """

    @abstractmethod
    def save(self, site: GeneratedWebsite) -> None:
        ...

    @abstractmethod
    def get(self, project_id: str) -> GeneratedWebsite:
        ...

    @abstractmethod
    def get_by_slug(self, slug: str) -> GeneratedWebsite:
        ...

    @abstractmethod
    def exists(self, project_id: str) -> bool:
        ...

    @abstractmethod
    def delete(self, project_id: str) -> bool:
        ...


class JSONFileWebsiteStore(WebsiteStore):
    """
    Simple JSON-file-backed store for generated websites. One file per
    project at {storage_dir}/{project_id}.json, holding only the latest
    revision (edit_history keeps the instructions that led to it).
    Good enough for MVP / local development — PostgresWebsiteStore below
    is used automatically instead whenever DATABASE_URL is set.
    """

    def __init__(self, storage_dir: str):
        self._dir = Path(storage_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def _path(self, project_id: str) -> Path:
        return self._dir / f"{project_id}.json"

    def save(self, site: GeneratedWebsite) -> None:
        with self._lock:
            self._path(site.project_id).write_text(
                site.model_dump_json(indent=2),
                encoding="utf-8",
            )

    def get(self, project_id: str) -> GeneratedWebsite:
        path = self._path(project_id)

        if not path.exists():
            raise WebsiteNotFoundError(
                f"No generated website found for project_id={project_id!r}"
            )

        with self._lock:
            data = json.loads(path.read_text(encoding="utf-8"))

        return GeneratedWebsite(**data)

    def get_by_slug(self, slug: str) -> GeneratedWebsite:
        """
        Looks up a site by its public URL slug (GET /sites/{slug}).
        Linear scan over the storage dir — fine at JSON-file-store
        scale; PostgresWebsiteStore below does this with an indexed
        query instead.
        """

        with self._lock:
            for path in self._dir.glob("*.json"):
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    continue
                if data.get("slug") == slug:
                    return GeneratedWebsite(**data)

        raise WebsiteNotFoundError(f"No website found at slug={slug!r}")

    def exists(self, project_id: str) -> bool:
        return self._path(project_id).exists()

    def delete(self, project_id: str) -> bool:
        path = self._path(project_id)

        with self._lock:
            if not path.exists():
                return False
            path.unlink()
            return True


class PostgresWebsiteStore(WebsiteStore):
    """
    PostgreSQL-backed WebsiteStore for production — mirrors
    PostgresProjectStore in prompt_service.py exactly (same
    lazy-import, same engine setup, same auto-create-table-if-missing
    approach). One row per project (only the latest revision is kept,
    same as the JSON store); plan/design/edit_history are stored as
    JSONB so their shape can evolve without a migration. slug has a
    unique index (allowing multiple NULLs, since Postgres treats NULL
    as distinct from itself in a UNIQUE constraint) to back the public
    GET /sites/{slug} lookup with an indexed query instead of a scan.

    Table is created automatically on first use if it doesn't exist —
    fine at this scale; move to a real migration tool (e.g. Alembic) if
    the schema starts changing often.
    """

    def __init__(self, database_url: str):
        # Imported lazily so environments that never set DATABASE_URL
        # (e.g. local dev on the JSON-file store) don't need sqlalchemy
        # or psycopg2 installed at all.
        from sqlalchemy import create_engine, text

        self._text = text
        self._engine = create_engine(database_url, pool_pre_ping=True)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with self._engine.begin() as conn:
            conn.execute(
                self._text(
                    """
                    CREATE TABLE IF NOT EXISTS generated_websites (
                        project_id TEXT PRIMARY KEY,
                        site_id TEXT NOT NULL,
                        slug TEXT,
                        prompt TEXT NOT NULL,
                        plan JSONB NOT NULL,
                        design JSONB NOT NULL,
                        html TEXT NOT NULL,
                        revision INTEGER NOT NULL DEFAULT 1,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                        edit_history JSONB NOT NULL DEFAULT '[]'::jsonb
                    )
                    """
                )
            )
            conn.execute(
                self._text(
                    """
                    CREATE UNIQUE INDEX IF NOT EXISTS idx_generated_websites_slug
                    ON generated_websites (slug) WHERE slug IS NOT NULL
                    """
                )
            )

    def save(self, site: GeneratedWebsite) -> None:
        with self._engine.begin() as conn:
            conn.execute(
                self._text(
                    """
                    INSERT INTO generated_websites
                        (project_id, site_id, slug, prompt, plan, design, html,
                         revision, created_at, updated_at, edit_history)
                    VALUES
                        (:project_id, :site_id, :slug, :prompt,
                         CAST(:plan AS JSONB), CAST(:design AS JSONB), :html,
                         :revision, :created_at, :updated_at, CAST(:edit_history AS JSONB))
                    ON CONFLICT (project_id) DO UPDATE SET
                        site_id = EXCLUDED.site_id,
                        slug = EXCLUDED.slug,
                        prompt = EXCLUDED.prompt,
                        plan = EXCLUDED.plan,
                        design = EXCLUDED.design,
                        html = EXCLUDED.html,
                        revision = EXCLUDED.revision,
                        updated_at = EXCLUDED.updated_at,
                        edit_history = EXCLUDED.edit_history
                    """
                ),
                {
                    "project_id": site.project_id,
                    "site_id": site.site_id,
                    "slug": site.slug,
                    "prompt": site.prompt,
                    "plan": site.plan.model_dump_json(),
                    "design": json.dumps(site.design),
                    "html": site.html,
                    "revision": site.revision,
                    "created_at": site.created_at,
                    "updated_at": site.updated_at,
                    "edit_history": json.dumps(site.edit_history),
                },
            )

    @staticmethod
    def _row_to_site(row) -> GeneratedWebsite:
        return GeneratedWebsite(
            site_id=row.site_id,
            project_id=row.project_id,
            slug=row.slug,
            prompt=row.prompt,
            plan=row.plan,
            design=row.design,
            html=row.html,
            revision=row.revision,
            created_at=row.created_at.isoformat()
            if hasattr(row.created_at, "isoformat")
            else str(row.created_at),
            updated_at=row.updated_at.isoformat()
            if hasattr(row.updated_at, "isoformat")
            else str(row.updated_at),
            edit_history=row.edit_history or [],
        )

    def get(self, project_id: str) -> GeneratedWebsite:
        with self._engine.connect() as conn:
            row = conn.execute(
                self._text(
                    "SELECT * FROM generated_websites WHERE project_id = :pid"
                ),
                {"pid": project_id},
            ).fetchone()

        if row is None:
            raise WebsiteNotFoundError(
                f"No generated website found for project_id={project_id!r}"
            )

        return self._row_to_site(row)

    def get_by_slug(self, slug: str) -> GeneratedWebsite:
        with self._engine.connect() as conn:
            row = conn.execute(
                self._text("SELECT * FROM generated_websites WHERE slug = :slug"),
                {"slug": slug},
            ).fetchone()

        if row is None:
            raise WebsiteNotFoundError(f"No website found at slug={slug!r}")

        return self._row_to_site(row)

    def exists(self, project_id: str) -> bool:
        with self._engine.connect() as conn:
            row = conn.execute(
                self._text(
                    "SELECT 1 FROM generated_websites WHERE project_id = :pid"
                ),
                {"pid": project_id},
            ).fetchone()

        return row is not None

    def delete(self, project_id: str) -> bool:
        with self._engine.begin() as conn:
            result = conn.execute(
                self._text("DELETE FROM generated_websites WHERE project_id = :pid"),
                {"pid": project_id},
            )

        return result.rowcount > 0


_store: Optional[WebsiteStore] = None
_store_lock = Lock()


def get_website_store() -> WebsiteStore:
    """
    Returns a process-wide singleton WebsiteStore.

    Uses PostgresWebsiteStore when Settings.DATABASE_URL is set
    (production — same DATABASE_URL that already backs ProjectStore),
    otherwise falls back to JSONFileWebsiteStore at
    Settings.WEBSITE_STORAGE_DIR (zero-setup local dev). Same
    selection logic as get_project_store() in prompt_service.py.
    """

    global _store

    if _store is None:
        with _store_lock:
            if _store is None:
                settings = get_settings()
                db_url = settings.DATABASE_URL.strip()
                placeholder_urls = {
                    "postgresql+psycopg2://app_user:change_me@localhost:5432/integrated_ai_platform",
                    "postgresql://app_user:change_me@localhost:5432/integrated_ai_platform",
                }

                if db_url and db_url not in placeholder_urls:
                    try:
                        _store = PostgresWebsiteStore(settings.DATABASE_URL)
                    except Exception:
                        logging.warning(
                            "Could not initialize PostgresWebsiteStore; falling back to JSONFileWebsiteStore.",
                            exc_info=True,
                        )
                        _store = JSONFileWebsiteStore(settings.website_storage_dir_abs)
                else:
                    _store = JSONFileWebsiteStore(settings.website_storage_dir_abs)

    return _store