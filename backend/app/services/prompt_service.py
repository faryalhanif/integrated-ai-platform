import json
from abc import ABC, abstractmethod
from pathlib import Path
from threading import Lock
from typing import Optional

from app.core.config import get_settings
from app.schemas.business import BusinessProfile


class ProjectNotFoundError(Exception):
    """Raised when a project_id has no stored profile/system prompt."""


class ProjectStore(ABC):
    """
    Storage abstraction for a project's BusinessProfile + generated
    system prompt.

    Swap JSONFileProjectStore for a real database-backed implementation
    later (e.g. PostgresProjectStore) without touching the onboarding
    service, the graph nodes, or anything else that depends on this
    interface.
    """

    @abstractmethod
    def save_project(
        self,
        profile: BusinessProfile,
        system_prompt: str,
    ) -> None:
        ...

    @abstractmethod
    def get_business_profile(self, project_id: str) -> BusinessProfile:
        ...

    @abstractmethod
    def get_system_prompt(self, project_id: str) -> str:
        ...

    @abstractmethod
    def exists(self, project_id: str) -> bool:
        ...

    @abstractmethod
    def list_projects(self) -> list:
        """Returns lightweight summaries of every stored project, most
        recently created first."""
        ...

    @abstractmethod
    def delete_project(self, project_id: str) -> bool:
        """Permanently removes a project's stored profile + system
        prompt. Returns True if a project was deleted, False if no such
        project existed (idempotent — not an error)."""
        ...


class JSONFileProjectStore(ProjectStore):
    """
    Simple JSON-file-backed ProjectStore. Good enough for MVP / local
    development. One file per project at
    {storage_dir}/{project_id}.json.
    """

    def __init__(self, storage_dir: str):
        self._dir = Path(storage_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def _path(self, project_id: str) -> Path:
        return self._dir / f"{project_id}.json"

    def save_project(
        self,
        profile: BusinessProfile,
        system_prompt: str,
    ) -> None:

        payload = {
            "business_profile": json.loads(profile.model_dump_json()),
            "system_prompt": system_prompt,
        }

        with self._lock:
            self._path(profile.project_id).write_text(
                json.dumps(payload, indent=2),
                encoding="utf-8",
            )

    def _load(self, project_id: str) -> dict:

        path = self._path(project_id)

        if not path.exists():
            raise ProjectNotFoundError(
                f"No project found for project_id={project_id!r}"
            )

        with self._lock:
            return json.loads(path.read_text(encoding="utf-8"))

    def get_business_profile(self, project_id: str) -> BusinessProfile:

        data = self._load(project_id)

        return BusinessProfile(**data["business_profile"])

    def get_system_prompt(self, project_id: str) -> str:

        data = self._load(project_id)

        return data["system_prompt"]

    def exists(self, project_id: str) -> bool:

        return self._path(project_id).exists()

    def list_projects(self) -> list:

        summaries = []

        with self._lock:
            for path in self._dir.glob("*.json"):
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    profile = data.get("business_profile", {})
                    summaries.append(
                        {
                            "project_id": profile.get("project_id", path.stem),
                            "business_name": profile.get("business_name", "Untitled"),
                            "brand_tone": profile.get("brand_tone"),
                            "created_at": profile.get("created_at"),
                        }
                    )
                except (json.JSONDecodeError, OSError):
                    # Skip unreadable/corrupt files rather than failing
                    # the whole listing.
                    continue

        summaries.sort(key=lambda p: p.get("created_at") or "", reverse=True)

        return summaries

    def delete_project(self, project_id: str) -> bool:

        path = self._path(project_id)

        with self._lock:
            if not path.exists():
                return False
            path.unlink()
            return True


class PostgresProjectStore(ProjectStore):
    """
    PostgreSQL-backed ProjectStore for production. Stores each project
    as one row: the full BusinessProfile as JSONB (so new profile
    fields never require a schema migration), plus a couple of plain
    columns (business_name, created_at) so listing/sorting doesn't need
    to unpack JSON for every row.

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
                    CREATE TABLE IF NOT EXISTS chatbot_projects (
                        project_id TEXT PRIMARY KEY,
                        business_name TEXT NOT NULL,
                        business_profile JSONB NOT NULL,
                        system_prompt TEXT NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                    )
                    """
                )
            )
            conn.execute(
                self._text(
                    """
                    CREATE INDEX IF NOT EXISTS idx_chatbot_projects_created_at
                    ON chatbot_projects (created_at DESC)
                    """
                )
            )

    def save_project(
        self,
        profile: BusinessProfile,
        system_prompt: str,
    ) -> None:

        payload = json.dumps(json.loads(profile.model_dump_json()))

        with self._engine.begin() as conn:
            conn.execute(
                self._text(
                    """
                    INSERT INTO chatbot_projects
                        (project_id, business_name, business_profile, system_prompt, created_at)
                    VALUES
                        (:project_id, :business_name, CAST(:business_profile AS JSONB), :system_prompt, :created_at)
                    ON CONFLICT (project_id) DO UPDATE SET
                        business_name = EXCLUDED.business_name,
                        business_profile = EXCLUDED.business_profile,
                        system_prompt = EXCLUDED.system_prompt
                    """
                ),
                {
                    "project_id": profile.project_id,
                    "business_name": profile.business_name,
                    "business_profile": payload,
                    "system_prompt": system_prompt,
                    "created_at": profile.created_at,
                },
            )

    def get_business_profile(self, project_id: str) -> BusinessProfile:

        with self._engine.connect() as conn:
            row = conn.execute(
                self._text(
                    "SELECT business_profile FROM chatbot_projects WHERE project_id = :pid"
                ),
                {"pid": project_id},
            ).fetchone()

        if row is None:
            raise ProjectNotFoundError(
                f"No project found for project_id={project_id!r}"
            )

        return BusinessProfile(**row[0])

    def get_system_prompt(self, project_id: str) -> str:

        with self._engine.connect() as conn:
            row = conn.execute(
                self._text(
                    "SELECT system_prompt FROM chatbot_projects WHERE project_id = :pid"
                ),
                {"pid": project_id},
            ).fetchone()

        if row is None:
            raise ProjectNotFoundError(
                f"No project found for project_id={project_id!r}"
            )

        return row[0]

    def exists(self, project_id: str) -> bool:

        with self._engine.connect() as conn:
            row = conn.execute(
                self._text(
                    "SELECT 1 FROM chatbot_projects WHERE project_id = :pid"
                ),
                {"pid": project_id},
            ).fetchone()

        return row is not None

    def list_projects(self) -> list:

        with self._engine.connect() as conn:
            rows = conn.execute(
                self._text(
                    """
                    SELECT
                        project_id,
                        business_name,
                        business_profile ->> 'brand_tone' AS brand_tone,
                        business_profile ->> 'created_at' AS created_at
                    FROM chatbot_projects
                    ORDER BY created_at DESC
                    """
                )
            ).fetchall()

        return [
            {
                "project_id": r.project_id,
                "business_name": r.business_name,
                "brand_tone": r.brand_tone,
                "created_at": r.created_at,
            }
            for r in rows
        ]

    def delete_project(self, project_id: str) -> bool:

        with self._engine.begin() as conn:
            result = conn.execute(
                self._text("DELETE FROM chatbot_projects WHERE project_id = :pid"),
                {"pid": project_id},
            )

        return result.rowcount > 0


_store: Optional[ProjectStore] = None
_store_lock = Lock()


def get_project_store() -> ProjectStore:
    """
    Returns a process-wide singleton ProjectStore.

    Uses PostgresProjectStore when Settings.DATABASE_URL is set
    (production), otherwise falls back to JSONFileProjectStore at
    Settings.STORAGE_DIR (zero-setup local dev).
    """

    global _store

    if _store is None:
        with _store_lock:
            if _store is None:
                settings = get_settings()
                if settings.DATABASE_URL:
                    _store = PostgresProjectStore(settings.DATABASE_URL)
                else:
                    _store = JSONFileProjectStore(settings.STORAGE_DIR)

    return _store