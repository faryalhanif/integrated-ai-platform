import json
from pathlib import Path
from threading import Lock
from typing import Optional

from app.core.config import get_settings
from app.schemas.website import GeneratedWebsite


class WebsiteNotFoundError(Exception):
    """Raised when a project has no generated website yet."""


class JSONFileWebsiteStore:
    """
    Simple JSON-file-backed store for generated websites. One file per
    project at {storage_dir}/{project_id}.json, holding only the latest
    revision (edit_history keeps the instructions that led to it). Same
    zero-setup local-dev pattern as JSONFileProjectStore in
    prompt_service.py — swap in a database-backed implementation later
    without touching WebsiteGenerationService.
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

    def exists(self, project_id: str) -> bool:
        return self._path(project_id).exists()

    def delete(self, project_id: str) -> bool:
        path = self._path(project_id)

        with self._lock:
            if not path.exists():
                return False
            path.unlink()
            return True


_store: Optional[JSONFileWebsiteStore] = None
_store_lock = Lock()


def get_website_store() -> JSONFileWebsiteStore:
    """Returns a process-wide singleton WebsiteStore."""

    global _store

    if _store is None:
        with _store_lock:
            if _store is None:
                settings = get_settings()
                _store = JSONFileWebsiteStore(settings.WEBSITE_STORAGE_DIR)

    return _store
