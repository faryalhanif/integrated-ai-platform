from typing import Any, Dict, List

import httpx

from app.core.config import get_settings


class KnowledgeClientError(Exception):
    """Raised when the knowledge base service can't be reached or errors out."""


class KnowledgeClient:
    """
    Thin HTTP client around the Knowledge Base / vector-search service
    that your teammate's ingestion pipeline exposes.

    Expected contract (confirm/adjust with teammate as their API
    solidifies — nothing here is hardcoded, only the *shape* is assumed):

        POST {KNOWLEDGE_BASE_BASE_URL}{KNOWLEDGE_BASE_RETRIEVE_PATH}
        Headers: Authorization: Bearer {KNOWLEDGE_BASE_API_KEY}  (if set)
        Body:
            {
                "project_id": "<project_id>",
                "query": "<rewritten user query>",
                "top_k": 5
            }
        Response:
            {
                "results": [
                    {
                        "content": "...chunk text...",
                        "source": "policy.pdf",       # or a URL
                        "score": 0.83,
                        "metadata": {"page": 3}
                    },
                    ...
                ]
            }

    If the ingestion service isn't reachable, retrieve() raises
    KnowledgeClientError so the caller can decide how to degrade
    gracefully (e.g. answer with "no context found" instead of crashing
    the whole chat).
    """

    def __init__(self):
        settings = get_settings()

        self._base_url = settings.KNOWLEDGE_BASE_BASE_URL.rstrip("/")
        self._path = settings.KNOWLEDGE_BASE_RETRIEVE_PATH
        self._api_key = settings.KNOWLEDGE_BASE_API_KEY
        self._timeout = settings.KNOWLEDGE_BASE_TIMEOUT_SECONDS
        self._default_top_k = settings.KNOWLEDGE_BASE_TOP_K

    def _headers(self) -> Dict[str, str]:

        headers = {"Content-Type": "application/json"}

        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        return headers

    def retrieve(
        self,
        project_id: str,
        query: str,
        top_k: int = None,
    ) -> List[Dict[str, Any]]:
        """
        Returns a list of {content, source, score, metadata} dicts, most
        relevant first. Returns an empty list (rather than raising) if
        the service responds but has nothing relevant.
        """

        url = f"{self._base_url}{self._path}"

        payload = {
            "project_id": project_id,
            "query": query,
            "top_k": top_k or self._default_top_k,
        }

        try:
            with httpx.Client(timeout=self._timeout) as client:
                response = client.post(
                    url,
                    json=payload,
                    headers=self._headers(),
                )
                response.raise_for_status()
                data = response.json()

        except httpx.HTTPError as exc:
            raise KnowledgeClientError(
                f"Failed to reach knowledge base at {url}: {exc}"
            ) from exc

        return data.get("results", [])

    def delete_project(self, project_id: str) -> bool:
        """
        Best-effort delete of this project's knowledge base collection
        (DELETE {KNOWLEDGE_BASE_BASE_URL}/projects/{project_id}).

        Deliberately swallows connection errors and returns False rather
        than raising: deleting a chatbot project should still succeed
        (removing its profile/system prompt) even if the knowledge base
        service happens to be down, rather than blocking the whole
        delete on a service outage.
        """

        url = f"{self._base_url}/projects/{project_id}"

        try:
            with httpx.Client(timeout=self._timeout) as client:
                response = client.delete(url, headers=self._headers())
                response.raise_for_status()
                data = response.json()

        except httpx.HTTPError:
            return False

        return bool(data.get("deleted", False))