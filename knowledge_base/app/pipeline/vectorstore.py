import re
import logging
import uuid
from threading import Lock
from typing import Any, Dict, List

import chromadb

from app.core.config import get_settings
from app.pipeline.state import PipelineState

logger = logging.getLogger(__name__)

_client = None
_client_lock = Lock()


def sanitize_collection_name(raw_name: str) -> str:
    """
    Chroma collection names must be 3-63 characters and contain only
    alphanumerics, underscores, and hyphens. Applied identically at
    ingest time and retrieval time so a given project_id always maps
    to the same collection.
    """
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", raw_name).strip("_-").lower()

    if len(safe) < 3:
        safe = (safe + "_kb")[:63]

    return safe[:63]


def get_chroma_client() -> chromadb.ClientAPI:
    global _client

    if _client is None:
        with _client_lock:
            if _client is None:
                settings = get_settings()
                _client = chromadb.PersistentClient(
                    path=settings.CHROMA_PERSIST_DIR
                )

    return _client


def get_or_create_collection(project_id: str):
    settings = get_settings()
    client = get_chroma_client()

    return client.get_or_create_collection(
        name=sanitize_collection_name(project_id),
        metadata={"hnsw:space": settings.CHROMA_DISTANCE_METRIC},
    )


def upsert_node(state: PipelineState) -> PipelineState:
    """
    LangGraph node: writes embedded chunks into this project's Chroma
    collection.
    """
    chunked_docs = state.get("chunked_documents", [])
    project_id = state["project_id"]

    if not chunked_docs:
        return {"vector_store_ready": False, "chunks_ingested": 0}

    collection = get_or_create_collection(project_id)
    collection_name = sanitize_collection_name(project_id)

    ids: List[str] = []
    embeddings: List[List[float]] = []
    metadatas: List[Dict[str, Any]] = []
    documents: List[str] = []

    for doc in chunked_docs:
        ids.append(str(uuid.uuid4()))
        embeddings.append(doc["embedding"])
        documents.append(doc.get("page_content", ""))

        meta = {
            k: v
            for k, v in doc.get("metadata", {}).items()
            if isinstance(v, (str, int, float, bool)) and v is not None
        }
        meta["project_id"] = project_id
        metadatas.append(meta)

    collection.add(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas,
        documents=documents,
    )

    return {
        "vector_store_ready": True,
        "chunks_ingested": len(ids),
        "collection_name": collection_name,
    }


def query_collection(
    project_id: str,
    query_embedding: List[float],
    top_k: int,
) -> List[Dict[str, Any]]:
    """
    Queries the given project's collection and returns
    {content, source, score, metadata} dicts, most relevant first.
    Returns [] if the collection doesn't exist yet (e.g. nothing has
    been ingested for this project) rather than raising.
    """

    client = get_chroma_client()
    collection_name = sanitize_collection_name(project_id)

    try:
        collection = client.get_collection(name=collection_name)
    except Exception:
        logger.warning(
            "query_collection: no collection named %r exists (from "
            "project_id=%r) — nothing has been ingested for this "
            "project yet, or project_id doesn't match what was used "
            "at ingest time.",
            collection_name,
            project_id,
        )
        return []

    if collection.count() == 0:
        logger.warning(
            "query_collection: collection %r exists but is empty (0 chunks).",
            collection_name,
        )
        return []

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    formatted: List[Dict[str, Any]] = []

    for content, metadata, distance in zip(documents, metadatas, distances):
        metadata = metadata or {}
        # Collection uses cosine space, so distance = 1 - cosine_similarity.
        score = max(0.0, min(1.0, 1.0 - distance))

        formatted.append(
            {
                "content": content,
                "source": metadata.get("source", "unknown source"),
                "score": round(score, 4),
                "metadata": metadata,
            }
        )

    return formatted


def list_projects() -> List[Dict[str, Any]]:
    client = get_chroma_client()
    collections = client.list_collections()

    out = []
    for col in collections:
        collection = client.get_collection(name=col.name)
        out.append({"collection_name": col.name, "chunk_count": collection.count()})

    return out


def delete_project(project_id: str) -> bool:
    client = get_chroma_client()
    collection_name = sanitize_collection_name(project_id)

    try:
        client.delete_collection(name=collection_name)
        return True
    except Exception:
        return False