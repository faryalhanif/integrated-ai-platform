from threading import Lock
from typing import List

from sentence_transformers import SentenceTransformer

from app.core.config import get_settings
from app.pipeline.state import PipelineState

_model = None
_model_lock = Lock()


def get_embedding_model() -> SentenceTransformer:
    """
    Returns a process-wide singleton SentenceTransformer, configured
    from Settings.EMBEDDING_MODEL / EMBEDDING_DEVICE. Loaded lazily so
    the API can start up without waiting on the model download.
    """

    global _model

    if _model is None:
        with _model_lock:
            if _model is None:
                settings = get_settings()
                _model = SentenceTransformer(
                    settings.EMBEDDING_MODEL,
                    device=settings.EMBEDDING_DEVICE,
                )

    return _model


def embed_texts(texts: List[str]) -> List[List[float]]:
    model = get_embedding_model()
    vectors = model.encode(texts, normalize_embeddings=True)
    return vectors.tolist()


def embedder_node(state: PipelineState) -> PipelineState:
    chunked_docs = state.get("chunked_documents", [])

    if not chunked_docs:
        return {"vector_store_ready": False, "error": "No chunks to embed"}

    texts = [doc["page_content"] for doc in chunked_docs]
    vectors = embed_texts(texts)

    for doc, vector in zip(chunked_docs, vectors):
        doc["embedding"] = vector

    return {
        "embeddings": vectors,
        "chunked_documents": chunked_docs,
    }
