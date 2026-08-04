from typing import Any, Dict, List, Literal, Optional, TypedDict


class PipelineState(TypedDict, total=False):
    """
    Shared state passed between the LangGraph ingestion pipeline nodes.
    """

    file_path: Optional[str]
    url: Optional[str]
    project_id: str

    file_type: Literal[
        "pdf", "pptx", "docx", "markdown", "text", "image", "web", "unsupported"
    ]

    valid: bool
    error: Optional[str]

    documents: List[Dict[str, Any]]
    chunked_documents: List[Dict[str, Any]]

    embeddings: List[List[float]]
    vector_store_ready: bool

    chunks_ingested: int
    collection_name: str
