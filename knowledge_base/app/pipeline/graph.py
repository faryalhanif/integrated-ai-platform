from langgraph.graph import END, StateGraph

from app.pipeline.chunking import chunking_node
from app.pipeline.embedding import embedder_node
from app.pipeline.parsers import PARSER_NODES, validator_node
from app.pipeline.state import PipelineState
from app.pipeline.vectorstore import upsert_node


class IngestionError(Exception):
    """Raised when the ingestion pipeline can't process the given input."""


def _route_after_validation(state: PipelineState) -> str:
    if not state.get("valid"):
        return "invalid"
    return state.get("file_type", "unsupported")


def build_ingestion_graph():
    """
    Builds and compiles the ingestion pipeline:

        validator -> (per-type parser) -> chunker -> embedder -> upsert

    This is the same pipeline as the original notebook, but wired as a
    reusable, importable graph instead of a script that blocks on
    input(). Callers pass file_path/url + project_id explicitly via
    run_ingestion_pipeline().
    """

    graph = StateGraph(PipelineState)

    graph.add_node("validator", validator_node)

    for name, node_fn in PARSER_NODES.items():
        graph.add_node(f"{name}_parser", node_fn)

    graph.add_node("chunker", chunking_node)
    graph.add_node("embedder", embedder_node)
    graph.add_node("upsert", upsert_node)

    graph.set_entry_point("validator")

    graph.add_conditional_edges(
        "validator",
        _route_after_validation,
        {
            "invalid": END,
            **{name: f"{name}_parser" for name in PARSER_NODES},
        },
    )

    for name in PARSER_NODES:
        graph.add_edge(f"{name}_parser", "chunker")

    graph.add_edge("chunker", "embedder")
    graph.add_edge("embedder", "upsert")
    graph.add_edge("upsert", END)

    return graph.compile()


_compiled_graph = None


def get_ingestion_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_ingestion_graph()
    return _compiled_graph


def run_ingestion_pipeline(
    project_id: str,
    file_path: str = None,
    url: str = None,
) -> PipelineState:
    """
    Runs the full ingestion pipeline for a single file or URL, storing
    the resulting chunks into the project's vector collection.

    Raises IngestionError if the input is invalid or nothing could be
    extracted; otherwise returns the final pipeline state (including
    chunks_ingested / collection_name).
    """

    if not file_path and not url:
        raise IngestionError("Either file_path or url must be provided.")

    graph = get_ingestion_graph()

    initial_state: PipelineState = {
        "project_id": project_id,
        "file_path": file_path,
        "url": url,
        "valid": False,
        "error": None,
        "documents": [],
        "chunked_documents": [],
        "vector_store_ready": False,
    }

    final_state = graph.invoke(initial_state)

    if not final_state.get("valid"):
        raise IngestionError(final_state.get("error") or "Invalid input.")

    if not final_state.get("vector_store_ready"):
        raise IngestionError(
            final_state.get("error")
            or "No content could be extracted from this input."
        )

    return final_state
