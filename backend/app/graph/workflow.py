from langgraph.graph import END, START, StateGraph

from app.graph.nodes import (
    guardrail_node,
    load_project_node,
    response_node,
    retrieve_node,
    rewrite_query_node,
)
from app.graph.routing import guardrail_router, load_project_router
from app.graph.state import RAGState


class RAGWorkflow:
    """
    Builds and compiles the LangGraph "Runtime Chat" workflow:

        Load Project Context
              -> Query Rewriter (resolves follow-ups/pronouns/typos
                 against conversation history into a standalone query)
                    -> Guardrails (judges the resolved query, so short
                       follow-ups and casual replies aren't misread as
                       unrelated/blocked in isolation)
                          -> Knowledge Client (retrieve, skipped for
                             casual chit-chat)
                                -> Prompt Builder + Response Agent
                                      -> Answer + Citations

    Note: "Load Project Context" runs first because both later stages
    need the project's business profile (restricted_topics for
    guardrails, brand_tone/system_prompt for responses).
    """

    def __init__(self):

        self.builder = StateGraph(RAGState)

    def build(self):

        self.builder.add_node(
            "load_project",
            load_project_node,
        )

        self.builder.add_node(
            "rewrite_query",
            rewrite_query_node,
        )

        self.builder.add_node(
            "guardrail",
            guardrail_node,
        )

        self.builder.add_node(
            "retrieve",
            retrieve_node,
        )

        self.builder.add_node(
            "response",
            response_node,
        )

        self.builder.add_edge(
            START,
            "load_project",
        )

        self.builder.add_conditional_edges(
            "load_project",
            load_project_router,
            {
                "ok": "rewrite_query",
                "error": END,
            },
        )

        self.builder.add_edge(
            "rewrite_query",
            "guardrail",
        )

        self.builder.add_conditional_edges(
            "guardrail",
            guardrail_router,
            {
                "allowed": "retrieve",
                "blocked": END,
            },
        )

        self.builder.add_edge(
            "retrieve",
            "response",
        )

        self.builder.add_edge(
            "response",
            END,
        )

        return self.builder.compile()