"""
src/graph/state.py

The shared state dictionary for the entire RAG pipeline.
Every node reads from this and writes back to it.

Think of it as the single source of truth — like a baton
passed between runners. Each runner (node) picks it up,
does their job, adds their result, passes it on.
"""

from typing import TypedDict, Optional, List, Dict, Any
from src.generation.decision_schema import PolicyDecision


class RAGState(TypedDict):
    """
    Full pipeline state — all fields a node might need.

    Convention:
      - Set at start     → query (never changes)
      - Filled by nodes  → everything else (starts as None)
      - Read by router   → needs_more_retrieval, escalate
    """

    # ── Input ──────────────────────────────────────────────────────────
    query: str
    # The original ad text. Set once when pipeline starts. Never modified.

    query_type: Optional[str]
    # "simple" | "complex" | "borderline"
    # Set by query_analyzer node based on keywords in the query.
    # Used by retriever to decide how many chunks to fetch.

    expanded_query: Optional[str]
    # If query_analyzer rewrites the query for better retrieval,
    # it stores the expanded version here.
    # Example: "bitcoin investment" → "bitcoin investment cryptocurrency financial services"

    # ── Retrieval ──────────────────────────────────────────────────────
    retrieved_chunks: Optional[List[Dict[str, Any]]]
    # Raw output from HybridSearch.search()
    # List of dicts with keys: content, metadata, score, rerank_score
    # Filled by retriever node.

    reranked_chunks: Optional[List[Dict[str, Any]]]
    # Same chunks after cross-encoder reranking.
    # Filled by reranker node.
    # This is what gets passed to the LLM as context.

    retrieval_attempts: int
    # How many times retriever has run.
    # Starts at 0. Incremented each time retriever node runs.
    # Used to prevent infinite retry loops (max 2 attempts).

    retrieval_scores: Optional[List[float]]
    # Top rerank scores from the cross-encoder.
    # Example: [0.0045, 0.0032, 0.0021, 0.0018, 0.0012]
    # Used by reranker node to decide if quality is good enough.

    # ── Generation ─────────────────────────────────────────────────────
    decision: Optional[PolicyDecision]
    # The Pydantic PolicyDecision object returned by the LLM.
    # Filled by llm_generator node.
    # Modified by validator node (confidence recalculation, policy override).

    raw_llm_response: Optional[str]
    # Raw string from Ollama before Pydantic parsing.
    # Kept for debugging — useful when LLM returns malformed JSON.

    # ── Routing flags ──────────────────────────────────────────────────
    needs_more_retrieval: bool
    # Set by reranker node.
    # True  → retrieval quality too low → go back to retriever
    # False → quality OK → proceed to llm_generator

    escalate: bool
    # Set by validator node.
    # True  → confidence too low → route to escalation node
    # False → confidence OK → route to END

    # ── Observability ──────────────────────────────────────────────────
    error: Optional[str]
    # Any error message from any node.
    # If set, pipeline can short-circuit to END gracefully.

    latency_ms: Optional[Dict[str, float]]
    # Per-node timing in milliseconds.
    # Example: {"query_analyzer": 2.1, "retriever": 340.5, "reranker": 2800.2}
    # Filled by each node. Used for P95 latency evaluation.

    node_trace: Optional[List[str]]
    # List of node names in the order they ran.
    # Example: ["query_analyzer", "retriever", "reranker", "llm_generator", "validator"]
    # Useful for debugging conditional routing decisions.