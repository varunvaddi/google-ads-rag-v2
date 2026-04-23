"""
src/graph/nodes.py

All LangGraph nodes for the RAG pipeline.
Each function = one node.
Each node receives full RAGState, returns dict of only what it changes.

We build this file step by step:
  Step 6  → query_analyzer_node
  Step 7  → retriever_node
  Step 8  → reranker_node
  Step 9  → llm_generator_node
  Step 10 → validator_node + escalation_node
"""

import time
from typing import Any, Dict
from src.graph.state import RAGState


# ─────────────────────────────────────────────────────────────────────────────
# Node 1: Query Analyzer
# ─────────────────────────────────────────────────────────────────────────────

def query_analyzer_node(state: RAGState) -> Dict[str, Any]:
    """
    Classifies the query and optionally expands it.

    INPUT  (reads from state): query
    OUTPUT (writes to state):  query_type, expanded_query, latency_ms, node_trace

    No LLM needed here — pure rule-based logic.
    Fast, deterministic, zero cost.
    """
    t0 = time.time()
    query = state["query"]
    query_lower = query.lower()

    print(f"\n[QueryAnalyzer] query: '{query}'")

    # ── Keyword signal lists ───────────────────────────────────────────────
    # These match the same patterns your v1 hybrid_search.py already used
    COMPLEX_SIGNALS = [
        "miracle", "cure", "guaranteed return", "100% profit",
        "no prescription", "counterfeit", "fake", "get rich quick",
        "lose weight fast", "instant cure", "risk-free",
    ]
    BORDERLINE_SIGNALS = [
        "crypto", "bitcoin", "ethereum", "alcohol", "whiskey",
        "beer", "wine", "political", "election", "vote",
        "gambling", "pharmacy", "prescription", "forex",
    ]

    # Financial guarantee pattern — same logic as your v1 search expansion
    FINANCIAL_GUARANTEE = (
        any(w in query_lower for w in ["guarantee", "guaranteed", "promised", "risk-free"])
        and
        any(w in query_lower for w in ["return", "profit", "income", "%", "apy", "apr"])
    )

    # ── Classify ──────────────────────────────────────────────────────────
    if any(sig in query_lower for sig in COMPLEX_SIGNALS) or FINANCIAL_GUARANTEE:
        query_type = "complex"
    elif any(sig in query_lower for sig in BORDERLINE_SIGNALS):
        query_type = "borderline"
    else:
        query_type = "simple"

    # ── Expand query for better retrieval ─────────────────────────────────
    # We add synonyms so BM25 keyword search AND semantic search
    # both fire on the right policy sections
    expanded_query = query   # default: no expansion

    if FINANCIAL_GUARANTEE:
        expanded_query += " unreliable claims improbable result misrepresentation"
    elif "miracle" in query_lower or "cure" in query_lower:
        expanded_query += " unapproved health claims prohibited medical treatment"
    elif "crypto" in query_lower or "bitcoin" in query_lower:
        expanded_query += " cryptocurrency digital currency financial services"

    was_expanded = expanded_query != query

    # ── Observability: timing + trace ─────────────────────────────────────
    latency = state.get("latency_ms") or {}
    latency["query_analyzer"] = round((time.time() - t0) * 1000, 1)

    trace = state.get("node_trace") or []
    trace.append("query_analyzer")

    print(f"[QueryAnalyzer] type={query_type} | expanded={was_expanded}")
    if was_expanded:
        print(f"[QueryAnalyzer] expanded: '{expanded_query}'")

    # ── Return only what this node changed ────────────────────────────────
    return {
        "query_type": query_type,
        "expanded_query": expanded_query,
        "latency_ms": latency,
        "node_trace": trace,
    }