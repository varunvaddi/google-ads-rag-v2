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

# ─────────────────────────────────────────────────────────────────────────────
# Singleton loader — loads HybridSearch once, reuses across all queries
# ─────────────────────────────────────────────────────────────────────────────

_hybrid_search = None

def _get_search():
    """
    Lazy singleton for HybridSearch.
    First call: loads BGE + FAISS + BM25 (~20s)
    All subsequent calls: returns cached instance instantly
    """
    global _hybrid_search
    if _hybrid_search is None:
        print("[Retriever] Loading HybridSearch (first time only)...")
        from src.retrieval.hybrid_search import HybridSearch
        _hybrid_search = HybridSearch()
        print("[Retriever] HybridSearch loaded and cached")
    return _hybrid_search


# ─────────────────────────────────────────────────────────────────────────────
# Node 2: Retriever
# ─────────────────────────────────────────────────────────────────────────────

def retriever_node(state: RAGState) -> Dict[str, Any]:
    """
    Runs hybrid search (BM25 + FAISS + RRF) on the query.
    Uses expanded_query if query_analyzer produced one.

    INPUT  (reads from state): expanded_query, query, retrieval_attempts, query_type
    OUTPUT (writes to state):  retrieved_chunks, retrieval_attempts, latency_ms, node_trace

    Your existing HybridSearch code is unchanged.
    This node is just a clean wrapper around it.
    """
    t0 = time.time()
    search = _get_search()

    # Use expanded query if available, fall back to original
    query = state.get("expanded_query") or state["query"]

    # Track how many times retriever has run
    attempts = state.get("retrieval_attempts", 0) + 1

    # On second attempt (retry), fetch more chunks to cast wider net
    if attempts == 1:
        top_k = 5
    else:
        top_k = 8
        print(f"[Retriever] Second attempt — widening to top_k={top_k}")

    print(f"\n[Retriever] attempt={attempts}, top_k={top_k}")
    print(f"[Retriever] query: '{query[:70]}...'")

    chunks = search.search(query, top_k=top_k)

    print(f"[Retriever] got {len(chunks)} chunks")
    if chunks:
        top_score = chunks[0].get("rerank_score", chunks[0].get("score", 0))
        print(f"[Retriever] top chunk score: {top_score:.4f}")
        top_hierarchy = " > ".join(chunks[0].get("metadata", {}).get("hierarchy", []))
        print(f"[Retriever] top chunk: {top_hierarchy[:60]}")

    # ── Observability ─────────────────────────────────────────────────────
    latency = state.get("latency_ms") or {}
    latency[f"retriever_attempt_{attempts}"] = round((time.time() - t0) * 1000, 1)

    trace = state.get("node_trace") or []
    trace.append(f"retriever_attempt_{attempts}")

    return {
        "retrieved_chunks": chunks,
        "retrieval_attempts": attempts,
        "latency_ms": latency,
        "node_trace": trace,
    }

# ─────────────────────────────────────────────────────────────────────────────
# Node 3: Reranker
# ─────────────────────────────────────────────────────────────────────────────

def reranker_node(state: RAGState) -> Dict[str, Any]:
    """
    Reads rerank scores from retrieved chunks and decides retrieval quality.

    Cross-encoder reranking already happened inside HybridSearch.search().
    This node's job is to:
      1. Extract the scores
      2. Decide if quality is good enough to proceed to LLM
      3. Set needs_more_retrieval flag for the conditional edge

    INPUT  (reads from state): retrieved_chunks, retrieval_attempts
    OUTPUT (writes to state):  reranked_chunks, retrieval_scores,
                               needs_more_retrieval, latency_ms, node_trace

    ROUTING LOGIC:
      top_score >= 0.001 AND attempts < 2  → proceed to LLM
      top_score <  0.001 AND attempts < 2  → retry retrieval
      attempts >= 2                         → proceed anyway (avoid infinite loop)
    """
    t0 = time.time()
    chunks = state.get("retrieved_chunks", [])
    attempts = state.get("retrieval_attempts", 1)

    # Extract rerank scores from chunks
    # rerank_score is set by cross-encoder inside HybridSearch
    scores = [
        chunk.get("rerank_score", chunk.get("combined_score", 0.0))
        for chunk in chunks
    ]

    top_score = scores[0] if scores else 0.0

    print(f"\n[Reranker] {len(chunks)} chunks | top_score={top_score:.4f} | attempts={attempts}")

    # Print top 3 with their scores and hierarchy
    for i, (chunk, score) in enumerate(zip(chunks[:3], scores[:3]), 1):
        hierarchy = " > ".join(chunk.get("metadata", {}).get("hierarchy", []))
        print(f"  {i}. [{score:.4f}] {hierarchy[:55]}")

    # ── Routing decision ──────────────────────────────────────────────────
    # THRESHOLD: 0.001 is the minimum acceptable rerank score
    # Below this = cross-encoder found no meaningful relevance
    # Above this = at least one chunk is genuinely relevant
    QUALITY_THRESHOLD = 0.001

    if top_score < QUALITY_THRESHOLD and attempts < 2:
        # Quality too low AND we haven't retried yet → retry
        needs_more = True
        print(f"[Reranker] score {top_score:.4f} < {QUALITY_THRESHOLD} → RETRY retrieval")
    else:
        # Either quality is OK, or we've already retried → proceed
        needs_more = False
        reason = "quality OK" if top_score >= QUALITY_THRESHOLD else "max attempts reached"
        print(f"[Reranker] {reason} → PROCEED to LLM")

    # ── Observability ─────────────────────────────────────────────────────
    latency = state.get("latency_ms") or {}
    latency["reranker"] = round((time.time() - t0) * 1000, 1)

    trace = state.get("node_trace") or []
    trace.append("reranker")

    return {
        "reranked_chunks": chunks,       # same chunks — scores already inside them
        "retrieval_scores": scores,
        "needs_more_retrieval": needs_more,
        "latency_ms": latency,
        "node_trace": trace,
    }

# ─────────────────────────────────────────────────────────────────────────────
# Conditional Edge: after reranker, retry or proceed?
# ─────────────────────────────────────────────────────────────────────────────

def should_retrieve_more(state: RAGState) -> str:
    """
    Called by LangGraph after reranker_node runs.
    Returns a string — the name of the next node.

    "retriever"     → go back and retry with wider search
    "llm_generator" → quality is good, proceed to generation
    """
    if state.get("needs_more_retrieval", False):
        return "retriever"
    return "llm_generator"