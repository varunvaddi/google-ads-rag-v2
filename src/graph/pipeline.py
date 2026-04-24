"""
src/graph/pipeline.py

Assembles all nodes into a LangGraph state machine.
This is the v2 replacement for run_phase4_Generation.py

GRAPH TOPOLOGY:
                    ┌─────────────────┐
                    │  query_analyzer  │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
               ┌───►│    retriever    │
               │    └────────┬────────┘
               │             │
               │    ┌────────▼────────┐
               │    │    reranker     │
               │    └────────┬────────┘
               │             │
               │    quality too low?
               └────────────YES
                             │NO
                    ┌────────▼────────┐
                    │  llm_generator  │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │    validator    │
                    └────────┬────────┘
                             │
                    confidence < 0.7?
                    YES              NO
                     │               │
            ┌────────▼───┐        ┌──▼──┐
            │ escalation │        │ END │
            └────────────┘        └─────┘
"""

import time
from langgraph.graph import StateGraph, END

from src.graph.state import RAGState
from src.graph.nodes import (
    query_analyzer_node,
    retriever_node,
    reranker_node,
    llm_generator_node,
    validator_node,
    escalation_node,
    should_retrieve_more,
    should_escalate,
)
from src.generation.decision_schema import PolicyDecision


def build_graph():
    """
    Assembles and compiles the LangGraph pipeline.
    Called once at startup — reused for every query.
    """
    graph = StateGraph(RAGState)

    # ── Add nodes ──────────────────────────────────────────────────────────
    graph.add_node("query_analyzer", query_analyzer_node)
    graph.add_node("retriever",      retriever_node)
    graph.add_node("reranker",       reranker_node)
    graph.add_node("llm_generator",  llm_generator_node)
    graph.add_node("validator",      validator_node)
    graph.add_node("escalation",     escalation_node)

    # ── Entry point ────────────────────────────────────────────────────────
    graph.set_entry_point("query_analyzer")

    # ── Fixed edges ────────────────────────────────────────────────────────
    # These always go A → B with no conditions
    graph.add_edge("query_analyzer", "retriever")
    graph.add_edge("retriever",      "reranker")
    graph.add_edge("llm_generator",  "validator")
    graph.add_edge("escalation",     END)

    # ── Conditional edge 1: after reranker ────────────────────────────────
    # should_retrieve_more() returns "retriever" or "llm_generator"
    graph.add_conditional_edges(
        "reranker",
        should_retrieve_more,
        {
            "retriever":     "retriever",     # retry retrieval
            "llm_generator": "llm_generator", # proceed to LLM
        }
    )

    # ── Conditional edge 2: after validator ───────────────────────────────
    # should_escalate() returns "escalation" or "end"
    graph.add_conditional_edges(
        "validator",
        should_escalate,
        {
            "escalation": "escalation", # low confidence → human review
            "end":        END,          # high confidence → done
        }
    )

    return graph.compile()


class RAGPipeline:
    """
    Clean public interface for the v2 RAG pipeline.
    Drop-in replacement for GeminiPolicyEngine from v1.

    Usage:
        pipeline = RAGPipeline()
        decision = pipeline.run("your ad text here")
        pipeline.print_decision(decision)
    """

    def __init__(self):
        print("🚀 Building LangGraph RAG pipeline...")
        print("   LLM: Ollama llama3.2 (local, free, unlimited)")
        print("   Retrieval: BGE-large + FAISS + BM25 + RRF + Cross-Encoder")
        print("   Orchestration: LangGraph state machine")
        self.graph = build_graph()
        print("✅ Pipeline ready!\n")

    def _make_initial_state(self, query: str) -> RAGState:
        """Build clean initial state for a new query."""
        return {
            "query": query,
            "query_type": None,
            "expanded_query": None,
            "retrieved_chunks": None,
            "reranked_chunks": None,
            "retrieval_attempts": 0,
            "retrieval_scores": None,
            "decision": None,
            "raw_llm_response": None,
            "needs_more_retrieval": False,
            "escalate": False,
            "error": None,
            "latency_ms": {},
            "node_trace": [],
        }

    def run(self, query: str) -> PolicyDecision:
        """
        Run the full pipeline for a single query.

        Args:
            query: Ad text or policy question

        Returns:
            PolicyDecision with decision, confidence, citations, etc.
        """
        t_total = time.time()
        print(f"\n{'='*65}")
        print(f"🔍 Query: {query}")
        print(f"{'='*65}")

        # Build initial state and invoke the graph
        initial_state = self._make_initial_state(query)
        final_state = self.graph.invoke(initial_state)

        # Attach total latency
        total_ms = round((time.time() - t_total) * 1000, 1)
        if final_state.get("latency_ms"):
            final_state["latency_ms"]["total"] = total_ms

        # Print pipeline summary
        trace = " → ".join(final_state.get("node_trace", []))
        print(f"\n📊 Trace:   {trace}")
        print(f"⏱  Total:   {total_ms:.0f}ms")

        return final_state["decision"]

    def print_decision(self, decision: PolicyDecision):
        """Pretty print a PolicyDecision."""
        print("\n" + "=" * 65)
        print("📋 POLICY DECISION  [v2 — Ollama + LangGraph]")
        print("=" * 65)
        print(f"🎯 Decision:     {decision.decision.upper()}")
        print(f"📊 Confidence:   {decision.confidence:.1%}")
        print(f"📂 Policy:       {decision.policy_section}")
        print(f"🔗 Source:       {decision.citation_url}")
        print(f"💬 Justification:\n   {decision.justification[:200]}")

        if decision.risk_factors:
            print(f"⚠️  Risk Factors:")
            for f in decision.risk_factors:
                print(f"   • {f}")

        if decision.escalation_required:
            print(f"🚨 ESCALATED — routed for human review")

        print("=" * 65)