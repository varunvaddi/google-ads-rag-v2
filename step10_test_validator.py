"""
Step 10 — Test validator_node
Key thing to watch: crypto should get overridden from ALLOWED → RESTRICTED
"""

from src.graph.nodes import (
    query_analyzer_node,
    retriever_node,
    reranker_node,
    llm_generator_node,
    validator_node,
    should_escalate,
)

def make_state(query):
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

test_queries = [
    "Lose 15 pounds in one week with this miracle pill! Guaranteed!",
    "Learn cryptocurrency trading from certified experts",
    "Buy our new laptop - Intel i7, 16GB RAM, free shipping",
]

for query in test_queries:
    print(f"\n{'='*65}")
    print(f"Query: {query}")
    print(f"{'='*65}")

    state = make_state(query)
    state.update(query_analyzer_node(state))
    state.update(retriever_node(state))
    state.update(reranker_node(state))
    state.update(llm_generator_node(state))
    state.update(validator_node(state))

    decision = state["decision"]
    next_node = should_escalate(state)

    print(f"\n{'─'*65}")
    print(f"FINAL Decision:   {decision.decision.upper()}")
    print(f"FINAL Confidence: {decision.confidence:.1%}")
    print(f"Escalate:         {state['escalate']} → next: '{next_node}'")
    print(f"Trace:            {state['node_trace']}")