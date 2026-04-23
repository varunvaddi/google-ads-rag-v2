"""
Step 9 — Test llm_generator_node
Runs the first 3 nodes + LLM generation end to end
No graph yet — just chaining nodes manually
"""

from src.graph.nodes import (
    query_analyzer_node,
    retriever_node,
    reranker_node,
    llm_generator_node,
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

    # Chain all nodes manually — same order the graph will use
    state.update(query_analyzer_node(state))
    state.update(retriever_node(state))
    state.update(reranker_node(state))
    state.update(llm_generator_node(state))

    decision = state["decision"]
    print(f"\n{'─'*65}")
    print(f"Decision:      {decision.decision.upper()}")
    print(f"Confidence:    {decision.confidence:.1%}")
    print(f"Policy:        {decision.policy_section}")
    print(f"Justification: {decision.justification[:120]}")
    print(f"Escalation:    {decision.escalation_required}")
    print(f"Trace:         {state['node_trace']}")
    print(f"Latency:       {state['latency_ms']}")