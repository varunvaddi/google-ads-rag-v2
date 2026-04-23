"""
Step 8 — Test reranker_node
Shows routing decision based on score quality
"""

from src.graph.nodes import query_analyzer_node, retriever_node, reranker_node, should_retrieve_more

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
    "Lose 15 pounds in one week with this miracle pill!",
    "Buy our new laptop - Intel i7, 16GB RAM",
]

for query in test_queries:
    print(f"\n{'='*65}")
    print(f"Query: {query}")
    print(f"{'='*65}")

    state = make_state(query)

    # Run nodes in sequence, updating state each time
    state.update(query_analyzer_node(state))
    state.update(retriever_node(state))
    state.update(reranker_node(state))

    # Check routing decision
    next_node = should_retrieve_more(state)

    print(f"\n→ Router decision:       '{next_node}'")
    print(f"→ needs_more_retrieval:  {state['needs_more_retrieval']}")
    print(f"→ top rerank score:      {state['retrieval_scores'][0]:.4f}")
    print(f"→ node_trace:            {state['node_trace']}")