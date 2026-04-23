"""
Step 6 — Test query_analyzer_node in isolation
We don't need the full graph to test a node — just call it like a function
"""

from src.graph.nodes import query_analyzer_node

# Manually build a minimal state to pass in
test_queries = [
    "Buy our new laptop - Intel i7, 16GB RAM",
    "Learn cryptocurrency trading from experts",
    "Lose 15 pounds in one week with this miracle pill! Guaranteed!",
    "100% guaranteed returns on Bitcoin investment",
]

for query in test_queries:
    print(f"\n{'='*60}")

    # Build minimal state — only fields this node reads
    state = {
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

    result = query_analyzer_node(state)

    print(f"query_type:      {result['query_type']}")
    print(f"expanded_query:  {result['expanded_query']}")
    print(f"latency_ms:      {result['latency_ms']}")
    print(f"node_trace:      {result['node_trace']}")