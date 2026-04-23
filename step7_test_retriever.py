"""
Step 7 — Test retriever_node in isolation
This will actually load your BGE model + FAISS + BM25
and run real retrieval on your 341 chunks
"""

from src.graph.nodes import query_analyzer_node, retriever_node

# Base state template — reuse for all tests
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
    "Learn cryptocurrency trading from certified experts",
    "Buy our new laptop - Intel i7, 16GB RAM",
]

for query in test_queries:
    print(f"\n{'='*65}")
    print(f"Query: {query}")
    print(f"{'='*65}")

    state = make_state(query)

    # Run node 1 first to get expanded query
    state.update(query_analyzer_node(state))

    # Run node 2 — real retrieval
    result = retriever_node(state)

    chunks = result["retrieved_chunks"]
    print(f"\nChunks retrieved: {len(chunks)}")
    print(f"Latency: {result['latency_ms']}")
    print(f"Trace: {result['node_trace']}")

    print(f"\nTop 3 results:")
    for i, chunk in enumerate(chunks[:3], 1):
        hierarchy = " > ".join(chunk.get("metadata", {}).get("hierarchy", []))
        score = chunk.get("rerank_score", chunk.get("score", 0))
        print(f"  {i}. [{score:.4f}] {hierarchy[:60]}")