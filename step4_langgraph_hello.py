"""
Step 4 — First LangGraph graph
Concepts: StateGraph, nodes, edges, state, invoke()

We build the simplest possible graph first:
    [node_a] → [node_b] → END

Then add a conditional edge so you can see how routing works.
"""

from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END


# ── 1. STATE ──────────────────────────────────────────────────────────────
# TypedDict = a dictionary with typed fields
# Every node reads from this and writes back to it
# Think of it as a baton passed between runners in a relay race
class MyState(TypedDict):
    query: str                    # set at the start, never changes
    retrieved: Optional[str]      # filled by node_a
    final_answer: Optional[str]   # filled by node_b
    quality_score: float          # filled by node_a, read by router


# ── 2. NODES ──────────────────────────────────────────────────────────────
# A node is just a plain Python function that:
#   - receives the full state as input
#   - returns a DICT of only the fields it wants to update
#   (you don't return the whole state — just your changes)

def node_retrieve(state: MyState) -> dict:
    """Simulates retrieval — pretends to find policy chunks"""
    query = state["query"]
    print(f"  [node_retrieve] got query: '{query}'")

    # Simulate finding something relevant
    retrieved_text = f"Policy chunk relevant to: {query}"
    quality = 0.85 if "crypto" in query.lower() else 0.3

    print(f"  [node_retrieve] quality_score = {quality}")

    # Return ONLY what this node changes
    return {
        "retrieved": retrieved_text,
        "quality_score": quality,
    }


def node_generate(state: MyState) -> dict:
    """Simulates LLM generation using retrieved context"""
    retrieved = state["retrieved"]
    print(f"  [node_generate] using context: '{retrieved}'")

    answer = f"Based on policy: {retrieved} → Decision: RESTRICTED"
    return {"final_answer": answer}


def node_retry(state: MyState) -> dict:
    """Called when quality is too low — expands the query and retrieves again"""
    print(f"  [node_retry] quality too low! Expanding query...")
    expanded = state["query"] + " policy rules regulations"
    retrieved_text = f"Expanded policy chunk for: {expanded}"

    return {
        "retrieved": retrieved_text,
        "quality_score": 0.9,   # assume retry improved quality
    }


# ── 3. CONDITIONAL EDGE FUNCTION ──────────────────────────────────────────
# This function reads the state and returns a STRING
# that string = the name of the next node to go to
def route_after_retrieval(state: MyState) -> str:
    if state["quality_score"] < 0.5:
        print(f"  [router] score={state['quality_score']} → retry")
        return "retry"
    print(f"  [router] score={state['quality_score']} → generate")
    return "generate"


# ── 4. BUILD THE GRAPH ────────────────────────────────────────────────────
graph = StateGraph(MyState)

# Add nodes (name → function)
graph.add_node("retrieve", node_retrieve)
graph.add_node("generate", node_generate)
graph.add_node("retry", node_retry)

# Set entry point
graph.set_entry_point("retrieve")

# Fixed edge: retry always goes back to generate
graph.add_edge("retry", "generate")

# Fixed edge: generate always goes to END
graph.add_edge("generate", END)

# Conditional edge: after retrieve, call route_after_retrieval()
# its return value ("retry" or "generate") picks the next node
graph.add_conditional_edges(
    "retrieve",                  # from this node
    route_after_retrieval,       # call this function
    {
        "retry": "retry",        # if function returns "retry" → go to retry node
        "generate": "generate",  # if function returns "generate" → go to generate node
    }
)

# Compile — turns the graph definition into a runnable object
pipeline = graph.compile()


# ── 5. RUN IT ─────────────────────────────────────────────────────────────
print("=" * 55)
print("TEST 1: High quality retrieval (should skip retry)")
print("=" * 55)
result1 = pipeline.invoke({
    "query": "crypto trading ads",
    "retrieved": None,
    "final_answer": None,
    "quality_score": 0.0,
})
print(f"\nFinal answer: {result1['final_answer']}")

print("\n" + "=" * 55)
print("TEST 2: Low quality retrieval (should trigger retry)")
print("=" * 55)
result2 = pipeline.invoke({
    "query": "laptops",
    "retrieved": None,
    "final_answer": None,
    "quality_score": 0.0,
})
print(f"\nFinal answer: {result2['final_answer']}")