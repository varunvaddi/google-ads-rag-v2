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
        "forex secrets", "trading secrets", "rich quick",
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
    elif "crypto" in query_lower or "bitcoin" in query_lower or "ethereum" in query_lower:
        expanded_query += " cryptocurrency digital currency financial services"
    elif any(w in query_lower for w in ["whiskey", "beer", "wine", "alcohol", "spirits", "liquor"]):
        expanded_query += " alcohol restricted content age-gating requirements"
    elif any(w in query_lower for w in ["vote", "election", "political", "candidate"]):
        expanded_query += " political content election ads restricted verification"
    elif any(w in query_lower for w in ["gambling", "casino", "betting", "poker"]):
        expanded_query += " gambling games restricted content wagering"
    elif any(w in query_lower for w in ["pharmacy", "prescription", "medication"]):
        expanded_query += " pharmaceutical drugs healthcare restricted content"
    elif any(w in query_lower for w in ["forex", "rich quick", "trading secrets"]):
        expanded_query += " unreliable claims misrepresentation improbable results prohibited"

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

# ─────────────────────────────────────────────────────────────────────────────
# Singleton loader — loads Ollama LLM once
# ─────────────────────────────────────────────────────────────────────────────

_llm = None

def _get_llm():
    """
    Lazy singleton for ChatOllama.
    Connects to Ollama server running locally on port 11434.
    No API key, no rate limits, free forever.
    """
    global _llm
    if _llm is None:
        from langchain_ollama import ChatOllama
        from src.generation.decision_schema import PolicyDecision
        print("[LLMGenerator] Connecting to Ollama llama3.2...")
        llm = ChatOllama(
            model="llama3.2",
            temperature=0.1,    # Low = consistent decisions
            num_predict=1024,   # Max tokens in response
        )
        _llm = llm.with_structured_output(PolicyDecision)
        print("[LLMGenerator] Connected!")
    return _llm


# ─────────────────────────────────────────────────────────────────────────────
# Node 4: LLM Generator
# ─────────────────────────────────────────────────────────────────────────────

def llm_generator_node(state: RAGState) -> Dict[str, Any]:
    """
    Calls Ollama llama3.2 with retrieved policy context.
    Returns a structured PolicyDecision object directly.

    Replaces GeminiPolicyEngine from v1.
    Same prompts from src/generation/prompts.py — nothing changed there.

    INPUT  (reads from state): reranked_chunks, query
    OUTPUT (writes to state):  decision, raw_llm_response, latency_ms, node_trace
    """
    t0 = time.time()

    from langchain_core.messages import SystemMessage, HumanMessage
    from src.generation.prompts import format_policy_review_prompt

    structured_llm = _get_llm()

    # Use reranked chunks if available, fall back to retrieved
    chunks = state.get("reranked_chunks") or state.get("retrieved_chunks", [])
    query = state["query"]

    print(f"\n[LLMGenerator] Calling Ollama llama3.2...")
    print(f"[LLMGenerator] query: '{query[:60]}'")
    print(f"[LLMGenerator] context: {len(chunks)} chunks")

    # Format prompt using your existing prompts.py — unchanged from v1
    prompts = format_policy_review_prompt(query, chunks)

    messages = [
        SystemMessage(content=prompts["system"]),
        HumanMessage(content=prompts["user"]),
    ]

    try:
        # with_structured_output handles JSON parsing + Pydantic validation
        # Response is already a PolicyDecision object — no manual parsing needed
        decision = structured_llm.invoke(messages)
        raw_response = "structured output — no raw text"

        print(f"[LLMGenerator] decision={decision.decision} | confidence={decision.confidence:.1%}")

    except Exception as e:
        print(f"[LLMGenerator] Error: {e}")
        from src.generation.decision_schema import PolicyDecision
        decision = PolicyDecision(
            decision="unclear",
            confidence=0.0,
            policy_section="Generation Error",
            citation_url="",
            justification=f"LLM error: {str(e)}",
            policy_quote="",
            escalation_required=True,
        )
        raw_response = str(e)

    # ── Observability ─────────────────────────────────────────────────────
    latency = state.get("latency_ms") or {}
    latency["llm_generator"] = round((time.time() - t0) * 1000, 1)

    trace = state.get("node_trace") or []
    trace.append("llm_generator")

    return {
        "decision": decision,
        "raw_llm_response": raw_response,
        "latency_ms": latency,
        "node_trace": trace,
    }

# ─────────────────────────────────────────────────────────────────────────────
# Node 5: Validator
# ─────────────────────────────────────────────────────────────────────────────

def validator_node(state: RAGState) -> Dict[str, Any]:
    """
    Post-processes the LLM decision with three jobs:
      1. Policy path override  — chunk hierarchy overrides LLM decision
      2. Confidence recalc     — combines retrieval scores + LLM confidence
      3. Escalation flag       — sets escalate=True if confidence < threshold

    This is a safety net. Even if the LLM makes a wrong call,
    the policy path override catches it deterministically.

    INPUT  (reads from state): decision, reranked_chunks, retrieval_scores
    OUTPUT (writes to state):  decision (updated), escalate, latency_ms, node_trace
    """
    t0 = time.time()

    from src.generation.decision_schema import PolicyDecision

    decision = state["decision"]
    chunks = state.get("reranked_chunks") or state.get("retrieved_chunks", [])
    scores = state.get("retrieval_scores", [])

    print(f"\n[Validator] incoming decision={decision.decision} | confidence={decision.confidence:.1%}")

    # ── 1. Policy path override ───────────────────────────────────────────
    # Find the first NON-JUNK chunk to use for override decision
    # Junk chunks = "Was this helpful?" UI remnants from scraping
    JUNK_PHRASES = ["was this helpful", "was this article helpful"]

    override_chunk = None
    for chunk in chunks:
        content = chunk.get("content", "").lower()
        is_junk = any(phrase in content for phrase in JUNK_PHRASES)
        if not is_junk:
            override_chunk = chunk
            break

    # Only override if retrieval quality is strong enough to trust
    # Low scores = no relevant policy found = likely allowed
    OVERRIDE_SCORE_THRESHOLD = 0.002

    top_score = scores[0] if scores else 0.0

    if override_chunk and top_score >= OVERRIDE_SCORE_THRESHOLD:
        hierarchy = override_chunk.get("metadata", {}).get("hierarchy", [])
        policy_path = " ".join(hierarchy).lower()
        print(f"[Validator] override chunk (score={top_score:.4f}): {' > '.join(hierarchy)}")

        if "prohibited content" in policy_path or "prohibited practices" in policy_path:
            if decision.decision != "disallowed":
                print(f"[Validator] OVERRIDE: {decision.decision} → disallowed (Prohibited Content)")
                decision.decision = "disallowed"

        elif "restricted content" in policy_path:
            if decision.decision not in ("restricted", "disallowed"):
                print(f"[Validator] OVERRIDE: {decision.decision} → restricted (Restricted Content)")
                decision.decision = "restricted"
        else:
            print(f"[Validator] No override needed")

    elif top_score < OVERRIDE_SCORE_THRESHOLD:
        print(f"[Validator] Score {top_score:.4f} too low — skipping override, trusting LLM")
    else:
        print(f"[Validator] All chunks are junk — skipping override")

    # ── 2. Confidence recalculation ───────────────────────────────────────
    # Same formula as v1 calculate_confidence() — now in validator node
    # Combines 4 factors:
    #   retrieval_factor : how good were the retrieved chunks?
    #   clarity_factor   : how clear is the decision?
    #   multi_source     : how many high-quality chunks agree?
    #   llm_confidence   : what did the LLM say its confidence was?

    if scores and len(scores) >= 2:
        margin = scores[0] - scores[1]
        if margin > 0.002:
            retrieval_factor = 0.8
        elif margin > 0.001:
            retrieval_factor = 0.65
        elif margin > 0.0005:
            retrieval_factor = 0.5
        else:
            retrieval_factor = 0.35

        if scores[0] > 0.003:
            retrieval_factor = min(1.0, retrieval_factor + 0.1)
    elif scores:
        retrieval_factor = 0.5
    else:
        retrieval_factor = 0.3

    clarity = 0.8 if decision.decision != "unclear" else 0.3
    high_quality_count = sum(1 for s in scores[:3] if s > 0.002)
    multi_source = high_quality_count / 3

    llm_conf = getattr(decision, "confidence", 0.5)
    # Clamp LLM confidence to 0-1 in case model returned 90 instead of 0.9
    if llm_conf > 1.0:
        llm_conf = llm_conf / 100.0
    llm_conf = max(0.0, min(1.0, llm_conf))

    final_confidence = (
        retrieval_factor * 0.25
        + clarity       * 0.25
        + multi_source  * 0.15
        + llm_conf      * 0.35
    )
    final_confidence = round(max(0.0, min(1.0, final_confidence)), 4)

    print(f"[Validator] confidence: {decision.confidence:.1%} → {final_confidence:.1%}")
    decision.confidence = final_confidence

    # ── 3. Escalation routing ─────────────────────────────────────────────
    # If confidence is below threshold → flag for human review
    ESCALATION_THRESHOLD = 0.70
    escalate = final_confidence < ESCALATION_THRESHOLD or decision.decision == "unclear"
    decision.escalation_required = escalate

    print(f"[Validator] final: decision={decision.decision} | confidence={final_confidence:.1%} | escalate={escalate}")

    # ── Observability ─────────────────────────────────────────────────────
    latency = state.get("latency_ms") or {}
    latency["validator"] = round((time.time() - t0) * 1000, 1)

    trace = state.get("node_trace") or []
    trace.append("validator")

    return {
        "decision": decision,
        "escalate": escalate,
        "latency_ms": latency,
        "node_trace": trace,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node 6: Escalation Handler
# ─────────────────────────────────────────────────────────────────────────────

def escalation_node(state: RAGState) -> Dict[str, Any]:
    """
    Handles low-confidence decisions.
    In production: writes to a review queue, sends a Slack alert, etc.
    For now: logs the escalation and updates the justification.

    INPUT  (reads from state): decision
    OUTPUT (writes to state):  decision (updated), node_trace
    """
    decision = state["decision"]

    print(f"\n[Escalation] 🚨 Routing for human review")
    print(f"[Escalation] decision={decision.decision} | confidence={decision.confidence:.1%}")

    # Tag the justification so reviewers know why it was escalated
    decision.justification = (
        f"[ESCALATED — confidence {decision.confidence:.1%} below threshold] "
        + decision.justification
    )
    decision.escalation_required = True

    trace = state.get("node_trace") or []
    trace.append("escalation")

    return {
        "decision": decision,
        "node_trace": trace,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Conditional Edge: after validator, escalate or finish?
# ─────────────────────────────────────────────────────────────────────────────

def should_escalate(state: RAGState) -> str:
    """
    Called by LangGraph after validator_node.
    Returns "escalation" or "end" — LangGraph routes accordingly.
    """
    if state.get("escalate", False):
        return "escalation"
    return "end"