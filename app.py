"""
Google Ads Policy RAG System v2 - Web Interface
LangGraph + Ollama + BGE-large + FAISS + BM25
Interactive demo for portfolio/interviews
"""

import streamlit as st
import sys
from pathlib import Path
import time
import json

sys.path.insert(0, str(Path(__file__).parent))
from src.graph.pipeline import RAGPipeline

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Google Ads Policy RAG v2",
    page_icon="🔍",
    layout="wide"
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS — warm beige / chocolate / pastel brown theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Force light mode + beige background ── */
    .stApp {
        background-color: #f5f0e8 !important;
    }
    .stApp * {
        color: #3d2b1f !important;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background-color: #ede4d3 !important;
    }
    [data-testid="stSidebar"] * {
        color: #3d2b1f !important;
    }

    /* ── Main content area ── */
    [data-testid="stMainBlockContainer"] {
        background-color: #f5f0e8 !important;
    }

    /* ── Tabs ── */
    [data-testid="stTabs"] {
        background-color: #f5f0e8 !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        background-color: #ede4d3 !important;
    }
    .stTabs [data-baseweb="tab"] {
        color: #3d2b1f !important;
    }

    /* ── Text area + inputs ── */
    .stTextArea textarea {
        background-color: #faf6f0 !important;
        color: #3d2b1f !important;
        border: 1px solid #c8a882 !important;
    }

    /* ── Buttons ── */
    .stButton button {
        background-color: #c8a882 !important;
        color: #3d2b1f !important;
        border: none !important;
    }
    .stButton button:hover {
        background-color: #a87850 !important;
        color: #f5f0e8 !important;
    }

    /* ── Primary button ── */
    .stButton button[kind="primary"] {
        background-color: #e8761a !important;
        color: #ffffff !important;
    }
    .stButton button[kind="primary"]:hover {
        background-color: #c85e10 !important;
    }

    /* ── Expanders ── */
    [data-testid="stExpander"] {
        background-color: #ede4d3 !important;
        border: 1px solid #c8a882 !important;
    }

    /* ── Metrics ── */
    [data-testid="metric-container"] {
        background-color: #ede4d3 !important;
        border-radius: 8px !important;
        padding: 0.5rem !important;
    }
    [data-testid="metric-container"] * {
        color: #3d2b1f !important;
    }

    /* ── Info/warning boxes ── */
    [data-testid="stAlert"] {
        background-color: #f0e6d4 !important;
        color: #3d2b1f !important;
    }

    /* ── Divider ── */
    hr {
        border-color: #c8a882 !important;
    }

    /* ── Headers ── */
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 0.5rem;
        color: #3d2b1f !important;
    }
    .sub-header {
        text-align: center;
        color: #7a5c44 !important;
        margin-bottom: 2rem;
    }

    /* ── Decision banners ── */
    .decision-allowed {
        background-color: #d6ead6 !important;
        border-left: 4px solid #5a8a5a;
        padding: 1rem;
        border-radius: 6px;
        color: #1a3a1a !important;
    }
    .decision-restricted {
        background-color: #f5e6c8 !important;
        border-left: 4px solid #c8883a;
        padding: 1rem;
        border-radius: 6px;
        color: #3a2a0a !important;
    }
    .decision-disallowed {
        background-color: #f0d4cc !important;
        border-left: 4px solid #a84a3a;
        padding: 1rem;
        border-radius: 6px;
        color: #3a0a0a !important;
    }
    .decision-unclear {
        background-color: #e8e0d4 !important;
        border-left: 4px solid #8a7a6a;
        padding: 1rem;
        border-radius: 6px;
        color: #3a3028 !important;
    }

    /* ── LangGraph trace box ── */
    .trace-box {
        background-color: #3d2b1f;
        color: #d4a96a !important;
        padding: 0.8rem 1rem;
        border-radius: 6px;
        font-family: monospace;
        font-size: 0.85rem;
        margin-top: 0.5rem;
    }

    /* ── Node badges ── */
    .node-badge {
        display: inline-block;
        background: #5c3d28;
        color: #f0d4a8 !important;
        border: 1px solid #c8883a;
        border-radius: 12px;
        padding: 2px 10px;
        font-size: 0.78rem;
        margin: 2px;
        font-family: monospace;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Cached pipeline — loads once, reused across all queries
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_pipeline():
    return RAGPipeline()

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">🔍 Google Ads Policy RAG</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">v2 · LangGraph Orchestration · Ollama llama3.2 · Local & Free</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ System Info")

    st.markdown("**v2 Stack**")
    st.markdown("""
    - 🧠 BGE-large-en-v1.5 (1024-dim)
    - 🗄️ FAISS + BM25 hybrid search
    - 🔄 RRF fusion + cross-encoder rerank
    - 🤖 Ollama llama3.2 (local, free)
    - 🕸️ LangGraph state machine
    - ✅ Pydantic v2 validation
    """)

    st.divider()
    st.markdown("**Pipeline Nodes**")
    nodes = [
        ("🔍", "query_analyzer", "classifies + expands"),
        ("📚", "retriever",      "BM25 + FAISS + RRF"),
        ("🎯", "reranker",       "cross-encoder scores"),
        ("🤖", "llm_generator",  "Ollama decision"),
        ("✅", "validator",      "override + confidence"),
        ("🚨", "escalation",     "human review queue"),
    ]
    for icon, name, desc in nodes:
        st.markdown(f"{icon} **{name}** — {desc}")

    st.divider()
    st.markdown("**Eval Metrics**")
    st.metric("Decision Accuracy", "90%")
    st.metric("Recall@5",          "77.8%")
    st.metric("Avg Confidence",    "81.4%")
    st.metric("Policy Chunks",     "316")

# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🧪 Ad Review", "📚 Example Cases", "📈 System Metrics"])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — AD REVIEW
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Ad Policy Review")

    if "ad_text" not in st.session_state:
        st.session_state.ad_text = ""

    def set_example(text):
        st.session_state.ad_text = text

    # ── Text area (full width) ─────────────────────────────────────────────
    ad_text = st.text_area(
        "Enter ad text to review:",
        placeholder="Example: Lose 15 pounds in one week with this miracle pill!",
        height=100,
        key="ad_text",
    )

    # ── Quick examples — horizontal row ───────────────────────────────────
    st.markdown("**Quick Examples:**")
    ex_cols = st.columns(6)
    examples_list = [
        ("🏥 Miracle Pill", "Lose 15 pounds in one week with this miracle pill! Guaranteed!"),
        ("💰 Crypto",       "Learn crypto trading from certified experts!"),
        ("📱 Product",      "Buy our new smartphone - 5G, free shipping over $50"),
        ("🍷 Alcohol",      "Premium craft whiskey delivered to your door. 21+ only."),
        ("📈 Forex Scam",   "Get rich quick with forex trading secrets!"),
        ("💊 Pharmacy",     "Online pharmacy - no prescription needed!"),
    ]
    for col, (label, text) in zip(ex_cols, examples_list):
        with col:
            st.button(label, on_click=set_example, args=(text,), use_container_width=True)

    # ── Review button ──────────────────────────────────────────────────────
    if st.button("🔍 Review Ad", type="primary", use_container_width=True):
        if not ad_text.strip():
            st.warning("⚠️ Please enter ad text to review")
        else:
            pipeline = load_pipeline()

            with st.spinner("Running LangGraph pipeline..."):
                import io, contextlib

                log_buffer = io.StringIO()
                start = time.time()

                with contextlib.redirect_stdout(log_buffer):
                    initial_state = pipeline._make_initial_state(ad_text)
                    final_state   = pipeline.graph.invoke(initial_state)

                elapsed = time.time() - start
                decision   = final_state["decision"]
                node_trace = final_state.get("node_trace", [])
                latency_ms = final_state.get("latency_ms", {})

            st.markdown("---")
            st.header("📋 Policy Decision")

            # ── Decision banner ────────────────────────────────────────────
            EMOJI = {
                "allowed":    "✅",
                "restricted": "⚠️",
                "disallowed": "❌",
                "unclear":    "❓",
            }
            emoji = EMOJI.get(decision.decision, "❓")
            decision_class = f"decision-{decision.decision}"

            st.markdown(
                f'<div class="{decision_class}"><h3>{emoji} {decision.decision.upper()}</h3></div>',
                unsafe_allow_html=True,
            )

            # ── Metrics row ────────────────────────────────────────────────
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Confidence",   f"{decision.confidence:.1%}")
            c2.metric("Latency",      f"{elapsed:.1f}s")
            c3.metric("Escalation",   "🚨 Yes" if decision.escalation_required else "No")
            c4.metric("Risk Factors", len(decision.risk_factors or []))

            # ── LangGraph trace ────────────────────────────────────────────
            st.markdown("### 🕸️ LangGraph Pipeline Trace")
            trace_html = " → ".join(
                f'<span class="node-badge">{n}</span>' for n in node_trace
            )
            st.markdown(f'<div class="trace-box">{trace_html}</div>', unsafe_allow_html=True)

            # ── Node timing breakdown ──────────────────────────────────────
            if latency_ms:
                with st.expander("⏱ Node Timing Breakdown"):
                    timing_data = {k: f"{v:.0f}ms" for k, v in latency_ms.items()}
                    st.json(timing_data)

            # ── Policy details ─────────────────────────────────────────────
            st.markdown("### 📂 Policy Section")
            st.info(decision.policy_section)

            st.markdown("### 💬 Justification")
            st.write(decision.justification)

            st.markdown("### 📝 Policy Citation")
            if decision.policy_quote:
                st.markdown(f"> {decision.policy_quote}")
            if decision.citation_url:
                st.markdown(f"[📖 View Official Policy]({decision.citation_url})")

            if decision.risk_factors:
                st.markdown("### ⚠️ Risk Factors")
                for rf in decision.risk_factors:
                    st.markdown(f"- `{rf}`")

            if decision.escalation_required:
                st.warning(
                    "🚨 This case requires **HUMAN REVIEW** — "
                    "confidence below threshold or decision is unclear."
                )

            # ── Query analysis ─────────────────────────────────────────────
            with st.expander("🔍 Query Analysis Details"):
                st.markdown(f"**Query type:** `{final_state.get('query_type', 'unknown')}`")
                expanded = final_state.get("expanded_query", "")
                if expanded and expanded != ad_text:
                    st.markdown("**Expanded query:**")
                    st.code(expanded)
                else:
                    st.markdown("No query expansion applied")

            # ── Top retrieved chunks ───────────────────────────────────────
            with st.expander("📚 Top Retrieved Policy Chunks"):
                chunks = final_state.get("reranked_chunks") or []
                if chunks:
                    for i, chunk in enumerate(chunks[:3], 1):
                        hierarchy = " > ".join(
                            chunk.get("metadata", {}).get("hierarchy", [])
                        )
                        score = chunk.get("rerank_score", 0)
                        st.markdown(f"**{i}. [{score:.4f}] {hierarchy}**")
                        st.caption(chunk.get("content", "")[:300] + "...")
                        st.divider()
                else:
                    st.info("No chunks available")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — EXAMPLES
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("📚 Example Test Cases")
    examples = [
        ("❌ Misleading Health Claims",  "Lose 15 pounds in one week with this miracle pill! Guaranteed!", "disallowed"),
        ("⚠️ Crypto Education",          "Learn cryptocurrency trading from certified experts",             "restricted"),
        ("✅ Standard Product",           "Buy our new smartphone - 5G, 128GB, free shipping over $50",    "allowed"),
        ("❌ Financial Guarantee",        "100% guaranteed 30% annual returns — invest with us today!",     "disallowed"),
        ("⚠️ Alcohol Ad",                "Premium craft whiskey delivered to your door. 21+ only.",        "restricted"),
        ("❌ No-Prescription Pharmacy",   "Online pharmacy — no prescription needed!",                      "disallowed"),
        ("❌ Forex Get Rich Quick",       "Get rich quick with forex trading secrets!",                     "disallowed"),
        ("⚠️ Political Ad",              "Vote for John Smith — best candidate for mayor!",                "restricted"),
    ]
    for name, ad, expected in examples:
        with st.expander(name):
            st.code(ad)
            color = {"allowed": "green", "restricted": "orange", "disallowed": "red"}[expected]
            st.markdown(f"Expected decision: :{color}[**{expected.upper()}**]")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — METRICS
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("📈 System Metrics")

    results_path = Path("evaluation/evaluation_results.json")
    if results_path.exists():
        with open(results_path) as f:
            eval_results = json.load(f)

        ret = eval_results.get("retrieval", {})
        dec = eval_results.get("decisions", {})
        rag = eval_results.get("ragas",     {})

        st.subheader("🔍 Retrieval")
        c1, c2, c3 = st.columns(3)
        c1.metric("Recall@5",    f"{ret.get('recall_at_5', 0):.1%}")
        c2.metric("MRR",         f"{ret.get('mrr', 0):.3f}")
        c3.metric("Precision@5", f"{ret.get('precision_at_5', 0):.1%}")

        st.subheader("🎯 Decisions")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy",        f"{dec.get('decision_accuracy', 0):.1%}")
        c2.metric("Avg Confidence",  f"{dec.get('avg_confidence', 0):.1%}")
        c3.metric("Escalation Rate", f"{dec.get('escalation_rate', 0):.1%}")
        c4.metric("Latency P50",     f"{dec.get('latency_p50_ms', 0)/1000:.1f}s")

        if rag and "faithfulness" in rag:
            st.subheader("🧪 LLM-Judge (RAGAS-equivalent)")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Faithfulness",      f"{rag.get('faithfulness', 0):.3f}")
            c2.metric("Answer Relevancy",  f"{rag.get('answer_relevancy', 0):.3f}")
            c3.metric("Context Recall",    f"{rag.get('context_recall', 0):.3f}")
            c4.metric("Context Precision", f"{rag.get('context_precision', 0):.3f}")

        st.subheader("📊 v1 vs v2 Comparison")
        comparison = {
            "Metric":                ["Decision Accuracy", "Avg Confidence", "Escalation Rate", "Recall@5"],
            "v1 (Gemini)":           ["80%",               "29%",            "100%",             "77.8%"],
            "v2 (Ollama+LangGraph)": [
                f"{dec.get('decision_accuracy', 0):.0%}",
                f"{dec.get('avg_confidence', 0):.0%}",
                f"{dec.get('escalation_rate', 0):.0%}",
                f"{ret.get('recall_at_5', 0):.1%}",
            ],
        }
        import pandas as pd
        st.dataframe(pd.DataFrame(comparison), hide_index=True, use_container_width=True)

    else:
        st.info("Run `python -m src.evaluation.evaluator` to generate metrics.")

    st.subheader("🏗️ Architecture")
    st.markdown("""
    **What changed from v1 → v2:**
    - ❌ Gemini API → ✅ Ollama llama3.2 (local, unlimited, free)
    - ❌ Sequential functions → ✅ LangGraph state machine
    - ❌ 100% escalation rate → ✅ 10% (calibrated confidence)
    - ❌ 29% avg confidence → ✅ 81% (recalibrated scoring)
    - ➕ Conditional retry logic (low quality retrieval → retry)
    - ➕ Full eval suite with 8 metrics
    - ➕ Auto-generated eval dataset via Ollama

    **What stayed the same (already production-grade):**
    - BGE-large-en-v1.5 embeddings (1024-dim)
    - FAISS vector store
    - BM25 keyword search
    - RRF fusion
    - Cross-encoder reranking
    - Pydantic v2 structured output
    """)

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#7a5c44'>"
    "Google Ads Policy RAG v2 · LangGraph + Ollama + BGE-large + FAISS + BM25 · "
    "316 clean policy chunks · Built by Varun Vaddi"
    "</p>",
    unsafe_allow_html=True,
)