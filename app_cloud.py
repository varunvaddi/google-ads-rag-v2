"""
app_cloud.py — Streamlit Cloud deployment
Same v2 UI as app2.py but uses Gemini instead of Ollama.
Ollama can't run on Streamlit Cloud — Gemini works from anywhere.

Local dev:   python app2.py        (Ollama — free, unlimited)
Cloud demo:  streamlit run app_cloud.py  (Gemini — works on Streamlit Cloud)
"""

import streamlit as st
import sys
import os
from pathlib import Path
import time
import json

sys.path.insert(0, str(Path(__file__).parent))

# ── Download embeddings from HuggingFace Hub if not present ──────────────
from huggingface_hub import hf_hub_download

def ensure_embeddings():
    embeddings_dir = Path("data/embeddings")
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    files = ["bm25.pkl", "embeddings.npy", "faiss.index", "metadata.json"]
    for filename in files:
        if not (embeddings_dir / filename).exists():
            with st.spinner(f"Downloading {filename}..."):
                hf_hub_download(
                    repo_id="varunvaddi/google-ads-rag-embeddings",
                    filename=filename,
                    repo_type="dataset",
                    local_dir=str(embeddings_dir),
                )

ensure_embeddings()

# ── Gemini decision engine (from v1, works on cloud) ─────────────────────
from src.generation.decision_engine import GeminiPolicyEngine

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
    .stApp { background-color: #f5f0e8 !important; }
    .stApp * { color: #3d2b1f !important; }
    [data-testid="stSidebar"] { background-color: #ede4d3 !important; }
    [data-testid="stSidebar"] * { color: #3d2b1f !important; }
    [data-testid="stMainBlockContainer"] { background-color: #f5f0e8 !important; }
    .stTabs [data-baseweb="tab-list"] { background-color: #ede4d3 !important; }
    .stTabs [data-baseweb="tab"] { color: #3d2b1f !important; }
    .stTextArea textarea {
        background-color: #faf6f0 !important;
        color: #3d2b1f !important;
        border: 1px solid #c8a882 !important;
    }
    .stButton button {
        background-color: #c8a882 !important;
        color: #3d2b1f !important;
        border: none !important;
    }
    .stButton button:hover {
        background-color: #a87850 !important;
        color: #f5f0e8 !important;
    }
    .stButton button[kind="primary"] {
        background-color: #e8761a !important;
        color: #ffffff !important;
    }
    .stButton button[kind="primary"]:hover {
        background-color: #c85e10 !important;
    }
    [data-testid="stExpander"] {
        background-color: #ede4d3 !important;
        border: 1px solid #c8a882 !important;
    }
    [data-testid="metric-container"] {
        background-color: #ede4d3 !important;
        border-radius: 8px !important;
        padding: 0.5rem !important;
    }
    [data-testid="metric-container"] * { color: #3d2b1f !important; }
    hr { border-color: #c8a882 !important; }

    .main-header {
        font-size: 3rem; font-weight: bold;
        text-align: center; margin-bottom: 0.5rem; color: #3d2b1f !important;
    }
    .sub-header {
        text-align: center; color: #7a5c44 !important; margin-bottom: 2rem;
    }
    .decision-allowed {
        background-color: #d6ead6 !important; border-left: 4px solid #5a8a5a;
        padding: 1rem; border-radius: 6px; color: #1a3a1a !important;
    }
    .decision-restricted {
        background-color: #f5e6c8 !important; border-left: 4px solid #c8883a;
        padding: 1rem; border-radius: 6px; color: #3a2a0a !important;
    }
    .decision-disallowed {
        background-color: #f0d4cc !important; border-left: 4px solid #a84a3a;
        padding: 1rem; border-radius: 6px; color: #3a0a0a !important;
    }
    .decision-unclear {
        background-color: #e8e0d4 !important; border-left: 4px solid #8a7a6a;
        padding: 1rem; border-radius: 6px; color: #3a3028 !important;
    }
    .trace-box {
        background-color: #3d2b1f; color: #d4a96a !important;
        padding: 0.8rem 1rem; border-radius: 6px;
        font-family: monospace; font-size: 0.85rem; margin-top: 0.5rem;
    }
    .node-badge {
        display: inline-block; background: #5c3d28;
        color: #f0d4a8 !important; border: 1px solid #c8883a;
        border-radius: 12px; padding: 2px 10px;
        font-size: 0.78rem; margin: 2px; font-family: monospace;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Cached engine
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_engine():
    return GeminiPolicyEngine()

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">🔍 Google Ads Policy RAG</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">v2 · Hybrid Retrieval · BGE-large + FAISS + BM25 · Gemini 2.5 Flash</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ System Info")
    st.markdown("**Stack**")
    st.markdown("""
    - 🧠 BGE-large-en-v1.5 (1024-dim)
    - 🗄️ FAISS + BM25 hybrid search
    - 🔄 RRF fusion + cross-encoder rerank
    - 🤖 Gemini 2.5 Flash (cloud)
    - ✅ Pydantic v2 validation
    """)
    st.divider()
    st.markdown("**Eval Metrics**")
    st.metric("Decision Accuracy", "90%")
    st.metric("Recall@5",          "77.8%")
    st.metric("Avg Confidence",    "81.4%")
    st.metric("Policy Chunks",     "316")
    st.divider()
    st.caption("Local version uses Ollama llama3.2 + LangGraph orchestration. See GitHub for full v2.")

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

    # Text area
    ad_text = st.text_area(
        "Enter ad text to review:",
        placeholder="Example: Lose 15 pounds in one week with this miracle pill!",
        height=100,
        key="ad_text",
    )

    # Quick examples — horizontal row
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

    # Review button
    if st.button("🔍 Review Ad", type="primary", use_container_width=True):
        if not ad_text.strip():
            st.warning("⚠️ Please enter ad text to review")
        else:
            engine = load_engine()

            with st.spinner("Reviewing ad against Google Ads policies..."):
                start = time.time()
                decision = engine.review_ad(ad_text)
                elapsed = time.time() - start

            st.markdown("---")
            st.header("📋 Policy Decision")

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

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Confidence",   f"{decision.confidence:.1%}")
            c2.metric("Latency",      f"{elapsed:.1f}s")
            c3.metric("Escalation",   "🚨 Yes" if decision.escalation_required else "No")
            c4.metric("Risk Factors", len(decision.risk_factors or []))

            # Pipeline trace (v2 style display even with Gemini backend)
            st.markdown("### 🕸️ Retrieval Pipeline")
            trace_nodes = ["query_analyzer", "hybrid_retriever", "cross_encoder_reranker", "gemini_generator", "validator"]
            trace_html = " → ".join(f'<span class="node-badge">{n}</span>' for n in trace_nodes)
            st.markdown(f'<div class="trace-box">{trace_html}</div>', unsafe_allow_html=True)

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
                st.warning("🚨 This case requires **HUMAN REVIEW** — confidence below threshold or decision is unclear.")

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
            st.subheader("🧪 LLM-Judge Metrics")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Faithfulness",      f"{rag.get('faithfulness', 0):.3f}")
            c2.metric("Answer Relevancy",  f"{rag.get('answer_relevancy', 0):.3f}")
            c3.metric("Context Recall",    f"{rag.get('context_recall', 0):.3f}")
            c4.metric("Context Precision", f"{rag.get('context_precision', 0):.3f}")

        st.subheader("📊 Architecture")
        st.markdown("""
        **Retrieval Pipeline (same in cloud + local):**
        - BGE-large-en-v1.5 embeddings (1024-dim)
        - FAISS vector store + BM25 keyword search
        - RRF fusion + BGE-reranker-large cross-encoder
        - 316 clean policy chunks (25 junk chunks removed)

        **LLM (differs by environment):**
        - ☁️ Cloud: Gemini 2.5 Flash (this demo)
        - 💻 Local: Ollama llama3.2 + LangGraph state machine

        **Full v2 (local) adds:**
        - LangGraph 6-node state machine
        - Conditional retry routing
        - Confidence-gated escalation
        - 8-metric eval suite
        """)
    else:
        st.info("Evaluation results not available in this deployment.")
        st.markdown("""
        **System specs:**
        - Recall@5: 77.8% · MRR: 0.778
        - Decision Accuracy: 90%
        - Avg Confidence: 81.4%
        - Escalation Rate: 10%
        """)

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#7a5c44'>"
    "Google Ads Policy RAG v2 · BGE-large + FAISS + BM25 + Gemini · "
    "316 clean policy chunks · Built by Varun Vaddi"
    "</p>",
    unsafe_allow_html=True,
)