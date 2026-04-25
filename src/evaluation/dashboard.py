"""
src/evaluation/dashboard.py
Streamlit evaluation dashboard for Google Ads Policy RAG v2.

Run with:
    streamlit run src/evaluation/dashboard.py

Shows all 8 metrics visually:
  - Gauge charts for key scores
  - Per-case decision table
  - Latency bar chart
  - LLM-Judge radar chart
  - Overall score card
"""

import json
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

RESULTS_PATH = Path("evaluation/evaluation_results.json")

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Eval Dashboard",
    page_icon="📊",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# Load results
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_results():
    if not RESULTS_PATH.exists():
        return None
    with open(RESULTS_PATH) as f:
        return json.load(f)

results = load_results()

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.title("📊 Google Ads Policy RAG — Evaluation Dashboard")
st.caption("v2 · Ollama llama3.2 · LangGraph · BGE-large + FAISS + BM25 + Cross-Encoder")

if results is None:
    st.warning(
        "No evaluation results found.\n\n"
        "Run first:\n```bash\npython -m src.evaluation.evaluator\n```"
    )
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Helper: gauge chart
# ─────────────────────────────────────────────────────────────────────────────
def gauge(value: float, title: str, as_percent: bool = True):
    display_val = value * 100 if as_percent else value * 100
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=display_val,
        title={"text": title, "font": {"size": 13}},
        number={"suffix": "%", "font": {"size": 22}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#4f8bf9"},
            "steps": [
                {"range": [0,  50], "color": "rgba(255, 75, 75, 0.15)"},
                {"range": [50, 75], "color": "rgba(255, 165, 0, 0.15)"},
                {"range": [75, 100], "color": "rgba(0, 200, 83, 0.15)"},
            ],
            "threshold": {
                "line": {"color": "green", "width": 3},
                "thickness": 0.75,
                "value": 80,
            },
        },
    ))
    fig.update_layout(height=200, margin=dict(t=50, b=0, l=20, r=20))
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# Metadata strip
# ─────────────────────────────────────────────────────────────────────────────
meta = results.get("metadata", {})
c1, c2, c3, c4 = st.columns(4)
c1.metric("Model", meta.get("model", "llama3.2"))
c2.metric("Total Chunks", meta.get("total_chunks", 316))
c3.metric("Version", meta.get("version", "v2"))
c4.metric("Eval Time", f"{meta.get('eval_time_s', 0):.0f}s")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Section 1: Retrieval metrics
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("🔍 Retrieval Quality")
ret = results.get("retrieval", {})

c1, c2, c3 = st.columns(3)
c1.plotly_chart(gauge(ret.get("recall_at_5", 0),    "Recall@5"),    use_container_width=True)
c2.plotly_chart(gauge(ret.get("mrr", 0),            "MRR"),         use_container_width=True)
c3.plotly_chart(gauge(ret.get("precision_at_5", 0), "Precision@5"), use_container_width=True)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Section 2: Decision accuracy
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("🎯 Decision Accuracy & Confidence")
dec = results.get("decisions", {})

c1, c2, c3, c4 = st.columns(4)
c1.metric("Decision Accuracy", f"{dec.get('decision_accuracy', 0):.1%}")
c2.metric("Policy Match",      f"{dec.get('policy_accuracy', 0):.1%}")
c3.metric("Avg Confidence",    f"{dec.get('avg_confidence', 0):.1%}")
c4.metric("Escalation Rate",   f"{dec.get('escalation_rate', 0):.1%}")

# Per-case table
per_case = dec.get("per_case", [])
if per_case:
    df = pd.DataFrame(per_case)
    df["correct"]   = df["correct"].map({True: "✅", False: "❌"})
    df["escalation"] = df["escalation"].map({True: "🚨", False: "—"})
    df["confidence"] = df["confidence"].map(lambda x: f"{x:.1%}")
    df["latency_ms"] = df["latency_ms"].map(lambda x: f"{x/1000:.1f}s")
    st.dataframe(
        df[["id", "query", "expected", "actual", "correct",
            "confidence", "escalation", "latency_ms"]],
        use_container_width=True,
        hide_index=True,
    )

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Section 3: Latency
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("⏱ Latency")

# Show P50 as the reliable number — P95/P99 may be skewed by stalls
p50 = dec.get("latency_p50_ms", 0) / 1000
st.info(f"📌 P50 (median) latency: **{p50:.1f}s** — this is the real-world typical query time on MacBook Air CPU. P95/P99 may be skewed by model loading on first call.")

if per_case:
    raw_latencies = [c["latency_ms"] for c in results["decisions"]["per_case"]]
    # Cap display at 3x median to avoid stall outliers skewing chart
    median_lat = float(np.median(raw_latencies))
    display_latencies = [min(l, median_lat * 3) for l in raw_latencies]

    fig = px.bar(
        x=[f"Case {c['id']}" for c in results["decisions"]["per_case"]],
        y=[l / 1000 for l in display_latencies],
        labels={"x": "Test Case", "y": "Latency (seconds)"},
        color=display_latencies,
        color_continuous_scale="Blues",
        title="Per-Query Latency (seconds)",
        text_auto=".1f",
    )
    fig.add_hline(
        y=p50,
        line_dash="dash",
        line_color="green",
        annotation_text=f"P50 = {p50:.1f}s",
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Section 4: LLM-Judge metrics
# ─────────────────────────────────────────────────────────────────────────────
ragas = results.get("ragas", {})
if ragas and "faithfulness" in ragas:
    st.subheader("🧪 LLM-Judge Metrics (RAGAS-equivalent)")
    st.caption(f"Evaluated on {ragas.get('n_samples', '?')} auto-generated samples using llama3.2 as judge")

    metrics = {
        "Faithfulness":      ragas.get("faithfulness",      0),
        "Answer Relevancy":  ragas.get("answer_relevancy",  0),
        "Context Recall":    ragas.get("context_recall",    0),
        "Context Precision": ragas.get("context_precision", 0),
    }

    # Radar chart
    fig = go.Figure(go.Scatterpolar(
        r=list(metrics.values()),
        theta=list(metrics.keys()),
        fill="toself",
        line_color="#4f8bf9",
        fillcolor="rgba(79,139,249,0.2)",
        name="v2 scores",
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        height=400,
        title="LLM-Judge Radar",
    )
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Faithfulness",      f"{metrics['Faithfulness']:.4f}")
    c2.metric("Answer Relevancy",  f"{metrics['Answer Relevancy']:.4f}")
    c3.metric("Context Recall",    f"{metrics['Context Recall']:.4f}")
    c4.metric("Context Precision", f"{metrics['Context Precision']:.4f}")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Section 5: Overall score card
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("🏆 Overall Score Card")

all_scores = {
    "Recall@5":          ret.get("recall_at_5",       0),
    "MRR":               ret.get("mrr",               0),
    "Decision Accuracy": dec.get("decision_accuracy", 0),
    "Avg Confidence":    dec.get("avg_confidence",    0),
}
if ragas and "faithfulness" in ragas:
    all_scores["Faithfulness"]      = ragas["faithfulness"]
    all_scores["Answer Relevancy"]  = ragas["answer_relevancy"]
    all_scores["Context Recall"]    = ragas["context_recall"]
    all_scores["Context Precision"] = ragas["context_precision"]

fig = px.bar(
    x=list(all_scores.keys()),
    y=[v * 100 for v in all_scores.values()],
    labels={"x": "Metric", "y": "Score (%)"},
    color=list(all_scores.values()),
    color_continuous_scale="Blues",
    title="All Metrics — v2 Pipeline",
    text_auto=".1f",
)
fig.update_layout(yaxis_range=[0, 100], showlegend=False)
st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Google Ads Policy RAG v2 · "
    "Ollama llama3.2 · LangGraph · BGE-large + FAISS + BM25 + RRF + Cross-Encoder · "
    "316 clean policy chunks"
)