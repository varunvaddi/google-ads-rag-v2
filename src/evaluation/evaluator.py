"""
src/evaluation/evaluator.py

Full evaluation suite for the v2 RAG pipeline.

Metrics:
  Retrieval  → Recall@5, MRR, Precision@5
  Decisions  → Accuracy, P50/P95/P99 latency
  RAGAS      → Faithfulness, Answer Relevancy, Context Recall, Context Precision

Usage:
  python src/evaluation/evaluator.py
  python run_evaluation.py --skip-ragas   (faster, no RAGAS)
  python run_evaluation.py --ragas-samples 20
"""

import json
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

from src.graph.pipeline import RAGPipeline
from src.retrieval.hybrid_search import HybridSearch


# ── Ground truth test cases for retrieval + decision evaluation ────────────
# Same 10 cases from your v1 evaluator — consistent across versions
DECISION_TEST_CASES = [
    {
        "id": 1,
        "query": "Lose 15 pounds in one week with this miracle pill",
        "expected_decision": "disallowed",
        "expected_policy": "Healthcare and medicines",
    },
    {
        "id": 2,
        "query": "100% guaranteed returns on Bitcoin investment",
        "expected_decision": "disallowed",
        "expected_policy": "Financial services",
    },
    {
        "id": 3,
        "query": "Learn cryptocurrency trading from certified experts",
        "expected_decision": "restricted",
        "expected_policy": "Cryptocurrencies",
    },
    {
        "id": 4,
        "query": "Buy our new laptop - Intel i7, 16GB RAM, free shipping",
        "expected_decision": "allowed",
        "expected_policy": None,
    },
    {
        "id": 5,
        "query": "Cure diabetes naturally with herbal tea",
        "expected_decision": "disallowed",
        "expected_policy": "Healthcare and medicines",
    },
    {
        "id": 6,
        "query": "Vote for John Smith - best candidate for mayor",
        "expected_decision": "restricted",
        "expected_policy": "Political content",
    },
    {
        "id": 7,
        "query": "Premium whiskey delivery - 21+ only",
        "expected_decision": "restricted",
        "expected_policy": "Alcohol",
    },
    {
        "id": 8,
        "query": "Counterfeit designer handbags wholesale",
        "expected_decision": "disallowed",
        "expected_policy": "Counterfeit goods",
    },
    {
        "id": 9,
        "query": "Online pharmacy - no prescription needed",
        "expected_decision": "disallowed",
        "expected_policy": "Healthcare and medicines",
    },
    {
        "id": 10,
        "query": "Get rich quick with forex trading secrets",
        "expected_decision": "disallowed",
        "expected_policy": "Financial services",
    },
]


class FullEvaluator:
    """
    Runs all evaluation sections and saves results to disk.
    """

    def __init__(
        self,
        eval_dataset_path: str = "evaluation/eval_dataset.json",
        results_path: str = "evaluation/evaluation_results.json",
    ):
        self.eval_dataset_path = Path(eval_dataset_path)
        self.results_path = Path(results_path)
        self.results_path.parent.mkdir(parents=True, exist_ok=True)

        print("=" * 65)
        print("FULL EVALUATION SUITE  v2")
        print("=" * 65)

        print("\nLoading pipeline...")
        self.pipeline = RAGPipeline()

        print("Loading retrieval system...")
        self.search = HybridSearch()

        print("✅ Ready!\n")

    # ─────────────────────────────────────────────────────────────────────
    # Section 1: Retrieval Metrics
    # ─────────────────────────────────────────────────────────────────────

    def evaluate_retrieval(self) -> Dict[str, float]:
        """
        Runs retrieval on all test cases and measures:
          Recall@5    — was correct policy in top 5?
          MRR         — how high was correct policy ranked?
          Precision@5 — how many of top 5 were relevant?
        """
        print("\n" + "=" * 65)
        print("SECTION 1: RETRIEVAL METRICS")
        print("=" * 65)

        recall_scores = []
        mrr_scores = []
        precision_scores = []

        # Only test cases that have an expected policy
        cases = [c for c in DECISION_TEST_CASES if c["expected_policy"]]

        for case in cases:
            retrieved = self.search.search(case["query"], top_k=5)
            expected = case["expected_policy"].lower()

            first_correct_rank = None
            relevant_count = 0

            for i, chunk in enumerate(retrieved, 1):
                hierarchy = " > ".join(
                    chunk.get("metadata", {}).get("hierarchy", [])
                )
                if expected in hierarchy.lower():
                    relevant_count += 1
                    if first_correct_rank is None:
                        first_correct_rank = i

            recall   = 1 if first_correct_rank is not None else 0
            mrr      = 1 / first_correct_rank if first_correct_rank else 0
            precision = relevant_count / len(retrieved) if retrieved else 0

            recall_scores.append(recall)
            mrr_scores.append(mrr)
            precision_scores.append(precision)

            status = "✅" if recall else "❌"
            rank_str = f"rank={first_correct_rank}" if first_correct_rank else "not found"
            print(f"  {status} [{case['id']:02d}] {case['query'][:50]}")
            print(f"        {rank_str} | precision={precision:.2f}")

        summary = {
            "recall_at_5":    round(float(np.mean(recall_scores)),    4),
            "mrr":            round(float(np.mean(mrr_scores)),        4),
            "precision_at_5": round(float(np.mean(precision_scores)), 4),
            "n_cases":        len(cases),
        }

        print(f"\n📊 Retrieval Results:")
        print(f"   Recall@5:     {summary['recall_at_5']:.1%}")
        print(f"   MRR:          {summary['mrr']:.3f}")
        print(f"   Precision@5:  {summary['precision_at_5']:.1%}")

        return summary

    # ─────────────────────────────────────────────────────────────────────
    # Section 2: Decision Accuracy + Latency
    # ─────────────────────────────────────────────────────────────────────

    def evaluate_decisions(self) -> Dict[str, Any]:
        """
        Runs all 10 test cases through the full LangGraph pipeline.
        Measures accuracy and P50/P95/P99 latency.
        """
        print("\n" + "=" * 65)
        print("SECTION 2: DECISION ACCURACY + LATENCY")
        print("=" * 65)

        correct_decisions = 0
        correct_policies  = 0
        confidences       = []
        latencies         = []
        escalations       = []
        per_case          = []

        for case in DECISION_TEST_CASES:
            print(f"\n  [{case['id']:02d}] {case['query'][:60]}")

            t0 = time.time()
            decision = self.pipeline.run(case["query"])
            elapsed_ms = round((time.time() - t0) * 1000, 1)
            latencies.append(elapsed_ms)

            decision_correct = decision.decision == case["expected_decision"]
            if decision_correct:
                correct_decisions += 1

            policy_correct = False
            if case["expected_policy"]:
                policy_correct = (
                    case["expected_policy"].lower()
                    in decision.policy_section.lower()
                )
                if policy_correct:
                    correct_policies += 1

            confidences.append(decision.confidence)
            escalations.append(1 if decision.escalation_required else 0)

            status = "✅" if decision_correct else "❌"
            print(f"        {status} expected={case['expected_decision']:<12} "
                  f"got={decision.decision:<12} "
                  f"conf={decision.confidence:.1%} "
                  f"latency={elapsed_ms:.0f}ms")

            per_case.append({
                "id":               case["id"],
                "query":            case["query"],
                "expected":         case["expected_decision"],
                "actual":           decision.decision,
                "correct":          decision_correct,
                "confidence":       round(decision.confidence, 4),
                "escalation":       decision.escalation_required,
                "latency_ms":       elapsed_ms,
            })

        total = len(DECISION_TEST_CASES)
        cases_with_policy = sum(
            1 for c in DECISION_TEST_CASES if c["expected_policy"]
        )
        lat = np.array(latencies)

        summary = {
            "decision_accuracy":  round(correct_decisions / total,             4),
            "policy_accuracy":    round(correct_policies  / cases_with_policy, 4),
            "avg_confidence":     round(float(np.mean(confidences)),           4),
            "escalation_rate":    round(float(np.mean(escalations)),           4),
            "latency_p50_ms":     round(float(np.percentile(lat, 50)),         1),
            "latency_p95_ms":     round(float(np.percentile(lat, 95)),         1),
            "latency_p99_ms":     round(float(np.percentile(lat, 99)),         1),
            "latency_mean_ms":    round(float(np.mean(lat)),                   1),
            "per_case":           per_case,
        }

        print(f"\n📊 Decision Results:")
        print(f"   Accuracy:       {summary['decision_accuracy']:.1%}")
        print(f"   Policy Match:   {summary['policy_accuracy']:.1%}")
        print(f"   Avg Confidence: {summary['avg_confidence']:.1%}")
        print(f"   Escalation Rate:{summary['escalation_rate']:.1%}")
        print(f"   Latency P50:    {summary['latency_p50_ms']:.0f}ms")
        print(f"   Latency P95:    {summary['latency_p95_ms']:.0f}ms")
        print(f"   Latency P99:    {summary['latency_p99_ms']:.0f}ms")

        return summary

    # ─────────────────────────────────────────────────────────────────────
    # Section 3: RAGAS Metrics
    # ─────────────────────────────────────────────────────────────────────

    def evaluate_ragas(self, max_samples: int = 20) -> Dict[str, Any]:
        """
        Runs RAGAS evaluation using the auto-generated dataset.
        Uses Ollama as judge — fully local, no API needed.

        RAGAS works by:
          1. Taking your question + answer + retrieved contexts + ground truth
          2. Using an LLM to judge quality across 4 dimensions
          3. Returning scores 0.0-1.0 for each dimension

        Args:
            max_samples: Cap to keep runtime reasonable
                         20 samples ≈ 10-15 min on CPU
        """
        print("\n" + "=" * 65)
        print("SECTION 3: RAGAS METRICS")
        print("=" * 65)

        if not self.eval_dataset_path.exists():
            print(f"⚠️  No eval dataset at {self.eval_dataset_path}")
            print(f"   Run first: python src/evaluation/dataset_generator.py")
            return {}

        with open(self.eval_dataset_path) as f:
            raw_dataset = json.load(f)

        samples = raw_dataset[:max_samples]
        print(f"Running RAGAS on {len(samples)} samples...")
        print("(Uses Ollama locally as judge — takes 10-15 min on CPU)\n")

        # Build RAGAS rows — run pipeline on each question
        ragas_rows = []
        for i, sample in enumerate(samples, 1):
            print(f"  [{i:02d}/{len(samples)}] {sample['question'][:60]}")
            try:
                # Get pipeline answer
                decision = self.pipeline.run(sample["question"])
                answer = decision.justification or decision.decision

                # Get retrieved contexts for this question
                retrieved = self.search.search(sample["question"], top_k=5)
                contexts  = [c["content"] for c in retrieved]

                ragas_rows.append({
                    "question":    sample["question"],
                    "answer":      answer,
                    "contexts":    contexts,
                    "ground_truth": sample["ground_truth"],
                })
            except Exception as e:
                print(f"         ⚠️  skipped: {e}")

        if not ragas_rows:
            print("❌ No valid samples")
            return {}

        # Run RAGAS
        try:
            from datasets import Dataset
            from ragas import evaluate
            from ragas.metrics import (
                faithfulness,
                answer_relevancy,
                context_recall,
                context_precision,
            )
            from langchain_ollama import ChatOllama
            from langchain_ollama import OllamaEmbeddings

            hf_dataset = Dataset.from_list(ragas_rows)

            # Use Ollama as judge — same model, fully local
            ragas_llm        = ChatOllama(model="llama3.2", temperature=0)
            ragas_embeddings = OllamaEmbeddings(model="nomic-embed-text")

            result = evaluate(
                dataset    = hf_dataset,
                metrics    = [
                    faithfulness,
                    answer_relevancy,
                    context_recall,
                    context_precision,
                ],
                llm         = ragas_llm,
                embeddings  = ragas_embeddings,
                raise_exceptions=False,
            )

            scores = result.to_pandas().mean().to_dict()
            summary = {
                "faithfulness":      round(float(scores.get("faithfulness",      0)), 4),
                "answer_relevancy":  round(float(scores.get("answer_relevancy",  0)), 4),
                "context_recall":    round(float(scores.get("context_recall",    0)), 4),
                "context_precision": round(float(scores.get("context_precision", 0)), 4),
                "n_samples":         len(ragas_rows),
            }

            print(f"\n📊 RAGAS Results:")
            print(f"   Faithfulness:      {summary['faithfulness']:.4f}")
            print(f"   Answer Relevancy:  {summary['answer_relevancy']:.4f}")
            print(f"   Context Recall:    {summary['context_recall']:.4f}")
            print(f"   Context Precision: {summary['context_precision']:.4f}")

            return summary

        except Exception as e:
            print(f"❌ RAGAS error: {e}")
            return {"error": str(e)}

    # ─────────────────────────────────────────────────────────────────────
    # Section 4: Full Run + Save
    # ─────────────────────────────────────────────────────────────────────

    def run_full_evaluation(
        self,
        run_ragas: bool = True,
        ragas_samples: int = 20,
    ) -> Dict[str, Any]:
        """
        Runs all sections and saves to evaluation/evaluation_results.json
        """
        t_start = time.time()

        all_results = {
            "metadata": {
                "version":      "v2",
                "model":        "llama3.2 (Ollama — local)",
                "architecture": "LangGraph + BGE-large + FAISS + BM25 + RRF + Cross-Encoder",
                "total_chunks": 316,
            }
        }

        # Run sections
        all_results["retrieval"] = self.evaluate_retrieval()
        all_results["decisions"] = self.evaluate_decisions()

        if run_ragas:
            all_results["ragas"] = self.evaluate_ragas(max_samples=ragas_samples)
        else:
            print("\n⏩ RAGAS skipped")

        # Save
        all_results["metadata"]["eval_time_s"] = round(time.time() - t_start, 1)
        with open(self.results_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\n💾 Results saved → {self.results_path}")

        # Final summary table
        r = all_results.get("retrieval", {})
        d = all_results.get("decisions", {})
        g = all_results.get("ragas", {})

        print(f"\n{'='*65}")
        print(f"FINAL SUMMARY")
        print(f"{'='*65}")
        print(f"  {'Metric':<28} {'Value':>10}")
        print(f"  {'-'*40}")
        print(f"  {'Recall@5':<28} {r.get('recall_at_5', 0):>10.1%}")
        print(f"  {'MRR':<28} {r.get('mrr', 0):>10.3f}")
        print(f"  {'Precision@5':<28} {r.get('precision_at_5', 0):>10.1%}")
        print(f"  {'Decision Accuracy':<28} {d.get('decision_accuracy', 0):>10.1%}")
        print(f"  {'Policy Match':<28} {d.get('policy_accuracy', 0):>10.1%}")
        print(f"  {'Avg Confidence':<28} {d.get('avg_confidence', 0):>10.1%}")
        print(f"  {'Escalation Rate':<28} {d.get('escalation_rate', 0):>10.1%}")
        print(f"  {'Latency P50 (ms)':<28} {d.get('latency_p50_ms', 0):>10.0f}")
        print(f"  {'Latency P95 (ms)':<28} {d.get('latency_p95_ms', 0):>10.0f}")
        print(f"  {'Latency P99 (ms)':<28} {d.get('latency_p99_ms', 0):>10.0f}")
        if g and "faithfulness" in g:
            print(f"  {'Faithfulness':<28} {g.get('faithfulness', 0):>10.4f}")
            print(f"  {'Answer Relevancy':<28} {g.get('answer_relevancy', 0):>10.4f}")
            print(f"  {'Context Recall':<28} {g.get('context_recall', 0):>10.4f}")
            print(f"  {'Context Precision':<28} {g.get('context_precision', 0):>10.4f}")
        print()

        return all_results


def main():
    evaluator = FullEvaluator()
    # Skip RAGAS first run — just verify retrieval + decisions work
    # Then rerun with run_ragas=True for full metrics
    evaluator.run_full_evaluation(run_ragas=False)


if __name__ == "__main__":
    main()