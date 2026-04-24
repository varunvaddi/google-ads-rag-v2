"""
run_pipeline.py
Main entrypoint for v2 RAG pipeline.
Replaces run_phase4_Generation.py from v1.

Usage:
    python run_pipeline.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.graph.pipeline import RAGPipeline


TEST_CASES = [
    {
        "name": "Misleading Health Claims",
        "ad": "Lose 15 pounds in one week with this miracle pill! Guaranteed!",
        "expected": "disallowed",
    },
    {
        "name": "Crypto Education",
        "ad": "Learn cryptocurrency trading from certified instructors.",
        "expected": "restricted",
    },
    {
        "name": "Standard Product",
        "ad": "Buy our smartphone - 5G, 128GB. Free shipping over $50.",
        "expected": "allowed",
    },
    {
        "name": "Financial Guarantee",
        "ad": "100% guaranteed 30% annual returns — invest with us today!",
        "expected": "disallowed",
    },
    {
        "name": "Alcohol Ad",
        "ad": "Premium craft whiskey delivered to your door. 21+ only.",
        "expected": "restricted",
    },
]


def main():
    print("=" * 65)
    print("GOOGLE ADS POLICY RAG  v2")
    print("LangGraph + Ollama llama3.2 + BGE-large + FAISS + BM25")
    print("=" * 65)

    pipeline = RAGPipeline()
    results = []

    for i, test in enumerate(TEST_CASES, 1):
        print(f"\n\nTEST {i}: {test['name']}")
        print(f"Expected: {test['expected']}")

        decision = pipeline.run(test["ad"])
        pipeline.print_decision(decision)

        match = decision.decision == test["expected"]
        results.append({
            "test": test["name"],
            "expected": test["expected"],
            "actual": decision.decision,
            "confidence": decision.confidence,
            "match": match,
        })

    # Summary
    print(f"\n\n{'='*65}")
    print("RESULTS SUMMARY")
    print(f"{'='*65}")
    for r in results:
        status = "✅" if r["match"] else "❌"
        print(f"  {status}  {r['test']:<30} "
              f"got={r['actual']:<12} "
              f"expected={r['expected']:<12} "
              f"conf={r['confidence']:.1%}")

    passed = sum(1 for r in results if r["match"])
    print(f"\n  Result: {passed}/{len(results)} passed ({passed/len(results):.0%})")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise