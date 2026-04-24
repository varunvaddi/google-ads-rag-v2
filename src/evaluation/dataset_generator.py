"""
src/evaluation/dataset_generator.py

Auto-generates RAGAS evaluation dataset from your 316 clean policy chunks.
Uses Ollama llama3.2 locally — free, unlimited, no API key needed.

OUTPUT FORMAT (one entry per chunk):
{
    "question":          "Can I advertise weight loss supplements on Google Ads?",
    "ground_truth":      "Weight loss supplements fall under restricted healthcare...",
    "contexts":          ["full policy chunk text"],
    "reference_url":     "https://support.google.com/adspolicy/...",
    "category":          "Restricted Content > Healthcare and medicines",
    "expected_decision": "restricted"
}

Run once, reuse forever.
Takes ~15-20 minutes for 50 samples on MacBook Air CPU.
"""

import json
import time
import re
import random
from pathlib import Path
from typing import List, Dict, Any, Optional

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage


# ── Prompt for generating Q&A from a policy chunk ─────────────────────────
SYSTEM_PROMPT = """You are a Google Ads policy expert generating realistic evaluation examples.

Given a policy chunk, you must generate:
1. A realistic question an advertiser would ask about this policy
2. The correct answer based strictly on the policy text
3. Whether an ad in this category would be allowed, restricted, or disallowed

Rules:
- Question must be specific and realistic — something a real advertiser would ask
- Answer must be based ONLY on the provided policy text
- expected_decision must be one of: allowed, restricted, disallowed

Respond ONLY with valid JSON — no markdown, no extra text:
{
    "question": "realistic advertiser question about this policy",
    "ground_truth": "correct answer based on policy text (2-3 sentences)",
    "expected_decision": "allowed" or "restricted" or "disallowed"
}"""


class EvalDatasetGenerator:
    """
    Generates RAGAS-compatible evaluation dataset from policy chunks.
    Uses Ollama llama3.2 — runs fully locally.
    """

    def __init__(self, chunks_path: str = "data/processed/chunks.json"):
        self.chunks_path = Path(chunks_path)

        # Slightly higher temperature for varied questions
        self.llm = ChatOllama(
            model="llama3.2",
            temperature=0.3,
            num_predict=512,
        )

    def _load_chunks(self) -> List[Dict]:
        """Load chunks from disk."""
        with open(self.chunks_path) as f:
            return json.load(f)

    def _select_chunks(self, chunks: List[Dict], n: int) -> List[Dict]:
        """
        Select n chunks that cover all policy categories evenly.

        WHY EVEN COVERAGE:
        If we just take the first n chunks, we might over-represent
        one category. Even coverage = more reliable eval metrics.
        """
        # Group by top-level category
        by_category: Dict[str, List[Dict]] = {}
        for chunk in chunks:
            hierarchy = chunk.get("metadata", {}).get("hierarchy", ["Unknown"])
            category = hierarchy[0] if hierarchy else "Unknown"
            by_category.setdefault(category, []).append(chunk)

        print(f"\nPolicy categories found:")
        for cat, cat_chunks in by_category.items():
            print(f"  {cat}: {len(cat_chunks)} chunks")

        # Take evenly from each category, prefer longer chunks
        selected = []
        per_category = max(1, n // len(by_category))

        for category, cat_chunks in by_category.items():
            # Sort by content length — longer = more policy detail = better Q&A
            sorted_chunks = sorted(
                cat_chunks,
                key=lambda c: len(c.get("content", "")),
                reverse=True
            )
            selected.extend(sorted_chunks[:per_category])

        # Fill remaining slots randomly from leftover chunks
        used = set(id(c) for c in selected)
        remaining = [c for c in chunks if id(c) not in used]
        random.shuffle(remaining)
        selected.extend(remaining[:max(0, n - len(selected))])

        return selected[:n]

    def _generate_one(self, chunk: Dict) -> Optional[Dict[str, Any]]:
        """
        Generate one Q&A triple from a single chunk.
        Returns None if generation fails.
        """
        hierarchy = " > ".join(chunk.get("metadata", {}).get("hierarchy", []))
        url = chunk.get("metadata", {}).get("url", "")
        content = chunk.get("content", "")

        # Skip very short chunks — not enough content for good Q&A
        if len(content) < 100:
            return None

        user_prompt = f"""Policy Category: {hierarchy}
Source URL: {url}

Policy Text:
{content[:800]}

Generate a realistic advertiser Q&A for this exact policy."""

        try:
            response = self.llm.invoke([
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ])

            raw = response.content.strip()

            # Strip markdown fences if present
            raw = re.sub(r"^```json\s*", "", raw)
            raw = re.sub(r"^```\s*",     "", raw)
            raw = re.sub(r"\s*```$",     "", raw)
            raw = raw.strip()

            parsed = json.loads(raw)

            # Validate required fields
            if not all(k in parsed for k in ["question", "ground_truth", "expected_decision"]):
                return None

            # Validate decision value
            if parsed["expected_decision"] not in ("allowed", "restricted", "disallowed"):
                parsed["expected_decision"] = "restricted"  # safe default

            return {
                "question":          parsed["question"],
                "ground_truth":      parsed["ground_truth"],
                "contexts":          [content],      # the chunk that answers this question
                "reference_url":     url,
                "category":          hierarchy,
                "expected_decision": parsed["expected_decision"],
            }

        except json.JSONDecodeError:
            return None
        except Exception:
            return None

    def generate(
        self,
        n_samples: int = 50,
        output_path: str = "evaluation/eval_dataset.json"
    ) -> List[Dict]:
        """
        Generate n_samples Q&A triples and save to disk.

        Args:
            n_samples:   How many eval examples to generate
                         50 = solid for RAGAS, takes ~15-20 min
                         20 = faster, takes ~5-8 min
            output_path: Where to save the dataset

        Returns:
            List of Q&A triples
        """
        print("=" * 65)
        print(f"EVAL DATASET GENERATOR")
        print(f"Target: {n_samples} samples | Model: llama3.2 | Local")
        print("=" * 65)

        # Load and select chunks
        chunks = self._load_chunks()
        print(f"\nLoaded {len(chunks)} chunks from {self.chunks_path}")

        selected = self._select_chunks(chunks, n=n_samples)
        print(f"\nSelected {len(selected)} chunks for generation")
        print(f"Estimated time: ~{len(selected) * 15 // 60} minutes\n")

        dataset = []
        failed = 0

        for i, chunk in enumerate(selected, 1):
            hierarchy = " > ".join(chunk.get("metadata", {}).get("hierarchy", []))
            print(f"[{i:02d}/{len(selected)}] {hierarchy[:60]}")

            t0 = time.time()
            result = self._generate_one(chunk)
            elapsed = round(time.time() - t0, 1)

            if result:
                dataset.append(result)
                print(f"         ✅ Q: {result['question'][:65]}")
                print(f"            decision={result['expected_decision']} | {elapsed}s")
            else:
                failed += 1
                print(f"         ❌ skipped ({elapsed}s)")

        # Save to disk
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(out_path, "w") as f:
            json.dump(dataset, f, indent=2)

        # Summary
        decisions = {}
        for entry in dataset:
            d = entry["expected_decision"]
            decisions[d] = decisions.get(d, 0) + 1

        print(f"\n{'='*65}")
        print(f"✅ Dataset saved → {out_path}")
        print(f"   Generated: {len(dataset)} samples")
        print(f"   Failed:    {failed} samples")
        print(f"   Decision breakdown: {decisions}")
        print(f"{'='*65}")

        return dataset


def main():
    gen = EvalDatasetGenerator()

    # Start with 20 samples to verify it works
    # Then rerun with 50 for full eval
    gen.generate(
        n_samples=20,
        output_path="evaluation/eval_dataset.json"
    )


if __name__ == "__main__":
    main()