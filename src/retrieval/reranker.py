"""
Cross-Encoder Reranking

WHAT IS CROSS-ENCODER?
A model that scores query + document as a PAIR

BI-ENCODER (Phase 2):
query → embedding_A
document → embedding_B
similarity = cosine(embedding_A, embedding_B)

CROSS-ENCODER (Phase 3):
model([query, document]) → relevance_score

WHY IS CROSS-ENCODER BETTER?
- Sees query + document TOGETHER
- Can understand relationships between them
- Much more accurate (but slower)

ANALOGY:
Bi-encoder: Looking at two photos separately and guessing if they're related
Cross-encoder: Looking at both photos side-by-side

WHEN TO USE:
- Bi-encoder: Initial retrieval (fast, approximate)
- Cross-encoder: Reranking top candidates (slow, precise)

TYPICAL PIPELINE:
1000 docs → Bi-encoder → Top 20 → Cross-encoder → Top 5
         (50ms)                      (100ms)
"""

from typing import List, Dict
from sentence_transformers import CrossEncoder
import numpy as np


class Reranker:
    """
    Cross-encoder reranking for improved precision
    
    Takes top-K candidates and re-scores them with more accurate model
    """
    
    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):  # was bge-reranker-large
        """
        Initialize reranker
        
        Args:
            model_name: Cross-encoder model to use
                - BGE-reranker-large is one of the best, but also slower and larger size on CPU. BGE-reranker-base is faster and bad.
                - ~500MB model size
                - Trained specifically for reranking
        """
        print(f"🎯 Loading reranker model: {model_name}")
        print("   (First run downloads ~500MB, then cached)")
        
        self.model = CrossEncoder(model_name)
        
        print("✅ Reranker loaded!")
    
    def rerank(
        self,
        query: str,
        candidates: List[Dict],
        top_k: int = 5
    ) -> List[Dict]:
        """
        Rerank candidates using cross-encoder
        
        PROCESS:
        1. Create [query, document] pairs
        2. Score each pair with cross-encoder
        3. Sort by new scores
        4. Return top K
        
        Args:
            query: Search query
            candidates: List of candidate chunks (from hybrid search)
            top_k: How many to return after reranking
        
        Returns:
            Reranked list of chunks
        
        Example:
            Before reranking:
            1. Chunk A (hybrid score: 0.023)
            2. Chunk B (hybrid score: 0.021)
            3. Chunk C (hybrid score: 0.019)
            
            After reranking:
            1. Chunk C (rerank score: 0.95) ← Actually most relevant!
            2. Chunk A (rerank score: 0.87)
            3. Chunk B (rerank score: 0.65) ← Was false positive
        """
        if not candidates:
            return []
        
        # Create query-document pairs
        pairs = [[query, candidate['content']] for candidate in candidates]
        
        # Score all pairs with cross-encoder
        # This is the expensive operation (but worth it!)
        scores = self.model.predict(pairs)
        
        # Get indices sorted by score (descending)
        ranked_indices = np.argsort(scores)[::-1][:top_k]
        
        # Build reranked results
        reranked = []
        for new_rank, idx in enumerate(ranked_indices, 1):
            result = candidates[idx].copy()
            result['rank'] = new_rank
            result['rerank_score'] = float(scores[idx])
            result['original_score'] = result.get('score', 0.0)  # Keep original
            result['score'] = float(scores[idx])  # Replace with rerank score
            reranked.append(result)
        
        return reranked
    
    def print_comparison(
        self,
        original_results: List[Dict],
        reranked_results: List[Dict]
    ):
        """
        Print before/after comparison
        
        Shows how reranking changed the order
        """
        print("\n" + "=" * 80)
        print("📊 RERANKING COMPARISON")
        print("=" * 80)
        
        print("\n🔵 BEFORE RERANKING (Hybrid Search):")
        print("─" * 80)
        for result in original_results[:5]:
            hierarchy = " > ".join(result['metadata']['hierarchy'])
            print(f"{result['rank']}. [{result['score']:.4f}] {hierarchy[:60]}")
        
        print("\n🟢 AFTER RERANKING (Cross-Encoder):")
        print("─" * 80)
        for result in reranked_results:
            hierarchy = " > ".join(result['metadata']['hierarchy'])
            orig_rank = next(
                i for i, r in enumerate(original_results, 1) 
                if r['chunk_id'] == result['chunk_id']
            )
            rank_change = orig_rank - result['rank']
            
            change_icon = "↑" if rank_change > 0 else "↓" if rank_change < 0 else "="
            print(f"{result['rank']}. [{result['score']:.4f}] {hierarchy[:60]}")
            print(f"   {change_icon} Was rank #{orig_rank} (moved {abs(rank_change)} positions)")
        
        print("\n" + "=" * 80)


def main():
    """Demo reranking"""
    from .hybrid_search import HybridSearch
    
    print("=" * 80)
    print("CROSS-ENCODER RERANKING DEMO")
    print("=" * 80)
    
    # Initialize
    hybrid = HybridSearch()
    reranker = Reranker()
    
    # Test query
    query = "Are cryptocurrency trading courses allowed in Google Ads?"
    
    print(f"\n🔎 Query: \"{query}\"")
    print("─" * 80)
    
    # Get hybrid results (more candidates for reranking)
    print("\n1️⃣  Running hybrid search (getting top 10 candidates)...")
    hybrid_results = hybrid.search(query, top_k=10)
    
    print(f"   ✅ Got {len(hybrid_results)} candidates")
    
    # Rerank
    print("\n2️⃣  Reranking with cross-encoder...")
    reranked_results = reranker.rerank(query, hybrid_results, top_k=5)
    
    print(f"   ✅ Reranked to top {len(reranked_results)}")
    
    # Show comparison
    reranker.print_comparison(hybrid_results, reranked_results)
    
    # Show detailed results
    print("\n📋 FINAL RESULTS (After Reranking):")
    print("=" * 80)
    for result in reranked_results:
        hierarchy = " > ".join(result['metadata']['hierarchy'])
        print(f"\n🏆 Rank #{result['rank']}")
        print(f"📊 Rerank Score: {result['rerank_score']:.4f}")
        print(f"📊 Original Score: {result['original_score']:.4f}")
        print(f"📂 {hierarchy}")
        print(f"💬 {result['content'][:200]}...")


if __name__ == "__main__":
    main()