"""
BM25 Search V2 - Using Clean Data
"""

import json
import pickle
from pathlib import Path
from typing import List, Dict


class BM25Search:
    """
    BM25 keyword search using clean data
    """
    
    def __init__(self):
        print("📊 Loading clean BM25 search...")
        
        # Load clean chunks
        with open('data/processed/chunks.json', 'r', encoding='utf-8') as f:
            self.chunks = json.load(f)
        
        # Load clean BM25 index
        with open('data/embeddings/bm25.pkl', 'rb') as f:
            self.bm25 = pickle.load(f)
        
        print(f"✅ Clean BM25 ready! Documents: {len(self.chunks)}")
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        text = text.lower()
        for char in '.,!?;:()[]{}"\'-':
            text = text.replace(char, ' ')
        tokens = [t for t in text.split() if t]
        return tokens
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search using BM25"""
        
        # Tokenize query
        query_tokens = self._tokenize(query)
        
        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top K indices
        import numpy as np
        top_indices = scores.argsort()[::-1][:top_k]
        
        # Format results
        results = []
        for rank, idx in enumerate(top_indices, 1):
            chunk = self.chunks[idx]
            score = float(scores[idx])
            
            results.append({
                'rank': rank,
                'score': score,
                'content': chunk['content'],
                'metadata': chunk['metadata']
            })
        
        return results


if __name__ == "__main__":
    search = BM25Search()
    results = search.search("cryptocurrency ads", top_k=3)
    
    print("\nBM25 RESULTS (CLEAN DATA):")
    for r in results:
        print(f"{r['rank']}. Score: {r['score']:.2f}")
        print(f"   {' > '.join(r['metadata']['hierarchy'][:3])}")
