"""
Semantic Search V2 - Using Clean Data
"""

import numpy as np
import faiss
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
from typing import List, Dict


class PolicySearch:
    """
    Search system using clean data (junk removed)
    """
    
    def __init__(self):
        embeddings_dir = Path("data/embeddings")
        
        print("🔍 Loading clean search system...")
        
        # Load model
        self.encoder = SentenceTransformer('BAAI/bge-large-en-v1.5')
        
        # Load FAISS index (clean)
        self.index = faiss.read_index(str(embeddings_dir / "faiss.index"))
        
        # Load metadata (clean)
        with open(embeddings_dir / "metadata.json", 'r') as f:
            self.metadata = json.load(f)
        
        # Load chunks (clean)
        with open('data/processed/chunks.json', 'r') as f:
            self.chunks = json.load(f)
        
        print(f"✅ Clean search system ready!")
        print(f"   Vectors: {self.index.ntotal}")
        print(f"   Chunks: {len(self.chunks)}")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for relevant policy chunks"""
        
        # Encode query
        query_vector = self.encoder.encode([query], normalize_embeddings=True)
        
        # Search FAISS index
        distances, indices = self.index.search(query_vector, top_k)
        
        # Format results
        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], distances[0]), 1):
            chunk = self.chunks[idx]
            
            results.append({
                'rank': rank,
                'score': float(score),
                'content': chunk['content'],
                'metadata': chunk['metadata']
            })
        
        return results
    
    def print_results(self, results: List[Dict]):
        """Pretty print search results"""
        print("\n" + "="*80)
        print("SEMANTIC SEARCH RESULTS (CLEAN DATA)")
        print("="*80)
        
        for result in results:
            hierarchy = " > ".join(result['metadata']['hierarchy'])
            
            print(f"\n🏆 Rank #{result['rank']} | Score: {result['score']:.4f}")
            print(f"📂 {hierarchy}")
            print(f"🔗 {result['metadata']['url']}")
            print(f"\n💬 {result['content'][:250]}...")
            print()


if __name__ == "__main__":
    search = PolicySearch()
    results = search.search("cryptocurrency ads", top_k=3)
    search.print_results(results)
