"""
Build FAISS Vector Store

WHY FAISS?
- Enables fast similarity search (find nearest neighbors)
- Optimized C++ implementation (50-100x faster than naive search)
- Scalable to millions of vectors
- Used by Facebook, Google, industry standard

INPUT: data/embeddings/embeddings.npy (367 × 1024 vectors)
OUTPUT: data/embeddings/faiss.index (FAISS search index)
"""

import numpy as np
import faiss
import json
from pathlib import Path
import time


class VectorStore:
    """
    Build FAISS index for fast similarity search
    
    INDEX TYPE: FlatIP (Flat Index with Inner Product)
    - "Flat" = Exact search, no approximation
    - "IP" = Inner Product (cosine similarity for normalized vectors)
    - Perfect for <100K vectors
    - 100% recall (finds true nearest neighbors)
    """
    
    def __init__(
        self,
        embeddings_dir: str = "data/embeddings"
    ):
        self.embeddings_dir = Path(embeddings_dir)
        
        print("="*80)
        print("PHASE D: BUILD FAISS VECTOR STORE")
        print("="*80)
        print(f"📁 Embeddings directory: {self.embeddings_dir}")
    
    def load_embeddings(self) -> np.ndarray:
        """Load embeddings from numpy file"""
        embeddings_path = self.embeddings_dir / "embeddings.npy"
        
        print(f"\n📂 Loading embeddings from: {embeddings_path}")
        
        embeddings = np.load(embeddings_path)
        
        print(f"✅ Loaded embeddings")
        print(f"   Shape: {embeddings.shape}")
        print(f"   Size: {embeddings.nbytes / 1024 / 1024:.2f} MB")
        
        return embeddings
    
    def build_index(self, embeddings: np.ndarray) -> faiss.Index:
        """
        Build FAISS index
        
        PROCESS:
        1. Create index (FlatIP for exact search)
        2. Add all vectors to index
        3. Verify index built correctly
        
        WHY FlatIP?
        - Exact search (100% recall)
        - Fast for <100K vectors (<100ms per query)
        - Inner product = cosine similarity (since vectors normalized)
        
        ALTERNATIVES (if we had millions of vectors):
        - IVF: Inverted file, approximate search
        - HNSW: Hierarchical graph, very fast
        - But for 367 vectors, Flat is perfect!
        """
        print(f"\n🏗️  Building FAISS index...")
        
        dim = embeddings.shape[1]  # 1024
        
        # Create index: FlatIP for exact inner product search
        index = faiss.IndexFlatIP(dim)
        
        print(f"   Index type: FlatIP (Flat Inner Product)")
        print(f"   Dimension: {dim}")
        print(f"   Vectors to add: {len(embeddings)}")
        
        # Add all vectors to index
        start = time.time()
        index.add(embeddings)
        elapsed = time.time() - start
        
        print(f"✅ Index built in {elapsed:.3f}s")
        print(f"   Vectors in index: {index.ntotal}")
        
        return index
    
    def test_index(self, index: faiss.Index, embeddings: np.ndarray):
        """
        Test the index with sample queries
        
        WHAT WE TEST:
        1. Can we retrieve vectors?
        2. Are similarities in expected range?
        3. Does it return correct number of results?
        """
        print(f"\n🧪 Testing index...")
        
        # Test query: Use first embedding as query
        query = embeddings[0:1]  # Shape: (1, 1024)
        k = 5  # Get top 5 results
        
        # Search
        start = time.time()
        distances, indices = index.search(query, k)
        elapsed = time.time() - start
        
        print(f"✅ Search completed in {elapsed*1000:.2f}ms")
        print(f"\n   Top {k} results:")
        for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
            print(f"   {i+1}. Index {idx}: similarity {dist:.4f}")
        
        # Verify
        assert indices[0][0] == 0, "First result should be query itself!"
        assert distances[0][0] > 0.99, "Self-similarity should be ~1.0"
        
        print(f"\n✅ Index working correctly!")
    
    def save_index(self, index: faiss.Index):
        """Save FAISS index to disk"""
        index_path = self.embeddings_dir / "faiss.index"
        
        print(f"\n💾 Saving index to: {index_path}")
        
        faiss.write_index(index, str(index_path))
        
        file_size = index_path.stat().st_size / 1024 / 1024
        print(f"✅ Saved! Size: {file_size:.2f} MB")
    
    def save_stats(self, embeddings: np.ndarray, index: faiss.Index):
        """Save index statistics"""
        stats = {
            "index_type": "FlatIP",
            "num_vectors": int(index.ntotal),
            "dimension": embeddings.shape[1],
            "index_size_mb": (self.embeddings_dir / "faiss.index").stat().st_size / 1024 / 1024,
            "embeddings_size_mb": embeddings.nbytes / 1024 / 1024,
            "built_at": time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        stats_path = self.embeddings_dir / "faiss_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"📊 Stats saved to: {stats_path}")


def main():
    """Build FAISS vector store"""
    
    store = VectorStore()
    
    # Load embeddings
    embeddings = store.load_embeddings()
    
    # Build index
    index = store.build_index(embeddings)
    
    # Test it works
    store.test_index(index, embeddings)
    
    # Save to disk
    store.save_index(index)
    store.save_stats(embeddings, index)
    
    print("\n" + "="*80)
    print("✅ PHASE D COMPLETE: VECTOR STORE")
    print("="*80)
    print(f"\n📊 Summary:")
    print(f"   Vectors indexed: {index.ntotal}")
    print(f"   Index type: FlatIP (exact search)")
    print(f"   Search speed: <10ms per query")
    print(f"   Files created:")
    print(f"      • faiss.index (search index)")
    print(f"      • faiss_stats.json (metadata)")
    print(f"\n🚀 Next: Build hybrid search (BM25 + Dense)")


if __name__ == "__main__":
    main()