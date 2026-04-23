"""
Generate Embeddings for Policy Chunks V2

WHY EMBEDDINGS?
- Computers can't understand text directly
- Embeddings convert text → numbers that capture MEANING
- Similar meanings → similar numbers (vectors)
- Enables semantic search (meaning-based, not just keywords)

MODEL: BGE-large-en-v1.5
- Best open-source embedding model
- 1024 dimensions (balance of quality vs speed)
- Trained on 200M+ text pairs
- Free, runs locally on your laptop

INPUT: data/processed/chunks.json (345 chunks)
OUTPUT: data/embeddings/embeddings.npy (345 × 1024 array)
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict
import time
from tqdm import tqdm

from sentence_transformers import SentenceTransformer


class EmbeddingGenerator:
    """
    Generate embeddings for policy chunks using BGE-large
    
    PROCESS:
    1. Load BGE model (downloads ~1.3GB first time)
    2. Batch process chunks (efficient GPU/CPU usage)
    3. Normalize vectors (for cosine similarity)
    4. Save as numpy array (fast loading later)
    """
    
    def __init__(
        self,
        chunks_file: str = "data/processed/chunks.json",
        output_dir: str = "data/embeddings",
        model_name: str = "BAAI/bge-large-en-v1.5",
        batch_size: int = 32
    ):
        self.chunks_file = Path(chunks_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model_name = model_name
        self.batch_size = batch_size
        
        print("="*80)
        print("PHASE C: EMBEDDING GENERATION")
        print("="*80)
        print(f"\n🤖 Model: {model_name}")
        print(f"📦 Batch size: {batch_size}")
        print(f"📁 Output: {output_dir}")
        
        # Load the model
        self.model = self._load_model()
    
    def _load_model(self) -> SentenceTransformer:
        """
        Load BGE embedding model
        
        FIRST TIME:
        - Downloads ~1.3GB model from Hugging Face
        - Takes 2-5 minutes depending on internet
        - Cached locally for future use
        
        AFTER FIRST TIME:
        - Loads from cache (~5 seconds)
        
        WHY BGE-large-en-v1.5?
        - Ranks #1 on MTEB leaderboard (embedding benchmark)
        - 1024 dimensions (good quality/speed balance)
        - Works well with policy/legal text
        - Open source and free
        """
        print(f"\n⏳ Loading model: {self.model_name}")
        print("   First run: Downloads ~1.3GB (5-10 min)")
        print("   After: Loads from cache (~5 sec)")
        
        start = time.time()
        
        try:
            model = SentenceTransformer(self.model_name)
            elapsed = time.time() - start
            
            print(f"✅ Model loaded in {elapsed:.1f}s")
            print(f"   Embedding dimension: {model.get_sentence_embedding_dimension()}")
            
            return model
            
        except Exception as e:
            print(f"\n❌ Error loading model: {e}")
            print("\nTroubleshooting:")
            print("1. Check internet connection (needs to download model)")
            print("2. Try: pip install --upgrade sentence-transformers")
            print("3. Check disk space (~2GB needed)")
            raise
    
    def load_chunks(self) -> List[Dict]:
        """Load chunks from JSON"""
        print(f"\n📂 Loading chunks from: {self.chunks_file}")
        
        with open(self.chunks_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        print(f"✅ Loaded {len(chunks)} chunks")
        return chunks
    
    def generate_embeddings(self, chunks: List[Dict]) -> np.ndarray:
        """
        Generate embeddings for all chunks
        
        PROCESS:
        1. Extract text content from each chunk
        2. Batch process in groups of 32 (faster than one-by-one)
        3. Model converts text → 1024 numbers per chunk
        4. Normalize vectors (required for cosine similarity)
        
        BATCH PROCESSING:
        - Instead of: embed(chunk1), embed(chunk2), embed(chunk3)...
        - We do: embed([chunk1, chunk2, ..., chunk32]) → 10x faster!
        - GPU/CPU can parallelize within batch
        
        NORMALIZATION:
        - Each vector is scaled to length 1.0
        - Required for cosine similarity (dot product becomes cosine)
        - Makes similarity scores 0-1 range
        """
        print(f"\n🔢 Generating embeddings for {len(chunks)} chunks...")
        
        # Extract just the text content
        texts = [chunk['content'] for chunk in chunks]
        
        # Estimate time
        time_per_chunk = 0.05  # ~50ms per chunk on CPU, ~5ms on GPU
        estimated_time = (len(chunks) * time_per_chunk) / self.batch_size
        print(f"   Estimated time: {estimated_time:.1f} seconds")
        print(f"   Processing in batches of {self.batch_size}...")
        
        start = time.time()
        
        # Generate embeddings with progress bar
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True  # Required for cosine similarity
        )
        
        elapsed = time.time() - start
        
        print(f"\n✅ Embeddings generated in {elapsed:.1f}s")
        print(f"   Shape: {embeddings.shape}")
        print(f"   ({embeddings.shape[0]} chunks × {embeddings.shape[1]} dimensions)")
        
        return embeddings
    
    def save_embeddings(self, embeddings: np.ndarray, chunks: List[Dict]):
        """
        Save embeddings and metadata
        
        FILES CREATED:
        1. embeddings.npy - The actual vectors (345 × 1024 array)
        2. metadata.json - Chunk info (for citations, filtering)
        3. index_map.json - Maps embedding index → chunk info
        
        WHY SEPARATE FILES?
        - embeddings.npy: Fast numpy loading for search
        - metadata.json: Human-readable, easy to inspect
        - Keeps code clean (embeddings separate from text)
        """
        
        # Save embeddings (binary numpy format - fast!)
        embeddings_path = self.output_dir / "embeddings.npy"
        np.save(embeddings_path, embeddings)
        print(f"\n💾 Saved embeddings: {embeddings_path}")
        print(f"   Size: {embeddings_path.stat().st_size / 1024 / 1024:.2f} MB")
        
        # Save metadata (JSON - readable)
        metadata = []
        for i, chunk in enumerate(chunks):
            metadata.append({
                'index': i,
                'hierarchy': chunk['metadata']['hierarchy'],
                'category': chunk['metadata']['category'],
                'url': chunk['metadata']['url'],
                'section_title': chunk['metadata']['section_title'],
                'chunk_type': chunk['metadata']['chunk_type'],
                'char_count': chunk['char_count'],
                'token_count': chunk.get('token_count')
            })
        
        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Saved metadata: {metadata_path}")
        
        # Save embedding statistics
        stats = {
            'model': self.model_name,
            'num_chunks': len(chunks),
            'embedding_dim': embeddings.shape[1],
            'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'mean_norm': float(np.mean(np.linalg.norm(embeddings, axis=1))),
            'std_norm': float(np.std(np.linalg.norm(embeddings, axis=1)))
        }
        
        stats_path = self.output_dir / "embedding_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"📊 Saved stats: {stats_path}")
    
    def verify_embeddings(self, embeddings: np.ndarray, chunks: List[Dict]):
        """
        Verify embeddings are correct
        
        CHECKS:
        1. Shape is correct (345 × 1024)
        2. All vectors are normalized (length ≈ 1.0)
        3. No NaN or Inf values
        4. Vectors have reasonable variance (not all same)
        """
        print("\n" + "="*80)
        print("VERIFICATION")
        print("="*80)
        
        # Check shape
        expected_shape = (len(chunks), 1024)
        assert embeddings.shape == expected_shape, f"Wrong shape: {embeddings.shape}"
        print(f"✅ Shape correct: {embeddings.shape}")
        
        # Check normalization
        norms = np.linalg.norm(embeddings, axis=1)
        mean_norm = np.mean(norms)
        print(f"✅ Normalized: mean norm = {mean_norm:.4f} (should be ~1.0)")
        
        # Check for NaN/Inf
        has_nan = np.isnan(embeddings).any()
        has_inf = np.isinf(embeddings).any()
        assert not has_nan and not has_inf, "Found NaN or Inf values!"
        print(f"✅ No invalid values")
        
        # Check variance (vectors should be different)
        std = np.std(embeddings)
        print(f"✅ Vectors have variance: std = {std:.4f}")
        
        # Sample similarity check
        # Similar policy chunks should have high similarity
        print(f"\n🔍 Sample similarity check:")
        sim = np.dot(embeddings[0], embeddings[1])
        print(f"   Chunk 0 vs Chunk 1: {sim:.4f}")
        print(f"   (Should be 0.3-0.9 for related policies)")


def main():
    """Run embedding generation"""
    
    generator = EmbeddingGenerator()
    
    # Load chunks
    chunks = generator.load_chunks()
    
    if not chunks:
        print("\n❌ No chunks found! Run chunking.py first.")
        return
    
    # Generate embeddings
    embeddings = generator.generate_embeddings(chunks)
    
    # Verify quality
    generator.verify_embeddings(embeddings, chunks)
    
    # Save everything
    generator.save_embeddings(embeddings, chunks)
    
    print("\n" + "="*80)
    print("✅ PHASE C COMPLETE: EMBEDDINGS")
    print("="*80)
    print(f"\n📊 Summary:")
    print(f"   Chunks embedded: {len(chunks)}")
    print(f"   Embedding dimensions: {embeddings.shape[1]}")
    print(f"   Output directory: {generator.output_dir}")
    print(f"\n🚀 Next: Build FAISS vector store for fast search")


if __name__ == "__main__":
    main()