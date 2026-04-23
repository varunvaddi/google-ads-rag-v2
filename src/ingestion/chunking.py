"""
Smart Chunking for Google Ads Policies V2

WHY CHUNKING?
- Embeddings work best on 500-1000 character chunks
- LLMs have context limits (we can't send all 160K chars)
- Smaller chunks = more precise retrieval
- BUT: Too small = loses context

STRATEGY:
- Keep sections intact if <1000 chars (most are ~830)
- Split larger sections but preserve meaning
- Add overlap to maintain context across chunks
- Preserve metadata for citations

INPUT: data/processed/parsed_sections.json (194 sections)
OUTPUT: data/processed/chunks.json (~200-250 chunks)
"""

import json
from pathlib import Path
from typing import List, Dict
from collections import Counter
import tiktoken


class PolicyChunker:
    """
    Intelligent chunking that preserves policy structure
    
    DESIGN PRINCIPLES:
    1. Respect section boundaries (don't split mid-topic)
    2. Target 600-1000 characters per chunk
    3. Add 100-char overlap for context continuity
    4. Enrich with hierarchical metadata
    5. Preserve citations (URL, category, hierarchy)
    """
    
    def __init__(
        self,
        input_file: str = "data/processed/parsed_sections.json",
        output_dir: str = "data/processed",
        chunk_size: int = 800,
        chunk_overlap: int = 100,
        min_chunk_size: int = 200
    ):
        self.input_file = Path(input_file)
        self.output_dir = Path(output_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        
        # For token counting (GPT tokenizer)
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except:
            self.tokenizer = None
        
        print("✅ Chunker V2 initialized")
        print(f"   Target chunk size: {chunk_size} chars")
        print(f"   Overlap: {chunk_overlap} chars")
        print(f"   Minimum chunk: {min_chunk_size} chars")
    
    def load_sections(self) -> List[Dict]:
        """Load parsed sections from JSON"""
        print(f"\n📂 Loading sections from: {self.input_file}")
        
        with open(self.input_file, 'r', encoding='utf-8') as f:
            sections = json.load(f)
        
        print(f"✅ Loaded {len(sections)} sections")
        return sections
    
    def chunk_all_sections(self, sections: List[Dict]) -> List[Dict]:
        """
        Process all sections into chunks
        
        LOGIC:
        - If section < 1000 chars → Keep as one chunk
        - If section > 1000 chars → Split intelligently
        - Always preserve hierarchy and metadata
        """
        all_chunks = []
        
        print("\n" + "="*80)
        print("CHUNKING POLICY SECTIONS")
        print("="*80)
        
        stats = {
            'kept_intact': 0,
            'split': 0,
            'total_chunks': 0
        }
        
        for section in sections:
            content_length = len(section['content'])
            
            # Decision: Keep intact or split?
            if content_length <= self.chunk_size:
                # Keep entire section as one chunk
                chunks = [self._create_chunk(
                    content=section['content'],
                    section=section,
                    chunk_index=0,
                    is_full_section=True
                )]
                stats['kept_intact'] += 1
            else:
                # Split into multiple chunks
                chunks = self._split_section(section)
                stats['split'] += 1
            
            all_chunks.extend(chunks)
            stats['total_chunks'] += len(chunks)
        
        print(f"\n✅ Chunking complete!")
        print(f"   Sections kept intact: {stats['kept_intact']}")
        print(f"   Sections split: {stats['split']}")
        print(f"   Total chunks created: {stats['total_chunks']}")
        
        return all_chunks
    
    def _create_chunk(
        self,
        content: str,
        section: Dict,
        chunk_index: int,
        is_full_section: bool = False
    ) -> Dict:
        """
        Create a chunk with full metadata
        
        METADATA INCLUDES:
        - content: The actual text for embedding
        - hierarchy: ["Category", "Page", "Section"] for citations
        - url: Link to original policy
        - category: Which of 4 main categories
        - chunk_type: full_section or partial_section
        - chunk_index: Position if split (0, 1, 2...)
        - char_count: Size for filtering
        - token_count: For LLM context management
        """
        
        # Add hierarchical context prefix
        # This helps embeddings understand the broader context
        hierarchy_str = " > ".join(section['hierarchy'])
        context_prefix = f"Policy: {hierarchy_str}\n\n"
        
        # Full content with context
        full_content = context_prefix + content
        
        # Count tokens if possible
        token_count = None
        if self.tokenizer:
            try:
                token_count = len(self.tokenizer.encode(full_content))
            except:
                pass
        
        chunk = {
            'content': full_content,
            'metadata': {
                'hierarchy': section['hierarchy'],
                'url': section['url'],
                'category': section['category'],
                'section_title': section['title'],
                'chunk_type': 'full_section' if is_full_section else 'partial_section',
                'chunk_index': chunk_index,
                'content_type': section.get('content_type', 'section')
            },
            'char_count': len(full_content),
            'token_count': token_count
        }
        
        return chunk
    
    def _split_section(self, section: Dict) -> List[Dict]:
        """
        Split large section into overlapping chunks
        
        WHY OVERLAP?
        - If we split: "...prohibited substances. Healthcare products..."
        - Without overlap: First chunk ends at "substances", second starts at "Healthcare"
        - With overlap: Both chunks include the boundary → better context
        
        ALGORITHM:
        1. Split content by paragraphs (preserve meaning)
        2. Group paragraphs into ~800 char chunks
        3. Add 100-char overlap from previous chunk
        4. Ensure no chunk is too small (<200 chars)
        """
        content = section['content']
        
        # Split by double newlines (paragraphs)
        paragraphs = content.split('\n\n')
        
        chunks = []
        current_chunk = ""
        chunk_index = 0
        
        for para in paragraphs:
            # Would adding this paragraph exceed chunk_size?
            if len(current_chunk) + len(para) + 2 > self.chunk_size and current_chunk:
                # Save current chunk
                if len(current_chunk.strip()) >= self.min_chunk_size:
                    chunks.append(self._create_chunk(
                        content=current_chunk.strip(),
                        section=section,
                        chunk_index=chunk_index,
                        is_full_section=False
                    ))
                    chunk_index += 1
                
                # Start new chunk with overlap
                # Take last 100 chars from previous chunk
                overlap = current_chunk[-self.chunk_overlap:] if len(current_chunk) > self.chunk_overlap else ""
                current_chunk = overlap + "\n\n" + para
            else:
                # Add paragraph to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
        
        # Don't forget the last chunk
        if current_chunk.strip() and len(current_chunk.strip()) >= self.min_chunk_size:
            chunks.append(self._create_chunk(
                content=current_chunk.strip(),
                section=section,
                chunk_index=chunk_index,
                is_full_section=False
            ))
        
        return chunks
    
    def save_chunks(self, chunks: List[Dict], filename: str = "chunks.json"):
        """Save chunks to JSON"""
        output_path = self.output_dir / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Saved to: {output_path}")
        
        # Print statistics
        self._print_statistics(chunks)
    
    def _print_statistics(self, chunks: List[Dict]):
        """Print chunking statistics"""
        print("\n" + "="*80)
        print("CHUNKING STATISTICS")
        print("="*80)
        
        # By category
        by_category = Counter(c['metadata']['category'] for c in chunks)
        print("\n📊 Chunks by category:")
        for cat, count in sorted(by_category.items()):
            cat_name = cat.replace('_', ' ').title()
            print(f"   {cat_name}: {count}")
        
        # By chunk type
        by_type = Counter(c['metadata']['chunk_type'] for c in chunks)
        print(f"\n📦 Chunk types:")
        for chunk_type, count in by_type.items():
            print(f"   {chunk_type}: {count}")
        
        # Size statistics
        char_counts = [c['char_count'] for c in chunks]
        token_counts = [c['token_count'] for c in chunks if c['token_count']]
        
        print(f"\n📏 Size distribution:")
        print(f"   Total chunks: {len(chunks)}")
        print(f"   Average chars: {sum(char_counts) / len(char_counts):.0f}")
        print(f"   Min chars: {min(char_counts)}")
        print(f"   Max chars: {max(char_counts)}")
        
        if token_counts:
            print(f"   Average tokens: {sum(token_counts) / len(token_counts):.0f}")
            print(f"   Max tokens: {max(token_counts)}")
        
        # Sample chunks
        print(f"\n🔍 Sample chunks:")
        for i, chunk in enumerate(chunks[:2], 1):
            hierarchy = ' > '.join(chunk['metadata']['hierarchy'])
            print(f"\n   {i}. {hierarchy}")
            print(f"      Type: {chunk['metadata']['chunk_type']}")
            print(f"      Size: {chunk['char_count']} chars")
            print(f"      Preview: {chunk['content'][:100]}...")


def main():
    """Run the chunker"""
    print("\n" + "="*80)
    print("PHASE B: SMART CHUNKING")
    print("="*80)
    
    chunker = PolicyChunker()
    
    # Load sections
    sections = chunker.load_sections()
    
    if not sections:
        print("\n❌ No sections found! Run parse_policies_v2.py first.")
        return
    
    # Chunk all sections
    chunks = chunker.chunk_all_sections(sections)
    
    if not chunks:
        print("\n❌ No chunks created!")
        return
    
    # Save results
    chunker.save_chunks(chunks)
    
    print("\n✅ Phase B Complete: Chunking")
    print(f"   Created {len(chunks)} embeddings-ready chunks")
    print(f"   Next: Generate embeddings with BGE-large-en-v1.5")


if __name__ == "__main__":
    main()