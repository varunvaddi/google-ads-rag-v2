"""
clean_chunks.py
Removes junk chunks from data/processed/chunks.json

Junk patterns identified:
- "Was this helpful?" — UI feedback button from Google's help pages
- "Yes No" — the yes/no buttons that follow "Was this helpful?"
- Very short chunks (< 50 chars) — usually navigation/UI remnants
"""

import json
from pathlib import Path


JUNK_PHRASES = [
    "was this helpful",
    "was this article helpful",
    "yes no",
    "send feedback",
    "except as otherwise noted",
]

MIN_CHUNK_LENGTH = 80  # characters


def is_junk(chunk: dict) -> bool:
    content = chunk.get("content", "").lower().strip()

    # Too short to be useful
    if len(content) < MIN_CHUNK_LENGTH:
        return True

    # Contains junk phrases
    if any(phrase in content for phrase in JUNK_PHRASES):
        return True

    return False


def main():
    input_path = Path("data/processed/chunks.json")
    output_path = Path("data/processed/chunks_clean.json")

    print(f"Loading chunks from {input_path}...")
    with open(input_path) as f:
        chunks = json.load(f)

    print(f"Total chunks before cleaning: {len(chunks)}")

    # Show some junk examples before removing
    junk_chunks = [c for c in chunks if is_junk(c)]
    clean_chunks = [c for c in chunks if not is_junk(c)]

    print(f"\nJunk chunks found: {len(junk_chunks)}")
    print(f"Clean chunks remaining: {len(clean_chunks)}")

    print(f"\nSample junk chunks being removed:")
    for chunk in junk_chunks[:5]:
        hierarchy = " > ".join(chunk.get("metadata", {}).get("hierarchy", []))
        content_preview = chunk.get("content", "")[:80].replace("\n", " ")
        print(f"  [{hierarchy[:50]}]")
        print(f"   '{content_preview}'")
        print()

    # Save clean chunks
    with open(output_path, "w") as f:
        json.dump(clean_chunks, f, indent=2)

    print(f"✅ Clean chunks saved to {output_path}")
    print(f"   Removed: {len(junk_chunks)} junk chunks")
    print(f"   Kept:    {len(clean_chunks)} clean chunks")
    print(f"\nNext step: rebuild embeddings with clean chunks")


if __name__ == "__main__":
    main()