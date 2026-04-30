"""
Run Phase 1: Google Ads Policy Ingestion
"""

from src.ingestion.scrape_policies import GoogleAdsPolicyScraper
from src.ingestion.parse_policies import PolicyParser
from src.ingestion.chunking import PolicyChunker

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def main():
    
    print("=" * 60)
    print("PHASE 1: DATA INGESTION")
    print("=" * 60)
    print()

    # Step 1: Scrape policies
    print("\n📥 Step 1: Scraping Google Ads Policies")
    print("-" * 60)
    pages = GoogleAdsPolicyScraper().run()



    # Step 2: Parse policies
    print("\n📄 Step 2: Parsing Policy Pages")
    print("-" * 60)
    sections = PolicyParser().run()



    # Step 3: Chunking
    print("\n✂️  Step 3: Chunking Policy Sections")
    print("-" * 60)
    chunks = PolicyChunker(
        input_file="data/processed/parsed_sections.json",
        output_file="data/processed/chunks.json"
    ).run()



    print("\n📊 PHASE I: SUMMARY")
    print("-" * 60)
    print()
    print(f"Pages: {len(pages)}")
    print(f"Sections: {len(sections)}")
    print(f"Chunks: {len(chunks)}")
    print("\n✅ Phase 1 complete")


if __name__ == "__main__":
    main()
