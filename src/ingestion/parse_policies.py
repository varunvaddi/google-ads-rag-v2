"""
Google Ads Policy Parser V2
Reads from raw_v2/ folder with 4-category structure

WHY THIS STEP?
- HTML files contain raw markup, ads, navigation
- We need clean policy text only
- Extract hierarchical structure (sections/subsections)
- Preserve metadata for citations

INPUT: data/raw/*.html (22 files)
OUTPUT: data/processed/parsed_sections.json (~60-80 sections)
"""

import os
import json
from pathlib import Path
from typing import List, Dict
from bs4 import BeautifulSoup
from datetime import datetime
from tqdm import tqdm


class PolicyParser:
    """
    Parses Google Ads policy HTML files into structured sections
    
    WHAT IT DOES:
    1. Reads HTML files from 4 category folders
    2. Extracts policy text (removes navigation, ads, etc.)
    3. Identifies section hierarchy (H2 → H3 → H4)
    4. Creates structured JSON with metadata
    """
    
    def __init__(
        self, 
        input_dir: str = "data/raw",
        output_dir: str = "data/processed"
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 4 main categories
        self.categories = [
            "prohibited_content",
            "prohibited_practices", 
            "restricted_content",
            "editorial_technical"
        ]
        
        print("✅ Parser V2 initialized")
        print(f"   Input: {self.input_dir}")
        print(f"   Output: {self.output_dir}")
    
    def parse_all_policies(self) -> List[Dict]:
        """
        Parse all HTML files from all categories
        
        PROCESS:
        1. Loop through 4 category folders
        2. Parse each HTML file
        3. Extract sections with hierarchy
        4. Combine into master list
        """
        all_sections = []
        
        print("\n" + "="*80)
        print("PARSING GOOGLE ADS POLICIES V2")
        print("="*80)
        
        for category in self.categories:
            category_path = self.input_dir / category
            
            if not category_path.exists():
                print(f"⚠️  Category not found: {category}")
                continue
            
            # Get all HTML files in this category
            html_files = list(category_path.glob("*.html"))
            
            print(f"\n📂 {category.upper()}: {len(html_files)} files")
            
            for html_file in tqdm(html_files, desc=f"  Parsing"):
                sections = self.parse_single_file(html_file, category)
                all_sections.extend(sections)
                
        print(f"\n✅ Parsing complete!")
        print(f"   Total sections extracted: {len(all_sections)}")
        
        return all_sections
    
    def parse_single_file(self, html_file: Path, category: str) -> List[Dict]:
        """
        Parse a single HTML file into sections
        
        WHY SECTION EXTRACTION?
        - Policy pages have multiple topics (e.g., "Unapproved Pharmaceuticals", "Weight Loss")
        - Each topic should be searchable independently
        - Preserves hierarchy for better context
        
        HIERARCHY STRUCTURE:
        Policy Page: "Healthcare and Medicines"
          ├── H2: "Unapproved pharmaceuticals"
          │   ├── H3: "Examples"
          │   └── H3: "What to do"
          └── H2: "Weight loss products"
              └── H3: "Requirements"
        
        Each H2 becomes a separate section with its H3/H4 content
        """
        
        with open(html_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Extract page title and URL from metadata
        page_title = self._extract_title(soup, html_file)
        url = self._extract_url(soup, html_file)
        
        # Find main content area
        main_content = self._find_main_content(soup)
        
        if not main_content:
            return []
        
        # Extract sections by hierarchy
        sections = self._extract_sections(main_content, page_title, url, category)
        
        return sections
    
    def _extract_title(self, soup: BeautifulSoup, html_file: Path) -> str:
        """Extract page title"""
        # Try <title> tag
        title_tag = soup.find('title')
        if title_tag:
            title = title_tag.get_text(strip=True)
            # Clean Google's title format
            title = title.replace(' - Google Ads Help', '')
            title = title.replace(' - Advertising Policies Help', '')
            return title
        
        # Try H1
        h1 = soup.find('h1')
        if h1:
            return h1.get_text(strip=True)
        
        # Fallback: Use filename
        return html_file.stem.replace('_', ' ').title()
    
    def _extract_url(self, soup: BeautifulSoup, html_file: Path) -> str:
        """Extract original URL from HTML"""
        # Try canonical link
        canonical = soup.find('link', rel='canonical')
        if canonical and canonical.get('href'):
            return canonical['href']
        
        # Try meta og:url
        og_url = soup.find('meta', property='og:url')
        if og_url and og_url.get('content'):
            return og_url['content']
        
        # Fallback: construct from filename
        return f"https://support.google.com/adspolicy/{html_file.stem}"
    
    def _find_main_content(self, soup: BeautifulSoup):
        """
        Find the main content area (policy text)
        
        WHY THIS MATTERS:
        - HTML contains navigation, footer, ads, etc.
        - We only want policy text
        - Google uses <article> or specific divs for content
        """
        # Try <article> tag (most pages)
        article = soup.find('article')
        if article:
            return article
        
        # Try main content div
        main = soup.find('div', {'class': ['article-content', 'main-content']})
        if main:
            return main
        
        # Fallback: use body
        return soup.find('body')
    
    def _extract_sections(
        self, 
        content, 
        page_title: str, 
        url: str, 
        category: str
    ) -> List[Dict]:
        """
        Extract hierarchical sections from content
        
        STRATEGY:
        1. Find all H2 headers (main sections)
        2. For each H2, collect content until next H2
        3. Within content, identify H3/H4 subsections
        4. Create hierarchy: [Category, Page, Section, Subsection]
        
        EXAMPLE OUTPUT:
        {
          "title": "Unapproved pharmaceuticals",
          "hierarchy": ["Restricted Content", "Healthcare", "Unapproved pharmaceuticals"],
          "content": "Google doesn't allow ads for...",
          "url": "https://...",
          "category": "restricted_content"
        }
        """
        sections = []
        
        # Find all H2 headers (main section divisions)
        h2_headers = content.find_all('h2')
        
        if not h2_headers:
            # No sections, treat entire page as one section
            full_text = self._extract_text(content)
            if full_text.strip():
                sections.append({
                    'title': page_title,
                    'hierarchy': [
                        self._format_category_name(category),
                        page_title
                    ],
                    'content': full_text,
                    'url': url,
                    'category': category,
                    'content_type': 'full_page',
                    'char_count': len(full_text)
                })
            return sections
        
        # Process each H2 section
        for i, h2 in enumerate(h2_headers):
            section_title = h2.get_text(strip=True)
            
            # Get content between this H2 and next H2
            section_content = []
            current = h2.find_next_sibling()
            
            # Collect elements until next H2 or end
            while current:
                if current.name == 'h2':
                    break
                section_content.append(current)
                current = current.find_next_sibling()
            
            # Extract text from collected elements
            section_text = self._extract_text_from_elements(section_content)
            
            if not section_text.strip():
                continue
            
            # Build hierarchy
            hierarchy = [
                self._format_category_name(category),
                page_title,
                section_title
            ]
            
            sections.append({
                'title': section_title,
                'hierarchy': hierarchy,
                'content': section_text,
                'url': url,
                'category': category,
                'content_type': 'section',
                'char_count': len(section_text)
            })
        
        return sections
    
    def _extract_text(self, element) -> str:
        """
        Extract clean text from HTML element
        
        WHY CAREFUL TEXT EXTRACTION?
        - Need to preserve structure (paragraphs, lists)
        - Remove navigation/ads/scripts
        - Keep meaningful whitespace
        """
        # Remove script and style elements
        for script in element(['script', 'style', 'nav', 'header', 'footer']):
            script.decompose()
        
        # Get text with some structure preserved
        text = element.get_text(separator='\n', strip=True)
        
        # Clean up excessive whitespace
        lines = [line.strip() for line in text.split('\n')]
        lines = [line for line in lines if line]
        
        return '\n\n'.join(lines)
    
    def _extract_text_from_elements(self, elements: List) -> str:
        """Extract text from list of BeautifulSoup elements"""
        texts = []
        for element in elements:
            text = self._extract_text(element)
            if text:
                texts.append(text)
        return '\n\n'.join(texts)
    
    def _format_category_name(self, category: str) -> str:
        """Convert category slug to display name"""
        return category.replace('_', ' ').title()
    
    def save_sections(self, sections: List[Dict], filename: str = "parsed_sections.json"):
        """Save parsed sections to JSON"""
        output_path = self.output_dir / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(sections, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Saved to: {output_path}")
        
        # Print statistics
        self._print_statistics(sections)
    
    def _print_statistics(self, sections: List[Dict]):
        """Print parsing statistics"""
        from collections import Counter
        
        print("\n" + "="*80)
        print("PARSING STATISTICS")
        print("="*80)
        
        # By category
        by_category = Counter(s['category'] for s in sections)
        print("\n📊 Sections by category:")
        for cat, count in sorted(by_category.items()):
            print(f"   {self._format_category_name(cat)}: {count}")
        
        # Content size
        total_chars = sum(s['char_count'] for s in sections)
        avg_chars = total_chars / len(sections) if sections else 0
        
        print(f"\n📏 Content size:")
        print(f"   Total characters: {total_chars:,}")
        print(f"   Average per section: {avg_chars:,.0f}")
        print(f"   Total sections: {len(sections)}")
        
        # Sample hierarchy
        print(f"\n🌳 Sample hierarchies:")
        for section in sections[:3]:
            hierarchy = ' > '.join(section['hierarchy'])
            print(f"   • {hierarchy}")


def main():
    """Run the parser"""
    parser = PolicyParser()
    
    # Parse all policies
    sections = parser.parse_all_policies()
    
    if not sections:
        print("\n❌ No sections parsed! Check HTML files.")
        return
    
    # Save results
    parser.save_sections(sections)
    
    print("\n✅ Phase A Complete: Parsing")
    print(f"   Next: Run chunking to create embeddings-ready chunks")


if __name__ == "__main__":
    main()