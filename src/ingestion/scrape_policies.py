"""
Complete Google Ads Policy Scraper

STRUCTURE UNDERSTANDING:
Level 1: Hub page (topic/1626336) - Lists 25 category pages
Level 2: Category pages (e.g., Alcohol) - Contains 3-10 policy sections each
Level 3: Anchor links within Level 2 (e.g., #alcohol-sale) - NOT separate pages

STRATEGY:
1. Fetch the 25 Level 2 pages
2. Parse ALL sections from each page
3. Each section becomes a searchable chunk

This gives us ~100-150 policy sections with full content.
"""

import os
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


class GoogleAdsPolicyScraper:
    """
    Complete scraper that gets all 25 policy pages with full section content
    """
    
    def __init__(self, output_dir: str = "data/raw"):
        self.output_dir = Path(output_dir)
        self.base_url = "https://support.google.com"
        
        # Create category directories
        self.categories = {
            "prohibited_content": self.output_dir / "prohibited_content",
            "prohibited_practices": self.output_dir / "prohibited_practices",
            "restricted_content": self.output_dir / "restricted_content",
            "editorial_technical": self.output_dir / "editorial_technical"
        }
        
        for category_path in self.categories.values():
            category_path.mkdir(parents=True, exist_ok=True)
        
        print("="*80)
        print("GOOGLE ADS POLICY SCRAPER - COMPLETE VERSION")
        print("="*80)
        print("Strategy: Fetch all 25 category pages with full section content")
        print(f"Output: {self.output_dir}")
    
    def get_all_policy_urls(self) -> Dict[str, List[Dict]]:
        """
        All 25 policy pages manually mapped
        
        These URLs come from: https://support.google.com/adspolicy/topic/1626336
        Each URL is a Level 2 page containing multiple policy sections.
        """
        return {
            "prohibited_content": [
                {"title": "Counterfeit goods", "url": "https://support.google.com/adspolicy/answer/176017"},
                {"title": "Dangerous products or services", "url": "https://support.google.com/adspolicy/answer/6014299"},
                {"title": "Enabling dishonest behavior", "url": "https://support.google.com/adspolicy/answer/6020955"},
                {"title": "Inappropriate content", "url": "https://support.google.com/adspolicy/answer/6015406"},
            ],
            
            "prohibited_practices": [
                {"title": "Abusing the ad network", "url": "https://support.google.com/adspolicy/answer/6020954"},
                {"title": "Data collection and use", "url": "https://support.google.com/adspolicy/answer/6020956"},
                {"title": "Misuse of ad network", "url": "https://support.google.com/adspolicy/answer/6020954"},
                {"title": "Misrepresentation", "url": "https://support.google.com/adspolicy/answer/6020955"},
            ],
            
            "restricted_content": [
                {"title": "Ad protection for children and teens", "url": "http://support.google.com/adspolicy/answer/15416897"},
                {"title": "Sexual content", "url": "https://support.google.com/adspolicy/answer/6023699"},
                {"title": "Alcohol", "url": "https://support.google.com/adspolicy/answer/6012382"},
                {"title": "Copyrights", "url": "https://support.google.com/adspolicy/answer/6018015"},
                {"title": "Gambling and games", "url": "https://support.google.com/adspolicy/answer/6018017"},
                {"title": "Healthcare and medicines", "url": "https://support.google.com/adspolicy/answer/176031"},
                {"title": "Political content", "url": "https://support.google.com/adspolicy/answer/6014595"},
                {"title": "Financial services", "url": "https://support.google.com/adspolicy/answer/2464998"},
                {"title": "Trademarks", "url": "https://support.google.com/adspolicy/answer/6118"},
                {"title": "Legal requirements", "url": "https://support.google.com/adspolicy/answer/6023676"},
                {"title": "Other restricted businesses", "url": "https://support.google.com/adspolicy/answer/6368711"},
                {"title": "Cryptocurrencies", "url": "https://support.google.com/adspolicy/answer/7648803"},
                {"title": "Limited Ad serving", "url": "http://support.google.com/adspolicy/answer/13889491"},
                
            ],
            
            "editorial_technical": [
                {"title": "Editorial and professional requirements", "url": "https://support.google.com/adspolicy/answer/6021546"},
                {"title": "Destination requirements", "url": "https://support.google.com/adspolicy/answer/6368661"},
                {"title": "Technical requirements", "url": "https://support.google.com/adspolicy/answer/176108"},
                {"title": "Ad format requirements", "url": "https://support.google.com/adspolicy/answer/6021630"},
            ]
        }
    
    def scrape_policy_page(self, url: str, title: str) -> tuple:
        """
        Fetch policy page and return HTML + parsed sections
        
        Returns: (html_content, section_count)
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            
            html_content = response.text
            
            # Parse to count sections
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Count H2 sections (main policy sections)
            h2_count = len(soup.find_all('h2'))
            
            # Be respectful to server
            time.sleep(1)
            
            return html_content, h2_count
            
        except requests.RequestException as e:
            print(f"\n   ❌ Error fetching {title}: {e}")
            return None, 0
    
    def scrape_all_policies(self):
        """
        Scrape all 25 policy pages
        """
        policies = self.get_all_policy_urls()
        
        total_policies = sum(len(policies[cat]) for cat in policies)
        print(f"\n📊 Total policy pages to scrape: {total_policies}")
        
        metadata = {
            "total_pages": 0,
            "total_sections": 0,
            "by_category": {},
            "scraped_at": datetime.now().isoformat(),
            "policies": []
        }
        
        # Scrape each category
        for category, policy_list in policies.items():
            category_path = self.categories[category]
            
            print(f"\n📂 {category.upper().replace('_', ' ')}")
            print(f"   Policies: {len(policy_list)}")
            
            category_sections = 0
            success_count = 0
            
            for policy in tqdm(policy_list, desc="  Scraping"):
                title = policy['title']
                url = policy['url']
                
                # Create safe filename
                filename = title.replace('/', '_').replace(' ', '_') + '.html'
                output_path = category_path / filename
                
                # Fetch content
                html_content, section_count = self.scrape_policy_page(url, title)
                
                if html_content:
                    # Save HTML
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(html_content)
                    
                    success_count += 1
                    category_sections += section_count
                    
                    # Add to metadata
                    metadata["policies"].append({
                        "title": title,
                        "url": url,
                        "category": category,
                        "filename": filename,
                        "sections_found": section_count
                    })
            
            metadata["by_category"][category] = {
                "pages": success_count,
                "sections": category_sections
            }
            metadata["total_pages"] += success_count
            metadata["total_sections"] += category_sections
            
            print(f"  ✅ Pages: {success_count}/{len(policy_list)}")
            print(f"  📄 Sections found: {category_sections}")
        
        # Save metadata
        self._save_metadata(metadata)
        
        # Print final summary
        self._print_summary(metadata)
    
    def _save_metadata(self, metadata: Dict):
        """Save scraping metadata"""
        # Full metadata
        metadata_path = self.output_dir / "metadata_final.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        # Summary
        summary = {
            "total_pages": metadata["total_pages"],
            "total_sections": metadata["total_sections"],
            "by_category": metadata["by_category"],
            "scraped_at": metadata["scraped_at"]
        }
        
        summary_path = self.output_dir / "scrape_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
    
    def _print_summary(self, metadata: Dict):
        """Print scraping summary"""
        print("\n" + "="*80)
        print("✅ SCRAPING COMPLETE")
        print("="*80)
        
        print(f"\n📊 Summary:")
        print(f"   Total pages scraped: {metadata['total_pages']}")
        print(f"   Total sections found: {metadata['total_sections']}")
        print(f"   Average sections per page: {metadata['total_sections'] / metadata['total_pages']:.1f}")
        
        print(f"\n📂 By category:")
        for cat, stats in metadata["by_category"].items():
            cat_name = cat.replace('_', ' ').title()
            print(f"   {cat_name}:")
            print(f"      Pages: {stats['pages']}")
            print(f"      Sections: {stats['sections']}")
        
        print(f"\n💾 Data saved to: {self.output_dir}")
        
        # Verification tip
        print(f"\n🔍 Verify content:")
        print(f"   cd {self.output_dir}/restricted_content")
        print(f"   head -50 Alcohol.html | grep '<h2'")
        print(f"   # Should show: Alcohol sale, Alcohol information, etc.")


def main():
    """Run the complete scraper"""
    scraper = GoogleAdsPolicyScraper()
    scraper.scrape_all_policies()
    
    print("\n🎯 Next steps:")
    print("1. Verify: Check a few HTML files for actual policy content")
    print("2. Update parser: Point parse_policies_v2.py to 'raw_final/'")
    print("3. Run: python src/ingestion/parse_policies_v2.py")
    print("4. Run: python src/ingestion/chunking_v2.py")


if __name__ == "__main__":
    main()