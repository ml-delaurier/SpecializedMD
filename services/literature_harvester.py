"""
Literature Harvester: Automatically fetches and processes medical literature
from PubMed related to colorectal surgery using DeepSeek API.
"""

import os
import json
import boto3
import requests
import PyPDF2
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urljoin
import logging
import fitz
import re
from typing import Tuple, Any
from core.settings import SettingsManager
from core.llm.deepseek_api import DeepSeekAPI

class LiteratureHarvester:
    """Harvests and processes medical literature using DeepSeek API."""
    
    def __init__(self, output_dir: str):
        """
        Initialize the harvester.
        
        Args:
            output_dir (str): Local directory for data
        """
        self.logger = logging.getLogger(__name__)
        self.settings = SettingsManager()
        
        # Configure DeepSeek
        deepseek_api_key = self.settings.get_api_key('DEEPSEEK_API_KEY')
        if deepseek_api_key:
            self.deepseek = DeepSeekAPI(api_key=deepseek_api_key)
        else:
            self.logger.warning("No DeepSeek API key found in settings")
            self.deepseek = None
        
        self.output_dir = Path(output_dir)
        self.pdf_dir = self.output_dir / "pdfs"
        self.mapping_file = self.output_dir / "pmid_mapping.json"
        self._setup_directories()
        self._setup_logging()

    def _setup_directories(self) -> None:
        """Create necessary directories."""
        self.pdf_dir.mkdir(parents=True, exist_ok=True)

    def _setup_logging(self) -> None:
        """Configure logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / "harvester.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("LiteratureHarvester")

    def fetch_new_publications(self,
                             days_back: int = 7,
                             max_results: int = 50) -> List[Dict]:
        """
        Fetch recent colorectal surgery publications using DeepSeek.
        
        Args:
            days_back (int): Number of days to look back
            max_results (int): Maximum number of results to fetch
            
        Returns:
            List[Dict]: List of publication metadata
        """
        if not self.deepseek:
            self.logger.error("DeepSeek API not configured")
            return []

        self.logger.info(f"Fetching publications from last {days_back} days")
        
        # Construct search query for DeepSeek
        query = {
            "topic": "colorectal surgery",
            "publication_types": ["randomized controlled trial", "systematic review"],
            "date_range": {
                "start": (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d"),
                "end": datetime.now().strftime("%Y-%m-%d")
            },
            "max_results": max_results
        }
        
        try:
            # Search using DeepSeek
            results = self.deepseek.search_medical_literature(query)
            
            publications = []
            for result in results:
                pub_data = self._extract_publication_data(result)
                if pub_data:
                    publications.append(pub_data)
            
            return publications
            
        except Exception as e:
            self.logger.error(f"Error fetching publications: {e}")
            return []

    def _extract_publication_data(self, article: Dict) -> Optional[Dict]:
        """Extract relevant data from DeepSeek article response."""
        try:
            return {
                "pmid": article.get("pmid", ""),
                "title": article.get("title", ""),
                "authors": article.get("authors", []),
                "journal": article.get("journal", ""),
                "publication_date": article.get("publication_date", {}),
                "abstract": article.get("abstract", ""),
                "mesh_terms": article.get("mesh_terms", []),
                "doi": article.get("doi")
            }
        except KeyError as e:
            self.logger.error(f"Error extracting data from article: {e}")
            return None

    def download_and_process_pdfs(self, publications: List[Dict]) -> None:
        """
        Download and process PDFs for publications.
        
        Args:
            publications (List[Dict]): List of publication metadata
        """
        s3 = self.s3_client
        mapping = self._load_mapping()
        
        for pub in publications:
            pmid = pub["pmid"]
            if pmid in mapping:
                self.logger.info(f"PMID {pmid} already processed, skipping")
                continue
                
            try:
                # Download PDF using Unpaywall or similar service
                pdf_url = self._get_pdf_url(pub["doi"])
                if not pdf_url:
                    continue
                    
                pdf_path = self.pdf_dir / f"pmid_{pmid}.pdf"
                self._download_pdf(pdf_url, pdf_path)
                
                # Extract text and data
                extracted_data = self.extract_content(str(pdf_path))
                
                # Upload to S3
                s3_key = f"pdfs/pmid_{pmid}.pdf"
                s3.upload_file(str(pdf_path), self.s3_bucket, s3_key)
                
                # Update mapping
                mapping[pmid] = {
                    "metadata": pub,
                    "s3_key": s3_key,
                    "extracted_data": extracted_data,
                    "processed_at": datetime.now().isoformat()
                }
                
                self._save_mapping(mapping)
                
            except Exception as e:
                self.logger.error(f"Error processing PMID {pmid}: {e}")

    def _get_pdf_url(self, doi: Optional[str]) -> Optional[str]:
        """Get PDF URL using Unpaywall API."""
        if not doi:
            return None
            
        response = requests.get(
            f"https://api.unpaywall.org/v2/{doi}",
            params={"email": "your_email@example.com"}
        )
        
        if response.status_code == 200:
            data = response.json()
            best_oa_location = data.get("best_oa_location", {})
            return best_oa_location.get("pdf_url")
        
        return None

    def _download_pdf(self, url: str, path: Path) -> None:
        """Download PDF from URL."""
        response = requests.get(url)
        response.raise_for_status()
        
        with open(path, "wb") as f:
            f.write(response.content)

    def extract_content(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract and structure content from medical literature PDF using DeepSeek.
        
        Args:
            pdf_path (str): Path to PDF file
        
        Returns:
            Dict: Structured content with sections, figures, references
        """
        if not self.deepseek:
            self.logger.error("DeepSeek API not configured")
            return {}

        try:
            # Extract text from PDF
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            
            # Use DeepSeek to analyze the content
            analysis = self.deepseek.analyze_medical_document({
                "text": text,
                "document_type": "medical_research",
                "analysis_types": [
                    "sections",
                    "figures",
                    "tables",
                    "references",
                    "key_findings"
                ]
            })
            
            return {
                'title': analysis.get('title', ''),
                'abstract': analysis.get('abstract', ''),
                'sections': analysis.get('sections', []),
                'figures': analysis.get('figures', []),
                'tables': analysis.get('tables', []),
                'references': analysis.get('references', []),
                'keywords': analysis.get('keywords', []),
                'key_findings': analysis.get('key_findings', []),
                'metadata': self._extract_metadata(doc)
            }
            
        except Exception as e:
            self.logger.error(f"Error extracting content from {pdf_path}: {e}")
            raise
        
        finally:
            if 'doc' in locals():
                doc.close()

    def _extract_metadata(self, doc: fitz.Document) -> Dict[str, Any]:
        """Extract metadata from PDF document."""
        metadata = {
            'title': '',
            'authors': [],
            'publication_date': None,
            'journal': '',
            'doi': '',
            'keywords': []
        }
        
        try:
            # Extract basic metadata
            meta = doc.metadata
            if meta:
                metadata.update({
                    'title': meta.get('title', ''),
                    'authors': [a.strip() for a in meta.get('author', '').split(';')],
                    'publication_date': meta.get('creationDate', '')
                })
                
            # Use DeepSeek to extract additional metadata
            if self.deepseek:
                first_page = doc[0].get_text()
                enhanced_metadata = self.deepseek.extract_document_metadata({
                    "text": first_page,
                    "document_type": "medical_research"
                })
                metadata.update(enhanced_metadata)
                
        except Exception as e:
            self.logger.warning(f"Error extracting metadata: {e}")
            
        return metadata

    def _load_mapping(self) -> Dict:
        """Load PMID mapping from file."""
        if self.mapping_file.exists():
            with open(self.mapping_file) as f:
                return json.load(f)
        return {}

    def _save_mapping(self, mapping: Dict) -> None:
        """Save PMID mapping to file."""
        with open(self.mapping_file, "w") as f:
            json.dump(mapping, f, indent=2)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Harvest medical literature")
    parser.add_argument("--initial-fetch", action="store_true",
                       help="Perform initial fetch of literature")
    parser.add_argument("--days-back", type=int, default=7,
                       help="Number of days to look back")
    parser.add_argument("--output-dir", type=str, default="data/external",
                       help="Output directory for data")
    args = parser.parse_args()
    
    harvester = LiteratureHarvester(output_dir=args.output_dir)
    
    if args.initial_fetch:
        # For initial fetch, look back further
        publications = harvester.fetch_new_publications(days_back=30, max_results=100)
    else:
        publications = harvester.fetch_new_publications(
            days_back=args.days_back,
            max_results=50
        )
    
    harvester.download_and_process_pdfs(publications)
