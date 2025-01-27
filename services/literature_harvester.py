"""
Literature Harvester: Automatically fetches and processes medical literature
from PubMed related to colorectal surgery.
"""

import os
import json
import boto3
import requests
import PyPDF2
from Bio import Entrez
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urljoin
import logging
import fitz
import re
from typing import Tuple, Any
from ..core.settings import SettingsManager

class LiteratureHarvester:
    """Harvests and processes medical literature from PubMed."""
    
    def __init__(self,
                 output_dir: str):
        """
        Initialize the harvester.
        
        Args:
            output_dir (str): Local directory for data
        """
        self.logger = logging.getLogger(__name__)
        self.settings = SettingsManager()
        
        # Configure Entrez
        email = self.settings.get_api_key('PUBMED_EMAIL')
        if email:
            Entrez.email = email
        else:
            self.logger.warning("No PubMed email found in settings")
        
        api_key = self.settings.get_api_key('PUBMED_API_KEY')
        if api_key:
            Entrez.api_key = api_key
        else:
            self.logger.warning("No PubMed API key found in settings")
        
        # Configure AWS client
        aws_access_key = self.settings.get_api_key('AWS_ACCESS_KEY_ID')
        aws_secret_key = self.settings.get_api_key('AWS_SECRET_ACCESS_KEY')
        if aws_access_key and aws_secret_key:
            self.s3_bucket = self.settings.get_api_key('AWS_S3_BUCKET')
            if self.s3_bucket:
                self.s3_client = boto3.client(
                    's3',
                    aws_access_key_id=aws_access_key,
                    aws_secret_access_key=aws_secret_key
                )
            else:
                self.logger.warning("No AWS S3 bucket found in settings")
                self.s3_client = None
        else:
            self.logger.warning("AWS credentials not found in settings")
            self.s3_client = None
        
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
        Fetch recent colorectal surgery publications.
        
        Args:
            days_back (int): Number of days to look back
            max_results (int): Maximum number of results to fetch
            
        Returns:
            List[Dict]: List of publication metadata
        """
        self.logger.info(f"Fetching publications from last {days_back} days")
        
        # Construct PubMed query
        query = (
            '(colorectal surgery[MeSH Terms]) AND '
            '("randomized controlled trial"[Publication Type] OR '
            '"systematic review"[Publication Type]) AND '
            f'("{(datetime.now() - timedelta(days=days_back)).strftime("%Y/%m/%d")}"'
            f'[Date - Publication] : "{datetime.now().strftime("%Y/%m/%d")}"[Date - Publication])'
        )
        
        # Search PubMed
        handle = Entrez.esearch(
            db="pubmed",
            term=query,
            retmax=max_results,
            sort="pub_date"
        )
        record = Entrez.read(handle)
        handle.close()
        
        if not record["IdList"]:
            self.logger.info("No new publications found")
            return []
        
        # Fetch details for each publication
        handle = Entrez.efetch(
            db="pubmed",
            id=record["IdList"],
            rettype="medline",
            retmode="text"
        )
        records = Entrez.read(handle)
        handle.close()
        
        publications = []
        for article in records["PubmedArticle"]:
            pub_data = self._extract_publication_data(article)
            if pub_data:
                publications.append(pub_data)
        
        return publications

    def _extract_publication_data(self, article: Dict) -> Optional[Dict]:
        """Extract relevant data from PubMed article."""
        try:
            article_data = article["MedlineCitation"]["Article"]
            pmid = str(article["MedlineCitation"]["PMID"])
            
            return {
                "pmid": pmid,
                "title": article_data["ArticleTitle"],
                "authors": [
                    f"{author['LastName']} {author['ForeName']}"
                    for author in article_data.get("AuthorList", [])
                ],
                "journal": article_data["Journal"]["Title"],
                "publication_date": article_data["Journal"]["JournalIssue"]["PubDate"],
                "abstract": article_data.get("Abstract", {}).get("AbstractText", [""])[0],
                "mesh_terms": [
                    mesh["DescriptorName"]
                    for mesh in article["MedlineCitation"].get("MeshHeadingList", [])
                ],
                "doi": next(
                    (id_obj["Value"] for id_obj in article_data.get("ELocationID", [])
                    if id_obj["EIdType"] == "doi"),
                    None
                )
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
            params={"email": Entrez.email}
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
        Extract and structure content from medical literature PDF.
        Uses advanced NLP and medical domain knowledge to identify
        and categorize key information.
        
        Args:
            pdf_path (str): Path to PDF file
        
        Returns:
            Dict: Structured content with sections, figures, references
        """
        try:
            # Extract raw text using PyMuPDF
            doc = fitz.open(pdf_path)
            
            # Initialize content structure
            content = {
                'title': '',
                'abstract': '',
                'sections': [],
                'figures': [],
                'tables': [],
                'references': [],
                'keywords': [],
                'metadata': {}
            }
            
            # Extract document metadata
            content['metadata'] = self._extract_metadata(doc)
            
            # Process each page
            text_blocks = []
            figures = []
            tables = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Extract text blocks with position information
                blocks = page.get_text("dict")["blocks"]
                for block in blocks:
                    if block.get("type") == 0:  # Text block
                        text_blocks.append({
                            'text': block.get("text", ""),
                            'bbox': block.get("bbox"),
                            'page': page_num
                        })
                    elif block.get("type") == 1:  # Image block
                        figures.append({
                            'bbox': block.get("bbox"),
                            'page': page_num
                        })
            
            # Process text content
            content.update(self._process_text_content(text_blocks))
            
            # Extract and process figures
            content['figures'] = self._process_figures(doc, figures)
            
            # Extract and process tables
            content['tables'] = self._process_tables(doc, tables)
            
            # Extract references
            content['references'] = self._extract_references(text_blocks)
            
            return content
            
        except Exception as e:
            self.logger.error(f"Error extracting content from {pdf_path}: {e}")
            raise
        
        finally:
            if 'doc' in locals():
                doc.close()

    def _extract_metadata(self, doc: fitz.Document) -> Dict[str, Any]:
        """
        Extract and process document metadata.
        
        Args:
            doc (fitz.Document): PDF document
        
        Returns:
            Dict: Document metadata
        """
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
                metadata['title'] = meta.get('title', '')
                if 'author' in meta:
                    metadata['authors'] = [a.strip() for a in meta['author'].split(';')]
                metadata['publication_date'] = meta.get('creationDate', '')
                
            # Extract DOI using regex
            first_page = doc[0].get_text()
            doi_match = re.search(r'doi:?\s*(10\.\d{4,}/[-._;()/:\w]+)', first_page, re.I)
            if doi_match:
                metadata['doi'] = doi_match.group(1)
                
            # Extract journal name and keywords from first page
            journal_patterns = [
                r'(published\s+in\s+)(.*?)(\d{4})',
                r'(journal\s+of\s+.*?)(\d{4}|\(|\n)',
            ]
            
            for pattern in journal_patterns:
                match = re.search(pattern, first_page, re.I)
                if match:
                    metadata['journal'] = match.group(2).strip()
                    break
                    
            # Extract keywords
            keywords_section = re.search(r'keywords?:?(.*?)(?:\n\n|\.|$)', 
                                       first_page, 
                                       re.I | re.S)
            if keywords_section:
                keywords = keywords_section.group(1)
                metadata['keywords'] = [k.strip() for k in keywords.split(',')]
                
        except Exception as e:
            self.logger.warning(f"Error extracting metadata: {e}")
            
        return metadata

    def _process_text_content(self, text_blocks: List[Dict]) -> Dict[str, Any]:
        """
        Process and structure text content from PDF blocks.
        
        Args:
            text_blocks (List[Dict]): List of text blocks with position info
        
        Returns:
            Dict: Processed text content
        """
        content = {
            'abstract': '',
            'sections': []
        }
        
        try:
            # Sort blocks by page and position
            text_blocks.sort(key=lambda x: (x['page'], x['bbox'][1]))
            
            current_section = {
                'title': '',
                'content': '',
                'subsections': []
            }
            
            # Process each block
            for block in text_blocks:
                text = block['text'].strip()
                
                # Skip empty blocks
                if not text:
                    continue
                    
                # Detect section headers
                if self._is_section_header(text, block):
                    # Save previous section if it exists
                    if current_section['title'] and current_section['content']:
                        content['sections'].append(current_section)
                        
                    current_section = {
                        'title': text,
                        'content': '',
                        'subsections': []
                    }
                    
                # Detect abstract
                elif 'abstract' in text.lower()[:20] and not content['abstract']:
                    content['abstract'] = text
                    
                # Add to current section
                else:
                    current_section['content'] += text + '\n'
            
            # Add final section
            if current_section['title'] and current_section['content']:
                content['sections'].append(current_section)
                
        except Exception as e:
            self.logger.warning(f"Error processing text content: {e}")
            
        return content

    def _is_section_header(self, text: str, block: Dict) -> bool:
        """
        Determine if a text block is a section header.
        
        Args:
            text (str): Text content
            block (Dict): Text block information
        
        Returns:
            bool: True if section header
        """
        # Common section headers in medical papers
        section_keywords = {
            'introduction', 'methods', 'results', 'discussion',
            'conclusion', 'background', 'materials', 'references'
        }
        
        # Check text properties
        is_short = len(text.split()) <= 5
        is_keyword = text.lower().split()[0] in section_keywords
        
        # Check formatting (assuming headers are often bold/larger)
        font_size = block.get('size', 0)
        is_large = font_size > 11  # Adjust threshold as needed
        
        return (is_short and (is_keyword or is_large))

    def _process_figures(self, doc: fitz.Document, figures: List[Dict]) -> List[Dict]:
        """
        Process and extract figures with captions.
        
        Args:
            doc (fitz.Document): PDF document
            figures (List[Dict]): List of figure blocks
        
        Returns:
            List[Dict]: Processed figures with metadata
        """
        processed_figures = []
        
        try:
            for fig in figures:
                page = doc[fig['page']]
                
                # Extract figure image
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                img_data = pix.tobytes()
                
                # Find caption
                caption = self._find_nearby_text(page, fig['bbox'], 
                                              pattern=r'Figure\s+\d+:?\s*(.*?)(?:\n\n|$)')
                
                processed_figures.append({
                    'page': fig['page'],
                    'caption': caption,
                    'image_data': img_data,
                    'bbox': fig['bbox']
                })
                
        except Exception as e:
            self.logger.warning(f"Error processing figures: {e}")
            
        return processed_figures

    def _process_tables(self, doc: fitz.Document, tables: List[Dict]) -> List[Dict]:
        """
        Extract and process tables from the document.
        
        Args:
            doc (fitz.Document): PDF document
            tables (List[Dict]): List of table blocks
        
        Returns:
            List[Dict]: Processed tables with data
        """
        processed_tables = []
        
        try:
            for table in tables:
                page = doc[table['page']]
                
                # Extract table caption
                caption = self._find_nearby_text(page, table['bbox'],
                                              pattern=r'Table\s+\d+:?\s*(.*?)(?:\n\n|$)')
                
                # Extract table data (simplified)
                table_data = []
                # TODO: Implement table structure extraction
                
                processed_tables.append({
                    'page': table['page'],
                    'caption': caption,
                    'data': table_data,
                    'bbox': table['bbox']
                })
                
        except Exception as e:
            self.logger.warning(f"Error processing tables: {e}")
            
        return processed_tables

    def _find_nearby_text(self, page: fitz.Page, bbox: Tuple[float, float, float, float],
                         pattern: str, max_distance: float = 50) -> str:
        """
        Find text matching pattern near a given bounding box.
        
        Args:
            page (fitz.Page): PDF page
            bbox (Tuple): Bounding box coordinates
            pattern (str): Regex pattern to match
            max_distance (float): Maximum distance to search
        
        Returns:
            str: Found text or empty string
        """
        try:
            # Get text blocks near the bbox
            x0, y0, x1, y1 = bbox
            nearby_text = page.get_text("dict", clip=(x0-max_distance,
                                                    y0-max_distance,
                                                    x1+max_distance,
                                                    y1+max_distance))
            
            # Search for pattern in nearby text
            text = ' '.join(block.get("text", "") for block in nearby_text.get("blocks", []))
            match = re.search(pattern, text, re.S)
            
            return match.group(1) if match else ""
            
        except Exception as e:
            self.logger.warning(f"Error finding nearby text: {e}")
            return ""

    def _extract_references(self, text_blocks: List[Dict]) -> List[Dict]:
        """
        Extract and process references section.
        
        Args:
            text_blocks (List[Dict]): List of text blocks
        
        Returns:
            List[Dict]: Structured references
        """
        references = []
        
        try:
            # Find references section
            ref_section = ''
            in_references = False
            
            for block in text_blocks:
                text = block['text'].lower()
                
                if 'references' in text[:15]:
                    in_references = True
                    continue
                    
                if in_references:
                    ref_section += block['text'] + '\n'
            
            # Parse individual references
            ref_patterns = [
                r'\[\d+\]\s*(.*?)(?=\[\d+\]|\Z)',  # [1] Style
                r'^\d+\.\s*(.*?)(?=^\d+\.|\Z)',    # 1. Style
                r'^[A-Z][a-z]+.*?\(\d{4}\).*?$'    # Author (Year) Style
            ]
            
            for pattern in ref_patterns:
                matches = re.finditer(pattern, ref_section, re.M | re.S)
                if matches:
                    for match in matches:
                        ref_text = match.group(1).strip()
                        references.append(self._parse_reference(ref_text))
                    break
                    
        except Exception as e:
            self.logger.warning(f"Error extracting references: {e}")
            
        return references

    def _parse_reference(self, ref_text: str) -> Dict[str, str]:
        """
        Parse a reference string into structured data.
        
        Args:
            ref_text (str): Reference text
        
        Returns:
            Dict: Structured reference data
        """
        ref_data = {
            'authors': [],
            'year': '',
            'title': '',
            'journal': '',
            'volume': '',
            'pages': '',
            'doi': ''
        }
        
        try:
            # Extract DOI if present
            doi_match = re.search(r'doi:?\s*(10\.\d{4,}/[-._;()/:\w]+)', ref_text, re.I)
            if doi_match:
                ref_data['doi'] = doi_match.group(1)
            
            # Extract year
            year_match = re.search(r'\((\d{4})\)', ref_text)
            if year_match:
                ref_data['year'] = year_match.group(1)
            
            # Extract authors (simplified)
            author_section = ref_text.split('(')[0]
            ref_data['authors'] = [a.strip() for a in author_section.split(',')]
            
            # Extract title (text between year and journal)
            if year_match:
                post_year = ref_text[ref_text.find(')') + 1:]
                title_end = post_year.find('.')
                if title_end > 0:
                    ref_data['title'] = post_year[:title_end].strip()
            
        except Exception as e:
            self.logger.warning(f"Error parsing reference: {e}")
            
        return ref_data

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
