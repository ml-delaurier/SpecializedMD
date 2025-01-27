"""
DeepSeekAPI: A wrapper around DeepSeek's API using the OpenAI client.
Provides specialized medical literature and document analysis capabilities.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from openai import OpenAI
import logging

class DeepSeekAPI:
    """API wrapper for DeepSeek's medical literature and document analysis capabilities."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the DeepSeek API client.
        
        Args:
            api_key (str, optional): DeepSeek API key
        """
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )
        self.logger = logging.getLogger(__name__)
        
    def search_medical_literature(self, query: Dict) -> List[Dict]:
        """
        Search for medical literature using DeepSeek's reasoner model.
        
        Args:
            query (Dict): Search parameters including topic, date range, etc.
            
        Returns:
            List[Dict]: List of publication metadata
        """
        try:
            prompt = f"""Search for medical literature with the following criteria:
            - Topic: {query.get('topic', '')}
            - Publication Types: {', '.join(query.get('publication_types', []))}
            - Date Range: {query.get('date_range', {}).get('start')} to {query.get('date_range', {}).get('end')}
            
            Return results as a list of publications with metadata including:
            - PMID
            - Title
            - Authors
            - Journal
            - Publication Date
            - Abstract
            - MeSH Terms
            - DOI
            """
            
            response = self.client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[{"role": "user", "content": prompt}],
                stream=True
            )
            
            # Process streaming response
            reasoning_content = ""
            content = ""
            
            for chunk in response:
                if chunk.choices[0].delta.reasoning_content:
                    reasoning_content += chunk.choices[0].delta.reasoning_content
                else:
                    content += chunk.choices[0].delta.content
            
            # Parse the content as a list of publications
            # Note: In a real implementation, this would need more robust parsing
            publications = []  # Parse content into structured data
            return publications
            
        except Exception as e:
            self.logger.error(f"Error searching medical literature: {e}")
            return []
            
    def analyze_medical_document(self, params: Dict) -> Dict[str, Any]:
        """
        Analyze medical document content using DeepSeek's reasoner model.
        
        Args:
            params (Dict): Analysis parameters including text and analysis types
            
        Returns:
            Dict: Analysis results including sections, figures, references, etc.
        """
        try:
            text = params.get('text', '')
            doc_type = params.get('document_type', '')
            analysis_types = params.get('analysis_types', [])
            
            prompt = f"""Analyze this {doc_type} document and extract the following components:
            {', '.join(analysis_types)}
            
            Document text:
            {text[:1000]}...  # Truncated for prompt length
            
            Provide a structured analysis with these components clearly separated.
            """
            
            response = self.client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[{"role": "user", "content": prompt}],
                stream=True
            )
            
            # Process streaming response
            reasoning_content = ""
            content = ""
            
            for chunk in response:
                if chunk.choices[0].delta.reasoning_content:
                    reasoning_content += chunk.choices[0].delta.reasoning_content
                else:
                    content += chunk.choices[0].delta.content
            
            # Parse the content into structured analysis results
            # Note: In a real implementation, this would need more robust parsing
            analysis_results = {}  # Parse content into structured data
            return analysis_results
            
        except Exception as e:
            self.logger.error(f"Error analyzing medical document: {e}")
            return {}
            
    def extract_document_metadata(self, params: Dict) -> Dict[str, Any]:
        """
        Extract metadata from document text using DeepSeek's reasoner model.
        
        Args:
            params (Dict): Parameters including text and document type
            
        Returns:
            Dict: Extracted metadata
        """
        try:
            text = params.get('text', '')
            doc_type = params.get('document_type', '')
            
            prompt = f"""Extract metadata from this {doc_type} document text:
            {text[:1000]}...  # Truncated for prompt length
            
            Include:
            - Title
            - Authors
            - Journal
            - Publication Date
            - DOI
            - Keywords
            """
            
            response = self.client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[{"role": "user", "content": prompt}],
                stream=True
            )
            
            # Process streaming response
            reasoning_content = ""
            content = ""
            
            for chunk in response:
                if chunk.choices[0].delta.reasoning_content:
                    reasoning_content += chunk.choices[0].delta.reasoning_content
                else:
                    content += chunk.choices[0].delta.content
            
            # Parse the content into metadata
            # Note: In a real implementation, this would need more robust parsing
            metadata = {}  # Parse content into structured data
            return metadata
            
        except Exception as e:
            self.logger.error(f"Error extracting document metadata: {e}")
            return {}
