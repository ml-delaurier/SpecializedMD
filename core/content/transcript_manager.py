"""
TranscriptManager: Handles transcript generation, enrichment, and medical term correction
using Groq's Whisper-large-v3 and UMLS Metathesaurus.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import requests
from concurrent.futures import ThreadPoolExecutor
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag, RegexpParser
from ..llm.deepseek_service import DeepSeekService
from ..audio.transcription_service import TranscriptionService

class TranscriptManager:
    """Manages lecture transcripts with medical term enrichment."""
    
    def __init__(self, 
                 umls_api_key: str,
                 groq_api_key: Optional[str] = None):
        """
        Initialize TranscriptManager.
        
        Args:
            umls_api_key (str): API key for UMLS Metathesaurus
            groq_api_key (str, optional): API key for Groq transcription service
        """
        self.umls_api_key = umls_api_key
        self.transcription_service = TranscriptionService(api_key=groq_api_key)
        self.umls_cache = {}
        self.logger = None  # Initialize logger

    def process_lecture_audio(self, 
                            audio_path: str,
                            lecture_id: str,
                            output_dir: str) -> Dict:
        """
        Generate and enrich transcript from lecture audio.
        
        Args:
            audio_path (str): Path to audio file
            lecture_id (str): Unique lecture identifier
            output_dir (str): Output directory for transcripts
            
        Returns:
            Dict: Processed transcript with metadata
        """
        output_path = Path(output_dir) / lecture_id
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Generate initial transcript using Groq's Whisper-large-v3
        raw_transcript = self._generate_transcript(audio_path)
        
        # Step 2: Enrich with medical terms
        enriched_segments = self._enrich_transcript(raw_transcript)
        
        # Step 3: Align timestamps
        aligned_transcript = self._align_timestamps(enriched_segments)
        
        # Save results
        transcript_data = {
            "lecture_id": lecture_id,
            "segments": aligned_transcript,
            "metadata": {
                "processed_at": datetime.now().isoformat(),
                "audio_file": audio_path,
                "model": "whisper-large-v3"
            }
        }
        
        self._save_transcript(output_path / "transcript.json", transcript_data)
        return transcript_data

    def _generate_transcript(self, audio_path: str) -> Dict:
        """
        Generate initial transcript using Groq's Whisper-large-v3.
        
        Args:
            audio_path (str): Path to audio file
            
        Returns:
            Dict: Raw transcript with timestamps
        """
        # Use Groq's Whisper-large-v3 to transcribe
        result = self.transcription_service.transcribe_file(
            file_path=audio_path,
            language="en",
            timestamp_granularities=["segment", "word"]
        )
        
        return result

    def _enrich_transcript(self, 
                          raw_transcript: Dict,
                          batch_size: int = 5) -> List[Dict]:
        """
        Enrich transcript with medical terminology.
        
        Args:
            raw_transcript (Dict): Raw Whisper transcript
            batch_size (int): Number of segments to process in parallel
            
        Returns:
            List[Dict]: Enriched transcript segments
        """
        enriched_segments = []
        
        # Process segments in parallel
        with ThreadPoolExecutor() as executor:
            segment_batches = [
                raw_transcript["segments"][i:i + batch_size]
                for i in range(0, len(raw_transcript["segments"]), batch_size)
            ]
            
            for batch in segment_batches:
                futures = [
                    executor.submit(self._enrich_segment, segment)
                    for segment in batch
                ]
                
                for future in futures:
                    enriched_segment = future.result()
                    enriched_segments.append(enriched_segment)
        
        return enriched_segments

    def _enrich_segment(self, segment: Dict) -> Dict:
        """
        Enrich a single transcript segment with medical terms.
        
        Args:
            segment (Dict): Transcript segment
            
        Returns:
            Dict: Enriched segment
        """
        text = segment["text"]
        
        # Find medical terms
        medical_terms = self.detect_medical_terms(text)
        
        # Add UMLS concepts
        umls_concepts = []
        for term in medical_terms:
            if term in self.umls_cache:
                concepts = self.umls_cache[term]
            else:
                concepts = self._query_umls(term)
                self.umls_cache[term] = concepts
            umls_concepts.extend(concepts)
        
        return {
            **segment,
            "medical_terms": medical_terms,
            "umls_concepts": umls_concepts
        }

    def detect_medical_terms(self, text: str) -> List[Dict[str, Any]]:
        """
        Detect and classify medical terms in the text using a combination of
        medical terminology databases and context-aware analysis.
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            List[Dict]: List of detected medical terms with metadata
        """
        detected_terms = []
        
        try:
            # Split text into sentences for better context
            sentences = sent_tokenize(text)
            
            for sentence in sentences:
                # Tokenize and POS tag words
                tokens = word_tokenize(sentence)
                pos_tags = pos_tag(tokens)
                
                # Extract noun phrases as potential medical terms
                grammar = r"""
                    MedicalTerm: {<JJ>*<NN.*>+}  # Adjective(s) + Noun(s)
                                {<NN.*><IN><NN.*>}  # Noun + Preposition + Noun
                """
                chunk_parser = RegexpParser(grammar)
                chunks = chunk_parser.parse(pos_tags)
                
                # Process each potential medical term
                for subtree in chunks.subtrees(filter=lambda t: t.label() == 'MedicalTerm'):
                    term = ' '.join([word for word, tag in subtree.leaves()])
                    
                    # Check against medical terminology database
                    term_info = self._lookup_medical_term(term)
                    
                    if term_info:
                        # Get sentence context
                        context = sentence
                        
                        detected_terms.append({
                            'term': term,
                            'category': term_info['category'],
                            'definition': term_info['definition'],
                            'context': context,
                            'position': {
                                'sentence': sentences.index(sentence),
                                'start': sentence.find(term),
                                'end': sentence.find(term) + len(term)
                            },
                            'confidence': term_info['confidence']
                        })
        
        except Exception as e:
            self.logger.error(f"Error in medical term detection: {e}")
        
        return detected_terms

    def _lookup_medical_term(self, term: str) -> Optional[Dict[str, Any]]:
        """
        Look up a potential medical term in medical terminology databases
        and classify it based on context.
        
        Args:
            term (str): Term to look up
            
        Returns:
            Optional[Dict]: Term information if found, None otherwise
        """
        # Initialize medical terminology databases
        medical_categories = {
            'anatomy': ['colon', 'rectum', 'anus', 'intestine', 'mucosa'],
            'procedures': ['colectomy', 'resection', 'anastomosis', 'excision'],
            'instruments': ['forceps', 'retractor', 'stapler', 'scalpel'],
            'pathology': ['tumor', 'polyp', 'inflammation', 'necrosis'],
            'techniques': ['dissection', 'ligation', 'suturing', 'stapling']
        }
        
        term_lower = term.lower()
        
        # Check each category
        for category, terms in medical_categories.items():
            if any(ref_term in term_lower for ref_term in terms):
                # Calculate confidence based on exact match vs partial match
                confidence = 1.0 if term_lower in terms else 0.7
                
                return {
                    'category': category,
                    'definition': self._get_term_definition(term),
                    'confidence': confidence
                }
        
        # Use medical NLP model for unknown terms
        try:
            # Initialize DeepSeek medical model
            llm = DeepSeekService(model="medical")
            
            # Query term classification
            response = llm.process_text(
                f"Classify the medical term '{term}' and provide its definition " +
                "in the context of colorectal surgery."
            )
            
            # Parse response
            if response and isinstance(response, dict):
                return {
                    'category': response.get('category', 'unknown'),
                    'definition': response.get('definition', ''),
                    'confidence': float(response.get('confidence', 0.5))
                }
        
        except Exception as e:
            self.logger.warning(f"Error in medical NLP classification: {e}")
        
        return None

    def _get_term_definition(self, term: str) -> str:
        """
        Get the medical definition of a term.
        
        Args:
            term (str): Term to define
            
        Returns:
            str: Medical definition of the term
        """
        # Basic definitions for common terms
        definitions = {
            'colon': 'The large intestine, part of the digestive system',
            'rectum': 'The final section of the large intestine',
            'anastomosis': 'Surgical connection of two structures or parts',
            'resection': 'Surgical removal of all or part of an organ or structure',
            'polyp': 'Abnormal tissue growth protruding from a mucous membrane'
        }
        
        # Return definition if found
        if term.lower() in definitions:
            return definitions[term.lower()]
        
        # For unknown terms, use medical NLP model
        try:
            llm = DeepSeekService(model="medical")
            response = llm.process_text(
                f"Provide a concise medical definition for the term '{term}' " +
                "in the context of colorectal surgery."
            )
            
            if response and isinstance(response, str):
                return response.strip()
        
        except Exception as e:
            self.logger.warning(f"Error getting term definition: {e}")
        
        return "Definition not available"

    def _query_umls(self, term: str) -> List[Dict]:
        """
        Query UMLS for medical term information.
        
        Args:
            term (str): Medical term to query
            
        Returns:
            List[Dict]: UMLS concepts
        """
        # UMLS API endpoint
        url = "https://uts-ws.nlm.nih.gov/rest/search/current"
        
        headers = {
            "Authorization": f"Bearer {self.umls_api_key}",
            "Content-Type": "application/json"
        }
        
        params = {
            "string": term,
            "searchType": "exact"
        }
        
        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            
            data = response.json()
            return [
                {
                    "cui": result["ui"],
                    "name": result["name"],
                    "semantic_type": result["semanticTypes"][0]
                }
                for result in data["result"]["results"]
            ]
        except Exception as e:
            self.logger.error(f"UMLS query failed for term '{term}': {e}")
            return []

    def _align_timestamps(self, 
                         enriched_segments: List[Dict]) -> List[Dict]:
        """
        Align ASR timestamps with lecture timeline.
        
        Args:
            enriched_segments (List[Dict]): Enriched transcript segments
            
        Returns:
            List[Dict]: Segments with aligned timestamps
        """
        aligned_segments = []
        lecture_start = enriched_segments[0]["start"] if enriched_segments else 0
        
        for segment in enriched_segments:
            # Calculate lecture timeline
            lecture_time = segment["start"] - lecture_start
            
            aligned_segments.append({
                **segment,
                "video_time": segment["start"],
                "lecture_time": lecture_time
            })
        
        return aligned_segments

    def _save_transcript(self, path: Path, data: Dict) -> None:
        """Save transcript data to JSON file."""
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

# Download required NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
