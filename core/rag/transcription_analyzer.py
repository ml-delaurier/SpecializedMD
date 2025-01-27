"""
TranscriptionAnalyzer: Analyzes lecture transcriptions to generate enhanced RAG data
with Q&A pairs, key concepts, and semantic relationships for improved retrieval.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime
import logging
from pydantic import BaseModel, Field

from ..llm.deepseek_service import DeepSeekService, MedicalContext

class QAPair(BaseModel):
    """Structure for Question-Answer pairs."""
    question: str = Field(..., description="Generated question")
    answer: str = Field(..., description="Corresponding answer")
    context: str = Field(..., description="Source context from transcription")
    concepts: List[str] = Field(default_factory=list, description="Key medical concepts")
    confidence: float = Field(..., description="Confidence score for the QA pair")
    metadata: Dict = Field(default_factory=dict, description="Additional metadata")

class TranscriptionSegment(BaseModel):
    """Structure for analyzed transcription segments."""
    text: str = Field(..., description="Original transcription text")
    start_time: float = Field(..., description="Start time in seconds")
    end_time: float = Field(..., description="End time in seconds")
    qa_pairs: List[QAPair] = Field(default_factory=list, description="Generated QA pairs")
    key_concepts: List[str] = Field(default_factory=list, description="Key concepts")
    clinical_pearls: List[str] = Field(default_factory=list, description="Clinical pearls")
    references: List[str] = Field(default_factory=list, description="Related references")

class TranscriptionAnalyzer:
    """Analyzes medical lecture transcriptions for enhanced RAG operations."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the transcription analyzer.
        
        Args:
            api_key (str, optional): DeepSeek API key
        """
        self.llm = DeepSeekService(api_key=api_key, model="medical")
        self.logger = logging.getLogger(__name__)
        
    def analyze_transcription(self,
                            transcription_file: Union[str, Path],
                            output_dir: Optional[Union[str, Path]] = None,
                            segment_length: int = 300,  # 5 minutes in seconds
                            min_confidence: float = 0.7) -> Dict:
        """
        Analyze a transcription file and generate enhanced RAG data.
        
        Args:
            transcription_file (Union[str, Path]): Path to transcription JSON
            output_dir (Union[str, Path], optional): Output directory
            segment_length (int): Length of segments in seconds
            min_confidence (float): Minimum confidence for QA pairs
            
        Returns:
            Dict: Analysis results with enhanced RAG data
        """
        transcription_file = Path(transcription_file)
        
        # Load transcription
        with open(transcription_file, "r", encoding="utf-8") as f:
            transcription_data = json.load(f)
        
        # Prepare output directory
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = transcription_file.parent / "enhanced_rag"
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract transcription text and segments from Groq's format
        if isinstance(transcription_data.get("transcription"), dict):
            # New Groq format
            segments = transcription_data["transcription"].get("segments", [])
            text = transcription_data["transcription"].get("text", "")
        else:
            # Legacy format
            segments = transcription_data.get("segments", [])
            text = transcription_data.get("transcription", "")
        
        # Process transcription in segments
        analyzed_segments = []
        for segment in segments:
            try:
                analyzed_segment = self._analyze_segment(
                    segment,
                    min_confidence
                )
                analyzed_segments.append(analyzed_segment)
            except Exception as e:
                self.logger.error(f"Failed to analyze segment: {str(e)}")
        
        # Compile results
        results = {
            "metadata": {
                "source_file": transcription_file.name,
                "analyzed_at": datetime.now().isoformat(),
                "segments_count": len(analyzed_segments),
                "total_qa_pairs": sum(len(s.qa_pairs) for s in analyzed_segments),
                "model": transcription_data.get("metadata", {}).get("model", "unknown")
            },
            "segments": [s.dict() for s in analyzed_segments]
        }
        
        # Save enhanced RAG data
        output_file = output_dir / f"{transcription_file.stem}_enhanced.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
            
        return results
    
    def _segment_transcription(self,
                             text: str,
                             segment_length: int) -> List[Dict]:
        """Split transcription into analyzable segments."""
        # Use DeepSeek to identify logical break points
        context = MedicalContext(
            context_type="procedure",
            educational_level="expert"
        )
        
        segmentation_prompt = (
            "Split this medical lecture transcription into logical segments. "
            "Identify natural break points between topics or concepts. "
            "Maintain context and coherence within each segment."
        )
        
        response = self.llm.process_medical_text(
            text=text,
            context=context,
            task="segment"
        )
        
        # Process segments
        segments = []
        current_time = 0
        
        for segment_text in response["output"].split("\n\n"):
            if not segment_text.strip():
                continue
                
            segments.append({
                "text": segment_text,
                "start_time": current_time,
                "end_time": current_time + segment_length
            })
            current_time += segment_length
        
        return segments
    
    def _analyze_segment(self,
                        segment: Dict,
                        min_confidence: float) -> TranscriptionSegment:
        """Analyze a transcription segment."""
        context = MedicalContext(
            context_type="procedure",
            educational_level="expert"
        )
        
        # Generate QA pairs
        qa_response = self.llm.process_medical_text(
            text=segment["text"],
            context=context,
            task="generate_qa"
        )
        
        qa_pairs = []
        for qa_data in qa_response["output"].split("\n\n"):
            if not qa_data.strip():
                continue
                
            # Parse QA data
            qa_dict = json.loads(qa_data)
            if qa_dict.get("confidence", 0) >= min_confidence:
                qa_pairs.append(QAPair(**qa_dict))
        
        # Extract key concepts
        concepts_response = self.llm.process_medical_text(
            text=segment["text"],
            context=context,
            task="extract_concepts"
        )
        
        # Generate clinical pearls
        pearls_response = self.llm.process_medical_text(
            text=segment["text"],
            context=context,
            task="extract_pearls"
        )
        
        # Find relevant references
        refs_response = self.llm.process_medical_text(
            text=segment["text"],
            context=context,
            task="find_references"
        )
        
        return TranscriptionSegment(
            text=segment["text"],
            start_time=segment["start_time"],
            end_time=segment["end_time"],
            qa_pairs=qa_pairs,
            key_concepts=concepts_response["output"].split("\n"),
            clinical_pearls=pearls_response["output"].split("\n"),
            references=refs_response["output"].split("\n")
        )
    
    def _save_enhanced_data(self,
                          results: Dict,
                          output_dir: Path,
                          base_name: str):
        """Save enhanced RAG data in multiple formats."""
        # Save complete results
        with open(output_dir / f"{base_name}_enhanced.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Save QA pairs in dedicated format
        qa_pairs = []
        for segment in results["segments"]:
            for qa in segment["qa_pairs"]:
                qa_pairs.append({
                    "question": qa["question"],
                    "answer": qa["answer"],
                    "context": qa["context"],
                    "concepts": qa["concepts"],
                    "confidence": qa["confidence"],
                    "timestamp": {
                        "start": segment["start_time"],
                        "end": segment["end_time"]
                    }
                })
        
        with open(output_dir / f"{base_name}_qa_pairs.json", "w", encoding="utf-8") as f:
            json.dump(qa_pairs, f, indent=2, ensure_ascii=False)
        
        # Save concepts index
        concepts_index = {}
        for segment in results["segments"]:
            for concept in segment["key_concepts"]:
                if concept not in concepts_index:
                    concepts_index[concept] = []
                concepts_index[concept].append({
                    "start_time": segment["start_time"],
                    "end_time": segment["end_time"],
                    "context": segment["text"][:200] + "..."  # Preview
                })
        
        with open(output_dir / f"{base_name}_concepts_index.json", "w", encoding="utf-8") as f:
            json.dump(concepts_index, f, indent=2, ensure_ascii=False)
