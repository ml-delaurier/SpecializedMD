"""
ContentManager: Handles the processing and management of educational content including videos,
audio, transcripts, and annotations.
"""

import os
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

class ContentManager:
    """Manages educational content including videos, transcripts, and annotations."""
    
    def __init__(self, base_path: str):
        """
        Initialize the ContentManager with a base path for content storage.
        
        Args:
            base_path (str): Base directory path for storing all content
        """
        self.base_path = Path(base_path)
        self.lectures_path = self.base_path / "lectures"
        self.annotations_path = self.base_path / "annotations"
        self._create_directories()

    def _create_directories(self) -> None:
        """Create necessary directory structure if it doesn't exist."""
        for path in [self.lectures_path, self.annotations_path]:
            path.mkdir(parents=True, exist_ok=True)

    def store_lecture(self, 
                     video_file: bytes,
                     metadata: Dict,
                     lecture_id: Optional[str] = None) -> str:
        """
        Store a new lecture video with associated metadata.
        
        Args:
            video_file (bytes): Raw video file data
            metadata (Dict): Lecture metadata including title, description, etc.
            lecture_id (Optional[str]): Custom lecture ID, generated if not provided
            
        Returns:
            str: Unique identifier for the stored lecture
        """
        # Generate unique lecture ID if not provided
        if not lecture_id:
            lecture_id = f"lecture_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create lecture directory
        lecture_dir = self.lectures_path / lecture_id
        lecture_dir.mkdir(exist_ok=True)
        
        # Save video file
        video_path = lecture_dir / "raw_video.mp4"
        with open(video_path, "wb") as f:
            f.write(video_file)
            
        # Store metadata
        self._save_metadata(lecture_dir, metadata)
        
        return lecture_id

    def add_annotation(self,
                      lecture_id: str,
                      timestamp: float,
                      annotation_type: str,
                      content: Dict) -> str:
        """
        Add an annotation to a specific point in a lecture.
        
        Args:
            lecture_id (str): ID of the lecture to annotate
            timestamp (float): Time in seconds where annotation occurs
            annotation_type (str): Type of annotation (cutaway, explanation, etc.)
            content (Dict): Annotation content and metadata
            
        Returns:
            str: Unique identifier for the annotation
        """
        annotation_id = f"annot_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create annotation directory if needed
        lecture_annot_dir = self.annotations_path / lecture_id
        lecture_annot_dir.mkdir(exist_ok=True)
        
        # Store annotation data
        annotation_data = {
            "id": annotation_id,
            "timestamp": timestamp,
            "type": annotation_type,
            "content": content,
            "created_at": datetime.now().isoformat()
        }
        
        annotation_path = lecture_annot_dir / f"{annotation_id}.json"
        self._save_metadata(annotation_path, annotation_data)
        
        return annotation_id

    def get_lecture_with_annotations(self, lecture_id: str) -> Dict:
        """
        Retrieve a lecture and all its associated annotations.
        
        Args:
            lecture_id (str): ID of the lecture to retrieve
            
        Returns:
            Dict: Lecture data including video path and annotations
        """
        lecture_dir = self.lectures_path / lecture_id
        if not lecture_dir.exists():
            raise ValueError(f"Lecture {lecture_id} not found")
            
        # Get lecture metadata
        metadata = self._load_metadata(lecture_dir / "metadata.json")
        
        # Get annotations
        annotations = []
        annot_dir = self.annotations_path / lecture_id
        if annot_dir.exists():
            for annot_file in annot_dir.glob("*.json"):
                annotation = self._load_metadata(annot_file)
                annotations.append(annotation)
                
        return {
            "lecture_id": lecture_id,
            "metadata": metadata,
            "video_path": str(lecture_dir / "raw_video.mp4"),
            "annotations": sorted(annotations, key=lambda x: x["timestamp"])
        }

    def _save_metadata(self, path: Path, data: Dict) -> None:
        """Save metadata to JSON file."""
        import json
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def _load_metadata(self, path: Path) -> Dict:
        """Load metadata from JSON file."""
        import json
        with open(path) as f:
            return json.load(f)
