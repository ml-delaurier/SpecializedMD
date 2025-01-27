"""
AnnotationProcessor: Handles the processing and integration of professor annotations,
external content, and educational resources.
"""

from typing import Dict, List, Optional, Union
from pathlib import Path
from datetime import datetime
import json

class AnnotationProcessor:
    """Processes and manages educational content annotations and external resources."""

    def __init__(self, external_content_path: str):
        """
        Initialize the AnnotationProcessor.
        
        Args:
            external_content_path (str): Base path for storing external content
        """
        self.external_path = Path(external_content_path)
        self.external_path.mkdir(parents=True, exist_ok=True)

    def create_cutaway_annotation(self,
                                timestamp: float,
                                content: Dict[str, Union[str, Dict]],
                                source_info: Dict[str, str]) -> Dict:
        """
        Create a cut-away annotation with external content reference.
        
        Args:
            timestamp (float): Time in lecture where cut-away occurs
            content (Dict): Cut-away content and explanation
            source_info (Dict): Source information for external content
            
        Returns:
            Dict: Processed annotation data
        """
        annotation = {
            "type": "cutaway",
            "timestamp": timestamp,
            "content": content,
            "source": source_info,
            "created_at": datetime.now().isoformat()
        }
        
        # Validate source information
        self._validate_source_info(source_info)
        
        return annotation

    def create_explanation_annotation(self,
                                   timestamp: float,
                                   explanation: str,
                                   references: List[Dict[str, str]] = None) -> Dict:
        """
        Create an explanation annotation with optional references.
        
        Args:
            timestamp (float): Time in lecture for explanation
            explanation (str): Detailed explanation text
            references (List[Dict]): List of reference sources
            
        Returns:
            Dict: Processed explanation annotation
        """
        annotation = {
            "type": "explanation",
            "timestamp": timestamp,
            "content": {
                "explanation": explanation,
                "references": references or []
            },
            "created_at": datetime.now().isoformat()
        }
        
        return annotation

    def add_external_resource(self,
                            resource_type: str,
                            content: bytes,
                            metadata: Dict) -> str:
        """
        Add external educational resource (video, paper, etc.).
        
        Args:
            resource_type (str): Type of resource (video, pdf, etc.)
            content (bytes): Resource content
            metadata (Dict): Resource metadata
            
        Returns:
            str: Resource identifier
        """
        resource_id = f"resource_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create resource directory
        resource_dir = self.external_path / resource_id
        resource_dir.mkdir(exist_ok=True)
        
        # Save resource content
        extension = self._get_extension(resource_type)
        resource_path = resource_dir / f"content{extension}"
        with open(resource_path, "wb") as f:
            f.write(content)
        
        # Save metadata
        metadata.update({
            "resource_id": resource_id,
            "resource_type": resource_type,
            "created_at": datetime.now().isoformat()
        })
        self._save_metadata(resource_dir / "metadata.json", metadata)
        
        return resource_id

    def link_external_literature(self,
                               lecture_id: str,
                               literature_refs: List[Dict[str, str]]) -> Dict:
        """
        Link external medical literature to a lecture.
        
        Args:
            lecture_id (str): ID of the lecture
            literature_refs (List[Dict]): List of literature references
            
        Returns:
            Dict: Processed literature links
        """
        links = {
            "lecture_id": lecture_id,
            "references": []
        }
        
        for ref in literature_refs:
            processed_ref = self._process_literature_reference(ref)
            links["references"].append(processed_ref)
            
        return links

    def _process_literature_reference(self, ref: Dict[str, str]) -> Dict:
        """Process and validate a literature reference."""
        required_fields = ["title", "authors", "publication", "year", "doi"]
        for field in required_fields:
            if field not in ref:
                raise ValueError(f"Missing required field: {field}")
        
        return {
            **ref,
            "processed_at": datetime.now().isoformat()
        }

    def _validate_source_info(self, source_info: Dict[str, str]) -> None:
        """Validate external content source information."""
        required_fields = ["type", "title", "source"]
        missing_fields = [field for field in required_fields if field not in source_info]
        if missing_fields:
            raise ValueError(f"Missing required source fields: {missing_fields}")

    def _get_extension(self, resource_type: str) -> str:
        """Get file extension for resource type."""
        extensions = {
            "video": ".mp4",
            "pdf": ".pdf",
            "image": ".png",
            "audio": ".mp3"
        }
        return extensions.get(resource_type, "")

    def _save_metadata(self, path: Path, data: Dict) -> None:
        """Save metadata to JSON file."""
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
