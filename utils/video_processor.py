"""
VideoProcessor: Handles video preprocessing, segmentation, and keyframe extraction
for medical educational content.
"""

import cv2
import numpy as np
import json
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

class VideoProcessor:
    """Processes educational videos for optimal learning experience."""
    
    def __init__(self, output_base_dir: str):
        """
        Initialize VideoProcessor with output directory.
        
        Args:
            output_base_dir (str): Base directory for processed videos
        """
        self.output_dir = Path(output_base_dir)
        self.processed_dir = self.output_dir / "processed"
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def process_lecture_video(self, 
                            input_path: str, 
                            lecture_id: str,
                            segment_length: int = 300) -> Dict:
        """
        Process a lecture video: segment, extract keyframes, add watermarks.
        
        Args:
            input_path (str): Path to input video file
            lecture_id (str): Unique identifier for the lecture
            segment_length (int): Length of each segment in seconds (default: 300)
            
        Returns:
            Dict: Processing metadata including segments and keyframes
        """
        lecture_dir = self.processed_dir / lecture_id
        lecture_dir.mkdir(exist_ok=True)
        
        metadata = {
            "lecture_id": lecture_id,
            "original_file": input_path,
            "segments": [],
            "keyframes": [],
            "processed_at": datetime.now().isoformat()
        }
        
        # Step 1: Segment the video
        segments = self._segment_video(input_path, lecture_dir, segment_length)
        metadata["segments"] = segments
        
        # Step 2: Extract and process keyframes
        keyframes = self._extract_keyframes(input_path, lecture_dir)
        metadata["keyframes"] = keyframes
        
        # Step 3: Add watermarks to segments
        self._add_watermarks(lecture_dir, lecture_id)
        
        # Save metadata
        self._save_metadata(lecture_dir / "processing_metadata.json", metadata)
        
        return metadata

    def _segment_video(self, 
                      input_path: str, 
                      output_dir: Path, 
                      segment_length: int) -> List[Dict]:
        """
        Split video into fixed-length segments using FFmpeg.
        
        Args:
            input_path (str): Input video path
            output_dir (Path): Output directory for segments
            segment_length (int): Segment length in seconds
            
        Returns:
            List[Dict]: List of segment metadata
        """
        segments = []
        
        # FFmpeg command for segmentation
        cmd = [
            "ffmpeg", "-i", input_path,
            "-c", "copy",  # Copy without re-encoding
            "-f", "segment",
            "-segment_time", str(segment_length),
            "-reset_timestamps", "1",
            str(output_dir / "segment_%03d.mp4")
        ]
        
        subprocess.run(cmd, check=True)
        
        # Collect segment metadata
        for segment_file in sorted(output_dir.glob("segment_*.mp4")):
            segment_num = int(segment_file.stem.split("_")[1])
            segments.append({
                "segment_number": segment_num,
                "filename": segment_file.name,
                "start_time": segment_num * segment_length,
                "duration": segment_length
            })
            
        return segments

    def _extract_keyframes(self, 
                         input_path: str, 
                         output_dir: Path,
                         threshold: float = 0.5) -> List[Dict]:
        """
        Extract keyframes focusing on surgical tool detection.
        
        Args:
            input_path (str): Input video path
            output_dir (Path): Output directory for keyframes
            threshold (float): Difference threshold for keyframe detection
            
        Returns:
            List[Dict]: Keyframe metadata
        """
        keyframes = []
        keyframe_dir = output_dir / "keyframes"
        keyframe_dir.mkdir(exist_ok=True)
        
        cap = cv2.VideoCapture(input_path)
        prev_frame = None
        frame_count = 0
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert to grayscale for comparison
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if prev_frame is not None:
                # Calculate frame difference
                diff = cv2.absdiff(gray, prev_frame)
                score = np.mean(diff)
                
                # Detect surgical tools using edge detection
                edges = cv2.Canny(gray, 100, 200)
                tool_score = np.sum(edges) / (edges.shape[0] * edges.shape[1])
                
                # Save frame if significant change or surgical tools detected
                if score > threshold or tool_score > 0.1:
                    timestamp = frame_count / fps
                    keyframe_path = keyframe_dir / f"keyframe_{frame_count:06d}.jpg"
                    cv2.imwrite(str(keyframe_path), frame)
                    
                    keyframes.append({
                        "frame_number": frame_count,
                        "timestamp": timestamp,
                        "filename": keyframe_path.name,
                        "tool_detection_score": float(tool_score)
                    })
            
            prev_frame = gray
            frame_count += 1
            
        cap.release()
        return keyframes

    def _add_watermarks(self, 
                       segment_dir: Path, 
                       lecture_id: str) -> None:
        """
        Add timestamped watermarks to video segments.
        
        Args:
            segment_dir (Path): Directory containing video segments
            lecture_id (str): Lecture identifier for watermark
        """
        for segment_file in segment_dir.glob("segment_*.mp4"):
            output_file = segment_dir / f"watermarked_{segment_file.name}"
            
            # FFmpeg command for adding watermark
            cmd = [
                "ffmpeg", "-i", str(segment_file),
                "-vf", f"drawtext=text='{lecture_id} - %{{pts\\:hms}}':x=10:y=10:"
                       "fontsize=24:fontcolor=white:box=1:boxcolor=black@0.5",
                "-codec:a", "copy",
                str(output_file)
            ]
            
            subprocess.run(cmd, check=True)
            
            # Replace original with watermarked version
            output_file.replace(segment_file)

    def _save_metadata(self, path: Path, data: Dict) -> None:
        """Save metadata to JSON file."""
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
