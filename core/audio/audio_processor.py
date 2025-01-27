"""
AudioProcessor: Handles audio processing and analysis for medical lectures,
focusing on high-quality transcription and speaker diarization.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime
import whisper
from pydub import AudioSegment
import numpy as np

class AudioProcessor:
    """Processes audio content from medical lectures."""
    
    def __init__(self, model_name: str = "base", device: str = "cpu"):
        """
        Initialize the audio processor.
        
        Args:
            model_name (str): Whisper model name (tiny, base, small, medium, large)
            device (str): Device to run model on (cpu/cuda)
        """
        self.logger = logging.getLogger(__name__)
        self.model_name = model_name
        self.device = device
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize Whisper model for transcription."""
        try:
            self.model = whisper.load_model(self.model_name, device=self.device)
            self.logger.info(f"Loaded Whisper model: {self.model_name}")
        except Exception as e:
            self.logger.error(f"Error loading Whisper model: {e}")
            self.model = None
    
    def process_audio(self, audio_path: Union[str, Path], output_dir: Optional[Path] = None) -> Dict:
        """
        Process an audio file for transcription and analysis.
        
        Args:
            audio_path (Union[str, Path]): Path to input audio file
            output_dir (Path, optional): Output directory for processed files
            
        Returns:
            Dict: Processing results and metadata
        """
        audio_path = Path(audio_path)
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Load and preprocess audio
            audio = AudioSegment.from_file(str(audio_path))
            
            # Convert to WAV if needed
            if audio_path.suffix.lower() != ".wav":
                wav_path = output_dir / f"{audio_path.stem}.wav" if output_dir else audio_path.with_suffix(".wav")
                audio.export(wav_path, format="wav")
                audio_path = wav_path
            
            # Transcribe audio
            result = self.transcribe_audio(audio_path)
            
            # Save transcription if output directory provided
            if output_dir and result.get("text"):
                transcript_path = output_dir / f"{audio_path.stem}_transcript.txt"
                with open(transcript_path, "w", encoding="utf-8") as f:
                    f.write(result["text"])
                
                # Save detailed segments
                segments_path = output_dir / f"{audio_path.stem}_segments.json"
                with open(segments_path, "w", encoding="utf-8") as f:
                    json.dump(result.get("segments", []), f, indent=2)
            
            return {
                "status": "success",
                "original_path": str(audio_path),
                "transcript_path": str(transcript_path) if output_dir else None,
                "segments_path": str(segments_path) if output_dir else None,
                "timestamp": datetime.now().isoformat(),
                "metadata": {
                    "duration": len(audio) / 1000,  # Convert to seconds
                    "sample_rate": audio.frame_rate,
                    "channels": audio.channels
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error processing audio {audio_path}: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def transcribe_audio(self, audio_path: Union[str, Path]) -> Dict:
        """
        Transcribe audio using Whisper model.
        
        Args:
            audio_path (Union[str, Path]): Path to input audio file
            
        Returns:
            Dict: Transcription results with text and segments
        """
        if not self.model:
            raise RuntimeError("Whisper model not initialized")
        
        try:
            # Transcribe with Whisper
            result = self.model.transcribe(str(audio_path))
            return result
        except Exception as e:
            self.logger.error(f"Error transcribing audio: {e}")
            return {"error": str(e)}
    
    def detect_speakers(self, audio_path: Union[str, Path]) -> List[Dict]:
        """
        Perform speaker diarization on audio.
        
        Args:
            audio_path (Union[str, Path]): Path to input audio file
            
        Returns:
            List[Dict]: Speaker segments with timestamps
        """
        # TODO: Implement speaker diarization
        self.logger.warning("Speaker diarization not yet implemented")
        return []
    
    def enhance_audio(self, audio_path: Union[str, Path], output_path: Optional[Path] = None) -> Path:
        """
        Enhance audio quality for better transcription.
        
        Args:
            audio_path (Union[str, Path]): Path to input audio file
            output_path (Path, optional): Path for enhanced audio output
            
        Returns:
            Path: Path to enhanced audio file
        """
        audio_path = Path(audio_path)
        if not output_path:
            output_path = audio_path.parent / f"enhanced_{audio_path.name}"
        
        try:
            # Load audio
            audio = AudioSegment.from_file(str(audio_path))
            
            # Basic enhancement: normalize volume
            enhanced = audio.normalize()
            
            # Export enhanced audio
            enhanced.export(str(output_path), format=output_path.suffix[1:])
            
            return output_path
        except Exception as e:
            self.logger.error(f"Error enhancing audio: {e}")
            return audio_path  # Return original path on error
