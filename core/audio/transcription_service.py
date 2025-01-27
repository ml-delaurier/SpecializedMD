
"""
TranscriptionService: Handles audio transcription using the Groq API.
Supports various audio formats and provides options for customizing transcription parameters.
"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Union
import logging
from openai import OpenAI
from pydub import AudioSegment

class TranscriptionService:
    """Service for transcribing audio files using Groq API."""
    
    SUPPORTED_FORMATS = {
        "flac", "mp3", "mp4", "mpeg", "mpga", "m4a", "ogg", "wav", "webm"
    }
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the transcription service.
        
        Args:
            api_key (str, optional): Groq API key. If not provided, will look for
                                   stored API key in settings.
        """
        from core.settings import SettingsManager
        
        if not api_key:
            settings = SettingsManager()
            api_key = settings.get_api_key("groq")
            
        if not api_key:
            raise ValueError("Groq API key not found. Please set it in Settings -> API Keys.")
            
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1"
        )
        self.logger = logging.getLogger(__name__)
        
    def transcribe_file(self,
                       file_path: Union[str, Path],
                       output_dir: Optional[Union[str, Path]] = None,
                       language: str = "en",
                       prompt: Optional[str] = None,
                       temperature: float = 0.0,
                       timestamp_granularities: Optional[list] = None) -> Dict:
        """
        Transcribe an audio file and save results.
        
        Args:
            file_path (Union[str, Path]): Path to audio file
            output_dir (Union[str, Path], optional): Directory to save transcription
            language (str): ISO-639-1 language code
            prompt (str, optional): Text to guide transcription
            temperature (float): Sampling temperature (0-1)
            timestamp_granularities (list, optional): Timestamp detail level
            
        Returns:
            Dict: Transcription results
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")
            
        # Validate and convert audio format if needed
        file_format = file_path.suffix.lower()[1:]
        if file_format not in self.SUPPORTED_FORMATS:
            self.logger.info(f"Converting {file_format} to wav format")
            file_path = self._convert_audio_format(file_path)
        
        # Prepare output directory
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = file_path.parent
            
        # Prepare transcription parameters
        timestamp_granularities = timestamp_granularities or ["segment"]
        response_format = "verbose_json" if timestamp_granularities else "json"
        
        try:
            # Perform transcription
            with open(file_path, "rb") as audio_file:
                transcription = self.client.audio.transcriptions.create(
                    file=(str(file_path), audio_file.read()),
                    model="whisper-large-v3",
                    prompt=prompt,
                    response_format=response_format,
                    language=language,
                    temperature=temperature,
                    timestamp_granularities=timestamp_granularities
                )
            
            # Prepare results
            results = {
                "metadata": {
                    "file_name": file_path.name,
                    "file_size": file_path.stat().st_size,
                    "transcribed_at": datetime.now().isoformat(),
                    "language": language,
                    "model": "whisper-large-v3",
                    "parameters": {
                        "temperature": temperature,
                        "timestamp_granularities": timestamp_granularities,
                        "prompt": prompt
                    }
                },
                "transcription": transcription.text,
                "x_groq": transcription.x_groq
            }
            
            # Save results
            output_file = output_dir / f"{file_path.stem}_transcription.json"
            self._save_transcription(results, output_file)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Transcription failed: {str(e)}")
            raise
            
    def transcribe_directory(self,
                           input_dir: Union[str, Path],
                           output_dir: Optional[Union[str, Path]] = None,
                           **kwargs) -> Dict[str, Dict]:
        """
        Transcribe all supported audio files in a directory.
        
        Args:
            input_dir (Union[str, Path]): Input directory
            output_dir (Union[str, Path], optional): Output directory
            **kwargs: Additional parameters for transcribe_file
            
        Returns:
            Dict[str, Dict]: Mapping of filenames to transcription results
        """
        input_dir = Path(input_dir)
        if not input_dir.is_dir():
            raise NotADirectoryError(f"Input directory not found: {input_dir}")
            
        output_dir = Path(output_dir) if output_dir else input_dir / "transcriptions"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {}
        for file_path in input_dir.iterdir():
            if file_path.suffix.lower()[1:] in self.SUPPORTED_FORMATS:
                try:
                    self.logger.info(f"Transcribing {file_path.name}")
                    result = self.transcribe_file(
                        file_path,
                        output_dir=output_dir,
                        **kwargs
                    )
                    results[file_path.name] = result
                except Exception as e:
                    self.logger.error(f"Failed to transcribe {file_path.name}: {str(e)}")
                    results[file_path.name] = {"error": str(e)}
        
        # Save batch results
        batch_results = {
            "metadata": {
                "processed_at": datetime.now().isoformat(),
                "input_directory": str(input_dir),
                "output_directory": str(output_dir),
                "files_processed": len(results),
                "parameters": kwargs
            },
            "results": results
        }
        
        batch_output = output_dir / "batch_transcription_results.json"
        self._save_transcription(batch_results, batch_output)
        
        return results
    
    def _convert_audio_format(self, file_path: Path) -> Path:
        """Convert audio to a supported format."""
        try:
            audio = AudioSegment.from_file(file_path)
            output_path = file_path.with_suffix(".wav")
            audio.export(output_path, format="wav")
            return output_path
        except Exception as e:
            self.logger.error(f"Audio conversion failed: {str(e)}")
            raise
    
    def _save_transcription(self, results: Dict, output_file: Path):
        """Save transcription results to JSON file."""
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Saved transcription to {output_file}")
        except Exception as e:
            self.logger.error(f"Failed to save transcription: {str(e)}")
            raise
