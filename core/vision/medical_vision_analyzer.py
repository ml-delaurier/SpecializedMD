"""
MedicalVisionAnalyzer: Provides automated medical image analysis capabilities using
the Moondream2 vision language model. This module helps annotate medical procedures,
identify surgical tools, and generate detailed descriptions of surgical techniques.
"""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import torch
from typing import Dict, List, Optional, Tuple, Union
import logging
from datetime import datetime
import time
import re
import json
import moondream as md
from transformers import AutoModelForCausalLM, AutoTokenizer

class MedicalVisionAnalyzer:
    """
    Analyzes medical images and video frames using Moondream vision language model.
    Provides capabilities for surgical tool detection, procedure step identification,
    and automated medical scene description.
    """
    
    def __init__(self, use_gpu: bool = False, model_path: Optional[str] = None):
        """
        Initialize the medical vision analyzer.
        
        Args:
            use_gpu (bool): Whether to use GPU acceleration
            model_path (str, optional): Path to local Moondream model. If None, will download from HF
        """
        self.logger = logging.getLogger(__name__)
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        
        try:
            if model_path:
                # Use local model with moondream package
                self.logger.info(f"Loading local Moondream model from {model_path}")
                self.model = md.vl(model=model_path)
            else:
                # Use HuggingFace model for GPU support
                self.logger.info(f"Loading Moondream model from HuggingFace")
                model_id = "vikhyatk/moondream2"
                revision = "2024-08-26"
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    trust_remote_code=True,
                    revision=revision,
                    torch_dtype=torch.float16 if use_gpu else torch.float32,
                    device_map={"": self.device} if use_gpu else None
                )
                self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
            
            self.logger.info("Model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading Moondream model: {e}")
            raise

    def detect_surgical_tools(self, image: Image.Image) -> List[Dict]:
        """
        Detect and locate surgical tools in the image.
        
        Args:
            image (Image.Image): Input image
            
        Returns:
            List[Dict]: Detected tools with locations and confidence scores
        """
        tools = []
        
        try:
            # Encode image
            if isinstance(self.model, md.VLModel):
                # Using moondream package
                encoded_image = self.model.encode_image(image)
                response = self.model.query(encoded_image, 
                    "List all surgical tools visible in this image. " +
                    "For each tool, specify your confidence level (0-1).")["answer"]
            else:
                # Using HuggingFace model
                encoded_image = self.model.encode_image(image)
                response = self.model.answer_question(encoded_image,
                    "List all surgical tools visible in this image. " +
                    "For each tool, specify your confidence level (0-1).",
                    self.tokenizer)
            
            # Parse response into structured format
            lines = response.split('\n')
            for line in lines:
                if ':' in line:
                    tool_name, details = line.split(':', 1)
                    tool_info = {
                        'name': tool_name.strip(),
                        'confidence': 0.0,
                        'location': {'x': 0, 'y': 0}
                    }
                    
                    # Extract confidence if provided
                    conf_match = re.search(r'(\d+(?:\.\d+)?)', details)
                    if conf_match:
                        tool_info['confidence'] = float(conf_match.group(1))
                    
                    tools.append(tool_info)
                    
        except Exception as e:
            self.logger.error(f"Error detecting surgical tools: {e}")
        
        return tools

    def identify_procedure_step(self, image: Image.Image) -> Dict:
        """
        Identify the current surgical procedure step.
        
        Args:
            image (Image.Image): Input image
            
        Returns:
            Dict: Identified procedure step and relevant details
        """
        result = {
            'step_name': '',
            'anatomical_structures': [],
            'technique': '',
            'confidence': 0.0,
            'details': ''
        }
        
        try:
            # Prepare prompt for procedure identification
            prompt = """Analyze this colorectal surgery image and identify:
            1. The current step in the surgical procedure
            2. Key anatomical structures visible
            3. The surgical technique being employed
            Be specific to colorectal surgery procedures."""
            
            # Get model response
            if isinstance(self.model, md.VLModel):
                # Using moondream package
                encoded_image = self.model.encode_image(image)
                response = self.model.query(encoded_image, prompt)["answer"]
            else:
                # Using HuggingFace model
                encoded_image = self.model.encode_image(image)
                response = self.model.answer_question(encoded_image, prompt, self.tokenizer)
            
            # Parse response
            lines = response.split('\n')
            current_section = None
            
            for line in lines:
                line = line.strip()
                if line.startswith('1.'):
                    current_section = 'step'
                    result['step_name'] = line[2:].strip()
                elif line.startswith('2.'):
                    current_section = 'anatomy'
                    structures = line[2:].strip().split(',')
                    result['anatomical_structures'] = [s.strip() for s in structures]
                elif line.startswith('3.'):
                    current_section = 'technique'
                    result['technique'] = line[2:].strip()
                elif line:
                    if current_section == 'step':
                        result['details'] += line + ' '
            
            # Clean up details
            result['details'] = result['details'].strip()
            
            # Estimate confidence based on response completeness
            confidence = 0.0
            if result['step_name']: confidence += 0.4
            if result['anatomical_structures']: confidence += 0.3
            if result['technique']: confidence += 0.3
            result['confidence'] = confidence
            
        except Exception as e:
            self.logger.error(f"Error identifying procedure step: {e}")
        
        return result

    def analyze_surgical_frame(self, 
                             frame: Union[np.ndarray, Image.Image],
                             analysis_types: List[str] = ["tools", "procedure", "description"]) -> Dict:
        """
        Perform comprehensive analysis of a surgical video frame.
        
        Args:
            frame (Union[np.ndarray, Image.Image]): Input frame (CV2 array or PIL Image)
            analysis_types (List[str]): Types of analysis to perform
            
        Returns:
            Dict: Analysis results including tools detected, procedure steps, etc.
        """
        # Convert CV2 array to PIL Image if needed
        if isinstance(frame, np.ndarray):
            frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        results = {
            'timestamp': time.time(),
            'tools': [],
            'procedure': {},
            'description': '',
            'warnings': []
        }
        
        try:
            # Perform requested analyses
            if "tools" in analysis_types:
                results['tools'] = self.detect_surgical_tools(frame)
                
            if "procedure" in analysis_types:
                results['procedure'] = self.identify_procedure_step(frame)
                
            if "description" in analysis_types:
                # Generate general scene description
                if isinstance(self.model, md.VLModel):
                    # Using moondream package
                    encoded_image = self.model.encode_image(frame)
                    results['description'] = self.model.query(encoded_image,
                        "Provide a detailed description of this surgical scene, " +
                        "focusing on the ongoing procedure, visible anatomy, and " +
                        "any notable surgical techniques or complications visible.")["answer"]
                else:
                    # Using HuggingFace model
                    encoded_image = self.model.encode_image(frame)
                    results['description'] = self.model.answer_question(encoded_image,
                        "Provide a detailed description of this surgical scene, " +
                        "focusing on the ongoing procedure, visible anatomy, and " +
                        "any notable surgical techniques or complications visible.",
                        self.tokenizer)
                
        except Exception as e:
            self.logger.error(f"Error during frame analysis: {e}")
            results['warnings'].append(str(e))
        
        return results

    def analyze_video_segment(self,
                            video_path: str,
                            start_time: float,
                            duration: float,
                            frame_interval: float = 1.0) -> List[Dict]:
        """
        Analyze a segment of surgical video.
        
        Args:
            video_path (str): Path to video file
            start_time (float): Start time in seconds
            duration (float): Duration to analyze in seconds
            frame_interval (float): Interval between analyzed frames
            
        Returns:
            List[Dict]: Analysis results for each processed frame
        """
        results = []
        
        try:
            # Open video file
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")
            
            # Set start position
            cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
            
            # Calculate frame properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_skip = int(frame_interval * fps)
            total_frames = int(duration * fps)
            
            frame_count = 0
            while frame_count < total_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Process frame at specified intervals
                if frame_count % frame_skip == 0:
                    # Calculate actual timestamp
                    timestamp = start_time + (frame_count / fps)
                    
                    # Analyze frame
                    frame_results = self.analyze_surgical_frame(frame)
                    frame_results['timestamp'] = timestamp
                    
                    results.append(frame_results)
                
                frame_count += 1
                
        except Exception as e:
            self.logger.error(f"Error analyzing video segment: {e}")
            raise
        
        finally:
            if 'cap' in locals():
                cap.release()
        
        return results

    def batch_process_keyframes(self,
                              keyframe_dir: str,
                              output_dir: str) -> Dict[str, List[Dict]]:
        """
        Process all keyframes in a directory.
        
        Args:
            keyframe_dir (str): Directory containing keyframe images
            output_dir (str): Directory to save analysis results
            
        Returns:
            Dict[str, List[Dict]]: Analysis results for each keyframe
        """
        keyframe_dir = Path(keyframe_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        for image_path in keyframe_dir.glob("*.jpg"):
            self.logger.info(f"Processing keyframe: {image_path.name}")
            
            # Load and analyze image
            image = Image.open(image_path)
            analysis = self.analyze_surgical_frame(image)
            
            # Save results
            results[image_path.name] = analysis
            
            # Save individual analysis to file
            output_file = output_dir / f"{image_path.stem}_analysis.json"
            with open(output_file, "w") as f:
                json.dump(analysis, f, indent=2)
        
        return results
