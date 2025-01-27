"""
DeepSeekService: Handles text-based LLM operations using the DeepSeek API.
Provides specialized medical text processing, analysis, and generation capabilities.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union
from tenacity import retry, stop_after_attempt, wait_exponential
import deepseek
from pydantic import BaseModel, Field

class ChatMessage(BaseModel):
    """Structure for chat messages."""
    role: str = Field(..., description="Role of the message sender (system/user/assistant)")
    content: str = Field(..., description="Content of the message")

class MedicalContext(BaseModel):
    """Structure for medical context information."""
    specialty: str = Field(default="colorectal_surgery", description="Medical specialty")
    procedure_type: Optional[str] = Field(None, description="Specific procedure type")
    context_type: str = Field(..., description="Type of context (diagnosis/procedure/followup)")
    educational_level: str = Field(default="expert", description="Target educational level")

class DeepSeekService:
    """Service for medical text processing using DeepSeek API."""
    
    # Available DeepSeek models
    MODELS = {
        "chat": "deepseek-chat",
        "coder": "deepseek-coder",
        "medical": "deepseek-medical"  # Specialized medical model
    }
    
    def __init__(self, api_key: Optional[str] = None, model: str = "medical"):
        """
        Initialize the DeepSeek service.
        
        Args:
            api_key (str, optional): DeepSeek API key
            model (str): Model to use (chat/coder/medical)
        """
        self.client = deepseek.Client(api_key=api_key)
        self.model = self.MODELS.get(model, self.MODELS["medical"])
        self.logger = logging.getLogger(__name__)
        
        # Load medical prompts
        self.prompts = self._load_medical_prompts()
        
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def process_medical_text(self,
                           text: str,
                           context: MedicalContext,
                           task: str,
                           temperature: float = 0.3) -> Dict:
        """
        Process medical text with specific context and task.
        
        Args:
            text (str): Input medical text
            context (MedicalContext): Medical context information
            task (str): Processing task (analyze/summarize/enhance)
            temperature (float): Model temperature
            
        Returns:
            Dict: Processed results with metadata
        """
        # Get task-specific prompt
        prompt_template = self.prompts.get(task, {}).get(context.context_type)
        if not prompt_template:
            raise ValueError(f"No prompt found for task '{task}' and context '{context.context_type}'")
            
        # Prepare messages
        messages = [
            ChatMessage(
                role="system",
                content=prompt_template.format(
                    specialty=context.specialty,
                    educational_level=context.educational_level
                )
            ),
            ChatMessage(role="user", content=text)
        ]
        
        try:
            # Call DeepSeek API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[msg.dict() for msg in messages],
                temperature=temperature,
                max_tokens=2000
            )
            
            # Process response
            result = {
                "input": {
                    "text": text,
                    "context": context.dict(),
                    "task": task
                },
                "output": response.choices[0].message.content,
                "metadata": {
                    "model": self.model,
                    "temperature": temperature,
                    "processed_at": datetime.now().isoformat(),
                    "tokens": {
                        "prompt": response.usage.prompt_tokens,
                        "completion": response.usage.completion_tokens,
                        "total": response.usage.total_tokens
                    }
                }
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Text processing failed: {str(e)}")
            raise
            
    def analyze_procedure_description(self,
                                   description: str,
                                   procedure_type: str,
                                   educational_level: str = "expert") -> Dict:
        """
        Analyze a surgical procedure description.
        
        Args:
            description (str): Procedure description
            procedure_type (str): Type of procedure
            educational_level (str): Target educational level
            
        Returns:
            Dict: Analysis results
        """
        context = MedicalContext(
            specialty="colorectal_surgery",
            procedure_type=procedure_type,
            context_type="procedure",
            educational_level=educational_level
        )
        
        return self.process_medical_text(
            text=description,
            context=context,
            task="analyze"
        )
        
    def enhance_medical_content(self,
                              content: str,
                              context_type: str,
                              educational_level: str = "expert") -> Dict:
        """
        Enhance medical educational content.
        
        Args:
            content (str): Medical content
            context_type (str): Type of content
            educational_level (str): Target educational level
            
        Returns:
            Dict: Enhanced content
        """
        context = MedicalContext(
            context_type=context_type,
            educational_level=educational_level
        )
        
        return self.process_medical_text(
            text=content,
            context=context,
            task="enhance"
        )
        
    def generate_educational_notes(self,
                                 procedure_data: Dict,
                                 educational_level: str = "expert") -> Dict:
        """
        Generate educational notes from procedure data.
        
        Args:
            procedure_data (Dict): Procedure information
            educational_level (str): Target educational level
            
        Returns:
            Dict: Generated educational notes
        """
        # Convert procedure data to text
        content = json.dumps(procedure_data, indent=2)
        
        context = MedicalContext(
            context_type="procedure",
            educational_level=educational_level
        )
        
        return self.process_medical_text(
            text=content,
            context=context,
            task="generate_notes"
        )
        
    def _load_medical_prompts(self) -> Dict:
        """Load specialized medical prompts."""
        return {
            "analyze": {
                "procedure": (
                    "As a colorectal surgery expert, analyze this {specialty} "
                    "procedure description for a {educational_level} audience. "
                    "Focus on technical precision, safety considerations, and "
                    "educational value."
                ),
                "diagnosis": (
                    "Review this {specialty} diagnosis information for a "
                    "{educational_level} audience. Highlight key findings, "
                    "differential diagnoses, and clinical implications."
                )
            },
            "enhance": {
                "procedure": (
                    "Enhance this {specialty} procedure description for a "
                    "{educational_level} audience. Add relevant anatomical "
                    "details, technical tips, and evidence-based context."
                ),
                "followup": (
                    "Enhance this {specialty} follow-up information for a "
                    "{educational_level} audience. Include post-operative "
                    "care guidelines and complication monitoring."
                )
            },
            "generate_notes": {
                "procedure": (
                    "Generate comprehensive educational notes for this "
                    "{specialty} procedure targeting a {educational_level} "
                    "audience. Include key learning points, technical steps, "
                    "and clinical pearls."
                )
            },
            "segment": {
                "procedure": (
                    "Split this medical lecture transcription into logical segments. "
                    "Each segment should cover a complete concept or technique. "
                    "Identify natural break points between topics while maintaining "
                    "context and coherence within each segment."
                )
            },
            "generate_qa": {
                "procedure": (
                    "Generate high-quality question-answer pairs from this medical "
                    "lecture segment. Each pair should test important concepts, "
                    "techniques, or decision-making points. Format as JSON with "
                    "fields: question, answer, context (relevant text), concepts "
                    "(list of key terms), and confidence (0-1 score)."
                )
            },
            "extract_concepts": {
                "procedure": (
                    "Extract key medical concepts, anatomical terms, surgical "
                    "techniques, and important terminology from this text. "
                    "Return as a list with one concept per line."
                )
            },
            "extract_pearls": {
                "procedure": (
                    "Identify clinical pearls, expert tips, and critical insights "
                    "from this surgical lecture segment. Focus on practical "
                    "wisdom that enhances surgical technique and decision-making."
                )
            },
            "find_references": {
                "procedure": (
                    "Identify relevant medical literature, guidelines, or evidence "
                    "that supports the concepts discussed in this segment. "
                    "Format as a list of citations."
                )
            }
        }
        
    def save_results(self, results: Dict, output_file: Union[str, Path]):
        """Save processing results to file."""
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Saved results to {output_file}")
        except Exception as e:
            self.logger.error(f"Failed to save results: {str(e)}")
            raise
