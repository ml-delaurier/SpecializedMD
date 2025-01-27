"""
DeepSeekService: Handles text-based LLM operations using the DeepSeek Reasoner model.
Provides medical text processing with chain-of-thought reasoning capabilities.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from tenacity import retry, stop_after_attempt, wait_exponential
from openai import OpenAI
from pydantic import BaseModel, Field

class ChatMessage(BaseModel):
    """Structure for chat messages."""
    role: str = Field(..., description="Role of the message sender (system/user/assistant)")
    content: str = Field(..., description="Content of the message")

class MedicalContext(BaseModel):
    """Structure for medical context information."""
    specialty: str = Field(..., description="Medical specialty or domain")
    topic: str = Field(..., description="Specific medical topic or condition")
    context_type: str = Field(..., description="Type of medical context (e.g., diagnosis, treatment, procedure)")
    key_terms: List[str] = Field(default_factory=list, description="Important medical terms")
    references: List[str] = Field(default_factory=list, description="Medical references or sources")
    confidence: float = Field(..., description="Confidence score for the context")

class DeepSeekService:
    """Service for medical text processing using DeepSeek Reasoner model."""
    
    MODEL = "deepseek-reasoner"
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the DeepSeek service using OpenAI client.
        
        Args:
            api_key (str, optional): DeepSeek API key. If not provided, will look for
                                   stored API key in settings.
        """
        from core.settings import SettingsManager
        
        if not api_key:
            settings = SettingsManager()
            api_key = settings.get_api_key("deepseek")
            
        if not api_key:
            raise ValueError("DeepSeek API key not found. Please set it in Settings -> API Keys.")
            
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )
        self.logger = logging.getLogger(__name__)
        
    def process_text(self, 
                    prompt: str,
                    text: str,
                    stream: bool = True) -> Tuple[str, str]:
        """
        Process text with chain-of-thought reasoning.
        
        Args:
            prompt (str): System prompt to guide the model
            text (str): Input text to process
            stream (bool): Whether to use streaming response
            
        Returns:
            Tuple[str, str]: (final_content, reasoning_content)
        """
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": text}
        ]
        
        try:
            # Call DeepSeek Reasoner API
            response = self.client.chat.completions.create(
                model=self.MODEL,
                messages=messages,
                stream=stream
            )
            
            if not stream:
                return (
                    response.choices[0].message.content,
                    response.choices[0].message.reasoning_content
                )
            
            # Process streaming response
            reasoning_content = ""
            content = ""
            
            for chunk in response:
                if chunk.choices[0].delta.reasoning_content:
                    reasoning_content += chunk.choices[0].delta.reasoning_content
                else:
                    content += chunk.choices[0].delta.content
                    
            return content, reasoning_content
            
        except Exception as e:
            self.logger.error(f"Text processing failed: {str(e)}")
            raise
            
    def chat(self, messages: List[Dict[str, str]], stream: bool = True) -> Tuple[str, str]:
        """
        Conduct a chat conversation with reasoning.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            stream: Whether to use streaming response
            
        Returns:
            Tuple[str, str]: (content, reasoning_content)
        """
        try:
            response = self.client.chat.completions.create(
                model=self.MODEL,
                messages=messages,
                stream=stream
            )
            
            if not stream:
                return (
                    response.choices[0].message.content,
                    response.choices[0].message.reasoning_content
                )
            
            reasoning_content = ""
            content = ""
            
            for chunk in response:
                if chunk.choices[0].delta.reasoning_content:
                    reasoning_content += chunk.choices[0].delta.reasoning_content
                else:
                    content += chunk.choices[0].delta.content
                    
            return content, reasoning_content
            
        except Exception as e:
            self.logger.error(f"Chat failed: {str(e)}")
            raise
