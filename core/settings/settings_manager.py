"""
Settings manager for handling application configuration and API keys.
"""
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging
from datetime import datetime

class SettingsManager:
    """
    Manages application settings and API keys, with secure storage and validation.
    """
    
    # Define required API keys and their descriptions
    REQUIRED_API_KEYS = {
        'ENTREZ_API_KEY': 'NCBI Entrez API key for accessing medical literature',
        'GROQ_API_KEY': 'Groq API key for AI model access',
        'AWS_ACCESS_KEY_ID': 'AWS access key ID for S3 storage',
        'AWS_SECRET_ACCESS_KEY': 'AWS secret access key for S3 storage',
        'DEEPSEEK_API_KEY': 'DeepSeek API key for medical AI models',
        'UMLS_API_KEY': 'UMLS API key for medical terminology access'
    }
    
    def __init__(self):
        """Initialize settings manager and load existing settings."""
        self.logger = logging.getLogger(__name__)
        self.settings_dir = Path.home() / '.specializedmd'
        self.settings_file = self.settings_dir / 'settings.json'
        self.settings: Dict[str, Any] = {}
        
        # Create settings directory if it doesn't exist
        self.settings_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing settings
        self.load_settings()
    
    def load_settings(self) -> None:
        """Load settings from file."""
        try:
            if self.settings_file.exists():
                with open(self.settings_file, 'r') as f:
                    self.settings = json.load(f)
                self.logger.info("Settings loaded successfully")
            else:
                self.logger.info("No existing settings file found")
        except Exception as e:
            self.logger.error(f"Error loading settings: {e}")
            self.settings = {}
    
    def save_settings(self) -> None:
        """Save settings to file."""
        try:
            # Create backup of existing settings
            if self.settings_file.exists():
                backup_file = self.settings_dir / f'settings_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
                with open(self.settings_file, 'r') as src, open(backup_file, 'w') as dst:
                    dst.write(src.read())
            
            # Save new settings
            with open(self.settings_file, 'w') as f:
                json.dump(self.settings, f, indent=4)
            self.logger.info("Settings saved successfully")
        except Exception as e:
            self.logger.error(f"Error saving settings: {e}")
            raise
    
    def get_api_key(self, key_name: str) -> Optional[str]:
        """
        Get an API key from settings.
        
        Args:
            key_name: Name of the API key to retrieve
            
        Returns:
            The API key if found, None otherwise
        """
        return self.settings.get('api_keys', {}).get(key_name)
    
    def set_api_key(self, key_name: str, value: str) -> None:
        """
        Set an API key in settings.
        
        Args:
            key_name: Name of the API key to set
            value: Value of the API key
        """
        if 'api_keys' not in self.settings:
            self.settings['api_keys'] = {}
        self.settings['api_keys'][key_name] = value
        self.save_settings()
    
    def validate_api_keys(self) -> Dict[str, bool]:
        """
        Check which required API keys are set.
        
        Returns:
            Dictionary mapping API key names to boolean indicating if they're set
        """
        return {
            key: bool(self.get_api_key(key))
            for key in self.REQUIRED_API_KEYS
        }
    
    def get_missing_api_keys(self) -> Dict[str, str]:
        """
        Get list of missing API keys and their descriptions.
        
        Returns:
            Dictionary of missing API key names and their descriptions
        """
        return {
            key: desc
            for key, desc in self.REQUIRED_API_KEYS.items()
            if not self.get_api_key(key)
        }
    
    def clear_api_key(self, key_name: str) -> None:
        """
        Remove an API key from settings.
        
        Args:
            key_name: Name of the API key to remove
        """
        if 'api_keys' in self.settings and key_name in self.settings['api_keys']:
            del self.settings['api_keys'][key_name]
            self.save_settings()
    
    def clear_all_settings(self) -> None:
        """Clear all settings and save empty configuration."""
        self.settings = {}
        self.save_settings()
