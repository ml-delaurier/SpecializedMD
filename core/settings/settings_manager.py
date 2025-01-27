"""
Settings manager for handling application configuration and API keys.

This module provides a centralized way to manage application settings and API keys.
It handles:
- Secure storage of API keys in the user's home directory
- Validation of required API keys
- Automatic backup of settings before changes
- Structured access to configuration parameters
"""
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime
import hashlib
import re

class SettingsManager:
    """
    Manages application settings and API keys, with secure storage and validation.
    
    This class provides a robust interface for managing application settings, with a focus
    on secure handling of API keys and critical configuration parameters. It includes:
    - Automatic backup creation before any settings changes
    - Validation of API key presence and format
    - Secure storage in the user's home directory
    - Structured access to nested configuration parameters
    """
    
    # Define required API keys and their descriptions
    REQUIRED_API_KEYS = {
        'PUBMED_EMAIL': 'Email address for PubMed/NCBI API access',
        'PUBMED_API_KEY': 'NCBI/PubMed API key for accessing medical literature',
        'GROQ_API_KEY': 'Groq API key for AI model access',
        'AWS_ACCESS_KEY_ID': 'AWS access key ID for S3 storage',
        'AWS_SECRET_ACCESS_KEY': 'AWS secret access key for S3 storage',
        'AWS_S3_BUCKET': 'AWS S3 bucket name for storing documents',
        'DEEPSEEK_API_KEY': 'DeepSeek API key for medical AI models',
        'UMLS_API_KEY': 'UMLS API key for medical terminology access'
    }
    
    # Define key format validation patterns
    KEY_VALIDATION_PATTERNS = {
        'AWS_ACCESS_KEY_ID': r'^[A-Z0-9]{20}$',
        'AWS_SECRET_ACCESS_KEY': r'^[A-Za-z0-9+/]{40}$',
        'PUBMED_EMAIL': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    }
    
    def __init__(self, settings_dir: Optional[str] = None):
        """
        Initialize settings manager and load existing settings.
        
        Args:
            settings_dir: Optional custom directory for settings storage.
                        If None, uses ~/.specializedmd
        """
        self.logger = logging.getLogger(__name__)
        
        # Set up settings directory
        if settings_dir:
            self.settings_dir = Path(settings_dir)
        else:
            self.settings_dir = Path.home() / '.specializedmd'
        
        self.settings_file = self.settings_dir / 'settings.json'
        self.settings: Dict[str, Any] = {}
        
        # Create settings directory if it doesn't exist
        self.settings_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing settings
        self.load_settings()
    
    def load_settings(self) -> None:
        """
        Load settings from file with error handling and logging.
        
        Raises:
            json.JSONDecodeError: If settings file is corrupted
            PermissionError: If unable to read settings file
        """
        try:
            if self.settings_file.exists():
                with open(self.settings_file, 'r') as f:
                    self.settings = json.load(f)
                self.logger.info("Settings loaded successfully")
            else:
                self.logger.info("No existing settings file found")
        except json.JSONDecodeError as e:
            self.logger.error(f"Error decoding settings file: {e}")
            self._handle_corrupted_settings()
        except PermissionError as e:
            self.logger.error(f"Permission denied accessing settings file: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error loading settings: {e}")
            self.settings = {}
    
    def _handle_corrupted_settings(self) -> None:
        """
        Handle corrupted settings file by creating backup and resetting settings.
        """
        try:
            # Create backup of corrupted file
            backup_file = self.settings_dir / f'settings_corrupted_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            self.settings_file.rename(backup_file)
            self.logger.info(f"Corrupted settings backed up to {backup_file}")
            
            # Reset settings
            self.settings = {}
            self.save_settings()
        except Exception as e:
            self.logger.error(f"Error handling corrupted settings: {e}")
    
    def save_settings(self) -> None:
        """
        Save settings to file with backup creation.
        
        Creates a backup of the existing settings file before saving new settings.
        
        Raises:
            PermissionError: If unable to write to settings file
            OSError: If unable to create backup
        """
        try:
            # Create backup of existing settings
            if self.settings_file.exists():
                backup_file = self.settings_dir / f'settings_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
                with open(self.settings_file, 'r') as src, open(backup_file, 'w') as dst:
                    dst.write(src.read())
                self.logger.info(f"Settings backup created at {backup_file}")
            
            # Save new settings
            with open(self.settings_file, 'w') as f:
                json.dump(self.settings, f, indent=4)
            self.logger.info("Settings saved successfully")
        except Exception as e:
            self.logger.error(f"Error saving settings: {e}")
            raise
    
    def get_api_key(self, key_name: str) -> Optional[str]:
        """
        Get an API key from settings with validation.
        
        Args:
            key_name: Name of the API key to retrieve
            
        Returns:
            The API key if found and valid, None otherwise
        
        Raises:
            ValueError: If key_name is not in REQUIRED_API_KEYS
        """
        if key_name not in self.REQUIRED_API_KEYS:
            raise ValueError(f"Unknown API key: {key_name}")
        
        value = self.settings.get('api_keys', {}).get(key_name)
        if value and not self._validate_api_key_format(key_name, value):
            self.logger.warning(f"Invalid format for {key_name}")
            return None
        return value
    
    def _validate_api_key_format(self, key_name: str, value: str) -> bool:
        """
        Validate the format of an API key.
        
        Args:
            key_name: Name of the API key to validate
            value: Value to validate
            
        Returns:
            True if valid, False otherwise
        """
        if key_name in self.KEY_VALIDATION_PATTERNS:
            pattern = self.KEY_VALIDATION_PATTERNS[key_name]
            if not re.match(pattern, value):
                return False
        return True
    
    def set_api_key(self, key_name: str, value: str) -> None:
        """
        Set an API key in settings with validation.
        
        Args:
            key_name: Name of the API key to set
            value: Value of the API key
            
        Raises:
            ValueError: If key_name is not in REQUIRED_API_KEYS or value format is invalid
        """
        if key_name not in self.REQUIRED_API_KEYS:
            raise ValueError(f"Unknown API key: {key_name}")
        
        if not self._validate_api_key_format(key_name, value):
            raise ValueError(f"Invalid format for {key_name}")
        
        if 'api_keys' not in self.settings:
            self.settings['api_keys'] = {}
        
        # Hash sensitive keys for logging
        log_value = self._hash_sensitive_value(value) if 'SECRET' in key_name else value
        self.logger.info(f"Setting {key_name} to {log_value}")
        
        self.settings['api_keys'][key_name] = value
        self.save_settings()
    
    def _hash_sensitive_value(self, value: str) -> str:
        """
        Create a short hash of a sensitive value for logging.
        
        Args:
            value: Value to hash
            
        Returns:
            First 8 characters of SHA-256 hash
        """
        return hashlib.sha256(value.encode()).hexdigest()[:8]
    
    def validate_api_keys(self) -> Dict[str, bool]:
        """
        Check which required API keys are set and valid.
        
        Returns:
            Dictionary mapping API key names to boolean indicating if they're set and valid
        """
        return {
            key: bool(self.get_api_key(key))
            for key in self.REQUIRED_API_KEYS
        }
    
    def get_missing_api_keys(self) -> Dict[str, str]:
        """
        Get list of missing or invalid API keys and their descriptions.
        
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
            
        Raises:
            ValueError: If key_name is not in REQUIRED_API_KEYS
        """
        if key_name not in self.REQUIRED_API_KEYS:
            raise ValueError(f"Unknown API key: {key_name}")
        
        if 'api_keys' in self.settings and key_name in self.settings['api_keys']:
            del self.settings['api_keys'][key_name]
            self.save_settings()
    
    def clear_all_settings(self) -> None:
        """
        Clear all settings and save empty configuration.
        Creates a backup before clearing.
        """
        self.logger.warning("Clearing all settings")
        self.settings = {}
        self.save_settings()
    
    def get_backup_files(self) -> List[Path]:
        """
        Get list of available settings backup files.
        
        Returns:
            List of backup file paths, sorted by creation time (newest first)
        """
        backup_files = list(self.settings_dir.glob('settings_backup_*.json'))
        return sorted(backup_files, key=lambda x: x.stat().st_mtime, reverse=True)
    
    def restore_from_backup(self, backup_file: Path) -> None:
        """
        Restore settings from a backup file.
        
        Args:
            backup_file: Path to backup file to restore from
            
        Raises:
            FileNotFoundError: If backup file doesn't exist
            json.JSONDecodeError: If backup file is corrupted
        """
        if not backup_file.exists():
            raise FileNotFoundError(f"Backup file not found: {backup_file}")
        
        try:
            with open(backup_file, 'r') as f:
                restored_settings = json.load(f)
            
            # Create backup of current settings before restore
            self.save_settings()
            
            # Restore from backup
            self.settings = restored_settings
            self.save_settings()
            self.logger.info(f"Settings restored from {backup_file}")
        except Exception as e:
            self.logger.error(f"Error restoring from backup: {e}")
            raise
