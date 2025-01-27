"""
Simple settings manager that stores settings in a local JSON file.
"""

import json
from pathlib import Path

class SettingsManager:
    """Manages application settings using a local JSON file."""
    
    def __init__(self):
        """Initialize settings manager."""
        self.settings_file = Path(__file__).parent.parent / "settings.json"
        self._load_settings()
    
    def _load_settings(self):
        """Load settings from JSON file."""
        if self.settings_file.exists():
            with open(self.settings_file, 'r') as f:
                self.settings = json.load(f)
        else:
            self.settings = {
                "api_keys": {
                    "deepseek": "",
                    "openai": ""
                },
                "paths": {
                    "content_dir": "content",
                    "transcripts_dir": "transcripts",
                    "media_dir": "media"
                },
                "app": {
                    "theme": "dark",
                    "window_size": "1200x800"
                }
            }
            self._save_settings()
    
    def _save_settings(self):
        """Save settings to JSON file."""
        with open(self.settings_file, 'w') as f:
            json.dump(self.settings, f, indent=4)
    
    def get_setting(self, *keys):
        """Get a setting value using dot notation."""
        value = self.settings
        for key in keys:
            value = value.get(key)
            if value is None:
                return None
        return value
    
    def set_setting(self, value, *keys):
        """Set a setting value using dot notation."""
        target = self.settings
        for key in keys[:-1]:
            target = target.setdefault(key, {})
        target[keys[-1]] = value
        self._save_settings()
    
    def get_api_key(self, service):
        """Get API key for a service."""
        return self.get_setting("api_keys", service)
    
    def set_api_key(self, service, key):
        """Set API key for a service."""
        self.set_setting(key, "api_keys", service)
