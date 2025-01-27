"""
Configuration UI for managing application settings and API keys.

This module provides a user-friendly interface for managing application settings
and API keys using customtkinter. Features include:
- Secure entry fields for API keys
- Show/hide functionality for sensitive data
- Input validation with visual feedback
- Backup management capabilities
"""
import customtkinter as ctk
from typing import Optional, Callable, Dict
import webbrowser
from pathlib import Path
from .settings_manager import SettingsManager
import re

class ConfigUI:
    """
    Configuration UI for managing application settings and API keys.
    
    Features:
    - Secure entry fields for API keys with show/hide functionality
    - Real-time validation of API key formats
    - Backup management capabilities
    - Visual feedback for validation status
    """
    
    def __init__(self, parent: Optional[ctk.CTk] = None):
        """
        Initialize the configuration UI.
        
        Args:
            parent: Parent window, if None creates a new window
        """
        self.settings = SettingsManager()
        
        # Create window
        if parent is None:
            self.window = ctk.CTk()
            self.window.title("SpecializedMD Settings")
            self.window.geometry("600x800")
        else:
            self.window = ctk.CTkToplevel(parent)
            self.window.title("Settings")
            self.window.geometry("600x800")
        
        # Configure grid
        self.window.grid_columnconfigure(0, weight=1)
        self.window.grid_columnconfigure(1, weight=2)
        
        # Create main frame with tabs
        self.tabview = ctk.CTkTabview(self.window)
        self.tabview.grid(row=0, column=0, columnspan=2, sticky="nsew", padx=20, pady=20)
        
        # Add tabs
        self.api_tab = self.tabview.add("API Keys")
        self.backup_tab = self.tabview.add("Backups")
        
        self._setup_api_tab()
        self._setup_backup_tab()
    
    def _setup_api_tab(self) -> None:
        """Set up the API keys configuration tab."""
        # Configure grid
        self.api_tab.grid_columnconfigure(0, weight=1)
        self.api_tab.grid_columnconfigure(1, weight=2)
        
        # Add title
        self.title = ctk.CTkLabel(
            self.api_tab,
            text="API Key Configuration",
            font=("Helvetica", 20, "bold")
        )
        self.title.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Add description
        self.description = ctk.CTkLabel(
            self.api_tab,
            text="Please enter your API keys for the required services.\n" +
                 "These will be stored securely in your user directory.",
            font=("Helvetica", 12),
            wraplength=500
        )
        self.description.grid(row=1, column=0, columnspan=2, pady=(0, 20))
        
        # Add API key entries
        self.api_entries: Dict[str, ctk.CTkEntry] = {}
        self.validation_labels: Dict[str, ctk.CTkLabel] = {}
        row = 2
        
        for key, description in self.settings.REQUIRED_API_KEYS.items():
            # Label frame for each API key
            frame = ctk.CTkFrame(self.api_tab)
            frame.grid(row=row, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
            frame.grid_columnconfigure(1, weight=1)
            
            # Key label
            label = ctk.CTkLabel(
                frame,
                text=f"{key}:",
                font=("Helvetica", 12, "bold")
            )
            label.grid(row=0, column=0, sticky="w", padx=5, pady=2)
            
            # Entry with validation
            entry = ctk.CTkEntry(
                frame,
                width=300,
                show="•"  # Hide API key
            )
            entry.grid(row=0, column=1, sticky="ew", padx=5, pady=2)
            
            # Show/Hide button
            show_btn = ctk.CTkButton(
                frame,
                text="👁",
                width=30,
                command=lambda e=entry: self.toggle_show_hide(e)
            )
            show_btn.grid(row=0, column=2, sticky="e", padx=5, pady=2)
            
            # Validation label
            validation_label = ctk.CTkLabel(
                frame,
                text="",
                font=("Helvetica", 10),
                text_color="gray"
            )
            validation_label.grid(row=1, column=0, columnspan=3, sticky="w", padx=25)
            
            # Description
            desc_label = ctk.CTkLabel(
                frame,
                text=description,
                font=("Helvetica", 10),
                wraplength=400
            )
            desc_label.grid(row=2, column=0, columnspan=3, sticky="w", padx=25, pady=(0, 5))
            
            # Set existing value if any
            existing_value = self.settings.get_api_key(key)
            if existing_value:
                entry.insert(0, existing_value)
                self._validate_entry(key, entry, validation_label)
            
            # Bind validation
            entry.bind('<KeyRelease>', lambda e, k=key, en=entry, vl=validation_label: 
                      self._validate_entry(k, en, vl))
            
            self.api_entries[key] = entry
            self.validation_labels[key] = validation_label
            
            row += 1
        
        # Add save button
        self.save_btn = ctk.CTkButton(
            self.api_tab,
            text="Save Settings",
            command=self.save_settings,
            font=("Helvetica", 12, "bold")
        )
        self.save_btn.grid(row=row, column=0, columnspan=2, pady=20)
        
        # Add status label
        self.status_label = ctk.CTkLabel(
            self.api_tab,
            text="",
            font=("Helvetica", 12),
            text_color="gray"
        )
        self.status_label.grid(row=row+1, column=0, columnspan=2, pady=5)
    
    def _setup_backup_tab(self) -> None:
        """Set up the backup management tab."""
        # Configure grid
        self.backup_tab.grid_columnconfigure(0, weight=1)
        
        # Add title
        title = ctk.CTkLabel(
            self.backup_tab,
            text="Backup Management",
            font=("Helvetica", 20, "bold")
        )
        title.grid(row=0, column=0, pady=(0, 20))
        
        # Add description
        description = ctk.CTkLabel(
            self.backup_tab,
            text="Manage your settings backups here.\n" +
                 "Backups are created automatically before each settings change.",
            font=("Helvetica", 12),
            wraplength=500
        )
        description.grid(row=1, column=0, pady=(0, 20))
        
        # Create backup list frame
        list_frame = ctk.CTkFrame(self.backup_tab)
        list_frame.grid(row=2, column=0, sticky="nsew", padx=20, pady=10)
        
        # Add refresh button
        refresh_btn = ctk.CTkButton(
            list_frame,
            text="Refresh Backup List",
            command=self._refresh_backup_list
        )
        refresh_btn.grid(row=0, column=0, pady=10, padx=10)
        
        # Add backup list
        self.backup_listbox = ctk.CTkTextbox(
            list_frame,
            height=200,
            width=500
        )
        self.backup_listbox.grid(row=1, column=0, pady=10, padx=10)
        
        # Add restore button
        self.restore_btn = ctk.CTkButton(
            list_frame,
            text="Restore Selected Backup",
            command=self._restore_selected_backup,
            state="disabled"
        )
        self.restore_btn.grid(row=2, column=0, pady=10, padx=10)
        
        # Initialize backup list
        self._refresh_backup_list()
    
    def _validate_entry(self, key: str, entry: ctk.CTkEntry, 
                       validation_label: ctk.CTkLabel) -> None:
        """
        Validate an API key entry field.
        
        Args:
            key: API key name
            entry: Entry widget to validate
            validation_label: Label to show validation status
        """
        value = entry.get().strip()
        if not value:
            validation_label.configure(text="Required", text_color="gray")
            return
        
        if key in self.settings.KEY_VALIDATION_PATTERNS:
            pattern = self.settings.KEY_VALIDATION_PATTERNS[key]
            if re.match(pattern, value):
                validation_label.configure(text="Valid format ✓", text_color="green")
            else:
                validation_label.configure(text="Invalid format ✗", text_color="red")
        else:
            validation_label.configure(text="No format validation", text_color="gray")
    
    def toggle_show_hide(self, entry: ctk.CTkEntry) -> None:
        """
        Toggle showing/hiding an API key.
        
        Args:
            entry: Entry widget to toggle
        """
        if entry.cget("show") == "•":
            entry.configure(show="")
        else:
            entry.configure(show="•")
    
    def save_settings(self) -> None:
        """Save all API keys to settings with validation."""
        try:
            # Validate and save each API key
            for key, entry in self.api_entries.items():
                value = entry.get().strip()
                if value:
                    self.settings.set_api_key(key, value)
            
            # Update status
            self.status_label.configure(
                text="Settings saved successfully!",
                text_color="green"
            )
            
            # Refresh backup list
            self._refresh_backup_list()
            
        except ValueError as e:
            self.status_label.configure(
                text=f"Validation error: {str(e)}",
                text_color="red"
            )
        except Exception as e:
            self.status_label.configure(
                text=f"Error saving settings: {str(e)}",
                text_color="red"
            )
    
    def _refresh_backup_list(self) -> None:
        """Refresh the list of available backups."""
        self.backup_listbox.delete("0.0", "end")
        backups = self.settings.get_backup_files()
        
        if not backups:
            self.backup_listbox.insert("0.0", "No backups available")
            self.restore_btn.configure(state="disabled")
            return
        
        for backup in backups:
            timestamp = backup.stem.replace("settings_backup_", "")
            size = backup.stat().st_size
            self.backup_listbox.insert("end", f"{timestamp} ({size} bytes)\n")
        
        self.restore_btn.configure(state="normal")
    
    def _restore_selected_backup(self) -> None:
        """Restore settings from the selected backup."""
        try:
            selection = self.backup_listbox.get("sel.first", "sel.last")
            if not selection:
                return
            
            timestamp = selection.split()[0]
            backup_file = self.settings.settings_dir / f"settings_backup_{timestamp}.json"
            
            if backup_file.exists():
                self.settings.restore_from_backup(backup_file)
                
                # Refresh UI
                for key in self.api_entries:
                    value = self.settings.get_api_key(key)
                    entry = self.api_entries[key]
                    entry.delete(0, "end")
                    if value:
                        entry.insert(0, value)
                
                self.status_label.configure(
                    text="Settings restored successfully!",
                    text_color="green"
                )
            
        except Exception as e:
            self.status_label.configure(
                text=f"Error restoring backup: {str(e)}",
                text_color="red"
            )
    
    def run(self) -> None:
        """Run the settings UI."""
        self.window.mainloop()

def show_settings(parent: Optional[ctk.CTk] = None) -> None:
    """
    Show the settings configuration UI.
    
    Args:
        parent: Parent window, if any
    """
    config_ui = ConfigUI(parent)
    config_ui.run()
