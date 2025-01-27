"""
Configuration UI for managing application settings and API keys.
"""
import customtkinter as ctk
from typing import Optional, Callable
import webbrowser
from .settings_manager import SettingsManager

class ConfigUI:
    """
    Configuration UI for managing application settings and API keys.
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
            self.window.geometry("600x700")
        else:
            self.window = ctk.CTkToplevel(parent)
            self.window.title("Settings")
            self.window.geometry("600x700")
        
        # Configure grid
        self.window.grid_columnconfigure(0, weight=1)
        self.window.grid_columnconfigure(1, weight=2)
        
        # Create scrollable frame
        self.scroll_frame = ctk.CTkScrollableFrame(self.window)
        self.scroll_frame.grid(row=0, column=0, columnspan=2, sticky="nsew", padx=20, pady=20)
        
        # Add title
        self.title = ctk.CTkLabel(
            self.scroll_frame,
            text="API Key Configuration",
            font=("Helvetica", 20, "bold")
        )
        self.title.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Add description
        self.description = ctk.CTkLabel(
            self.scroll_frame,
            text="Please enter your API keys for the required services.\n" +
                 "These will be stored securely in your user directory.",
            font=("Helvetica", 12),
            wraplength=500
        )
        self.description.grid(row=1, column=0, columnspan=2, pady=(0, 20))
        
        # Add API key entries
        self.api_entries = {}
        row = 2
        
        for key, description in self.settings.REQUIRED_API_KEYS.items():
            # Label
            label = ctk.CTkLabel(
                self.scroll_frame,
                text=f"{key}:",
                font=("Helvetica", 12, "bold")
            )
            label.grid(row=row, column=0, sticky="w", padx=5, pady=2)
            
            # Description
            desc_label = ctk.CTkLabel(
                self.scroll_frame,
                text=description,
                font=("Helvetica", 10),
                wraplength=400
            )
            desc_label.grid(row=row+1, column=0, columnspan=2, sticky="w", padx=25, pady=(0, 10))
            
            # Entry
            entry = ctk.CTkEntry(
                self.scroll_frame,
                width=300,
                show="•"  # Hide API key
            )
            entry.grid(row=row, column=1, sticky="ew", padx=5, pady=2)
            
            # Set existing value if any
            existing_value = self.settings.get_api_key(key)
            if existing_value:
                entry.insert(0, existing_value)
            
            self.api_entries[key] = entry
            
            # Show/Hide button
            show_btn = ctk.CTkButton(
                self.scroll_frame,
                text="👁",
                width=30,
                command=lambda e=entry: self.toggle_show_hide(e)
            )
            show_btn.grid(row=row, column=1, sticky="e", padx=5, pady=2)
            
            row += 3
        
        # Add save button
        self.save_btn = ctk.CTkButton(
            self.scroll_frame,
            text="Save Settings",
            command=self.save_settings,
            font=("Helvetica", 12, "bold")
        )
        self.save_btn.grid(row=row, column=0, columnspan=2, pady=20)
        
        # Add status label
        self.status_label = ctk.CTkLabel(
            self.scroll_frame,
            text="",
            font=("Helvetica", 12),
            text_color="green"
        )
        self.status_label.grid(row=row+1, column=0, columnspan=2, pady=5)
    
    def toggle_show_hide(self, entry: ctk.CTkEntry) -> None:
        """Toggle showing/hiding an API key."""
        if entry.cget("show") == "•":
            entry.configure(show="")
        else:
            entry.configure(show="•")
    
    def save_settings(self) -> None:
        """Save all API keys to settings."""
        try:
            # Save each API key
            for key, entry in self.api_entries.items():
                value = entry.get().strip()
                if value:
                    self.settings.set_api_key(key, value)
            
            # Update status
            self.status_label.configure(
                text="Settings saved successfully!",
                text_color="green"
            )
        except Exception as e:
            self.status_label.configure(
                text=f"Error saving settings: {str(e)}",
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
