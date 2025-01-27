import customtkinter as ctk
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
import sys
import os

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import components
from services.literature_harvester import LiteratureHarvester
from ui.video_annotator import VideoAnnotator
from scripts.process_lecture_transcriptions import TranscriptionProcessor
from utils.video_processor import VideoProcessor
from core.settings import SettingsManager
from core.rag.transcription_analyzer import TranscriptionAnalyzer
from core.vision.image_processor import ImageProcessor
from core.audio.audio_processor import AudioProcessor
from core.content.content_manager import ContentManager
from core.llm.llm_interface import LLMInterface

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Configure window
        self.title("SpecializedMD")
        self.geometry("1200x800")
        
        # Configure grid
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        
        # Create settings manager
        self.settings = SettingsManager()
        
        # Create main container
        self.main_container = ctk.CTkFrame(self)
        self.main_container.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.main_container.grid_rowconfigure(1, weight=1)
        self.main_container.grid_columnconfigure(0, weight=1)
        
        # Create tabview
        self.tabview = ctk.CTkTabview(self.main_container)
        self.tabview.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        
        # Create tabs
        self.create_literature_tab()
        self.create_video_annotator_tab()
        self.create_settings_tab()
        self.create_transcription_tab()
        self.create_video_processor_tab()
        self.create_rag_tab()
        self.create_vision_tab()
        self.create_audio_tab()
        self.create_content_tab()
        self.create_llm_tab()
        
        # Select first tab
        self.tabview.set("Literature")

    def create_literature_tab(self):
        tab = self.tabview.add("Literature")
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(1, weight=1)
        
        # Controls frame
        controls = ctk.CTkFrame(tab)
        controls.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        
        ctk.CTkLabel(controls, text="Days Back:").pack(side="left", padx=5)
        days_back = ctk.CTkEntry(controls, width=80)
        days_back.insert(0, "7")
        days_back.pack(side="left", padx=5)
        
        ctk.CTkLabel(controls, text="Max Results:").pack(side="left", padx=5)
        max_results = ctk.CTkEntry(controls, width=80)
        max_results.insert(0, "50")
        max_results.pack(side="left", padx=5)
        
        fetch_btn = ctk.CTkButton(
            controls, 
            text="Fetch Publications",
            command=lambda: self.fetch_publications(int(days_back.get()), int(max_results.get()))
        )
        fetch_btn.pack(side="left", padx=5)
        
        # Results frame
        results_frame = ctk.CTkFrame(tab)
        results_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        results_frame.grid_columnconfigure(0, weight=1)
        results_frame.grid_rowconfigure(0, weight=1)
        
        self.literature_text = ctk.CTkTextbox(results_frame)
        self.literature_text.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

    def create_video_annotator_tab(self):
        tab = self.tabview.add("Video Annotator")
        self.video_annotator = VideoAnnotator(tab)
        self.video_annotator.pack(expand=True, fill="both")

    def create_settings_tab(self):
        """Create the settings tab for managing API keys and preferences."""
        settings_tab = self.tabview.add("Settings")
        settings_tab.grid_columnconfigure(0, weight=1)
        
        # Create frame for API keys
        api_frame = ctk.CTkFrame(settings_tab)
        api_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        api_frame.grid_columnconfigure(1, weight=1)
        
        # Add title
        title = ctk.CTkLabel(api_frame, text="API Keys", font=("Arial", 16, "bold"))
        title.grid(row=0, column=0, columnspan=2, pady=(10, 20), sticky="w")
        
        # Add Groq API key input
        groq_label = ctk.CTkLabel(api_frame, text="Groq API Key:")
        groq_label.grid(row=1, column=0, padx=10, pady=5, sticky="w")
        
        self.groq_key_var = tk.StringVar(value=self.settings.get_api_key("groq"))
        groq_entry = ctk.CTkEntry(api_frame, show="•", textvariable=self.groq_key_var, width=300)
        groq_entry.grid(row=1, column=1, padx=10, pady=5, sticky="ew")
        
        # Add show/hide button for Groq key
        self.groq_show_var = tk.BooleanVar(value=False)
        groq_show_btn = ctk.CTkButton(
            api_frame, 
            text="Show", 
            width=60,
            command=lambda: self.toggle_key_visibility(groq_entry, self.groq_show_var)
        )
        groq_show_btn.grid(row=1, column=2, padx=10, pady=5)
        
        # Add DeepSeek API key input
        deepseek_label = ctk.CTkLabel(api_frame, text="DeepSeek API Key:")
        deepseek_label.grid(row=2, column=0, padx=10, pady=5, sticky="w")
        
        self.deepseek_key_var = tk.StringVar(value=self.settings.get_api_key("deepseek"))
        deepseek_entry = ctk.CTkEntry(api_frame, show="•", textvariable=self.deepseek_key_var, width=300)
        deepseek_entry.grid(row=2, column=1, padx=10, pady=5, sticky="ew")
        
        # Add show/hide button for DeepSeek key
        self.deepseek_show_var = tk.BooleanVar(value=False)
        deepseek_show_btn = ctk.CTkButton(
            api_frame, 
            text="Show", 
            width=60,
            command=lambda: self.toggle_key_visibility(deepseek_entry, self.deepseek_show_var)
        )
        deepseek_show_btn.grid(row=2, column=2, padx=10, pady=5)
        
        # Add save button
        save_btn = ctk.CTkButton(
            api_frame,
            text="Save API Keys",
            command=self.save_api_keys
        )
        save_btn.grid(row=3, column=0, columnspan=3, pady=20)

    def create_transcription_tab(self):
        tab = self.tabview.add("Transcription")
        self.transcription_processor = TranscriptionProcessor(tab)
        self.transcription_processor.pack(expand=True, fill="both")

    def create_video_processor_tab(self):
        tab = self.tabview.add("Video Processor")
        self.video_processor = VideoProcessor(tab)
        self.video_processor.pack(expand=True, fill="both")

    def create_rag_tab(self):
        tab = self.tabview.add("RAG")
        self.rag_analyzer = TranscriptionAnalyzer(tab)
        self.rag_analyzer.pack(expand=True, fill="both")

    def create_vision_tab(self):
        tab = self.tabview.add("Vision")
        self.image_processor = ImageProcessor(tab)
        self.image_processor.pack(expand=True, fill="both")

    def create_audio_tab(self):
        tab = self.tabview.add("Audio")
        self.audio_processor = AudioProcessor(tab)
        self.audio_processor.pack(expand=True, fill="both")

    def create_content_tab(self):
        tab = self.tabview.add("Content")
        self.content_manager = ContentManager(tab)
        self.content_manager.pack(expand=True, fill="both")

    def create_llm_tab(self):
        tab = self.tabview.add("LLM")
        self.llm_interface = LLMInterface(tab)
        self.llm_interface.pack(expand=True, fill="both")

    def toggle_key_visibility(self, entry_widget, show_var):
        """Toggle visibility of API key in entry widget."""
        if show_var.get():
            entry_widget.configure(show="")
            show_var.set(False)
        else:
            entry_widget.configure(show="•")
            show_var.set(True)
    
    def save_api_keys(self):
        """Save API keys to settings."""
        try:
            self.settings.set_api_key("groq", self.groq_key_var.get())
            self.settings.set_api_key("deepseek", self.deepseek_key_var.get())
            messagebox.showinfo("Success", "API keys saved successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save API keys: {str(e)}")

    def fetch_publications(self, days_back, max_results):
        try:
            harvester = LiteratureHarvester(output_dir="data/external")
            publications = harvester.fetch_new_publications(
                days_back=days_back,
                max_results=max_results
            )
            
            # Display results
            self.literature_text.delete("1.0", tk.END)
            for pub in publications:
                self.literature_text.insert(tk.END, f"Title: {pub['title']}\n")
                self.literature_text.insert(tk.END, f"Authors: {', '.join(pub['authors'])}\n")
                self.literature_text.insert(tk.END, f"Journal: {pub['journal']}\n")
                self.literature_text.insert(tk.END, f"Abstract: {pub['abstract']}\n")
                self.literature_text.insert(tk.END, "-" * 80 + "\n\n")
            
        except Exception as e:
            messagebox.showerror("Error", str(e))

if __name__ == "__main__":
    app = App()
    app.mainloop()
