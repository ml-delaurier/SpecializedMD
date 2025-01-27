"""
VideoAnnotator: A customtkinter-based UI for annotating medical education videos
with timestamps, cut-aways, and medical references.
"""

import customtkinter as ctk
import tkinter as tk
from tkinter import ttk
import vlc
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import threading
import time
from PIL import Image, ImageTk
import cv2
import numpy as np

from core.vision.medical_vision_analyzer import MedicalVisionAnalyzer

class VideoPlayer(ctk.CTkFrame):
    """Custom video player widget using VLC."""
    
    def __init__(self, master, video_path: str, **kwargs):
        """Initialize video player."""
        super().__init__(master, **kwargs)
        
        # Initialize VLC
        self.instance = vlc.Instance()
        self.player = self.instance.media_player_new()
        self.media = self.instance.media_new(video_path)
        self.player.set_media(self.media)
        
        # Create video frame
        self.video_frame = ctk.CTkFrame(self)
        self.video_frame.pack(expand=True, fill="both")
        
        # Create controls frame
        self.controls = ctk.CTkFrame(self)
        self.controls.pack(fill="x", pady=5)
        
        # Play/Pause button
        self.play_button = ctk.CTkButton(
            self.controls,
            text="Play",
            command=self.toggle_play
        )
        self.play_button.pack(side="left", padx=5)
        
        # Time slider
        self.time_slider = ctk.CTkSlider(
            self.controls,
            from_=0,
            to=1000,
            command=self.seek
        )
        self.time_slider.pack(side="left", fill="x", expand=True, padx=5)
        
        # Time label
        self.time_label = ctk.CTkLabel(self.controls, text="00:00 / 00:00")
        self.time_label.pack(side="left", padx=5)
        
        # Bind video frame
        self.video_frame.bind("<Configure>", self._on_configure)
        
        # Start update thread
        self.running = True
        self.update_thread = threading.Thread(target=self._update_ui)
        self.update_thread.daemon = True
        self.update_thread.start()

    def _on_configure(self, event):
        """Handle window resize."""
        if self.player:
            if event.widget == self.video_frame:
                x, y = event.width, event.height
                if sys.platform.startswith('linux'):
                    self.player.set_xwindow(self.video_frame.winfo_id())
                elif sys.platform == "win32":
                    self.player.set_hwnd(self.video_frame.winfo_id())
                elif sys.platform == "darwin":
                    self.player.set_nsobject(self.video_frame.winfo_id())

    def toggle_play(self):
        """Toggle play/pause."""
        if self.player.is_playing():
            self.player.pause()
            self.play_button.configure(text="Play")
        else:
            self.player.play()
            self.play_button.configure(text="Pause")

    def seek(self, value):
        """Seek to position."""
        self.player.set_position(float(value) / 1000.0)

    def get_time(self) -> float:
        """Get current time in seconds."""
        return self.player.get_time() / 1000.0

    def _update_ui(self):
        """Update UI elements."""
        while self.running:
            if self.player.is_playing():
                # Update slider
                pos = self.player.get_position()
                self.time_slider.set(pos * 1000)
                
                # Update time label
                current = self.player.get_time() / 1000
                duration = self.player.get_length() / 1000
                self.time_label.configure(
                    text=f"{int(current//60):02d}:{int(current%60):02d} / "
                         f"{int(duration//60):02d}:{int(duration%60):02d}"
                )
            time.sleep(0.1)

    def destroy(self):
        """Clean up resources."""
        self.running = False
        if self.player:
            self.player.stop()
        super().destroy()


class VideoAnnotator(ctk.CTk):
    """Main video annotation application."""
    
    def __init__(self):
        """Initialize the application."""
        super().__init__()
        
        # Initialize vision analyzer
        self.vision_analyzer = MedicalVisionAnalyzer(use_gpu=True)
        
        # Configure window
        self.title("SpecializedMD Video Annotator")
        self.geometry("1200x800")
        
        # Configure grid
        self.grid_columnconfigure(0, weight=7)
        self.grid_columnconfigure(1, weight=3)
        self.grid_rowconfigure(0, weight=1)
        
        # Create main containers
        self.video_container = ctk.CTkFrame(self)
        self.video_container.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        self.annotation_container = ctk.CTkFrame(self)
        self.annotation_container.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        
        self._setup_video_section()
        self._setup_annotation_section()
        self._setup_ai_analysis_section()
        
        # Initialize data
        self.annotations: List[Dict] = []
        self.current_lecture_id: Optional[str] = None
        self.current_frame: Optional[np.ndarray] = None
        self.analysis_thread: Optional[threading.Thread] = None
        self.analysis_running = False

    def _setup_video_section(self):
        """Set up video player section."""
        # Placeholder for video player
        self.video_player = None
        
        # Load video button
        self.load_button = ctk.CTkButton(
            self.video_container,
            text="Load Video",
            command=self._load_video
        )
        self.load_button.pack(pady=10)

    def _setup_annotation_section(self):
        """Set up annotation controls and list."""
        # Annotation type selector
        self.annotation_type = ctk.CTkComboBox(
            self.annotation_container,
            values=["WARNING", "BEST_PRACTICE", "HISTORICAL_CONTEXT", "VIDEO_CUTAWAY", "AI_ANALYSIS"]
        )
        self.annotation_type.pack(fill="x", padx=10, pady=5)
        
        # Annotation text
        self.annotation_text = ctk.CTkTextbox(
            self.annotation_container,
            height=100
        )
        self.annotation_text.pack(fill="x", padx=10, pady=5)
        
        # External reference
        self.ref_frame = ctk.CTkFrame(self.annotation_container)
        self.ref_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(self.ref_frame, text="External Reference:").pack(side="left")
        self.ref_entry = ctk.CTkEntry(self.ref_frame)
        self.ref_entry.pack(side="left", fill="x", expand=True, padx=5)
        
        # Add annotation button
        self.add_button = ctk.CTkButton(
            self.annotation_container,
            text="Add Annotation",
            command=self._add_annotation
        )
        self.add_button.pack(pady=10)
        
        # Annotations list
        self.annotations_frame = ctk.CTkFrame(self.annotation_container)
        self.annotations_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Create treeview for annotations
        self.tree = ttk.Treeview(
            self.annotations_frame,
            columns=("Time", "Type", "Content"),
            show="headings"
        )
        
        self.tree.heading("Time", text="Time")
        self.tree.heading("Type", text="Type")
        self.tree.heading("Content", text="Content")
        
        self.tree.column("Time", width=80)
        self.tree.column("Type", width=100)
        self.tree.column("Content", width=200)
        
        self.tree.pack(fill="both", expand=True)
        
        # Save button
        self.save_button = ctk.CTkButton(
            self.annotation_container,
            text="Save Annotations",
            command=self._save_annotations
        )
        self.save_button.pack(pady=10)

    def _setup_ai_analysis_section(self):
        """Set up AI analysis controls."""
        # AI Analysis Frame
        self.ai_frame = ctk.CTkFrame(self.annotation_container)
        self.ai_frame.pack(fill="x", padx=10, pady=5)
        
        # Analysis type checkboxes
        self.analysis_types = {
            "tools": tk.BooleanVar(value=True),
            "procedure": tk.BooleanVar(value=True),
            "description": tk.BooleanVar(value=True)
        }
        
        ctk.CTkLabel(self.ai_frame, text="AI Analysis:").pack(anchor="w")
        
        for analysis_type, var in self.analysis_types.items():
            ctk.CTkCheckBox(
                self.ai_frame,
                text=analysis_type.capitalize(),
                variable=var
            ).pack(anchor="w", padx=5)
        
        # Analysis buttons
        self.analyze_frame_btn = ctk.CTkButton(
            self.ai_frame,
            text="Analyze Current Frame",
            command=self._analyze_current_frame
        )
        self.analyze_frame_btn.pack(pady=5)
        
        self.analyze_segment_btn = ctk.CTkButton(
            self.ai_frame,
            text="Analyze Segment",
            command=self._analyze_segment
        )
        self.analyze_segment_btn.pack(pady=5)
        
        # Analysis results
        self.analysis_text = ctk.CTkTextbox(
            self.ai_frame,
            height=150,
            wrap="word"
        )
        self.analysis_text.pack(fill="x", pady=5)

    def _load_video(self):
        """Load a video file."""
        file_path = tk.filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.avi *.mkv")]
        )
        
        if file_path:
            # Clean up existing player
            if self.video_player:
                self.video_player.destroy()
            
            # Create new player
            self.video_player = VideoPlayer(
                self.video_container,
                file_path
            )
            self.video_player.pack(expand=True, fill="both")
            
            # Set lecture ID from filename
            self.current_lecture_id = Path(file_path).stem
            
            # Load existing annotations
            self._load_annotations()

    def _add_annotation(self):
        """Add a new annotation at current timestamp."""
        if not self.video_player:
            return
            
        current_time = self.video_player.get_time()
        annotation_type = self.annotation_type.get()
        content = self.annotation_text.get("1.0", "end-1c")
        reference = self.ref_entry.get()
        
        annotation = {
            "timestamp": current_time,
            "type": annotation_type,
            "content": content,
            "reference": reference,
            "created_at": datetime.now().isoformat()
        }
        
        self.annotations.append(annotation)
        
        # Add to treeview
        self.tree.insert(
            "",
            "end",
            values=(
                f"{int(current_time//60):02d}:{int(current_time%60):02d}",
                annotation_type,
                content[:50] + "..." if len(content) > 50 else content
            )
        )
        
        # Clear inputs
        self.annotation_text.delete("1.0", "end")
        self.ref_entry.delete(0, "end")

    def _save_annotations(self):
        """Save annotations to file."""
        if not self.current_lecture_id:
            return
            
        output_dir = Path("data/annotations")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f"{self.current_lecture_id}_annotations.json"
        
        with open(output_file, "w") as f:
            json.dump({
                "lecture_id": self.current_lecture_id,
                "annotations": self.annotations
            }, f, indent=2)

    def _load_annotations(self):
        """Load existing annotations for current lecture."""
        if not self.current_lecture_id:
            return
            
        annotation_file = Path(f"data/annotations/{self.current_lecture_id}_annotations.json")
        
        if annotation_file.exists():
            with open(annotation_file) as f:
                data = json.load(f)
                self.annotations = data["annotations"]
                
                # Clear and reload treeview
                for item in self.tree.get_children():
                    self.tree.delete(item)
                    
                for annotation in self.annotations:
                    time = annotation["timestamp"]
                    self.tree.insert(
                        "",
                        "end",
                        values=(
                            f"{int(time//60):02d}:{int(time%60):02d}",
                            annotation["type"],
                            annotation["content"][:50] + "..."
                            if len(annotation["content"]) > 50 else annotation["content"]
                        )
                    )

    def _analyze_current_frame(self):
        """Analyze the current video frame."""
        if not self.video_player or self.current_frame is None:
            return
            
        # Get selected analysis types
        analysis_types = [
            analysis_type
            for analysis_type, var in self.analysis_types.items()
            if var.get()
        ]
        
        # Perform analysis
        results = self.vision_analyzer.analyze_surgical_frame(
            self.current_frame,
            analysis_types=analysis_types
        )
        
        # Display results
        self._display_analysis_results(results)
        
        # Create annotation from analysis
        self._create_annotation_from_analysis(results)

    def _analyze_segment(self):
        """Analyze a segment of the video."""
        if not self.video_player:
            return
            
        # Get current position
        current_time = self.video_player.get_time()
        
        # Analyze next 30 seconds
        self.analysis_running = True
        self.analysis_thread = threading.Thread(
            target=self._run_segment_analysis,
            args=(current_time, 30.0)
        )
        self.analysis_thread.daemon = True
        self.analysis_thread.start()

    def _run_segment_analysis(self, start_time: float, duration: float):
        """Run analysis on a video segment in a separate thread."""
        try:
            results = self.vision_analyzer.analyze_video_segment(
                self.video_player.get_media().get_mrl(),
                start_time,
                duration
            )
            
            # Process results
            self.after(0, self._process_segment_results, results)
            
        finally:
            self.analysis_running = False

    def _process_segment_results(self, results: List[Dict]):
        """Process and display segment analysis results."""
        for result in results:
            self._display_analysis_results(result)
            self._create_annotation_from_analysis(result)

    def _display_analysis_results(self, results: Dict):
        """Display analysis results in the UI."""
        # Clear previous results
        self.analysis_text.delete("1.0", "end")
        
        # Format and display results
        if "tools" in results:
            self.analysis_text.insert("end", "Surgical Tools:\n")
            for tool in results["tools"]:
                self.analysis_text.insert(
                    "end",
                    f"- {tool['tool_type']}: {tool['description']}\n"
                )
        
        if "procedure" in results:
            self.analysis_text.insert("end", "\nProcedure:\n")
            self.analysis_text.insert(
                "end",
                f"- {results['procedure']['procedure_details']}\n"
            )
        
        if "description" in results:
            self.analysis_text.insert("end", "\nMedical Description:\n")
            self.analysis_text.insert(
                "end",
                f"- Anatomy: {results['description']['anatomical_description']}\n"
                f"- Technique: {results['description']['technique_description']}\n"
                f"- Safety: {results['description']['safety_considerations']}\n"
            )

    def _create_annotation_from_analysis(self, results: Dict):
        """Create an annotation from analysis results."""
        if not results:
            return
            
        # Create annotation text
        annotation_text = ""
        
        if "procedure" in results:
            annotation_text += f"Procedure: {results['procedure']['procedure_details']}\n\n"
            
        if "description" in results:
            annotation_text += (
                f"Technical Details:\n"
                f"- {results['description']['technique_description']}\n\n"
                f"Safety Considerations:\n"
                f"- {results['description']['safety_considerations']}"
            )
        
        # Add annotation
        self.annotation_text.delete("1.0", "end")
        self.annotation_text.insert("1.0", annotation_text)
        
        # Set annotation type
        self.annotation_type.set("AI_ANALYSIS")
        
        # Add annotation
        self._add_annotation()


if __name__ == "__main__":
    import sys
    
    app = VideoAnnotator()
    app.mainloop()
