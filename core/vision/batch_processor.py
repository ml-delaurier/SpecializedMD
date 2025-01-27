"""
BatchProcessor: Handles batch processing of multiple surgical videos and exports
analysis results in various formats.
"""

import os
import json
import csv
import yaml
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import logging
from tqdm import tqdm

from .medical_vision_analyzer import MedicalVisionAnalyzer
from .medical_queries import SurgicalQueries

class BatchProcessor:
    """Process multiple surgical videos and export results."""
    
    def __init__(self,
                 output_dir: str,
                 use_gpu: bool = False,
                 max_workers: int = 4):
        """
        Initialize batch processor.
        
        Args:
            output_dir (str): Directory for output files
            use_gpu (bool): Whether to use GPU acceleration
            max_workers (int): Maximum number of parallel workers
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.analyzer = MedicalVisionAnalyzer(use_gpu=use_gpu)
        self.max_workers = max_workers
        self.logger = logging.getLogger(__name__)
        
        # Load specialized queries
        self.queries = SurgicalQueries()

    def process_directory(self,
                        input_dir: str,
                        file_pattern: str = "*.mp4",
                        analysis_config: Optional[Dict] = None) -> Dict:
        """
        Process all videos in a directory.
        
        Args:
            input_dir (str): Input directory containing videos
            file_pattern (str): Pattern to match video files
            analysis_config (Dict): Configuration for analysis types
            
        Returns:
            Dict: Processing results and statistics
        """
        input_path = Path(input_dir)
        video_files = list(input_path.glob(file_pattern))
        
        if not video_files:
            self.logger.warning(f"No video files found in {input_dir}")
            return {}
        
        # Default analysis configuration
        if analysis_config is None:
            analysis_config = {
                "frame_interval": 1.0,  # Analyze every second
                "analysis_types": ["tools", "procedure", "description"],
                "specialized_queries": {
                    "anatomical": True,
                    "technical": True,
                    "safety": True,
                    "pathology": True,
                    "quality": True
                }
            }
        
        results = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_video = {
                executor.submit(
                    self._process_single_video,
                    video_file,
                    analysis_config
                ): video_file
                for video_file in video_files
            }
            
            for future in tqdm(
                as_completed(future_to_video),
                total=len(video_files),
                desc="Processing videos"
            ):
                video_file = future_to_video[future]
                try:
                    result = future.result()
                    results[video_file.name] = result
                except Exception as e:
                    self.logger.error(f"Error processing {video_file}: {str(e)}")
                    results[video_file.name] = {"error": str(e)}
        
        # Save batch results
        self._save_batch_results(results)
        
        return results

    def _process_single_video(self,
                            video_path: Path,
                            config: Dict) -> Dict:
        """Process a single video file."""
        self.logger.info(f"Processing video: {video_path.name}")
        
        results = {
            "video_file": video_path.name,
            "processed_at": datetime.now().isoformat(),
            "config": config,
            "analysis": []
        }
        
        # Analyze video segments
        segment_results = self.analyzer.analyze_video_segment(
            str(video_path),
            start_time=0,
            duration=float("inf"),  # Analyze entire video
            frame_interval=config["frame_interval"]
        )
        
        # Add specialized analysis
        for frame_result in segment_results:
            specialized_results = self._run_specialized_queries(
                frame_result["frame"],
                config["specialized_queries"]
            )
            frame_result.update(specialized_results)
            results["analysis"].append(frame_result)
        
        return results

    def _run_specialized_queries(self,
                               frame: Union[str, Path],
                               query_config: Dict) -> Dict:
        """Run specialized medical queries on a frame."""
        results = {}
        
        if query_config.get("anatomical"):
            results["anatomical"] = self._run_query_set(
                frame,
                self.queries.get_anatomical_queries()
            )
            
        if query_config.get("technical"):
            results["technical"] = self._run_query_set(
                frame,
                self.queries.get_technical_queries()
            )
            
        if query_config.get("safety"):
            results["safety"] = self._run_query_set(
                frame,
                self.queries.get_safety_queries()
            )
            
        if query_config.get("pathology"):
            results["pathology"] = self._run_query_set(
                frame,
                self.queries.get_pathology_queries()
            )
            
        if query_config.get("quality"):
            results["quality"] = self._run_query_set(
                frame,
                self.queries.get_quality_assessment_queries()
            )
        
        return results

    def _run_query_set(self,
                      frame: Union[str, Path],
                      queries: List[Dict[str, str]]) -> Dict:
        """Run a set of queries on a frame."""
        results = {}
        for query in queries:
            response = self.analyzer.model.query(
                frame,
                query["query"]
            )
            results[query["id"]] = response["answer"]
        return results

    def _save_batch_results(self, results: Dict):
        """Save batch processing results in multiple formats."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed JSON
        json_path = self.output_dir / f"batch_results_{timestamp}.json"
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        
        # Save summary CSV
        csv_path = self.output_dir / f"batch_summary_{timestamp}.csv"
        self._export_csv_summary(results, csv_path)
        
        # Save YAML format
        yaml_path = self.output_dir / f"batch_results_{timestamp}.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(results, f)
        
        # Generate Excel report
        excel_path = self.output_dir / f"batch_report_{timestamp}.xlsx"
        self._export_excel_report(results, excel_path)

    def _export_csv_summary(self, results: Dict, output_path: Path):
        """Export summary of results to CSV."""
        summary_data = []
        
        for video_name, video_results in results.items():
            if "error" in video_results:
                continue
                
            for frame_analysis in video_results["analysis"]:
                summary_data.append({
                    "video_name": video_name,
                    "timestamp": frame_analysis.get("timestamp"),
                    "procedure_type": frame_analysis.get("procedure", {}).get("procedure_details", ""),
                    "tools_detected": len(frame_analysis.get("tools", [])),
                    "safety_concerns": frame_analysis.get("description", {}).get("safety_considerations", "")
                })
        
        df = pd.DataFrame(summary_data)
        df.to_csv(output_path, index=False)

    def _export_excel_report(self, results: Dict, output_path: Path):
        """Export detailed analysis report to Excel."""
        with pd.ExcelWriter(output_path) as writer:
            # Summary sheet
            summary_data = []
            for video_name, video_results in results.items():
                if "error" in video_results:
                    continue
                    
                summary_data.append({
                    "video_name": video_name,
                    "processed_at": video_results["processed_at"],
                    "frame_count": len(video_results["analysis"]),
                    "config": str(video_results["config"])
                })
            
            pd.DataFrame(summary_data).to_excel(
                writer,
                sheet_name="Summary",
                index=False
            )
            
            # Detailed analysis sheets
            for video_name, video_results in results.items():
                if "error" in video_results:
                    continue
                    
                # Create separate sheets for different analysis types
                self._export_analysis_sheet(
                    writer,
                    f"{video_name[:28]}_tools",
                    video_results,
                    "tools"
                )
                self._export_analysis_sheet(
                    writer,
                    f"{video_name[:28]}_procedure",
                    video_results,
                    "procedure"
                )
                self._export_analysis_sheet(
                    writer,
                    f"{video_name[:28]}_safety",
                    video_results,
                    "safety"
                )

    def _export_analysis_sheet(self,
                             writer: pd.ExcelWriter,
                             sheet_name: str,
                             video_results: Dict,
                             analysis_type: str):
        """Export specific analysis type to Excel sheet."""
        analysis_data = []
        
        for frame_analysis in video_results["analysis"]:
            if analysis_type == "tools":
                for tool in frame_analysis.get("tools", []):
                    analysis_data.append({
                        "timestamp": frame_analysis["timestamp"],
                        "tool_type": tool["tool_type"],
                        "description": tool["description"],
                        "confidence": tool.get("confidence", "N/A")
                    })
            elif analysis_type == "procedure":
                analysis_data.append({
                    "timestamp": frame_analysis["timestamp"],
                    "procedure_details": frame_analysis.get("procedure", {}).get("procedure_details", ""),
                    "technical_description": frame_analysis.get("description", {}).get("technique_description", "")
                })
            elif analysis_type == "safety":
                analysis_data.append({
                    "timestamp": frame_analysis["timestamp"],
                    "safety_considerations": frame_analysis.get("description", {}).get("safety_considerations", ""),
                    "anatomical_description": frame_analysis.get("description", {}).get("anatomical_description", "")
                })
        
        if analysis_data:
            pd.DataFrame(analysis_data).to_excel(
                writer,
                sheet_name=sheet_name,
                index=False
            )
