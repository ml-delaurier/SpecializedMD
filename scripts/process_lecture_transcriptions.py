"""
Process Lecture Transcriptions Script

This script processes medical lecture transcriptions using the TranscriptionAnalyzer
to generate enhanced RAG data, including Q&A pairs, key concepts, and clinical pearls.
It handles batch processing of multiple transcription files and organizes the output
in a structured format for improved retrieval operations.
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from core.rag.transcription_analyzer import TranscriptionAnalyzer
from core.audio.transcription_service import TranscriptionService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def process_single_lecture(
    input_file: Path,
    output_dir: Path,
    analyzer: TranscriptionAnalyzer
) -> Dict:
    """
    Process a single lecture transcription file.
    
    Args:
        input_file (Path): Path to transcription file
        output_dir (Path): Directory for enhanced output
        analyzer (TranscriptionAnalyzer): Initialized analyzer instance
        
    Returns:
        Dict: Processing results and statistics
    """
    logger.info(f"Processing lecture transcription: {input_file.name}")
    
    try:
        # Create lecture-specific output directory
        lecture_output = output_dir / input_file.stem
        lecture_output.mkdir(parents=True, exist_ok=True)
        
        # Analyze transcription
        results = analyzer.analyze_transcription(
            transcription_file=input_file,
            output_dir=lecture_output,
            segment_length=300,  # 5 minutes
            min_confidence=0.7
        )
        
        # Generate processing summary
        summary = {
            "lecture_id": input_file.stem,
            "processed_at": datetime.now().isoformat(),
            "segments_analyzed": len(results["segments"]),
            "qa_pairs_generated": sum(
                len(segment["qa_pairs"]) 
                for segment in results["segments"]
            ),
            "unique_concepts": len({
                concept
                for segment in results["segments"]
                for concept in segment["key_concepts"]
            }),
            "clinical_pearls": sum(
                len(segment["clinical_pearls"])
                for segment in results["segments"]
            )
        }
        
        # Save processing summary
        with open(lecture_output / "processing_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
            
        return summary
        
    except Exception as e:
        logger.error(f"Failed to process {input_file.name}: {str(e)}")
        return {
            "lecture_id": input_file.stem,
            "error": str(e),
            "processed_at": datetime.now().isoformat()
        }

def batch_process_lectures(
    input_dir: Path,
    output_dir: Path,
    max_workers: int = 4
) -> Dict:
    """
    Process multiple lecture transcriptions in parallel.
    
    Args:
        input_dir (Path): Directory containing transcription files
        output_dir (Path): Base directory for enhanced output
        max_workers (int): Maximum number of parallel workers
        
    Returns:
        Dict: Batch processing results and statistics
    """
    # Initialize analyzer
    analyzer = TranscriptionAnalyzer()
    
    # Find all transcription files
    transcription_files = list(input_dir.glob("*_transcription.json"))
    
    if not transcription_files:
        logger.warning(f"No transcription files found in {input_dir}")
        return {}
    
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit processing tasks
        future_to_file = {
            executor.submit(
                process_single_lecture,
                input_file,
                output_dir,
                analyzer
            ): input_file
            for input_file in transcription_files
        }
        
        # Process results as they complete
        for future in tqdm(
            as_completed(future_to_file),
            total=len(transcription_files),
            desc="Processing lectures"
        ):
            file = future_to_file[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {file}: {str(e)}")
    
    # Generate batch summary
    batch_summary = {
        "processed_at": datetime.now().isoformat(),
        "total_lectures": len(transcription_files),
        "successful_processes": len([r for r in results if "error" not in r]),
        "failed_processes": len([r for r in results if "error" in r]),
        "total_qa_pairs": sum(
            r.get("qa_pairs_generated", 0)
            for r in results
            if "error" not in r
        ),
        "total_unique_concepts": sum(
            r.get("unique_concepts", 0)
            for r in results
            if "error" not in r
        ),
        "lecture_summaries": results
    }
    
    # Save batch summary
    with open(output_dir / "batch_processing_summary.json", "w") as f:
        json.dump(batch_summary, f, indent=2)
    
    return batch_summary

def create_rag_index(output_dir: Path) -> Dict:
    """
    Create a consolidated RAG index from processed lectures.
    
    Args:
        output_dir (Path): Directory containing processed lecture data
        
    Returns:
        Dict: Consolidated RAG index
    """
    logger.info("Creating consolidated RAG index")
    
    rag_index = {
        "qa_pairs": [],
        "concepts": {},
        "clinical_pearls": [],
        "references": set()
    }
    
    # Process each lecture directory
    for lecture_dir in output_dir.iterdir():
        if not lecture_dir.is_dir():
            continue
            
        enhanced_file = lecture_dir / f"{lecture_dir.name}_enhanced.json"
        if not enhanced_file.exists():
            continue
            
        try:
            with open(enhanced_file, "r") as f:
                lecture_data = json.load(f)
                
            # Process each segment
            for segment in lecture_data["segments"]:
                # Add QA pairs with lecture context
                for qa in segment["qa_pairs"]:
                    qa["lecture_id"] = lecture_dir.name
                    qa["timestamp"] = {
                        "start": segment["start_time"],
                        "end": segment["end_time"]
                    }
                    rag_index["qa_pairs"].append(qa)
                
                # Update concepts index
                for concept in segment["key_concepts"]:
                    if concept not in rag_index["concepts"]:
                        rag_index["concepts"][concept] = []
                    rag_index["concepts"][concept].append({
                        "lecture_id": lecture_dir.name,
                        "timestamp": {
                            "start": segment["start_time"],
                            "end": segment["end_time"]
                        },
                        "context": segment["text"][:200] + "..."
                    })
                
                # Add clinical pearls
                for pearl in segment["clinical_pearls"]:
                    rag_index["clinical_pearls"].append({
                        "pearl": pearl,
                        "lecture_id": lecture_dir.name,
                        "timestamp": {
                            "start": segment["start_time"],
                            "end": segment["end_time"]
                        }
                    })
                
                # Add references
                rag_index["references"].update(segment["references"])
                
        except Exception as e:
            logger.error(f"Error processing {lecture_dir.name}: {str(e)}")
    
    # Convert references set to list for JSON serialization
    rag_index["references"] = list(rag_index["references"])
    
    # Save consolidated index
    index_file = output_dir / "consolidated_rag_index.json"
    with open(index_file, "w") as f:
        json.dump(rag_index, f, indent=2)
    
    logger.info(f"RAG index created with:")
    logger.info(f"- {len(rag_index['qa_pairs'])} QA pairs")
    logger.info(f"- {len(rag_index['concepts'])} unique concepts")
    logger.info(f"- {len(rag_index['clinical_pearls'])} clinical pearls")
    logger.info(f"- {len(rag_index['references'])} references")
    
    return rag_index

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Process medical lecture transcriptions for enhanced RAG operations"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing transcription files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory for enhanced output"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum number of parallel workers"
    )
    
    args = parser.parse_args()
    
    # Convert paths
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process lectures
    batch_summary = batch_process_lectures(
        input_dir=input_dir,
        output_dir=output_dir,
        max_workers=args.max_workers
    )
    
    # Create consolidated RAG index
    rag_index = create_rag_index(output_dir)
    
    logger.info("Processing complete!")
    logger.info(f"Processed {batch_summary['total_lectures']} lectures")
    logger.info(f"Generated {batch_summary['total_qa_pairs']} QA pairs")
    logger.info(f"Identified {batch_summary['total_unique_concepts']} unique concepts")

if __name__ == "__main__":
    main()
