#!/usr/bin/env python3
"""
Script to analyze training time discrepancies between speedrun_results.json files and wandb data.

This script:
1. Recursively finds all speedrun_results.json files in the repository
2. Extracts training_time and wandb_run_link from each file
3. Calculates training time from wandb using get_step_times_from_wandb
4. Creates comparison plots showing the discrepancy
"""

import json
import logging
import os
from pathlib import Path
from urllib.parse import urlparse
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import wandb
from matplotlib.patches import Rectangle

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_step_times_from_wandb(run_id: str, entity: str, project: str) -> List[float]:
    """Get step times from a Weights & Biases run.

    Args:
        run_id: The WandB run id
        entity: The WandB entity (user or org)
        project: The WandB project name
    """
    try:
        run = wandb.Api().run(f"{entity}/{project}/{run_id}")
        return [
            row["throughput/duration"]
            for row in run.scan_history(keys=["throughput/duration"])
            if "throughput/duration" in row
        ]
    except Exception as e:
        logger.error(f"Failed to fetch step times for run {entity}/{project}/{run_id}: {e}")
        return []


def find_speedrun_files(repo_root: str) -> List[Path]:
    """Find all speedrun_results.json files in the repository"""
    repo_path = Path(repo_root)
    speedrun_files = list(repo_path.rglob("speedrun_results.json"))
    logger.info(f"Found {len(speedrun_files)} speedrun_results.json files")
    return speedrun_files


def parse_wandb_link(wandb_link: str) -> Tuple[str, str, str]:
    """Extract (entity, project, run_id) from a WandB run URL.

    Expected formats include:
      - https://wandb.ai/<entity>/<project>/runs/<run_id>
      - https://wandb.ai/<entity>/<project>/runs/<run_id>/
    """
    try:
        parsed = urlparse(wandb_link)
        segments = [seg for seg in parsed.path.split("/") if seg]
        # Typical path: [entity, project, 'runs', run_id]
        if len(segments) >= 4 and segments[2] == "runs":
            entity, project, run_id = segments[0], segments[1], segments[3]
            return entity, project, run_id
    except Exception:
        pass

    # Fallback parser using simple splitting
    parts = [seg for seg in wandb_link.strip("/").split("/") if seg]
    if "runs" in parts:
        idx = parts.index("runs")
        if idx >= 2 and idx + 1 < len(parts):
            entity = parts[idx - 2]
            project = parts[idx - 1]
            run_id = parts[idx + 1]
            return entity, project, run_id

    raise ValueError(f"Unrecognized WandB link format: {wandb_link}")


def process_speedrun_file(file_path: Path, repo_root: str) -> Dict:
    """Process a single speedrun_results.json file and return relevant data"""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Extract data from the first run
        if not data.get("runs") or len(data["runs"]) == 0:
            logger.warning(f"No runs found in {file_path}")
            return None
            
        run_info = data["runs"][0]["run_info"]
        
        # Get relative path from repo root
        rel_path = file_path.relative_to(Path(repo_root))
        
        # Extract training time and wandb link
        training_time = run_info.get("training_time")
        wandb_link = run_info.get("wandb_run_link")
        
        if training_time is None or wandb_link is None:
            logger.warning(f"Missing training_time or wandb_run_link in {file_path}")
            return None
            
        # Extract entity/project/run_id and calculate wandb training time
        entity, project, run_id = parse_wandb_link(wandb_link)
        step_times = get_step_times_from_wandb(run_id, entity=entity, project=project)
        
        if not step_times:
            logger.warning(f"Could not fetch step times for run {run_id} from {file_path}")
            return None
            
        wandb_training_time = sum(step_times)
        
        return {
            "file_path": str(rel_path),
            "training_time": training_time,
            "wandb_training_time": wandb_training_time,
            "run_id": run_id,
            "wandb_entity": entity,
            "wandb_project": project,
            "discrepancy_percent": ((training_time - wandb_training_time) / wandb_training_time) * 100 if wandb_training_time > 0 else 0
        }
        
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        return None


def create_comparison_plots(data: List[Dict], output_path: str = "training_time_discrepancy_analysis.png"):
    """Create comparison plots showing training time discrepancies"""
    
    if not data:
        logger.error("No data to plot")
        return
        
    # Sort data by file path for consistent ordering
    data = sorted(data, key=lambda x: x["file_path"])
    
    # Extract data for plotting
    file_paths = [d["file_path"].removesuffix("/speedrun_results.json") for d in data]
    training_times = [d["training_time"] for d in data]
    wandb_times = [d["wandb_training_time"] for d in data]
    discrepancies = [d["discrepancy_percent"] for d in data]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), constrained_layout=True)
    
    # Top plot: Training times comparison
    x_pos = np.arange(len(file_paths))
    width = 0.35
    
    bars1 = ax1.bar(x_pos - width/2, training_times, width, label='speedrun_results.json', alpha=0.8, color='skyblue')
    bars2 = ax1.bar(x_pos + width/2, wandb_times, width, label='wandb calculated', alpha=0.8, color='lightcoral')
    
    ax1.set_xlabel('Speedrun Files')
    ax1.set_ylabel('Training Time (seconds)')
    ax1.set_title('Training Time Comparison: speedrun_results.json vs wandb calculated')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(file_paths, rotation=90, ha='right')
    # Move file path labels further down to avoid overlap with discrepancy labels
    ax1.tick_params(axis='x', which='major', pad=20)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars (rotated 90 degrees)
    max_height = max(max(training_times), max(wandb_times))
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        height1 = bar1.get_height()
        height2 = bar2.get_height()
        ax1.text(bar1.get_x() + bar1.get_width()/2., height1 + max_height * 0.01,
                f'{height1:.0f}', ha='center', va='bottom', fontsize=7, rotation=90)
        ax1.text(bar2.get_x() + bar2.get_width()/2., height2 + max_height * 0.01,
                f'{height2:.0f}', ha='center', va='bottom', fontsize=7, rotation=90)
        
        # Add discrepancy percentage label below the bars (horizontal)
        discrepancy = discrepancies[i]
        # Position label well below the bars to avoid overlap with file path labels
        label_y = -max_height * 0.05  # Position further below the bars
        # Color code: red for positive, blue for negative
        discrepancy_color = 'red' if discrepancy >= 0 else 'blue'
        ax1.text(i, label_y, f'{discrepancy:+.1f}%', ha='center', va='top', 
                fontsize=6, color=discrepancy_color, weight='bold')
    
    # Bottom plot: Histogram of discrepancies
    n, bins, patches = ax2.hist(discrepancies, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
    ax2.set_xlabel('Discrepancy Percentage (%)')
    ax2.set_ylabel('Number of Files')
    ax2.set_title('Distribution of Training Time Discrepancies')
    ax2.grid(True, alpha=0.3)
    
    # Add bin count annotations
    for i, (count, bin_left, bin_right) in enumerate(zip(n, bins[:-1], bins[1:])):
        if count > 0:  # Only annotate bins with files
            bin_center = (bin_left + bin_right) / 2
            ax2.text(bin_center, count + 0.1, f'{int(count)}', ha='center', va='bottom', 
                    fontsize=8, weight='bold')
    
    # Add statistics text
    mean_discrepancy = np.mean(discrepancies)
    std_discrepancy = np.std(discrepancies)
    max_discrepancy = np.max(np.abs(discrepancies))
    
    stats_text = f'Mean: {mean_discrepancy:.2f}%\nStd: {std_discrepancy:.2f}%\nMax: {max_discrepancy:.2f}%'
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Plots saved to {output_path}")
    
    # Print summary statistics
    print(f"\n=== Training Time Discrepancy Analysis ===")
    print(f"Total files analyzed: {len(data)}")
    print(f"Mean discrepancy: {mean_discrepancy:.2f}%")
    print(f"Standard deviation: {std_discrepancy:.2f}%")
    print(f"Maximum absolute discrepancy: {max_discrepancy:.2f}%")
    print(f"Files with >10% discrepancy: {sum(1 for d in discrepancies if abs(d) > 10)}")
    print(f"Files with >5% discrepancy: {sum(1 for d in discrepancies if abs(d) > 5)}")
    
    # Show files with largest discrepancies
    print(f"\n=== Files with Largest Discrepancies ===")
    sorted_data = sorted(data, key=lambda x: abs(x['discrepancy_percent']), reverse=True)
    for i, d in enumerate(sorted_data[:10]):  # Top 10
        print(f"{i+1:2d}. {d['file_path']}: {d['discrepancy_percent']:+.2f}% "
              f"(speedrun: {d['training_time']:.0f}s, wandb: {d['wandb_training_time']:.0f}s)")


def main():
    """Main function to run the analysis"""
    # Get repository root (assuming script is run from repo root)
    repo_root = os.getcwd()
    logger.info(f"Analyzing repository: {repo_root}")
    
    # Find all speedrun files
    speedrun_files = find_speedrun_files(repo_root)
    
    if not speedrun_files:
        logger.error("No speedrun_results.json files found")
        return
    
    # Process each file
    data = []
    for file_path in speedrun_files:
        logger.info(f"Processing {file_path}")
        result = process_speedrun_file(file_path, repo_root)
        if result:
            data.append(result)
    
    if not data:
        logger.error("No valid data found")
        return
    
    logger.info(f"Successfully processed {len(data)} files")
    
    # Create plots
    create_comparison_plots(data)


if __name__ == "__main__":
    main()
