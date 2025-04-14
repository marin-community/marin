"""
Leaderboard system for Marin speedruns.
"""

import json
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional
from pathlib import Path
import fsspec
import logging

logger = logging.getLogger(__name__)

@dataclass
class LeaderboardEntry:
    run_name: str
    compute_budget_track: str
    model_size: int
    total_training_time: float
    submission_timestamp: str
    final_flops_used: float
    submitted_by: str
    storage_path: str  # Path to the analysis file
    wandb_run_id: Optional[str] = None
    lm_eval_macro_avg_acc: Optional[float] = None
    eval_paloma_c4_en_bpb: Optional[float] = None

class Leaderboard:
    def __init__(self, base_path: str):
        """
        Initialize leaderboard with path containing run directories.
        
        Args:
            base_path: Storage path like 'gs://bucket/path/to/runs' or '/path/to/runs'
        """
        self.base_path = base_path.rstrip('/')
        scheme = base_path.split('://', 1)[0] if '://' in base_path else 'file'
        self.fs = fsspec.filesystem(scheme)
        
    def _find_analysis_files(self) -> List[str]:
        """Find all speedrun analysis JSON files in storage."""
        pattern = f"{self.base_path}/**/speedrun_analysis.json"
        return self.fs.glob(pattern)
    
    def _load_analysis_file(self, path: str) -> dict:
        """Load a single analysis file from storage."""
        with self.fs.open(path, 'r') as f:
            return json.load(f)
            
    def _create_entry_from_analysis(self, analysis: dict, storage_path: str) -> LeaderboardEntry:
        """Create a leaderboard entry from analysis data."""
        # Extract run name from storage path
        run_name = Path(storage_path).parent.name
        
        budget_flops = analysis['compute_budget']['flops_budget_for_track']
        actual_compute = analysis['actual_stats']['final_flops_estimate']
        
        # Convert to float and scientific notation
        if isinstance(actual_compute, str):
            if 'e' in actual_compute.lower():
                actual_flops = float(actual_compute.replace('e', 'E'))
            else:
                actual_flops = float(actual_compute)
        else:
            actual_flops = float(actual_compute)
            
        # Ensure both are in same format (scientific notation)
        budget_flops = float(f"{budget_flops:.2e}".replace('e', 'E'))
        actual_flops = float(f"{actual_flops:.2e}".replace('e', 'E'))
                
        # Get training time (handle both field names)
        training_time = analysis['actual_stats'].get('training_time_in_minutes', 
                                                    analysis['actual_stats'].get('training_time', 0.0))
        
        # Get eval metrics
        lm_eval_macro_avg_acc = analysis['actual_stats'].get('lm_eval/averages/macro_avg_acc')
        eval_paloma_c4_en_bpb = analysis['actual_stats'].get('eval/paloma/c4_en/bpb')
        
        return LeaderboardEntry(
            run_name=run_name,
            compute_budget_track=analysis['compute_budget']['track'],
            model_size=analysis['run_related_info']['num_parameters'],
            total_training_time=training_time,
            submission_timestamp=analysis.get('speedrun_analysis_timestamp', datetime.now().isoformat()),
            final_flops_used=actual_flops,
            submitted_by=analysis['run_related_info'].get('submitted_by', 'unknown'),
            storage_path=storage_path,
            wandb_run_id=None,
            lm_eval_macro_avg_acc=float(lm_eval_macro_avg_acc) if lm_eval_macro_avg_acc is not None else None,
            eval_paloma_c4_en_bpb=float(eval_paloma_c4_en_bpb) if eval_paloma_c4_en_bpb is not None else None
        )
    
    def get_entries(self) -> List[LeaderboardEntry]:
        """Load and return all leaderboard entries from storage."""
        entries = []
        analysis_files = self._find_analysis_files()
        
        for file_path in analysis_files:
            try:
                analysis = self._load_analysis_file(file_path)
                entry = self._create_entry_from_analysis(analysis, file_path)
                entries.append(entry)
            except Exception as e:
                logger.warning(f"Failed to process {file_path}: {e}")
                
        return entries
    
    def get_entries_by_track(self, track: str) -> List[LeaderboardEntry]:
        """Get entries filtered by compute budget track."""
        return [e for e in self.get_entries() if e.compute_budget_track == track]
    
    def format_leaderboard(self, track: Optional[str] = None) -> str:
        """Format the leaderboard as a markdown table."""
        entries = self.get_entries_by_track(track) if track else self.get_entries()
        
        if not entries:
            return "No entries found."
            
        # Sort by FLOPs used (lower is better)
        entries.sort(key=lambda x: x.final_flops_used, reverse=True)
        
        header = "| Rank | Run Name | Budget Track | Model Size | Training Time (min) | FLOPs Used | LM Eval Acc | C4 BPB |\n"
        separator = "|------|----------|--------------|------------|-------------------|-------------|------------|---------|\n"
        
        rows = []
        for i, entry in enumerate(entries, 1):
            model_size_str = f"{entry.model_size/1e6:.1f}M" if entry.model_size < 1e9 else f"{entry.model_size/1e9:.1f}B"
            eval_acc = f"{entry.lm_eval_macro_avg_acc:.3f}" if entry.lm_eval_macro_avg_acc is not None else "N/A"
            c4_bpb = f"{entry.paloma_c4_en_bpb:.3f}" if entry.paloma_c4_en_bpb is not None else "N/A"
            row = f"| {i} | {entry.run_name} | {entry.compute_budget_track} | {model_size_str} | {entry.total_training_time:.1f} | {entry.final_flops_used:.2e} | {eval_acc} | {c4_bpb} |"
            rows.append(row)
            
        return header + separator + "\n".join(rows)

def serve_leaderboard(storage_path: str, port: int = 8000):
    """
    Serve the leaderboard UI reading from storage.
    Creates a temporary JSON file for the web UI to read.
    """
    from pathlib import Path
    import http.server
    import socketserver
    import os
    
    # Initialize leaderboard
    leaderboard = Leaderboard(storage_path)
    
    # Ensure static/data directory exists
    static_dir = Path(__file__).parent / "static"
    data_dir = static_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Write current entries to JSON file for UI
    entries = leaderboard.get_entries()
    with open(data_dir / "leaderboard.json", 'w') as f:
        json.dump([{
            "run_name": e.run_name,
            "compute_budget_track": e.compute_budget_track,
            "model_size": e.model_size,
            "total_training_time": e.total_training_time,
            "final_flops_used": e.final_flops_used,
            "submission_timestamp": e.submission_timestamp,
            "submitted_by": e.submitted_by,
            "storage_path": e.storage_path,
            "wandb_run_id": e.wandb_run_id,
            "lm_eval_macro_avg_acc": e.lm_eval_macro_avg_acc,
            "eval_paloma_c4_en_bpb": e.eval_paloma_c4_en_bpb
        } for e in entries], f, indent=2)
    
    # Change to the static directory
    os.chdir(static_dir)
    
    # Start server
    Handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", port), Handler) as httpd:
        print(f"Serving leaderboard at http://localhost:{port}")
        print(f"Reading runs from: {storage_path}")
        print("Press Ctrl+C to stop")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server...")
