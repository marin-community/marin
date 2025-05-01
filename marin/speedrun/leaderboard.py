"""
Leaderboard system for Marin speedruns.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import fsspec

logger = logging.getLogger(__name__)


@dataclass
class LeaderboardEntry:
    run_name: str
    compute_budget_track: str
    model_size: int
    total_training_time: float
    final_flops_used: float
    submitted_by: str
    storage_path: str  # Path to the analysis file
    wandb_run_id: str | None = None
    lm_eval_macro_avg_acc: float | None = None
    eval_paloma_c4_en_bpb: float | None = None
    eval_fineweb_edu_loss: float | None = None

    def to_json(self) -> dict:
        return {
            "run_name": self.run_name,
            "compute_budget_track": self.compute_budget_track,
            "model_size": self.model_size,
            "total_training_time": self.total_training_time,
            "final_flops_used": self.final_flops_used,
            "submitted_by": self.submitted_by,
            "storage_path": self.storage_path,
            "wandb_run_id": self.wandb_run_id,
            "lm_eval_macro_avg_acc": self.lm_eval_macro_avg_acc,
            "eval_paloma_c4_en_bpb": self.eval_paloma_c4_en_bpb,
            "eval_fineweb_edu_loss": self.eval_fineweb_edu_loss,
        }


class Leaderboard:
    def __init__(self, base_path: str):
        self.base_path = base_path.rstrip("/")
        scheme = base_path.split("://", 1)[0] if "://" in base_path else "file"
        self.fs = fsspec.filesystem(scheme)

    def _find_analysis_files(self) -> list[str]:
        pattern = f"{self.base_path}/**/speedrun_analysis.json"
        return self.fs.glob(pattern)

    def _load_analysis_file(self, path: str) -> dict:
        with self.fs.open(path, "r") as f:
            data = json.load(f)
            # Handle both old and new format
            if "runs" in data:
                return data
            else:
                # Convert old format to new format
                return {"runs": [data]}

    def _create_entry_from_analysis(self, analysis: dict, storage_path: str) -> LeaderboardEntry:
        run_name = Path(storage_path).parent.name

        budget_flops = analysis["compute_budget"]["flops_budget_for_track"]
        actual_compute = analysis["actual_stats"]["final_flops_estimate"]

        # Convert to float and scientific notation
        if isinstance(actual_compute, str):
            if "e" in actual_compute.lower():
                actual_flops = float(actual_compute.replace("e", "E"))
            else:
                actual_flops = float(actual_compute)
        else:
            actual_flops = float(actual_compute)

        # Ensure both are in same format (scientific notation)
        budget_flops = float(f"{budget_flops:.2e}".replace("e", "E"))
        actual_flops = float(f"{actual_flops:.2e}".replace("e", "E"))

        # Get training time (handle both field names)
        training_time = analysis["actual_stats"].get(
            "training_time_in_minutes", analysis["actual_stats"].get("training_time", 0.0)
        )

        # Get eval metrics
        lm_eval_macro_avg_acc = analysis["actual_stats"].get("lm_eval/averages/macro_avg_acc")
        eval_paloma_c4_en_bpb = analysis["actual_stats"].get("eval/paloma/c4_en/bpb")

        return LeaderboardEntry(
            run_name=run_name,
            compute_budget_track=analysis["compute_budget"]["track"],
            model_size=analysis["run_related_info"]["num_parameters"],
            total_training_time=training_time,
            final_flops_used=actual_flops,
            submitted_by=analysis["run_related_info"].get("submitted_by", "unknown"),
            storage_path=storage_path,
            wandb_run_id=analysis["run_related_info"].get("wandb_run_id", None),
            lm_eval_macro_avg_acc=float(lm_eval_macro_avg_acc) if lm_eval_macro_avg_acc is not None else None,
            eval_paloma_c4_en_bpb=float(eval_paloma_c4_en_bpb) if eval_paloma_c4_en_bpb is not None else None,
            eval_fineweb_edu_loss=(
                float(analysis["actual_stats"].get("eval/fineweb-edu/loss"))
                if analysis["actual_stats"].get("eval/fineweb-edu/loss") is not None
                else None
            ),
        )

    def get_entries(self) -> list[LeaderboardEntry]:
        entries = []
        analysis_files = self._find_analysis_files()

        for file_path in analysis_files:
            try:
                analysis_data = self._load_analysis_file(file_path)
                # Process each run in the runs list
                for i, run_analysis in enumerate(analysis_data["runs"]):
                    # For multiple runs in the same file, we need to create unique paths
                    # to differentiate between them in the leaderboard
                    run_path = f"{file_path}#{i}" if i > 0 else file_path
                    entry = self._create_entry_from_analysis(run_analysis, run_path)
                    entries.append(entry)
            except Exception as e:
                logger.warning(f"Failed to process {file_path}: {e}")

        return entries

    def get_entries_by_track(self, track: str) -> list[LeaderboardEntry]:
        return [e for e in self.get_entries() if e.compute_budget_track == track]

    def format_leaderboard(self, track: str | None = None) -> str:
        entries = self.get_entries_by_track(track) if track else self.get_entries()

        if not entries:
            return "No entries found."

        # Sort by FLOPs used (lower is better)
        entries.sort(key=lambda x: x.final_flops_used, reverse=True)

        header = "| Rank | Run Name | Budget Track | Model Size | Training Time (min) | FLOPs Used |\n"
        header += "| LM Eval Acc | C4-EN BPB | Fineweb-Edu Loss |\n"
        separator = "|------|----------|--------------|------------|-------------------|-------------|\n"
        separator += "|------------|---------|----------------|\n"

        rows = []
        for i, entry in enumerate(entries, 1):
            model_size_str = f"{entry.model_size/1e6:.1f}M" if entry.model_size < 1e9 else f"{entry.model_size/1e9:.1f}B"
            eval_acc = f"{entry.lm_eval_macro_avg_acc:.3f}" if entry.lm_eval_macro_avg_acc is not None else "N/A"
            c4_bpb = f"{entry.eval_paloma_c4_en_bpb:.3f}" if entry.eval_paloma_c4_en_bpb is not None else "N/A"
            fineweb_loss = f"{entry.eval_fineweb_edu_loss:.3f}" if entry.eval_fineweb_edu_loss is not None else "N/A"
            row = (
                f"| {i} | {entry.run_name} | {entry.compute_budget_track} | {model_size_str} | "
                f"{entry.total_training_time:.1f} | {entry.final_flops_used:.2e} | {eval_acc} | "
                f"{c4_bpb} | {fineweb_loss} |"
            )
            rows.append(row)

        return header + separator + "\n".join(rows)


def serve_leaderboard(storage_path: str, port: int = 8000):
    """
    Serve the leaderboard UI reading from storage.
    Creates a temporary JSON file for the web UI to read.
    NOTE: this can eventually be the JSON file that users upload their run info to,
    but for now this is auto-populated by us.
    """
    import http.server
    import os
    import socketserver
    from pathlib import Path

    # Initialize leaderboard
    leaderboard = Leaderboard(storage_path)

    # Ensure static/data directory exists
    static_dir = Path(__file__).parent / "static"
    data_dir = static_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Write current entries to JSON file for UI
    entries = leaderboard.get_entries()
    with open(data_dir / "runs.json", "w") as f:
        json.dump(
            [e.to_json() for e in entries],
            f,
            indent=2,
        )

    # Change to the static directory
    os.chdir(static_dir)

    # Start server
    Handler = http.server.SimpleHTTPRequestHandler
    # Allow port reuse
    socketserver.TCPServer.allow_reuse_address = True

    with socketserver.TCPServer(("", port), Handler) as httpd:
        print(f"Serving leaderboard at http://localhost:{port}")
        print(f"Reading runs from: {storage_path}")
        print("Press Ctrl+C to stop")
        try:
            print(f"Serving leaderboard at http://localhost:{port}")
            print(f"Reading runs from: {storage_path}")
            print("Press Ctrl+C to stop")
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server...")
        except OSError as e:
            if e.errno == 48:  # Address already in use
                print(f"\nError: Port {port} is already in use. Try a different port with --port <number>")
            else:
                raise e
        finally:
            httpd.server_close()
