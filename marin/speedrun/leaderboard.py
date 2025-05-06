"""
Leaderboard system for Marin speedruns.
"""

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path

import fsspec
import humanfriendly

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LeaderboardEntry:
    run_name: str # name that appears on the leaderboard
    model_size: int # number of parameters; for MoEs, total number of trainable params
    total_training_time: float # sum of times for training steps, in minutes
    total_training_flops: float # estimate of total FLOPs used for training
    submitted_by: str # who submitted the run
    results_filepath: str  # path to the results JSON file
    wandb_run_id: str | None = None
    eval_paloma_c4_en_bpb: float | None = None


class Leaderboard:
    def __init__(self, base_path: str, result_files: list[str] | None = None):
        """
        base_path: path to the directory containing speedrun_results.json files
        result_files: list of paths to speedrun_results.json files
        """
        self.base_path = str(base_path).rstrip("/")
        scheme = base_path.split("://", 1)[0] if "://" in base_path else "file"
        self.fs = fsspec.filesystem(scheme)
        self.result_files = result_files

    def _find_results_files(self) -> list[str]:
        if self.result_files:
            return self.result_files
        pattern = f"{self.base_path}/**/speedrun_results.json"
        return self.fs.glob(pattern)

    def _load_results_file(self, path: str) -> dict:
        with self.fs.open(path, "r") as f:
            data = json.load(f)
            # Handle both old and new format
            if "runs" in data:
                return data
            else:
                # Convert old format to new format
                return {"runs": [data]}

    def _create_entry_from_results(self, results: dict, results_filepath: str) -> LeaderboardEntry:
        run_name = Path(results_filepath).parent.name

        actual_compute = results["run_stats"]["final_flops_estimate"]

        # Convert to float
        if isinstance(actual_compute, str):
            if "e" in actual_compute.lower():
                actual_flops = float(actual_compute.replace("e", "E"))
            else:
                actual_flops = float(actual_compute)
        else:
            actual_flops = float(actual_compute)
            
        # Log the human-friendly version for debugging
        flops_str = humanfriendly.format_number(actual_flops)
        print(f"FLOPS for {run_name}: {flops_str}")

        # Get training time (handle both field names)
        training_time = results["run_stats"].get(
            "training_time_in_minutes", results["run_stats"].get("training_time", 0.0)
        )

        # Get eval metrics
        eval_paloma_c4_en_bpb = results["run_stats"].get("eval/paloma/c4_en/bpb")

        return LeaderboardEntry(
            run_name=run_name,
            model_size=results["run_related_info"]["num_parameters"],
            total_training_time=training_time,
            total_training_flops=actual_flops,
            submitted_by=results["run_related_info"].get("submitted_by", "unknown"),
            results_filepath=results_filepath,
            wandb_run_id=results["run_related_info"].get("wandb_run_id", None),
            eval_paloma_c4_en_bpb=float(eval_paloma_c4_en_bpb) if eval_paloma_c4_en_bpb is not None else None,
        )

    def get_entries(self) -> list[LeaderboardEntry]:
        entries = []
        results_files = self._find_results_files()
        print(f"Found {len(results_files)} results files: {results_files}")

        for file_path in results_files:
            try:
                print(f"Processing file: {file_path}")
                results_data = self._load_results_file(file_path)
                print(f"Loaded results data with keys: {list(results_data.keys())}")
                print(f"Number of runs in file: {len(results_data.get('runs', []))}")

                # Process each run in the runs list
                for i, run_results in enumerate(results_data["runs"]):
                    # For multiple runs in the same file, we need to create unique paths
                    run_path = f"{file_path}#{i}" if i > 0 else file_path
                    entry = self._create_entry_from_results(run_results, run_path)
                    entries.append(entry)
            except Exception as e:
                logger.warning(f"Failed to process {file_path}: {e}")
                import traceback

                traceback.print_exc()

        print(f"Total entries found: {len(entries)}")
        return entries

    # No filtering by categories needed

    def format_leaderboard(self) -> str:
        entries = self.get_entries()

        if not entries:
            return "No entries found."

        # Sort by FLOPs used (lower is better)
        entries.sort(key=lambda x: x.total_training_flops, reverse=True)

        header = "| Rank | Run Name | Model Size | Training Time | FLOPs Used | C4-EN BPB |"
        separator = "|------|----------|------------|-------------------|-------------|---------|"

        rows = []
        for i, entry in enumerate(entries, 1):
            model_size_str = humanfriendly.format_size(entry.model_size, binary=False).replace("bytes", "params")
            training_time = humanfriendly.format_timespan(entry.total_training_time)
            flops_str = humanfriendly.format_number(entry.total_training_flops)
            c4_bpb = f"{entry.eval_paloma_c4_en_bpb:.3f}" if entry.eval_paloma_c4_en_bpb is not None else "N/A"
            row = (
                f"| {i} | {entry.run_name} | {model_size_str} | "
                f"{training_time} | {flops_str} | {c4_bpb} |"
            )
            rows.append(row)

        return header + separator + "\n".join(rows)


def serve_leaderboard(leaderboard: Leaderboard, port: int = 8000):
    """
    Serve the leaderboard UI reading from storage.
    Creates a temporary JSON file for the web UI to read.
    """
    import http.server
    import os
    import socketserver
    from pathlib import Path

    # Ensure static/data directory exists
    static_dir = Path(__file__).parent / "static"
    data_dir = static_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Write current entries to JSON file for UI
    entries = leaderboard.get_entries()
    print(f"Writing {len(entries)} entries to runs.json")

    # Convert entries to JSON
    entries_json = [asdict(e) for e in entries]

    # Write to file
    with open(data_dir / "runs.json", "w") as f:
        json.dump(entries_json, f, indent=2)

    print(f"Wrote {len(entries_json)} entries to {data_dir / 'runs.json'}")

    # Change to the static directory
    os.chdir(static_dir)

    # Start server
    Handler = http.server.SimpleHTTPRequestHandler
    # Allow port reuse
    socketserver.TCPServer.allow_reuse_address = True

    with socketserver.TCPServer(("", port), Handler) as httpd:
        try:
            print(f"Serving leaderboard at http://localhost:{port}")
            print(f"Reading runs from: {leaderboard.base_path}")
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
