"""
Updates and serves the Marin speedrun leaderboard.
"""

import argparse
import glob
import os
from pathlib import Path
from marin.speedrun.leaderboard import Leaderboard, serve_leaderboard

def find_speedrun_results(base_path):
    print(f"Searching for speedrun_results.json files in: {base_path}")
    result_files = glob.glob(f"{base_path}/**/speedrun_results.json", recursive=True)
    print(f"Found {len(result_files)} files: {result_files}")
    return result_files

def main():
    parser = argparse.ArgumentParser(description="Updates and serves the Marin speedrun leaderboard")

    # this just finds the path to the experiments/speedrun directory
    current_file = Path(__file__).resolve()
    marin_root = current_file.parent.parent.parent
    experiments_path = marin_root / "experiments" / "speedrun"

    parser.add_argument(
        "--storage-path",
        type=str,
        default=experiments_path,
        help="Storage path containing run directories (e.g., gs://bucket/path/to/runs or /path/to/runs)",
    )
    parser.add_argument("--port", type=int, default=8000, help="Port to serve the leaderboard UI on")
    parser.add_argument("--print-only", action="store_true", help="Only print the leaderboard, don't start the server")

    args = parser.parse_args()
    
    result_files = find_speedrun_results(args.storage_path)
    storage_path = str(args.storage_path) if isinstance(args.storage_path, Path) else args.storage_path
    leaderboard = Leaderboard(storage_path, result_files=result_files)

    print("\nSpeedrun Leaderboard")
    print("===================\n")
    print(leaderboard.format_leaderboard())

    if not args.print_only:
        serve_leaderboard(leaderboard, port=args.port)

if __name__ == "__main__":
    main()
