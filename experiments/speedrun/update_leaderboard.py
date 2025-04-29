"""
Update and serve the Marin speedrun leaderboard.

This script manages the Marin speedrun leaderboard by collecting run results from a specified storage location
(local directory or Google Cloud Storage bucket), generating formatted leaderboard outputs, and optionally
serving them via a web UI.

Main functionalities:
- Aggregates and formats leaderboard results from experiment run directories.
- Prints the full leaderboard and per-track leaderboards (TINY, SMALL, MEDIUM) to the console.
- Can launch a web server to display the leaderboard UI for interactive viewing.

Arguments:
    --storage-path (str, required): Path to the directory or GCS bucket containing run results.
        Example: 'gs://bucket/path/to/runs' or '/path/to/runs'.
    --port (int, optional): Port to serve the leaderboard UI on (default: 8000).
    --print-only (flag): If set, only prints the leaderboard to the console and does not start the server.

Usage examples:
    python update_leaderboard.py --storage-path gs://my-bucket/speedrun/runs
    python update_leaderboard.py --storage-path ./runs --print-only

This script relies on the `Leaderboard` and `serve_leaderboard` utilities from the `leaderboard` module
for core logic and serving. It is intended to be run as a standalone script for updating and visualizing
speedrun results.
"""

import argparse

from leaderboard import Leaderboard, serve_leaderboard


def main():
    parser = argparse.ArgumentParser(description="Update and serve Marin speedrun leaderboard")
    parser.add_argument(
        "--storage-path",
        type=str,
        required=True,
        help="Storage path containing run directories (e.g., gs://bucket/path/to/runs or /path/to/runs)",
    )
    parser.add_argument("--port", type=int, default=8000, help="Port to serve the leaderboard UI on")
    parser.add_argument("--print-only", action="store_true", help="Only print the leaderboard, don't start the server")

    args = parser.parse_args()

    # Initialize leaderboard
    leaderboard = Leaderboard(args.storage_path)

    if args.print_only:
        print("\nFull Leaderboard:")
        print(leaderboard.format_leaderboard())

        print("\nTINY Track:")
        print(leaderboard.format_leaderboard("TINY"))

        print("\nSMALL Track:")
        print(leaderboard.format_leaderboard("SMALL"))

        print("\nMEDIUM Track:")
        print(leaderboard.format_leaderboard("MEDIUM"))
    else:
        # Start the web server
        serve_leaderboard(args.storage_path, args.port)


if __name__ == "__main__":
    main()
