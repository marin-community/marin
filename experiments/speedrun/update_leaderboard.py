"""
Update and serve the Marin speedrun leaderboard.
"""

from leaderboard import Leaderboard, serve_leaderboard
import argparse

def main():
    parser = argparse.ArgumentParser(description="Update and serve Marin speedrun leaderboard")
    parser.add_argument(
        "--storage-path",
        type=str,
        required=True,
        help="Storage path containing run directories (e.g., gs://bucket/path/to/runs or /path/to/runs)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to serve the leaderboard UI on"
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Only print the leaderboard, don't start the server"
    )
    
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
