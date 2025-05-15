#!/usr/bin/env python3
"""Script to generate static leaderboard data."""

import argparse
import datetime
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import fsspec


@dataclass(frozen=True)
class LeaderboardEntry:
    run_name: str
    model_size: int
    total_training_time: float
    total_training_flops: float
    submitted_by: str
    results_filepath: str
    wandb_link: str | None = None
    eval_paloma_c4_en_bpb: float | None = None
    run_timestamp: datetime.datetime | None = None
    contributor_name: str | None = None
    contributor_affiliation: str | None = None
    description: str | None = None


def find_speedrun_results(base_path: str) -> list[str]:
    fs = fsspec.filesystem(base_path.split("://", 1)[0] if "://" in base_path else "file")
    pattern = f"{base_path}/**/speedrun_results.json"
    return fs.glob(pattern)


def load_results_file(path: str) -> dict:
    fs = fsspec.filesystem(path.split("://", 1)[0] if "://" in path else "file")
    with fs.open(path, "r") as f:
        data = json.load(f)
        return {"runs": [data]} if "runs" not in data else data


def create_entry_from_results(results: dict, results_filepath: str) -> LeaderboardEntry:
    run_name = Path(results_filepath).parent.name

    # Extract from speedrun_config
    speedrun_config = results.get("speedrun_config", {})
    model_size = speedrun_config.get("num_parameters")
    contributor_name = speedrun_config.get("contributor_name", "marin-community")
    contributor_affiliation = speedrun_config.get("contributor_affiliation")
    description = speedrun_config.get("description")

    # Extract from speedrun_results
    speedrun_results = results.get("speedrun_results", {})
    total_training_flops = speedrun_results.get("total_training_flops")
    training_time = speedrun_results.get("training_time_in_minutes")
    eval_paloma_c4_en_bpb = speedrun_results.get("eval/paloma/c4_en/bpb")
    wandb_link = speedrun_results.get("wandb_run_link")
    run_timestamp_str = speedrun_results.get("run_completion_timestamp")

    # submitted_by will be the contributor_name from the config
    submitted_by = contributor_name
    
    # Convert timestamp string to datetime object if present
    run_timestamp = None
    if run_timestamp_str:
        try:
            run_timestamp = datetime.datetime.strptime(run_timestamp_str, "%Y-%m-%d %H:%M:%S UTC")
        except ValueError:
            print(f"Warning: Could not parse timestamp '{run_timestamp_str}' for {run_name}. Leaving as None.")
            run_timestamp = None # Or keep as string if preferred, but dataclass expects datetime

    return LeaderboardEntry(
        run_name=run_name,
        model_size=int(model_size) if model_size is not None else 0, # Ensure int
        total_training_time=float(training_time) if training_time is not None else 0.0,
        total_training_flops=float(total_training_flops) if total_training_flops is not None else 0.0,
        submitted_by=submitted_by,
        results_filepath=results_filepath,
        wandb_link=wandb_link,
        eval_paloma_c4_en_bpb=float(eval_paloma_c4_en_bpb) if eval_paloma_c4_en_bpb is not None else None,
        run_timestamp=run_timestamp, # Already a datetime object or None
        contributor_name=contributor_name,
        contributor_affiliation=contributor_affiliation,
        description=description,
    )


def get_entries(base_path: str) -> list[LeaderboardEntry]:
    entries = []
    results_files = find_speedrun_results(base_path)

    for file_path in results_files:
        try:
            results_data = load_results_file(file_path)
            for _i, run_results in enumerate(results_data["runs"]):
                run_path = file_path
                entry = create_entry_from_results(run_results, run_path)
                entries.append(entry)
        except Exception as e:
            print(f"Failed to process {file_path}: {e}")

    return entries


def main():
    parser = argparse.ArgumentParser(description="Generate static leaderboard data")
    current_file = Path(__file__).resolve()
    marin_root = current_file.parent.parent.parent
    experiments_path = marin_root / "experiments" / "speedrun"

    parser.add_argument(
        "--results-dir",
        type=str,
        default=experiments_path,
        help="Storage directory containing runs, which then contain JSONs (usually experiments/speedrun/)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="static/data/runs.json",
        help="Output path for the leaderboard data",
    )
    args = parser.parse_args()

    entries = get_entries(str(args.results_dir))
    entries_json = [asdict(e) for e in entries]

    output_path = Path(__file__).parent / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(entries_json, f, indent=2)
    print(f"Wrote {len(entries)} entries to {output_path}")


if __name__ == "__main__":
    main()
