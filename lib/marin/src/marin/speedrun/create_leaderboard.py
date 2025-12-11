#!/usr/bin/env python3
# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Script to generate static leaderboard data."""

import argparse
import datetime
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import fsspec

CURRENT_FILE = Path(__file__).resolve()


def _find_repo_root(start_file: Path) -> Path | None:
    """Return the repository root that also contains the experiments directory if possible."""
    for parent in start_file.parents:
        if (parent / "pyproject.toml").exists() and (parent / "experiments").exists():
            return parent
    return None


REPO_ROOT = _find_repo_root(CURRENT_FILE)
DEFAULT_RESULTS_DIR = (REPO_ROOT / "experiments" / "speedrun") if REPO_ROOT else Path("experiments") / "speedrun"


@dataclass(frozen=True)
class SpeedrunAuthor:
    """Author information displayed in the leaderboard."""

    name: str
    affiliation: str
    url: str | None = None


@dataclass(frozen=True)
class LeaderboardEntry:
    # Run identification
    run_name: str
    results_filepath: str
    author: SpeedrunAuthor

    # Model/hardware specs
    model_size: int
    training_hardware_flops: float

    # Training metrics and FLOPs
    training_time: float  # in seconds
    model_flops: float

    # Optional fields
    eval_paloma_c4_en_bpb: float | None = None
    wandb_link: str | None = None
    run_completion_timestamp: datetime.datetime | None = None
    description: str | None = None


# Speedruns to exclude from the leaderboard (used for tutorials, etc.,
# or generally when, we for some reason don't want to include a run)
EXCLUDED_SPEEDRUNS = {
    "hello_world_gpu_speedrun",
}


def find_speedrun_results(base_path: str) -> list[str]:
    fs = fsspec.filesystem(base_path.split("://", 1)[0] if "://" in base_path else "file")
    pattern = f"{base_path}/**/speedrun_results.json"
    all_results = fs.glob(pattern)

    # Filter out excluded speedruns by checking the run name (directory name)
    return [path for path in all_results if Path(path).parent.name not in EXCLUDED_SPEEDRUNS]


def load_results_file(path: str) -> dict:
    fs = fsspec.filesystem(path.split("://", 1)[0] if "://" in path else "file")
    with fs.open(path, "r") as f:
        data = json.load(f)
        return {"runs": [data]} if "runs" not in data else data


def create_entry_from_results(results: dict, results_filepath: str, base_path: str) -> LeaderboardEntry:
    base_path_abs = Path(base_path).resolve()
    filepath = Path(results_filepath).resolve()
    run_dir = filepath.parent

    # Path information
    run_name = run_dir.relative_to(base_path_abs).as_posix()
    relative_path = run_dir
    if REPO_ROOT and run_dir.is_relative_to(REPO_ROOT):
        relative_path = run_dir.relative_to(REPO_ROOT)

    # Get the run info which contains all the information
    run_data = results["run_info"]

    # Training metrics and FLOPs
    training_hardware_flops = run_data["training_hardware_flops"]
    training_time = run_data["training_time"]
    eval_paloma_c4_en_bpb = run_data["eval/paloma/c4_en/bpb"]
    model_flops = run_data["model_flops"]

    # Model specification
    model_size = run_data["model_size"]

    # Run metadata
    wandb_link = run_data.get("wandb_run_link")
    run_completion_timestamp = run_data.get("run_completion_timestamp")
    description = run_data["description"]

    # Author information
    author_info = run_data["author"]
    author = SpeedrunAuthor(name=author_info["name"], affiliation=author_info["affiliation"], url=author_info.get("url"))

    return LeaderboardEntry(
        run_name=run_name,
        results_filepath=str(relative_path),
        model_size=model_size or 0,
        training_hardware_flops=training_hardware_flops or 0.0,
        training_time=training_time or 0.0,
        model_flops=model_flops,
        eval_paloma_c4_en_bpb=eval_paloma_c4_en_bpb,
        author=author,
        wandb_link=wandb_link,
        run_completion_timestamp=run_completion_timestamp,
        description=description,
    )


def get_entries(base_path: str) -> list[LeaderboardEntry]:
    entries = []
    results_files = find_speedrun_results(base_path)
    print(f"Found {len(results_files)} result files under {base_path}")

    for file_path in results_files:
        try:
            results_data = load_results_file(file_path)
            for _i, run_results in enumerate(results_data["runs"]):
                run_path = file_path
                entry = create_entry_from_results(run_results, run_path, base_path)
                entries.append(entry)
        except Exception as e:
            print(f"Failed to process {file_path}: {e}")

    return entries


def main():
    parser = argparse.ArgumentParser(description="Generate static leaderboard data")

    parser.add_argument(
        "--results-dir",
        type=str,
        default=str(DEFAULT_RESULTS_DIR),
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
