"""
When we run HELM to get MMLU results, HELM does not provide a direct way to get the average exact match.
This script post-processes the results files to calculate the final MMLU metric.

Usage: python3 process_mmlu_results.py --results-path <path to the results folder (local or GCS)>

Example: python3 process_mmlu_results.py
            --results-path "gs://marin-us-central2/evaluation/helm/dclm_1b_1x_replication_oct26-a28b1e/step-54931-d64b9b"
"""

import json
import os
import tempfile

from utils import download_from_gcs, is_remote_path


def process_mmlu_results_local(local_results_path: str) -> float:

    # make sure it is a local path
    assert not is_remote_path(local_results_path), "Local path expected."

    total_exact_match = 0
    count = 0  # Track the number of subjects
    for run_folder in os.listdir(local_results_path):

        run_path: str = os.path.join(local_results_path, run_folder)
        if not os.path.isdir(run_path) or not run_folder.startswith("mmlu"):
            continue

        stats_path: str = os.path.join(run_path, "stats.json")
        assert os.path.exists(stats_path), f"Stats file not found: {stats_path}"

        subject: str = run_folder.split(",")[0].replace("mmlu:subject=", "")

        with open(stats_path, "r") as f:
            stats: list = json.load(f)
            for stat in stats:
                if stat["name"] == {"name": "exact_match", "split": "test"}:
                    exact_match: float = stat["sum"]

                    # Display as a percentage to 2 decimal places
                    exact_match_percentage = round(exact_match * 100, 2)
                    print(f"{subject}: {exact_match_percentage}%")
                    total_exact_match += exact_match_percentage
                    count += 1
                    break

    if count > 0:
        average_exact_match = round(total_exact_match / count, 2)
        print(f"Average Exact Match: {average_exact_match}%")
    else:
        average_exact_match = 0.0
        print("No results found.")

    return average_exact_match


def compute_and_print_mmlu_stats(results_path: str) -> float:

    # results_path may be a local path or a GCS path; handle both cases
    if is_remote_path(results_path):
        with tempfile.TemporaryDirectory() as temp_dir:
            download_from_gcs(results_path, temp_dir)
            results_path = temp_dir
            return process_mmlu_results_local(results_path)
    else:
        return process_mmlu_results_local(results_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-path",
        help="Where the HELM mmlu results are",
        default="mmlu",
    )

    args = parser.parse_args()

    compute_and_print_mmlu_stats(args.results_path)
