# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Canary ferry regression gate: validate training metrics against thresholds.

Reads tracker_metrics.jsonl from the canary ferry's GCS output directory and
checks final metrics against hardcoded thresholds. Exits non-zero if any
threshold is breached. Intended to run as a post-training step in the
marin-canary-ferry GitHub Actions workflow.
"""

import json
import operator
import sys

import fsspec

from marin.execution.executor import Executor

MARIN_PREFIX = "gs://marin-us-central1"

THRESHOLDS = [
    # (summary_key, display_name, check, threshold)
    # check(actual, threshold) → True means passing
    ("train/loss", "Final loss", operator.le, 4.0),
    ("throughput/mfu", "MFU (%)", operator.ge, 25.0),
    ("_step", "Steps completed", operator.ge, 3000),
    ("_runtime", "Wall-clock (s)", operator.le, 3600),
]


def resolve_canary_output_path() -> str:
    """Resolve the canary ferry's GCS output path via the executor's version hash."""
    from experiments.ferries.canary_ferry import make_training_step

    training_step = make_training_step()
    executor = Executor(
        prefix=MARIN_PREFIX,
        executor_info_base_path=f"{MARIN_PREFIX}/experiments",
    )
    executor.compute_version(training_step, is_pseudo_dep=False)
    return executor.output_paths[training_step]


def read_summary(output_path: str) -> dict:
    """Read the summary dict from tracker_metrics.jsonl on GCS."""
    metrics_file = f"{output_path}/tracker_metrics.jsonl"
    with fsspec.open(metrics_file, "r") as f:
        record = json.loads(f.read().strip())
    return record["summary"]


def lookup_metric(summary: dict, key: str):
    """Look up a slash-separated metric key in a potentially nested dict.

    WandB metrics like "train/loss" may be stored nested as {"train": {"loss": val}}
    in the replicated summary. Falls back to flat key lookup.
    """
    parts = key.split("/")
    current = summary
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return summary.get(key)
    return current


def validate_metrics(summary: dict) -> list[tuple[str, float | None, float, bool]]:
    """Check each metric against its threshold. Returns (name, actual, threshold, passed)."""
    results = []
    for summary_key, display_name, check, threshold in THRESHOLDS:
        actual = lookup_metric(summary, summary_key)
        if actual is None:
            results.append((display_name, None, threshold, False))
        else:
            actual = float(actual)
            passed = check(actual, threshold)
            results.append((display_name, actual, threshold, passed))
    return results


def print_report(results: list[tuple[str, float | None, float, bool]]) -> None:
    """Print a human-readable pass/fail table to stdout."""
    print(f"\n{'Metric':<20} {'Actual':>12} {'Threshold':>12} {'Status':>8}")
    print("-" * 56)
    for display_name, actual, threshold, passed in results:
        actual_str = f"{actual:.4f}" if actual is not None else "MISSING"
        status = "PASS" if passed else "FAIL"
        print(f"{display_name:<20} {actual_str:>12} {threshold:>12.4f} {status:>8}")
    print()


def main():
    output_path = resolve_canary_output_path()
    print(f"Canary output path: {output_path}")

    summary = read_summary(output_path)
    results = validate_metrics(summary)
    print_report(results)

    if any(not passed for _, _, _, passed in results):
        print("FAILED: One or more metrics breached thresholds.")
        sys.exit(1)
    print("PASSED: All metrics within thresholds.")


if __name__ == "__main__":
    main()
