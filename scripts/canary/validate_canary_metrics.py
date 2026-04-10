# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Canary ferry regression gate: validate training metrics against thresholds.

Reads tracker_metrics.jsonl from the canary ferry's GCS output directory and
checks final metrics against coarse thresholds. Exits non-zero if any
threshold is breached. Intended to run as a post-training step in the
marin-canary-ferry GitHub Actions workflow.

Thresholds are deliberately loose — they catch total failures (job hung,
loss exploded, zero throughput) not quality regressions. Tighten after
calibrating from several successful MoE canary runs.
"""

import json
import operator
import os
import sys
from collections.abc import Callable

from rigging.filesystem import open_url

from marin.execution.executor import Executor


def _env_float(key: str, default: float) -> float:
    raw = os.environ.get(key, "")
    return float(raw) if raw else default


def _thresholds() -> list[tuple[str, str, Callable[[float, float], bool], float]]:
    return [
        ("_step", "Steps completed", operator.ge, _env_float("CANARY_MIN_STEPS", 400)),
        ("train/loss", "Final loss", operator.le, _env_float("CANARY_MAX_LOSS", 8.0)),
    ]


def resolve_canary_output_path() -> str:
    """Resolve the canary ferry's output path via the executor's version hash.

    Uses mirror:// so the read works regardless of which region the canary
    wrote to.
    """
    from experiments.ferries.canary_ferry import canary_moe_step

    executor = Executor(
        prefix="mirror://",
        executor_info_base_path="mirror://experiments",
    )
    executor.compute_version(canary_moe_step, is_pseudo_dep=False)
    return executor.output_paths[canary_moe_step]


def read_summary(output_path: str) -> dict:
    """Read the summary dict from tracker_metrics.jsonl."""
    metrics_file = f"{output_path}/tracker_metrics.jsonl"
    with open_url(metrics_file, "r") as f:
        record = json.loads(f.read().strip())
    return record["summary"]


def lookup_metric(summary: dict, key: str):
    """Look up a slash-separated metric key in a potentially nested dict."""
    parts = key.split("/")
    current = summary
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return summary.get(key)
    return current


def validate_metrics(summary: dict) -> list[tuple[str, float | None, float, bool]]:
    """Returns (display_name, actual_value, threshold, passed) per metric."""
    results = []
    for summary_key, display_name, check, threshold in _thresholds():
        actual = lookup_metric(summary, summary_key)
        if actual is None:
            results.append((display_name, None, threshold, False))
        else:
            actual = float(actual)
            passed = check(actual, threshold)
            results.append((display_name, actual, threshold, passed))
    return results


def print_report(results: list[tuple[str, float | None, float, bool]]) -> None:
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
