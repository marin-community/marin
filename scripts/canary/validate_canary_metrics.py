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
from dataclasses import dataclass

from marin.execution.executor import Executor
from rigging.filesystem import open_url

BYTES_PER_GIB = 1024**3
STATUS_FAIL = "FAIL"
STATUS_PASS = "PASS"
STATUS_SKIP = "SKIP"
STATUS_WARN = "WARN"


def _env_float(key: str, default: float) -> float:
    raw = os.environ.get(key, "")
    return float(raw) if raw else default


def _env_float_or_none(key: str) -> float | None:
    raw = os.environ.get(key, "")
    return float(raw) if raw else None


@dataclass(frozen=True)
class ValidationResult:
    display_name: str
    actual: float | None
    threshold: float | None
    status: str
    detail: str = ""


def _required_thresholds() -> list[tuple[str, str, Callable[[float, float], bool], float]]:
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


def _lookup_first_float(summary: dict, *keys: str) -> float | None:
    for key in keys:
        actual = lookup_metric(summary, key)
        if actual is not None:
            return float(actual)
    return None


def _lookup_gib(summary: dict, *, gib_keys: tuple[str, ...], byte_keys: tuple[str, ...]) -> float | None:
    value_gib = _lookup_first_float(summary, *gib_keys)
    if value_gib is not None:
        return value_gib

    value_bytes = _lookup_first_float(summary, *byte_keys)
    if value_bytes is None:
        return None
    return value_bytes / BYTES_PER_GIB


def _threshold_status(actual: float, *, warn_threshold: float, fail_threshold: float) -> tuple[str, float]:
    if actual > fail_threshold:
        return STATUS_FAIL, fail_threshold
    if actual > warn_threshold:
        return STATUS_WARN, warn_threshold
    return STATUS_PASS, warn_threshold


def _validate_required_metrics(summary: dict) -> list[ValidationResult]:
    results = []
    for summary_key, display_name, check, threshold in _required_thresholds():
        actual = lookup_metric(summary, summary_key)
        if actual is None:
            results.append(ValidationResult(display_name, None, threshold, STATUS_FAIL, "required metric missing"))
        else:
            actual = float(actual)
            passed = check(actual, threshold)
            results.append(ValidationResult(display_name, actual, threshold, STATUS_PASS if passed else STATUS_FAIL))
    return results


def validate_memory_metrics(summary: dict) -> list[ValidationResult]:
    """Validate canary memory metrics when present.

    Older canary summaries do not contain these metrics; missing memory metrics
    are skipped so the existing loss/step gate remains backward compatible.
    """
    results: list[ValidationResult] = []

    compile_temp_gib = _lookup_gib(
        summary,
        gib_keys=("canary/compile/temp_gib", "canary/compile/temp_size_gib"),
        byte_keys=("canary/compile/temp_bytes", "canary/compile/temp_size_in_bytes"),
    )
    if compile_temp_gib is None:
        results.append(ValidationResult("Compile temp memory", None, None, STATUS_SKIP, "metric absent"))
    else:
        warn_threshold = _env_float("CANARY_COMPILE_TEMP_WARN_GIB", 60.0)
        fail_threshold = _env_float("CANARY_COMPILE_TEMP_FAIL_GIB", 68.0)
        status, threshold = _threshold_status(
            compile_temp_gib,
            warn_threshold=warn_threshold,
            fail_threshold=fail_threshold,
        )
        results.append(ValidationResult("Compile temp memory", compile_temp_gib, threshold, status, "GiB"))

    compile_temp_ratio = _lookup_first_float(
        summary,
        "canary/compile/temp_baseline_ratio",
        "canary/compile/temp_ratio",
    )
    if compile_temp_ratio is None and compile_temp_gib is not None:
        baseline_gib = _env_float_or_none("CANARY_COMPILE_TEMP_BASELINE_GIB")
        if baseline_gib is not None and baseline_gib > 0:
            compile_temp_ratio = compile_temp_gib / baseline_gib

    if compile_temp_ratio is None:
        results.append(ValidationResult("Compile temp ratio", None, None, STATUS_SKIP, "baseline absent"))
    else:
        warn_threshold = _env_float("CANARY_COMPILE_TEMP_RATIO_WARN", 1.15)
        fail_threshold = _env_float("CANARY_COMPILE_TEMP_RATIO_FAIL", 1.30)
        status, threshold = _threshold_status(
            compile_temp_ratio,
            warn_threshold=warn_threshold,
            fail_threshold=fail_threshold,
        )
        results.append(ValidationResult("Compile temp ratio", compile_temp_ratio, threshold, status))

    gpu_peak_gib = _lookup_gib(
        summary,
        gib_keys=("canary/gpu/peak_used_gib", "canary/gpu/peak_memory_used_gib"),
        byte_keys=("canary/gpu/peak_used_bytes", "canary/gpu/peak_memory_used_bytes"),
    )
    if gpu_peak_gib is None:
        results.append(ValidationResult("GPU peak used", None, None, STATUS_SKIP, "metric absent"))
    else:
        warn_threshold = _env_float("CANARY_GPU_PEAK_WARN_GIB", 70.0)
        fail_threshold = _env_float("CANARY_GPU_PEAK_FAIL_GIB", 76.0)
        status, threshold = _threshold_status(
            gpu_peak_gib,
            warn_threshold=warn_threshold,
            fail_threshold=fail_threshold,
        )
        results.append(ValidationResult("GPU peak used", gpu_peak_gib, threshold, status, "GiB"))

    return results


def validate_metrics(summary: dict) -> list[ValidationResult]:
    return [*_validate_required_metrics(summary), *validate_memory_metrics(summary)]


def print_report(results: list[ValidationResult]) -> None:
    print(f"\n{'Metric':<24} {'Actual':>12} {'Threshold':>12} {'Status':>8}  Detail")
    print("-" * 72)
    for result in results:
        display_name = result.display_name
        actual = result.actual
        threshold = result.threshold
        status = result.status
        detail = result.detail
        actual_str = f"{actual:.4f}" if actual is not None else "MISSING"
        threshold_str = f"{threshold:.4f}" if threshold is not None else "-"
        print(f"{display_name:<24} {actual_str:>12} {threshold_str:>12} {status:>8}  {detail}")
    print()


def main():
    output_path = resolve_canary_output_path()
    print(f"Canary output path: {output_path}")

    summary = read_summary(output_path)
    results = validate_metrics(summary)
    print_report(results)

    if any(result.status == STATUS_FAIL for result in results):
        print("FAILED: One or more metrics breached thresholds.")
        sys.exit(1)
    if any(result.status == STATUS_WARN for result in results):
        print("PASSED WITH WARNINGS: Memory metrics crossed warning thresholds.")
        return
    print("PASSED: All required metrics within thresholds.")


if __name__ == "__main__":
    main()
