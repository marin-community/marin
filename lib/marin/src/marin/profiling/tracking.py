# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Regression tracking utilities for profile-summary before/after loops."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from marin.profiling.query import compare_profile_summaries
from marin.profiling.schema import ProfileSummary


@dataclass(frozen=True)
class RegressionThresholds:
    """
    Thresholds for classifying before/after profile comparisons.

    The percentage fields are expressed as percentages (for example `5.0` means 5%).
    """

    max_step_median_regression_pct: float = 5.0
    max_step_p90_regression_pct: float = 10.0
    max_communication_share_regression_abs: float = 0.05
    max_stall_share_regression_abs: float = 0.05


def assess_profile_regression(
    before: ProfileSummary,
    after: ProfileSummary,
    *,
    thresholds: RegressionThresholds,
    top_k: int = 10,
) -> dict[str, Any]:
    """
    Compare two summaries and classify the result as `pass`, `warn`, or `fail`.

    Classification rules:
    - `fail` if step-time median or p90 regresses beyond threshold.
    - `warn` if communication/stall share increases beyond threshold.
    - otherwise `pass`.
    """
    comparison = compare_profile_summaries(before, after, top_k=top_k)
    step_time = comparison["step_time"]

    median_before = _as_float(step_time.get("steady_state_median_before"))
    median_after = _as_float(step_time.get("steady_state_median_after"))
    p90_before = _as_float(step_time.get("steady_state_p90_before"))
    p90_after = _as_float(step_time.get("steady_state_p90_after"))

    median_regression_pct = _pct_delta(median_before, median_after)
    p90_regression_pct = _pct_delta(p90_before, p90_after)

    time_breakdown_delta = comparison["time_breakdown_share_delta"]
    comm_share_delta = _as_float(time_breakdown_delta.get("communication"))
    stall_share_delta = _as_float(time_breakdown_delta.get("stall"))

    failures: list[str] = []
    warnings: list[str] = []

    if median_regression_pct is not None and median_regression_pct > thresholds.max_step_median_regression_pct:
        failures.append(
            f"Steady-state median step time regressed by {median_regression_pct:.2f}% "
            f"(threshold {thresholds.max_step_median_regression_pct:.2f}%)."
        )

    if p90_regression_pct is not None and p90_regression_pct > thresholds.max_step_p90_regression_pct:
        failures.append(
            f"Steady-state p90 step time regressed by {p90_regression_pct:.2f}% "
            f"(threshold {thresholds.max_step_p90_regression_pct:.2f}%)."
        )

    if comm_share_delta is not None and comm_share_delta > thresholds.max_communication_share_regression_abs:
        warnings.append(
            f"Communication share increased by {comm_share_delta:.4f} "
            f"(threshold {thresholds.max_communication_share_regression_abs:.4f})."
        )

    if stall_share_delta is not None and stall_share_delta > thresholds.max_stall_share_regression_abs:
        warnings.append(
            f"Stall share increased by {stall_share_delta:.4f} "
            f"(threshold {thresholds.max_stall_share_regression_abs:.4f})."
        )

    if failures:
        status = "fail"
    elif warnings:
        status = "warn"
    else:
        status = "pass"

    return {
        "status": status,
        "failures": failures,
        "warnings": warnings,
        "metrics": {
            "steady_state_median_regression_pct": median_regression_pct,
            "steady_state_p90_regression_pct": p90_regression_pct,
            "communication_share_delta": comm_share_delta,
            "stall_share_delta": stall_share_delta,
        },
        "comparison": comparison,
    }


def make_regression_record(
    *,
    before: ProfileSummary,
    after: ProfileSummary,
    assessment: dict[str, Any],
    label: str | None = None,
) -> dict[str, Any]:
    """Create a JSON-serializable regression record for history tracking."""
    return {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "label": label,
        "before": {
            "source_path": before.source_path,
            "run_path": before.run_metadata.run_path,
            "run_id": before.run_metadata.run_id,
            "artifact_ref": before.run_metadata.artifact_ref,
        },
        "after": {
            "source_path": after.source_path,
            "run_path": after.run_metadata.run_path,
            "run_id": after.run_metadata.run_id,
            "artifact_ref": after.run_metadata.artifact_ref,
        },
        "assessment": assessment,
    }


def append_regression_record(history_path: Path, record: dict[str, Any]) -> None:
    """Append a regression record to a JSONL history file."""
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True))
        handle.write("\n")


def load_regression_history(history_path: Path) -> list[dict[str, Any]]:
    """Load regression records from a JSONL history file."""
    if not history_path.exists():
        return []

    records: list[dict[str, Any]] = []
    for line in history_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        decoded = json.loads(line)
        if isinstance(decoded, dict):
            records.append(decoded)
    return records


def summarize_regression_history(history_path: Path, *, tail: int = 20) -> dict[str, Any]:
    """
    Summarize a regression-tracking history JSONL file.

    Args:
        history_path: Path to JSONL file produced by `append_regression_record`.
        tail: Number of most recent records to include in the detailed tail.
    """
    records = load_regression_history(history_path)
    status_counter: Counter[str] = Counter()
    failure_counter: Counter[str] = Counter()
    warning_counter: Counter[str] = Counter()

    for record in records:
        assessment = record.get("assessment")
        if not isinstance(assessment, dict):
            continue
        status = assessment.get("status")
        if isinstance(status, str):
            status_counter[status] += 1
        for failure in assessment.get("failures", []):
            if isinstance(failure, str):
                failure_counter[failure] += 1
        for warning in assessment.get("warnings", []):
            if isinstance(warning, str):
                warning_counter[warning] += 1

    tail_records = records[-tail:] if tail > 0 else []

    return {
        "history_path": str(history_path),
        "num_records": len(records),
        "status_counts": dict(status_counter),
        "top_failures": [{"message": message, "count": count} for message, count in failure_counter.most_common(10)],
        "top_warnings": [{"message": message, "count": count} for message, count in warning_counter.most_common(10)],
        "latest_record": records[-1] if records else None,
        "recent_records": tail_records,
    }


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _pct_delta(before: float | None, after: float | None) -> float | None:
    if before is None or after is None:
        return None
    if before <= 0:
        return None
    return ((after - before) / before) * 100.0
