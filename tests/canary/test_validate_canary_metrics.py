# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from scripts.canary.validate_canary_metrics import STATUS_FAIL, STATUS_SKIP, STATUS_WARN, validate_metrics


def _statuses_by_name(summary: dict) -> dict[str, str]:
    return {result.display_name: result.status for result in validate_metrics(summary)}


def test_missing_memory_metrics_are_skipped():
    statuses = _statuses_by_name({"_step": 400, "train/loss": 5.0})

    assert statuses["Steps completed"] != STATUS_FAIL
    assert statuses["Final loss"] != STATUS_FAIL
    assert statuses["Compile temp memory"] == STATUS_SKIP
    assert statuses["Compile temp ratio"] == STATUS_SKIP
    assert statuses["GPU peak used"] == STATUS_SKIP


def test_memory_metrics_warn_before_fail():
    statuses = _statuses_by_name(
        {
            "_step": 400,
            "train/loss": 5.0,
            "canary/compile/temp_gib": 61.0,
            "canary/compile/temp_baseline_ratio": 1.2,
            "canary/gpu/peak_used_gib": 71.0,
        }
    )

    assert statuses["Compile temp memory"] == STATUS_WARN
    assert statuses["Compile temp ratio"] == STATUS_WARN
    assert statuses["GPU peak used"] == STATUS_WARN


def test_memory_metrics_fail_at_hard_thresholds():
    statuses = _statuses_by_name(
        {
            "_step": 400,
            "train/loss": 5.0,
            "canary/compile/temp_gib": 69.0,
            "canary/compile/temp_baseline_ratio": 1.31,
            "canary/gpu/peak_used_gib": 77.0,
        }
    )

    assert statuses["Compile temp memory"] == STATUS_FAIL
    assert statuses["Compile temp ratio"] == STATUS_FAIL
    assert statuses["GPU peak used"] == STATUS_FAIL


def test_compile_ratio_can_use_env_baseline(monkeypatch):
    monkeypatch.setenv("CANARY_COMPILE_TEMP_BASELINE_GIB", "50")

    statuses = _statuses_by_name(
        {
            "_step": 400,
            "train/loss": 5.0,
            "canary/compile/temp_gib": 60.0,
        }
    )

    assert statuses["Compile temp ratio"] == STATUS_WARN
