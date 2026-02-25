# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gzip
import json
from pathlib import Path

from marin.profiling.ingest import normalize_run_target, summarize_trace
from marin.profiling.tracking import (
    RegressionThresholds,
    append_regression_record,
    assess_profile_regression,
    load_regression_history,
    make_regression_record,
    summarize_regression_history,
)


def test_normalize_run_target_variants() -> None:
    assert normalize_run_target(
        "https://wandb.ai/marin-community/marin/runs/abc123",
        entity=None,
        project=None,
    ) == ("marin-community", "marin", "abc123")
    assert normalize_run_target("marin-community/marin/abc123", entity=None, project=None) == (
        "marin-community",
        "marin",
        "abc123",
    )
    assert normalize_run_target("abc123", entity="marin-community", project="marin") == (
        "marin-community",
        "marin",
        "abc123",
    )


def test_assess_profile_regression_and_history_tracking(tmp_path: Path) -> None:
    before_trace = tmp_path / "before_trace.json.gz"
    after_trace = tmp_path / "after_trace.json.gz"
    _write_trace(before_trace, step_durations=[100, 110, 120, 130, 140, 150])
    _write_trace(after_trace, step_durations=[130, 140, 150, 160, 170, 180])

    before = summarize_trace(before_trace, warmup_steps=2, hot_op_limit=10)
    after = summarize_trace(after_trace, warmup_steps=2, hot_op_limit=10)

    thresholds = RegressionThresholds(
        max_step_median_regression_pct=5.0,
        max_step_p90_regression_pct=5.0,
    )
    assessment = assess_profile_regression(before, after, thresholds=thresholds)
    assert assessment["status"] == "fail"
    assert len(assessment["failures"]) >= 1

    record = make_regression_record(before=before, after=after, assessment=assessment, label="unit-test")
    history_path = tmp_path / "history.jsonl"
    append_regression_record(history_path, record)

    lines = history_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    decoded = json.loads(lines[0])
    assert decoded["label"] == "unit-test"
    assert decoded["assessment"]["status"] == "fail"

    loaded = load_regression_history(history_path)
    assert len(loaded) == 1
    summary = summarize_regression_history(history_path, tail=5)
    assert summary["num_records"] == 1
    assert summary["status_counts"]["fail"] == 1


def _write_trace(path: Path, *, step_durations: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "displayTimeUnit": "ns",
        "traceEvents": [
            {"ph": "M", "pid": 1, "name": "process_name", "args": {"name": "/device:TPU:0"}},
            {"ph": "M", "pid": 1, "tid": 1, "name": "thread_name", "args": {"name": "Steps"}},
            {"ph": "M", "pid": 1, "tid": 2, "name": "thread_name", "args": {"name": "XLA Ops"}},
            {"ph": "M", "pid": 2, "name": "process_name", "args": {"name": "/host:CPU"}},
            {"ph": "M", "pid": 2, "tid": 1, "name": "thread_name", "args": {"name": "main"}},
        ],
    }

    for idx, duration in enumerate(step_durations):
        payload["traceEvents"].append(
            {"ph": "X", "pid": 1, "tid": 1, "name": str(idx), "ts": idx * 1000, "dur": duration}
        )

    payload["traceEvents"].extend(
        [
            {"ph": "X", "pid": 1, "tid": 2, "name": "fusion.1", "ts": 0, "dur": 100},
            {"ph": "X", "pid": 1, "tid": 2, "name": "all-reduce.1", "ts": 20, "dur": 40},
            {"ph": "X", "pid": 1, "tid": 2, "name": "softmax", "ts": 120, "dur": 60},
            {"ph": "X", "pid": 1, "tid": 2, "name": "dependency-wait", "ts": 190, "dur": 30},
            {"ph": "X", "pid": 2, "tid": 1, "name": "python_host_compute", "ts": 0, "dur": 80},
        ]
    )

    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)
