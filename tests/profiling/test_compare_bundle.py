# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gzip
import json
from pathlib import Path

from marin.profiling.compare_bundle import run_profile_comparison_bundle
from marin.profiling.ingest import summarize_trace
from marin.profiling.tracking import RegressionThresholds, summarize_regression_history


def test_run_profile_comparison_bundle_writes_expected_outputs(tmp_path: Path) -> None:
    before_trace = tmp_path / "before.trace.json.gz"
    after_trace = tmp_path / "after.trace.json.gz"
    _write_trace(before_trace, step_durations=[100, 110, 120, 130, 140, 150], softmax_duration=60)
    _write_trace(after_trace, step_durations=[90, 100, 110, 120, 130, 140], softmax_duration=40)

    before_summary = summarize_trace(before_trace, warmup_steps=2)
    after_summary = summarize_trace(after_trace, warmup_steps=2)

    history_path = tmp_path / "history.jsonl"
    result = run_profile_comparison_bundle(
        before=before_summary,
        after=after_summary,
        output_dir=tmp_path / "bundle",
        thresholds=RegressionThresholds(),
        top_k=3,
        label="bundle-test",
        history_path=history_path,
    )

    assert result.before_summary_path.exists()
    assert result.after_summary_path.exists()
    assert result.comparison_path.exists()
    assert result.tracking_path.exists()
    assert result.before_report_path.exists()
    assert result.after_report_path.exists()
    assert result.status in {"pass", "warn", "fail"}

    history_summary = summarize_regression_history(history_path, tail=5)
    assert history_summary["num_records"] == 1
    assert history_summary["latest_record"]["label"] == "bundle-test"


def _write_trace(path: Path, *, step_durations: list[float], softmax_duration: float) -> None:
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
            {"ph": "X", "pid": 1, "tid": 2, "name": "softmax", "ts": 120, "dur": softmax_duration},
            {"ph": "X", "pid": 1, "tid": 2, "name": "dependency-wait", "ts": 190, "dur": 30},
            {"ph": "X", "pid": 2, "tid": 1, "name": "python_host_compute", "ts": 0, "dur": 80},
        ]
    )

    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)
