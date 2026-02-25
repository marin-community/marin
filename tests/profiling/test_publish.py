# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gzip
import json
from pathlib import Path

from marin.profiling.ingest import summarize_trace
from marin.profiling.publish import (
    PROFILE_SUMMARY_ARTIFACT_TYPE,
    build_profile_summary_artifact_metadata,
    default_profile_summary_artifact_name,
    publish_profile_summary_artifact,
)


def test_publish_profile_summary_artifact_dry_run(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.json.gz"
    summary_path = tmp_path / "summary.json"
    report_path = tmp_path / "report.md"

    _write_trace(trace_path)
    summary = summarize_trace(trace_path)
    summary_path.write_text(summary.to_json() + "\n", encoding="utf-8")
    report_path.write_text("# report\n", encoding="utf-8")

    metadata = build_profile_summary_artifact_metadata(summary)
    assert metadata["schema_version"] == summary.schema_version
    assert metadata["source_format"] == summary.source_format

    artifact_name = default_profile_summary_artifact_name(summary)
    assert artifact_name.startswith("profile-summary-")

    response = publish_profile_summary_artifact(
        summary_path=summary_path,
        report_path=report_path,
        entity="marin-community",
        project="marin",
        artifact_name="profile-summary-unit-test",
        aliases=["latest", "unit"],
        dry_run=True,
    )

    assert response["status"] == "dry_run"
    assert response["artifact_type"] == PROFILE_SUMMARY_ARTIFACT_TYPE
    assert response["artifact_name"] == "profile-summary-unit-test"
    assert response["aliases"] == ["latest", "unit"]
    assert response["summary_path"] == str(summary_path)


def _write_trace(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "displayTimeUnit": "ns",
        "traceEvents": [
            {"ph": "M", "pid": 1, "name": "process_name", "args": {"name": "/device:TPU:0"}},
            {"ph": "M", "pid": 1, "tid": 1, "name": "thread_name", "args": {"name": "Steps"}},
            {"ph": "M", "pid": 1, "tid": 2, "name": "thread_name", "args": {"name": "XLA Ops"}},
            {"ph": "M", "pid": 2, "name": "process_name", "args": {"name": "/host:CPU"}},
            {"ph": "M", "pid": 2, "tid": 1, "name": "thread_name", "args": {"name": "main"}},
            {"ph": "X", "pid": 1, "tid": 1, "name": "0", "ts": 0, "dur": 100},
            {"ph": "X", "pid": 1, "tid": 2, "name": "fusion.1", "ts": 0, "dur": 100},
            {"ph": "X", "pid": 1, "tid": 2, "name": "all-reduce.1", "ts": 20, "dur": 40},
            {"ph": "X", "pid": 1, "tid": 2, "name": "dependency-wait", "ts": 190, "dur": 30},
            {"ph": "X", "pid": 2, "tid": 1, "name": "python_host_compute", "ts": 0, "dur": 80},
        ],
    }

    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)
