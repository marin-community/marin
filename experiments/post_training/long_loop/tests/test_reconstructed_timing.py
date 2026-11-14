# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from experiments.post_training.long_loop.reconstruct_timing import attempt_measurements
from experiments.post_training.long_loop.timing import durable_event_log


def test_running_occupancy_excludes_queue_and_missing_finish_stays_unknown():
    attempts = [
        {
            "job_id": "train",
            "task_id": "train/0",
            "attempt_id": "0",
            "submitted_at_ms": "0",
            "created_at_ms": "3600000",
            "started_at_ms": "7200000",
            "finished_at_ms": "10800000",
        },
        {
            "job_id": "train",
            "task_id": "train/1",
            "attempt_id": "0",
            "submitted_at_ms": "0",
            "created_at_ms": "3600000",
            "started_at_ms": "7200000",
            "finished_at_ms": "",
        },
    ]
    rows = attempt_measurements(attempts, [{"job_id": "train", "resources": {"device": {"count": 8}}}])
    assert rows[0]["submit_to_attempt_seconds"] == 3600
    assert rows[0]["attempt_to_running_seconds"] == 3600
    assert rows[0]["running_gpu_hours"] == 8
    assert rows[0]["attempt_envelope_gpu_hours"] == 16
    assert rows[1]["running_gpu_hours"] is None
    assert rows[1]["attempt_envelope_gpu_hours"] is None


def test_worker_timing_is_uploaded_when_phase_fails(tmp_path, monkeypatch):
    task_outputs = tmp_path / "task-outputs"
    task_outputs.mkdir()
    monkeypatch.setenv("IRIS_OUTPUT_DIR", str(task_outputs))
    destination = tmp_path / "durable/events.jsonl"
    with pytest.raises(RuntimeError, match="work failed"):
        with durable_event_log(str(destination), "run", "export", "attempt") as log:
            with log.span("upload"):
                raise RuntimeError("work failed")
    events = [json.loads(line) for line in destination.read_text().splitlines()]
    assert [row["event"] for row in events] == ["start", "end"]
    assert events[-1]["status"] == "failed"
    assert events[-1]["monotonic"] >= events[0]["monotonic"]
    journals = list(task_outputs.glob("native-timing-*/events.jsonl"))
    assert len(journals) == 1
    assert journals[0].read_bytes() == destination.read_bytes()
