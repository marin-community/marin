# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from iris.rpc import job_pb2, time_pb2

from marin.mcp.babysitter import (
    classify_diagnosis,
    parse_zephyr_progress,
    task_status_to_json,
)


def _timestamp(epoch_ms: int):
    return time_pb2.Timestamp(epoch_ms=epoch_ms)


def test_task_status_json_includes_attempts_timestamps_and_usage():
    task = job_pb2.TaskStatus(
        task_id="/alice/train/0",
        state=job_pb2.TASK_STATE_FAILED,
        worker_id="worker-a",
        worker_address="worker-a:1234",
        exit_code=137,
        error="OOMKilled",
        started_at=_timestamp(1_000),
        finished_at=_timestamp(2_500),
        current_attempt_id=1,
        pending_reason="",
        can_be_scheduled=True,
        resource_usage=job_pb2.ResourceUsage(
            memory_mb=2048,
            memory_peak_mb=4096,
            cpu_millicores=1500,
            disk_mb=512,
            process_count=4,
        ),
        attempts=[
            job_pb2.TaskAttempt(
                attempt_id=0,
                worker_id="worker-old",
                state=job_pb2.TASK_STATE_PREEMPTED,
                exit_code=143,
                error="preempted",
                started_at=_timestamp(100),
                finished_at=_timestamp(900),
                is_worker_failure=True,
            ),
            job_pb2.TaskAttempt(
                attempt_id=1,
                worker_id="worker-a",
                state=job_pb2.TASK_STATE_FAILED,
                exit_code=137,
                error="OOMKilled",
                started_at=_timestamp(1_000),
                finished_at=_timestamp(2_500),
            ),
        ],
    )

    payload = task_status_to_json(task)

    assert payload["task_id"] == "/alice/train/0"
    assert payload["state"] == "failed"
    assert payload["exit_code"] == 137
    assert payload["started_at_ms"] == 1_000
    assert payload["finished_at_ms"] == 2_500
    assert payload["duration_ms"] == 1_500
    assert payload["resource_usage"]["memory_peak_mb"] == 4096
    assert payload["attempts"][0]["state"] == "preempted"
    assert payload["attempts"][0]["is_worker_failure"] is True
    assert payload["attempts"][1]["exit_code"] == 137


def test_parse_zephyr_progress_keeps_latest_stage_snapshot():
    lines = [
        "noise: pull_task worker-7",
        "[stage0-Map -> Scatter] 12/20 complete, 3 in-flight, 5 queued, 8/9 workers alive, 1 dead",
        "[stage1-Reduce] 4/10 complete, 1 in-flight, 5 queued, 8/8 workers alive, 0 dead",
        "[stage0-Map -> Scatter] 15/20 complete, 2 in-flight, 3 queued, 8/9 workers alive, 1 dead",
    ]

    progress = parse_zephyr_progress(lines)

    assert len(progress) == 2
    assert progress[0] == {
        "stage": "stage0-Map -> Scatter",
        "completed": 15,
        "total": 20,
        "in_flight": 2,
        "queued": 3,
        "workers_alive": 8,
        "workers_total": 9,
        "workers_dead": 1,
    }
    assert progress[1]["stage"] == "stage1-Reduce"


def test_classify_diagnosis_reports_common_babysitting_signals():
    job = {
        "state": "failed",
        "error": "Terminated by user",
        "failure_count": 3,
        "preemption_count": 1,
        "pending_reason": "Quota exceeded for v5litepod",
        "tasks": [
            {
                "task_id": "/alice/train/0",
                "state": "failed",
                "exit_code": 137,
                "error": "container OOMKilled",
                "pending_reason": "",
                "attempts": [{"attempt_id": 0}, {"attempt_id": 1}, {"attempt_id": 2}],
            }
        ],
    }
    logs = [
        {"task_id": "/alice/train/0", "data": "RESOURCE_EXHAUSTED: TPU quota exceeded"},
        {"task_id": "/alice/train/0", "data": "XLA detected bad TPU node"},
    ]
    workers = [
        {
            "worker_id": "worker-a",
            "healthy": False,
            "status_message": "Heartbeat timeout",
        }
    ]

    signals = classify_diagnosis(job=job, logs=logs, workers=workers, thread_dump="")
    names = {signal["signal"] for signal in signals}

    assert "oom_or_exit_137" in names
    assert "quota_or_backoff" in names
    assert "tpu_xla_bad_node" in names
    assert "dead_worker" in names
    assert "repeated_retries" in names
    assert "misleading_terminated_by_user" in names
