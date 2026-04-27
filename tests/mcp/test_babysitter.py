# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from iris.cli.token_store import store_token
from iris.rpc import controller_pb2, job_pb2, time_pb2

from marin.mcp.babysitter import (
    IrisBabysitter,
    IrisConnectionConfig,
    _job_summary_payload,
    _token_provider,
    classify_diagnosis,
    parse_zephyr_progress,
    parse_zephyr_thread_state,
    task_status_to_json,
)


def _timestamp(epoch_ms: int):
    return time_pb2.Timestamp(epoch_ms=epoch_ms)


class _ListJobsController:
    def __init__(self, jobs: list[job_pb2.JobStatus]):
        self.jobs = jobs

    def list_jobs(self, _request):
        return controller_pb2.Controller.ListJobsResponse(jobs=self.jobs, has_more=False)


def _service_with_controller(controller):
    service = object.__new__(IrisBabysitter)
    service.config = IrisConnectionConfig(controller_url="http://controller", cluster="test")
    service.controller = controller
    return service


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


def test_job_summary_payload_preserves_summary_task_fields():
    job = job_pb2.JobStatus(
        job_id="/alice/train",
        name="train",
        state=job_pb2.JOB_STATE_RUNNING,
        task_count=1,
    )
    running_task = job_pb2.TaskStatus(
        task_id="/alice/train/0",
        state=job_pb2.TASK_STATE_RUNNING,
        exit_code=0,
    )

    payload = _job_summary_payload(job, [running_task])

    assert payload["tasks"][0]["index"] == "0"
    assert payload["tasks"][0]["exit_code"] is None
    assert "resource_requests" in payload


def test_jobs_with_prefix_excludes_string_prefix_siblings():
    service = _service_with_controller(
        _ListJobsController(
            [
                job_pb2.JobStatus(job_id="/alice/train"),
                job_pb2.JobStatus(job_id="/alice/train/child"),
                job_pb2.JobStatus(job_id="/alice/train-v2"),
            ]
        )
    )

    jobs = service._jobs_with_prefix("/alice/train")

    assert [job.job_id for job in jobs] == ["/alice/train", "/alice/train/child"]


def test_token_provider_loads_iris_token_store(tmp_path):
    store_path = tmp_path / "tokens.json"
    store_token("iris-prod", "https://controller.example.com", "stored-token", store_path=store_path)

    provider = _token_provider("iris-prod", store_path=store_path)

    assert provider is not None
    assert provider.get_token() == "stored-token"


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


def test_parse_zephyr_thread_state_classifies_active_and_zombie_dumps():
    active = parse_zephyr_thread_state(
        """
        Thread actor-method_0:
          File "zephyr/execution.py", line 873, in _wait_for_stage
        Thread zephyr-coordinator-loop:
          File "zephyr/execution.py", line 444, in _coordinator_loop
        """
    )
    zombie = parse_zephyr_thread_state(
        """
        Thread worker-pool-0:
          File "concurrent/futures/thread.py", line 58, in _worker
        """
    )

    assert active["state"] == "active"
    assert "waiting for stage completion" in active["evidence"]
    assert zombie["state"] == "zombie_suspected"
    assert "worker pool frames without coordinator loop" in zombie["evidence"]


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

    thread_dump = 'File "concurrent/futures/thread.py", line 58, in _worker'

    signals = classify_diagnosis(job=job, logs=logs, workers=workers, thread_dump=thread_dump)
    names = {signal["signal"] for signal in signals}

    assert "oom_or_exit_137" in names
    assert "quota_or_backoff" in names
    assert "tpu_xla_bad_node" in names
    assert "dead_worker" in names
    assert "zombie_coordinator" in names
    assert "repeated_retries" in names
    assert "misleading_terminated_by_user" in names
