# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import marin.mcp.babysitter as babysitter
from iris.cluster.resources.attempt import AttemptSummary
from iris.cluster.resources.identity import AttemptIdentity, JobIdentity, ResourceKey, ResourceKind, TaskIdentity
from iris.cluster.resources.job import JobDetail, JobSpec, JobSummary
from iris.cluster.resources.task import TaskDetail, TaskSummary
from iris.cluster.types import ResourceSpec
from iris.rpc import job_pb2
from marin.mcp.babysitter import (
    _token_provider,
    classify_diagnosis,
    parse_zephyr_progress,
    parse_zephyr_thread_state,
    task_status_to_json,
)
from rigging.credentials import MARIN_CLUSTER_TOKEN_ENV
from rigging.timing import Timestamp

_NOW = Timestamp(1_000)


def _job_identity() -> JobIdentity:
    return JobIdentity(ResourceKey("prod", ResourceKind.JOB, "/alice/train"), "job-uid")


def _task_identity() -> TaskIdentity:
    return TaskIdentity(ResourceKey("prod", ResourceKind.TASK, "/alice/train/0"), "task-uid")


def _job_detail() -> JobDetail:
    return JobDetail(
        summary=JobSummary(
            identity=_job_identity(),
            owner_id="alice",
            parent=None,
            state=job_pb2.JOB_STATE_RUNNING,
            execution_cluster_id="prod",
            backend_id="east",
            num_tasks=1,
            submitted_at=_NOW,
            started_at=_NOW,
            finished_at=None,
            error_message="",
            pending_reason="",
        ),
        spec=JobSpec(
            version=1,
            name="train",
            entrypoint=job_pb2.RuntimeEntrypoint(),
            resources=ResourceSpec(cpu=1, memory=1024, disk=2048),
            environment=job_pb2.EnvironmentConfig(),
            bundle_id="bundle",
            scheduling_timeout=None,
            ports=(),
            max_task_failures=0,
            max_retries_failure=0,
            max_retries_preemption=1,
            constraints=(),
            coscheduling=None,
            replicas=1,
            timeout=None,
            fail_if_exists=False,
            preemption_policy=job_pb2.JOB_PREEMPTION_POLICY_UNSPECIFIED,
            existing_job_policy=job_pb2.EXISTING_JOB_POLICY_UNSPECIFIED,
            priority_band=job_pb2.PRIORITY_BAND_INHERIT,
            task_image="",
            submit_argv=(),
            client_revision_date="",
            container_profile=job_pb2.CONTAINER_PROFILE_UNSPECIFIED,
        ),
    )


def test_task_status_json_preserves_exact_identity_and_attempt_history():
    task_identity = _task_identity()
    attempt_identity = AttemptIdentity(task_identity.key, 1, "attempt-uid")
    attempt = AttemptSummary(
        identity=attempt_identity,
        state=job_pb2.TASK_STATE_FAILED,
        execution_cluster_id="prod",
        backend_id="east",
        node=None,
        created_at=_NOW,
        started_at=_NOW,
        finished_at=Timestamp(2_500),
        exit_code=137,
        error_message="OOMKilled",
        terminal_reason="application",
    )
    task = TaskDetail(
        summary=TaskSummary(
            identity=task_identity,
            job=_job_identity(),
            task_index=0,
            state=job_pb2.TASK_STATE_FAILED,
            execution_cluster_id="prod",
            backend_id="east",
            current_attempt=attempt_identity,
            current_node=None,
            failure_count=1,
            preemption_count=0,
            submitted_at=_NOW,
            started_at=_NOW,
            finished_at=Timestamp(2_500),
            status_message="",
            error_message="OOMKilled",
        ),
        attempts=(attempt,),
        source_statuses=(),
        root_cause_highlights=("container exited 137",),
    )

    payload = task_status_to_json(task)

    assert payload["task_id"] == "/alice/train/0"
    assert payload["task_uid"] == "task-uid"
    assert payload["state"] == "failed"
    assert payload["started_at_ms"] == 1_000
    assert payload["finished_at_ms"] == 2_500
    assert payload["duration_ms"] == 1_500
    assert payload["attempts"][0]["attempt_uid"] == "attempt-uid"
    assert payload["attempts"][0]["exit_code"] == 137
    assert payload["root_cause_highlights"] == ["container exited 137"]


def test_job_summary_payload_preserves_summary_task_fields():
    running_task = TaskSummary(
        identity=_task_identity(),
        job=_job_identity(),
        task_index=0,
        state=job_pb2.TASK_STATE_RUNNING,
        execution_cluster_id="prod",
        backend_id="east",
        current_attempt=None,
        current_node=None,
        failure_count=0,
        preemption_count=0,
        submitted_at=_NOW,
        started_at=_NOW,
        finished_at=None,
        status_message="running",
        error_message="",
    )

    payload = babysitter._job_summary_payload(_job_detail(), [running_task])

    assert payload["tasks"][0]["index"] == "0"
    assert payload["job_uid"] == "job-uid"
    assert payload["tasks"][0]["task_uid"] == "task-uid"
    assert "resource_requests" in payload
    assert "resource_usage" not in payload


def test_token_provider_uses_env_override(monkeypatch):
    # Pure-IAP: the controller mints no user token, so the only Authorization bearer
    # is the explicit $MARIN_CLUSTER_TOKEN escape hatch (e.g. a worker JWT for CI).
    monkeypatch.setenv(MARIN_CLUSTER_TOKEN_ENV, "env-token")
    provider = _token_provider()
    assert provider is not None
    assert provider.get_token() == "env-token"


def test_token_provider_none_without_env(monkeypatch):
    monkeypatch.delenv(MARIN_CLUSTER_TOKEN_ENV, raising=False)
    assert _token_provider() is None


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
