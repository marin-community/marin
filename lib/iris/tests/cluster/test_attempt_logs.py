# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for task attempt state tracking through the service layer.

Verifies that task status correctly records multiple attempts after
failure + retry, using the ServiceTestHarness (parameterized GCP + K8s).
"""

from finelog.rpc import logging_pb2
from finelog.rpc.logging_connect import LogServiceClientSync
from iris.cluster.log_keys import task_log_key
from iris.cluster.types import TaskAttempt
from iris.rpc import controller_pb2, job_pb2

from .conftest import ServiceTestHarness


def _push_task_logs(log_service: LogServiceClientSync, task_id, lines: list[str]) -> None:
    """Durably push ``lines`` as the first attempt's logs for ``task_id``."""
    entries = []
    for i, data in enumerate(lines):
        entry = logging_pb2.LogEntry(source="stdout", data=data)
        entry.timestamp.epoch_ms = 1000 + i
        entries.append(entry)
    key = task_log_key(TaskAttempt(task_id=task_id, attempt_id=0))
    log_service.push_logs(logging_pb2.PushLogsRequest(key=key, entries=entries))


def test_task_status_shows_attempts(harness: ServiceTestHarness):
    """Retried task has multiple attempts visible in task status."""
    if harness.provider_type == "gcp":
        harness.register_gcp_worker("w1")

    job_id = harness.submit("retry-status", max_retries_failure=1, max_task_failures=1)
    tasks = harness._query_tasks(job_id)
    assert len(tasks) == 1
    task_id = tasks[0].task_id

    # Drive first attempt to FAILED
    harness.drive_task_state(task_id, job_pb2.TASK_STATE_FAILED)

    # After failure with retries remaining, task goes back to PENDING.
    # Drive the retry attempt to SUCCEEDED.
    harness.drive_task_state(task_id, job_pb2.TASK_STATE_SUCCEEDED)

    # Check via RPC that attempts are visible
    req = controller_pb2.Controller.GetTaskStatusRequest(task_id=task_id.to_wire())
    resp = harness.service.get_task_status(req, None)
    attempts = resp.task.attempts

    assert len(attempts) >= 2, f"Expected >=2 attempts, got {len(attempts)}"

    # First attempt should have failed
    assert attempts[0].state in (
        job_pb2.TASK_STATE_FAILED,
        job_pb2.TASK_STATE_WORKER_FAILED,
    )
    # Last attempt should have succeeded
    assert attempts[-1].state == job_pb2.TASK_STATE_SUCCEEDED


def test_get_task_status_surfaces_root_cause_highlights(harness: ServiceTestHarness, log_service: LogServiceClientSync):
    """A failed task's status distills the crash out of its noisy logs."""
    if harness.provider_type == "gcp":
        harness.register_gcp_worker("w1")

    job_id = harness.submit("root-cause")
    task_id = harness._query_tasks(job_id)[0].task_id
    harness.drive_task_state(task_id, job_pb2.TASK_STATE_FAILED)

    _push_task_logs(
        log_service,
        task_id,
        [
            " 50%|#####     | 500/1000 [00:10<00:10,  5.0it/s]",
            "Traceback (most recent call last):",
            "RuntimeError: CUDA error: an illegal memory access was encountered",
        ],
    )

    resp = harness.service.get_task_status(
        controller_pb2.Controller.GetTaskStatusRequest(task_id=task_id.to_wire()), None
    )

    assert resp.task.state == job_pb2.TASK_STATE_FAILED
    assert "RuntimeError: CUDA error: an illegal memory access was encountered" in resp.root_cause_highlights
    # The tqdm progress bar is noise and must not appear.
    assert not any("500/1000" in line for line in resp.root_cause_highlights)


def test_get_task_status_no_highlights_for_succeeded_task(
    harness: ServiceTestHarness, log_service: LogServiceClientSync
):
    """A task that did not fail gets no root-cause highlights, even with logs."""
    if harness.provider_type == "gcp":
        harness.register_gcp_worker("w1")

    job_id = harness.submit("no-root-cause")
    task_id = harness._query_tasks(job_id)[0].task_id
    harness.drive_task_state(task_id, job_pb2.TASK_STATE_SUCCEEDED)

    _push_task_logs(log_service, task_id, ["RuntimeError: this ran on an earlier, since-retried attempt"])

    resp = harness.service.get_task_status(
        controller_pb2.Controller.GetTaskStatusRequest(task_id=task_id.to_wire()), None
    )

    assert resp.task.state == job_pb2.TASK_STATE_SUCCEEDED
    assert list(resp.root_cause_highlights) == []
