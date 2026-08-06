# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for task attempt state tracking through the service layer.

Verifies that task status correctly records multiple attempts after
failure + retry, using the ServiceTestHarness (parameterized GCP + K8s).
"""

from finelog.rpc import logging_pb2
from finelog.rpc.logging_connect import LogServiceClientSync
from iris.cluster.log_keys import task_log_key
from iris.cluster.types import JobName, TaskAttempt
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


def _only_task_id(harness: ServiceTestHarness, job_id: JobName) -> JobName:
    response = harness.service.list_tasks(
        controller_pb2.Controller.ListTasksRequest(job_id=job_id.to_wire()),
        None,
    )
    assert len(response.tasks) == 1
    return JobName.from_wire(response.tasks[0].task_id)


def test_get_task_status_after_retry_returns_attempt_history(harness: ServiceTestHarness):
    if harness.provider_type == "gcp":
        harness.register_gcp_worker("w1")

    job_id = harness.submit("retry-status", max_retries_failure=1, max_task_failures=1)
    task_id = _only_task_id(harness, job_id)

    harness.drive_task_state(task_id, job_pb2.TASK_STATE_FAILED)
    harness.drive_task_state(task_id, job_pb2.TASK_STATE_SUCCEEDED)

    req = controller_pb2.Controller.GetTaskStatusRequest(task_id=task_id.to_wire())
    resp = harness.service.get_task_status(req, None)
    attempts = resp.task.attempts

    assert len(attempts) == 2
    assert attempts[0].state in (
        job_pb2.TASK_STATE_FAILED,
        job_pb2.TASK_STATE_WORKER_FAILED,
    )
    assert attempts[1].state == job_pb2.TASK_STATE_SUCCEEDED


def test_get_task_status_surfaces_root_cause_highlights(harness: ServiceTestHarness, log_service: LogServiceClientSync):
    """A failed task's status distills the crash out of its noisy logs."""
    if harness.provider_type == "gcp":
        harness.register_gcp_worker("w1")

    job_id = harness.submit("root-cause")
    task_id = _only_task_id(harness, job_id)
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
    task_id = _only_task_id(harness, job_id)
    harness.drive_task_state(task_id, job_pb2.TASK_STATE_SUCCEEDED)

    _push_task_logs(log_service, task_id, ["RuntimeError: this ran on an earlier, since-retried attempt"])

    resp = harness.service.get_task_status(
        controller_pb2.Controller.GetTaskStatusRequest(task_id=task_id.to_wire()), None
    )

    assert resp.task.state == job_pb2.TASK_STATE_SUCCEEDED
    assert list(resp.root_cause_highlights) == []
