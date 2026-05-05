# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for task attempt state tracking through the service layer.

Verifies that task status correctly records multiple attempts after
failure + retry, using the ServiceTestHarness (parameterized GCP + K8s).
"""

from iris.rpc import controller_pb2, job_pb2

from .conftest import ServiceTestHarness


def test_task_status_shows_attempts(harness: ServiceTestHarness):
    """Retried task has multiple attempts visible in task status."""
    if harness.provider_type == "gcp":
        harness.register_gcp_worker("w1")

    job_id = harness.submit("retry-status", max_retries_failure=1)
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
