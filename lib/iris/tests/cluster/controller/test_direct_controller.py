# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for KubernetesProvider integration with controller and transitions."""

from iris.cluster.controller.schema import TASK_DETAIL_PROJECTION
from iris.cluster.controller.transitions import (
    DirectProviderBatch,
    DirectProviderSyncResult,
    TaskUpdate,
)
from iris.cluster.types import JobName
from iris.rpc import logging_pb2
from iris.rpc import job_pb2
from rigging.timing import Timestamp

from .conftest import (
    make_direct_job_request,
    query_attempt,
    query_task,
    submit_direct_job,
)


class FakeDirectProvider:
    """Minimal KubernetesProvider-like implementation for testing."""

    def __init__(self):
        self.sync_calls: list[DirectProviderBatch] = []
        self.sync_result = DirectProviderSyncResult()
        self.closed = False

    def sync(self, batch: DirectProviderBatch) -> DirectProviderSyncResult:
        self.sync_calls.append(batch)
        return self.sync_result

    def fetch_live_logs(
        self,
        task_id: str,
        attempt_id: int,
        cursor: int,
        max_lines: int,
    ) -> tuple[list[logging_pb2.LogEntry], int]:
        return [], cursor

    def close(self) -> None:
        self.closed = True


# =============================================================================
# Transition-level tests: drain_for_direct_provider
# =============================================================================


def test_drain_pending_creates_attempt_rows(state):
    """Pending tasks are promoted to ASSIGNED with NULL worker_id and an attempt row is created."""
    [task_id] = submit_direct_job(state, "drain-pending")

    task_before = query_task(state, task_id)
    assert task_before.state == job_pb2.TASK_STATE_PENDING

    batch = state.drain_for_direct_provider()

    assert len(batch.tasks_to_run) == 1
    assert batch.tasks_to_run[0].task_id == task_id.to_wire()
    assert batch.tasks_to_run[0].attempt_id == 0

    task_after = query_task(state, task_id)
    assert task_after.state == job_pb2.TASK_STATE_ASSIGNED
    assert task_after.current_attempt_id == 0

    attempt = query_attempt(state, task_id, 0)
    assert attempt is not None
    assert attempt.worker_id is None


def test_drain_skips_already_assigned(state):
    """Already ASSIGNED tasks appear in running_tasks, not tasks_to_run."""
    [task_id] = submit_direct_job(state, "drain-skip")

    # First drain promotes to ASSIGNED.
    batch1 = state.drain_for_direct_provider()
    assert len(batch1.tasks_to_run) == 1
    assert len(batch1.running_tasks) == 0

    # Second drain: task is already ASSIGNED, so appears only in running_tasks.
    batch2 = state.drain_for_direct_provider()
    assert len(batch2.tasks_to_run) == 0
    assert len(batch2.running_tasks) == 1
    assert batch2.running_tasks[0].task_id == task_id


def test_drain_kill_queue(state):
    """Kill requests buffered via buffer_direct_kill appear in tasks_to_kill."""
    [task_id] = submit_direct_job(state, "drain-kill")

    # Promote to ASSIGNED first.
    state.drain_for_direct_provider()

    state.buffer_direct_kill(task_id.to_wire())

    batch = state.drain_for_direct_provider()
    assert task_id.to_wire() in batch.tasks_to_kill


# =============================================================================
# Transition-level tests: apply_direct_provider_updates
# =============================================================================


def test_apply_running(state):
    """ASSIGNED -> RUNNING via direct provider update."""
    [task_id] = submit_direct_job(state, "apply-running")
    batch = state.drain_for_direct_provider()
    attempt_id = batch.tasks_to_run[0].attempt_id

    result = state.apply_direct_provider_updates(
        [
            TaskUpdate(task_id=task_id, attempt_id=attempt_id, new_state=job_pb2.TASK_STATE_RUNNING),
        ]
    )

    task = query_task(state, task_id)
    assert task.state == job_pb2.TASK_STATE_RUNNING
    assert not result.tasks_to_kill


def test_apply_succeeded(state):
    """RUNNING -> SUCCEEDED via direct provider update."""
    [task_id] = submit_direct_job(state, "apply-succeeded")
    batch = state.drain_for_direct_provider()
    attempt_id = batch.tasks_to_run[0].attempt_id

    # First move to RUNNING.
    state.apply_direct_provider_updates(
        [
            TaskUpdate(task_id=task_id, attempt_id=attempt_id, new_state=job_pb2.TASK_STATE_RUNNING),
        ]
    )

    # Then to SUCCEEDED.
    state.apply_direct_provider_updates(
        [
            TaskUpdate(task_id=task_id, attempt_id=attempt_id, new_state=job_pb2.TASK_STATE_SUCCEEDED),
        ]
    )

    task = query_task(state, task_id)
    assert task.state == job_pb2.TASK_STATE_SUCCEEDED
    assert task.exit_code == 0


def test_apply_failed_with_retry(state):
    """FAILED with retries remaining returns task to PENDING."""
    jid = JobName.root("test-user", "retry-job")
    req = make_direct_job_request("retry-job")
    req.max_retries_failure = 2
    state.submit_job(jid, req, Timestamp.now())
    with state._db.snapshot() as q:
        tasks = TASK_DETAIL_PROJECTION.decode(q.fetchall("SELECT * FROM tasks WHERE job_id = ?", (jid.to_wire(),)))
    task_id = tasks[0].task_id

    batch = state.drain_for_direct_provider()
    attempt_id = batch.tasks_to_run[0].attempt_id

    state.apply_direct_provider_updates(
        [
            TaskUpdate(task_id=task_id, attempt_id=attempt_id, new_state=job_pb2.TASK_STATE_RUNNING),
        ]
    )
    state.apply_direct_provider_updates(
        [
            TaskUpdate(task_id=task_id, attempt_id=attempt_id, new_state=job_pb2.TASK_STATE_FAILED, error="boom"),
        ]
    )

    task = query_task(state, task_id)
    # Should be back to PENDING because failure_count(1) <= max_retries_failure(2).
    assert task.state == job_pb2.TASK_STATE_PENDING
    assert task.failure_count == 1


def test_apply_failed_no_retry(state):
    """FAILED with no retries remaining stays terminal."""
    jid = JobName.root("test-user", "no-retry-job")
    req = make_direct_job_request("no-retry-job")
    req.max_retries_failure = 0
    state.submit_job(jid, req, Timestamp.now())
    with state._db.snapshot() as q:
        tasks = TASK_DETAIL_PROJECTION.decode(q.fetchall("SELECT * FROM tasks WHERE job_id = ?", (jid.to_wire(),)))
    task_id = tasks[0].task_id

    batch = state.drain_for_direct_provider()
    attempt_id = batch.tasks_to_run[0].attempt_id

    state.apply_direct_provider_updates(
        [
            TaskUpdate(task_id=task_id, attempt_id=attempt_id, new_state=job_pb2.TASK_STATE_RUNNING),
        ]
    )
    state.apply_direct_provider_updates(
        [
            TaskUpdate(task_id=task_id, attempt_id=attempt_id, new_state=job_pb2.TASK_STATE_FAILED, error="fatal"),
        ]
    )

    task = query_task(state, task_id)
    assert task.state == job_pb2.TASK_STATE_FAILED
    assert task.failure_count == 1


def test_apply_failed_directly_from_assigned(state):
    """ASSIGNED -> FAILED without going through RUNNING (e.g. ConfigMap too large)."""
    [task_id] = submit_direct_job(state, "fail-on-apply")
    batch = state.drain_for_direct_provider()
    attempt_id = batch.tasks_to_run[0].attempt_id

    state.apply_direct_provider_updates(
        [
            TaskUpdate(
                task_id=task_id,
                attempt_id=attempt_id,
                new_state=job_pb2.TASK_STATE_FAILED,
                error="kubectl apply failed: RequestEntityTooLarge",
            ),
        ]
    )

    task = query_task(state, task_id)
    assert task.state == job_pb2.TASK_STATE_FAILED
    assert task.error == "kubectl apply failed: RequestEntityTooLarge"


def test_apply_worker_failed_from_running_retries(state):
    """WORKER_FAILED from RUNNING with retries remaining returns to PENDING."""
    jid = JobName.root("test-user", "wf-retry")
    req = make_direct_job_request("wf-retry")
    req.max_retries_preemption = 5
    state.submit_job(jid, req, Timestamp.now())
    with state._db.snapshot() as q:
        tasks = TASK_DETAIL_PROJECTION.decode(q.fetchall("SELECT * FROM tasks WHERE job_id = ?", (jid.to_wire(),)))
    task_id = tasks[0].task_id

    batch = state.drain_for_direct_provider()
    attempt_id = batch.tasks_to_run[0].attempt_id

    state.apply_direct_provider_updates(
        [
            TaskUpdate(task_id=task_id, attempt_id=attempt_id, new_state=job_pb2.TASK_STATE_RUNNING),
        ]
    )
    state.apply_direct_provider_updates(
        [
            TaskUpdate(task_id=task_id, attempt_id=attempt_id, new_state=job_pb2.TASK_STATE_WORKER_FAILED),
        ]
    )

    task = query_task(state, task_id)
    assert task.state == job_pb2.TASK_STATE_PENDING
    assert task.preemption_count == 1


def test_apply_worker_failed_from_assigned(state):
    """WORKER_FAILED from ASSIGNED returns to PENDING without incrementing preemption_count."""
    [task_id] = submit_direct_job(state, "wf-assigned")
    batch = state.drain_for_direct_provider()
    attempt_id = batch.tasks_to_run[0].attempt_id

    # Task is ASSIGNED after drain (not yet RUNNING).
    state.apply_direct_provider_updates(
        [
            TaskUpdate(task_id=task_id, attempt_id=attempt_id, new_state=job_pb2.TASK_STATE_WORKER_FAILED),
        ]
    )

    task = query_task(state, task_id)
    assert task.state == job_pb2.TASK_STATE_PENDING
    assert task.preemption_count == 0


def test_buffer_direct_kill(state):
    """buffer_direct_kill inserts a kill entry with NULL worker_id."""
    state.buffer_direct_kill("some-task-id")

    rows = state._db.fetchall(
        "SELECT worker_id, kind, task_id FROM dispatch_queue WHERE worker_id IS NULL",
        (),
    )
    assert len(rows) == 1
    assert rows[0]["kind"] == "kill"
    assert rows[0]["task_id"] == "some-task-id"
    assert rows[0]["worker_id"] is None


# =============================================================================
# Controller-level tests
# =============================================================================


def test_drain_multiple_tasks(state):
    """Multiple pending tasks are all promoted in a single drain call."""
    task_ids = submit_direct_job(state, "multi-task", replicas=3)
    assert len(task_ids) == 3

    batch = state.drain_for_direct_provider()
    assert len(batch.tasks_to_run) == 3

    promoted_ids = {req.task_id for req in batch.tasks_to_run}
    expected_ids = {tid.to_wire() for tid in task_ids}
    assert promoted_ids == expected_ids


def test_apply_ignores_stale_attempt(state):
    """Updates with a mismatched attempt_id are silently skipped."""
    [task_id] = submit_direct_job(state, "stale-attempt")
    batch = state.drain_for_direct_provider()
    attempt_id = batch.tasks_to_run[0].attempt_id

    # Apply with wrong attempt_id.
    result = state.apply_direct_provider_updates(
        [
            TaskUpdate(task_id=task_id, attempt_id=attempt_id + 99, new_state=job_pb2.TASK_STATE_RUNNING),
        ]
    )

    task = query_task(state, task_id)
    # Should still be ASSIGNED (the update was skipped).
    assert task.state == job_pb2.TASK_STATE_ASSIGNED
    assert not result.tasks_to_kill


def test_apply_ignores_finished_task(state):
    """Updates to already-finished tasks are silently skipped."""
    [task_id] = submit_direct_job(state, "finished-task")
    batch = state.drain_for_direct_provider()
    attempt_id = batch.tasks_to_run[0].attempt_id

    # Move to SUCCEEDED.
    state.apply_direct_provider_updates(
        [
            TaskUpdate(task_id=task_id, attempt_id=attempt_id, new_state=job_pb2.TASK_STATE_RUNNING),
        ]
    )
    state.apply_direct_provider_updates(
        [
            TaskUpdate(task_id=task_id, attempt_id=attempt_id, new_state=job_pb2.TASK_STATE_SUCCEEDED),
        ]
    )

    # Try to move to FAILED after already succeeded.
    result = state.apply_direct_provider_updates(
        [
            TaskUpdate(task_id=task_id, attempt_id=attempt_id, new_state=job_pb2.TASK_STATE_FAILED),
        ]
    )

    task = query_task(state, task_id)
    assert task.state == job_pb2.TASK_STATE_SUCCEEDED
    assert not result.tasks_to_kill
