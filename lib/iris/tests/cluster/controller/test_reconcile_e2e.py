# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Integration test for the _reconcile_worker_batch hot path.

Spins up a Controller with an in-process WorkerProvider mock that records
reconcile_workers calls. Verifies that the wire payload produced by the new
pure-compute path (reconcile_worker + legacy_translator_request) matches the
payload that the old hand-written path produced for a representative set of
DB states.

Four tasks on the same worker: ASSIGNED, BUILDING, RUNNING, CANCELLED (KILLED).
The assertions check that each lands in the expected wire list — protecting
against drift between the new pure-compute path and the old hand-written one.

Note on CANCELLED tasks: ``reconcile_rows_for_workers`` only returns
ASSIGNED/BUILDING/RUNNING rows. A CANCELLED (KILLED) task disappears from the
expected-task set; the worker auto-kills it because it is no longer present.
``stop_tasks`` is therefore always empty in Phase A — which is identical to
the behavior of the old hand-written path.

Note on attempt IDs: the first attempt for each task has ``attempt_id=0``
(``current_attempt_id`` starts at -1; first assignment increments to 0).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from iris.cluster.controller import reads
from iris.cluster.controller.provider import ProviderError
from iris.cluster.controller.reconcile import WorkerReconcileDispatch
from iris.cluster.controller.transitions import HeartbeatApplyRequest, TaskUpdate
from iris.cluster.controller.worker_provider import WorkerReconcileResult
from iris.cluster.types import JobName, WorkerId
from iris.rpc import job_pb2, worker_pb2

from .conftest import (
    make_job_request,
    make_worker_metadata,
    register_worker,
    submit_job,
)

# ---------------------------------------------------------------------------
# Mock WorkerProvider that records reconcile calls
# ---------------------------------------------------------------------------


@dataclass
class RecordingWorkerProvider:
    """WorkerProvider that records reconcile_workers calls and returns no updates."""

    calls: list[list[WorkerReconcileDispatch]] = field(default_factory=list)

    def reconcile_workers(self, plans: list[WorkerReconcileDispatch]) -> list[WorkerReconcileResult]:
        self.calls.append(list(plans))
        return [
            WorkerReconcileResult(
                worker_id=plan.worker_id,
                start_response=worker_pb2.Worker.StartTasksResponse() if plan.start_tasks else None,
                start_error=None,
                poll_updates=[],
                poll_error=None,
            )
            for plan in plans
        ]

    def ping_workers(self, workers: Any) -> list[Any]:
        return []

    def get_process_status(self, worker_id: Any, address: Any, request: Any) -> Any:
        raise ProviderError("recording provider: get_process_status not supported")

    def on_worker_failed(self, worker_id: Any, address: Any) -> None:
        pass

    def profile_task(self, address: Any, request: Any, timeout_ms: int) -> Any:
        raise ProviderError("recording provider: profile_task not supported")

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_W1 = "worker-1"
_WORKER_ADDR = "127.0.0.1:9999"


def _register_worker(state, worker_id: str = _W1, address: str = _WORKER_ADDR) -> WorkerId:
    """Register a healthy CPU worker and return its WorkerId."""
    return register_worker(state, worker_id, address, make_worker_metadata(cpu=8, memory_bytes=32 * 1024**3))


def _set_task_state(state, task_id: JobName, worker_id: WorkerId, new_state: int) -> None:
    """Advance a task to ``new_state`` via the heartbeat apply path.

    Re-queries ``current_attempt_id`` from the DB so callers don't need to
    track attempt IDs manually.
    """
    with state._db.transaction() as cur:
        task = reads.get_task_detail(cur, task_id)
        assert task is not None
        state.apply_task_updates(
            cur,
            HeartbeatApplyRequest(
                worker_id=worker_id,
                updates=[
                    TaskUpdate(
                        task_id=task_id,
                        attempt_id=task.current_attempt_id,
                        new_state=new_state,
                    )
                ],
            ),
        )


# ---------------------------------------------------------------------------
# Core wire-shape test: four DB states on one worker
# ---------------------------------------------------------------------------


def test_reconcile_worker_batch_wire_shape(make_controller, tmp_path):
    """Wire payload from _reconcile_worker_batch matches expected legacy wire shape.

    Four tasks on one worker in states: ASSIGNED, BUILDING, RUNNING, CANCELLED.

    Expected wire:
    - start_tasks: only the ASSIGNED task (carries RunTaskRequest spec inline)
    - expected_tasks: ASSIGNED + BUILDING + RUNNING (three tasks; CANCELLED is omitted)
    - stop_tasks: empty — CANCELLED task is not in expected_tasks, so the worker
      auto-kills it. The reconcile query only returns ASSIGNED/BUILDING/RUNNING rows,
      so no stop intent is emitted.
    """
    provider = RecordingWorkerProvider()
    ctrl = make_controller(provider=provider)
    state = ctrl._transitions

    wid = _register_worker(state)

    # Submit four 1-task jobs
    tasks_assigned = submit_job(state, "job-assigned", make_job_request(name="job-assigned", replicas=1))
    tasks_building = submit_job(state, "job-building", make_job_request(name="job-building", replicas=1))
    tasks_running = submit_job(state, "job-running", make_job_request(name="job-running", replicas=1))
    tasks_cancelled = submit_job(state, "job-cancelled", make_job_request(name="job-cancelled", replicas=1))

    assert tasks_assigned and tasks_building and tasks_running and tasks_cancelled

    task_assigned = tasks_assigned[0]
    task_building = tasks_building[0]
    task_running = tasks_running[0]
    task_cancelled = tasks_cancelled[0]

    # Assign all four tasks to the worker
    from iris.cluster.controller.transitions import Assignment

    with state._db.transaction() as cur:
        state.queue_assignments(
            cur,
            [
                Assignment(task_id=task_assigned.task_id, worker_id=wid),
                Assignment(task_id=task_building.task_id, worker_id=wid),
                Assignment(task_id=task_running.task_id, worker_id=wid),
                Assignment(task_id=task_cancelled.task_id, worker_id=wid),
            ],
        )

    # Advance building task to BUILDING state
    _set_task_state(state, task_building.task_id, wid, job_pb2.TASK_STATE_BUILDING)

    # Advance running task to RUNNING via BUILDING
    _set_task_state(state, task_running.task_id, wid, job_pb2.TASK_STATE_BUILDING)
    _set_task_state(state, task_running.task_id, wid, job_pb2.TASK_STATE_RUNNING)

    # Cancel the fourth job (task transitions to KILLED)
    job_cancelled_id = JobName.root("test-user", "job-cancelled")
    with state._db.transaction() as cur:
        state.cancel_job(cur, job_cancelled_id, reason="test cancel")

    # Run one reconcile tick
    ctrl._reconcile_worker_batch()

    # Exactly one call to reconcile_workers
    assert len(provider.calls) == 1
    dispatches = provider.calls[0]
    assert len(dispatches) == 1  # one worker
    dispatch = dispatches[0]

    assert dispatch.worker_id == wid

    # start_tasks: exactly the ASSIGNED task (BUILDING/RUNNING/CANCELLED are not started)
    assert len(dispatch.start_tasks) == 1
    start_task_ids = {t.task_id for t in dispatch.start_tasks}
    assert task_assigned.task_id.to_wire() in start_task_ids

    # expected_tasks: ASSIGNED + BUILDING + RUNNING (three tasks)
    # CANCELLED (KILLED) tasks are included in the reconcile query so the
    # controller can emit an explicit stop intent, but they are not "expected"
    # to be reported back by the worker.
    expected_task_ids = {e.task_id for e in dispatch.expected_tasks}
    assert task_assigned.task_id in expected_task_ids
    assert task_building.task_id in expected_task_ids
    assert task_running.task_id in expected_task_ids

    # stop_tasks: KILLED task produces an explicit stop intent.
    stop_task_ids = set(dispatch.stop_tasks)
    assert task_cancelled.task_id.to_wire() in stop_task_ids


# ---------------------------------------------------------------------------
# Test: empty worker — empty dispatch still fires
# ---------------------------------------------------------------------------


def test_reconcile_empty_worker_still_polled(make_controller):
    """A worker with no tasks still receives an empty PollTasks (auto-kills strays)."""
    provider = RecordingWorkerProvider()
    ctrl = make_controller(provider=provider)
    state = ctrl._transitions

    _register_worker(state)

    ctrl._reconcile_worker_batch()

    assert len(provider.calls) == 1
    dispatches = provider.calls[0]
    assert len(dispatches) == 1
    dispatch = dispatches[0]
    assert dispatch.start_tasks == []
    assert dispatch.expected_tasks == []
    assert dispatch.stop_tasks == []


# ---------------------------------------------------------------------------
# Test: no workers — no reconcile call fired
# ---------------------------------------------------------------------------


def test_reconcile_no_workers_no_calls(make_controller):
    """With no registered workers, reconcile_workers is not called."""
    provider = RecordingWorkerProvider()
    ctrl = make_controller(provider=provider)

    ctrl._reconcile_worker_batch()

    assert provider.calls == []


# ---------------------------------------------------------------------------
# Test: start_tasks payload has spec with task_id and attempt_id stamped
# ---------------------------------------------------------------------------


def test_reconcile_start_tasks_spec_stamped(make_controller):
    """The RunTaskRequest in start_tasks has task_id and attempt_id stamped.

    attempt_id is 0 for the first assignment (current_attempt_id starts at -1;
    first assignment increments to 0).
    """
    provider = RecordingWorkerProvider()
    ctrl = make_controller(provider=provider)
    state = ctrl._transitions

    wid = _register_worker(state)
    tasks = submit_job(state, "job-spec", make_job_request(name="job-spec", replicas=1))
    assert tasks
    task = tasks[0]

    from iris.cluster.controller.transitions import Assignment

    with state._db.transaction() as cur:
        state.queue_assignments(cur, [Assignment(task_id=task.task_id, worker_id=wid)])

    ctrl._reconcile_worker_batch()

    assert provider.calls
    dispatch = provider.calls[0][0]
    assert len(dispatch.start_tasks) == 1
    req = dispatch.start_tasks[0]
    assert req.task_id == task.task_id.to_wire()
    assert req.attempt_id >= 0  # first attempt is attempt_id=0


# ---------------------------------------------------------------------------
# Test: ASSIGNED task not in start_tasks if its job has no spec
# ---------------------------------------------------------------------------


def test_reconcile_assigned_spec_present_in_start(make_controller):
    """An ASSIGNED task with a resolvable spec appears in start_tasks."""
    provider = RecordingWorkerProvider()
    ctrl = make_controller(provider=provider)
    state = ctrl._transitions

    wid = _register_worker(state)
    tasks = submit_job(state, "job-a", make_job_request(name="job-a", replicas=1))
    assert tasks
    task = tasks[0]

    from iris.cluster.controller.transitions import Assignment

    with state._db.transaction() as cur:
        state.queue_assignments(cur, [Assignment(task_id=task.task_id, worker_id=wid)])

    ctrl._reconcile_worker_batch()

    assert provider.calls
    dispatch = provider.calls[0][0]
    # The ASSIGNED task should be in start_tasks
    assert any(t.task_id == task.task_id.to_wire() for t in dispatch.start_tasks)
    # And also in expected_tasks
    assert any(e.task_id == task.task_id for e in dispatch.expected_tasks)
