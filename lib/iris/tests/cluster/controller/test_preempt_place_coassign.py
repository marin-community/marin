# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Regression for weaver #439: preempt-and-place must be atomic within a tick.

The scheduler runs placement and preemption in two passes per tick, but the
preemptor is unplaced by construction in the pass that triggers the eviction —
placement already failed for it, which is *why* it drives a preemption. On
current ``main`` the preemptor is only placed on a *later* tick, after the
victim's attempt asynchronously finalizes and frees its worker. Nothing
reserves the freed worker in the meantime, so two failures follow:

* the freed worker sits idle while the preemptor stays pending (the "stranded
  free worker" the incident reported), and
* the still-pending preemptor keeps selecting *fresh* victims tick after tick,
  over-preempting workers it will never use.

The fix commits the preemptor as ASSIGNED onto the worker it frees in the same
transaction as the victim's PREEMPT, so a second tick neither over-preempts nor
leaves the worker claimable.
"""

from iris.cluster.controller import ops
from iris.cluster.controller.ops.task import Assignment
from iris.cluster.controller.reconcile.snapshot import TaskUpdate
from iris.cluster.types import WorkerId
from iris.rpc import job_pb2
from rigging.timing import Timestamp
from tests.cluster.controller._test_support import ControllerTestState
from tests.cluster.controller.transition_driver import WorkerTaskUpdates, apply_task_observations

from .conftest import (
    make_job_request,
    make_worker_metadata,
    query_task,
    register_worker,
    submit_job,
)


def _register_cpu_worker(state: ControllerTestState, wid: str) -> WorkerId:
    """A worker sized to hold exactly one ``cpu=1`` task."""
    return register_worker(state, wid, f"{wid}:8080", make_worker_metadata(cpu=1))


def _dispatch_running(state: ControllerTestState, task, worker_id: WorkerId, band: int) -> None:
    """Assign ``task`` to ``worker_id`` with a stamped band and advance to RUNNING."""
    with state._db.transaction() as cur:
        ops.task.assign(
            cur,
            [Assignment(task_id=task.task_id, worker_id=worker_id, priority_band=band)],
            health=state._health,
        )
    attempt_id = query_task(state, task.task_id).current_attempt_id
    with state._db.transaction() as cur:
        apply_task_observations(
            cur,
            [
                WorkerTaskUpdates(
                    worker_id=worker_id,
                    updates=[
                        TaskUpdate(
                            task_id=task.task_id,
                            attempt_id=attempt_id,
                            new_state=job_pb2.TASK_STATE_RUNNING,
                        )
                    ],
                )
            ],
            health=state._health,
            now=Timestamp.now(),
        )


def _submit_solo(state: ControllerTestState, name: str, band: int):
    tasks = submit_job(
        state,
        name,
        make_job_request(name, cpu=1, priority_band=band, max_retries_preemption=5),
    )
    return tasks[0]


def _setup_two_full_workers_and_preemptor(make_controller):
    """Two single-slot workers each running a BATCH victim, plus a pending
    PRODUCTION preemptor that needs one slot. Returns the bound state and the
    task rows ``(victim_1, victim_2, preemptor)``."""
    ctrl = make_controller(remote_state_dir="file:///tmp/iris-439-coassign")
    state = ControllerTestState(ctrl._db, health=ctrl.provider.health)

    w1 = _register_cpu_worker(state, "w1")
    w2 = _register_cpu_worker(state, "w2")

    victim_1 = _submit_solo(state, "/alice/victim-1", job_pb2.PRIORITY_BAND_BATCH)
    victim_2 = _submit_solo(state, "/alice/victim-2", job_pb2.PRIORITY_BAND_BATCH)
    _dispatch_running(state, victim_1, w1, job_pb2.PRIORITY_BAND_BATCH)
    _dispatch_running(state, victim_2, w2, job_pb2.PRIORITY_BAND_BATCH)

    preemptor = _submit_solo(state, "/bob/preemptor", job_pb2.PRIORITY_BAND_PRODUCTION)
    return ctrl, state, victim_1, victim_2, preemptor


def _running_victims(state, victim_1, victim_2) -> list:
    return [
        t
        for t in (query_task(state, victim_1.task_id), query_task(state, victim_2.task_id))
        if t.state == job_pb2.TASK_STATE_RUNNING
    ]


def test_preemptor_is_coassigned_to_the_worker_it_frees(make_controller):
    """The preempting tick also places the preemptor onto the freed worker."""
    ctrl, state, victim_1, victim_2, preemptor = _setup_two_full_workers_and_preemptor(make_controller)

    ctrl._run_scheduling()

    placed = query_task(state, preemptor.task_id)
    assert placed.state == job_pb2.TASK_STATE_ASSIGNED, (
        "preemptor must be ASSIGNED onto the freed worker in the same tick it triggers the "
        f"preemption, not left pending; got state={placed.state}"
    )
    running = _running_victims(state, victim_1, victim_2)
    assert len(running) == 1, "exactly one victim should be evicted"
    freed_worker = running[0].current_worker_id  # the survivor's worker
    assert placed.current_worker_id != freed_worker
    assert placed.current_worker_id in {WorkerId("w1"), WorkerId("w2")}


def test_pending_preemptor_does_not_over_preempt_a_second_victim(make_controller):
    """A preemptor already satisfied by tick 1 must not evict a second victim on
    tick 2 while the first victim's attempt is still finalizing."""
    ctrl, state, victim_1, victim_2, _preemptor = _setup_two_full_workers_and_preemptor(make_controller)

    ctrl._run_scheduling()
    assert len(_running_victims(state, victim_1, victim_2)) == 1, "tick 1 should evict exactly one victim"

    # No terminal heartbeat for the evicted victim: its worker's chips stay
    # reserved. The preemptor is already placed, so tick 2 must be a no-op.
    ctrl._run_scheduling()

    assert (
        len(_running_victims(state, victim_1, victim_2)) == 1
    ), "the second victim must not be over-preempted for an already-placed preemptor"
