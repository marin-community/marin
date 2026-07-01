# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the commit-ownership pivot: the controller-side job-DAG fold and
backend-owned placement validation.

``test_multi_backend_control_loop.py`` covers the cross-backend cascade case
(the fold reaching rows owned by a different backend than the one that
authored the triggering transition). These tests cover the other two
guarantees the pivot depends on: a stale scheduler placement is rejected at
commit rather than silently dispatched into a re-registered worker's new
session, and a reader can never observe a job's task counts mid-fold.
"""

from iris.cluster.controller import ops, reads
from iris.cluster.controller.ops.reconcile import fold_direct_results
from iris.cluster.controller.ops.task import Assignment
from iris.cluster.controller.reconcile.commit import commit_effects
from iris.cluster.controller.reconcile.snapshot import TaskUpdate
from iris.cluster.controller.schema import workers_table
from iris.cluster.types import JobName, WorkerId
from iris.rpc import job_pb2
from rigging.timing import Timestamp
from sqlalchemy import select
from tests.cluster.controller._test_support import ControllerTestState
from tests.cluster.controller.conftest import (
    dispatch_task,
    make_job_request,
    make_worker_metadata,
    query_job,
    query_task,
    register_worker,
    submit_job,
)
from tests.cluster.controller.transition_driver import WorkerTaskUpdates, author_task_observations


def _incarnation(state: ControllerTestState, worker_id: WorkerId) -> int:
    with state._db.read_snapshot() as tx:
        return tx.execute(
            select(workers_table.c.registered_at_ms).where(workers_table.c.worker_id == worker_id)
        ).scalar_one()


def _register_at(state: ControllerTestState, worker_id: str, address: str, ts_ms: int) -> WorkerId:
    wid = WorkerId(worker_id)
    with state._db.transaction() as cur:
        ops.worker.register(
            cur,
            worker_id=wid,
            address=address,
            metadata=make_worker_metadata(),
            ts=Timestamp.from_ms(ts_ms),
            health=state._health,
            worker_attrs=state._worker_attrs,
            slice_id="",
            scale_group="",
        )
    return wid


def test_stale_assignment_incarnation_rejected_at_commit(state: ControllerTestState) -> None:
    """A scheduler-computed ``Assignment`` against a worker's old incarnation
    is dropped at commit once the worker has re-registered (crashed and
    rejoined) — the task stays unscheduled rather than being dispatched into
    the new worker session. A fresh assignment against the current
    incarnation still succeeds.
    """
    worker_id = _register_at(state, "w1", "host:8080", 1_000)
    stale_incarnation = _incarnation(state, worker_id)

    tasks = submit_job(state, "job", make_job_request("job"))
    task_id = tasks[0].task_id

    # Worker crashes and rejoins — same worker_id and address, new incarnation.
    _register_at(state, "w1", "host:8080", 2_000)
    assert _incarnation(state, worker_id) != stale_incarnation

    stale_assignment = Assignment(
        task_id=task_id, worker_id=worker_id, address="host:8080", incarnation=stale_incarnation
    )
    with state._db.transaction() as cur:
        ops.task.assign(cur, [stale_assignment], backend=state)
    assert query_task(state, task_id).state == job_pb2.TASK_STATE_PENDING
    assert query_task(state, task_id).current_worker_id is None

    fresh_assignment = Assignment(
        task_id=task_id,
        worker_id=worker_id,
        address="host:8080",
        incarnation=_incarnation(state, worker_id),
    )
    with state._db.transaction() as cur:
        ops.task.assign(cur, [fresh_assignment], backend=state)
    assert query_task(state, task_id).state == job_pb2.TASK_STATE_ASSIGNED
    assert query_task(state, task_id).current_worker_id == str(worker_id)


def test_reader_never_sees_cross_backend_fold_partially_applied(state: ControllerTestState) -> None:
    """Two independently authored direct results for the two tasks of one job
    settle to a fully consistent terminal state in a single fold+commit — a
    reader can never observe the job counted as complete for only one task.
    """
    w1 = register_worker(state, "w1", "host1:8080", make_worker_metadata())
    w2 = register_worker(state, "w2", "host2:8080", make_worker_metadata())

    tasks = submit_job(state, "job", make_job_request("job", replicas=2))
    job_id = JobName.root("test-user", "job")
    dispatch_task(state, tasks[0], w1)
    dispatch_task(state, tasks[1], w2)
    assert query_job(state, job_id).state == job_pb2.JOB_STATE_RUNNING

    attempt_0 = query_task(state, tasks[0].task_id).current_attempt_id
    attempt_1 = query_task(state, tasks[1].task_id).current_attempt_id

    with state._db.transaction() as cur:
        now = Timestamp.now()
        direct_a = author_task_observations(
            cur,
            [
                WorkerTaskUpdates(
                    worker_id=w1,
                    updates=[
                        TaskUpdate(
                            task_id=tasks[0].task_id, attempt_id=attempt_0, new_state=job_pb2.TASK_STATE_SUCCEEDED
                        )
                    ],
                )
            ],
            now=now,
        )
        direct_b = author_task_observations(
            cur,
            [
                WorkerTaskUpdates(
                    worker_id=w2,
                    updates=[
                        TaskUpdate(
                            task_id=tasks[1].task_id, attempt_id=attempt_1, new_state=job_pb2.TASK_STATE_SUCCEEDED
                        )
                    ],
                )
            ],
            now=now,
        )
        # Authoring alone doesn't recompute or finalize the job — the fold
        # over the union of both direct results hasn't run yet.
        assert reads.get_job_detail(cur, job_id).state == job_pb2.JOB_STATE_RUNNING

        effects = fold_direct_results(cur, [direct_a, direct_b], now=now)
        commit_effects(cur, effects, endpoints=state._endpoints)

    assert query_job(state, job_id).state == job_pb2.JOB_STATE_SUCCEEDED
    assert query_task(state, tasks[0].task_id).state == job_pb2.TASK_STATE_SUCCEEDED
    assert query_task(state, tasks[1].task_id).state == job_pb2.TASK_STATE_SUCCEEDED
