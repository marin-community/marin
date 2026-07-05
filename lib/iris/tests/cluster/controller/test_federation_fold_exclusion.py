# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Federated tasks are structurally invisible to the control-plane fold.

A federated task (``cluster != 'local'``) is owned by a peer cluster: it lives
in the local ``jobs``/``tasks`` rows only so listings render, but the local
scheduler must never act on it. Every control-plane reader sources the
``local_tasks`` selectable (``cluster = 'local'``), so these tests assert that
synthetic federated rows are never routed, dispatched, finalized, timed out,
counted as local budget/admission spend, or pruned — while the parallel *local*
rows still flow through those same readers (the positive control).

No peer can be configured yet, so in production zero federated rows exist and
this is a behavior-preserving refactor; the tests inject the rows the future
sync will write and pin the boundary now.
"""

from iris.cluster.controller import reads
from iris.cluster.controller.reads import TaskScope
from iris.cluster.controller.reconcile import dispatch, loader
from iris.cluster.controller.reconcile.policy import NON_TERMINAL_TASK_STATES
from iris.cluster.controller.schema import job_config_table, jobs_table, task_attempts_table, tasks_table
from iris.cluster.controller.task_state import ACTIVE_TASK_STATES
from iris.cluster.types import LOCAL_CLUSTER, TERMINAL_JOB_STATES, JobName
from iris.rpc import job_pb2
from rigging.timing import Timestamp
from sqlalchemy import insert as sa_insert
from sqlalchemy import update as sa_update

from .conftest import make_worker_metadata, query_tasks_for_job, register_worker, submit_direct_job

PEER = "peer-west"


def _mark_federated(state, job_id: JobName, *, task_states: dict[int, int]) -> None:
    """Turn an already-submitted local job into a peer-owned federated job.

    Sets ``cluster`` on the job and all its tasks, clears ``backend_id``
    and any local worker binding, and stamps each task_index with the state in
    ``task_states``. This is the shape the future federation sync mirrors from
    the peer: rows present locally, but owned elsewhere.
    """
    with state._db.transaction() as cur:
        cur.execute(sa_update(jobs_table).where(jobs_table.c.job_id == job_id).values(cluster=PEER))
        cur.execute(
            sa_update(tasks_table)
            .where(tasks_table.c.job_id == job_id)
            .values(cluster=PEER, backend_id="", current_worker_id=None)
        )
        for task_index, task_state in task_states.items():
            cur.execute(
                sa_update(tasks_table)
                .where((tasks_table.c.job_id == job_id) & (tasks_table.c.task_index == task_index))
                .values(state=task_state)
            )


def test_federated_pending_task_is_not_routed_or_dispatched(state):
    # Local control job (its PENDING task must still flow through every reader).
    [local_task] = submit_direct_job(state, "local-job")

    # Federated job: task 0 PENDING, task 1 RUNNING, both peer-owned.
    fed_tasks = submit_direct_job(state, "fed-job", replicas=2)
    fed_job = JobName.root("test-user", "fed-job")
    _mark_federated(
        state,
        fed_job,
        task_states={0: job_pb2.TASK_STATE_PENDING, 1: job_pb2.TASK_STATE_RUNNING},
    )

    # Routing: the scheduler's pending projection sees the local task, never the peer's.
    with state._db.read_snapshot() as tx:
        pending = reads.pending_tasks_with_jobs(tx)
    pending_ids = {t.task_id for t in pending}
    assert local_task in pending_ids
    assert pending_ids.isdisjoint(fed_tasks)

    # Dispatch: the direct-provider drain is the canonical silent break — a
    # federated PENDING row would be promoted to ASSIGNED and run locally.
    with state._db.transaction() as cur:
        batch = dispatch.drain_for_dispatch(cur)
    dispatched = {r.task_id for r in batch.tasks_to_run}
    assert local_task.to_wire() in dispatched
    assert dispatched.isdisjoint({t.to_wire() for t in fed_tasks})
    # The RUNNING federated task must not enter the poll/redrive set either.
    running_ids = {e.task_id for e in batch.running_tasks}
    assert running_ids.isdisjoint(fed_tasks)

    # The promotion only touched the local task; the federated rows are untouched.
    fed_rows = {r.task_index: r for r in query_tasks_for_job(state, fed_job)}
    assert fed_rows[0].state == job_pb2.TASK_STATE_PENDING
    assert fed_rows[1].state == job_pb2.TASK_STATE_RUNNING
    assert fed_rows[0].current_worker_id is None and fed_rows[1].current_worker_id is None


def test_federated_tasks_excluded_from_budget_and_admission(state):
    submit_direct_job(state, "local-job", replicas=2)
    local_job = JobName.root("test-user", "local-job")
    # A local RUNNING task is real budget spend (spend counts ACTIVE states);
    # its federated twin below must not add to the same total.
    with state._db.transaction() as cur:
        cur.execute(
            sa_update(tasks_table)
            .where((tasks_table.c.job_id == local_job) & (tasks_table.c.task_index == 1))
            .values(state=job_pb2.TASK_STATE_RUNNING)
        )

    fed_job = JobName.root("test-user", "fed-job")
    submit_direct_job(state, "fed-job", replicas=2)
    _mark_federated(
        state,
        fed_job,
        task_states={0: job_pb2.TASK_STATE_PENDING, 1: job_pb2.TASK_STATE_RUNNING},
    )

    with state._db.read_snapshot() as tx:
        spend_jobs = {row.job_id for row in reads.user_spend_rows(tx)}
        active_count = reads.count_active_tasks_for_user(tx, "test-user")
        active_by_job = reads.list_active_tasks_for_jobs(tx, [local_job, fed_job], states=NON_TERMINAL_TASK_STATES)

    # Budget spend and the admission cap count only the two local tasks.
    assert spend_jobs == {local_job}
    assert active_count == 2
    assert {r.task_id for r in active_by_job[local_job]} and active_by_job[fed_job] == ()


def test_federated_running_row_excluded_from_dispatch_poll_set(state):
    [local_task] = submit_direct_job(state, "local-run")
    # Promote the local task to ASSIGNED so it populates the null-worker poll set.
    with state._db.transaction() as cur:
        dispatch.drain_for_dispatch(cur)

    fed_job = JobName.root("test-user", "fed-run")
    [fed_task] = submit_direct_job(state, "fed-run")
    _mark_federated(state, fed_job, task_states={0: job_pb2.TASK_STATE_RUNNING})

    # The null-worker active scan (dispatch poll set) sees the local ASSIGNED
    # row but not the peer-owned RUNNING one.
    with state._db.read_snapshot() as tx:
        active = reads.list_active_tasks(tx, TaskScope(null_worker=True), states=ACTIVE_TASK_STATES)
    active_ids = {r.task_id for r in active}
    assert local_task in active_ids
    assert fed_task not in active_ids


def test_federated_row_excluded_even_with_local_worker_attempt(state):
    """Adversarial: even a federated row that wrongly carries a local worker and
    a stale worker-bound attempt is excluded by the ``local_tasks`` boundary.

    The attempt/worker-joining readers (timeout scan, reconcile rows) drop
    federated tasks structurally, not merely because a real federated task has
    no local attempts — so the guard holds even in this impossible-but-defended
    state.
    """
    worker = register_worker(state, "w-local", "w-local:8080", make_worker_metadata())

    fed_job = JobName.root("test-user", "fed-exec")
    [fed_task] = submit_direct_job(state, "fed-exec")
    _mark_federated(state, fed_job, task_states={0: job_pb2.TASK_STATE_RUNNING})

    now_ms = Timestamp.now().epoch_ms()
    long_ago = now_ms - 3_600_000
    with state._db.transaction() as cur:
        # Give it a local worker binding + a running attempt started an hour ago,
        # and a 1s job timeout, so a raw-`tasks` reader WOULD flag it.
        cur.execute(
            sa_update(tasks_table)
            .where(tasks_table.c.task_id == fed_task)
            .values(current_worker_id=worker, current_attempt_id=0)
        )
        cur.execute(sa_update(job_config_table).where(job_config_table.c.job_id == fed_job).values(timeout_ms=1000))
        cur.execute(
            sa_insert(task_attempts_table).values(
                task_id=fed_task,
                attempt_id=0,
                worker_id=worker,
                state=job_pb2.TASK_STATE_RUNNING,
                created_at_ms=long_ago,
                started_at_ms=long_ago,
                attempt_uid="fed-exec-uid",
            )
        )

    with state._db.read_snapshot() as tx:
        timed_out = {row.task_id for row in reads.scan_execution_timeout_rows(tx)}
        reconcile = {row.task_id for row in reads.load_reconcile_rows(tx, [worker])}

    assert fed_task not in timed_out
    assert fed_task not in reconcile


def test_federated_terminal_job_is_not_pruned_locally(state):
    long_ago = Timestamp.now().epoch_ms() - 7 * 24 * 3_600_000
    cutoff = Timestamp.now()

    # A local terminal job (prunable) and a federated terminal job (never pruned
    # locally — a peer tombstone is its only deletion path).
    submit_direct_job(state, "local-done")
    local_done = JobName.root("test-user", "local-done")
    submit_direct_job(state, "fed-done")
    fed_done = JobName.root("test-user", "fed-done")
    _mark_federated(state, fed_done, task_states={0: job_pb2.TASK_STATE_SUCCEEDED})

    with state._db.transaction() as cur:
        for job_id in (local_done, fed_done):
            cur.execute(
                sa_update(jobs_table)
                .where(jobs_table.c.job_id == job_id)
                .values(state=job_pb2.JOB_STATE_SUCCEEDED, finished_at_ms=long_ago)
            )

    # The pruner picks the local job; the federated one is never a candidate,
    # even after the local job is deleted.
    with state._db.read_snapshot() as tx:
        first = reads.find_prunable_job(tx, TERMINAL_JOB_STATES, cutoff)
    assert first == local_done

    with state._db.transaction() as cur:
        cur.execute(tasks_table.delete().where(tasks_table.c.job_id == local_done))
        cur.execute(jobs_table.delete().where(jobs_table.c.job_id == local_done))

    with state._db.read_snapshot() as tx:
        assert reads.find_prunable_job(tx, TERMINAL_JOB_STATES, cutoff) is None
        # The federated job and its task are still present.
        assert reads.get_job_state(tx, fed_done) == job_pb2.JOB_STATE_SUCCEEDED
        assert query_tasks_for_job(state, fed_done)[0].cluster == PEER


def test_status_projections_surface_cluster(state):
    """The job/task detail reads carry ``cluster``, so a peer-owned row is
    distinguishable from a locally unrouted one on the status API — a local job
    reports ``'local'``, a federated one reports its peer."""
    submit_direct_job(state, "local-status")
    local_job = JobName.root("test-user", "local-status")

    fed_job = JobName.root("test-user", "fed-status")
    [fed_task] = submit_direct_job(state, "fed-status")
    _mark_federated(state, fed_job, task_states={0: job_pb2.TASK_STATE_RUNNING})

    with state._db.read_snapshot() as tx:
        assert reads.get_job_detail(tx, local_job).cluster == LOCAL_CLUSTER
        assert reads.get_job_detail(tx, fed_job).cluster == PEER
        assert reads.get_task_detail(tx, fed_task).cluster == PEER


def test_reconcile_snapshot_loader_excludes_federated_tasks(state):
    """The reconcile snapshot's per-job task load excludes federated rows, so a
    federated job seeded into the fold contributes no local task histogram."""
    fed_job = JobName.root("test-user", "fed-loader")
    submit_direct_job(state, "fed-loader", replicas=2)
    _mark_federated(
        state,
        fed_job,
        task_states={0: job_pb2.TASK_STATE_PENDING, 1: job_pb2.TASK_STATE_RUNNING},
    )

    with state._db.read_snapshot() as tx:
        by_job = loader._load_all_tasks_for_jobs(tx, [fed_job])
    assert by_job[fed_job] == ()
