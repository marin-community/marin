# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""``AttemptCountsProjection`` invalidation at the ``delete_job`` chokepoint.

The per-job failure/preemption memo is derived from ``task_attempts``. Deleting a
job CASCADEs those attempt rows away, so ``writes.delete_job`` drops the memo via
``tx.caches`` — otherwise a later job minted with the same id (a federation
set-replace that drops a handle, then a re-handoff that re-mints it) would serve
the dead job's cached counts until the next attempt mirror.
"""

from iris.cluster.controller import writes
from iris.cluster.controller.attempt_counts import AttemptCounts
from iris.cluster.controller.projections.attempt_counts import AttemptCountsProjection
from iris.cluster.controller.schema import jobs_table, task_attempts_table, tasks_table
from iris.cluster.types import JobName
from iris.rpc import job_pb2
from sqlalchemy import insert


def _seed_job_with_attempt(state, job_id: JobName, *, attempt_state: int) -> None:
    """Insert a one-task job whose single attempt is in ``attempt_state``.

    ``started_at_ms`` is set so an executing-phase preemption would charge its
    budget; a FAILED attempt charges failure regardless of phase.
    """
    task_id = job_id.task(0)
    with state._db.transaction() as cur:
        cur.execute(
            insert(jobs_table).values(
                job_id=job_id,
                user_id="u1",
                root_job_id=job_id.to_wire(),
                depth=0,
                state=job_pb2.JOB_STATE_RUNNING,
                submitted_at_ms=2_000,
                root_submitted_at_ms=2_000,
                num_tasks=1,
            )
        )
        cur.execute(
            insert(tasks_table).values(
                task_id=task_id,
                job_id=job_id,
                task_index=0,
                state=job_pb2.TASK_STATE_RUNNING,
                submitted_at_ms=2_000,
                max_retries_failure=0,
                max_retries_preemption=0,
                priority_neg_depth=0,
                priority_root_submitted_ms=2_000,
                priority_insertion=0,
                current_attempt_id=0,
            )
        )
        cur.execute(
            insert(task_attempts_table).values(
                task_id=task_id,
                attempt_id=0,
                state=attempt_state,
                created_at_ms=2_000,
                started_at_ms=2_000,
                attempt_uid=task_id.to_wire(),
            )
        )


def _counts(state, job_id: JobName) -> AttemptCounts:
    """Read ``job_id``'s counts through the projection (warming/serving the memo)."""
    with state._db.read_snapshot() as snap:
        return state._db.caches[AttemptCountsProjection].get_jobs(snap, [job_id])[job_id]


def test_delete_job_invalidates_attempt_counts_cache(state):
    """A re-minted job id must not serve the deleted job's cached counts.

    ``writes.delete_job`` invalidates the memo, so the guarantee is intrinsic to
    the delete chokepoint and independent of any caller wrapping it.
    """
    job_id = JobName.root("u1", "counts-job")

    _seed_job_with_attempt(state, job_id, attempt_state=job_pb2.TASK_STATE_FAILED)
    # Warm the memo with the dead job's non-zero counts.
    assert _counts(state, job_id) == AttemptCounts(failure_count=1)

    with state._db.transaction() as cur:
        writes.delete_job(cur, job_id)

    # A later job re-mints the same id with a clean attempt; the memo must reflect
    # it, not the dead job's cached failure.
    _seed_job_with_attempt(state, job_id, attempt_state=job_pb2.TASK_STATE_SUCCEEDED)
    assert _counts(state, job_id) == AttemptCounts()
