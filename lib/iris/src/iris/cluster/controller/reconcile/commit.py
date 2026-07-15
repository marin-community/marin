# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Persist a :class:`ControllerEffects` to SQL — "record results".

The pure kernel produces a :class:`ControllerEffects` (typed per-entity row
deltas + cross-aggregate effect categories); ``commit_effects`` is the I/O sink
that drains it into the caller's write transaction. Kept separate from
``effects.py`` so the pure kernel (overlay, task/job/peers rules) never imports
``db``/``schema``/``projections``.

Row deltas flush via bulk ``executemany`` statements (one per entity group);
audit log lines are deferred to ``cur``'s post-commit hooks so a rolled-back
transaction leaves no observable trace. Health is owned by the controller and
folded through a single ``apply`` site, so this sink never touches it.
"""

from sqlalchemy import bindparam, func
from sqlalchemy import update as sa_update

from iris.cluster.controller.audit_logging import log_event
from iris.cluster.controller.db import Tx
from iris.cluster.controller.projections.attempt_counts import AttemptCountsProjection
from iris.cluster.controller.reconcile.effects import (
    AttemptRowDelta,
    ControllerEffects,
    JobRowDelta,
    TaskRowDelta,
)
from iris.cluster.controller.schema import jobs_table, task_attempts_table, tasks_table
from iris.cluster.controller.task_state import ACTIVE_TASK_STATES
from iris.cluster.controller.writes import record_federation_change
from iris.rpc import job_pb2


def _flush_tasks(cur: Tx, deltas: list[TaskRowDelta]) -> None:
    """Bulk-flush task deltas.

    Split by terminal vs active state: terminal rows additionally null out
    ``current_worker_id`` / ``current_worker_address`` so a finished task no
    longer points at a worker. ``container_id`` is applied in a small secondary
    executemany over only the rows that carry one, so the main statement shape
    stays uniform across rows.
    """
    if not deltas:
        return

    active_params: list[dict] = []
    terminal_params: list[dict] = []
    container_params: list[dict] = []
    for d in deltas:
        params = {
            "b_task_id": d.task_id,
            "b_state": d.state,
            "b_error": d.error,
            "b_exit_code": d.exit_code,
            "b_started_at": d.started_at.epoch_ms() if d.started_at is not None else None,
            "b_finished_at": d.finished_at.epoch_ms() if d.finished_at is not None else None,
        }
        if d.state in ACTIVE_TASK_STATES:
            active_params.append(params)
        else:
            terminal_params.append(params)
        if d.container_id is not None:
            container_params.append({"b_task_id": d.task_id, "b_container_id": d.container_id})

    base_values = {
        "state": bindparam("b_state"),
        "error": func.coalesce(bindparam("b_error"), tasks_table.c.error),
        "exit_code": func.coalesce(bindparam("b_exit_code"), tasks_table.c.exit_code),
        "started_at_ms": func.coalesce(tasks_table.c.started_at_ms, bindparam("b_started_at")),
        "finished_at_ms": bindparam("b_finished_at"),
    }

    if active_params:
        cur.execute(
            sa_update(tasks_table).where(tasks_table.c.task_id == bindparam("b_task_id")).values(**base_values),
            active_params,
        )
    if terminal_params:
        cur.execute(
            sa_update(tasks_table)
            .where(tasks_table.c.task_id == bindparam("b_task_id"))
            .values(current_worker_id=None, current_worker_address=None, **base_values),
            terminal_params,
        )
    if container_params:
        cur.execute(
            sa_update(tasks_table)
            .where(tasks_table.c.task_id == bindparam("b_task_id"))
            .values(container_id=bindparam("b_container_id")),
            container_params,
        )

    # Record each task transition for a parent federating with this peer (a no-op
    # unless the task's root was received via handoff).
    for d in deltas:
        record_federation_change(cur, d.task_id.parent, task_index=d.task_id.task_index)


def _flush_attempts(cur: Tx, deltas: list[AttemptRowDelta]) -> None:
    """Bulk-flush attempt deltas. Rows where nothing is set are skipped.

    All set columns use the same statement shape (uniform bound params); a None
    bound param leaves the column unchanged via the coalesce expression.
    """
    params: list[dict] = []
    for d in deltas:
        if (
            d.state is None
            and d.started_at is None
            and d.finished_at is None
            and d.exit_code is None
            and d.error is None
        ):
            continue
        params.append(
            {
                "b_task_id": d.task_id,
                "b_attempt_id": d.attempt_id,
                "b_state": d.state,
                "b_started_at": d.started_at.epoch_ms() if d.started_at is not None else None,
                "b_finished_at": d.finished_at.epoch_ms() if d.finished_at is not None else None,
                "b_exit_code": d.exit_code,
                "b_error": d.error,
            }
        )
    if not params:
        return
    c = task_attempts_table.c
    cur.execute(
        sa_update(task_attempts_table)
        .where(c.task_id == bindparam("b_task_id"), c.attempt_id == bindparam("b_attempt_id"))
        .values(
            state=func.coalesce(bindparam("b_state"), c.state),
            started_at_ms=func.coalesce(c.started_at_ms, bindparam("b_started_at")),
            finished_at_ms=func.coalesce(c.finished_at_ms, bindparam("b_finished_at")),
            exit_code=func.coalesce(bindparam("b_exit_code"), c.exit_code),
            error=func.coalesce(bindparam("b_error"), c.error),
        ),
        params,
    )
    # An attempt's state / started_at drives the derived failure & preemption
    # counts, so invalidate the owning jobs' cached totals via the cursor's memo.
    cur.caches[AttemptCountsProjection].invalidate_for_tasks(cur, [d.task_id for d in deltas])


def _flush_jobs(cur: Tx, deltas: list[JobRowDelta]) -> None:
    """Bulk-flush job deltas: recompute writes, then cascade kills."""
    # Record each job transition for a parent federating with this peer (a no-op
    # unless this job's root was received via handoff).
    for d in deltas:
        record_federation_change(cur, d.job_id)

    recompute = [d for d in deltas if not d.is_cascade_kill]
    if recompute:
        params = [
            {
                "b_job_id": d.job_id,
                "b_state": d.state,
                "b_started_at": d.started_at.epoch_ms() if d.started_at is not None else None,
                "b_finished_at": d.finished_at.epoch_ms() if d.finished_at is not None else None,
                "b_error": d.error,
            }
            for d in recompute
        ]
        c = jobs_table.c
        cur.execute(
            sa_update(jobs_table)
            .where(c.job_id == bindparam("b_job_id"))
            .values(
                state=bindparam("b_state"),
                started_at_ms=func.coalesce(c.started_at_ms, bindparam("b_started_at")),
                # recompute finished_at is last-wins (direct set, may be NULL).
                finished_at_ms=bindparam("b_finished_at"),
                error=func.coalesce(bindparam("b_error"), c.error),
            ),
            params,
        )

    # Cascade kills flush unconditionally: merge_cascade_kill already drops a kill
    # onto an already-terminal job, and snapshot load and this flush share one write
    # transaction, so the live row cannot diverge from the guarded basis.
    kills = [d for d in deltas if d.is_cascade_kill]
    if kills:
        params = [
            {
                "b_job_id": d.job_id,
                "b_error": d.error,
                "b_started_at": d.started_at.epoch_ms() if d.started_at is not None else None,
                "b_finished_at": d.finished_at.epoch_ms() if d.finished_at is not None else None,
            }
            for d in kills
        ]
        c = jobs_table.c
        cur.execute(
            sa_update(jobs_table)
            .where(c.job_id == bindparam("b_job_id"))
            .values(
                state=job_pb2.JOB_STATE_KILLED,
                error=bindparam("b_error"),
                # Preserve a started_at stamp carried over from a same-batch
                # recompute->RUNNING that this kill replaced (first-wins).
                started_at_ms=func.coalesce(c.started_at_ms, bindparam("b_started_at")),
                finished_at_ms=func.coalesce(c.finished_at_ms, bindparam("b_finished_at")),
            ),
            params,
        )


def commit_effects(
    cur: Tx,
    effects: ControllerEffects,
) -> None:
    """Record a batch's ``effects`` within the caller's write transaction.

    Row deltas flush via bulk ``executemany`` statements (one per entity group).
    Endpoint deletions write within the Tx. Audit log lines are deferred to
    ``cur``'s post-commit hooks so a rolled-back transaction leaves no observable
    trace. Health is NOT mutated here: ``effects.health.build_failed`` rides back
    to the controller, which folds it (with the backend's transport-observed
    events) through the single ``WorkerHealthTracker.apply`` site.

    Attempt writes invalidate the derived-count cache through the cursor
    (``cur.caches[AttemptCountsProjection]``) — no cache reference is threaded here.
    """
    _flush_tasks(cur, list(effects.tasks.values()))
    _flush_attempts(cur, list(effects.attempts.values()))
    _flush_jobs(cur, list(effects.jobs.values()))

    log_events = effects.log_events
    if log_events:

        def _post_commit() -> None:
            for ev in log_events:
                details = {k: v for k, v in ev.details}
                log_event(ev.action, ev.entity_id, trigger=ev.trigger, **details)

        cur.register(_post_commit)
