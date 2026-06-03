# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Persist a :class:`ControllerEffects` to SQL — "record results".

The pure kernel produces a :class:`ControllerEffects` (typed per-entity row
deltas + cross-aggregate effect categories); ``commit_effects`` is the I/O sink
that drains it into the caller's write transaction. Kept separate from
``effects.py`` so the pure kernel (overlay, task/job/peers rules) never imports
``db``/``schema``/``projections``.

Row deltas flush via bulk ``executemany`` statements (one per entity group);
endpoint deletions write within the Tx; in-memory health bumps and audit log
lines are deferred to ``cur``'s post-commit hooks so a rolled-back transaction
leaves no observable trace.
"""

from __future__ import annotations

from rigging.timing import Timestamp
from sqlalchemy import bindparam, func
from sqlalchemy import literal as sa_literal
from sqlalchemy import update as sa_update

from iris.cluster.controller.audit_logging import log_event
from iris.cluster.controller.db import Tx
from iris.cluster.controller.projections.endpoints import EndpointsProjection
from iris.cluster.controller.reconcile.effects import (
    AttemptRowDelta,
    ControllerEffects,
    JobRowDelta,
    TaskRowDelta,
)
from iris.cluster.controller.reconcile.policy import CANCEL_GUARD_STATES
from iris.cluster.controller.schema import jobs_table, task_attempts_table, tasks_table
from iris.cluster.controller.task_state import ACTIVE_TASK_STATES
from iris.cluster.controller.worker_health import WorkerHealthTracker
from iris.cluster.types import TERMINAL_JOB_STATES
from iris.rpc import job_pb2


def _flush_tasks(cur: Tx, deltas: list[TaskRowDelta]) -> None:
    """Bulk-flush task deltas.

    Split by terminal vs active state: terminal rows additionally null out
    ``current_worker_id`` / ``current_worker_address`` (matching the legacy
    ``TaskMutation.apply``). ``container_id`` is applied in a small secondary
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
            # failure/preemption counts are set unconditionally; None means
            # "leave column unchanged" via coalesce(new, col).
            "b_failure_count": d.failure_count,
            "b_preemption_count": d.preemption_count,
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
        "failure_count": func.coalesce(bindparam("b_failure_count"), tasks_table.c.failure_count),
        "preemption_count": func.coalesce(bindparam("b_preemption_count"), tasks_table.c.preemption_count),
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


def _flush_jobs(cur: Tx, deltas: list[JobRowDelta]) -> None:
    """Bulk-flush job deltas in two groups: recompute writes and cascade kills.

    The kill group keeps the SQL ``WHERE state NOT IN guard`` guard for safety;
    the in-memory merge already enforces it. Cancel kills (which widen the guard)
    are issued separately so each executemany uses a single guard set.
    """
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

    # Cascade kills split by guard width so each executemany uses one guard set.
    for allow_overwrite in (False, True):
        kills = [d for d in deltas if d.is_cascade_kill and d.allow_overwrite_worker_failed == allow_overwrite]
        if not kills:
            continue
        guard_states = CANCEL_GUARD_STATES if allow_overwrite else TERMINAL_JOB_STATES
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
        # Render the guard as inline literals (NOT an expanding ``IN``): expanding
        # bind params are incompatible with executemany. The in-memory merge in
        # ``merge_cascade_kill`` already enforces this guard; the SQL guard is a
        # harmless safety net.
        guard_clause = c.state.not_in([sa_literal(s) for s in sorted(guard_states)])
        cur.execute(
            sa_update(jobs_table)
            .where(c.job_id == bindparam("b_job_id"), guard_clause)
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
    *,
    health: WorkerHealthTracker,
    endpoints: EndpointsProjection,
    now: Timestamp,
) -> None:
    """Record a batch's ``effects`` within the caller's write transaction.

    Row deltas flush via bulk ``executemany`` statements (one per entity group).
    Endpoint deletions write within the Tx. In-memory health bumps and audit log
    lines are deferred to ``cur``'s post-commit hooks so a rolled-back
    transaction leaves no observable trace.
    """
    _flush_tasks(cur, list(effects.tasks.values()))
    _flush_attempts(cur, list(effects.attempts.values()))
    _flush_jobs(cur, list(effects.jobs.values()))

    for d in effects.endpoint_deletions:
        endpoints.remove_by_task(cur, d.task_id)

    health_heartbeat = effects.health.heartbeat
    health_build_failed = effects.health.build_failed
    health_make_unhealthy = effects.health.make_unhealthy
    log_events = effects.log_events
    if health_heartbeat or health_build_failed or health_make_unhealthy or log_events:
        commit_ms = now.epoch_ms()

        def _post_commit() -> None:
            if health_heartbeat:
                health.heartbeat(list(health_heartbeat), commit_ms)
            for wid in health_build_failed:
                health.build_failed(wid)
            for wid in health_make_unhealthy:
                health.mark_unhealthy(wid)
            for ev in log_events:
                details = {k: v for k, v in ev.details}
                log_event(ev.action, ev.entity_id, trigger=ev.trigger, **details)

        cur.register(_post_commit)
