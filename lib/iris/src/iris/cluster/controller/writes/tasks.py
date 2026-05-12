# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SA Core write helpers for the ``tasks`` table.

Stage 11 of the SA Core migration. Ports every write on
:class:`iris.cluster.controller.stores.TaskStore` into module-level
functions taking a :class:`iris.cluster.controller.db_v2.Tx`. The
``set_state_for_test`` helper intentionally remains on ``TaskStore``;
it is a test-only fixture, not part of production transitions.
"""

from collections.abc import Sequence

from sqlalchemy import text

from iris.cluster.controller.db_v2 import Tx
from iris.cluster.controller.schema_v2 import task_attempts_table, tasks_table
from iris.cluster.controller.stores import TaskAttemptInsertParams, TaskInsertParams, TaskStateUpdateParams
from iris.cluster.controller.writes import writes_to
from iris.cluster.controller.writes.task_attempts import insert_attempt
from iris.cluster.types import JobName, WorkerId
from iris.rpc import job_pb2

_INSERT_TASK_SQL = text(
    "INSERT INTO tasks("
    "task_id, job_id, task_index, state, error, exit_code, submitted_at_ms, started_at_ms, "
    "finished_at_ms, max_retries_failure, max_retries_preemption, failure_count, preemption_count, "
    "current_attempt_id, priority_neg_depth, priority_root_submitted_ms, "
    "priority_insertion, priority_band"
    ") VALUES ("
    ":task_id, :job_id, :task_index, :state, NULL, NULL, :submitted_at_ms, NULL, NULL, "
    ":max_retries_failure, :max_retries_preemption, 0, 0, -1, "
    ":priority_neg_depth, :priority_root_submitted_ms, :priority_insertion, :priority_band)"
)

_UPDATE_CONTAINER_ID_SQL = text("UPDATE tasks SET container_id = :container_id WHERE task_id = :task_id")


@writes_to(tasks_table)
def insert_task(tx: Tx, params: TaskInsertParams) -> None:
    """Insert one row into ``tasks``."""
    tx.execute(
        _INSERT_TASK_SQL,
        {
            "task_id": params.task_id.to_wire(),
            "job_id": params.job_id.to_wire(),
            "task_index": params.task_index,
            "state": params.state,
            "submitted_at_ms": params.submitted_at_ms,
            "max_retries_failure": params.max_retries_failure,
            "max_retries_preemption": params.max_retries_preemption,
            "priority_neg_depth": params.priority_neg_depth,
            "priority_root_submitted_ms": params.priority_root_submitted_ms,
            "priority_insertion": params.priority_insertion,
            "priority_band": params.priority_band,
        },
    )


@writes_to(tasks_table)
def mark_assigned(
    tx: Tx,
    task_id: JobName,
    attempt_id: int,
    worker_id: WorkerId | None,
    worker_address: str | None,
    now_ms: int,
    priority_band: int | None = None,
) -> None:
    """Move a task to ``TASK_STATE_ASSIGNED`` and stamp worker / attempt fields.

    ``priority_band`` is stamped at assign time so the preemption pass
    treats a running task's band as fixed. ``None`` leaves the column
    untouched (paths that do not run the budget computation).
    """
    band_set = "" if priority_band is None else ", priority_band = :priority_band"
    params: dict[str, object] = {
        "state": job_pb2.TASK_STATE_ASSIGNED,
        "attempt_id": attempt_id,
        "now_ms": now_ms,
        "task_id": task_id.to_wire(),
    }
    if priority_band is not None:
        params["priority_band"] = priority_band
    if worker_id is not None:
        stmt = text(
            "UPDATE tasks SET state = :state, current_attempt_id = :attempt_id, "
            "current_worker_id = :worker_id, current_worker_address = :worker_address, "
            f"started_at_ms = COALESCE(started_at_ms, :now_ms){band_set} WHERE task_id = :task_id"
        )
        params["worker_id"] = str(worker_id)
        params["worker_address"] = worker_address
        tx.execute(stmt, params)
        return
    stmt = text(
        "UPDATE tasks SET state = :state, current_attempt_id = :attempt_id, "
        f"started_at_ms = COALESCE(started_at_ms, :now_ms){band_set} WHERE task_id = :task_id"
    )
    tx.execute(stmt, params)


@writes_to(tasks_table, task_attempts_table)
def assign_task(
    tx: Tx,
    task_id: JobName,
    worker_id: WorkerId | None,
    worker_address: str | None,
    attempt_id: int,
    now_ms: int,
    priority_band: int | None = None,
) -> None:
    """Insert a fresh ``task_attempts`` row and move the task to ASSIGNED.

    Composite write that mirrors :meth:`TaskStore.assign`: a single
    transaction creates the attempt row and stamps the task with the
    worker / attempt fields.
    """
    insert_attempt(
        tx,
        TaskAttemptInsertParams(
            task_id=task_id,
            attempt_id=attempt_id,
            worker_id=worker_id,
            state=job_pb2.TASK_STATE_ASSIGNED,
            created_at_ms=now_ms,
        ),
    )
    mark_assigned(tx, task_id, attempt_id, worker_id, worker_address, now_ms, priority_band=priority_band)


@writes_to(tasks_table)
def apply_state_update(
    tx: Tx,
    params: TaskStateUpdateParams,
    active_states: set[int],
) -> None:
    """Apply a computed task state update.

    Active target states preserve ``current_worker_id`` /
    ``current_worker_address``; non-active states clear them so the row
    is consistent with terminal-transition writes.
    """
    bind = {
        "state": params.state,
        "error": params.error,
        "exit_code": params.exit_code,
        "started_at_ms": params.started_at_ms,
        "finished_at_ms": params.finished_at_ms,
        "failure_count": params.failure_count,
        "preemption_count": params.preemption_count,
        "task_id": params.task_id.to_wire(),
    }
    if params.state in active_states:
        tx.execute(
            text(
                "UPDATE tasks SET state = :state, error = COALESCE(:error, error), "
                "exit_code = COALESCE(:exit_code, exit_code), "
                "started_at_ms = COALESCE(started_at_ms, :started_at_ms), "
                "finished_at_ms = :finished_at_ms, "
                "failure_count = :failure_count, preemption_count = :preemption_count "
                "WHERE task_id = :task_id"
            ),
            bind,
        )
        return
    tx.execute(
        text(
            "UPDATE tasks SET state = :state, error = COALESCE(:error, error), "
            "exit_code = COALESCE(:exit_code, exit_code), "
            "started_at_ms = COALESCE(started_at_ms, :started_at_ms), "
            "finished_at_ms = :finished_at_ms, "
            "failure_count = :failure_count, preemption_count = :preemption_count, "
            "current_worker_id = NULL, current_worker_address = NULL "
            "WHERE task_id = :task_id"
        ),
        bind,
    )


@writes_to(tasks_table)
def mark_terminal(
    tx: Tx,
    task_id: JobName,
    state: int,
    error: str | None,
    finished_at_ms: int | None,
    *,
    failure_count: int | None = None,
    preemption_count: int | None = None,
    active_states: set[int],
) -> None:
    """Move a task to a terminal-style state, optionally updating counters.

    Mirrors :meth:`TaskStore.mark_terminal`: clears ``current_worker_*``
    when the target state is not active; preserves the existing
    ``finished_at_ms`` if one is already set and ``finished_at_ms`` is
    not None (COALESCE), otherwise sets it directly.
    """
    if finished_at_ms is not None:
        set_clauses = [
            "state = :state",
            "error = :error",
            "finished_at_ms = COALESCE(finished_at_ms, :finished_at_ms)",
        ]
    else:
        set_clauses = ["state = :state", "error = :error", "finished_at_ms = :finished_at_ms"]
    params: dict[str, object] = {
        "state": state,
        "error": error,
        "finished_at_ms": finished_at_ms,
        "task_id": task_id.to_wire(),
    }
    if failure_count is not None:
        set_clauses.append("failure_count = :failure_count")
        params["failure_count"] = failure_count
    if preemption_count is not None:
        set_clauses.append("preemption_count = :preemption_count")
        params["preemption_count"] = preemption_count
    if state not in active_states:
        set_clauses.append("current_worker_id = NULL")
        set_clauses.append("current_worker_address = NULL")
    tx.execute(
        text(f"UPDATE tasks SET {', '.join(set_clauses)} WHERE task_id = :task_id"),
        params,
    )


@writes_to(tasks_table)
def bulk_kill_non_terminal(
    tx: Tx,
    job_ids: Sequence[JobName],
    reason: str,
    finished_at_ms: int,
    terminal_states: set[int],
) -> None:
    """Mark all non-terminal tasks under ``job_ids`` as ``TASK_STATE_KILLED``."""
    if not job_ids:
        return
    wire_ids = [jid.to_wire() for jid in job_ids]
    terminal = tuple(terminal_states)
    job_keys = [f"j{i}" for i in range(len(wire_ids))]
    term_keys = [f"t{i}" for i in range(len(terminal))]
    stmt = text(
        "UPDATE tasks SET state = :state, error = :error, "
        "finished_at_ms = COALESCE(finished_at_ms, :finished_at_ms), "
        "current_worker_id = NULL, current_worker_address = NULL "
        f"WHERE job_id IN ({','.join(f':{k}' for k in job_keys)}) "
        f"AND state NOT IN ({','.join(f':{k}' for k in term_keys)})"
    )
    params: dict[str, object] = {
        "state": job_pb2.TASK_STATE_KILLED,
        "error": reason,
        "finished_at_ms": finished_at_ms,
    }
    for k, v in zip(job_keys, wire_ids, strict=True):
        params[k] = v
    for k, v in zip(term_keys, terminal, strict=True):
        params[k] = v
    tx.execute(stmt, params)


@writes_to(tasks_table)
def update_container_id(tx: Tx, task_id: JobName, container_id: str) -> None:
    """Update ``tasks.container_id`` for ``task_id``."""
    tx.execute(_UPDATE_CONTAINER_ID_SQL, {"container_id": container_id, "task_id": task_id.to_wire()})
