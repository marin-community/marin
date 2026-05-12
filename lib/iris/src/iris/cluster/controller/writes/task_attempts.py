# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SA Core write helpers for ``task_attempts``.

Stage 11 of the SA Core migration. Ports every write on
:class:`iris.cluster.controller.stores.TaskAttemptStore` into
module-level functions taking a :class:`iris.cluster.controller.db_v2.Tx`.
"""

from collections.abc import Sequence

from sqlalchemy import text

from iris.cluster.controller.db_v2 import Tx
from iris.cluster.controller.schema_v2 import task_attempts_table
from iris.cluster.controller.stores import TaskAttemptInsertParams, TaskAttemptUpdateParams
from iris.cluster.controller.writes import writes_to
from iris.cluster.types import JobName

_INSERT_ATTEMPT_SQL = text(
    "INSERT INTO task_attempts(task_id, attempt_id, worker_id, state, created_at_ms) "
    "VALUES (:task_id, :attempt_id, :worker_id, :state, :created_at_ms)"
)

_MARK_FINISHED_SQL = text(
    "UPDATE task_attempts SET state = :state, "
    "finished_at_ms = COALESCE(finished_at_ms, :finished_at_ms), error = :error "
    "WHERE task_id = :task_id AND attempt_id = :attempt_id"
)

_APPLY_ATTEMPT_STATE_SQL = text(
    "UPDATE task_attempts SET state = :state, error = COALESCE(:error, error) "
    "WHERE task_id = :task_id AND attempt_id = :attempt_id"
)

_APPLY_UPDATE_SQL = text(
    "UPDATE task_attempts SET state = :state, "
    "started_at_ms = COALESCE(started_at_ms, :started_at_ms), "
    "finished_at_ms = COALESCE(finished_at_ms, :finished_at_ms), "
    "exit_code = COALESCE(:exit_code, exit_code), "
    "error = COALESCE(:error, error) "
    "WHERE task_id = :task_id AND attempt_id = :attempt_id"
)


@writes_to(task_attempts_table)
def insert_attempt(tx: Tx, params: TaskAttemptInsertParams) -> None:
    """Insert one row into ``task_attempts``."""
    tx.execute(
        _INSERT_ATTEMPT_SQL,
        {
            "task_id": params.task_id.to_wire(),
            "attempt_id": params.attempt_id,
            "worker_id": str(params.worker_id) if params.worker_id is not None else None,
            "state": params.state,
            "created_at_ms": params.created_at_ms,
        },
    )


@writes_to(task_attempts_table)
def mark_finished(
    tx: Tx,
    task_id: JobName,
    attempt_id: int,
    state: int,
    finished_at_ms: int,
    error: str | None,
) -> None:
    """Stamp ``finished_at_ms`` and final state on an attempt."""
    tx.execute(
        _MARK_FINISHED_SQL,
        {
            "state": state,
            "finished_at_ms": finished_at_ms,
            "error": error,
            "task_id": task_id.to_wire(),
            "attempt_id": attempt_id,
        },
    )


@writes_to(task_attempts_table)
def apply_attempt_state(
    tx: Tx,
    task_id: JobName,
    attempt_id: int,
    state: int,
    error: str | None,
) -> None:
    """Update an attempt's reporting state without stamping ``finished_at_ms``.

    Producing transitions (cancel, preempt, timeout, gang cascade) call
    this to record the controller's intent while the worker still holds
    the container. The heartbeat path stamps ``finished_at_ms`` via
    :func:`mark_finished` once the worker confirms termination.
    """
    tx.execute(
        _APPLY_ATTEMPT_STATE_SQL,
        {
            "state": state,
            "error": error,
            "task_id": task_id.to_wire(),
            "attempt_id": attempt_id,
        },
    )


@writes_to(task_attempts_table)
def apply_update(tx: Tx, params: TaskAttemptUpdateParams) -> None:
    """Apply a worker/direct-provider attempt update."""
    tx.execute(
        _APPLY_UPDATE_SQL,
        {
            "state": params.state,
            "started_at_ms": params.started_at_ms,
            "finished_at_ms": params.finished_at_ms,
            "exit_code": params.exit_code,
            "error": params.error,
            "task_id": params.task_id.to_wire(),
            "attempt_id": params.attempt_id,
        },
    )


@writes_to(task_attempts_table)
def bulk_apply_attempt_state(
    tx: Tx,
    job_ids: Sequence[JobName],
    state: int,
    error: str,
    active_states: set[int],
) -> None:
    """Update reporting state on every active attempt under ``job_ids``."""
    if not job_ids:
        return
    wire_ids = [jid.to_wire() for jid in job_ids]
    active = tuple(active_states)
    job_keys = [f"j{i}" for i in range(len(wire_ids))]
    active_keys = [f"a{i}" for i in range(len(active))]
    stmt = text(
        "UPDATE task_attempts SET state = :state, error = COALESCE(error, :error) "
        "WHERE task_id IN ("
        f"  SELECT task_id FROM tasks WHERE job_id IN ({','.join(f':{k}' for k in job_keys)})"
        f") AND state IN ({','.join(f':{k}' for k in active_keys)})"
    )
    params: dict[str, object] = {"state": state, "error": error}
    for k, v in zip(job_keys, wire_ids, strict=True):
        params[k] = v
    for k, v in zip(active_keys, active, strict=True):
        params[k] = v
    tx.execute(stmt, params)
