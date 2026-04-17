# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Test-only DB mutation helpers for controller tests.

These functions reach directly into the SQLite DB to set state that is
difficult or impossible to reach through normal controller transitions.
They live here (not in production code) to keep transitions.py clean.
"""

from iris.cluster.constraints import AttributeValue
from iris.cluster.controller.db import ControllerDB
from iris.cluster.controller.schema import ACTIVE_TASK_STATES
from iris.cluster.controller.store import ControllerStores, WorkerAssignment
from iris.cluster.types import JobName, WorkerId
from rigging.timing import Timestamp


def set_worker_health(db: ControllerDB, worker_id: WorkerId, *, healthy: bool) -> None:
    """Set worker health directly in DB."""
    db.execute(
        "UPDATE workers SET healthy = ?, consecutive_failures = ? WHERE worker_id = ?",
        (1 if healthy else 0, 0 if healthy else 1, str(worker_id)),
    )


def set_worker_attribute(db: ControllerDB, worker_id: WorkerId, key: str, value: AttributeValue) -> None:
    """Upsert one worker attribute directly in DB."""
    str_value = int_value = float_value = None
    value_type = "str"
    if isinstance(value.value, int):
        value_type = "int"
        int_value = int(value.value)
    elif isinstance(value.value, float):
        value_type = "float"
        float_value = float(value.value)
    else:
        str_value = str(value.value)

    db.execute(
        "INSERT INTO worker_attributes(worker_id, key, value_type, str_value, int_value, float_value) "
        "VALUES (?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(worker_id, key) DO UPDATE SET "
        "value_type=excluded.value_type, "
        "str_value=excluded.str_value, "
        "int_value=excluded.int_value, "
        "float_value=excluded.float_value",
        (str(worker_id), key, value_type, str_value, int_value, float_value),
    )


def set_worker_consecutive_failures(db: ControllerDB, worker_id: WorkerId, consecutive_failures: int) -> None:
    """Set worker consecutive failure count directly in DB."""
    db.execute(
        "UPDATE workers SET consecutive_failures = ? WHERE worker_id = ?",
        (consecutive_failures, str(worker_id)),
    )


def set_task_state(
    db: ControllerDB,
    task_id: JobName,
    state: int,
    *,
    error: str | None = None,
    exit_code: int | None = None,
) -> None:
    """Set task state directly in DB."""
    if state in ACTIVE_TASK_STATES:
        db.execute(
            "UPDATE tasks SET state = ?, error = ?, exit_code = ? WHERE task_id = ?",
            (state, error, exit_code, task_id.to_wire()),
        )
    else:
        db.execute(
            "UPDATE tasks SET state = ?, error = ?, exit_code = ?, "
            "current_worker_id = NULL, current_worker_address = NULL WHERE task_id = ?",
            (state, error, exit_code, task_id.to_wire()),
        )


def create_attempt(stores: ControllerStores, task_id: JobName, worker_id: WorkerId) -> int:
    """Append a new task_attempt without finalizing the prior attempt."""
    db = stores.db
    task = db.fetchone("SELECT current_attempt_id FROM tasks WHERE task_id = ?", (task_id.to_wire(),))
    if task is None:
        raise ValueError(f"unknown task: {task_id}")
    worker_row = db.fetchone("SELECT address FROM workers WHERE worker_id = ?", (str(worker_id),))
    worker_address = str(worker_row["address"]) if worker_row is not None else str(worker_id)
    next_attempt_id = int(task["current_attempt_id"]) + 1
    now_ms = Timestamp.now().epoch_ms()
    with stores.transact() as ctx:
        ctx.tasks.assign_to_worker(
            ctx.cur,
            WorkerAssignment(
                task_id=task_id.to_wire(),
                attempt_id=next_attempt_id,
                worker_id=str(worker_id),
                worker_address=worker_address,
                now_ms=now_ms,
            ),
        )
    return next_attempt_id
