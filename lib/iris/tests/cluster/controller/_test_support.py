# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Test-support helpers for ControllerTransitions.

Module-level functions that previously lived on ControllerTransitions as
``*_for_test`` methods. Each function takes ``ctrl: ControllerTransitions``
as its first argument and mutates state directly via the controller's DB.
"""

from iris.cluster.constraints import AttributeValue
from iris.cluster.controller import writes
from iris.cluster.controller.schema import (
    tasks_table,
    worker_attributes_table,
    workers_table,
)
from iris.cluster.controller.task_state import ACTIVE_TASK_STATES
from iris.cluster.controller.transitions import ControllerTransitions
from iris.cluster.types import JobName, WorkerId
from rigging.timing import Timestamp
from sqlalchemy import bindparam, select
from sqlalchemy import update as sa_update
from sqlalchemy.dialects.sqlite import insert as sqlite_insert


def set_worker_health_for_test(ctrl: ControllerTransitions, worker_id: WorkerId, healthy: bool) -> None:
    """Set worker health in the in-memory tracker."""
    ctrl._health.set_health_for_test(worker_id, healthy)


def set_worker_attribute_for_test(
    ctrl: ControllerTransitions, worker_id: WorkerId, key: str, value: AttributeValue
) -> None:
    """Upsert one worker attribute in DB and mirror it into the in-memory projection."""
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

    with ctrl._db.transaction() as cur:
        cur.execute(
            sqlite_insert(worker_attributes_table)
            .values(
                worker_id=worker_id,
                key=key,
                value_type=value_type,
                str_value=str_value,
                int_value=int_value,
                float_value=float_value,
            )
            .on_conflict_do_update(
                index_elements=["worker_id", "key"],
                set_=dict(
                    value_type=value_type,
                    str_value=str_value,
                    int_value=int_value,
                    float_value=float_value,
                ),
            )
        )
        existing = ctrl._worker_attrs.get(worker_id)
        merged = {**existing, key: value}
        ctrl._worker_attrs.set(cur, worker_id, merged)


def set_worker_consecutive_failures_for_test(
    ctrl: ControllerTransitions, worker_id: WorkerId, consecutive_failures: int
) -> None:
    """Set worker consecutive failure count in the in-memory tracker."""
    ctrl._health.set_consecutive_failures_for_test(worker_id, consecutive_failures)


def set_task_state_for_test(
    ctrl: ControllerTransitions,
    task_id: JobName,
    state: int,
    *,
    error: str | None = None,
    exit_code: int | None = None,
) -> None:
    """Set task state directly in DB."""
    with ctrl._db.transaction() as cur:
        values: dict = {"state": state, "error": error, "exit_code": exit_code}
        if state not in ACTIVE_TASK_STATES:
            values["current_worker_id"] = None
            values["current_worker_address"] = None
        cur.execute(sa_update(tasks_table).where(tasks_table.c.task_id == task_id).values(**values))


def create_attempt_for_test(ctrl: ControllerTransitions, task_id: JobName, worker_id: WorkerId) -> int:
    """Append a new task_attempt without finalizing the prior attempt."""
    with ctrl._db.read_snapshot() as snap:
        _attempt_row = snap.execute(
            select(tasks_table.c.current_attempt_id).where(tasks_table.c.task_id == bindparam("task_id")),
            {"task_id": task_id},
        ).first()
    current_attempt_id = int(_attempt_row.current_attempt_id) if _attempt_row is not None else None
    if current_attempt_id is None:
        raise ValueError(f"unknown task: {task_id}")
    with ctrl._db.read_snapshot() as snap:
        _addr_row = snap.execute(
            select(workers_table.c.address).where(workers_table.c.worker_id == bindparam("worker_id")),
            {"worker_id": worker_id},
        ).first()
        worker_address = str(_addr_row.address) if _addr_row is not None else str(worker_id)
    next_attempt_id = current_attempt_id + 1
    now_ms = Timestamp.now().epoch_ms()
    with ctrl._db.transaction() as cur:
        writes.assign_to_worker(
            cur,
            task_id,
            worker_id,
            worker_address,
            next_attempt_id,
            now_ms,
            0,
        )
    return next_attempt_id
