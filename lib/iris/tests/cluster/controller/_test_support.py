# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Test-support helpers and a controller projection carrier for tests.

After Stage 7 (transitions-pure.md), production no longer has a
``ControllerTransitions`` class to hang test helpers off of. Tests still
want a single handle to pass through helpers that drive ``commands.*`` /
``reconcile_io.*`` flows, so this module defines :class:`ControllerTestState`: a thin
read-only bundle of the projections + DB + cache that those helpers need.

``ControllerTestState`` is intentionally test-only — production wires the same
projections through ``Controller`` constructor arguments.
"""

from dataclasses import dataclass

from iris.cluster.constraints import AttributeValue
from iris.cluster.controller import writes
from iris.cluster.controller.db import ControllerDB
from iris.cluster.controller.projections.attempt_counts import AttemptCountsProjection
from iris.cluster.controller.projections.endpoints import EndpointsProjection
from iris.cluster.controller.projections.run_templates import RunTemplatesProjection
from iris.cluster.controller.projections.worker_attrs import WorkerAttrsProjection
from iris.cluster.controller.schema import (
    tasks_table,
    workers_table,
)
from iris.cluster.controller.task_state import ACTIVE_TASK_STATES
from iris.cluster.controller.worker_health import WorkerHealthTracker
from iris.cluster.types import JobName, WorkerId
from rigging.timing import Timestamp
from sqlalchemy import bindparam, select
from sqlalchemy import update as sa_update


@dataclass
class ControllerTestState:
    """Projection bundle for tests that drive ``commands.*`` / ``reconcile_io.*``
    without booting a full ``Controller``.

    Field names match the underscored ones a single-backend :class:`Controller`
    exposes through its default backend (``_db``, ``_health``) so the same helpers
    work against either. ``_endpoints`` is a convenience property backed by the DB
    cache registry.
    """

    _db: ControllerDB
    _health: WorkerHealthTracker
    _worker_attrs: WorkerAttrsProjection

    def __init__(
        self,
        db: ControllerDB,
        *,
        health: WorkerHealthTracker | None = None,
    ) -> None:
        self._db = db
        # Mirror a real Controller: each Projection self-registers into
        # ``db.caches`` on construction; cursors the DB mints reach them as
        # ``tx.caches[Projection]`` when helpers commit effects / purge jobs /
        # read derived counts.
        AttemptCountsProjection(db)
        self._health = health or WorkerHealthTracker()
        EndpointsProjection(db)
        RunTemplatesProjection(db)
        self._worker_attrs = WorkerAttrsProjection(db)

    @property
    def _endpoints(self) -> EndpointsProjection:
        """The endpoints projection, looked up from the DB cache registry."""
        return self._db.caches[EndpointsProjection]


def set_worker_health_for_test(ctrl: ControllerTestState, worker_id: WorkerId, healthy: bool) -> None:
    """Set worker health in the in-memory tracker."""
    ctrl._health.set_health_for_test(worker_id, healthy)


def set_worker_attribute_for_test(
    ctrl: ControllerTestState, worker_id: WorkerId, key: str, value: AttributeValue
) -> None:
    """Upsert one worker attribute in DB and mirror it into the in-memory projection."""
    with ctrl._db.transaction() as cur:
        existing = ctrl._worker_attrs.get(worker_id)
        merged = {**existing, key: value}
        ctrl._worker_attrs.set(cur, worker_id, merged)


def set_worker_consecutive_failures_for_test(
    ctrl: ControllerTestState, worker_id: WorkerId, consecutive_failures: int
) -> None:
    """Set worker consecutive failure count in the in-memory tracker."""
    ctrl._health.set_consecutive_failures_for_test(worker_id, consecutive_failures)


def set_task_state_for_test(
    ctrl: ControllerTestState,
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


def create_attempt_for_test(ctrl: ControllerTestState, task_id: JobName, worker_id: WorkerId) -> int:
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
