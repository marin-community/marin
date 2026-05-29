# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Test driver for landing task-state updates through the production path.

The live controller lands worker-reported task states through the reconcile
loop (``ops.worker.apply_reconcile_observations``). To keep tests exercising
the same code the controller runs, ``apply_task_observations`` rebuilds a
per-worker batch of :class:`WorkerTaskUpdates` into reconcile
``AttemptObservation`` protos and applies them through that production verb.
"""

from dataclasses import dataclass

from iris.cluster.controller.db import Tx
from iris.cluster.controller.ops.worker import apply_reconcile_observations
from iris.cluster.controller.projections.endpoints import EndpointsProjection
from iris.cluster.controller.reconcile.effects import ControllerEffects
from iris.cluster.controller.reconcile.snapshot import TaskUpdate
from iris.cluster.controller.reconcile.worker import ReconcileResult, WorkerReconcilePlan
from iris.cluster.controller.schema import task_attempts_table
from iris.cluster.controller.worker_health import WorkerHealthTracker
from iris.cluster.types import JobName, WorkerId
from iris.rpc import worker_pb2
from rigging.timing import Timestamp
from sqlalchemy import select


@dataclass(frozen=True)
class WorkerTaskUpdates:
    """A worker reporting observed states for a batch of its attempts.

    A worker id plus the neutral per-task updates to land.
    """

    worker_id: WorkerId
    updates: list[TaskUpdate]


def _attempt_uid(cur: Tx, task_id: JobName, attempt_id: int) -> str:
    """Read the controller-minted attempt_uid that routes an observation."""
    row = cur.execute(
        select(task_attempts_table.c.attempt_uid).where(
            task_attempts_table.c.task_id == task_id,
            task_attempts_table.c.attempt_id == attempt_id,
        )
    ).first()
    assert row is not None, f"no task_attempts row for {task_id.to_wire()}/{attempt_id}"
    return row.attempt_uid


def _observation(uid: str, update: TaskUpdate) -> worker_pb2.Worker.AttemptObservation:
    return worker_pb2.Worker.AttemptObservation(
        attempt_uid=uid,
        state=update.new_state,
        exit_code=update.exit_code if update.exit_code is not None else 0,
        error=update.error or "",
        container_id=update.container_id or "",
    )


def apply_task_observations(
    cur: Tx,
    requests: list[WorkerTaskUpdates],
    *,
    health: WorkerHealthTracker,
    endpoints: EndpointsProjection,
    now: Timestamp,
) -> ControllerEffects:
    """Land ``requests`` through the production reconcile-observation verb.

    Builds one ``(WorkerReconcilePlan, ReconcileResult)`` pair per worker: the
    plan lists each touched attempt's uid as desired (so the production filter
    accepts the observation) and the result reports the observed state. An
    empty ``updates`` list still records a worker heartbeat, matching the old
    behaviour.
    """
    plans_by_worker: dict[WorkerId, WorkerReconcilePlan] = {}
    results: list[ReconcileResult] = []
    for req in requests:
        observations: list[worker_pb2.Worker.AttemptObservation] = []
        desired: list[worker_pb2.Worker.DesiredAttempt] = []
        for update in req.updates:
            uid = _attempt_uid(cur, update.task_id, update.attempt_id)
            observations.append(_observation(uid, update))
            desired.append(worker_pb2.Worker.DesiredAttempt(attempt_uid=uid, run=worker_pb2.Worker.AttemptSpec()))
        plans_by_worker[req.worker_id] = WorkerReconcilePlan(
            worker_id=req.worker_id,
            request=worker_pb2.Worker.ReconcileRequest(worker_id=str(req.worker_id), desired=desired),
        )
        results.append(ReconcileResult(worker_id=req.worker_id, observations=observations, error=None))

    return apply_reconcile_observations(cur, plans_by_worker, results, health=health, endpoints=endpoints, now=now)
