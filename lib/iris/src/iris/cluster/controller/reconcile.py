# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pure-compute reconcile layer for the Iris controller.

``reconcile_worker`` takes a ``WorkerReconcileInputs`` snapshot and returns
a ``WorkerReconcilePlan`` carrying the ``Worker.ReconcileRequest`` proto
to send to one worker.
"""

from __future__ import annotations

from dataclasses import dataclass

from iris.cluster.controller.task_state import ACTIVE_TASK_STATES, EXECUTING_TASK_STATES
from iris.cluster.types import JobName, WorkerId
from iris.rpc import job_pb2, worker_pb2


@dataclass(frozen=True, slots=True)
class WorkerRow:
    """Durable worker columns: identity and address."""

    worker_id: WorkerId
    address: str


@dataclass(frozen=True, slots=True)
class ReconcileRow:
    """One (task, attempt, worker) tuple driving per-worker reconcile."""

    worker_id: WorkerId
    task_id: JobName
    attempt_id: int
    task_state: int
    attempt_state: int
    job_id: JobName


@dataclass(frozen=True)
class WorkerReconcileInputs:
    """All state needed to decide one worker's next desired set.

    Sole input to ``reconcile_worker``. ``job_specs`` maps ``JobName`` to
    the cached ``RunTaskRequest`` (or ``None`` if the spec is unavailable
    this tick).
    """

    worker: WorkerRow
    rows: list[ReconcileRow]
    job_specs: dict[JobName, job_pb2.RunTaskRequest]


@dataclass(frozen=True)
class WorkerReconcilePlan:
    """The reconcile decision for one worker: the proto payload to send."""

    worker_id: WorkerId
    request: worker_pb2.Worker.ReconcileRequest


# ASSIGNED is the one active state not in EXECUTING_TASK_STATES.
_ASSIGNED_STATES: frozenset[int] = ACTIVE_TASK_STATES - EXECUTING_TASK_STATES


def reconcile_worker(inputs: WorkerReconcileInputs) -> WorkerReconcilePlan:
    """Pure function: compute the reconcile plan for one worker.

    No DB, no RPC, no clock. ``AttemptSpec.request`` is populated only on
    the ASSIGNED dispatch tick; subsequent ticks rely on the worker's
    cached spec. ``attempt_uid`` is left empty — workers route by the
    legacy (task_id, attempt_id) composite key until UID routing lands.
    """
    desired: list[worker_pb2.Worker.DesiredAttempt] = []

    for row in inputs.rows:
        wire_task_id = row.task_id.to_wire()
        if row.task_state in _ASSIGNED_STATES:
            spec = inputs.job_specs.get(row.job_id)
            if spec is None:
                # Reservation holder or job disappeared mid-tick; the
                # scheduler reissues on a subsequent tick.
                continue
            req = job_pb2.RunTaskRequest()
            req.CopyFrom(spec)
            req.task_id = wire_task_id
            req.attempt_id = row.attempt_id
            desired.append(
                worker_pb2.Worker.DesiredAttempt(
                    attempt_uid="",
                    run=worker_pb2.Worker.AttemptSpec(request=req),
                    task_id=wire_task_id,
                    attempt_id=row.attempt_id,
                )
            )
        elif row.task_state in EXECUTING_TASK_STATES:
            desired.append(
                worker_pb2.Worker.DesiredAttempt(
                    attempt_uid="",
                    run=worker_pb2.Worker.AttemptSpec(),
                    task_id=wire_task_id,
                    attempt_id=row.attempt_id,
                )
            )
        elif row.task_state == job_pb2.TASK_STATE_KILLED:
            desired.append(
                worker_pb2.Worker.DesiredAttempt(
                    attempt_uid="",
                    stop=worker_pb2.Worker.STOP_REASON_CANCELLED,
                    task_id=wire_task_id,
                    attempt_id=row.attempt_id,
                )
            )
        elif row.task_state == job_pb2.TASK_STATE_PREEMPTED:
            desired.append(
                worker_pb2.Worker.DesiredAttempt(
                    attempt_uid="",
                    stop=worker_pb2.Worker.STOP_REASON_PREEMPTED,
                    task_id=wire_task_id,
                    attempt_id=row.attempt_id,
                )
            )
        # Terminal and unrecognised states are omitted from desired.

    return WorkerReconcilePlan(
        worker_id=inputs.worker.worker_id,
        request=worker_pb2.Worker.ReconcileRequest(
            worker_id=inputs.worker.worker_id,
            desired=desired,
        ),
    )
