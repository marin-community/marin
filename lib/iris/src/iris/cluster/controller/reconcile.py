# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pure-compute reconcile layer for the Iris controller."""

from __future__ import annotations

from dataclasses import dataclass

from iris.cluster.controller.task_state import ACTIVE_TASK_STATES, EXECUTING_TASK_STATES
from iris.cluster.types import TERMINAL_TASK_STATES, JobName, WorkerId
from iris.rpc import job_pb2, worker_pb2


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
class ReconcileInputs:
    """Snapshot driving one reconcile tick across all active workers."""

    job_specs: dict[JobName, job_pb2.RunTaskRequest]
    worker_ids: list[WorkerId]
    rows_by_worker: dict[WorkerId, list[ReconcileRow]]


@dataclass(frozen=True)
class WorkerReconcilePlan:
    """Reconcile decision for one worker: the proto payload to send."""

    worker_id: WorkerId
    request: worker_pb2.Worker.ReconcileRequest


@dataclass(frozen=True)
class ReconcileResult:
    """Unified per-worker reconcile outcome.

    ``observations`` is the (possibly empty) list of proto observations the
    apply layer should consume. ``error`` is set when the reconcile RPC
    outright failed; ``observations`` is then empty.
    """

    worker_id: WorkerId
    observations: list[worker_pb2.Worker.AttemptObservation]
    error: str | None = None


_ASSIGNED_STATES: frozenset[int] = ACTIVE_TASK_STATES - EXECUTING_TASK_STATES
_TERMINAL_EXPECTED_STATES: frozenset[int] = TERMINAL_TASK_STATES - {
    job_pb2.TASK_STATE_KILLED,
    job_pb2.TASK_STATE_PREEMPTED,
}


def reconcile_workers(inputs: ReconcileInputs) -> list[WorkerReconcilePlan]:
    """Compute one reconcile plan per worker from a batch snapshot."""
    return [_reconcile_worker(wid, inputs.rows_by_worker.get(wid, []), inputs.job_specs) for wid in inputs.worker_ids]


def _reconcile_worker(
    worker_id: WorkerId,
    rows: list[ReconcileRow],
    job_specs: dict[JobName, job_pb2.RunTaskRequest],
) -> WorkerReconcilePlan:
    desired: list[worker_pb2.Worker.DesiredAttempt] = []

    for row in rows:
        wire_task_id = row.task_id.to_wire()
        if row.task_state in _ASSIGNED_STATES:
            spec = job_specs.get(row.job_id)
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
        elif row.task_state in _TERMINAL_EXPECTED_STATES:
            desired.append(
                worker_pb2.Worker.DesiredAttempt(
                    attempt_uid="",
                    run=worker_pb2.Worker.AttemptSpec(),
                    task_id=wire_task_id,
                    attempt_id=row.attempt_id,
                )
            )
        # Unrecognised states are omitted from desired.

    return WorkerReconcilePlan(
        worker_id=worker_id,
        request=worker_pb2.Worker.ReconcileRequest(
            worker_id=worker_id,
            desired=desired,
        ),
    )
