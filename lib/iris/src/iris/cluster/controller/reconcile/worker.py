# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pure rules for the worker aggregate: planning + per-worker observation processing.

The primitives are otherwise pure, but emit free-form diagnostic logs inline
for dropped/unresolvable observations. Those logs are observability, not state,
so they do not flow through ``ControllerEffects``.
"""

import logging
from dataclasses import dataclass

from iris.cluster.controller.reconcile.snapshot import TaskUpdate, TransitionSnapshot
from iris.cluster.controller.task_state import (
    ACTIVE_TASK_STATES,
    EXECUTING_TASK_STATES,
)
from iris.cluster.types import (
    TERMINAL_TASK_STATES,
    AttemptUid,
    JobName,
    WorkerId,
)
from iris.rpc import job_pb2, worker_pb2

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_ASSIGNED_STATES: frozenset[int] = ACTIVE_TASK_STATES - EXECUTING_TASK_STATES
_TERMINAL_EXPECTED_STATES: frozenset[int] = TERMINAL_TASK_STATES - {
    job_pb2.TASK_STATE_KILLED,
    job_pb2.TASK_STATE_PREEMPTED,
}


# ---------------------------------------------------------------------------
# Reconcile planner inputs/outputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ReconcileRow:
    """One (task, attempt, worker) tuple driving per-worker reconcile."""

    worker_id: WorkerId
    task_id: JobName
    attempt_id: int
    task_state: int
    attempt_state: int
    job_id: JobName
    attempt_uid: AttemptUid


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


# ---------------------------------------------------------------------------
# Reconcile planning
# ---------------------------------------------------------------------------


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
            req.attempt_uid = row.attempt_uid
            desired.append(
                worker_pb2.Worker.DesiredAttempt(
                    attempt_uid=row.attempt_uid,
                    task_id=wire_task_id,
                    attempt_id=row.attempt_id,
                    run=worker_pb2.Worker.AttemptSpec(request=req),
                )
            )
        elif row.task_state in EXECUTING_TASK_STATES:
            desired.append(
                worker_pb2.Worker.DesiredAttempt(
                    attempt_uid=row.attempt_uid,
                    task_id=wire_task_id,
                    attempt_id=row.attempt_id,
                    run=worker_pb2.Worker.AttemptSpec(),
                )
            )
        elif row.task_state == job_pb2.TASK_STATE_KILLED:
            desired.append(
                worker_pb2.Worker.DesiredAttempt(
                    attempt_uid=row.attempt_uid,
                    task_id=wire_task_id,
                    attempt_id=row.attempt_id,
                    stop=worker_pb2.Worker.STOP_REASON_CANCELLED,
                )
            )
        elif row.task_state == job_pb2.TASK_STATE_PREEMPTED:
            desired.append(
                worker_pb2.Worker.DesiredAttempt(
                    attempt_uid=row.attempt_uid,
                    task_id=wire_task_id,
                    attempt_id=row.attempt_id,
                    stop=worker_pb2.Worker.STOP_REASON_PREEMPTED,
                )
            )
        elif row.task_state in _TERMINAL_EXPECTED_STATES:
            desired.append(
                worker_pb2.Worker.DesiredAttempt(
                    attempt_uid=row.attempt_uid,
                    task_id=wire_task_id,
                    attempt_id=row.attempt_id,
                    run=worker_pb2.Worker.AttemptSpec(),
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


# ---------------------------------------------------------------------------
# Plan/observation primitives (consumed by batches.py)
# ---------------------------------------------------------------------------


def filter_observations_to_plan(
    plan: WorkerReconcilePlan,
    observations: list[worker_pb2.Worker.AttemptObservation],
    worker_id: WorkerId,
) -> list[worker_pb2.Worker.AttemptObservation]:
    """Drop observations whose attempt is not in the per-worker plan we sent."""
    plan_uids: set[str] = {desired.attempt_uid for desired in plan.request.desired if desired.attempt_uid}

    kept: list[worker_pb2.Worker.AttemptObservation] = []
    dropped = 0
    for obs in observations:
        if obs.attempt_uid and obs.attempt_uid in plan_uids:
            kept.append(obs)
        else:
            dropped += 1
    if dropped:
        logger.warning("apply_reconcile: worker %s sent %d observations outside the plan; dropping", worker_id, dropped)
    return kept


def observations_to_updates(
    snapshot: TransitionSnapshot,
    observations: list[worker_pb2.Worker.AttemptObservation],
) -> list[TaskUpdate]:
    """Translate ``AttemptObservation`` protos to ``TaskUpdate``s."""
    updates: list[TaskUpdate] = []
    for obs in observations:
        if not obs.attempt_uid:
            logger.warning("AttemptObservation missing attempt_uid; skipping: %s", obs)
            continue
        resolved = snapshot.attempt_uid_index.get(AttemptUid(obs.attempt_uid))
        if resolved is None:
            logger.warning("AttemptObservation uid=%s did not resolve to an attempt row; skipping", obs.attempt_uid)
            continue
        task_id, attempt_id = resolved
        exit_code: int | None = obs.exit_code if obs.exit_code != 0 else None
        error: str | None = obs.error or None
        container_id: str | None = obs.container_id or None
        if obs.state == job_pb2.TASK_STATE_MISSING:
            updates.append(
                TaskUpdate(
                    task_id=task_id,
                    attempt_id=attempt_id,
                    new_state=job_pb2.TASK_STATE_FAILED,
                    error="worker_lost_spec",
                )
            )
        else:
            updates.append(
                TaskUpdate(
                    task_id=task_id,
                    attempt_id=attempt_id,
                    new_state=obs.state,
                    error=error,
                    exit_code=exit_code,
                    container_id=container_id,
                )
            )
    return updates


def assigned_updates_from_plan(
    snapshot: TransitionSnapshot,
    candidates: list[tuple[JobName, int]],
    error: str,
) -> list[TaskUpdate]:
    """Return synthetic WORKER_FAILED updates for ASSIGNED attempts in the plan."""
    updates: list[TaskUpdate] = []
    for task_id, attempt_id in candidates:
        task = snapshot.tasks.get(task_id)
        if task is None:
            continue
        if task.state != job_pb2.TASK_STATE_ASSIGNED:
            continue
        updates.append(
            TaskUpdate(
                task_id=task_id,
                attempt_id=attempt_id,
                new_state=job_pb2.TASK_STATE_WORKER_FAILED,
                error=f"Reconcile RPC failed: {error}",
            )
        )
    return updates
