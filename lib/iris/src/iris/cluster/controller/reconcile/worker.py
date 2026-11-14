# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pure rules for the worker aggregate: planning + per-worker observation processing.

The primitives are otherwise pure, but emit free-form diagnostic logs inline
for dropped/unresolvable observations. Those logs are observability, not state,
so they do not flow through ``ControllerEffects``.
"""

import logging
from dataclasses import dataclass

from rigging.timing import Timestamp

from iris.cluster.controller.reconcile.snapshot import TaskUpdate
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
_STOP_REASON_BY_STATE: dict[int, worker_pb2.Worker.StopReason] = {
    job_pb2.TASK_STATE_KILLED: worker_pb2.Worker.STOP_REASON_CANCELLED,
    job_pb2.TASK_STATE_PREEMPTED: worker_pb2.Worker.STOP_REASON_PREEMPTED,
    # A controller-induced FAILED on a worker-bound attempt is an execution timeout.
    job_pb2.TASK_STATE_FAILED: worker_pb2.Worker.STOP_REASON_TASK_TIMEOUT,
}


def _stop_reason(row: "ReconcileRow") -> worker_pb2.Worker.StopReason | None:
    """Return the reason to stop this attempt, or ``None`` to leave it running."""
    # The attempt's own state decides: the task state can describe a newer
    # attempt or a rollback to PENDING. A terminal attempt in a reconcile row is
    # one the worker has not confirmed as gone, and only a 'stop' makes the
    # worker send the terminal observation that finalizes it.
    if row.attempt_state in TERMINAL_TASK_STATES:
        return _STOP_REASON_BY_STATE.get(row.attempt_state, worker_pb2.Worker.STOP_REASON_JOB_TERMINATED)
    if row.superseded:
        return worker_pb2.Worker.STOP_REASON_SUPERSEDED
    if row.task_state in (job_pb2.TASK_STATE_KILLED, job_pb2.TASK_STATE_PREEMPTED):
        return _STOP_REASON_BY_STATE[row.task_state]
    return None


def _holds_assigned_dispatch(row: "ReconcileRow") -> bool:
    if row.attempt_state == job_pb2.TASK_STATE_PREEMPTED:
        return True
    return row.superseded and row.attempt_state in TERMINAL_TASK_STATES


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
    # The task has a newer attempt, so ``task_state`` does not describe this one.
    superseded: bool


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
    attempts: tuple[ReconcileRow, ...]


@dataclass(frozen=True)
class WorkerReconcileResult:
    """Unified per-worker reconcile outcome.

    ``observations`` is the (possibly empty) list of proto observations the
    apply layer should consume. ``error`` is set when the reconcile RPC
    outright failed; ``observations`` is then empty. ``self_healthy`` is the
    worker's own health verdict from the Reconcile response (always ``True`` on
    an RPC error, where it is meaningless): a worker that responds but reports
    unhealthy — e.g. a failed disk — is still counted as a liveness failure so
    it is eventually reaped.

    ``responder_worker_id`` is the ``worker_id`` the responding daemon stamped on
    its Reconcile response. It is the *answerer's* identity, which can differ
    from the targeted :attr:`worker_id` if the controller dialed a stale address
    that a different live worker now owns (GCP recycles a deleted worker's
    internal IP onto a new VM). ``None`` on an RPC error or when the daemon did
    not report an id.
    """

    worker_id: WorkerId
    observations: list[worker_pb2.Worker.AttemptObservation]
    error: str | None = None
    self_healthy: bool = True
    responder_worker_id: str | None = None


# ---------------------------------------------------------------------------
# Reconcile planning
# ---------------------------------------------------------------------------


def build_reconcile_plans(inputs: ReconcileInputs) -> list[WorkerReconcilePlan]:
    """Compute one reconcile plan per worker from a batch snapshot."""
    return [_reconcile_worker(wid, inputs.rows_by_worker.get(wid, []), inputs.job_specs) for wid in inputs.worker_ids]


def _reconcile_worker(
    worker_id: WorkerId,
    rows: list[ReconcileRow],
    job_specs: dict[JobName, job_pb2.RunTaskRequest],
) -> WorkerReconcilePlan:
    desired: list[worker_pb2.Worker.DesiredAttempt] = []

    # The scheduler can commit a task ASSIGNED onto a worker before the process
    # it replaces is gone: a preemptor onto its victim's worker, or a retry onto
    # the worker of its own old attempt. While such an attempt still occupies
    # this worker, withhold the ASSIGNED run-intents so the worker does not start
    # a process against a live one (libtpu fixed-port crash loop). The terminal
    # observation stamps ``finished_at_ms``, which drops the old attempt from the
    # next snapshot — in the healthy case, a hold of one reconcile cycle. Only a
    # terminal attempt holds: the controller drops a report for a nonterminal
    # superseded attempt, so that attempt would hold the worker forever.
    hold_assigned_dispatch = any(_holds_assigned_dispatch(row) for row in rows)

    for row in rows:
        stop_reason = _stop_reason(row)
        if stop_reason is not None:
            desired.append(worker_pb2.Worker.DesiredAttempt(attempt_uid=row.attempt_uid, stop=stop_reason))
        elif row.task_state in _ASSIGNED_STATES:
            if hold_assigned_dispatch:
                continue
            spec = job_specs.get(row.job_id)
            if spec is None:
                # Job disappeared mid-tick; the scheduler reissues on a
                # subsequent tick.
                continue
            req = job_pb2.RunTaskRequest()
            req.CopyFrom(spec)
            req.task_id = row.task_id.to_wire()
            req.attempt_id = row.attempt_id
            req.attempt_uid = row.attempt_uid
            desired.append(
                worker_pb2.Worker.DesiredAttempt(
                    attempt_uid=row.attempt_uid,
                    run=worker_pb2.Worker.AttemptSpec(request=req),
                )
            )
        elif row.task_state in EXECUTING_TASK_STATES or row.task_state in TERMINAL_TASK_STATES:
            # Executing, or stranded: the task is terminal but its attempt is
            # not. Poll without a spec so the worker reports its real status, or
            # MISSING if it has forgotten the attempt, which finalizes it.
            desired.append(
                worker_pb2.Worker.DesiredAttempt(attempt_uid=row.attempt_uid, run=worker_pb2.Worker.AttemptSpec())
            )
        # A PENDING task with a live attempt is not planned for.

    return WorkerReconcilePlan(
        worker_id=worker_id,
        request=worker_pb2.Worker.ReconcileRequest(
            worker_id=worker_id,
            desired=desired,
        ),
        attempts=tuple(rows),
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
        logger.warning("worker reconcile: worker %s sent %d observations outside the plan; dropping", worker_id, dropped)
    return kept


def observations_to_updates(
    plan: WorkerReconcilePlan,
    observations: list[worker_pb2.Worker.AttemptObservation],
    *,
    observed_at: Timestamp,
) -> list[TaskUpdate]:
    """Translate worker protocol observations into exact task observations."""
    attempts_by_uid = {row.attempt_uid: row for row in plan.attempts}
    updates: list[TaskUpdate] = []
    for obs in observations:
        if not obs.attempt_uid:
            logger.warning("AttemptObservation missing attempt_uid; skipping: %s", obs)
            continue
        attempt_uid = AttemptUid(obs.attempt_uid)
        row = attempts_by_uid.get(attempt_uid)
        if row is None:
            logger.warning("AttemptObservation uid=%s was not in the reconcile plan; skipping", obs.attempt_uid)
            continue
        # proto3 has no presence for scalar ``exit_code``: an unset field and a
        # genuine 0 both arrive as 0. Treat 0 as "no exit code reported" so a
        # default-valued observation doesn't overwrite a previously recorded
        # exit code; a real exit 0 is conveyed by the SUCCEEDED terminal state.
        exit_code: int | None = obs.exit_code if obs.exit_code != 0 else None
        error: str | None = obs.error or None
        container_id: str | None = obs.container_id or None
        output_archive = obs.output_archive if obs.HasField("output_archive") else None
        updates.append(
            TaskUpdate(
                task_id=row.task_id,
                attempt_id=row.attempt_id,
                attempt_uid=attempt_uid,
                worker_id=plan.worker_id,
                execution_started_at=observed_at if obs.state == job_pb2.TASK_STATE_BUILDING else None,
                new_state=obs.state,
                error=error,
                exit_code=exit_code,
                container_id=container_id,
                status_message=obs.status_message,
                output_archive=output_archive,
            )
        )
    return updates


def failed_launch_updates(plan: WorkerReconcilePlan, error: str) -> list[TaskUpdate]:
    """Report launches rejected because their worker RPC failed."""
    attempts_by_uid = {row.attempt_uid: row for row in plan.attempts}
    updates: list[TaskUpdate] = []
    for desired in plan.request.desired:
        if not desired.HasField("run") or not desired.run.HasField("request"):
            continue
        row = attempts_by_uid.get(AttemptUid(desired.attempt_uid))
        if row is None:
            continue
        updates.append(
            TaskUpdate(
                task_id=row.task_id,
                attempt_id=row.attempt_id,
                attempt_uid=row.attempt_uid,
                worker_id=plan.worker_id,
                new_state=job_pb2.TASK_STATE_WORKER_FAILED,
                observed_task_state=job_pb2.TASK_STATE_ASSIGNED,
                error=f"Reconcile RPC failed: {error}",
            )
        )
    return updates


def task_updates_from_result(
    plan: WorkerReconcilePlan,
    result: WorkerReconcileResult,
    *,
    observed_at: Timestamp,
) -> list[TaskUpdate]:
    """Normalize one worker RPC result into the shared task observation shape."""
    if result.error is not None:
        return failed_launch_updates(plan, result.error)
    observations = filter_observations_to_plan(plan, result.observations, plan.worker_id)
    return observations_to_updates(plan, observations, observed_at=observed_at)
