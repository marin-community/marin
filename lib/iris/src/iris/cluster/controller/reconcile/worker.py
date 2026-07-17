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
# Terminal states that require a worker-directed stop (timeout, cosched
# failure, unschedulable, failed). KILLED and PREEMPTED are excluded: each is
# handled by its own explicit branch above.
_TERMINAL_EXPECTED_STATES: frozenset[int] = TERMINAL_TASK_STATES - {
    job_pb2.TASK_STATE_KILLED,
    job_pb2.TASK_STATE_PREEMPTED,
}


def _holds_preemption_victim(row: "ReconcileRow") -> bool:
    """Whether this row is a preemption victim still occupying its worker.

    The scheduler commits a preemptor ASSIGNED onto the worker its victim frees
    before the victim's process is actually gone; the victim's attempt is
    PREEMPTED but not yet finalized, so it stays worker-bound (reconcile rows are
    the live, ``finished_at_ms IS NULL`` attempts). Covers both the budget-exhausted
    victim (task PREEMPTED) and the retry/coscheduled-requeue victim (task rolled
    back to PENDING) — both leave the attempt in PREEMPTED. It is the one signal
    that a device is held by a process the preemptor must not race.
    """
    return row.attempt_state == job_pb2.TASK_STATE_PREEMPTED


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

    # A preemptor is committed ASSIGNED onto the worker its victim frees before
    # the victim's process is actually gone. While a preemption victim still
    # occupies this worker, withhold the ASSIGNED run-intents so the worker does
    # not start the preemptor against the live victim (libtpu fixed-port crash
    # loop). The victim's terminal observation stamps ``finished_at_ms``, which
    # drops it from the next reconcile snapshot and lets the run-intent through —
    # a hold of, in the healthy case, a single reconcile cycle.
    hold_assigned_dispatch = any(_holds_preemption_victim(row) for row in rows)

    for row in rows:
        wire_task_id = row.task_id.to_wire()
        if row.task_state in _ASSIGNED_STATES:
            if hold_assigned_dispatch:
                continue
            spec = job_specs.get(row.job_id)
            if spec is None:
                # Job disappeared mid-tick; the scheduler reissues on a
                # subsequent tick.
                continue
            req = job_pb2.RunTaskRequest()
            req.CopyFrom(spec)
            req.task_id = wire_task_id
            req.attempt_id = row.attempt_id
            req.attempt_uid = row.attempt_uid
            desired.append(
                worker_pb2.Worker.DesiredAttempt(
                    attempt_uid=row.attempt_uid,
                    run=worker_pb2.Worker.AttemptSpec(request=req),
                )
            )
        elif row.task_state in EXECUTING_TASK_STATES:
            desired.append(
                worker_pb2.Worker.DesiredAttempt(
                    attempt_uid=row.attempt_uid,
                    run=worker_pb2.Worker.AttemptSpec(),
                )
            )
        elif row.task_state == job_pb2.TASK_STATE_KILLED:
            desired.append(
                worker_pb2.Worker.DesiredAttempt(
                    attempt_uid=row.attempt_uid,
                    stop=worker_pb2.Worker.STOP_REASON_CANCELLED,
                )
            )
        elif row.task_state == job_pb2.TASK_STATE_PREEMPTED:
            desired.append(
                worker_pb2.Worker.DesiredAttempt(
                    attempt_uid=row.attempt_uid,
                    stop=worker_pb2.Worker.STOP_REASON_PREEMPTED,
                )
            )
        elif row.task_state in _TERMINAL_EXPECTED_STATES:
            if row.attempt_state in TERMINAL_TASK_STATES:
                # Controller-induced terminal whose attempt is itself terminal:
                # execution timeout (-> FAILED) or coscheduled-sibling cascade
                # (-> COSCHED_FAILED) moved the task terminal while the worker
                # may still be running the process. A 'run' is a no-op for an
                # attempt the worker already holds, so it would never stop the
                # process nor finalize the attempt (the worker keeps reporting
                # RUNNING, which does not stamp finished_at_ms). Send 'stop' so
                # the worker tears the process down; its resulting terminal
                # observation finalizes the attempt and releases capacity.
                stop_reason = (
                    worker_pb2.Worker.STOP_REASON_TASK_TIMEOUT
                    if row.task_state == job_pb2.TASK_STATE_FAILED
                    else worker_pb2.Worker.STOP_REASON_JOB_TERMINATED
                )
                desired.append(
                    worker_pb2.Worker.DesiredAttempt(
                        attempt_uid=row.attempt_uid,
                        stop=stop_reason,
                    )
                )
            else:
                # Stranded attempt: the task is terminal but the attempt itself
                # is not finalized and the worker may have lost the spec. Re-poll
                # so the worker re-reports its real terminal status, or — if the
                # worker has forgotten the attempt — the daemon synthesizes
                # MISSING, which the controller treats as terminal and stamps.
                desired.append(
                    worker_pb2.Worker.DesiredAttempt(
                        attempt_uid=row.attempt_uid,
                        run=worker_pb2.Worker.AttemptSpec(),
                    )
                )
        elif row.task_state == job_pb2.TASK_STATE_PENDING and row.attempt_state in TERMINAL_TASK_STATES:
            # The task rolled back to PENDING for retry — preemption with budget,
            # or a coscheduled-sibling requeue (the sibling's own attempt is left
            # unfinished) — but current_attempt_id still points at the old
            # worker-bound attempt, which is already terminal (PREEMPTED). Its
            # chips stay reserved until that attempt is finalized. Omitting it (as
            # an unrecognised state) is the leak: the worker zombie-kills the
            # attempt and reports it terminal, but ``filter_observations_to_plan``
            # drops the observation because the attempt isn't in the plan, so the
            # controller never stamps finished_at_ms and the slot leaks forever.
            # Emit 'stop' so the attempt stays in the plan; the worker tears the
            # process down and its terminal observation finalizes the attempt and
            # releases capacity (the reserved-until-heartbeats contract).
            desired.append(
                worker_pb2.Worker.DesiredAttempt(
                    attempt_uid=row.attempt_uid,
                    stop=worker_pb2.Worker.STOP_REASON_PREEMPTED,
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
        # proto3 has no presence for scalar ``exit_code``: an unset field and a
        # genuine 0 both arrive as 0. Treat 0 as "no exit code reported" so a
        # default-valued observation doesn't overwrite a previously recorded
        # exit code; a real exit 0 is conveyed by the SUCCEEDED terminal state.
        exit_code: int | None = obs.exit_code if obs.exit_code != 0 else None
        error: str | None = obs.error or None
        container_id: str | None = obs.container_id or None
        if obs.state == job_pb2.TASK_STATE_MISSING:
            # A worker reports MISSING when it can't resolve a desired attempt to
            # a live local one. While the task is still ACTIVE this is worker loss
            # (the worker restarted and failed to re-adopt a still-running
            # container) -> WORKER_FAILED so it consumes the preemption budget
            # rather than going terminal at max_retries_failure=0. Once the task
            # is already TERMINAL in the snapshot, MISSING is the stranded
            # terminal-attempt finalize case -> FAILED stamps the dead attempt.
            snapshot_task = snapshot.tasks.get(task_id)
            task_active = snapshot_task is not None and snapshot_task.state in ACTIVE_TASK_STATES
            missing_state = job_pb2.TASK_STATE_WORKER_FAILED if task_active else job_pb2.TASK_STATE_FAILED
            updates.append(
                TaskUpdate(
                    task_id=task_id,
                    attempt_id=attempt_id,
                    new_state=missing_state,
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
