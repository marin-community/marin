# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pure-compute reconcile layer for the Iris controller.

The central function ``reconcile_worker`` takes a ``WorkerReconcileInputs``
snapshot and returns a ``WorkerReconcilePlan`` describing:

  - the wire payload to send to the worker (``desired`` attempts)
  - any DB writes to apply if the RPC succeeds (``db_writes``)
  - audit events to record (``events``)

The wire-payload types (``DesiredAttempt``, ``AttemptSpec``, etc.) mirror
the proto shape from the Reconcile RPC but are plain dataclasses.

The legacy wire translators (``legacy_translator_request`` /
``legacy_translator_response``) translate between the pure ``Reconcile``
plan and the legacy ``StartTasks`` / ``PollTasks`` / ``StopTasks`` wires.
They exist as long as both wires coexist; once only the ``Reconcile`` RPC
is in use they can be removed.

See spec.md §4.2-4.6 and sub/transitions-split.md for design context.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from rigging.timing import Timestamp

from iris.cluster.controller.db import TASK_STATE_KILLED, TASK_STATE_PREEMPTED
from iris.cluster.controller.task_state import ACTIVE_TASK_STATES
from iris.cluster.controller.transitions import EXECUTING_TASK_STATES, RunningTaskEntry
from iris.cluster.types import AttemptUid, JobName, WorkerId
from iris.rpc import job_pb2, worker_pb2


@dataclass(frozen=True, slots=True)
class WorkerRow:
    """Durable worker columns: identity and capability."""

    worker_id: WorkerId
    address: str
    total_cpu_millicores: int
    total_memory_bytes: int
    total_gpu_count: int
    total_tpu_count: int
    device_type: str
    device_variant: str
    attributes: dict = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ReconcileRow:
    """One (task, attempt, worker) tuple driving per-worker reconcile.

    Returned by ``TaskAttemptStore.reconcile_rows_for_workers``; rows whose
    task is in ASSIGNED produce start payloads, rows in BUILDING/RUNNING
    populate the worker's expected-task set.
    """

    worker_id: WorkerId
    task_id: JobName
    attempt_id: int
    task_state: int
    attempt_state: int
    job_id: JobName


# ---------------------------------------------------------------------------
# StopReason — mirrors the proto enum, but as a StrEnum for type safety
# ---------------------------------------------------------------------------


class StopReason(StrEnum):
    """Reason the controller is asking the worker to stop an attempt.

    Mirrors ``Worker.StopReason`` proto enum. Values are the canonical
    lower-case names used in log messages and audit events.
    """

    UNSPECIFIED = "unspecified"
    CANCELLED = "cancelled"
    PREEMPTED = "preempted"
    SUPERSEDED = "superseded"
    JOB_TERMINATED = "job_terminated"
    TASK_TIMEOUT = "task_timeout"
    WORKER_DRAIN = "worker_drain"


# ---------------------------------------------------------------------------
# Wire-payload plain-dataclass shapes (NOT proto types)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AttemptSpec:
    """Spec for an attempt the worker should start.

    ``request`` is populated only when the DB attempt state is ``ASSIGNED``
    (the single dispatch tick that hands the worker the spec). On every
    subsequent tick the field is ``None``; the worker is expected to have
    the spec cached from the assignment tick.

    See spec.md §4.3 for the dispatch invariant.
    """

    # Any here: the reconcile_worker pure function is agnostic to the proto shape;
    # legacy_translator_request stamps the concrete RunTaskRequest at the wire boundary.
    request: Any | None = None


@dataclass(frozen=True)
class DesiredAttempt:
    """One entry in the controller's desired-set for a worker.

    Exactly one of ``intent_run`` or ``intent_stop`` is set.

    Compat fields ``task_id`` and ``attempt_id`` are carried alongside
    ``attempt_uid`` for routing while workers still index by the composite
    key; the legacy translator uses them to build StartTasks / PollTasks
    calls without re-looking up the row.
    """

    attempt_uid: AttemptUid
    # Exactly one of:
    intent_run: AttemptSpec | None = None
    intent_stop: StopReason | None = None
    # Routing keys carried on the wire for legacy lookups.
    task_id: str = ""
    attempt_id: int = 0


@dataclass(frozen=True)
class AttemptObservation:
    """One observation reported by the worker.

    Mirrors ``Worker.AttemptObservation`` proto. Carries composite-key
    compat fields alongside ``attempt_uid``.
    """

    attempt_uid: AttemptUid
    # state is the proto int (job_pb2.TaskState)
    state: int = 0
    exit_code: int | None = None
    error: str | None = None
    container_id: str | None = None
    # Routing keys carried on the wire for legacy lookups.
    task_id: str = ""
    attempt_id_compat: int = 0


@dataclass(frozen=True)
class ReconcileRequest:
    """Plain-dataclass mirror of ``Worker.ReconcileRequest`` proto."""

    worker_id: str
    desired: list[DesiredAttempt] = field(default_factory=list)


@dataclass(frozen=True)
class ReconcileResponse:
    """Plain-dataclass mirror of ``Worker.ReconcileResponse`` proto."""

    worker_id: str
    observed: list[AttemptObservation] = field(default_factory=list)


# ---------------------------------------------------------------------------
# WorkerReconcileDispatch — wire payload for the legacy StartTasks/PollTasks path
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WorkerReconcileDispatch:
    """Wire-level per-worker reconcile work for one polling tick.

    ``start_tasks`` is the list of fresh ASSIGNED rows to dispatch (empty if
    none). ``expected_tasks`` is the full set the worker should currently
    have running — anything outside this set the worker auto-kills locally.
    ``stop_tasks`` is the list of task_id wire strings for which the
    controller is requesting a stop (CANCELLED or PREEMPTED rows).
    """

    worker_id: WorkerId
    address: str | None
    start_tasks: list[job_pb2.RunTaskRequest]
    expected_tasks: list[RunningTaskEntry]
    stop_tasks: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# TransitionDelta — DB write instructions produced by the pure layer
# ---------------------------------------------------------------------------


@runtime_checkable
class TransitionDelta(Protocol):
    """A single DB-mutating effect of a reconcile decision.

    Implementations are frozen dataclasses; the apply layer in
    ``transitions.py`` dispatches on the concrete type.
    """


@dataclass(frozen=True)
class AttemptObserved:
    """Recorded when a worker reports observing this attempt.

    The apply layer writes the observed state and fires cascades if the
    new state is terminal.
    """

    attempt_uid: AttemptUid
    # state as proto int (job_pb2.TaskState)
    state: int
    container_id: str | None = None
    finished_at: Timestamp | None = None
    exit_code: int | None = None
    error: str | None = None


@dataclass(frozen=True)
class AttemptMissingOnWorker:
    """Worker reported MISSING — spec cache lost mid-attempt.

    Apply layer: transition attempt to FAILED("worker_lost_spec"); fire
    cascades. Scheduler reissues under a new uid on a subsequent tick.
    """

    attempt_uid: AttemptUid


# ---------------------------------------------------------------------------
# WorkerReconcileInputs / WorkerReconcilePlan
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WorkerReconcileInputs:
    """All state needed to decide one worker's next desired set.

    This is the sole input to ``reconcile_worker``; no DB access, no I/O.

    ``job_specs`` maps ``JobName`` → the ``RunTaskRequest``-shaped object
    (or ``None`` if the spec is unavailable — reservation holder or job
    disappeared mid-tick). The type is ``Any`` here so this module stays
    proto-free; the call site provides the concrete proto.
    """

    worker: WorkerRow
    rows: list[ReconcileRow]
    job_specs: dict[JobName, Any]
    now: Timestamp


@dataclass(frozen=True)
class WorkerReconcilePlan:
    """The reconcile decision for one worker: wire payload + DB writes.

    ``request`` is the ``ReconcileRequest`` to send to the worker.
    ``db_writes`` are applied to the DB only if the RPC succeeds.
    ``events`` are audit-log entries regardless of RPC outcome.
    """

    request: ReconcileRequest
    db_writes: list[TransitionDelta] = field(default_factory=list)
    events: list[Any] = field(default_factory=list)


# ---------------------------------------------------------------------------
# reconcile_worker
# ---------------------------------------------------------------------------

# ASSIGNED is the one active state not in EXECUTING_TASK_STATES.
_ASSIGNED_STATES: frozenset[int] = ACTIVE_TASK_STATES - EXECUTING_TASK_STATES


def _stop_reason_from_state(task_state: int) -> StopReason:
    """Map a DB task state requiring a stop-intent to the appropriate StopReason.

    Only TASK_STATE_PREEMPTED and TASK_STATE_KILLED are expected callers.
    KILLED corresponds to a user-cancelled task (spec §5.3 "CANCELLED" row).
    """
    if task_state == TASK_STATE_PREEMPTED:
        return StopReason.PREEMPTED
    if task_state == TASK_STATE_KILLED:
        return StopReason.CANCELLED
    raise ValueError(f"no StopReason for task_state={task_state!r}")


def reconcile_worker(inputs: WorkerReconcileInputs) -> WorkerReconcilePlan:
    """Pure function: compute the reconcile plan for one worker.

    No DB, no RPC, no time.time() — inputs.now is the clock.

    Spec dispatch invariant: ``AttemptSpec.request`` is set exactly when
    the DB attempt state is ``ASSIGNED``. Every other dispatched state
    sends an empty ``AttemptSpec``; the worker is expected to have the
    spec cached from the assignment tick.

    Rows in terminal states are omitted from ``desired`` entirely; the worker
    auto-stops any zombie attempt not present in the desired set.
    """
    desired: list[DesiredAttempt] = []

    for row in inputs.rows:
        if row.task_state in _ASSIGNED_STATES:
            spec = inputs.job_specs.get(row.job_id)
            if spec is None:
                # Reservation holder or job disappeared mid-tick; skip.
                # The scheduler reissues on a subsequent tick.
                continue
            desired.append(
                DesiredAttempt(
                    attempt_uid=AttemptUid(""),
                    intent_run=AttemptSpec(request=spec),
                    task_id=row.task_id.to_wire(),
                    attempt_id=row.attempt_id,
                )
            )
        elif row.task_state in EXECUTING_TASK_STATES:
            desired.append(
                DesiredAttempt(
                    attempt_uid=AttemptUid(""),
                    intent_run=AttemptSpec(),
                    task_id=row.task_id.to_wire(),
                    attempt_id=row.attempt_id,
                )
            )
        elif row.task_state in (TASK_STATE_KILLED, TASK_STATE_PREEMPTED):
            desired.append(
                DesiredAttempt(
                    attempt_uid=AttemptUid(""),
                    intent_stop=_stop_reason_from_state(row.task_state),
                    task_id=row.task_id.to_wire(),
                    attempt_id=row.attempt_id,
                )
            )
        # Terminal states and any unrecognised states: omit from desired.

    return WorkerReconcilePlan(
        request=ReconcileRequest(
            worker_id=inputs.worker.worker_id,
            desired=desired,
        ),
    )


# ---------------------------------------------------------------------------
# Legacy wire translators (StartTasks / PollTasks / StopTasks)
# ---------------------------------------------------------------------------


def legacy_translator_request(plan: WorkerReconcilePlan, address: str | None) -> WorkerReconcileDispatch:
    """Translate a pure ``WorkerReconcilePlan`` into a legacy ``WorkerReconcileDispatch``.

    This is the first place plain-dataclass payloads cross into proto. It
    builds the three legacy wire lists from the plan's desired set:

    - ``start_tasks``: one ``RunTaskRequest`` for each ASSIGNED-with-spec
      entry (``intent_run`` with a non-None ``request``). These also land in
      ``expected_tasks`` so the worker polls them on the same tick.
    - ``expected_tasks``: every non-terminal run-intent entry (ASSIGNED and
      BUILDING/RUNNING alike). The worker auto-kills anything it has running
      that is not in this set.
    - ``stop_tasks``: task_id wire strings for each stop-intent entry
      (CANCELLED / PREEMPTED rows).

    The legacy ``PollTasks`` wire has no ``MISSING`` state: when the worker
    returns no status for an entry in ``expected_tasks``, the controller
    receives no observation for it on this tick. The worker learns about the
    attempt on the next tick when it fetches the spec via
    ``GetTaskAttemptInfo`` and enqueues it itself, reporting BUILDING via the
    existing pull path. The MISSING→FAILED spec-loss path is only reachable
    via the ``Reconcile`` RPC.
    """
    # TODO(Reconcile-RPC-default): collapse once StartTasks/PollTasks retire (kata 5hzc).
    worker_id = WorkerId(plan.request.worker_id)
    start_tasks: list[job_pb2.RunTaskRequest] = []
    expected_tasks: list[RunningTaskEntry] = []
    stop_tasks: list[str] = []

    for desired in plan.request.desired:
        if desired.intent_run is not None:
            entry = RunningTaskEntry(
                task_id=JobName.from_wire(desired.task_id),
                attempt_id=desired.attempt_id,
            )
            expected_tasks.append(entry)
            if desired.intent_run.request is not None:
                req = job_pb2.RunTaskRequest()
                req.CopyFrom(desired.intent_run.request)
                req.task_id = desired.task_id
                req.attempt_id = desired.attempt_id
                start_tasks.append(req)
        elif desired.intent_stop is not None:
            stop_tasks.append(desired.task_id)

    return WorkerReconcileDispatch(
        worker_id=worker_id,
        address=address,
        start_tasks=start_tasks,
        expected_tasks=expected_tasks,
        stop_tasks=stop_tasks,
    )


_STOP_REASON_PROTO: dict[StopReason, worker_pb2.Worker.StopReason] = {
    StopReason.CANCELLED: worker_pb2.Worker.STOP_REASON_CANCELLED,
    StopReason.PREEMPTED: worker_pb2.Worker.STOP_REASON_PREEMPTED,
    StopReason.SUPERSEDED: worker_pb2.Worker.STOP_REASON_SUPERSEDED,
    StopReason.JOB_TERMINATED: worker_pb2.Worker.STOP_REASON_JOB_TERMINATED,
    StopReason.TASK_TIMEOUT: worker_pb2.Worker.STOP_REASON_TASK_TIMEOUT,
    StopReason.WORKER_DRAIN: worker_pb2.Worker.STOP_REASON_WORKER_DRAIN,
}


def reconcile_request_from_plan(plan: WorkerReconcilePlan) -> worker_pb2.Worker.ReconcileRequest:
    """Translate a plain-dataclass ``WorkerReconcilePlan`` into the Reconcile RPC proto.

    Pure struct copy — no DB access, no I/O.

    Maps each ``DesiredAttempt`` to the proto ``Worker.DesiredAttempt``:

    - ``intent_run`` with a spec → ``run`` field with the embedded
      ``RunTaskRequest`` (present only for ASSIGNED rows with a spec).
    - ``intent_run`` without a spec → ``run`` field with an empty
      ``AttemptSpec`` (BUILDING/RUNNING rows; worker uses its cached spec).
    - ``intent_stop`` → ``stop`` field with the mapped ``StopReason`` proto.

    The composite-key compat fields ``task_id`` and ``attempt_id`` on
    ``DesiredAttempt`` are forwarded verbatim so the worker-side handler can
    resolve the spec from ``GetTaskAttemptInfo`` using the same routing keys.
    """
    desired_protos: list[worker_pb2.Worker.DesiredAttempt] = []
    for da in plan.request.desired:
        kwargs: dict = {
            "attempt_uid": da.attempt_uid,
            "task_id": da.task_id,
            "attempt_id": da.attempt_id,
        }
        if da.intent_run is not None:
            spec = worker_pb2.Worker.AttemptSpec()
            if da.intent_run.request is not None:
                req = job_pb2.RunTaskRequest()
                req.CopyFrom(da.intent_run.request)
                req.task_id = da.task_id
                req.attempt_id = da.attempt_id
                spec = worker_pb2.Worker.AttemptSpec(request=req)
            kwargs["run"] = spec
        elif da.intent_stop is not None:
            kwargs["stop"] = _STOP_REASON_PROTO.get(da.intent_stop, worker_pb2.Worker.STOP_REASON_UNSPECIFIED)
        desired_protos.append(worker_pb2.Worker.DesiredAttempt(**kwargs))
    return worker_pb2.Worker.ReconcileRequest(
        worker_id=plan.request.worker_id,
        desired=desired_protos,
    )


def observations_from_reconcile_response(
    response: worker_pb2.Worker.ReconcileResponse,
) -> list[AttemptObservation]:
    """Translate a Reconcile RPC response into an ``AttemptObservation`` list.

    Each ``Worker.AttemptObservation`` proto entry maps to one
    ``AttemptObservation`` plain-dataclass. ``TASK_STATE_MISSING`` passes
    through as-is; ``apply_reconcile_observations`` converts it to
    ``AttemptMissingOnWorker`` → ``FAILED("worker_lost_spec")``.

    ``finished_at`` from the proto is not forwarded — the apply layer stamps
    the transaction timestamp itself for terminal transitions (consistent with
    the legacy path).
    """
    observations: list[AttemptObservation] = []
    for obs in response.observed:
        exit_code: int | None = obs.exit_code if obs.exit_code != 0 else None
        error: str | None = obs.error if obs.error else None
        container_id: str | None = obs.container_id if obs.container_id else None
        observations.append(
            AttemptObservation(
                attempt_uid=AttemptUid(obs.attempt_uid),
                state=int(obs.state),
                exit_code=exit_code,
                error=error,
                container_id=container_id,
                task_id=obs.task_id,
                attempt_id_compat=obs.attempt_id,
            )
        )
    return observations


def legacy_translator_response(
    plan: WorkerReconcilePlan,
    result: Any,
) -> list[AttemptObservation]:
    """Translate a legacy ``WorkerReconcileResult`` into ``AttemptObservation`` list.

    ``result`` is a ``WorkerReconcileResult`` (from ``worker_provider``). The
    type is ``Any`` here to avoid a circular import; the caller provides the
    concrete instance.

    Conversion rules:

    - ``poll_error is not None`` → return empty list; the apply layer
      (A.4) handles the error via ``apply_reconcile_failure``.
    - ``start_error is not None`` → that path is handled by A.4's
      synthetic-WORKER_FAILED logic; no observations fabricated here.
    - ``poll_updates`` (``list[TaskUpdate]``) → one ``AttemptObservation``
      per update with state / exit_code / error / container_id populated.
      Entries absent from the poll response produce no observation — the
      worker learns about them on the next tick via the existing pull path
      (see ``legacy_translator_request`` for the MISSING-state explanation).

    ``finished_at`` is not populated because the legacy ``TaskUpdate`` type
    does not carry it; the apply layer fills in the timestamp itself.
    """
    if result.poll_error is not None:
        return []

    poll_updates = result.poll_updates
    if not poll_updates:
        return []

    observations: list[AttemptObservation] = []
    for update in poll_updates:
        observations.append(
            AttemptObservation(
                attempt_uid=AttemptUid(""),
                state=update.new_state,
                exit_code=update.exit_code,
                error=update.error,
                container_id=update.container_id,
                task_id=update.task_id.to_wire(),
                attempt_id_compat=update.attempt_id,
            )
        )
    return observations
