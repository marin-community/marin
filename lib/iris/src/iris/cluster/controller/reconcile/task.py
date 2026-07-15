# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pure rules for tasks and attempts: per-transition primitives.

The transition primitives are otherwise pure, but emit free-form diagnostic
logs inline for dropped stale/late attempt updates. Those logs are
observability, not state, so they do not flow through ``ControllerEffects``.
"""

import logging
from dataclasses import dataclass
from enum import StrEnum

from rigging.timing import Timestamp

from iris.cluster.controller.reconcile.effects import AttemptRowDelta, TaskRowDelta
from iris.cluster.controller.reconcile.overlay import Overlay
from iris.cluster.controller.reconcile.policy import PEER_CASCADE_TRIGGER_STATES
from iris.cluster.controller.reconcile.snapshot import TaskUpdate, TransitionSnapshot
from iris.cluster.controller.task_state import (
    ACTIVE_TASK_STATES,
    EXECUTING_TASK_STATES,
    ActiveTaskRow,
    TaskDetailRow,
    task_is_finished,
)
from iris.cluster.types import (
    TERMINAL_TASK_STATES,
    JobName,
    WorkerId,
)
from iris.rpc import job_pb2

logger = logging.getLogger(__name__)


# ─── Inputs ───
#
# ``TaskUpdate`` lives in ``snapshot.py`` (a leaf module) so reconcile-plan and
# direct-provider callers can build it without an aggregate cross-import.
# Import it from there.


class TerminalKind(StrEnum):
    """Which terminal transition a :class:`TerminalDecision` requests.

    - ``PREEMPT``: the task should be preempted (retried if budget remains).
    - ``TIMEOUT``: the task should fail without retry.
    - ``UNSCHEDULABLE``: the task can never be placed.
    """

    PREEMPT = "preempt"
    TIMEOUT = "timeout"
    UNSCHEDULABLE = "unschedulable"


class TransitionSource(StrEnum):
    """Caller policy for side effects attached to task-state updates."""

    WORKER_RECONCILE = "worker_reconcile"
    DISPATCH = "dispatch"


@dataclass(frozen=True, slots=True)
class TerminalDecision:
    """One task → terminal assertion to be applied as part of a batch."""

    kind: TerminalKind
    task_id: JobName
    reason: str


# ─── Per-update transition outcome ───


@dataclass(frozen=True, slots=True)
class TransitionOutcome:
    """Result of one ``apply_one_transition`` call, consumed by batches.py.

    Carries the information the orchestrator needs to drive cross-aggregate
    cascades (peer cascade, job recompute) without re-reading state.
    """

    task_id: JobName
    job_id: JobName
    prior_state: int
    new_task_state: int
    cascade_to_peers: bool


# ─── Snapshot lookups ───


def task_is_finished_row(task: TaskDetailRow) -> bool:
    return task_is_finished(
        task.state,
        task.failure_count,
        task.max_retries_failure,
        task.preemption_count,
        task.max_retries_preemption,
    )


def active_row_from_snapshot(snapshot: TransitionSnapshot, task_id: JobName) -> ActiveTaskRow | None:
    """Resolve the snapshot's active-task row for ``task_id``."""
    task = snapshot.tasks.get(task_id)
    if task is None:
        return None
    rows = snapshot.active_tasks_by_job.get(task.job_id, ())
    for row in rows:
        if row.task_id == task_id:
            return row
    return None


# ─── Per-attempt transitions ───


def _backfill_attempt_finished_at(state: Overlay, task_id: JobName, attempt_id: int, now_ms: int) -> None:
    """Stamp ``finished_at_ms`` on an attempt whose overlay value is still NULL.

    Producer-style terminations leave the attempt's ``finished_at_ms`` NULL,
    expecting the worker's next terminal status push to land the timestamp. When
    that push is dropped we backfill it here so the scheduler releases capacity.
    No-op if the overlay already has a ``finished_at`` for the attempt.
    """
    if state.attempt_finished_at(task_id, attempt_id) is not None:
        return
    state.merge_attempt(
        AttemptRowDelta(
            task_id=task_id,
            attempt_id=attempt_id,
            finished_at=Timestamp.from_ms(now_ms),
        )
    )


def merge_task_termination(
    state: Overlay,
    task_id: str,
    attempt_id: int | None,
    task_state: int,
    error: str | None,
    now_ms: int,
    *,
    stamp_attempt_finished: bool,
    attempt_state: int | None = None,
) -> None:
    """Move a task to ``task_state`` and record its attempt.

    ``stamp_attempt_finished`` controls whether the attempt's ``finished_at_ms``
    is stamped: finalizing callers stamp it (the attempt is truly done);
    producer-style terminations leave it NULL so the worker's next terminal
    status update lands the timestamp.

    No retry counter is carried: both the failure and preemption counts derive
    from the attempt rows this records (see
    ``iris.cluster.controller.attempt_counts``).

    An already-terminal attempt is left untouched: killing a PENDING task (a
    cancel or a job-failure cascade) must not overwrite the historical outcome of
    its last, already-finished attempt (e.g. rewrite a FAILED attempt to KILLED),
    which would both lose history and, since counts derive from attempt state,
    miscount that terminal as a preemption instead of the failure it was.
    """
    now = Timestamp.from_ms(now_ms)
    task_finished_at = None if task_state in ACTIVE_TASK_STATES or task_state == job_pb2.TASK_STATE_PENDING else now
    effective_attempt_state = attempt_state if attempt_state is not None else task_state
    task_name = JobName.from_wire(task_id)

    if attempt_id is not None and attempt_id >= 0:
        existing_attempt_state = state.attempt_state(task_name, attempt_id)
        if existing_attempt_state is None or existing_attempt_state not in TERMINAL_TASK_STATES:
            state.merge_attempt(
                AttemptRowDelta(
                    task_id=task_name,
                    attempt_id=attempt_id,
                    state=effective_attempt_state,
                    finished_at=now if stamp_attempt_finished else None,
                    error=error,
                )
            )

    state.merge_task(
        TaskRowDelta(
            task_id=task_name,
            state=task_state,
            error=error,
            finished_at=task_finished_at,
        )
    )


# ─── Per-task decision helpers ───


def resolve_task_failure_state(
    prior_state: int,
    preemption_count: int,
    max_preemptions: int,
    terminal_state: int,
) -> int:
    """Determine the new task state after a worker failure or preemption.

    Assigned tasks always retry. Executing tasks retry while the preemption
    budget remains (this attempt would be the ``preemption_count + 1``-th),
    otherwise go to the given terminal state.
    """
    if prior_state == job_pb2.TASK_STATE_ASSIGNED:
        return job_pb2.TASK_STATE_PENDING
    if prior_state in EXECUTING_TASK_STATES and preemption_count + 1 <= max_preemptions:
        return job_pb2.TASK_STATE_PENDING
    return terminal_state


# ─── Per-task terminal entry points ───
#
# These produce the per-task mutations but do NOT run the cross-aggregate
# cascade (peers / job). The orchestrator in ``batches.py`` drives that.


def unschedulable_one(
    state: Overlay,
    snapshot: TransitionSnapshot,
    task_id: JobName,
    reason: str,
) -> JobName | None:
    """Mark one task UNSCHEDULABLE; return parent job_id for caller-driven recompute."""
    task = snapshot.tasks.get(task_id)
    if task is None:
        return None
    now_ms = snapshot.now.epoch_ms()
    merge_task_termination(
        state,
        task_id.to_wire(),
        None,
        job_pb2.TASK_STATE_UNSCHEDULABLE,
        reason,
        now_ms,
        stamp_attempt_finished=True,
    )
    return task.job_id


def preempt_one(
    state: Overlay,
    snapshot: TransitionSnapshot,
    task_id: JobName,
    reason: str,
    *,
    row: ActiveTaskRow | None,
) -> TransitionOutcome | None:
    """Preempt one task on the shared ``state``. Pure per-task mutation only."""
    if row is None:
        return None
    prior_state = row.state
    if prior_state not in ACTIVE_TASK_STATES:
        return None

    now_ms = snapshot.now.epoch_ms()
    new_state = resolve_task_failure_state(
        prior_state,
        row.preemption_count,
        row.max_retries_preemption,
        job_pb2.TASK_STATE_PREEMPTED,
    )
    merge_task_termination(
        state,
        task_id.to_wire(),
        row.current_attempt_id,
        new_state,
        reason,
        now_ms,
        stamp_attempt_finished=False,
        attempt_state=job_pb2.TASK_STATE_PREEMPTED,
    )
    return TransitionOutcome(
        task_id=task_id,
        job_id=row.job_id,
        prior_state=prior_state,
        new_task_state=new_state,
        cascade_to_peers=row.has_coscheduling,
    )


# ─── The per-update transition core ───


def apply_one_transition(
    state: Overlay,
    snapshot: TransitionSnapshot,
    update: TaskUpdate,
    now_ms: int,
    *,
    source: TransitionSource = TransitionSource.WORKER_RECONCILE,
) -> TransitionOutcome | None:
    """Apply one ``TaskUpdate`` against ``state``: write attempt + task mutations.

    This is the single per-update transition core. Worker reconcile updates
    charge build failures to worker health so hosts that keep failing builds
    get reaped; direct providers manage their own hosts. ``update.container_id``
    is folded into the task row when present.

    Returns a :class:`TransitionOutcome` describing the change so the
    orchestrator can drive the peer-cascade and job-recompute. Returns
    ``None`` when the update is dropped (no-op, stale attempt, task already
    finished without state delta, etc.).

    NOTE: This function does NOT run peer cascades or job recompute — those
    are orchestrator concerns and live in ``batches.py``.
    """
    task_map = snapshot.tasks
    attempt_map = snapshot.attempts

    task = task_map.get(update.task_id)
    if task is None:
        return None

    # Overlay-aware terminal guard. An earlier item in the same batch (e.g. a
    # peer cascade that moved this task to COSCHED_FAILED, or another update for
    # the same task) may already have finalized it in the overlay even though the
    # snapshot row is stale. Re-applying from the stale row would overwrite the
    # cascade mutation, so drop the update once the overlay shows a terminal state.
    overlay_state = state.task_state(update.task_id)
    if overlay_state is not None and overlay_state in TERMINAL_TASK_STATES and overlay_state != task.state:
        return None

    if task_is_finished_row(task) or update.new_state in (
        job_pb2.TASK_STATE_UNSPECIFIED,
        job_pb2.TASK_STATE_PENDING,
    ):
        # Stranded-attempt finalization: producer transitions move the task
        # to a terminal state but leave the attempt's ``finished_at_ms`` NULL,
        # expecting the worker's next terminal status update to stamp it. If
        # that push was dropped, the reconcile planner re-polls the still
        # worker-bound attempt and we land here with the task already finished.
        # Stamp ``finished_at_ms`` on the attempt so the scheduler releases
        # capacity.
        if (
            task_is_finished_row(task)
            and update.new_state in TERMINAL_TASK_STATES
            and update.attempt_id == task.current_attempt_id
        ):
            attempt = attempt_map.get((update.task_id, update.attempt_id))
            if attempt is not None and attempt.worker_id is not None:
                _backfill_attempt_finished_at(state, update.task_id, update.attempt_id, now_ms)
        return None

    if update.attempt_id != task.current_attempt_id:
        stale_state = state.attempt_state(update.task_id, update.attempt_id)
        if stale_state is not None and stale_state not in TERMINAL_TASK_STATES:
            logger.error(
                "Stale attempt precondition violation: task=%s reported=%d current=%d stale_state=%s",
                update.task_id,
                update.attempt_id,
                task.current_attempt_id,
                stale_state,
            )
        return None

    # Overlay-aware prior state: a same-batch cascade may have already moved this
    # task (e.g. requeued a coscheduled sibling to PENDING). Fall back to the
    # snapshot row when no overlay entry exists.
    prior_state = overlay_state if overlay_state is not None else task.state

    # Fast path: task already in the reported state with no new data to apply.
    has_new_data = update.error is not None or update.exit_code is not None
    if update.new_state == prior_state and not has_new_data:
        return None

    attempt = attempt_map.get((update.task_id, update.attempt_id))
    if attempt is None:
        return None
    # Overlay-aware attempt state: a same-batch cascade (e.g. a coscheduled
    # sibling requeue that drove this attempt PREEMPTED in the overlay) must be
    # visible here, so read through the accessor rather than the stale snapshot.
    overlay_attempt_state = state.attempt_state(update.task_id, update.attempt_id)
    # The attempt is already terminal (e.g. preempted, killed) but the task
    # has been rolled back to PENDING for retry and current_attempt_id still
    # points at the dead attempt. Reviving it would produce an inconsistent
    # row where state contradicts finished_at_ms/error.
    if overlay_attempt_state is not None and overlay_attempt_state in TERMINAL_TASK_STATES:
        if update.new_state in TERMINAL_TASK_STATES:
            _backfill_attempt_finished_at(state, update.task_id, update.attempt_id, now_ms)
        logger.warning(
            "Dropping late update for terminal attempt: task=%s attempt=%d attempt_state=%d reported=%d",
            update.task_id,
            update.attempt_id,
            overlay_attempt_state,
            update.new_state,
        )
        return None
    attempt_worker_id = attempt.worker_id
    terminal_ms: int | None = None
    started_ms: int | None = None
    task_state = prior_state
    task_error = update.error
    task_exit = update.exit_code
    # Committed-derived per-task retry counters read from the snapshot row, used only
    # for this attempt's local retry decision (failure gates on max_retries_failure,
    # preemption on max_retries_preemption). Nothing prospective is carried across the
    # batch: the job-wide failure budget derives from the FAILED attempt this records
    # (see Overlay.job_basis), symmetric with preemption.
    failure_count = task.failure_count
    preemption_count = task.preemption_count
    charge_worker_build_failures = source is TransitionSource.WORKER_RECONCILE

    if update.new_state == job_pb2.TASK_STATE_RUNNING:
        started_ms = now_ms
        task_state = job_pb2.TASK_STATE_RUNNING
    elif update.new_state == job_pb2.TASK_STATE_BUILDING:
        # Stamp started_at_ms on BUILDING so the execution-timeout scan
        # (gated on started_at_ms IS NOT NULL) can finalize wedged builds.
        # COALESCE on the RUNNING write preserves this stamp. Issue #6077.
        started_ms = now_ms
        task_state = job_pb2.TASK_STATE_BUILDING
    elif update.new_state in (
        job_pb2.TASK_STATE_FAILED,
        job_pb2.TASK_STATE_WORKER_FAILED,
        job_pb2.TASK_STATE_KILLED,
        job_pb2.TASK_STATE_UNSCHEDULABLE,
        job_pb2.TASK_STATE_SUCCEEDED,
    ):
        terminal_ms = now_ms
        task_state = update.new_state
        if update.new_state == job_pb2.TASK_STATE_SUCCEEDED and task_exit is None:
            task_exit = 0
        if update.new_state == job_pb2.TASK_STATE_UNSCHEDULABLE and task_error is None:
            task_error = "Scheduling timeout exceeded"

        # Charge the host build-failure reaper when a worker failed to bring the
        # attempt up — a launch (ASSIGNED) or build/setup (BUILDING) failure — so
        # a host that keeps failing launches gets reaped. A WORKER_FAILED here is
        # infra (bad/missing node); a FAILED-from-BUILDING is a failed image pull
        # or runtime setup. A FAILED-from-ASSIGNED is not a host fault (and the
        # worker never reports it: it announces BUILDING before running). Direct
        # providers manage their own hosts, so this is gated on
        # charge_worker_build_failures (worker path only).
        launch_or_build_failure = (
            update.new_state == job_pb2.TASK_STATE_WORKER_FAILED
            and prior_state in (job_pb2.TASK_STATE_ASSIGNED, job_pb2.TASK_STATE_BUILDING)
        ) or (update.new_state == job_pb2.TASK_STATE_FAILED and prior_state == job_pb2.TASK_STATE_BUILDING)
        if charge_worker_build_failures and launch_or_build_failure and attempt_worker_id is not None:
            state.emit_worker_build_failed(WorkerId(str(attempt_worker_id)))

        if update.new_state == job_pb2.TASK_STATE_FAILED:
            # Application failure (non-zero exit / setup error): failure budget.
            failure_count += 1
            if failure_count <= task.max_retries_failure:
                task_state = job_pb2.TASK_STATE_PENDING
                terminal_ms = None
        elif update.new_state in (job_pb2.TASK_STATE_WORKER_FAILED, job_pb2.TASK_STATE_KILLED):
            # Worker loss / infra (WORKER_FAILED) or an out-of-band container stop
            # the worker reports as KILLED — a higher-priority job reclaiming the
            # slice, a node drain, a spot/preemptible reclaim, or a stop directive
            # the controller issued without recording a matching task transition.
            # None of these are application failures, so both share the preemption
            # budget. A genuine user/controller cancel never reaches here: cancel_job
            # marks the task terminal first, so the worker's echoing KILLED lands on
            # an already-finished row above and is dropped; a stale stop of an
            # abandoned attempt arrives under an old attempt_id and is dropped too.
            # A KILLED that survives to this branch is therefore always the *current*
            # live attempt stopped out-of-band — which must retry, not fail the job.
            # ASSIGNED retries without charge (the worker never ran the process);
            # EXECUTING (BUILDING/RUNNING) charges and gates on max_retries_preemption.
            # A truly-dead worker also misses its next ping/heartbeat (bumped
            # observer-side), so we don't double-count here.
            task_state = resolve_task_failure_state(
                prior_state,
                preemption_count,
                task.max_retries_preemption,
                terminal_state=job_pb2.TASK_STATE_WORKER_FAILED,
            )
            if task_state == job_pb2.TASK_STATE_PENDING:
                terminal_ms = None

    # An attempt is terminal whenever the update itself is terminal, even
    # if the TASK rolls back to PENDING for a retry. terminal_ms above
    # tracks the task's finished_at_ms; the attempt needs its own stamp.
    attempt_finished_at = Timestamp.from_ms(now_ms) if update.new_state in TERMINAL_TASK_STATES else None
    started_at = Timestamp.from_ms(started_ms) if started_ms is not None else None
    task_finished_at = Timestamp.from_ms(terminal_ms) if terminal_ms is not None else None

    state.merge_attempt(
        AttemptRowDelta(
            task_id=update.task_id,
            attempt_id=update.attempt_id,
            state=update.new_state,
            started_at=started_at,
            finished_at=attempt_finished_at,
            exit_code=task_exit,
            error=update.error,
        )
    )
    state.merge_task(
        TaskRowDelta(
            task_id=update.task_id,
            state=task_state,
            error=task_error,
            exit_code=task_exit,
            started_at=started_at,
            finished_at=task_finished_at,
            container_id=update.container_id,
        )
    )

    jc = state.job_config(task.job_id)
    has_cosched = bool(jc is not None and jc.has_coscheduling and update.new_state in PEER_CASCADE_TRIGGER_STATES)

    return TransitionOutcome(
        task_id=update.task_id,
        job_id=task.job_id,
        prior_state=prior_state,
        new_task_state=task_state,
        cascade_to_peers=has_cosched,
    )


# ─── Timeout per-task primitive (batched cascade lives in batches.py) ───


def timeout_one(
    state: Overlay,
    row: ActiveTaskRow,
    reason: str,
    now_ms: int,
) -> None:
    """Mark one task FAILED via timeout by recording a FAILED attempt. Per-task mutation only."""
    merge_task_termination(
        state,
        row.task_id.to_wire(),
        row.current_attempt_id,
        job_pb2.TASK_STATE_FAILED,
        reason,
        now_ms,
        stamp_attempt_finished=False,
    )
