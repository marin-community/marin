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
from typing import Any

from rigging.timing import Timestamp

from iris.cluster.controller.reconcile.effects import AttemptRowDelta, TaskRowDelta
from iris.cluster.controller.reconcile.policy import FAILURE_TASK_STATES
from iris.cluster.controller.reconcile.snapshot import TaskUpdate, TransitionSnapshot
from iris.cluster.controller.reconcile.working_state import WorkingState
from iris.cluster.controller.task_state import (
    ACTIVE_TASK_STATES,
    EXECUTING_TASK_STATES,
    ActiveTaskRow,
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
    """Variant tag for :class:`TerminalDecision`.

    Each kind drives a different per-task terminal transition inside
    :func:`apply_terminal_decisions_batch`:

    - ``PREEMPT``: mark the task PREEMPTED (or retry to PENDING if budget remains).
    - ``TIMEOUT``: mark the task FAILED with no retry; cascade siblings.
    - ``UNSCHEDULABLE``: mark the task UNSCHEDULABLE and recompute job state.
    """

    PREEMPT = "preempt"
    TIMEOUT = "timeout"
    UNSCHEDULABLE = "unschedulable"


class TransitionSource(StrEnum):
    """Caller policy for side effects attached to task-state updates."""

    WORKER_RECONCILE = "worker_reconcile"
    DIRECT_PROVIDER = "direct_provider"


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
    has_coscheduling: bool


# ─── Snapshot lookups ───


def task_is_finished_row(task: Any) -> bool:
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


def _record_task_termination(
    state: WorkingState,
    task_id: str,
    attempt_id: int | None,
    task_state: int,
    error: str | None,
    now_ms: int,
    *,
    stamp_attempt_finished: bool,
    attempt_state: int | None = None,
    failure_count: int | None = None,
    preemption_count: int | None = None,
) -> None:
    """Move a task to ``task_state`` and record its attempt + endpoint deletion.

    ``stamp_attempt_finished`` controls whether the attempt's ``finished_at_ms``
    is stamped: finalizers stamp it (the attempt is truly done); producer-style
    terminations leave it NULL so the worker's next terminal status update lands
    the timestamp.
    """
    now = Timestamp.from_ms(now_ms)
    task_finished_at = None if task_state in ACTIVE_TASK_STATES or task_state == job_pb2.TASK_STATE_PENDING else now
    effective_attempt_state = attempt_state if attempt_state is not None else task_state
    task_name = JobName.from_wire(task_id)

    if attempt_id is not None and attempt_id >= 0:
        state.record_attempt(
            AttemptRowDelta(
                task_id=task_name,
                attempt_id=attempt_id,
                state=effective_attempt_state,
                finished_at=now if stamp_attempt_finished else None,
                error=error,
            )
        )

    state.record_task(
        TaskRowDelta(
            task_id=task_name,
            state=task_state,
            error=error,
            finished_at=task_finished_at,
            failure_count=failure_count,
            preemption_count=preemption_count,
        )
    )
    state.record_endpoint_deletion(task_name)


def finalize_attempt(
    state: WorkingState,
    task_id: str,
    attempt_id: int | None,
    task_state: int,
    error: str | None,
    now_ms: int,
    *,
    attempt_state: int | None = None,
    failure_count: int | None = None,
    preemption_count: int | None = None,
) -> None:
    """Stamp ``finished_at_ms`` on the attempt and move the task to ``task_state``."""
    _record_task_termination(
        state,
        task_id,
        attempt_id,
        task_state,
        error,
        now_ms,
        stamp_attempt_finished=True,
        attempt_state=attempt_state,
        failure_count=failure_count,
        preemption_count=preemption_count,
    )


def mark_task_terminating(
    state: WorkingState,
    task_id: str,
    attempt_id: int | None,
    task_state: int,
    error: str | None,
    now_ms: int,
    *,
    attempt_state: int | None = None,
    failure_count: int | None = None,
    preemption_count: int | None = None,
) -> None:
    """Update the attempt's reporting state without stamping ``finished_at_ms``."""
    _record_task_termination(
        state,
        task_id,
        attempt_id,
        task_state,
        error,
        now_ms,
        stamp_attempt_finished=False,
        attempt_state=attempt_state,
        failure_count=failure_count,
        preemption_count=preemption_count,
    )


# ─── Per-task decision helpers ───


def resolve_task_failure_state(
    prior_state: int,
    preemption_count: int,
    max_preemptions: int,
    terminal_state: int,
) -> tuple[int, int]:
    """Determine new task state after a worker failure or preemption.

    Assigned tasks always retry. Executing tasks retry if preemption budget remains,
    otherwise go to the given terminal state.

    Returns (new_task_state, updated_preemption_count).
    """
    if prior_state == job_pb2.TASK_STATE_ASSIGNED:
        return job_pb2.TASK_STATE_PENDING, preemption_count
    if prior_state in EXECUTING_TASK_STATES:
        preemption_count += 1
        if preemption_count <= max_preemptions:
            return job_pb2.TASK_STATE_PENDING, preemption_count
    return terminal_state, preemption_count


# ─── Per-task terminal entry points ───
#
# These produce the per-task mutations but do NOT run the cross-aggregate
# cascade (peers / job). The orchestrator in ``batches.py`` drives that.


def unschedulable_one(
    state: WorkingState,
    snapshot: TransitionSnapshot,
    task_id: JobName,
    reason: str,
) -> JobName | None:
    """Mark one task UNSCHEDULABLE; return parent job_id for caller-driven recompute."""
    task = snapshot.tasks.get(task_id)
    if task is None:
        return None
    now_ms = snapshot.now.epoch_ms()
    finalize_attempt(
        state,
        task_id.to_wire(),
        None,
        job_pb2.TASK_STATE_UNSCHEDULABLE,
        reason,
        now_ms,
    )
    return task.job_id


@dataclass(frozen=True, slots=True)
class PreemptOutcome:
    """Result of ``preempt_one``, consumed by batches.py to drive cascade."""

    job_id: JobName
    new_state: int
    has_coscheduling: bool


def preempt_one(
    state: WorkingState,
    snapshot: TransitionSnapshot,
    task_id: JobName,
    reason: str,
    *,
    row: ActiveTaskRow | None,
) -> PreemptOutcome | None:
    """Preempt one task on the shared ``state``. Pure per-task mutation only."""
    if row is None:
        return None
    prior_state = row.state
    if prior_state not in ACTIVE_TASK_STATES:
        return None

    now_ms = snapshot.now.epoch_ms()
    new_state, preemption_count = resolve_task_failure_state(
        prior_state,
        row.preemption_count,
        row.max_retries_preemption,
        job_pb2.TASK_STATE_PREEMPTED,
    )
    mark_task_terminating(
        state,
        task_id.to_wire(),
        row.current_attempt_id,
        new_state,
        reason,
        now_ms,
        attempt_state=job_pb2.TASK_STATE_PREEMPTED,
        preemption_count=preemption_count,
    )
    return PreemptOutcome(job_id=row.job_id, new_state=new_state, has_coscheduling=row.has_coscheduling)


# ─── The per-update transition core ───


def apply_one_transition(
    state: WorkingState,
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
        # that push was dropped, the poll loop re-asks via ``expected_tasks``
        # and we land here with the task already finished. Stamp
        # ``finished_at_ms`` on the attempt so the scheduler releases capacity.
        if (
            task_is_finished_row(task)
            and update.new_state in TERMINAL_TASK_STATES
            and update.attempt_id == task.current_attempt_id
        ):
            attempt = attempt_map.get((update.task_id, update.attempt_id))
            if attempt is not None and attempt.worker_id is not None and attempt.finished_at_ms is None:
                state.record_attempt(
                    AttemptRowDelta(
                        task_id=update.task_id,
                        attempt_id=update.attempt_id,
                        finished_at=Timestamp.from_ms(now_ms),
                    )
                )
        return None

    if update.attempt_id != task.current_attempt_id:
        stale_attempt = attempt_map.get((update.task_id, update.attempt_id))
        stale_state = stale_attempt.state if stale_attempt is not None else None
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
    # The attempt is already terminal (e.g. preempted, killed) but the task
    # has been rolled back to PENDING for retry and current_attempt_id still
    # points at the dead attempt. Reviving it would produce an inconsistent
    # row where state contradicts finished_at_ms/error.
    if attempt.state in TERMINAL_TASK_STATES:
        if attempt.finished_at_ms is None and int(update.new_state) in TERMINAL_TASK_STATES:
            state.record_attempt(
                AttemptRowDelta(
                    task_id=update.task_id,
                    attempt_id=update.attempt_id,
                    finished_at=Timestamp.from_ms(now_ms),
                )
            )
        logger.warning(
            "Dropping late update for terminal attempt: task=%s attempt=%d attempt_state=%d reported=%d",
            update.task_id,
            update.attempt_id,
            attempt.state,
            int(update.new_state),
        )
        return None
    attempt_worker_id = attempt.worker_id
    terminal_ms: int | None = None
    started_ms: int | None = None
    task_state = prior_state
    task_error = update.error
    task_exit = update.exit_code
    failure_count = task.failure_count
    preemption_count = task.preemption_count
    charge_worker_build_failures = source is TransitionSource.WORKER_RECONCILE

    if update.new_state == job_pb2.TASK_STATE_RUNNING:
        started_ms = now_ms
        task_state = job_pb2.TASK_STATE_RUNNING
    elif update.new_state == job_pb2.TASK_STATE_BUILDING:
        task_state = job_pb2.TASK_STATE_BUILDING
    elif update.new_state in (
        job_pb2.TASK_STATE_FAILED,
        job_pb2.TASK_STATE_WORKER_FAILED,
        job_pb2.TASK_STATE_KILLED,
        job_pb2.TASK_STATE_UNSCHEDULABLE,
        job_pb2.TASK_STATE_SUCCEEDED,
    ):
        terminal_ms = now_ms
        task_state = int(update.new_state)
        if update.new_state == job_pb2.TASK_STATE_SUCCEEDED and task_exit is None:
            task_exit = 0
        if update.new_state == job_pb2.TASK_STATE_UNSCHEDULABLE and task_error is None:
            task_error = "Scheduling timeout exceeded"
        if update.new_state == job_pb2.TASK_STATE_FAILED:
            failure_count += 1
            # A FAILED originating while the task was still BUILDING almost
            # always means the worker couldn't pull the image or set up the
            # runtime. Mark the worker as build-failed so the scheduler
            # avoids it.
            if (
                charge_worker_build_failures
                and prior_state == job_pb2.TASK_STATE_BUILDING
                and attempt_worker_id is not None
            ):
                state.record_worker_build_failed(WorkerId(str(attempt_worker_id)))
        if update.new_state == job_pb2.TASK_STATE_WORKER_FAILED and prior_state in EXECUTING_TASK_STATES:
            # A worker that truly died will also miss its next ping/heartbeat
            # RPC, which bumps the tracker on the observer side. We don't
            # double-count that signal here.
            preemption_count += 1
        if update.new_state == job_pb2.TASK_STATE_WORKER_FAILED and prior_state == job_pb2.TASK_STATE_ASSIGNED:
            task_state = job_pb2.TASK_STATE_PENDING
            terminal_ms = None
            # ASSIGNED -> WORKER_FAILED means the worker accepted the task but
            # couldn't bring it up. Attribute the failure to the worker so a
            # host that keeps failing launches gets reaped.
            if charge_worker_build_failures and attempt_worker_id is not None:
                state.record_worker_build_failed(WorkerId(str(attempt_worker_id)))
        if update.new_state == job_pb2.TASK_STATE_FAILED and failure_count <= task.max_retries_failure:
            task_state = job_pb2.TASK_STATE_PENDING
            terminal_ms = None
        if (
            update.new_state == job_pb2.TASK_STATE_WORKER_FAILED
            and preemption_count <= task.max_retries_preemption
            and prior_state in EXECUTING_TASK_STATES
        ):
            task_state = job_pb2.TASK_STATE_PENDING
            terminal_ms = None

    # An attempt is terminal whenever the update itself is terminal, even
    # if the TASK rolls back to PENDING for a retry. terminal_ms above
    # tracks the task's finished_at_ms; the attempt needs its own stamp.
    attempt_finished_at = Timestamp.from_ms(now_ms) if int(update.new_state) in TERMINAL_TASK_STATES else None
    started_at = Timestamp.from_ms(started_ms) if started_ms is not None else None
    task_finished_at = Timestamp.from_ms(terminal_ms) if terminal_ms is not None else None

    state.record_attempt(
        AttemptRowDelta(
            task_id=update.task_id,
            attempt_id=update.attempt_id,
            state=int(update.new_state),
            started_at=started_at,
            finished_at=attempt_finished_at,
            exit_code=task_exit,
            error=update.error,
        )
    )
    state.record_task(
        TaskRowDelta(
            task_id=update.task_id,
            state=task_state,
            error=task_error,
            exit_code=task_exit,
            started_at=started_at,
            finished_at=task_finished_at,
            failure_count=failure_count,
            preemption_count=preemption_count,
            container_id=update.container_id,
        )
    )

    if update.new_state in TERMINAL_TASK_STATES:
        state.record_endpoint_deletion(update.task_id)

    jc = state.job_config(task.job_id)
    has_cosched = bool(jc is not None and jc.has_coscheduling and int(update.new_state) in FAILURE_TASK_STATES)

    return TransitionOutcome(
        task_id=update.task_id,
        job_id=task.job_id,
        prior_state=prior_state,
        new_task_state=task_state,
        has_coscheduling=has_cosched,
    )


# ─── Timeout per-task primitive (batched cascade lives in batches.py) ───


def timeout_one(
    state: WorkingState,
    row: ActiveTaskRow,
    reason: str,
    now_ms: int,
) -> None:
    """Mark one task FAILED via timeout. Per-task mutation only."""
    mark_task_terminating(
        state,
        row.task_id.to_wire(),
        row.current_attempt_id,
        job_pb2.TASK_STATE_FAILED,
        reason,
        now_ms,
        failure_count=row.failure_count + 1,
    )
