# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Cross-aggregate rules for coscheduled task peers."""

from collections.abc import Sequence
from typing import Protocol

from rigging.timing import Timestamp

from iris.cluster.controller.reconcile.effects import TaskActionEvent, TaskActionReason
from iris.cluster.controller.reconcile.overlay import Overlay
from iris.cluster.controller.reconcile.task import merge_task_termination
from iris.cluster.controller.task_state import (
    ACTIVE_TASK_STATES,
    ActiveTaskRow,
)
from iris.cluster.stats.tables import TaskEventSeverity
from iris.cluster.types import JobName
from iris.rpc import job_pb2


class CoscheduledSibling(Protocol):
    """Task fields required by a coscheduled peer cascade."""

    @property
    def task_id(self) -> JobName: ...

    @property
    def current_attempt_id(self) -> int: ...


def find_coscheduled_siblings(
    state: Overlay,
    job_id: JobName,
    exclude_task_id: JobName,
) -> list[ActiveTaskRow]:
    """Find active siblings in a coscheduled job, reading from the prospective overlay.

    Callers are responsible for gating on coscheduling: this returns every active
    sibling regardless, so a non-coscheduled caller must avoid invoking it.
    """
    return state.active_tasks_for_job(
        job_id,
        states=ACTIVE_TASK_STATES,
        exclude=exclude_task_id,
    )


def find_coscheduled_requeue_siblings(
    state: Overlay,
    job_id: JobName,
    exclude_task_id: JobName,
) -> Sequence[CoscheduledSibling]:
    """Find siblings that must return to PENDING for an atomic gang restart.

    A succeeded sibling may have exited because another gang member vanished.
    Its task result is therefore provisional until the whole coscheduled job
    completes. Include those siblings so a retriable peer failure can revive
    their tasks without rewriting the succeeded attempt.
    """
    active_siblings = state.active_tasks_for_job(
        job_id,
        states=ACTIVE_TASK_STATES,
        exclude=exclude_task_id,
    )
    succeeded_siblings = state.task_details_for_job(
        job_id,
        states={job_pb2.TASK_STATE_SUCCEEDED},
        exclude=exclude_task_id,
    )
    return (*active_siblings, *succeeded_siblings)


def terminate_coscheduled_siblings(
    state: Overlay,
    siblings: Sequence[ActiveTaskRow],
    failed_task_id: JobName,
    now_ms: int,
) -> None:
    """Terminate coscheduled siblings.

    Each sibling is moved to ``TASK_STATE_COSCHED_FAILED``, which is
    unconditionally terminal. The attempt is left unfinished
    (``finished_at_ms`` NULL) so the sibling's chips stay accounted for until
    its process is actually stopped: because the attempt is terminal but still
    worker-bound, the reconcile planner sends the worker a ``stop`` for it
    (``reconcile/worker.py``), and the worker's resulting terminal observation
    finalizes the attempt and releases capacity.
    """
    error = f"Coscheduled sibling {failed_task_id.to_wire()} failed"

    for sib in siblings:
        merge_task_termination(
            state,
            sib.task_id.to_wire(),
            sib.current_attempt_id,
            job_pb2.TASK_STATE_COSCHED_FAILED,
            error,
            now_ms,
            stamp_attempt_finished=False,
        )
        state.emit_task_event(
            TaskActionEvent(
                task_id=sib.task_id,
                attempt_id=sib.current_attempt_id,
                ts=Timestamp.from_ms(now_ms),
                reason=TaskActionReason.COSCHEDULED_SIBLING_TERMINATED,
                message=error,
                severity=TaskEventSeverity.WARNING,
            )
        )


def requeue_coscheduled_siblings(
    state: Overlay,
    siblings: Sequence[CoscheduledSibling],
    failed_task_id: JobName,
    now_ms: int,
) -> None:
    """Bounce coscheduled siblings to PENDING so the job re-coschedules atomically.

    An active sibling's attempt is stamped ``COSCHED_FAILED``, not ``PREEMPTED``:
    an atomic gang restart is not the sibling's own preemption, so its budget
    stays unchanged. A succeeded attempt is already terminal and remains
    ``SUCCEEDED`` in the attempt history while its task returns to ``PENDING``.
    """
    error = f"Coscheduled sibling {failed_task_id.to_wire()} bounced for atomic re-scheduling"

    for sib in siblings:
        merge_task_termination(
            state,
            sib.task_id.to_wire(),
            sib.current_attempt_id,
            job_pb2.TASK_STATE_PENDING,
            error,
            now_ms,
            stamp_attempt_finished=False,
            attempt_state=job_pb2.TASK_STATE_COSCHED_FAILED,
        )
        state.emit_task_event(
            TaskActionEvent(
                task_id=sib.task_id,
                attempt_id=sib.current_attempt_id,
                ts=Timestamp.from_ms(now_ms),
                reason=TaskActionReason.COSCHEDULED_SIBLING_REQUEUED,
                message=error,
            )
        )
