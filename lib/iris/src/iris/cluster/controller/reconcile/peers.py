# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Cross-aggregate rules for coscheduled task peers."""

from collections.abc import Iterable

from iris.cluster.controller.reconcile.task import mark_task_terminating
from iris.cluster.controller.reconcile.working_state import WorkingState
from iris.cluster.controller.task_state import (
    ACTIVE_TASK_STATES,
    ActiveTaskRow,
)
from iris.cluster.types import JobName
from iris.rpc import job_pb2


def find_coscheduled_siblings(
    state: WorkingState,
    job_id: JobName,
    exclude_task_id: JobName,
    has_coscheduling: bool,
) -> list[ActiveTaskRow]:
    """Find active siblings in a coscheduled job, reading from the prospective overlay."""
    if not has_coscheduling:
        return []
    return state.active_tasks_for_job(
        job_id,
        states=ACTIVE_TASK_STATES,
        exclude=exclude_task_id,
    )


def terminate_coscheduled_siblings(
    state: WorkingState,
    siblings: Iterable[ActiveTaskRow],
    failed_task_id: JobName,
    now_ms: int,
) -> None:
    """Terminate coscheduled siblings.

    Each sibling is moved to ``TASK_STATE_COSCHED_FAILED``, which is
    unconditionally terminal. Capacity stays held by the unfinished attempt
    rows until the worker's next poll diffs the task out of its
    ``expected_tasks`` and the heartbeat path finalizes the attempt.
    """
    error = f"Coscheduled sibling {failed_task_id.to_wire()} failed"

    for sib in siblings:
        mark_task_terminating(
            state,
            sib.task_id.to_wire(),
            sib.current_attempt_id,
            job_pb2.TASK_STATE_COSCHED_FAILED,
            error,
            now_ms,
        )


def requeue_coscheduled_siblings(
    state: WorkingState,
    siblings: Iterable[ActiveTaskRow],
    failed_task_id: JobName,
    now_ms: int,
) -> None:
    """Bounce coscheduled siblings to PENDING so the job re-coschedules atomically.

    Reservation-holder siblings are skipped; they never hold worker resources
    and don't participate in the slice.
    """
    error = f"Coscheduled sibling {failed_task_id.to_wire()} bounced for atomic re-scheduling"

    for sib in siblings:
        if sib.is_reservation_holder:
            continue
        mark_task_terminating(
            state,
            sib.task_id.to_wire(),
            sib.current_attempt_id,
            job_pb2.TASK_STATE_PENDING,
            error,
            now_ms,
            attempt_state=job_pb2.TASK_STATE_PREEMPTED,
        )
