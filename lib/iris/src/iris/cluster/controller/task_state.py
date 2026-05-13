# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Controller task and attempt state predicates."""

from typing import Any

from rigging.timing import Deadline, Duration, Timestamp

from iris.cluster.types import TERMINAL_TASK_STATES
from iris.rpc import job_pb2

ACTIVE_TASK_STATES: frozenset[int] = frozenset(
    {
        job_pb2.TASK_STATE_ASSIGNED,
        job_pb2.TASK_STATE_BUILDING,
        job_pb2.TASK_STATE_RUNNING,
    }
)


def task_is_finished(
    state: int,
    failure_count: int,
    max_retries_failure: int,
    preemption_count: int,
    max_retries_preemption: int,
) -> bool:
    """Whether a task has reached a terminal state with no remaining retries."""
    if state == job_pb2.TASK_STATE_SUCCEEDED:
        return True
    if state in (job_pb2.TASK_STATE_KILLED, job_pb2.TASK_STATE_UNSCHEDULABLE, job_pb2.TASK_STATE_COSCHED_FAILED):
        return True
    if state == job_pb2.TASK_STATE_FAILED:
        return failure_count > max_retries_failure
    if state in (job_pb2.TASK_STATE_WORKER_FAILED, job_pb2.TASK_STATE_PREEMPTED):
        return preemption_count > max_retries_preemption
    return False


def task_row_can_be_scheduled(task: Any) -> bool:
    if task.state != job_pb2.TASK_STATE_PENDING:
        return False
    return task.current_attempt_id < 0 or not task_is_finished(
        task.state,
        task.failure_count,
        task.max_retries_failure,
        task.preemption_count,
        task.max_retries_preemption,
    )


def job_scheduling_deadline(scheduling_deadline_epoch_ms: int | None) -> Deadline | None:
    """Compute scheduling deadline from epoch ms."""
    if scheduling_deadline_epoch_ms is None:
        return None
    return Deadline.after(Timestamp.from_ms(scheduling_deadline_epoch_ms), Duration.from_ms(0))


def attempt_is_terminal(state: int) -> bool:
    """Check if an attempt is in a terminal state."""
    return state in TERMINAL_TASK_STATES


def attempt_is_worker_failure(state: int) -> bool:
    """Check if an attempt is a worker failure or preemption."""
    return state in (job_pb2.TASK_STATE_WORKER_FAILED, job_pb2.TASK_STATE_PREEMPTED)
