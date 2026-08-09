# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Controller task and attempt state predicates."""

from dataclasses import dataclass
from typing import NamedTuple, Protocol

from rigging.timing import Deadline, Duration, Timestamp

from iris.cluster.types import JobName, WorkerId
from iris.resources.state import TaskState

ACTIVE_TASK_STATES: frozenset[int] = frozenset(
    {
        TaskState.ASSIGNED,
        TaskState.BUILDING,
        TaskState.RUNNING,
    }
)

# Subset of ACTIVE that excludes ASSIGNED — i.e. tasks already on a worker.
EXECUTING_TASK_STATES: frozenset[int] = frozenset(
    {
        TaskState.BUILDING,
        TaskState.RUNNING,
    }
)

# Subset of ACTIVE that excludes RUNNING — dispatched to a worker (or pod) but
# not yet observed running. These tasks age from their current attempt's
# creation in the ``iris.task_state`` wait-age columns.
DISPATCHED_TASK_STATES: frozenset[int] = frozenset(
    {
        TaskState.ASSIGNED,
        TaskState.BUILDING,
    }
)


class RunningTaskEntry(NamedTuple):
    """Task ID and attempt ID pair captured at snapshot time.

    ``attempt_uid`` is the incarnation key the K8s provider needs to rebuild the
    pod name (which embeds it): a resubmit reuses (task_id, attempt_id) but mints
    a new uid, so poll must target this attempt's pod, not a stale one. Empty off
    the direct-dispatch path.
    """

    task_id: JobName
    attempt_id: int
    attempt_uid: str = ""


@dataclass(frozen=True, slots=True)
class AttemptRecord:
    """One persisted Attempt history record."""

    task_id: JobName
    attempt_id: int
    worker_id: WorkerId | None
    state: int
    created_at_ms: Timestamp
    started_at_ms: Timestamp | None
    finished_at_ms: Timestamp | None
    exit_code: int | None
    error: str | None
    attempt_uid: str
    backend_id: str
    pod_name: str
    pod_uid: str
    node_name: str
    terminal_reason: str


def task_is_finished(
    state: int,
    failure_count: int,
    max_retries_failure: int,
    preemption_count: int,
    max_retries_preemption: int,
) -> bool:
    """Whether a task has reached a terminal state with no remaining retries."""
    if state == TaskState.SUCCEEDED:
        return True
    if state in (TaskState.KILLED, TaskState.UNSCHEDULABLE, TaskState.COSCHED_FAILED):
        return True
    if state == TaskState.FAILED:
        return failure_count > max_retries_failure
    if state in (TaskState.WORKER_FAILED, TaskState.PREEMPTED):
        return preemption_count > max_retries_preemption
    return False


class TaskStateRow(Protocol):
    """Minimal row shape for state-only predicates."""

    state: int


def task_row_can_be_scheduled(task: TaskStateRow) -> bool:
    # Only PENDING tasks are schedulable; a PENDING task is never finished and
    # never has retries exhausted, so state is the sole discriminator here.
    return task.state == TaskState.PENDING


def job_scheduling_deadline(scheduling_deadline_epoch_ms: int | None) -> Deadline | None:
    """Compute scheduling deadline from epoch ms."""
    if scheduling_deadline_epoch_ms is None:
        return None
    return Deadline.after(Timestamp.from_ms(scheduling_deadline_epoch_ms), Duration.from_ms(0))


@dataclass(frozen=True, slots=True)
class ActiveTaskRow:
    """Task projection joined with ``jobs`` + ``job_config``.

    Shared by every cascade/scheduling query (``_kill_non_terminal_tasks``,
    ``peers.find_coscheduled_siblings``, ``ReconcileState`` verbs, poll paths).
    Callers that need resource info for RPC payloads use ``PendingDispatchRow``
    instead; ``ActiveTaskRow`` carries only the fields needed for state-machine
    and cascade logic.
    """

    task_id: JobName
    job_id: JobName
    state: int
    current_attempt_id: int
    current_worker_id: WorkerId | None
    preemption_count: int
    max_retries_failure: int
    max_retries_preemption: int
    has_coscheduling: bool


@dataclass(frozen=True, slots=True)
class TaskDetailRow:
    """Task-detail projection: ``TASK_DETAIL_COLS`` plus the federated worker label.

    ``failure_count`` / ``preemption_count`` are derived from the task's attempt
    rows — there are no such columns on ``tasks``.
    """

    task_id: JobName
    job_id: JobName
    task_index: int
    state: int
    current_attempt_id: int
    failure_count: int
    preemption_count: int
    max_retries_failure: int
    max_retries_preemption: int
    submitted_at_ms: Timestamp
    priority_band: int
    error: str | None
    exit_code: int | None
    started_at_ms: Timestamp | None
    finished_at_ms: Timestamp | None
    current_worker_id: WorkerId | None
    current_worker_address: str | None
    container_id: str | None
    # Backend status one-liner for a waiting/building task (why it is not running
    # yet); None/"" when running or quiet. See tasks.status_message.
    status_message: str | None
    backend_id: str
    cluster: str
    # Federated task's peer-side worker label ("" for a local task); NULL from the
    # outer join when absent.
    peer_worker_label: str | None
