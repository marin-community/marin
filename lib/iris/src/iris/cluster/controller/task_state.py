# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Controller task and attempt state predicates."""

from dataclasses import dataclass
from typing import NamedTuple, Protocol

from rigging.timing import Deadline, Duration, Timestamp
from sqlalchemy import Row

from iris.cluster.types import AttemptUid, JobName, WorkerId
from iris.rpc import job_pb2

ACTIVE_TASK_STATES: frozenset[int] = frozenset(
    {
        job_pb2.TASK_STATE_ASSIGNED,
        job_pb2.TASK_STATE_BUILDING,
        job_pb2.TASK_STATE_RUNNING,
    }
)

# Subset of ACTIVE that excludes ASSIGNED — i.e. tasks already on a worker.
EXECUTING_TASK_STATES: frozenset[int] = frozenset(
    {
        job_pb2.TASK_STATE_BUILDING,
        job_pb2.TASK_STATE_RUNNING,
    }
)

# Subset of ACTIVE that excludes RUNNING — dispatched to a worker (or pod) but
# not yet observed running. These tasks age from their current attempt's
# creation in the ``iris.task_state`` wait-age columns.
DISPATCHED_TASK_STATES: frozenset[int] = frozenset(
    {
        job_pb2.TASK_STATE_ASSIGNED,
        job_pb2.TASK_STATE_BUILDING,
    }
)


class RunningTaskEntry(NamedTuple):
    """Active task attempt captured at snapshot time.

    ``attempt_uid`` is the incarnation key the K8s provider needs to rebuild the
    pod name (which embeds it): a resubmit reuses (task_id, attempt_id) but mints
    a new uid, so poll must target this attempt's pod, not a stale one. Empty off
    the direct-dispatch path. ``state`` lets a provider preserve the controller's
    current state when an observation carries no actionable phase.
    """

    task_id: JobName
    attempt_id: int
    attempt_uid: str = ""
    state: int = job_pb2.TASK_STATE_RUNNING


@dataclass(frozen=True, slots=True)
class RuntimeReleaseTarget:
    """Exact external runtime awaiting backend release confirmation."""

    task_id: JobName
    attempt_id: int
    attempt_uid: AttemptUid
    worker_id: WorkerId | None = None
    worker_address: str | None = None


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


def attempt_is_worker_failure(state: int) -> bool:
    return state in (job_pb2.TASK_STATE_WORKER_FAILED, job_pb2.TASK_STATE_PREEMPTED)


class TaskStateRow(Protocol):
    """Minimal row shape for state-only predicates."""

    state: int


def task_row_can_be_scheduled(task: TaskStateRow) -> bool:
    # Only PENDING tasks are schedulable; a PENDING task is never finished and
    # never has retries exhausted, so state is the sole discriminator here.
    return task.state == job_pb2.TASK_STATE_PENDING


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


@dataclass(frozen=True, slots=True)
class AttemptDetailRow:
    """Attempt projection shared by task, worker, and federation views."""

    task_id: JobName
    attempt_id: int
    worker_id: WorkerId | None
    state: int
    started_at_ms: Timestamp | None
    finished_at_ms: Timestamp | None
    exit_code: int | None
    error: str | None
    attempt_uid: str
    pod_name: str | None
    pod_uid: str | None
    node_name: str | None
    terminal_reason: str | None
    output_archive_json: str | None

    @classmethod
    def from_row(cls, row: Row) -> "AttemptDetailRow":
        return cls(
            task_id=row.task_id,
            attempt_id=row.attempt_id,
            worker_id=row.worker_id,
            state=row.state,
            started_at_ms=row.started_at_ms,
            finished_at_ms=row.finished_at_ms,
            exit_code=row.exit_code,
            error=row.error,
            attempt_uid=row.attempt_uid,
            pod_name=row.pod_name,
            pod_uid=row.pod_uid,
            node_name=row.node_name,
            terminal_reason=row.terminal_reason,
            output_archive_json=row.output_archive_json,
        )


def task_is_finished_row(task: TaskDetailRow) -> bool:
    """Whether a task-detail row is terminal with no retries left."""
    return task_is_finished(
        task.state,
        task.failure_count,
        task.max_retries_failure,
        task.preemption_count,
        task.max_retries_preemption,
    )
