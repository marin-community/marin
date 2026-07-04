# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Controller task and attempt state predicates."""

from dataclasses import dataclass
from typing import NamedTuple, Protocol

from rigging.timing import Deadline, Duration, Timestamp

from iris.cluster.types import JobName, WorkerId
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


class RunningTaskEntry(NamedTuple):
    """Task ID and attempt ID pair captured at snapshot time.

    ``coscheduled`` lets the direct (K8s) provider classify a vanished pod as
    a gang preemption (WORKER_FAILED) rather than an application failure: when
    Kueue preempts a pod group it deletes every pod, leaving no terminal pod
    status to read — only the absence. See K8sTaskProvider._poll_pods.
    """

    task_id: JobName
    attempt_id: int
    coscheduled: bool = False


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
    failure_count: int
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
    backend_id: str
    cluster: str
    # Federated task's peer-side worker label ("" for a local task); NULL from the
    # outer join when absent.
    peer_worker_label: str | None
