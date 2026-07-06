# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pure data shapes for one pure-function call into the state machine.

`TransitionSnapshot` is the closed input bundle; the leaf dataclasses are its
row shapes.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from rigging.timing import Timestamp

from iris.cluster.controller.task_state import ActiveTaskRow, TaskDetailRow
from iris.cluster.types import TERMINAL_TASK_STATES, AttemptUid, JobName, WorkerId
from iris.rpc import job_pb2


@dataclass(frozen=True)
class TaskUpdate:
    """Neutral single-task state update consumed by the transition kernel.

    Reconcile-plan observations and direct-provider reports both produce one
    of these; ``batches.py`` runs a shared kernel over them.
    Lives in ``snapshot.py`` (a leaf) so both ``task.py`` and ``worker.py``
    can build/consume it without an aggregate cross-import.
    """

    task_id: JobName
    attempt_id: int
    new_state: int
    error: str | None = None
    exit_code: int | None = None
    container_id: str | None = None


@dataclass(frozen=True, slots=True)
class JobConfigRow:
    job_id: JobName
    has_coscheduling: bool
    max_task_failures: int
    preemption_policy: int  # JOB_PREEMPTION_POLICY_*
    num_tasks: int


@dataclass(frozen=True, slots=True)
class JobStateBasis:
    job_id: JobName
    state: int
    started_at: Timestamp | None
    max_task_failures: int
    task_state_counts: dict[int, int]  # task state → count
    total_failures: int  # committed-derived cumulative FAILED attempts across the job (loader-summed)
    first_task_error: str | None  # the error of the task that failed first (the root cause), not task index 0


@dataclass(frozen=True, slots=True)
class JobDescendants:
    job_id: JobName
    descendants: tuple[JobName, ...]


@dataclass(frozen=True, slots=True)
class TaskHistogramRow:
    task_id: JobName
    task_index: int
    state: int
    failure_count: int
    error: str | None
    # Set only once the task reaches a terminal state; None while it is still
    # active or bounced back to PENDING for a retry.
    finished_at: Timestamp | None = None


def pick_earliest_task_error(candidates: Iterable[tuple[int, int, Timestamp | None, str | None]]) -> str | None:
    """Return the error of the failed task that finished first among ``candidates``.

    ``candidates`` is ``(task_index, state, finished_at, error)`` per task.
    Considers only tasks that finished in a failed terminal state with a
    recorded error, then returns the earliest-finishing one's error (ties break
    by ``task_index``). This picks a coscheduled gang's true root cause — the
    sibling that crashed first — over a follower that only timed out waiting on
    it. Tasks still retrying (no ``finished_at``) and tasks that ultimately
    succeeded (a stale error preserved from an earlier failed attempt) are
    excluded.
    """
    failed = [
        (finished_at, task_index, error)
        for task_index, state, finished_at, error in candidates
        if error is not None
        and finished_at is not None
        and state in TERMINAL_TASK_STATES
        and state != job_pb2.TASK_STATE_SUCCEEDED
    ]
    if not failed:
        return None
    return min(failed, key=lambda c: (c[0].epoch_ms(), c[1]))[2]


@dataclass(frozen=True)
class TransitionSnapshot:
    """Pre-loaded inputs for one pure-function call into the state machine."""

    now: Timestamp
    tasks: dict[JobName, TaskDetailRow]
    attempts: dict[tuple[JobName, int], Any]
    attempt_uid_index: dict[AttemptUid, tuple[JobName, int]]
    job_configs: dict[JobName, JobConfigRow]
    job_state_basis: dict[JobName, JobStateBasis]
    job_descendants: dict[JobName, JobDescendants]
    all_tasks_by_job: dict[JobName, tuple[TaskHistogramRow, ...]]
    active_tasks_by_job: dict[JobName, tuple[ActiveTaskRow, ...]]
    active_workers: frozenset[WorkerId]
