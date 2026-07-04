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
from iris.cluster.types import AttemptUid, JobName, WorkerId


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
    total_failures: int  # cumulative failed attempts across the job (sum of task failure_count)
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
    # None while the task has not reached a genuinely terminal state (e.g. a
    # coscheduled sibling bounced back to PENDING for retry, whose error records
    # the cascade without ever finishing it).
    finished_at: Timestamp | None = None


def task_error_rank(finished_at: Timestamp | None, task_index: int) -> tuple[int, int, int]:
    """Sort key ordering tasks by which one failed first.

    Orders by ``finished_at`` so the task that actually failed first outranks
    siblings that only fail later after timing out waiting on it (e.g. a JAX
    shutdown barrier). A task with no ``finished_at`` sorts after every task
    that did finish; ties break by ``task_index`` for determinism.
    """
    if finished_at is None:
        return (1, 0, task_index)
    return (0, finished_at.epoch_ms(), task_index)


def pick_earliest_task_error(candidates: Iterable[tuple[int, Timestamp | None, str | None]]) -> str | None:
    """Return the error of the task that failed first among ``candidates``.

    ``candidates`` is ``(task_index, finished_at, error)`` per task; entries
    with a ``None`` error are skipped. Ranks by ``finished_at`` (earliest
    wins, a task with none sorts last, ties break by ``task_index``), so the
    result is a job's root-cause task, not necessarily the lowest-indexed task
    with a non-null error.
    """
    best_rank: tuple[int, int, int] | None = None
    best_error: str | None = None
    for task_index, finished_at, error in candidates:
        if error is None:
            continue
        rank = task_error_rank(finished_at, task_index)
        if best_rank is None or rank < best_rank:
            best_rank = rank
            best_error = error
    return best_error


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
