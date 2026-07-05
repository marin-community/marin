# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pure data shapes for one pure-function call into the state machine.

`TransitionSnapshot` is the closed input bundle; the leaf dataclasses are its
row shapes.
"""

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
    total_failures: int  # committed-derived cumulative FAILED attempts across the job (loader-summed)
    first_task_error: str | None


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
