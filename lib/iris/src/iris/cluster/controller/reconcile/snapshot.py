# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pure data shapes for one pure-function call into the state machine.

`TransitionSnapshot` is the closed input bundle; the leaf dataclasses are its
row shapes.
"""

import re
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
    # Backend status one-liner for a waiting/building task. Tri-state: ``None`` means
    # "this update does not speak to the status message; leave it unchanged"; ``""``
    # clears it (task now running/quiet); a string sets it. A provider that reports it
    # (k8s) sets it on every update so it clears naturally on RUNNING.
    status_message: str | None = None


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


# The controller stamps a coscheduled gang's siblings with a derived error when
# the gang unwinds: ``Coscheduled sibling <id> failed`` / ``... bounced for
# atomic re-scheduling`` (reconcile/peers.py). These only echo the one sibling
# that actually crashed, so when a job's root cause is chosen they must not mask
# it. Scoped deliberately to the coscheduled cascade: a scheduling timeout,
# preemption, or cancellation is a standalone reason a task failed — often the
# state-driving one — so demoting those would detach ``job.error`` from the
# job's terminal state. Anchored so an application error that merely quotes the
# phrase is not misread as derived.
_DERIVED_ERROR_PATTERNS = (re.compile(r"^Coscheduled sibling\b"),)

# COSCHED_FAILED is the cascade by construction: the task was torn down or
# bounced because a sibling failed, not on its own merits. Caught by state even
# if the recorded error text drifts from the pattern above.
_DERIVED_ERROR_STATES = frozenset({job_pb2.TASK_STATE_COSCHED_FAILED})


def is_derived_task_error(state: int, error: str) -> bool:
    """Whether a terminal task's error only echoes a coscheduled sibling's
    failure, carrying no root-cause signal of its own.

    Used to keep a coscheduled gang's cascade — every sibling stamped
    ``Coscheduled sibling ... bounced for atomic re-scheduling`` — from masking
    the one real crash when a job's root cause is chosen.
    """
    if state in _DERIVED_ERROR_STATES:
        return True
    return any(pattern.search(error) for pattern in _DERIVED_ERROR_PATTERNS)


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

    Derived errors (``is_derived_task_error`` — a coscheduled sibling bounce or
    teardown) are deprioritized against genuine failures: they only echo the
    sibling that actually crashed, so a real application error is preferred even
    when a derived one finished first. A derived error surfaces only when it is
    the sole thing recorded — better than an empty error.
    """
    failed = [
        (finished_at, task_index, state, error)
        for task_index, state, finished_at, error in candidates
        if error is not None
        and finished_at is not None
        and state in TERMINAL_TASK_STATES
        and state != job_pb2.TASK_STATE_SUCCEEDED
    ]
    if not failed:
        return None
    primary = [c for c in failed if not is_derived_task_error(c[2], c[3])]
    return min(primary or failed, key=lambda c: (c[0].epoch_ms(), c[1]))[3]


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
