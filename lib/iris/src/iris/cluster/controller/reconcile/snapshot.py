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


# Errors the controller stamps to describe the *consequence* of another task's
# failure — or an orchestration action — rather than an application fault. When
# a coscheduled gang crash-loops, every sibling is stamped one of these (see
# reconcile/peers.py), so they swamp the single real crash. They carry no
# root-cause signal of their own, so ``pick_earliest_task_error`` deprioritizes
# them: they represent a job only when nothing better was recorded. Anchored so
# an application error that merely quotes one of these phrases is not misread as
# derived. Each generator is cited next to its pattern.
_DERIVED_ERROR_PATTERNS = (
    re.compile(r"^Coscheduled sibling\b"),  # gang unwind cascade — reconcile/peers.py
    re.compile(r"^Preempted by\b"),  # preemption — controller/backend.py
    re.compile(r"^Scheduling timeout exceeded\b"),  # scheduling gave up — reconcile/task.py
    re.compile(r"^Cancelled\b"),  # federation/user cancel — federation_store.py
    re.compile(r"^worker_lost_spec$"),  # worker dropped the task spec — reconcile/worker.py
    re.compile(r"^Reconcile RPC failed:"),  # worker reconcile RPC error — reconcile/worker.py
)

# States whose recorded error is derived by construction: the task did not fail
# on its own merits but was torn down because a coscheduled sibling did. Caught
# by state even if the error text drifts from the patterns above.
_DERIVED_ERROR_STATES = frozenset({job_pb2.TASK_STATE_COSCHED_FAILED})


def is_derived_task_error(state: int, error: str) -> bool:
    """Whether a terminal task's error only echoes another task's failure or an
    orchestration action, carrying no root-cause signal of its own.

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

    Derived errors (``is_derived_task_error`` — a coscheduled sibling bounce, a
    preemption, a scheduling giveup) are deprioritized against genuine failures:
    they only echo another task's failure or an orchestration action, so a real
    application error is preferred even when a derived one finished first. A
    derived error surfaces only when it is the sole thing recorded — better than
    an empty error.
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
