# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Controller task and attempt state predicates."""

from typing import Any, NamedTuple

from rigging.timing import Deadline, Duration, Timestamp
from sqlalchemy import func, literal_column
from sqlalchemy.sql.elements import ColumnElement

from iris.cluster.types import TERMINAL_TASK_STATES, JobName
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

# SQLite planner hint: `tasks.state IN (<active states>)` selects ~0.5% of rows
# on a populated controller DB (~3.7k of ~773k). `sqlite_stat1` only stores the
# average rows-per-distinct-value of an index's leading column, so the planner
# estimates an active-state predicate as ~14% of the table and full-scans
# instead of driving off the small active set. Wrap such predicates with
# `hint_rare_state(...)` so the planner picks the state-driven plan.
#
# The probability argument to SQLite's `likelihood()` must be a literal constant
# in the SQL text, not a bound parameter — `literal_column` inlines it.
# See https://www.sqlite.org/lang_corefunc.html#likelihood.
_RARE_STATE_PROBABILITY = literal_column("0.005")


def hint_rare_state(predicate: ColumnElement[bool]) -> ColumnElement[bool]:
    """Wrap a `state IN (<rare states>)` predicate in SQLite's `likelihood()` hint.

    Used by scheduling-loop queries (per-tick budget spend, per-minute timeout
    enforcement) so the planner drives off the active-state index instead of
    full-scanning the tasks table.
    """
    return func.likelihood(predicate, _RARE_STATE_PROBABILITY)


class RunningTaskEntry(NamedTuple):
    """Task ID and attempt ID pair captured at snapshot time."""

    task_id: JobName
    attempt_id: int


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
