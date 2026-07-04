# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Derive a task's failure/preemption counts from its persisted attempt rows.

The attempt rows (``task_attempts``) are the authoritative record of what
happened to a task, so the two retry counters are a pure function of them —
there is no denormalized ``tasks.failure_count`` / ``tasks.preemption_count`` to
keep in sync. This module is the single definition of that function, exposed two
ways so every path computes identical numbers:

- :func:`counts_from_attempts` for callers that already hold the attempt rows
  (per-task RPC, tests).
- :func:`failure_count_expr` / :func:`preemption_count_expr` — SQLAlchemy
  aggregate expressions over ``task_attempts`` for GROUP-BY derivation (the
  reconcile snapshot loader, the projection rehydrate, the job-list sort).

Semantics (see ``reconcile/task.py`` for the increment logic they reproduce):

- **failure** — an attempt with ``state == TASK_STATE_FAILED``. Application
  failures charge the failure budget regardless of phase.
- **preemption** — an attempt in :data:`PREEMPTION_ATTEMPT_STATES` that reached
  the executing phase (``started_at_ms IS NOT NULL``). BUILDING and RUNNING are
  the only transitions that stamp an attempt's ``started_at``, so this predicate
  is exactly the ``prior_state in EXECUTING_TASK_STATES`` gate in
  ``resolve_task_failure_state``: an ASSIGNED-phase worker/kill/preempt failure
  retries without charging and has ``started_at_ms IS NULL``.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

from sqlalchemy import case, func
from sqlalchemy.sql.elements import ColumnElement

from iris.cluster.controller.schema import task_attempts_table
from iris.rpc import job_pb2

# An attempt in one of these terminal states charges the preemption budget when
# it reached the executing phase. KILLED shares the budget with WORKER_FAILED: an
# out-of-band container stop of a live attempt (higher-priority reclaim, drain,
# spot reclaim) is a preemption, not an application failure.
PREEMPTION_ATTEMPT_STATES: frozenset[int] = frozenset(
    {
        job_pb2.TASK_STATE_WORKER_FAILED,
        job_pb2.TASK_STATE_KILLED,
        job_pb2.TASK_STATE_PREEMPTED,
    }
)


@dataclass(frozen=True, slots=True)
class AttemptCounts:
    """Retry counters derived from a task's attempt rows."""

    failure_count: int = 0
    preemption_count: int = 0


class AttemptCountRow(Protocol):
    """Minimal attempt-row shape the pure derivation reads."""

    state: int
    # ``TimestampMsType`` decodes to a ``Timestamp``; typed as object so this leaf
    # module needs no timing import. Only its None-ness is inspected.
    started_at_ms: object | None


def _is_executing_preemption(state: int, started_at_ms: object | None) -> bool:
    return state in PREEMPTION_ATTEMPT_STATES and started_at_ms is not None


def counts_from_attempts(attempts: Iterable[AttemptCountRow]) -> AttemptCounts:
    """Derive :class:`AttemptCounts` from an iterable of attempt rows."""
    failure = 0
    preemption = 0
    for attempt in attempts:
        state = int(attempt.state)
        if state == job_pb2.TASK_STATE_FAILED:
            failure += 1
        elif _is_executing_preemption(state, attempt.started_at_ms):
            preemption += 1
    return AttemptCounts(failure_count=failure, preemption_count=preemption)


def failure_count_expr() -> ColumnElement[int]:
    """SQL sum of an attempt group's FAILED attempts (mirrors the failure branch)."""
    return func.coalesce(
        func.sum(case((task_attempts_table.c.state == job_pb2.TASK_STATE_FAILED, 1), else_=0)),
        0,
    )


def preemption_count_expr() -> ColumnElement[int]:
    """SQL sum of an attempt group's executing-phase preemption attempts."""
    return func.coalesce(
        func.sum(
            case(
                (
                    task_attempts_table.c.state.in_(PREEMPTION_ATTEMPT_STATES)
                    & task_attempts_table.c.started_at_ms.is_not(None),
                    1,
                ),
                else_=0,
            )
        ),
        0,
    )
