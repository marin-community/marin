# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the attempt-count derivation (the single source of truth).

Covers the pure :func:`counts_from_attempts` and asserts the SQL aggregate
expressions produce identical numbers over the same attempt rows.
"""

from dataclasses import dataclass

import pytest
from iris.cluster.controller.attempt_counts import (
    AttemptCounts,
    counts_from_attempts,
    failure_count_expr,
    preemption_count_expr,
)
from iris.cluster.controller.schema import task_attempts_table
from iris.rpc import job_pb2
from sqlalchemy import MetaData, Table, create_engine, insert, select

FAILED = job_pb2.TASK_STATE_FAILED
WORKER_FAILED = job_pb2.TASK_STATE_WORKER_FAILED
KILLED = job_pb2.TASK_STATE_KILLED
PREEMPTED = job_pb2.TASK_STATE_PREEMPTED
SUCCEEDED = job_pb2.TASK_STATE_SUCCEEDED
RUNNING = job_pb2.TASK_STATE_RUNNING
ASSIGNED = job_pb2.TASK_STATE_ASSIGNED

_STARTED = 1_000  # any non-null started_at_ms (executing phase)


@dataclass(frozen=True)
class _Attempt:
    state: int
    started_at_ms: int | None


# (label, attempts, expected failure_count, expected preemption_count)
_CASES = [
    ("empty", [], 0, 0),
    ("single_success", [_Attempt(SUCCEEDED, _STARTED)], 0, 0),
    ("app_failure_executing", [_Attempt(FAILED, _STARTED)], 1, 0),
    # An application failure charges the failure budget regardless of phase.
    ("app_failure_no_start", [_Attempt(FAILED, None)], 1, 0),
    # Executing-phase worker/kill/preempt → preemption.
    ("worker_failed_executing", [_Attempt(WORKER_FAILED, _STARTED)], 0, 1),
    ("killed_executing", [_Attempt(KILLED, _STARTED)], 0, 1),
    ("preempted_executing", [_Attempt(PREEMPTED, _STARTED)], 0, 1),
    # ASSIGNED-phase worker/kill/preempt retries WITHOUT charging (started_at NULL).
    ("worker_failed_assigned", [_Attempt(WORKER_FAILED, None)], 0, 0),
    ("killed_assigned", [_Attempt(KILLED, None)], 0, 0),
    ("preempted_assigned", [_Attempt(PREEMPTED, None)], 0, 0),
    (
        "mixed_assigned_and_executing",
        [
            _Attempt(WORKER_FAILED, None),  # assigned-phase: not charged
            _Attempt(WORKER_FAILED, _STARTED),  # executing: preemption
            _Attempt(FAILED, _STARTED),  # app failure
            _Attempt(PREEMPTED, _STARTED),  # executing: preemption
            _Attempt(RUNNING, _STARTED),  # in-flight: neither
        ],
        1,
        2,
    ),
]


@pytest.mark.parametrize("label,attempts,failures,preemptions", _CASES, ids=[c[0] for c in _CASES])
def test_counts_from_attempts(label, attempts, failures, preemptions):
    assert counts_from_attempts(attempts) == AttemptCounts(failure_count=failures, preemption_count=preemptions)


@pytest.mark.parametrize("label,attempts,failures,preemptions", _CASES, ids=[c[0] for c in _CASES])
def test_sql_exprs_match_pure(label, attempts, failures, preemptions):
    """The GROUP-BY SQL expressions must agree with the pure function."""
    engine = create_engine("sqlite://")
    scratch = MetaData()
    # Copy just task_attempts into a scratch metadata so we can CREATE it without
    # its FK target (tasks); SQLite does not enforce the FK without PRAGMA.
    Table("task_attempts", scratch, *(c.copy() for c in task_attempts_table.c))
    scratch.tables["task_attempts"].create(engine)

    with engine.begin() as conn:
        for i, a in enumerate(attempts):
            conn.execute(
                insert(task_attempts_table).values(
                    task_id="/j/0",
                    attempt_id=i,
                    worker_id=None,
                    state=a.state,
                    created_at_ms=0,
                    started_at_ms=a.started_at_ms,
                    attempt_uid=f"uid{i}",
                )
            )
        row = conn.execute(
            select(
                failure_count_expr().label("failure_count"),
                preemption_count_expr().label("preemption_count"),
            )
        ).one()

    # Differential: the production SQL must reproduce the pure reference exactly
    # (which the case table independently pins to the expected numbers).
    sql_counts = AttemptCounts(failure_count=int(row.failure_count), preemption_count=int(row.preemption_count))
    assert (
        sql_counts
        == counts_from_attempts(attempts)
        == AttemptCounts(failure_count=failures, preemption_count=preemptions)
    )
