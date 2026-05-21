# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Regression: scheduling-loop queries pin SQLite to a state-driven plan.

`tasks.state` distribution is extreme (terminal states ~99% of rows, active
states ~0.5%) and `sqlite_stat1` only stores average rows-per-distinct-value,
so the planner mis-estimates `state IN (<rare states>)` and full-scans the
tasks table on every scheduling tick. Both queries here wrap the predicate
in :func:`hint_rare_state`, which emits the `likelihood(..., 0.005)` planner
hint; these tests pin both that the hint is present in the compiled SQL and
that EXPLAIN QUERY PLAN no longer scans tasks.
"""

from collections.abc import Iterator
from pathlib import Path

import pytest
from iris.cluster.controller.budget import _USER_SPEND_QUERY
from iris.cluster.controller.controller import _EXECUTION_TIMEOUT_QUERY
from iris.cluster.controller.db import ControllerDB
from iris.cluster.controller.task_state import ACTIVE_TASK_STATES
from iris.rpc import job_pb2
from sqlalchemy import text
from sqlalchemy.dialects import sqlite


def _compiled_sql(query) -> str:
    """Render a query against the SQLite dialect for substring assertions."""
    return str(query.compile(dialect=sqlite.dialect()))


def test_user_spend_query_carries_likelihood_hint() -> None:
    sql = _compiled_sql(_USER_SPEND_QUERY)
    assert "likelihood(tasks.state IN" in sql
    assert "0.005" in sql


def test_execution_timeout_query_carries_likelihood_hint() -> None:
    sql = _compiled_sql(_EXECUTION_TIMEOUT_QUERY)
    assert "likelihood(tasks.state IN" in sql
    assert "0.005" in sql


# --- EXPLAIN QUERY PLAN integration -----------------------------------------


_TERMINAL_JOBS = 50
_TASKS_PER_TERMINAL_JOB = 100  # 5000 terminal tasks
_ACTIVE_JOBS = 2
_TASKS_PER_ACTIVE_JOB = 10  # 20 active tasks


def _seed_skewed_tasks(db: ControllerDB) -> None:
    """Seed a controller DB with a realistic active/terminal task skew.

    The active set is ~0.4% of total tasks — close to the production skew
    that triggers the planner mis-estimate. After seeding the test must run
    ANALYZE so the planner sees the new distribution.
    """
    with db.transaction() as cur:
        cur.execute(
            text("INSERT INTO users (user_id, created_at_ms, role) VALUES (:uid, :ts, :role)"),
            {"uid": "u1", "ts": 1_000, "role": "user"},
        )
        for j in range(_TERMINAL_JOBS):
            job_id = f"/u1/term-{j:04d}"
            cur.execute(
                text(
                    "INSERT INTO jobs ("
                    "  job_id, user_id, root_job_id, depth, state,"
                    "  submitted_at_ms, root_submitted_at_ms, num_tasks,"
                    "  is_reservation_holder, has_reservation"
                    ") VALUES (:jid, 'u1', :jid, 0, :state, 2000, 2000, :n, 0, 0)"
                ),
                {"jid": job_id, "state": job_pb2.JOB_STATE_SUCCEEDED, "n": _TASKS_PER_TERMINAL_JOB},
            )
            cur.execute(
                text("INSERT INTO job_config (job_id, name) VALUES (:jid, :name)"),
                {"jid": job_id, "name": f"term-{j}"},
            )
            for t in range(_TASKS_PER_TERMINAL_JOB):
                cur.execute(
                    text(
                        "INSERT INTO tasks ("
                        "  task_id, job_id, task_index, state, submitted_at_ms,"
                        "  max_retries_failure, max_retries_preemption,"
                        "  failure_count, preemption_count,"
                        "  priority_neg_depth, priority_root_submitted_ms, priority_insertion,"
                        "  current_attempt_id"
                        ") VALUES (:tid, :jid, :idx, :state, 2000, 0, 0, 0, 0, 0, 2000, 0, 0)"
                    ),
                    {
                        "tid": f"{job_id}/{t}",
                        "jid": job_id,
                        "idx": t,
                        "state": job_pb2.TASK_STATE_SUCCEEDED,
                    },
                )
        for j in range(_ACTIVE_JOBS):
            job_id = f"/u1/act-{j:04d}"
            cur.execute(
                text(
                    "INSERT INTO jobs ("
                    "  job_id, user_id, root_job_id, depth, state,"
                    "  submitted_at_ms, root_submitted_at_ms, num_tasks,"
                    "  is_reservation_holder, has_reservation"
                    ") VALUES (:jid, 'u1', :jid, 0, :state, 2000, 2000, :n, 0, 0)"
                ),
                {"jid": job_id, "state": job_pb2.JOB_STATE_RUNNING, "n": _TASKS_PER_ACTIVE_JOB},
            )
            cur.execute(
                text(
                    "INSERT INTO job_config ("
                    "  job_id, name, res_cpu_millicores, res_memory_bytes, timeout_ms"
                    ") VALUES (:jid, :name, 1000, 1073741824, 60000)"
                ),
                {"jid": job_id, "name": f"act-{j}"},
            )
            for t in range(_TASKS_PER_ACTIVE_JOB):
                task_id = f"{job_id}/{t}"
                cur.execute(
                    text(
                        "INSERT INTO tasks ("
                        "  task_id, job_id, task_index, state, submitted_at_ms,"
                        "  max_retries_failure, max_retries_preemption,"
                        "  failure_count, preemption_count,"
                        "  priority_neg_depth, priority_root_submitted_ms, priority_insertion,"
                        "  current_attempt_id"
                        ") VALUES (:tid, :jid, :idx, :state, 2000, 0, 0, 0, 0, 0, 2000, 0, 0)"
                    ),
                    {
                        "tid": task_id,
                        "jid": job_id,
                        "idx": t,
                        "state": job_pb2.TASK_STATE_RUNNING,
                    },
                )
                cur.execute(
                    text(
                        "INSERT INTO task_attempts ("
                        "  task_id, attempt_id, state, created_at_ms, started_at_ms, attempt_uid"
                        ") VALUES (:tid, 0, :state, 2000, 2500, :uid)"
                    ),
                    {
                        "tid": task_id,
                        "state": job_pb2.TASK_STATE_RUNNING,
                        "uid": f"a{j:08x}{t:08x}",
                    },
                )

    # ControllerDB.__init__ runs ANALYZE on empty tables; re-run so
    # sqlite_stat1 reflects the seeded skew.
    raw = db._sa_write_engine.raw_connection()
    try:
        raw.execute("ANALYZE")
    finally:
        raw.close()


@pytest.fixture
def skewed_db(tmp_path: Path) -> Iterator[ControllerDB]:
    db = ControllerDB(db_dir=tmp_path)
    try:
        _seed_skewed_tasks(db)
        yield db
    finally:
        db.close()


def _explain_query_plan(db: ControllerDB, query, postcompile_name: str, state_values: list[int]) -> str:
    """Return the joined `detail` column of EXPLAIN QUERY PLAN as a string.

    SQLAlchemy renders `IN (bindparam(..., expanding=True))` as a postcompile
    token; we substitute it with an inline integer list so EXPLAIN can parse
    the SQL.
    """
    compiled = str(query.compile(dialect=sqlite.dialect(), compile_kwargs={"literal_binds": True}))
    inlined = ",".join(str(int(v)) for v in state_values)
    sql = compiled.replace(f"(__[POSTCOMPILE_{postcompile_name}])", f"({inlined})")
    with db.read_snapshot() as tx:
        rows = tx.execute(text(f"EXPLAIN QUERY PLAN {sql}")).all()
    return "\n".join(str(row[3]) for row in rows)


def test_user_spend_plan_drives_off_state_index(skewed_db: ControllerDB) -> None:
    """With the hint, the planner must search tasks via a state index, not scan."""
    plan = _explain_query_plan(skewed_db, _USER_SPEND_QUERY, "states", list(ACTIVE_TASK_STATES))
    # Must use an index keyed on state; must not scan tasks.
    assert "SCAN tasks" not in plan, f"Unexpected tasks scan in plan:\n{plan}"
    assert "SEARCH tasks" in plan and "state=?" in plan, f"Expected state-driven search:\n{plan}"


def test_execution_timeout_plan_drives_off_state_index(skewed_db: ControllerDB) -> None:
    plan = _explain_query_plan(
        skewed_db,
        _EXECUTION_TIMEOUT_QUERY,
        "executing_states",
        [int(job_pb2.TASK_STATE_BUILDING), int(job_pb2.TASK_STATE_RUNNING)],
    )
    assert "SCAN tasks" not in plan, f"Unexpected tasks scan in plan:\n{plan}"
    assert "SEARCH tasks" in plan and "state=?" in plan, f"Expected state-driven search:\n{plan}"
