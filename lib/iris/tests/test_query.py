# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the raw SQL query executor."""

import json
from pathlib import Path

import pytest
from iris.cluster.controller.db import ControllerDB
from iris.cluster.controller.query import execute_raw_query


@pytest.fixture
def db(tmp_path: Path) -> ControllerDB:
    cdb = ControllerDB(db_dir=tmp_path)
    _seed_test_data(cdb)
    return cdb


def _seed_test_data(db: ControllerDB) -> None:
    """Insert minimal data into the migrated schema for query testing."""
    with db.transaction() as cur:
        cur.execute(
            "INSERT INTO users (user_id, created_at_ms, display_name, role) VALUES (?, ?, ?, ?)",
            ("alice", 1000, "Alice", "admin"),
        )
        cur.execute(
            "INSERT INTO users (user_id, created_at_ms, display_name, role) VALUES (?, ?, ?, ?)",
            ("bob", 2000, "Bob", "user"),
        )

        for i, (user, state) in enumerate(
            [
                ("alice", 3),  # RUNNING
                ("alice", 4),  # SUCCEEDED
                ("bob", 3),  # RUNNING
            ]
        ):
            job_id = f"/user/job-{i}"
            cur.execute(
                "INSERT INTO jobs (job_id, user_id, parent_job_id, root_job_id, depth, state,"
                " submitted_at_ms, root_submitted_at_ms, started_at_ms, finished_at_ms, error, exit_code,"
                " num_tasks, is_reservation_holder) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    job_id,
                    user,
                    None,
                    job_id,
                    0,
                    state,
                    1000 + i * 100,
                    1000 + i * 100,
                    2000 + i * 100,
                    None if state == 3 else 3000 + i * 100,
                    None,
                    None if state == 3 else 0,
                    2,
                    0,
                ),
            )
            cur.execute(
                "INSERT INTO job_config (job_id, name) VALUES (?, ?)",
                (job_id, f"job-{i}"),
            )
            for t in range(2):
                cur.execute(
                    "INSERT INTO tasks (task_id, job_id, task_index, state, error, exit_code,"
                    " submitted_at_ms, started_at_ms, finished_at_ms, max_retries_failure,"
                    " max_retries_preemption, failure_count, preemption_count, current_attempt_id,"
                    " priority_neg_depth, priority_root_submitted_ms, priority_insertion)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        f"{job_id}/task-{t}",
                        job_id,
                        t,
                        state,
                        None,
                        None if state == 3 else 0,
                        1000 + i * 100,
                        2000 + i * 100,
                        None if state == 3 else 3000 + i * 100,
                        3,
                        5,
                        0,
                        0,
                        0,
                        0,
                        1000 + i * 100,
                        t,
                    ),
                )


def _parse_rows(result) -> list[list]:
    return [json.loads(r) for r in result.rows]


def test_raw_query_select(db: ControllerDB) -> None:
    result = execute_raw_query(db, "SELECT job_id, state FROM jobs")
    assert len(result.rows) == 3
    col_names = [c.name for c in result.columns]
    assert col_names == ["job_id", "state"]


def test_raw_query_reject_insert(db: ControllerDB) -> None:
    with pytest.raises(ValueError, match="Only SELECT"):
        execute_raw_query(db, "INSERT INTO jobs (job_id) VALUES ('x')")


def test_raw_query_reject_update(db: ControllerDB) -> None:
    with pytest.raises(ValueError, match="Only SELECT"):
        execute_raw_query(db, "UPDATE jobs SET state = 0 WHERE 1=1")


def test_raw_query_reject_delete(db: ControllerDB) -> None:
    with pytest.raises(ValueError, match="Only SELECT"):
        execute_raw_query(db, "DELETE FROM jobs WHERE 1=1")


def test_raw_query_reject_drop(db: ControllerDB) -> None:
    with pytest.raises(ValueError, match="Only SELECT"):
        execute_raw_query(db, "DROP TABLE jobs")


def test_raw_query_reject_select_with_drop_keyword(db: ControllerDB) -> None:
    """A SELECT statement that embeds a forbidden keyword is also rejected."""
    with pytest.raises(ValueError, match="Forbidden SQL keyword"):
        execute_raw_query(db, "SELECT * FROM jobs WHERE DROP = 1")


def test_raw_query_with_aggregation(db: ControllerDB) -> None:
    result = execute_raw_query(
        db,
        "SELECT user_id, COUNT(*) as cnt FROM jobs GROUP BY user_id ORDER BY cnt DESC",
    )
    rows = _parse_rows(result)
    assert rows[0][0] == "alice"
    assert rows[0][1] == 2


@pytest.mark.parametrize("keyword", ["PRAGMA", "VACUUM", "REINDEX", "SAVEPOINT"])
def test_raw_query_reject_additional_forbidden_keywords(db: ControllerDB, keyword: str) -> None:
    with pytest.raises(ValueError, match="Forbidden SQL keyword"):
        execute_raw_query(db, f"SELECT * FROM jobs WHERE {keyword} = 1")
