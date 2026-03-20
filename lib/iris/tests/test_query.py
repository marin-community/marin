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
    cdb = ControllerDB(tmp_path / "test.db", auth_db_path=tmp_path / "auth.sqlite3")
    _seed_test_data(cdb)
    return cdb


def _seed_test_data(db: ControllerDB) -> None:
    """Insert minimal data into the migrated schema for query testing."""
    with db.transaction() as cur:
        cur.insert(
            "users",
            {
                "user_id": "alice",
                "created_at_ms": 1000,
                "display_name": "Alice",
                "role": "admin",
            },
        )
        cur.insert(
            "users",
            {
                "user_id": "bob",
                "created_at_ms": 2000,
                "display_name": "Bob",
                "role": "user",
            },
        )

        for i, (user, state) in enumerate(
            [
                ("alice", 3),  # RUNNING
                ("alice", 4),  # SUCCEEDED
                ("bob", 3),  # RUNNING
            ]
        ):
            job_id = f"/user/job-{i}"
            cur.insert(
                "jobs",
                {
                    "job_id": job_id,
                    "user_id": user,
                    "parent_job_id": None,
                    "root_job_id": job_id,
                    "depth": 0,
                    "request_proto": b"fake-proto",
                    "state": state,
                    "submitted_at_ms": 1000 + i * 100,
                    "root_submitted_at_ms": 1000 + i * 100,
                    "started_at_ms": 2000 + i * 100,
                    "finished_at_ms": None if state == 3 else 3000 + i * 100,
                    "error": None,
                    "exit_code": None if state == 3 else 0,
                    "num_tasks": 2,
                    "is_reservation_holder": 0,
                },
            )
            for t in range(2):
                cur.insert(
                    "tasks",
                    {
                        "task_id": f"{job_id}/task-{t}",
                        "job_id": job_id,
                        "task_index": t,
                        "state": state,
                        "error": None,
                        "exit_code": None if state == 3 else 0,
                        "submitted_at_ms": 1000 + i * 100,
                        "started_at_ms": 2000 + i * 100,
                        "finished_at_ms": None if state == 3 else 3000 + i * 100,
                        "max_retries_failure": 3,
                        "max_retries_preemption": 5,
                        "failure_count": 0,
                        "preemption_count": 0,
                        "current_attempt_id": 0,
                        "priority_neg_depth": 0,
                        "priority_root_submitted_ms": 1000 + i * 100,
                        "priority_insertion": t,
                    },
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
