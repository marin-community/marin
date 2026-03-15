# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the raw SQL query executor."""

import json
from pathlib import Path

import pytest
from iris.cluster.controller.db import ControllerDB
from iris.cluster.controller.query import execute_raw_query
from iris.cluster.log_store import LogStore
from iris.rpc import logging_pb2


@pytest.fixture
def db(tmp_path: Path) -> ControllerDB:
    cdb = ControllerDB(tmp_path / "test.db")
    _seed_test_data(cdb)
    return cdb


@pytest.fixture
def log_store(tmp_path: Path) -> LogStore:
    store = LogStore(db_path=tmp_path / "logs.db")
    _seed_log_data(store)
    return store


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


def _seed_log_data(store: LogStore) -> None:
    """Insert test data into the log store."""
    entries = []
    for i in range(5):
        entry = logging_pb2.LogEntry(
            source="test",
            data=f"log line {i}",
            level=20,
        )
        entry.timestamp.epoch_ms = 1000 + i * 100
        entries.append(entry)
    store.append("/test/task-0", entries)


def _parse_rows(result) -> list[list]:
    return [json.loads(r) for r in result.rows]


# ---- Raw query: basic SELECT ----


def test_raw_query_select(db: ControllerDB) -> None:
    result = execute_raw_query(db, "SELECT job_id, state FROM jobs")
    assert len(result.rows) == 3
    col_names = [c.name for c in result.columns]
    assert col_names == ["job_id", "state"]


def test_raw_query_with_aggregation(db: ControllerDB) -> None:
    result = execute_raw_query(
        db,
        "SELECT user_id, COUNT(*) as cnt FROM jobs GROUP BY user_id ORDER BY cnt DESC",
    )
    rows = _parse_rows(result)
    assert rows[0][0] == "alice"
    assert rows[0][1] == 2


def test_raw_query_with_where(db: ControllerDB) -> None:
    result = execute_raw_query(db, "SELECT job_id FROM jobs WHERE state = 4")
    rows = _parse_rows(result)
    assert len(rows) == 1
    assert rows[0][0] == "/user/job-1"


def test_raw_query_with_join(db: ControllerDB) -> None:
    result = execute_raw_query(
        db,
        "SELECT t.task_id, j.user_id FROM tasks t JOIN jobs j ON t.job_id = j.job_id WHERE j.user_id = 'alice'",
    )
    rows = _parse_rows(result)
    assert len(rows) == 4  # alice has 2 jobs * 2 tasks
    for row in rows:
        assert row[1] == "alice"


# ---- Raw query: rejection ----


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


def test_raw_query_reject_multi_statement(db: ControllerDB) -> None:
    with pytest.raises(ValueError, match="Multiple SQL"):
        execute_raw_query(db, "SELECT 1; SELECT 2")


# ---- Snapshot isolation ----


def test_snapshot_isolation_sees_consistent_data(db: ControllerDB) -> None:
    """Verify snapshot reads work and return expected data."""
    result = execute_raw_query(db, "SELECT COUNT(*) FROM jobs")
    rows = _parse_rows(result)
    assert rows[0][0] == 3


# ---- Log store queries ----


def test_log_store_select(db: ControllerDB, log_store: LogStore) -> None:
    result = execute_raw_query(db, "SELECT * FROM logs", log_store=log_store, database="logs")
    rows = _parse_rows(result)
    assert len(rows) == 5
    col_names = [c.name for c in result.columns]
    assert "key" in col_names
    assert "data" in col_names


def test_log_store_with_filter(db: ControllerDB, log_store: LogStore) -> None:
    result = execute_raw_query(
        db,
        "SELECT data FROM logs WHERE epoch_ms > 1200",
        log_store=log_store,
        database="logs",
    )
    rows = _parse_rows(result)
    assert len(rows) == 2  # epoch_ms 1300, 1400


def test_log_store_count(db: ControllerDB, log_store: LogStore) -> None:
    result = execute_raw_query(
        db,
        "SELECT COUNT(*) as total FROM logs",
        log_store=log_store,
        database="logs",
    )
    rows = _parse_rows(result)
    assert rows[0][0] == 5


def test_log_store_not_available(db: ControllerDB) -> None:
    with pytest.raises(ValueError, match="Log store not available"):
        execute_raw_query(db, "SELECT * FROM logs", database="logs")


# ---- truncation ----


def test_not_truncated_when_under_limit(db: ControllerDB) -> None:
    result = execute_raw_query(db, "SELECT job_id FROM jobs")
    assert not result.truncated
    assert len(result.rows) == 3


def test_unknown_database_rejected(db: ControllerDB) -> None:
    with pytest.raises(ValueError, match="Unknown database"):
        execute_raw_query(db, "SELECT 1", database="typo")


# ---- Blob encoding ----


def test_blob_encoded_as_placeholder(db: ControllerDB) -> None:
    result = execute_raw_query(db, "SELECT request_proto FROM jobs LIMIT 1")
    rows = _parse_rows(result)
    assert rows[0][0].startswith("<blob:")
