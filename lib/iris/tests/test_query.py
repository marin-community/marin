# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the structured query executor."""

import json
from pathlib import Path

import pytest
from iris.cluster.controller.db import ControllerDB
from iris.cluster.controller.query import (
    DEFAULT_LIMIT,
    MAX_JOINS,
    MAX_LIMIT,
    execute_query,
    execute_raw_query,
)
from iris.cluster.log_store import LogStore
from iris.rpc import logging_pb2, query_pb2


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

        cur.insert(
            "workers",
            {
                "worker_id": "w-1",
                "address": "10.0.0.1:8080",
                "metadata_proto": b"meta",
                "healthy": 1,
                "active": 1,
                "consecutive_failures": 0,
                "last_heartbeat_ms": 5000,
                "committed_cpu_millicores": 4000,
                "committed_mem_bytes": 8_000_000_000,
                "committed_gpu": 1,
                "committed_tpu": 0,
            },
        )

        cur.insert(
            "endpoints",
            {
                "endpoint_id": "ep-1",
                "name": "svc/model-a",
                "address": "10.0.0.1:9090",
                "job_id": "/user/job-0",
                "task_id": "/user/job-0/task-0",
                "metadata_json": "{}",
                "registered_at_ms": 3000,
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


def _make_filter(**kwargs) -> query_pb2.QueryFilter:
    """Build a QueryFilter, handling reserved Python keywords ('in', 'not')."""
    renamed = {}
    for k, v in kwargs.items():
        key = k.rstrip("_") if k in ("in_", "not_") else k
        renamed[key] = v
    return query_pb2.QueryFilter(**renamed)


def _make_query(**kwargs) -> query_pb2.Query:
    """Build a Query proto, handling the reserved 'from' keyword."""
    from_table = kwargs.pop("from_table", None)
    q = query_pb2.Query(**kwargs)
    if from_table is not None:
        getattr(q, "from").CopyFrom(from_table)
    return q


def _parse_rows(result) -> list[list]:
    return [json.loads(r) for r in result.rows]


# ---- Basic SELECT ----


def test_select_all_from_users(db: ControllerDB) -> None:
    query = _make_query(from_table=query_pb2.QueryTable(name="users"))
    result = execute_query(db, query, is_admin=False)
    rows = _parse_rows(result)
    assert len(rows) == 2
    col_names = [c.name for c in result.columns]
    assert "user_id" in col_names
    assert "display_name" in col_names


def test_select_specific_columns(db: ControllerDB) -> None:
    query = _make_query(
        from_table=query_pb2.QueryTable(name="jobs"),
        columns=[
            query_pb2.QueryColumn(name="job_id"),
            query_pb2.QueryColumn(name="state"),
        ],
    )
    result = execute_query(db, query, is_admin=False)
    col_names = [c.name for c in result.columns]
    assert col_names == ["job_id", "state"]
    rows = _parse_rows(result)
    assert len(rows) == 3


# ---- WHERE filters ----


def test_where_comparison_eq(db: ControllerDB) -> None:
    query = _make_query(
        from_table=query_pb2.QueryTable(name="jobs"),
        columns=[query_pb2.QueryColumn(name="job_id")],
        where=query_pb2.QueryFilter(
            comparison=query_pb2.ComparisonFilter(
                column="state",
                op=query_pb2.CMP_EQ,
                value=query_pb2.QueryValue(int_value=4),
            )
        ),
    )
    result = execute_query(db, query, is_admin=False)
    rows = _parse_rows(result)
    assert len(rows) == 1
    assert rows[0][0] == "/user/job-1"


def test_where_comparison_gt(db: ControllerDB) -> None:
    query = _make_query(
        from_table=query_pb2.QueryTable(name="jobs"),
        columns=[query_pb2.QueryColumn(name="job_id")],
        where=query_pb2.QueryFilter(
            comparison=query_pb2.ComparisonFilter(
                column="submitted_at_ms",
                op=query_pb2.CMP_GT,
                value=query_pb2.QueryValue(int_value=1050),
            )
        ),
    )
    result = execute_query(db, query, is_admin=False)
    rows = _parse_rows(result)
    assert len(rows) == 2


def test_where_logical_and(db: ControllerDB) -> None:
    query = _make_query(
        from_table=query_pb2.QueryTable(name="jobs"),
        columns=[query_pb2.QueryColumn(name="job_id")],
        where=query_pb2.QueryFilter(
            logical=query_pb2.LogicalFilter(
                op=query_pb2.LOGICAL_AND,
                operands=[
                    query_pb2.QueryFilter(
                        comparison=query_pb2.ComparisonFilter(
                            column="user_id",
                            op=query_pb2.CMP_EQ,
                            value=query_pb2.QueryValue(string_value="alice"),
                        )
                    ),
                    query_pb2.QueryFilter(
                        comparison=query_pb2.ComparisonFilter(
                            column="state",
                            op=query_pb2.CMP_EQ,
                            value=query_pb2.QueryValue(int_value=3),
                        )
                    ),
                ],
            )
        ),
    )
    result = execute_query(db, query, is_admin=False)
    rows = _parse_rows(result)
    assert len(rows) == 1
    assert rows[0][0] == "/user/job-0"


def test_where_logical_or(db: ControllerDB) -> None:
    query = _make_query(
        from_table=query_pb2.QueryTable(name="jobs"),
        columns=[query_pb2.QueryColumn(name="job_id")],
        where=query_pb2.QueryFilter(
            logical=query_pb2.LogicalFilter(
                op=query_pb2.LOGICAL_OR,
                operands=[
                    query_pb2.QueryFilter(
                        comparison=query_pb2.ComparisonFilter(
                            column="state",
                            op=query_pb2.CMP_EQ,
                            value=query_pb2.QueryValue(int_value=4),
                        )
                    ),
                    query_pb2.QueryFilter(
                        comparison=query_pb2.ComparisonFilter(
                            column="user_id",
                            op=query_pb2.CMP_EQ,
                            value=query_pb2.QueryValue(string_value="bob"),
                        )
                    ),
                ],
            )
        ),
    )
    result = execute_query(db, query, is_admin=False)
    rows = _parse_rows(result)
    assert len(rows) == 2


def test_where_in_filter(db: ControllerDB) -> None:
    query = _make_query(
        from_table=query_pb2.QueryTable(name="jobs"),
        columns=[query_pb2.QueryColumn(name="job_id")],
        where=_make_filter(
            in_=query_pb2.InFilter(
                column="job_id",
                values=[
                    query_pb2.QueryValue(string_value="/user/job-0"),
                    query_pb2.QueryValue(string_value="/user/job-2"),
                ],
            )
        ),
    )
    result = execute_query(db, query, is_admin=False)
    rows = _parse_rows(result)
    assert len(rows) == 2


def test_where_like_filter(db: ControllerDB) -> None:
    query = _make_query(
        from_table=query_pb2.QueryTable(name="endpoints"),
        columns=[query_pb2.QueryColumn(name="name")],
        where=query_pb2.QueryFilter(
            like=query_pb2.LikeFilter(
                column="name",
                pattern="svc/%",
            )
        ),
    )
    result = execute_query(db, query, is_admin=False)
    rows = _parse_rows(result)
    assert len(rows) == 1
    assert rows[0][0] == "svc/model-a"


def test_where_is_null(db: ControllerDB) -> None:
    query = _make_query(
        from_table=query_pb2.QueryTable(name="jobs"),
        columns=[query_pb2.QueryColumn(name="job_id")],
        where=query_pb2.QueryFilter(
            null_check=query_pb2.NullCheckFilter(
                column="finished_at_ms",
                op=query_pb2.NULL_IS_NULL,
            )
        ),
    )
    result = execute_query(db, query, is_admin=False)
    rows = _parse_rows(result)
    # job-0 (alice, RUNNING) and job-2 (bob, RUNNING) have no finished_at_ms
    assert len(rows) == 2


def test_where_is_not_null(db: ControllerDB) -> None:
    query = _make_query(
        from_table=query_pb2.QueryTable(name="jobs"),
        columns=[query_pb2.QueryColumn(name="job_id")],
        where=query_pb2.QueryFilter(
            null_check=query_pb2.NullCheckFilter(
                column="finished_at_ms",
                op=query_pb2.NULL_IS_NOT_NULL,
            )
        ),
    )
    result = execute_query(db, query, is_admin=False)
    rows = _parse_rows(result)
    assert len(rows) == 1


def test_where_between(db: ControllerDB) -> None:
    query = _make_query(
        from_table=query_pb2.QueryTable(name="jobs"),
        columns=[query_pb2.QueryColumn(name="job_id")],
        where=query_pb2.QueryFilter(
            between=query_pb2.BetweenFilter(
                column="submitted_at_ms",
                low=query_pb2.QueryValue(int_value=1000),
                high=query_pb2.QueryValue(int_value=1100),
            )
        ),
    )
    result = execute_query(db, query, is_admin=False)
    rows = _parse_rows(result)
    assert len(rows) == 2  # job-0 at 1000, job-1 at 1100


# ---- JOINs ----


def test_join_tasks_to_jobs(db: ControllerDB) -> None:
    query = _make_query(
        from_table=query_pb2.QueryTable(name="tasks", alias="t"),
        columns=[
            query_pb2.QueryColumn(name="task_id", table="t"),
            query_pb2.QueryColumn(name="user_id", table="j"),
        ],
        joins=[
            query_pb2.QueryJoin(
                table=query_pb2.QueryTable(name="jobs", alias="j"),
                kind=query_pb2.JOIN_INNER,
                left_column="job_id",
                left_table="t",
                right_column="job_id",
                right_table="j",
            )
        ],
        where=query_pb2.QueryFilter(
            comparison=query_pb2.ComparisonFilter(
                column="user_id",
                table="j",
                op=query_pb2.CMP_EQ,
                value=query_pb2.QueryValue(string_value="alice"),
            )
        ),
    )
    result = execute_query(db, query, is_admin=False)
    rows = _parse_rows(result)
    assert len(rows) == 4  # alice has 2 jobs * 2 tasks
    for row in rows:
        assert row[1] == "alice"


# ---- GROUP BY with aggregates ----


def test_group_by_with_count(db: ControllerDB) -> None:
    query = _make_query(
        from_table=query_pb2.QueryTable(name="jobs"),
        columns=[
            query_pb2.QueryColumn(name="user_id"),
            query_pb2.QueryColumn(name="job_id", func=query_pb2.AGG_COUNT, alias="job_count"),
        ],
        group_by=query_pb2.QueryGroupBy(columns=[query_pb2.QueryColumn(name="user_id")]),
    )
    result = execute_query(db, query, is_admin=False)
    rows = _parse_rows(result)
    col_names = [c.name for c in result.columns]
    assert "job_count" in col_names
    counts = {r[0]: r[1] for r in rows}
    assert counts["alice"] == 2
    assert counts["bob"] == 1


def test_count_star(db: ControllerDB) -> None:
    query = _make_query(
        from_table=query_pb2.QueryTable(name="jobs"),
        columns=[
            query_pb2.QueryColumn(func=query_pb2.AGG_COUNT_STAR, alias="total"),
        ],
    )
    result = execute_query(db, query, is_admin=False)
    rows = _parse_rows(result)
    assert rows[0][0] == 3


# ---- ORDER BY ----


def test_order_by_desc(db: ControllerDB) -> None:
    query = _make_query(
        from_table=query_pb2.QueryTable(name="jobs"),
        columns=[query_pb2.QueryColumn(name="job_id")],
        order_by=[query_pb2.QueryOrderBy(column="submitted_at_ms", direction=query_pb2.SORT_DESC)],
    )
    result = execute_query(db, query, is_admin=False)
    rows = _parse_rows(result)
    assert rows[0][0] == "/user/job-2"
    assert rows[-1][0] == "/user/job-0"


# ---- LIMIT/OFFSET ----


def test_limit(db: ControllerDB) -> None:
    query = _make_query(
        from_table=query_pb2.QueryTable(name="jobs"),
        columns=[query_pb2.QueryColumn(name="job_id")],
        limit=2,
    )
    result = execute_query(db, query, is_admin=False)
    rows = _parse_rows(result)
    assert len(rows) == 2
    # total_count reflects all matching rows (first page)
    assert result.total_count == 3


def test_offset(db: ControllerDB) -> None:
    query = _make_query(
        from_table=query_pb2.QueryTable(name="jobs"),
        columns=[query_pb2.QueryColumn(name="job_id")],
        order_by=[query_pb2.QueryOrderBy(column="submitted_at_ms")],
        limit=2,
        offset=1,
    )
    result = execute_query(db, query, is_admin=False)
    rows = _parse_rows(result)
    assert len(rows) == 2
    assert rows[0][0] == "/user/job-1"


def test_default_limit_applied(db: ControllerDB) -> None:
    """When no limit is specified, DEFAULT_LIMIT is used."""
    query = _make_query(
        from_table=query_pb2.QueryTable(name="jobs"),
        columns=[query_pb2.QueryColumn(name="job_id")],
    )
    result = execute_query(db, query, is_admin=False)
    # We have only 3 jobs, so all are returned, but the limit was applied.
    assert len(result.rows) == 3
    assert DEFAULT_LIMIT == 100  # sanity check default


def test_max_limit_cap(db: ControllerDB) -> None:
    """Requesting a limit above MAX_LIMIT is capped."""
    query = _make_query(
        from_table=query_pb2.QueryTable(name="jobs"),
        columns=[query_pb2.QueryColumn(name="job_id")],
        limit=5000,
    )
    result = execute_query(db, query, is_admin=False)
    assert len(result.rows) == 3
    assert MAX_LIMIT == 1000


# ---- Total count pagination ----


def test_total_count_on_first_page(db: ControllerDB) -> None:
    query = _make_query(
        from_table=query_pb2.QueryTable(name="jobs"),
        columns=[query_pb2.QueryColumn(name="job_id")],
        limit=1,
        offset=0,
    )
    result = execute_query(db, query, is_admin=False)
    assert len(result.rows) == 1
    assert result.total_count == 3


def test_total_count_computed_on_subsequent_page(db: ControllerDB) -> None:
    """total_count is always computed via COUNT(*) when limit > 0, even with offset."""
    query = _make_query(
        from_table=query_pb2.QueryTable(name="jobs"),
        columns=[query_pb2.QueryColumn(name="job_id")],
        limit=1,
        offset=1,
    )
    result = execute_query(db, query, is_admin=False)
    assert len(result.rows) == 1
    assert result.total_count == 3


# ---- Access control: denylist-based ----


def test_sensitive_table_blocked_for_non_admin(db: ControllerDB) -> None:
    """Non-admin users cannot query sensitive tables."""
    query = _make_query(from_table=query_pb2.QueryTable(name="api_keys"))
    with pytest.raises(ValueError, match="not accessible"):
        execute_query(db, query, is_admin=False)


def test_sensitive_table_controller_secrets_blocked(db: ControllerDB) -> None:
    """controller_secrets is blocked for non-admins."""
    query = _make_query(from_table=query_pb2.QueryTable(name="controller_secrets"))
    with pytest.raises(ValueError, match="not accessible"):
        execute_query(db, query, is_admin=False)


def test_sensitive_table_allowed_for_admin(db: ControllerDB) -> None:
    """Admin users can query sensitive tables."""
    query = _make_query(
        from_table=query_pb2.QueryTable(name="api_keys"),
        columns=[query_pb2.QueryColumn(name="key_id")],
    )
    result = execute_query(db, query, is_admin=True)
    assert result.columns[0].name == "key_id"


def test_blocked_column_key_hash(db: ControllerDB) -> None:
    """key_hash is permanently blocked, even for admins."""
    query = _make_query(
        from_table=query_pb2.QueryTable(name="api_keys"),
        columns=[query_pb2.QueryColumn(name="key_hash")],
    )
    with pytest.raises(ValueError, match="not accessible"):
        execute_query(db, query, is_admin=True)


def test_non_sensitive_table_accessible(db: ControllerDB) -> None:
    """Non-sensitive tables like scaling_groups are accessible to all users."""
    query = _make_query(
        from_table=query_pb2.QueryTable(name="scaling_groups"),
        columns=[query_pb2.QueryColumn(name="name")],
    )
    # Should not raise — scaling_groups is not in SENSITIVE_TABLES.
    result = execute_query(db, query, is_admin=False)
    assert result.columns[0].name == "name"


def test_any_column_accessible_on_non_sensitive_table(db: ControllerDB) -> None:
    """Any column (including request_proto) is accessible when the table is not sensitive."""
    query = _make_query(
        from_table=query_pb2.QueryTable(name="jobs"),
        columns=[query_pb2.QueryColumn(name="request_proto")],
    )
    result = execute_query(db, query, is_admin=False)
    assert result.columns[0].name == "request_proto"


def test_missing_from_table(db: ControllerDB) -> None:
    query = _make_query()
    with pytest.raises(ValueError, match="FROM table"):
        execute_query(db, query, is_admin=False)


def test_too_many_joins(db: ControllerDB) -> None:
    joins = [
        query_pb2.QueryJoin(
            table=query_pb2.QueryTable(name="tasks", alias=f"t{i}"),
            left_column="job_id",
            right_column="job_id",
        )
        for i in range(MAX_JOINS + 1)
    ]
    query = _make_query(
        from_table=query_pb2.QueryTable(name="jobs"),
        joins=joins,
    )
    with pytest.raises(ValueError, match="Maximum"):
        execute_query(db, query, is_admin=False)


# ---- Raw query ----


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


def test_raw_query_reject_multi_statement(db: ControllerDB) -> None:
    with pytest.raises(ValueError, match="Multiple SQL"):
        execute_raw_query(db, "SELECT 1; SELECT 2")


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


# ---- NOT filter ----


def test_where_not_filter(db: ControllerDB) -> None:
    query = _make_query(
        from_table=query_pb2.QueryTable(name="jobs"),
        columns=[query_pb2.QueryColumn(name="job_id")],
        where=_make_filter(
            not_=query_pb2.NotFilter(
                operand=query_pb2.QueryFilter(
                    comparison=query_pb2.ComparisonFilter(
                        column="state",
                        op=query_pb2.CMP_EQ,
                        value=query_pb2.QueryValue(int_value=3),
                    )
                )
            )
        ),
    )
    result = execute_query(db, query, is_admin=False)
    rows = _parse_rows(result)
    # Only job-1 (SUCCEEDED, state=4) should remain
    assert len(rows) == 1
    assert rows[0][0] == "/user/job-1"


# ---- Identifier injection ----


def test_table_alias_injection_rejected(db: ControllerDB) -> None:
    query = _make_query(
        from_table=query_pb2.QueryTable(name="jobs", alias="j; DROP TABLE jobs--"),
    )
    with pytest.raises(ValueError, match="Invalid table alias"):
        execute_query(db, query, is_admin=False)


def test_column_alias_injection_rejected(db: ControllerDB) -> None:
    query = _make_query(
        from_table=query_pb2.QueryTable(name="jobs"),
        columns=[
            query_pb2.QueryColumn(name="job_id", alias="x; DROP TABLE jobs--"),
        ],
    )
    with pytest.raises(ValueError, match="Invalid column alias"):
        execute_query(db, query, is_admin=False)


def test_join_alias_injection_rejected(db: ControllerDB) -> None:
    query = _make_query(
        from_table=query_pb2.QueryTable(name="jobs"),
        joins=[
            query_pb2.QueryJoin(
                table=query_pb2.QueryTable(name="tasks", alias="t OR 1=1"),
                left_column="job_id",
                right_column="job_id",
            )
        ],
    )
    with pytest.raises(ValueError, match="Invalid table alias"):
        execute_query(db, query, is_admin=False)


# ---- AGG_COUNT_STAR with GROUP BY ----


def test_count_star_with_group_by(db: ControllerDB) -> None:
    query = _make_query(
        from_table=query_pb2.QueryTable(name="jobs"),
        columns=[
            query_pb2.QueryColumn(name="user_id"),
            query_pb2.QueryColumn(func=query_pb2.AGG_COUNT_STAR, alias="cnt"),
        ],
        group_by=query_pb2.QueryGroupBy(columns=[query_pb2.QueryColumn(name="user_id")]),
        order_by=[query_pb2.QueryOrderBy(column="user_id")],
    )
    result = execute_query(db, query, is_admin=False)
    rows = _parse_rows(result)
    counts = {r[0]: r[1] for r in rows}
    assert counts["alice"] == 2
    assert counts["bob"] == 1


# ---- Log store queries ----


def test_log_store_select_all(db: ControllerDB, log_store: LogStore) -> None:
    """Query all columns from the logs table."""
    query = _make_query(from_table=query_pb2.QueryTable(name="logs"))
    result = execute_query(db, query, is_admin=False, log_store=log_store, database="logs")
    rows = _parse_rows(result)
    assert len(rows) == 5
    col_names = [c.name for c in result.columns]
    assert "key" in col_names
    assert "data" in col_names


def test_log_store_with_filter(db: ControllerDB, log_store: LogStore) -> None:
    """Query logs with a WHERE filter."""
    query = _make_query(
        from_table=query_pb2.QueryTable(name="logs"),
        columns=[query_pb2.QueryColumn(name="data")],
        where=query_pb2.QueryFilter(
            comparison=query_pb2.ComparisonFilter(
                column="epoch_ms",
                op=query_pb2.CMP_GT,
                value=query_pb2.QueryValue(int_value=1200),
            )
        ),
    )
    result = execute_query(db, query, is_admin=False, log_store=log_store, database="logs")
    rows = _parse_rows(result)
    assert len(rows) == 2  # epoch_ms 1300, 1400


def test_log_store_count(db: ControllerDB, log_store: LogStore) -> None:
    """Aggregate query on log store."""
    query = _make_query(
        from_table=query_pb2.QueryTable(name="logs"),
        columns=[query_pb2.QueryColumn(func=query_pb2.AGG_COUNT_STAR, alias="total")],
    )
    result = execute_query(db, query, is_admin=False, log_store=log_store, database="logs")
    rows = _parse_rows(result)
    assert rows[0][0] == 5


def test_log_store_not_available(db: ControllerDB) -> None:
    """Querying logs without a log_store raises ValueError."""
    query = _make_query(from_table=query_pb2.QueryTable(name="logs"))
    with pytest.raises(ValueError, match="Log store not available"):
        execute_query(db, query, is_admin=False, database="logs")


def test_log_store_rejects_main_table(db: ControllerDB, log_store: LogStore) -> None:
    """Querying a main DB table via database='logs' is rejected."""
    query = _make_query(from_table=query_pb2.QueryTable(name="jobs"))
    with pytest.raises(ValueError, match="does not exist in the logs database"):
        execute_query(db, query, is_admin=False, log_store=log_store, database="logs")


def test_log_store_rejects_cross_db_join(db: ControllerDB, log_store: LogStore) -> None:
    """Joining a main DB table from the logs database is rejected."""
    query = _make_query(
        from_table=query_pb2.QueryTable(name="logs"),
        joins=[
            query_pb2.QueryJoin(
                table=query_pb2.QueryTable(name="jobs", alias="j"),
                left_column="key",
                right_column="job_id",
            )
        ],
    )
    with pytest.raises(ValueError, match="does not exist in the logs database"):
        execute_query(db, query, is_admin=False, log_store=log_store, database="logs")
