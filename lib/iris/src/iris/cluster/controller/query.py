# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Raw SQL query executor for the controller query API.

All queries run inside snapshot isolation (BEGIN + ROLLBACK), so writes
are impossible even if the SQL contains mutation statements.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass

from iris.cluster.controller.db import ControllerDB
from iris.cluster.log_store import LogStore
from iris.rpc import query_pb2

MAX_RESULT_ROWS = 1000


@dataclass(frozen=True)
class QueryResult:
    columns: list[query_pb2.ColumnMeta]
    rows: list[str]  # JSON-encoded arrays
    truncated: bool


def execute_raw_query(
    db: ControllerDB,
    sql: str,
    *,
    log_store: LogStore | None = None,
    database: str = "main",
) -> QueryResult:
    """Execute a raw SQL query (admin-only). Only SELECT statements allowed.

    Runs inside a snapshot that is always rolled back, preventing any writes.
    """
    stripped = sql.strip()
    if not stripped.upper().startswith("SELECT"):
        raise ValueError("Only SELECT statements are allowed")
    if ";" in stripped:
        raise ValueError("Multiple SQL statements are not allowed")

    if database not in ("main", "logs"):
        raise ValueError(f"Unknown database: {database!r}. Must be 'main' or 'logs'.")

    if database == "logs":
        if log_store is None:
            raise ValueError("Log store not available")
        with log_store.snapshot() as conn:
            return _execute_on_conn(conn, stripped)

    with db.snapshot() as q:
        cursor = q.execute_sql(stripped)
        col_descriptions = cursor.description or []
        raw_rows = cursor.fetchmany(MAX_RESULT_ROWS + 1)

    truncated = len(raw_rows) > MAX_RESULT_ROWS
    if truncated:
        raw_rows = raw_rows[:MAX_RESULT_ROWS]

    return _build_result(col_descriptions, raw_rows, truncated=truncated)


def _execute_on_conn(conn: sqlite3.Connection, sql: str) -> QueryResult:
    """Execute SQL on a raw connection within a snapshot."""
    cursor = conn.execute(sql)
    col_descriptions = cursor.description or []
    raw_rows = cursor.fetchmany(MAX_RESULT_ROWS + 1)
    truncated = len(raw_rows) > MAX_RESULT_ROWS
    if truncated:
        raw_rows = raw_rows[:MAX_RESULT_ROWS]
    return _build_result(col_descriptions, raw_rows, truncated=truncated)


def _build_result(
    col_descriptions: list[tuple],
    raw_rows: list[tuple],
    *,
    truncated: bool,
) -> QueryResult:
    columns = [query_pb2.ColumnMeta(name=desc[0], type="unknown") for desc in col_descriptions]
    rows = [json.dumps([_encode_cell(row[i]) for i in range(len(columns))]) for row in raw_rows]
    return QueryResult(columns=columns, rows=rows, truncated=truncated)


def _encode_cell(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, bytes):
        return f"<blob:{len(value)} bytes>"
    return value
