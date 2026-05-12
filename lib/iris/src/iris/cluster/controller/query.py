# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Raw SQL query executor for the generic query API.

Validates and executes raw SELECT statements against the controller's
SQLite database via snapshot isolation.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from sqlalchemy import text

from iris.cluster.controller.db import ControllerDB
from iris.rpc import query_pb2

# Keywords that must never appear in raw SQL queries.
_FORBIDDEN_SQL_KEYWORDS = (
    "INSERT",
    "UPDATE",
    "DELETE",
    "DROP",
    "ALTER",
    "CREATE",
    "ATTACH",
    "DETACH",
    "PRAGMA",
    "VACUUM",
    "REINDEX",
    "SAVEPOINT",
)


@dataclass(frozen=True)
class QueryResult:
    columns: list[query_pb2.ColumnMeta]
    rows: list[str]  # JSON-encoded arrays


def execute_raw_query(
    db: ControllerDB,
    sql: str,
) -> QueryResult:
    """Execute a raw SQL query (admin-only). Only SELECT statements allowed."""
    stripped = sql.strip()
    if not stripped.upper().startswith("SELECT"):
        raise ValueError("Only SELECT statements are allowed")

    upper = stripped.upper()
    for keyword in _FORBIDDEN_SQL_KEYWORDS:
        if re.search(rf"\b{keyword}\b", upper):
            raise ValueError(f"Forbidden SQL keyword: {keyword}")

    with db.read_snapshot() as tx:
        result = tx.execute(text(stripped))
        col_names = list(result.keys())
        raw_rows = result.all()

    return _build_result(col_names, raw_rows)


def _build_result(
    col_names: list[str],
    raw_rows: list,
) -> QueryResult:
    """Encode cursor results into a QueryResult with JSON-serialized rows."""
    columns = [query_pb2.ColumnMeta(name=name, type="unknown") for name in col_names]
    rows = [json.dumps([_encode_cell(row[i]) for i in range(len(columns))]) for row in raw_rows]
    return QueryResult(columns=columns, rows=rows)


def _encode_cell(value: object) -> object:
    """Encode a SQLite cell value for JSON serialization."""
    if value is None:
        return None
    if isinstance(value, bytes):
        return f"<blob:{len(value)} bytes>"
    return value
