# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Structured query executor for the generic query API.

Validates and converts proto Query messages into parameterized SQL,
executing them against the controller's SQLite database (or the log store)
via snapshot isolation. Uses a denylist to block sensitive tables from
non-admin users while allowing unrestricted access to all other tables.
"""

from __future__ import annotations

import json
import re
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from collections.abc import Iterator

from iris.cluster.controller.db import ControllerDB
from iris.cluster.log_store import LogStore
from iris.rpc import query_pb2

MAX_LIMIT = 1000
DEFAULT_LIMIT = 100
MAX_JOINS = 3

_IDENTIFIER_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")

# Tables containing secrets — only admin users may query these.
SENSITIVE_TABLES = frozenset(
    {
        "controller_secrets",
        "api_keys",
    }
)

# Columns permanently blocked from the query API, even for admins.
BLOCKED_COLUMNS: dict[str, frozenset[str]] = {
    "api_keys": frozenset({"key_hash"}),
    "controller_secrets": frozenset({"value"}),
}

# The log store DB only has one table.
LOG_STORE_TABLES = frozenset({"logs"})


def _validate_identifier(name: str, label: str = "identifier") -> str:
    """Reject identifiers that could allow SQL injection when interpolated into queries."""
    if not _IDENTIFIER_RE.match(name):
        raise ValueError(f"Invalid {label}: {name!r}")
    return name


def _check_table_access(table_name: str, is_admin: bool, database: str) -> None:
    """Raise if the table is not accessible for the given role and database target."""
    _validate_identifier(table_name, "table name")
    if database == "logs":
        if table_name not in LOG_STORE_TABLES:
            raise ValueError(f"Table {table_name!r} does not exist in the logs database")
        return
    if table_name in SENSITIVE_TABLES and not is_admin:
        raise ValueError(f"Table {table_name!r} is not accessible")


def _check_column_access(column: str, table_name: str) -> None:
    """Raise if the column is permanently blocked (e.g. key_hash)."""
    _validate_identifier(column, "column")
    blocked = BLOCKED_COLUMNS.get(table_name, frozenset())
    if column in blocked:
        raise ValueError(f"Column {column!r} not accessible on table {table_name!r}")


def _resolve_column(
    column: str,
    table_alias: str,
    alias_to_table: dict[str, str],
) -> str:
    """Validate and return qualified column reference."""
    _validate_identifier(column, "column")
    table_name = alias_to_table.get(table_alias)
    if table_name is None:
        raise ValueError(f"Unknown table alias: {table_alias!r}")
    _check_column_access(column, table_name)
    return f"{table_alias}.{column}"


_CMP_OPS = {
    query_pb2.CMP_EQ: "=",
    query_pb2.CMP_NE: "!=",
    query_pb2.CMP_LT: "<",
    query_pb2.CMP_LE: "<=",
    query_pb2.CMP_GT: ">",
    query_pb2.CMP_GE: ">=",
}

_AGG_FUNCS = {
    query_pb2.AGG_COUNT: "COUNT",
    query_pb2.AGG_SUM: "SUM",
    query_pb2.AGG_AVG: "AVG",
    query_pb2.AGG_MIN: "MIN",
    query_pb2.AGG_MAX: "MAX",
    query_pb2.AGG_COUNT_STAR: "COUNT",
}

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
    total_count: int


def _encode_value(v: query_pb2.QueryValue) -> object:
    """Extract the scalar value from a QueryValue proto."""
    which = v.WhichOneof("value")
    if which == "string_value":
        return v.string_value
    if which == "int_value":
        return v.int_value
    if which == "float_value":
        return v.float_value
    if which == "bool_value":
        return int(v.bool_value)
    raise ValueError("QueryValue has no value set")


def _compile_filter(
    f: query_pb2.QueryFilter,
    alias_to_table: dict[str, str],
    default_alias: str,
) -> tuple[str, list[object]]:
    """Compile a QueryFilter tree into (sql, params)."""
    which = f.WhichOneof("filter")

    if which == "comparison":
        c = f.comparison
        alias = c.table or default_alias
        col = _resolve_column(c.column, alias, alias_to_table)
        op = _CMP_OPS.get(c.op)
        if op is None:
            raise ValueError(f"Unknown comparison op: {c.op}")
        return f"{col} {op} ?", [_encode_value(c.value)]

    if which == "logical":
        lg = f.logical
        if not lg.operands:
            raise ValueError("Logical filter requires at least one operand")
        op_str = "AND" if lg.op == query_pb2.LOGICAL_AND else "OR"
        parts: list[str] = []
        params: list[object] = []
        for operand in lg.operands:
            s, p = _compile_filter(operand, alias_to_table, default_alias)
            parts.append(f"({s})")
            params.extend(p)
        return f" {op_str} ".join(parts), params

    if which == "not":
        not_filter: query_pb2.NotFilter = getattr(f, "not")
        s, p = _compile_filter(not_filter.operand, alias_to_table, default_alias)
        return f"NOT ({s})", p

    if which == "in":
        inf: query_pb2.InFilter = getattr(f, "in")
        alias = inf.table or default_alias
        col = _resolve_column(inf.column, alias, alias_to_table)
        if not inf.values:
            return "0", []  # Empty IN is always false
        placeholders = ", ".join("?" for _ in inf.values)
        return f"{col} IN ({placeholders})", [_encode_value(v) for v in inf.values]

    if which == "like":
        lf = f.like
        alias = lf.table or default_alias
        col = _resolve_column(lf.column, alias, alias_to_table)
        return f"{col} LIKE ?", [lf.pattern]

    if which == "null_check":
        nc = f.null_check
        alias = nc.table or default_alias
        col = _resolve_column(nc.column, alias, alias_to_table)
        op = "IS NULL" if nc.op == query_pb2.NULL_IS_NULL else "IS NOT NULL"
        return f"{col} {op}", []

    if which == "between":
        bf = f.between
        alias = bf.table or default_alias
        col = _resolve_column(bf.column, alias, alias_to_table)
        return f"{col} BETWEEN ? AND ?", [
            _encode_value(bf.low),
            _encode_value(bf.high),
        ]

    raise ValueError(f"Unknown filter type: {which}")


@contextmanager
def _log_store_snapshot(log_store: LogStore) -> Iterator[sqlite3.Connection]:
    """Provide a read-only snapshot context for log store queries."""
    with log_store._read_lock:
        log_store._read_conn.execute("BEGIN")
        try:
            yield log_store._read_conn
        finally:
            log_store._read_conn.rollback()


def _introspect_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    """Get column names for a table via PRAGMA table_info."""
    cursor = conn.execute(f"PRAGMA table_info({table_name})")
    return {row[1] for row in cursor.fetchall()}


def execute_query(
    db: ControllerDB,
    query: query_pb2.Query,
    is_admin: bool,
    *,
    log_store: LogStore | None = None,
    database: str = "main",
) -> QueryResult:
    """Validate and execute a structured query against the controller DB or log store."""
    if database == "logs":
        if log_store is None:
            raise ValueError("Log store not available")
        return _execute_on_log_store(log_store, query, is_admin)
    return _execute_on_main(db, query, is_admin)


def _execute_on_main(
    db: ControllerDB,
    query: query_pb2.Query,
    is_admin: bool,
) -> QueryResult:
    """Execute a structured query against the main controller DB."""
    from_table, primary_alias, alias_to_table, join_clauses = _build_from_joins(query, is_admin, "main")
    # For SELECT *, introspect columns from the DB.
    select_cols: set[str] | None = None
    if not query.columns:
        with db.snapshot() as q:
            cursor = q.execute_sql(f"PRAGMA table_info({from_table.name})")
            all_cols = {row[1] for row in cursor.fetchall()}
        blocked = BLOCKED_COLUMNS.get(from_table.name, frozenset())
        select_cols = all_cols - blocked

    select_parts = _build_select(query, primary_alias, alias_to_table, select_cols)
    where_sql, params = _build_where(query, alias_to_table, primary_alias)
    group_sql = _build_group_by(query, alias_to_table, primary_alias)
    order_sql = _build_order_by(query, alias_to_table, primary_alias)
    limit, limit_sql = _build_limit(query)

    sql, base_sql = _assemble_sql(
        select_parts,
        from_table.name,
        primary_alias,
        join_clauses,
        where_sql,
        group_sql,
        order_sql,
        limit_sql,
    )

    with db.snapshot() as q:
        cursor = q.execute_sql(sql, tuple(params))
        col_descriptions = cursor.description
        raw_rows = cursor.fetchall()
        total_count = len(raw_rows)
        if limit > 0:
            count_cursor = q.execute_sql(f"SELECT COUNT(*) {base_sql}", tuple(params))
            total_count = count_cursor.fetchone()[0]

    return _build_result(col_descriptions, raw_rows, total_count)


def _execute_on_log_store(
    log_store: LogStore,
    query: query_pb2.Query,
    is_admin: bool,
) -> QueryResult:
    """Execute a structured query against the log store DB."""
    from_table, primary_alias, alias_to_table, join_clauses = _build_from_joins(query, is_admin, "logs")

    select_cols: set[str] | None = None
    if not query.columns:
        with log_store._read_lock:
            all_cols = _introspect_columns(log_store._read_conn, from_table.name)
        select_cols = all_cols

    select_parts = _build_select(query, primary_alias, alias_to_table, select_cols)
    where_sql, params = _build_where(query, alias_to_table, primary_alias)
    group_sql = _build_group_by(query, alias_to_table, primary_alias)
    order_sql = _build_order_by(query, alias_to_table, primary_alias)
    limit, limit_sql = _build_limit(query)

    sql, base_sql = _assemble_sql(
        select_parts,
        from_table.name,
        primary_alias,
        join_clauses,
        where_sql,
        group_sql,
        order_sql,
        limit_sql,
    )

    with _log_store_snapshot(log_store) as conn:
        cursor = conn.execute(sql, tuple(params))
        col_descriptions = cursor.description
        raw_rows = cursor.fetchall()
        total_count = len(raw_rows)
        if limit > 0:
            count_cursor = conn.execute(f"SELECT COUNT(*) {base_sql}", tuple(params))
            total_count = count_cursor.fetchone()[0]

    return _build_result(col_descriptions, raw_rows, total_count)


def _build_from_joins(
    query: query_pb2.Query,
    is_admin: bool,
    database: str,
) -> tuple[query_pb2.QueryTable, str, dict[str, str], list[str]]:
    """Validate FROM and JOINs. Returns (from_table, primary_alias, alias_to_table, join_clauses)."""
    from_table: query_pb2.QueryTable = getattr(query, "from")
    if not from_table.name:
        raise ValueError("Query must specify a FROM table")
    _check_table_access(from_table.name, is_admin, database)
    primary_alias = from_table.alias or from_table.name
    _validate_identifier(primary_alias, "table alias")
    alias_to_table: dict[str, str] = {primary_alias: from_table.name}

    if len(query.joins) > MAX_JOINS:
        raise ValueError(f"Maximum {MAX_JOINS} joins allowed")
    join_clauses: list[str] = []
    for j in query.joins:
        if not j.table.name:
            raise ValueError("Join table name required")
        _check_table_access(j.table.name, is_admin, database)
        j_alias = j.table.alias or j.table.name
        _validate_identifier(j_alias, "table alias")
        if j_alias in alias_to_table:
            raise ValueError(f"Duplicate table alias: {j_alias!r}")
        alias_to_table[j_alias] = j.table.name
        kind = "LEFT JOIN" if j.kind == query_pb2.JOIN_LEFT else "JOIN"
        l_alias = j.left_table or primary_alias
        r_alias = j.right_table or j_alias
        l_col = _resolve_column(j.left_column, l_alias, alias_to_table)
        r_col = _resolve_column(j.right_column, r_alias, alias_to_table)
        join_clauses.append(f"{kind} {j.table.name} {j_alias} ON {l_col} = {r_col}")

    return from_table, primary_alias, alias_to_table, join_clauses


def _build_select(
    query: query_pb2.Query,
    primary_alias: str,
    alias_to_table: dict[str, str],
    introspected_cols: set[str] | None,
) -> list[str]:
    """Build SELECT column list."""
    select_parts: list[str] = []
    if not query.columns:
        if introspected_cols is None:
            raise ValueError("No columns available for SELECT *")
        for col in sorted(introspected_cols):
            select_parts.append(f"{primary_alias}.{col}")
    else:
        for qc in query.columns:
            alias = qc.table or primary_alias
            if qc.func == query_pb2.AGG_COUNT_STAR:
                expr = "COUNT(*)"
            elif qc.func and qc.func != query_pb2.AGG_NONE:
                col = _resolve_column(qc.name, alias, alias_to_table)
                func_name = _AGG_FUNCS[qc.func]
                expr = f"{func_name}({col})"
            else:
                col = _resolve_column(qc.name, alias, alias_to_table)
                expr = col
            label = qc.alias or qc.name
            _validate_identifier(label, "column alias")
            select_parts.append(f"{expr} AS {label}")
    return select_parts


def _build_where(
    query: query_pb2.Query,
    alias_to_table: dict[str, str],
    primary_alias: str,
) -> tuple[str, list[object]]:
    params: list[object] = []
    where_sql = ""
    if query.HasField("where"):
        w_sql, w_params = _compile_filter(query.where, alias_to_table, primary_alias)
        where_sql = f" WHERE {w_sql}"
        params.extend(w_params)
    return where_sql, params


def _build_group_by(
    query: query_pb2.Query,
    alias_to_table: dict[str, str],
    primary_alias: str,
) -> str:
    if not query.HasField("group_by") or not query.group_by.columns:
        return ""
    group_cols = []
    for gc in query.group_by.columns:
        alias = gc.table or primary_alias
        col = _resolve_column(gc.name, alias, alias_to_table)
        group_cols.append(col)
    return f" GROUP BY {', '.join(group_cols)}"


def _build_order_by(
    query: query_pb2.Query,
    alias_to_table: dict[str, str],
    primary_alias: str,
) -> str:
    if not query.order_by:
        return ""
    order_parts = []
    for ob in query.order_by:
        alias = ob.table or primary_alias
        col = _resolve_column(ob.column, alias, alias_to_table)
        direction = "DESC" if ob.direction == query_pb2.SORT_DESC else "ASC"
        order_parts.append(f"{col} {direction}")
    return f" ORDER BY {', '.join(order_parts)}"


def _build_limit(query: query_pb2.Query) -> tuple[int, str]:
    limit = min(query.limit, MAX_LIMIT) if query.limit > 0 else DEFAULT_LIMIT
    limit_sql = f" LIMIT {limit}"
    if query.offset > 0:
        limit_sql += f" OFFSET {query.offset}"
    return limit, limit_sql


def _assemble_sql(
    select_parts: list[str],
    table_name: str,
    primary_alias: str,
    join_clauses: list[str],
    where_sql: str,
    group_sql: str,
    order_sql: str,
    limit_sql: str,
) -> tuple[str, str]:
    """Assemble full SELECT and base (no LIMIT) SQL strings."""
    select_sql = ", ".join(select_parts)
    joins_sql = (" " + " ".join(join_clauses)) if join_clauses else ""
    sql = (
        f"SELECT {select_sql} FROM {table_name} {primary_alias}"
        f"{joins_sql}{where_sql}{group_sql}{order_sql}{limit_sql}"
    )
    base_sql = f"FROM {table_name} {primary_alias}{joins_sql}{where_sql}{group_sql}"
    return sql, base_sql


def execute_raw_query(
    db: ControllerDB,
    sql: str,
) -> QueryResult:
    """Execute a raw SQL query (admin-only). Only SELECT statements allowed."""
    stripped = sql.strip()
    if not stripped.upper().startswith("SELECT"):
        raise ValueError("Only SELECT statements are allowed")

    if ";" in stripped:
        raise ValueError("Multiple SQL statements are not allowed")

    upper = stripped.upper()
    for keyword in _FORBIDDEN_SQL_KEYWORDS:
        if re.search(rf"\b{keyword}\b", upper):
            raise ValueError(f"Forbidden SQL keyword: {keyword}")

    with db.snapshot() as q:
        cursor = q.execute_sql(stripped)
        col_descriptions = cursor.description
        raw_rows = cursor.fetchall()

    return _build_result(col_descriptions, raw_rows, len(raw_rows))


def _build_result(
    col_descriptions: list[tuple],
    raw_rows: list[tuple],
    total_count: int,
) -> QueryResult:
    """Encode cursor results into a QueryResult with JSON-serialized rows."""
    columns = [query_pb2.ColumnMeta(name=desc[0], type="unknown") for desc in col_descriptions]
    rows = [json.dumps([_encode_cell(row[i]) for i in range(len(columns))]) for row in raw_rows]
    return QueryResult(columns=columns, rows=rows, total_count=total_count)


def _encode_cell(value: object) -> object:
    """Encode a SQLite cell value for JSON serialization."""
    if value is None:
        return None
    if isinstance(value, bytes):
        return f"<blob:{len(value)} bytes>"
    return value
