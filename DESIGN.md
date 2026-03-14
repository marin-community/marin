# Iris Generic Query API

Design document for [#3492](https://github.com/marin-community/marin/issues/3492): adding a structured query DSL to the iris controller.

## 1. Current State / Problem

The `ControllerService` (cluster.proto:951-1003) has 15 query-oriented RPCs, each with bespoke proto messages, handler code, and pagination/filtering logic:

| RPC | Proto Line | Handler (`service.py`) | Pattern |
|-----|-----------|----------------------|---------|
| `GetJobStatus` | 954 | L657-715 | Single row + task aggregation |
| `ListJobs` | 956 | L756-907 | Paginated listing with sort/filter/state |
| `GetTaskStatus` | 959 | L911-934 | Single row + attempt join |
| `ListTasks` | 960 | L936-957 | Listing with optional job filter |
| `ListWorkers` | 964 | L996-1018 | Full scan + running task count join |
| `ListEndpoints` | 968 | L1079-1103 | Name prefix/exact filter |
| `GetAutoscalerStatus` | 971 | L1107-1140 | Autoscaler state + worker enrichment |
| `GetTransactions` | 974 | L1317-1334 | Recent actions with limit |
| `ListUsers` | 976 | L1336-1360 | GROUP BY user with state counts |
| `GetTaskLogs` | 979 | L1146-1237 | Log store query (not DB) |
| `FetchLogs` | 982 | L1362-1397 | Log store or worker proxy |
| `GetWorkerStatus` | 988 | L1401-1462 | Single worker + task history + resource history + worker proxy |
| `GetProcessStatus` | 994 | L1479-1513 | Local process info or worker proxy |
| `GetAuthInfo` | 997 | L1517-1525 | Config metadata |
| `GetCurrentUser` | 1002 | L1632-1644 | Current user identity |

Each listing RPC re-implements filtering, sorting, and pagination in Python. `ListJobs` alone is 150 lines (service.py:756-907) handling sort fields, state filters, name filters, pagination with parent-child grouping, and per-job task summary aggregation. `ListUsers` (service.py:1336-1360) does a manual GROUP BY in Python across two separate queries.

The DB layer (`db.py`) already has a rich query DSL with `Table`, `Column`, `Predicate`, `Join`, `Order`, `SelectExpr`, and `QuerySnapshot.select()`. But each RPC builds its own bespoke query and post-processing.

Adding a new read pattern (e.g., "jobs by user sorted by failure count", "tasks running longer than 1 hour") requires a new proto message pair, a new handler method, dashboard wiring, and CLI plumbing.

## 2. Goals and Non-Goals

### Goals

- **Structured protobuf query DSL** that covers SELECT/FROM/WHERE/JOIN/GROUP BY/ORDER BY/LIMIT against the controller's SQLite database.
- **Raw SQL admin bypass** for ad-hoc queries that the DSL cannot express.
- **Unified pagination** via LIMIT/OFFSET in the DSL, replacing per-RPC pagination.
- **Table-level access control**: restrict which tables each role can query.

### Non-Goals

- **Full SQL parser** or **GraphQL** — the DSL is a protobuf message tree, not a string language.
- **Write operations** — the query API is read-only. All mutations go through existing RPCs.
- **Replacing worker-proxied RPCs** — `FetchLogs` (worker mode), `ProfileTask`, `GetProcessStatus` (worker mode), and `GetWorkerStatus` (worker log/resource proxy) involve live worker I/O. These stay as dedicated RPCs.
- **Replacing `GetAutoscalerStatus`** — autoscaler state lives in the autoscaler's in-memory structures, not in the DB.
- **Replacing auth RPCs** — `GetAuthInfo`, `GetCurrentUser`, `Login`, `CreateApiKey`, `RevokeApiKey`, `ListApiKeys` are auth lifecycle operations, not generic queries.
- **Replacing log RPCs** — `GetTaskLogs` and `FetchLogs` query the in-memory `LogStore`, not the SQLite DB. The `logs` table is used for persistence but log access is through the log store API.

## 3. Query DSL (Proposed Solution)

### Proto Messages

Add to `cluster.proto`:

```protobuf
// ============================================================================
// GENERIC QUERY API
// ============================================================================

// Structured query against the controller database.
// Equivalent to: SELECT <columns> FROM <table> [JOIN ...] [WHERE ...] [GROUP BY ...] [ORDER BY ...] [LIMIT ... OFFSET ...]
message Query {
  // Columns to select. Empty = all columns from the primary table.
  repeated QueryColumn columns = 1;

  // Primary table to query.
  QueryTable from = 2;

  // Filter conditions (AND of all provided).
  QueryFilter where = 3;

  // Joins against other tables.
  repeated QueryJoin joins = 4;

  // Group-by columns with optional aggregate functions.
  QueryGroupBy group_by = 5;

  // Sort order.
  repeated QueryOrderBy order_by = 6;

  // Pagination.
  int32 limit = 7;   // Max rows (0 = server default, capped at 1000)
  int32 offset = 8;  // Skip first N rows
}

// A column reference, optionally with an aggregate function.
message QueryColumn {
  string name = 1;          // Column name (e.g., "job_id", "state")
  string table = 2;         // Table alias (empty = primary table)
  AggregateFunc func = 3;   // Aggregate function (NONE for plain column)
  string alias = 4;         // Output alias (e.g., "job_count")
}

enum AggregateFunc {
  AGG_NONE = 0;
  AGG_COUNT = 1;
  AGG_SUM = 2;
  AGG_AVG = 3;
  AGG_MIN = 4;
  AGG_MAX = 5;
  AGG_COUNT_STAR = 6;  // COUNT(*) — ignores column name
}

// Table reference.
message QueryTable {
  string name = 1;   // Table name (must be in the allowlist)
  string alias = 2;  // Optional alias for use in joins/filters
}

// Filter expression tree.
message QueryFilter {
  oneof filter {
    ComparisonFilter comparison = 1;
    LogicalFilter logical = 2;
    NotFilter not = 3;
    InFilter in = 4;
    LikeFilter like = 5;
    NullCheckFilter null_check = 6;
    BetweenFilter between = 7;
  }
}

message ComparisonFilter {
  string column = 1;
  string table = 2;        // Table alias (empty = primary table)
  ComparisonOp op = 3;
  QueryValue value = 4;
}

enum ComparisonOp {
  CMP_EQ = 0;
  CMP_NE = 1;
  CMP_LT = 2;
  CMP_LE = 3;
  CMP_GT = 4;
  CMP_GE = 5;
}

message LogicalFilter {
  LogicalOp op = 1;
  repeated QueryFilter operands = 2;
}

enum LogicalOp {
  LOGICAL_AND = 0;
  LOGICAL_OR = 1;
}

message NotFilter {
  QueryFilter operand = 1;
}

message InFilter {
  string column = 1;
  string table = 2;
  repeated QueryValue values = 3;
}

message LikeFilter {
  string column = 1;
  string table = 2;
  string pattern = 3;  // SQL LIKE pattern (e.g., "%train%")
}

message NullCheckFilter {
  string column = 1;
  string table = 2;
  NullOp op = 3;
}

enum NullOp {
  NULL_IS_NULL = 0;
  NULL_IS_NOT_NULL = 1;
}

message BetweenFilter {
  string column = 1;
  string table = 2;
  QueryValue low = 3;
  QueryValue high = 4;
}

// Typed scalar value.
message QueryValue {
  oneof value {
    string string_value = 1;
    int64 int_value = 2;
    double float_value = 3;
    bool bool_value = 4;
  }
}

message QueryJoin {
  QueryTable table = 1;
  JoinKind kind = 2;
  // Join condition: left_table.left_column = right_table.right_column
  string left_column = 3;
  string left_table = 4;   // Table alias (empty = primary table)
  string right_column = 5;
  string right_table = 6;  // Table alias of joined table
}

enum JoinKind {
  JOIN_INNER = 0;
  JOIN_LEFT = 1;
}

message QueryGroupBy {
  repeated QueryColumn columns = 1;  // Columns to group by (func must be AGG_NONE)
}

message QueryOrderBy {
  string column = 1;
  string table = 2;
  SortDir direction = 3;
}

enum SortDir {
  SORT_ASC = 0;
  SORT_DESC = 1;
}

// ---- RPC messages ----

message QueryRequest {
  Query query = 1;
}

message QueryResponse {
  // Column metadata for the result set.
  repeated ColumnMeta columns = 1;
  // Rows as JSON-encoded arrays (each row is a JSON array of values).
  repeated string rows = 2;
  // Total row count before LIMIT (populated when offset=0 and limit>0).
  int32 total_count = 3;
}

message ColumnMeta {
  string name = 1;   // Column name or alias
  string type = 2;   // "text", "integer", "real", "blob"
}

// Admin-only raw SQL query.
message RawQueryRequest {
  string sql = 1;    // Raw SELECT statement
}

message RawQueryResponse {
  repeated ColumnMeta columns = 1;
  repeated string rows = 2;
}
```

### Service Definition

Add two new RPCs to `ControllerService` (cluster.proto):

```protobuf
service ControllerService {
  // ... existing RPCs ...

  // Structured query against the controller database (authenticated users).
  rpc ExecuteQuery(QueryRequest) returns (QueryResponse);

  // Raw SQL query (admin-only).
  rpc ExecuteRawQuery(RawQueryRequest) returns (RawQueryResponse);
}
```

## 4. SQL Generation

The query executor validates and converts `Query` into parameterized SQL. It reuses the existing `db.snapshot()` mechanism for transactional consistency.

### Validation Rules

1. **Table allowlist** — only tables in the allowlist can appear in `from` or `join`.
2. **Column allowlist** — only columns in each table's schema can be referenced. BLOB columns are excluded by default.
3. **SELECT only** — the generated SQL always starts with `SELECT`. No DDL/DML.
4. **No subqueries** — the proto structure makes subqueries impossible by design.
5. **Limit cap** — server enforces a maximum of 1000 rows.
6. **Join limit** — maximum 3 joins per query.
7. **Single aggregate mode** — if any column has an aggregate function, all non-aggregated columns must appear in the GROUP BY.

### Python Implementation

New file: `lib/iris/src/iris/cluster/controller/query.py`

```python
"""Structured query executor for the generic query API."""

import json
import re
from dataclasses import dataclass

from iris.cluster.controller.db import ControllerDB
from iris.rpc import cluster_pb2

MAX_LIMIT = 1000
DEFAULT_LIMIT = 100
MAX_JOINS = 3

# Tables accessible to all authenticated users.
USER_TABLES: dict[str, set[str]] = {
    "jobs": {
        "job_id", "user_id", "parent_job_id", "root_job_id", "depth",
        "state", "submitted_at_ms", "root_submitted_at_ms",
        "started_at_ms", "finished_at_ms", "error", "exit_code",
        "num_tasks", "is_reservation_holder", "scheduling_deadline_epoch_ms",
    },
    "tasks": {
        "task_id", "job_id", "task_index", "state", "error", "exit_code",
        "submitted_at_ms", "started_at_ms", "finished_at_ms",
        "max_retries_failure", "max_retries_preemption",
        "failure_count", "preemption_count", "current_attempt_id",
        "priority_neg_depth", "priority_root_submitted_ms",
        "priority_insertion",
    },
    "task_attempts": {
        "task_id", "attempt_id", "worker_id", "state",
        "created_at_ms", "started_at_ms", "finished_at_ms",
        "exit_code", "error",
    },
    "workers": {
        "worker_id", "address", "healthy", "active",
        "consecutive_failures", "last_heartbeat_ms",
        "committed_cpu_millicores", "committed_mem_bytes",
        "committed_gpu", "committed_tpu",
    },
    "worker_attributes": {
        "worker_id", "key", "value_type",
        "str_value", "int_value", "float_value",
    },
    "endpoints": {
        "endpoint_id", "name", "address", "job_id",
        "task_id", "metadata_json", "registered_at_ms",
    },
    "txn_actions": {
        "id", "txn_id", "action", "entity_id",
        "details_json", "created_at_ms",
    },
    "users": {
        "user_id", "created_at_ms", "display_name", "role",
    },
}

# Additional tables accessible only to admins.
ADMIN_TABLES: dict[str, set[str]] = {
    "api_keys": {
        "key_id", "key_prefix", "user_id", "name",
        "created_at_ms", "last_used_at_ms", "expires_at_ms",
        "revoked_at_ms",
        # key_hash intentionally excluded — never exposed via query API
    },
    "scaling_groups": {
        "name", "consecutive_failures", "backoff_until_ms",
        "last_scale_up_ms", "last_scale_down_ms",
        "quota_exceeded_until_ms", "quota_reason", "updated_at_ms",
    },
    "slices": {
        "slice_id", "scale_group", "lifecycle", "worker_ids",
        "created_at_ms", "last_active_ms", "error_message",
    },
    "tracked_workers": {
        "worker_id", "slice_id", "scale_group", "internal_address",
    },
    "reservation_claims": {"worker_id", "job_id", "entry_idx"},
    "dispatch_queue": {
        "id", "worker_id", "kind", "task_id", "created_at_ms",
    },
    "txn_log": {"id", "kind", "payload_json", "created_at_ms"},
    "worker_task_history": {
        "id", "worker_id", "task_id", "assigned_at_ms",
    },
    "worker_resource_history": {
        "id", "worker_id", "timestamp_ms",
    },
    "task_profiles": {"id", "task_id", "captured_at_ms"},
    "schema_migrations": {"name", "applied_at_ms"},
    "meta": {"key", "value"},
}

# BLOB columns that cannot be meaningfully queried via SQL.
# These are excluded from all allowlists above.
BLOB_COLUMNS = {
    ("jobs", "request_proto"),
    ("workers", "metadata_proto"),
    ("workers", "resource_snapshot_proto"),
    ("tasks", "resource_usage_proto"),
    ("worker_resource_history", "snapshot_proto"),
    ("dispatch_queue", "payload_proto"),
    ("task_profiles", "profile_data"),
}

_CMP_OPS = {
    cluster_pb2.CMP_EQ: "=",
    cluster_pb2.CMP_NE: "!=",
    cluster_pb2.CMP_LT: "<",
    cluster_pb2.CMP_LE: "<=",
    cluster_pb2.CMP_GT: ">",
    cluster_pb2.CMP_GE: ">=",
}

_AGG_FUNCS = {
    cluster_pb2.AGG_COUNT: "COUNT",
    cluster_pb2.AGG_SUM: "SUM",
    cluster_pb2.AGG_AVG: "AVG",
    cluster_pb2.AGG_MIN: "MIN",
    cluster_pb2.AGG_MAX: "MAX",
    cluster_pb2.AGG_COUNT_STAR: "COUNT",
}


@dataclass(frozen=True)
class QueryResult:
    columns: list[cluster_pb2.ColumnMeta]
    rows: list[str]  # JSON-encoded arrays
    total_count: int


def _allowed_tables(is_admin: bool) -> dict[str, set[str]]:
    tables = dict(USER_TABLES)
    if is_admin:
        tables.update(ADMIN_TABLES)
    return tables


def _resolve_column(
    column: str,
    table_alias: str,
    alias_to_table: dict[str, str],
    allowed: dict[str, set[str]],
) -> str:
    """Validate and return qualified column reference."""
    table_name = alias_to_table.get(table_alias)
    if table_name is None:
        raise ValueError(f"Unknown table alias: {table_alias!r}")
    if column not in allowed.get(table_name, set()):
        raise ValueError(
            f"Column {column!r} not accessible on table {table_name!r}"
        )
    return f"{table_alias}.{column}"


def _encode_value(v: cluster_pb2.QueryValue) -> object:
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
    f: cluster_pb2.QueryFilter,
    alias_to_table: dict[str, str],
    default_alias: str,
    allowed: dict[str, set[str]],
) -> tuple[str, list[object]]:
    """Compile a QueryFilter tree into (sql, params)."""
    which = f.WhichOneof("filter")

    if which == "comparison":
        c = f.comparison
        alias = c.table or default_alias
        col = _resolve_column(c.column, alias, alias_to_table, allowed)
        op = _CMP_OPS.get(c.op)
        if op is None:
            raise ValueError(f"Unknown comparison op: {c.op}")
        return f"{col} {op} ?", [_encode_value(c.value)]

    if which == "logical":
        lg = f.logical
        if not lg.operands:
            raise ValueError("Logical filter requires at least one operand")
        op_str = "AND" if lg.op == cluster_pb2.LOGICAL_AND else "OR"
        parts, params = [], []
        for operand in lg.operands:
            s, p = _compile_filter(
                operand, alias_to_table, default_alias, allowed
            )
            parts.append(f"({s})")
            params.extend(p)
        return f" {op_str} ".join(parts), params

    if which == "not":
        s, p = _compile_filter(
            f.not_.operand, alias_to_table, default_alias, allowed
        )
        return f"NOT ({s})", p

    if which == "in":
        inf = f.in_
        alias = inf.table or default_alias
        col = _resolve_column(inf.column, alias, alias_to_table, allowed)
        if not inf.values:
            return "0", []  # Empty IN is always false
        placeholders = ", ".join("?" for _ in inf.values)
        return f"{col} IN ({placeholders})", [
            _encode_value(v) for v in inf.values
        ]

    if which == "like":
        lf = f.like
        alias = lf.table or default_alias
        col = _resolve_column(lf.column, alias, alias_to_table, allowed)
        return f"{col} LIKE ?", [lf.pattern]

    if which == "null_check":
        nc = f.null_check
        alias = nc.table or default_alias
        col = _resolve_column(nc.column, alias, alias_to_table, allowed)
        op = "IS NULL" if nc.op == cluster_pb2.NULL_IS_NULL else "IS NOT NULL"
        return f"{col} {op}", []

    if which == "between":
        bf = f.between
        alias = bf.table or default_alias
        col = _resolve_column(bf.column, alias, alias_to_table, allowed)
        return f"{col} BETWEEN ? AND ?", [
            _encode_value(bf.low),
            _encode_value(bf.high),
        ]

    raise ValueError(f"Unknown filter type: {which}")


def execute_query(
    db: ControllerDB,
    query: cluster_pb2.Query,
    is_admin: bool,
) -> QueryResult:
    """Validate and execute a structured query against the controller DB."""
    allowed = _allowed_tables(is_admin)

    # Validate FROM table
    from_table = query.from_
    if not from_table.name:
        raise ValueError("Query must specify a FROM table")
    if from_table.name not in allowed:
        raise ValueError(f"Table {from_table.name!r} is not accessible")
    primary_alias = from_table.alias or from_table.name
    alias_to_table: dict[str, str] = {primary_alias: from_table.name}

    # Validate JOINs
    if len(query.joins) > MAX_JOINS:
        raise ValueError(f"Maximum {MAX_JOINS} joins allowed")
    join_clauses: list[str] = []
    for j in query.joins:
        if not j.table.name or j.table.name not in allowed:
            raise ValueError(f"Join table {j.table.name!r} is not accessible")
        j_alias = j.table.alias or j.table.name
        if j_alias in alias_to_table:
            raise ValueError(f"Duplicate table alias: {j_alias!r}")
        alias_to_table[j_alias] = j.table.name
        kind = "LEFT JOIN" if j.kind == cluster_pb2.JOIN_LEFT else "JOIN"
        l_alias = j.left_table or primary_alias
        r_alias = j.right_table or j_alias
        l_col = _resolve_column(
            j.left_column, l_alias, alias_to_table, allowed
        )
        r_col = _resolve_column(
            j.right_column, r_alias, alias_to_table, allowed
        )
        join_clauses.append(f"{kind} {j.table.name} {j_alias} ON {l_col} = {r_col}")

    # Build SELECT columns
    select_parts: list[str] = []
    if not query.columns:
        # Default: all allowed columns from primary table
        for col in sorted(allowed[from_table.name]):
            select_parts.append(f"{primary_alias}.{col}")
    else:
        for qc in query.columns:
            alias = qc.table or primary_alias
            if qc.func == cluster_pb2.AGG_COUNT_STAR:
                expr = "COUNT(*)"
            elif qc.func and qc.func != cluster_pb2.AGG_NONE:
                col = _resolve_column(
                    qc.name, alias, alias_to_table, allowed
                )
                func_name = _AGG_FUNCS[qc.func]
                expr = f"{func_name}({col})"
            else:
                col = _resolve_column(
                    qc.name, alias, alias_to_table, allowed
                )
                expr = col
            label = qc.alias or qc.name
            select_parts.append(f"{expr} AS {label}")

    # Build WHERE
    params: list[object] = []
    where_sql = ""
    if query.HasField("where"):
        w_sql, w_params = _compile_filter(
            query.where, alias_to_table, primary_alias, allowed
        )
        where_sql = f" WHERE {w_sql}"
        params.extend(w_params)

    # Build GROUP BY
    group_sql = ""
    if query.HasField("group_by") and query.group_by.columns:
        group_cols = []
        for gc in query.group_by.columns:
            alias = gc.table or primary_alias
            col = _resolve_column(
                gc.name, alias, alias_to_table, allowed
            )
            group_cols.append(col)
        group_sql = f" GROUP BY {', '.join(group_cols)}"

    # Build ORDER BY
    order_sql = ""
    if query.order_by:
        order_parts = []
        for ob in query.order_by:
            alias = ob.table or primary_alias
            col = _resolve_column(
                ob.column, alias, alias_to_table, allowed
            )
            direction = "DESC" if ob.direction == cluster_pb2.SORT_DESC else "ASC"
            order_parts.append(f"{col} {direction}")
        order_sql = f" ORDER BY {', '.join(order_parts)}"

    # Build LIMIT/OFFSET
    limit = min(query.limit, MAX_LIMIT) if query.limit > 0 else DEFAULT_LIMIT
    limit_sql = f" LIMIT {limit}"
    if query.offset > 0:
        limit_sql += f" OFFSET {query.offset}"

    # Assemble final SQL
    select_sql = ", ".join(select_parts)
    joins_sql = " ".join(join_clauses)
    if joins_sql:
        joins_sql = " " + joins_sql

    sql = f"SELECT {select_sql} FROM {from_table.name} {primary_alias}{joins_sql}{where_sql}{group_sql}{order_sql}{limit_sql}"

    # Build the base query (without LIMIT/OFFSET) for counting total matching rows.
    base_sql = f"FROM {from_table.name} {primary_alias}{joins_sql}{where_sql}{group_sql}"

    # Execute under snapshot.
    # TODO: Add an `execute_raw` method to QuerySnapshot during implementation
    # so we don't access the private _conn attribute:
    #   def execute_raw(self, sql: str, params: tuple) -> sqlite3.Cursor
    with db.snapshot() as q:
        cursor = q._conn.execute(sql, tuple(params))
        col_descriptions = cursor.description
        raw_rows = cursor.fetchall()

        # Compute pre-LIMIT total count on the first page so clients know
        # how many rows match the query for pagination purposes.
        total_count = len(raw_rows)
        if query.offset == 0 and limit > 0:
            count_sql = f"SELECT COUNT(*) {base_sql}"
            count_cursor = q._conn.execute(count_sql, tuple(params))
            total_count = count_cursor.fetchone()[0]

    # Build column metadata
    columns = [
        cluster_pb2.ColumnMeta(name=desc[0], type=_sqlite_type(desc))
        for desc in col_descriptions
    ]

    # Encode rows as JSON arrays
    rows = [json.dumps([_encode_cell(row[i]) for i in range(len(columns))]) for row in raw_rows]

    return QueryResult(columns=columns, rows=rows, total_count=total_count)


def execute_raw_query(
    db: ControllerDB,
    sql: str,
) -> QueryResult:
    """Execute a raw SQL query (admin-only). Only SELECT statements allowed."""
    stripped = sql.strip()
    if not stripped.upper().startswith("SELECT"):
        raise ValueError("Only SELECT statements are allowed")

    # Reject multi-statement execution
    if ";" in stripped:
        raise ValueError("Multiple SQL statements are not allowed")

    # Block dangerous keywords using word-boundary matching
    upper = stripped.upper()
    for keyword in ("INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "ATTACH", "DETACH"):
        if re.search(rf"\b{keyword}\b", upper):
            raise ValueError(f"Forbidden SQL keyword: {keyword}")

    # TODO: Use QuerySnapshot.execute_raw() once added (see execute_query above)
    with db.snapshot() as q:
        cursor = q._conn.execute(stripped)
        col_descriptions = cursor.description
        raw_rows = cursor.fetchall()

    columns = [
        cluster_pb2.ColumnMeta(name=desc[0], type=_sqlite_type(desc))
        for desc in col_descriptions
    ]
    rows = [json.dumps([_encode_cell(row[i]) for i in range(len(columns))]) for row in raw_rows]

    return QueryResult(columns=columns, rows=rows, total_count=len(rows))


def _sqlite_type(desc: tuple) -> str:
    """Infer a type string from sqlite3 cursor.description."""
    # SQLite cursor.description doesn't reliably report types,
    # so we return "unknown" and let the client infer from values.
    return "unknown"


def _encode_cell(value: object) -> object:
    """Encode a SQLite cell value for JSON serialization."""
    if value is None:
        return None
    if isinstance(value, bytes):
        return f"<blob:{len(value)} bytes>"
    return value
```

## 5. Privileged Tables

### User-Accessible Tables (all authenticated users)

| Table | Excluded Columns | Notes |
|-------|-----------------|-------|
| `jobs` | `request_proto` (BLOB) | Core job metadata. `request_proto` is a serialized `LaunchJobRequest` — not queryable via SQL. |
| `tasks` | `resource_usage_proto` (BLOB) | Task state and retry metadata. |
| `task_attempts` | — | Attempt history per task. |
| `workers` | `metadata_proto` (BLOB), `resource_snapshot_proto` (BLOB) | Worker health and committed resources. |
| `worker_attributes` | — | Key-value attributes for constraint matching. |
| `endpoints` | — | Service discovery entries. |
| `txn_actions` | — | Controller action log. |
| `users` | — | User list with roles. |

### Admin-Only Tables

| Table | Excluded Columns | Reason |
|-------|-----------------|--------|
| `api_keys` | `key_hash` | Contains credential material. `key_hash` is **never** exposed. |
| `scaling_groups` | — | Autoscaler internals. All scalar columns exposed after migration 0003 normalization. |
| `slices` | — | Slice lifecycle tracking (added in migration 0003). |
| `tracked_workers` | — | Autoscaler worker tracking. |
| `reservation_claims` | — | Reservation state. |
| `dispatch_queue` | `payload_proto` (BLOB) | Pending worker dispatches. |
| `txn_log` | — | Raw transaction log. |
| `worker_task_history` | — | Historical task assignments. |
| `worker_resource_history` | `snapshot_proto` (BLOB) | Resource snapshots (BLOBs excluded). |
| `task_profiles` | `profile_data` (BLOB) | Profile captures (BLOBs excluded). |
| `schema_migrations` | — | Migration state. |
| `meta` | — | Key-value metadata. |

### Excluded Tables

The `logs` table is intentionally excluded from both allowlists. Log access goes through the in-memory `LogStore` API via `GetTaskLogs` and `FetchLogs` RPCs, not through direct SQL queries. The `logs` table is a persistence backing store and its schema is an implementation detail of the log store.

### BLOB Column Limitation

Columns storing serialized protobufs (`request_proto`, `metadata_proto`, `resource_snapshot_proto`, `resource_usage_proto`, `payload_proto`, `profile_data`) are excluded from query results. They contain binary data that is meaningless as SQL values. The dedicated RPCs (`GetJobStatus`, `GetWorkerStatus`, etc.) remain the correct way to retrieve deserialized protobuf data.

## 6. Raw SQL Bypass

Admin-only RPC for queries that the structured DSL cannot express (complex CTEs, window functions, ad-hoc debugging).

### Proto Messages

See `RawQueryRequest` / `RawQueryResponse` in Section 3.

### Handler

```python
def execute_raw_query_rpc(
    self,
    request: cluster_pb2.RawQueryRequest,
    ctx: Any,
) -> cluster_pb2.RawQueryResponse:
    """Execute raw SQL (admin-only)."""
    caller = self._require_auth()
    self._require_admin(caller)

    result = execute_raw_query(self._db, request.sql)
    return cluster_pb2.RawQueryResponse(
        columns=result.columns,
        rows=result.rows,
    )
```

### Safety

- Auth: uses existing `_require_admin()` (service.py:1658-1660) which checks `role == 'admin'` via `db.get_user_role()`.
- Read-only: `db.snapshot()` opens a transaction with `BEGIN` and rolls back on exit (db.py:360-368). Even if a DML statement slipped through validation, it would be rolled back.
- Keyword blocklist: rejects SQL containing `INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, `CREATE`, `ATTACH`, `DETACH` as standalone words.
- No `ATTACH DATABASE`: prevents reading other SQLite files on the filesystem.

## 7. Existing RPC Deprecation Plan

### Phase 1: Deprecate (add `ExecuteQuery`, keep existing RPCs)

Existing RPCs continue to work. New dashboard features prefer `ExecuteQuery`.

### Phase 2: Migrate dashboard components to `ExecuteQuery`

### Phase 3: Remove deprecated RPCs (major version bump)

### RPC → Query Mapping

**ListJobs** — paginated job listing with sort/filter:

```json
{
  "query": {
    "from": {"name": "jobs", "alias": "j"},
    "columns": [
      {"name": "job_id"},
      {"name": "user_id"},
      {"name": "state"},
      {"name": "submitted_at_ms"},
      {"name": "started_at_ms"},
      {"name": "finished_at_ms"},
      {"name": "error"},
      {"name": "exit_code"},
      {"name": "num_tasks"}
    ],
    "where": {
      "comparison": {
        "column": "state",
        "op": "CMP_EQ",
        "value": {"int_value": 3}
      }
    },
    "order_by": [{"column": "submitted_at_ms", "direction": "SORT_DESC"}],
    "limit": 50,
    "offset": 0
  }
}
```

Note: `ListJobs` also includes task state aggregation and pending diagnostics that require joins and Python post-processing. The query API replaces the raw data fetch; the dashboard may issue a follow-up query for task counts per job:

```json
{
  "query": {
    "from": {"name": "tasks", "alias": "t"},
    "columns": [
      {"name": "job_id"},
      {"name": "state"},
      {"name": "task_id", "func": "AGG_COUNT", "alias": "count"}
    ],
    "where": {
      "in": {
        "column": "job_id",
        "values": [{"string_value": "/user/job1"}, {"string_value": "/user/job2"}]
      }
    },
    "group_by": {"columns": [{"name": "job_id"}, {"name": "state"}]}
  }
}
```

**ListUsers** — per-user aggregate counts:

```json
{
  "query": {
    "from": {"name": "jobs", "alias": "j"},
    "columns": [
      {"name": "user_id"},
      {"name": "state"},
      {"name": "job_id", "func": "AGG_COUNT", "alias": "job_count"}
    ],
    "group_by": {"columns": [{"name": "user_id"}, {"name": "state"}]}
  }
}
```

**GetTransactions** — recent controller actions:

```json
{
  "query": {
    "from": {"name": "txn_actions"},
    "order_by": [{"column": "created_at_ms", "direction": "SORT_DESC"}],
    "limit": 50
  }
}
```

**ListWorkers** — all workers with health status:

```json
{
  "query": {
    "from": {"name": "workers"},
    "order_by": [{"column": "last_heartbeat_ms", "direction": "SORT_DESC"}]
  }
}
```

**ListEndpoints** — prefix-filtered endpoints:

```json
{
  "query": {
    "from": {"name": "endpoints"},
    "where": {
      "like": {
        "column": "name",
        "pattern": "abc123/%"
      }
    },
    "order_by": [{"column": "registered_at_ms", "direction": "SORT_DESC"}]
  }
}
```

**ListTasks** — tasks for a job:

```json
{
  "query": {
    "from": {"name": "tasks"},
    "where": {
      "comparison": {
        "column": "job_id",
        "op": "CMP_EQ",
        "value": {"string_value": "/user/my-job"}
      }
    },
    "order_by": [{"column": "task_index", "direction": "SORT_ASC"}]
  }
}
```

### RPCs NOT Deprecated

| RPC | Reason |
|-----|--------|
| `GetJobStatus` | Returns deserialized `request_proto` (BLOB), task statuses with attempt details, pending diagnostics from autoscaler/scheduler (live state, not DB). |
| `GetTaskStatus` | Returns deserialized `resource_usage_proto` (BLOB), attempts with converted timestamps. |
| `GetWorkerStatus` | Proxies to worker for live logs, fetches resource history BLOBs, resolves VM info from autoscaler. |
| `GetAutoscalerStatus` | Autoscaler state is in-memory, not in DB. Worker enrichment from DB is secondary. |
| `GetTaskLogs` | Queries `LogStore` (in-memory ring buffer), not the DB `logs` table. |
| `FetchLogs` | Same — `LogStore` or worker proxy. |
| `GetProcessStatus` | Live `psutil` data or worker proxy. |
| `ProfileTask` | Worker proxy for live profiling. |
| `GetAuthInfo` | Config metadata, not DB query. |
| `GetCurrentUser` | ContextVar lookup + role check. |
| `Login` | Auth lifecycle mutation. |
| `CreateApiKey` / `RevokeApiKey` / `ListApiKeys` | Auth lifecycle mutations. |
| `LaunchJob` / `TerminateJob` | Write operations. |
| `Register` / `RegisterEndpoint` / `UnregisterEndpoint` | Write operations. |
| `BeginCheckpoint` | Write operation. |

## 8. Dashboard Updates

Dashboard components that call deprecated RPCs need a migration path. The `useControllerRpc` composable (`dashboard/src/composables/useRpc.ts`) already supports arbitrary method names and JSON bodies.

### New Composable

```typescript
// dashboard/src/composables/useQuery.ts
import { controllerRpcCall } from './useRpc'

export interface QueryRequest {
  query: {
    from: { name: string; alias?: string }
    columns?: Array<{ name: string; table?: string; func?: string; alias?: string }>
    where?: object
    joins?: object[]
    group_by?: { columns: Array<{ name: string; table?: string }> }
    order_by?: Array<{ column: string; table?: string; direction?: string }>
    limit?: number
    offset?: number
  }
}

export interface QueryResponse {
  columns: Array<{ name: string; type: string }>
  rows: string[]  // JSON-encoded arrays
  totalCount: number
}

export async function executeQuery(req: QueryRequest): Promise<QueryResponse> {
  return controllerRpcCall<QueryResponse>('ExecuteQuery', req)
}
```

### Component Migration Plan

| Component | File | Current RPC | Migration |
|-----------|------|-------------|-----------|
| `JobsTab.vue` | `dashboard/src/components/controller/JobsTab.vue` | `ListJobs` | Phase 2 — replace with `ExecuteQuery` for jobs table + separate task count query |
| `FleetTab.vue` | `dashboard/src/components/controller/FleetTab.vue` | `ListWorkers` | Phase 2 — replace with `ExecuteQuery` for workers table |
| `EndpointsTab.vue` | `dashboard/src/components/controller/EndpointsTab.vue` | `ListEndpoints` | Phase 2 — replace with `ExecuteQuery` |
| `UsersTab.vue` | `dashboard/src/components/controller/UsersTab.vue` | `ListUsers` | Phase 2 — replace with `ExecuteQuery` GROUP BY |
| `TransactionsTab.vue` | `dashboard/src/components/controller/TransactionsTab.vue` | `GetTransactions` | Phase 2 — replace with `ExecuteQuery` |
| `JobDetail.vue` | `dashboard/src/components/controller/JobDetail.vue` | `GetJobStatus` | **Keep** — needs deserialized protos + scheduler state |
| `TaskDetail.vue` | `dashboard/src/components/controller/TaskDetail.vue` | `GetTaskStatus` | **Keep** — needs attempt details |
| `WorkerDetail.vue` | `dashboard/src/components/controller/WorkerDetail.vue` | `GetWorkerStatus` | **Keep** — needs live worker proxy |
| `AutoscalerTab.vue` | `dashboard/src/components/controller/AutoscalerTab.vue` | `GetAutoscalerStatus` | **Keep** — in-memory state |
| `StatusTab.vue` | `dashboard/src/components/controller/StatusTab.vue` | `GetProcessStatus` | **Keep** — live process info |
| `AccountTab.vue` | `dashboard/src/components/controller/AccountTab.vue` | `ListApiKeys`, `GetCurrentUser` | **Keep** — auth lifecycle |

### New Dashboard Feature: Query Explorer

Add a new tab or admin page with a query builder UI for ad-hoc queries. Admin users also get a raw SQL input box.

## 9. CLI Updates

### New `iris query` Command

```bash
# Structured query via JSON
iris query '{"from": {"name": "jobs"}, "where": {"comparison": {"column": "state", "op": "CMP_EQ", "value": {"int_value": 3}}}, "limit": 10}'

# Raw SQL (admin-only)
iris query --raw "SELECT user_id, COUNT(*) as cnt FROM jobs GROUP BY user_id ORDER BY cnt DESC"

# Convenience: table scan with optional filters
iris query jobs --where "state=3" --limit 10
iris query tasks --where "job_id=/user/my-job" --order "submitted_at_ms:desc"
```

### Implementation

Add to `cli/rpc.py` or new `cli/query.py`:

```python
@click.command("query")
@click.argument("query_json", required=False)
@click.option("--raw", "raw_sql", default=None, help="Raw SQL query (admin-only)")
@click.option("--format", "fmt", type=click.Choice(["table", "json", "csv"]), default="table")
@click.pass_context
def query_cmd(ctx, query_json: str | None, raw_sql: str | None, fmt: str):
    """Execute a query against the controller database."""
    controller_url = require_controller_url(ctx)
    tp = ctx.obj.get("token_provider") if ctx.obj else None

    if raw_sql:
        request = cluster_pb2.RawQueryRequest(sql=raw_sql)
        response = call_rpc("controller", "ExecuteRawQuery", controller_url, request, token_provider=tp)
    elif query_json:
        query = json_format.ParseDict(json.loads(query_json), cluster_pb2.Query())
        request = cluster_pb2.QueryRequest(query=query)
        response = call_rpc("controller", "ExecuteQuery", controller_url, request, token_provider=tp)
    else:
        raise click.UsageError("Provide a query JSON or --raw SQL")

    # Format output
    ...
```

### Dynamic RPC Exposure

The existing `ServiceCommands` in `cli/rpc.py` auto-discovers all RPC methods on `ControllerServiceClientSync`. Once `ExecuteQuery` and `ExecuteRawQuery` are added to the proto and the client is regenerated, they automatically appear as:

```bash
iris rpc controller execute-query --json '...'
iris rpc controller execute-raw-query --sql '...'
```

## 10. Implementation Plan

### Phase 1: Core (can be parallelized)

**Unit A — Proto + codegen** (1 file)
- Add query messages to `lib/iris/src/iris/rpc/cluster.proto`
- Add `ExecuteQuery` and `ExecuteRawQuery` to `ControllerService`
- Run `scripts/generate_protos.py`

**Unit B — Query executor** (1 new file)
- Create `lib/iris/src/iris/cluster/controller/query.py`
- Implement `execute_query()` and `execute_raw_query()` with validation
- Unit tests: `lib/iris/tests/test_query.py`

**Depends on A + B:**

**Unit C — Service handlers** (1 file)
- Add `execute_query()` and `execute_raw_query()` handlers to `ControllerServiceImpl` in `service.py`
- Wire auth checks (`_require_auth()` for structured, `_require_admin()` for raw)

**Unit D — CLI** (1 new file)
- Create `lib/iris/src/iris/cli/query.py`
- Register on main CLI group

### Phase 2: Dashboard (after Phase 1)

**Unit E — Dashboard composable** (1 new file)
- Create `lib/iris/dashboard/src/composables/useQuery.ts`
- Add `QueryResponse` type to `types/rpc.ts`

**Unit F — Query Explorer page** (1 new file)
- New dashboard component for ad-hoc queries

### Phase 3: Migration (after Phase 2, incremental)

- Migrate `JobsTab.vue` to use `ExecuteQuery` for the primary listing
- Migrate `UsersTab.vue`, `TransactionsTab.vue`, `EndpointsTab.vue`, `FleetTab.vue`
- Deprecate old RPCs in proto with `[deprecated = true]`
- Remove deprecated RPCs and handlers in a future major version

## 11. Risks and Open Questions

### Risks

1. **Query performance** — Complex queries against large tables (e.g., `tasks` with 100k+ rows) could saturate the SQLite WAL or hold the snapshot lock too long. Mitigations: enforce `MAX_LIMIT = 1000`, require `LIMIT` on all queries, add query timeout via SQLite's `progress_handler`.

2. **SQL injection in raw queries** — The raw SQL bypass accepts arbitrary strings. The keyword blocklist + snapshot rollback provide defense-in-depth, but a determined admin could still construct harmful reads. This is acceptable because admin role already implies full trust.

3. **Schema drift** — The column allowlists in `query.py` must stay in sync with migration files. A test should verify that `USER_TABLES` and `ADMIN_TABLES` match the actual schema.

4. **BLOB column exposure** — If a user queries `SELECT *` on a table with BLOB columns, those columns are excluded from the allowlist so the query will succeed but only return non-BLOB columns. This is the correct behavior but may surprise users.

### Open Questions

1. **Should `ExecuteQuery` support `HAVING` clauses?** — `HAVING` filters after `GROUP BY` aggregation. Useful for queries like "users with more than 10 running jobs". Could be added as a `QueryFilter having` field on `QueryGroupBy`. Recommend: defer to Phase 2, add if dashboard needs it.

2. **Should results include row count before LIMIT?** — The implementation computes `total_count` via a `SELECT COUNT(*)` of the same query (without LIMIT) only when `offset == 0` and `limit > 0`, so the first page knows total pages without doubling cost on subsequent pages.

3. **Should `QueryResponse.rows` use a more structured format?** — JSON-encoded string arrays are simple but require client-side parsing. Alternatives: repeated `QueryRow` with `repeated QueryValue` fields (more protobuf-native), or a columnar format. Recommend: start with JSON arrays for simplicity; the overhead is negligible for <1000 rows.

4. **Per-user row filtering?** — Should non-admin users only see their own jobs/tasks? Currently all listing RPCs return all users' data. The query API inherits this. If row-level security is needed, it can be added as an implicit WHERE clause injected by the executor.

5. **Rate limiting?** — Should `ExecuteQuery` have per-user rate limits to prevent abuse? Recommend: not in Phase 1. The controller is single-tenant and the dashboard already polls at fixed intervals.
