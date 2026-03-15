# Iris Generic Query API

Design document for [#3492](https://github.com/marin-community/marin/issues/3492): adding a structured query DSL to the iris controller.

## 1. Current State / Problem

The `ControllerService` has 15+ query-oriented RPCs, each with bespoke proto messages, handler code, and pagination/filtering logic. `ListJobs` alone is ~150 lines handling sort fields, state filters, pagination with parent-child grouping, and per-job task summary aggregation. `ListUsers` does a manual GROUP BY in Python across two separate queries.

The DB layer (`db.py`) already has a rich query DSL with `Table`, `Column`, `Predicate`, `Join`, `Order`, `SelectExpr`, and `QuerySnapshot.select()`. But each RPC builds its own bespoke query and post-processing.

Adding a new read pattern (e.g., "jobs by user sorted by failure count") requires a new proto message pair, a new handler method, dashboard wiring, and CLI plumbing.

## 2. Goals and Non-Goals

### Goals

- **Structured protobuf query DSL** that covers SELECT/FROM/WHERE/JOIN/GROUP BY/ORDER BY/LIMIT against the controller's SQLite database.
- **Raw SQL admin bypass** for ad-hoc queries that the DSL cannot express.
- **Unified pagination** via LIMIT/OFFSET in the DSL, replacing per-RPC pagination.
- **Denylist access control**: block sensitive tables (`api_keys`, `controller_secrets`) from non-admin users; block dangerous columns (`key_hash`, secret `value`) from all users including admins.
- **Log store query support**: query the log store's SQLite DB via `database: "logs"`.

### Non-Goals

- **Full SQL parser** or **GraphQL** — the DSL is a protobuf message tree, not a string language.
- **Write operations** — the query API is read-only. All mutations go through existing RPCs.
- **Replacing worker-proxied RPCs** — `FetchLogs` (worker mode), `ProfileTask`, `GetProcessStatus` (worker mode), and `GetWorkerStatus` (worker log/resource proxy) involve live worker I/O. These stay as dedicated RPCs.
- **Replacing `GetAutoscalerStatus`** — autoscaler state lives in the autoscaler's in-memory structures, not in the DB.
- **Replacing auth RPCs** — `GetAuthInfo`, `GetCurrentUser`, `Login`, `CreateApiKey`, `RevokeApiKey`, `ListApiKeys` are auth lifecycle operations, not generic queries.

## 3. Query DSL

### Proto Messages

Defined in `lib/iris/src/iris/rpc/query.proto`:

```protobuf
message Query {
  repeated QueryColumn columns = 1;  // Empty = all columns from the primary table
  QueryTable from = 2;
  QueryFilter where = 3;
  repeated QueryJoin joins = 4;
  QueryGroupBy group_by = 5;
  repeated QueryOrderBy order_by = 6;
  int32 limit = 7;   // Max rows (0 = server default 100, capped at 1000)
  int32 offset = 8;
}
```

The filter tree supports: comparison (`=`, `!=`, `<`, `<=`, `>`, `>=`), logical (`AND`, `OR`), `NOT`, `IN`, `LIKE`, `IS NULL`/`IS NOT NULL`, and `BETWEEN`. Aggregates: `COUNT`, `SUM`, `AVG`, `MIN`, `MAX`, `COUNT(*)`.

### Service RPCs

Added to `ControllerService` in `cluster.proto`:

```protobuf
rpc ExecuteQuery(iris.cluster.QueryRequest) returns (iris.cluster.QueryResponse);
rpc ExecuteRawQuery(iris.cluster.RawQueryRequest) returns (iris.cluster.RawQueryResponse);
```

`QueryRequest` includes a `database` field: `"main"` (default) queries the controller DB, `"logs"` queries the log store DB.

## 4. Access Control

The implementation uses a **denylist** model rather than per-table column allowlists. Any table in the database is queryable unless explicitly blocked. This works because auth secrets are isolated to two tables, keeping the denylist small.

### Sensitive Tables (admin-only)

| Table | Reason |
|-------|--------|
| `api_keys` | Contains credential metadata |
| `controller_secrets` | Contains JWT signing keys |

Non-admin users receive an error when querying these tables. Admin users can query them but with column restrictions (see below).

### Blocked Columns (all users, including admins)

| Table | Column | Reason |
|-------|--------|--------|
| `api_keys` | `key_hash` | Raw credential hash — never exposed |
| `controller_secrets` | `value` | Secret key material — never exposed |

### Schema Introspection

For `SELECT *` queries, columns are discovered at runtime via `PRAGMA table_info()` rather than maintained in a static allowlist. Blocked columns are excluded from the introspected set. This means the query API automatically picks up new columns added by migrations without code changes.

### Identifier Validation

All table names, column names, and aliases are validated against `^[a-zA-Z_][a-zA-Z0-9_]*$` before being interpolated into SQL, preventing injection through identifiers.

## 5. SQL Generation

The query executor (`lib/iris/src/iris/cluster/controller/query.py`) validates and converts `Query` messages into parameterized SQL.

### Validation Rules

1. **Table denylist** — sensitive tables are blocked for non-admin users.
2. **Column denylist** — permanently blocked columns (e.g., `key_hash`) are rejected for all users.
3. **Identifier regex** — all names must match `[a-zA-Z_][a-zA-Z0-9_]*`.
4. **SELECT only** — the generated SQL always starts with `SELECT`. No DDL/DML.
5. **No subqueries** — the proto structure makes subqueries impossible by design.
6. **Limit cap** — server enforces a maximum of 1000 rows (default 100).
7. **Join limit** — maximum 3 joins per query.

### Execution

Queries execute under snapshot isolation via `db.snapshot()` (main DB) or a read-only transaction on the log store's `_read_conn`. The snapshot rolls back on exit, so even if a DML statement somehow slipped through validation, it would be rolled back.

Total row count (for pagination) is computed via a parallel `SELECT COUNT(*)` of the same base query (without LIMIT/OFFSET).

## 6. Raw SQL Bypass

Admin-only RPC for queries the structured DSL cannot express (complex CTEs, window functions, ad-hoc debugging).

### Safety

- **Auth**: `require_identity()` + `role == "admin"` check.
- **Read-only**: snapshot isolation with rollback.
- **SELECT only**: rejects SQL not starting with `SELECT`.
- **No multi-statement**: rejects SQL containing `;`.
- **Keyword blocklist**: rejects `INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, `CREATE`, `ATTACH`, `DETACH`, `PRAGMA`, `VACUUM`, `REINDEX`, `SAVEPOINT` as standalone words.

## 7. Log Store Support

The `QueryRequest.database` field selects the target database:

- `"main"` (default) — controller SQLite DB with all job/task/worker tables.
- `"logs"` — log store SQLite DB, restricted to the `logs` table.

The log store path uses `_log_store_snapshot()` which acquires the log store's read lock and opens a read-only transaction.

## 8. Existing RPC Deprecation Plan

### Phase 1 (done): Add `ExecuteQuery`, keep existing RPCs

Existing RPCs continue to work. The query API is available for ad-hoc use and the Query Explorer dashboard tab.

### Phase 2: Migrate dashboard listing components

Migrate `JobsTab`, `FleetTab`, `EndpointsTab`, `UsersTab`, `TransactionsTab` to use `ExecuteQuery` instead of their dedicated listing RPCs.

### Phase 3: Remove deprecated RPCs

### RPCs NOT Deprecated

| RPC | Reason |
|-----|--------|
| `GetJobStatus` | Returns deserialized `request_proto` (BLOB), task statuses with attempt details, pending diagnostics from autoscaler/scheduler (live state, not DB). |
| `GetTaskStatus` | Returns deserialized `resource_usage_proto` (BLOB), attempts with converted timestamps. |
| `GetWorkerStatus` | Proxies to worker for live logs, fetches resource history BLOBs, resolves VM info from autoscaler. |
| `GetAutoscalerStatus` | Autoscaler state is in-memory, not in DB. |
| `GetTaskLogs` / `FetchLogs` | Query `LogStore` in-memory ring buffer or worker proxy. |
| `GetProcessStatus` | Live `psutil` data or worker proxy. |
| `ProfileTask` | Worker proxy for live profiling. |
| `GetAuthInfo` / `GetCurrentUser` | Config metadata / context var lookup. |
| `Login` / `CreateApiKey` / `RevokeApiKey` / `ListApiKeys` | Auth lifecycle mutations. |
| `LaunchJob` / `TerminateJob` | Write operations. |
| `Register` / `RegisterEndpoint` / `UnregisterEndpoint` | Write operations. |
| `BeginCheckpoint` | Write operation. |

## 9. Dashboard

### Query Explorer Tab

A new `QueryExplorerTab.vue` provides ad-hoc query access. Admin users also get a raw SQL input. Results are displayed in a paginated table.

### `useQuery` Composable

`dashboard/src/composables/useQuery.ts` provides:

- Full TypeScript types mirroring the proto query DSL.
- `executeQuery(request)` and `executeRawQuery(request)` — typed wrappers around the RPC.
- `parseRows(columns, rows)` — converts JSON-encoded row arrays into keyed `Record<string, unknown>` objects for template rendering.

## 10. CLI

### `iris query` Command

```bash
# Structured query via JSON
iris query '{"from": {"name": "jobs"}, "limit": 10}'

# Raw SQL (admin-only)
iris query --raw "SELECT count(*) FROM jobs"

# Output formats: table (default), json, csv
iris query -f json '{"from": {"name": "workers"}}'
```

Implemented in `lib/iris/src/iris/cli/query.py`. Supports `--format`/`-f` with `table`, `json`, and `csv` output. Shows pagination info (`N of M rows`) when results are truncated.

## 11. Query Examples

**Jobs by state:**
```json
{
  "from": {"name": "jobs"},
  "where": {"comparison": {"column": "state", "op": "CMP_EQ", "value": {"intValue": 3}}},
  "orderBy": [{"column": "submitted_at_ms", "direction": "SORT_DESC"}],
  "limit": 50
}
```

**Task counts per job:**
```json
{
  "from": {"name": "tasks", "alias": "t"},
  "columns": [
    {"name": "job_id"},
    {"name": "state"},
    {"name": "task_id", "func": "AGG_COUNT", "alias": "count"}
  ],
  "where": {"in": {"column": "job_id", "values": [{"stringValue": "/user/job1"}]}},
  "groupBy": {"columns": [{"name": "job_id"}, {"name": "state"}]}
}
```

**Per-user job counts:**
```json
{
  "from": {"name": "jobs"},
  "columns": [
    {"name": "user_id"},
    {"name": "state"},
    {"name": "job_id", "func": "AGG_COUNT", "alias": "job_count"}
  ],
  "groupBy": {"columns": [{"name": "user_id"}, {"name": "state"}]}
}
```

**Recent transactions:**
```json
{
  "from": {"name": "txn_actions"},
  "orderBy": [{"column": "created_at_ms", "direction": "SORT_DESC"}],
  "limit": 50
}
```

**Prefix-filtered endpoints:**
```json
{
  "from": {"name": "endpoints"},
  "where": {"like": {"column": "name", "pattern": "abc123/%"}},
  "orderBy": [{"column": "registered_at_ms", "direction": "SORT_DESC"}]
}
```

## 12. Risks

1. **Query performance** — Complex queries against large tables could saturate the SQLite WAL or hold the snapshot lock too long. Mitigated by `MAX_LIMIT = 1000` and required LIMIT on all queries.

2. **SQL injection in raw queries** — The keyword blocklist + snapshot rollback provide defense-in-depth. Acceptable because admin role already implies full trust.

3. **Schema drift** — The denylist approach means new tables are automatically queryable. New sensitive tables must be added to `SENSITIVE_TABLES`, and new secret columns to `BLOCKED_COLUMNS`. A test should verify these lists stay current.
