# Iris Raw Query API

Tracking issue: [#3492](https://github.com/marin-community/marin/issues/3492).

## Problem

The `ControllerService` has 15+ query-oriented RPCs, each with bespoke proto messages, handler code, and pagination logic. Adding a new read pattern requires a new proto message pair, handler method, dashboard wiring, and CLI plumbing.

## Solution

A single `ExecuteRawQuery` RPC that accepts a raw SQL `SELECT` string and returns column metadata + JSON-encoded rows. Admin-only.

### Proto (`query.proto`)

```protobuf
message ColumnMeta {
  string name = 1;
  string type = 2;
}

message RawQueryRequest {
  string sql = 1;
}

message RawQueryResponse {
  repeated ColumnMeta columns = 1;
  repeated string rows = 2;  // JSON-encoded arrays
}
```

### RPC (`cluster.proto`)

```protobuf
rpc ExecuteRawQuery(iris.cluster.RawQueryRequest) returns (iris.cluster.RawQueryResponse);
```

## Safety

The executor (`lib/iris/src/iris/cluster/controller/query.py`) enforces:

- **Admin-only**: `require_identity()` + `role == "admin"` check in the service handler.
- **SELECT only**: rejects SQL not starting with `SELECT`.
- **Single statement**: rejects SQL containing `;`.
- **Keyword blocklist**: rejects `INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, `CREATE`, `ATTACH`, `DETACH`, `PRAGMA`, `VACUUM`, `REINDEX`, `SAVEPOINT`.
- **Snapshot isolation**: executes under `db.snapshot()` which rolls back on exit, so even if DML slipped through it would be discarded.
- **BLOB encoding**: binary cell values are rendered as `<blob:N bytes>` rather than leaked raw.

## CLI

```bash
iris query "SELECT * FROM jobs LIMIT 10"
iris query "SELECT user_id, count(*) FROM jobs GROUP BY user_id"
iris query -f json "SELECT job_id, state FROM jobs"
iris query -f csv "SELECT * FROM workers"
```

Implemented in `lib/iris/src/iris/cli/query.py`. Output formats: `table` (default), `json`, `csv`.

## Future Work

[#3706](https://github.com/marin-community/marin/issues/3706): Move `api_keys` and `controller_secrets` into a separate `auth.db`, attach the log store as `logs.db` via `ATTACH DATABASE`. This eliminates the need for the admin-only restriction on the query API (no secrets in the main DB) and allows querying logs via normal SQL (`SELECT * FROM logs.logs`).
