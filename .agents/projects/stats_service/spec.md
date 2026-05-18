# Stats Service — Spec

Concrete contracts for the design in [`design.md`](./design.md). This doc names every public surface the implementation has to deliver: the proto, the Python client API, the on-disk shapes, and the error types. It is **not** an implementation plan; algorithm choices and sequencing belong in PRs that ship the code.

## File layout

```text
lib/finelog/src/finelog/
  proto/
    logging.proto          # existing
    finelog_stats.proto    # NEW (renamed from `stats.proto` to avoid descriptor-pool collision with iris's existing `iris/rpc/stats.proto` (same precedent as `time.proto`).)
  rpc/                     # buf-generated stubs (already present for logging)
    finelog_stats_pb2.py     # generated
    finelog_stats_pb2.pyi    # generated
    finelog_stats_connect.py # generated
  client/
    __init__.py            # MODIFIED: re-export LogClient (LogPusher removed)
    log_client.py          # NEW: top-level LogClient + Table
    proxy.py               # existing — read-side helper, unchanged
    remote_log_handler.py  # MODIFIED: now takes (LogClient, key), no LogPusher
    # NOTE: pusher.py is DELETED. The Table-owned buffer in log_client.py is
    # the only batcher.
  server/
    asgi.py                # MODIFIED: build_log_server_asgi mounts a second Connect app for StatsService
    service.py             # existing — LogServiceImpl
    stats_service.py       # NEW: StatsServiceImpl
  store/
    duckdb_store.py        # MODIFIED: becomes the NamespaceRegistry; per-namespace state lives in LogNamespace
    log_namespace.py       # NEW: LogNamespace (per-namespace pending/segments/flush/compaction/offload)
    schema.py              # NEW: Schema/Column dataclasses, validation, Arrow-schema bridge
```

## Proto: `lib/finelog/src/finelog/proto/finelog_stats.proto`

```protobuf
edition = "2023";

package finelog.stats;

// Column type for a registered table schema. Maps 1:1 to a subset of
// pyarrow types and Postgres-flavored DuckDB types.
enum ColumnType {
  COLUMN_TYPE_UNKNOWN = 0;
  COLUMN_TYPE_STRING = 1;       // pa.string()
  COLUMN_TYPE_INT64 = 2;        // pa.int64()
  COLUMN_TYPE_FLOAT64 = 3;      // pa.float64()
  COLUMN_TYPE_BOOL = 4;         // pa.bool_()
  COLUMN_TYPE_TIMESTAMP_MS = 5; // pa.timestamp("ms")
  COLUMN_TYPE_BYTES = 6;        // pa.binary()
  COLUMN_TYPE_INT32 = 7;        // pa.int32()
}

message Column {
  string name = 1;
  ColumnType type = 2;
  bool nullable = 3;
}

message Schema {
  repeated Column columns = 1;

  // Column to sort by during compaction. Must name an existing column of
  // type INT64 or TIMESTAMP_MS. If empty, the server requires a column
  // named "timestamp_ms" of type INT64 or TIMESTAMP_MS and uses it
  // implicitly. A schema with neither is rejected at register time.
  string key_column = 2;
}

// Write batches are Arrow IPC RecordBatches on the wire — columnar, with
// the batch's schema embedded in the IPC header. There is no proto-level
// Row or Value type; the column types in the table above (ColumnType)
// describe the Arrow types the server accepts.

// ============================================================================
// STATS SERVICE
// ============================================================================

message RegisterTableRequest {
  string namespace = 1;     // e.g. "iris.worker"
  Schema schema = 2;
}

message RegisterTableResponse {
  Schema effective_schema = 1; // The schema now in force. Differs from the
                               // requested one when the server merged the
                               // request as an additive-nullable extension
                               // of a previously-registered schema, or when
                               // the request was a subset of the existing
                               // registered schema.
}

message WriteRowsRequest {
  string namespace = 1;
  // Arrow IPC stream containing exactly one RecordBatch. The batch's
  // Arrow schema must be a subset of (or equal to) the registered schema:
  // every column name must exist in the registry with a matching Arrow
  // type. Missing nullable columns are filled with NULL on append.
  // Unknown column names raise SchemaValidationError.
  bytes arrow_ipc = 2;
}

message WriteRowsResponse {
  int64 rows_written = 1;
}

message QueryRequest {
  string sql = 1;           // Postgres-flavored SQL passed to DuckDB.
                            // Tables in the FROM clause must be registered
                            // namespaces; the server resolves them to the
                            // backing per-namespace Parquet directories.
}

message QueryResponse {
  bytes arrow_ipc = 1;      // Arrow IPC stream serialization of the result.
  int64 row_count = 2;      // Number of rows in the result. Lets callers
                            // size buffers / decide whether to decode without
                            // walking the IPC stream first.
}

message DropTableRequest {
  string namespace = 1;
}

message DropTableResponse {
  // Empty. Server has removed the registry entry and deleted the local
  // segment directory. GCS-archived data for the namespace is left in place.
}

service StatsService {
  rpc RegisterTable(RegisterTableRequest) returns (RegisterTableResponse);
  rpc WriteRows(WriteRowsRequest) returns (WriteRowsResponse);
  rpc Query(QueryRequest) returns (QueryResponse);
  rpc DropTable(DropTableRequest) returns (DropTableResponse);
}
```

## Python API

### `finelog.client.LogClient` (new top-level client)

```python
class LogClient:
    """Domain client for the finelog process. Hides Connect/RPC and proto
    details; safe to import from worker code.

    Both LogService methods (write_batch, query) and StatsService methods
    (get_table) are exposed here. There is no separate StatsClient — logs
    are one namespace among many.
    """

    @staticmethod
    def connect(endpoint: str | tuple[str, int]) -> "LogClient": ...

    # --- log-side (existing semantics, lifted into LogClient) ---

    def write_batch(self, key: str, messages: Sequence[LogEntry]) -> None: ...
    def query(self, request: FetchLogsRequest) -> FetchLogsResponse: ...

    # --- stats-side (new) ---

    def drop_table(self, namespace: str) -> None:
        """Remove `namespace` from the registry and delete its local segment
        directory. GCS-archived data for the namespace is left in place;
        callers who want it deleted must clean the bucket themselves.

        Subsequent get_table on the same namespace registers it fresh —
        old archived data and the new registration are independent.

        No-op (does not raise) if the namespace was not registered.
        """

    def get_table(
        self,
        namespace: str,
        schema: type | Schema,
    ) -> "Table":
        """Idempotently register `namespace` with `schema` and return a
        Table handle.

        `schema` may be either an explicit `Schema` instance or a dataclass
        class (the common case). When a dataclass is passed, fields are
        mapped to columns in declaration order using the type-annotation
        rules in "Dataclass schema inference" below.

        Register is evolve-by-default: if `namespace` already has a
        registered schema, the server returns a Table whose `.schema` is
        the union of the requested schema and the registered one
        (additive-nullable merge). The caller's writes can use either the
        narrower or wider view.

        Raises:
            SchemaConflictError: the requested schema differs from the
                registered one in a non-additive way (rename, type change,
                or a new non-nullable column).
        """
```

### `finelog.client.Table`

```python
@dataclass(frozen=True)
class Schema:
    columns: tuple[Column, ...]
    key_column: str = ""        # empty → server uses implicit "timestamp_ms"

@dataclass(frozen=True)
class Column:
    name: str
    type: ColumnType
    nullable: bool = False

class ColumnType(StrEnum):
    STRING = "string"
    INT64 = "int64"
    FLOAT64 = "float64"
    BOOL = "bool"
    TIMESTAMP_MS = "timestamp_ms"
    BYTES = "bytes"

class Table:
    """Handle returned by LogClient.get_table(). Lifecycle: a Table owns a
    per-table client-side write buffer (parallel to LogPusher's batcher,
    not shared) that flushes on flush_interval or batch_size, whichever
    comes first. Closing the LogClient drains all open Tables.
    """

    @property
    def namespace(self) -> str: ...

    @property
    def schema(self) -> Schema: ...

    def write(self, rows: Sequence[Any]) -> None:
        """Buffer rows for write. Each row must have attributes (or dict
        keys) matching schema column names. Missing nullable columns are
        sent as null; missing non-nullable columns raise SchemaValidationError
        client-side before the buffer flush.

        Write semantics match LogPusher: a background flusher retries
        transient server failures with backoff; rows persist in the
        in-memory buffer across a finelog restart as long as the client
        process is alive. If the client process exits with rows still
        buffered, those rows are lost (acceptable for stats — the writer
        is expected to re-emit on next sample).

        Common shapes accepted: dataclass instances, NamedTuple, dict, or
        any object with __getattr__ matching column names.
        """

    def query(self, sql: str, *, max_rows: int = 100_000) -> pa.Table:
        """Run Postgres-flavored SQL. Reference namespaces by name in the
        FROM clause (e.g. `FROM "iris.worker"`); the server registers a
        DuckDB view per registered namespace before executing the query
        and never rewrites the user SQL string. Returns an Arrow table;
        SQL syntax is DuckDB's. Coupling to DuckDB syntax is deliberate
        (see design.md "Query execution").

        If the result exceeds `max_rows`, raises QueryResultTooLargeError
        rather than silently truncating. Caller can re-issue with a higher
        cap (or a `LIMIT`/aggregation in the SQL) if they really want it.
        Reads have no fallback during a finelog outage — failures surface
        as exceptions for the caller to handle.
        """

    def close(self) -> None:
        """Flush the write buffer and release client-side resources."""
```

### Errors (in `finelog.client`)

Stage 2 ships these in `finelog.store.schema`; Stage 4 moves them to `finelog.client` alongside the new client surface.

```python
class StatsError(Exception): ...

class SchemaConflictError(StatsError):
    """Raised by LogClient.get_table() when the requested schema differs
    from the registered one in a non-additive way: rename, type change,
    or a new non-nullable column. Additive-nullable differences are
    merged silently and do not raise."""

class SchemaValidationError(StatsError):
    """Raised by:
      - Table.write() when a row is missing a non-nullable column, has a
        type mismatch, or contains an unknown column name. Validation
        happens client-side before flush; the server re-validates and
        rejects the batch with the same error (defense in depth).
      - LogClient.get_table() when the schema is structurally invalid:
        unsupported dataclass field type, or no ordering key (neither
        an explicit Schema.key_column nor a TIMESTAMP_MS column named
        'timestamp_ms')."""

class NamespaceNotFoundError(StatsError):
    """Raised by Table.query() when the SQL references an unregistered
    namespace."""

class QueryResultTooLargeError(StatsError):
    """Raised by Table.query() when the result row count exceeds `max_rows`.
    Caller should add a LIMIT, aggregate further, or pass a higher cap."""

class InvalidNamespaceError(StatsError):
    """Raised by LogClient.get_table() (and DropTable) when the namespace
    name does not resolve to a path strictly inside the finelog data dir,
    e.g. it contains '..' or absolute components. The check is path-
    containment: (data_dir / namespace).resolve() must be a subdir of
    data_dir.resolve(). No regex beyond what that check implies."""
```

### Dataclass schema inference

When `LogClient.get_table(namespace, schema=SomeDataclass)` is called with a dataclass class, fields are mapped to columns in declaration order:

| Annotation | `ColumnType` | `nullable` |
|---|---|---|
| `str` | `STRING` | `False` |
| `int` | `INT64` | `False` |
| `float` | `FLOAT64` | `False` |
| `bool` | `BOOL` | `False` |
| `datetime` | `TIMESTAMP_MS` | `False` (microseconds truncated) |
| `bytes` | `BYTES` | `False` |
| `T \| None` (or `Optional[T]`) | as `T` | `True` |

Dataclasses with unsupported field types (collections, nested dataclasses, custom classes) raise `SchemaValidationError` at `get_table` time, not at first write. Construct an explicit `Schema` if you need finer control than the inference gives you.

The inferred schema's `key_column` defaults to `""` (empty). To pick a non-default key column, declare it as a `ClassVar[str]` on the dataclass:

```python
from typing import ClassVar
from datetime import datetime

@dataclass
class WorkerStat:
    key_column: ClassVar[str] = "ts"   # opts out of the implicit "timestamp_ms"
    worker_id: str
    ts: datetime
    mem_bytes: int
```

`ClassVar` is idiomatic for schema-level metadata and doesn't appear as a column. The inference code reads `getattr(cls, "key_column", "")`.

When `key_column` is empty, the server requires a column named `timestamp_ms` of type `INT64` or `TIMESTAMP_MS` (Python `int` annotated as `timestamp_ms` works; so does `datetime`). A dataclass with neither a `timestamp_ms` field nor a `key_column` ClassVar raises `SchemaValidationError` at `get_table` time.

`datetime` columns store at millisecond precision; sub-millisecond components of a Python `datetime` are silently truncated. If a namespace ever needs microsecond accuracy we add a `TIMESTAMP_US` `ColumnType`; until then this is documented and accepted.

## Persisted shapes

### Schema registry (sidecar DuckDB DB)

The registry lives in a DuckDB database file in the finelog data directory. It is **not** inferred from Parquet footers on startup.

```sql
-- Path: {data_dir}/_finelog_registry.duckdb
CREATE TABLE namespaces (
    namespace        TEXT PRIMARY KEY,
    schema_json      TEXT NOT NULL,         -- JSON serialization of Schema proto (includes key_column)
    key_column       TEXT NOT NULL,         -- Resolved ordering key (explicit or implicit "timestamp_ms")
    registered_at_ms BIGINT NOT NULL,
    last_modified_ms BIGINT NOT NULL        -- Updated on additive evolution
);
```

`schema_json` uses the Schema proto's standard JSON encoding, so the registry can be inspected with `duckdb` directly without finelog code.

### Per-namespace Parquet layout

```text
{data_dir}/
  _finelog_registry.duckdb
  log/
    tmp_{seq:019d}.parquet               # in-flight (renamed from flat layout at startup)
    logs_{seq:019d}.parquet              # sealed/compacted segments
  iris.worker/
    tmp_{seq:019d}.parquet
    logs_{seq:019d}.parquet
  iris.task/
    ...
```

Namespace names must resolve to a subdirectory strictly inside `{data_dir}` — `(data_dir / namespace).resolve()` must be a subdir of `data_dir.resolve()`. Names containing `..` or absolute components raise `InvalidNamespaceError` at register time. No regex beyond that path-containment check.

**GCS upload layout** mirrors the local layout: `{remote_log_dir}/{namespace}/{filename}`. `_offload_to_gcs` (`duckdb_store.py:860`) is updated to include the namespace in the upload path; recovery from GCS reads the same per-namespace prefix. Future direction: Hive-style daily partitioning (`{remote_log_dir}/{namespace}/dt=YYYY-MM-DD/`); v1 keeps a flat per-namespace layout.

**Storage caps** stay global (`DEFAULT_MAX_LOCAL_SEGMENTS` / `DEFAULT_MAX_LOCAL_BYTES` at `duckdb_store.py:122`) — one budget shared across all namespaces. Eviction drops the oldest segments first, regardless of namespace. Per-namespace quotas are deferred until evidence of starvation forces them.

**Namespace deletion** removes the registry row and deletes the local segment directory. GCS-archived data is *not* deleted by drop — it is the caller's responsibility to clean up bucket contents if desired. A subsequent `get_table` on the same namespace registers it fresh; the new registration shares no state with the old archived data.

**One-time migration of existing logs**: see `design.md` "Migration: flat → per-namespace layout" for the full state machine. The sentinel file is `{data_dir}/.layout-migration` (not `.migration_lock`); it carries a JSON `state` field (`in-progress` or `done`) and the migration walk is idempotent so a crashed run resumes cleanly.

### Sequence numbers

`_next_seq` is per-namespace, not global. `_recover_max_seq()` runs once per namespace at startup, walking only that namespace's segment directory. Cross-namespace queries do not require a global sequence.

### Compaction across schema versions

Segments within a namespace may have been written under different (additively-evolved) schemas. Compaction reads them with `union_by_name` and projects the result to the *currently-registered* schema before re-writing the compacted segment. Newer columns missing from old segments come back as NULL. The compacted segment carries the current registered schema's column set.

Namespaces may declare a non-key compaction order. The `log` namespace orders by `(key, seq)` to preserve per-key read locality for `get_logs`. Other namespaces order by their declared `key_column`.

## Concurrent-register and validation behavior

Register is evolve-by-default; there is no opt-in flag.

- **Identical schema**: idempotent, no-op (returns the registered schema as `effective_schema`).
- **Subset (caller's columns ⊆ registered)**: accepted, registered schema unchanged. The Table's `.schema` reflects the registered (wider) one. The caller's narrower writes are still valid — missing columns serialize as NULL.
- **Additive-nullable extension (caller's columns ⊃ registered, all extras are nullable)**: server merges (UNION on column set), updates `last_modified_ms`, returns the merged schema as `effective_schema`.
- **Non-additive change** — rename, type change, or a new non-nullable column: raises `SchemaConflictError`. The migration path is "register a new namespace and dual-write."
- **All registry mutations** are guarded by a process-level lock on the `namespaces` row; the DuckDB sidecar's transaction guarantees serialization.

## Endpoint and resolution

The stats service is hosted by the finelog process at the existing logical endpoint:

```text
iris://marin?endpoint=/system/log_server
```

There is no `/system/stats` endpoint. `LogClient.connect()` resolves once and dispatches both LogService and StatsService methods to the same `(host, port)`.
