# Stats Service — Spec

Concrete contracts for the design in [`design.md`](./design.md). This doc names every public surface the implementation has to deliver: the proto, the Python client API, the on-disk shapes, and the error types. It is **not** an implementation plan; algorithm choices and sequencing belong in PRs that ship the code.

## File layout

```text
lib/finelog/src/finelog/
  proto/
    logging.proto          # existing
    stats.proto            # NEW
  rpc/                     # buf-generated stubs (already present for logging)
    stats_pb2.py           # generated
    stats_pb2.pyi          # generated
    stats_connect.py       # generated
  client/
    __init__.py            # MODIFIED: re-export LogClient
    log_client.py          # NEW: top-level LogClient
    table.py               # NEW: Table handle
    pusher.py              # existing — used by LogClient internally
    proxy.py               # existing
  server/
    asgi.py                # MODIFIED: register StatsService alongside LogService
    service.py             # existing — LogServiceImpl
    stats_service.py       # NEW: StatsServiceImpl
  store/
    duckdb_store.py        # MODIFIED: per-namespace registry + segments
    schema.py              # NEW: Schema/Column dataclasses, validation
```

## Proto: `lib/finelog/src/finelog/proto/stats.proto`

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
}

message Column {
  string name = 1;
  ColumnType type = 2;
  bool nullable = 3;
}

message Schema {
  repeated Column columns = 1;
}

// One row, ordered to match the registered Schema.columns. Missing nullable
// columns can be represented either by a Value with null=true or by absence
// (server pads to schema length).
message Value {
  oneof kind {
    string string_value = 1;
    int64 int_value = 2;
    double float_value = 3;
    bool bool_value = 4;
    int64 timestamp_ms = 5;
    bytes bytes_value = 6;
    bool null = 7;          // Explicit null marker
  }
}

message Row {
  repeated Value values = 1;
}

// ============================================================================
// STATS SERVICE
// ============================================================================

message RegisterTableRequest {
  string namespace = 1;     // e.g. "iris.worker"
  Schema schema = 2;
  bool allow_evolve = 3;    // If true, accept additive-nullable extensions of
                            // a previously-registered schema. (See Open Q in
                            // design.md — default false; behavior may change.)
}

message RegisterTableResponse {
  Schema effective_schema = 1; // The schema now in force. May differ from the
                               // requested one if allow_evolve=true and the
                               // server merged in additive columns.
}

message WriteRowsRequest {
  string namespace = 1;
  repeated Row rows = 2;
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
}

service StatsService {
  rpc RegisterTable(RegisterTableRequest) returns (RegisterTableResponse);
  rpc WriteRows(WriteRowsRequest) returns (WriteRowsResponse);
  rpc Query(QueryRequest) returns (QueryResponse);
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

    def write_batch(self, key: str, messages: Sequence[LogMessage]) -> None: ...
    def query(self, query: LogQuery) -> Sequence[LogRecord]: ...

    # --- stats-side (new) ---

    def get_table(
        self,
        namespace: str,
        schema: Schema,
        *,
        allow_evolve: bool = False,
    ) -> "Table":
        """Idempotently register `namespace` with `schema` and return a
        Table handle.

        Raises:
            SchemaConflictError: a different schema is already registered
                for `namespace` and `allow_evolve` is False (or the requested
                schema is not an additive-nullable extension).
        """
```

### `finelog.client.Table`

```python
@dataclass(frozen=True)
class Schema:
    columns: tuple[Column, ...]

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

        Common shapes accepted: dataclass instances, NamedTuple, dict, or
        any object with __getattr__ matching column names.
        """

    def query(self, sql: str) -> pa.Table:
        """Run Postgres-flavored SQL. The string `t` (or the namespace name)
        in the SQL is rewritten to the backing per-namespace Parquet path.
        Returns an Arrow table; SQL syntax is DuckDB's. Coupling to DuckDB
        syntax is deliberate (see design.md "Queries").
        """

    def close(self) -> None:
        """Flush the write buffer and release client-side resources."""
```

### Errors (in `finelog.client`)

```python
class StatsError(Exception): ...

class SchemaConflictError(StatsError):
    """Raised by LogClient.get_table() when the requested schema differs
    from the registered one and the difference is not an additive-nullable
    extension (or allow_evolve is False)."""

class SchemaValidationError(StatsError):
    """Raised by Table.write() when a row is missing a non-nullable column,
    has a type mismatch, or contains an unknown column name. Validation
    happens client-side before flush; the server re-validates and rejects
    the batch with the same error if it sees a violation (defense in depth)."""

class NamespaceNotFoundError(StatsError):
    """Raised by Table.query() when the SQL references an unregistered
    namespace."""
```

## Persisted shapes

### Schema registry (sidecar DuckDB DB)

The registry lives in a DuckDB database file in the finelog data directory. It is **not** inferred from Parquet footers on startup.

```sql
-- Path: {data_dir}/_finelog_registry.duckdb
CREATE TABLE namespaces (
    namespace        TEXT PRIMARY KEY,
    schema_json      TEXT NOT NULL,         -- JSON serialization of Schema proto
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
    tmp_{seq}_{uuid}.parquet            # in-flight (renamed from existing flat layout at startup)
    logs_{seq_lo}_{seq_hi}.parquet      # compacted segments
  iris.worker/
    tmp_{seq}_{uuid}.parquet
    logs_{seq_lo}_{seq_hi}.parquet
  iris.task/
    ...
```

**One-time migration of existing logs**: at startup, if `{data_dir}/log/` does not exist but flat `{data_dir}/{tmp,logs}_*.parquet` files do, the server takes a lock at `{data_dir}/.migration_lock` and renames the flat files into `log/`. The lock file persists if the rename is interrupted; on next startup, the recovery walk completes any partial moves before serving traffic.

### Sequence numbers

`_next_seq` is per-namespace, not global. `_recover_max_seq()` runs once per namespace at startup, walking only that namespace's segment directory. Cross-namespace queries do not require a global sequence.

## Concurrent-register and validation behavior

- **Concurrent register, identical schema**: idempotent, no-op (returns the registered schema).
- **Concurrent register, additive-nullable extension, `allow_evolve=False`**: raises `SchemaConflictError`.
- **Concurrent register, additive-nullable extension, `allow_evolve=True`**: server merges (UNION on column set, all new columns must be nullable), updates `last_modified_ms`, returns merged schema as `effective_schema`.
- **Concurrent register, non-additive change (rename, type-change, removed column)**: raises `SchemaConflictError` regardless of `allow_evolve`.
- **All registry mutations** are guarded by a process-level lock on the `namespaces` row; the DuckDB sidecar's transaction guarantees serialization.

## Endpoint and resolution

The stats service is hosted by the finelog process at the existing logical endpoint:

```text
iris://marin?endpoint=/system/log_server
```

There is no `/system/stats` endpoint. `LogClient.connect()` resolves once and dispatches both LogService and StatsService methods to the same `(host, port)`.
