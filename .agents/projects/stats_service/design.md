# Stats Service

_Why are we doing this? What's the benefit?_

Iris emits operational stats — worker heartbeats, container utilization, scheduling decisions — across three uncoordinated places: in-memory RPC counters (`lib/iris/src/iris/rpc/stats.proto:21`), live-computed user/job/task rollups (`lib/iris/src/iris/cluster/controller/db.py:192`), and ephemeral worker heartbeats (`lib/iris/src/iris/cluster/worker/worker.py:67`). None are queryable historically. The dashboard's worker pane reads worker roster state from the controller's sqlite via `_worker_roster_cached()` / `list_workers()` in `lib/iris/src/iris/cluster/controller/service.py`, which couples the dashboard's data model to controller restarts and makes "what was this worker doing yesterday" effectively unanswerable.

This adds a small **stats service** co-hosted in the existing finelog process: typed, schema-registered tables that callers write rows into and query with SQL. It also sets up moving the dashboard's worker pane off the controller sqlite onto a service that outlives any single controller restart.

## Background

File refs, prior-art comparison, and Q&A summary live in [`research.md`](./research.md); concrete contracts (proto, public API, persisted shapes) in [`spec.md`](./spec.md). The load-bearing change is generalizing the DuckDB backend's hardcoded log row schema (`lib/finelog/src/finelog/store/duckdb_store.py:79`) into a per-namespace schema registry. Co-hosting inside finelog keeps the operational footprint small and shares the storage layer; we revisit splitting if stats traffic ever dominates.

## Challenges

The schema model is the load-bearing call. Per-namespace registered schemas with typed columns sit between schema-on-read (typing bugs at query time) and one-table-per-signal (heavier evolution); they need `DuckDBLogStore` to grow a schema registry. Once that lands, logs are one namespace among many.

## Non-goals

The stats service is scoped to *post-hoc query of typed, consistent-schema time-series*. Out of scope for v1:

- **Server-side rollups, materialized views, or aggregation engines.** If a query gets expensive, callers can write a periodic job that reads raw rows and writes summary rows into a separate namespace. We may surface DuckDB-builtin views later as an optional optimization.
- **Rich metric types** (`counter.inc()`, histograms, gauges with semantics). Schema is plain typed columns; if histograms become common we add them later.
- **Hot-path event counters.** Zephyr counters stay in Zephyr; per-increment writes don't belong here. Final/summary stats from a job *do* belong here.
- **Query builder DSL.** `Table.query(sql: str)` takes raw Postgres-flavored SQL. Typed query builders are unstable to design ahead of demand; we add them only once a clear boundary emerges.

## Costs / Risks

- Storage-layer coupling: segment recovery (`duckdb_store.py:182`), compaction (`:728`), and GCS offload (`:860`) all assume a single global namespace and one schema. The refactor encapsulates per-namespace state in a new `LogNamespace` object (see "Storage architecture" below) so each namespace owns its own copy of this state.
- Eviction-across-tenants: a noisy stats namespace can evict log segments from local storage and degrade `FetchLogs` queryability for a window. We accept this tradeoff for v1 because we don't have multiple high-volume namespaces yet; per-namespace quotas land if it becomes real. This is a product tradeoff, not just an implementation detail.
- Cardinality is unbounded without per-namespace TTL; worker stats at 1Hz reach ~150GB/year. Ship without TTL, revisit when a namespace nears storage caps.
- The dashboard's worker pane gains a stats-service-down failure mode that today's controller-coupled path doesn't have. Mitigation under "Availability" below.
- Diagnostic queries can return arbitrarily large result sets. Default per-query row cap (`spec.md`); abusive callers get a hard error, not silent truncation.

## Design

A new `StatsService` proto in `lib/finelog/src/finelog/proto/stats.proto`, co-hosted with `LogService` on the same finelog process. Co-hosting is a small refactor of `build_log_server_asgi()` (`lib/finelog/src/finelog/server/asgi.py:42`), which today mounts exactly one Connect app for `LogService`; we extend it to mount a second app for `StatsService` at the same endpoint. Public API stays on `LogClient`:

```python
@dataclass
class WorkerStat:
    worker_id: str
    mem_bytes: int
    cpu_pct: float
    note: str | None = None     # nullable column

client = LogClient.connect("iris://marin?endpoint=/system/log_server")
table = client.get_table("iris.worker", schema=WorkerStat)           # dataclass class drives schema inference
table.write([WorkerStat(worker_id="w-1", mem_bytes=...)])            # instances are the row payload
rows = table.query("SELECT worker_id, AVG(mem_bytes) ...")           # returns pa.Table
```

### Storage architecture

The current `DuckDBLogStore` holds all per-namespace state at the top level: `_pending`, `_chunks`, `_flushing`, `_local_segments`, `_next_seq`, the flush thread, the compaction connection, and the GCS offload state. Per-namespace tables means redesigning that, not just adding a registry on top.

The refactor introduces a `LogNamespace` object — one instance per registered namespace — that owns:

- the in-memory pending rows and chunked flushable batches
- the local segment list (oldest → newest), sequence counter
- a dedicated background flush thread
- compaction state (sealed-segment threshold, last-compacted seq)
- GCS offload state (last-uploaded seq, retry backoff)
- a cached copy of the registered Arrow schema and key column

`DuckDBLogStore` becomes a thin **`NamespaceRegistry`** that holds the schema sidecar DB, instantiates `LogNamespace` lazily on startup or on first write, routes RPCs by namespace, and owns the two global semaphores (see "Concurrency" below). One DuckDB compaction connection is shared across namespaces because compactions are serialized by the insertion semaphore.

Per-namespace rate limiters are marked-run at namespace construction so the first bg tick is a quiet wait — otherwise `should_run()` returns true on the first iteration and a freshly-constructed namespace immediately compacts/flushes.

The registry persists in a sidecar DuckDB DB in the finelog data dir and rehydrates on startup. Each `LogNamespace` calls its own `_recover_max_seq` walking only its own segment directory (`{data_dir}/{namespace}/`). Namespace count is small (tens), so sequential startup recovery is fine.

### Concurrency

Two global locks live on `NamespaceRegistry`:

- **Insertion mutex** — held by every `WriteRows` call and every compaction step's slow phase (read inputs, write `.tmp`). Serializes appends and compactions across all namespaces.
- **Query-visibility rwlock** — queries hold the read side for their full duration (including `fetch_arrow_table()`). Compaction's commit step (rename `.tmp` → final, unlink inputs) holds the write side. `DropTable`'s segment-directory delete holds the write side. Multiple queries run concurrently; commits and drops briefly block on in-flight queries.

The locks are orthogonal: writes don't block queries and vice versa. Today's `_segments_rwlock` in `duckdb_store.py` provides exactly this guarantee for the log namespace; the new design generalizes it to "across all namespaces."

**`DropTable` lifecycle** is the trickiest piece. The race we have to prevent: a `WriteRows` looks up the `LogNamespace`, gets a reference to it, then waits on the insertion mutex while `DropTable` runs ahead of it. After `DropTable` releases, the write would proceed against a deleted directory.

The fix: namespace lookup is itself guarded by the insertion mutex. `WriteRows` acquires the mutex *first*, then looks up the namespace from the registry. If the registry row is gone, the lookup raises `NamespaceNotFoundError` and the write fails cleanly. This means namespace lookup is on the hot path — but registry reads are O(1) hashmap lookups, so the cost is negligible.

`DropTable` order of operations. Holding both locks end-to-end deadlocks because the bg flush thread itself takes the insertion mutex on every iteration; joining it under the mutex would never return. Instead:

1. Acquire the insertion mutex.
2. Verify the namespace exists; remove the registry row and the in-memory dict entry. Subsequent `WriteRows` lookups under the same mutex fail with `NamespaceNotFoundError`.
3. Release the insertion mutex.
4. Stop the flush thread without flushing; in-memory data is discarded by design — a write captured under the insertion mutex but not yet appended evaporates. This is the explicit drop contract: stats data in flight at the moment of drop is not durable.
5. Acquire the query-visibility write lock. In-flight queries drain.
6. Delete the local segment directory.
7. Release the write lock.

Drop on the privileged `log` namespace is rejected with `InvalidNamespaceError`. Drop does not delete remote (GCS-archived) objects — see "Namespace deletion" below.

Per-namespace state inside `LogNamespace` is not separately locked — the global locks serialize all access. Each namespace also holds a small per-namespace flush lock that prevents the bg flush thread and any explicit `flush()` call from racing on the same `tmp_{seq}.parquet` filename derivation. This is independent of the global locks and exists because the filename is derived from per-namespace `_next_seq`. The throughput cap (one writer at a time across all namespaces) is acceptable at expected aggregate rates (single-digit kHz); we revisit if real load pushes it.

### Wire format

Write batches are Arrow IPC RecordBatches (`bytes arrow_ipc` in `WriteRowsRequest`, see `spec.md`) — columnar, not row-by-row. Each batch carries its own schema in the IPC header, so the server validates by comparing Arrow schemas against the registered one rather than walking per-row maps. Columnar also gives us null bitmaps and per-column compression for free, and matches `QueryResponse`'s shape so the read and write paths are symmetric. A writer that knows about a subset of columns sends a RecordBatch with only those columns; the server fills missing nullable columns with NULL on append. Unknown column names are rejected. The columnar shape (rather than ordinal rows) avoids the correctness bug where two writers agree on column *names* but disagree on order after an additive merge.

**Resource limits**:

- Max request size: 16 MiB (gRPC default; the proto carries one IPC stream per request and we don't want a single misbehaving caller to balloon server memory). Larger logical batches must be split client-side; the `Table` buffer respects this implicitly via its 16 MiB queue cap.
- Max rows per RecordBatch: 1,000,000. Above this we reject with `SchemaValidationError` to keep server-side decode bounded; the `Table` buffer flushes well below this in practice.

**Dictionary-encoded columns**: PyArrow may produce dictionary arrays for low-cardinality strings (e.g. category-typed columns). A dictionary-encoded column does *not* structurally compare equal to `pa.string()` even though it logically is. The server policy is to **decode dictionary arrays to their value type before schema comparison**, accepting them transparently. The decode cost is paid once per batch on the server; the wire savings of dictionary encoding are still realized over the network. Reject only nested-list / struct / union types; those are out of scope and raise `SchemaValidationError`.

### Schemas and ordering keys

Every registered schema must declare an **ordering key column** — the column compaction sorts by, equivalent to today's `epoch_ms` for logs. The schema either:

- explicitly names a key column (`Schema.key_column = "ts"`), which must exist in the schema with type `INT64` or `TIMESTAMP_MS`, or
- omits `key_column`, in which case the registry requires a column named `timestamp_ms` of type `INT64` *or* `TIMESTAMP_MS` and uses it as the implicit key. (Both are accepted because some callers carry epoch-ms ints directly without round-tripping through `datetime`.)

A schema declaring neither (no `key_column` and no `timestamp_ms` column) is rejected at register time. The `log` namespace declares `epoch_ms` (INT64) as its explicit key, preserving today's compaction order.

When inferring a schema from a dataclass, the caller can override the implicit-`timestamp_ms` rule with a `ClassVar`:

```python
from typing import ClassVar
from datetime import datetime

@dataclass
class WorkerStat:
    key_column: ClassVar[str] = "ts"   # opts out of the timestamp_ms default
    worker_id: str
    ts: datetime
    mem_bytes: int
```

`key_column` as a `ClassVar` is idiomatic for "schema-level, not row-level metadata"; it doesn't appear as a column. A dataclass with neither a `timestamp_ms` field nor a `key_column` ClassVar raises `SchemaValidationError` at `get_table` time.

The key column is recorded in the registry alongside the schema and never changes for a namespace. A "schema evolution that changes the key" is treated as a non-additive change — register a new namespace and dual-write.

### Namespace names

Validated at register time. Namespaces must match `^[a-z][a-z0-9_.-]{0,63}$` — lowercase ASCII alphanumerics, underscore, dot, hyphen, starting with a letter, max 64 chars. The regex is restrictive enough to be safe as both a directory name and a double-quoted DuckDB identifier without further escaping; path-containment-alone is *not* sufficient because names like `a"b` would still resolve to a subdir but break the `CREATE VIEW "{ns}"` SQL.

In addition, the resolved path must be strictly inside the data dir — `(data_dir / namespace).resolve()` is a subdir of `data_dir.resolve()` — as a defense-in-depth check against symlink games. Invalid names raise `InvalidNamespaceError`.

The existing log namespace's flat parquet files migrate into `{data_dir}/log/` on first startup; see "Migration: flat → per-namespace layout" below for the state machine. Row layout is unchanged.

### Schema enforcement

The server validates every batch against the registered schema by comparing the IPC RecordBatch's Arrow schema against the registry's. A batch missing a nullable column is accepted (NULL on append); a batch missing a non-nullable column, with an unknown column name, or with a type mismatch is rejected. Validation lives server-side specifically because trusting the client risks namespace corruption from a misbehaving worker. Schema lookup is in-memory after first registration; validation cost is one Arrow-schema comparison per batch, not per row.

### Schema evolution

Register is evolve-by-default. A caller's schema that adds nullable columns to the registered one is silently merged; the registry stores the union and `RegisterTableResponse.effective_schema` returns it. A caller whose schema is a *subset* of the registered one is also accepted as-is (older clients during a rolling upgrade write rows with NULL for the columns they don't know about). Non-additive changes — rename, type change, or a new non-nullable column — are rejected with `SchemaConflictError`; the migration path is "register a new namespace (`iris.worker.v2`), dual-write through a transition, retire the old". DuckDB's `union_by_name` handles the cross-segment read. No migration tooling; namespace bumps are caller-driven.

A caller who needs strict-equality guarantees (e.g. a test) can compare `Table.schema` against their requested schema after `get_table` and assert.

### Compaction

Each `LogNamespace` triggers compaction when its sealed-segment count exceeds a threshold. The compaction step:

1. Acquires the insertion semaphore.
2. Snapshots the list of sealed segments to compact (by sequence range).
3. Inspects each input segment's Parquet footer schema (`pyarrow.parquet.read_schema(path)`) to compute the union of columns present across all inputs. This matters because `union_by_name=true` only fills NULL for a column missing from *some* segments; a column missing from *all* compaction inputs causes DuckDB to error with `Referenced column "X" not found`. The compaction projection synthesizes those columns explicitly.
4. Issues to the shared compaction connection:

   ```sql
   COPY (
       SELECT
           {col1_expr},        -- "col1" if col1 is present in any input segment,
           {col2_expr},        -- "NULL::TYPE2 AS col2" if absent from all inputs.
           ...,
           {colN_expr}
       FROM read_parquet([{seg_paths}], union_by_name=true)
       ORDER BY {key_column}
   ) TO '{compacted_path}.tmp' (FORMAT PARQUET, COMPRESSION ZSTD);
   ```

   The column list is the *currently-registered* schema in registered order — never `SELECT *`. `key_column` is the namespace's declared ordering key. For columns that exist in at least one input, `union_by_name=true` fills NULL where missing per-row. For columns that exist in zero inputs (an additive evolution that hasn't yet been written by anyone whose data we're now compacting), we emit `NULL::TYPE AS col` to give DuckDB the correct typed null.

5. Acquires the query-visibility write lock briefly (see "Concurrency" below), atomically renames `{compacted_path}.tmp` to its final `logs_{seq_lo:019d}.parquet` name, and unlinks the input segments.
6. Releases both locks.

Step 1's insertion mutex is the load-bearing serialization across namespaces — without it, two namespaces' bg flush threads could hit the shared compaction connection concurrently. Today's single-namespace finelog does not need it for correctness; the lock matters once Stage 2+ admits multiple namespaces.

For the `log` namespace, this matches today's compaction (`ORDER BY epoch_ms`) with explicit projection and missing-column synthesis added.

Namespaces may declare a non-key compaction order. The `log` namespace orders by `(key, seq)` to preserve per-key read locality for `get_logs`. Other namespaces order by their declared `key_column`.

The compaction-input list is never empty by construction (compaction is gated on a sealed-segment count threshold). DuckDB rejects `read_parquet([])`, so an explicit empty-list guard is unnecessary here but is necessary in the query path — see below.

### Query execution

Queries do not rewrite SQL strings. The server registers a DuckDB view per registered namespace on a fresh connection at query time. There are three subtleties that have to be right:

1. **`CREATE VIEW` does not accept prepared parameters in DuckDB 1.5.** We must inline the path list as a SQL literal. Paths are server-built (`{data_dir}/{namespace}/{filename}`) and namespace + filename are validated, so escaping is straightforward: SQL-escape any `'` in each path defensively and emit a literal `[...]` list.
2. **Empty path lists fail.** `read_parquet([], union_by_name=true)` is rejected by DuckDB (it can't infer a schema from nothing). Namespaces with zero sealed segments must still get a typed empty view so user queries against them return zero rows instead of erroring. We synthesize the registered schema as a `SELECT NULL::TYPE AS col, ... WHERE FALSE`.
3. **The query-visibility lock must be held through the entire query, including `fetch_arrow_table()`** — not just during view setup. DuckDB opens Parquet files lazily during execution; releasing the lock after `CREATE VIEW` would let compaction unlink those files mid-scan.

Concretely:

```python
con = duckdb.connect()
with registry.query_lock.read():           # held until fetch returns
    for ns_name, ns in registry.namespaces.items():
        ns_quoted = _sql_quote_ident(ns_name)        # doubles any embedded "
        paths = ns.sealed_segments()                 # snapshot under the read lock
        if paths:
            paths_literal = "[" + ", ".join(_sql_quote_lit(str(p)) for p in paths) + "]"
            con.execute(
                f'CREATE VIEW {ns_quoted} AS '
                f'SELECT * FROM read_parquet({paths_literal}, union_by_name=true)'
            )
        else:
            cols_sql = ", ".join(f"NULL::{duckdb_type_for(c)} AS {_sql_quote_ident(c.name)}"
                                 for c in ns.schema.columns)
            con.execute(f'CREATE VIEW {ns_quoted} AS SELECT {cols_sql} WHERE FALSE')
    result = con.execute(user_sql).fetch_arrow_table()
```

Caller SQL references namespaces by name (`SELECT * FROM "iris.worker"`); DuckDB handles identifier quoting, CTEs, subqueries, multi-namespace joins. The server never substitutes strings into user SQL.

**Concurrency on segment visibility**: the "query semaphore" of earlier drafts is more precisely a single `query_visibility` rwlock. Queries acquire the read side for their full duration. Compaction's commit step (rename + unlink) and `DropTable`'s segment-dir delete acquire the write side. Multiple queries run concurrently; commits and drops wait briefly for in-flight queries to drain. The insertion semaphore stays a plain mutex.

Properties:

- **Lock held through fetch.** Segment paths can't be unlinked under a running query because the rwlock blocks the commit-step writer.
- **Sealed-only.** Tmp (in-flight) segments are excluded; queries see flushed and compacted data, not the in-memory buffer.
- **Per-query connection.** No connection pooling in v1; cost is dominated by Parquet scan, not connect.
- **Two query flavors.** Exact lookups — "fetch logs for this set of tasks" — keep the existing `LogClient.query(LogQuery)` shape: typed filters, no SQL surface, no DuckDB exposure. Diagnostic queries go through `table.query(SQL)`. Coupling diagnostic SQL to DuckDB syntax is a deliberate trade-off; a future backend migration only updates dashboard code, not callers' write code.

### Endpoint

Stays `/system/log_server`; logs and stats share the process under one logical name.

### Storage caps and eviction

Global cap, not per-namespace. We keep `DEFAULT_MAX_LOCAL_SEGMENTS` / `DEFAULT_MAX_LOCAL_BYTES` (`duckdb_store.py:122`) as a single budget across all namespaces. Eviction runs on the registry: it walks every `LogNamespace`'s segment list, picks the globally-oldest sealed segment (by sequence + namespace registration time as tiebreak), and drops it from local disk; the namespace's own `_local_segments` is updated. This is a product tradeoff — a noisy stats namespace can evict log segments and degrade local `FetchLogs` queryability. We accept it for v1 because we don't have multiple high-volume namespaces yet; per-namespace quotas land if it becomes real.

### GCS layout

Per-namespace prefix on the remote bucket (`{remote_log_dir}/{namespace}/{filename}`), mirroring the local layout. `_offload_to_gcs` (`duckdb_store.py:860`) is updated to include the namespace in the upload path; recovery from GCS reads the same per-namespace prefix. Future direction: Hive-style partitioning (`{remote_log_dir}/{namespace}/dt=YYYY-MM-DD/`) once a namespace's daily volume justifies it; v1 keeps a flat per-namespace layout.

GCS offload runs only from `_compaction_step`, not from `_flush_step`. `tmp_*.parquet` files are not offloaded; only sealed (compacted-or-flushed-final) segments make it to GCS.

### Namespace deletion

In scope for v1. `LogClient.drop_table(namespace)` removes the registry entry, deletes the local segment directory, and refuses subsequent reads/writes against that namespace. Archived data already pushed to GCS is left in place — the contract is "drop frees local capacity; the historical record on the bucket is your problem to clean up if you want it gone." A subsequent `get_table` on the same namespace registers it fresh.

### Datetime precision

`datetime` columns are stored at millisecond precision (`pa.timestamp("ms")`). Python `datetime` carries microseconds, which are silently truncated. Documented in the spec; if microsecond precision becomes load-bearing for a namespace we add a `TIMESTAMP_US` column type.

Datetime values are stored as tz-naive `pa.timestamp("ms")`. Callers should
normalize to UTC before storage (`dt.astimezone(timezone.utc).replace(tzinfo=None)`).
Server-side queries that compare against `now()` must explicitly cast: `(now() AT
TIME ZONE 'UTC')::TIMESTAMP`. Otherwise DuckDB's session timezone causes a
non-UTC controller to filter with a wrong window.

### Client architecture: LogClient subsumes LogPusher

`LogPusher` is removed in this change. The new `LogClient` is the only write path:

```python
class LogClient:
    @staticmethod
    def connect(endpoint: str | tuple[str, int]) -> "LogClient": ...

    # stats-side
    def get_table(self, namespace: str, schema: type | Schema) -> Table: ...
    def drop_table(self, namespace: str) -> None: ...

    # log-side: sugar over the "log" namespace
    def write_batch(self, key: str, messages: Sequence[LogMessage]) -> None: ...
    def query(self, q: LogQuery) -> Sequence[LogRecord]: ...

    def close(self) -> None: ...
```

Internally, every write goes through a `Table` — `write_batch` wires the `log` namespace's table. There is no separate batcher class. Each `Table` owns:

- a bounded in-memory queue (default: 10k rows or 16 MiB, whichever first), oldest-drop on overflow (matching today's `LogPusher` overflow policy)
- a background flush thread that flushes on size threshold, time interval (default 1s), or explicit `flush()`
- retry/backoff on transient server failures, with resolver invalidation on connection-refused (matching today's `LogPusher` behavior)
- `close()` drains the queue; closing the `LogClient` drains all open tables and joins all flush threads

`RemoteLogHandler` is rewritten to take a `LogClient` and a `key`:

```python
class RemoteLogHandler(logging.Handler):
    def __init__(self, client: LogClient, key: str): ...
```

Existing call sites (worker, controller, tests) update in one pass — no compat shim, no kept-around `LogPusher` class. Tests that pass a fake `LogPusher` switch to a fake `LogClient`. The breaking change is contained to direct `LogPusher` constructors and `RemoteLogHandler(LogPusher, ...)` callers; both are in-repo and migrated together with this PR.

`LogServiceProxy` (the read-side helper) stays as-is — it does not own write state and is unrelated to the batching consolidation.

### Migration: flat → per-namespace layout

When the store is initialized with `data_dir`, finelog's flat layout (`{data_dir}/tmp_*.parquet`, `{data_dir}/logs_*.parquet`) is migrated once into `{data_dir}/log/`. Migration runs synchronously inside `run_log_server` *before* `LogServiceImpl` is constructed and *before* uvicorn binds the port; the server simply does not accept ASGI traffic until the function returns. No separate health-check gating is needed — the kernel hasn't accepted the listen socket yet. Multi-minute migrations are acceptable: a TCP `connect` that races startup either fails or blocks until uvicorn is up, which is the same observable behavior as a slow boot.

**Sentinel.** A single file `{data_dir}/.layout-migration` encodes state as one JSON object on one line: `{"version": 1, "state": "<state>", "started_at": <epoch_ms>, "finished_at": <epoch_ms|null>}`. States: `in-progress`, `done`. Absence of the file means "unknown — inspect dir." `not-needed` and `done` collapse to the same on-disk representation (sentinel present, `state=done`); a crash mid-migration leaves the sentinel in `state=in-progress`, and the next boot resumes the same idempotent walk — no separate `partial-recovery` state.

**Fast path.** First action on startup: `os.stat("{data_dir}/.layout-migration")`. If it exists and parses to `state=done`, return immediately — no directory scan, no per-file stat. This is the steady-state cost forever after.

**Cold-start decision.** If the sentinel is missing:
- If `{data_dir}/log/` does not exist *and* no `tmp_*.parquet` / `logs_*.parquet` exist directly under `{data_dir}` (`glob` on the two prefixes only — `O(matches)`, not full readdir), this is a fresh install. Create `{data_dir}/log/`, write sentinel `state=done`, return.
- Otherwise migration is needed: write sentinel `state=in-progress` (via `os.replace` of a sibling tmp file for atomicity), then run the walk below.

If the sentinel exists with `state=in-progress`, the prior process crashed mid-migration; re-run the same walk. The walk is fully idempotent.

**Atomic move protocol.** Each segment is moved with `os.rename(src, dst)` where `src = {data_dir}/{name}` and `dst = {data_dir}/log/{name}`. Filenames embed `min_seq` (`tmp_{seq:019d}.parquet`, `logs_{seq:019d}.parquet`) and are unique within the `log` namespace, so a destination collision means a half-finished prior run already moved it. `os.rename` is atomic on POSIX local filesystems within one mount, the only supported deployment for `data_dir` (asserted at startup; see edge cases). We do **not** journal individual moves — the source/destination pair *is* the journal. A crash leaves each file either fully at `src` or fully at `dst`.

**Pre-flight checks** (fail fast, before any move):
1. `{data_dir}/log/` exists as a non-directory → raise `RuntimeError`. Refuse to guess.
2. `{data_dir}/log/` and `{data_dir}` are on different filesystems (`os.stat(...).st_dev` differs) → raise. Cross-mount rename is not atomic and is not supported.
3. `mkdir({data_dir}/log/, exist_ok=True)`.

**Walk.**

```python
def migrate_to_namespaced_layout(data_dir: Path) -> None:
    sentinel = data_dir / ".layout-migration"
    if _read_sentinel(sentinel) == "done":
        return

    log_dir = data_dir / "log"
    if log_dir.exists() and not log_dir.is_dir():
        raise RuntimeError(f"{log_dir} exists but is not a directory")

    flat = sorted(data_dir.glob("tmp_*.parquet")) + sorted(data_dir.glob("logs_*.parquet"))
    if not flat and not log_dir.exists():
        log_dir.mkdir(parents=True, exist_ok=True)
        _write_sentinel(sentinel, state="done")
        return

    log_dir.mkdir(parents=True, exist_ok=True)
    _assert_same_filesystem(data_dir, log_dir)
    _write_sentinel(sentinel, state="in-progress")

    logger.info("layout migration: %d flat segments to move into %s", len(flat), log_dir)
    moved = skipped = 0
    for i, src in enumerate(flat, start=1):
        dst = log_dir / src.name
        if dst.exists():
            # Prior crashed run already moved this one. Source is the duplicate
            # to discard; sizes must match or we abort loudly.
            if src.stat().st_size != dst.stat().st_size:
                raise RuntimeError(f"size mismatch on resume: {src} vs {dst}")
            src.unlink()
            skipped += 1
        else:
            os.rename(src, dst)
            moved += 1
        if i % 500 == 0:
            logger.info("layout migration: %d/%d processed", i, len(flat))

    _write_sentinel(sentinel, state="done")
    logger.info("layout migration: complete (moved=%d skipped=%d)", moved, skipped)
```

`_write_sentinel` writes to `{sentinel}.tmp` and `os.replace`s it onto the final path so readers never see a partial line.

**Other-file policy.** Files under `data_dir` that do not match `tmp_*.parquet` or `logs_*.parquet` (DuckDB metadata, hidden files, the sentinel itself) are left in place and not logged per-file — only the two glob patterns are migrated. This is intentional: only the `log` namespace has flat files, and any other namespace will be created post-migration directly under its own subdir.

**Stale tmp segments.** Unfinalized `tmp_*.parquet` files are migrated as-is. Compaction handles them after startup exactly as today (`_discover_segments` reads from `log_dir`); migration does not interpret them.

**ENOSPC / unexpected I/O error.** `os.rename` within one filesystem does not consume free space, so ENOSPC should not occur. If any rename raises, the exception propagates, the sentinel stays `in-progress`, and the next boot resumes the walk. We do not catch.

**Observability (INFO).**
- `layout migration: N flat segments to move into <log_dir>` at start.
- `layout migration: i/N processed` every 500 files.
- `layout migration: complete (moved=M skipped=K)` at end.
- On fast-path boots: nothing.

**Done state.** Sentinel: `{"version": 1, "state": "done", "started_at": ..., "finished_at": ...}`. Subsequent startups stat the sentinel, parse one line, and return in microseconds.

### Availability

The service runs on a single VM with health-checks and Docker auto-restart — same operational posture as logs today. We do not replicate. The contract callers see is "available almost always; tolerate transient outages." Concretely:

- **Writes** use the per-`Table` in-memory buffer described in "Client architecture" above. A client process that survives a finelog restart loses no rows; a client process that crashes mid-buffer drops what's in flight (acceptable for stats).
- **Reads** (`table.query`) have no fallback — a query during a finelog outage returns an error to the caller. The dashboard's worker pane treats this as a soft failure (renders a "stats unavailable" banner rather than blocking the rest of the page).

## Implementation stages

The change is large; we land it in five PRs in this order:

1. **Layout migration + LogNamespace skeleton**: introduce `LogNamespace`, refactor `DuckDBLogStore` into `NamespaceRegistry`, ship the migration state machine, single namespace (`log`) routed through the new path. The log schema stays hardcoded inside the `LogNamespace` for now — no schema registry yet, no `RegisterTable`, no Arrow-IPC writes. The PR is pure storage-internal restructuring with the migration shipped alongside. No public API change. Tests cover migration recovery and parity with today's log behavior.
2. **Schema registry + Arrow IPC writes**: add the sidecar registry DB, `RegisterTable` / `WriteRows` proto and server impl, schema validation. Still single client surface (`LogClient` not yet introduced).
3. **Stats RPC end-to-end**: add `Query` and `DropTable` to the proto, DuckDB-view-based query execution, namespace-name validation, eviction across namespaces.
4. **Client consolidation**: introduce `LogClient`, delete `LogPusher`, rewrite `RemoteLogHandler`, migrate every in-repo call site in one pass. `Table` buffers and flush threads land here.
5. **Iris dashboard cutover**: worker pane reads from `iris.worker` namespace; sqlite-backed read path retired after the parity check passes for 24h.

Stages 1 and 2 are storage/server-only and ship without API changes. Stage 4 is the breaking client change; stage 5 is the user-visible feature.

## Testing

**Integration**: on the iris dev cluster, a worker registers `iris.worker`, emits one row per heartbeat, and the dashboard reads from the stats service in place of sqlite. Regression check: the rendered worker pane matches the sqlite-backed version pre-cutover for a 24h window.

**Concurrency / lifecycle** (process-level tests against a real `DuckDBLogStore`):

- Concurrent `RegisterTable` + `WriteRows` + `Query` + `DropTable` against the same namespace; assert no segment loss, no half-written files, and that drop-mid-write fails the in-flight write cleanly with `NamespaceNotFoundError`.
- `Query` running while compaction unlinks segments in the same namespace — query must complete on the snapshot it captured at entry.
- One namespace under heavy ingest while another is queried — query latency stays bounded by the query semaphore, not by the writer.
- One namespace's segments evicted by global cap while another is being read — read sees the surviving segments only, no stale paths.

**Schema**:

- Arrow batches with reordered columns, missing nullable columns, missing non-nullable columns, unknown columns, and type mismatches — only the missing-nullable case is accepted.
- Additive evolution across multiple sealed segments: registered `(a, b)` → `(a, b, c)` → `(a, b, c, d)`. Compaction projects to the current schema and the result has the column set in registered order, with NULLs in older rows for late-added columns.
- A schema with no `key_column` and no `timestamp_ms` is rejected at register time.
- A schema whose `key_column` references a non-existent or wrong-typed column is rejected.

**Migration**:

- Cold start with flat `tmp_*.parquet` / `logs_*.parquet` files present produces `{data_dir}/log/` with all files moved and the sentinel in `state=done`.
- Crash mid-migration (kill -9 between two renames) followed by restart resumes cleanly, with size-mismatch on a duplicate aborting loudly.
- Cross-filesystem `data_dir` (different `st_dev`) is refused at startup.

**GCS**:

- Per-namespace prefix appears in upload calls; drop_table does not delete remote objects.

**Client**:

- `Table` overflow policy under back-pressure (server unreachable for N seconds, queue saturates, oldest rows are dropped).
- `RemoteLogHandler` constructed with the new `(LogClient, key)` signature emits to the `log` namespace.
- Existing tests that constructed a fake `LogPusher` are migrated to a fake `LogClient`; their assertions remain meaningful.

**Dashboard**:

- Soft-failure: simulate finelog outage; the worker pane renders the "stats unavailable" banner without breaking the rest of the page.

## Open Questions

None outstanding for v1.

Resolved during review:
- **Register during rolling upgrades** — evolve-by-default. Server merges additive-nullable extensions silently; non-additive changes still error.
- **Per-namespace storage caps** — global cap, oldest-first eviction across all namespaces. Per-namespace quotas deferred until a noisy namespace forces the question.
- **Wire format for batches** — Arrow IPC RecordBatch (columnar), not row-by-row protos. Symmetric with the read path and lets schema validation reduce to an Arrow-schema comparison.
- **Namespace deletion** — in scope, drops local data only; GCS-archived data is the caller's to manage.
- **Storage architecture** — `LogNamespace` per registered namespace owns all per-namespace state; `NamespaceRegistry` is a thin top-level router.
- **Concurrency model** — insertion mutex + query-visibility rwlock. Queries hold read for their full duration; commits and drops hold write briefly.
- **Query SQL strategy** — DuckDB views per namespace, inlined literal path lists (CREATE VIEW rejects prepared params), typed empty views for namespaces with zero sealed segments.
- **Compaction projection** — `union_by_name=true` plus explicit `NULL::TYPE` synthesis for columns missing from *all* compaction inputs; column list always in registered order.
- **Namespace name validation** — restrictive regex `^[a-z][a-z0-9_.-]{0,63}$` plus path-containment defense in depth.
- **Ordering key** — `Schema.key_column` (INT64 or TIMESTAMP_MS) or implicit `timestamp_ms` column. ClassVar opt-in for dataclass inference.
- **LogPusher removal** — deleted entirely; `LogClient.Table` is the only batcher.
