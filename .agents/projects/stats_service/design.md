# Stats Service

_Why are we doing this? What's the benefit?_

Iris emits operational stats — worker heartbeats, container utilization, scheduling decisions — across three uncoordinated places: in-memory RPC counters (`lib/iris/src/iris/rpc/stats.proto:21`), live-computed dashboard rollups (`lib/iris/src/iris/cluster/controller/db.py:192`), and ephemeral worker heartbeats (`lib/iris/src/iris/cluster/worker/worker.py:67`). None are queryable historically. The dashboard reads worker info from the controller's sqlite, which couples the dashboard's data model to controller restarts and makes "what was this worker doing yesterday" effectively unanswerable.

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

- Storage-layer coupling: the existing segment lifecycle (`duckdb_store.py:728`), recovery (`:182`), and compaction (`:342`) all assume one schema; per-namespace segments need each adapted.
- Cardinality is unbounded without per-namespace TTL; worker stats at 1Hz reach ~150GB/year. Ship without TTL, revisit when a namespace nears storage caps.
- The dashboard's worker pane gains a stats-service-down failure mode that today's controller-coupled path doesn't have. Mitigation under "Availability" below.
- Diagnostic queries can return arbitrarily large result sets. Default per-query row cap (`spec.md`); abusive callers get a hard error, not silent truncation.

## Design

A new `StatsService` proto in `lib/finelog/src/finelog/proto/stats.proto`, co-hosted with `LogService` on the same finelog process (`lib/finelog/src/finelog/server/asgi.py:42`). Public API stays on `LogClient`:

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

`DuckDBLogStore` generalizes into a schema registry keyed by namespace, with per-namespace Parquet segment directories (`{log_dir}/{namespace}/`). The registry persists as a sidecar DuckDB table in the finelog data dir and rehydrates on startup — not inferred from Parquet footers. Concurrent registers are guarded by a server-side lock; the registered schema is the union of every schema seen so far for that namespace (additive-nullable evolution is silent — see "Schema evolution" below). Sequence numbers are per-namespace; `_recover_max_seq` runs once per namespace at startup.

The existing log namespace migrates with a one-time directory restructure at startup (move `{log_dir}/{tmp,logs}_*.parquet` into `{log_dir}/log/`), gated on a lock-file so an interrupted move is recoverable on restart. Row layout is unchanged. `LogClient.write_batch(...)` keeps working unchanged on top with its own per-table buffer; stats writes get a parallel buffer per `Table` rather than sharing the log batcher.

**Schema enforcement**: the server validates every batch against the registered schema. A row missing a nullable column is accepted (stored as NULL); a row missing a non-nullable column or with a type mismatch is rejected. Validation lives server-side specifically because trusting the client risks namespace corruption from a misbehaving worker. Schema lookup is in-memory after first registration, so the per-batch cost is constant.

**Schema evolution**: register is evolve-by-default. A caller's schema that adds nullable columns to the registered one is silently merged; the registry stores the union and `RegisterTableResponse.effective_schema` returns it. A caller whose schema is a *subset* of the registered one is also accepted as-is (older clients during a rolling upgrade write rows with NULL for the columns they don't know about). Non-additive changes — rename, type change, or a new non-nullable column — are rejected with `SchemaConflictError`; the migration path is "register a new namespace (`iris.worker.v2`), dual-write through a transition, retire the old". DuckDB's `union_by_name` handles the cross-segment read. No migration tooling; namespace bumps are caller-driven.

A caller who needs strict-equality guarantees (e.g. a test) can compare `Table.schema` against their requested schema after `get_table` and assert.

**Queries** come in two flavors. Exact lookups — "fetch logs for this set of tasks" — keep the existing `LogClient.query(LogQuery)` shape: typed filters, no SQL surface. Diagnostic queries — "what was the p95 task runtime by region last week" — go through `table.query(SQL)` returning an `pa.Table`, with Postgres-flavored SQL passed through to DuckDB. Queries are informational, so coupling them to DuckDB syntax is a deliberate trade: a future migration to a different backend or a typed query builder only requires updating the dashboard, not callers' write code.

Endpoint stays `/system/log_server`; logs and stats share the process under one logical name.

**Availability**: the service runs on a single VM with health-checks and Docker auto-restart — same operational posture as logs today. We do not replicate. The contract callers see is "available almost always; tolerate transient outages." Concretely:

- **Writes** reuse the `LogPusher` pattern: per-`Table` in-memory buffer, background flush, retry on transient failure. A client process that survives a finelog restart loses no rows; a client process that crashes mid-buffer drops what's in flight (acceptable for stats).
- **Reads** (`table.query`) have no fallback — a query during a finelog outage returns an error to the caller. The dashboard's worker pane treats this as a soft failure (renders a "stats unavailable" banner rather than blocking the rest of the page).

## Testing

Integration test on the iris dev cluster: a worker registers `iris.worker`, emits one row per heartbeat, and the dashboard reads from the stats service in place of sqlite. Regression check: the rendered worker pane matches the sqlite-backed version pre-cutover for a 24h window. Unit-grained tests cover schema registry round-trip, add-nullable across segments with `union_by_name`, and rejected writes for type mismatches.

## Open Questions

- **Per-namespace storage caps and retention.** Existing finelog has global storage caps (`DEFAULT_MAX_LOCAL_SEGMENTS`/`DEFAULT_MAX_LOCAL_BYTES` at `duckdb_store.py:122`); per-namespace dirs need either a shared cap (one noisy namespace can starve others) or per-namespace quotas declared at register time. Retention/TTL is the same question viewed from the other side — once a namespace is at cap, do we drop oldest segments (TTL by displacement) or refuse new writes? We're shipping without an opinion here; needs revision before any namespace nears caps.

Resolved during review: register-during-rolling-upgrades — evolve-by-default. Server merges additive-nullable extensions silently; non-additive changes still error.
