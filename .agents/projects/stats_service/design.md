# Stats Service

_Why are we doing this? What's the benefit?_

Iris emits operational stats — worker heartbeats, container utilization, scheduling decisions — across three uncoordinated places: in-memory RPC counters (`lib/iris/src/iris/rpc/stats.proto:21`), live-computed dashboard rollups (`lib/iris/src/iris/cluster/controller/db.py:192`), and ephemeral worker heartbeats (`lib/iris/src/iris/cluster/worker/worker.py:67`). None are queryable historically. The dashboard reads worker info from the controller's sqlite, which couples the dashboard's data model to controller restarts and makes "what was this worker doing yesterday" effectively unanswerable.

This adds a small **stats service** co-hosted in the existing finelog process: typed, schema-registered tables that callers write rows into and query with SQL. It also sets up moving the dashboard's worker pane off the controller sqlite onto a service that outlives any single controller restart.

## Background

File refs, prior-art comparison, and Q&A summary live in [`research.md`](./research.md); concrete contracts (proto, public API, persisted shapes) in [`spec.md`](./spec.md). The load-bearing change is generalizing the DuckDB backend's hardcoded log row schema (`lib/finelog/src/finelog/store/duckdb_store.py:79`) into a per-namespace schema registry. Co-hosting inside finelog keeps the operational footprint small and shares the storage layer; we revisit splitting if stats traffic ever dominates.

## Challenges

The schema model is the load-bearing call. Per-namespace registered schemas with typed columns sit between schema-on-read (typing bugs at query time) and one-table-per-signal (heavier evolution); they need `DuckDBLogStore` to grow a schema registry. Once that lands, logs are one namespace among many.

## Costs / Risks

- Storage-layer coupling: the existing segment lifecycle (`duckdb_store.py:728`), recovery (`:182`), and compaction (`:342`) all assume one schema; per-namespace segments need each adapted.
- Cardinality is unbounded without per-namespace TTL; worker stats at 1Hz reach ~150GB/year. Ship without TTL, revisit when a namespace nears storage caps.
- The dashboard's worker pane gains a stats-service-down failure mode that today's controller-coupled path doesn't have.

## Design

A new `StatsService` proto in `lib/finelog/src/finelog/proto/stats.proto`, co-hosted with `LogService` on the same finelog process (`lib/finelog/src/finelog/server/asgi.py:42`). Public API stays on `LogClient`:

```python
client = LogClient.connect("iris://marin?endpoint=/system/log_server")
table = client.get_table("iris.worker", schema=WORKER_SCHEMA)        # idempotent register
table.write([WorkerStat(worker_id="w-1", mem_bytes=...)])            # dataclass mirror, not generated proto
rows = table.query("SELECT worker_id, AVG(mem_bytes) ...")           # returns pa.Table
```

`DuckDBLogStore` generalizes into a schema registry keyed by namespace, with per-namespace Parquet segment directories (`{log_dir}/{namespace}/`). The registry persists as a sidecar DuckDB table in the finelog data dir and rehydrates on startup — not inferred from Parquet footers. Concurrent registers are guarded by a server-side lock; conflicting schemas are rejected. Sequence numbers are per-namespace; `_recover_max_seq` runs once per namespace at startup.

The existing log namespace migrates with a one-time directory restructure at startup (move `{log_dir}/{tmp,logs}_*.parquet` into `{log_dir}/log/`), gated on a lock-file so an interrupted move is recoverable on restart. Row layout is unchanged. `LogClient.write_batch(...)` keeps working unchanged on top with its own per-table buffer; stats writes get a parallel buffer per `Table` rather than sharing the log batcher.

**Schema enforcement**: the server validates every batch against the registered schema. A row missing a nullable column is accepted (stored as NULL); a row missing a non-nullable column or with a type mismatch is rejected. Validation lives server-side specifically because trusting the client risks namespace corruption from a misbehaving worker. Schema lookup is in-memory after first registration, so the per-batch cost is constant.

**Schema evolution**: add-nullable and drop-column are supported via DuckDB's `union_by_name`. Rename and type change are unsupported — register a new namespace, dual-write through a transition, retire the old one. We ship without migration tooling; namespace bumps are caller-driven.

**Queries** come in two flavors. Exact lookups — "fetch logs for this set of tasks" — keep the existing `LogClient.query(LogQuery)` shape: typed filters, no SQL surface. Diagnostic queries — "what was the p95 task runtime by region last week" — go through `table.query(SQL)` returning an `pa.Table`, with Postgres-flavored SQL passed through to DuckDB. Queries are informational, so coupling them to DuckDB syntax is a deliberate trade: a future migration to a different backend or a typed query builder only requires updating the dashboard, not callers' write code.

Endpoint stays `/system/log_server`; logs and stats share the process under one logical name.

## Testing

Integration test on the iris dev cluster: a worker registers `iris.worker`, emits one row per heartbeat, and the dashboard reads from the stats service in place of sqlite. Regression check: the rendered worker pane matches the sqlite-backed version pre-cutover for a 24h window. Unit-grained tests cover schema registry round-trip, add-nullable across segments with `union_by_name`, and rejected writes for type mismatches.

## Open Questions

- **Register semantics during rolling upgrades.** The Design rejects "conflicting schemas" on concurrent register, but rolling upgrades produce schemas that legitimately differ — worker v2 adds a nullable column; worker v1 still calls `get_table` with the older schema. Should register accept additive-nullable extensions of the stored schema and silently upgrade, require strict equality (and break rollouts), or accept-as-superset only when the caller passes an explicit `allow_evolve=True`? The first is most ergonomic but means register can mutate stored schema as a side effect, which is surprising; the third is explicit but adds API surface.
- **Per-namespace storage caps vs global caps.** Existing finelog has global storage caps (`_DEFAULT_MAX_LOCAL_*` at `duckdb_store.py:122`); per-namespace dirs need either a shared cap (one noisy namespace can starve others) or per-namespace quotas declared at register time. We're shipping without TTL, so this matters sooner rather than later.
