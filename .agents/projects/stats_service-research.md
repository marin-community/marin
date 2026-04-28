# Stats Service — Research Notes

Background research for [`stats_service.md`](./stats_service.md). In-repo findings, prior-art digest, and the Q&A that shaped the design.

## In-repo: finelog today

`finelog` came out of [#5212](https://github.com/marin-community/marin/pull/5212). It's a Python service with a clean proto / storage / client split, currently hosting one RPC service:

- **Service registration** — `LogServiceWSGIApplication` wired via `build_log_server_asgi()` at `lib/finelog/src/finelog/server/asgi.py:42`. Starlette ASGI app over Connect/RPC; concurrency interceptors plug in transparently. `/health` already exposed.
- **Storage abstraction** — `LogStore` (`lib/finelog/src/finelog/store/__init__.py:37`) is generic: `append(key, entries)` + `get_logs(...)`. Auto-selects `MemStore` (tests) or `DuckDBLogStore` (prod).
- **DuckDB backend** — `lib/finelog/src/finelog/store/duckdb_store.py:79-88` hardcodes a Parquet schema `[seq, key, source, data, epoch_ms, level]`. Arrow → Parquet flush pipeline expects these fields exactly. **This is the load-bearing thing that has to generalize for stats.**
- **Proto** — `lib/finelog/src/finelog/proto/logging.proto` defines `LogEntry` + `LogService { PushLogs, FetchLogs }`. Adding `stats.proto` alongside is mechanical given PR #5212 already set the cross-package import patterns.
- **Process entry point** — `lib/finelog/src/finelog/server/main.py:35-68`. Co-hosting a second proto service is straightforward — register on the same ASGI app.

## In-repo: current iris stats sources

Three uncoordinated emitters, none persisted:

- **RPC introspection** — `lib/iris/src/iris/rpc/stats.proto:21-70` (`RpcMethodStats`, percentiles, histograms). Already has its own proto. In-memory only via `RpcStatsCollector` (`lib/iris/src/iris/cluster/controller/dashboard.py:79-81`). **Stays as-is** — different concern (live RPC introspection vs persisted operational stats).
- **User/job/task counts** — `lib/iris/src/iris/cluster/controller/db.py:192-195` (`UserStats`), computed live via `_live_user_stats()` (`controller/service.py:853-879`). Ephemeral, returned by `GetClusterStatus` RPC.
- **Worker heartbeats** — `lib/iris/src/iris/cluster/worker/worker.py:67,226`. Tracked as deadlines, not logged.
- **Container metrics** — `lib/iris/src/iris/cluster/runtime/docker.py:278-330` (`ContainerStats`). Sampled on-demand.

The iris dashboard's worker pane reads worker info from the controller's sqlite (`lib/iris/src/iris/cluster/controller/db.py`). MVP cutover replaces that read path with the stats service.

## Prior art (web pass)

Five categories worth comparing against:

| System | Schema model | Ingest | Storage | Worth borrowing | Gotcha |
|---|---|---|---|---|---|
| Prometheus + Pushgateway | fixed (metric+labels+float) | pull, push escape hatch | custom TSDB | labels-as-dimensions | Pushgateway misuse → stale series |
| InfluxDB / VictoriaMetrics | schema-on-read line proto | batched push | columnar TSM | dead-simple emission | unbounded tag cardinality silently kills perf |
| Honeycomb / Scuba | schema-on-read wide events | batched JSON push | segmented columnar | wide events + ad-hoc GROUP BY | typing bugs land at query time |
| OpenTelemetry Collector | fixed OTLP proto | batched gRPC/HTTP push | none (pipeline) | proto contract decouples producer/storage | enormous config surface |
| ClickHouse-as-observability (SigNoz, M3) | typed user-defined tables | batched INSERT | columnar MergeTree | typed columns + SQL is cheap | naive single-row inserts collapse merge engine |

**Synthesis.** Convergent shape at our scale: typed proto contract on the wire (OTel pattern), batched push, columnar per-signal tables (ClickHouse pattern). Honeycomb/Scuba validate co-locating wide-event tables for debug stats. Genuinely contested: (1) fixed schema vs schema-on-read, (2) push vs pull, (3) does the service own aggregation or just store raw events.

We pick the typed-per-namespace position deliberately. Schema-on-read is tempting for v0 simplicity but loses column pruning on the dimensions stats queries actually filter by. Pull doesn't fit Iris workers well (short-lived tasks). Aggregation in-service is unnecessary at our row volumes.

## Q&A summary

Decisions that shaped the doc:

1. **Schema model** — per-namespace registered schemas with typed columns. Logs become the namespace `"log"` with a fixed registered schema. Not schema-on-read.
2. **Client** — single `LogClient`, extended with `get_table(name, schema)` returning a typed handle. No separate `StatsClient` until ergonomic pressure justifies it.
3. **`iris.rpc.stats`** — stays as-is for RPC introspection. Post-MVP, worker stats writes go direct to the stats service rather than via the controller; that's a follow-up, not part of this design.
4. **Schema evolution** — add-nullable and drop-column supported via DuckDB `union_by_name` across segments. Rename / type-change unsupported; register `iris.worker.v2`, dual-write, retire the old. No migration tooling — namespace bump is the tool.
5. **MVP** — wire iris worker stats end-to-end: worker → stats service → dashboard pane. The dashboard pane currently reading from controller sqlite cuts over to the stats service.
6. **Open Questions** — only things genuinely uncertain to us. Three made the cut: schema enforcement strictness, retention, batching reuse with `LogPusher`.

## Cardinality back-of-envelope

Worker stats at 1Hz × 100 workers × 24h = ~8.6M rows/day. At ~50 bytes/row in Parquet (compressed, typed columns) that's ~400MB/day, ~150GB/year per namespace. Manageable but not free; retention is the question, not capacity.
