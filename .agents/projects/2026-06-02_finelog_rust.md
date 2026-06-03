# Finelog Rust Rewrite — Project Proposal

**Status:** Draft proposal (investigation)
**Author:** weaver/finelog-rust
**Date:** 2026-06-02
**Build plan:** see the synthesized incremental roadmap with the per-stage
acceptance matrix at `2026-06-02_finelog_rust_roadmap.md` — it supersedes this
doc for sequencing and **corrects the dependency versions below** (the roadmap's
recon resolved the real crate graph: DataFusion **53.1**, arrow/parquet **58**,
`object_store` **0.13** — not the 52.x/57.1 first assumed here).

## 1. Summary

Rewrite the core finelog **server and storage engine** in Rust, behind the
existing protobuf RPC contract, so that the Python side becomes a thin client +
deploy-tooling layer. Two capabilities motivate going native:

- **Native compaction.** Today compaction is a DuckDB `COPY (... ORDER BY ...)`
  driven from a Python background thread. In Rust we merge parquet segments
  directly with the `arrow`/`parquet` crates (already vendored for `dupekit`),
  no embedded SQL engine on the write path.
- **Native SQL via DataFusion.** Replace the read-side embedded DuckDB +
  per-namespace view registration with [Apache DataFusion](https://datafusion.apache.org/),
  exposing each namespace as a custom `TableProvider` that unions the on-disk
  parquet segments with the in-RAM write buffer.

The **proto contract is the migration boundary and is frozen** (see §3). The
Python client (`connect-python`) and the Python test harness keep working
unchanged; tests shift to exercise the system **over real HTTP/RPC** so the same
suite validates both the Python and Rust servers during the transition.

## 2. Current architecture (what we are replacing)

Grounded in `lib/finelog/`. Two services over Connect/RPC, one shared store.

```
            ┌─────────────────────── Python (today) ───────────────────────┐
 client →   │  ASGI (starlette/uvicorn)                                     │
            │   ├─ LogService     (service.py)      push/fetch logs         │
            │   ├─ StatsService   (stats_service.py) register/write/query   │
            │   └─ interceptors: slow-RPC, concurrency caps (fetch=4,query=4)│
            │         │                                                      │
            │  DuckDBLogStore (duckdb_store.py, 715 loc)  ── shared ─────────│
            │   ├─ Catalog (catalog.py): sidecar DuckDB registry            │
            │   │     namespaces + segments tables; drop reservations       │
            │   ├─ DiskLogNamespace (log_namespace.py, 2170 loc)            │
            │   │     RamBuffers (LSM chunks) → flush loop → parquet         │
            │   │     compaction loop (leveled) ; maintenance loop          │
            │   │       (eviction + GCS sync via fsspec)                     │
            │   ├─ Compactor (compactor.py): leveled-merge planner          │
            │   ├─ Schema (schema.py): Arrow↔proto↔json, additive merge     │
            │   └─ ConnectionPool: single read DuckDB conn + query watchdog  │
            └───────────────────────────────────────────────────────────────┘
                              on disk:  {data_dir}/<namespace>/seg_L<n>_<minseq>.parquet
                                        {data_dir}/_finelog_registry.duckdb
                              remote:    gs://.../<namespace>/...  (BOTH→REMOTE)
```

Core engine facts the rewrite must reproduce exactly:

- **Seq numbering.** A monotonic per-row `seq` is stamped at append time; the
  `cursor` in `FetchLogsResponse` is the max seq seen and is the resume token.
  Must survive restarts.
- **Durability contract.** `PushLogs` / `WriteRows` return only **after** the
  written rows are persisted to an L0 parquet segment (handler awaits
  `await_persisted` / `request_persistence`, ~30s timeout). Writes are
  RAM-buffered but the RPC is synchronous w.r.t. durability.
- **Query visibility.** A query sees sealed parquet segments **plus** the
  in-RAM buffer. Today: snapshot segments under the catalog lock, register a
  DuckDB view per namespace, run the **user SQL verbatim** (no rewrite),
  serialize the result as Arrow IPC.
- **Schema evolution is additive-only.** New nullable columns OK; renames /
  type changes / key-column changes / non-nullable additions rejected. Missing
  columns filled with NULL on append.
- **Leveled compaction.** `level_targets=(64MiB,256MiB,256MiB)`,
  `max_segments_per_level=4`; longest contiguous run at a level merges up;
  terminal level never re-compacts and is the only eviction-eligible level.
- **Storage policy / eviction.** Per-namespace `max_segments` / `max_bytes` /
  `max_age_seconds`; eviction flips segment location `LOCAL→BOTH→REMOTE` after
  GCS upload, then unlinks local bytes.
- **Concurrency caps.** `_MAX_CONCURRENT_FETCH_LOGS=4`, `_MAX_CONCURRENT_QUERY=4`
  to bound the parquet working set against page-cache thrash.
- **Wire format.** `WriteRows.arrow_ipc` is one Arrow IPC RecordBatch;
  `QueryResponse.arrow_ipc` is the result table as Arrow IPC. Limits:
  `MAX_WRITE_ROWS_BYTES=16MiB`, `MAX_WRITE_ROWS_ROWS=1Mi`.
- **LogService is a specialized table.** Logs live in a reserved `"log"`
  namespace; `FetchLogs` is key-pattern (EXACT/PREFIX/REGEX) + `since_ms` +
  cursor + substring + `min_level` + `tail` over that namespace.

### Size / complexity (drives sequencing)

| Component | File | LOC | Notes |
|---|---|---:|---|
| Per-namespace engine | `store/log_namespace.py` | 2170 | **heaviest**; buffers, 3 bg loops |
| Store wrapper / query | `store/duckdb_store.py` | 715 | RPC-facing orchestration |
| Catalog / registry | `store/catalog.py` | 489 | sidecar DuckDB |
| Schema | `store/schema.py` | 460 | validation + merge |
| Compaction planner | `store/compactor.py` | 267 | leveled policy |
| Server (asgi/service/stats/main) | `server/*.py` | 719 | RPC shell |
| Client | `client/log_client.py` | 848 | **stays Python** |
| Migrations / layout | `store/migrations`, `layout_migration.py` | ~600 | |

## 3. Frozen contract

The two proto files are the spec and **do not change** during this project:

- `finelog.logging` — `LogService{PushLogs, FetchLogs}`,
  enums `LogLevel`, `MatchScope`.
- `finelog.stats` — `StatsService{RegisterTable, WriteRows, Query, DropTable,
  ListNamespaces, GetTableSchema}`, enum `ColumnType`, `StoragePolicy`.

Wire-format invariants that are part of the contract (not just the proto):
Arrow IPC framing for `WriteRows`/`Query`, the seq/cursor semantics, the
durability-before-ack guarantee, additive schema evolution, and the
`/iris.logging.LogService/*` → `/finelog.logging.LogService/*` legacy path
rewrite (already CRON-marked for removal 2026-05-12; confirm gone before cutover).

## 4. Target architecture (Rust)

```
            ┌──────────────────────── Rust (target) ───────────────────────┐
 client →   │  axum + hyper + tokio                                         │
 (Python    │   connectrpc Router  (anthropics/connect-rust)                │
  unchanged)│    ├─ LogService impl                                          │
            │    ├─ StatsService impl                                        │
            │    ├─ tower middleware: zstd/gzip, concurrency-limit, slow-RPC │
            │    └─ /health, /static (Vue SPA served as before)             │
            │         │                                                      │
            │   Store (Rust)                                                 │
            │    ├─ Catalog: segment+namespace registry (rusqlite sidecar    │
            │    │     OR rebuilt-from-disk manifest; see §6.3)              │
            │    ├─ Namespace engine (tokio tasks, not OS threads):          │
            │    │     RAM buffer (Arrow) → flush → parquet (arrow/parquet)  │
            │    │     compaction task: native arrow k-way merge → parquet   │
            │    │     maintenance task: eviction + object_store upload      │
            │    ├─ Query: DataFusion SessionContext                         │
            │    │     each namespace = custom TableProvider unioning        │
            │    │     ListingTable(parquet segments) + MemTable(RAM buffer) │
            │    └─ object_store crate for GCS (replaces fsspec/gcsfs)       │
            └───────────────────────────────────────────────────────────────┘
```

### Crate layout (extends the existing `rust/` workspace)

`rust/` already hosts a maturin/PyO3 workspace (`dupekit`, arrow 57.1,
parquet 57.1). Add a **binary** crate — finelog is a server, not a Python
extension module, so unlike `dupekit` it needs **no PyO3**; the RPC socket is
the language boundary.

```
rust/
  Cargo.toml                 # add "finelog" to workspace members
  dupekit/                   # unchanged
  finelog/
    Cargo.toml               # [[bin]] finelog-server  +  lib for unit tests
    build.rs                 # connectrpc-build: compile the two .proto files
    src/
      main.rs                # CLI (clap): --port --log-dir --remote-log-dir
      proto/                 # generated (buffa views + connect stubs)
      server/                # LogService / StatsService trait impls, middleware
      store/                 # catalog, namespace, ram_buffer, flush, compaction
      query/                 # DataFusion SessionContext + NamespaceTableProvider
      schema.rs              # Arrow↔proto, additive-merge, validate/align batch
      remote.rs              # object_store GCS sync
```

**Build toolchain (proto codegen).** We use connect-rust's `build.rs` path
(`connectrpc_build::Config` + `connectrpc::include_generated!`) — the documented
"Option B". `connectrpc-build` internally shells out to `protoc` to parse the
`.proto` files (the "no binary plugins" note refers to `protoc-gen-*` codegen
plugins, not to protoc the parser). Our protos use `edition = "2023"`, which
needs protoc ≥ 27. To keep the build hermetic — no dependency on whatever protoc
a host happens to have — `build.rs` vendors protoc via the `protoc-bin-vendored`
crate (3.2.0 → libprotoc 31.1) and hands its path to connectrpc-build through
the `PROTOC` env var. `cargo build` is fully self-contained; nothing is checked
in and there is no system-protoc requirement. (Verified: builds with no protoc
on `PATH`.)

Key dependencies (versions corrected by the roadmap's recon against the real
crate graph): `connectrpc` 0.6 (+`axum`,`gzip`,`zstd`), `buffa`/`buffa-types`
0.6, `tokio`, `axum`, `datafusion` **53.1**, `arrow`/`arrow-ipc` **58**,
`parquet` **58** (features `async`,`object_store`) — DataFusion 53 re-exports
arrow 58, so the finelog binary links exactly one arrow (58); dupekit's 57.1 is
a separate cdylib, no skew. Plus `object_store` **0.13** (gcp feature),
`rusqlite` (bundled, catalog sidecar — distinct filename from Python's DuckDB
registry), `clap`, `tracing`.

### Notable design decisions

- **Threads → tokio tasks.** The three Python background loops (flush /
  compact / maintenance) become per-namespace tokio tasks. Blocking parquet I/O
  and DataFusion execution run on `spawn_blocking` / a dedicated rayon pool so
  they don't stall the async reactor.
- **Durability await is natural in async.** The handler awaits a
  `tokio::sync::watch` / `Notify` on the namespace's `persisted_seq` instead of
  Python's poll-with-backoff.
- **Compaction is native, not SQL.** A k-way merge of sorted parquet segments
  by `(key_column, seq)` using `arrow` row interleave + `parquet` writer —
  removes the 8GB-DuckDB-for-compaction memory tuning entirely.
- **Concurrency caps preserved** via `tower::limit` / a `Semaphore`
  (fetch=4, query=4), matching today's page-cache rationale.
- **Object store.** `object_store` crate (native GCS) replaces fsspec/gcsfs;
  same `LOCAL→BOTH→REMOTE` location state machine.

## 5. Why these libraries

- **anthropics/connect-rust (`connectrpc`).** Tower-based, runs on axum/hyper,
  serves Connect + gRPC + gRPC-Web, passes the full conformance suite, has
  built-in zstd/gzip (we use zstd today), interceptors, and a client. Protos
  compile via `connectrpc-build` in `build.rs` (no external plugin binaries) or
  `buf generate`. Connect-protocol unary is ~20% faster than tonic and the
  decode-heavy log-ingest path is materially faster — a good fit for a
  write-heavy log server.
- **DataFusion.** First-class custom `TableProvider`/`ExecutionPlan` extension
  points let us model "parquet segments ∪ live RAM buffer" as one logical table
  per namespace and run **user SQL verbatim** (the current contract). It already
  has parquet readers, predicate/projection pushdown, parquet footer/metadata
  caching, and partitioned parallel execution — the things the DuckDB read path
  gives us today.

## 6. Key risks & decisions

### 6.1 SQL dialect: DuckDB → DataFusion (TOP RISK)

The contract says the server runs the **user's SQL verbatim** against a
Postgres-flavored DuckDB engine. DataFusion uses a Postgres-ish dialect
(`sqlparser-rs`) but is **not** DuckDB-compatible: function names, casts,
identifier-case handling, and some sugar differ. This is the only place a
rewrite can silently change user-visible behavior.

Mitigation — the actual corpus is small. Observed query shapes across the repo
(`tests/`, `deploy/cli.py:255`, `iris/scripts/job_profile_summary.py:158`,
`iris/src/iris/cli/query.py`) are projections, `WHERE` filters, `ORDER BY`,
`COUNT(*) AS n`, quoted dotted identifiers like `"iris.worker"` — all squarely
within DataFusion's support.

**Decision / spike (Phase 0):** extract the real query corpus from iris/zephyr
callers, build a golden-query parity test that runs each against both engines
and diffs the Arrow result, and register DataFusion UDFs for any DuckDB
function gaps. **Fallback:** if dialect gaps prove expensive, the `duckdb-rs`
crate keeps the exact DuckDB dialect natively while still removing Python from
the path — DataFusion stays the goal but is not a hard blocker for going native.
*(Flag for the user: DataFusion is an explicit request; confirm we accept
possible minor query-dialect changes, recorded via weaver note.)*

### 6.2 Identifier case-folding

DataFusion (unlike Postgres) does not lower-case unquoted identifiers; our
identifiers are already quoted (`"iris.worker"`). Verify column refs in the
corpus are consistent; covered by the golden-query parity test.

### 6.3 On-disk / catalog compatibility for production cutover

Parquet segments are language-neutral. The catalog is a DuckDB sidecar
(`_finelog_registry.duckdb`). Two options for the Rust server to adopt existing
data:

- **(A) Rebuild catalog from disk** (recommended). Segment filenames encode
  level + min_seq and the parquet footer carries row counts and key-column
  stats; a one-time directory scan reconstructs the registry into a Rust-owned
  sidecar (`rusqlite` or a parquet/JSON manifest). No DuckDB dependency in Rust;
  precedent exists in `layout_migration.py`.
- **(B) Read the DuckDB sidecar** via `duckdb-rs` once at startup, then own it
  natively.

Decision deferred to the cutover phase; (A) is preferred. Either way the cutover
is a flag-day per deployment — finelog is single-instance, no replication, so no
dual-write/dual-read window is needed.

### 6.4 Arrow version skew

`dupekit` pins arrow/parquet 57.1; DataFusion 52 re-exports its own (older)
arrow. Keep the finelog crate on **DataFusion's arrow** to avoid two arrow
versions linked in one binary; dupekit is a separate cdylib and unaffected.

### 6.5 Async durability & backpressure

The synchronous-ack-after-persist contract must hold under the write path's new
async model; the 16MiB/1Mi request caps and per-namespace flush target
(100MiB / 5s) carry over. Covered by the durability parity tests.

## 7. Incremental migration plan

Principle: **the RPC socket is the seam.** We build a native Rust server
bottom-up, RPC-family by RPC-family, and a dual-backend parity harness runs the
growing Rust subset against the *same* HTTP tests that pin the Python server.
Production keeps running Python until Phase 6 flips one deployment at a time.

We deliberately **do not** take the PyO3-embedding route (moving leaf functions
into a Python-loaded `.so` à la dupekit). The data plane — write buffer, flush,
compaction, query, catalog — shares on-disk state and cannot be cleanly split
write-vs-query across a language boundary; embedding would mean building a FFI
facade we throw away at cutover. A single cohesive Rust engine validated through
RPC is less total work and matches the stated goal.

### Phase 0 — Scaffolding & the parity harness (no behavior change)
- Add `rust/finelog` to the workspace; `build.rs` compiles both protos; a
  `/health` + empty-service binary builds and boots.
- **Parity harness (linchpin):** parametrize the Python RPC tests over a
  `server_backend` fixture that yields a base URL for either (a) the in-process
  Python ASGI app, or (b) a freshly-spawned `finelog-server` subprocess on a
  tmp data dir. Tests talk over real HTTP (`httpx`) — no `DuckDBLogStore`
  imports. Rust-unimplemented RPCs are `xfail`-marked and flip to pass as each
  phase lands.
- Extract the **golden SQL corpus** (§6.1) from iris/zephyr callers.

### Phase 1 — Catalog & metadata RPCs
`RegisterTable`, `GetTableSchema`, `ListNamespaces`, `DropTable` (catalog side),
schema additive-merge + validation. No data yet. Parity tests:
`test_register`, schema-merge cases, namespace listing.

### Phase 2 — Write path & durability
`WriteRows` + `PushLogs`: Arrow IPC decode, validate/align batch, seq stamping,
RAM buffer, flush task → parquet L0, persisted-seq await. Verify via
`ListNamespaces` stats (row_count / max_seq / segment_count) without needing
Query yet. Parity tests: `test_write_rows`, `test_durable_writes`,
`test_ram_buffers` (reframed as RPC/stats assertions), `test_concurrency`.

### Phase 3 — Query path (DataFusion)
`Query` + `FetchLogs`: `NamespaceTableProvider` unioning parquet segments +
RAM buffer; run user SQL; Arrow IPC result. Log-service key matching
(EXACT/PREFIX/REGEX), substring, `min_level`, `tail`, cursor. Parity tests:
`test_query`, golden-query corpus, `test_duckdb_store` (log fetch semantics),
`test_drop` (query-after-drop).

### Phase 4 — Compaction & eviction (native)
Leveled compaction task (native arrow merge), maintenance task (eviction by
count/bytes/age), `object_store` GCS sync with `LOCAL→BOTH→REMOTE`. Parity
tests: `test_compactor` (reframed: write enough to trigger promotion, assert via
stats + post-compaction query correctness), `test_eviction`,
`test_storage_policy`, `test_catalog_stats`, `test_offload`.

### Phase 5 — Server hardening
Concurrency caps, slow-RPC logging, zstd/gzip, static SPA serving, graceful
shutdown, pool/RSS diagnostics, legacy path rewrite (if still needed). Parity
tests: `test_asgi`, `test_server`, `test_interceptors`.

### Phase 6 — Production cutover
Catalog adoption (§6.3), deploy image builds the Rust binary, flip one
deployment, bake, then roll. Delete the Python store/server (`store/`,
`server/`); keep `client/`, `deploy/`, protos, tests. Remove DuckDB + fsspec
Python deps.

## 8. Testing strategy

Goal stated by the user: **shift testing to exercise the server over RPC** so
the Python harness keeps working against a Rust backend.

- **Dual-backend RPC parity harness** (Phase 0). The same test body runs against
  Python ASGI and the Rust binary via a parametrized base-URL fixture. This is
  what makes the migration safe and is the long-term test surface.
- **Convert direct-store tests to RPC/stats assertions.** Tests that today call
  `store.query(...)`, `store.catalog[...]`, `ns.flush()/force_compact_l0()`
  (e.g. `test_catalog_stats`, `test_compactor`, `test_ram_buffers`,
  `test_offload`) move to asserting on `ListNamespaces` stats and query results.
  Where a test needs to force a background action (flush/compact) that the RPC
  surface doesn't expose, add **test-only admin RPCs or env knobs** (e.g. a
  debug `Flush`/`Compact` trigger gated to test builds) rather than reaching
  into internals — keeps the harness language-agnostic.
- **Rust unit tests** (`#[cfg(test)]`) cover algorithmic internals that don't
  belong on the wire: compaction planner selection, RAM-buffer LSM invariant,
  schema-merge edge cases, sql_escape. These replace the Python direct-import
  unit tests for those modules.
- **Golden SQL parity** (§6.1): run the corpus against both engines, diff Arrow
  output.
- **Keep as-is, Python-only:** `test_client.py` (client logic), `test_config`,
  `test_deploy_*` — they test Python code that survives.
- Avoid the slop-test pitfalls in `AGENTS.md`: assert on structured Arrow
  output and `NamespaceInfo` fields, not log strings or rendered text.

## 9. Decisions (resolved with user, 2026-06-02)

1. **DataFusion dialect — DECIDED: DataFusion + golden-query gate.** Go native
   on DataFusion as the primary engine. Minor user-visible dialect differences
   vs DuckDB are acceptable, guarded by the golden-query parity test (§6.1) that
   diffs the real corpus against both engines; register UDFs for any gaps.
   `duckdb-rs` remains a documented fallback only if the spike surfaces costly
   gaps.
2. **Catalog adoption — DECIDED: rebuild from disk (option A).** At cutover the
   Rust server reconstructs the registry by scanning the parquet directory +
   footers into a Rust-owned sidecar; no DuckDB dependency in the Rust build.
3. **Native compaction.** Pure arrow k-way merge (no SQL on the write path).
   Revisit only if merge correctness/perf argues for routing through DataFusion.
4. **Distribution.** Rust binary ships in the existing Docker image; no
   pip-installable server wheel planned.

## 10. Rough sequencing

Phases 0–2 deliver a Rust server that registers, writes, and durably persists,
validated by the existing suite over RPC — the highest-risk integration
(connectrpc wiring + Arrow IPC + durability) proven early. Phase 3 (DataFusion
query) carries the dialect risk and gets the golden-corpus gate. Phases 4–5 are
mechanical given the engine. Phase 6 is operational. Each phase ends with the
parity harness green for its RPC family; no phase merges with regressions in the
Python suite.
