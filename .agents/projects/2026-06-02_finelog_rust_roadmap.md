<!-- Synthesized 2026-06-02 by the finelog-rust-roadmap workflow (9 agents: 2 API
recon + 6 per-phase design + 1 synthesis), grounded in the actual rust/Cargo.lock
resolution and the Python source. Companion to the proposal
.agents/projects/2026-06-02_finelog_rust.md, which it supersedes for sequencing
and corrects on dependency versions. -->

# Finelog Rust Rewrite — Incremental Roadmap

**Status:** Synthesized build plan (supersedes the per-phase JSON drafts for sequencing)
**Branch:** `weaver/finelog-rust` · **Worktree:** `/home/power/code/marin/.worktrees/finelog-rust`
**Source proposal:** `.agents/projects/2026-06-02_finelog_rust.md`

---

## 1. Intro & the always-green / parity-gated principle

We rewrite the finelog **server + storage engine** in Rust behind the **frozen** protobuf RPC
contract (`finelog.logging.LogService`, `finelog.stats.StatsService`). Python becomes a thin
**client + deploy** layer. Phase 0 (scaffolding + parity harness) is already in the tree and
green.

Two non-negotiable invariants govern the entire sequence:

1. **Always buildable.** Every sub-stage ends with `cargo build -p finelog` green. Pure-library
   sub-stages (schema, ram_buffer, segment, planner, merge, UDFs) land with `cargo test` gates
   *before* any RPC wiring, so the hardest CPU code is validated in isolation.
2. **Parity-gated, family-by-family.** The dual-backend parity suite
   (`lib/finelog/tests/parity/`) runs the *same* HTTP test body against the Python ASGI server
   and the Rust binary. The Rust backend auto-skips when the binary is absent. Each RPC family
   flips from `rust_pending(...)` (xfail) to **green on both backends** exactly when its
   sub-stage lands. We never relax a Python-side assertion; we never merge with a regression in
   the Python suite.

**The seam is the RPC socket.** No test imports `DuckDBLogStore` or store internals. Where a
test must force a background action the proto doesn't expose (flush/compact/evict/backdate), we
add a **flag-gated, non-proto** debug surface (`--debug-admin`: `POST /debug/maintain`,
`GET /debug/segments`, `POST /debug/backdate`) on **both** backends, driving the *same*
maintenance code path — never a parallel implementation.

**RPC family → phase map:**

| RPC family | Lands in | Until then |
|---|---|---|
| `RegisterTable`, `GetTableSchema`, `ListNamespaces`, `DropTable` | Phase 1 | Unimplemented |
| `WriteRows`, `PushLogs` | Phase 2 | Unimplemented |
| `Query`, `FetchLogs` | Phase 3 | Unimplemented |
| compaction/eviction/remote (observed via stats + `/debug/*`) | Phase 4 | n/a (no new RPC) |
| concurrency caps, slow-RPC, SPA, legacy-path, shutdown | Phase 5 | n/a (hardening) |
| rebuild-from-disk adoption + deploy cutover | Phase 6 | n/a (operational) |

---

## 2. Consolidated acceptance matrix

`R=` cargo from `…/rust`; `P=` parity pytest from `…/lib/finelog`. The standard parity command
(abbreviated **`PARITY`** below) is:

```
cd /home/power/code/marin/.worktrees/finelog-rust/lib/finelog && \
  uv run --group dev --with httpx pytest <paths> -p no:xdist -o addopts="" -v
```

The standard cargo build (**`BUILD`**):
```
cd /home/power/code/marin/.worktrees/finelog-rust/rust && cargo build -p finelog --bin finelog-server
```

| Phase | Sub-stage | Acceptance command | What turns green | Parity test(s) |
|---|---|---|---|---|
| 1 | S1 schema types + ColumnType↔Arrow + proto/json conv | `R cargo test -p finelog schema::` | `schema.rs` conversions, `resolve_key_column`, `with_implicit_seq`, json round-trip | — (cargo only) |
| 1 | S2 errors + `StatsError→ConnectError` | `R cargo test -p finelog errors::` | error code mapping | — |
| 1 | S3 merge_schemas + namespace-name validation | `R cargo test -p finelog merge_schemas namespace_name` | additive-merge branches, name regex + path-traversal | — |
| 1 | S4 policy.rs | `R cargo test -p finelog policy::` | StoragePolicy zero↔None, is_empty | — |
| 1 | S5 catalog.rs (rusqlite sidecar) | `R cargo test -p finelog catalog::` | namespaces/policies/empty-segments tables, drop-fencing, aggregate stats=0 | — |
| 1 | S6 store.rs orchestration | `R cargo test -p finelog store::` | register/get/list/drop at store layer, `log` bootstrap | — |
| 1 | S7 wire 4 metadata RPCs + raised Limits | `BUILD` | RegisterTable/GetTableSchema/ListNamespaces/DropTable over RPC | — |
| 1 | S8 metadata parity (flip rust_pending) | `BUILD` then `P test_metadata.py test_smoke.py` | **metadata RPC family green on rust** | `test_metadata.py` (register/merge/name/get/list/drop) |
| 2 | 2a arrow/parquet deps + validate_and_align_batch + IPC decode | `R cargo build && cargo test -p finelog schema::` | batch alignment, null-fill, dictionary decode, IPC decode; single arrow 58 | — |
| 2 | 2b RAM buffer (LSM) + seq stamping | `R cargo test -p finelog ram_buffer::` | chunk invariant, byte/row accounting, seq stamp + additive null-fill | — |
| 2 | 2c parquet L0 writer (UNSORTED) + footer recovery | `R cargo test -p finelog segment::` | seg filename, write_segment, footer bounds, recover_next_seq | — |
| 2 | 2d namespace engine: lock + persisted_seq watch + flush task + stats | `R cargo test -p finelog namespace::` | durability await, restart seq recovery, stats seq-window math | — |
| 2 | 2e wire WriteRows + PushLogs | `BUILD` | WriteRows/PushLogs handlers compile, server boots, 16MiB body OK | — |
| 2 | 2f write-path parity (flip rust_pending) | `BUILD` then `P test_write_path.py` | **WriteRows/PushLogs family green on rust** | `test_write_path.py` (round-trip stats, null-fill, violations, dictionary, durable-before-ack, concurrency, seq-window) |
| 3 | 3.0 DataFusion deps + read-only SessionContext + IPC encode | `R cargo build && cargo test -p finelog query::tests::select_one_roundtrips arrow_ipc::` | DataFusion linked, single arrow 58, SQL→IPC round-trip | — |
| 3 | 3.1 NamespaceProvider over sealed segments | `R cargo test -p finelog query::provider::` | per-ns TableProvider (sealed parquet only), typed empty, JOIN | — |
| 3 | 3.2 wire `StatsService::query` (flip smoke) | `BUILD` then `P test_smoke.py` | **Query reachable on rust** | `test_constant_query_round_trips` (un-xfail) |
| 3 | 3.3 convert test_query → RPC parity | `P test_query.py` | Query write→read visibility, typed-empty, WHERE/ORDER/JOIN, unknown-ns | `test_query.py` (round-trip, empty, where, join, unknown-ns) |
| 3 | 3.4 golden-query corpus + compat UDFs | `R cargo test -p finelog query::udf::` then `P test_golden_corpus.py` | DuckDB↔DataFusion Arrow parity; prefix/regexp_matches/contains UDFs | `test_golden_corpus.py` |
| 3 | 3.5 `LogService::fetch_logs` (pure-logic ports) | `R cargo test -p finelog store::log_read::` then `BUILD` | scope predicates, attempt_id, level map, regex-prefix, tail/cursor shaping | — |
| 3 | 3.6 convert FetchLogs → RPC parity | `P test_fetch_logs.py test_smoke.py test_query.py test_golden_corpus.py` | **FetchLogs family green on rust** | `test_fetch_logs.py` (roundtrip/tail/cursor/regex/prefix/metachars/empty-prefix/substring) |
| 3 | 3.7 query-after-drop + log queryable | `P tests/parity/` (whole dir) | drop→query error parity; `log` queryable; FetchLogs/Query agree | extend `test_query.py` |
| 4 | 4a CompactionConfig + planner (pure) | `R cargo test -p finelog planner:: compaction:: policy::` | run selection, byte/count caps, terminal-level exclusion, aggregate_key_bounds | — |
| 4 | 4b native k-way merge + level-bump | `R cargo test -p finelog merge:: executor::` | (key,seq) merge, rename-on-bump, additive null-fill | — |
| 4 | 4c test-only admin surface (`--debug-admin`) | `BUILD` then `P test_compaction.py::test_debug_maintain_promotes_l0` | both backends expose `/debug/maintain` + `/debug/segments` | `test_debug_maintain_promotes_l0` |
| 4 | 4d wire compaction + maintenance task (local) | `BUILD` then `P test_compaction.py test_catalog_stats.py` | per-ns maintenance task, atomic commit_swap, LOCAL evict, stats roll-up | compaction-via-stats, post-compaction query, catalog-stats lifecycle |
| 4 | 4e object_store GCS sync + boot reconcile | `BUILD` then `P test_offload.py test_eviction.py test_storage_policy.py` | **compaction/eviction/remote family green on rust** | offload/eviction/policy/age/wiped-catalog-recover/redundancy-drop |
| 5 | 5a app builder: ConnectRpcService + 64MB Limits + zstd/gzip | `BUILD` then `P tests/parity/` | RPC services on one Router via `register`, large-write OK | `test_health`, `test_constant_query_round_trips`, `test_large_write_rows_within_limit` |
| 5 | 5b SlowRpcInterceptor | `R cargo test -p finelog server::interceptors` | per-method WARN threshold (default 7000, ≤0 disables) | — (unit only; no log-string parity) |
| 5 | 5c ConcurrencyInterceptor (fetch=4, query=4) + deadline shed | `R cargo test -p finelog server::interceptors` then `P test_server.py` | per-method semaphore caps, deadline shedding | `test_fetch_logs_concurrency_cap`, `test_query_concurrency_cap` |
| 5 | 5d static Vue SPA + base-href rewrite | `R cargo test -p finelog server::spa` then `P test_spa.py` | `/static`, `/favicon.ico`, SPA fallback, `X-Forwarded-Prefix` | `test_spa_and_static_served` |
| 5 | 5e legacy `/iris.logging.LogService/*` rewrite | `BUILD` then `P test_server.py::test_legacy_iris_logging_path_compat` | transport-layer path rewrite | `test_legacy_iris_logging_path_compat` |
| 5 | 5f graceful SIGTERM/SIGINT shutdown drain | `BUILD` then `P test_server.py::test_clean_shutdown_after_durable_write` | clean exit-0 drain of bg tasks | `test_clean_shutdown_after_durable_write` |
| 5 | 5g periodic pool/RSS diagnostics line | `R cargo test -p finelog server::diagnostics` | `/proc/self/status` parse + memory_summary line | — (unit only) |
| 6 | 6a seg-filename + footer-bounds primitives | `R cargo test -p finelog footer:: segment_name::` | filename grammar, footer-only metadata | — |
| 6 | 6b directory scan → in-memory catalog | `R cargo test -p finelog adopt::` | adopt namespace/store from disk, recover_next_seq, aggregates | — |
| 6 | 6c schema recovery from parquet footer | `R cargo test -p finelog adopt::schema` | recover proto Schema + key_column from footer | — |
| 6 | 6d sentinel state machine + idempotent adoption | `R cargo test -p finelog adopt::sentinel` | crash-safe boot adoption before bind | — |
| 6 | 6e remote (GCS) segment adoption | `R cargo test -p finelog adopt::remote` | REMOTE row adoption + redundancy prune | — |
| 6 | 6f cross-backend cutover parity (THE GATE) | `P test_cutover.py` | **Python-writes → Rust-reads identical** | cutover stats/query/log/schema/idempotent |
| 6 | 6g deploy: Rust binary in image | `docker build … && finelog-server --help`; `P test_deploy_k8s.py` | image builds Rust server; manifests render | `test_deploy_k8s_manifests_render` |
| 6 | 6h delete Python store/server; drop deps | `P tests/parity/ test_client.py test_config.py test_deploy_cli.py test_deploy_k8s.py`; `uv run pyrefly`; `./infra/pre-commit.py --all-files` | dead code removed, deps dropped, gcs-query rerouted | full surviving suite |

---

## 3. Per-phase sections

### Phase 1 — Catalog & metadata RPCs

**Goal.** Stand up the StatsService metadata surface so RegisterTable / GetTableSchema /
ListNamespaces / DropTable behave byte-for-byte like Python over RPC, backed by a **Rust-owned
catalog** (`{data_dir}/_finelog_catalog.sqlite` via `rusqlite` bundled). No data path yet:
WriteRows/Query/PushLogs/FetchLogs return `Unimplemented`; `ListNamespaces` stats are all-zero.
The catalog and on-disk layout are laid down now so Phase 2 inserts segment rows and Phase 6
rebuilds from disk **without a schema change**.

**Ordered sub-stages (acceptance commands inline):**

- **S1** `store/schema.rs` — `Column`, `Schema`, `ColumnType↔Arrow DataType` map (all 7 types,
  `TimestampMs→Timestamp(Millisecond,None)`), `schema_from_proto_view` (reject `COLUMN_TYPE_UNKNOWN`,
  unknown int, reserved `seq`), `schema_to_proto_owned` (**strips** implicit seq — wire schema
  never carries seq), `schema_to/from_json` (proto enum NAME + legacy lowercase fallback),
  `with_implicit_seq`, `resolve_key_column` (**presence-only**, default `timestamp_ms`; do NOT
  enforce the INT64/TIMESTAMP_MS proto comment — match Python). Constants
  `IMPLICIT_KEY_COLUMN="timestamp_ms"`, `IMPLICIT_SEQ_COLUMN="seq"`, `MAX_WRITE_ROWS_BYTES=16MiB`,
  `MAX_WRITE_ROWS_ROWS=1<<20` (declared now). `arrow=58` added now to avoid a mid-stream bump.
  Acc: `cargo test -p finelog schema::`.
- **S2** `errors.rs` — `StatsError` enum + `From<StatsError> for ConnectError`. **Mapping is
  load-bearing:** `SchemaConflict→failed_precondition` (NOT already_exists), `SchemaValidation`/
  `InvalidNamespace→invalid_argument`, `NamespaceNotFound→not_found`, `QueryResultTooLarge→
  resource_exhausted`, `Internal→internal`. Acc: `cargo test -p finelog errors::`.
- **S3** `merge_schemas` (additive-only; differing `key_column` is a **hint → coerce to
  registered with a warn, not reject**) + `store/namespace_name.rs` (`^[a-z][a-z0-9_.-]{0,63}$`,
  strict path-containment rejecting `..`/escapes, `data_dir=None` still enforces the regex). Adds
  `regex="1"`. Acc: `cargo test -p finelog merge_schemas namespace_name`.
- **S4** `store/policy.rs` — `StoragePolicy{Option<i32/i64/i64>}`, proto3 `0↔None`, `is_empty`.
  Acc: `cargo test -p finelog policy::`.
- **S5** `store/catalog.rs` — rusqlite sidecar with the **final** logical tables
  (`namespaces`, `storage_policies`, `segments`). `segments` is created but **empty** in P1 so
  `list_segments`/`aggregate_namespace_stats` return zeros and Phase 2 just inserts. One
  `std::sync::Mutex` over `{conn, live: BTreeMap<String,RegisteredNamespace>, registered_at,
  next_ordinal, dropping}`. `register_or_evolve`, `begin_drop/finish_drop/is_dropping`,
  `upsert/get_policy/upsert_policy`, `aggregate_namespace_stats`, `snapshot_live` (registration
  order). `data_dir=None` → in-memory rusqlite. Acc: `cargo test -p finelog catalog::`.
- **S6** `store/store.rs` — construct (mkdir, rehydrate from `list_all`, ensure privileged `log`
  namespace via `with_implicit_seq(LOG_REGISTERED_SCHEMA)`), `register_table`,
  `get_table_schema`, `list_namespaces_with_stats`, `drop_table` (reject `log`; begin→delete→
  finish). **Re-register with empty policy keeps existing**; register returns store-form
  effective schema **with seq** (the wire encoder strips it). Acc: `cargo test -p finelog store::`.
- **S7** `server/stats_service.rs` + `server/log_service.rs` + `server/mod.rs` — implement the 4
  metadata RPCs (others `Unimplemented`); build **`ConnectRpcService::new(router).with_limits(
  Limits::default().max_message_size(64<<20).max_request_body_size(64<<20))`** as
  `.fallback_service` (NOT `into_axum_service()`). Wrap blocking rusqlite calls in
  `spawn_blocking`. Return **owned** response messages. Acc: `BUILD`.
- **S8** `tests/parity/test_metadata.py` — RPC-driven register/merge/name/get/list/drop; assert
  on the **wire** effective_schema (seq stripped). Drop `rust_pending` for the metadata family.
  Acc: `BUILD` then `PARITY test_metadata.py test_smoke.py`.

**Rust module map:** `store/schema.rs`, `errors.rs`, `store/policy.rs`, `store/namespace_name.rs`,
`store/catalog.rs`, `store/types.rs` (SegmentLocation/SegmentRow/NamespaceStats declared now),
`store/store.rs`, `server/{stats_service,log_service,mod}.rs`.

**Python-reference map:** `store/schema.py` (Column/Schema, `_ARROW_TYPE_FOR`, proto/json conv,
`with_implicit_seq`, `resolve_key_column` presence-only, `merge_schemas`), `store/catalog.py`
(state model + method surface; **do not** port the DuckDB connection — Rust sidecar, different
filename), `store/policy.py`, `store/duckdb_store.py` (`_NAMESPACE_NAME_RE`,
`_validate_namespace_name`, `register_table`, `drop_table`, `_ensure_log_namespace_registered`,
`_rehydrate_from_registry`), `server/stats_service.py` (exact error→code mapping),
`store/log_namespace.py` (`LOG_REGISTERED_SCHEMA` only), `errors.py`, migrations 0001–0007 (final
table shapes only — no migration runner), `store/compactor.py` (`seg_filename`/`parse_seg_filename`
declared now for P2/P6).

**Parity gates (S8):** valid/invalid names, path traversal, missing-ordering-key, implicit/explicit
key, idempotent re-register, subset-returns-full, additive-nullable merge, type-change→
`FAILED_PRECONDITION`, non-nullable-new→`FAILED_PRECONDITION`, key-hint coerce, GetTableSchema
round-trip + unknown→`NOT_FOUND`, ListNamespaces includes `log` + registered ns with zeroed stats,
drop registered-empty/unknown→`NOT_FOUND`/`log`→`INVALID_ARGUMENT`.

**Risks:** (1) wire-vs-store schema (parity asserts wire seq-stripped, cargo asserts store
seq-present); (2) SchemaConflict→`failed_precondition` not `already_exists`; (3) catalog filename
must differ from DuckDB's `_finelog_registry.duckdb`; (4) empty-policy-keeps-existing lives in the
store, not the decoder; (5) buffa view shapes per recon §3/§4; (6) return owned messages
(JSON-codec trap); (7) `resolve_key_column` presence-only.

---

### Phase 2 — Write path & durability

**Goal.** WriteRows + PushLogs end-to-end: IPC decode → validate/align → stamp monotonic `seq`
under a per-namespace insertion lock → LSM RAM buffer → per-namespace flush task draining to an
**UNSORTED** L0 parquet segment + catalog row → handler blocks on a `persisted_seq` watch so the
RPC returns **only after** the rows are durable. Verified externally via `ListNamespaces` stats.

**Ordered sub-stages:**

- **2a** Add `arrow=58, arrow-ipc=58, parquet={58, features=["async","object_store"]}, bytes,
  futures` (NOT datafusion). `store/schema.rs`: `arrow_to_column_type` (decode Dictionary to value
  type; reject List/LargeList/Struct/Union/Map), `decode_dictionary_columns`, `AlignedBatch`,
  `validate_and_align_batch` (schema.py:386 exactly), `MAX_WRITE_ROWS_*`. `store/ipc.rs`:
  `decode_one_record_batch` (exactly-one-batch contract) + `encode_ipc` (placed now, used by P3).
  Acc: `cargo build && cargo test -p finelog schema::`; verify `cargo tree -p finelog | grep
  'arrow v'` shows exactly one (58).
- **2b** `store/ram_buffer.rs` — `RamBuffers` LSM (`chunks[i-1].rows > chunks[i].rows`),
  `allocate_seq`, `append_table` (cascade-merge tail via `concat_batches`), explicit O(1)
  byte/row accounting (`added_bytes = AlignedBatch.byte_size + 8*num_rows`), `seal/commit_flush/
  restore_flush`, `stamp_seq_and_build` (seq `Int64 [first..first+n)` in registered column order +
  additive NULL-fill). Acc: `cargo test -p finelog ram_buffer::`.
- **2c** `store/segment.rs` — `seg_filename(level,min_seq)="seg_L{level}_{min_seq:019}.parquet"`,
  `parse_seg_filename`, `write_segment` (**UNSORTED L0**, `set_max_row_group_row_count(Some(16384))`,
  zstd, staging `.parquet.tmp`+atomic rename), `read_segment_footer` (num_rows; **min_seq from
  FILENAME**; max_seq=min_seq+rows-1; key Int64 min/max via `Statistics::min_opt/max_opt`),
  `recover_next_seq`. Acc: `cargo test -p finelog segment::`.
- **2d** `store/namespace.rs` — `DiskNamespace` with `Mutex<NsInner{buffers, local_segments:
  VecDeque, next_seq}>` + `watch::Sender<i64> persisted_seq` (init `-1`) + flush `Notify`.
  `append_aligned_batch`, `append_log_batch` (prep 5 log columns outside the lock),
  `await_persisted(target)` (watch + `tokio::time::timeout`), flush task (seal → `spawn_blocking`
  parquet write+rename → under lock push LocalSegment + commit_flush + catalog upsert → **THEN**
  `watch.send(max_seq)`), `stats()` with the exact Python seq-window math, boot
  `recover_next_seq` + local-segment adoption. Acc: `cargo test -p finelog namespace::`.
- **2e** Wire `write_rows` (size/row caps → decode → validate/align → append → durability await
  bounded by `ctx.time_remaining()` default 30s → `WriteRowsResponse`) and `push_logs` (empty→
  empty; else append to reserved `log` ns + await). Ensure `Store::open` auto-registers `log`
  (idempotent with P1's S6). Acc: `BUILD`.
- **2f** `tests/parity/test_write_path.py` — RPC parity; drop `rust_pending` for the write family.
  Acc: `BUILD` then `PARITY test_write_path.py`.

**Rust module map:** `store/{schema,ipc,ram_buffer,segment,namespace,types}.rs`,
`server/{stats_service,log_service,mod}.rs`.

**Python-reference map:** `store/schema.py` (`validate_and_align_batch`, `_arrow_to_column_type`,
`_decode_dictionary_columns`), `store/log_namespace.py` (`RamBuffers`,
`_maintain_chunk_invariant`, `_stamp_seq_and_build`, `_SealedBuffer`, `append_aligned_batch`,
`append_log_batch`, `_write_new_segment` (UNSORTED), `_recover_next_seq`,
`_read_segment_metadata`, `stats()` lines 1725–1753, `flush()`, `LOG_REGISTERED_SCHEMA`,
`_ROW_GROUP_SIZE=16384`), `store/duckdb_store.py` (`write_rows`, `_decode_single_record_batch`,
`SEGMENT_TARGET_BYTES=100MiB`, `DEFAULT_FLUSH_INTERVAL_SEC=5.0`), `server/stats_service.py`
(`write_rows` handler), `server/service.py` (`await_persisted`,
`DEFAULT_PERSIST_TIMEOUT_SEC=30.0`, `push_logs`), `store/compactor.py` (`seg_filename`),
`tests/conftest.py` (`_ipc_bytes`, `_worker_schema`, `_worker_batch`).

**Parity gates (2f):** round-trip stats (row_count/min_seq/max_seq), missing-nullable null-fill,
schema-violation codes (unknown-ns→`NOT_FOUND`; missing-non-nullable/unknown-col/type-mismatch/
nested/oversize/too-many→`INVALID_ARGUMENT`), dictionary decode, **durable-before-ack via restart**
(second process on same `--log-dir` sees the rows), PushLogs durability, 8-way concurrent writes
(distinct monotonic seq 1..8), RAM-buffer stats seq-window (3-then-2 → row_count 5, min 1, max 5).

**Risks:** L0 must stay UNSORTED (sort is Phase-4 only); `watch.send` strictly after file+catalog
land (durability-before-ack ordering); stats seq-window off-by-one; 64MB limits set at 2e; return
owned messages; deadline handling (`tokio::time::timeout`, no auto-cancel); pin arrow/parquet 58
now to avoid Phase-3 churn.

---

### Phase 3 — Query path (DataFusion) & FetchLogs

**Goal.** `StatsService.Query` (user SQL verbatim against **every live namespace**, result as one
Arrow IPC stream + row_count) and `LogService.FetchLogs` (EXACT/PREFIX/REGEX + since_ms + cursor
(exclusive) + substring + min_level + tail + max_lines over the reserved `log` namespace).
**Query visibility = sealed L≥0 parquet segments ONLY** (NOT the RAM buffer — matches
`query_snapshot`); the durability contract makes this invisible to clients because writes are
sealed before ack. DataFusion replaces DuckDB; dialect diffs are gated by the golden corpus.

**Ordered sub-stages:**

- **3.0** Add `datafusion=53, object_store={0.13, gcp}, async-trait, url, bytes, futures`
  (**version correction:** DataFusion **53.1**, arrow/parquet **58**, NOT the proposal's 52/57.1).
  `query/mod.rs::make_ctx()` (`sql_parser.map_string_types_to_utf8view=false` for Utf8 result
  parity, `dialect="DuckDB"`, leave `enable_ident_normalization` at the DF53 default true since the
  corpus quotes dotted identifiers); `arrow_ipc.rs` `encode_ipc`/`decode_ipc` (empty result still
  emits schema+EOS). Acc: `cargo build && cargo test -p finelog query::tests::select_one_roundtrips
  arrow_ipc::`; verify single arrow 58.
- **3.1** `query/provider.rs::NamespaceProvider` — per-namespace `TableProvider` over the snapshot
  of **sealed** segment paths (`ListingTable` with `.with_schema`); empty segment list → typed
  empty `MemTable`/`EmptyExec` carrying the registered schema (incl. `seq`); `supports_filters_
  pushdown=Inexact`. `run_query_over(ctx, namespaces, sql)` registers all live namespaces, runs
  SQL, collects, deregisters. Acc: `cargo test -p finelog query::provider::`.
- **3.2** `StatsService::query` handler — snapshot live registry under the **query-visibility read
  guard**, register all namespaces, run SQL, encode IPC, return **owned** `QueryResponse`.
  Errors: plan/parse/resolve → `invalid_argument` (the DuckDB `CatalogException` slot);
  oversize-IPC > max_message_size → `resource_exhausted` (the `QueryResultTooLargeError` analog —
  **no server row cap**, matching Python). Un-xfail `test_constant_query_round_trips`. Acc:
  `BUILD` then `PARITY test_smoke.py`.
- **3.3** `tests/parity/test_query.py` — RPC parity: write→query (no manual seal), typed-empty,
  WHERE, multi-ns JOIN, unknown-ns→`invalid_argument`. Acc: `PARITY test_query.py`.
- **3.4** `query/udf.rs` — register `prefix(text,prefix)->bool`, `regexp_matches(text,pattern)
  ->bool`, `contains(text,sub)->bool` via `create_udf` (used by the corpus **and** FetchLogs).
  `tests/parity/test_golden_corpus.py` seeds a fixed dataset and diffs decoded Arrow on both
  backends (this diff **is** the dialect gate). Corpus sources:
  `iris/scripts/job_profile_summary.py:158`, `deploy/cli.py`, `test_query` shapes. Acc:
  `cargo test -p finelog query::udf::` then `PARITY test_golden_corpus.py`.
- **3.5** `store/log_read.rs` (pure) — `build_log_predicates` (EXACT `key=source AND seq>cursor`,
  literal — metachars never reinterpreted; PREFIX `prefix(key,source)`, empty source→
  `invalid_argument`; REGEX leading-literal-prefix prune + `regexp_matches`), `add_common_filters`
  (since_ms/substring(`contains`)/min_level `(level=0 OR level>=min)`), `shape_log_read_result`
  (tail-reverse, cursor=max(seq) or default), `parse_attempt_id`, `str_to_log_level`,
  `regex_literal_prefix`. `LogService::fetch_logs` maps wire UNSPECIFIED→REGEX, max_lines≤0→1000.
  Acc: `cargo test -p finelog store::log_read::` then `BUILD`.
- **3.6** `tests/parity/test_fetch_logs.py` — PushLogs+FetchLogs RPC parity; drop `rust_pending`
  for the fetch family. Acc: `PARITY test_fetch_logs.py test_smoke.py test_query.py
  test_golden_corpus.py`.
- **3.7** Extend `test_query.py` — query-after-drop→`invalid_argument`, `log` queryable via SQL,
  FetchLogs+Query agree on `log` row count. Acc: `PARITY tests/parity/` (whole dir) — **Phase-3
  exit gate**.

**Rust module map:** `query/{mod,provider,udf}.rs`, `arrow_ipc.rs`,
`server/{stats_service,log_service,errors}.rs`, `store/log_read.rs`.

**Python-reference map:** `store/duckdb_store.py` (`query` lines 546–592: verbatim SQL,
sealed-only visibility, all-namespaces registration, typed-empty), `store/log_namespace.py`
(`query_snapshot` 1659–1664 = local segments only, `get_logs`, `_scope_query`,
`_add_common_filters`, `_shape_log_read_result`, `_execute_read`), `server/service.py`
(`fetch_logs` UNSPECIFIED→REGEX, max_lines 1000), `server/stats_service.py` (`query` error
mapping, `_arrow_table_to_ipc_bytes`), `types.py` (`parse_attempt_id`, `str_to_log_level`),
`store/cursor.py` (cursor-advance contract), `store/schema.py` (`schema_to_arrow` for typed-empty),
`iris/scripts/job_profile_summary.py:158` (golden corpus), `deploy/cli.py` (`query_cmd` — server
has no row cap).

**Parity gates:** the matrix rows for 3.2–3.7 (Query round-trips, typed-empty, WHERE/JOIN,
unknown-ns, golden corpus, FetchLogs roundtrip/tail/cursor-exclusive/regex/prefix/metachar-safety/
empty-prefix/substring-literal, query-after-drop, log-queryable-consistent).

**Risks:** **TOP** — SQL dialect divergence, gated by the golden corpus + compat UDFs;
`enable_ident_normalization=true` in DF53 (corpus quotes idents, safe); **visibility must be
sealed-only (no RAM union)** — building a MemTable union over-exposes vs Python; Utf8 vs Utf8View
(`map_string_types_to_utf8view=false`); regex-metachar literal safety (#5392); no server row cap;
hard dependency on Phases 1+2; hold the read guard across `collect()` so compaction (Phase 4)
can't unlink a file mid-scan.

---

### Phase 4 — Compaction & eviction (native) + remote sync

**Goal.** Per-namespace maintenance pipeline: (1) native arrow k-way-merge leveled compactor
(planner + executor, no SQL), (2) maintenance task evicting terminal-level BOTH segments by
count/bytes/age under per-namespace StoragePolicy, (3) `object_store` GCS sync flipping
`LOCAL→BOTH→REMOTE`. Driven over RPC via the flag-gated `--debug-admin` surface so the harness can
force flush/compact/sync/evict deterministically and read per-segment level+location.

**Ordered sub-stages:**

- **4a** `store/compaction/config.rs` (`CompactionConfig`: `level_targets=[64,256,256]MiB`,
  **`max_segments_per_level=32`** (the proposal's "4" is stale; Python uses 32),
  `max_segments_per_namespace=1000`, `max_bytes_per_namespace=100GiB`) + `store/policy.rs` (reuse
  P1) + `store/compaction/planner.rs` (`plan`: contiguous-run selection, byte-target prefix cap,
  count cap, terminal-level never selected; `compaction_sort_keys`, `aggregate_key_bounds`). Pure.
  Acc: `cargo test -p finelog planner:: compaction:: policy::`.
- **4b** `store/compaction/merge.rs` (`kway_merge` via `RowConverter` + `BinaryHeap` +
  `interleave`, chunked at 16384; `project_to_schema` additive null-fill) + `executor.rs`
  (`apply_level_bump` = **rename** single input, no rewrite, preserve created_at_ms; `apply_merge`
  = read inputs via `ParquetRecordBatchReaderBuilder` under `spawn_blocking`, merge, write via
  `ArrowWriter` rg=16384 zstd) + `segment::segment_bounds`. Acc: `cargo test -p finelog merge::
  executor::`.
- **4c** `--debug-admin` flag + `server/debug.rs` (`POST /debug/maintain{namespace,
  force_compact_l0?}` runs synchronous flush→compact→sync→evict via the **same** body the bg task
  uses; `GET /debug/segments?namespace` → JSON `[{path, level, min_seq, max_seq, row_count,
  byte_size, location, created_at_ms}]`). **Mount before the connect fallback** (recon §9). Add the
  matching `--debug-admin` handler to the Python `server/main.py` (calls existing
  `flush()/force_compact_l0()/compact()` + `catalog.list_segments`). Python helpers
  `maintain()/segments()` in `conftest.py`, thread `--debug-admin` into both backends' commands.
  Acc: `BUILD` then `PARITY test_compaction.py::test_debug_maintain_promotes_l0`.
- **4d** Per-namespace maintenance tokio task (replaces `_maint_loop`): drain planner → run_job →
  `eviction_step`. `commit_swap` (atomic deque+catalog splice under query-visibility write lock +
  insertion lock; swap the `ListingTable` so queries never see a half-spliced set). LOCAL-only
  eviction (`evict_segment`: BOTH→REMOTE+unlink; LOCAL-only→drop row+unlink). `stats()` excludes
  REMOTE. Acc: `BUILD` then `PARITY test_compaction.py test_catalog_stats.py`.
- **4e** `store/remote.rs` (`build_remote_store`: `gs://`→`GoogleCloudStorageBuilder`, else
  `LocalFileSystem` for tests; upload/list/delete via `ObjectStoreExt`) + `sync_step` (two-phase:
  upload all L≥1 LOCAL rows tracking `all_durable`; then orphan-delete remote files with no
  catalog row **only if all_durable**) + age-trim + `store/reconcile.rs::reconcile_remote_segments`
  (boot: adopt unknown remote parquet as REMOTE via async footer; redundancy-drop covered
  segments). `/debug/backdate?namespace&created_at_ms` (RPC-only age tests, no sleep). Acc:
  `BUILD` then `PARITY test_offload.py test_eviction.py test_storage_policy.py`.

**Rust module map:** `store/compaction/{config,planner,merge,executor}.rs`, `store/segment.rs`,
`store/policy.rs`, `store/remote.rs`, `store/reconcile.rs`, `store/namespace.rs` (maintenance
task), `store/catalog.rs` (`replace_segments`, `select_eviction_candidate`,
`select_aged_eviction_candidate`, `set_location`, `remove_segment`), `server/debug.rs`,
`main.rs` (`--debug-admin`).

**Python-reference map:** `store/compactor.py` (`plan`, `_contiguous_runs`, `_take_until_target`,
`_build_job`, `compaction_sort_keys`, `aggregate_key_bounds`, filename helpers — **NOT**
`merge_sql`), `store/log_namespace.py` (`_compaction_step`, `_run_job`, `_apply_level_bump`,
`_apply_merge`, `_commit_swap`, `_sync_step`, `_upload`, `_mark_uploaded`, `_eviction_step`,
`evict_segment`, `_reconcile_remote_segments`, `stats()` REMOTE-excluded, `force_compact_l0`,
`compact()`), `store/policy.py`, `store/types.py` (`SegmentLocation`, SegmentRow fields),
`store/catalog.py` (selectors, `replace_segments`, `set_location`), `store/duckdb_store.py`
(`list_namespaces_with_stats` roll-up).

**Parity gates (4c–4e):** compaction-promotes-L0 (stats+query), eviction-drops-oldest-BOTH,
eviction-skips-not-yet-uploaded, per-namespace-policy-overrides-global, age-eviction-by-created_at,
sync-uploads+deletes-orphans, eviction-preserves-remote-archive, wiped-catalog-recovers-at-boot,
stale-input-redundancy-drop.

**Risks:** RPC can't force/observe bg actions → flag-gated admin surface (sanctioned by §8);
native merge correctness (gated by merge unit tests + post-compaction query parity);
query-visibility race during commit_swap (write lock drains readers); object_store 0.13 API drift
(`ObjectStoreExt` import; `set_max_row_group_row_count`); two-phase sync ordering (data-safety);
age tests need `/debug/backdate` (no sleep); boot reconcile needs a **restartable-backend
fixture** (extend the harness to respawn on the same `log_dir`+`remote_log_dir`).

---

### Phase 5 — Server hardening

**Goal.** Behavioral parity with the Python ASGI shell: per-method concurrency caps (FetchLogs=4,
Query=4) with post-acquire deadline shedding, slow-RPC WARNING logging (default 7000ms, ≤0
disables), zstd+gzip, ≥64MB limits, static Vue SPA serving (+`X-Forwarded-Prefix` base-href),
graceful SIGTERM/SIGINT shutdown draining bg tasks, periodic pool/RSS diagnostics, legacy
`/iris.logging.LogService/*`→`/finelog.logging.LogService/*` rewrite.

**Ordered sub-stages:**

- **5a** `server/app.rs::build_app(store, config)` — register **both** services on one
  `connectrpc::Router` via the generated `register()` (required so `ctx.spec()`/`ctx.path()` are
  populated for the interceptors), wrap in `ConnectRpcService` with 64MB limits + default
  zstd/gzip; `axum::Router` with `/health` + `.fallback_service(connect)`. Switch `main.rs` off the
  bare `/health` router. Acc: `BUILD` then `PARITY tests/parity/` (incl. `test_large_write_rows_
  within_limit`).
- **5b** `server/interceptors.rs::SlowRpcInterceptor` — time `next.run`, look up per-method ms
  threshold via `ctx.spec().method()`, `tracing::warn!` over threshold, ≤0 disables. Registered
  **first** (outermost). Acc: `cargo test -p finelog server::interceptors`.
- **5c** `ConcurrencyInterceptor` — two `tokio::sync::Semaphore` (fetch=4, query=4) keyed by
  method; acquire **before** `next.run`, hold across the handler; **post-acquire deadline shed**
  (`time_remaining().is_some_and(|d| d.is_zero())` → `deadline_exceeded`). Caps are constructor
  params so the parity test can lower them. Registered **after** SlowRpc. Acc: `cargo test -p
  finelog server::interceptors` then `PARITY test_server.py`.
- **5d** `server/spa.rs` — `vue_dist_dir()` (dist OR `/app/dashboard/dist` OR env override),
  `index_html_with_base()` (byte-exact base-href rewrite), `/static` via
  `tower_http::services::ServeDir` (add `tower-http={"0.6", features=["fs"]}`), `/favicon.ico`,
  `/` + `/{*rest}` SPA fallback, `NOT_BUILT_HTML` placeholder. Register **before** the connect
  fallback. Acc: `cargo test -p finelog server::spa` then `PARITY test_spa.py`.
- **5e** `server/legacy_path.rs` — `axum::middleware::from_fn` mutating `req.uri()` for the legacy
  prefix (**transport-layer**, not an Interceptor). **Still live** (iris dashboard alias +
  test_server exercise it); port it, flag for later removal. Acc: `BUILD` then `PARITY
  test_server.py::test_legacy_iris_logging_path_compat`.
- **5f** `main.rs` — `axum::serve(...).with_graceful_shutdown(shutdown_signal())` (SIGTERM/SIGINT);
  after serve returns, `store.shutdown().await` (cooperatively cancel+join per-namespace tasks with
  a bounded timeout). Acc: `BUILD` then `PARITY test_server.py::test_clean_shutdown_after_durable_
  write`.
- **5g** `server/diagnostics.rs` — 60s task: `/proc/self/status` VmRSS/VmSize parse +
  `Store::memory_summary()` (namespaces/ram_bytes/chunks). No pyarrow pool fields. Cancelled on
  shutdown. Acc: `cargo test -p finelog server::diagnostics`.

**Rust module map:** `server/{mod,app,interceptors,spa,legacy_path,diagnostics}.rs`, `main.rs`.

**Python-reference map:** `server/asgi.py` (`build_log_server_asgi` wiring, `_DEFAULT_
COMPRESSIONS`, `_MAX_CONCURRENT_*=4`, `_vue_dist_dir`, `_index_html_with_base`, `_NOT_BUILT_HTML`,
SPA routes, `_LegacyIrisLoggingPathMiddleware`, `/health`), `server/interceptors.py`
(`SlowRpcInterceptor`, `DEFAULT_SLOW_RPC_THRESHOLD_MS=7000`), `server/main.py` (`run_log_server`,
`_emit_pool_diagnostics`, `_read_proc_self_status_kb`), `rigging/log_setup.py` (`slow_log`),
`rigging/rpc.py` (`ConcurrencyLimitInterceptor`, `_deadline_expired`), `store/duckdb_store.py`
(`memory_summary()` shape).

**Parity gates:** concurrency caps (peak in-flight == cap), legacy-path round-trip, push→fetch
round-trip, UNSPECIFIED→REGEX through the stack, register/write/query/drop via RPC,
large-write-within-limit, clean-shutdown, SPA+static served (and RPC POSTs still reach connect).

**Risks:** concurrency parity is timing-sensitive → assert exact cap in **cargo** via `run_chain`
+ a parked terminal (no sleep), lower the cap + env-gated test delay for the RPC test; `ctx.spec()`
only populated via `register()` (never `into_axum_service`); SPA route precedence (connect must be
fallback); legacy path may be removable but verify first; shutdown must not hang (bounded join +
outer timeout); axum 0.7/0.8 dup is pre-existing; diagnostics/slow-RPC are log lines → **no
log-string parity tests** (cargo unit tests only).

---

### Phase 6 — Production cutover (rebuild-from-disk + deploy)

**Goal.** Cut prod from Python to Rust with no data loss and no change in queryable contents by
(a) reconstructing the Rust catalog **purely by scanning the on-disk parquet layout + footers**
(never reading the DuckDB sidecar) as a one-time, idempotent, sentinel-gated boot step, and (b)
building the Rust binary into the Docker image, doing a flag-day Recreate cutover, then deleting
the Python store/server and dropping unused Python deps.

**Ordered sub-stages:**

- **6a** `store/segment_name.rs` (reuse the P2/P4 helper — search first) + `store/footer.rs`
  (`read_segment_footer`: footer-only, `min_seq` from filename, `max_seq=min_seq+rows-1`, key
  bounds; empty/corrupt → empty marker, never hard error). Acc: `cargo test -p finelog footer::
  segment_name::`.
- **6b** `store/adopt.rs::adopt_namespace_from_disk` (sorted by min_seq, `created_at_ms` from
  mtime, location=LOCAL) + `adopt_store_from_disk` (enumerate ns subdirs, populate catalog +
  `recover_next_seq`). Aggregates must equal `aggregate_namespace_stats`. Footer reads under
  `spawn_blocking`. Acc: `cargo test -p finelog adopt::`.
- **6c** `recover_schema_from_segments` — Arrow schema from the newest segment's footer →
  proto Schema (reuse P1 conversion + `resolve_key_column`); re-mark implicit `seq`. Adopted
  namespaces start with **empty (inherit) StoragePolicy** (re-established by deploy's startup
  RegisterTable). Acc: `cargo test -p finelog adopt::schema`.
- **6d** Sentinel `{data_dir}/.finelog-rust-catalog` (single-line JSON, atomic tmp+rename, fast
  path on `done`, in-progress/missing re-runs idempotently — the directory IS the journal).
  `ensure_catalog_adopted` runs **before** axum binds; **assert** the per-namespace seg_L layout
  and **fail loudly** on a flat layout (do NOT port `layout_migration`). Acc: `cargo test -p
  finelog adopt::sentinel`.
- **6e** `adopt_remote_segments` — when `--remote-log-dir` set, list remote, adopt unknown
  parquet as REMOTE via async footer, redundancy-drop covered segments (reuse P4 `remote.rs`).
  No-op when unset. Acc: `cargo test -p finelog adopt::remote`.
- **6f** `tests/parity/test_cutover.py` — **two-phase fixture**: Python writes a corpus → stop →
  Rust boots on the same `log_dir` (no sidecar/sentinel) → assert identical NamespaceInfo
  (exact row_count/min_seq/max_seq/segment_count; byte_size>0 not equal), identical Query Arrow,
  matching GetTableSchema, log round-trip, idempotent second boot. **THE PHASE GATE.** Acc:
  `PARITY test_cutover.py`.
- **6g** `deploy/Dockerfile` — add a `rust:1.95` build stage (`cargo build --release -p finelog
  --bin finelog-server`), copy the binary into runtime, switch CMD to the binary (same `FINELOG_*`
  env), keep the dashboard SPA stage. `02-deployment.yaml.tmpl` keeps `replicas=1`/`Recreate`
  (single-writer; cutover = Recreate onto the same PV → triggers adoption). Acc: `docker build -f
  lib/finelog/deploy/Dockerfile -t finelog:rust .` + `finelog-server --help`; `PARITY
  test_deploy_k8s.py`.
- **6h** Delete `src/finelog/store/` + `src/finelog/server/`; delete the ~19 internal-store tests
  (behavior now covered by parity). Drop `duckdb`/`fsspec`/`gcsfs` from base deps **after**
  rerouting `deploy/cli.py` `gcs-query` through the live server's `Query` RPC (LogClient). Keep
  `client/`, `deploy/`, protos, `tests/parity/`, `types.py`/`errors.py` if still imported. Acc:
  `PARITY tests/parity/ test_client.py test_config.py test_deploy_cli.py test_deploy_k8s.py`;
  `uv run pyrefly`; `./infra/pre-commit.py --all-files`.

**Rust module map:** `store/{segment_name,footer,adopt,reconcile,key_bound}.rs`, `main.rs`
bootstrap, `deploy/Dockerfile`, `tests/parity/test_cutover.py`.

**Python-reference map:** `store/layout_migration.py` (sentinel pattern ONLY), `store/log_
namespace.py` (`_read_segment_metadata`, `_key_bounds_from_parquet`, `_discover_segments`,
`_recover_next_seq`, boot reconcile passes, `_segment_to_row`, `_reconcile_remote_segments`),
`store/duckdb_store.py` (`_rehydrate_from_registry`, `_ensure_log_namespace_registered`),
`store/catalog.py` (`aggregate_namespace_stats`), `store/compactor.py` (filename helpers),
`store/types.py`, `store/schema.py` (`resolve_key_column`, arrow→proto), migrations 0003
(historical naming reference only), `deploy/Dockerfile` + `02-deployment.yaml.tmpl`,
`deploy/cli.py` (`query_cmd`/`_register_namespace_views` → reroute to Query RPC).

**Parity gates:** cutover stats (exact int equality), cutover query identical, cutover log
round-trip, cutover schema recovered, idempotent second boot, deploy manifests render.

**Risks:** schema recovery is **lossy** vs the sidecar (additive-merge history, non-default
key_column) — mitigated by full-arrow-schema read + `resolve_key_column` + deploy's startup
RegisterTable + the cutover parity gate; min_seq from filename (not footer); policy not in parquet
(re-established via RegisterTable); two-phase fixture needs harness surgery (restartable backend);
dropping deps breaks `gcs-query` → reroute through Query RPC first; adoption runs before bind (bump
readinessProbe for first rollout; idempotent on probe-kill); flat-layout dirs → hard error;
cross-writer byte_size differs → assert presence not equality.

---

## 4. Shared types & cross-phase decisions

**Shared Rust types — defined ONCE (Phase 1 unless noted), reused everywhere:**

- `store/schema.rs`: `Column`, `Schema`, `ColumnType↔Arrow DataType` map, `arrow_type_for`,
  `schema_to_arrow`, proto/json conversions, `with_implicit_seq`, `resolve_key_column`,
  `merge_schemas`, `MAX_WRITE_ROWS_BYTES/ROWS`, `IMPLICIT_KEY_COLUMN/SEQ_COLUMN`. (Phase 2 *adds*
  `validate_and_align_batch`/`AlignedBatch`/`arrow_to_column_type` to the SAME module — `arrow`
  must already be a dep in Phase 1 for the type map, so add `arrow=58` in Phase 1.)
- `store/types.rs`: `SegmentLocation{Local,Remote,Both}` (`LOCAL/REMOTE/BOTH` strings),
  `SegmentRow`, `LocalSegment`, `NamespaceStats` (+ `::empty()`). Declared in Phase 1 (segments
  empty), populated in Phase 2, extended in Phase 4 (`created_at_ms`, key bounds, level).
- `store/policy.rs`: `StoragePolicy` — defined in Phase 1, reused unchanged by Phase 4.
- `store/catalog.rs`: `Catalog` + `RegisteredNamespace` — Phase 1 owns it; Phases 2/4/6 add
  segment ops (`replace_segments`, `set_location`, eviction selectors, segment inserts/aggregates)
  to the SAME struct. **One** catalog type for the whole project.
- `store/segment.rs` / `store/segment_name.rs`: `seg_filename`/`parse_seg_filename` — defined
  ONCE in Phase 2; Phase 4 and Phase 6 reuse (Phase 6 must `grep store/` before re-adding).
- `arrow_ipc.rs` (or `store/ipc.rs`): `encode_ipc`/`decode_ipc` — the decode side is Phase 2, the
  encode side is Phase 3, both in ONE module.

**Cross-phase decisions (resolved):**

1. **Durability-await primitive = `tokio::sync::watch<i64> persisted_seq`** (init `-1`), one per
   namespace. The flush task `send`s the new high-water seq **only after** the parquet file is
   renamed into place AND the catalog row is committed. The write/push handlers
   `await_persisted(target)` = subscribe + `while *rx.borrow() < target { notify_flush;
   rx.changed().await }` bounded by `tokio::time::timeout(ctx.time_remaining().unwrap_or(30s))`.
   This is the single primitive for the whole project — no poll-with-backoff anywhere.
2. **Catalog = Rust-owned `rusqlite` sidecar** at `{data_dir}/_finelog_catalog.sqlite` (bundled
   sqlite, hermetic). **Distinct filename** from Python's `_finelog_registry.duckdb` so the two
   backends never alias a file during the dual-backend window, and rebuild-from-disk (Phase 6) is
   independent of any DuckDB sidecar. `data_dir=None` → in-memory rusqlite. **Never read the DuckDB
   sidecar.**
3. **Async/threading model.** One multi-threaded tokio runtime. DataFusion `sql()/collect()` are
   awaited directly (DF schedules its own CPU tasks). **Sync parquet I/O and the k-way merge run
   under `spawn_blocking`** so they don't stall the reactor. The Python flush/compact/maintenance
   loops become **per-namespace tokio tasks**; `Store::shutdown()` cooperatively cancels+joins them
   on SIGTERM.
4. **Versions (corrected from the proposal).** `datafusion=53.1`, `arrow=58`, `parquet=58`,
   `object_store=0.13`. Inside the finelog binary there is exactly ONE arrow (58). dupekit's 57.1
   is a separate cdylib and irrelevant. Add `arrow=58` in **Phase 1** (for the type map) to avoid a
   mid-stream re-pin.
5. **`max_segments_per_level=32`** (the proposal §2 "4" is stale — confirmed in `compactor.py`).
   `level_targets=(64,256,256)MiB`, terminal level = `len(level_targets)`.
6. **Query visibility = sealed segments ONLY** (no RAM MemTable union). The durability contract
   (write acks after L0 persist) makes this complete for RPC clients. (Reconciles the cookbook's
   "union sealed+RAM" framing against `query_snapshot`'s sealed-only contract — sealed-only wins.)
7. **L0 is written UNSORTED.** Sorting happens only at L0→L1 compaction (Phase 4). Phase 2 flush
   must NOT apply `compaction_sort_keys`.
8. **Errors:** `SchemaConflict→failed_precondition` (NOT already_exists); DataFusion plan/parse/
   resolve → `invalid_argument` (the DuckDB error slot); oversize-IPC → `resource_exhausted`
   (no server-side row cap).
9. **Test-forcing seam = flag-gated non-proto `--debug-admin`** (`/debug/maintain`,
   `/debug/segments`, `/debug/backdate`) on BOTH backends, driving the same code path. Off the
   frozen contract, disabled in production.
10. **Always return OWNED response messages** (JSON-codec safety) and **always build
    `ConnectRpcService` explicitly with 64MB Limits** via `register()` (never `into_axum_service`,
    so `ctx.spec()` is populated for interceptors).

---

## 5. Build / test quickref

```bash
# --- Build the Rust server (prereq for the rust parity leg) ---
cd /home/power/code/marin/.worktrees/finelog-rust/rust
cargo build -p finelog --bin finelog-server          # debug
cargo build --release -p finelog --bin finelog-server # release (deploy / faster parity)

# --- Rust unit tests (per module; the cargo-side gates) ---
cd /home/power/code/marin/.worktrees/finelog-rust/rust
cargo test -p finelog                  # all
cargo test -p finelog schema::         # Phase 1/2
cargo test -p finelog catalog::        # Phase 1
cargo test -p finelog ram_buffer:: segment:: namespace::   # Phase 2
cargo test -p finelog query:: arrow_ipc:: store::log_read::# Phase 3
cargo test -p finelog planner:: merge:: executor::         # Phase 4
cargo test -p finelog server::interceptors server::spa server::diagnostics  # Phase 5
cargo test -p finelog footer:: adopt::  # Phase 6
# Single-arrow check (must print exactly one line):
cargo tree -p finelog | grep 'arrow v'

# --- Parity suite (dual-backend over HTTP); rust leg auto-skips if binary absent ---
cd /home/power/code/marin/.worktrees/finelog-rust/lib/finelog
uv run --group dev --with httpx pytest tests/parity/ -p no:xdist -o addopts="" -v
# Per-phase narrowing:
#   Phase 1: tests/parity/test_metadata.py tests/parity/test_smoke.py
#   Phase 2: tests/parity/test_write_path.py
#   Phase 3: tests/parity/test_query.py tests/parity/test_golden_corpus.py tests/parity/test_fetch_logs.py
#   Phase 4: tests/parity/test_compaction.py tests/parity/test_catalog_stats.py \
#            tests/parity/test_offload.py tests/parity/test_eviction.py tests/parity/test_storage_policy.py
#   Phase 5: tests/parity/test_server.py tests/parity/test_spa.py
#   Phase 6: tests/parity/test_cutover.py
# (--with httpx because finelog doesn't declare httpx; -p no:xdist and -o addopts="" override the repo's -n auto)

# --- Lint / types (before any PR) ---
cd /home/power/code/marin/.worktrees/finelog-rust
./infra/pre-commit.py --all-files --fix
uv run pyrefly

# --- Deploy image (Phase 6g) ---
docker build -f lib/finelog/deploy/Dockerfile -t finelog:rust .
docker run --rm finelog:rust finelog-server --help
```

---

## 6. Residual risk register (operational — track through cutover)

These are the cross-cutting risks the synthesis flagged beyond the per-phase
risk lists above. They are mostly operational and must be tracked into Phase 6.

1. SQL dialect divergence DuckDB->DataFusion is the TOP residual risk. The golden-query corpus parity test (Phase 3.4) is the executable gate, but it can only catch query shapes that actually appear in the extracted corpus; a production query shape not represented in the corpus could diverge silently after cutover. Mitigation: extract the corpus broadly (iris/zephyr callers + deploy/cli + tests) and keep duckdb-rs as the documented fallback if a costly gap surfaces.

2. Schema recovery at cutover (Phase 6c) is lossy relative to the DuckDB sidecar: additive-merge history and any non-default key_column that the parquet footer + resolve_key_column do not imply are unrecoverable from parquet alone. Mitigated by deploy's startup RegisterTable re-establishing the registered schema/policy, but a namespace that was NOT re-registered at startup would adopt the footer-inferred key_column. Operational check required: confirm the cutover deployment re-registers all its known tables.

3. Concurrency-cap parity over RPC is inherently timing-sensitive and AGENTS.md forbids time.sleep in tests. The exact 'peak in-flight == cap' assertion is moved to a cargo unit test (run_chain + parked terminal), but the RPC-level parity test still needs an env-gated server-side delay knob to hold permits deterministically; if that knob is mis-scoped the test could flake or become a slop test.

4. Boot adoption (Phase 6) runs synchronously before the listener binds; a very large data dir (thousands of segments x footer reads) could push readiness past the k8s readinessProbe window and get the pod killed mid-adoption on the first Rust rollout. Footer-only reads under bounded-parallelism spawn_blocking + the idempotent sentinel mitigate this, but the first-rollout readinessProbe initialDelaySeconds likely needs a manual bump.

5. The two-phase (Python-writes->Rust-reads) cutover fixture and the restartable-backend fixture require non-trivial surgery to the existing single-backend, fresh-tmp-dir parity harness. If that harness work slips, the Phase-4 boot-reconcile and Phase-6 cutover gates (the most important data-safety gates) cannot run.

6. Legacy /iris.logging.LogService/* path rewrite: the CRON removal marker (2026-05-12) has passed but the rewrite is still live (iris dashboard alias + worker push path). Porting it is correct, but it is dead-code risk: it must be re-verified against current worker images before it can ever be deleted, and that verification is an operational step outside this roadmap.

7. axum 0.7 (finelog direct dep) vs axum 0.8 (connectrpc internal) duplication is pre-existing and benign for compilation, but tower-http 0.6 'fs' (added in Phase 5d) must remain compatible with whichever axum the app Router uses; a future connectrpc bump that forces axum 0.8 on the finelog crate could require revalidating ServeDir/middleware::from_fn.

### Harness notes
- `httpx` is now declared in `lib/finelog`'s `dev` group, so the `--with httpx`
  in the commands above is **optional** (kept for copy-paste robustness on a
  bare venv).
- The dual-backend parity harness currently spawns each backend fresh on a
  tmp dir. Phases 4 and 6 require a **restartable-backend fixture** (respawn on
  the same `log_dir`/`remote_log_dir`) and a **two-phase cross-backend fixture**
  (Python writes → Rust reads). Building those fixtures is itself gated work
  inside Phases 4c/4e and 6f.
