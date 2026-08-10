# Finelog Agent Notes

Standalone log store + log service. Originally lifted out of `lib/iris`
(`iris/cluster/log_store/` and `iris/log_server/`); see the
[worked Finelog proposal](../../.agents/projects/design-template.md#worked-example)
or the original extraction PR for context.

Start with the shared instructions in `/AGENTS.md`. Finelog-specific notes:

## Source Layout

- `src/finelog/proto/logging.proto` — log-service RPC definitions (package `finelog.logging`)
- `src/finelog/proto/finelog_stats.proto` — stats-service RPC definitions (package `finelog.stats`)
- `src/finelog/rpc/` — generated `_pb2`/`_connect` modules
- `src/finelog/types.py` — shared types: `LogReadResult`, `LogWriterProtocol`, key-related constants
- `src/finelog/store/` — `MemStore` (in-memory) and `DuckDBLogStore` (Parquet + DuckDB)
- `src/finelog/server/` — `LogServiceImpl`, `StatsServiceImpl`, ASGI builder, CLI launcher
- `src/finelog/client/` — `LogClient` (single user-facing entry; covers logs and stats),
  `RemoteLogHandler`, error types in `errors.py`.
- `tests/` — store + server tests
- `deploy/` — Dockerfile, k8s manifests, GCP snippets

## Boundaries

- Finelog has no `iris.*` imports. Iris-specific helpers (`worker_log_key`,
  `task_log_key`, `build_log_source`, anything that takes `JobName`/`TaskAttempt`)
  live under `iris/cluster/log_store_helpers.py` and call into finelog with opaque
  string keys.
- Finelog's server gates every RPC with an authenticated-ingress stack
  (`rust/src/server/auth.rs`): an ordered, **default-deny** list of `cidr` and
  `jwt` layers (`FINELOG_AUTH_POLICY`), defaulting to loopback-only. The `jwt`
  layer verifies **EdDSA (Ed25519)** tokens with `aud="finelog"` against each
  cluster's inline **public** key(s) — a sending finelog signs with its private
  key, the receiving one holds only the public half. Deployments still secure the
  network layer (k8s NetworkPolicy, GCP firewall, VPC) as defense in depth. The
  policy schema lives in `deploy/config.py`.
- **The admitting layer names the caller.** A `jwt` layer's matched key binds the
  request to that key's `cluster`; a `cidr` match binds to nothing (see
  `AuthIdentity` in `rust/src/server/auth.rs`). `PushLogs` stamps each row with the
  authenticated cluster. Cross-cluster forwarding instead writes through the generic
  `WriteRows` path and stamps the origin into the row itself, so the `cluster` column
  is a label, not a trust boundary — any admitted sender can write any cluster's rows.
- Keys are opaque strings. Any structure (`/system/...`, `/user/<job>/<task>:<attempt>`)
  is iris-side convention; finelog does not parse keys.

## Cross-cluster forwarding

A per-cluster finelog ships its rows to a hub finelog itself; no other process
relays them. `forwarding:` in its deploy config names the hub, this cluster's
name, and a `rigging.secrets` reference to its Ed25519 private key; the hub adds
one `jwt` key entry per sender. Each server therefore owns a keypair, distinct
from the iris controller's signing key.

The forwarder (`rust/src/server/forwarding.rs`) forwards **every table**, not just
logs. Each round it lists the live namespaces and gives each one a batch-sized turn,
then immediately starts another round while any namespace remains backlogged. Per
namespace, it reads the rows past a durable per-`(target, namespace)` cursor
(`forward_state` in the catalog) and ships them through the generic `RegisterTable` +
`WriteRows` (Arrow IPC) path — a namespace the hub lacks is created there first. Rows
of a table with a `cluster` column are stamped with the origin and skipped if they
already carry a foreign one, so a hub's own relayed rows never loop. The cursor is
durable, so a restart resumes rather than replays.

Forwarding is **best-effort by construction**: the sending store holds the record,
the hub a convenience copy. A backlog is a durable cursor into the sender's bounded
local retention rather than a separate queue, so the forwarder drains it without an
age or row-count cap. Non-log chunks from one read turn may wait for hub durability
concurrently; log chunks stay serial to preserve line order. Rows are skipped only
after local eviction makes them unreadable or the hub permanently rejects a malformed
batch. A hub outage therefore cannot consume extra sender memory, but a long enough
outage can still outlive local retention.

Only the k8s backend can forward — it projects the key through a Secret. The gcp
backend refuses, because its only channel to the server is world-readable
startup-script metadata.

## Packaging

Finelog ships as two PyPI dists, released in lockstep by
`marin-release-libs-wheels.yaml`:

- `marin-finelog` — pure Python (this directory; hatchling).
- `marin-finelog-server` — the native in-process server ext, importable as
  top-level `finelog_server` (maturin project at `rust/`; the cdylib crate is
  `rust/pyext`). Only `src/finelog/embedded.py` imports it.

`marin-finelog` does **not** depend on `marin-finelog-server` at runtime — the
pure client never needs the in-process server. Consumers that do (the iris
controller) depend on `marin-finelog-server` explicitly. Here it is only a
`dev` dependency, pulled in for the embedded-server smoke test and the
dashboard demo.

By default the extension comes from the pre-built PyPI wheel, so in-dir
`uv run` never compiles Rust. To build it from source (live Rust dev), run
`python scripts/rust_mode.py dev` at the repo root — it points
`marin-finelog-server` at the local `rust/` tree in both the root and
`lib/finelog` pyprojects. Run `python scripts/rust_mode.py user` before
committing.

## Development

```bash
cd lib/finelog
uv run --group dev pytest --tb=short tests/
```

Regenerate protos after editing `proto/logging.proto`:

```bash
cd lib/finelog && buf generate
```

### Benchmarks

Three harnesses under `src/finelog/benchmarks/`, each driving a real
`finelog-server` and writing a JSON result with the server build, storage
layout, and per-query `EXPLAIN ANALYZE` metrics.

- `log_query_bench` — the operator query corpus for `log`: job substring
  scoping, task tails, first-error lookups, body search. `generate` builds a
  corpus; `measure` runs it over a directory that already holds segments.
- `grafana_dashboard_bench` — every query in a checked-in Grafana dashboard.
- `telemetry_layout_bench` — the storage-layout candidates for `telemetry_v1`.

Point `--log-dir` at a **disposable copy**: starting Finelog activates
compaction, layout rewrites, and index backfill.

`log_query_bench` writes to the `log` namespace the server auto-registers rather
than registering its own, so the same corpus under two binaries measures a
schema change. Backfill must be finished before measuring; maintenance running
alongside the queries moves every number by 2-4x.

`EXPLAIN ANALYZE` counters are decimal, `bytes_scanned` included: `1.16 B` is
1.16 billion.

### Dashboard

`npm run dev` serves the SPA with HMR and proxies RPC to a finelog on port
10001 (`FINELOG_DEV_SERVER` to point elsewhere), so frontend work does not need
a `npm run build` round trip into the `dist/` the Rust server reads from disk.

Two surfaces are plain axum JSON rather than proto, because they describe this
process and its files rather than the wire contract: `GET /api/server` (build
revision, uptime, store paths, cache diagnostics, the writer's format policy)
and `GET /api/segments?namespace=NS` (catalog rows, plus footer and index-bundle
detail under `physical=true`). Both sit behind the same default-deny
auth gate as the RPCs. `build.rs` stamps the git commit, its tree hash, and a
dirty flag into the binary; all three are empty when the build had no checkout
to read, as in a wheel built from an sdist.

The SQL editor completes identifiers from `ListNamespaces`, so the vocabulary is
this store's namespaces and columns rather than a general SQL dictionary.
`utils/sqlComplete.ts` holds the ranking and `utils/chart.ts` the axis, EMA, and
decimation maths; both are pure and tested under `npm test` without mounting a
component.

`npm run test:e2e` drives the **built** dashboard with Playwright against an
already-running server; it does not start one. `scripts/demo.py --keep` serves a
seeded store on the default port. Point it at a store with real segments via
`FINELOG_BASE_URL` and `FINELOG_TEST_NAMESPACE`.

## Ingest health

`/health` returns 200 whenever the server is listening. It is the Kubernetes
liveness, readiness, and startup probe, so it cannot fail on a condition that
survives a restart; the verdict is in the body: `ok`, or `degraded:` followed by
each namespace this process registers for itself that is not registered.
`server/ingest_health.rs` holds that state. `/api/server`'s `ingest` block
carries the per-namespace error, first-failure time, and attempt count, and the
dashboard's System page renders it.

The deploy paths gate on the body: the VM bootstrap loop, `_wait_health_via_ssh`
(which is what makes `safe_deploy` auto-rollback fire), and `k8s_up` /
`k8s_restart` via a post-rollout `kubectl exec`. A binary that cannot register
`telemetry_v1` fails its own deploy.

## Changing a server-owned schema

`log` and `telemetry_v1` are registered by the server itself, and every boot
re-merges this binary's definition against the schema that deployment's catalog
persisted. A merge that fails wedges the namespace for as long as the image is
deployed, so `/health` reports it (`server/ingest_health.rs`) and `safe_deploy
rollout` rolls back on it. To decide a schema change ahead of a deploy, boot
the candidate over a copy of that deployment's catalog with `--mode shadow` and
read `/health`. Register through `schema::stored_form` so what you check is the
schema `register_table` merges.

`--mode shadow` is for booting a server over a copy of a real store: it serves
reads from `--log-dir` and refuses a `gs://`/`s3://` remote or a forwarding
target at startup. It resolves once into the store's `ServeMode`, so no
namespace starts a maintenance task — including one registered at runtime,
which otherwise starts its own. Use it for any local benchmark over a copied
store, and pass the mode down rather than re-deriving it at a call site.

## Secondary indexes

A segment with any configured method gets one `.fidx` bundle. The bundle is
bound to the immutable segment identity and contains a checksummed internal directory
of typed sections. Trigram Blooms, exact postings, and value counts stay inside
the bundle. Named covering projections remain narrow Parquet files referenced
by it. Missing, stale, or corrupt derived data always falls back to source
Parquet.

The runtime family is closed: `TrigramBloom`, `ExactPostings`, `ValueCounts`,
and `CoveringProjection`. User-facing `ColumnIndex` flags and
`Schema.projections` compile into those specs. Adding a method means adding a
typed enum variant, format version, validation, planner rule, and copied-shard
benchmark; there is no free-form plugin registry.

A column declared with `ColumnIndex.trigram` gets a span-granular substring
section. That index makes `contains(col, …)` and `col LIKE '%…%'` prune instead
of full-scan. Today it is on `log.key`, `log.data`, and `telemetry_v1.name`.

Sorting by a column does not cover substring search of it. A log key is
`/user/<job>-coord/<job>/<task>:<attempt>`, so the job an operator searches for
is not a prefix and min/max statistics cannot bound it. `log.key` needs its own
trigram section for that.

A `LIKE` pattern contributes every literal run between its wildcards, all
required: `%CUDA_ERROR%` prunes on `CUDA` and `ERROR` separately, while the
escaped `%CUDA\_ERROR%` prunes on the single run `CUDA_ERROR`. Runs under three
bytes carry no trigram and drop out. `NOT LIKE`, `ILIKE`, and an explicit
`ESCAPE` never prune.

Enabling an index is additive and can be done on a live namespace. Maintenance
backfills L≥1 segments a few at a time, reading all required index columns once
and publishing projections before the bundle. Query methods apply per segment,
so partial backfill gives partial benefit except for exact aggregates, which
require complete snapshot coverage.

A string column can also declare `exact_values` and/or `value_counts`.
`exact_values` writes source-row postings. Equality and same-column `IN` or `OR`
predicates use them only while the selected fraction is at most 25%; denser
matches retain the contiguous source scan. The residual predicate always
remains.

`Schema.projections` declares independent named covering projections with one
predicate and an explicit included-column list. The planner substitutes one
only when both the predicate values and every referenced query column are
covered. Covered segments use the projection while uncovered segments retain
source Parquet. The initial `training-status` projection covers three metric
names and seven columns.

Redefining a projection is not a conflict. `merge_schemas` supersedes the
registered definition with the requested one, unless the registered one already
covers it: a superset keeps its place, so an older binary re-registering a
narrower definition against a newer catalog does not churn a namespace's derived
state mid-rollout. Segments written under the superseded definition stay
queryable untouched, because each `.fidx` `CoveringProjection` section describes
its own coverage; the backfill rebuilds them at its usual few per tick and
unlinks the superseded Parquet files. Index hints follow the same rule.

A column *type* mismatch is still a hard `SchemaConflict`: the registered layout
cannot hold the requested rows, so it fails at registration. A new column
declared non-nullable is adopted as nullable, since every already-stored row is
missing it.

`value_counts` records a complete low-cardinality histogram. A DataFusion
optimizer rule replaces a qualifying unfiltered one-column `GROUP BY` with
`COUNT(*)` or `COUNT(column)` when every visible segment is covered. The rewrite
appears as `FinelogIndexAggregate` in `EXPLAIN` and composes with outer
projections, ordering, and limits. Columns over 4,096 distinct values omit the
summary; a combined result above 16,384 values falls back to DataFusion.

A trigram Bloom covers a fixed 16,384-row span, deliberately independent of the
Parquet row-group size. The prune maps its source-row mask onto row groups and
can emit a partial-row selection. Every `.fidx` section has its own format
version and checksum, so changing one method invalidates only that method.

Every segment's parquet footer carries the physical layout revision it was
written with, and maintenance re-encodes stale ones in place a couple per
namespace per tick. This exists because the terminal level never re-compacts, so
without it a writer-policy change would only reach a namespace's bulk as eviction
aged it out. The rewrite preserves the rows, their order, and the filename, which
is what makes it free: the archive keys objects by basename and the sync step
only uploads segments the catalog still marks `Local`, so nothing is re-uploaded.
Staging and committing are separate, with the rename taken under the insertion
lock, because eviction may drop a segment during the seconds a rewrite runs and
renaming over a dropped path would resurrect an untracked file. UUID-stamped
bundles remain valid because the rewrite preserves the segment ID, rows, and row
order. Older unstamped segments use a local file-generation identity until
rewritten; their old bundle falls back and is rebuilt after the rewrite assigns
a UUID.

No segment carries a parquet bloom filter. Writing them for every column cost 15%
of each segment and pruned nothing measurable; the key-column bloom that outlived
that only served exact-key lookups against unsorted L0, which is a few hundred
KiB that compaction consumes within a tick or two, against a write cost on every
flush. L1+ is sorted by `(key, seq)` and prunes the key band from min/max
statistics; substring queries prune from the trigram bundle section.

A starts-with predicate — `prefix(col, P)`, `col LIKE 'P%'`, or
`regexp_matches(col, '^P…')` — prunes only because `PrefixRangeRewrite` ANDs the
implied `[P, succ(P))` range onto it, since min/max statistics key on whole
values. That rule is an `AnalyzerRule`, so it runs *before* the optimizer folds
constants: a column and literal of different string types leave the literal
wrapped in a coercion `Cast`, which the rule has to see through. It does not
share this hazard with the trigram needles, which are extracted from the
optimized plan where such casts are already folded away.
