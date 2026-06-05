# Finelog dashboard query benchmark — FetchLogs PREFIX slowness

Date: 2026-06-05. Branch `weaver/finelog-rust`.

## What this is

Replays every distinct query the iris dashboard sends to the finelog server,
against a **real** prod dataset on a **local** server, to find which shapes are
slow and why. Driven by `lib/finelog/scripts/bench_dashboard_queries.py` (extended
here with FAMILY D — the PREFIX FetchLogs cases). Artifacts:
`/home/power/finelog-bench/results/dashboard-bench.json` + `server-{rust,duckdb}-*.log`.

## Setup

- **Data:** real segments from `gs://marin-us-central2/finelog/marin`, bounded to
  recent per-namespace slices (the full `log` archive is 27 GB/163 segs; we use the
  most-recent 5.5 GB/54 segs = 201M rows). iris.* 2.4–3.9 GB each.
- **Servers (one at a time, same data):** Rust `finelog-server` (fast build) and
  Python/DuckDB, 21 query shapes × 3 reps + warmup, query pool capped 8 GB.

## Headline: FetchLogs PREFIX is the slow shape

EXACT scope prunes; PREFIX scope does not. Rust p50, recent 5.5 GB slice:

| FetchLogs shape | scope | rust p50 | duck p50 |
|---|---|---:|---:|
| `/system/worker/<id>` | EXACT | **29 ms** | 204 ms |
| `<task>:<attempt>` | EXACT | **49 ms** | 411 ms |
| `/system/controller` | EXACT | **61 ms** | 718 ms |
| `<task>:` all-attempts | PREFIX | **1763 ms** | 643 ms |
| `<run>/` whole run | PREFIX | **1522 ms** | 634 ms |
| `<run>/` + `minLevel=INFO` | PREFIX | **1531 ms** | 668 ms |

This reproduces the prod report exactly: `/system/controller` (EXACT) was fine;
`{source:"/bizon/iris-run-cli-.../", matchScope:PREFIX, minLevel:INFO}` was slow
(prod logged 12.7 s over the full 27 GB; ~1.5 s here over 5.5 GB — scales with data).

## Root cause (verified in code + by the captured SQL)

`build_log_predicates` (`store/log_read.rs`) emits, for `MATCH_SCOPE_PREFIX`:

```sql
SELECT seq,key,source,data,epoch_ms,level FROM "log"
WHERE seq > 0 AND prefix(key, '/benjaminfeuer/.../140858/')
      [AND (level = 0 OR level >= 2)]      -- when minLevel=INFO
ORDER BY seq DESC LIMIT 500
```

- `prefix(key, P)` is an **opaque scalar UDF**. DataFusion's `ParquetExec` can't
  derive key bounds from it, so it prunes **no** row groups and decodes
  key+source+data+epoch_ms+level for **every** row of every segment, then filters.
  Full scan of the namespace.
- EXACT emits `key = P` — a stats-comparable predicate. Compaction sorts the log by
  `[key, seq]` (`compaction/planner.rs`: *"key_column first so range scans prune row
  groups"*), so on the L1+ segments only the matching row groups are read. Fast.
- `minLevel` adds `(level = 0 OR level >= 2)` on a non-cluster column — doesn't
  prune, and slightly compounds cost (more rows scanned to gather 500 INFO+ lines).
- `tail` → `ORDER BY seq DESC LIMIT 500` (TopK) can only push `seq > threshold`
  *after* 500 matches are found; a sparse prefix forces scanning far back, so TopK
  doesn't rescue it.

## The fix (quantified directly on the log)

Rewrite PREFIX (and the REGEX literal-prefix) from the `prefix()` UDF to the
**half-open key range** `key >= P AND key < succ(P)` — semantically identical to a
prefix match, but stats-prunable on the `[key,seq]`-sorted segments. Keep the
`prefix()` UDF as a cheap residual on the already-pruned rows for exactness on the
byte-wrap edge (log keys are ASCII paths, so the simple last-byte+1 successor is
safe; the residual is belt-and-suspenders).

Measured on the real 201M-row log slice, same tail-500 query:

| variant | p50 | rows |
|---|---:|---:|
| `prefix(key,P)` (current) | 1554 ms | 500 |
| `key >= P AND key < succ(P)` (**fix**) | **79 ms** | 500 |
| `key = E` (EXACT control) | 50 ms | 500 |

**~20× faster, identical result, down to EXACT speed.** FAMILY B corroborates on
iris.task: a prunable key predicate (`LIKE 'Y%'` / range) is 185–199 ms vs the
namespace full scan.

Fix site: `build_log_predicates` `MATCH_SCOPE_PREFIX` and `MATCH_SCOPE_REGEX`
arms in `rust/finelog/src/store/log_read.rs`.

### Implemented + validated end-to-end (through the FetchLogs RPC)

`build_log_predicates` now emits, for PREFIX (and the REGEX literal prefix):
`key >= P` + (when a finite successor exists) `key < succ(P)` + the `prefix()`
residual. `succ(P)` = increment the last byte below `0xFF`, drop the rest;
`None` (residual-only) on the empty / all-`0xFF` / non-UTF-8 edge. Re-running the
*actual* FetchLogs RPC against the same slice with the fixed server:

| FetchLogs PREFIX shape | before | after |
|---|---:|---:|
| `<task>:` all-attempts | 1763 ms | **97 ms** |
| `<run>/` whole run | 1522 ms | **100 ms** |
| `<run>/` + `minLevel=INFO` | 1531 ms | **107 ms** |

EXACT controls unchanged (27–59 ms). Tests: `prefix_scope_emits_prunable_key_range_plus_residual`,
`key_prefix_upper_bound_cases`, and an end-to-end `prefix_fetch_returns_exactly_the_prefix_rows`
(keys straddling the range boundary — guards against a successor that drops/leaks rows).

## Does rolling back to Python/DuckDB help? (it's a trap)

DuckDB **is** ~2.4× faster on the PREFIX FetchLogs (634–668 ms vs Rust ~1500 ms).
But it is dramatically **slower on nearly everything else** the dashboard runs:

| query | rust p50 | duck p50 |
|---|---:|---:|
| worker_detail_history (keyed) | 108 ms | 441 ms |
| profile_download_one | 21 ms | 249 ms |
| canary_probes_recent | 8 ms | 252 ms |
| worker_recent_window | 108 ms | **7102 ms** |
| log_tail_controller (EXACT) | 61 ms | 718 ms |

Net: a Python rollback trades one slow shape for regressions across the whole
dashboard. The key-range fix makes Rust fast on PREFIX too (~79 ms) — strictly
better than rolling back. (Rollback to Python `05aad929` remains available as an
operational safety valve; this is the argument for *not* needing it for perf.)

## Secondary findings (separate from the FetchLogs fix)

1. **`profile_history_real` is slow on Rust (1665 ms) but fast on DuckDB (359 ms).**
   The `length(profile_data)` blob query — Rust's parquet pushdown isn't pruning the
   blob reads as well as DuckDB's late materialization here. Worth a separate look;
   the dashboard-side query is being changed independently.
2. **`worker_recent_window` correctness discrepancy:** Rust returns 500 rows,
   DuckDB returns 0, for the *same* `ts > now() - INTERVAL '15 minutes'` query.
   Likely legacy µs-vs-ms timestamp handling (see the microsecond-timestamp note).
   Potential real bug — needs separate investigation; not part of the FetchLogs fix.
3. **`fleet_status_rollup` hit ResourcesExhausted** under the 8 GB test pool (full
   iris.worker GROUP BY + external sort). Prod's larger pool may absorb it, but it's
   memory-heavy; the bounded pool did its job (failed cleanly, server survived).
4. **`iris.task_status` has 1522 tiny segments** (compaction lag); keyed queries
   still prune and stay fast (67–84 ms), but per-segment open cost is a latent risk.

## Tooling added this session

- `query/mod.rs`: slow-query WARN logging that prints the **executed SQL** for any
  `Query`/`FetchLogs` over `FINELOG_SLOW_QUERY_LOG_MS` (default = the 7 s RPC bar) —
  the interceptor's "Slow RPC" line can't show SQL. This is what captured the exact
  predicate above.
- `bench_dashboard_queries.py`: FAMILY D (PREFIX FetchLogs: task-prefix, run-prefix,
  run-prefix+INFO), discovering a real run prefix from the data; slow-line capture
  broadened to the new SQL-bearing logs.
