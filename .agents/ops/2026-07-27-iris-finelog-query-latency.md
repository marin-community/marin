---
date: 2026-07-27
system: finelog
severity: degraded
resolution: investigating
pr: none
issue: none
---

# TL;DR

- The Iris controller RPC panel issued an unbounded latest-per-series query that took 16.954 seconds in DataFusion and 21.469 seconds at the RPC boundary.
- The live `telltale` namespace held 988,423,984 rows and 11.85 GB across 111 durable segments. Its key column was correctly set to `name`.
- A five-minute timestamp predicate pruned 60,390 row groups to 7 and read 254 KiB, but Parquet metadata loading still took 2.29 seconds.
- The one-hour bound merged in #7645 used `epoch_ms(ts)`, which the live and current DataFusion engines did not implement. `ts >= now() - INTERVAL '5 minutes'` planned successfully and pushed into Parquet pruning.
- No namespace was reset. The proposed sequence was valid server-clock SQL, metadata-cache measurement and a Marin-only override, then cancellation coverage before a 9-second hub query cap.

# Original problem report

The Iris dashboard emitted:

```text
WARN finelog::query: slow Query: 16954ms rows=545
WARN finelog::server::interceptors: Slow RPC Query: 21469ms (threshold: 7000ms)
```

The SQL read six `iris_rpc_*` metric names from `telltale` and applied:

```sql
QUALIFY row_number() OVER (
  PARTITION BY name, json_get(labels, 'service'), json_get(labels, 'method'),
               json_get(labels, 'upstream'), json_get(labels, 'status'), json_get(labels, 'le')
  ORDER BY ts DESC
) = 1
```

It had no timestamp predicate.

# Investigation path

1. The dashboard query was traced to `lib/iris/dashboard/src/components/controller/RpcStatsPanel.vue:58`. Commit #7645 had already added a one-hour cutoff and a local-cluster filter, but the deployed warning still showed the older unbounded form.

2. `TelltaleMetric` declared `key_column = "name"` at `lib/rigging/src/rigging/telltale.py:217`. `ListNamespaces` confirmed that the live table retained that key. The suspected wrong-key namespace configuration was ruled out.

3. The live namespace reported 988,423,984 rows, 11,849,890,300 bytes, and 111 durable segments with no per-namespace policy. The table also held training and inference metrics, so resetting it would have deleted data outside the controller panel.

4. The #7645 query failed during live planning because `epoch_ms(ts)` was unknown. The current DataFusion UDF registry under `lib/finelog/rust/src/query/` also contained no `epoch_ms` function.

5. A direct timestamp literal was accepted and pushed into `TableScan.partial_filters`. One-hour, ten-minute, five-minute, and approximately two-minute windows returned the same 506 rows in 2.001-2.253 seconds.

6. `EXPLAIN ANALYZE` on the five-minute query reported 60,390 row groups reduced to 7 and 254 KiB scanned. It spent 2.29 seconds loading file metadata and 2.02 seconds opening files. Data decoding was no longer the limiting cost.

7. A scalar subquery that found `max(ts)` and fetched that scrape returned the same rows in 2.866 seconds. The second scan cost more than the removed window work.

8. `ts >= now() - INTERVAL '5 minutes'` was tested on the live server. DataFusion folded `now()` to a timestamp literal and emitted a Parquet `ts_max >= cutoff` pruning predicate, avoiding browser-clock skew.

9. Finelog's existing timeout was found at `lib/finelog/rust/src/server/stats_service.rs:188`. It wrapped the DataFusion future in `tokio::time::timeout`; `FINELOG_QUERY_TIMEOUT_MS` defaulted to 60 seconds. The Python client defaulted to 10 seconds, so a 10-second server limit would race the client.

10. Codex reviewed the Weaver design against the repository. The review found that namespace policies were eventual local-cache eviction targets, not hard row-age or archive-retention guarantees; a new dotted namespace required registration and migration handling; and cache capacity needed measurement before choosing 512 MiB.

# User course corrections

- The user authorized bounded live queries through the Marin dashboard endpoint. This allowed the investigation to replace assumptions about pruning with `EXPLAIN ANALYZE` evidence.
- The user allowed Finelog changes or a stats-namespace reset. The live schema and shared training-data use showed that reset was unnecessary and destructive, so the proposal kept the namespace intact.

# Root cause

The deployed dashboard selected the latest value for every RPC metric-label tuple over the full history of a table approaching one billion rows. The query therefore decoded labels and executed a repartitioned window over historical samples that were irrelevant to a current-status panel.

The merged cutoff was not deployable because it used an unregistered `epoch_ms(timestamp)` function. A valid cutoff exposed a second latency floor: `NamespaceProvider` constructed a `ListingTable` over every segment path (`lib/finelog/rust/src/query/provider.rs:63`), and DataFusion spent about 2.3 seconds loading metadata even though row-group pruning reduced data reads to 254 KiB.

# Fix

No production fix was applied during the investigation. The revised Weaver design proposed:

```sql
ts >= now() - INTERVAL '5 minutes'
```

for the dashboard, metadata-cache hit/byte instrumentation followed by a measured Marin-only cache override, and cancellation/resource-release tests before setting a 9-second Marin server cap. A dedicated `"iris.telltale"` namespace remained conditional on the warm-cache result.

# How OPS.md could have shortened this

- Add a `Query latency` section to `lib/finelog/OPS.md`: run `ListNamespaces` before changing a namespace, and record row count, bytes, segments, key column, and policy. This distinguishes wrong-key scans from large-table metadata costs.
- Add a bounded `EXPLAIN ANALYZE` example that calls out `row_groups_pruned_statistics`, `bytes_scanned`, `metadata_load_time`, and `time_elapsed_opening`. High metadata time with low scanned bytes indicates footer/cache work, not failed row-group pruning.
- State that timestamp columns should use native timestamp comparisons such as `ts >= now() - INTERVAL '5 minutes'`. Integer log columns named `epoch_ms` are different from a nonexistent `epoch_ms(timestamp)` conversion function.
- Document that `StoragePolicy` evicts eligible uploaded local segments. It does not guarantee row-age retention or delete remote archive objects.

# Artifacts

- Weaver proposal and measurements: https://loom.oa.dev/s/n2cerqw4/artifacts/design
- Dashboard query: `lib/iris/dashboard/src/components/controller/RpcStatsPanel.vue`
- Finelog query runtime and timeout: `lib/finelog/rust/src/query/mod.rs`
- DataFusion table provider: `lib/finelog/rust/src/query/provider.rs`
