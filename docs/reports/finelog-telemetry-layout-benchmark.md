# Finelog telemetry layout benchmark

Status: decision record
Date: 2026-07-31
Issue: Weaver #205
Production anchor: [Echo wiki/38](https://echo.oa.dev/wiki/38)

## Decision

The telemetry foundation should implement the smallest change that attacks the
measured failure mode:

1. Bind only SQL-referenced namespaces.
2. Keep the shared DataFusion Parquet metadata cache and expose its hit/size
   metrics.
3. Add a generation-keyed segment manifest with event-time and clustering
   bounds. Apply usable time/name/entity predicates before constructing the
   `ListingTable`; keep DataFusion residual filtering.
4. Split logical namespaces by access pattern for schema, retention, ownership,
   and failure isolation. Do not claim that splitting alone solves metadata
   latency.
5. Keep current name-keyed physical segments for the first foundation change.
   Do not land multi-column clustering, day partitions, day+hash partitions, or
   generic minute/hour rollups from this benchmark.
6. Add the 100,000-ID run catalog. When descriptor-specific rollups become
   necessary, prefer the watermark materializer over compaction coupling and
   benchmark it on production cadence first.

The manifest clears the query gate. At the 1,000,000-row, 111-file local stress
point, the 15-minute run query read one row group and 2.4 KiB but the current
provider still opened all 111 files. Manifest selection reduced its cold p50
from 150.5 ms to 19.9 ms and warm p50 from 58.1 ms to 11.1 ms in the
same-command control. The production incident measured the same shape at much
larger scale: seven of 60,390 row groups and 254 KiB survived a five-minute
predicate, while metadata loading and file opening still cost about two
seconds.

The 64/128 principal-hash candidates are not selected. Across a 14-day,
500,000-row exact projection, all 896/1,792 partitions were nonempty before
segment multiplicity. Day-only already selected one idealized file for a
one-day query. Hashing retained one file for exact-run and exact-worker node
queries but expanded a one-day fleet/alert query to 64/128 files and weekly
accounting to 448/896 files. It therefore shifts the metadata problem to many
small partitions.

## Evidence classes and limits

- **M — StatsService measurement:** SQL ran through the current embedded
  Finelog server and public StatsService. Candidate stores still use the
  unchanged Finelog provider and DataFusion engine.
- **P — local proxy:** local DuckDB aggregation, hardlink-based candidate
  preparation, or in-memory manifest/partition selection. Proxy latency is not
  Finelog query latency.
- **I — production incident:** observations copied from Echo wiki/38; this
  session did not reread production GCS.
- **E — extrapolation:** arithmetic projection, never presented as a
  measurement.

Process-cold means a new Python process, embedded server, and DataFusion
metadata cache. It does not drop the host page cache. “Planning” is StatsService
wall time for `EXPLAIN` without `ANALYZE`. Metadata/file-open/scan values come
from `EXPLAIN ANALYZE`. RSS includes the Python harness and embedded Rust
server. Each cold p95 is the maximum of only three process-cold samples; it is a
tail indicator, not a stable production percentile.

Synthetic rows have the flattened Telltale schema, 34 representative signals,
a hot multi-worker run, and rotating identities configured over a 100,000-ID
range. The numeric scheme caps achieved distinct IDs at 99,993. The 500k corpus
has 230 scrapes about 88 minutes apart and reaches 11,027 distinct run IDs. The
1M corpus has 460 scrapes about 44 minutes apart and reaches 22,053 run IDs.
Both span 14 days. They compress much better than production: the 1M corpus is
5.61 MB of Parquet. Results are useful for provider/file-count components, not
storage-capacity or production-cadence planning.

No Iris job or remote object read was used.

## Query coverage

| Design requirement | Checked query |
|---|---|
| Exact run, ten families | 15 minutes, 24 hours, and seven days over ten Levanter families |
| Run-prefix search | Bounded prefix lookup over a separate 100,000-row run catalog |
| Worker/rank outliers | Exact run, latest CPU/RSS by worker |
| vLLM distribution | One-hour TTFT histogram buckets for an exact run |
| Fleet | Worker-state/failure-counter scan-shape proxy by cluster/region |
| Node/device | 24-hour device history with run attribution |
| Errors/log scan-shape proxy | Six-hour errors plus exact-run `LIKE '%OOM%'` body search |
| Alerts | One-hour latest progress, producer heartbeat, queue, GPU, and forwarding-gap states |
| Accounting | Weekly user/project accelerator accounting |

Every fact query has an upper and lower timestamp bound. The checked SQL is
stored in the raw JSON outputs. The fleet SQL intentionally exercises the
required names and grouping shape, but `max(worker_state)` is not latest-value
semantics and summing counter samples is not a counter delta. Its values are not
used as correctness evidence. The errors SQL likewise sums `*_total` counter
samples instead of taking a delta; it exercises counter/body-filter scan shape,
and its values are not correctness evidence.

## Production anchor (I)

No benchmark command was run. This is the 2026-07-27 incident observation from
[Echo wiki/38](https://echo.oa.dev/wiki/38). The corrected predicate was:

```sql
ts >= now() - INTERVAL '5 minutes'
```

| Rows | Durable segments | Parquet bytes | Row groups matched | Bytes scanned | Metadata load | File open | Unbounded DataFusion / RPC |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 988,423,984 | 111 | 11,849,890,300 | 7 / 60,390 | 254 KiB | ~2.29 s | ~2.02 s | 16.954 s / 21.469 s |

This establishes the production failure mode but does not attribute the entire
unbounded wall time to metadata.

## Current-layout scale sweep (M)

Exact commands/configurations:

```bash
cd lib/finelog
uv run --group dev python -m finelog.benchmarks.telemetry_layout_bench baseline \
  --rows 100000 --duration-days 14 --batch-rows 25000 --segments 8 \
  --decoy-namespaces 0 --distinct-run-ids 100000 \
  --cold-iterations 3 --warm-iterations 7 --warmup-iterations 1 \
  --concurrency 1 4 8 --concurrency-queries-per-worker 2 \
  --work-dir /tmp/finelog-baseline-final-205-100k \
  --output ../../docs/reports/finelog-telemetry-benchmark-results/baseline-100k-8-files.json

uv run --group dev python -m finelog.benchmarks.telemetry_layout_bench baseline \
  --rows 500000 --duration-days 14 --batch-rows 25000 --segments 32 \
  --decoy-namespaces 0 --distinct-run-ids 100000 \
  --cold-iterations 3 --warm-iterations 7 --warmup-iterations 1 \
  --concurrency 1 4 8 --concurrency-queries-per-worker 2 \
  --work-dir /tmp/finelog-baseline-final-205-500k \
  --output ../../docs/reports/finelog-telemetry-benchmark-results/baseline-500k-32-files.json

uv run --group dev python -m finelog.benchmarks.telemetry_layout_bench baseline \
  --rows 1000000 --duration-days 14 --batch-rows 25000 --segments 111 \
  --decoy-namespaces 128 --distinct-run-ids 100000 \
  --cold-iterations 3 --warm-iterations 7 --warmup-iterations 1 \
  --concurrency 1 4 8 --concurrency-queries-per-worker 2 \
  --work-dir /tmp/finelog-baseline-final-205-1m-111 \
  --output ../../docs/reports/finelog-telemetry-benchmark-results/baseline-1m-111-files-128-decoys.json
```

The latency columns are the median across the ten workload-specific p50 or p95
statistics; they are not pooled percentiles. Each cold p95 is the maximum of
three samples.

| Rows / files / empty decoys | Parquet | Cold p50 / p95 | Warm p50 / p95 | Cold / warm planning p50 | Max query RSS growth |
|---|---:|---:|---:|---:|---:|
| 100k / 8 / 0 | 0.58 MB | 30.1 / 32.8 ms | 17.6 / 19.8 ms | 21.2 / 13.6 ms | 52.1 MiB |
| 500k / 32 / 0 | 2.74 MB | 54.7 / 56.9 ms | 26.4 / 28.3 ms | 22.3 / 13.5 ms | 54.3 MiB |
| 1M / 111 / 128 | 5.61 MB | 153.7 / 163.9 ms | 62.0 / 68.9 ms | 51.8 / 16.0 ms | 67.6 MiB |

The sweep intentionally changes rows, files, and namespace count together. It
shows a metadata/file-count stress, not an isolated coefficient.

### 1M/111-file workload latency (M)

Exact command/configuration:

```bash
cd lib/finelog
uv run --group dev python -m finelog.benchmarks.telemetry_layout_bench baseline \
  --rows 1000000 --duration-days 14 --batch-rows 25000 --segments 111 \
  --decoy-namespaces 128 --distinct-run-ids 100000 \
  --cold-iterations 3 --warm-iterations 7 --warmup-iterations 1 \
  --concurrency 1 4 8 --concurrency-queries-per-worker 2 \
  --work-dir /tmp/finelog-baseline-final-205-1m-111 \
  --output ../../docs/reports/finelog-telemetry-benchmark-results/baseline-1m-111-files-128-decoys.json
```

| Workload | Cold p50 / p95 | Warm p50 / p95 | Cold plan p50 / p95 | Warm plan p50 / p95 | RSS growth |
|---|---:|---:|---:|---:|---:|
| Run 15m | 159.8 / 164.7 ms | 62.7 / 70.3 ms | 52.1 / 55.1 ms | 16.8 / 19.4 ms | 56.5 MiB |
| Run 24h | 162.9 / 163.2 ms | 74.3 / 82.6 ms | 55.5 / 66.8 ms | 16.7 / 17.9 ms | 58.9 MiB |
| Run 7d | 255.0 / 285.7 ms | 210.4 / 228.6 ms | 61.8 / 65.7 ms | 16.5 / 18.8 ms | 61.1 MiB |
| Worker outliers | 149.3 / 206.0 ms | 46.8 / 47.8 ms | 64.6 / 70.1 ms | 20.7 / 23.3 ms | 59.3 MiB |
| vLLM histogram | 124.8 / 130.3 ms | 41.4 / 44.0 ms | 54.2 / 58.3 ms | 14.7 / 14.8 ms | 54.4 MiB |
| Fleet scan-shape proxy | 135.8 / 138.6 ms | 43.6 / 45.6 ms | 42.3 / 48.6 ms | 13.4 / 13.8 ms | 58.3 MiB |
| Node/device | 158.1 / 204.2 ms | 66.3 / 77.2 ms | 51.6 / 55.3 ms | 15.5 / 16.1 ms | 56.6 MiB |
| Errors/log scan-shape proxy | 143.3 / 148.1 ms | 43.3 / 46.6 ms | 46.1 / 50.6 ms | 13.5 / 14.1 ms | 61.2 MiB |
| Alert evaluation | 145.0 / 152.8 ms | 61.2 / 67.5 ms | 46.3 / 47.1 ms | 20.1 / 20.8 ms | 62.6 MiB |
| Accounting | 209.2 / 211.4 ms | 124.9 / 136.3 ms | 41.2 / 42.1 ms | 15.3 / 19.2 ms | 67.6 MiB |

### 1M/111-file pruning and file-open metrics (M)

Exact command/configuration:

```bash
cd lib/finelog
uv run --group dev python -m finelog.benchmarks.telemetry_layout_bench baseline \
  --rows 1000000 --duration-days 14 --batch-rows 25000 --segments 111 \
  --decoy-namespaces 128 --distinct-run-ids 100000 \
  --cold-iterations 3 --warm-iterations 7 --warmup-iterations 1 \
  --concurrency 1 4 8 --concurrency-queries-per-worker 2 \
  --work-dir /tmp/finelog-baseline-final-205-1m-111 \
  --output ../../docs/reports/finelog-telemetry-benchmark-results/baseline-1m-111-files-128-decoys.json
```

| Workload | Files pruned | Row groups pruned / matched | Scan | Metadata | File open |
|---|---:|---:|---:|---:|---:|
| Run 15m | 0 / 111 | 110 / 1 | 2.4 KiB | 70.7 ms | 83.9 ms |
| Run 24h | 0 / 111 | 103 / 8 | 18.8 KiB | 79.1 ms | 84.7 ms |
| Run 7d | 0 / 111 | 55 / 56 | 134.4 KiB | 81.9 ms | 45.4 ms |
| Worker outliers | 0 / 111 | 110 / 1 | 3.3 KiB | 60.5 ms | 72.0 ms |
| vLLM histogram | 0 / 111 | 110 / 1 | 6.6 KiB | 65.6 ms | 77.3 ms |
| Fleet scan-shape proxy | 0 / 111 | 108 / 3 | 6.0 KiB | 65.0 ms | 69.1 ms |
| Node/device | 0 / 111 | 103 / 8 | 60.3 KiB | 64.0 ms | 70.3 ms |
| Errors/log scan-shape proxy | 0 / 111 | 108 / 3 | 19.6 KiB | 57.8 ms | 60.6 ms |
| Alert evaluation | 0 / 111 | 110 / 1 | 8.6 KiB | 63.3 ms | 73.2 ms |
| Accounting | 0 / 111 | 55 / 56 | 344.9 KiB | 66.3 ms | 33.5 ms |

This is the central result: row-group pruning is effective, but every bounded
query still registers/opens 111 files.

### 1M mixed concurrency (M)

Exact command/configuration:

```bash
cd lib/finelog
uv run --group dev python -m finelog.benchmarks.telemetry_layout_bench baseline \
  --rows 1000000 --duration-days 14 --batch-rows 25000 --segments 111 \
  --decoy-namespaces 128 --distinct-run-ids 100000 \
  --cold-iterations 3 --warm-iterations 7 --warmup-iterations 1 \
  --concurrency 1 4 8 --concurrency-queries-per-worker 2 \
  --work-dir /tmp/finelog-baseline-final-205-1m-111 \
  --output ../../docs/reports/finelog-telemetry-benchmark-results/baseline-1m-111-files-128-decoys.json
```

| Concurrency | Queries | p50 / p95 | Throughput | Peak RSS |
|---:|---:|---:|---:|---:|
| 1 | 2 | 66.9 / 76.4 ms | 14.9 q/s | 357.1 MiB |
| 4 | 8 | 105.0 / 327.2 ms | 24.3 q/s | 365.8 MiB |
| 8 | 16 | 198.8 / 343.2 ms | 30.8 q/s | 372.4 MiB |

These are warm mixed queries over one embedded server.

## Namespace binding and metadata cache (M)

The manifest command’s all-files control has one referenced namespace and no
unrelated providers, so it is not used to decide lazy binding. The targeted
comparison below gives every unrelated namespace one nonempty hardlinked
segment.

Exact command/configuration:

```bash
cd lib/finelog
uv run --group dev python -m finelog.benchmarks.telemetry_layout_bench binding \
  --source-log-dir /tmp/finelog-baseline-final-205-1m-111/current \
  --rows 1000000 --duration-days 14 --batch-rows 25000 --segments 111 \
  --distinct-run-ids 100000 --decoy-counts 32 128 \
  --cold-iterations 3 --warm-iterations 7 --warmup-iterations 1 \
  --concurrency 1 4 8 --concurrency-queries-per-worker 2 \
  --work-dir /tmp/finelog-binding-final-205-1m-111-v2 \
  --output ../../docs/reports/finelog-telemetry-benchmark-results/binding-nonempty-decoys-1m-111-files.json
```

| Variant | Providers / total segments | Workload | Cold query p50 / p95 | Cold plan p50 / p95 | Warm plan p50 / p95 | RSS growth |
|---|---:|---|---:|---:|---:|---:|
| Referenced only | 1 / 111 | Run 15m | 153.5 / 322.3 ms | 39.6 / 87.8 ms | 30.9 / 36.0 ms | 56.0 MiB |
| 32 nonempty decoys | 33 / 143 | Run 15m | 232.5 / 251.2 ms | 49.9 / 50.3 ms | 16.9 / 22.2 ms | 60.2 MiB |
| 128 nonempty decoys | 129 / 239 | Run 15m | 148.5 / 161.2 ms | 51.7 / 53.7 ms | 28.2 / 35.0 ms | 57.0 MiB |
| Referenced only | 1 / 111 | Alert | 246.7 / 256.2 ms | 58.5 / 58.9 ms | 19.6 / 21.6 ms | 61.6 MiB |
| 32 nonempty decoys | 33 / 143 | Alert | 144.1 / 153.8 ms | 56.7 / 62.2 ms | 19.7 / 20.0 ms | 61.4 MiB |
| 128 nonempty decoys | 129 / 239 | Alert | 204.8 / 273.8 ms | 73.0 / 92.1 ms | 21.8 / 22.9 ms | 63.6 MiB |

Process-cold planning median across the two query shapes rises from 49.1 ms
referenced-only to 62.4 ms with 128 nonempty decoys. Query latency is noisy and
not monotonic, so it is not used as a coefficient. Warm planning stays roughly
flat, confirming that the existing shared metadata cache helps after startup.
The variants ran sequentially in one host process order, so the comparison does
not isolate page-cache or allocator history. Lazy binding is a structural guard
against provider growth before namespace splitting, not a fitted latency
coefficient.

## Segment manifest (M for queries, P for selection)

The prototype reads local Parquet footer time/name bounds, selects files in
memory, and exposes the selected hardlinks to the unchanged Finelog
`NamespaceProvider`. Result digests are checked before timing; digest
normalization rounds floating aggregates to 1e-9 to tolerate reduction-order
noise.

Exact command/configuration:

```bash
cd lib/finelog
uv run --group dev python -m finelog.benchmarks.telemetry_layout_bench manifest \
  --source-log-dir /tmp/finelog-baseline-final-205-1m-111/current \
  --rows 1000000 --duration-days 14 --batch-rows 25000 --segments 111 \
  --decoy-namespaces 128 --distinct-run-ids 100000 \
  --cold-iterations 3 --warm-iterations 7 --warmup-iterations 1 \
  --concurrency 1 4 8 --concurrency-queries-per-worker 2 \
  --work-dir /tmp/finelog-manifest-final-205-1m-111 \
  --output ../../docs/reports/finelog-telemetry-benchmark-results/candidates-binding-manifest-1m-111-files.json
```

| Workload | Selected files | Selection p50 / p95 | Same-command all-files control → manifest cold p50 | Control → manifest warm p50 | Manifest metadata / open |
|---|---:|---:|---:|---:|---:|
| Run 15m | 1 | 8 / 14 µs | 150.5 → 19.9 ms | 58.1 → 11.1 ms | 1.2 / 2.7 ms |
| Run 24h | 8 | 14 / 18 µs | 158.9 → 42.7 ms | 74.0 → 31.2 ms | 8.5 / 4.9 ms |
| Run 7d | 56 | 59 / 74 µs | 247.6 → 199.3 ms | 185.1 → 201.7 ms | 54.7 / 4.7 ms |
| Worker outliers | 1 | 14 / 17 µs | 138.6 → 66.0 ms | 96.0 → 21.7 ms | 1.2 / 2.8 ms |
| vLLM histogram | 1 | 14 / 17 µs | 125.8 → 52.4 ms | 43.6 → 26.4 ms | 1.0 / 10.2 ms |
| Fleet scan-shape proxy | 3 | 17 / 19 µs | 138.7 → 71.4 ms | 41.0 → 22.2 ms | 6.6 / 6.4 ms |
| Node/device | 8 | 25 / 28 µs | 153.3 → 139.4 ms | 64.0 → 52.7 ms | 14.6 / 13.9 ms |
| Errors/log scan-shape proxy | 3 | 17 / 19 µs | 129.9 → 108.8 ms | 52.4 → 54.9 ms | 16.3 / 4.4 ms |
| Alert evaluation | 1 | 14 / 15 µs | 144.5 → 85.0 ms | 58.3 → 55.9 ms | 1.1 / 10.6 ms |
| Accounting | 56 | 101 / 128 µs | 211.9 → 365.8 ms | 124.4 → 225.8 ms | 27.6 / 2.3 ms |

The manifest wins short bounded queries and is neutral/mixed for wide scans.
Accounting’s regression prevents a blanket latency claim; the production
implementation should use the manifest to bound registration and retain
ordinary DataFusion scans for the selected files.

Both sides of this table come from the `manifest` command above. The control
registers all 111 Telltale files under only the referenced namespace. The
targeted binding section provides the unrelated-provider evidence.

The manifest alert-only concurrency proxy recorded p95 66.5/149.9/407.2 ms at
1/4/8 workers. It is not directly comparable with the baseline mixed-workload
concurrency table.

## Logical split, clustering, and four-bucket mechanics (M/P)

These are 100k mechanics measurements, not the production decision point.
Candidate query latency is StatsService-measured; preparation uses local
hardlinks/current compaction. Values below are medians of the ten workload
p50/p95 statistics.

Exact command/configuration:

```bash
cd lib/finelog
uv run --group dev python -m finelog.benchmarks.telemetry_layout_bench layouts \
  --source-log-dir /tmp/finelog-baseline-final-205-100k/current \
  --rows 100000 --duration-days 14 --batch-rows 25000 --segments 8 \
  --distinct-run-ids 100000 --split-files-per-group 1 --partition-buckets 4 \
  --cold-iterations 3 --warm-iterations 7 --warmup-iterations 1 \
  --concurrency 1 4 8 --concurrency-queries-per-worker 2 \
  --work-dir /tmp/finelog-layouts-final-205-100k-v3 \
  --output ../../docs/reports/finelog-telemetry-benchmark-results/candidates-layouts-100k-8-files.json
```

| Layout | Files built | Cold p50 / p95 | Warm p50 / p95 | Cold planning p50 | Matched files / row groups | Preparation |
|---|---:|---:|---:|---:|---:|---:|
| Current name-keyed | 8 | 30.1 / 32.8 ms | 17.6 / 19.8 ms | 21.2 ms | 8 / 1 | 11.8 s |
| Access-pattern split | 8 | 23.7 / 25.7 ms | 13.9 / 14.7 ms | 21.1 ms | 1 / 1 | 28.3 s |
| Composite cluster key | 8 | 28.0 / 28.6 ms | 14.9 / 17.4 ms | 21.2 ms | 8 / 1 | 12.7 s |
| Hidden day+4 proxy | 56 | 37.6 / 47.5 ms | 16.9 / 28.7 ms | 33.2 ms | 2.5 / 2.5 | 72.7 s |

The split changes a measured component—one file instead of eight—and is already
required for schema/retention isolation. Composite clustering does not improve
planning or median latency because current name sorting and predicates already
prune to one row group for most queries. The four-bucket proxy improves
exact-run and exact-worker node selection; the corrected node query selects one
of 56 files. The layout still multiplies physical files sevenfold.

Current, split, and clustered mixed-load p95 at concurrency 8 were 61.6, 51.3,
and 76.4 ms. The day+4 concurrency measurement is alert-only and is not
comparable.

## Hidden day versus day+64/128 (P)

This command generates the exact 500k identities and counts nonempty partition
keys. It does not create/query 896 or 1,792 Parquet files. “Payload” divides
source Parquet bytes proportionally and excludes footer/small-file overhead.

Exact command/configuration:

```bash
cd lib/finelog
uv run --group dev python -m finelog.benchmarks.telemetry_layout_bench \
  partition-projection \
  --source-log-dir /tmp/finelog-baseline-final-205-500k/current \
  --rows 500000 --duration-days 14 --batch-rows 25000 --segments 32 \
  --distinct-run-ids 100000 --bucket-counts 64 128 \
  --output ../../docs/reports/finelog-telemetry-benchmark-results/partition-projection-500k-day-64-128.json
```

| Transform | Nonempty / maximum before segment multiplicity | Rows per partition min / p50 / p95 / max | Proportional payload | Selection p50 range | Run 15m / node 24h / alert 1h / accounting 7d files |
|---|---:|---:|---:|---:|---:|
| Day only | 14 / 14 | 34,816 / 34,816 / 36,992 / 36,992 | 195.5 KiB | 0.018–0.026 ms | 1 / 1 / 1 / 7 |
| Day + bucket 64 | 896 / 896 | 276 / 276 / 3,348 / 5,466 | 3.1 KiB | 0.58–0.84 ms | 1 / 1 / 64 / 448 |
| Day + bucket 128 | 1,792 / 1,792 | 114 / 147.5 / 184 / 5,352 | 1.5 KiB | 1.16–3.34 ms | 1 / 1 / 128 / 896 |

The exact node predicate `worker = 'worker-0007'` maps to worker-principal
bucket 7, so it selects one idealized file at both 64 and 128 buckets. The
rejection is driven by fleet/alert and accounting fan-out, not exact-node
history.

**E — incident-byte projection:** distributing 11,849,890,300 incident bytes
evenly over 896/1,792 ideal partitions yields 12.6/6.3 MiB per partition before
segment multiplicity. The incident’s time span and distribution are not known,
so this is not a capacity prediction. It shows that 64/128 buckets do not
naturally produce large compacted files and make non-run scans worse.

Day-only remains a distinct future option if retention deletion needs physical
time partitions. It is not needed for the query gate because the manifest
already supplies event-time file pruning.

## Minute/hour rollup methods (P)

This is not query latency. The materializer issues 14 day-bounded StatsService
aggregate queries and concatenates their states. The compaction-time proxy reads
each local Parquet segment through DuckDB, computes associative partial states,
then merges them. Both implement the same fixed-dimension projection:
sum/min/max/count grouped by bucket, name, source, run, job, worker, region,
cluster, and nine allowlisted label values. The projection omits
`task_index`, `attempt`, `process_index`, and every non-allowlisted label. For
minute and hour, a full outer comparison between those two implementations
returned aggregate-state `difference_count=0`; this does not validate the
projection’s semantic completeness.

For day `d`, the materializer predicate is
`ts >= start + d days AND ts < min(start + (d + 1) days, corpus end)`. The 14
nonoverlapping responses are concatenated; no unbounded response is used.

Exact command/configuration:

```bash
cd lib/finelog
uv run --group dev python -m finelog.benchmarks.telemetry_layout_bench rollups \
  --source-log-dir /tmp/finelog-baseline-final-205-500k/current \
  --rows 500000 --duration-days 14 --batch-rows 25000 --segments 32 \
  --distinct-run-ids 100000 --resolutions minute hour \
  --work-dir /tmp/finelog-rollups-final-205-500k-v2 \
  --output ../../docs/reports/finelog-telemetry-benchmark-results/rollup-proxies-500k-32-files.json
```

| Resolution / method | Time | Output rows | Output Parquet | Peak RSS | Fixed-projection aggregate-state `difference_count` |
|---|---:|---:|---:|---:|---:|
| Minute / 14 day-bounded materializer queries | 2.76 s | 470,588 | 3.88 MB | 669 MB | 0 |
| Minute / local per-segment compaction proxy | 8.95 s | 470,588 | 10.41 MB | 1,102 MB | 0 |
| Hour / 14 day-bounded materializer queries | 2.73 s | 470,588 | 3.88 MB | 1,139 MB | 0 |
| Hour / local per-segment compaction proxy | 7.42 s | 470,588 | 10.42 MB | 1,306 MB | 0 |

Peak RSS is process-order-sensitive because methods run sequentially in one
process. The timing includes rereading completed data for both local proxies;
real compaction integration could reuse in-flight batches. It would also couple
watermarks, late-data recomputation, and compaction correctness.

**Negative result:** the initial unbounded minute materializer attempted to
return 123,825,416 bytes (123.8 MB / 118.1 MiB) through a StatsService response
with a 67,108,864-byte (64 MiB) cap and failed. The checked harness therefore
uses 14 day-bounded queries, matching a watermark materializer’s bounded input
contract.

At this cadence, minute and hour both retain 470,588 rows from 500,000 raw rows;
the 500k corpus has only 230 scrapes about 88 minutes apart, so neither minute
nor hour can coalesce successive scrapes. The apparent reduction is mostly the
excluded error/log group, not useful temporal compression. Do not land generic
rollups on this result. If production descriptor cadence justifies rollups, use
the standalone materializer first and measure its dimensions,
late-data/completeness behavior, and compression at production cadence.

## Run catalog (M)

Exact command/configuration:

```bash
cd lib/finelog
uv run --group dev python -m finelog.benchmarks.telemetry_layout_bench catalog \
  --run-ids 100000 --cold-iterations 3 --warm-iterations 7 \
  --warmup-iterations 1 --concurrency 1 4 8 \
  --concurrency-queries-per-worker 2 \
  --work-dir /tmp/finelog-catalog-final-205-100k \
  --output ../../docs/reports/finelog-telemetry-benchmark-results/run-catalog-100k.json
```

| IDs / files | Cold p50 / p95 | Warm p50 / p95 | Cold / warm planning p50 | Files / row groups matched | Scan | RSS growth |
|---|---:|---:|---:|---:|---:|---:|
| 100,000 / 1 | 24.4 / 30.5 ms | 9.4 / 12.6 ms | 17.3 / 10.2 ms | 1 / 1 | 14.1 KiB | 44.2 MiB |

This clears the sub-second warm run-prefix target with wide margin.

## Extrapolation and stop rule

No 1B synthetic run was attempted. **E — linear generator projection:** the
1M/111-file build took 164.5 seconds. Scaling only its local generation work to
988,423,984 rows projects about 45.2 hours and 5.16 GiB of unusually
compressible synthetic Parquet. The production incident already supplies the
1B/111-file observation, and the local 1M stress isolates the file-registration
component. A 10M or Iris run would not change the selected smallest fix.

The result does not prove production storage capacity, 64/128-bucket query
latency, or rollup compression. Those are explicitly unmeasured.

## Reproduction and raw results

The harness is under `finelog.benchmarks`:

- `telemetry_workload_corpus.py`: deterministic generator and bounded SQL;
- `query_measurement.py`: StatsService cold/warm/planning/scan/RSS/load metrics;
- `layout_candidates.py`: current stores, manifests, layouts, partition
  projections, catalog, and rollup proxies;
- `telemetry_layout_bench.py`: checked CLI.

Raw outputs:

- [100k baseline](finelog-telemetry-benchmark-results/baseline-100k-8-files.json)
- [500k baseline](finelog-telemetry-benchmark-results/baseline-500k-32-files.json)
- [1M/111-file baseline](finelog-telemetry-benchmark-results/baseline-1m-111-files-128-decoys.json)
- [nonempty binding](finelog-telemetry-benchmark-results/binding-nonempty-decoys-1m-111-files.json)
- [manifest](finelog-telemetry-benchmark-results/candidates-binding-manifest-1m-111-files.json)
- [layout mechanics](finelog-telemetry-benchmark-results/candidates-layouts-100k-8-files.json)
- [64/128 projection](finelog-telemetry-benchmark-results/partition-projection-500k-day-64-128.json)
- [rollup proxies](finelog-telemetry-benchmark-results/rollup-proxies-500k-32-files.json)
- [100k run catalog](finelog-telemetry-benchmark-results/run-catalog-100k.json)

Relevant engine behavior is in
`lib/finelog/rust/src/store/store.rs::query_providers`,
`lib/finelog/rust/src/query/provider.rs`, and
`lib/finelog/rust/src/query/mod.rs`. DataFusion documents the shared metadata
cache and Parquet pruning controls in its
[configuration reference](https://datafusion.apache.org/user-guide/configs.html)
and explains why footer/open latency matters in
[Parquet pruning](https://datafusion.apache.org/blog/2025/03/20/parquet-pruning/)
and [external indexes](https://datafusion.apache.org/blog/2025/08/15/external-parquet-indexes/).
Iceberg’s [hidden partition evolution](https://iceberg.apache.org/docs/1.7.0/docs/evolution/)
supports keeping logical SQL independent from any future physical transform.

## Foundation integration gate

The foundation branch should integrate this benchmark plus:

- lazy referenced-namespace binding;
- generation-keyed catalog/manifest caching;
- manifest prefiltering by event time and available clustering/entity
  summaries;
- access-pattern logical namespaces;
- the run catalog and restricted bounded query templates.

It should not integrate a production physical partition/clustering layout or
compaction-time rollup from this record. Re-run the checked commands after the
foundation implementation; the production gate remains first useful panels
within two seconds warm/five seconds cold, sub-second run lookup, and alert
queries well inside their evaluation interval.
