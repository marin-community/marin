# Finelog Telemetry Layout Research

## Background Research Brief

- Effort: medium
- Stop rule: reproduce on the full hot set, find a treatment above 10x, compare
  physical layouts on production rows, and rule on GCS-only and Lance options
- Date: 2026-08-17

### Question

How should Finelog index, group, sort, and structure `telemetry_v1` so exact
metric and run-scoped queries remain sub-second at the current ingest rate?

### Current Marin Context

The copied production hot set contained 1,360,540,454 rows in 244 Parquets
(16.06 GB), with 232 `.fidx` bundles (1.135 GB). It covered about 40 hours.
Levanter accounted for 1.051 billion rows, so splitting files by service did not
balance the dominant partition. GCS held about 17 days of source Parquets while
ordinary SQL exposed local and `BOTH` segments as the hot table.

The submitted equality literal contained backslashes. DataFusion treats those
as literal characters for `=`, so the corrected query is
`name = 'grad_norm_layers_0_ln2_b'`. The corrected query remained slow because
the exact-pruning planner opened postings payloads before learning whether a
value was indexed in that segment.

The recurring Zephyr selector exposed a second planning boundary. Its
five-minute predicate overlapped only 28 query-visible files, but the custom
exact/projection path constructed a 259-file listing before DataFusion could
apply Parquet pruning. Carrying catalog key bounds into the table provider
reduced its warm median from 13,275.8 to 10.5 milliseconds on the same copied
1.36-billion-row corpus.

### Internal Prior Work

- [Telemetry layout benchmark](https://github.com/marin-community/marin/blob/57151a89f4/docs/reports/finelog-telemetry-layout-benchmark.md)
  rejected day-plus-run-hash partitioning because fleet queries fanned out over
  many small files. It also established a separate run catalog as the better
  discovery path.
- [Echo wiki/90](https://echo.oa.dev/wiki/90) found trigram indexes ineffective
  for common metric families and measured a filtered projection at 403 ms.
- [Echo wiki/51](https://echo.oa.dev/wiki/51) records the Finelog canary gate:
  health alone is insufficient when Docker can mask a first-process panic;
  restart count and startup logs are also required.
- [Echo wiki/25](https://echo.oa.dev/wiki/25) records a forward-only catalog
  migration. This design avoids a new enum value and uses nullable columns plus
  unknown-field-tolerant JSON/protobuf additions so an old binary can reopen the
  catalog.

### External Prior Art

- [DataFusion Parquet pruning](https://datafusion.apache.org/blog/2025/03/20/parquet-pruning/)
  describes the file, row-group, page, and bloom-filter pruning hierarchy. It
  supports clustering common predicates while limiting metadata fanout.
- [Apache Iceberg partition evolution](https://iceberg.apache.org/docs/latest/partitioning/)
  provides the conceptual split between logical predicates and evolving
  physical layout. Finelog can keep that separation without adopting Iceberg.
- [Lance index format](https://lance.org/format/index/) supports paged and
  coalesced index access across fragments. Those ideas fit archive serving, but
  a format migration is unnecessary for the measured hot-query problem.

### Negative / Failed Leads

- Correcting the escaped SQL fixed semantics but left the query at 8.56 seconds
  warm in the production replay.
- A timestamp lower bound did not reduce planning until segment bounds were
  applied before constructing the source and secondary-index scans.
- Combining eight source files into one time-ordered file made exact lookup
  slower.
- Separate service/time files grew bytes by 31% and made exact lookup 9% slower.
- Global name sorting with 128K row groups was fast but grew bytes by 61%.
- Direct GCS serving took 19.42 seconds for a bounded recent query.
- Lance was not benchmarked because the Parquet planner and layout treatments
  already cleared the 10x target.

### Evidence Map

#### Claim: Header coverage removes the dominant exact-planning cost

- Support: the full replay fell from 8,563.6 to 228.0 milliseconds when the same
  Parquets were served without irrelevant index bundles. The implemented header
  gate measured 189.3 milliseconds.
- Contradiction: an escaped absent value still requires a source scan to prove
  absence and measured 1.86 seconds after planning became cheap.
- Directness to Marin: full copied hot set and production release baseline.
- Confidence: high.
- Action: version exact coverage and backfill existing bundles.

#### Claim: Service/run/name sorting with 128K row groups is the best tested layout

- Support: on 48.1 million production rows it answered the exact query in 15.8
  milliseconds versus 194.8 milliseconds and grew Parquet bytes by 6.6%.
- Contradiction: recent service aggregation regressed from 28.5 to 34.9
  milliseconds.
- Directness to Marin: production rows, same Finelog query path.
- Confidence: medium-high; layout sample covered eight terminal files.
- Action: apply to new telemetry merges and let the old hot layout age out.

#### Claim: GCS should remain archive storage rather than the normal SQL path

- Support: a bounded direct GCS query took 19.42 seconds; the local equivalent
  took about 240 milliseconds.
- Contradiction: a generation manifest and co-located query service could reduce
  listing and cross-region costs, but neither exists in this change.
- Directness to Marin: current bucket, VM region, and stored objects.
- Confidence: high for the current deployment.
- Action: retain local hot serving and pursue archive manifests separately.

#### Claim: Catalog key bounds must filter files before secondary-index planning

- Support: the unchanged five-minute Zephyr selector fell from 13,275.8 to
  10.5 milliseconds warm p50. Plan construction fell from 13,161.6 to 11.0
  milliseconds, with 28 candidate files instead of 259.
- Contradiction: segments with missing or invalid bounds must remain in the
  candidate set, so legacy or corrupt metadata weakens performance rather than
  correctness.
- Directness to Marin: full copied hot set and exact production alert SQL.
- Confidence: high.
- Action: capture key bounds with the segment-path snapshot and apply them
  before creating the listing table, projections, or postings plans.

### Recommended Next Experiments

#### 1. Archive manifest prototype

- Minimum experiment: publish event-time and identity bounds for one archive
  generation and query selected objects without bucket globbing.
- Baseline/control: 19.42-second direct GCS query.
- Expected signal: bounded object count and sub-ten-second cold response.
- Falsifier: footer/network latency remains above the server deadline.
- Cost/risk: medium; requires generation and cache consistency contracts.

### Hypothesis Queue Update

- Promote: exact header coverage, catalog-bound file elimination, promoted
  dimensions, and telemetry sort policy.
- Revise: archive work starts with a manifest, not a storage-format migration.
- Stop: physical service splitting, global name-only sorting, and raw GCS serving.

### Source Ledger

| Source | Type | Claim used for | Confidence | Notes |
|---|---|---|---|---|
| 2026-08-17 production replay | benchmark | exact planner and full-set latency | high | Copied hot set; raw JSON retained in Weaver session |
| 2026-08-18 Zephyr replay | benchmark | bounded planner latency and file selection | high | Full copied hot set; unchanged production selector |
| 48.1M-row layout sample | benchmark | sort/row-group choice | medium-high | Eight production terminal files |
| Telemetry layout benchmark | Marin report | prior partition/run-catalog decisions | high | Synthetic predecessor study |
| Echo wiki/90 | incident | trigram and projection behavior | high | Production alert workload |
| Echo wiki/51 and wiki/25 | incidents | rollout and rollback gates | high | Prior Finelog deployments |
| DataFusion pruning docs | official docs | physical pruning mechanism | high | Engine-level description |
| Iceberg and Lance docs | official docs | manifest/index prior art | medium | No adoption proposed |

### Handoff

- Open question: force an old-layout rewrite only if canary results justify its
  CPU and page-cache cost.
- Stop reason: exact-header, segment-bound, and physical-layout treatments all
  exceeded the 10x target; remaining work concerns rollout and archive serving.
