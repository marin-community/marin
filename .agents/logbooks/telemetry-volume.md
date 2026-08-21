---
topic: telemetry-volume
issue: https://github.com/marin-community/marin/issues/8563
description: Attribute and reduce Finelog telemetry bytes per day with semantic child namespaces.
author: power
---

# Telemetry Volume: Task Logbook

## Current TL;DR

- 2026-08-21 live baseline: 643,485,005 rows and 14.94 GiB across 28.54 hours, or 541.2 million rows and 12.57 GiB/day.
- Levanter accounts for 66.2% of retained rows. A 704-process run duplicates `step` across every process; node telemetry accounts for another 28.0% at a 30-second cadence.
- The selected producer changes reduced the copied sample by 68.6% compressed
  bytes. A 60-second compaction policy reduced it by 63.9% and changed
  five-minute GPU-utilization means by 1.12 percentage points on average.
- Phase-a Loom report: `telemetry-influx-phase-a`.
- Coordinating issue: https://github.com/marin-community/marin/issues/8563

## Scope

- Goal: reduce compressed telemetry bytes/day and retain core training signals longer.
- Primary metrics: compressed bytes/day, rows/day, child retention budgets, and representative query latency.
- Constraints: production Finelog and its archive are read-only experiment inputs; copied stores run locally; no Iris cluster restart or Finelog rollout is authorized.
- Coordinating issue: https://github.com/marin-community/marin/issues/8563

## Current Baseline

- Date: 2026-08-21 21:31 UTC
- Code ref: `origin/main` at task start
- Retained window: 2026-08-20 16:58:52.701 through 2026-08-21 21:31:06.468 UTC
- Rows: 643,485,005
- Bytes: 16,043,909,016
- Inherited local cap: 15 GiB
- Implied rate: 541.2 million rows/day and 12.57 GiB/day

## Hypothesis Queue

### Active

- None.

### Blocked

- None.

### Falsified / Dead End

- `TEL-VOL-005`: generic compaction-time last-gauge downsampling is not selected
  for production. At 60 seconds it saved fewer bytes than the producer arm and
  changed five-minute hardware aggregates. At 300 seconds it saved 6.1
  percentage points more than the producer arm but increased p95
  GPU-utilization mean error to 28.1 points.

### Promoted

- `TEL-VOL-001`: publish Levanter tracker metrics only from process 0 and retain
  core metrics every step while sampling extra mappings and reduced moments
  every 10 steps.
- `TEL-VOL-002`: collect node metrics every 60 seconds and scrape vLLM and Iris
  controller metrics every 60 seconds.
- `TEL-VOL-003`: route client-selected records to semantic `telemetry_v1.*`
  children while keeping `telemetry_v1` as their query rollup.
- `TEL-VOL-004`: divide a 50 GiB local-cache budget across semantic children.

## Decision Log

- 2026-08-21: use semantic child names such as `telemetry_v1.levanter.core`
  and `telemetry_v1.vllm`; keep `telemetry_v1` as a logical rollup for existing
  and composite queries.
- 2026-08-21: select the newest immutable Parquets by sequence until the copy reaches at least 10 GiB. This samples the current writer mix and avoids GCS-side mutation.
- 2026-08-21: require every client sample to name a finite semantic group;
  generic probes receive the owning application's group explicitly.
- 2026-08-21: include client-registered `telemetry_v1.*` children in the
  composite view when required columns and types are compatible. Align fields
  by name and null-fill optional additions; an incompatible child disables the
  composite view without affecting direct or unrelated queries.
- 2026-08-21: evaluate compaction downsampling, but require a metric-kind-aware
  policy before production use because imported counters and histograms are
  stored as gauges with their source semantics in attributes.

## Negative Results Index

- Whole-sample exact downsampling required a 29.8 GiB DuckDB spill and exceeded
  the experiment disk budget. The replacement evaluates each immutable
  compaction segment independently, matching an incremental compactor policy.

## Entry Log

### 2026-08-21 21:31 - TEL-VOL-000 live attribution

- Hypothesis: a small number of services or metric families account for most `telemetry_v1` volume.
- Commit Hash: none; read-only production measurement.
- Commands: `uv run finelog namespaces marin`; `uv run finelog schema marin telemetry_v1`; the SQL recorded in Loom artifact `telemetry-influx-phase-a`.
- Config: production `marin` Finelog through the IAP endpoint; retained-window and latest-30-minute aggregations.
- Result: Levanter contributed 425,993,068 rows (66.2%), node telemetry 180,268,843 (28.0%), vLLM 22,307,818 (3.5%), and Iris controller 14,942,065 (2.3%). `step` contributed 1,445,263 rows in the latest 30 minutes. The process grouping returned 704 active process indices for one run.
- Interpretation: the first treatments should remove cross-process training duplicates and lower node cadence. Histogram volume must be identified through metric names or `attributes_json.source_kind`, because imported snapshots persist as gauges.
- Next action: copy and rewrite a representative local production sample.

### 2026-08-21 21:34 - TEL-VOL-001 sample selection

- Hypothesis: the newest 10 GiB of archive segments represents the writer mix causing the current local-cache churn.
- Commit Hash: none; read-only object listing.
- Command: `gcloud storage ls --long 'gs://marin-us-central2/finelog/marin/telemetry_v1/*.parquet'`; select descending segment sequence until cumulative size is at least 10 GiB; `gcloud storage cp <selected objects> /tmp/marin-telemetry-volume-20260821/telemetry_v1/`.
- Config: 110 Parquets, 10,823,295,403 bytes (10.080 GiB), sequence endpoints 16,739,693,499 through 16,308,042,696.
- Result: copy started with 71 GiB free in `/tmp`.
- Interpretation: the sample is large enough to measure Parquet compression effects by service, cadence, and tier without reading the full archive.
- Next action: validate row/timestamp coverage and run rewrite treatments after the copy completes.

### 2026-08-21 22:03 - TEL-VOL-001 through TEL-VOL-004 producer replay

- Hypothesis: replicated-item suppression, lower writer cadence, and namespace
  routing reduce bytes while preserving core metrics.
- Commit Hash: none; local rewrite of the production sample.
- Command: DuckDB rewrites recorded in Loom artifact
  `telemetry-volume-mitigation-analysis`.
- Config: 110 Parquets covering 16.90 hours, 431,783,340 rows, and
  10,823,295,403 compressed bytes. Core names were `train_loss`, `step`,
  `phase`, `progress_time_seconds`, and `global_step`. Levanter extra used a
  10-step modulus; node telemetry used 2; vLLM and controller snapshots used 4.
- Result: suppressing replicated progress rows alone reduced compressed bytes
  6.4%. Cadence alone reduced them 62.0%. Combined, the sample fell to
  3,399,290,916 bytes and 137,648,462 rows, reductions of 68.6% and 68.1%.
  The resulting core candidate was 1,582,551,970 bytes over 16.90 hours. The
  final design replaces the experimental two-tier names with semantic children.
- Interpretation: producer cadence carries most of the reduction; grouping
  gives each producer a directly queryable ownership boundary.
- Next action: finish the compaction-downsampling arm and representative query
  comparisons, then publish the final artifact.

### 2026-08-21 22:40 - TEL-VOL-005 compaction downsampling

- Hypothesis: compaction-time last-gauge downsampling gives an additive storage
  arm with acceptable error in the five-minute aggregates used by dashboards.
- Commit Hash: none; local rewrite of the production sample.
- Command: `.venv/bin/python /tmp/telemetry_downsample.py`; analysis recorded in
  Loom artifact `telemetry-volume-mitigation-analysis`.
- Config: last sample per complete series key in 60- and 300-second wall-clock
  buckets; counters, events, and native histogram observations unchanged;
  independent processing of each of the 110 immutable source segments.
- Result: the 60-second policy retained 72,649,205 bulk rows and 1,604,504,870
  high-volume bytes. With core data unchanged, it reduced source bytes by 63.9%.
  The 300-second policy retained 20,704,087 high-volume rows and 436,680,325
  bytes, a 74.7% total reduction. At 60 seconds, five-minute GPU-utilization
  mean error was 1.12 percentage points on average and 6.43 at p95; at 300
  seconds it was 4.16 on average and 28.1 at p95.
- Interpretation: producer changes give a larger reduction than 60-second
  compaction without altering emitted samples. The 300-second policy adds only
  6.1 percentage points versus the producer arm and loses transient hardware
  information. A future policy needs per-family reducers and a recent unsampled
  horizon.
- Next action: publish the report and implementation PR; keep downsampling as a
  follow-up design rather than part of this rollout.

### 2026-08-21 23:37 - staged Finelog rollout

- Hypothesis: the semantic namespace and 50 GiB policies can be introduced
  without interrupting ingest or existing composite queries.
- Commit Hash: `5ec73d87df`.
- Command: `uv run python lib/finelog/scripts/safe_deploy.py rollout marin-dev`,
  then promote the verified digest with `rollout marin --no-build`.
- Config: release image
  `ghcr.io/marin-community/finelog@sha256:3e2e2c4d19871801df0dc73fa6e7df86bdd10fef8077d5e3e1e62e4991032d2c`;
  prior production digest `29f9b266acb42b39aa2c0190b94ef46f37378a5bd934409b586ab3f65f79af0f`.
- Result: the first dev gate exposed a legacy-only nullable `alert_tag` column
  and a different column order. Production remained untouched while the
  rollup was changed to align compatible schemas by name. The corrected image
  passed health, namespace-policy, direct-child, and composite-query checks on
  `marin-dev`, then on `marin`. Production's bounded one-minute composite query
  returned 182,997 rows. The parent policy is 50 GiB; the seven server-owned
  child budgets sum to 50 GiB.
- Interpretation: the retention change is active before client cadence changes
  percolate, and schema evolution no longer requires identical physical column
  order across semantic children.
- Next action: monitor PR #8571 and watch child namespaces populate as updated
  clients launch.
