---
topic: harrier-fuzzy-duplicates
issue: https://github.com/marin-community/marin/issues/8162
description: Produce Harrier embeddings for noncanonical fuzzy-duplicate documents.
author: Rafal Wojdyla
---

# Harrier Fuzzy-Duplicate Embeddings: Run Logbook

## Run Contract

- DRI: Rafal Wojdyla.
- Goal: Produce co-partitioned Harrier embeddings for noncanonical fuzzy-duplicate documents in all 292 Datakit sources.
- Stop or escalation criteria: Escalate repeated application failures, output conflicts, or no worker allocation after 30 minutes.
- Issue: https://github.com/marin-community/marin/issues/8162
- Source: `48ef14ea8e62d15fee2d1db101feae216ca04fbc` on `origin/rav/harrier-all`.
- Output root: `s3://marin-us-east-02a/marin/datakit/embed/harrier-fuzzy-duplicates/`.
- W&B: Not used by this embedding job.
- Checkpoints: Not used. Zephyr keeps completed Parquet shards and skips them after a restart.
- Completion target: 292 source artifacts across two deterministic partitions.
- Monitor owner: Codex, with a 15-minute check cadence after the initial checks.

## Entry Log

### 2026-08-11 19:54 UTC - Launch plan

- Commit hash: `48ef14ea8e62d15fee2d1db101feae216ca04fbc`.
- Dirty tree: No.
- Runtime source: Iris repository sync from `origin/rav/harrier-all`.
- Hardware: 96 H100 TEI workers in `cw-us-east-02a` and 96 H100 TEI workers in `cw-rno2a`.
- Source concurrency: One source in each region.
- East command: `uv run iris --cluster marin job run --no-wait --target-cluster cw-us-east-02a --job-name harrier-fuzzy-dups-east-p0-20260811 --priority interactive --cpu 2 --memory 8GB --disk 16GB --enable-extra-resources --extra datakit -e MARIN_PREFIX s3://marin-us-east-02a/marin -- python -m experiments.datakit.embeddings.harrier.run --document-set fuzzy_duplicates --partition-index 0 --partition-count 2 --tei-instances 96 --max-concurrent 1`.
- RNO command: `uv run iris --cluster marin job run --no-wait --target-cluster cw-rno2a --job-name harrier-fuzzy-dups-rno-p1-20260811 --priority interactive --cpu 2 --memory 8GB --disk 16GB --enable-extra-resources --extra datakit -e MARIN_PREFIX s3://marin-us-east-02a/marin -- python -m experiments.datakit.embeddings.harrier.run --document-set fuzzy_duplicates --partition-index 1 --partition-count 2 --tei-instances 96 --max-concurrent 1`.
- Result: The launch is not started.
- Next action: Commit and push this run contract, then submit both Iris jobs.

### 2026-08-11 19:56 UTC - Dual-region launch

- Runtime source SHA: `1bb93a759198bc463a87514ac6b421731ffdd6f2`.
- Code commit: `48ef14ea8e62d15fee2d1db101feae216ca04fbc`.
- Dirty tree: No.
- East job: `/rav/harrier-fuzzy-dups-east-p0-20260811`.
- RNO job: `/rav/harrier-fuzzy-dups-rno-p1-20260811`.
- Priority: Interactive.
- Hardware: 96 H100 TEI workers per region, for 192 H100 workers total.
- Partitions: East uses partition 0 of 2. RNO uses partition 1 of 2.
- Output root: `s3://marin-us-east-02a/marin/datakit/embed/harrier-fuzzy-duplicates/`.
- Result: Both Iris submissions succeeded.
- Next action: Check startup state after 120 seconds, then monitor at a 15-minute cadence.

### 2026-08-11 20:00 UTC - TEI port collisions

- Status: Both roots are running, but the TEI pools are degraded.
- Evidence: RNO has 74 failed descendants and 25 running descendants. East has repeated `Address already in use` errors.
- Root cause: All TEI children bind fixed ports 8080 and 18080 on host-networked CoreWeave nodes.
- Prior record: https://echo.oa.dev/wiki/66 documents the same host-global port failure class.
- Decision: Request Iris task ports for TEI HTTP and metrics endpoints.
- Recovery: Validate and push the fix. Then stop both degraded roots and relaunch the same partitions.
- Output safety: Zephyr skips completed Parquet shards, so a relaunch can reuse valid output.

### 2026-08-11 20:05 UTC - Allocated-port repair validated

- Commit hash: `daa2275a0cb63ca5ffed9fb64d59394d777a31f8`.
- Change: Fray forwards named port requests to Iris. Harrier TEI uses allocated HTTP and metrics ports.
- Regression evidence: Both focused tests failed before the repair and passed after it.
- Validation: Fray has 89 passing tests. Datakit has 248 passing tests and five expected failures.
- Validation: Pre-commit and Pyrefly passed.
- Next action: Push the repair, stop both degraded roots, and relaunch both partitions from the new SHA.

### 2026-08-11 20:07 UTC - Repaired dual-region launch

- Runtime source SHA: `2e23fbe3d9b65ec7095b13a0dd94cf3ef6cb221a`.
- Code commit: `daa2275a0cb63ca5ffed9fb64d59394d777a31f8`.
- Dirty tree: No.
- Old east job: `/rav/harrier-fuzzy-dups-east-p0-20260811`, stopped.
- Old RNO job: `/rav/harrier-fuzzy-dups-rno-p1-20260811`, stopped.
- East job: `/rav/harrier-fuzzy-dups-east-p0-20260811-v2`.
- RNO job: `/rav/harrier-fuzzy-dups-rno-p1-20260811-v2`.
- Priority: Interactive.
- Hardware: 96 H100 TEI workers per region, for 192 H100 workers total.
- Output root: `s3://marin-us-east-02a/marin/datakit/embed/harrier-fuzzy-duplicates/`.
- Result: Both repaired Iris submissions succeeded.
- Next action: Check startup state after 120 seconds and verify that port collisions do not recur.

### 2026-08-11 20:10 UTC - Kubernetes port-zero behavior

- Status: Both v2 roots are stopped before source processing started.
- Evidence: All 96 descendants ran in each region without bind errors, but TEI logged HTTP and metrics ports as zero.
- Root cause: The CoreWeave Kubernetes backend does not allocate numeric named ports. It supplies zero for automatic binding.
- Effect: TEI selects a port, but the parent health check and endpoint registry cannot discover that port.
- Decision: Use Iris ports when they are nonzero. Use a deterministic, run-specific port block when Iris returns zero.
- Next action: Add fallback-port tests, validate the repair, and launch v3.

### 2026-08-11 20:16 UTC - CoreWeave fallback ports validated

- Commit hash: `fd073d80eb9e632db84e2ff7e03d823681d7dfc2`.
- Change: Each pool selects a run-specific block in ports 12000 through 13999 when Iris returns zero.
- Port coverage: A 96-instance pool gets 192 unique HTTP and metrics ports.
- Regression evidence: The port-zero tests failed before the repair and passed after it.
- Validation: Datakit has 250 passing tests and five expected failures.
- Validation: Pre-commit and Pyrefly passed.
- Next action: Push the repair and launch v3 in both regions.

### 2026-08-11 20:17 UTC - CoreWeave fallback launch

- Runtime source SHA: `e9b9368d49dd49942edd4f16f457a73cefb97215`.
- Code commit: `fd073d80eb9e632db84e2ff7e03d823681d7dfc2`.
- Dirty tree: No.
- East job: `/rav/harrier-fuzzy-dups-east-p0-20260811-v3`.
- RNO job: `/rav/harrier-fuzzy-dups-rno-p1-20260811-v3`.
- Priority: Interactive.
- Hardware: 96 H100 TEI workers per region, for 192 H100 workers total.
- Output root: `s3://marin-us-east-02a/marin/datakit/embed/harrier-fuzzy-duplicates/`.
- Result: Both v3 Iris submissions succeeded.
- Next action: Verify nonzero TEI ports, service registration, source-job startup, and error-free logs.
