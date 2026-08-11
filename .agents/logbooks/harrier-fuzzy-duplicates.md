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

### 2026-08-11 20:26 UTC - First sources completed

- East result: `agenttrove_e27f2c9e` completed all 43 shards. The source step succeeded in 6 minutes and 52 seconds.
- RNO result: `biocorpus_41bbbae7` completed all 27 shards. The source step succeeded.
- Current work: East started `cp/biodiversity_23880f3b`. RNO started `climblab-ja_edc88cea` and reached 112 of 1,787 output shards.
- Service health: Both 96-worker TEI pools use distinct nonzero ports. Logs show successful requests and no port collisions, HTTP 429 responses, connection failures, dead workers, or memory errors.
- Result: Each region has written and sealed its first complete source artifact.
- Next action: Continue 15-minute monitoring. Post only major changes to issue #8162.

### 2026-08-11 20:38 UTC - Routine health check

- Root health: Both roots are running with zero failures and zero preemptions.
- East progress: Three source jobs succeeded. The fourth source job is running.
- RNO progress: One source job succeeded. `climblab-ja_edc88cea` reached 1,131 of 1,787 output shards with live workers and no dead workers.
- Error scan: No port collisions, HTTP 429 responses, connection failures, tracebacks, or memory errors.
- Next action: Continue the 15-minute check cadence.

### 2026-08-11 20:53 UTC - Routine health check

- Root health: Both roots are running with zero failures and zero preemptions.
- East progress: Three source jobs succeeded. `common-crawl-focus-2026-22_7a12c4bf` reached 1,661 of 4,573 output shards with 32 live workers and no dead workers.
- RNO progress: Two source jobs succeeded. `common_corpus-english_0e1cf2c4` reached 5,838 of 8,018 join-side tasks with 32 live workers and no dead workers.
- Error scan: No port collisions, HTTP 429 responses, connection failures, tracebacks, or memory errors.
- Next action: Continue the 15-minute check cadence.

### 2026-08-11 21:09 UTC - Routine health check

- Root health: Both roots are running with zero failures and zero preemptions.
- East progress: Five source jobs succeeded. `docx-corpus-en_1223b38b` is pending. This is the first check with that pending state.
- RNO progress: Two source jobs succeeded. `common_corpus-english_0e1cf2c4` entered its output stage and reached 184 of 8,018 shards with 32 live workers and no dead workers.
- Error scan: No port collisions, HTTP 429 responses, connection failures, tracebacks, or memory errors.
- Next action: Continue the 15-minute check cadence. Escalate if the East pending state lasts 30 minutes.

### 2026-08-11 21:25 UTC - Routine health check

- Root health: Both roots are running with zero failures and zero preemptions.
- East progress: Six source jobs succeeded. The prior pending state cleared. `eai-taxonomy-code-w-dclm_39527a3d` reached 2,753 of 5,872 join-side tasks with 32 live workers and no dead workers.
- RNO progress: Two source jobs succeeded. `common_corpus-english_0e1cf2c4` reached 455 of 8,018 output shards with 32 live workers and no dead workers.
- Retry check: RNO reports 59 one-time shard retries. The count stayed unchanged for at least five minutes. No matching exception, timeout, request, connection, or memory error was present.
- Next action: Continue the 15-minute check cadence. Track whether the RNO retry count grows.

### 2026-08-11 21:55 UTC - TEI transport retry safeguard

- Root health: Both roots are running with zero failures and zero preemptions.
- East progress: Six source jobs succeeded. `eai-taxonomy-code-w-dclm_39527a3d` reached 250 of 5,872 output shards with 32 live workers and no dead workers.
- RNO progress: Two source jobs succeeded. `common_corpus-english_0e1cf2c4` reached 858 of 8,018 output shards with 32 live workers and no dead workers.
- Retry diagnosis: TEI closed some HTTP connections without a response. This raised `http.client.RemoteDisconnected` outside the client retry loop. East has six one-time shard retries. RNO has 109 one-time shard retries. No shard has retried twice.
- Safeguard commit: `fdea5ebfd9cda42a7d37c61dbaea83d5b4914392` retries transport errors in the TEI request and health-check paths.
- Regression evidence: The observed disconnect test failed before the change and passed after it. Datakit has 253 passing tests and five expected failures. Pre-commit and Pyrefly passed. An independent review led to coverage of reset, timeout, and incomplete-response errors.
- Decision: Keep both v3 roots running because they continue to write shards with all workers live. Use the safeguard only if a recovery launch becomes necessary.
- Next action: Continue the 15-minute check cadence. Escalate if a shard reaches a second retry or either root stops making progress.

### 2026-08-11 22:11 UTC - Routine health check

- Root health: Both roots are running with zero failures and zero preemptions.
- East progress: Six source jobs succeeded. `eai-taxonomy-code-w-dclm_39527a3d` reached 434 of 5,872 output shards with 32 live workers and no dead workers.
- RNO progress: Two source jobs succeeded. `common_corpus-english_0e1cf2c4` reached 1,076 of 8,018 output shards with 32 live workers and no dead workers.
- Retry check: East stayed at six one-time retries. RNO stayed at 109 one-time retries. No shard has retried twice.
- Next action: Continue the 15-minute check cadence.

### 2026-08-11 22:30 UTC - Routine health check

- Root health: Both roots are running with zero failures and zero preemptions.
- East progress: Six source jobs succeeded. `eai-taxonomy-code-w-dclm_39527a3d` reached 627 of 5,872 output shards with 32 live workers and no dead workers.
- RNO progress: Two source jobs succeeded. `common_corpus-english_0e1cf2c4` reached 1,312 of 8,018 output shards with 32 live workers and no dead workers.
- Retry check: East increased from six to 22 one-time retries. RNO increased from 109 to 131 one-time retries. No shard has retried twice.
- Decision: Keep both roots running because shard output continues and all workers are healthy.
- Next action: Continue the 15-minute check cadence. Escalate if a shard reaches a second retry or either root stops making progress.

### 2026-08-11 22:45 UTC - Routine health check

- Root health: Both roots are running with zero failures and zero preemptions.
- East progress: Six source jobs succeeded. `eai-taxonomy-code-w-dclm_39527a3d` reached 790 of 5,872 output shards with 32 live workers and no dead workers.
- RNO progress: Two source jobs succeeded. `common_corpus-english_0e1cf2c4` reached 1,559 of 8,018 output shards with 32 live workers and no dead workers.
- Retry check: East increased from 22 to 24 one-time retries after two TEI connections closed without a response. RNO stayed at 131 one-time retries. No shard has retried twice.
- Decision: Keep both roots running because Zephyr requeued the affected East shards and output continues.
- Next action: Continue the 15-minute check cadence.
