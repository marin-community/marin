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

### 2026-08-11 23:01 UTC - Routine health check

- Root health: Both roots are running with zero failures and zero preemptions.
- East progress: Six source jobs succeeded. `eai-taxonomy-code-w-dclm_39527a3d` reached 950 of 5,872 output shards with 32 live workers and no dead workers.
- RNO progress: Two source jobs succeeded. `common_corpus-english_0e1cf2c4` reached 1,782 of 8,018 output shards with 32 live workers and no dead workers.
- Retry check: East stayed at 24 one-time retries. RNO increased from 131 to 154 one-time retries. No shard has retried twice.
- Decision: Keep both roots running because output continues and retry depth is unchanged.
- Next action: Continue the 15-minute check cadence.

### 2026-08-11 23:25 UTC - Routine health check

- Root health: Both roots are running with zero failures and zero preemptions.
- East progress: Six source jobs succeeded. `eai-taxonomy-code-w-dclm_39527a3d` reached 1,295 of 5,872 output shards with 32 live workers and no dead workers.
- RNO progress: Two source jobs succeeded. `common_corpus-english_0e1cf2c4` reached 2,188 of 8,018 output shards with 32 live workers and no dead workers.
- Retry check: East stayed at 24 one-time retries. RNO stayed at 154 one-time retries. No shard has retried twice.
- Next action: Continue the 15-minute check cadence.

### 2026-08-11 23:56 UTC - Routine health check

- Root health: Both roots are running with zero failures and zero preemptions.
- East progress: Six source jobs succeeded. `eai-taxonomy-code-w-dclm_39527a3d` reached 1,739 of 5,872 output shards with 32 live workers and no dead workers.
- RNO progress: Two source jobs succeeded. `common_corpus-english_0e1cf2c4` reached 2,664 of 8,018 output shards with 32 live workers and no dead workers.
- Retry check: East stayed at 24 one-time retries. RNO increased from 154 to 174 one-time retries. No shard has retried twice.
- Decision: Keep both roots running because output continues and retry depth is unchanged.
- Next action: Resume the 15-minute check cadence after the PR review pause.

### 2026-08-12 00:12 UTC - Routine health check

- Root health: Both roots are running with zero failures and zero preemptions.
- East progress: Six source jobs succeeded. `eai-taxonomy-code-w-dclm_39527a3d` reached 1,972 of 5,872 output shards with 32 live workers and no dead workers.
- RNO progress: Two source jobs succeeded. `common_corpus-english_0e1cf2c4` reached 2,904 of 8,018 output shards with 32 live workers and no dead workers.
- Retry check: East stayed at 24 one-time retries. RNO stayed at 174 one-time retries. No shard has retried twice.
- Next action: Continue the 15-minute check cadence.

### 2026-08-12 00:28 UTC - Routine health check

- Root health: Both roots are running with zero failures and zero preemptions.
- East progress: Six source jobs succeeded. `eai-taxonomy-code-w-dclm_39527a3d` reached 2,218 of 5,872 output shards with 32 live workers and no dead workers.
- RNO progress: Two source jobs succeeded. `common_corpus-english_0e1cf2c4` reached 3,153 of 8,018 output shards with 32 live workers and no dead workers.
- Retry check: East stayed at 24 one-time retries. RNO stayed at 174 one-time retries. No shard has retried twice.
- Next action: Continue the 15-minute check cadence.

### 2026-08-12 00:52 UTC - Routine health check

- Root health: Both roots are running with zero failures and zero preemptions.
- East progress: Six source jobs succeeded. `eai-taxonomy-code-w-dclm_39527a3d` reached 2,581 of 5,872 output shards with 32 live workers and no dead workers.
- RNO progress: Two source jobs succeeded. `common_corpus-english_0e1cf2c4` reached 3,525 of 8,018 output shards with 32 live workers and no dead workers.
- Retry check: East stayed at 24 one-time retries. RNO stayed at 174 one-time retries. No shard has retried twice.
- Next action: Continue the 15-minute check cadence.

### 2026-08-12 01:16 UTC - TEI worker utilization and RNO service loss

- East service capacity: All 96 TEI workers are running. Three GPU samples across all workers show 93.4% mean utilization and 98.3% median utilization. No worker stayed idle across the sample window.
- RNO service capacity: 82 of 96 TEI workers are running. Three GPU samples across the live workers show 96.7% mean utilization and 100% median utilization. No live worker stayed idle across the sample window.
- Failure evidence: The other 14 RNO TEI jobs became terminal after four to six cluster preemptions. Their last attempts could not read the staged model archive because the object was missing.
- Recovery: Restored the 1,208,135,680-byte model archive from a live East worker to its original East-region S3 path. This protects later TEI restarts from the missing-object failure.
- Source health: Both source jobs continue to run. No Parquet merge has started.
- Next action: Continue source monitoring and decide whether to replace the 14 terminal RNO services or keep the saturated 82-worker pool.

### 2026-08-12 01:27 UTC - RNO TEI service replay

- Source health: East reached 3,013 of 5,872 output shards. RNO reached 3,967 of 8,018 output shards. Each source has 32 of 32 Zephyr workers alive and zero dead workers.
- Recovery method: Read each terminal TEI job request from the `cw-rno2a` controller and replayed the same request with the same job ID.
- Scope: Replayed TEI indices 067, 070-073, 075, 078-082, 088, 090, and 091. The live 82-worker pool stayed in service.
- Scheduler state: All 96 TEI jobs are in the running job state. Kubernetes has 82 running TEI pods and 14 pending pods with `SchedulingGated`.
- Archive state: The restored model archive remains at the original S3 path for these worker starts.
- Next action: Wait for RNO interactive capacity. Validate one new endpoint and then validate the full 96-worker pool.

### 2026-08-12 01:32 UTC - Longer model staging lifetime

- Prior evidence: A review of the first Harrier pipeline identified the one-day archive lifetime as unsafe for longer embedding runs: https://github.com/marin-community/marin/pull/7977#issuecomment-5186200794.
- Failure match: The RNO TEI restart failure matched that prediction. The current backfill has a 3-to-5-day estimate.
- Change: Commit `b48a81225` keeps future Harrier model archives for 14 days instead of one day.
- Validation: All nine focused Harrier pipeline tests passed. Ruff, Black, Pyrefly, and the other changed-file checks passed.
- Current-run caveat: The v3 workers still use the original one-day path. The monitor must keep that restored object available until v3 completes.
- Next action: Continue the RNO capacity wait and source monitoring.

### 2026-08-12 01:46 UTC - Routine health check

- Root health: Both roots are running with zero failures and zero preemptions.
- East progress: Six source jobs succeeded. `eai-taxonomy-code-w-dclm_39527a3d` reached 3,419 of 5,872 output shards with 32 live workers and no dead workers.
- RNO progress: Two source jobs succeeded. `common_corpus-english_0e1cf2c4` reached 4,372 of 8,018 output shards with 32 live workers and no dead workers.
- Service capacity: East has 96 running TEI pods. RNO has 82 running TEI pods and 14 replacement pods waiting for interactive capacity.
- Error check: No new transport, rate-limit, memory, dead-worker, or missing-file error matched the last 15 minutes of coordinator logs.
- Next action: Continue the 15-minute check cadence and validate RNO replacements when capacity becomes available.

### 2026-08-12 02:01 UTC - Routine health check

- Root health: Both roots are running with zero failures and zero preemptions.
- East progress: Six source jobs succeeded. `eai-taxonomy-code-w-dclm_39527a3d` reached 3,659 of 5,872 output shards with 32 live workers and no dead workers.
- RNO progress: Two source jobs succeeded. `common_corpus-english_0e1cf2c4` reached 4,612 of 8,018 output shards with 32 live workers and no dead workers.
- Service capacity: East has 96 running TEI pods. RNO has 82 running TEI pods and 14 replacement pods waiting for interactive capacity.
- Archive check: The restored model archive still exists at the original v3 S3 path.
- Error check: No new transport, rate-limit, memory, dead-worker, or missing-file error matched the last 15 minutes of coordinator logs.
- Next action: Continue the 15-minute check cadence and validate RNO replacements when capacity becomes available.

### 2026-08-12 02:17 UTC - Routine health check

- Root health: Both roots are running with zero failures and zero preemptions.
- East progress: Six source jobs succeeded. `eai-taxonomy-code-w-dclm_39527a3d` reached 3,896 of 5,872 output shards with 32 live workers and no dead workers.
- RNO progress: Two source jobs succeeded. `common_corpus-english_0e1cf2c4` reached 4,848 of 8,018 output shards with 32 live workers and no dead workers.
- Service capacity: East has 96 running TEI pods. RNO has 82 running TEI pods and 14 replacement pods waiting for interactive capacity.
- Error check: No new transport, rate-limit, memory, dead-worker, or missing-file error matched the last 15 minutes of coordinator logs.
- Next action: Continue the 15-minute check cadence and validate RNO replacements when capacity becomes available.

### 2026-08-12 02:38 UTC - Embedding throughput profile

- Current health: Both roots have zero failures and zero preemptions. East reached 4,216 of 5,872 output shards. RNO reached 5,171 of 8,018 output shards. Each stage has 32 live workers and no dead workers.
- Thread profiles: Worker 0 in each region had all 16 embedding request threads blocked in `http.client.getresponse()` from `TEIClient.embed()`. The active shard thread waited for those futures. No active Parquet or S3 call was in that path.
- CPU profiles: The 10-second East capture contained 2.05 CPU-seconds across all threads. The RNO capture contained 0.20 CPU-seconds. Minor East samples were in Arrow conversion. CPU work is not the limit.
- GPU samples: A 20-second East TEI sample had 86.6% mean utilization, 98% median utilization, and 565 W mean power. RNO had 99.6% mean utilization, 100% median utilization, and 670 W mean power.
- East TEI timing: Across 1,325 requests, mean total time was 451 ms. Mean queue time was 180 ms. Mean inference time was 244 ms. Queue plus inference was 93.9% of total time.
- RNO TEI timing: Across 389 requests, mean total time was 2.006 seconds. Mean queue time was 1.175 seconds. Mean inference time was 692 ms. Queue plus inference was 93.1% of total time.
- Result: Both Zephyr stages wait for GPU embedding responses. RNO has the larger TEI queue because 82 live GPUs are saturated while 14 replacement pods wait for interactive capacity.
- Captures: `scratch/harrier-profile-20260812-0232-east-zephyr-threads.txt`, `scratch/harrier-profile-20260812-0232-rno-zephyr-threads.txt`, `scratch/harrier-profile-20260812-0236-east-zephyr.speedscope.json`, and `scratch/harrier-profile-20260812-0236-rno-zephyr.speedscope.json`.
- Next action: Keep the live configuration. Continue to wait for the 14 RNO replacement workers and monitor source progress.

### 2026-08-12 02:54 UTC - Routine health check

- Root health: Both roots are running with zero failures and zero preemptions.
- East progress: Six source jobs succeeded. `eai-taxonomy-code-w-dclm_39527a3d` reached 4,466 of 5,872 output shards with 32 live workers and no dead workers.
- RNO progress: Two source jobs succeeded. `common_corpus-english_0e1cf2c4` reached 5,424 of 8,018 output shards with 32 live workers and no dead workers.
- Service capacity: East has 96 running TEI pods. RNO has 82 running TEI pods and 14 replacement pods waiting for interactive capacity.
- Error check: No new transport, rate-limit, memory, dead-worker, or missing-file error matched the last 15 minutes of coordinator logs.
- Next action: Continue the 15-minute check cadence and validate RNO replacements when capacity becomes available.

### 2026-08-12 03:11 UTC - Routine health check

- Root health: Both roots are running with zero failures and zero preemptions.
- East progress: Six source jobs succeeded. `eai-taxonomy-code-w-dclm_39527a3d` reached 4,716 of 5,872 output shards with 32 live workers and no dead workers.
- RNO progress: Two source jobs succeeded. `common_corpus-english_0e1cf2c4` reached 5,684 of 8,018 output shards with 32 live workers and no dead workers.
- Service capacity: East has 96 running TEI pods. RNO has 82 running TEI pods and 14 replacement pods waiting for interactive capacity.
- Error check: No new transport, rate-limit, memory, dead-worker, or missing-file error matched the last 15 minutes of coordinator logs.
- Next action: Continue the 15-minute check cadence and validate RNO replacements when capacity becomes available.

### 2026-08-12 03:27 UTC - Routine health check

- Root health: Both roots are running with zero failures and zero preemptions.
- East progress: Six source jobs succeeded. `eai-taxonomy-code-w-dclm_39527a3d` reached 4,970 of 5,872 output shards with 32 live workers and no dead workers.
- RNO progress: Two source jobs succeeded. `common_corpus-english_0e1cf2c4` reached 5,945 of 8,018 output shards with 32 live workers and no dead workers.
- Service capacity: East has 96 running TEI pods. RNO has 82 running TEI pods and 14 replacement pods waiting for interactive capacity.
- Error check: No new transport, rate-limit, memory, dead-worker, or missing-file error matched the last 15 minutes of coordinator logs.
- Next action: Continue the 15-minute check cadence and validate RNO replacements when capacity becomes available.

### 2026-08-12 03:43 UTC - Routine health check

- Root health: Both roots are running with zero failures and zero preemptions.
- East progress: Six source jobs succeeded. `eai-taxonomy-code-w-dclm_39527a3d` reached 5,219 of 5,872 output shards with 32 live workers and no dead workers.
- RNO progress: Two source jobs succeeded. `common_corpus-english_0e1cf2c4` reached 6,194 of 8,018 output shards with 32 live workers and no dead workers.
- Service capacity: East has 96 running TEI pods. RNO has 82 running TEI pods and 14 replacement pods waiting for interactive capacity.
- Error check: No new transport, rate-limit, memory, dead-worker, or missing-file error matched the last 15 minutes of coordinator logs.
- Next action: Continue the 15-minute check cadence and validate RNO replacements when capacity becomes available.

### 2026-08-12 05:30 UTC - East source transition

- Root health: Both roots are running with zero failures and zero preemptions.
- Completed artifacts: East has 11 completed source artifacts. RNO has two completed source artifacts. The full run has 13 of 292 completed source artifacts.
- East result: `eai-taxonomy-code-w-dclm_39527a3d`, `ghalogs-public_414133b4`, `gpt-oss-rollouts_733e1cb8`, `identity-data-content_49ce1d92`, and `massive_function_calling_e02ff837` succeeded.
- East progress: `nemotron_code_v1-content_085c2c96` got to 636 of 2,341 output shards with 32 live workers and no dead workers.
- RNO progress: `common_corpus-english_0e1cf2c4` got to 7,859 of 8,018 output shards with 32 live workers and no dead workers.
- Service capacity: East has 96 running TEI pods. RNO has 82 running TEI pods. Fourteen replacement jobs still wait for interactive capacity.
- Archive check: The restored model archive still exists at the original v3 S3 path.
- Error check: No new transport, rate-limit, memory, dead-worker, or missing-file error matched the last 15 minutes of coordinator logs.
- Next action: Verify the RNO source transition, continue the 15-minute check cadence, and validate replacement services when RNO capacity becomes available.

### 2026-08-12 05:43 UTC - RNO source transition

- Root health: Both roots are running with zero failures and zero preemptions.
- Completed artifacts: East has 11 completed source artifacts. RNO has three completed source artifacts. The full run has 14 of 292 completed source artifacts.
- RNO result: `common_corpus-english_0e1cf2c4` succeeded after 8 hours and 55 minutes. It processed 8,017 output shards and skipped one existing output shard.
- Retry result: 174 RNO shards retried once. No shard needed a second retry.
- RNO progress: `davinci-dev-ctx-native_f5bd4268` started. It has 298 output shards, with 32 in flight, 32 live workers, and no dead workers.
- East progress: `nemotron_code_v1-content_085c2c96` got to 856 of 2,341 output shards with 32 live workers and no dead workers.
- Service capacity: East has 96 running TEI pods. RNO has 82 running TEI pods. Fourteen replacement jobs still wait for interactive capacity.
- Error check: No new transport, rate-limit, memory, dead-worker, or missing-file error matched the active coordinator logs.
- Next action: Continue the 15-minute check cadence and validate replacement services when RNO capacity becomes available.

### 2026-08-12 06:02 UTC - RNO completes ctx-native

- Root health: Both roots are running with zero failures and zero preemptions.
- Completed artifacts: East has 11 completed source artifacts. RNO has four completed source artifacts. The full run has 15 of 292 completed source artifacts.
- RNO result: `davinci-dev-ctx-native_f5bd4268` succeeded in 17 minutes. It processed all 298 output shards and 1,739,770 documents.
- RNO progress: `dna-functional-regions_d1bafea8` started. It has 38 output shards, with 32 in flight, 32 live workers, and no dead workers.
- East progress: `nemotron_code_v1-content_085c2c96` got to 1,100 of 2,341 output shards with 32 live workers and no dead workers.
- Service capacity: East has 96 running TEI pods. RNO has 82 running TEI pods. Fourteen replacement jobs still wait for interactive capacity.
- Archive status: The model archive was last confirmed at the original v3 S3 path at 05:30 UTC.
- Error check: No new transport, rate-limit, memory, dead-worker, or missing-file error matched the active coordinator logs.
- Next action: Continue the 15-minute check cadence and validate replacement services when RNO capacity becomes available.

### 2026-08-12 06:22 UTC - RNO completes functional regions

- Root health: Both roots are running with zero failures and zero preemptions.
- Completed artifacts: East has 11 completed source artifacts. RNO has five completed source artifacts. The full run has 16 of 292 completed source artifacts.
- RNO result: `dna-functional-regions_d1bafea8` succeeded. It processed all 38 output shards and 56,432,718 documents.
- RNO progress: `dolma_code_prose_013b53af` started. It has 224 output shards, with 32 in flight, 32 live workers, and no dead workers.
- East progress: `nemotron_code_v1-content_085c2c96` got to 1,468 of 2,341 output shards with 32 live workers and no dead workers.
- Service capacity: East has 96 running TEI pods. RNO has 82 running TEI pods. Fourteen replacement jobs still wait for interactive capacity.
- Error check: No new transport, rate-limit, memory, dead-worker, or missing-file error matched the active coordinator logs.
- Next action: Continue the 15-minute check cadence and validate replacement services when RNO capacity becomes available.

### 2026-08-12 06:47 UTC - RNO completes dolma code prose

- Root health: Both roots are running with zero failures and zero preemptions.
- Completed artifacts: East has 11 completed source artifacts. RNO has six completed source artifacts. The full run has 17 of 292 completed source artifacts.
- RNO result: `dolma_code_prose_013b53af` succeeded. It processed all 224 output shards and 12,827,072 documents.
- RNO progress: `finetranslations_6ce00a47` started. Its first join stage got to 4,009 of 25,962 tasks with 32 live workers and no dead workers.
- East progress: `nemotron_code_v1-content_085c2c96` got to 1,862 of 2,341 output shards with 32 live workers and no dead workers.
- Error check: No new transport, rate-limit, memory, dead-worker, or missing-file error matched the active coordinator logs.
- Next action: Monitor the `finetranslations` join stages and verify the transition to embedding.

### 2026-08-12 07:23 UTC - East completes three sources

- Root health: Both roots are running with zero failures and zero preemptions.
- Completed artifacts: East has 14 completed source artifacts. RNO has six completed source artifacts. The full run has 20 of 292 completed source artifacts.
- East result: `nemotron_code_v1-content_085c2c96` succeeded with all 2,341 output shards and 160,750,583 documents.
- East result: `nemotron-terminal_08c05d68` succeeded with all 30 output shards and 92,616 documents.
- East result: `numinamath-1.5_a479d052` succeeded with both output shards and 50,999 documents.
- East progress: `sec-edgar_e59004ca` started and got to 32 of 923 output shards with 32 live workers and no dead workers.
- RNO progress: `finetranslations_6ce00a47` completed its join work and got to 1,144 of 25,962 output shards in the embedding stage. It has 32 live workers and no dead workers.
- Service capacity: East has 96 running TEI jobs. RNO has 82 running TEI jobs. Fourteen RNO TEI jobs remain failed after replay.
- Error check: No new transport, rate-limit, memory, dead-worker, or missing-file error matched the last 15 minutes of root logs.
- Next action: Continue the 15-minute check cadence and monitor the long `finetranslations` source.

### 2026-08-12 07:50 UTC - East completes sec-edgar and superior reasoning

- Root health: Both roots are running with zero failures and zero preemptions.
- Completed artifacts: East has 16 completed source artifacts. RNO has six completed source artifacts. The full run has 22 of 292 completed source artifacts.
- East result: `sec-edgar_e59004ca` succeeded with all 923 output shards and 7,239,019 documents.
- East result: `superior-reasoning_13a85562` succeeded with all 42 output shards and 43,227 documents.
- East progress: `swe-rebench-contree_fc148d90` started. It has 318 output shards, with 32 in flight, 32 live workers, and no dead workers.
- RNO progress: `finetranslations_6ce00a47` got to 2,871 of 25,962 output shards with 32 live workers and no dead workers.
- Service capacity: East has 96 running TEI jobs. RNO has 82 running TEI jobs.
- Error check: No new transport, rate-limit, memory, dead-worker, or missing-file error matched the last 15 minutes of root logs.
- Next action: Continue the 15-minute check cadence and monitor the long `finetranslations` source.

### 2026-08-12 08:06 UTC - East completes SWE rebench contree

- Root health: Both roots are running with zero failures and zero preemptions.
- Completed artifacts: East has 17 completed source artifacts. RNO has six completed source artifacts. The full run has 23 of 292 completed source artifacts.
- East result: `swe-rebench-contree_fc148d90` succeeded with all 318 output shards and 1,591,909 documents.
- East progress: `swe-zero-12m_243316c3` started. It has 470 output shards, with 32 in flight, 32 live workers, and no dead workers.
- RNO progress: `finetranslations_6ce00a47` got to 4,245 of 25,962 output shards with 32 live workers and no dead workers.
- Error check: No new transport, rate-limit, memory, dead-worker, or missing-file error matched the last 15 minutes of root logs.
- Next action: Continue the 15-minute check cadence and monitor both active sources.

### 2026-08-12 09:47 UTC - East completes three sources

- Root health: Both roots are running with zero failures and zero preemptions.
- Completed artifacts: East has 20 completed source artifacts. RNO has six completed source artifacts. The full run has 26 of 292 completed source artifacts.
- East result: `swe-zero-12m_243316c3` succeeded with all 470 output shards and 11,549,124 documents.
- East result: `starcoder2-documentation_871525e3` succeeded with all six output shards and 29,667 documents.
- East result: `starcoder2-ir_low_resource_c2f6ee37` succeeded with all nine output shards and 259,025 documents.
- East progress: `starcoder2-ir_rust_f431fcdb` started. It has four output shards, with all four workers alive and no dead workers.
- RNO progress: `finetranslations_6ce00a47` got to 11,192 of 25,962 output shards with 32 live workers and no dead workers.
- Service capacity: East has 96 running TEI jobs. RNO has 82 running TEI jobs.
- Shutdown diagnostic: Five late `swe-zero-12m` replacement workers tried to register endpoints after Iris had made their worker-group tasks terminal. Iris rejected the registrations. The source and coordinator succeeded, and East has no failed descendants. No recovery was necessary.
- Error check: No error matched the active East or RNO source logs after the source transitions.
- Next action: Continue the 15-minute check cadence and monitor both active sources.

### 2026-08-12 09:56 UTC - East completes StarCoder Rust

- Root health: Both roots are running with zero failures and zero preemptions.
- Completed artifacts: East has 21 completed source artifacts. RNO has six completed source artifacts. The full run has 27 of 292 completed source artifacts.
- East result: `starcoder2-ir_rust_f431fcdb` succeeded with all four output shards and 27,118 documents.
- East progress: `biocollection-free_text_stream_29c0f030` started. It has 59 output shards, with 32 in flight, 32 live workers, and no dead workers.
- RNO progress: `finetranslations_6ce00a47` got to 12,015 of 25,962 output shards with 32 live workers and no dead workers.
- Service capacity: East has 96 running TEI jobs. RNO has 82 running TEI jobs.
- Error check: No new transport, rate-limit, memory, dead-worker, or missing-file error matched the last 15 minutes of root logs.
- Next action: Continue the 15-minute check cadence and monitor the first large `biocollection-free_text_stream` shards.

### 2026-08-12 10:24 UTC - East completes BioCollection and arXiv abstracts

- Root health: Both roots are running with zero failures and zero preemptions.
- Completed artifacts: East has 23 completed source artifacts. RNO has six completed source artifacts. The full run has 29 of 292 completed source artifacts.
- East result: `biocollection-free_text_stream_29c0f030` succeeded with all 59 output shards and 34,417,546 documents.
- East result: `cp-arxiv_abstracts_3d8157c9` succeeded with all four output shards and 744,059 documents.
- East progress: `cp-caselaw_8eb43b0b` started. It has 89 output shards, with 32 in flight, 32 live workers, and no dead workers.
- RNO progress: `finetranslations_6ce00a47` got to 13,832 of 25,962 output shards with 32 live workers and no dead workers.
- Error check: No new transport, rate-limit, memory, dead-worker, or missing-file error matched the last 15 minutes of root logs.
- Next action: Continue the 15-minute check cadence and monitor both active sources.

### 2026-08-12 10:40 UTC - East completes case law and DOAB

- Root health: Both roots are running with zero failures and zero preemptions.
- Completed artifacts: East has 25 completed source artifacts. RNO has six completed source artifacts. The full run has 31 of 292 completed source artifacts.
- East result: `cp-caselaw_8eb43b0b` succeeded with all 89 output shards and 4,261,234 documents.
- East result: `cp-doab_b9b530ea` succeeded with all 12 output shards and 81,119 documents.
- East progress: `cp-github_archive_540aca13` started. It has 56 output shards, with 32 in flight, 32 live workers, and no dead workers.
- RNO progress: `finetranslations_6ce00a47` got to 15,174 of 25,962 output shards with 32 live workers and no dead workers.
- Service capacity: East has 96 running TEI jobs. RNO has 82 running TEI jobs.
- Error check: No new transport, rate-limit, memory, dead-worker, or missing-file error matched the last 15 minutes of root logs.
- Next action: Continue the 15-minute check cadence and monitor both active sources.

### 2026-08-12 10:48 UTC - East completes four Common Pile sources

- Root health: Both roots are running with zero failures and zero preemptions.
- Completed artifacts: East has 29 completed source artifacts. RNO has six completed source artifacts. The full run has 35 of 292 completed source artifacts.
- East result: `cp-github_archive_540aca13` succeeded with all 56 output shards and 1,598,567 documents.
- East result: `cp-libretexts_9aa4c8ae` succeeded with its one output shard and 26,947 documents.
- East result: `cp-oercommons_26367272` succeeded with its one output shard and 1,766 documents.
- East result: `cp-peps_c79187ca` succeeded with its one output shard and 593 documents.
- East progress: `cp-pressbooks_4e5ce25d` started. It has one output shard and its worker is alive.
- RNO progress: `finetranslations_6ce00a47` got to 15,673 of 25,962 output shards with 32 live workers and no dead workers.
- Error check: No new transport, rate-limit, memory, dead-worker, or missing-file error matched the active source logs.
- Next action: Continue the 15-minute check cadence and monitor both active sources.

### 2026-08-12 10:56 UTC - East completes four more Common Pile sources

- Root health: Both roots are running with zero failures and zero preemptions.
- Completed artifacts: East has 33 completed source artifacts. RNO has six completed source artifacts. The full run has 39 of 292 completed source artifacts.
- East result: `cp-pressbooks_4e5ce25d` succeeded with its one output shard and 32,931 documents.
- East result: `cp-public_domain_review_9221823d` succeeded with its one output shard and 1,038 documents.
- East result: `cp-regulations_99079523` succeeded with all four output shards and 150,779 documents.
- East result: `cp-ubuntu_irc_e64c4be0` succeeded with all six output shards and 27,525 documents.
- East progress: `cp-usgpo_1e7881f3` started. It has 36 output shards, with 32 in flight, 32 live workers, and no dead workers.
- RNO progress: `finetranslations_6ce00a47` got to approximately 16,333 of 25,962 output shards with 32 live workers and no dead workers.
- Service capacity: East has 96 running TEI jobs. RNO has 82 running TEI jobs.
- Error check: No new transport, rate-limit, memory, dead-worker, or missing-file error matched the last 15 minutes of root logs.
- Next action: Continue the 15-minute check cadence and monitor both active sources.

### 2026-08-12 11:03 UTC - East completes USGPO

- Root health: Both roots are running with zero failures and zero preemptions.
- Completed artifacts: East has 34 completed source artifacts. RNO has six completed source artifacts. The full run has 40 of 292 completed source artifacts.
- East result: `cp-usgpo_1e7881f3` succeeded with all 36 output shards and 1,291,904 documents.
- East progress: `cp-wikiteam_7721bc04` started. It has 13 output shards, with all 13 workers alive and no dead workers.
- RNO progress: `finetranslations_6ce00a47` got to 16,781 of 25,962 output shards with 32 live workers and no dead workers.
- Error check: No new transport, rate-limit, memory, dead-worker, or missing-file error matched the active source logs.
- Next action: Continue the 15-minute check cadence and monitor both active sources.

### 2026-08-12 11:11 UTC - East completes WikiTeam

- Root health: Both roots are running with zero failures and zero preemptions.
- Completed artifacts: East has 35 completed source artifacts. RNO has six completed source artifacts. The full run has 41 of 292 completed source artifacts.
- East result: `cp-wikiteam_7721bc04` succeeded with all 13 output shards and 3,781,780 documents.
- East progress: `finepdfs_cf4aed04` started. It has 9,244 output shards, with 32 in flight, 32 live workers, and no dead workers.
- RNO progress: `finetranslations_6ce00a47` got to approximately 17,439 of 25,962 output shards with 32 live workers and no dead workers.
- Service capacity: East has 96 running TEI jobs. RNO has 82 running TEI jobs.
- Error check: No new transport, rate-limit, memory, dead-worker, or missing-file error matched the last 15 minutes of root logs.
- Next action: Continue the 15-minute check cadence and monitor the first FinePDFs shard completions.

### 2026-08-12 11:26 UTC - Long-source health check

- Root health: Both roots are running with zero failures and zero preemptions.
- Completed artifacts: East has 35 completed source artifacts. RNO has six completed source artifacts. The full run has 41 of 292 completed source artifacts.
- East progress: `finepdfs_cf4aed04` got to 475 of 9,244 output shards with 32 live workers and no dead workers.
- RNO progress: `finetranslations_6ce00a47` got to 18,527 of 25,962 output shards with 32 live workers and no dead workers.
- Service capacity: East has 96 running TEI jobs. RNO has 82 running TEI jobs.
- Error check: No new transport, rate-limit, memory, dead-worker, or missing-file error matched the last 15 minutes of root logs.
- Next action: Continue the 15-minute check cadence and monitor both long-running sources.

### 2026-08-12 11:42 UTC - Long-source health check

- Root health: Both roots are running with zero failures and zero preemptions.
- Completed artifacts: East has 35 completed source artifacts. RNO has six completed source artifacts. The full run has 41 of 292 completed source artifacts.
- East progress: `finepdfs_cf4aed04` got to 960 of 9,244 output shards with 32 live workers and no dead workers.
- RNO progress: `finetranslations_6ce00a47` got to 19,606 of 25,962 output shards with 32 live workers and no dead workers.
- Service capacity: East has 96 running TEI jobs. RNO has 82 running TEI jobs.
- Error check: No new transport, rate-limit, memory, dead-worker, or missing-file error matched the last 15 minutes of root logs.
- Next action: Continue the 15-minute check cadence and monitor both long-running sources.

### 2026-08-12 11:57 UTC - Long-source health check

- Root health: Both roots are running with zero failures and zero preemptions.
- Completed artifacts: East has 35 completed source artifacts. RNO has six completed source artifacts. The full run has 41 of 292 completed source artifacts.
- East progress: `finepdfs_cf4aed04` got to 1,437 of 9,244 output shards with 32 live workers and no dead workers.
- RNO progress: `finetranslations_6ce00a47` got to 20,717 of 25,962 output shards with 32 live workers and no dead workers.
- Service capacity: East has 96 running TEI jobs. RNO has 82 running TEI jobs.
- Error check: No new transport, rate-limit, memory, dead-worker, or missing-file error matched the last 15 minutes of root logs.
- Next action: Continue the 15-minute check cadence and monitor both long-running sources.

### 2026-08-12 12:12 UTC - Long-source health check

- Root health: Both roots are running with zero failures and zero preemptions.
- Completed artifacts: East has 35 completed source artifacts. RNO has six completed source artifacts. The full run has 41 of 292 completed source artifacts.
- East progress: `finepdfs_cf4aed04` got to 1,918 of 9,244 output shards with 32 live workers and no dead workers.
- RNO progress: `finetranslations_6ce00a47` got to 21,842 of 25,962 output shards with 32 live workers and no dead workers.
- Service capacity: East has 96 running TEI jobs. RNO has 82 running TEI jobs.
- Error check: No new transport, rate-limit, memory, dead-worker, or missing-file error matched the last 15 minutes of root logs.
- Next action: Continue the 15-minute check cadence and monitor both long-running sources.

### 2026-08-12 12:29 UTC - RNO transport event recovers

- Root health: Both roots are running with zero failures and zero preemptions.
- Completed artifacts: East has 35 completed source artifacts. RNO has six completed source artifacts. The full run has 41 of 292 completed source artifacts.
- East progress: `finepdfs_cf4aed04` got to 2,428 of 9,244 output shards with 32 live workers and no dead workers.
- RNO progress: `finetranslations_6ce00a47` got to 22,383 of 25,962 output shards with 32 live workers and no dead workers.
- Transport diagnostic: From 12:17:37 through 12:18:12, 25 shard attempts on 24 RNO workers exhausted TEI request retries with `RemoteDisconnected`. Zephyr requeued each shard at task error one of three. No shard reached task error two or three.
- Recovery evidence: All 32 Zephyr workers remain running, no new TEI job failed, and RNO resumed normal shard progress. The 25 failed shards remain queued for later retry. No manual recovery was necessary.
- Service capacity: East has 96 running TEI jobs. RNO has 82 running TEI jobs.
- Next action: Continue the 15-minute check cadence. Escalate if transport errors repeat, a shard reaches task error two, or worker health changes.

### 2026-08-12 12:45 UTC - Long-source health check

- Root health: Both roots are running with zero failures and zero preemptions.
- Completed artifacts: East has 35 completed source artifacts. RNO has six completed source artifacts. The full run has 41 of 292 completed source artifacts.
- East progress: `finepdfs_cf4aed04` got to 2,975 of 9,244 output shards with 32 live workers and no dead workers.
- RNO progress: `finetranslations_6ce00a47` got to 23,511 of 25,962 output shards with 32 live workers and no dead workers.
- Recovery evidence: No new transport error, task error two or three, dead worker, or other selected error matched either active source log during the last 15 minutes.
- Service capacity: East has 96 running TEI jobs. RNO has 82 running TEI jobs.
- Next action: Continue the 15-minute check cadence. Monitor the expected FineTranslations completion and the next RNO source start.

### 2026-08-12 13:01 UTC - Long-source health check

- Root health: Both roots are running with zero failures and zero preemptions.
- Completed artifacts: East has 35 completed source artifacts. RNO has six completed source artifacts. The full run has 41 of 292 completed source artifacts.
- East progress: `finepdfs_cf4aed04` got to 3,459 of 9,244 output shards with 32 live workers and no dead workers.
- RNO progress: `finetranslations_6ce00a47` got to 24,549 of 25,962 output shards with 32 live workers and no dead workers.
- Error check: No new transport, retry-two, rate-limit, memory, dead-worker, or missing-file error matched either active source log during the last 15 minutes.
- Service capacity: East has 96 running TEI jobs. RNO has 82 running TEI jobs.
- Next action: Continue the 15-minute check cadence. Monitor the expected FineTranslations completion near 13:22 UTC and the next RNO source start.

### 2026-08-12 13:29 UTC - RNO completes FineTranslations and GLM KernelGym

- Root health: Both roots are running with zero failures and zero preemptions.
- Completed artifacts: East has 35 completed source artifacts. RNO has eight completed source artifacts. The full run has 43 of 292 completed source artifacts.
- RNO result: `finetranslations_6ce00a47` succeeded with all 25,962 output shards and 111,482,016 duplicate documents. Final counters prove that all 25 shards from the 12:18 transport event completed after one shard retry.
- RNO result: `glm-5.2-kernelgym-rollouts_70abd4f9` succeeded with its one output shard and 456 duplicate documents.
- RNO progress: `hplt_v3_528b745e` started. Its 6,330-shard join stage got to 510 completed shards with 32 live workers and no dead workers.
- Service capacity: East has 96 running TEI jobs. RNO has 82 running TEI jobs.
- Next action: Continue the 15-minute check cadence and monitor HPLT through its join and embedding stages.

### 2026-08-12 13:45 UTC - Long-source health check

- Root health: Both roots are running with zero failures and zero preemptions.
- Completed artifacts: East has 35 completed source artifacts. RNO has eight completed source artifacts. The full run has 43 of 292 completed source artifacts.
- East progress: `finepdfs_cf4aed04` got to 4,848 of 9,244 output shards with 32 live workers and no dead workers.
- RNO progress: `hplt_v3_528b745e` completed its 6,330-shard join stage and got to 224 of 6,330 embedding shards with 32 live workers and no dead workers.
- Error check: No new transport, retry-two, rate-limit, memory, dead-worker, or missing-file error matched either active source log during the last 15 minutes.
- Service capacity: East has 96 running TEI jobs. RNO has 82 running TEI jobs.
- Next action: Continue the 15-minute check cadence and refine the FinePDFs and HPLT completion estimates.

### 2026-08-12 14:01 UTC - Long-source health check

- Root health: Both roots are running with zero failures and zero preemptions.
- Completed artifacts: East has 35 completed source artifacts. RNO has eight completed source artifacts. The full run has 43 of 292 completed source artifacts.
- East progress: `finepdfs_cf4aed04` got to 5,376 of 9,244 output shards with 32 live workers and no dead workers. Its recent rate gives a rough 16:03 UTC completion estimate.
- RNO progress: `hplt_v3_528b745e` got to 590 of 6,330 embedding shards with 32 live workers and no dead workers. Its recent rate gives a rough 18:23 UTC completion estimate.
- Error check: No new transport, retry-two, rate-limit, memory, dead-worker, or missing-file error matched either active source log during the last 15 minutes.
- Service capacity: East has 96 running TEI jobs. RNO has 82 running TEI jobs.
- Next action: Continue the 15-minute check cadence and monitor both long-running sources.

### 2026-08-12 14:17 UTC - Long-source health check

- Root health: Both roots are running with zero failures and zero preemptions.
- Completed artifacts: East has 35 completed source artifacts. RNO has eight completed source artifacts. The full run has 43 of 292 completed source artifacts.
- East progress: `finepdfs_cf4aed04` got to 5,860 of 9,244 output shards with 32 live workers and no dead workers. Its recent rate gives a rough 16:04 UTC completion estimate.
- RNO progress: `hplt_v3_528b745e` got to 937 of 6,330 embedding shards with 32 live workers and no dead workers. Its recent rate gives a rough 18:15 UTC completion estimate.
- Error check: No new transport, retry-two, rate-limit, memory, dead-worker, or missing-file error matched either active source log during the last 15 minutes.
- Service capacity: East has 96 running TEI jobs. RNO has 82 running TEI jobs.
- Next action: Continue the 15-minute check cadence and monitor both long-running sources.

### 2026-08-12 14:33 UTC - Long-source health check

- Root health: Both roots are running with zero failures and zero preemptions.
- Completed artifacts: East has 35 completed source artifacts. RNO has eight completed source artifacts. The full run has 43 of 292 completed source artifacts.
- East progress: `finepdfs_cf4aed04` got to 6,377 of 9,244 output shards with 32 live workers and no dead workers. Its recent rate gives a rough 16:03 UTC completion estimate.
- RNO progress: `hplt_v3_528b745e` got to 1,308 of 6,330 embedding shards with 32 live workers and no dead workers. Its recent rate gives a rough 18:14 UTC completion estimate.
- Error check: No new transport, retry-two, rate-limit, memory, dead-worker, or missing-file error matched either active source log during the last 15 minutes.
- Service capacity: East has 96 running TEI jobs. RNO has 82 running TEI jobs.
- Next action: Continue the 15-minute check cadence and monitor both long-running sources.

### 2026-08-12 14:49 UTC - Long-source health check

- Root health: Both roots are running with zero failures and zero preemptions.
- Completed artifacts: East has 35 completed source artifacts. RNO has eight completed source artifacts. The full run has 43 of 292 completed source artifacts.
- East progress: `finepdfs_cf4aed04` got to 6,891 of 9,244 output shards with 32 live workers and no dead workers. Its recent rate gives a rough 16:03 UTC completion estimate.
- RNO progress: `hplt_v3_528b745e` got to 1,668 of 6,330 embedding shards with 32 live workers and no dead workers. Its recent rate gives a rough 18:14 UTC completion estimate.
- Error check: No new transport, retry-two, rate-limit, memory, dead-worker, or missing-file error matched either active source log during the last 15 minutes.
- Service capacity: East has 96 running TEI jobs. RNO has 82 running TEI jobs.
- Next action: Continue the 15-minute check cadence and monitor both long-running sources.

### 2026-08-12 15:06 UTC - Long-source health check

- Root health: Both roots are running with zero failures and zero preemptions.
- Completed artifacts: East has 35 completed source artifacts. RNO has eight completed source artifacts. The full run has 43 of 292 completed source artifacts.
- East progress: `finepdfs_cf4aed04` got to 7,405 of 9,244 output shards with 32 live workers and no dead workers. Its recent rate gives a rough 16:04 UTC completion estimate.
- RNO progress: `hplt_v3_528b745e` got to 2,041 of 6,330 embedding shards with 32 live workers and no dead workers. Its recent rate gives a rough 18:14 UTC completion estimate.
- Error check: No new transport, retry-two, rate-limit, memory, dead-worker, or missing-file error matched either active source log during the last 15 minutes.
- Service capacity: East has 96 running TEI jobs. RNO has 82 running TEI jobs.
- Next action: Continue the 15-minute check cadence and monitor both long-running sources.

### 2026-08-12 15:22 UTC - Long-source health check

- Root health: Both roots are running with zero failures and zero preemptions.
- Completed artifacts: East has 35 completed source artifacts. RNO has eight completed source artifacts. The full run has 43 of 292 completed source artifacts.
- East progress: `finepdfs_cf4aed04` got to 7,917 of 9,244 output shards with 32 live workers and no dead workers. Its recent rate gives a rough 16:04 UTC completion estimate.
- RNO progress: `hplt_v3_528b745e` got to 2,406 of 6,330 embedding shards with 32 live workers and no dead workers. Its recent rate gives a rough 18:14 UTC completion estimate.
- Error check: No new transport, retry-two, rate-limit, memory, dead-worker, or missing-file error matched either active source log during the last 15 minutes.
- Service capacity: East has 96 running TEI jobs. RNO has 82 running TEI jobs.
- Next action: Continue the 15-minute check cadence and monitor both long-running sources.

### 2026-08-12 15:38 UTC - Long-source health check

- Root health: Both roots are running with zero failures and zero preemptions.
- Completed artifacts: East has 35 completed source artifacts. RNO has eight completed source artifacts. The full run has 43 of 292 completed source artifacts.
- East progress: `finepdfs_cf4aed04` got to 8,427 of 9,244 output shards with 32 live workers and no dead workers. Its recent rate gives a rough 16:04 UTC completion estimate.
- RNO progress: `hplt_v3_528b745e` got to 2,777 of 6,330 embedding shards with 32 live workers and no dead workers. Its recent rate gives a rough 18:14 UTC completion estimate.
- Error check: No new transport, retry-two, rate-limit, memory, dead-worker, or missing-file error matched either active source log during the last 15 minutes.
- Service capacity: East has 96 running TEI jobs. RNO has 82 running TEI jobs.
- Next action: Continue the 15-minute check cadence and monitor the expected FinePDFs completion.

### 2026-08-12 15:54 UTC - Long-source health check

- Root health: Both roots are running with zero failures and zero preemptions.
- Completed artifacts: East has 35 completed source artifacts. RNO has eight completed source artifacts. The full run has 43 of 292 completed source artifacts.
- East progress: `finepdfs_cf4aed04` got to 8,940 of 9,244 output shards with 32 live workers and no dead workers. Its recent rate gives a rough 16:04 UTC completion estimate.
- RNO progress: `hplt_v3_528b745e` got to 3,141 of 6,330 embedding shards with 32 live workers and no dead workers. Its recent rate gives a rough 18:14 UTC completion estimate.
- Error check: No new transport, retry-two, rate-limit, memory, dead-worker, or missing-file error matched either active source log during the last 15 minutes.
- Service capacity: East has 96 running TEI jobs. RNO has 82 running TEI jobs.
- Next action: Watch FinePDFs through completion, then return to the 15-minute check cadence.

### 2026-08-12 16:06 UTC - FinePDFs completed

- Root health: Both roots are running with zero failures and zero preemptions.
- Completed artifacts: East has 36 completed source artifacts. RNO has eight completed source artifacts. The full run has 44 of 292 completed source artifacts.
- East result: `finepdfs_cf4aed04` completed all 9,244 output shards with 63,732,737 duplicate documents. The source step succeeded, and its worker group succeeded.
- East progress: `finepdfs-ces_latn_b5488bfa` started with a running coordinator and worker group.
- RNO progress: `hplt_v3_528b745e` got to 3,401 of 6,330 embedding shards with 32 live workers and no dead workers. Its recent rate gives a rough 18:18 UTC completion estimate.
- Error check: No retry-two, rate-limit, memory, dead-worker, or missing-file error matched the active RNO source log during the last 15 minutes.
- Service capacity: East has 96 running TEI jobs. RNO has 82 running TEI jobs.
- Next action: Check FinePDFs CES Latin startup and continue the 15-minute check cadence.

### 2026-08-12 16:10 UTC - FinePDFs CES Latin completed

- Completed artifacts: East has 37 completed source artifacts. RNO has eight completed source artifacts. The full run has 45 of 292 completed source artifacts.
- East result: `finepdfs-ces_latn_b5488bfa` completed all 194 output shards with 596,382 duplicate documents. The source step succeeded.
- East progress: `finepdfs-deu_latn_142650e9` started with a running coordinator and worker group.
- Next action: Return to the 15-minute health-check cadence.

### 2026-08-12 16:22 UTC - FinePDFs German join tail cleared

- Root health: Both roots are running with zero failures and zero preemptions.
- Completed artifacts: East has 37 completed source artifacts. RNO has eight completed source artifacts. The full run has 45 of 292 completed source artifacts.
- East progress: `finepdfs-deu_latn_142650e9` completed its 1,170-shard join stage after two long-tail shards. One tail shard completed at 16:20:22 UTC, and the embedding stage then started. Embedding got to 73 of 1,170 shards with 32 live workers and no dead workers.
- RNO progress: `hplt_v3_528b745e` got to 3,774 of 6,330 embedding shards with 32 live workers and no dead workers. Its recent rate gives a rough 18:14 UTC completion estimate.
- Error check: No retry-two, rate-limit, memory, dead-worker, or missing-file error matched either active source log during the check window.
- Service capacity: East has 96 running TEI jobs. RNO has 82 running TEI jobs.
- Decision: Take no recovery action because the East join tail cleared and both sources continue to write output.
- Next action: Continue the 15-minute check cadence.

### 2026-08-12 16:39 UTC - FinePDFs German completed

- Root health: Both roots are running with zero failures and zero preemptions.
- Completed artifacts: East has 38 completed source artifacts. RNO has eight completed source artifacts. The full run has 46 of 292 completed source artifacts.
- East result: `finepdfs-deu_latn_142650e9` completed all 1,170 output shards with 4,907,656 duplicate documents. The source step succeeded.
- East progress: `finepdfs-hun_latn_5e462038` started its 191-shard join stage with live workers and no dead workers.
- RNO progress: `hplt_v3_528b745e` got to 4,074 of 6,330 embedding shards with 32 live workers and no dead workers at 16:35 UTC.
- Error check: No retry-two, rate-limit, memory, dead-worker, or missing-file error matched either active source log during the check window.
- Next action: Watch the short FinePDFs Hungarian source through completion, then continue the regular check cadence.

### 2026-08-12 16:43 UTC - FinePDFs Hungarian completed

- Completed artifacts: East has 39 completed source artifacts. RNO has eight completed source artifacts. The full run has 47 of 292 completed source artifacts.
- East result: `finepdfs-hun_latn_5e462038` completed all 191 output shards with 412,840 duplicate documents. The source step succeeded.
- East progress: `finepdfs-ita_latn_224a9bc3` started with a running coordinator and worker group.
- Next action: Check FinePDFs Italian startup and continue the regular health-check cadence.

### 2026-08-12 16:54 UTC - Routine health check

- Root health: Both roots are running with zero failures and zero preemptions.
- Completed artifacts: East has 39 completed source artifacts. RNO has eight completed source artifacts. The full run has 47 of 292 completed source artifacts.
- East progress: `finepdfs-ita_latn_224a9bc3` got to 391 of 618 embedding shards with 32 live workers and no dead workers. Its recent rate gives a rough 17:00 UTC completion estimate.
- RNO progress: `hplt_v3_528b745e` got to 4,489 of 6,330 embedding shards with 32 live workers and no dead workers. Its recent rate gives a rough 18:17 UTC completion estimate.
- Error check: No retry-two, rate-limit, memory, dead-worker, or missing-file error matched either active source log during the check window.
- Service capacity: East has 96 running TEI jobs. RNO has 82 running TEI jobs.
- Next action: Watch FinePDFs Italian through completion, then continue the regular check cadence.

### 2026-08-12 17:24 UTC - Three FinePDFs language sources completed

- Root health: Both roots are running with zero failures and zero preemptions.
- Completed artifacts: East has 42 completed source artifacts. RNO has eight completed source artifacts. The full run has 50 of 292 completed source artifacts.
- East results: `finepdfs-ita_latn_224a9bc3` completed 618 shards with 2,997,795 duplicate documents. `finepdfs-nld_latn_b55acf48` completed 295 shards with 1,073,748 duplicate documents. `finepdfs-por_latn_554504de` completed 564 shards with 2,562,012 duplicate documents. All three source steps succeeded.
- East progress: `finepdfs-rus_cyrl_88010315` started.
- RNO progress: `hplt_v3_528b745e` got to 5,107 of 6,330 embedding shards with 32 live workers and no dead workers. Its recent rate gives a rough 18:15 UTC completion estimate.
- Error check: No retry-two, rate-limit, memory, dead-worker, or missing-file error matched either active source log during the check window.
- Service capacity: East has 96 running TEI jobs. RNO has 82 running TEI jobs.
- Issue update: None. These are routine source completions, not a major run-state change.
- Next action: Check FinePDFs Russian startup and continue the regular health-check cadence.

### 2026-08-12 17:38 UTC - FinePDFs Russian completed

- Root health: Both roots are running with zero failures and zero preemptions.
- Completed artifacts: East has 43 completed source artifacts. RNO has eight completed source artifacts. The full run has 51 of 292 completed source artifacts.
- East result: `finepdfs-rus_cyrl_88010315` completed all 1,182 output shards with 2,424,394 duplicate documents. The source step succeeded.
- East progress: `finepdfs-swe_latn_3f198a8d` started its 148-shard join stage with 32 live workers and no dead workers.
- RNO progress: `hplt_v3_528b745e` got to 5,441 of 6,330 embedding shards with 32 live workers and no dead workers at 17:36 UTC. Its recent rate gives a rough 18:16 UTC completion estimate.
- Error check: No retry-two, rate-limit, memory, dead-worker, or missing-file error matched either active source log during the check window.
- Service capacity: East has 96 running TEI jobs. RNO has 82 running TEI jobs.
- Issue update: None. This is a routine source completion, not a major run-state change.
- Next action: Continue the regular health-check cadence and watch RNO HPLT through completion.

### 2026-08-12 17:45 UTC - FinePDFs Swedish and Ukrainian completed

- Root health: Both roots are running with zero failures and zero preemptions.
- Completed artifacts: East has 45 completed source artifacts. RNO has eight completed source artifacts. The full run has 53 of 292 completed source artifacts.
- East results: `finepdfs-swe_latn_3f198a8d` completed 148 shards with 526,543 duplicate documents. `finepdfs-ukr_cyrl_85c35eab` completed 214 shards with 394,995 duplicate documents. Both source steps succeeded.
- East progress: `nemotron_cc_v2-diverse_qa_016d1909` started with a running coordinator and worker group.
- RNO progress: `hplt_v3_528b745e` got to 5,659 of 6,330 embedding shards with 32 live workers and no dead workers. Its recent rate gives a rough 18:15 UTC completion estimate.
- Error check: No retry-two, rate-limit, memory, dead-worker, or missing-file error matched either active source log during the check window.
- Issue update: None. These are routine source completions, not a major run-state change.
- Next action: Continue the regular health-check cadence and watch RNO HPLT through completion.

### 2026-08-12 18:03 UTC - Routine health check

- Root health: Both roots are running with zero failures and zero preemptions.
- Completed artifacts: East has 45 completed source artifacts. RNO has eight completed source artifacts. The full run has 53 of 292 completed source artifacts.
- East progress: `nemotron_cc_v2-diverse_qa_016d1909` completed its 4,416-shard join and got to 192 of 4,416 embedding shards with 32 live workers and no dead workers.
- RNO progress: `hplt_v3_528b745e` got to 6,024 of 6,330 embedding shards with 32 live workers and no dead workers. Its recent rate gives a rough 18:15-18:20 UTC completion estimate.
- Error check: No retry-two, rate-limit, memory, dead-worker, or missing-file error matched either active source log during the check window.
- Service capacity: East has 96 running TEI jobs. RNO has 82 running TEI jobs.
- Issue update: None. There is no major run-state change.
- Next action: Watch RNO HPLT through completion and continue the regular East health checks.

### 2026-08-12 18:17 UTC - HPLT completed

- Root health: Both roots are running with zero failures and zero preemptions.
- Completed artifacts: East has 45 completed source artifacts. RNO has nine completed source artifacts. The full run has 54 of 292 completed source artifacts.
- RNO result: `hplt_v3_528b745e` completed all 6,330 output shards with 148,355,626 duplicate documents and 1,169,175,357,373 input text bytes. The source step and all 32 worker tasks succeeded.
- RNO progress: `institutional_books_63f8aca7` started its 1,832-shard join and passed shard 300.
- East progress: `nemotron_cc_v2-diverse_qa_016d1909` got to 416 of 4,416 embedding shards with 32 live workers and no dead workers.
- Issue update: None. This is a routine source completion, not a major run-state change.
- Next action: Check Institutional Books startup and continue the regular health-check cadence.

### 2026-08-12 18:36 UTC - Institutional Books and Molmo2 Cap completed

- Root health: Both roots are running with zero failures and zero preemptions.
- Completed artifacts: East has 45 completed source artifacts. RNO has 11 completed source artifacts. The full run has 56 of 292 completed source artifacts.
- RNO results: `institutional_books_63f8aca7` completed 1,832 shards with 603,657 duplicate documents. `molmo2-cap_a25dc16d` completed two shards with 11,287 duplicate documents. Both source steps succeeded.
- RNO progress: `nemotron_code_v2-content_459eda75` started its 517-shard join and passed shard 138.
- East progress: `nemotron_cc_v2-diverse_qa_016d1909` got to 714 of 4,416 embedding shards with 32 live workers and no dead workers.
- Error check: No selected error matched the new RNO source log.
- Issue update: None. These are routine source completions, not a major run-state change.
- Next action: Check Nemotron Code Content startup and continue the regular health-check cadence.

### 2026-08-12 18:50 UTC - Routine health check

- Root health: Both roots are running with zero failures and zero preemptions.
- Completed artifacts: East has 45 completed source artifacts. RNO has 11 completed source artifacts. The full run has 56 of 292 completed source artifacts.
- East progress: `nemotron_cc_v2-diverse_qa_016d1909` got to 940 of 4,416 embedding shards with 32 live workers and no dead workers.
- RNO progress: `nemotron_code_v2-content_459eda75` completed its 517-shard join and got to 163 of 517 embedding shards with 32 live workers and no dead workers. Uneven shard times give a rough 19:03 UTC completion estimate.
- Error check: No retry-two, rate-limit, memory, dead-worker, or missing-file error matched either active source log during the check window.
- Service capacity: East has 96 running TEI jobs. RNO has 82 running TEI jobs.
- Issue update: None. There is no major run-state change.
- Next action: Continue the regular health-check cadence and watch Nemotron Code Content through completion.

### 2026-08-12 19:17 UTC - Three RNO sources completed

- Root health: Both roots are running with zero failures and zero preemptions.
- Completed artifacts: East has 45 completed source artifacts. RNO has 14 completed source artifacts. The full run has 59 of 292 completed source artifacts.
- RNO results: `nemotron_code_v2-content_459eda75` completed 517 shards with 44,697,176 duplicate documents. `nsf_awards_0f5c5fa9` completed three shards with 62,181 duplicate documents. `numinamath-tir_1451911b` completed one shard with 470 duplicate documents. All three source steps succeeded.
- RNO progress: `stack-v3_6ac1a286` started its 12,818-shard join and passed shard 59.
- East progress: `nemotron_cc_v2-diverse_qa_016d1909` got to 1,360 of 4,416 embedding shards with 32 live workers and no dead workers.
- Issue update: None. These are routine source completions, not a major run-state change.
- Next action: Monitor both long sources and check Stack v3 through its join transition.

### 2026-08-12 19:28 UTC - Stack v3 started embedding

- Root health: Both roots are running with zero failures and zero preemptions.
- Completed artifacts: East has 45 completed source artifacts. RNO has 14 completed source artifacts. The full run has 59 of 292 completed source artifacts.
- RNO progress: `stack-v3_6ac1a286` completed its 12,818-shard join and started embedding at about 19:26 UTC. It got to 154 of 12,818 embedding shards with 32 live workers and no dead workers. The initial rate gives a rough 21:00 UTC completion estimate.
- East progress: `nemotron_cc_v2-diverse_qa_016d1909` got to 1,541 of 4,416 embedding shards with 32 live workers and no dead workers. Its recent rate gives a rough 22:30 UTC completion estimate.
- Error check: No retry-two, rate-limit, memory, dead-worker, or missing-file error matched either active source log during the check window.
- Service capacity: East has 96 running TEI jobs. RNO has 82 running TEI jobs.
- Issue update: None. The stage change is routine and does not change the run state.
- Next action: Measure the stable Stack v3 embedding rate and continue the regular health-check cadence.

### 2026-08-12 19:58 UTC - Routine health check

- Root health: Both roots are running with zero failures and zero preemptions.
- Completed artifacts: East has 45 completed source artifacts. RNO has 14 completed source artifacts. The full run has 59 of 292 completed source artifacts.
- East progress: `nemotron_cc_v2-diverse_qa_016d1909` got to 1,993 of 4,416 embedding shards with 32 live workers and no dead workers. Its stable rate gives a rough 22:30-22:45 UTC completion estimate.
- RNO progress: `stack-v3_6ac1a286` got to 4,237 of 12,818 embedding shards with 32 live workers and no dead workers. Its stable rate gives a rough 21:00 UTC completion estimate.
- Error check: No retry-two, rate-limit, memory, dead-worker, or missing-file error matched either active source log during the check window.
- Service capacity: East has 96 running TEI jobs. RNO has 82 running TEI jobs.
- Issue update: None. There is no major run-state change.
- Next action: Continue the regular health-check cadence and watch both long sources through completion.

### 2026-08-12 20:40 UTC - Three RNO TEI services recovered

- Root health: Both roots are running with zero failures and zero preemptions. Both active sources have 32 live workers and no dead workers.
- Completed artifacts: East has 45 completed source artifacts. RNO has 14 completed source artifacts. The full run has 59 of 292 completed source artifacts.
- RNO service event: TEI jobs 028, 030, and 032 reached their retry limit after three cluster preemptions. RNO capacity fell from 82 to 79 live TEI services.
- Recovery: The saved controller request did not include the callable work files. A replay without those files failed before TEI started. A live TEI service supplied the original runner and callable template. Each replacement then received its deterministic port pair. All three replacements passed their local health checks, and RNO returned to 82 live TEI services.
- RNO progress: `stack-v3_6ac1a286` got to 9,446 of 12,818 embedding shards. A set of endpoint disconnects was requeued at retry one of three. No shard reached retry two, the source kept 32 live workers, and its stable rate gives a rough 21:05-21:10 UTC completion estimate.
- East progress: `nemotron_cc_v2-diverse_qa_016d1909` got to 2,648 of 4,416 embedding shards with 32 live workers and no dead workers. Its stable rate gives a rough 22:25-22:35 UTC completion estimate.
- Issue update: Post one major update for the RNO capacity loss and recovery.
- Next action: Confirm that the disconnect burst has ended, then continue the regular health-check cadence.

### 2026-08-12 20:44 UTC - RNO recovery remained stable

- Root health: Both roots remain running with zero failures and zero preemptions.
- RNO progress: `stack-v3_6ac1a286` got to 9,995 of 12,818 embedding shards with 32 live workers and no dead workers. No new disconnect or retry-two event appeared after 20:41:06 UTC. RNO still has 82 live TEI services.
- East progress: `nemotron_cc_v2-diverse_qa_016d1909` got to 2,742 of 4,416 embedding shards with 32 live workers and no dead workers. East still has 96 live TEI services.
- Issue update: None. The major recovery update is already on issue #8162.
- Next action: Continue the regular health-check cadence and watch Stack v3 through completion.

### 2026-08-12 20:57 UTC - More RNO TEI capacity recovered

- Root health: Both roots remain running with zero failures and zero preemptions. Both active sources have 32 live workers and no dead workers.
- RNO service event: Another `RemoteDisconnected` burst appeared at 20:49 UTC. All affected shards were requeued at retry one of three, and no shard reached retry two.
- Recovery: The remaining 14 terminal TEI jobs were resubmitted with their original callable data and deterministic port pairs. Six passed local health checks, which increased healthy RNO TEI capacity from 82 to 88. Eight remain in the build state while they wait for H100 capacity.
- RNO progress: `stack-v3_6ac1a286` got to 11,758 of 12,818 embedding shards with 32 live workers and no dead workers. Its current rate gives a rough 21:05 UTC completion estimate.
- Issue update: None. Wait for full TEI recovery, a source-state change, or a new error level before another issue update.
- Next action: Watch Stack v3 through completion and check the eight queued TEI jobs as H100 capacity changes.

### 2026-08-12 21:11 UTC - Stack v3 and SVG completed

- Root health: Both roots are running with zero failures and zero preemptions.
- Completed artifacts: East has 45 completed source artifacts. RNO has 16 completed source artifacts. The full run has 61 of 292 completed source artifacts.
- RNO results: `stack-v3_6ac1a286` completed all 12,818 output shards with 26,075,490 duplicate documents and 314,413,350,203 input text bytes. The source step and all 32 worker tasks succeeded. Forty-four shards used retry one, and no shard reached retry two. `svg_5cac82e4` then completed all 27 scheduled tasks, and its source step succeeded.
- RNO progress: `swe-rebench-openhands_8decfaa0` started with 12 live workers and no dead workers.
- East progress: `nemotron_cc_v2-diverse_qa_016d1909` got to 3,137 of 4,416 embedding shards with 32 live workers and no dead workers. Its stable rate gives a rough 22:30 UTC completion estimate.
- Service capacity: East has 96 healthy TEI services. RNO has 88 healthy TEI services. The remaining eight RNO services are still in the build state with zero failures and zero preemptions.
- Issue update: None. These are routine source completions, and the partial RNO capacity recovery is already in the task record.
- Next action: Check SWE-rebench OpenHands through completion and continue to watch the eight queued RNO TEI services.

### 2026-08-12 21:27 UTC - Aggregate shard baseline

- Root health: Both roots are running with zero failures and zero preemptions. Both active sources have 32 live workers and no dead workers.
- Completed artifacts: East has 45 completed source artifacts. RNO has 18 completed source artifacts. The full run has 63 of 292 completed source artifacts.
- Aggregate progress: The 292 normalized sources contain 166,775 input shards. A direct object count found 90,802 fuzzy-duplicate embedding Parquet shards, or 54.45%. There are 75,973 shards left. This count includes completed and active source outputs.
- RNO results: `swe-rebench-openhands_8decfaa0` and `synthetic-1_9c07e65b` succeeded. `starcoder2-ir_cpp_f66a5c36` reached 64 of 97 output shards.
- East progress: `nemotron_cc_v2-diverse_qa_016d1909` reached 3,420 of 4,416 output shards.
- Error check: No retry-two, rate-limit, memory, dead-worker, or missing-file error matched either active source log during the check window.
- Issue update: None. There is no major run-state change.
- Next action: Use aggregate shard counts for future progress reports and continue the regular health-check cadence.

### 2026-08-12 21:47 UTC - Three StarCoder sources completed

- Root health: Both roots are running with zero failures and zero preemptions.
- Completed artifacts: East has 45 completed source artifacts. RNO has 21 completed source artifacts. The full run has 66 of 292 completed source artifacts.
- Aggregate progress: A fresh object count found 91,183 of 166,775 output shards, or 54.67%. There are 75,592 shards left.
- RNO results: `starcoder2-ir_cpp_f66a5c36` completed 97 shards, `starcoder2-ir_python_f334a918` completed 11 shards, and `starcoder2-kaggle_f7b0c8ab` completed eight shards. All three source steps succeeded.
- RNO progress: `biocollection-instruction_stream_003a7575` started 36 large shards. Its normalized input has 23,588,716 records. All 32 workers are alive, and four shards are queued.
- RNO profile: An on-demand thread capture from worker zero found the shard task in `TeiEmbeddingClient.embed`. Its 16 request threads were waiting for TEI HTTP responses. A sampled TEI service was returning successful requests with active queue and inference time. This confirms GPU embedding work rather than a stalled join.
- East progress: `nemotron_cc_v2-diverse_qa_016d1909` reached 3,744 of 4,416 output shards with 32 live workers and no dead workers.
- Error check: No retry-two, rate-limit, memory, dead-worker, or missing-file error matched either active source log during the check window.
- Service capacity: East has 96 healthy TEI services. RNO has 88 healthy TEI services. The remaining eight RNO services are still in the build state with zero failures and zero preemptions.
- Issue update: None. There is no major run-state change.
- Next action: Watch the large BioCollection shards for their first completion and continue the regular East health checks.

### 2026-08-12 22:08 UTC - Aggregate shard progress reached 54.91%

- Root health: Both roots are running with zero source failures.
- Completed artifacts: East has 45 completed source artifacts. RNO has 25 completed source artifacts. The full run has 70 of 292 completed source artifacts.
- Aggregate progress: A direct object count found 91,569 of 166,775 output shards, or 54.91%. There are 75,206 shards left. East has written 33,452 shards, and RNO has written 58,117 shards.
- RNO results: `biocollection-instruction_stream_003a7575` completed 36 shards, `cp/arxiv_papers_bcf7caef` completed 22 shards, `cp/data_provenance_6de533c9` completed four shards, and `cp/foodista_dfde536c` completed one shard. All four source steps succeeded.
- RNO progress: `cp/library_of_congress_8e80a121` is running.
- East progress: `nemotron_cc_v2-diverse_qa_016d1909` reached 4,031 of 4,416 output shards with 32 live workers and no dead workers at 22:05 UTC.
- Issue update: None. These are routine source completions, not a major run-state change.
- Next action: Continue the regular health checks and count all output objects for each aggregate progress request.

### 2026-08-12 22:31 UTC - Nemotron diverse QA completed

- Root health: Both roots are running with zero source failures.
- Completed artifacts: East has 46 completed source artifacts. RNO has 27 completed source artifacts. The full run has 73 of 292 completed source artifacts.
- Aggregate progress: A direct object count found 92,097 of 166,775 output shards, or 55.22%. There are 74,678 shards left. East has written 33,804 shards, and RNO has written 58,293 shards.
- East result: `nemotron_cc_v2-diverse_qa_016d1909` completed all 4,416 output shards with 506,433,587 duplicate documents and 1,410,848,683,168 input text bytes. The source step succeeded with zero failures and zero preemptions.
- East progress: `nemotron_cc_v2-high_quality_synthetic_ea28c25e` started.
- RNO results: `cp/library_of_congress_8e80a121` and `cp/news_b412d327` succeeded.
- RNO progress: `cp/pes2o_c403b7bb` reached 116 of 211 output shards with 32 live workers and no dead workers.
- Error check: No retry-two, rate-limit, memory, dead-worker, or missing-file error matched either active source log during the check window.
- Issue update: None. These are routine source completions, not a major run-state change.
- Next action: Watch `cp/pes2o` through completion and confirm that the new east source starts embedding.

### 2026-08-12 23:04 UTC - RNO TEI capacity fell to 80

- Root health: Both roots are running with zero failures and zero preemptions.
- Completed artifacts: East has 46 completed source artifacts. RNO has 30 completed source artifacts. The full run has 76 of 292 completed source artifacts.
- Aggregate progress: A direct object count found 93,645 of 166,775 output shards, or 56.15%. There are 73,130 shards left. East has written 35,084 shards, and RNO has written 58,561 shards.
- RNO service event: Eight previously queued TEI services obtained H100 workers. Sixteen other TEI services were then preempted and moved to the build state. Healthy RNO capacity fell from 88 to 80 and stayed there for more than ten minutes. No TEI service job failed.
- RNO results: `cp/pes2o_c403b7bb` completed 211 shards with 3,836,569 duplicate documents, `cp/pre_1929_books_aa6450f5` completed 61 shards with 107,100 duplicate documents, and `cp/project_gutenberg_4b66f21a` completed 28 shards with 54,293 duplicate documents. All three source steps succeeded.
- RNO progress: `cp/pubmed_1a33b0a8` is running with 32 live workers and no dead workers. No retry-two, rate-limit, memory, dead-worker, or missing-file error matched its logs.
- East progress: `nemotron_cc_v2-high_quality_synthetic_ea28c25e` is embedding 8,778 shards with 32 live workers and no dead workers.
- Issue update: Post one major update for the sustained RNO capacity loss.
- Next action: Watch the 16 building RNO services and recover only if a service job becomes terminal.

### 2026-08-12 23:52 UTC - RNO TEI capacity returned to 88

- Root health: Both roots are running with zero failures and zero preemptions.
- Completed artifacts: East has 46 completed source artifacts. RNO has 33 completed source artifacts. The full run has 79 of 292 completed source artifacts, and 81 sources have written output.
- Aggregate progress: A direct object count at 23:49 UTC found 95,986 of 166,775 output shards, or 57.55%. There are 70,789 shards left. East has written 37,090 shards, and RNO has written 58,896 shards.
- RNO service recovery: Eight TEI services recovered automatically. RNO now has 88 running services and eight services in the build state. This state stayed stable for more than ten minutes. No TEI service job failed.
- RNO results: `cp/pubmed_1a33b0a8` completed 162 shards, `cp/stackexchange_89d5f2ba` completed 130 shards, and `cp/uk_hansard_1e827dc0` completed 11 shards. All three source steps succeeded.
- RNO progress: `cp/uspto_97add1bd` reached 128 of 551 shards with 32 live workers and no dead workers. The service transition caused eight `RemoteDisconnected` errors. Zephyr requeued all eight shards at retry one of three. The burst ended at 23:38:41 UTC, and no shard reached retry two.
- East progress: `nemotron_cc_v2-high_quality_synthetic_ea28c25e` reached 3,353 of 8,778 shards with 32 live workers and no dead workers.
- Issue update: Post one major update for the stable RNO capacity recovery.
- Next action: Continue to watch both active sources and the eight building RNO services.

### 2026-08-13 00:33 UTC - RNO USPTO coordinator recovered automatically

- Root health: Both roots remain running with zero failures and zero preemptions. East has 46 completed source artifacts, and RNO has 33. The full run has 79 of 292 completed source artifacts.
- RNO recovery: The first USPTO coordinator lost its actor endpoint at 00:22:55 UTC after two preemptions. Iris ended its workers and started a replacement coordinator at 00:22:57 UTC. The source job stayed running, so no manual recovery was necessary.
- Resume check: The replacement coordinator found the completed output shards through `skip_existing`, restored 32 of 32 workers, and advanced USPTO from 320 to 352 of 551 shards. No worker is dead, and no retry-two, rate-limit, memory, or missing-file error matched the replacement logs.
- Aggregate progress: A direct object count found 97,973 of 166,775 output shards, or 58.75%. There are 68,802 shards left. East has written 38,853 shards, and RNO has written 59,120 shards.
- East progress: `nemotron_cc_v2-high_quality_synthetic_ea28c25e` reached 5,071 of 8,778 shards with 32 live workers and no dead workers.
- Service capacity: RNO has 88 running TEI services and eight services in the build state. No TEI service job failed.
- Issue update: Post one major update for the automatic coordinator recovery.
- Next action: Watch USPTO through completion and confirm that later sources start without another coordinator preemption.

### 2026-08-13 01:13 UTC - RNO service recovered and USPTO completed

- Root health: Both roots remain running with zero failures and zero preemptions. East has 46 completed source artifacts, and RNO has 35. The full run has 81 of 292 completed source artifacts.
- RNO service event: A preemption wave reduced the healthy TEI pool from 88 to 80 services. Fifteen services returned to the build queue. Service 030 then failed when TEI could not bind fixed port 12252 because the address was in use.
- Service recovery: Service 030 was resubmitted with the original callable data and fixed ports 12252 and 12253. It is in the interactive build queue. RNO has 80 running services, 16 building services, and zero failed services.
- RNO result: `cp/uspto_97add1bd` completed all 551 output shards. Closed TEI connections caused retry-one events during the final wave. All affected shards completed, and no retry-two or retry-three event occurred.
- RNO progress: `cp/youtube_54df62a2` then completed all 25 output shards. `finepdfs-arb_arab_603af2a3` started next.
- East progress: `nemotron_cc_v2-high_quality_synthetic_ea28c25e` reached 6,664 of 8,778 output shards with 32 live workers and no dead workers at 01:10 UTC.
- Aggregate progress: A direct object count found 99,918 of 166,775 output shards, or 59.91%. There are 66,857 shards left. East has written 40,574 shards, and RNO has written 59,344 shards.
- Issue update: Posted the major RNO service recovery and USPTO completion update at https://github.com/marin-community/marin/issues/8162#issuecomment-5274704587.
- Next action: Watch the 16 queued RNO services, both active sources, and all source or root failure counters.

### 2026-08-13 02:09 UTC - RNO TEI pool fully recovered

- Root health: Both roots remain running with zero failures and zero preemptions. East has 47 completed source artifacts, and RNO has 40. The full run has 87 of 292 completed source artifacts.
- Service recovery: All 16 queued RNO TEI services obtained H100 capacity. RNO has 96 running services, zero building services, and zero failed services. This state stayed stable for more than ten minutes.
- Service 030 check: The recovered service is running with zero failures and zero preemptions. Its local TEI health check on fixed port 12252 succeeded.
- East result: `nemotron_cc_v2-high_quality_synthetic_ea28c25e` completed all 8,778 output shards. `nemotron_cc_v2-medium_quality_edb2d9ae` started next.
- RNO results: `finepdfs-fra_latn_a27736a4`, `finepdfs-ind_latn_866435e0`, and `finepdfs-jpn_jpan_e6d158b1` succeeded. `finepdfs-pol_latn_51d3c17a` started next.
- Aggregate progress: A direct object count found 104,318 of 166,775 output shards, or 62.55%. There are 62,457 shards left. East has written 42,582 shards, and RNO has written 61,736 shards.
- Issue update: Posted the stable full-capacity recovery at https://github.com/marin-community/marin/issues/8162#issuecomment-5275096234.
- Next action: Continue the regular root, source, service, and error checks.

### 2026-08-13 02:30 UTC - Routine health check

- Root health: Both roots remain running with zero failures and zero preemptions.
- Completed artifacts: East has 47 completed source artifacts. RNO has 42. The full run has 89 of 292 completed source artifacts.
- Aggregate progress: A direct object count found 105,188 of 166,775 output shards, or 63.07%. There are 61,587 shards left. East has written 42,836 shards, and RNO has written 62,352 shards. Ninety-one sources have output, and no Parquet path is unknown.
- East progress: `nemotron_cc_v2-medium_quality_edb2d9ae` reached 219 of 14,843 output shards with 32 live workers and no dead workers.
- RNO results: `finepdfs-ron_latn_a135b740` completed. `finepdfs-spa_latn_6fe75099` reached 167 of 1,409 output shards with 32 live workers and no dead workers.
- Service capacity: East and RNO each have 96 running TEI services. Neither pool has a building or failed service. East has seven total service preemptions, and RNO has 161.
- Error check: No retry, rate-limit, connection, memory, dead-worker, or missing-file error matched either active source log.
- Issue update: None. There is no major run-state change.
- Next action: Continue the regular root, source, service, and error checks.

### 2026-08-13 03:07 UTC - RNO TEI capacity fell to 80

- Root health: Both roots remain running with zero failures and zero preemptions.
- RNO service event: A preemption wave moved 16 TEI services from the running state to the build state. Healthy RNO capacity fell from 96 to 80 services and stayed there for ten minutes. No TEI service failed.
- Service retries: The RNO TEI pool now has 185 total preemptions, an increase of 24 since the 02:30 UTC check. Services 083 and 092 each have nine preemptions and remain below the limit.
- RNO progress: `finepdfs-spa_latn_6fe75099` reached 1,311 of 1,409 output shards with 32 live workers and no dead workers. No retry, connection, rate-limit, memory, or missing-file error matched its log after the capacity loss.
- East progress: `nemotron_cc_v2-medium_quality_edb2d9ae` reached 869 of 14,843 output shards with 32 live workers and no dead workers.
- Decision: Keep both roots running. The surviving RNO pool remains healthy, and the source continues to write output.
- Issue update: Post one major update for the sustained RNO capacity loss.
- Next action: Watch the 16 building services and recover only a terminal service. Watch Spanish FinePDF through completion.

### 2026-08-13 05:04 UTC - RNO capacity recovered to 95 services

- Root health: Both roots remain running with zero failures and zero preemptions. East has 47 completed source artifacts, and RNO has 44. The full run has 91 of 292 completed source artifacts.
- Aggregate progress: A direct object count at 04:50 UTC found 110,609 of 166,775 output shards, or 66.32%. There are 56,166 shards left. East has written 45,078 shards, and RNO has written 65,531 shards. Ninety-three sources have output, and no Parquet path is unknown.
- RNO results: Spanish FinePDF completed all 1,409 shards with 7,309,974 duplicate documents. Thai FinePDF completed all 110 shards with 159,172 duplicate documents. `dolma4pdfs_75504c36` started next.
- Service event: Fifteen of the 16 queued RNO TEI services recovered. Service 090 then failed because TEI could not bind port 12372. The job had three preemptions before the port collision.
- Service recovery: The original controller request and stored callable files were resubmitted for service 090. Its fixed ports remain 12372 and 12373. The local health check succeeded, and the service stayed running with zero failures and zero preemptions for ten minutes.
- Service capacity: RNO has 95 running services, one building service, and zero failed services. Service 092 has nine preemptions and remains in the build state.
- Active progress: East Nemotron medium-quality reached 2,720 of 14,843 shards. RNO Dolma4PDFs reached 2,284 of 8,683 shards. Both sources have 32 live workers and no dead workers.
- Issue update: Post one major update for the stable capacity recovery and terminal-service repair.
- Next action: Watch service 092 for capacity or a terminal state. Continue the regular root, source, service, and error checks.

### 2026-08-13 05:32 UTC - Routine health check

- Root health: Both roots remain running with zero failures and zero preemptions.
- East progress: `nemotron_cc_v2-medium_quality_edb2d9ae` reached 3,165 of 14,843 output shards with 32 live workers and no dead workers.
- RNO progress: `dolma4pdfs_75504c36` reached 3,070 of 8,683 output shards with 32 live workers and no dead workers.
- Service capacity: East has 96 running TEI services. RNO has 95 running services and one building service. Neither pool has a failed service.
- Error check: No retry, rate-limit, connection, memory, dead-worker, or missing-file error matched either active source log.
- Issue update: None. There is no major run-state change.
- Next action: Continue the regular root, source, service, and error checks.

### 2026-08-13 06:30 UTC - RNO TEI pool returned to full capacity

- Root health: Both roots remain running with zero failures and zero preemptions.
- Service fault: Service 092 stayed in the build state after nine preemptions. Its attempt-nine pod was scheduled at 04:24 UTC, but the init containers did not start. A Kubernetes event showed that the pod could not mount its `workdir-files` volume because the Iris-managed ConfigMap did not exist. The assigned node and GPU were healthy.
- Service recovery: The Iris controller moved only task `tei-harrier-2d9c6923-092/0` to a preempted state with a recovery reason. Attempt 10 started with a new ConfigMap and became ready without a direct Kubernetes change.
- Recovery check: Service 092 stayed running for more than ten minutes. Its TEI process uses fixed ports 12376 and 12377, and the local health check on port 12376 succeeded.
- Service capacity: East and RNO each have 96 running TEI services, zero building services, and zero failed services.
- Aggregate progress: A direct object count at 06:32 UTC found 115,124 of 166,775 output shards, or 69.03%. There are 51,651 shards left. East has written 46,720 shards, and RNO has written 68,404 shards. Ninety-three sources have output, and no Parquet path is unknown.
- Active progress: East Nemotron medium-quality reached 4,094 of 14,843 shards. RNO Dolma4PDFs reached 4,696 of 8,683 shards. Both sources have 32 live workers and no dead workers.
- Issue update: Post one major update for the full-capacity recovery.
- Next action: Continue the regular root, source, service, and error checks.

### 2026-08-13 07:03 UTC - Relaunched at batch cluster maxima

- User direction: Stop both interactive roots and restart the same deterministic partitions at batch priority with the maximum H100 request in each target cluster.
- Stopped roots: `/rav/harrier-fuzzy-dups-east-p0-20260811-v3` and `/rav/harrier-fuzzy-dups-rno-p1-20260811-v3` were stopped at 06:40 UTC. All 96 TEI descendants in each region became inactive.
- Maximum request: The cluster backend reported 256 total H100s in East and 512 total H100s in RNO. The final jobs request those full counts.
- Final east root: `/rav/harrier-fuzzy-dups-east-p0-20260813-batch-max-v2`.
- Final RNO root: `/rav/harrier-fuzzy-dups-rno-p1-20260813-batch-max-v2`.
- East command: `uv run iris --cluster marin job run --no-wait --target-cluster cw-us-east-02a --job-name harrier-fuzzy-dups-east-p0-20260813-batch-max-v2 --priority batch --cpu 2 --memory 8GB --disk 16GB --enable-extra-resources --extra datakit -e MARIN_PREFIX s3://marin-us-east-02a/marin -- python -m experiments.datakit.embeddings.harrier.run --document-set fuzzy_duplicates --partition-index 0 --partition-count 2 --tei-instances 256 --max-concurrent 8`.
- RNO command: `uv run iris --cluster marin job run --no-wait --target-cluster cw-rno2a --job-name harrier-fuzzy-dups-rno-p1-20260813-batch-max-v2 --priority batch --cpu 2 --memory 8GB --disk 16GB --enable-extra-resources --extra datakit -e MARIN_PREFIX s3://marin-us-east-02a/marin -- python -m experiments.datakit.embeddings.harrier.run --document-set fuzzy_duplicates --partition-index 1 --partition-count 2 --tei-instances 512 --max-concurrent 8`.
- Launch correction: The first batch replacements requested the program default of 128 TEI services per region. They were stopped when the cluster maxima were confirmed, and the final roots above replaced them.
- Root health: Both final roots stayed running with zero failures and zero root preemptions for more than ten minutes.
- Service state: All 256 East and 512 RNO service jobs are active at batch priority. At 06:57 UTC, East had 192 running service tasks and 64 building tasks. At 07:02 UTC, RNO had 104 running service tasks and 408 building tasks. Batch capacity can yield to higher-priority work.
- RNO service repair: Service 025 failed once because port 12050 was in use. Its exact batch request and stored callable files were replayed. The replacement is active in the batch queue with zero failures.
- Source concurrency: East and RNO each have eight running source jobs, with no pending or failed source job. Visible Zephyr coordinators have 32 live workers and no dead worker.
- Output safety: The write stage keeps `skip_existing=True`. A direct object count at 06:59 UTC found 115,640 of 166,775 output shards, or 69.34%. There are 51,135 shards left. East has 46,898 shards, RNO has 68,742 shards, 95 sources have output, and no Parquet path is unknown.
- Issue update: Post one major update for the user-directed batch restart.
- Next action: Watch batch preemptions, terminal service failures, all 16 active sources, and aggregate output growth.

### 2026-08-13 08:00 UTC - Batch restart hourly checkpoint

- Root health: Both batch roots remain running with zero failures and zero root preemptions after 70 minutes.
- Completed artifacts: The final roots completed nine new source artifacts. East completed five, and RNO completed four. The full run has 100 of 292 completed source artifacts.
- Source concurrency: East and RNO each have eight running source jobs. No source job is pending or failed.
- Aggregate progress: A direct object count at 07:46 UTC found 119,853 of 166,775 output shards, or 71.87%. There are 46,922 shards left. East has 49,249 shards, RNO has 70,604 shards, 111 sources have output, and no Parquet path is unknown.
- Output rate: The jobs wrote 3,149 shards between 07:11 UTC and 07:46 UTC.
- Service state: All 256 East and 512 RNO service jobs remain active. East has 192 running service tasks and 64 building tasks. RNO has 112 running service tasks and 400 building tasks.
- RNO service 025: The recovered service remains active in the batch queue. It has no second failure.
- TEI health: East service logs show successful embedding requests. Short `no permits available` events occur during queue pressure, and later requests succeed.
- Issue update: None. The user-directed restart update already contains the major state change.
- Next action: Continue the root, source, service, Zephyr, and output checks. Recover only a terminal in-scope job failure.

### 2026-08-13 09:32 UTC - Batch output passed 77 percent

- Root health: Both batch roots remain running with zero failures and zero root preemptions after 2 hours and 42 minutes.
- Completed artifacts: The final roots completed 65 new source artifacts. East completed 61, and RNO completed four. The full run has 156 of 292 completed source artifacts.
- Source concurrency: RNO has eight running source jobs. East has six running sources and one pending source during rapid source turnover. No source job failed.
- Aggregate progress: A direct object count at 09:31 UTC found 129,544 of 166,775 output shards, or 77.68%. There are 37,231 shards left. East has 53,840 shards, RNO has 75,704 shards, 162 sources have output, and no Parquet path is unknown.
- Output rate: The jobs wrote 4,286 shards between 08:43 UTC and 09:31 UTC. A simple rate estimate gives 6.8 hours for the remaining shards. Source sizes and batch capacity can change this estimate.
- Zephyr progress: East Nemotron medium-quality reached 5,116 of 14,843 shards. RNO Dolma4PDFs reached 7,098 of 8,683 shards. Each source has 32 live workers and zero dead workers.
- Service state: All 256 East and 512 RNO service jobs remain active. East has 192 running service tasks and 64 building tasks. RNO has 112 running service tasks and 400 building tasks.
- Error check: The last ten-minute root scans found no retry, connection, port, rate-limit, memory, dead-worker, missing-file, traceback, or exception event.
- Restart status note: The root logs show `previous status: FAILED` for sources from the stopped interactive roots. The final roots use force-run mode and start those sources. Current source failures remain zero.
- Issue update: None. This is routine progress after the batch restart.
- Next action: Continue the root, source, service, Zephyr, and output checks. Recover only a terminal in-scope job failure.

### 2026-08-13 11:14 UTC - Batch output passed 82 percent

- Root health: Both batch roots remain running. Their waiter sessions have no terminal output.
- Completed artifacts: The final roots completed 104 new source artifacts. East completed 97, and RNO completed seven. With the artifacts from the stopped roots, East has 144 of 146 artifacts and RNO has 51 of 146 artifacts. The full run has 195 of 292 completed source artifacts.
- Source concurrency: East has two running source jobs, and RNO has eight. No source job is pending or failed.
- Aggregate progress: A direct object count at 11:12 UTC found 137,266 of 166,775 output shards, or 82.30%. There are 29,509 shards left. East has 57,950 shards, RNO has 79,316 shards, 203 sources have output, and no Parquet path is unknown.
- Output rate: The jobs wrote 7,722 shards between 09:31 UTC and 11:12 UTC. A simple rate estimate gives 6.4 hours for the remaining shards. Source sizes and batch capacity can change this estimate.
- Resume evidence: The completed Dolma4PDFs stage reported 5,036 skipped partitions and 3,647 processed shards. These counts sum to its 8,683 partitions and show that the restart skipped its existing output.
- Service state: All 256 East and 512 RNO service jobs remain active. East has 192 running service tasks and 64 building tasks. RNO has 112 running service tasks and 400 building tasks.
- Zephyr health: The checked coordinators have 32 live workers and zero dead workers. One RNO source reported five shards at retry attempt one. No retry attempt two is present.
- Issue update: None. This is routine progress after the batch restart.
- Next action: Watch the final two East sources, the eight active RNO sources, service state, and output growth. Recover only a terminal in-scope job failure.

### 2026-08-13 12:02 UTC - Batch output passed 84 percent

- Root health: Both batch roots remain running with zero failures and zero root preemptions.
- Completed artifacts: The final roots completed 108 new source artifacts. East completed 97, and RNO completed 11. With the artifacts from the stopped roots, East has 144 of 146 artifacts and RNO has 55 of 146 artifacts. The full run has 199 of 292 completed source artifacts.
- Source concurrency: East has two running source jobs, and RNO has eight. No source job is pending or failed.
- Aggregate progress: A direct object count at 12:00 UTC found 140,074 of 166,775 output shards, or 83.99%. There are 26,701 shards left. East has 59,825 shards, RNO has 80,249 shards, 206 sources have output, and no Parquet path is unknown.
- Output rate: The jobs wrote 2,808 shards between 11:12 UTC and 12:00 UTC. A simple rate estimate gives 7.6 hours for the remaining shards. Source sizes and batch capacity can change this estimate.
- Service state: All 256 East and 512 RNO TEI service jobs remain active. Neither pool has a failed service job.
- Zephyr health: The checked coordinators have live workers. One translated source had a worker heartbeat timeout, but its other workers completed all 267 shards. The source then sealed without manual recovery. No shard has reached retry attempt two.
- Large-shard check: The scientific-coding source has two active workers on two large shards. A five-second process sample showed increasing CPU time and I/O. The source is active, although its completed-shard counter remains at zero.
- Issue update: None. This is routine progress after the batch restart.
- Next action: Continue the root, source, service, Zephyr, and output checks. Recover only a terminal in-scope job failure.

### 2026-08-13 12:34 UTC - Reduced RNO source concurrency after endpoint-list stall

- RNO symptom: All eight active source counters stayed unchanged from 12:12 through 12:31 UTC. Many Zephyr workers reported `Operation list_endpoints failed ... Request timed out`. The source jobs and all 512 TEI service jobs stayed live.
- Retry state: Some endpoint-list operations reached client retry attempt two. No output shard reached Zephyr retry attempt two, and no source job failed.
- Controller evidence: Direct endpoint listing succeeded. Direct controller logs showed large bursts of `ListEndpoints` calls that completed in about 1.0 to 2.0 seconds. A ten-second CPU profile sampled most controller CPU time in Kubernetes reconciliation serialization and deserialization. This supports a control-plane load diagnosis, but it does not prove the root cause.
- Decision: Eight source jobs can start as many as 256 Zephyr workers, and each worker can start 16 TEI requests. Stop only the RNO root and reduce source concurrency before reducing the requested GPU capacity. Do not restart the Iris controller.
- Stopped root: `/rav/harrier-fuzzy-dups-rno-p1-20260813-batch-max-v2`.
- Replacement root: `/rav/harrier-fuzzy-dups-rno-p1-20260813-batch-max-v3`.
- Replacement command: `uv run iris --cluster marin job run --no-wait --target-cluster cw-rno2a --job-name harrier-fuzzy-dups-rno-p1-20260813-batch-max-v3 --priority batch --cpu 2 --memory 8GB --disk 16GB --enable-extra-resources --extra datakit -e MARIN_PREFIX s3://marin-us-east-02a/marin -- python -m experiments.datakit.embeddings.harrier.run --document-set fuzzy_duplicates --partition-index 1 --partition-count 2 --tei-instances 512 --max-concurrent 4`.
- Output safety: The replacement uses the same output paths and `skip_existing=True`. Completed Parquet shards remain in place.
- East state: `/rav/harrier-fuzzy-dups-east-p0-20260813-batch-max-v2` continues without a restart.
- Incident record: https://echo.oa.dev/wiki/141.
- Issue update: https://github.com/marin-community/marin/issues/8162#issuecomment-5280478157.
- Next action: Confirm all 512 RNO service jobs become active, then confirm four source jobs resume shard progress without a new endpoint-list timeout storm.

### 2026-08-13 12:55 UTC - RNO resumed with one source

- Four-source result: `/rav/harrier-fuzzy-dups-rno-p1-20260813-batch-max-v3` reproduced the synchronized endpoint-list timeout wave before it completed a new shard. The root was stopped.
- Final RNO root: `/rav/harrier-fuzzy-dups-rno-p1-20260813-batch-max-v4`.
- Final RNO command: `uv run iris --cluster marin job run --no-wait --target-cluster cw-rno2a --job-name harrier-fuzzy-dups-rno-p1-20260813-batch-max-v4 --priority batch --cpu 2 --memory 8GB --disk 16GB --enable-extra-resources --extra datakit -e MARIN_PREFIX s3://marin-us-east-02a/marin -- python -m experiments.datakit.embeddings.harrier.run --document-set fuzzy_duplicates --partition-index 1 --partition-count 2 --tei-instances 512 --max-concurrent 1`.
- Capacity choice: The root still requests all 512 RNO TEI service jobs. One source starts as many as 32 Zephyr workers, and each worker starts as many as 16 TEI requests. One source can therefore issue as many as 512 concurrent TEI requests with fewer endpoint-list calls.
- Validation: The active high-quality Nemotron source resumed from 416 shards and reached 608 of 4,136. All 32 Zephyr workers were alive, and no endpoint-list timeout matched the replacement logs.
- Service admission: At validation time, 32 TEI tasks were running and 480 were in the build state at batch priority.
- Incident update: https://echo.oa.dev/wiki/141.
- Issue update: https://github.com/marin-community/marin/issues/8162#issuecomment-5280678102.
- Next action: Keep the one-source RNO root running. Watch shard rate, endpoint-list warnings, batch service admission, and both root states.

### 2026-08-13 13:05 UTC - Corrected RNO stage interpretation

- Correction: The 608 and 3,244 counters observed during the RNO v4 validation were join-side stage tasks, not written output Parquet shards. The active source output directory still had 416 Parquet shards at 13:03 UTC.
- Current validation: All 32 Zephyr workers are live, the join-side stage is active, and no endpoint-list timeout matched v4 logs. Output recovery remains unconfirmed until the write stage adds Parquet shards.
- Aggregate progress: A direct object count found 142,697 of 166,775 output shards, or 85.56%. There are 24,078 shards left. East has 62,287 shards, RNO has 80,410 shards, 208 sources have output, and no Parquet path is unknown.
- RNO service admission: The v4 root requests all 512 batch TEI services. Thirty-two TEI tasks are running, and 480 are in the build state.
- Published correction: Edited the existing issue update at https://github.com/marin-community/marin/issues/8162#issuecomment-5280678102 and updated https://echo.oa.dev/wiki/141.
- Next action: Verify the first new RNO Parquet shard, then continue root, source, service, and output checks.

### 2026-08-13 13:08 UTC - RNO output recovery verified

- Output proof: A direct S3 check found 430 Parquet shards for the active high-quality Nemotron source, up from its 416 checkpoint shards.
- Inference proof: TEI service logs showed successful embedding requests across the 32 admitted RNO GPUs. Request times ranged from tens of milliseconds to about 1.4 seconds in the checked sample.
- Error check: No endpoint-list timeout matched the RNO v4 logs after the single-source restart.
- Incident state: The single-source mitigation is effective. Updated https://echo.oa.dev/wiki/141 and edited the existing issue update at https://github.com/marin-community/marin/issues/8162#issuecomment-5280678102.
- Next action: Continue the root, source, service, and output checks. Watch for another endpoint-list timeout as batch TEI capacity changes.

### 2026-08-13 14:00 UTC - Batch output passed 87 percent

- Root health: The East and RNO roots remain running with zero failures and zero root preemptions.
- Completed artifacts: East has 144 of 146 completed source artifacts. RNO has 57 of 146. The full run has 201 of 292 completed source artifacts.
- Aggregate progress: A direct object count found 145,145 of 166,775 output shards, or 87.03%. There are 21,630 shards left. East has written 64,547 shards, and RNO has written 80,598 shards. Two hundred eight sources have output, and no Parquet path is unknown.
- Output rate: The jobs wrote 1,095 shards between 13:35 UTC and 14:00 UTC. A simple rate estimate gives about 8.2 hours for the remaining shards. Source sizes and batch capacity can change this estimate.
- East progress: The final two sources reached 10,170 of 14,843 shards and 6,183 of 14,285 shards. Each source has 32 live workers and zero dead workers.
- RNO progress: The active source reached 602 of 4,136 shards with 32 live workers and zero dead workers. The RNO pool has 24 running TEI tasks and 488 building tasks at batch priority.
- Error check: No endpoint-list timeout, shard retry, source failure, or root failure matched the current checks. TEI logs contain `no permits available` events during request pressure, but the output counter continues to increase.
- Issue update: None. This is routine progress after the RNO control-plane mitigation.
- Next action: Continue the root, source, service, and output checks. Recover only a terminal in-scope job failure.

### 2026-08-13 14:27 UTC - RNO batch capacity fell to zero

- Capacity event: RNO had 24 running TEI tasks at 14:10 UTC. By 14:18 UTC, all 512 TEI tasks were in the build state. No RNO TEI endpoint remained active.
- Cause: The task-attempt records state that Kueue preempted the running tasks to admit higher-priority workloads in the same cluster queue. The backend reported zero free H100s. Interactive jobs held 416 H100s, and batch jobs held the other 96 H100s.
- Pipeline effect: The active RNO source stayed at 638 of 4,136 output shards. Eight shards entered retry attempt one after the endpoints stopped. No shard entered retry attempt two, and no source or root job failed.
- East state: The East sources continued to write output and reached 10,694 of 14,843 shards and 6,748 of 14,285 shards.
- Decision: Keep the RNO root in the batch queue. A restart cannot create GPU capacity and would add another endpoint-pool transition.
- Issue update: Post one major update for the full RNO capacity loss.
- Next action: Watch for RNO task admission. When capacity returns, confirm that the eight retried shards complete and that no endpoint-list timeout returns.

### 2026-08-13 14:49 UTC - RNO capacity and terminal services recovered

- Capacity recovery: Kueue admitted 80 RNO TEI tasks. The other 432 tasks remain in the batch queue. This state stayed stable for more than ten minutes.
- Service failures: Services 055, 061, 064, 070, 077, and 090 failed after admission because their fixed ports were in use. The failures occurred on three hosts.
- Service recovery: The six terminal jobs were replayed with their original batch requests and stored `_callable.pkl` and `_callable_runner.py` files. All 512 service jobs returned to the running job state. The six replacements are in the batch queue.
- Pipeline recovery: The active RNO source advanced from 638 to 798 of 4,136 output shards. It has 32 live workers and zero dead workers.
- Retry state: Thirty-eight shards entered retry attempt one during the endpoint loss. No shard entered retry attempt two, and the retry count stayed unchanged during the recovery check.
- East state: The East sources reached 11,118 of 14,843 shards and 7,185 of 14,285 shards. Each source has 32 live workers and zero dead workers.
- Incident record: https://echo.oa.dev/wiki/142.
- Issue update: Post one major update for the stable capacity and service recovery.
- Next action: Continue the root, source, service, and output checks. Watch the six queued replacements and the RNO retry depth.

### 2026-08-13 15:54 UTC - Output passed 90 percent

- Root health: The East and RNO roots remain running with zero root failures. No current source job has failed.
- Completed artifacts: East has 144 of 146 sealed source artifacts. RNO has 57 of 146. The full run has 201 of 292 sealed source artifacts.
- Aggregate progress: A direct object count found 150,128 of 166,775 output shards, or 90.02%. There are 16,647 shards left. East has written 68,632 shards, and RNO has written 81,496 shards. Two hundred eight sources have output, and no Parquet path is unknown.
- Output rate: The jobs wrote 2,357 shards from 15:00 UTC to 15:54 UTC. A constant-rate estimate gives about 6.4 hours for the remaining shards, or about 22:15 UTC. Batch preemption and source size can change this estimate.
- Active stages: The East sources reached 12,202 of 14,843 shards and 8,324 of 14,285 shards. The RNO source reached 1,510 of 4,136 shards. Each stage has 32 live workers and zero dead workers.
- Service admission: East has 184 running TEI tasks and 72 building tasks. RNO has 168 running TEI tasks and 344 building tasks. All 256 East service jobs and all 512 RNO service jobs remain in the running job state.
- Error check: Short endpoint-list timeout waves occurred in both clusters and recovered without a job action. RNO still reports 38 shards at retry attempt one. No shard entered retry attempt two, and no service job became terminal.
- Issue update: https://github.com/marin-community/marin/issues/8162#issuecomment-5282903873.
- Next action: Continue the root, source, service, Zephyr, and output checks. Recover only a terminal in-scope job failure.

### 2026-08-13 16:16 UTC - Replayed RNO service 404 after a port collision

- Detection: RNO TEI service 404 entered the failed job state. Its only task reported `Job exceeded max_task_failures`.
- Cause: TEI could not bind port 12808 on host `g53121e` because the address was in use. This matches the fixed-port collision recorded in https://echo.oa.dev/wiki/142.
- Recovery: Reconstructed the exact stored batch request for service 404, verified `_callable.pkl` and `_callable_runner.py`, and replayed only that terminal service.
- Validation: All 512 RNO service jobs returned to the running job state. The replacement task is building in the batch queue. The pool has 168 running tasks and 344 building tasks.
- Pipeline state: The active RNO source continued from 1,700 to 1,708 of 4,136 shards with 32 live workers and zero dead workers. The retry state remains 38 shards at attempt one, with no attempt two.
- Issue update: None. One service replay did not stop source progress and is not a major run update.
- Next action: Continue root, source, service, Zephyr, and output checks. Watch for another fixed-port collision after batch admission.

### 2026-08-13 17:15 UTC - Output passed 91 percent

- Root health: The East and RNO roots remain running with zero root failures. No current source job has failed.
- Completed artifacts: East has 144 of 146 sealed source artifacts. RNO has 57 of 146. The full run has 201 of 292 sealed source artifacts.
- Aggregate progress: A direct object count found 153,288 of 166,775 output shards, or 91.91%. There are 13,487 shards left. East has written 70,970 shards, and RNO has written 82,318 shards. Two hundred eight sources have output, and no Parquet path is unknown.
- Output rate: The jobs wrote 3,160 shards from 15:54 UTC to 17:15 UTC. A constant-rate estimate gives about 5.8 hours for the remaining shards, or about 23:00 UTC. Batch preemption and source size can change this estimate.
- Active stages: The East sources reached 13,369 of 14,843 shards and 9,536 of 14,285 shards. The RNO source reached 2,345 of 4,136 shards. Each stage has 32 live workers and zero dead workers.
- Service admission: East has 192 running TEI tasks and 64 building tasks. RNO has 184 running TEI tasks and 328 building tasks. All 256 East service jobs and all 512 RNO service jobs remain in the running job state.
- Retry state: RNO still reports 38 shards at retry attempt one. No shard entered retry attempt two.
- Issue update: None. This is a routine checkpoint after the 90-percent issue update.
- Next action: Continue root, source, service, Zephyr, and output checks. Recover only a terminal in-scope job failure.

### 2026-08-13 18:15 UTC - Output passed 93 percent

- Root health: The East and RNO roots remain running with zero root failures. No current source job has failed.
- Completed artifacts: East has 144 of 146 sealed source artifacts. RNO has 57 of 146. The full run has 201 of 292 sealed source artifacts.
- Aggregate progress: A direct object count found 156,361 of 166,775 output shards, or 93.76%. There are 10,414 shards left. East has written 73,128 shards, and RNO has written 83,233 shards. Two hundred eight sources have output, and no Parquet path is unknown.
- Output rate: The jobs wrote 1,477 shards from 17:45 UTC to 18:15 UTC. A constant-rate estimate gives about 3.5 hours for the remaining shards, or about 21:45 UTC. Batch preemption and source size can change this estimate.
- Active stages: The East sources reached 14,314 of 14,843 shards and 10,541 of 14,285 shards. The RNO source reached 3,174 of 4,136 shards. Each stage has 32 live workers and zero dead workers.
- Service admission: East has 192 running TEI tasks and 64 building tasks. RNO has 184 running TEI tasks and 328 building tasks. All 256 East service jobs and all 512 RNO service jobs remain in the running job state.
- Retry state: RNO still reports 38 shards at retry attempt one. No shard entered retry attempt two.
- Issue update: None. This is a routine checkpoint after the 90-percent issue update.
- Next action: Continue root, source, service, Zephyr, and output checks. Recover only a terminal in-scope job failure.

### 2026-08-13 19:15 UTC - Output passed 95 percent

- Root health: The East and RNO roots remain running with zero root failures. No current source job has failed.
- Completed artifacts: The medium-quality Nemotron source succeeded at 18:44 UTC. East now has 145 of 146 sealed source artifacts. RNO has 57 of 146. The full run has 202 of 292 sealed source artifacts.
- Aggregate progress: A direct object count found 158,749 of 166,775 output shards, or 95.19%. There are 8,026 shards left. East has written 75,006 shards, and RNO has written 83,743 shards. Two hundred eight sources have output, and no Parquet path is unknown.
- Output rate: The jobs wrote 916 shards from 18:46 UTC to 19:15 UTC. A constant-rate estimate gives about 4.2 hours for the remaining shards, or about 23:30 UTC. Batch preemption and source size can change this estimate.
- Active stages: The final East source reached 12,055 of 14,285 shards. The RNO source reached 3,757 of 4,136 shards. Each stage has 32 live workers and zero dead workers.
- Service admission: East has 184 running TEI tasks and 72 building tasks. RNO capacity fell from 184 running tasks to 80 running tasks, with 432 building tasks. All service jobs remain non-terminal.
- Retry state: RNO still reports 38 shards at retry attempt one. No shard entered retry attempt two.
- Issue update: None. One East source completion and a recoverable batch capacity reduction are not partition-level events.
- Next action: Continue root, source, service, Zephyr, and output checks. Watch RNO capacity and recover only a terminal in-scope job failure.

### 2026-08-13 21:01 UTC - Output passed 97 percent

- Aggregate progress: A direct object count at 20:55 UTC found 161,869 of 166,775 output shards, or 97.06%. There are 4,906 shards left. East has written 77,248 shards, and RNO has written 84,621 shards. Two hundred eight sources have output, and no Parquet path is unknown.
- Completed artifacts: East has 145 of 146 sealed source artifacts. RNO has 58 of 146. The full run has 203 of 292 sealed source artifacts.
- East recovery: The final source completed its first write stage and shut down its coordinator at 20:38 UTC. The source pod was then preempted during cleanup. Its automatic retry scanned the join data and checked existing output, but the East root pod was preempted at 20:54 UTC. The root resumed on a new node and launched the final source again. The new join scan reached 3,821 of 14,285 shards with 32 live workers and zero dead workers.
- East data safety: The output writer has `skip_existing=True`. Each retry checks the target file before it evaluates the embedding stream, so it does not embed an existing output shard again.
- RNO progress: The active medium-high-quality Nemotron source reached 984 of 3,679 output shards with 32 live workers and zero dead workers.
- RNO service recovery: Service 026 failed after two batch preemptions because TEI could not bind port 12052 on a reused node. Reconstructed its exact stored batch request, checked the dry run, and replayed only that service. All 512 service jobs are active again. At 20:54 UTC, 144 service tasks were running and 368 were building.
- Shared record: Echo entry 2613 records the service replay. The fixed-port incident remains at https://echo.oa.dev/wiki/142.
- Issue update: None. Both recoveries were automatic or limited to one service, and neither stopped RNO output.
- Next action: Let the East retry seal its final source and root. Continue RNO source, service, worker, and output checks.

### 2026-08-13 22:07 UTC - Two-source merge smoke passed

- Test scope: Merged the completed production inputs for `nsf_awards` and `agenttrove`. The output uses the two-day TTL prefix `s3://marin-us-east-02a/tmp/ttl=2d/harrier-merge-smoke/`. No full production merge was launched.
- First result: The first smoke run failed safely because the old and backfill inputs overlap on some IDs. A full input scan found 2 overlapping IDs in `nsf_awards` and 24 in `agenttrove`. The normalized data contains each ID once, and removal of the overlap gives its exact ID sequence.
- Merge rule: The revised merge keeps the old embedding when both inputs contain the same ID. It counts and removes the backfill copy. It then requires the complete output ID sequence to equal the matching normalized shard.
- Successful run: `/rav/harrier-merge-smoke-v2-20260813` wrote 3 of 3 `nsf_awards` shards with 381,789 rows and 43 of 43 `agenttrove` shards with 781,076 rows.
- Read-back check: `/rav/harrier-merge-verify-nsf-20260813` and `/rav/harrier-merge-verify-agenttrove-20260813` read the written Parquet files. Both checks confirmed the shard names, canonical schema, row counts, exact normalized ID order, and exact embedding rows from the two input datasets.
- Shared record: Echo entry 2627 records the smoke result.
- Issue update: None. This test is not a major backfill checkpoint.
- Next action: Finish the merge change checks. Continue to monitor the East and RNO backfill roots. Do not launch the full merge until the backfill is complete and verified.

### 2026-08-13 22:22 UTC - East partition completed and passed shard verification

- Root result: `/rav/harrier-fuzzy-dups-east-p0-20260813-batch-max-v2` succeeded.
- Source artifacts: All 146 expected East source artifacts have `SUCCESS` status. No source is missing, and no source has more than one completed artifact.
- Shard verification: A direct East-region check compared each output artifact with its normalized source. All 77,248 expected Parquet shard names are present. There are no missing or extra shards.
- Merge verification: The merge now reopens its output and checks the exact shard names, canonical embedding schema, row count for each shard, total row count, and complete ID order against normalized data.
- Final smoke: `/rav/harrier-merge-smoke-v3-20260813` passed these checks on production inputs. `nsf_awards` verified 3 shards and 381,789 rows. `agenttrove` verified 43 shards and 781,076 rows.
- Issue update: https://github.com/marin-community/marin/issues/8162#issuecomment-5287072480.
- Next action: Continue the RNO root through completion. Run the same partition verification before the full merge.

### 2026-08-13 22:52 UTC - RNO paused after East S3 write suspension

- Failure: At 22:48 UTC, all RNO workers began to fail `CreateMultipartUpload` calls to `s3://marin-us-east-02a`. S3 returned HTTP 405 with `Account is write suspended for this availability zone` and asked for a capacity-quota check.
- Effect: Embedding continued, but no new shard could seal. The coordinator remained at 2,052 of 3,679 shards for `nemotron_cc_v2-medium_high_quality` and re-queued each failed write.
- Safe action: Stopped `/rav/harrier-fuzzy-dups-rno-p1-20260813-batch-max-v4` at 22:52 UTC to prevent repeated GPU work. All 162,969 completed output shards remain in place. East remains complete and verified.
- Shared record: Echo entry 2635 records the storage suspension and safe stop.
- Issue update: https://github.com/marin-community/marin/issues/8162#issuecomment-5287292405.
- Next action: Use a small multipart write under a TTL prefix to probe the East bucket. Restart the RNO root from its existing output only after the write suspension clears.

### 2026-08-13 23:42 UTC - Remaining work split across East and RNO

- Storage recovery: A 64 MiB multipart write from the RNO controller to `s3://marin-us-east-02a/tmp/ttl=1d/harrier-write-probe-20260813-2316/probe.bin` succeeded at 23:16 UTC. This proved that East S3 multipart writes had recovered.
- Partition proof: The old RNO partition `1/2` has 146 sources. Partitions `1/4` and `3/4` have 73 sources each, are disjoint, and their union is exactly partition `1/2`.
- Stop race: `/rav/harrier-fuzzy-dups-rno-p1-20260813-batch-max-v5` reached RNO as its parent stop request arrived. The RNO controller reported the root as killed before it created child jobs. Two first replacement submissions were stopped before handoff to prevent an overlap.
- Final East command: `uv run iris --cluster marin job run --no-wait --target-cluster cw-us-east-02a --job-name harrier-fuzzy-dups-east-p1of4-20260813-batch-max-v4 --priority batch --cpu 2 --memory 8GB --disk 16GB --enable-extra-resources --extra datakit -e MARIN_PREFIX s3://marin-us-east-02a/marin -- python -m experiments.datakit.embeddings.harrier.run --document-set fuzzy_duplicates --partition-index 1 --partition-count 4 --tei-instances 256 --max-concurrent 1`.
- Final RNO command: `uv run iris --cluster marin job run --no-wait --target-cluster cw-rno2a --job-name harrier-fuzzy-dups-rno-p3of4-20260813-batch-max-v7 --priority batch --cpu 2 --memory 8GB --disk 16GB --enable-extra-resources --extra datakit -e MARIN_PREFIX s3://marin-us-east-02a/marin -- python -m experiments.datakit.embeddings.harrier.run --document-set fuzzy_duplicates --partition-index 3 --partition-count 4 --tei-instances 512 --max-concurrent 1`.
- Admission result: Both final roots reached their target clusters. East created all 256 TEI service jobs. RNO created all 512 TEI service jobs. The East root had one batch preemption and began its automatic retry. The RNO root remained running with no failures or preemptions.
- Issue update: None. Wait for confirmed new output before posting the storage-recovery update.
- Next action: Confirm that both roots select their first unfinished source, skip existing shards, and write new shards. Then post one major recovery update.

### 2026-08-13 23:54 UTC - Split restart writes new output

- Root health: The East `1/4` and RNO `3/4` replacement roots are running. East has zero failures and one recovered batch preemption. RNO has zero failures and zero preemptions.
- Resume proof: Each root selected the first incomplete source in its disjoint partition. Earlier complete source steps did not run again.
- Durable output proof: A direct S3 object count from the RNO controller found 163,020 of 166,775 output Parquet shards. This is 51 more than the 162,969 shards present before the split restart. The run is 97.75 percent complete, with 3,755 shards left.
- Active work: RNO is in the embed and write stage for `nemotron_sft/sft_math`. East is rebuilding the join state for `nemotron_cc_v2/medium_high_quality`; completed embedding output remains in place and the write stage uses `skip_existing=True`.
- Storage result: The new RNO output proves that multipart writes to the East bucket work after the storage suspension.
- Shared records: Echo entry 2640 records the recovery proof. The major issue update is https://github.com/marin-community/marin/issues/8162#issuecomment-5287731523.
- Next action: Continue both roots through completion. Verify all 292 source artifacts and all 166,775 shard names before the production merge.

### 2026-08-14 00:23 UTC - Five East TEI services recovered

- Failure: East TEI services `006`, `175`, `184`, `185`, and `186` failed because their fixed ports were already in use. The affected ports were `12524`, `12862`, `12880`, `12882`, and `12884`.
- Recovery: Replayed only the five terminal services from their saved controller requests. The model, resources, batch priority, bundle, ports, and TTL policy were unchanged.
- Result: All five service jobs returned to the running job state and entered their first build attempt. The active embedding source continued to write output.
- Issue update: None. This was a limited service recovery and did not stop source progress.

### 2026-08-14 01:04 UTC - Shared-pool merge smoke passed

- Change: Commit `2d629dc6b` makes every source pipeline borrow one top-level Zephyr context. The selected sources share one bounded worker pool and one coordinator. Temporary Zephyr data uses the output region.
- Test scope: `/rav/harrier-merge-shared-smoke-20260814` used the production inputs for `nsf_awards` and `agenttrove`. It wrote only to `s3://marin-us-east-02a/tmp/ttl=2d/harrier-merge-shared-smoke-20260814/`.
- Pool proof: The root created one Zephyr coordinator and one group of eight workers. Both source jobs ran against this pool. `nsf_awards` finished while `agenttrove` continued, and the same coordinator and worker group stayed active.
- Verification: `nsf_awards` passed the read-back check for all 3 shards and 381,789 normalized rows. `agenttrove` passed it for all 43 shards and 781,076 normalized rows. Each check compared shard names, schema, row counts, and the exact ID order with normalized data.
- Result: Both source jobs and the root succeeded. The root then closed the shared context and stopped its coordinator and worker group. No production merge was launched.
- Issue update: None. This smoke result does not change backfill progress.
- Next action: Continue both backfill roots through completion. Run the full partition verification before a production merge.

### 2026-08-14 01:23 UTC - Backfill passed 98 percent

- Durable output: A direct S3 object count found 164,066 of 166,775 output Parquet shards. This is 98.38 percent complete, with 2,709 shards left.
- Rate: The output gained 240 shards after the 01:10 UTC count of 163,826 shards.
- Active work: East continues `nemotron_cc_v2/medium_high_quality`. RNO continues `nemotron_specialized_v1_2/fact_seeking`. Both source jobs have zero failures and zero preemptions.
- RNO service recovery: TEI service `219` failed because port `12438` was already in use. Reconstructed its exact saved batch request and replayed only that service. The job returned to the running state and entered its first build attempt with the same H100, CPU, memory, disk, image, bundle, and callable files.
- Issue update: None. This service recovery did not stop source output and is not a major run checkpoint.
- Next action: Continue both roots through completion. Verify all 292 source artifacts and all 166,775 shard names before the production merge.

### 2026-08-14 02:01 UTC - Merge uses one embedding per ID

- Merge rule: The normalized ID stream controls output order and row multiplicity. The merge selects the first deduplicated embedding for an ID, or the first fuzzy-duplicate embedding when the deduplicated input does not contain that ID. This uses the agreed invariant that equal IDs have equal embeddings.
- Input I/O: Normalized Parquet reads request only the `id` column. The embedding streams remain sorted and are reduced to one row per ID with Arrow run-end encoding, including duplicate runs that cross Parquet row-group boundaries.
- Lookup: Each normalized row group uses run-end indices to select embeddings in Arrow. The job fails when a normalized ID has no embedding or when either embedding input has an extra ID.
- Counters: The job records dropped duplicate rows separately for the deduplicated and fuzzy-duplicate inputs. It records cross-input overlap as IDs, not rows.
- Verification decision: Removed the automatic post-write `verify_merged_output` reread and its helper subtree. Production output verification will run as a separate operation after the merge.
- Checks: All 15 tests in `tests/datakit/test_harrier_pipeline.py` passed. Changed-file formatting, lint, and type checks passed. An independent review found the hot-path and dead-verifier issues that this revision removes.
- Production state: No production merge was launched.

### 2026-08-14 02:31 UTC - Backfill passed 99 percent

- Root health: The East and RNO roots remain running. Neither root has a failed child job.
- Durable output: A direct S3 object count found 165,212 of 166,775 Parquet shards. This is 99.06 percent complete, with 1,563 shards left.
- Completed artifacts: An exact source-name check found 229 of 292 successful source artifacts. There are 63 source artifacts left.
- East progress: `nemotron_cc_v2/medium_high_quality` reached 3,528 of 3,679 shards. The stage has 32 live workers and zero dead workers.
- RNO progress: RNO continues through the small `penfever-traces` sources. The active source changed several times during this check.
- Issue update: None. Wait for full completion and partition verification before the next issue update.
- Next action: Continue both roots through completion. Then verify all 292 source artifacts and all 166,775 shard names.

### 2026-08-14 02:43 UTC - Twelve RNO TEI services recovered

- Detection: Twelve RNO TEI service jobs became terminal after new batch capacity was admitted. No source job failed.
- Cause: Eleven services failed during task setup because the NVIDIA CUDA package index changed during a mirror sync. Service `440` failed during setup because the `uv` download from GitHub timed out.
- Recovery: Reconstructed each exact saved batch request and attached its stored `_callable.pkl` and `_callable_runner.py` files. Replayed only services `355`, `356`, `360`, `361`, `363`, `375`, `376`, `380`, `388`, `392`, `398`, and `440`.
- Validation: All 512 RNO TEI service jobs returned to the running job state. RNO has 120 running service tasks and 392 building service tasks.
- Pipeline state: RNO continued through the small `penfever-traces` sources. A direct output count found 165,255 of 166,775 shards at 02:36 UTC.
- Incident record: https://echo.oa.dev/wiki/149.
- Issue update: None. Add the incident link to the next major completion update.
- Next action: Confirm that no setup failure repeats. Continue both roots through completion and full shard verification.

### 2026-08-14 03:08 UTC - Backfill reached 99.42 percent

- Root health: The East and RNO roots remain running. Neither root has a failed child job.
- Durable output: A direct S3 object count found 165,805 of 166,775 Parquet shards. This is 99.42 percent complete, with 970 shards left.
- Completed artifacts: An exact status check found 249 of 292 successful source artifacts. There are 43 source artifacts left.
- East progress: `nemotron_cc_code_v1/all` reached 1,344 of 2,098 shards. The stage has 32 live workers and zero dead workers.
- RNO progress: RNO completed `safety_pt/refuseweb/score_5_refusal` at 03:06 UTC and is moving between the remaining small sources.
- Service recovery check: None of the earlier RNO setup failures repeated as a terminal job failure.
- Issue update: None. Wait for full completion and verification before the next major issue update.
- Next action: Continue both roots through completion. Then verify all 292 source artifacts and all 166,775 shard names.

### 2026-08-14 03:24 UTC - RNO replacement partition completed and verified

- Failure: RNO root `v7` failed after its assigned sources finished. An S3 `GetObject` response ended early while StepRunner released the lock for `nemotron_sft/sft_math`. The source status was already `SUCCESS`, and no lock holder remained.
- Recovery: Started root `v8` with the same batch resources, target cluster, source partition, and 512-service request. It created all 512 service jobs and skipped the completed `nemotron_sft/sft_math` artifact and all other completed source artifacts.
- Service event: TEI service `040` hit the known fixed-port collision on port `12080`. The saved request was replayed, but the root completed its cache scan and closed the pool before the replay became necessary.
- Root result: `/rav/harrier-fuzzy-dups-rno-p3of4-20260813-batch-max-v8` succeeded.
- Partition verification: All 73 sources assigned to partition `3/4` have one successful artifact. Their normalized inputs contain 18,274 Parquet shard names, and the fuzzy-duplicate embedding outputs contain the same 18,274 names. There are no missing or extra shards.
- Aggregate progress: A direct S3 count found 166,093 of 166,775 output shards, or 99.59 percent. There are 682 shards left.
- East progress: `nemotron_cc_code_v1/all` reached 1,600 of 2,098 shards with 32 live workers and zero dead workers. East is now the only active partition.
- Issue update: None. Wait for the full 292-source and 166,775-shard verification before the final major update.
- Next action: Continue the East root through its final 43 source artifacts, then run the full output verification.

### 2026-08-14 04:13 UTC - Partial production merge started

- Selection: Chose 128 completed sources with the smallest shard counts. An exact preflight check found one successful canonical Harrier artifact and one successful fuzzy-duplicate artifact for each source. Normalized, canonical, and fuzzy-duplicate Parquet shard names match for all 128 sources.
- Exclusion: The selector and launch command exclude `common-crawl-focus-2026-22`. This source is not part of the job.
- Scope: The selected sources contain 128 shards in total. The job writes to the production `datakit/embed/harrier-all` prefix.
- Job: `/rav/harrier-merge-128-complete-no-focus-20260814-batch-v1` targets East-02 at batch priority. It uses eight concurrent source steps and one shared Zephyr pool with 16 CPU workers.
- Backfill state: The East fuzzy-duplicate root remains active. The durable backfill count reached 166,656 of 166,775 shards before this merge launch.
- Issue update: None. Report the partial merge after it completes, or include it with the final backfill verification.
- Next action: Monitor both jobs. Verify the 128 merged artifacts after the merge succeeds. Complete the full 292-source and 166,775-shard backfill verification after the East root succeeds.
