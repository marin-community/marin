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
