---
topic: hero-run-4-code-glm52
issue: https://github.com/marin-community/marin/issues/7697
description: Generate GLM-5.2 responses for every hero_run_4_code prompt.
author: will-held
---

# hero_run_4_code GLM-5.2: Hero Run Logbook

## Run Contract

- DRI: Will Held
- Goal: Generate one GLM-5.2 FP8 response for all 959,216 `instruction_seed` prompts.
- Stop/escalation criteria: Stop on corrupt output, repeated deterministic failure, or sustained throughput more than 30% below the pilot baseline. Batch preemption is recoverable.
- Issue: https://github.com/marin-community/marin/issues/7697
- W&B: not applicable; progress is recorded in S3 manifests and Iris logs.
- Output root: `s3://marin-us-east-02a/marin/rollouts/glm-5.2/hero-run-4-code-20260727`
- Final step: 959,216 complete response records.
- Checkpoint policy: Immutable compressed JSONL chunks retained permanently. Progress manifests are updated after each chunk; restart skips complete chunks.

## Launched Instances

### 2026-07-27 23:48 PDT - `/held/hero-run-4-code-glm52-cache-20260727`

- Command: `uv run iris --controller-url http://127.0.0.1:18100 job run --no-wait --job-name hero-run-4-code-glm52-cache-20260727 --priority batch --enable-extra-resources --cpu 8 --memory 64GB --disk 100GB --extra marin-core:cpu --timeout 259200 -- python -m experiments.rollout_data.collect_hero_run_4_code_glm52 prepare --output-path s3://marin-us-east-02a/marin/rollouts/glm-5.2/hero-run-4-code-20260727`
- Git SHA: `5f4f66c4a27f4e2c5ac1fec56dbeb76bd82f01f0`
- Dirty tree: no
- Source bundle/container: Iris workspace bundle from the Git SHA above; default Iris CPU task image.
- Hardware/topology: 8 CPU, 64 GB memory, 100 GB disk; Iris batch priority.
- `initialize_from`: not applicable
- Final step: regional cache `.cache_complete` plus `model-cache.json` at the output root.
- Checkpoint policy: not applicable; the pinned model mirror has a 30-day regional cache TTL.
- Babysitter/check cadence: 2-minute startup check, then 15 minutes.

## Event Log

### 2026-07-27 23:00 PDT - Run contract

- Status: implementation in progress
- Evidence: 202 schedulable GB200 nodes were ready. The live allocation left 756 B200 GPUs free and 92 same-domain TP8 placements; the requested run is capped at 85 TP8 instances (680 GPUs).
- Decision: Use the working two-node Ray/vLLM GLM-5.2 topology from the KernelGym collector. Run an 8-GPU smoke and a 64-GPU pilot before the 680-GPU launch.
- Next: Implement deterministic Parquet row-group sharding, model-cache preparation, resumable output chunks, and progress manifests.

### 2026-07-27 23:47 PDT - Implementation ready

- Status: ready for model-cache preparation
- Evidence: Source commit `7141baeee6a97d467fd806402270b51cd5bcd675` is pushed in PR #7698. `tests/experiment/test_collect_hero_run_4_code_glm52.py` passes, Pyrefly reports no errors, and the branch lint-catalog review reports no findings.
- Command: `uv run iris --controller-url http://127.0.0.1:18100 job run --no-wait --job-name hero-run-4-code-glm52-cache-20260727 --priority batch --enable-extra-resources --cpu 8 --memory 64GB --disk 100GB --extra marin-core:cpu --timeout 259200 -- python -m experiments.rollout_data.collect_hero_run_4_code_glm52 prepare --output-path s3://marin-us-east-02a/marin/rollouts/glm-5.2/hero-run-4-code-20260727`
- Config: Model `zai-org/GLM-5.2-FP8` at revision `ba978f7d347eaf65d22f1a86833408afdb953541`; regional model cache TTL 30 days.
- Decision: Populate the distributed regional model cache before reserving GPUs.
- Next: Submit the cache job, verify `model-cache.json`, then launch an 8-GPU smoke.

### 2026-07-28 06:45 PDT - Partial cache recovery

- Status: blocked on the original cache lease
- Evidence: The first cache attempt uploaded 97 of 141 weight shards but did not write `.cache_complete` or `model-cache.json`. No production or smoke output exists. The original east-08 worker still refreshes its distributed lease while the object count remains unchanged.
- Command: `uv run iris --controller-url http://127.0.0.1:18100 job run --no-wait --job-name hero-run-4-code-glm52-cache-resume-20260728 --priority batch --enable-extra-resources --cpu 8 --memory 64GB --disk 100GB --extra marin-core:cpu --timeout 259200 -- python -m experiments.rollout_data.collect_hero_run_4_code_glm52 prepare --output-path s3://marin-us-east-02a/marin/rollouts/glm-5.2/hero-run-4-code-20260727`
- Config: Source commit `c62eda3fb`; preserve existing cache files and download only missing files; batch priority; no GPUs.
- Decision: Keep the active lease intact. The resume job and automatic 8-GPU smoke launcher remain queued.
- Next: Terminate the stalled original cache task through the east-08 control plane, then let the retry fetch the remaining 44 weight shards.
