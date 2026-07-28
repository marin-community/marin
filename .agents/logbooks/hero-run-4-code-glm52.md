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

## Event Log

### 2026-07-27 23:00 PDT - Run contract

- Status: implementation in progress
- Evidence: 202 schedulable GB200 nodes were ready. The live allocation left 756 B200 GPUs free and 92 same-domain TP8 placements; the requested run is capped at 85 TP8 instances (680 GPUs).
- Decision: Use the working two-node Ray/vLLM GLM-5.2 topology from the KernelGym collector. Run an 8-GPU smoke and a 64-GPU pilot before the 680-GPU launch.
- Next: Implement deterministic Parquet row-group sharding, model-cache preparation, resumable output chunks, and progress manifests.
