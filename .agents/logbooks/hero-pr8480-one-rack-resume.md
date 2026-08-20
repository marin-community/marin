---
topic: hero-pr8480-one-rack-resume
issue: https://github.com/marin-community/marin/issues/8492
description: Validate d6144 EP hero checkpoint restore and host memory on one GB200 rack.
author: rav
---

# One-Rack d6144 Checkpoint Restore: Hero Run Logbook

## Current TL;DR

The cold GPU child completed five steps and committed the step-5 checkpoint. Its coordinator then failed as planned. The restore-to-15 phase is pending at production priority. Both successful phases run from `891d7d8c60`, which descends from the #8480 merge commit. The reference production run remains read-only.

## Run Contract

- DRI: rav.
- Goal: Run five d6144 EP hero steps with synthetic data, commit a checkpoint, restart the process, restore step 5, and finish at step 15.
- Stop or escalation criteria: Stop on OOM, non-finite loss, incomplete checkpoint metadata, wrong restore lineage, unexpected access to the production output, task retry, or repeated infrastructure failure. Do not mutate the Iris cluster.
- Issue: [#8492](https://github.com/marin-community/marin/issues/8492).
- W&B: Disabled after the first cold attempt exposed missing cluster credentials during tracker initialization. Iris and checkpoint telemetry are the sources of record.
- Output root: `s3://marin-us-east-02a/tmp/ttl=1d/users/rav/hero-checkpoint-restore/hero-pr8480-1rack-resume-20260820-1914`.
- Resume checkpoint root / retention / projected bytes: `<output root>/checkpoints`; two complete checkpoints at steps 5 and 15; about 4,993 GiB each and 9.75 TiB total, estimated from the matching production shape's save report. Bucket lifecycle removes them after one day.
- Canonical export: None. This diagnostic has no durable model export.
- Raw trace / rendezvous / Ray spill: XProf disabled; tracker mirror and all run artifacts stay below the output root; Grug uses no Ray spill path.
- Final step: Phase 1 stops at 5 and intentionally aborts its coordinator after the checkpoint commits. Phase 2 restores step 5 and stops at 15.
- Hardware and topology: `cw-us-east-08a`; 16 tasks with four GB200 GPUs and four JAX ranks each; one NVL72 rack; 850 GiB host-memory request for each task.
- Source: `891d7d8c605af2a625b201180b82770799b70a45`, pushed to `origin/weaver/marin-test-iris-run`; ancestor `b3c5884a5c45d3bf3c84afd2d68eed1655447419` is the #8480 merge commit.
- Babysitter: This session owns the monitor. Check once after 120 seconds, then every 570 seconds until each phase reaches its required terminal state.

## Launched Instances

- Cancelled cold coordinator: [`/rav/hero-pr8480-1rack-resume-20260820-1914-cold`](https://iris.oa.dev/#/job/%2Frav%2Fhero-pr8480-1rack-resume-20260820-1914-cold), submitted at 2026-08-20 19:18 UTC with production priority.
- Successful cold coordinator: [`/rav/hero-pr8480-1rack-resume-20260820-1914-cold-nowandb`](https://iris.oa.dev/#/job/%2Frav%2Fhero-pr8480-1rack-resume-20260820-1914-cold-nowandb), submitted at 2026-08-20 19:23 UTC with production priority.
- Restore coordinator: [`/rav/hero-pr8480-1rack-resume-20260820-1914-resume`](https://iris.oa.dev/#/job/%2Frav%2Fhero-pr8480-1rack-resume-20260820-1914-resume), submitted at 2026-08-20 19:36 UTC with production priority.

## Event Log

### 2026-08-20 19:14 UTC - Preflight completed

- Status: Ready to validate and snapshot the dedicated phase harness.
- Evidence: Local `HEAD` is `b3c5884a5c`; GitHub reports #8480 merged at that commit. The reference coordinator and its 176-task GPU child are still running under `/rav/hero-12d8b6f0-dee637-coord-resume10`.
- Decision: Use a new run ID, W&B identity, JAX port, coordinator job IDs, and one-day temporary prefix. Preserve the production run as read-only.
- Checkpoint estimate: The matching d6144 checkpoint reports 4,993.38 GiB of serialized state. Two checkpoints fit the hero-run rollback limit and expire after one day.
- Next: Run local validation, commit and push the harness and logbook, then append the exact launch snapshot.

### 2026-08-20 19:18 UTC - Cold phase submitted

- Status: Pending on `cw-us-east-08a` at production priority.
- Source: `751f8ab08829cc882929344b5b082996bd6461ee`; changed-file lint and the harness dry plan passed before the snapshot.
- Command contract: Synthetic input, stop after step 5, then raise an intentional coordinator error after Grug completes and flushes the checkpoint.
- Isolation: The phase uses its own coordinator job, run ID, W&B identity, JAX port 32647, and one-day temporary output root.
- Next: Confirm the 16-task GPU child starts, reaches step 5 without retries or non-finite metrics, and commits non-temporary checkpoint metadata.

### 2026-08-20 19:22 UTC - First cold attempt cancelled

- Status: Cancelled the coordinator and its descendants before any training step or checkpoint.
- Evidence: Rank 0 failed in `wandb.init` with `No API key configured`; the other ranks waited in global tracker initialization.
- Decision: Set the diagnostic's `WandbConfig.mode` to `disabled`, snapshot the change, and relaunch from the same isolated one-day path.

### 2026-08-20 19:35 UTC - Cold phase completed and intentionally aborted

- Status: The 16-task GPU child succeeded with zero failures or preemptions. The coordinator then raised the planned error after child completion.
- Training: Five synthetic steps completed with finite loss; step 4 reported loss 11.8.
- Checkpoint: Rank 0 logged `Saved checkpoint` for `<output root>/checkpoints/step-5` at 19:34:53 UTC. The serialized state was 4,993.38 GiB, about 78 GiB per rank.
- Memory: Finelog's 30-second task samples ranged from 754,785 to 782,855 MiB at this point. The maximum was 764.5 GiB against an 850 GiB request. Serialization reported about 40 GiB of host memory in flight per rank.
- Limitation: Local `fsutil` lacks CoreWeave credentials, so the commit marker and the next phase's successful load are the completeness evidence.

### 2026-08-20 19:36 UTC - Restore phase submitted

- Status: Pending on `cw-us-east-08a` at production priority.
- Contract: Load the latest complete checkpoint from the same isolated prefix, start at step 5, run ten more synthetic steps, and stop at step 15.
- Isolation: New coordinator job and JAX port 32649; W&B remains disabled.
- Next: Verify the step-5 load, finite progress through step 15, final checkpoint commit, terminal success, and peak task memory.
