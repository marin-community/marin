---
topic: hero-pr8480-one-rack-resume
issue: https://github.com/marin-community/marin/issues/8492
description: Validate d6144 EP hero checkpoint restore and host memory on one GB200 rack.
author: rav
---

# One-Rack d6144 Checkpoint Restore: Hero Run Logbook

## Current TL;DR

Preflight confirms that merged `main` is `b3c5884a5c`, the merge commit for #8480. No diagnostic job has launched. The reference production run remains active and is read-only for this test.

## Run Contract

- DRI: rav.
- Goal: Run five d6144 EP hero steps with synthetic data, commit a checkpoint, restart the process, restore step 5, and finish at step 15.
- Stop or escalation criteria: Stop on OOM, non-finite loss, incomplete checkpoint metadata, wrong restore lineage, unexpected access to the production output, task retry, or repeated infrastructure failure. Do not mutate the Iris cluster.
- Issue: [#8492](https://github.com/marin-community/marin/issues/8492).
- W&B: `marin-community/marin_moe`, run ID and display name `hero-pr8480-1rack-resume-20260820-1914`, group `moe-hero-ep-checkpoint-restore`, resume `allow`.
- Output root: `s3://marin-us-east-02a/tmp/ttl=1d/users/rav/hero-checkpoint-restore/hero-pr8480-1rack-resume-20260820-1914`.
- Resume checkpoint root / retention / projected bytes: `<output root>/checkpoints`; two complete checkpoints at steps 5 and 15; about 4,993 GiB each and 9.75 TiB total, estimated from the matching production shape's save report. Bucket lifecycle removes them after one day.
- Canonical export: None. This diagnostic has no durable model export.
- Raw trace / rendezvous / Ray spill: XProf disabled; tracker mirror and all run artifacts stay below the output root; Grug uses no Ray spill path.
- Final step: Phase 1 stops at 5 and intentionally aborts its coordinator after the checkpoint commits. Phase 2 restores step 5 and stops at 15.
- Hardware and topology: `cw-us-east-08a`; 16 tasks with four GB200 GPUs and four JAX ranks each; one NVL72 rack; 850 GiB host-memory request for each task.
- Source: Clean pushed snapshot pending.
- Babysitter: This session owns the monitor. Check once after 120 seconds, then every 570 seconds until each phase reaches its required terminal state.

## Launched Instances

None.

## Event Log

### 2026-08-20 19:14 UTC - Preflight completed

- Status: Ready to validate and snapshot the dedicated phase harness.
- Evidence: Local `HEAD` is `b3c5884a5c`; GitHub reports #8480 merged at that commit. The reference coordinator and its 176-task GPU child are still running under `/rav/hero-12d8b6f0-dee637-coord-resume10`.
- Decision: Use a new run ID, W&B identity, JAX port, coordinator job IDs, and one-day temporary prefix. Preserve the production run as read-only.
- Checkpoint estimate: The matching d6144 checkpoint reports 4,993.38 GiB of serialized state. Two checkpoints fit the hero-run rollback limit and expire after one day.
- Next: Run local validation, commit and push the harness and logbook, then append the exact launch snapshot.
