---
topic: ragged-ep64
issue: https://github.com/marin-community/marin/issues/8077
description: One-rack GB200 investigation of EP64 ragged all-to-all correctness and performance
author: rjpower
---

# Ragged EP64: Task Logbook

## Scope

- Goal: Reproduce the selected EP hero with `ragged_all_to_all`, identify the cause of its performance or failures, and test the obvious EP-specific runtime and NCCL controls.
- Primary metrics: median step duration and MFU over steps 5 through 24, token throughput, routing-drop fraction, peak HBM, and terminal correctness.
- Constraints: one GB200 rack, interactive priority, serialized arms, no periodic watch/eval/profile steps, no checkpoints, and no Iris cluster lifecycle changes.
- Coordinating issue/PR: experiment [#8077](https://github.com/marin-community/marin/issues/8077); source PR [#8013](https://github.com/marin-community/marin/pull/8013).
- Experiment prefix: `RA2A`.

## Current TL;DR

The current selected latent-E192 EP hero has no ragged baseline. Historical run MHEP-001 completed the older E128 full-width shape at 14.9614% median MFU and 2.4099% final assignment drops. The first new run will change only the transport to `ragged_all_to_all` on PR #8013, disable periodic metric/eval/profile work, and score steps 5 through 24.

## Current Baseline

- Date: 2026-08-08.
- Code refs: historical baseline `120ccfbe2`; candidate source PR #8013 head `b80b35887c6e7d523a5440f63c80aba21092f2d1` before integration with this research branch.
- Historical numbers: MHEP-001, E128 x i3072, d6144, top-4, capacity factor 1.0, EP64, 25 steps, 14.9614% median MFU, 2.4099% final drops. This is not shape-matched to the selected E192 latent hero.
- Current baseline: pending `RA2A-001`.

## Hypothesis Queue

### Active

- `RA2A-001`: The current PR #8013 ragged flavor completes 25 steps and establishes a usable one-rack baseline. Next test: selected E192 latent hero with only `--flavor ep-ragged` changed.
- `RA2A-002`: XLA latency hiding and four-way collective overlap add scheduling cost or excessive concurrency to dynamic EP collectives. Next test: disable latency hiding, then reduce overlap independently if the baseline completes.
- `RA2A-003`: NCCL NVLS/SHARP settings help only if the ragged lowering uses NCCL. Next test: inspect compile/runtime evidence before allocating a rack to the setting.
- `RA2A-004`: Allocator and NCCL buffer configuration determines whether the selected model reaches the ragged collective. Next test: classify any failure as XLA-pool, NCCL, compiler support, or coordination before changing memory controls.

### Blocked

- None.

### Falsified / Dead End

- None.

### Promoted

- None.

## Decision Log

- 2026-08-08: Run arms serially on one interactive-priority rack. Score steps 5 through 24; treat effects below the ±1.57% single-reading resolution measured in #8054 as unresolved without replication.
- 2026-08-08: Pass `--watch-interval 0 --eval-every 0 --profile-steps 0 --no-save-checkpoints` to keep periodic metric, eval, profile, and checkpoint work out of the timing window.
- 2026-08-08: Test NCCL/SHARP controls only after lowering or logs show that the ragged path reaches NCCL. JAX's GPU implementation may use a peer-pointer kernel instead.

## Negative Results Index

- None.

## Entry Log

### 2026-08-08 22:12 UTC - Investigation opened

- Hypothesis: The selected EP hero can reproduce the historical ragged behavior on the current runtime.
- Commit Hash: PR #8013 head `b80b35887c6e7d523a5440f63c80aba21092f2d1`; launch snapshot pending.
- Command: Pending preflight and issue creation.
- Config: one GB200 rack, EP64, selected E192 x i6272 latent-3072 hero, top-4, capacity factor 1.33, 25 steps, interactive priority, no watch/eval/profile/checkpoint work.
- Result: Historical evidence recovered. MHEP-001 completed a different E128 shape at 14.9614% median MFU and 2.4099% final drops. PR #8013 reports two newer attempts that failed before a valid ragged measurement, one from OOM and one from repeated Gloo startup timeouts.
- Interpretation: Ragged all-to-all is not universally broken. The investigation must first separate current-shape/runtime failures from steady-state collective performance.
- Next action: Create the coordinating experiment issue, integrate PR #8013 onto the current main lineage, validate the launch contract, and submit `RA2A-001`.
