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

The selected latent-E192 EP hero cannot reach its first ragged step with JAX 0.11's default one-shot kernel. `RA2A-001` completed the 64-rank clique, then hosts 8 through 15 failed with `CUDA_ERROR_ILLEGAL_ADDRESS`, beginning at the device-32 boundary. JAX 0.11 removed the old eight-output limit that had forced EP64 onto NCCL send/recv, making the newly reachable one-shot symmetric-memory path the leading cause. `RA2A-002` will disable that kernel and retain every other runtime setting.

## Current Baseline

- Date: 2026-08-08.
- Code refs: historical baseline `120ccfbe2`; launch source `67c78093d7a3fb464a339ba168e68d1178d157ac`, which integrates PR #8013's EP lineage while retaining main's FSDP launcher.
- Historical numbers: MHEP-001, E128 x i3072, d6144, top-4, capacity factor 1.0, EP64, 25 steps, 14.9614% median MFU, 2.4099% final drops. This is not shape-matched to the selected E192 latent hero.
- Current baseline: `RA2A-001` produced no step metrics. The first executable invocation failed at the device-32 boundary with `CUDA_ERROR_ILLEGAL_ADDRESS` after the 64-rank clique initialized.

## Hypothesis Queue

### Active

- `RA2A-005`: JAX 0.11's newly reachable EP64 one-shot ragged kernel causes the device-32 illegal-address failure. Next test: disable `xla_gpu_unsupported_use_ragged_all_to_all_one_shot_kernel` so the same model uses NCCL send/recv.
- `RA2A-002`: XLA latency hiding and four-way collective overlap add scheduling cost or excessive concurrency to dynamic EP collectives. Next test: disable latency hiding, then reduce overlap independently if the baseline completes.
- `RA2A-003`: NCCL NVLS/SHARP settings help only if the ragged lowering uses NCCL. Next test: inspect compile/runtime evidence before allocating a rack to the setting.
- `RA2A-004`: Allocator and NCCL buffer configuration determines whether the selected model reaches the ragged collective. Next test: classify any failure as XLA-pool, NCCL, compiler support, or coordination before changing memory controls.

### Blocked

- None.

### Falsified / Dead End

- `RA2A-001`: The default JAX 0.11 ragged flavor does not establish a performance baseline. It fails before step 0 with an illegal GPU address.

### Promoted

- `RA2A-005`: The first failure begins on host 8, whose local devices are global devices 32 through 35, immediately after the EP64 clique initializes. This matches the prior rank-32 symmetric-memory failure signature. JAX 0.11 also removed the eight-output one-shot limit that made older EP64 runs fall back to NCCL send/recv.

## Decision Log

- 2026-08-08: Run arms serially on one interactive-priority rack. Score steps 5 through 24; treat effects below the ±1.57% single-reading resolution measured in #8054 as unresolved without replication.
- 2026-08-08: Pass `--watch-interval 0 --eval-every 0 --profile-steps 0 --no-save-checkpoints` to keep periodic metric, eval, profile, and checkpoint work out of the timing window.
- 2026-08-08: Test NCCL/SHARP controls only after lowering or logs show that the ragged path reaches NCCL. JAX's GPU implementation may use a peer-pointer kernel instead.
- 2026-08-08: Stop the default one-shot baseline after its first synchronized illegal-address failure rather than consume ten automatic retries. Treat reachability as the baseline result and test NCCL send/recv next.

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

### 2026-08-08 22:24 UTC - RA2A-001 launch contract

- Hypothesis: The current E192 latent hero completes 25 steps with `ragged_all_to_all`, producing a clean one-rack performance baseline.
- Commit Hash: `67c78093d7a3fb464a339ba168e68d1178d157ac`, pushed to `origin/weaver/hero-run-why-can-t-we-ragged-all` with a clean worktree before this logbook update.
- Command:

  ```bash
  UV_CACHE_DIR=/tmp/marin-ragged-uv-cache uv run iris \
    --config lib/iris/config/marin.yaml job run --no-wait \
    --enable-extra-resources --target-cluster cw-us-east-08a \
    --priority interactive --cpu 2 --memory 8GB --disk 32GB \
    --timeout 21600 --max-retries 10 \
    --job-name ra2a-001-ragged-baseline-20260808-coord \
    -e WANDB_API_KEY '<redacted>' \
    -e WANDB_PROJECT marin_moe \
    -e MARIN_PREFIX s3://marin-us-east-02a/marin \
    -- python -m experiments.grug.moe_hero_ep.launch \
    --run-id ra2a-001-ragged-baseline-20260808 \
    --dp-racks 1 --num-steps 25 --flavor ep-ragged \
    --watch-interval 0 --eval-every 0 --profile-steps 0 \
    --no-save-checkpoints --version 2026.08.08 --run
  ```

- Output root: `s3://marin-us-east-02a/marin/grug/ra2a-001-ragged-baseline-20260808/2026.08.08`.
- Tracking identity: W&B entity `marin-community`, project `marin_moe`, group `moe-hero-ep`, run name and ID `ra2a-001-ragged-baseline-20260808`, resume policy `allow`.
- Initialization: no checkpoint; optimizer and model initialize from scratch.
- Stop boundary: final step 25. Checkpoints, eval, watch metrics, and profiling are disabled.
- Hardware and topology: 16 workers with four GB200 GPUs each on `cw-us-east-08a`; EP64; one data-parallel rack; interactive priority.
- Runtime choice: let Iris choose the JAX coordinator port through the rank-zero port-selection path from #7994. Do not pin the stale README port. Retain the EP runtime defaults from PR #8013: PGLE, `cuda_async`, latency hiding, overlap limit 4, and command buffers disabled.
- DRI and monitoring: DRI `rjpower`; Codex owns monitoring. Check after startup and then continuously through the terminal state. Stop on non-finite training, allocator or NCCL failure, unsupported ragged lowering, exhausted retries, or failure to complete step 25.
- Preflight evidence: `./infra/pre-commit.py --changed-files --fix` passed; 38 focused Grug tests passed with one skipped; eight Levanter eval tests passed. The dry run resolved the selected E192 x i6272 latent-3072, top-4, capacity-factor-1.33 model and the output/tracking identities above.
- Source bundle and Iris job: pending submission.
- Next action: Commit this contract, submit the coordinator once, record its source bundle and canonical job IDs, and monitor to a terminal state.

### 2026-08-08 22:43 UTC - RA2A-001 failed at the device-32 boundary

- Hypothesis: The current E192 latent hero completes 25 steps with JAX 0.11's default ragged lowering.
- Commit Hash: `d4d05861bb4a50c279c9f4f833a93064cd6c8191`.
- Jobs: coordinator `/power/ra2a-001-ragged-baseline-20260808-coord`; child `/power/ra2a-001-ragged-baseline-20260808-coord/grug-train-ra2a-001-ragged-baseline-20260808`; 9.8 MB source bundle; no retries before the first failure.
- Runtime: JAX and jaxlib 0.11.0, NCCL 2.30.7, 16 hosts and 64 GB200 GPUs. PGLE, `cuda_async`, latency hiding, overlap limit 4, and disabled command buffers retained the PR defaults.
- Result: Failed before step 0. XLA spent about seven minutes compiling `jit_train_step` on the slowest hosts and warned that rematerialization reduced its estimate only to 191.82 GiB against a 171.87 GiB target. The 64-rank clique nevertheless initialized successfully at 22:42:32 UTC. At 22:42:34, host 8 reported the first `CUDA_ERROR_ILLEGAL_ADDRESS` while `AsyncExecution` recorded completion events; hosts 8 through 15 then reported the same error. Host 8 owns global devices 32 through 35. W&B recorded the full topology and 546.292B parameter count but no `global_step`, duration, MFU, throughput, or drop metric.
- Terminal state: Iris began its first automatic retry. The coordinator was stopped at 22:43:51 UTC under the launch contract's illegal-address stop condition.
- Interpretation: This is a reachability failure, not a slow performance baseline. The log only exposes the downstream async-event failure, so it does not name the offending kernel. The device-32 boundary matches the earlier symmetric-memory rank-32 incident, while JAX 0.11 newly allows the one-shot NCCL-LSA ragged kernel at EP64. JAX 0.10.1's eight-output cap would instead have selected the private NCCL send/recv fallback. Disabling one-shot is therefore the highest-confidence treatment and also converts later NCCL knobs into meaningful tests.
- Next action: Run `RA2A-002` with only the one-shot ragged kernel disabled.

### 2026-08-08 22:46 UTC - RA2A-002 launch contract

- Hypothesis: Disabling JAX 0.11's one-shot ragged kernel avoids the device-32 symmetric-memory fault and produces a measurable EP64 baseline through the private NCCL send/recv fallback.
- Commit Hash: `d4d05861bb4a50c279c9f4f833a93064cd6c8191`; this launch changes process-start configuration only.
- Command:

  ```bash
  UV_CACHE_DIR=/tmp/marin-ragged-uv-cache uv run iris \
    --config lib/iris/config/marin.yaml job run --no-wait \
    --enable-extra-resources --target-cluster cw-us-east-08a \
    --priority interactive --cpu 2 --memory 8GB --disk 32GB \
    --timeout 21600 --max-retries 10 \
    --job-name ra2a-002-nccl-sendrecv-20260808-coord \
    -e WANDB_API_KEY '<redacted>' \
    -e WANDB_PROJECT marin_moe \
    -e MARIN_PREFIX s3://marin-us-east-02a/marin \
    -e XLA_FLAGS '--xla_gpu_unsupported_use_ragged_all_to_all_one_shot_kernel=false' \
    -- python -m experiments.grug.moe_hero_ep.launch \
    --run-id ra2a-002-nccl-sendrecv-20260808 \
    --dp-racks 1 --num-steps 25 --flavor ep-ragged \
    --watch-interval 0 --eval-every 0 --profile-steps 0 \
    --no-save-checkpoints --version 2026.08.08 --run
  ```

- Output root: `s3://marin-us-east-02a/marin/grug/ra2a-002-nccl-sendrecv-20260808/2026.08.08`.
- Tracking identity: W&B entity `marin-community`, project `marin_moe`, group `moe-hero-ep`, run name and ID `ra2a-002-nccl-sendrecv-20260808`, resume policy `allow`.
- Initialization and stop boundary: initialize from scratch and stop after step 25. Checkpoints, eval, watch metrics, and profiling remain disabled.
- Hardware and topology: 16 workers with four GB200 GPUs each on `cw-us-east-08a`; EP64; one data-parallel rack; interactive priority.
- Controlled change: add only `--xla_gpu_unsupported_use_ragged_all_to_all_one_shot_kernel=false`. The EP runtime configuration will append its existing latency-hiding, overlap-limit-4, and command-buffer flags while retaining PGLE and `cuda_async`.
- DRI and monitoring: DRI `rjpower`; Codex owns monitoring. Stop on non-finite training, allocator or NCCL failure, exhausted retries, or failure to complete step 25. If it completes, score exactly steps 5 through 24.
- Source bundle and Iris job: pending submission.
- Next action: Commit and push this milestone, publish the failed baseline to #8077, then submit the treatment once.
