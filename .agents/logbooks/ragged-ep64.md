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

The selected latent-E192 EP hero cannot reach its first ragged step with either JAX 0.11 transport at default settings. The one-shot path fails with an illegal address beginning at device 32. Disabling one-shot avoids that signature, completes the 64-rank clique, and then fails in `ncclGroupEnd()` because every rank runs out of HBM while NCCL allocates its send/recv buffers. `RA2A-003` will reduce the per-peer NCCL FIFO from 4 MiB to 1 MiB while retaining the same executable and scheduler settings.

## Current Baseline

- Date: 2026-08-08.
- Code refs: historical baseline `120ccfbe2`; launch source `67c78093d7a3fb464a339ba168e68d1178d157ac`, which integrates PR #8013's EP lineage while retaining main's FSDP launcher.
- Historical numbers: MHEP-001, E128 x i3072, d6144, top-4, capacity factor 1.0, EP64, 25 steps, 14.9614% median MFU, 2.4099% final drops. This is not shape-matched to the selected E192 latent hero.
- Current baseline: no step metrics. `RA2A-001` fails at the device-32 boundary in one-shot; `RA2A-002` avoids the illegal address but fails on every rank with `ncclCuMemAlloc` out of memory in the send/recv fallback.

## Hypothesis Queue

### Active

- `RA2A-004`: Default 4 MiB per-peer NCCL FIFOs exhaust the small amount of HBM left after loading the selected model. Next test: reduce `NCCL_BUFFSIZE` to 1 MiB with one-shot still disabled.
- `RA2A-002`: XLA latency hiding and four-way collective overlap add scheduling cost or excessive concurrency to dynamic EP collectives. Next test: disable latency hiding, then reduce overlap independently if the baseline completes.
- `RA2A-003`: NCCL NVLS/SHARP settings help only if the ragged lowering uses NCCL. Next test: if reducing the peer FIFO is insufficient, disable NVLS resource allocation because send/recv cannot use NVLink SHARP.

### Blocked

- None.

### Falsified / Dead End

- `RA2A-001`: The default JAX 0.11 ragged flavor does not establish a performance baseline. It fails before step 0 with an illegal GPU address.
- `RA2A-002`: Disabling one-shot alone does not establish a performance baseline. It replaces the illegal address with an explicit NCCL allocation OOM before step 0.

### Promoted

- `RA2A-005`: The first failure begins on host 8, whose local devices are global devices 32 through 35, immediately after the EP64 clique initializes. This matches the prior rank-32 symmetric-memory failure signature. JAX 0.11 also removed the eight-output one-shot limit that made older EP64 runs fall back to NCCL send/recv.
- `RA2A-004`: One-shot-off reaches `ncclGroupEnd()` and all ranks report `ncclCuMemAlloc` out of memory. NVIDIA documents a dedicated `NCCL_BUFFSIZE` FIFO for each send/recv source-destination pair and recommends reducing the 4 MiB default under memory pressure.

## Decision Log

- 2026-08-08: Run arms serially on one interactive-priority rack. Score steps 5 through 24; treat effects below the ±1.57% single-reading resolution measured in #8054 as unresolved without replication.
- 2026-08-08: Pass `--watch-interval 0 --eval-every 0 --profile-steps 0 --no-save-checkpoints` to keep periodic metric, eval, profile, and checkpoint work out of the timing window.
- 2026-08-08: Test NCCL/SHARP controls only after lowering or logs show that the ragged path reaches NCCL. JAX's GPU implementation may use a peer-pointer kernel instead.
- 2026-08-08: Stop the default one-shot baseline after its first synchronized illegal-address failure rather than consume ten automatic retries. Treat reachability as the baseline result and test NCCL send/recv next.
- 2026-08-08: Reduce the send/recv peer FIFO before changing the XLA memory fraction. The failing executable already exceeds XLA's rematerialization target, while `NCCL_BUFFSIZE` directly controls the late allocation that failed.

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

### 2026-08-08 23:05 UTC - RA2A-002 reaches NCCL send/recv and exhausts HBM

- Hypothesis: Disabling the one-shot ragged kernel avoids the device-32 illegal address and produces a measurable EP64 baseline through NCCL send/recv.
- Commit Hash: `a7e2f24475`; process-start configuration added only `--xla_gpu_unsupported_use_ragged_all_to_all_one_shot_kernel=false`.
- Jobs: coordinator `/power/ra2a-002-nccl-sendrecv-20260808-coord`; child `/power/ra2a-002-nccl-sendrecv-20260808-coord/grug-train-ra2a-002-nccl-sendrecv-20260808`; 9.8 MB source bundle; no preemption or retry before failure.
- Result: Failed before step 0. The HLO estimate and rematerialization warning were byte-for-byte identical to `RA2A-001`: 205.91 GiB before rematerialization, 191.82 GiB after, and a 171.87 GiB target. The 64-rank clique initialized, and the device-32 illegal-address signature did not recur. Six seconds later every rank logged `include/alloc.h:320 (ncclCuMemAlloc) NCCL WARN Cuda failure 2 'out of memory'`. The training call raised `JaxRuntimeError: INTERNAL: NCCL operation ncclGroupEnd() failed: unhandled cuda error` from `jit_train_step`.
- Metrics: W&B finished with topology and parameter metadata but no global step, duration, MFU, throughput, or drop metric.
- Terminal state: Iris began an automatic gang retry after the first failure. The coordinator was stopped at 23:05:24 UTC under the launch contract's NCCL-failure stop condition.
- Interpretation: Disabling one-shot changes the failure class without changing the XLA memory plan, strongly isolating the illegal address to the one-shot path. The fallback is viable far enough to enter NCCL send/recv, but the model leaves no room for its late communication allocations. NVIDIA's NCCL documentation states that send/recv allocates a dedicated FIFO for each source-destination pair, sized by `NCCL_BUFFSIZE`; the default is 4 MiB and the documented response to memory constraints is a smaller power-of-two value. At EP64 this can allocate one FIFO for each of 63 peers per rank, unlike the fixed collective path.
- Next action: Keep one-shot disabled and reduce the peer FIFO to 1 MiB. If that remains insufficient, test 256 KiB, then disable unused NVLS resource allocation before changing model memory or scheduling.

### 2026-08-08 23:09 UTC - RA2A-003 launch contract

- Hypothesis: Reducing each NCCL send/recv peer FIFO from the 4 MiB default to 1 MiB leaves enough HBM for the EP64 communicator and produces the first measurable ragged baseline.
- Commit Hash: `a7e2f24475`; this launch changes process-start configuration only.
- Command:

  ```bash
  UV_CACHE_DIR=/tmp/marin-ragged-uv-cache uv run iris \
    --config lib/iris/config/marin.yaml job run --no-wait \
    --enable-extra-resources --target-cluster cw-us-east-08a \
    --priority interactive --cpu 2 --memory 8GB --disk 32GB \
    --timeout 21600 --max-retries 10 \
    --job-name ra2a-003-nccl-buf1m-20260808-coord \
    -e WANDB_API_KEY '<redacted>' \
    -e WANDB_PROJECT marin_moe \
    -e MARIN_PREFIX s3://marin-us-east-02a/marin \
    -e XLA_FLAGS '--xla_gpu_unsupported_use_ragged_all_to_all_one_shot_kernel=false' \
    -e NCCL_BUFFSIZE 1048576 \
    -- python -m experiments.grug.moe_hero_ep.launch \
    --run-id ra2a-003-nccl-buf1m-20260808 \
    --dp-racks 1 --num-steps 25 --flavor ep-ragged \
    --watch-interval 0 --eval-every 0 --profile-steps 0 \
    --no-save-checkpoints --version 2026.08.08 --run
  ```

- Output root: `s3://marin-us-east-02a/marin/grug/ra2a-003-nccl-buf1m-20260808/2026.08.08`.
- Tracking identity: W&B entity `marin-community`, project `marin_moe`, group `moe-hero-ep`, run name and ID `ra2a-003-nccl-buf1m-20260808`, resume policy `allow`.
- Initialization and stop boundary: initialize from scratch and stop after step 25. Checkpoints, eval, watch metrics, and profiling remain disabled.
- Hardware and topology: 16 workers with four GB200 GPUs each on `cw-us-east-08a`; EP64; one data-parallel rack; interactive priority.
- Controlled change: relative to `RA2A-002`, add only `NCCL_BUFFSIZE=1048576`. One-shot remains disabled. PGLE, `cuda_async`, latency hiding, overlap limit 4, disabled command buffers, model, batch, routing, and metric schedule remain fixed.
- Expected memory effect: with one FIFO for each of 63 peers, the documented per-rank FIFO allocation falls from about 252 MiB to about 63 MiB, excluding other NCCL allocations.
- DRI and monitoring: DRI `rjpower`; Codex owns monitoring. Stop on non-finite training, allocator or NCCL failure, exhausted retries, or failure to complete step 25. If it completes, score exactly steps 5 through 24.
- Source bundle and Iris job: pending submission.
- Next action: Commit and push this milestone, publish the allocation result to #8077, then submit the treatment once.
