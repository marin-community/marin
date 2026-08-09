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

The selected latent-E192 EP hero cannot reach its first ragged step with either JAX 0.11 transport at default settings. The one-shot path fails with an illegal address beginning at device 32. Disabling one-shot avoids that signature but exhausts HBM while allocating 4 MiB peer FIFOs. Reducing them to 1 MiB removes the explicit OOM and creates the 64-rank communicator, then every sampled GPU spins at 100% utilization and about 203-232 W without completing step 0. `RA2A-004` will test NVIDIA's documented `NCCL_LAUNCH_MODE=GROUP` workaround for multi-device, single-process hangs.

## Current Baseline

- Date: 2026-08-08.
- Code refs: historical baseline `120ccfbe2`; launch source `67c78093d7a3fb464a339ba168e68d1178d157ac`, which integrates PR #8013's EP lineage while retaining main's FSDP launcher.
- Historical numbers: MHEP-001, E128 x i3072, d6144, top-4, capacity factor 1.0, EP64, 25 steps, 14.9614% median MFU, 2.4099% final drops. This is not shape-matched to the selected E192 latent hero.
- Current baseline: no step metrics. `RA2A-001` fails at the device-32 boundary in one-shot; `RA2A-002` avoids the illegal address but fails on every rank with `ncclCuMemAlloc` out of memory; `RA2A-003` removes that allocator failure but spins in the first executable call without optimizer progress.

## Hypothesis Queue

### Active

- `RA2A-006`: One process manages four GPUs per worker, and the reachable send/recv path now hangs after its 64-rank communicator appears. Next test: retain the 1 MiB FIFO and set `NCCL_LAUNCH_MODE=GROUP`, NVIDIA's documented workaround for this topology and symptom.
- `RA2A-002`: XLA latency hiding and four-way collective overlap add scheduling cost or excessive concurrency to dynamic EP collectives. Next test: disable latency hiding, then reduce overlap independently if the baseline completes.
- `RA2A-003`: NCCL NVLS/SHARP settings help only if the ragged lowering uses NCCL. Next test: if launch grouping does not restore progress, disable unused NVLS resource allocation because send/recv cannot use NVLink SHARP.

### Blocked

- None.

### Falsified / Dead End

- `RA2A-001`: The default JAX 0.11 ragged flavor does not establish a performance baseline. It fails before step 0 with an illegal GPU address.
- `RA2A-002`: Disabling one-shot alone does not establish a performance baseline. It replaces the illegal address with an explicit NCCL allocation OOM before step 0.
- `RA2A-004`: A 1 MiB peer FIFO does not establish a performance baseline. It removes the allocation OOM but leaves all ranks in a low-power GPU spin before step 0.

### Promoted

- `RA2A-005`: The first failure begins on host 8, whose local devices are global devices 32 through 35, immediately after the EP64 clique initializes. This matches the prior rank-32 symmetric-memory failure signature. JAX 0.11 also removed the eight-output one-shot limit that made older EP64 runs fall back to NCCL send/recv.
- `RA2A-004`: One-shot-off reaches `ncclGroupEnd()` and all ranks report `ncclCuMemAlloc` out of memory. NVIDIA documents a dedicated `NCCL_BUFFSIZE` FIFO for each send/recv source-destination pair and recommends reducing the 4 MiB default under memory pressure.
- `RA2A-006`: With 1 MiB FIFOs, two distant hosts show identical 100%-utilization, 203-232 W GPU spin while all Python main threads wait inside the first `pjit`. NVIDIA specifically recommends `NCCL_LAUNCH_MODE=GROUP` for hangs when one process manages multiple Blackwell GPUs.

## Decision Log

- 2026-08-08: Run arms serially on one interactive-priority rack. Score steps 5 through 24; treat effects below the ±1.57% single-reading resolution measured in #8054 as unresolved without replication.
- 2026-08-08: Pass `--watch-interval 0 --eval-every 0 --profile-steps 0 --no-save-checkpoints` to keep periodic metric, eval, profile, and checkpoint work out of the timing window.
- 2026-08-08: Test NCCL/SHARP controls only after lowering or logs show that the ragged path reaches NCCL. JAX's GPU implementation may use a peer-pointer kernel instead.
- 2026-08-08: Stop the default one-shot baseline after its first synchronized illegal-address failure rather than consume ten automatic retries. Treat reachability as the baseline result and test NCCL send/recv next.
- 2026-08-08: Reduce the send/recv peer FIFO before changing the XLA memory fraction. The failing executable already exceeds XLA's rematerialization target, while `NCCL_BUFFSIZE` directly controls the late allocation that failed.
- 2026-08-08: Do not reduce the FIFO below 1 MiB after `RA2A-003`: allocation already succeeded, and a smaller FIFO cannot explain or repair the new low-power collective spin. Test grouped NCCL launch next.

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

### 2026-08-08 23:20 UTC - Background research brief and throughput arm queue

- Effort: medium. Stop rule: stop external and internal foraging when new sources no longer change the first six single-variable arms.
- Question: Once NCCL send/recv reaches step 0, which EP-specific controls could plausibly raise the selected ragged hero above 20% MFU?
- Current context: the shape-matched fixed-all-to-all reference reaches about 23.47% median MFU but drops about 5.28% of routed assignments. The older, smaller EP64 ragged result reached 14.96% MFU. The target therefore likely requires a large scheduling or transport correction, not a sub-resolution collective tweak.
- Internal prior work: the EP4 profile in [#7279](https://github.com/marin-community/marin/issues/7279#issuecomment-5064115832) measured 472.7 ms in 512 ragged send/recv kernels and found 6.96% more compute plus 11.56% more stall time than ring; communication time alone did not explain the gap. The FSDP sweep in [#8054](https://github.com/marin-community/marin/issues/8054) found latency hiding harmful and NCCL protocol/NVLS changes small, but that evidence is only a ranking prior because FSDP uses AG/RS/AR rather than ragged P2P.
- External prior art: NVIDIA's [JAX Toolbox GPU guide](https://docs.nvidia.com/jax-toolbox/performance-profiling/gpu-performance) recommends `NCCL_PROTO`, local memcpy P2P, NCCL communicator/channel controls, and partial CUDA graphs. It says NCCL user buffers support send/recv, but require a separate memory pool. It also says NVLS accelerates reduction collectives, and explicitly warns that `CUDA_DEVICE_MAX_CONNECTIONS=1` is slower on Blackwell.
- Negative leads: `NCCL_ALGO=NVLS,Ring` and SHARP do not change NCCL send/recv, so they are not direct ragged arms. AG/RS/AR combine thresholds are FSDP controls. `xla_gpu_enable_pipelined_p2p` documents collective-permute patterns, not ragged-all-to-all. `NCCL_LAUNCH_MODE=GROUP` is a hang workaround, not a throughput claim. The one-shot kernel remains excluded until its device-32 illegal-address fault is fixed.

#### Ranked experiments after a stable baseline

1. Find the largest non-OOM FIFO: bracket `NCCL_BUFFSIZE` at 2 MiB after a 1 MiB success, or fall to 256 KiB after a 1 MiB OOM. Score the largest reachable value; falsifier is no step-time improvement or another allocation OOM.
2. Disable the latency-hiding scheduler with overlap otherwise unchanged. The EP default forces it on, while prior Marin measurements show ragged has excess stall/compute and FSDP shows the scheduler can lose performance. A result below the baseline falsifies scheduler interference.
3. Keep latency hiding on and reduce `--xla_gpu_experimental_parallel_collective_overlap_limit` from 4 to 1. This tests whether concurrent dynamic P2P operations consume HBM or serialize badly without conflating the result with disabling scheduling entirely.
4. Disable automatic PGLE. Every current rank reports an empty PGLE trace, so it has supplied no measured latency model. This arm tests whether its profiling/recompilation path is contaminating the short run.
5. Set `NCCL_PROTO=Simple`. Large P2P messages may benefit from avoiding LL protocol overhead. Prior FSDP evidence was neutral, but ragged send/recv is a different NCCL operation.
6. Enable `--xla_gpu_use_memcpy_local_p2p=true`. This moves only the three same-process peers per GPU to copy engines, so the upside is bounded and the arm follows global scheduler/protocol tests.
7. Sweep `--xla_gpu_nccl_p2p_max_nchannels` serially, starting at 1 and then 4 only if 1 changes the result. Sixty-three peer relationships per rank may otherwise reserve too many SMs or launch resources.
8. Test `--xla_gpu_enable_nccl_per_stream_comms=false` and communicator splitting as memory controls. Promote only if they permit a larger FIFO or materially improve peak HBM; they are not expected to create a large throughput win alone.
9. Test `--xla_gpu_enable_command_buffer=FUSION,CUSTOM_CALL` once. NVIDIA recommends this on B200 for launch overhead, but Marin has a multi-node command-buffer crash record in #5675, so stop immediately on graph-instantiation or illegal-address failure.
10. Test NCCL user buffers only after reserving a bounded collective arena and preserving NCCL headroom. This is the remaining zero-copy send/recv path, but its extra pool makes it a high-OOM-risk arm.
11. Test the analytical SOL estimator or O1 only if latency hiding remains enabled and a profile shows poor collective scheduling. SOL primarily models collectives, not private send/recv, and the FSDP sweep found this family below the resolution floor or harmful.
12. Test explicit NVLS/SHARP only if a profile shows non-ragged AR/AG/RS dominating the remaining step. It cannot directly accelerate the defining ragged P2P traffic.

- Success rule: a single 20-step window above 20% median MFU is a candidate win; effects below the ±1.57% reading resolution from #8054 remain unresolved without replication.
- Source ledger: NVIDIA JAX Toolbox GPU guide (official docs, current 2026-08-08); NVIDIA NCCL 2.30.7 environment guide (official docs); Marin issues #7012, #7279, #8054; current RA2A-001/002 logs and W&B records.
- Next action: finish the FIFO reachability bracket, publish the ranked queue to #8077, then serialize the highest-signal arms against the first stable baseline.

### 2026-08-08 23:36 UTC - RA2A-003 removes the OOM and exposes a collective spin

- Hypothesis: A 1 MiB NCCL peer FIFO leaves enough HBM for the private EP64 send/recv fallback to complete 25 steps.
- Commit Hash: `58ce5d056b`; process-start configuration retained one-shot-off and added only `NCCL_BUFFSIZE=1048576`.
- Jobs: coordinator `/power/ra2a-003-nccl-buf1m-20260808-coord`; child `/power/ra2a-003-nccl-buf1m-20260808-coord/grug-train-ra2a-003-nccl-buf1m-20260808`; 16 of 16 workers remained running with no retry, preemption, or logged exception.
- Result: Failed before step 0 by loss of progress. Unlike `RA2A-002`, no rank logged `ncclCuMemAlloc`, another NCCL warning, a CUDA error, or a traceback. Rank 0 and rank 15 Python main threads both waited in the first `pjit` call. At 23:33-23:36 UTC, all four GPUs on each sampled host reported 100% utilization, 185,173-185,195 MiB allocated, and only 203-232 W. The 1,200 W GB200 power limit makes this the established low-power collective-spin signature rather than useful compute or CPU compilation.
- RAS: periodic snapshots at 23:22 and 23:32 UTC succeeded in about 0.21 s. The later snapshot found five valid communicators, including a 64-rank communicator, with zero invalid or omitted records and no collection timeout. Sparse periodic RAS reported no rank-count mismatch, so it did not classify the hang; it proves the RAS service and communicator metadata remained responsive.
- Metrics: Levanter phase stayed `initializing=0` on every rank. W&B contained topology and parameter metadata but no global step, duration, MFU, throughput, or drop metric after more than 20 minutes in the first executable call.
- Terminal state: the coordinator was stopped at 23:36:21 UTC under the launch contract's no-progress condition, before Iris retried the same configuration.
- Interpretation: 1 MiB is enough to cross the explicit allocation failure, so a smaller 256 KiB FIFO attacks the wrong failure and is likely slower. The topology is exactly NVIDIA's documented Blackwell hang case: one process owns four devices while distributed communication is active. Group launch is now a reachability control, not a speculative performance tweak.
- Next action: Retain the 1 MiB FIFO and test only `NCCL_LAUNCH_MODE=GROUP`.

### 2026-08-08 23:38 UTC - RA2A-004 launch contract

- Hypothesis: `NCCL_LAUNCH_MODE=GROUP` prevents the multi-device process synchronization hang and produces the first measurable NCCL send/recv baseline.
- Commit Hash: `04a50faef8`; this launch changes process-start configuration only.
- Command:

  ```bash
  UV_CACHE_DIR=/tmp/marin-ragged-uv-cache uv run iris \
    --config lib/iris/config/marin.yaml job run --no-wait \
    --enable-extra-resources --target-cluster cw-us-east-08a \
    --priority interactive --cpu 2 --memory 8GB --disk 32GB \
    --timeout 21600 --max-retries 10 \
    --job-name ra2a-004-nccl-group-20260808-coord \
    -e WANDB_API_KEY '<redacted>' \
    -e WANDB_PROJECT marin_moe \
    -e MARIN_PREFIX s3://marin-us-east-02a/marin \
    -e XLA_FLAGS '--xla_gpu_unsupported_use_ragged_all_to_all_one_shot_kernel=false' \
    -e NCCL_BUFFSIZE 1048576 \
    -e NCCL_LAUNCH_MODE GROUP \
    -- python -m experiments.grug.moe_hero_ep.launch \
    --run-id ra2a-004-nccl-group-20260808 \
    --dp-racks 1 --num-steps 25 --flavor ep-ragged \
    --watch-interval 0 --eval-every 0 --profile-steps 0 \
    --no-save-checkpoints --version 2026.08.08 --run
  ```

- Output root: `s3://marin-us-east-02a/marin/grug/ra2a-004-nccl-group-20260808/2026.08.08`.
- Tracking identity: W&B entity `marin-community`, project `marin_moe`, group `moe-hero-ep`, run name and ID `ra2a-004-nccl-group-20260808`, resume policy `allow`.
- Initialization and stop boundary: initialize from scratch and stop after step 25. Checkpoints, eval, watch metrics, and profiling remain disabled.
- Hardware and topology: 16 workers with four GB200 GPUs each on `cw-us-east-08a`; one process per worker; EP64; one data-parallel rack; interactive priority.
- Controlled change: relative to `RA2A-003`, add only `NCCL_LAUNCH_MODE=GROUP`. One-shot remains disabled and the 1 MiB FIFO remains fixed. PGLE, `cuda_async`, latency hiding, overlap limit 4, disabled command buffers, model, batch, routing, and metric schedule remain unchanged.
- DRI and monitoring: DRI `rjpower`; Codex owns monitoring. Stop on allocator or CUDA failure, another low-power no-progress interval after communicator initialization, exhausted retries, or failure to complete step 25. If it completes, score exactly steps 5 through 24.
- Source bundle and Iris job: pending submission.
- Next action: Commit and push this milestone, publish `RA2A-003` to #8077, then submit this one-variable treatment once.

### 2026-08-08 23:48 UTC - RA2A-004 grouped launch does not break the spin

- Hypothesis: `NCCL_LAUNCH_MODE=GROUP` prevents the multi-device-process synchronization hang and produces the first measurable NCCL send/recv baseline.
- Commit Hash: `4b50a95a18`; the submitted source bundle retained one process per four-GPU worker.
- Jobs: coordinator `/power/ra2a-004-nccl-group-20260808-coord`; child `/power/ra2a-004-nccl-group-20260808-coord/grug-train-ra2a-004-nccl-group-20260808`; all 16 workers remained running with no retry, preemption, or logged exception.
- Result: Failed before step 0 by the same loss-of-progress signature as `RA2A-003`. Every rank entered the training executable at 23:41:14 UTC, but none completed a first step. At 23:47 UTC, all four GPUs on task 0 reported 100% utilization, 185,173-185,195 MiB allocated, and only 203-223 W against a 1,200 W limit.
- Thread evidence: three task-1 dumps at 23:46:49, 23:46:56, and 23:47:23 UTC held the Python main thread at `pxla.py:420` below `_pjit_call_impl_python`. In the exact JAX 0.11.0 source, line 420 is `self.xla_executable.execute_sharded(input_bufs)`, so the process had completed lowering/compilation and crossed into executable dispatch. The Python-only dump cannot identify the native CUDA/NCCL frame, but it rules against the initial interpretation that the process was merely still compiling.
- Metrics: W&B produced no step duration, MFU, throughput, or routing-drop observation. No rank logged a CUDA error, NCCL warning, allocation failure, or traceback.
- Terminal state: the coordinator was stopped at 23:47:34 UTC under the launch contract's no-progress condition.
- Interpretation: grouped NCCL launch did not produce a step in the four-device JAX process. The NVIDIA guide's stronger topology workaround is one process per device, and this topology is already known internally to reduce these stalls. All subsequent EP arms will use that mode; no further rack time will be spent adjudicating the obsolete topology.
- Profiling command: `uv run iris --cluster=marin process profile threads --target <task-id>` captures the on-demand dumps used above.
- Queue correction: remove `--xla_gpu_use_memcpy_local_p2p=true` from the ranked queue because each JAX process will own only one local device. `NCCL_LAUNCH_MODE=GROUP` also leaves the performance queue; it was a reachability control for the discarded topology.
- Next action: Change the EP launchers to set `processes_per_task` equal to the task's GPU count, validate and snapshot the source, then run `RA2A-005` with one-shot off and the 1 MiB FIFO but without `NCCL_LAUNCH_MODE`.

### 2026-08-08 23:48 UTC - RA2A-005 process-per-GPU launch contract

- Hypothesis: one JAX process per GB200 GPU avoids the synchronized first-step stall seen when one process drives four devices, producing the first measurable ragged NCCL send/recv baseline.
- Source change: `experiments/grug/moe_hero_ep/launch.py` now uses four processes per four-GPU task. The small-scale EP launcher derives its process count from the selected target so eight-GPU H100 tasks continue to satisfy the same one-process-per-GPU invariant.
- Controlled change: relative to `RA2A-003`, change only the process topology from one process per task to four. Retain one-shot-off and `NCCL_BUFFSIZE=1048576`; do not set `NCCL_LAUNCH_MODE`. Model, routing, batch, rack, runtime defaults, and metric schedule remain fixed.
- Validation: `uv run pytest tests/test_moe_hero_ep.py -q` passed all 22 tests. The required changed-file pre-commit pass completed successfully after downloading its pinned tools outside the restricted network sandbox.
- Hardware and topology: 16 workers with four GB200 GPUs and four JAX processes each on `cw-us-east-08a`; EP64; one data-parallel rack; interactive priority.
- Initialization and stop boundary: initialize from scratch and stop after step 25. Checkpoints, eval, watch metrics, and profiling remain disabled. Score exactly steps 5 through 24 if the run completes.
- DRI and monitoring: DRI `rjpower`; Codex owns monitoring. Stop on non-finite training, allocator or NCCL failure, exhausted retries, or a repeated low-power no-progress interval after communicator initialization.
- Source commit: `ad3e341524` on `weaver/hero-run-why-can-t-we-ragged-all`; pushed before submission. Source bundle size: 9.8 MB.
- Command:

  ```bash
  UV_CACHE_DIR=/tmp/marin-ragged-uv-cache uv run iris \
    --config lib/iris/config/marin.yaml job run --no-wait \
    --enable-extra-resources --target-cluster cw-us-east-08a \
    --priority interactive --cpu 2 --memory 8GB --disk 32GB \
    --timeout 21600 --max-retries 10 \
    --job-name ra2a-005-process-per-gpu-20260808-coord \
    -e WANDB_API_KEY '<redacted>' \
    -e WANDB_PROJECT marin_moe \
    -e MARIN_PREFIX s3://marin-us-east-02a/marin \
    -e XLA_FLAGS '--xla_gpu_unsupported_use_ragged_all_to_all_one_shot_kernel=false' \
    -e NCCL_BUFFSIZE 1048576 \
    -- python -m experiments.grug.moe_hero_ep.launch \
    --run-id ra2a-005-process-per-gpu-20260808 \
    --dp-racks 1 --num-steps 25 --flavor ep-ragged \
    --watch-interval 0 --eval-every 0 --profile-steps 0 \
    --no-save-checkpoints --version 2026.08.08 --run
  ```

- Output root: `s3://marin-us-east-02a/marin/grug/ra2a-005-process-per-gpu-20260808/2026.08.08`.
- Tracking identity: W&B entity `marin-community`, project `marin_moe`, group `moe-hero-ep`, run name and ID `ra2a-005-process-per-gpu-20260808`, resume policy `allow`.
- Iris coordinator: `/power/ra2a-005-process-per-gpu-20260808-coord`, submitted at 23:51:41 UTC. Child training job pending coordinator dispatch.

### 2026-08-09 01:06 UTC - EP16 screening plan while the rack is production-gated

- Capacity observation: 197 of 202 GB200 nodes had four requested GPUs, one node had one requested GPU, and four whole nodes were free. The rack job remained queued after three production preemptions. Reducing its 120-CPU request would not make 16 whole nodes available.
- Purpose: use four GB200 nodes as an EP16 relative screen while `RA2A-005` waits. With 1,048,576 tokens per step, every EP16 sender shard carries 65,536 tokens, matching the d6144 hero. E192 top-4 therefore matches its routing-cell load while the d768 model keeps the screen cheap.
- Limitation: EP16 has 15 peers per rank rather than EP64's 63. It cannot reproduce the rack's peer-FIFO memory pressure or establish the final MFU. Promote only large relative changes to EP64.
- Source change: add the `gb200-4node` target and `ep-ragged` flavor to the existing small-scale EP launcher. The target retains one JAX process per GPU.
- Validation: `uv run pytest tests/test_moe_hero_ep.py -q` passed all 22 tests; the required changed-file pre-commit gate passed.
- Baseline configuration: d768, EP16, E192, top-4, capacity factor 1.33, sequence length 4096, batch 256, 1,048,576 tokens per step, and a 1x active-parameter token budget (52 optimizer steps). Score steps 5-24.
- Runtime controls: ragged one-shot disabled, `NCCL_BUFFSIZE=1048576`, no grouped launch override, watch interval 0, profiler disabled, eval interval 1000, and no expected periodic checkpoint before completion.
- Serialization: launch one EP16 baseline only. It may occupy the currently free four nodes, but no second small arm launches until it terminates. The EP64 job remains queued at interactive priority.
- Source commit, command, output root, W&B identity, and Iris jobs: pending snapshot and submission.
