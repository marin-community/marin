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

### 2026-08-09 01:25 UTC - Retire the full-rack run and launch the EP16 screen

- `RA2A-005` topology result: every worker started four JAX processes with one visible GB200 each, for 64 one-device processes. An on-demand thread profile during attempt 0 placed the sampled main thread in `backend_compile_and_load`, with idle GPUs, distinguishing normal compilation from the executable-dispatch spin in `RA2A-003/004`.
- Capacity result: production repeatedly preempted or gated the interactive full-rack job. Iris recorded 19 task preemptions across the 16-worker gang by its final attempt. Attempt 3 obtained all 16 nodes at 01:19 UTC, but no optimizer step completed before the run was retired.
- Terminal state: at user direction, stop coordinator `/power/ra2a-005-process-per-gpu-20260808-coord` at 01:22:50 UTC and continue only with the four-node screen. Iris reports the child killed, all 16 tasks terminal, and `Terminated by user`; it no longer consumes GPUs. This is an operational retirement, not evidence that process-per-GPU failed.
- EP16 source: commit `eb3a2c1d20` adds the four-node GB200 target and ragged EP flavor to the small-scale launcher. The model is explicitly d768 rather than auto-sized: 8 layers, hidden size 768, E192, and expert intermediate size 384. It preserves 65,536 tokens per EP shard but is approximately 30 times lighter per GPU than the d6144 hero model.
- EP16 jobs: coordinator `/power/ra2a-s01-ep16-baseline-20260809-coord`; child `/power/ra2a-s01-ep16-baseline-20260809-coord/grug-train-ra2a-s01-ep16-baseline-20260809`; submitted at 01:13:43 UTC. The four workers remain Kueue-gated pending four whole nodes.
- EP16 output and tracking: `s3://marin-us-east-02a/marin/grug/ra2a-s01-ep16-baseline-20260809/2026.08.09`; W&B entity `marin-community`, project `marin_moe`, group `moe-hero-ep-small-abl`, run `ra2a-s01-ep16-baseline-20260809`.
- Placement control: before accepting any timing window, record the assigned worker nodes and their rack/topology labels. Compare treatments only under the same in-rack or cross-rack condition. Rerun or report separately when placement changes, because a cross-rack baseline is not exchangeable with an in-rack treatment.
- Next action: wait for the serialized EP16 baseline to admit, verify four processes per node and placement, then score steps 5-24. Future experiments use this four-node screen; no full-rack relaunch is queued.

### 2026-08-09 01:30 UTC - EP16 baseline admits in one rack

- Kueue admitted the four-worker gang at 01:26:48 UTC. The workload requested `required: ds.coreweave.com/nvlink.domain`, as Fray derives for a GB200 gang that fits one rack.
- Nodes: `s3bsxs64`, `s3jtxs64`, `s1mwxs64`, and `s1nrxs64`. All four report NVLink domain `DH1-392-US-EAST-08A`, leafgroup `14052417914232`, and superpod `0`. This baseline is explicitly single-rack.
- Process topology: logs show process IDs 0-15, four per worker, with one `IRIS_MULTIGPU_LOCAL_DEVICE_IDS` value and one visible device per process. No task failure or preemption occurred during admission.
- Initialization observation: at 01:29 UTC, a sampled worker's four GPUs were idle with 143,255 MiB allocated. An on-demand thread dump placed the sampled main process waiting for the background data loader; its active loader thread was inside the first JIT-backed `get_batch`. This is data initialization rather than the prior training-executable spin.
- PGLE observation: all sampled ranks warned that PGLE collected an empty trace. This makes PGLE-off a concrete early treatment after a stable baseline, not merely a generic knob.
- Next action: continue monitoring through the first optimizer step and score steps 5-24 if the run completes.

### 2026-08-09 01:32 UTC - EP16 attempt 0 is preempted during initialization

- Event: at 01:30:49 UTC, one process disappeared while ranks were compiling the first data-loader batch. The remaining processes reported lost coordination-service RPCs and aborted as expected when a distributed peer vanished.
- Classification: Iris recorded one preemption and recreated the four-worker gang as workload attempt 1. It did not record a task failure. The fatal Python abort and segmentation-fault dumps are teardown fallout, not evidence of a ragged collective or model failure.
- Progress: no optimizer step completed. The single-rack placement and one-process-per-GPU topology checks from attempt 0 remain valid, but no timing sample exists.
- Next action: wait for attempt 1 at interactive priority without changing the model, rack placement, or NCCL controls.

### 2026-08-09 01:45 UTC - EP16 remains production-preempted

- Attempts: Iris now records three preemptions. One later admission launched all 16 local ranks at 01:36:30-32 UTC and received SIGTERM at 01:36:46 UTC, before JAX initialization completed. The other admission was also too short to emit a training marker.
- Metrics: none of attempts 0-2 completed an optimizer step. No NCCL, CUDA, allocator, or model error precedes their termination signals.
- Capacity: attempt 3 is Kueue-gated. Its current diagnosis excludes 200 of 202 nodes on CPU and the other two on memory. No request-shape change can create four same-domain nodes while production holds them.
- Decision: retain the same interactive EP16 baseline and retry budget. Do not downsize further or change performance controls before a clean baseline.
- Next action: continue the single monitor at the normal babysitting cadence.

### 2026-08-09 02:06 UTC - EP16 retry budget remains healthy

- State: Iris records seven production preemptions, zero failures, and attempt 7 Kueue-gated.
- Retry semantics: the coordinator's launch-time `--max-retries 10` controls failure retries. The Fray child request separately defaults `max_retries_preemption` to 100. The current churn is therefore not near the child retry ceiling.
- Decision: no relaunch or retry-budget change. Continue waiting with the exact baseline.

### 2026-08-09 02:26 UTC - EP16 process-per-GPU baseline reproduces the first-step stall

- Run: `ra2a-s01-ep16-baseline-20260809`; source commit `eb3a2c1d20`; one-shot fallback enabled by setting the one-shot kernel flag false; `NCCL_BUFFSIZE=1048576`; latency hiding enabled; collective overlap limit 4.
- Surviving attempt: after eight production-preempted gang attempts, attempt 8 ran on `s45txs64`, `s5xm6j84`, `s6htxs64`, and `s6rrxs64`. All four share NVLink domain `DH1-125-US-EAST-08A` and leafgroup `3799780806351`.
- Reachability: all 16 one-device JAX processes initialized, loaded NCCL 2.30.7, completed first-batch JIT, and traced/compiled the training step. This rules out startup compilation and the discarded four-device-process topology as the sole cause.
- Result: failed before step 0 by loss of progress. Multiple on-demand thread samples placed rank main threads at JAX 0.11 `pxla.py:420`, the `execute_sharded` call. Three GPU samples during the confirmation window were stable at 100% utilization, 149,435-149,447 MiB allocated, and 211-232 W. No step metric, NCCL warning, CUDA error, OOM, or traceback preceded the stall.
- Terminal state: stopped coordinator `/power/ra2a-s01-ep16-baseline-20260809-coord` at 02:25:44 UTC under the no-progress contract. Iris reports all four child tasks killed by user; the run no longer consumes GPUs.
- Interpretation: one process per GPU avoids the multi-device-process ambiguity but does not by itself make ragged fallback progress, even at EP16. The remaining high-signal hypothesis is an XLA scheduling deadlock or incompatible overlap schedule around the ragged private send/recv fallback.
- Next action: test latency hiding off as the first one-variable reachability treatment.

### 2026-08-09 02:27 UTC - EP16 latency-hiding-off launch contract

- Hypothesis: disabling XLA's latency-hiding scheduler prevents the low-power first-executable stall by avoiding an invalid or circular overlap schedule around ragged private send/recv.
- Controlled change: add only `--xla_gpu_enable_latency_hiding_scheduler=false`. `_apply_hero_ep_runtime_defaults` recognizes the explicit flag name and does not append its `true` default. Process-per-GPU, one-shot off, 1 MiB FIFO, overlap limit 4, disabled command buffers, PGLE, allocator, d768 EP16 model, E192 top-4 routing, batch, and metric schedule remain fixed.
- Jobs: coordinator `/power/ra2a-s02-ep16-lhs-off-20260809-coord`; child `/power/ra2a-s02-ep16-lhs-off-20260809-coord/grug-train-ra2a-s02-ep16-lhs-off-20260809`; submitted at 02:26:09 UTC. The child had all four tasks assigned at 02:26:59 UTC.
- Output and tracking: `s3://marin-us-east-02a/marin/grug/ra2a-s02-ep16-lhs-off-20260809/2026.08.09`; W&B entity `marin-community`, project `marin_moe`, group `moe-hero-ep-small-abl`, run `ra2a-s02-ep16-lhs-off-20260809`.
- Stop and score: stop on allocator/CUDA/NCCL failure or the same confirmed low-power no-progress signature. If it progresses, score steps 5-24 before any step-20-adjacent periodic work; watch interval is 0, profiler off, and eval interval 1000.
- Next action: verify the assigned rack and effective XLA flag, then monitor to the first step.

### 2026-08-09 02:48 UTC - Latency-hiding attempt is confounded by preemption and stale bootstrap state

- Placement and flags: the first S02 attempt used `s2zpxs64`, `s5trxs64`, `s5wvxs64`, and `s1wvxs64`, all in NVLink domain `DH1-394-US-EAST-08A`. The pod carried one-shot false, latency hiding false, collective overlap limit 4, disabled command buffers, and `NCCL_BUFFSIZE=1048576`.
- Partial reachability result: attempt 1 completed training-step compilation and entered `pxla.py:420` on sampled ranks. All four GPUs on task 2 were at 100% utilization, 150,477-150,489 MiB allocated, and 207-233 W. No step, warning, error, or traceback appeared before production preempted the gang roughly one minute later. This matches the baseline signature but is shorter than the no-progress window, so it is not a completed treatment result.
- Retry confounder: attempt 2 was preempted during JAX initialization. On attempt 3, the fifteen nonzero processes resolved coordinator port `27055`, while process 0 selected and started the new service on `47647`. Thread dumps put all processes in `jax.distributed.initialize`; this generation could never form a JAX world and says nothing about ragged all-to-all.
- Decision: stop S02 at 02:47:37 UTC and relaunch the identical treatment under a fresh job identity instead of allowing the stale endpoint generation to consume GPUs.
- Fresh job: coordinator `/power/ra2a-s02b-ep16-lhs-off-20260809-coord`; child `/power/ra2a-s02b-ep16-lhs-off-20260809-coord/grug-train-ra2a-s02b-ep16-lhs-off-20260809`; submitted at 02:47:57 UTC. The output and W&B run ID are `ra2a-s02b-ep16-lhs-off-20260809`.
- Next action: verify S02b placement and bootstrap port agreement, then require the full no-progress window or steps 5-24 before classifying latency hiding.

### 2026-08-09 02:59 UTC - Disabling latency hiding does not restore ragged progress

- Clean run: `ra2a-s02b-ep16-lhs-off-20260809`. Attempt 0 was production-preempted during pod initialization. Attempt 1 ran on `s1nrxs64`, `s3bsxs64`, `s3jtxs64`, and `s1mwxs64`, all in NVLink domain `DH1-392-US-EAST-08A`.
- Bootstrap control: process 0 selected and started coordinator port `21765`; all 16 processes used `10.186.213.65:21765`. This removes the stale-endpoint confounder found in S02.
- Result: no optimizer step completed. Repeated thread samples placed the main threads at JAX 0.11 `pxla.py:420`. GPU samples over the confirmation interval remained at 100% utilization, 150,461-150,469 MiB allocated, and 210-240 W. No NCCL warning, CUDA error, OOM, traceback, or step metric appeared.
- Terminal state: stopped coordinator `/power/ra2a-s02b-ep16-lhs-off-20260809-coord` at 02:59:16 UTC after more than three minutes in the stable execute signature.
- Interpretation: XLA latency hiding is not the necessary cause of the ragged fallback stall. S02 and S02b also confirm that the empty PGLE warnings are not merely masking a latency-hiding schedule that would progress with the scheduler disabled.
- Next arm: restore latency hiding and change only `--xla_gpu_experimental_parallel_collective_overlap_limit` from 4 to 1.

### 2026-08-09 03:00 UTC - EP16 overlap-limit-one launch contract

- Hypothesis: limiting XLA to one parallel collective prevents the ragged send/receive fallback from oversubscribing or circularly ordering concurrent NCCL work.
- Controlled change: relative to S01, set `--xla_gpu_experimental_parallel_collective_overlap_limit=1`. Retain latency hiding true, one-shot false, process-per-GPU, the 1 MiB FIFO, disabled command buffers, PGLE, allocator, model, routing, batch, and metric controls.
- Jobs: coordinator `/power/ra2a-s03-ep16-overlap1-20260809-coord`; child `/power/ra2a-s03-ep16-overlap1-20260809-coord/grug-train-ra2a-s03-ep16-overlap1-20260809`; submitted at 02:59:31 UTC.
- Output and tracking: `s3://marin-us-east-02a/marin/grug/ra2a-s03-ep16-overlap1-20260809/2026.08.09`; W&B run `ra2a-s03-ep16-overlap1-20260809`.
- Next action: verify one-rack placement, effective flags, and coordinator-port agreement before accepting reachability or timing evidence.

### 2026-08-09 03:11 UTC - Collective overlap limit one reduces memory but not the stall

- Run: `ra2a-s03-ep16-overlap1-20260809`; nodes `s1mwxs64`, `s1nrxs64`, `s3bsxs64`, and `s3jtxs64`; all in NVLink domain `DH1-392-US-EAST-08A`; no preemption or task retry.
- Effective control: the pod carried overlap limit 1, latency hiding true, one-shot false, disabled command buffers, and the 1 MiB NCCL FIFO. Resident memory fell to 149,443-149,455 MiB, approximately 1 GiB below S02b, confirming that the treatment changed the compiled/runtime footprint.
- Result: no optimizer step completed. Main-thread samples remained at `pxla.py:420` for more than three minutes after compilation. GPUs held 100% utilization and 210-240 W. No NCCL warning, CUDA error, OOM, traceback, or metric appeared.
- Terminal state: stopped coordinator `/power/ra2a-s03-ep16-overlap1-20260809-coord` at 03:11:11 UTC.
- Interpretation: reducing XLA's parallel collective overlap does not restore progress. The stable memory delta suggests the flag is effective, so this is not a no-op negative.
- Next arm: restore overlap limit 4 and enable `NCCL_LAUNCH_ORDER_IMPLICIT=1`, which NCCL 2.30 documents as ordering operations from different communicators on the same device to prevent deadlock while permitting overlap on CUDA 12.3 and newer.

### 2026-08-09 03:12 UTC - EP16 implicit NCCL ordering launch contract

- Hypothesis: host-program ordering across NCCL communicators prevents an inconsistent ragged send/receive launch order from deadlocking the GPU.
- Controlled change: relative to S01, add only `NCCL_LAUNCH_ORDER_IMPLICIT=1`. Restore overlap limit 4; retain latency hiding, one-shot false, process-per-GPU, the 1 MiB FIFO, disabled command buffers, PGLE, allocator, model, routing, batch, and metric controls.
- Jobs: coordinator `/power/ra2a-s04-ep16-implicit-order-20260809-coord`; child `/power/ra2a-s04-ep16-implicit-order-20260809-coord/grug-train-ra2a-s04-ep16-implicit-order-20260809`; submitted at 03:11:31 UTC.
- Output and tracking: `s3://marin-us-east-02a/marin/grug/ra2a-s04-ep16-implicit-order-20260809/2026.08.09`; W&B run `ra2a-s04-ep16-implicit-order-20260809`.
- Next action: verify placement, effective environment, and clean bootstrap, then apply the same reachability contract.

### 2026-08-09 03:21 UTC - Implicit NCCL launch ordering does not restore progress

- Run: `ra2a-s04-ep16-implicit-order-20260809`; nodes `s1nrxs64`, `s3bsxs64`, `s1mwxs64`, and `s3jtxs64`; all in NVLink domain `DH1-392-US-EAST-08A`; no preemption or task retry.
- Effective control: the pod carried `NCCL_LAUNCH_ORDER_IMPLICIT=1`, one-shot false, overlap limit 4, latency hiding true, disabled command buffers, and the 1 MiB FIFO.
- Result: no optimizer step completed. Main-thread samples stayed at `pxla.py:420` for more than three minutes. GPUs remained at 100% utilization, 149,435-149,447 MiB allocated, and 210-235 W. No NCCL warning, CUDA error, OOM, traceback, or metric appeared.
- Terminal state: stopped coordinator `/power/ra2a-s04-ep16-implicit-order-20260809-coord` at 03:21:09 UTC.
- Interpretation: NCCL's implicit ordering across communicators does not resolve this stall. Either the problematic operations do not span separately ordered communicators, or the loss of progress is below/elsewhere than this host launch-order safeguard.
- Next arm: disable multi-node NVLink with `NCCL_MNNVL_ENABLE=0`. NCCL documents MNNVL as requiring cuMem support, and the earliest ragged fallback attempt failed in `ncclCuMemAlloc`, so this isolates the transport family most directly implicated by prior evidence.

### 2026-08-09 03:22 UTC - EP16 MNNVL-off launch contract

- Hypothesis: the ragged fallback stalls specifically on the cuMem-backed multi-node NVLink send/receive path; forcing another transport restores progress.
- Controlled change: relative to S01, add only `NCCL_MNNVL_ENABLE=0`. Retain one-shot false, process-per-GPU, the 1 MiB FIFO, overlap limit 4, latency hiding, disabled command buffers, PGLE, allocator, model, routing, batch, and metric controls.
- Jobs: coordinator `/power/ra2a-s05-ep16-mnnvl-off-20260809-coord`; child `/power/ra2a-s05-ep16-mnnvl-off-20260809-coord/grug-train-ra2a-s05-ep16-mnnvl-off-20260809`; submitted at 03:21:24 UTC.
- Output and tracking: `s3://marin-us-east-02a/marin/grug/ra2a-s05-ep16-mnnvl-off-20260809/2026.08.09`; W&B run `ra2a-s05-ep16-mnnvl-off-20260809`.
- Next action: verify placement, effective environment, and clean bootstrap, then apply the same reachability contract.

### 2026-08-09 03:30 UTC - Disabling MNNVL changes memory but not progress

- Run: `ra2a-s05-ep16-mnnvl-off-20260809`; nodes `s69vxs64`, `s38vxs64`, `s5xvxs64`, and `s45sxs64`; all in NVLink domain `DH1-129-US-EAST-08A`; no preemption or task retry.
- Effective control: the pod carried `NCCL_MNNVL_ENABLE=0`, one-shot false, overlap limit 4, latency hiding true, disabled command buffers, and the 1 MiB FIFO.
- Result: no optimizer step completed. Main-thread samples remained at `pxla.py:420` for more than three minutes. GPUs held 100% utilization, 145,011-145,023 MiB allocated, and 195-231 W. No NCCL warning, CUDA error, OOM, traceback, or metric appeared.
- Terminal state: stopped coordinator `/power/ra2a-s05-ep16-mnnvl-off-20260809-coord` at 03:30:22 UTC.
- Interpretation: the approximately 4.4 GiB memory reduction confirms the transport control is effective, but progress remains lost. The stall is therefore not specific to the cuMem-backed MNNVL path.
- Next arm: set `NCCL_RUNTIME_CONNECT=0` so NCCL establishes peer connections during communicator initialization rather than lazily on first send/receive.

### 2026-08-09 03:31 UTC - EP16 eager-connect launch contract

- Hypothesis: first-use lazy peer connection interacts badly with the ragged send/receive launch set; establishing all peers during communicator initialization avoids the first-step stall.
- Controlled change: relative to S01, add only `NCCL_RUNTIME_CONNECT=0`. Retain one-shot false, process-per-GPU, the 1 MiB FIFO, overlap limit 4, latency hiding, disabled command buffers, PGLE, allocator, model, routing, batch, and metric controls.
- Jobs: coordinator `/power/ra2a-s06-ep16-eager-connect-20260809-coord`; child `/power/ra2a-s06-ep16-eager-connect-20260809-coord/grug-train-ra2a-s06-ep16-eager-connect-20260809`; submitted at 03:30:38 UTC.
- Output and tracking: `s3://marin-us-east-02a/marin/grug/ra2a-s06-ep16-eager-connect-20260809/2026.08.09`; W&B run `ra2a-s06-ep16-eager-connect-20260809`.
- Next action: verify placement, effective environment, and clean bootstrap, then apply the same reachability contract.

### 2026-08-09 03:42 UTC - Re-rank the queue from the JAX 0.11 ragged lowering

- Exact source: the GPU environment resolves JAX/jaxlib 0.11.0, whose pinned OpenXLA revision is `131bf41acb46`. Its defaults leave the dense ragged decomposer off, enable one-shot, enable the NCCL-backed one-shot barrier, and select private ragged memory. The current fallback arms override only one-shot to false.
- Fallback mechanics: JAX copies the four metadata arrays to the host, performs a dense all-to-all on sender-side output offsets, then launches one grouped send and receive for every `(local update, peer)` pair. E192/EP16 has 12 local expert updates per peer and 16 peers, or 192 send/receive pairs per ragged call. The three-minute signature is operationally unusable, but a Python frame at `execute_sharded` cannot distinguish a hard deadlock from catastrophic serialized progress.
- Next reachability arm: after S06, enable `--xla_gpu_unsupported_enable_ragged_all_to_all_multi_host_decomposer=true` while retaining one-shot false. This XLA pass rewrites dispatch into an inter-host all-gather plus an intra-host ragged all-to-all, directly reducing cross-host P2P launch fanout and affecting only graphs that contain ragged all-to-all.
- Following arms: disable async wrapping specifically with `--xla_gpu_disable_async_collectives=RAGGEDALLTOALL`; restore one-shot at EP16 under process-per-GPU to test whether the EP64 device-32 illegal address is size-specific; use the dense ragged decomposer only as a correctness diagnostic.
- Performance gate: the completed fixed-EP64 ladder run `mhep-ladder-20260808c-ep64-d768` reports 6.5845% p50 MFU over 500 samples, so d768 is too small to be a credible absolute 20% target. Once a treatment reaches steps, repeat it on the same four-node domain with a d6144/L48, E48, top-4, latent-3072, i6272, batch-256 EP16 proxy. Dividing both hero experts and global batch by four preserves three resident experts, 65,536 tokens, and 16 sequences per GPU, plus pooled ragged receiver load; this makes per-GPU model memory, active compute, and routing load representative. Only then tune FIFO size, `NCCL_PROTO=Simple`, P2P channels, and any NVLS/SHARP setting relevant to ordinary collectives introduced by decomposition.
- Queue state: S06 remains Kueue-gated with zero preemptions. Its workload excludes 200 of 202 nodes on CPU and the remaining two on memory. A direct join of live GB200 nodes and pod GPU requests finds only five fully idle nodes: at most two in `DH1-125` and two in `DH1-126`, plus one in `DH1-122`; one additional `DH1-393` node has one of four GPUs occupied. No rack has four whole idle nodes, so reducing the inherited 120-CPU/850-GiB request would not admit the gang.
- Issue update: https://github.com/marin-community/marin/issues/8077#issuecomment-5229621669
- MFU calibration update: https://github.com/marin-community/marin/issues/8077#issuecomment-5229650847

### 2026-08-09 04:09 UTC - Eager peer connection does not restore useful progress

- Run: `ra2a-s06-ep16-eager-connect-20260809`; nodes `s5trxs64`, `s1wvxs64`, `s5wvxs64`, and `s2zpxs64`; all in NVLink domain `DH1-394-US-EAST-08A`.
- Bootstrap control: all 16 one-device JAX processes used coordinator `10.186.213.145:64279`. The pod carried `NCCL_RUNTIME_CONNECT=0`, `NCCL_BUFFSIZE=1048576`, one-shot false, overlap limit 4, latency hiding true, disabled command buffers, PGLE, and the `cuda_async` allocator.
- Result: no optimizer step completed during approximately eight minutes after the first training execute began. The sampled worker main thread remained at JAX 0.11 `pxla.py:420`; GPUs stayed at 100% utilization, 150,583-150,603 MiB allocated, and 207-232 W. Two samples 30 seconds apart of every NVLink counter on GPU 0 showed zero transmitted and received byte delta despite the 100% busy reading. No NCCL warning, CUDA error, OOM, traceback, or W&B step metric appeared.
- Terminal state: production preempted the gang at 04:07:00 UTC, before the planned ten-minute extended observation ended. The coordinator was explicitly stopped at 04:09:38 UTC so the currently deployed Iris controller could not retry this job through the stale JAX endpoint bug.
- Interpretation: eager NCCL connection setup does not produce useful first-step progress. The production preemption prevents calling this a treatment-originated terminal failure, but eight minutes of invariant GPU state and zero NVLink payload movement is stronger evidence of a fixed device-side spin than the earlier three-minute observations.
- Next arm: enable the JAX 0.11 multi-host ragged decomposer while retaining the non-one-shot path. This is the first treatment that changes the collective graph rather than scheduling or transport around the same 192-pair-per-call fallback.

### 2026-08-09 04:10 UTC - EP16 multi-host ragged decomposition launch contract

- Hypothesis: JAX 0.11's multi-host ragged decomposer restores progress by replacing the fallback's cross-host peer fanout with an inter-host all-gather followed by intra-host ragged all-to-all.
- Controlled change: relative to S01, add only `--xla_gpu_unsupported_enable_ragged_all_to_all_multi_host_decomposer=true`. Retain one-shot false, process-per-GPU, `NCCL_BUFFSIZE=1048576`, overlap limit 4, latency hiding, disabled command buffers, PGLE, allocator, model, routing, batch, and metric controls.
- Jobs: coordinator `/power/ra2a-s07-ep16-multihost-decomp-20260809-coord`; expected child `/power/ra2a-s07-ep16-multihost-decomp-20260809-coord/grug-train-ra2a-s07-ep16-multihost-decomp-20260809`; submitted at 04:10:44 UTC.
- Output and tracking: `s3://marin-us-east-02a/marin/grug/ra2a-s07-ep16-multihost-decomp-20260809/2026.08.09`; W&B run `ra2a-s07-ep16-multihost-decomp-20260809`.
- Next action: verify one-domain placement, exact flags, all 16 bootstrap endpoints, and whether the rewritten graph reaches the first step.

### 2026-08-09 04:20 UTC - Exact-shape four-node hero gate is ready

- Launcher: `experiments.grug.moe_hero_ep.launch` now accepts an explicit `ep_nodes` value for expert-parallel flavors. It derives both the expert axis and requested node replicas from that value while retaining the existing 16-node default and four processes per task.
- Proxy contract: `--ep-nodes 4 --batch-size 256 --num-experts 48 --num-steps 25 --flavor ep-ragged`. Relative to the EP64 hero, dividing nodes, global batch, and routed experts by four preserves d6144/L48, intermediate 6272, latent 3072, top-4, capacity 1.33, three resident experts per GPU, 65,536 tokens and 16 sequences per GPU, and pooled receiver load.
- Metric hygiene: default watch interval, eval, profiler, and checkpoint saving are all zero/off. Score steps 5-24. The W&B tag records the EP node count.
- Verification: all 23 focused EP hero tests pass; the required changed-file Ruff, Black, Pyrefly, header, AST, conflict, whitespace, and Markdown checks pass.
- Queue: S07 remains Kueue-gated. Kueue currently reports CPU as the first exclusion on all 202 nodes, but the most recent direct GPU/rack inventory found only five whole idle GB200 nodes split 2+2+1 across domains, so reducing CPU would not admit a same-domain four-node gang.
- Recheck at 04:43 UTC: the live node/pod join is unchanged in substance. `DH1-126` and `DH1-125` each have two whole idle nodes, `DH1-122` has one, and `DH1-393` has only three free GPUs on a partially occupied node. No domain can fit S07, independent of its CPU request.
- Next action: launch this exact-shape gate only after a reachability treatment completes optimizer steps on d768.

### 2026-08-09 04:24 UTC - Attempt-scoped JAX coordinator endpoint fix merged

- Resolution: https://github.com/marin-community/marin/pull/8079 merged. Iris now includes the task attempt ID in the JAX coordinator endpoint key, preventing a retry from resolving the prior attempt's port before rank 0 publishes the replacement.
- Verification: the regression injects stale port 27055 and fresh port 47647 and verifies all ranks resolve 47647. Fifteen focused tests and the full Iris suite passed; required CI was green at merge.
- Durable record: https://echo.oa.dev/wiki/100
- Deployment caveat: merge does not imply that today's controller has rolled out. Continue stopping preempted experiment parents and using fresh job identities until deployment is verified.
- Issue update: https://github.com/marin-community/marin/issues/8077#issuecomment-5229748426

### 2026-08-09 05:05 UTC - Multi-host decomposition does not restore progress

- Run: `ra2a-s07-ep16-multihost-decomp-20260809`; nodes `s3bsxs64`, `s3jtxs64`, `s4lqxs64`, and `s62pys64`; all in NVLink domain `DH1-392-US-EAST-08A`; no preemption or retry.
- Effective control: the pod carried `--xla_gpu_unsupported_enable_ragged_all_to_all_multi_host_decomposer=true`, one-shot false, overlap limit 4, latency hiding true, disabled command buffers, `NCCL_BUFFSIZE=1048576`, PGLE, and `cuda_async`.
- Bootstrap: all 16 one-device processes covered global process IDs 0-15 and used coordinator `10.186.213.59:42079`. NCCL was 2.30.7.
- Compile control: the first 05:00 UTC thread sample put all task-0 worker mains in `backend_compile_and_load` at `compiler.py:350`. The apparent loader warning came from the separate prefetch thread compiling its batch transform. By 05:02 UTC the worker mains had moved to `pxla.py:420`, establishing the first-execute boundary.
- Result: no optimizer step completed in more than three minutes of execute. The 05:05 thread sample still placed worker mains at `pxla.py:420`; GPUs remained at 100%, 149,467-149,475 MiB, and 211-235 W. All 36 Tx/Rx counters for GPU 0 had exactly zero delta from the first-execute sample. No NCCL warning, CUDA error, OOM, traceback, or W&B step metric appeared.
- Terminal state: stopped coordinator `/power/ra2a-s07-ep16-multihost-decomp-20260809-coord` at 05:05:26 UTC under the no-progress contract.
- Interpretation: the JAX 0.11 multi-host decomposer changes compilation but does not change the fixed execute signature. Either the decomposed graph still reaches a broken intra-host ragged thunk, or the loss of progress occurs in a shared schedule/runtime layer before payload movement.
- Next arm: retain the baseline graph and disable async wrapping specifically for `RAGGEDALLTOALL`.

### 2026-08-09 05:06 UTC - EP16 synchronous-ragged launch contract

- Hypothesis: XLA's async collective schedule wraps ragged all-to-all in a dependency cycle or unsupported execution path; keeping ragged synchronous restores progress without disabling async handling for other collective kinds.
- Controlled change: relative to S01, add only `--xla_gpu_disable_async_collectives=RAGGEDALLTOALL`. Retain one-shot false, process-per-GPU, `NCCL_BUFFSIZE=1048576`, overlap limit 4, latency hiding, disabled command buffers, PGLE, allocator, model, routing, batch, and metric controls.
- Jobs: coordinator `/power/ra2a-s08-ep16-sync-ragged-20260809-coord`; expected child `/power/ra2a-s08-ep16-sync-ragged-20260809-coord/grug-train-ra2a-s08-ep16-sync-ragged-20260809`; submitted at 05:06:13 UTC.
- Output and tracking: `s3://marin-us-east-02a/marin/grug/ra2a-s08-ep16-sync-ragged-20260809/2026.08.09`; W&B run `ra2a-s08-ep16-sync-ragged-20260809`.
- Next action: verify one-domain placement, exact flags, and bootstrap agreement, then distinguish compile from execute with thread samples.

### 2026-08-09 05:18 UTC - Synchronous ragged scheduling does not restore progress

- Run: `ra2a-s08-ep16-sync-ragged-20260809`; reused S07's `s3bsxs64`, `s3jtxs64`, `s4lqxs64`, and `s62pys64` hosts in `DH1-392-US-EAST-08A`, giving a direct same-hardware comparison; no preemption or retry.
- Effective control: the pod carried `--xla_gpu_disable_async_collectives=RAGGEDALLTOALL`, one-shot false, overlap limit 4, latency hiding true, disabled command buffers, `NCCL_BUFFSIZE=1048576`, PGLE, and `cuda_async`.
- Bootstrap: global process IDs 0-15 were each bound to one GPU and used coordinator `10.186.213.59:21727`. NCCL was 2.30.7.
- Compile control: worker mains progressed from initial batch wait through CuTeDSL compilation and `backend_compile_and_load`, then entered `pxla.py:420` by 05:14:07 UTC.
- Result: no optimizer step completed in more than four minutes of execute. At the cutoff, the sampled worker main remained at `pxla.py:420`; GPUs stayed at 100%, 149,475-149,485 MiB, and 207-235 W. All 36 GPU-0 Tx/Rx counters had zero delta from the first-execute sample. No NCCL warning, CUDA error, OOM, traceback, or W&B step metric appeared.
- Terminal state: stopped coordinator `/power/ra2a-s08-ep16-sync-ragged-20260809-coord` at 05:18:35 UTC.
- Interpretation: ragged-specific async wrapping is not the necessary cause. The approximately 10 MiB footprint delta relative to S07 confirms a distinct executable, but it retains the same no-payload device spin.
- Next arm: restore JAX 0.11's default one-shot ragged kernel on EP16/process-per-GPU under a bounded crash/no-progress contract. This isolates whether S01's EP64 device-32 illegal address was specific to the larger world or old multi-GPU process topology.

### 2026-08-09 05:19 UTC - EP16 one-shot ragged launch contract

- Hypothesis: the one-shot kernel's earlier CUDA illegal address was specific to EP64 or the discarded one-process-per-four-GPU topology; on EP16/process-per-GPU the fused implementation may be both correct and substantially faster than the send/receive fallback.
- Controlled change: relative to S01, remove only `--xla_gpu_unsupported_use_ragged_all_to_all_one_shot_kernel=false`, restoring the JAX 0.11 default. Retain process-per-GPU, `NCCL_BUFFSIZE=1048576`, overlap limit 4, latency hiding, disabled command buffers, PGLE, allocator, model, routing, batch, and metric controls.
- Jobs: coordinator `/power/ra2a-s09-ep16-oneshot-20260809-coord`; expected child `/power/ra2a-s09-ep16-oneshot-20260809-coord/grug-train-ra2a-s09-ep16-oneshot-20260809`; submitted at 05:19:32 UTC.
- Output and tracking: `s3://marin-us-east-02a/marin/grug/ra2a-s09-ep16-oneshot-20260809/2026.08.09`; W&B run `ra2a-s09-ep16-oneshot-20260809`.
- Stop contract: stop immediately on the prior illegal-address signature or any allocator/NCCL failure. Otherwise require an optimizer step or the same confirmed execute no-progress window.

### 2026-08-09 05:27 UTC - One-shot arm preempted before model execution

- Run: `ra2a-s09-ep16-oneshot-20260809`; nodes `s1wvxs64`, `s5trxs64`, `s2zpxs64`, and `s5wvxs64`; all in NVLink domain `DH1-394-US-EAST-08A`.
- Effective control: all 16 one-device processes used coordinator `10.186.213.145:19563`. The pod omitted the one-shot override, restoring JAX 0.11's default fused ragged kernel; it retained overlap limit 4, latency hiding, disabled command buffers, `NCCL_BUFFSIZE=1048576`, PGLE, and `cuda_async`.
- Compile boundary: training loops opened at 05:23:01 UTC, but most worker mains remained at line 677 waiting for their first batch. Background loader threads were still compiling the CPU `stack_tree` transform, with first-batch stalls of 146-161 seconds. One sampled worker reached line 688, but the distributed model execute could not begin before all ranks supplied a batch.
- Result: production preempted the gang at 05:25:31 UTC. No optimizer step, CUDA illegal address, NCCL error, OOM, or confirmed model-execute window occurred, so S09 is not evidence for or against one-shot.
- Terminal state: stopped coordinator `/power/ra2a-s09-ep16-oneshot-20260809-coord` at 05:26:56 UTC to prevent retries while the merged attempt-scoped endpoint fix remains undeployed.
- Source refinement: pinned OpenXLA names the dense diagnostic `--xla_gpu_unsupported_enable_ragged_all_to_all_decomposer=true`. In process-per-GPU mode, multi-host one-shot depends on the NCCL-backed barrier to obtain clique-wide symmetric memory; disabling that barrier would fall through to the already-stalled generic path rather than provide an independent one-shot implementation.
- Issue update: https://github.com/marin-community/marin/issues/8077#issuecomment-5229949762
- Next action: wait behind production and rerun the one-shot arm under a fresh job identity. Do not infer a treatment result from this scheduling interruption.

### 2026-08-09 05:30 UTC - Fresh EP16 one-shot retry queued

- Controlled retry: `ra2a-s09b-ep16-oneshot-20260809` repeats S09 under a fresh identity with no treatment change: d768/L8, E192, expert width 384, top-4, capacity 1.33, EP16, 1,048,576 tokens per step, 52 steps, process-per-GPU, the default one-shot kernel, and `NCCL_BUFFSIZE=1048576`.
- Metric controls: watch and profiling are disabled; eval remains outside the 52-step screen at interval 1000; checkpoints cannot reach their 30-minute interval before the bounded reachability decision. Score steps 5-24 if it progresses.
- Jobs: coordinator `/power/ra2a-s09b-ep16-oneshot-20260809-coord`; child `/power/ra2a-s09b-ep16-oneshot-20260809-coord/grug-train-ra2a-s09b-ep16-oneshot-20260809`; submitted at 05:29:45 UTC. The child was pending on `cw-us-east-08a` at 05:30:27 UTC with no failure or preemption.
- Scheduling contract: keep interactive priority and the hard single-NVLink-domain gang constraint. Wait behind production rather than lowering CPU or accepting cross-rack placement. No other experiment is queued.
- Issue update: https://github.com/marin-community/marin/issues/8077#issuecomment-5229960988

### 2026-08-09 05:34 UTC - A post-0.11 device kernel outranks generic NCCL tuning

- Version boundary: jaxlib 0.11.0 pins OpenXLA `131bf41acb46` from 2026-07-16. PyPI has no newer stable JAX release as of this check.
- Upstream change: OpenXLA PR [#46116](https://github.com/openxla/xla/pull/46116), merged 2026-07-24 as `acb5aaffe`, adds an opt-in device-initiated RA2A kernel using NCCL 2.29 LSA/GIN. Its stated design is one CUDA launch, no per-call host coordination, local LSA copies plus GIN put/signal for remote peers, and vectorized transfers. This directly replaces both pinned paths implicated here: the illegal-address one-shot LSA path and the host-synchronized send/receive fallback.
- Required pair: the new path needs both `--xla_gpu_experimental_ragged_all_to_all_use_device_kernel=true` and `--xla_gpu_experimental_enable_nccl_symmetric_buffers=true`. The first flag alone is a silent no-op under private memory because `FindSymmetricMemory` returns null. This was independently recorded in [#7891](https://github.com/marin-community/marin/issues/7891#issuecomment-5174781337), and an OpenXLA contributor confirmed the pair in the [follow-up](https://github.com/marin-community/marin/issues/7891#issuecomment-5175507921).
- Delivery: official JAX documentation publishes CUDA 13 nightlies through its non-PyPI index, but warns nightlies may not pass the full suite. A nightly dependency snapshot and focused compatibility test are therefore required before spending the four-node allocation.
- Revised queue: finish the current pinned one-shot retry; if it does not progress, test the pinned dense decomposer as the last no-version-change reachability diagnostic. Then prioritize a post-2026-07-24 nightly with the device-kernel/symmetric-buffer pair over PGLE, protocol, channel, user-buffer, or SHARP tuning.
- Scope exclusions: SHARP and NVLS optimize reductions and do not directly accelerate either ragged peer copies or NCCL send/receive. FSDP combine thresholds affect AG/RS/AR rather than RA2A. `CUDA_DEVICE_MAX_CONNECTIONS=1` is explicitly slower on Blackwell, and local memcpy P2P does not apply when every process owns one device.
- Issue update: https://github.com/marin-community/marin/issues/8077#issuecomment-5229980232

### 2026-08-09 05:43 UTC - Pinned nightly worker runtime is ready

- Implementation: EP run configs can now carry explicit worker pip packages through the shared Grug dispatcher. Both the small-screen and exact hero launchers accept `--jax-nightly-version`; omitted retains the locked stable environment.
- Reproducibility: the selected runtime is `0.11.1.dev20260808` for `jax`, `jaxlib`, `jax-cuda13-plugin[with-cuda]`, and `jax-cuda13-pjrt`, resolved from JAX's official nightly-only index. The generated worker setup installs all four after the locked GPU sync, so the coordinator remains on the repository environment and every training process gets one coherent nightly.
- Verification: all 25 focused `tests/test_moe_hero_ep.py` tests pass. The required changed-file Ruff, Black, Pyrefly, license, AST, conflict, whitespace, and Markdown checks pass. Tests cover the stable no-override default, the exact four-package nightly set, and rejection of non-dated nightly names.
- Experimental contract: when this arm launches, pair the runtime with `--xla_gpu_experimental_ragged_all_to_all_use_device_kernel=true --xla_gpu_experimental_enable_nccl_symmetric_buffers=true`. Do not launch it concurrently with S09b.
- Issue update: https://github.com/marin-community/marin/issues/8077#issuecomment-5230007264

### 2026-08-09 05:54 UTC - Auto-PGLE invalidates the process-per-GPU baseline

- Run: `ra2a-s09b-ep16-oneshot-20260809`; all 16 one-device processes bootstrapped with a fresh coordinator and entered their first-batch compile. The model execute was not reached cleanly.
- Failure: concurrent workers raised `jax.errors.JaxRuntimeError: ALREADY_EXISTS: Another profiling session active` from the first compiled training call. The grug loop reported a fatal error and Iris recorded a task failure before the parent was stopped.
- Root cause: the EP runtime unconditionally defaulted `JAX_ENABLE_PGLE=true`. In process-per-GPU mode this starts four concurrent auto-PGLE/CUPTI sessions on each node even though the explicit experiment profiler is disabled. The single-process-per-node FSDP topology does not have the same intra-node profiler collision.
- Consequence: S01-S08 all inherited auto-PGLE. Their no-progress observations remain useful signatures, but none is a clean process-per-GPU ragged result. Do not rank NCCL treatments from them until the baseline is repeated without auto-PGLE.
- Runtime fix: multi-process tasks now default `JAX_ENABLE_PGLE=false`; single-process tasks retain the existing true default, and explicit settings are preserved. The focused suite passes with 26 tests.
- Terminal state: stopped coordinator `/power/ra2a-s09b-ep16-oneshot-20260809-coord`; the summary records one failure and four later preemptions/killed attempts. No optimizer step or ragged treatment result was scored.
- Clean retry: submitted `/power/ra2a-s09c-ep16-oneshot-pgle-off-20260809-coord` at 05:54:18 UTC with an explicit `JAX_ENABLE_PGLE=false`. Every model, routing, one-shot, NCCL, allocator, metric, and scheduling parameter is otherwise unchanged.
- Next action: wait behind production, verify the retry's one-domain placement and effective environment, then distinguish compile from execute. Only after this clean baseline should the pinned dense decomposer or nightly device kernel run.

### 2026-08-09 06:00 UTC - PGLE-off restores ragged one-shot progress

- Run: `ra2a-s09c-ep16-oneshot-pgle-off-20260809`; initial training hosts `s2grxs64`, `s1zsxs64`, `s33xxs64`, and `s24qxs64`, all in `DH1-137-US-EAST-08A`. The coordinator ran separately on `s2rsxs64`; the GPU gang itself was not cross-rack. No production preemption affected the first attempt.
- Compile boundary: a 05:58:47 UTC thread profile showed all four process-per-GPU worker mains on task 0 in `backend_compile_and_load` at `compiler.py:350`, confirming the earlier `pxla.py:420` samples represented later execution rather than compile.
- Reachability result: with `JAX_ENABLE_PGLE=false`, every rank completed the first optimizer step. Logs report finite loss 11.8 and one displayed iteration at about 91.8 seconds including compilation. This is the first process-per-GPU ragged one-shot progress in the investigation and reverses the contaminated S01-S08 no-progress result.
- Numerical result: the second attempted step produced `Non-finite loss (nan) at step 2` on all ranks, so the run cannot supply steady-state timing. The 52-step screen used `tokens_per_active_param=1`; the optimizer heuristic therefore hit its maximum MuonH rate on an intentionally abbreviated schedule.
- Terminal state: stopped `/power/ra2a-s09c-ep16-oneshot-pgle-off-20260809-coord` at 06:00:01 UTC to prevent retries. No communication failure, OOM, CUDA illegal address, or NCCL error appeared.
- Interpretation: pinned JAX 0.11's one-shot path is reachable on EP16/process-per-GPU when auto-PGLE is disabled. The immediate next baseline should keep communication identical and use the standard 60-token-per-active-parameter optimizer schedule, then stop manually after step 25.
