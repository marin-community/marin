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

The process-per-GPU, four-node exact proxy now trains with finite loss and zero routing drops. The original ragged transfer launched only one block per peer and reached 11.28% MFU. Splitting each peer slice into 32 logical updates raises the grid from 16 to 512 blocks and reaches 18.08% MFU. Reusing the existing QuACK/CuTe SM100 grouped GEMMs for activation-path expert compute reaches the best stable result, 19.63% mean MFU. The first alternative Wgrad screen is positive: cuDNN Frontend reduces the exact standalone `dw13`/`dw2` pair by 29.04%, enough to project 20.26% MFU before adapter overhead. Transformer Engine and Mixture-of-Kittens screens remain active before a four-node integration.

## Current Baseline

- Date: 2026-08-10.
- Code refs: PR #8013 EP lineage plus implementation PR #8081 on `weaver/hero-run-why-can-t-we-ragged-all`.
- Exact safe baseline: S24, d6144/L48/E48, EP16, 1,048,576 tokens/step, one process per GPU, latency hiding off, overlap limit one, one update per peer: 33.8187 seconds/step and 11.2834% MFU.
- Selected result: S33, the same shape and runtime with 32 updates per peer and the QuACK/CuTe activation-path expert backend: 19.4429 seconds/step, 53,939.7 tokens/s, and 19.6291% MFU, with finite loss and zero routing drops.

## Hypothesis Queue

### Active

- `RA2A-TE`: Transformer Engine JAX BF16 grouped GEMM supports the exact `dw13` and `dw2` weight-gradient layouts and reduces their combined time by at least 17.3%. Next test: exact one-GB200 harness.
- `RA2A-CUDNN`: cuDNN Frontend grouped Wgrad supports the exact layouts and clears the target in a one-GB200 screen. Next test: a JAX FFI adapter with per-expert 256-row alignment, then the exact four-node run if adapter overhead remains below 0.2436 seconds/step.
- `RA2A-MOK`: Mixture-of-Kittens' persistent forward/backward schedule retains a material advantage at the exact hero-layer geometry. Next test: four-GB200 process-per-GPU oracle before JAX integration.
- `RA2A-OFFLOAD`: step-boundary optimizer-state staging can hide at least 0.3606 seconds/step without exceeding HBM. Next test: semantic host/device trace or Nsight Systems after compute screening.

### Blocked

- Full-rack EP64 validation depends on production capacity. It is not expected to improve the equal per-GPU routed-token and local-expert shapes through scale alone.

### Falsified / Dead End

- The private NCCL send/recv fallback OOMs with 4 MiB peer FIFOs and stalls with 1 MiB FIFOs, even in process-per-GPU mode.
- Restoring `NCCL_BUFFSIZE` from 1 MiB to 4 MiB is neutral on the working one-shot path. NVLS is already selected for the largest NCCL all-gathers, while SHARP, protocol, channel, and launch settings cannot tune the custom peer-write kernel.
- XLA `ragged_dot_general` OOMs on one 159.32 GiB allocation. The generic symmetric backend rejects the source window, and the pinned GXL backend is unimplemented.
- Command buffers complete cleanly but regress mean MFU from 19.6291% to 19.3369%.

### Promoted

- Disable latency hiding and use collective overlap limit one for ragged EP. The default scheduler corrupts the first backward at larger expert banks, while overlap four is 24% slower on the clean small proxy.
- Run one JAX process per GPU and reserve 850 GiB of host memory per four-GPU worker. Disable auto-PGLE because four concurrent CUPTI sessions collide.
- Split each peer slice into 32 updates for the exact hero geometry, and use the QuACK/CuTe ragged expert backend. Keep the library split default at one until other shapes are profiled.

## Decision Log

- 2026-08-08: Run arms serially on one interactive-priority rack. Score steps 5 through 24; treat effects below the ±1.57% single-reading resolution measured in #8054 as unresolved without replication.
- 2026-08-08: Pass `--watch-interval 0 --eval-every 0 --profile-steps 0 --no-save-checkpoints` to keep periodic metric, eval, profile, and checkpoint work out of the timing window.
- 2026-08-08: Test NCCL/SHARP controls only after lowering or logs show that the ragged path reaches NCCL. JAX's GPU implementation may use a peer-pointer kernel instead.
- 2026-08-08: Stop the default one-shot baseline after its first synchronized illegal-address failure rather than consume ten automatic retries. Treat reachability as the baseline result and test NCCL send/recv next.
- 2026-08-08: Reduce the send/recv peer FIFO before changing the XLA memory fraction. The failing executable already exceeds XLA's rematerialization target, while `NCCL_BUFFSIZE` directly controls the late allocation that failed.
- 2026-08-08: Do not reduce the FIFO below 1 MiB after `RA2A-003`: allocation already succeeded, and a smaller FIFO cannot explain or repair the new low-power collective spin. Test grouped NCCL launch next.
- 2026-08-10: Reopen #8077 for alternative grouped-MoE implementations outside the sealed flag sweep. Require a 17.3% combined exact-shape Wgrad reduction before a four-node training arm; run Transformer Engine, cuDNN Frontend, and Mixture-of-Kittens screens serially and retain process-per-GPU for distributed tests.
- 2026-08-10: Promote cuDNN Frontend's 256x256 grouped-Wgrad configuration to JAX-adapter work. Its exact one-GB200 result exceeds the kernel gate by 11.7 percentage points and projects 0.2436 seconds/step of integration-overhead budget before falling back below 20% MFU.

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
- Durable metric refinement: W&B recorded step 0 at 41.9956 seconds, 24,968.7 tokens/s, and 6.3880% MFU with loss 11.8062. The progress-line 91.8-second rate included surrounding first-iteration wall time. One sample is not a steady-state baseline, but the recorded MFU is close to the fixed-all-to-all d768 control's 6.5845% median and shows no order-of-magnitude ragged transport penalty at EP16.
- Routing sanity: the same step dropped 9,689 assignments, `moe/drop_fraction=0.000288755` (0.0289%). Seven of eight layers had zero capacity overflow; layer 1 was 0.2310%. Ragged receiver pooling at capacity 1.33 therefore avoids the massive token-loss regime that forced the fixed-bucket backend toward 4,096 buckets.
- Numerical result: the second attempted step produced `Non-finite loss (nan) at step 2` on all ranks, so the run cannot supply steady-state timing. The 52-step screen used `tokens_per_active_param=1`; the optimizer heuristic therefore hit its maximum MuonH rate on an intentionally abbreviated schedule.
- Terminal state: stopped `/power/ra2a-s09c-ep16-oneshot-pgle-off-20260809-coord` at 06:00:01 UTC to prevent retries. No communication failure, OOM, CUDA illegal address, or NCCL error appeared.
- Interpretation: pinned JAX 0.11's one-shot path is reachable on EP16/process-per-GPU when auto-PGLE is disabled. The immediate next baseline should keep communication identical and use the standard 60-token-per-active-parameter optimizer schedule, then stop manually after step 25.

### 2026-08-09 06:02 UTC - Stable-schedule timing baseline queued

- Run: `ra2a-s10-ep16-oneshot-pgle-off-standard-schedule-20260809`; coordinator `/power/ra2a-s10-ep16-oneshot-pgle-off-standard-schedule-20260809-coord` and expected four-node child of the same run ID.
- Controlled change from S09c: increase `tokens_per_active_param` from 1 to the ladder's standard 60. This restores the intended optimizer token budget and learning-rate schedule. Process-per-GPU, JAX 0.11 one-shot ragged, d768/L8, E192, expert width 384, top-4, capacity 1.33, 1,048,576 tokens per step, `NCCL_BUFFSIZE=1048576`, allocator, latency hiding, command buffers, and metric controls are unchanged.
- Stop/scoring contract: score steps 5-24 if loss remains finite, then stop manually. Watch and explicit profiling remain disabled; eval is at step 1000 and the 30-minute checkpoint is outside the expected timing window.
- Scheduling state: the same 4 x GB200, 120 CPU, 850 GiB task shape admitted S09c immediately and is known schedulable. S10 is currently Kueue-gated behind production, with zero failures or preemptions. Retain the single-NVLink-domain gang constraint and wait.

### 2026-08-09 06:20 UTC - Four-node host reservations corrected

- S10 at 120 CPU / 850 GiB was not merely waiting for GPUs: Kueue excluded all 202 nodes on CPU. It was stopped before admission.
- S10b at 32 CPU / 850 GiB reduced the CPU exclusion to 199 nodes, but still left only three candidates. S10c at 16 CPU / 850 GiB moved the dominant exclusion to memory: 4 CPU and 198 memory. Both were stopped before admission.
- S10d at 16 CPU / 256 GiB left 4 CPU, 195 memory, and 3 GPU exclusions. The four-node exact d6144 proxy retains this 256 GiB profile because its offloaded optimizer state is about 112 GiB/node before loader and runtime headroom.
- Final small-screen profile: 16 CPU and 128 GiB per four-GPU node. Four CPU cores per JAX process matches the H100 screen's ratio; the d768 screen does not offload optimizer state. The full-rack hero remains at 120 CPU / 850 GiB.
- Active run: `/power/ra2a-s10e-ep16-oneshot-pgle-off-standard-schedule-20260809-coord`. Kueue still reports 4 CPU, 195 memory, and 3 GPU exclusions across 202 nodes, so production reservations/topology now dominate even at the bounded small-screen profile. Leave the interactive-priority one-domain gang queued and wait; do not lower host resources further or permit cross-rack placement.
- Verification: the focused suite passes 26 tests and asserts both the small-screen and exact-proxy resource profiles. Required changed-file checks pass.

### 2026-08-09 12:24 UTC - Queue-only timeout and fresh baseline retry

- S10e never received a quota reservation. Its four training tasks stayed `building` for the full wait with zero failures, zero preemptions, and no GPU allocation; Kueue consistently reported that production occupancy left no four-node `multinode-nvlink-ib` domain with the requested resources.
- At 12:19 UTC the coordinator reached its 21,600-second execution timeout. Iris marked the coordinator failed with `Execution timeout exceeded` and killed the still-gated child with `Job exceeded max_task_failures`; neither terminal label represents a training, NCCL, or ragged-collective attempt.
- Fresh retry: coordinator `/power/ra2a-s10f-ep16-oneshot-pgle-off-standard-schedule-20260809-coord`; child `/power/ra2a-s10f-ep16-oneshot-pgle-off-standard-schedule-20260809-coord/grug-train-ra2a-s10f-ep16-oneshot-pgle-off-standard-schedule-20260809`; workload `iris-pg-f9e9858a58dea817-0`.
- Controlled change: extend only the coordinator lifetime from 6 to 12 hours. The training tuple remains d768/L8, E192, expert width 384, top-4, capacity 1.33, EP16, 1,048,576 tokens/step, 60 tokens/active parameter, process-per-GPU, pinned JAX 0.11 one-shot ragged, PGLE off, `NCCL_BUFFSIZE=1048576`, and watch interval zero.
- Resource verification: Kueue records four pods at interactive priority, each requesting four GB200 GPUs, 16 CPU, 128 GiB RAM, 1 TiB ephemeral storage, and four RDMA devices. The fresh workload is again pending behind production with zero failures/preemptions and no allocated GPUs.
- Next action: preserve the fresh queue position; after admission, verify all four training nodes share one NVLink domain, score steps 5-24, and stop the coordinator manually.

### 2026-08-09 13:42 UTC - Stable schedules reproduce a deterministic step-2 NaN

- S10f admitted at 13:28 UTC on `scspxs64`, `sjjvxs64`, `scssxs64`, and `s38vxs64`, all in `DH1-129-US-EAST-08A`. It retained the 60-token-per-active-parameter schedule, process-per-GPU, pinned JAX 0.11 one-shot ragged path, PGLE off, and the baseline NCCL/XLA controls.
- S10f completed a finite first step and then every rank raised `Non-finite loss (nan) at step 2`. The parent was stopped before an identical retry.
- S10g changed only the optimizer token budget to 750 tokens per active parameter. Its four workers `s7htxs64`, `s45sxs64`, `s5xvxs64`, and `s69vxs64` were also all in `DH1-129-US-EAST-08A`; the CPU coordinator was outside the domain and irrelevant to the GPU collective.
- The effective S10g worker environment was `JAX_ENABLE_PGLE=false`, `NCCL_BUFFSIZE=1048576`, overlap limit 4, latency hiding enabled, command buffers disabled, and NCCL termination timeout 600 seconds.
- S10g again completed only step 0: 41.3079 seconds, 25,384.4 tokens/s, 6.4944% MFU, loss 11.8062, and drop fraction 0.000288755. Every rank then raised the same step-2 NaN. The parent was stopped at 13:41:46 UTC before its retry consumed the same nodes.
- The resolved S10g optimizer was MuonH LR 0.00696971 and Adam LR 0.00160840, down from the short probe; both actual step-0 rates were zero during warmup. This rules out the abbreviated schedule and high peak LR as the immediate cause.
- Interpretation: a first backward can still create non-finite optimizer values because IEEE zero times NaN remains NaN. The current ragged tests compare forward values but do not compare gradients or reproduce the QB router-bias transition, which is the meaningful state change before the second forward. The next diagnostic should compare ragged gradients with a stable EP backend on the smallest useful process-per-GPU slice.
- Issue update: https://github.com/marin-community/marin/issues/8077#issuecomment-5231838613

### 2026-08-09 14:09 UTC - One-node EP4 ragged gradients remain finite

- Run: `ra2a-s10k-ep4-ragged-gradient-watch-20260809`; one GB200 node `s53txs64` in `DH1-122-US-EAST-08A`; four processes with one GPU each; pinned JAX 0.11.0 and NCCL 2.30.7.
- Geometry: d768/L8, E12, top-4, capacity 1.33, 262,144 tokens per step, 4-way expert axis, 65,536 tokens and three resident experts per rank. This is the one-node control for the four-node E48 hero-routing proxy.
- Result: the run remained finite through step 37. Loss moved from 11.8038 at step 0 to 11.7899 at step 37; total gradient norm stayed between 0.2056 and 0.3738. Expert `w_down`, `w_gate`, and `w_up` gradients and router gradients were finite on every watched step. Router-bias gradient was zero as designed. No assignment was dropped.
- Timing: steps 5-24 averaged 2.6778 seconds and 97,898.6 tokens/s. W&B reported 99.54% mean MFU, but this tiny one-node proxy is not comparable with the EP16 hero timing gate; inline per-step watch statistics also make it a correctness run rather than a transport benchmark.
- Terminal state: stopped coordinator `/power/ra2a-s10k-ep4-ragged-gradient-watch-20260809-coord` at 14:08:55 UTC after the requested measurement window, releasing four GPUs.
- Interpretation: ragged autodiff is not generically corrupt on process-per-GPU, and the per-rank 65,536-token/three-expert routing geometry is stable within one host. The next controlled arm scales this exact geometry to four hosts and E48; it changes the global expert axis and crosses hosts without returning to E192.

### 2026-08-09 14:38 UTC - Two-node EP8 cross-host ragged gradients remain finite

- Queue refinement: the four-node E48 matched-family control `ra2a-s10l-ep16-e48-gradient-watch-20260809` could not fit any complete NVLink domain. It requested the validated 16 CPU/128 GiB/four-GPU node shape and consumed no GPUs. It was stopped before admission in favor of a useful two-node cross-host control.
- Run: `ra2a-s10m-ep8-e24-gradient-watch-20260809`; nodes `sbxsxs64` and `sdgwxs64`, both in `DH1-126-US-EAST-08A`; eight one-device processes; pinned JAX 0.11.0 and NCCL 2.30.7.
- Geometry: d768/L8, E24, top-4, capacity 1.33, 524,288 tokens per step, 8-way expert axis, 65,536 tokens and three resident experts per rank. This scales the stable EP4/E12 control by two and exercises inter-host ragged traffic.
- Compile control: the 14:35:28 UTC thread profile placed a sampled rank at `train.py:681` waiting on the background data loader. At 14:36:25 UTC sampled rank mains had moved to `backend_compile_and_load` at `train.py:692`, confirming the first batch arrived and separating loader/compile time from collective execution.
- Result: finite through step 38. Loss was 11.7510 at the last sampled step; total gradient norm stayed between 0.1986 and 0.2722; expert-weight and router gradients were finite; no assignment was dropped.
- Timing: steps 5-24 averaged 1.5892 seconds and 329,957 tokens/s. W&B's 167.81% mean MFU is another small-model accounting artifact and is not a hero-performance result.
- Terminal state: stopped coordinator `/power/ra2a-s10m-ep8-e24-gradient-watch-20260809-coord` at 14:38:02 UTC after the requested window, releasing eight GPUs.
- Interpretation: process-per-GPU ragged gradients also remain correct across hosts at EP8. The next two-node arm should use E96: relative to failing EP16/E192, it preserves 65,536 tokens, 12 resident experts, and tokens per routed expert per rank while halving world size and peer count.

### 2026-08-09 14:45 UTC - EP8 E96 reproduces first-backward corruption

- Run: `ra2a-s10n-ep8-e96-gradient-watch-20260809`; reused `sbxsxs64` and `sdgwxs64` in `DH1-126-US-EAST-08A`, giving a same-host comparison with stable S10m. Eight one-device processes covered IDs 0-7 and used fresh coordinator `10.186.210.129:23189`; JAX 0.11.0 and NCCL 2.30.7.
- Controlled geometry: E24 -> E96 while holding d768/L8, EP8, 524,288 tokens per step, 65,536 tokens per rank, top-4, capacity 1.33, optimizer schedule, runtime flags, and hardware fixed. E96 gives 12 resident experts per rank and the same routed-token/expert load as failing EP16/E192.
- Step 0 had finite loss 11.8041, finite parameter norms, zero drops, and a 35.1001-second compile-inclusive duration. Its total gradient norm was NaN.
- Gradient localization: final gated norm, final norm, and output-projection gradients were finite; router-bias gradient was zero as designed. Embeddings and every watched transformer-stack family were NaN, including attention, shared MLP, routed expert, router, and block normalization gradients. This places corruption below the final projection during the first backward rather than in parameter initialization.
- Terminal signature: every rank raised `Non-finite loss (nan) at step 2` at 14:44:03 UTC. No CUDA illegal address, NCCL transport error, OOM, or preemption preceded the numerical failure. The later coordinator connection errors followed process termination and are consequences, not causes.
- Terminal state: stopped coordinator `/power/ra2a-s10n-ep8-e96-gradient-watch-20260809-coord` at 14:44:36 UTC before a retry.
- Interpretation: EP16 world size is not required. On identical EP8 hosts, E24 is stable and E96 corrupts the first backward, so the trigger follows expert/cell geometry. The next arm keeps EP8/E96 fixed and moves to the post-0.11 device-initiated ragged kernel with NCCL symmetric buffers.

### 2026-08-09 14:52 UTC - The nightly device kernel is incompatible with process-per-GPU

- Run: `ra2a-s11-ep8-e96-nightly-device-kernel-watch-20260809`; reused `sbxsxs64` and `sdgwxs64` in `DH1-126-US-EAST-08A`, preserving the failing EP8/E96 model, token geometry, and process-per-GPU topology.
- Runtime verification: every worker installed `jax`, `jaxlib`, `jax-cuda13-plugin`, and `jax-cuda13-pjrt` at `0.11.1.dev20260808`. Effective flags enabled both the post-0.11 device kernel and NCCL symmetric buffers; `JAX_ENABLE_PGLE=false` and `NCCL_BUFFSIZE=1048576` remained fixed. NCCL was 2.30.7.
- Result: all ranks aborted at the first ragged collective, before W&B recorded a training step, with `RET_CHECK failure ... ragged_all_to_all_thunk.cc:943 ... Peer access must be enabled`.
- Source confirmation: the current OpenXLA `RaggedAllToAllThunk::RunCollective` enters the device path only with device communication, symmetric collective memory, and an LSA size, then requires `peer_access_enabled` before locating the symmetric input and output buffers. This is a hard precondition rather than a tunable NCCL transport warning.
- Interpretation: one-device-per-process workers expose no local peer device to XLA, so the device kernel cannot satisfy its intra-host LSA peer-access requirement. This arm conflicts with the required process-per-GPU topology. It does not test the nightly default one-shot or private NCCL send/receive implementations and is not evidence that the nightly itself is numerically bad.
- Terminal state: stopped coordinator `/power/ra2a-s11-ep8-e96-nightly-device-kernel-watch-20260809-coord` at 14:51:31 UTC before Iris retried the fatal abort, releasing eight GPUs.
- Next arm: keep nightly `0.11.1.dev20260808` and EP8/E96 but remove the device-kernel and symmetric-buffer flags. This isolates post-0.11 default one-shot ragged value/gradient behavior under the long-term process-per-GPU topology.

### 2026-08-09 15:01 UTC - The nightly default one-shot path reproduces the E96 gradient corruption

- Run: `ra2a-s12-ep8-e96-nightly-default-watch-20260809`; the exact S10n EP8/E96 geometry ran again on `sbxsxs64` and `sdgwxs64` in `DH1-126-US-EAST-08A` with eight one-device processes. This was a same-host comparison with stable E24, failing stable E96, and the nightly device-kernel screen.
- Controlled change from S10n: install the four JAX packages at `0.11.1.dev20260808`; do not enable the device kernel or symmetric buffers. NCCL 2.30.7, `NCCL_BUFFSIZE=1048576`, PGLE off, model, routing, optimizer, data, and watch interval remained fixed.
- Initialization: all ranks used fresh coordinator `10.186.210.129:39535`. A 14:58 UTC thread profile placed the sampled worker mains at `train.py:681` waiting for their background loaders; it did not mistake the loader delay for a collective stall.
- Result: step 0 had finite loss 11.8090, zero drops, 18,757 tokens/s, and a NaN total gradient norm. Final gated norm, final norm, and output-projection gradients were finite; embeddings and every watched transformer-stack family were NaN, matching S10n. Every rank then raised `Non-finite loss (nan) at step 2`.
- Interpretation: post-0.11 default one-shot ragged reproduces the corruption exactly, while the new device kernel cannot run in process-per-GPU mode. The nightly is therefore not a correctness fix for this geometry. The error remains below the final projection on the first backward and follows the E96/12-local-expert geometry rather than EP8 transport placement.
- Terminal state: stopped coordinator `/power/ra2a-s12-ep8-e96-nightly-default-watch-20260809-coord` at 15:00:48 UTC before its retry, releasing eight GPUs. Later coordinator connection failures followed rank 0's explicit non-finite-loss exception.
- Next diagnostic: run E96 through the ring EP backend on the same two-node process-per-GPU shape. Ring shares the local `ragged_dot` expert computation but replaces ragged all-to-all dispatch/combine with all-gather plus psum-scatter, isolating the collective from the grouped matmul and model.

### 2026-08-09 15:09 UTC - Ring isolates the corruption to ragged all-to-all

- Run: `ra2a-s13-ep8-e96-ring-gradient-control-20260809`; reused `sbxsxs64` and `sdgwxs64` in `DH1-126-US-EAST-08A`, preserving the failing EP8/E96 process-per-GPU geometry, stable JAX 0.11.0 runtime, and NCCL controls.
- Controlled change: set the EP implementation to `ring`. This retains 8-way expert sharding, 96 experts, 12 resident experts per rank, and the same local `haliax.nn.ragged_dot` grouped matmuls. It replaces only ragged all-to-all dispatch/combine with all-gather plus psum-scatter.
- Result: gradients remained finite through the last recorded step 153. Total gradient norm moved from 0.2402 at step 0 to 0.6746 at step 153, loss fell from 11.8090 to 10.7428, and no assignment was dropped. The exact geometry that corrupted the first ragged backward is therefore stable when the collective is removed.
- Timing: steps 5-24 averaged 0.4791 seconds, 1,094,494 tokens/s, and W&B's model-accounting MFU value 5.9965. Ring duplicates tokens through all-gather, and this d768 diagnostic is not a hero-performance candidate; its purpose is the value-and-gradient control.
- Terminal state: stopped coordinator `/power/ra2a-s13-ep8-e96-ring-gradient-control-20260809-coord` at 15:08:34 UTC, releasing eight GPUs.
- Interpretation: shared model math, optimizer, router, and `ragged_dot` are exonerated at EP8/E96. The numerical failure is specific to ragged all-to-all dispatch/combine or its reverse-mode transpose/metadata. The next arm forces XLA's dense ragged-all-to-all decomposer on the otherwise identical stable-runtime ragged run; a finite result would isolate the one-shot kernel, while another NaN would implicate shared primitive transpose/metadata.

### 2026-08-09 15:20 UTC - Dense decomposition fixes the E96 ragged gradient

- Run: `ra2a-s14-ep8-e96-dense-decomposer-watch-20260809`; corrected coordinator `/power/ra2a-s14-ep8-e96-dense-decomposer-watch-20260809-r1-coord`. An initial CPU-only coordinator rejected the non-calendar artifact version before dispatch and consumed no GPUs.
- Placement: `sbxsxs64` and `sdgwxs64`, both in `DH1-126-US-EAST-08A`, exactly match S10m-S13. Eight one-device processes used stable JAX 0.11.0, PGLE off, `NCCL_BUFFSIZE=1048576`, latency hiding enabled, command buffers disabled, and watch-induced collective overlap limit 1.
- Controlled change from failing S10n: add only `--xla_gpu_unsupported_enable_ragged_all_to_all_decomposer=true`. The pass rewrites ragged all-to-all instead of executing the default one-shot kernel.
- Compile verification: a 15:17 UTC thread profile put all four sampled rank mains in `backend_compile_and_load` at `train.py:594`, distinguishing initialization compilation from collective execution.
- Result: finite through the last recorded step 58. Total gradient norm stayed finite from 0.2402 at step 0 through 0.2874 at step 58, with a 0.1945-0.4096 range. Loss fell from 11.8090 to 11.6758, and no assignment was dropped.
- Timing: steps 5-24 averaged 0.8686 seconds, 603,665 tokens/s, and W&B's small-model MFU value 3.3074. Inline gradient-watch work is included, so this is a correctness result rather than the clean transport baseline.
- Terminal state: stopped the corrected coordinator at 15:19:10 UTC after the requested scoring window, releasing eight GPUs.
- Interpretation: the routing metadata, JAX reverse-mode rule, and local MoE math are correct when XLA decomposes the primitive. The default one-shot lowering is the numerical fault at E96/12 local experts. Promote the decomposer to an EP16/E192 four-node no-watch baseline before tuning NCCL/XLA performance knobs.

### 2026-08-09 15:29 UTC - The isolated lowering is XLA's zero-copy direct-P2P path

- Pinned source: JAX 0.11.0 pins OpenXLA `131bf41acb4650e4391a640c3f1859c1c86ad74b` from 2026-07-16. The default multi-host one-shot path uses an NCCL barrier and a CUDA kernel that writes each sender's slices directly into peer symmetric output memory.
- Relevant history: OpenXLA commit `c1fcf2507528eec72374aeb2eddd85d028939cc2` made the zero-copy symmetric-output implementation mandatory on 2026-06-17, removing the earlier double-copy scratch path. The pinned revision includes that change. Nightly S12 still reproduces the NaN, so no later fix through 2026-08-08 closes it.
- Decomposer cost: the pinned dense pass pads every per-peer slice to the full ragged input length, exchanges the resulting dense tuple with ordinary all-to-all, and reconstructs the output with masks. Its full-size padding explains why S14 is a correctness oracle rather than the expected final EP64 performance path.
- Inference: the failing surface is the one-shot symmetric-output kernel, its zero-copy buffer assignment, or its async lifetime/synchronization. It is not the JAX transpose algebra itself: the same transpose metadata is finite under dense decomposition, and ring matches the decomposed step-0 gradient norm (`0.240191` versus `0.240176`).
- Diagnostic queue: after the four-node decomposer baseline, run the process-per-GPU private NCCL send/receive fallback with PGLE off and the 1 MiB FIFO. Then test synchronous one-shot (`--xla_gpu_disable_async_collectives=RAGGEDALLTOALL`) and optional VLOG-5 bounds checking to separate async buffer lifetime from offset bounds. Only finite paths qualify for performance tuning.

### 2026-08-09 15:38 UTC - Four-node slot promoted to the exact hero proxy

- Queue correction: stopped the unadmitted d768/E192 decomposer job `/power/ra2a-s15-ep16-e192-dense-decomposer-perf-20260809-coord`. It allocated no GPUs and had no task failure or preemption.
- Rationale: E192 on EP16 has 12 resident experts per rank and deliberately reproduces the diagnostic geometry. The selected EP64 hero has three. S14 already established dense-decomposer correctness; another small-model window would not address the requested >20% hero MFU gate.
- Active run: `/power/ra2a-s15b-exact-ep16-e48-oneshot-perf-20260809-coord`; child `/power/ra2a-s15b-exact-ep16-e48-oneshot-perf-20260809-coord/grug-train-ra2a-s15b-exact-ep16-e48-oneshot-perf-20260809`.
- Exact geometry: d6144/L48, E48, expert width 6272, latent width 3072, top-4, capacity 1.33, batch 256 x sequence 4096, EP16, and four one-device-per-process GB200 nodes. This preserves 65,536 tokens and three resident experts per GPU from the E192/EP64 hero.
- Optimizer/timing contract: `schedule_steps=17,652,512` represents approximately 750 tokens per 24.680B active parameters at 1,048,576 tokens/step, while `stop_after_steps=25` bounds the experiment. Watch, eval, profiler, checkpoint, and periodic metric work are disabled. Score steps 5-24.
- Runtime: pinned JAX 0.11 default one-shot, PGLE off, `NCCL_BUFFSIZE=1048576`, latency hiding on, overlap limit 4, command buffers off. This is the ideal-kernel exact baseline before forcing fallback or decomposition.
- Scheduling: four pods request four GB200s, 16 CPU, 256 GiB RAM, 1 TiB disk, and one NVLink domain. Kueue reports only three CPU and four GPU exclusions; production memory occupancy excludes 195 nodes. Leave the bounded shape queued rather than weakening placement.

### 2026-08-09 16:23 UTC - Private NCCL fallback stalls at EP8 E96

- Run: `ra2a-s16-ep8-e96-nccl-fallback-watch-20260809`; workers `sbxsxs64` and `sdgwxs64`, both in `DH1-126-US-EAST-08A`. This reused the same hardware as stable E24, failing one-shot E96, ring, and dense-decomposer controls while the exact four-node baseline stayed queued.
- Geometry/runtime: d768/L8, E96, EP8, 524,288 tokens/step, 65,536 tokens and 12 resident experts per rank, eight one-device processes, stable JAX 0.11.0, PGLE off, `NCCL_BUFFSIZE=1048576`, latency hiding on, overlap limit 1 for inline watch, and command buffers off.
- Controlled change from failing one-shot S10n: `--xla_gpu_unsupported_use_ragged_all_to_all_one_shot_kernel=false`, selecting the private grouped NCCL send/receive implementation.
- Progress boundary: data loading completed and a thread sample moved the worker mains into the first `train_step` executable at `train.py:691` / `pxla.py:420`. More than three minutes later W&B still had no step, loss, timing, or gradient metric.
- Device signature: sampled GPUs were all at 100% utilization with 146,687 MiB allocated and 195-227 W power. Bounded logs contained no allocator, CUDA, NCCL, or numerical error. This matches the generic fallback's low-power no-progress signature at a smaller peer count.
- Terminal state: stopped the coordinator at 16:22:43 UTC. Iris killed both workers and released the eight GPUs; no retry ran. The job summary's two preemptions reflect the two user-terminated tasks.
- Interpretation: the private fallback cannot provide a correctness or performance baseline even at EP8/process-per-GPU. Restore one-shot and disable asynchronous `RAGGEDALLTOALL` wrapping next to test whether the zero-copy gradient corruption depends on async buffer lifetime.
- Issue update: https://github.com/marin-community/marin/issues/8077#issuecomment-5232524890

### 2026-08-09 16:30 UTC - Synchronous one-shot reproduces the E96 NaN

- Run: `ra2a-s17-ep8-e96-sync-oneshot-watch-20260809`; same `sbxsxs64` and `sdgwxs64` workers in `DH1-126-US-EAST-08A` as S10m-S14 and S16. The pods initially waited for S16 memory release, then ran normally with no production preemption.
- Controlled change from failing S10n: add only `--xla_gpu_disable_async_collectives=RAGGEDALLTOALL`. Default one-shot remained enabled; process-per-GPU, stable JAX 0.11.0, PGLE off, `NCCL_BUFFSIZE=1048576`, latency hiding on, overlap 1, command buffers off, model, data, and gradient watch remained fixed.
- Compile boundary: 16:27 and 16:28 UTC thread samples showed `backend_compile_and_load`, first for state initialization at line 593 and then for the watched train step at line 691. No no-progress clock was started during compilation.
- Result: step 0 loss 11.8040695, zero drops, 32.275512 seconds, and 16,244.14 tokens/s. Total gradient norm was NaN. Token embeddings, routed expert `w_down`, and router gradients were NaN; final norm and output-projection gradients were finite at 0.00299966 and 0.1653454. This matches the S10n/S12 localization.
- Terminal state: the non-finite update aborted the first attempt and Iris began an identical retry. Stopped the coordinator at 16:30:24 UTC before the retry reached model execution.
- Interpretation: XLA's asynchronous collective wrapper is not necessary for the corruption. Remaining suspects are inside the zero-copy one-shot implementation: symmetric output buffer assignment, direct-P2P kernel synchronization, or offset/index handling. Run VLOG-5 bounds checking next if the exact four-node job remains gated.
- Issue update: https://github.com/marin-community/marin/issues/8077#issuecomment-5232556528

### 2026-08-09 16:43 UTC - VLOG-5 bounds checking is itself racy

- Run: `ra2a-s18-ep8-e96-oneshot-bounds-watch-20260809`; same EP8/E96 process-per-GPU diagnostic geometry and stable JAX 0.11.0 one-shot path as S10n/S17. The only new process setting was `TF_CPP_VMODULE=ragged_all_to_all_thunk=5`.
- Result: all eight ranks failed before W&B recorded a step at `ragged_all_to_all_thunk.cc:297` with `send_sz >= 0 && recv_sz >= 0` and `RaggedAllToAll: Negative sizes detected!`. The failure then produced rank-local segmentation faults and coordinator disconnects. Iris began a retry; the coordinator was stopped at 16:38:59 UTC, leaving the job terminal as user-killed.
- Pinned-source check: `CheckRaggedAllToAllBounds` runs after the pre-kernel NCCL barrier and before the one-shot CUDA kernel. It copies each metadata buffer with `StreamExecutor::SynchronousMemcpyD2H`, which is not enqueued on the collective's execution stream and does not first block that stream. The metadata-producing GPU work can therefore still be in flight when the checker reads it.
- Dtype check: JAX 0.11 accepts any integer metadata and initially lowers the four arrays unchanged, but OpenXLA's `RaggedAllToAllCanonicalizer` converts all four to S64 before `RaggedAllToAllThunk::CheckImplementable`. The runtime's `int64_t` reads are therefore not an i32/i64 mismatch in the executable.
- Interpretation: this arm exposes a broken VLOG diagnostic, not yet a production metadata bug. The negative values are consistent with the checker racing the metadata producers; its failure happens before the one-shot kernel that normally returns finite forward values and corrupts the first backward. Do not use S18 to claim that Marin produced negative routing counts.
- Next action: force CUDA launch synchronization on the same EP8/E96 arm. If that restores finite gradients, the one-shot kernel has a stream-ordering bug. If it still produces NaNs, combine launch blocking with VLOG-5 so the bounds checker can inspect initialized metadata and validate offsets/output-buffer bounds.

### 2026-08-09 16:53 UTC - CUDA launch blocking does not repair one-shot gradients

- Run: `ra2a-s19-ep8-e96-cuda-launch-blocking-watch-20260809`; two workers `sbxsxs64` and `sdgwxs64` in `DH1-126-US-EAST-08A`, matching S10m-S18. Source bundle was clean at `ff8b14c432`, and the effective worker environment added only `CUDA_LAUNCH_BLOCKING=1` to the S10n one-shot controls.
- Result: W&B recorded one row. Step 0 loss was 11.8040695 with zero drops, 23.2055 seconds, 22,593 tokens/s, and NaN total gradient norm. Token embedding, routed expert, router, attention, and normalization gradients were NaN; final norm and output-projection gradients were finite at 0.00299966 and 0.1653454. Every rank then raised `Non-finite loss (nan) at step 2`.
- Teardown: after the explicit non-finite-loss exception, the distributed coordinator closed and rank 0 emitted `FATAL: exception not rethrown`. This occurred after the root training error and is shutdown noise. Iris began attempt 1; the coordinator was stopped at 16:52:04 UTC to release both nodes.
- Interpretation: synchronizing every CUDA launch does not change the value/gradient corruption. Ordinary producer/consumer stream launch overlap is not sufficient to explain the production bug. It also confirms that S18's checker race and the production NaN are separable.
- Next action: add VLOG-5 to the launch-blocking case. With producers synchronized, the checker should read initialized metadata and either pass or identify a real offset/symmetric-output bound violation before the kernel.

### 2026-08-09 17:00 UTC - Synchronized bounds checks pass before the same NaN

- Run: `ra2a-s20-ep8-e96-blocking-bounds-watch-20260809`; same two hosts/domain and EP8/E96 process-per-GPU geometry as S19. Effective worker settings included both `CUDA_LAUNCH_BLOCKING=1` and `TF_CPP_VMODULE=ragged_all_to_all_thunk=5`, verified from the admitted workload.
- Bounds result: the VLOG-5 check ran before every one-shot kernel. No rank reported negative offsets/sizes, input read violation, symmetric-output write violation, or receive-capacity violation. Unlike racy S18, every rank completed the forward and backward executable.
- Numerical result: W&B step 0 loss was 11.8040695 with zero drops, 23.0627 seconds, 22,733 tokens/s, and NaN total gradient norm. Token embedding, expert `w_down`, and router gradients were NaN; final norm and output projection were finite at 0.00299966 and 0.1653454. Every rank then raised the same step-2 NaN.
- Terminal state: Iris began retry 1 after the deterministic failure. Stopped the coordinator at 16:59:17 UTC before the retry executed, releasing eight GPUs.
- Interpretation: producer timing, metadata sign/range, and the checker's declared input/output bounds are not the production fault. The evidence now isolates corruption to the zero-copy direct peer write or NCCL symmetric-window address translation: the kernel receives valid metadata and writes in declared bounds, but the backward output is wrong. Model size changes the allocator/window layout while the ragged data geometry remains nearly identical between stable E24 and failing E96, consistent with an address/offset-sensitive fault.
- Next action: test allocator/window-layout sensitivity with latency hiding disabled, then the allocator choice if needed. Separately preserve the queued exact E48 baseline; three local experts may avoid the bad layout and is the only candidate for a clean hero-scale timing window.

### 2026-08-09 17:07 UTC - Disabling latency hiding fixes EP8 E96

- Run: `ra2a-s21-ep8-e96-lhs-off-watch-20260809`; same `sbxsxs64` and `sdgwxs64` workers in `DH1-126-US-EAST-08A`, stable JAX 0.11.0, process-per-GPU, one-shot ragged kernel, allocator, 1 MiB NCCL FIFO, overlap limit 1, disabled command buffers, model, data, optimizer, and inline gradient watch as failing S10n. The only controlled change was `--xla_gpu_enable_latency_hiding_scheduler=false`.
- Correctness: finite through the last observed step 55. Step 0 total gradient norm was 0.236894, matching the healthy range from ring/decomposer controls instead of NaN. Across steps 5-24 it stayed in 0.198239-0.271718; loss moved from 11.8043 to 11.7833; no assignments were dropped.
- Timing window: steps 5-24 averaged 1.21903 seconds, 430,139 tokens/s, and 2.19333 small-model MFU; median duration was 1.21855 seconds. Inline watch work is included, and this d768 accounting is not the hero MFU gate.
- Terminal state: the run had advanced to step 55 when queried. Stopped the coordinator at 17:06:58 UTC after the requested measurement window, releasing eight GPUs.
- Interpretation: latency hiding is the causal knob for the E96 first-backward corruption. The exact mechanism is likely XLA scheduling/buffer-lifetime interaction with the zero-copy symmetric-output ragged kernel: metadata and declared bounds pass, disabling async wrapping or synchronizing CUDA launches does not help, while disabling the scheduler does. This is specific to the one-shot ragged path; ring and dense decomposition are finite with latency hiding enabled.
- Decision: keep `--xla_gpu_enable_latency_hiding_scheduler=false` for all remaining ragged EP treatments. Preserve the queued LHS-on E48 run as the exact default baseline, then compare an otherwise identical four-node LHS-off run on the same single-domain placement. Only clean exact runs determine whether MFU exceeds 20%.

### 2026-08-09 17:13 UTC - Clean overlap-four replication queued

- Run: `ra2a-s22-ep8-e96-lhs-off-clean-20260809`; coordinator `/power/ra2a-s22-ep8-e96-lhs-off-clean-20260809-coord`; two-node child of the same run ID.
- Controlled change from S21: disable inline gradient watch, which restores the production collective overlap limit from 1 to 4. Keep stable JAX 0.11.0, process-per-GPU, default one-shot, PGLE off, 1 MiB NCCL FIFO, disabled command buffers, latency hiding off, EP8/E96 geometry, data, and 750-token optimizer schedule fixed.
- Purpose: verify the latency-hiding fix survives the clean production-overlap configuration and obtain an uncontaminated small-slice timing window. This is a correctness and relative-throughput screen; its d768 MFU does not count toward the hero's 20% target.
- Scheduling: the coordinator dispatched successfully at interactive priority. Both GPU tasks are pending with zero failures or preemptions. The exact S15b four-node LHS-on baseline remains Kueue-gated with no GPU allocation; production retains priority.
- Launch correction: the first attempt admitted on two nodes but inherited `WANDB_MODE=offline`; it was stopped before step 0 while loading data. Resubmitted the identical configuration as `ra2a-s22-ep8-e96-lhs-off-clean-20260809-r1` with online W&B logging. Its two tasks are temporarily waiting for the stopped pods' memory to drain, with zero training failures.

### 2026-08-09 17:22 UTC - Clean overlap-four run is finite but slower

- Run: `ra2a-s22-ep8-e96-lhs-off-clean-20260809-r1`; workers `sbxsxs64` and `sdgwxs64`, both in rack 126 and NVLink domain `DH1-126-US-EAST-08A`. This exactly matches the S10n-S21 hardware.
- Correctness: finite through the final W&B step 65, with loss moving from 11.8043 at step 5 to 11.7833 at step 24 and zero dropped assignments. No watch metrics were computed. The run was stopped after the requested window to release eight GPUs.
- Timing: steps 5-24 averaged 1.60172 seconds and 327,371 tokens/s; median duration was 1.59992 seconds. W&B's d768 MFU was 1.66930 and is not the hero gate.
- Comparison: S21 averaged 1.21903 seconds despite inline watch work. Its watch configuration forces collective overlap limit 1, while this clean run restored overlap limit 4. The overlap-four clean path is 31.4% longer per step, making overlap serialization the highest-value remaining control.
- Next arm: submitted clean `ra2a-s23-ep8-e96-lhs-off-overlap1-clean-20260809`, changing only the overlap limit from 4 to 1 while retaining latency hiding off and online W&B. This directly measures the overlap effect without watch contamination.

### 2026-08-09 17:27 UTC - Overlap limit one recovers 31.6% throughput

- Run: `ra2a-s23-ep8-e96-lhs-off-overlap1-clean-20260809`; same `sbxsxs64` and `sdgwxs64` hosts and `DH1-126-US-EAST-08A` domain as S22. Effective environment verified latency hiding off, overlap limit 1, PGLE off, `NCCL_BUFFSIZE=1048576`, command buffers off, and online W&B.
- Correctness: finite through final W&B step 50. Loss moved from 11.8043 at step 5 to 11.7833 at step 24, with zero dropped assignments. Stopped the coordinator after the measurement window and released eight GPUs.
- Timing: steps 5-24 averaged 1.21747 seconds and 430,693 tokens/s; median duration was 1.21697 seconds and sample standard deviation 0.01416 seconds. W&B's d768 MFU was 2.19616 and is not the hero gate.
- Controlled comparison: S22's overlap-four mean was 1.60172 seconds and 327,371 tokens/s. Overlap one therefore reduced step duration by 24.0% and increased throughput by 31.6%. It also reproduces S21's 1.21903-second watched result, confirming that the gain is the overlap setting rather than watch contamination.
- Decision: exact ragged treatment uses both `--xla_gpu_enable_latency_hiding_scheduler=false` for correctness and `--xla_gpu_experimental_parallel_collective_overlap_limit=1` for throughput. Preserve S15b's enabled/four-way defaults as the paired baseline.

### 2026-08-09 17:39 UTC - Ragged-only runtime defaults prepared

- Change: `run_grug` now selects latency hiding off and collective overlap limit 1 only when `model.moe_implementation == "ragged_all_to_all"`. Fixed all-to-all, ring, and the FSDP control retain latency hiding on and their prior overlap behavior. Explicit user `XLA_FLAGS` still override every default.
- Regression coverage: the new ragged runtime test failed against the prior latency-hiding-on/overlap-four behavior, then passed after the change. The 31-test EP suite passes serially; 15 relevant Grug variant contracts pass with one platform-specific skip. The unrelated base JSON-tracker contract exceeded its existing 60-second timeout when the full contract file was included and was excluded from the targeted variant run.
- Documentation: the EP hero README records the measured ragged settings and their correctness/performance rationale.
- Experiment isolation: S15b was bundled before this change and retains latency hiding on/overlap four. The exact treatment will set both flags explicitly, so neither member of the pair depends on the moving branch default.

### 2026-08-09 20:26 UTC - Exact control exposed stale retry coordination

- Admission and placement: S15b admitted after waiting behind production. Its four workers `s38vxs64`, `s45sxs64`, `s5xvxs64`, and `s69vxs64` all shared NVLink domain `DH1-129-US-EAST-08A`, rack `dh1-r129-us-east-08a`, and IB leaf group `3799788302995`.
- Runtime verification: all 16 JAX ranks ran one process per GPU. The exact control retained stable JAX 0.11.0, PGLE off, `NCCL_BUFFSIZE=1048576`, latency hiding on, overlap limit 4, command buffers off, and no watch/eval/profile/checkpoint work.
- Attempt 0: all ranks reached `backend_compile_and_load`. Task 1 then exited 137 before a training step, causing Iris to atomically bounce its three siblings and retry the gang. No W&B step, CUDA/NCCL error, Kubernetes OOM event, or production preemption was recorded, so this attempt is not a ragged correctness result.
- Attempt 1 coordination failure: the retry's ranks 0 and 3 selected the stale attempt-0 coordinator at `10.186.210.229:54939`, while ranks 1 and 2 selected the current attempt-1 coordinator at `10.186.210.229:51993`. This reproduces the job-scoped endpoint bug fixed by merged PR #8079. The coordinator was stopped at 20:24 UTC, releasing all 16 GPUs.
- Source fix: cherry-picked merged commit `eafa4d49f7` into this experiment branch as `e00d556874`. The change scopes `jax_coordinator` endpoint names by Iris attempt ID and lives entirely in worker-bundle code; no controller deployment is required. Thirty relevant tests pass. One unrelated compilation-cache contract fails because the locally installed JAX no longer excludes `--xla_gpu_per_fusion_autotune_cache_dir` from its cache key.
- Replacement control: launch S15c from `e00d556874` with the identical model, data, resource, placement, and timing tuple. Explicitly force latency hiding on and overlap limit 4 so the ragged-safe branch defaults do not change the control. Keep the same attempt-scoped coordinator fix in the later off/1 treatment.

### 2026-08-09 21:56 UTC - Process-per-GPU exact proxy needs the hero memory reservation

- S15c admitted on `s38vxs64`, `s45sxs64`, `s5xvxs64`, and `s69vxs64`, preserving the S15b rack-129 placement. All 16 one-device JAX processes joined `jax_coordinator-attempt-0` at one address.
- Task 1 was Kubernetes `OOMKilled` twice after roughly six minutes, at the 256 GiB pod limit. Iris atomically restarted the gang. Attempt 1 proved the coordination fix under a real retry: every replacement process selected `jax_coordinator-attempt-1` and the same fresh port instead of the stale attempt-0 endpoint.
- Neither attempt reached a training step, so S15c provides no ragged correctness or timing result. The coordinator was stopped after the second OOM to end the retry loop and release the nodes.
- The 256 GiB four-node special case assumed node-shared host state. Process-per-GPU runs four independent Python/JAX processes per node, including four copies of initialization and host-offload state. Restored the four-node proxy to the standard hero reservation of 850 GiB. The assigned GB200 nodes expose about 955 GiB allocatable each, leaving about 105 GiB for system headroom.
- Replacement S15d keeps the exact d6144/L48/E48 model, four-node EP16 process topology, latency hiding on, overlap limit 4, PGLE off, 1 MiB NCCL FIFO, and clean steps-5-through-24 timing contract. Only the per-worker host-memory reservation changes from 256 to 850 GiB.

### 2026-08-09 23:36 UTC - Exact default control reaches 11.32% MFU

- Run: `ra2a-s15d-exact-ep16-e48-oneshot-perf-20260809`; workers `s38vxs64`, `s45sxs64`, `s5xvxs64`, and `s69vxs64`, all previously verified in rack 129, NVLink domain `DH1-129-US-EAST-08A`, and IB leaf group `3799788302995`.
- Runtime: d6144/L48, E48, expert width 6272, latent width 3072, top-4, capacity 1.33, EP16, 1,048,576 tokens/step, 16 one-device processes, stable JAX 0.11, PGLE off, 1 MiB NCCL FIFO, latency hiding on, overlap limit 4, command buffers off, and 850 GiB host RAM per node. Watch, eval, profile, and checkpoints were disabled.
- Correctness: all 25 steps completed with finite loss. Across steps 5-24, mean loss was 11.8084345 and both dropped-assignment fraction and router capacity-overflow rate were exactly zero.
- Timing: steps 5-24 averaged 33.7039685 seconds, 31,112.2092 tokens/s, and 11.3219861% MFU. Median duration was 33.669853 seconds. This is the requested exact clean baseline and is 8.68 percentage points below the 20% target.
- W&B: https://wandb.ai/marin-community/marin_moe/runs/ra2a-s15d-exact-ep16-e48-oneshot-perf-20260809
- Serialized treatment S24 changes only latency hiding to off and collective overlap limit from 4 to 1. It retains the exact model, process topology, 850 GiB worker reservation, and metric controls.

### 2026-08-10 01:35 UTC - Ragged-safe flags do not recover exact-shape MFU

- Run: `ra2a-s24-exact-ep16-e48-lhs-off-overlap1-perf-20260809`; the same `s38vxs64`, `s45sxs64`, `s5xvxs64`, and `s69vxs64` rack-129 workers as S15d. Effective process command and environment verified four one-device processes per node, PGLE off, 1 MiB NCCL FIFO, latency hiding off, overlap limit 1, and command buffers off.
- Correctness: all 25 steps completed with finite loss. Steps 5-24 had mean loss 11.8084362, zero dropped assignments, and zero router overflow.
- Timing: steps 5-24 averaged 33.818663 seconds, 31,006.1827 tokens/s, and 11.2834022% MFU. Median duration was 33.8144083 seconds.
- Paired result: relative to S15d's 33.7039685-second mean and 11.3219861% MFU, the safe flags were 0.34% slower and 0.0386 MFU points lower. The exact-shape effect is within noise and does not reproduce the small EP8/E96 overlap-limit gain.
- W&B: https://wandb.ai/marin-community/marin_moe/runs/ra2a-s24-exact-ep16-e48-lhs-off-overlap1-perf-20260809
- Next arm: S25 is a nine-step exact-shape run on the safe flags with a rank-0 XProf capture over steps 5-7. It preserves disabled watch/eval/checkpoint work and is diagnostic rather than a timing candidate. Use the trace to distinguish exposed ragged transport from model compute/rematerialization before selecting any NCCL arm.

### 2026-08-10 02:00 UTC - XProf isolates a 16-block ragged transfer kernel

- Run: `ra2a-s25-exact-ep16-e48-profile-20260810`; the same exact d6144/L48/E48 process-per-GPU configuration and rack-129 topology as S24, shortened to nine steps with a rank-0 XProf capture over steps 5-7. The run completed with finite loss.
- Profile: https://iris.oa.dev/proxy/xprof/open?uri=s3%3A%2F%2Fmarin-us-east-02a%2Ftmp%2Fttl%3D30d%2Fxprof%2Fra2a-s25-exact-ep16-e48-profile-20260810. Durable artifact: `s3://marin-us-east-02a/tmp/ttl=30d/xprof/ra2a-s25-exact-ep16-e48-profile-20260810/plugins/profile/steps-5-to-8`.
- One full traced step attributes 43.92% of exclusive device time to communication. `RaggedAllToAllWithSymmetricMemoryKernelImpl<8>` alone consumes 13.6599 seconds, or 41.3% of the 33.1261-second device timeline. The two NCCL barrier kernels around each transfer consume 0.9519 seconds; all-gather, reduce-scatter, send/recv, and all-reduce are individually smaller.
- The complete three-step XProf kernel table contains 864 ragged transfer launches totaling 41.4599 seconds: 47.986 milliseconds average, 43.744 minimum, and 55.200 maximum. Every launch uses only `grid=(16,1,1)` and `block=(128,1,1)` on a 152-SM GB200. Event metadata reports a 304-block minimum occupancy grid and a suggested 1,024-thread block.
- HLO metadata shows each transfer moves a BF16 buffer with either 262,144 or 348,652 rows of width 3,072, approximately 1.61 or 2.14 GB per invocation. The six ragged calls per layer split evenly between those sizes, yielding roughly 39 GB/s of effective source-byte throughput per GPU.
- The pinned OpenXLA implementation fixes the thread count at 128 and sets the block grid from `num_outputs` and `num_updates_per_output`. This workload has one update per destination, so it launches one block for each of 16 ranks; each block loops across its entire destination slice. Current OpenXLA `main` retains the same launch geometry. There is no XLA flag that increases blocks per large update.
- This changes the experiment ranking. The dominant transfer is a custom direct peer-write kernel, not a NCCL all-to-all. NVLS, IB SHARP, NCCL protocols, and NCCL channel counts can only affect the surrounding barriers and smaller collectives. Even removing all measured communication gives an idealized ceiling of about 20.2% MFU from the 11.32% baseline, so an ordinary NCCL arm cannot plausibly clear 20%.
- Next arm: on a two-node correctness/performance screen, disable the one-shot kernel and select the generic symmetric put/signal backend. This differs from the already-stalled private NCCL fallback and directly bypasses the 16-block kernel. If unavailable or slower, test whether the GXL backend is built before treating an upstream multi-block kernel change as the remaining path.

### 2026-08-10 02:22 UTC - Generic symmetric put/signal rejects the source buffer

- Run: `ra2a-s26-ep8-e96-generic-symmetric-clean-20260810`; d768/L8, E96, top-4, capacity 1.33, EP8 across two four-GPU hosts, 524,288 tokens/step, process-per-GPU, watch disabled, latency hiding off, overlap limit 1, and command buffers off.
- Treatment flags disabled the one-shot kernel and selected `--xla_gpu_ragged_all_to_all_mode=symmetric`. The job admitted on `s5xvxs64` and `s69vxs64`, compiled, and reached its first `jit_train_step` execution. It produced no completed training step.
- Every rank failed in `ncclPutSignal` with `invalid argument`; NCCL reported `srcWinHost is not in a valid symmetric window`. Iris retried the gang once, and the second attempt failed identically. This backend's put/signal lowering requires the source allocation to be in a symmetric NCCL window, which the ordinary XLA input buffer is not.
- The stock GXL collectives factory in the pinned OpenXLA source returns null and its interface marks the backend unimplemented, so there is no second packaged transport backend to screen.
- `save_moe` is not an experiment arm for this ragged implementation: the ragged path does not attach the checkpoint names consumed by that policy. Even if added, the exact per-rank dispatch tensors are approximately 1.61-2.14 GB each per layer; retaining the four named MoE intermediates across 48 layers would exceed GB200 HBM.
- Next arm: keep the working one-shot backend but split each destination's contiguous slice into 32 logical ragged updates. XLA's existing multi-update launch heuristic then raises the EP8 grid from 8 to 256 blocks and the exact EP16 grid from 16 to 512 blocks without changing routing capacity or tensor layout.

### 2026-08-10 03:00 UTC - Multi-update shaping gives a 2.46x proxy speedup

- Run: `ra2a-s27-ep8-e96-split32-clean-20260810`; the same `sbxsxs64` and `sdgwxs64` hosts as the S23 clean baseline. Their logged task IPs map back to the identical nodes, eliminating placement as an explanation for the result.
- Treatment: keep the working one-shot symmetric-memory backend but divide each sender-to-peer slice into 32 contiguous logical updates. The logical routed tensor and per-peer capacity are unchanged; the pinned XLA launch heuristic raises the EP8 kernel grid from 8 to 256 cooperating blocks. Latency hiding remained off, overlap limit remained one, command buffers and PGLE remained off, and watch work remained disabled.
- Correctness: the run remained finite through roughly 2,500 steps before it was stopped to release the two nodes. Across scored steps 5-24, mean loss was 11.7971860 and dropped assignments, dropped-token fraction, and router capacity overflow were all exactly zero.
- Timing: steps 5-24 averaged 0.4951376 seconds and 1,059,088.99 tokens/s; median duration was 0.4979239 seconds. W&B's small-model MFU was 5.80256%, compared with 2.19616% for S23.
- Controlled comparison: S23 averaged 1.21747 seconds and 430,693 tokens/s on the same nodes and runtime settings. Splitting the transfer therefore reduced mean step duration by 59.3% and increased throughput by 2.46x. This is large enough to justify the serialized exact-shape validation.
- W&B: https://wandb.ai/marin-community/marin_moe/runs/ra2a-s27-ep8-e96-split32-clean-20260810
- Next arm: run the exact four-node d6144/L48/E48 configuration with 32 updates per peer and score clean steps 5-24. On EP16, XLA should launch 512 blocks per ragged call instead of 16. Verify all four hosts share one NVLink domain before comparing with S24's 11.28% MFU.

### 2026-08-10 03:18 UTC - Exact multi-update shaping recovers 60.2% throughput

- Run: `ra2a-s28-exact-ep16-e48-split32-perf-20260810`; workers `s38vxs64`, `s69vxs64`, `s7htxs64`, and `s8mtxs64`. All four share rack 129, NVLink domain `DH1-129-US-EAST-08A`, and IB leaf group `3799788302995`, matching the S24 topology class. The first two workers also appeared in S24.
- Runtime: exact d6144/L48/E48, expert width 6272, latent width 3072, top-4, capacity 1.33, EP16, 1,048,576 tokens/step, 16 one-device processes, 850 GiB host RAM per node, latency hiding off, overlap limit one, command buffers and PGLE off, and 1 MiB NCCL FIFO. Watch, eval, profile, and checkpoints were disabled. The only model/runtime change from S24 was 32 ragged updates per peer instead of one.
- Correctness: all 25 steps completed with finite loss. Across steps 5-24, mean loss was 11.8084259 and dropped assignments, dropped-token fraction, and router capacity overflow were exactly zero.
- Timing: steps 5-24 averaged 21.1090181 seconds, 49,674.6925 tokens/s, and 18.0770248% MFU. Median duration was 21.0983272 seconds and duration standard deviation was 0.0597790 seconds.
- Paired result: S24 averaged 33.818663 seconds, 31,006.1827 tokens/s, and 11.2834022% MFU. Splitting each peer transfer into 32 updates therefore reduced step time by 37.58%, raised throughput by 60.21%, and recovered 6.794 MFU points. It remains 1.923 MFU points below the 20% target.
- W&B: https://wandb.ai/marin-community/marin_moe/runs/ra2a-s28-exact-ep16-e48-split32-perf-20260810
- Next arm: repeat a nine-step exact run with an XProf capture over steps 5-7. Confirm the expected 512-block launch, measure the residual ragged transfer and barriers, and use that evidence to decide whether 64 or 128 updates per peer can plausibly close the remaining 1.923-point gap.

### 2026-08-10 03:34 UTC - Post-treatment XProf retires larger split counts

- Run: `ra2a-s29-exact-ep16-e48-split32-profile-20260810`; the exact S28 configuration shortened to nine steps with rank-0 profiling over steps 5-7. It reused the exact same four workers as S28 and completed with finite loss.
- Profile: https://iris.oa.dev/proxy/xprof/open?uri=s3%3A%2F%2Fmarin-us-east-02a%2Ftmp%2Fttl%3D30d%2Fxprof%2Fra2a-s29-exact-ep16-e48-split32-profile-20260810. Durable artifact: `s3://marin-us-east-02a/tmp/ttl=30d/xprof/ra2a-s29-exact-ep16-e48-split32-profile-20260810/plugins/profile/steps-5-to-8`.
- Mechanism: the ragged kernel retained `block=(128,1,1)` and changed from `grid=(16,1,1)` to `grid=(16,1,32)`, exactly 512 cooperating blocks. Both profiles contain 864 launches over three steps. Average launch duration fell from 47.9861 to 2.6984 milliseconds, a 17.78x kernel speedup.
- Per-step kernel totals: ragged transfer fell from 13.8200 to 0.7771 seconds and its NCCL barrier kernels fell from 0.8064 to 0.2949 seconds. Their combined exposed cost fell by 13.5543 seconds per step, from 14.6264 to 1.0721 seconds.
- Ceiling: ragged transfer plus barriers now accounts for only 5.08% of S28's 21.1090-second clean step. Removing both entirely would reach 20.0370 seconds and about 19.044% MFU, still below the target. Splits 64 and 128 therefore cannot close the gap and are retired without another GPU run.
- Remaining aggregate kernel time is 89.36% compute and 10.64% communication. Excluding the custom ragged transfer, ordinary NCCL kernels total about 1.4244 seconds per step; all communication plus the ragged barriers totals 2.4964 seconds. Reaching 20% needs 2.0296 seconds, or removal of 81% of all remaining communication, which rules out protocol/channel tuning as a target-closing arm.
- Final NCCL arm: the working one-shot path still carries `NCCL_BUFFSIZE=1048576`, inherited from the OOMing fallback where smaller per-peer FIFOs were necessary. S30 changes only that value to NCCL's 4 MiB default. This cannot reach 20% by itself, but it tests the last clear EP-specific self-imposed penalty before sealing the sweep.

### 2026-08-10 03:56 UTC - NCCL buffer size is neutral; sweep sealed

- Run: `ra2a-s30-exact-ep16-e48-split32-buff4m-20260810`; the exact S28 split-32 configuration on the same four rack-129 workers as S28 and S29. The only treatment was increasing `NCCL_BUFFSIZE` from 1 MiB to 4 MiB.
- Correctness: all 25 steps completed with finite loss. Across steps 5-24, mean loss was 11.8084264 and dropped assignments, dropped-token fraction, and router capacity overflow were exactly zero.
- Timing: steps 5-24 averaged 21.1213289 seconds, 49,646.8362 tokens/s, and 18.0668877% MFU. Median duration was 21.1042245 seconds and duration standard deviation was 0.1192565 seconds.
- Controlled comparison: S28's 1 MiB run averaged 21.1090181 seconds and 18.0770248% MFU. The 4 MiB treatment was 0.058% slower and 0.010 MFU points lower, so the effect is neutral.
- W&B: https://wandb.ai/marin-community/marin_moe/runs/ra2a-s30-exact-ep16-e48-split32-buff4m-20260810
- Decision: stop the serialized sweep. XProf shows that the custom ragged transfer is no longer the target-closing bottleneck after split-32, and the final conventional NCCL correction was neutral. Clearing 20% on this proxy now requires reducing model compute or a structural communication change capable of removing more than 81% of all remaining communication, not another obvious NCCL protocol, channel, buffer, NVLS, or SHARP setting.
- Final report: https://github.com/marin-community/marin/issues/8077. Reusable incident record: https://echo.oa.dev/wiki/101.

### 2026-08-10 04:47 UTC - Fixed XProf moves the bottleneck to ragged expert compute

- Prompt: the full-rack fixed backend has reported roughly 23-26% MFU. Check whether the four-node ragged proxy should improve at the larger global batch or whether the backend performs different work.
- Geometry: the four-node EP16/E48 proxy and full-rack EP64/E192 hero both carry 65,536 tokens and three resident experts per GPU. Their routed expert width, latent width, top-k, attention shape, and expected assignments per local expert are equal. EP64 raises global batch and GPU count by the same factor, so it does not enlarge per-GPU compute shapes.
- Profile regeneration: regenerated both the existing MHEP-193 selected-shape XProf and S29 with the merged profile generator. S29's regenerated `kernel_stats.json` is byte-for-byte identical to the previous table. The fixed artifact captured four GPU devices for five steps, established by 960 attention-backward launches divided by 48 layers; S29 captured one device for three steps. All comparisons below normalize by 20 and three device-steps respectively.
- Kernel comparison: fixed all-to-all averages 16.3015 seconds of aggregate device-kernel time per step; split-32 ragged averages 20.6900 seconds. `moe_up_down` accounts for 4.5023 seconds fixed and 8.9647 seconds ragged, a 4.4624-second difference that explains the entire 4.3884-second kernel gap.
- Compute mechanism: ragged spends 7.7073 seconds per step in Pallas-Triton `ragged_dot` kernels, whose leading kernels report 12.5-18.75% theoretical occupancy, plus about 1.26 seconds in padding and other expert-scope work. Fixed uses dense NVJet GEMMs and spends 2.8717 seconds in expert `dot_general` kernels, or 4.5023 seconds for the complete expert scope.
- Communication control: all communication kernels total 2.4964 seconds per step for ragged and 3.3829 seconds for fixed. The fixed backend's dispatch and combine scopes are also larger, 2.2981 versus 1.2436 seconds. The residual MFU gap is not a larger-rack or NCCL advantage.
- Caveat: the commonly cited 26.02% mean run was MHEP-009, a smaller 128-expert, width-3072 model that dropped 9.97% of assignments. The selected E192/width-6272 fixed run is the relevant profile and reports about 23.47% median MFU before automatic PGLE; PGLE later improved selected-shape throughput by 3.40%.
- Decision: reopen #8077. The next exact four-node arm changes only `RAGGED_DOT_IMPL=xla`, retaining process-per-GPU, split-32 transport, latency hiding off, overlap one, and clean steps 5-24. This directly tests the packaged alternative grouped-matmul lowering before starting kernel work.

### 2026-08-10 04:51 UTC - Exact XLA grouped-matmul treatment admitted

- Run: `ra2a-s31b-exact-ep16-e48-split32-xla-gmm-20260810`; child job `/power/ra2a-s31b-exact-ep16-e48-split32-xla-gmm-20260810-coord/grug-train-ra2a-s31b-exact-ep16-e48-split32-xla-gmm-20260810`.
- Controlled change: relative to S28, set only `RAGGED_DOT_IMPL=xla`. Retain the exact d6144/L48/E48 shape, split-32 ragged transport, 16 one-device JAX processes, latency hiding off, overlap limit one, command buffers and PGLE off, `NCCL_BUFFSIZE=1048576`, and clean steps 5-24. Watch, eval, profiling, and checkpoints remain disabled.
- Placement: workers `s7nqxs64`, `s62xxs64`, `s14fys64`, and `s2brxs64` all share rack `dh1-r397-us-east-08a`, fabric `US-EAST-08A-FAB27`, leaf `400.6-DH1`, and topology zone `397`.
- Runtime verification: task logs enumerate global process IDs 0-15 with one visible GPU each. A narrow worker environment check confirms `RAGGED_DOT_IMPL=xla`, `NCCL_BUFFSIZE=1048576`, and `XLA_FLAGS='--xla_gpu_experimental_parallel_collective_overlap_limit=1 --xla_gpu_enable_latency_hiding_scheduler=false --xla_gpu_enable_command_buffer= --xla_gpu_nccl_termination_timeout_seconds=600'`.
- Launch correction: the first S31 coordinator stopped before worker creation because the label `2026.08.10-s31` failed the launcher's CalVer validation. S31b uses valid version `2026.08.10.31`; the failed coordinator consumed no GPU time or experiment data.
- Monitoring: a 30-minute terminal-state monitor is armed. Score only completed steps 5-24 and require finite loss plus zero unexpected routing drops before comparing with S28's 21.1090-second, 18.0770% MFU baseline.

### 2026-08-10 05:22 UTC - Packaged XLA ragged dot OOMs before step zero

- Run: `ra2a-s31b-exact-ep16-e48-split32-xla-gmm-20260810`; same-rack placement and effective runtime controls as the launch contract above.
- Result: no optimizer step completed. Multiple ranks failed in `jit_train_step` while requesting one 171,068,784,344-byte allocation, reported by JAX as 159.32 GiB. `cuMemAllocAsync` returned `CUDA_ERROR_OUT_OF_MEMORY`; sibling ranks then reported coordination connection failures after the first processes exited.
- Interpretation: `jax.lax.ragged_dot_general` is not a viable packaged replacement at the exact d6144/E48 shape. The coordination errors are secondary, not the root cause. Repeating the same exact arm or raising the allocator ceiling cannot service a single 159.32 GiB temporary alongside resident model state.
- Resource state: Iris marked the four-node child `worker_failed`, the coordinator failed, and all GPU workers were released. The terminal monitor initially omitted Iris's `worker_failed` spelling from its grep predicate; the explicit 30-minute inspection recovered the already-terminal root cause.
- Next arm: reuse the existing SM100 QuACK/CuTe grouped GEMMs for the ragged EP local expert MLP. Preserve XLA `ragged_dot` only for the two weight-gradient contractions as the existing `sonic_cute` path does. Validate values/gradients and a small accelerator shape before promoting to the exact four-node performance arm.

### 2026-08-10 05:42 UTC - CuTe expert backend passes the four-node correctness screen

- Integration: added an explicit `ragged_all_to_all_cute` MoE implementation. It retains the split-32 ragged dispatch/combine path, runs the forward, down, `dh`, and `dx` grouped GEMMs through the existing QuACK/CuTe SM100 kernels, and retains XLA `ragged_dot` only for the two weight-gradient contractions. The original ragged implementation remains unchanged, and unsupported activations fail fast instead of silently falling back.
- Run: `ra2a-s32-ep16-e48-split32-cute-correctness-20260810`; child `/power/ra2a-s32-ep16-e48-split32-cute-correctness-20260810-coord/grug-train-ra2a-s32-ep16-e48-split32-cute-correctness-20260810`.
- Geometry: d768/L8, E48, top-4, capacity 1.33, 1,048,576 tokens per step, EP16 across four GB200 nodes, split-32 ragged transport, and 16 one-device JAX processes. This preserves the exact proxy's 65,536 tokens and three resident experts per GPU while reducing only model width and depth.
- Placement: workers `s53txs64`, `s4150t64`, `s1csxs64`, and `s1b62nb4` all shared rack `dh1-r122-us-east-08a`, NVLink domain `DH1-122-US-EAST-08A`, fabric `US-EAST-08A-FAB27`, and IB leaf `130.1-DH1`.
- Result: all 52 optimizer steps completed and every recorded training loss was finite, decreasing from 11.806656 at step 0 to 5.592094 at step 51. The job and all four workers succeeded without retry, preemption, or kernel/runtime error.
- Diagnostic timing: steps 5-24 averaged 0.506422 seconds, 2,072,229 tokens/s, and 5.2741% small-model MFU. The narrow-model routing distribution dropped up to 14.0052% of assignments at capacity 1.33, so this run is a value/gradient execution screen rather than a performance or loss comparison. The exact E48 proxy has already demonstrated zero drops at the same capacity.
- W&B: https://wandb.ai/marin-community/marin_moe/runs/ra2a-s32-ep16-e48-split32-cute-correctness-20260810
- Decision: promote the explicit CuTe backend to the exact d6144/L48/E48 four-node timing arm, with watch, eval, profiling, checkpoints, and periodic metrics disabled. Score only clean steps 5-24 and compare with S28's 21.1090-second, 18.0770% MFU split-32 baseline.

### 2026-08-10 05:45 UTC - Exact CuTe performance arm admitted

- Run: `ra2a-s33-exact-ep16-e48-split32-cute-perf-20260810`; child `/power/ra2a-s33-exact-ep16-e48-split32-cute-perf-20260810-coord/grug-train-ra2a-s33-exact-ep16-e48-split32-cute-perf-20260810`.
- Controlled change: relative to S28, select only the explicit `ragged_all_to_all_cute` expert backend. Retain the exact d6144/L48/E48 shape, expert width 6272, latent width 3072, top-4 routing, capacity 1.33, split-32 transport, EP16, 1,048,576 tokens per step, 16 one-device JAX processes, latency hiding off, overlap limit one, command buffers and PGLE off, `NCCL_BUFFSIZE=1048576`, and 850 GiB host memory per node. Watch, evaluation, profiling, checkpoints, and periodic metrics are disabled.
- Placement: S33 reused the S32 workers `s53txs64`, `s4150t64`, `s1csxs64`, and `s1b62nb4`. All four share rack `dh1-r122-us-east-08a`, NVLink domain `DH1-122-US-EAST-08A`, fabric `US-EAST-08A-FAB27`, and IB leaf `130.1-DH1`.
- Runtime verification: all 16 one-GPU processes started. The effective environment is `JAX_ENABLE_PGLE=false`, `NCCL_BUFFSIZE=1048576`, and `XLA_FLAGS='--xla_gpu_experimental_parallel_collective_overlap_limit=1 --xla_gpu_enable_latency_hiding_scheduler=false --xla_gpu_enable_command_buffer= --xla_gpu_nccl_termination_timeout_seconds=600'`.
- Monitoring: poll terminal state every 30 minutes. On success, require finite loss and zero unexpected routing drops, then score clean steps 5-24 against S28's 21.1090181-second, 49,674.6925-token/s, 18.0770248% MFU baseline. Profile the exact CuTe path before selecting any further runtime or NCCL treatment.

### 2026-08-10 06:17 UTC - CuTe reaches 19.63% MFU without dropping tokens

- Completion: S33 and all four workers succeeded with exit 0, zero failures, and zero preemptions. All 25 optimizer steps completed on the verified rack-122 placement with 16 one-device JAX processes.
- Correctness: every scored loss was finite. Across steps 5-24, mean loss was 11.8084195, and dropped assignments, drop fraction, and capacity overflow were exactly zero.
- Timing: steps 5-24 averaged 19.4428557 seconds, 53,939.6848 tokens/s, and 19.6290902% MFU. Median duration was 19.3697725 seconds, median MFU was 19.7000425%, and duration sample standard deviation was 0.2561011 seconds.
- Controlled comparison: S28's split-32 ragged-dot backend averaged 21.1090181 seconds, 49,674.6925 tokens/s, and 18.0770248% MFU. QuACK/CuTe reduced mean step time by 7.893%, raised throughput by 8.586%, and recovered 1.5521 MFU points. It remains 0.3709 MFU points below the 20% target.
- W&B: https://wandb.ai/marin-community/marin_moe/runs/ra2a-s33-exact-ep16-e48-split32-cute-perf-20260810
- Issue update: https://github.com/marin-community/marin/issues/8077#issuecomment-5236592758

### 2026-08-10 06:18 UTC - Exact CuTe XProf submitted

- Run: `ra2a-s34-exact-ep16-e48-split32-cute-profile-20260810`; coordinator `/power/ra2a-s34-exact-ep16-e48-split32-cute-profile-20260810-coord`.
- Diagnostic contract: retain S33's exact model, routing, split-32 transport, CuTe expert backend, process topology, memory reservation, PGLE-off, 1 MiB NCCL FIFO, latency-hiding-off, overlap-one, and disabled command buffers. Shorten the run to nine steps and capture rank 0 over steps 5-7 with HLO metadata. Watch, evaluation, checkpoints, and periodic metric work remain disabled.
- Decision rule: use the profile to measure residual expert compute, ragged transport, ordinary NCCL, and launch gaps. Test Blackwell command buffers or another runtime treatment only if the trace exposes a mechanism capable of recovering the remaining 0.3709 MFU points.

### 2026-08-10 06:58 UTC - CuTe XProf confirms the residual weight-gradient hotspot

- Run: `ra2a-s34-exact-ep16-e48-split32-cute-profile-20260810`; all four workers succeeded on the same `s53txs64`, `s4150t64`, `s1csxs64`, and `s1b62nb4` rack-122 placement as S33. The rank-0 XProf capture covers steps 5-7 and is available at https://iris.oa.dev/proxy/xprof/open?uri=s3%3A%2F%2Fmarin-us-east-02a%2Ftmp%2Fttl%3D30d%2Fxprof%2Fra2a-s34-exact-ep16-e48-split32-cute-profile-20260810.
- Regeneration: reran the merged fast profile generator from `a5f0269edc` against the uncapped XPlane protobuf. The structured summary is `/tmp/marin-ragged-profile-summary-s34-fast.json`; the raw xprof tables are `/tmp/marin-ragged-xprof-tables-s34-fast`.
- Total device-kernel time fell from 20.6900 seconds/step in S29 to 18.8072 seconds/step in S34, a 1.8828-second reduction consistent with S33's clean timing uplift. The S34 trace is 84.96% compute and 15.04% communication.
- The two retained Pallas weight-gradient kernels are now the leading structural expert hotspot: `dw13` is 1.4678 seconds/step and `dw2` is 0.6159 seconds/step, 2.0837 seconds/step combined. Both use a 128x128 output tile, 65,552 bytes of shared memory, 154-155 registers per thread, and only 18.75% theoretical occupancy.
- The split-32 custom ragged transfer is 0.7881 seconds/step and its barriers are 0.4138 seconds/step. The trace already contains `ncclSymkDevKernel_AllGather_STMC`, confirming that NCCL selected its NVLS multicast path automatically; forcing NVLS or SHARP cannot accelerate the custom direct peer-write kernel.
- Target math: S33 needs 19.0823 seconds/step to reach 20% MFU, a 0.3606-second improvement. A 17.3% reduction in the two weight-gradient kernels would suffice, so screen exact weight-gradient tiles on one GB200 before spending another four-node run.

### 2026-08-10 07:12 UTC - Exact one-GPU tile sweeps find only a sub-target win

- Harness: added `ragged_weight_grad_benchmark.py`, which instantiates the production Pallas-Triton kernel body at the exact S34 HLO shapes: `dw13` `(348672,3072) x (348672,12544)` and `dw2` `(348672,6272) x (348672,3072)`, each with three local expert groups. It records compile and five-run steady-state timings plus exact-output deviation in machine-readable rows.
- Submission corrections: a direct module invocation used Marin's CPU fallback client and consumed no GPU; a federated `marin` root could not schedule a GB200 child and also consumed no GPU. The valid runs submitted their whole trees directly to `cw-us-east-08a` at interactive priority.
- S35 (`ra2a-s35-weight-grad-tile-bench-20260810-coord-r1`) swept 64/128 output tiles and two/four stages. The production 128x128, K32, four-warp/four-stage tile won both shapes at 32.5246 ms for `dw13` and 12.6110 ms for `dw2`; every smaller tile or two-stage treatment was slower. Every candidate matched the baseline output exactly.
- S36 (`ra2a-s36-weight-grad-large-tile-bench-20260810-coord`) swept K16/K64, eight warps, and 256-wide output tiles. `dw2` again favored the production tile at 12.3833 ms. `dw13` improved from 30.3105 to 28.0236 ms with a 128x256, K32, eight-warp/four-stage tile, a 7.54% kernel gain; every output again matched exactly.
- Ceiling: applying the `dw13` microbenchmark gain to S34 saves approximately 0.1107 seconds/step and predicts 19.74% MFU, still below the 20% target. Do not promote this tile alone to four nodes. The remaining bounded runtime arm is command-buffer enablement because S34 leaves only about 0.56 seconds/step between aggregate kernel time and clean wall time; this arm has explicit prior-risk evidence and must remain isolated from kernel changes.

### 2026-08-10 07:14 UTC - Exact command-buffer arm submitted

- Run: `ra2a-s37-exact-ep16-e48-split32-cute-command-buffers-20260810`; coordinator `/power/ra2a-s37-exact-cute-command-buffers-perf-20260810-coord` on `cw-us-east-08a` at interactive priority.
- Controlled change: relative to S33, enable XLA GPU command buffers for `FUSION,CUSTOM_CALL`. Retain the exact d6144/L48/E48 model, split-32 transport, CuTe expert backend, process-per-GPU topology, latency hiding off, overlap limit one, PGLE off, 1 MiB NCCL FIFO, and 850 GiB per worker. Watch, evaluation, profiling, checkpoints, and periodic metric work remain disabled.
- Decision rule: require all 25 finite steps, zero routing drops, and same-rack placement before scoring steps 5-24. If the arm fails with the known CUDA graph/custom-call failure, test `FUSION` alone only when the failure implicates custom-call capture. If it succeeds below 19.89% MFU, the independently measured 0.1107-second kernel-tile ceiling cannot make a combined arm cross 20%, so stop without spending that run.

### 2026-08-10 07:34 UTC - Command buffers regress the exact proxy

- Completion: S37 and all four workers succeeded with exit 0 after 25 optimizer steps. Every recorded loss was finite; dropped assignments, drop fraction, and capacity overflow were zero. The post-rank-zero coordination warnings occurred during normal teardown, and all 16 GPU processes exited cleanly.
- Placement: workers `sbmvxs64`, `sc8qxs64`, `scypxs64`, and `sfjrxs64` all shared rack `dh1-r126-us-east-08a`, NVLink domain `DH1-126-US-EAST-08A`, fabric `US-EAST-08A-FAB27`, and IB leaf `130.5-DH1`.
- Timing: steps 5-24 averaged 19.7338456 seconds, 53,136.8825 tokens/s, and 19.3369439% MFU. Median duration was 19.7102675 seconds and median MFU was 19.3597243%.
- Controlled comparison: command-buffer-off S33 averaged 19.4428557 seconds and 19.6290902% MFU. Enabling `FUSION,CUSTOM_CALL` increased duration by 1.497%, reduced throughput by 1.488%, and lost 0.2921 MFU points.
- Combined ceiling: stacking S36's independently measured 0.1107-second `dw13` tile estimate perfectly with S37 would predict only about 19.45% MFU. No combined four-node arm is justified.
- Decision: keep command buffers disabled. The investigation has exhausted the obvious EP runtime, conventional NCCL, transport-parallelism, packaged grouped-matmul, existing CuTe, and bounded Pallas tile changes. Preserve S33 as the selected configuration and require a full-rack EP64 validation before production adoption.
- W&B: https://wandb.ai/marin-community/marin_moe/runs/ra2a-s37-exact-ep16-e48-split32-cute-command-buffers-20260810
- Issue update: https://github.com/marin-community/marin/issues/8077#issuecomment-5237216568

### 2026-08-10 17:06 UTC - Alternative grouped-MoE campaign opened

- Hypothesis: a grouped-Wgrad or persistent MoE implementation outside JAX's packaged ragged contractions can remove at least 0.3606 seconds/step and raise the exact proxy above 20% MFU.
- Commit Hash: `b5ecce804e` before the new harness work.
- Control: S33 at 19.4428557 seconds/step, 53,939.6848 tokens/s, and 19.6290902% mean MFU over steps 5-24. The two S34 Pallas weight-gradient contractions consume 2.0837 seconds/step, so the target is a 17.3% combined reduction.
- XPlane idle analysis: merging all physical GPU compute/copy intervals across the three-step S34 trace gives 58.8417 seconds of wall span, 57.3471 seconds busy, and 1.4946 seconds idle. That is 0.4982 seconds idle per traced step and 97.46% physical GPU utilization. The largest 85-163 millisecond gaps cluster around step-boundary D2H/H2D copies, consistent with offloaded optimizer-state staging but not yet semantically attributed. Removing 72% of all observed idle time would be required to cross 20%; this is an upper bound, not an expected gain.
- Internal prior work: commit `0dd141a03e` contains a four-GB200 Mixture-of-Kittens/DeepEP prototype. At its different T2048/rank, E384/96-local, top-6, H7168, I3072 shape, the MoK oracle measured 3.613 milliseconds forward and 9.077 milliseconds backward; the generated DeepEP plus standalone-MoK path measured 4.478 milliseconds sequential and 4.268 milliseconds with shared-expert overlap. These numbers establish feasibility but do not predict the hero shape.
- External candidates: Transformer Engine releases include JAX BF16 grouped GEMM with device-side group sizes; cuDNN Frontend exposes Blackwell grouped GEMM plus Wgrad operations; Mixture-of-Kittens supplies a deterministic SM100 forward/backward megakernel; CUTLASS is the custom-kernel fallback. MegaBlocks is retained as an algorithmic reference, not the first JAX/GB200 port.
- Transport boundary: checked-in DeepEP changes dispatch/combine but retains the same `ragged_dot` expert compute, so DeepEP alone cannot remove the S34 Wgrad hotspot. Compose it with a winning compute backend only after the kernel gate.
- Serialized queue: Transformer Engine API/exact-shape screen; cuDNN API/exact-shape screen; four-GPU exact-shape MoK oracle; winning four-node same-rack S33 integration; one semantic optimizer-offload profile and targeted overlap treatment.
- Issue update: https://github.com/marin-community/marin/issues/8077#issuecomment-5243442771
- Next action: probe the installed GPU image and dependency lock for Transformer Engine/cuDNN APIs, then extend the one-GB200 harness with the smallest viable backend adapter.

### 2026-08-10 17:17 UTC - cuDNN grouped Wgrad clears the kernel gate

- Run: `ra2a-s39-cudnn-wgrad-bench-20260810-coord`; child `/power/ra2a-s39-cudnn-wgrad-bench-20260810-coord/cudnn-ragged-weight-grad-gb200`. Both tasks succeeded without failure, retry, or preemption on one NVIDIA GB200.
- Harness: commit `1b0d2bf9f2` dynamically installs `nvidia-cudnn-frontend==1.27.0` and measures the exact S34 physical rows and Wgrad dimensions. The active group sizes `[116218, 116217, 116217]` become three 116224-row groups by placing 6, 7, and 7 zeros in the existing 20-row routing tail; no additional physical rows are allocated. Every tested output matched a Torch BF16 dense reference exactly.
- Best configuration: 256x256 MMA tile with a 2x1 cluster. `dw13` fell from the production Pallas microbenchmark's 30.3105 milliseconds to 20.6860 milliseconds, a 31.75% reduction. `dw2` fell from 12.3833 to 9.6076 milliseconds, a 22.41% reduction. Combined time fell from 42.6938 to 30.2937 milliseconds, a 29.04% reduction. The 128x128 alternatives were slower on both shapes.
- Projection: scaling the S34 `dw13` and `dw2` kernel totals by their respective measured ratios saves 0.6041 seconds/step. Applied to S33, that predicts 18.8387 seconds/step and 20.2586% MFU before padding, adapter, or launch overhead. The path can absorb at most 0.2436 seconds/step of integration overhead and still reach 20%.
- Confidence: exploratory. This is one standalone Torch/cuDNN Frontend run with five steady-state samples per shape, not yet a JAX FFI or full-training measurement. The result is large enough to justify adapter work, but the projection is not a training result.
- Artifact: `s3://marin-us-east-02a/marin/benchmarks/cudnn-ragged-weight-grad-gb200/2026.08.10.39`.
- Next action: run the serialized Transformer Engine exact-shape feasibility probe, then design the smallest JAX FFI wrapper around the cuDNN Frontend operation while the four-GPU MoK oracle remains queued.

### 2026-08-10 18:02 UTC - Transformer Engine is not an available JAX arm

- Probe: `ra2a-s40i-transformer-engine-wgrad-bench-20260810-coord`, following setup-only S40-S40h attempts. The final child completed on one GB200 without retry or preemption and stored `s3://marin-us-east-02a/marin/benchmarks/transformer-engine-ragged-weight-grad-gb200/2026.08.10.50`.
- Packaging result: Transformer Engine 2.17.1's JAX source extension requires the CUDA 13, CCCL, cuDNN, NCCL, NVTX, and cuDNN Frontend development headers plus an unversioned NCCL linker alias. With those supplied from the pinned split Python SDKs, the aarch64 extension built successfully in 153.7 seconds.
- Availability result: the successful target install did not expose an importable `transformer_engine_jax` module, so no capability or kernel timing row could be produced. Earlier failures were build-environment omissions rather than backend measurements; the final result establishes that the current aarch64 Python install route is not usable by Marin without more packaging work.
- Decision: stop spending GPU time on the redundant TE wrapper. The native cuDNN Frontend Wgrad kernel already supplies the exact positive timing and has a much smaller JAX CuTe bridge. Keep TE as a future packaging/upstream option, not a candidate for this campaign.
- Issue update: https://github.com/marin-community/marin/issues/8077#issuecomment-5244031038.
- Next arm: S41 runs the pinned Mixture-of-Kittens main branch on one four-GB200 node using four `torchrun` processes, exact per-GPU token/hidden/expert geometry, an intermediate width padded from 6272 to 6400, and the upstream shared-expert work. Six communication/minibatch schedules run serially inside the allocation.

### 2026-08-10 18:29 UTC - Mixture-of-Kittens exact-local oracle is correct but not directly adoptable

- Run: `ra2a-s41f-mok-hero-bench-20260810-coord`; child `/power/ra2a-s41f-mok-hero-bench-20260810-coord/mok-hero-layer-gb200x4/0`. One GB200 node ran four `torchrun` ranks, one process per GPU, with no cross-node or cross-rack traffic.
- Harness: pinned `cursor/mixture-of-kittens` at `6438bf48f88094d305972fbe0fa6deba0f7d4d1a`, Torch 2.10.0/CUDA 13.0, and built the SM100 extension in 181.5 seconds. The installed extension import probe and exact benchmark both completed. Earlier S41 attempts were harness setup or package-resolution failures and contain no implementation timing result.
- Geometry: 65,536 local tokens, hidden size 6,144, top-4, three local and 12 global routed experts. MoK requires its intermediate dimension to divide 256, so the hero width was padded from 6,272 to 6,400. The current API and megakernel also require one shared expert, which the hero model does not have.
- Correctness: every rank reported finite output and gradients, and repeating the upstream-default schedule produced bitwise-identical output on all ranks.
- Schedule result: the best of six serialized schedules was 20 forward communication SMs, 28 backward communication SMs, minibatch 8,192, and macrobatch 262,144. Its median forward, backward, and combined times were 70.9924, 138.1374, and 210.6111 milliseconds. That is 8.30% faster than the 229.6828-millisecond upstream default.
- Comparison: S34's current expert compute scope is approximately 8.96 seconds over 48 layers, or 186.7 milliseconds/layer. The measured MoK path is 12.8% slower, but it performs a mandatory fifth, shared expert and 2.04% wider expert matrices. Removing those costs is not a configuration option: the current Python signatures, buffer allocation, scheduler task counts, epilogues, and forward/backward megakernels all explicitly include shared-expert tensors and work.
- Decision: do not integrate the current upstream MoK package into the hero run. It remains the strongest persistent-megakernel reference and could be competitive after a routed-only upstream kernel variant, but that requires nontrivial CUDA kernel surgery rather than a JAX adapter or launch knob. Promote the already-positive cuDNN grouped-Wgrad result to its JAX bridge instead.
- Artifact: `s3://marin-us-east-02a/marin/benchmarks/mok-hero-layer-gb200x4/2026.08.10.56`.
