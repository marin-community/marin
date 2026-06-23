# Pallas MoE Dispatch-Up Mosaic GPU: Research Logbook

## Scope

- Goal: Build a DeepEP-free single-node EP8 Grug MoE backend using Pallas
  Mosaic GPU remote refs. The first subkernel target is dispatch + W13/SiLU
  (`moe_dispatch_up`), before W2/combine and backward integration.
- Primary metrics: fused dispatch + W13/SiLU time, MoE forward time,
  MoE forward+backward time, end-to-end fwd+bwd+SGD MFU.
- Constraints: single-node 8xH100 first, `model_axis=1`, no DeepEP/NCCL
  EP/FFI transport/NIC/RDMA/NVSHMEM in this path.
- Experiment issue: https://github.com/marin-community/marin/issues/6597
- Spec: `.agents/projects/pallas_mosaic_mgpu_moe_spec.md`

## Baseline

- Date: 2026-06-23
- Code refs:
  - `lib/levanter/src/levanter/grug/_moe/ep_ring.py`
  - `lib/levanter/src/levanter/grug/_moe/ep_ragged_all_to_all.py`
  - `lib/levanter/src/levanter/grug/_moe/ep_padded_all_to_all.py`
  - `lib/levanter/src/levanter/grug/_moe/sonic.py`
  - `lib/levanter/src/levanter/grug/grug_moe.py`
- Baseline numbers:
  - May387 DeepEP EP16/N2 B64 `save_moe`: about 17 MFU, but DeepEP remains
    fragile.
  - May363/May378 EP8/B16 readable profiles: MoE around 37-40% of step time.
  - Ring/ragged A2A comparison: `ragged_all_to_all` was slower than ring in the
    tested shape.

## Experiment Log

### 2026-06-23 - Kickoff spec handoff

- Hypothesis: A Pallas Mosaic GPU dispatch-up subkernel can route token blocks directly into
  destination expert-major tiles and consume them with W13/SiLU, avoiding
  DeepEP's opaque FFI/runtime fragility and avoiding high-level collective
  permutation overhead.
- Command: No benchmark command yet; this entry creates the research branch,
  spec, and issue.
- Config: Target is single-node EP8 8xH100, top-k 4, bf16 activations/weights,
  May D2560 Grug MoE shape.
- Result: Created issue #6597 and checked in the initial implementation spec.
- Interpretation: The spec deliberately distinguishes a remote-dispatch
  validation slice from the product target. The first product target is fused
  dispatch + W13/SiLU; training integration also requires backward/custom VJP.
- Next action: Assign an implementation agent to start with the routing/layout
  JAX reference and Mosaic remote-dispatch validation slice.

### 2026-06-23 - Milestones 1 and 2 validation slice

- Hypothesis: A shard-local Mosaic GPU remote-ref dispatch can reproduce the
  prepacked expert-major layout exactly, and a local W13/SiLU Pallas kernel can
  consume that layout for top-k 1 and top-k 4 routing.
- Commands:
  - `uv run --package marin-levanter --group test pytest lib/levanter/tests/kernels/test_pallas_moe_dispatch_up.py -q`
  - `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job run --gpu H100x8 --enable-extra-resources --cpu 16 --memory 128GB --disk 50GB --extra gpu --timeout 1800 --job-name 6597-mosaic-mgpu-smoke-20 -- ... bench_moe_dispatch_up_mosaic_gpu.py --ep-size 8 --tokens-per-rank 8 --experts-per-rank 1 --top-k 1 --hidden 64 --intermediate 64 --dtype bf16 --run-pallas`
  - `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job run --gpu H100x8 --enable-extra-resources --cpu 16 --memory 128GB --disk 50GB --extra gpu --timeout 1800 --job-name 6597-mosaic-mgpu-smoke-21 -- ... bench_moe_dispatch_up_mosaic_gpu.py --ep-size 8 --tokens-per-rank 8 --experts-per-rank 4 --top-k 4 --hidden 64 --intermediate 64 --dtype bf16 --run-pallas`
- Config: Single-node CoreWeave `cw-us-east-02a`, H100x8, explicit `expert`
  mesh, bf16 activations/weights, `H=64`, `I=64`, `T/rank=8`.
- Result:
  - Local tests: 8 passed.
  - Top-k 1: dispatch payload max error 0, dispatch metadata errors
    `valid=0 local_expert=0 src_rank=0`, W13/SiLU max bf16 error 2.
  - Top-k 4: dispatch payload max error 0, dispatch metadata errors
    `valid=0 local_expert=0 src_rank=0`, W13/SiLU max bf16 error 2.
- Interpretation: Milestone 1 is validated for the prepacked remote-ref layout.
  Milestone 2 is validated for local dispatch-up W13/SiLU on the dispatched layout at
  the requested top-k 1 and top-k 4 slices. The dispatch kernel is still a
  deliberately serial validation primitive, not a performance implementation.
- Next action: Replace serial scalar remote writes with tiled/staged copies and
  add backward/custom-VJP coverage before Grug integration.

### 2026-06-23 - Backward oracle kickoff

- Hypothesis: A hand-written VJP for dispatch + W13/SiLU can match JAX autodiff
  on small routed shapes and serve as the target contract for a Mosaic GPU
  backward kernel.
- Command: `uv run --package marin-levanter --group test pytest lib/levanter/tests/kernels/test_pallas_moe_dispatch_up.py -q`
- Config: CPU reference, EP2, top-k 2, two local experts per rank, explicit
  receive capacity larger than per-sender send capacity.
- Result: 9 tests passed. The explicit backward oracle matches autodiff for
  gradients with respect to source activations and W13 weights.
- Interpretation: Backward can be decomposed into local W13/SiLU VJP followed
  by reverse dispatch from destination rows to source-rank token gradients.
- Next action: Port the W13/SiLU VJP and reverse dispatch to Mosaic GPU, then
  profile the serial forward dispatch to prioritize tiled/staged copies.

### 2026-06-23 - Steady-state perf harness kickoff

- Hypothesis: Separating compile-including timings from steady-state timings
  will make the serial dispatch bottleneck visible enough to guide the next
  tiled/staged-copy implementation.
- Command: `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job run --gpu H100x8 --enable-extra-resources --cpu 16 --memory 128GB --disk 50GB --extra gpu --timeout 1800 --job-name 6597-moe-dispatch-up-perf-1 -- ... bench_moe_dispatch_up_mosaic_gpu.py --ep-size 8 --tokens-per-rank 8 --experts-per-rank 4 --top-k 4 --hidden 64 --intermediate 64 --dtype bf16 --run-pallas --bench-iters 3 --warmup-steps 1`
- Config: Single-node CoreWeave `cw-us-east-02a`, H100x8, explicit `expert`
  mesh, bf16, `H=64`, `I=64`, `T/rank=8`, top-k 4, four local experts per
  rank.
- Result:
  - Dispatch compile-including: 106.7 s.
  - Dispatch steady-state: mean 1.897 ms, min 1.776 ms, max 2.000 ms.
  - W13/SiLU compile-including: 615 ms.
  - W13/SiLU steady-state: mean 0.201 ms, min 0.188 ms, max 0.213 ms.
  - Correctness remained clean: dispatch max error 0, metadata errors 0,
    W13/SiLU max bf16 error 2.
- Interpretation: The deliberately serial validation dispatch is already the
  dominant steady-state cost on this small shape. The next performance step
  should replace scalar row writes with tiled SMEM-staged remote copies.
- Next action: Implement tiled dispatch copy path and rerun the same steady-state
  command for an apples-to-apples comparison.

### 2026-06-23 - Baseline and roofline comparison

- Hypothesis: On the validation shape, JIT reference baselines and roofline math
  will show whether Mosaic GPU is compute-limited or dominated by the current
  scalar remote-write validation structure.
- Command: `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job run --gpu H100x8 --enable-extra-resources --cpu 16 --memory 128GB --disk 50GB --extra gpu --timeout 1800 --job-name 6597-moe-dispatch-up-perf-2 -- ... bench_moe_dispatch_up_mosaic_gpu.py --ep-size 8 --tokens-per-rank 8 --experts-per-rank 4 --top-k 4 --hidden 64 --intermediate 64 --dtype bf16 --run-pallas --bench-iters 5 --warmup-steps 2`
- Config: Single-node CoreWeave `cw-us-east-02a`, H100x8, explicit `expert`
  mesh, bf16, `H=64`, `I=64`, `T/rank=8`, top-k 4, four local experts per
  rank. This is still the validation shape, not the May D2560 shape.
- Result:
  - JIT reference dispatch steady-state: mean 0.638 ms.
  - Mosaic GPU dispatch steady-state: mean 1.796 ms, or 0.355x reference.
  - JIT reference W13/SiLU steady-state: mean 0.160 ms.
  - Mosaic GPU W13/SiLU steady-state: mean 0.214 ms, or 0.751x reference.
  - Correctness remained clean: dispatch max error 0, metadata errors 0,
    W13/SiLU max bf16 error 2.
  - Roofline summary, using H100 SXM order-of-magnitude peaks: 256 routed rows,
    dispatch payload 32 KiB, dispatch payload bandwidth 0.018 GB/s, W13 work
    4.19 MFLOP, W13 measured 0.0196 TFLOP/s, estimated W13 roofline
    190.6 TFLOP/s.
- Interpretation: The current Mosaic GPU kernels are not near hardware limits.
  Dispatch is dominated by serial scalar remote writes and synchronization, not
  payload bandwidth. W13/SiLU is far below roofline because this shape is tiny
  and launch/scheduling overhead dominates. The JIT reference remains faster on
  both subphases for this validation shape.
- Next action: Do not tune block sizes yet. First replace scalar dispatch with
  tiled SMEM-staged remote copies, then rerun this exact comparison. Only after
  that should we scale toward the May D2560 target.
