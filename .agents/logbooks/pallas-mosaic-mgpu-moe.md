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

### 2026-06-23 - Relevant-shape W13 probe

- Hypothesis: At the requested MoE shape (`H=2560`, `I=2560`, 256 global
  experts, top-k 4), the local Mosaic GPU W13/SiLU kernel should become useful
  even if the serial validation dispatch is not yet scalable.
- Commands:
  - Integrated dispatch+W13 probe: `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job run --gpu H100x8 --enable-extra-resources --cpu 16 --memory 160GB --disk 80GB --extra gpu --timeout 3600 --job-name 6597-moe-dispatch-up-relevant-h2560-i2560-e256 -- ... bench_moe_dispatch_up_mosaic_gpu.py --ep-size 8 --tokens-per-rank 8 --experts-per-rank 32 --top-k 4 --hidden 2560 --intermediate 2560 --dtype bf16 --run-pallas --bench-iters 3 --warmup-steps 1`
  - Split W13 probe: `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job run --gpu H100x8 --enable-extra-resources --cpu 16 --memory 160GB --disk 80GB --extra gpu --timeout 1800 --job-name 6597-moe-w13-relevant-h2560-i2560-e256-roofline -- ... bench_moe_dispatch_up_mosaic_gpu.py --ep-size 8 --tokens-per-rank 8 --experts-per-rank 32 --top-k 4 --hidden 2560 --intermediate 2560 --dtype bf16 --run-pallas --pallas-w13-from-reference-layout --bench-iters 5 --warmup-steps 2 --w13-atol 256`
- Config: Single-node CoreWeave `cw-us-east-02a`, H100x8, explicit `expert`
  mesh, bf16, `H=2560`, `I=2560`, `T/rank=8`, top-k 4, 32 local experts per
  rank, 256 global experts.
- Result:
  - Integrated path reached JIT baselines but produced no Mosaic dispatch timing
    after roughly nine minutes in dispatch compile/lowering; the job was
    terminated before the 3600 s timeout.
  - Integrated JIT baselines before termination: dispatch steady-state mean
    0.784 ms; W13/SiLU steady-state mean 10.504 ms.
  - Split W13 JIT baselines: dispatch steady-state mean 0.742 ms; W13/SiLU
    steady-state mean 10.479 ms.
  - Split Mosaic GPU W13/SiLU, using the reference dispatch layout:
    compile-including 911 ms; steady-state mean 0.646 ms, min 0.640 ms,
    max 0.655 ms; 16.23x faster than the JIT W13/SiLU baseline.
  - W13/SiLU max bf16 absolute error was 128 at this accumulation size. The
    performance probe used `--w13-atol 256`.
  - Roofline summary, using the benchmark's H100 SXM assumptions: 256 routed
    rows, dispatch payload 1280 KiB, W13 work 6.71 GFLOP, W13 bytes 6.25 GiB,
    arithmetic intensity 1.0 flop/byte, measured W13 10.39 TFLOP/s, estimated
    HBM-bound W13 roofline 26.79 TFLOP/s.
- Interpretation: The W13/SiLU subkernel is now meaningfully faster than the
  JIT baseline at the requested expert/hidden/intermediate shape, reaching about
  39% of the simple HBM-bound roofline estimate. The integrated `moe_dispatch_up`
  kernel is still blocked by the serial scalar remote dispatch lowering path, so
  it cannot yet provide a full relevant-shape end-to-end timing.
- Next action: Keep W13/SiLU as the viable fast subkernel and prioritize tiled
  SMEM-staged remote dispatch. Rerun the integrated shape after dispatch no
  longer lowers as a hidden-dimension scalar loop.

### 2026-06-23 - Grug-consistent W13 correctness check

- Hypothesis: The previous relevant-shape W13/SiLU max error of 128 may be an
  artifact of using unrealistic standard-normal expert weights instead of the
  Grug MoE initialization scale.
- Command: `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job run --gpu H100x8 --enable-extra-resources --cpu 16 --memory 160GB --disk 80GB --extra gpu --timeout 1800 --job-name 6597-moe-w13-gruginit-h2560-i2560-e256-summary -- ... bench_moe_dispatch_up_mosaic_gpu.py --ep-size 8 --tokens-per-rank 8 --experts-per-rank 32 --top-k 4 --hidden 2560 --intermediate 2560 --dtype bf16 --weight-init grug_truncated --run-pallas --pallas-w13-from-reference-layout --bench-iters 5 --warmup-steps 2`
- Config: Single-node CoreWeave `cw-us-east-02a`, H100x8, explicit `expert`
  mesh, bf16, `H=2560`, `I=2560`, `T/rank=8`, top-k 4, 32 local experts per
  rank, 256 global experts. Expert weights used the Grug MoE distribution:
  truncated normal in `[-3, 3] * (0.5 / sqrt(2560))`.
- Result:
  - JIT W13/SiLU steady-state mean 10.524 ms.
  - Mosaic GPU W13/SiLU steady-state mean 0.640 ms, min 0.637 ms, max 0.648 ms,
    or 16.44x faster than the JIT W13/SiLU baseline.
  - Error summary: max abs 0.015625, mean abs 2.28e-05, RMS abs 1.77e-04,
    max relative 0.0141, mean relative 2.27e-05. The max-error element had
    expected abs 2.015625 and actual abs 2.0.
  - Roofline summary, using the benchmark's H100 SXM assumptions: measured W13
    10.48 TFLOP/s, estimated HBM-bound W13 roofline 26.79 TFLOP/s.
- Interpretation: The prior max abs error of 128 was caused by the deliberately
  unrealistic standard-normal weight scale. With Grug-consistent weight scale,
  the W13/SiLU error is at bf16 rounding scale for this probe. The subkernel
  still needs broader numerical sweeps before production use, but this run does
  not indicate a routing or matmul correctness bug.
- Next action: Keep Grug-style random inputs as the default comparison for
  relevant-shape correctness claims, and continue prioritizing tiled dispatch.

### 2026-06-24 - XLA baseline flag check

- Hypothesis: Enabling Triton GEMM and latency-hiding scheduler flags, and then
  lowering JAX optimization level to O1, may speed up the JIT W13/SiLU baseline
  enough to narrow the Mosaic GPU advantage.
- Commands:
  - Invalid first attempt: `XLA_FLAGS="-O1 --xla_gpu_triton_gemm_any=True --xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_cuda_data_dir=..." ...`
  - Valid subset: `XLA_FLAGS="--xla_gpu_triton_gemm_any=True --xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_cuda_data_dir=..." ...`
  - Corrected O1: `JAX_OPTIMIZATION_LEVEL=O1 XLA_FLAGS="--xla_gpu_triton_gemm_any=True --xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_cuda_data_dir=..." ...`
- Config: Same split W13/SiLU probe as the Grug-consistent relevant-shape check:
  single-node CoreWeave `cw-us-east-02a`, H100x8, bf16, `H=2560`, `I=2560`,
  `T/rank=8`, top-k 4, 32 local experts per rank, 256 global experts,
  `--weight-init grug_truncated`, reference dispatch layout.
- Result:
  - `-O1` inside `XLA_FLAGS` failed at startup: `Unknown flag in XLA_FLAGS: -O1`.
  - Valid subset run: JIT W13/SiLU steady-state mean 10.526 ms; Mosaic GPU
    W13/SiLU steady-state mean 0.647 ms; speedup 16.27x.
  - Corrected O1 run: JIT W13/SiLU steady-state mean 10.492 ms; Mosaic GPU
    W13/SiLU steady-state mean 0.637 ms; speedup 16.47x.
  - Correctness stayed unchanged: max abs 0.015625, mean abs 2.28e-05,
    RMS abs 1.77e-04, max relative 0.0141.
  - Corrected O1 roofline summary: measured W13 10.54 TFLOP/s, estimated
    HBM-bound W13 roofline 26.79 TFLOP/s.
- Interpretation: These flags do not materially improve the JIT W13/SiLU
  baseline for this split relevant-shape probe. The corrected O1 run improved
  the baseline by roughly 0.3%, which is within run-to-run noise for this
  benchmark. The Mosaic GPU W13/SiLU timing is also essentially unchanged.
- Next action: Stop pursuing these XLA flags for this microbench. For baseline
  pressure, compare against Grug's production MoE implementation directly; for
  Mosaic work, continue with tiled dispatch.

### 2026-06-24 - Vectorized remote dispatch validation

- Hypothesis: The relevant-shape integrated dispatch hang is caused by scalar
  hidden-dimension remote writes in the validation dispatch kernel, not by a
  semantic routing bug. Replacing the inner scalar loop with a row-slice remote
  copy should compile at `H=2560` and preserve dispatch correctness.
- Commands:
  - H128 row-vector probe: `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job run --gpu H100x8 --enable-extra-resources --cpu 16 --memory 128GB --disk 50GB --extra gpu --timeout 1800 --job-name dlwh-6597-moe-dispatch-row-vector-h128 -- ... bench_moe_dispatch_up_mosaic_gpu.py --ep-size 8 --tokens-per-rank 8 --experts-per-rank 4 --top-k 4 --hidden 128 --intermediate 64 --dtype bf16 --run-pallas --dispatch-copy-mode row_vector --bench-iters 2 --warmup-steps 1`
  - H128 SMEM probe: `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job run --gpu H100x8 --enable-extra-resources --cpu 16 --memory 128GB --disk 50GB --extra gpu --timeout 1800 --job-name dlwh-6597-moe-dispatch-row-smem-h128 -- ... bench_moe_dispatch_up_mosaic_gpu.py --ep-size 8 --tokens-per-rank 8 --experts-per-rank 4 --top-k 4 --hidden 128 --intermediate 64 --dtype bf16 --run-pallas --dispatch-copy-mode row_smem --bench-iters 2 --warmup-steps 1`
  - Relevant-shape row-vector probe: `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job run --gpu H100x8 --enable-extra-resources --cpu 16 --memory 128GB --disk 50GB --extra gpu --timeout 2400 --job-name dlwh-6597-moe-dispatch-row-vector-h2560-grug -- ... bench_moe_dispatch_up_mosaic_gpu.py --ep-size 8 --tokens-per-rank 8 --experts-per-rank 32 --top-k 4 --hidden 2560 --intermediate 2560 --dtype bf16 --weight-init grug_truncated --run-pallas --dispatch-copy-mode row_vector --bench-iters 3 --warmup-steps 1`
- Result:
  - H128 row-vector dispatch compiled and matched the reference exactly:
    dispatch max error 0, metadata errors 0. Steady-state dispatch was
    1.753 ms versus 0.739 ms for the JIT reference.
  - H128 SMEM copy failed in ptxas with `Entry function ... uses too much
    parameter space (0x8080 bytes, 0x7ffc max)`. An earlier H64 SMEM attempt
    also failed before the peer copy because lane lowering requires vector
    element counts that are multiples of 128. This path was removed instead of
    retaining dead diagnostic code.
  - Relevant-shape row-vector dispatch compiled in 16.1 s and ran correctly:
    dispatch max error 0, metadata errors 0. Steady-state dispatch was
    1.739 ms versus 0.783 ms for the JIT reference.
  - Relevant-shape integrated W13/SiLU stayed correct under Grug weights:
    Mosaic GPU W13/SiLU steady-state mean 0.639 ms versus 10.518 ms for the
    JIT baseline, with max abs error 0.015625.
  - Relevant-shape roofline summary: 256 routed rows, dispatch payload
    1280 KiB, dispatch payload bandwidth 0.754 GB/s, W13 measured
    10.51 TFLOP/s, estimated HBM-bound W13 roofline 26.79 TFLOP/s.
- Interpretation: The dispatch correctness issue was the scalar lowering path,
  not the routing data. Row-vector remote stores make the current milestone-1
  integrated path compile and validate at the requested `H=2560`, `I=2560`,
  256-expert, top-k 4 shape. It is still slower than the JIT reference dispatch
  because each routed row is copied serially and synchronized coarsely. The JAX
  `collective_matmul_mgpu.py` pattern points to the next production design:
  structured symmetric scratch/ring exchange or multimem-style collectives,
  then local compaction/W13, rather than per-row direct remote writes into the
  final receive buffer.
- Next action: Use `row_vector` as the default validation dispatch path for the
  milestone branch. For the perf phase, prototype a scratch-mediated dispatch
  modeled after collective matmul's remote-ref exchange.

### 2026-06-24 - Symmetric scratch dispatch prototype

- Hypothesis: A destination-owned symmetric scratch exchange, following the
  remote-ref pattern in JAX's `collective_matmul_mgpu.py`, should reduce the
  direct final-buffer remote-write bottleneck and provide the right structure
  for eventual dispatch/W13 overlap. The communication remains sparse
  all-to-all: each source writes only rows routed to a destination rank.
- Commands:
  - Serial scratch H128 probe: `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job run --gpu H100x8 --enable-extra-resources --cpu 16 --memory 128GB --disk 50GB --extra gpu --timeout 1800 --job-name dlwh-6597-moe-dispatch-scratch-h128 -- ... bench_moe_dispatch_up_mosaic_gpu.py --ep-size 8 --tokens-per-rank 8 --experts-per-rank 4 --top-k 4 --hidden 128 --intermediate 64 --dtype bf16 --run-pallas --dispatch-copy-mode scratch --bench-iters 2 --warmup-steps 1`
  - Serial scratch-count H128/relevant probes: same command shape, with per-row
    valid scratch metadata replaced by one per-source scratch count.
  - Parallel scratch H128 probe: `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job run --gpu H100x8 --enable-extra-resources --cpu 16 --memory 128GB --disk 50GB --extra gpu --timeout 1800 --job-name dlwh-6597-moe-dispatch-scratch-parallel-h128 -- ... bench_moe_dispatch_up_mosaic_gpu.py --ep-size 8 --tokens-per-rank 8 --experts-per-rank 4 --top-k 4 --hidden 128 --intermediate 64 --dtype bf16 --run-pallas --dispatch-copy-mode scratch_parallel --bench-iters 2 --warmup-steps 1`
  - Parallel scratch relevant probe: `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job run --gpu H100x8 --enable-extra-resources --cpu 16 --memory 128GB --disk 50GB --extra gpu --timeout 2400 --job-name dlwh-6597-moe-dispatch-scratch-parallel-h2560-grug -- ... bench_moe_dispatch_up_mosaic_gpu.py --ep-size 8 --tokens-per-rank 8 --experts-per-rank 32 --top-k 4 --hidden 2560 --intermediate 2560 --dtype bf16 --weight-init grug_truncated --run-pallas --dispatch-copy-mode scratch_parallel --bench-iters 3 --warmup-steps 1`
- Config: Single-node CoreWeave `cw-us-east-02a`, H100x8, explicit `expert`
  mesh. H128 probes used bf16 standard-normal weights and are dispatch
  correctness/perf probes; the relevant probe used Grug-truncated weights at
  `H=2560`, `I=2560`, `T/rank=8`, top-k 4, 32 local experts per rank, 256
  global experts.
- Result:
  - Serial scratch H128 compiled and matched dispatch exactly, but dispatch
    steady-state was 1.793 ms versus 0.723 ms for the JIT reference. The later
    W13 failure was the known standard-normal max-error tolerance issue.
  - Replacing per-row scratch-valid flags with one per-source scratch count did
    not improve H128 dispatch: 1.795 ms versus 0.683 ms for the JIT reference.
  - Count-based serial scratch at the relevant Grug shape matched dispatch
    exactly and preserved W13 parity, but dispatch was 1.800 ms versus 0.747 ms
    for the JIT reference.
  - Parallel scratch, using one Pallas program per `(peer_rank, send_row)`,
    improved H128 dispatch to 1.539 ms versus 0.721 ms for the JIT reference.
  - Parallel scratch at the relevant Grug shape matched dispatch exactly:
    dispatch max error 0, metadata errors 0. Steady-state dispatch was
    1.512 ms versus 0.745 ms for the JIT reference.
  - After renaming the best variant to the checked-in `scratch` mode, a final
    H128 Grug smoke passed with dispatch max error 0, metadata errors 0, and
    W13 max abs error 0.0078125.
  - Relevant integrated W13/SiLU remained correct: steady-state mean 0.657 ms
    versus 10.478 ms for the JIT baseline, max abs error 0.015625.
  - Relevant roofline summary for the parallel scratch run: dispatch payload
    1280 KiB, dispatch payload bandwidth 0.867 GB/s, W13 measured
    10.21 TFLOP/s, estimated HBM-bound W13 roofline 26.79 TFLOP/s.
- Interpretation: The scratch implementation now follows the symmetric
  remote-ref exchange shape and validates the sparse all-to-all semantics, but
  it is still not close enough to the JIT dispatch baseline to justify overlap.
  The best current dispatch is 1.512 ms versus 0.745 ms, roughly 2.0x slower and
  outside the requested within-30% threshold. Parallelizing per row helped
  compile time and steady-state time, but the design still pays for separate
  scratch writes, local compaction, and a full fan-in semaphore before W13.
- Next action: Do not implement overlap yet. Keep the parallel scratch path as
  the default `scratch` validation mode and use it as the structural base for a
  later overlapped prototype only after dispatch is improved, likely by
  coarsening row programs into larger transfer tiles or fusing compaction with
  W13 consumption.

### 2026-06-24 - Large-token grouped W13 baseline

- Hypothesis: At the more relevant per-rank token count
  `T/rank=4096 * 8 = 32768`, the W13/SiLU grouped matmul should be compute
  intensive, not HBM-bound. Built-in Haliax `ragged_dot(auto)` should provide a
  stronger target than `ragged_dot_general` XLA lowering.
- Commands:
  - XLA grouped matmul baseline: `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job run --gpu H100x8 --enable-extra-resources --cpu 16 --memory 160GB --disk 80GB --extra gpu --timeout 3600 --job-name dlwh-6597-moe-bigshape-ragged-dot-xla -- ... bench_moe_dispatch_up_mosaic_gpu.py --synthetic-layout --ep-size 8 --tokens-per-rank 32768 --recv-capacity 131072 --experts-per-rank 32 --top-k 4 --hidden 2560 --intermediate 2560 --dtype bf16 --weight-init grug_truncated --w13-impl ragged_dot --ragged-dot-impl xla --bench-iters 3 --warmup-steps 1`
  - Triton/auto grouped matmul baseline: `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job run --gpu H100x8 --enable-extra-resources --cpu 16 --memory 160GB --disk 80GB --extra gpu --timeout 3600 --job-name dlwh-6597-moe-bigshape-ragged-dot-auto -- ... bench_moe_dispatch_up_mosaic_gpu.py --synthetic-layout --ep-size 8 --tokens-per-rank 32768 --recv-capacity-factor 1.0 --experts-per-rank 32 --top-k 4 --hidden 2560 --intermediate 2560 --dtype bf16 --weight-init grug_truncated --w13-impl ragged_dot --ragged-dot-impl auto --bench-iters 3 --warmup-steps 1`
- Config: Single-node CoreWeave `cw-us-east-02a`, H100x8, synthetic
  expert-major layout, bf16, `H=2560`, `I=2560`, top-k 4, 32 local experts per
  rank, 256 global experts, Grug-truncated expert weights. Capacity was sized
  to 0% buffer for the balanced synthetic layout:
  `recv_capacity=32768 * 4 = 131072`, so each local expert received 4096 rows.
- Result:
  - XLA `ragged_dot_general` W13/SiLU steady-state mean was 139.973 ms,
    measuring 196.38 TFLOP/s.
  - Haliax `ragged_dot(auto)`, which selects the Pallas Triton grouped matmul
    on GPU, compiled in 370.7 ms and reached steady-state mean 7.321 ms,
    measuring 3754.64 TFLOP/s.
  - Large-shape roofline summary: 1,048,576 routed rows globally; W13 work
    27.49 PFLOP; estimated W13 bytes 17.45 GB; arithmetic intensity
    1575 flop/byte including activations and weights; simple 8xH100 BF16
    peak estimate 7912 TFLOP/s.
- Interpretation: The earlier `~1 flop/byte` estimate was only true for the
  tiny `T/rank=8` probe. At the large token count, W13/SiLU is compute
  intensive and the built-in Triton grouped matmul is the relevant baseline.
  The measured 7.321 ms is about 47% of the simple BF16 peak estimate; the
  W13-only compute lower bound is roughly 3.5 ms, so a `<2 ms` end-to-end
  target for this full W13 shape is below roofline.
- Next action: Treat `ragged_dot(auto)` as the grouped W13 baseline and focus
  Mosaic work on dispatch/overlap only where it can improve end-to-end time.
  Benchmark capacity should be expressed as a per-rank routed-token factor:
  1.0 for no buffer and roughly 1.1-1.25 for normal imbalance.

### 2026-06-24 - Source/expert readiness metadata

- Hypothesis: The overlapped dispatch/W13 kernel needs explicit
  `(src_rank, local_expert)` row ranges before Mosaic semaphores can signal
  readiness at the granularity described in the spec. Encoding those ranges in
  the prepacked reference representation should make the later Pallas scheduler
  less error-prone.
- Commands:
  - `uv run --package marin-levanter --group test pytest lib/levanter/tests/kernels/test_pallas_moe_dispatch_up.py -q`
  - `uv run --package marin-levanter --group test python lib/levanter/scripts/bench/bench_moe_dispatch_up_mosaic_gpu.py --synthetic-layout --ep-size 1 --tokens-per-rank 8 --experts-per-rank 2 --top-k 2 --hidden 8 --intermediate 8 --dtype fp32 --weight-init grug_truncated --w13-impl ragged_dot --ragged-dot-impl xla --bench-iters 1 --warmup-steps 1`
  - `uv run --package marin-levanter --group test python lib/levanter/scripts/bench/bench_moe_dispatch_up_mosaic_gpu.py --ep-size 1 --tokens-per-rank 8 --experts-per-rank 2 --top-k 2 --hidden 8 --intermediate 8 --dtype fp32 --weight-init grug_truncated --w13-impl ragged_dot --ragged-dot-impl xla --bench-iters 1 --warmup-steps 1`
- Config: CPU smoke shapes only. The new metadata records source-side send
  ranges `[src_rank, dst_rank, local_expert]` and destination receive ranges
  `[dst_rank, src_rank, local_expert]`.
- Result: Kernel tests passed (`11 passed`). Synthetic and routed benchmark
  smokes completed successfully. The hand-authored routing test now pins exact
  send and receive range matrices for a two-rank, two-local-expert example.
- Interpretation: This does not implement overlap yet, but it establishes the
  static range contract required by the spec's `ready[src_rank, local_expert]`
  design. The next Mosaic step is to use these ranges in a scratch/ready
  dispatch variant whose producer signals per source/expert range instead of a
  single full-buffer fan-in.
- Next action: Add the Mosaic source/expert ready dispatch variant, validate it
  against the existing layout reference, then attach W13 scheduling to those
  ready ranges.

### 2026-06-24 - Scratch-ready Pallas dispatch

- Hypothesis: A Pallas scratch dispatch variant can expose
  `ready_count[src_rank, local_expert]` matching the clipped source/expert
  receive ranges. This is the producer-side contract needed before W13 can wait
  on source/expert readiness instead of full-buffer completion.
- Commands:
  - H128 smoke: `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job run --gpu H100x8 --enable-extra-resources --cpu 16 --memory 128GB --disk 50GB --extra gpu --timeout 1800 --job-name dlwh-6597-moe-dispatch-scratch-ready-h128 -- ... bench_moe_dispatch_up_mosaic_gpu.py --ep-size 8 --tokens-per-rank 8 --experts-per-rank 4 --top-k 4 --hidden 128 --intermediate 64 --dtype bf16 --weight-init grug_truncated --run-pallas --dispatch-copy-mode scratch_ready --bench-iters 1 --warmup-steps 1`
  - H2560 relevant-shape smoke: `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job run --gpu H100x8 --enable-extra-resources --cpu 16 --memory 160GB --disk 80GB --extra gpu --timeout 2400 --job-name dlwh-6597-moe-dispatch-scratch-ready-h2560-grug -- ... bench_moe_dispatch_up_mosaic_gpu.py --ep-size 8 --tokens-per-rank 8 --experts-per-rank 32 --top-k 4 --hidden 2560 --intermediate 2560 --dtype bf16 --weight-init grug_truncated --run-pallas --dispatch-copy-mode scratch_ready --bench-iters 2 --warmup-steps 1`
- Config: Single-node CoreWeave `cw-us-east-02a`, H100x8, explicit `expert`
  mesh. Both runs use Grug-truncated weights and 1.25 capacity factor
  (`recv_capacity=40` for `T/rank=8`, top-k 4).
- Result:
  - H128: dispatch max error 0, metadata errors 0, ready-count max error 0.
    Dispatch steady-state mean 1.689 ms versus 0.689 ms reference JIT.
    W13/SiLU max abs error 0.0078125.
  - H2560: dispatch max error 0, metadata errors 0, ready-count max error 0.
    Dispatch steady-state mean 1.685 ms versus 0.782 ms reference JIT.
    W13/SiLU steady-state mean 0.667 ms versus 8.528 ms reference JIT.
    W13/SiLU max abs error 0.015625.
- Interpretation: The producer side can now materialize exact source/expert
  readiness counts on the Mosaic path. This still uses a full-kernel barrier
  before marking ranges ready; it is not the final overlap design. The next
  step is to replace the whole-buffer barrier with range/block-local
  coordination, then schedule W13 over ready ranges.
- Next action: Add a W13 consumer mode that consumes source/expert ranges from
  `ready_count` and validate it against the current whole-buffer W13 result
  before attempting to remove the full barrier.

### 2026-06-24 - Ready-range W13 consumer

- Hypothesis: A W13/SiLU Mosaic kernel scheduled over flattened
  source/expert ready ranges can consume `ready_count` directly and preserve
  correctness. This tests the consumer-side scheduling contract before removing
  the full producer barrier.
- Commands:
  - H128 smoke: `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job run --gpu H100x8 --enable-extra-resources --cpu 16 --memory 128GB --disk 50GB --extra gpu --timeout 1800 --job-name dlwh-6597-moe-ready-w13-h128 -- ... bench_moe_dispatch_up_mosaic_gpu.py --ep-size 8 --tokens-per-rank 8 --experts-per-rank 4 --top-k 4 --hidden 128 --intermediate 64 --dtype bf16 --weight-init grug_truncated --run-pallas --dispatch-copy-mode scratch_ready --w13-impl mosaic_gpu_ready --bench-iters 1 --warmup-steps 1`
  - H2560 relevant-shape smoke: `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job run --gpu H100x8 --enable-extra-resources --cpu 16 --memory 160GB --disk 80GB --extra gpu --timeout 2400 --job-name dlwh-6597-moe-ready-w13-h2560-grug -- ... bench_moe_dispatch_up_mosaic_gpu.py --ep-size 8 --tokens-per-rank 8 --experts-per-rank 32 --top-k 4 --hidden 2560 --intermediate 2560 --dtype bf16 --weight-init grug_truncated --run-pallas --dispatch-copy-mode scratch_ready --w13-impl mosaic_gpu_ready --bench-iters 2 --warmup-steps 1`
- Config: Single-node CoreWeave `cw-us-east-02a`, H100x8, explicit `expert`
  mesh, Grug-truncated weights, top-k 4, `T/rank=8`, capacity factor 1.25.
- Result:
  - H128: dispatch max error 0, metadata errors 0, ready-count max error 0.
    Ready-range W13/SiLU max abs error 0.0078125; steady-state mean
    0.231 ms versus 0.174 ms reference JIT.
  - H2560: dispatch max error 0, metadata errors 0, ready-count max error 0.
    Ready-range W13/SiLU max abs error 0.015625; steady-state mean
    5.370 ms versus 8.506 ms reference JIT.
- Interpretation: The consumer-side ready-count contract is correct, but the
  naive source/expert range schedule fragments W13 badly on the small-token
  relevant-shape probe. The earlier expert-major W13 kernel ran about 0.667 ms
  on the same shape; splitting into 256 source/expert groups costs most of the
  TensorCore efficiency. The final overlap path should not simply run one
  independent W13 schedule per source/expert range. It needs either coarser
  block readiness, expert-local coalescing of ready ranges, or enough
  communication/compute pipeline overlap at the large-token shape to offset the
  scheduling fragmentation.
- Next action: For the large-token shape, prototype block-level readiness with
  expert-major coalescing, or keep W13 as the built-in grouped matmul and focus
  on overlapping dispatch with the non-fragmented expert-major consumer.

### 2026-06-24 - Coalesced block-ready dispatch metadata

- Hypothesis: A coalesced `ready_block_count[recv_block]` surface can represent
  row/block readiness without fragmenting W13 by source rank. This keeps the
  readiness contract aligned with expert-major W13 block scheduling.
- Commands:
  - Failed first H128 attempt: `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job run --gpu H100x8 --enable-extra-resources --cpu 16 --memory 128GB --disk 50GB --extra gpu --timeout 1800 --job-name dlwh-6597-moe-block-ready-h128 -- ... bench_moe_dispatch_up_mosaic_gpu.py --ep-size 8 --tokens-per-rank 8 --experts-per-rank 4 --top-k 4 --hidden 128 --intermediate 64 --dtype bf16 --weight-init grug_truncated --run-pallas --dispatch-copy-mode scratch_ready --bench-iters 1 --warmup-steps 1`
  - Fixed H128 smoke: `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job run --gpu H100x8 --enable-extra-resources --cpu 16 --memory 128GB --disk 50GB --extra gpu --timeout 1800 --job-name dlwh-6597-moe-block-ready-h128-2 -- ... bench_moe_dispatch_up_mosaic_gpu.py --ep-size 8 --tokens-per-rank 8 --experts-per-rank 4 --top-k 4 --hidden 128 --intermediate 64 --dtype bf16 --weight-init grug_truncated --run-pallas --dispatch-copy-mode scratch_ready --bench-iters 1 --warmup-steps 1`
  - H2560 relevant-shape smoke: `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job run --gpu H100x8 --enable-extra-resources --cpu 16 --memory 160GB --disk 80GB --extra gpu --timeout 2400 --job-name dlwh-6597-moe-block-ready-h2560-grug -- ... bench_moe_dispatch_up_mosaic_gpu.py --ep-size 8 --tokens-per-rank 8 --experts-per-rank 32 --top-k 4 --hidden 2560 --intermediate 2560 --dtype bf16 --weight-init grug_truncated --run-pallas --dispatch-copy-mode scratch_ready --bench-iters 2 --warmup-steps 1`
- Config: Single-node CoreWeave `cw-us-east-02a`, H100x8, explicit `expert`
  mesh, Grug-truncated weights, top-k 4, `T/rank=8`, capacity factor 1.25.
  Block-ready counts use the W13 `block_m` tile height.
- Result:
  - The first H128 attempt failed during Pallas lowering because the kernel
    captured non-scalar `rows_per_expert` constants while computing
    block-ready counts. Passing the total ready row count as an explicit scalar
    input fixed the issue.
  - Fixed H128: dispatch max error 0, metadata errors 0, source/expert
    ready-count max error 0, block-ready-count max error 0. W13 max abs error
    0.0078125.
  - H2560: dispatch max error 0, metadata errors 0, source/expert ready-count
    max error 0, block-ready-count max error 0. Dispatch steady-state mean
    1.713 ms, W13 steady-state mean 0.613 ms, W13 max abs error 0.015625.
- Interpretation: The Mosaic dispatch path now emits both fine-grained
  source/expert readiness and coalesced receive-block readiness. The latter
  preserves the expert-major W13 schedule and avoids the severe fragmentation
  observed in the source/expert ready-range W13 consumer. The path still marks
  blocks after a full dispatch barrier; the next step is to move block-ready
  signaling earlier, at row/block copy completion, and have W13 wait on those
  block signals.
- Next action: Replace the full-kernel ready marking with a row/block
  coordinator that signals block readiness as soon as each receive block is
  complete, then run W13 from the block-ready schedule.

### 2026-06-24 - Block-ready W13 consumer

- Hypothesis: A W13/SiLU consumer that still schedules expert-major blocks but
  masks stores with `ready_block_count[recv_block]` should preserve the fast
  non-fragmented Mosaic W13 schedule while testing the consumer-side
  block-readiness contract.
- Commands:
  - Local validation:
    `uv run --package marin-levanter --group test pytest lib/levanter/tests/kernels/test_pallas_moe_dispatch_up.py -q`
  - H128 smoke: `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job run --gpu H100x8 --enable-extra-resources --cpu 16 --memory 160GB --disk 80GB --extra gpu --timeout 2400 --job-name dlwh-6597-moe-block-ready-w13-h128 -- ... bench_moe_dispatch_up_mosaic_gpu.py --ep-size 8 --tokens-per-rank 8 --experts-per-rank 4 --top-k 4 --hidden 128 --intermediate 64 --dtype bf16 --weight-init grug_truncated --run-pallas --dispatch-copy-mode scratch_ready --w13-impl mosaic_gpu_block_ready --bench-iters 1 --warmup-steps 1`
  - H2560 relevant-shape smoke: `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job run --gpu H100x8 --enable-extra-resources --cpu 16 --memory 160GB --disk 80GB --extra gpu --timeout 2400 --job-name dlwh-6597-moe-block-ready-w13-h2560-grug -- ... bench_moe_dispatch_up_mosaic_gpu.py --ep-size 8 --tokens-per-rank 8 --experts-per-rank 32 --top-k 4 --hidden 2560 --intermediate 2560 --dtype bf16 --weight-init grug_truncated --run-pallas --dispatch-copy-mode scratch_ready --w13-impl mosaic_gpu_block_ready --bench-iters 2 --warmup-steps 1`
- Config: Single-node CoreWeave `cw-us-east-02a`, H100x8, explicit `expert`
  mesh, Grug-truncated weights, top-k 4, `T/rank=8`, capacity factor 1.25.
  This matches the normal 10-25% imbalance buffer; use
  `--recv-capacity-factor 1.0` when probing the no-buffer case.
- Result:
  - Local tests passed (`11 passed`), and `./infra/pre-commit.py
    --changed-files --fix` passed.
  - H128: dispatch max error 0, metadata errors 0, source/expert ready-count
    max error 0, block-ready-count max error 0. Block-ready W13/SiLU
    steady-state mean 0.210 ms versus 0.169 ms reference JIT, with max abs
    error 0.0078125.
  - H2560: dispatch max error 0, metadata errors 0, source/expert ready-count
    max error 0, block-ready-count max error 0. Dispatch steady-state mean
    1.680 ms versus 0.772 ms reference JIT. Block-ready W13/SiLU
    steady-state mean 0.610 ms versus 8.514 ms reference JIT, with max abs
    error 0.015625.
  - H2560 roofline summary for this small-token probe: 256 routed rows,
    dispatch payload 1280 KiB, W13 work 6710.886 MFLOP, arithmetic intensity
    1.000 flop/byte, measured W13 11.00 TFLOP/s, estimated HBM-bound W13
    roofline 26.79 TFLOP/s.
- Interpretation: The block-ready consumer fixes the fragmentation regression
  from source/expert range scheduling and returns W13 to the expected
  expert-major Mosaic speed on the relevant H2560 shape. This is still not the
  overlapped kernel: readiness is produced after the full dispatch barrier. The
  result does say that overlap should be built around coalesced block readiness
  and a non-fragmented W13 consumer. For production, the built-in grouped
  matmul path remains the better consumer target unless we need custom
  synchronization inside the W13 kernel itself.
- Next action: Implement the actual overlapped path by signaling block
  readiness at receive-block completion and feeding a non-fragmented W13
  consumer. Keep benchmark capacity factors at 1.0 for no-buffer probes and
  1.1-1.25 for realistic imbalance.

### 2026-06-24 - Candidate-only built-in GMM probes

- Hypothesis: The large-token path should use the built-in grouped matmul
  (`ragged_dot(auto)`) for W13 and focus Mosaic work on dispatch/overlap. The
  benchmark needs candidate-only execution and realistic per-peer send capacity
  before it can probe those shapes without pure-reference overhead or 8x
  send-buffer over-allocation.
- Changes:
  - Added `--skip-reference-checks` for candidate-only perf runs after
    correctness has been established on smaller shapes.
  - Added `--send-capacity` / `--send-capacity-factor`, where the factor is
    applied to balanced `T*K/EP` source/destination traffic. The default remains
    conservative `T*K` for skewed smoke tests.
  - Added `--dispatch-rows-per-program` for `scratch_ready`, so one Pallas
    program can copy a small block of send rows instead of exactly one row.
- Commands:
  - Local validation: `uv run --package marin-levanter python -m py_compile lib/levanter/src/levanter/kernels/pallas/moe_dispatch_up/mosaic_gpu.py lib/levanter/scripts/bench/bench_moe_dispatch_up_mosaic_gpu.py`
  - Local smoke: `uv run --package marin-levanter --group test python lib/levanter/scripts/bench/bench_moe_dispatch_up_mosaic_gpu.py --ep-size 1 --tokens-per-rank 32 --experts-per-rank 2 --top-k 2 --hidden 8 --intermediate 8 --dtype fp32 --send-capacity-factor 1.0 --skip-reference-checks`
  - H128 row-blocked dispatch smoke: `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job run --gpu H100x8 --enable-extra-resources --cpu 16 --memory 128GB --disk 50GB --extra gpu --timeout 1800 --job-name dlwh-6597-moe-scratch-ready-rows16-h128 -- ... bench_moe_dispatch_up_mosaic_gpu.py --ep-size 8 --tokens-per-rank 8 --experts-per-rank 4 --top-k 4 --hidden 128 --intermediate 64 --dtype bf16 --weight-init grug_truncated --run-pallas --dispatch-copy-mode scratch_ready --dispatch-rows-per-program 16 --w13-impl mosaic_gpu_block_ready --bench-iters 1 --warmup-steps 1`
  - Stopped medium row-program candidate: `... --job-name dlwh-6597-moe-ragged-dot-t4096-candidate -- ... --tokens-per-rank 4096 --experts-per-rank 32 --hidden 2560 --intermediate 2560 --recv-capacity-factor 1.25 --send-capacity-factor 1.25 --dispatch-copy-mode scratch_ready --w13-impl ragged_dot --ragged-dot-impl auto --skip-reference-checks`
  - Stopped medium row-blocked candidate: same shape with
    `--dispatch-rows-per-program 16`, job
    `dlwh-6597-moe-ragged-dot-t4096-rows16`.
- Result:
  - Local compile/smoke and `./infra/pre-commit.py --changed-files --fix`
    passed.
  - H128 rows16: dispatch max error 0, metadata errors 0, source/expert
    ready-count max error 0, block-ready-count max error 0. Dispatch
    steady-state mean 1.637 ms. Block-ready W13/SiLU steady-state mean
    0.183 ms, max abs error 0.0078125.
  - T/rank=4096 candidate setup used `recv_capacity=20480` and
    `send_capacity=2560` from 25% buffers. Prepack completed in about 22 s.
    Both row-program and rows16 scratch dispatch attempts produced no dispatch
    timing after several minutes and were stopped to avoid burning H100 time.
- Interpretation: Using built-in GMM for W13 is still the right direction, but
  the current scratch dispatch transport is the large-shape blocker. Batching
  rows per Pallas program preserves correctness but does not fix the fundamental
  cost of copying every 2560-wide row through remote scratch and then compacting
  it. The next overlap prototype should avoid this row-wise remote scratch
  transport, likely by writing/coalescing receive blocks directly and signaling
  block readiness, or by using a collective-style primitive closer to the JAX
  all-gather matmul pattern.
- Next action: Replace the scratch row-copy transport with block/coalesced
  receive writes before rerunning T/rank=4096 or the full T/rank=32768 shape.

### 2026-06-24 - Direct-ready dispatch transport probe

- Hypothesis: The stalled T/rank=4096 probes may be dominated by the scratch
  transport's second local compaction pass. A direct-ready transport that writes
  routed rows straight into final receive buffers, then emits the same
  source/expert and block readiness metadata, should isolate whether scratch
  compaction or row-wise remote writes are the large-shape blocker.
- Changes:
  - Added `direct_ready` dispatch mode. It uses the same dynamic row-program
    structure as `scratch_ready`, but writes remote rows directly into
    destination `recv_x` and metadata buffers.
  - Added an explicit zero barrier before remote writes so destination-local
    output initialization cannot race with incoming writes.
- Commands:
  - Local validation: `uv run --package marin-levanter python -m py_compile lib/levanter/src/levanter/kernels/pallas/moe_dispatch_up/mosaic_gpu.py lib/levanter/scripts/bench/bench_moe_dispatch_up_mosaic_gpu.py`
  - Local checks: `./infra/pre-commit.py --changed-files --fix`
  - H128 smoke: `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job run --gpu H100x8 --enable-extra-resources --cpu 16 --memory 128GB --disk 50GB --extra gpu --timeout 1800 --job-name dlwh-6597-moe-direct-ready-h128 -- ... bench_moe_dispatch_up_mosaic_gpu.py --ep-size 8 --tokens-per-rank 8 --experts-per-rank 4 --top-k 4 --hidden 128 --intermediate 64 --dtype bf16 --weight-init grug_truncated --run-pallas --dispatch-copy-mode direct_ready --dispatch-rows-per-program 16 --w13-impl mosaic_gpu_block_ready --bench-iters 1 --warmup-steps 1`
  - Required H2560 validation: `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job run --gpu H100x8 --enable-extra-resources --cpu 16 --memory 160GB --disk 80GB --extra gpu --timeout 2400 --job-name dlwh-6597-moe-direct-ready-h2560 -- ... bench_moe_dispatch_up_mosaic_gpu.py --ep-size 8 --tokens-per-rank 8 --experts-per-rank 32 --top-k 4 --hidden 2560 --intermediate 2560 --dtype bf16 --weight-init grug_truncated --run-pallas --dispatch-copy-mode direct_ready --dispatch-rows-per-program 16 --w13-impl mosaic_gpu_block_ready --bench-iters 2 --warmup-steps 1`
  - Stopped medium candidate: `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job run --gpu H100x8 --enable-extra-resources --cpu 16 --memory 160GB --disk 80GB --extra gpu --timeout 2400 --job-name dlwh-6597-moe-direct-ready-t4096-ragged -- ... bench_moe_dispatch_up_mosaic_gpu.py --ep-size 8 --tokens-per-rank 4096 --experts-per-rank 32 --top-k 4 --hidden 2560 --intermediate 2560 --dtype bf16 --weight-init grug_truncated --recv-capacity-factor 1.25 --send-capacity-factor 1.25 --run-pallas --dispatch-copy-mode direct_ready --dispatch-rows-per-program 16 --w13-impl ragged_dot --ragged-dot-impl auto --skip-reference-checks --bench-iters 2 --warmup-steps 1`
- Result:
  - H128: dispatch max error 0, metadata errors 0, source/expert ready-count
    max error 0, block-ready-count max error 0. Dispatch steady-state mean
    1.628 ms. Block-ready W13/SiLU steady-state mean 0.193 ms, max abs error
    0.0078125.
  - H2560 required validation: dispatch max error 0, metadata errors 0,
    source/expert ready-count max error 0, block-ready-count max error 0.
    Dispatch steady-state mean 1.733 ms versus 0.790 ms reference JIT.
    Block-ready W13/SiLU steady-state mean 0.611 ms versus 8.522 ms reference
    JIT, max abs error 0.015625.
  - H2560 roofline summary: 256 routed rows, dispatch payload 1280 KiB,
    W13 work 6710.886 MFLOP, arithmetic intensity 1.000 flop/byte, measured W13
    10.99 TFLOP/s, estimated HBM-bound W13 roofline 26.79 TFLOP/s.
  - T/rank=4096 direct-ready candidate used `recv_capacity=20480` and
    `send_capacity=2560` from 25% buffers. Prepack completed in 21.2 s, but no
    dispatch timing appeared after several minutes; the job was stopped.
- Interpretation: Direct final-row writes preserve correctness on the required
  H2560 validation shape, but they do not fix the large-token scaling failure.
  This narrows the bottleneck from "scratch compaction" to the broader row-wise
  remote-write transport and synchronization pattern. The next implementation
  should move data in coalesced receive blocks or use a collective-style
  primitive closer to the JAX all-gather matmul example, then signal
  `ready_block_count` as blocks complete.
- Next action: Prototype block/coalesced remote movement for expert-major
  receive blocks. Do not spend more H100 time on row-wise scratch or direct
  final-row transport at T/rank=4096+ unless the transport changes.

### 2026-06-24 - Failed direct block-copy lowering

- Hypothesis: A `direct_ready` fast path that copies a contiguous
  `[rows_per_program, hidden]` slice when send rows map to contiguous
  destination rows could reduce row-wise remote write overhead without changing
  the layout contract.
- Command: `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job run --gpu H100x8 --enable-extra-resources --cpu 16 --memory 128GB --disk 50GB --extra gpu --timeout 1800 --job-name dlwh-6597-moe-direct-ready-blockcopy-h128 -- ... bench_moe_dispatch_up_mosaic_gpu.py --ep-size 8 --tokens-per-rank 8 --experts-per-rank 4 --top-k 4 --hidden 128 --intermediate 64 --dtype bf16 --weight-init grug_truncated --run-pallas --dispatch-copy-mode direct_ready --dispatch-rows-per-program 16 --w13-impl mosaic_gpu_block_ready --bench-iters 1 --warmup-steps 1`
- Result: The candidate failed during Mosaic lowering before dispatch timing:
  `ValueError: Layout inference failed to find a solution`. The failing
  fast-path code was not kept.
- Interpretation: Naively assigning dynamic remote GMEM slices for
  `[rows_per_program, hidden]` blocks is not enough; the coalesced transport
  likely needs explicit layout annotations or a structure closer to the
  existing collective matmul pipeline.
- Next action: Revisit block movement with a minimal standalone remote block
  copy kernel or an all-gather-style pipeline before reintroducing it into the
  MoE dispatch benchmark.
