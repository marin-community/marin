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
