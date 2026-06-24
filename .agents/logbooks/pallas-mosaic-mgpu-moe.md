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

### 2026-06-24 - Block transport TMA and built-in GMM capacity probes

- Hypothesis: A coalesced SMEM-mediated remote block transport could replace
  the row-wise MoE dispatch transport, then feed built-in grouped matmul with
  realistic 0-25% capacity buffers.
- Changes:
  - Added `bench_moe_mgpu_block_transport.py`, a standalone transport probe
    that copies packed `[source, destination, rows, hidden]` blocks through
    SMEM to remote refs and checks the all-to-all transpose on small shapes.
  - Added hidden-column tiling, scratch-output mode, and loop-tile mode to
    isolate Mosaic GPU TMA lowering constraints.
- Commands:
  - Local validation: `uv run --package marin-levanter python -m py_compile lib/levanter/scripts/bench/bench_moe_mgpu_block_transport.py`
  - Local checks: `./infra/pre-commit.py --changed-files --fix`
  - H128 direct transport: `... --job-name dlwh-6597-moe-block-transport-h128-hostcheck -- ... bench_moe_mgpu_block_transport.py --ep-size 8 --rows 32 --hidden 128 --block-rows 16 --dtype bf16 --bench-iters 2 --warmup-steps 1`
  - H128 column-tiled/flat/loop/scratch variants:
    `dlwh-6597-moe-block-transport-h128-coltile`,
    `dlwh-6597-moe-block-transport-h128-flattiles`,
    `dlwh-6597-moe-block-transport-h128-scratchout`,
    `dlwh-6597-moe-block-transport-h128-looptiles`.
  - Large packed transport failures used the relevant T/rank=4096, topk=4,
    25% send-buffer shape: `rows=2560`, `hidden=2560`, `block_cols=256`,
    jobs `dlwh-6597-moe-block-transport-t4096buf125-b64`,
    `dlwh-6597-moe-block-transport-t4096buf125-b32`,
    `dlwh-6597-moe-block-transport-t4096buf125-flat-b32c256`,
    `dlwh-6597-moe-block-transport-t4096buf125-scratch-b32c256`, and
    `dlwh-6597-moe-block-transport-t4096buf125-loop-b32c256`.
  - Built-in grouped matmul capacity probes:
    `dlwh-6597-moe-ragged-dot-synth-t4096-buf125` and
    `dlwh-6597-moe-ragged-dot-synth-t4096-buf100-110`, with
    `JAX_OPTIMIZATION_LEVEL=O1` and XLA flags
    `--xla_gpu_triton_gemm_any=True --xla_gpu_enable_latency_hiding_scheduler=true`.
- Result:
  - H128 transport is correct for all retained variants. Direct host-check:
    max error 0, steady mean 1.545 ms for 0.5 MiB. Flattened column-tiled:
    max error 0, steady mean 1.409 ms. Scratch-output: max error 0, steady
    mean 1.456 ms. Loop-tiles: max error 0, steady mean 1.619 ms.
  - Large direct block copy with `block_rows=64` exceeded SMEM:
    `smem_bytes=327688 > max_smem_bytes=232448`.
  - Large `block_rows=32`, untiled hidden copy failed because Mosaic async
    copies require each dimension to be <=256 elements: shape `(1, 32, 2560)`.
  - Large column-tiled copies failed TMA lowering when the remote peer id or
    remote TMA descriptor depended on grid ids: failures reported
    `Failed to recompute the async_copy peer id on the host` for
    `gpu.block_id z`, then `gpu.block_id y`, and loop-tiles reduced this to
    `gpu.block_id x` because the peer id was still `program_id(0)`.
  - Writing to a separate scratch output did not change the large TMA failure.
  - Built-in `ragged_dot(auto)` W13/SiLU on the relevant H=I=2560,
    E=256, topk=4, T/rank=4096 synthetic receive layout:
    - 0% buffer (`recv_capacity=16384`, 512 rows/expert): 1.145 ms steady,
      3001 TFLOP/s measured, AI 426.7 flop/byte.
    - ~10% buffer (`recv_capacity=18048`, 564 rows/expert): 1.402 ms steady,
      2700 TFLOP/s measured, AI 462.2 flop/byte.
    - 25% buffer (`recv_capacity=20480`, 640 rows/expert): 1.363 ms steady,
      3150 TFLOP/s measured, AI 512.0 flop/byte.
- Interpretation: Direct arbitrary-peer Mosaic GPU remote TMA copies are not a
  viable dispatch transport with the current structure because the peer id must
  be host-recomputable and cannot come from a grid/program id. The JAX
  all-gather matmul avoids this by sending to a fixed neighbor derived from
  `axis_index`, so an overlapped Pallas transport would need a ring/all-gather
  style schedule or multiple fixed-peer kernels. For the target shape, built-in
  grouped matmul is already around 1.1-1.4 ms across realistic 0-25% buffers;
  the path to <2 ms end-to-end depends on using an efficient collective
  transport rather than replacing NCCL with arbitrary-peer Pallas TMA.
- Next action: Keep built-in `ragged_dot` as the W13 consumer and switch the
  dispatch/transport plan to either a built-in collective all-to-all/all-gather
  route or an all-gather-style fixed-neighbor ring if overlap is still required.

### 2026-06-24 - Built-in collective dispatch-up probes

- Hypothesis: If arbitrary-peer Pallas TMA is blocked, the practical path to
  fast dispatch-up may be built-in collective transport followed by built-in
  grouped matmul. Grug's existing `ragged_all_to_all` and fixed-bucket
  `all_to_all` helpers provide good baselines for this direction.
- Changes:
  - Added `--run-ragged-a2a-dispatch-up` to
    `bench_moe_dispatch_up_mosaic_gpu.py`. It uses Grug's global-expert sort,
    `lax.ragged_all_to_all`, local expert-major permute, and
    `ragged_dot(auto)` W13/SiLU.
  - Added `--run-padded-a2a-dispatch-up`. It uses Grug's fixed-bucket
    `lax.all_to_all` dispatch helper, reports dropped assignments, and runs
    `ragged_dot(auto)` W13/SiLU.
  - Candidate-only collective runs now skip reference prepack when
    `--skip-reference-checks` is set and no Mosaic dispatch mode is requested.
- Commands:
  - Local validation: `uv run --package marin-levanter python -m py_compile lib/levanter/scripts/bench/bench_moe_dispatch_up_mosaic_gpu.py`
  - Local checks: `./infra/pre-commit.py --changed-files --fix`
  - H128 ragged correctness: `... --job-name dlwh-6597-moe-ragged-a2a-up-h128 -- ... bench_moe_dispatch_up_mosaic_gpu.py --ep-size 8 --tokens-per-rank 8 --experts-per-rank 4 --top-k 4 --hidden 128 --intermediate 64 --dtype bf16 --weight-init grug_truncated --recv-capacity-factor 1.25 --run-ragged-a2a-dispatch-up --w13-impl ragged_dot --ragged-dot-impl auto --bench-iters 1 --warmup-steps 1`
  - Required H2560 ragged correctness: `... --job-name dlwh-6597-moe-ragged-a2a-up-h2560-t8 -- ... --ep-size 8 --tokens-per-rank 8 --experts-per-rank 32 --top-k 4 --hidden 2560 --intermediate 2560 --dtype bf16 --weight-init grug_truncated --recv-capacity-factor 1.25 --run-ragged-a2a-dispatch-up --w13-impl ragged_dot --ragged-dot-impl auto --bench-iters 2 --warmup-steps 1`
  - T/rank=4096 ragged capacity sweep: `... --job-name dlwh-6597-moe-ragged-a2a-up-t4096-buf-sweep -- ... for factor in 1.0 1.1 1.25; do ... --tokens-per-rank 4096 --hidden 2560 --intermediate 2560 --recv-capacity-factor $factor --run-ragged-a2a-dispatch-up --skip-reference-checks --bench-iters 2 --warmup-steps 1; done`
  - H128 padded correctness/drop probe: `... --job-name dlwh-6597-moe-padded-a2a-up-h128 -- ... --hidden 128 --intermediate 64 --recv-capacity-factor 1.25 --run-padded-a2a-dispatch-up --bench-iters 1 --warmup-steps 1`
  - T/rank=4096 padded 0% probe: `... --job-name dlwh-6597-moe-padded-a2a-up-t4096-buf-sweep -- ... --run-padded-a2a-dispatch-up --skip-reference-checks --bench-iters 2 --warmup-steps 1`; stopped during the 10% compile after the 0% result.
- Result:
  - H128 ragged all-to-all dispatch-up: max abs error `1.90735e-06`, steady
    end-to-end `0.629 ms`.
  - Required H2560/T=8 ragged all-to-all dispatch-up: max abs error
    `0.0078125`, steady end-to-end `0.840 ms`. Reference JIT dispatch was
    `0.751 ms`; reference W13 was `8.507 ms`.
  - T/rank=4096 ragged all-to-all dispatch-up candidate-only:
    - 0% buffer (`recv_capacity=16384`): `5.999 ms`.
    - 10% buffer (`recv_capacity=18023`): `6.122 ms`.
    - 25% buffer (`recv_capacity=20480`): `6.190 ms`.
  - H128 padded all-to-all dispatch-up: steady `0.424 ms`, but max abs error
    `1.78906`; the small random shape can exceed fixed per-peer buckets and
    drop routed rows.
  - T/rank=4096 padded all-to-all 0%: dropped assignments `0`, but steady
    end-to-end `12.821 ms`. The 10% compile was stopped because the zero-drop
    0% result was already far above target and slower than ragged all-to-all.
- Interpretation: Built-in `ragged_all_to_all + ragged_dot` is a correct
  end-to-end collective baseline, but it is ~6 ms at the relevant T/rank=4096
  shape. Fixed-bucket `all_to_all` is not a shortcut to <2 ms here: it is fast
  on tiny shapes but either drops rows under small skewed capacity or is much
  slower at the large no-drop shape. Since isolated `ragged_dot` is ~1.1-1.4 ms,
  the remaining gap is dispatch transport plus permutation/sort overhead, not
  W13 math.
- Next action: Use the collective baseline as the correctness/perf comparator,
  then either profile the ragged path to split sort/collective/W13 costs or
  prototype a fixed-neighbor ring/all-gather schedule that avoids arbitrary-peer
  TMA while enabling overlap.

### 2026-06-24 - Ragged all-to-all dispatch-up stage breakdown

- Hypothesis: The ~6 ms T/rank=4096 ragged collective dispatch-up baseline is
  dominated by either the source-side sort/compact, the collective/local
  receive permute, or W13 on the real routed layout. Splitting those stages
  should tell us whether overlap can plausibly reach <2 ms or whether transport
  must be replaced.
- Changes:
  - Added `--run-ragged-a2a-breakdown` stage splits to
    `bench_moe_dispatch_up_mosaic_gpu.py`.
  - The split reports full ragged dispatch-only, W13 on the dispatched layout,
    pre-collective sort/compact/all-gather, ragged-all-to-all plus local
    permute, and W13 on the transport output.
- Commands:
  - Local validation: `uv run --package marin-levanter python -m py_compile lib/levanter/scripts/bench/bench_moe_dispatch_up_mosaic_gpu.py`
  - Local checks: `./infra/pre-commit.py --changed-files --fix`
  - H128 split correctness: `... --job-name dlwh-6597-moe-ragged-a2a-breakdown-h128 -- ... bench_moe_dispatch_up_mosaic_gpu.py --ep-size 8 --tokens-per-rank 8 --experts-per-rank 4 --top-k 4 --hidden 128 --intermediate 64 --dtype bf16 --weight-init grug_truncated --recv-capacity-factor 1.25 --run-ragged-a2a-breakdown --w13-impl ragged_dot --ragged-dot-impl auto --bench-iters 1 --warmup-steps 1`
  - H128 deep split correctness: `... --job-name dlwh-6597-moe-ragged-a2a-deep-breakdown-h128 -- ... same command after adding pre-collective/transport split`
  - T/rank=4096 split: `... --job-name dlwh-6597-moe-ragged-a2a-breakdown-t4096 -- ... for factor in 1.0 1.25; do ... --tokens-per-rank 4096 --hidden 2560 --intermediate 2560 --recv-capacity-factor $factor --run-ragged-a2a-breakdown --skip-reference-checks --bench-iters 2 --warmup-steps 1; done`
  - T/rank=4096 deep split: `... --job-name dlwh-6597-moe-ragged-a2a-deep-breakdown-t4096-buf100 -- ... --recv-capacity-factor 1.0 --run-ragged-a2a-breakdown --skip-reference-checks --bench-iters 2 --warmup-steps 1`
- Result:
  - H128 split remained correct: max abs error `1.90735e-06`.
    Full dispatch-only `0.584-0.597 ms`; W13 on dispatched layout
    `0.187-0.217 ms`. Deep split: pre-collective `0.420 ms`,
    transport/local-permute `0.438 ms`, W13 `0.157 ms`. Separate launches sum
    higher than fused end-to-end, so use large-shape splits for bottleneck
    allocation.
  - T/rank=4096 top-level split:
    - 0% buffer: dispatch-only `5.036-5.038 ms`, W13 `1.138-1.149 ms`,
      sum `6.174-6.187 ms`.
    - 25% buffer: dispatch-only `5.140 ms`, W13 `1.347 ms`, sum `6.487 ms`.
  - T/rank=4096 deep split at 0% buffer:
    - pre-collective sort/compact/all-gather: `0.796 ms`.
    - ragged-all-to-all plus local receive permute: `4.595 ms`.
    - W13 on transport layout: `1.137 ms`.
    - deep split sum: `6.527 ms` (extra launch overhead versus fused).
- Interpretation: The front sort/compact is not the blocker. The target gap is
  almost entirely the built-in ragged transport/local-permute stage. Even perfect
  overlap of W13 under current ragged dispatch would leave roughly a 5 ms
  dispatch floor, so <2 ms requires replacing that transport/local-permute
  structure. The next viable direction is a fixed-neighbor/ring/all-gather style
  schedule that avoids arbitrary-peer TMA and avoids the expensive ragged
  all-to-all/local re-permute, or a lower-level collective primitive that can
  produce the expert-major layout directly.
- Next action: Prototype a fixed-neighbor ring/all-gather dispatch layout for
  dispatch-up, using the ragged collective baseline as the correctness oracle
  and preserving built-in `ragged_dot` for W13.

### 2026-06-24 - Fixed-neighbor and ring-gather dispatch-up probes

- Hypothesis: A fixed-neighbor transport pattern might avoid the arbitrary-peer
  Mosaic GPU remote-ref lowering failure, and an all-gather-style dispatch-up
  path might be a better built-in-collective baseline than ragged all-to-all for
  the relevant Grug MoE shape.
- Changes:
  - Added `--fixed-neighbor` to `bench_moe_mgpu_block_transport.py`. It copies
    each rank's `[rows, hidden]` payload to `(rank + 1) % EP` with Pallas
    Mosaic GPU remote refs, then checks against `np.roll` when reference checks
    are enabled.
  - Added `--run-ring-gather-dispatch-up` to
    `bench_moe_dispatch_up_mosaic_gpu.py`. It mirrors Grug's `ep_ring`
    dispatch-up structure: all-gather token/routing inputs, select this rank's
    local expert assignments, run `ragged_dot(auto)` W13/SiLU, and report
    dropped assignments.
  - For dispatch-up-only timing, the ring-gather path keeps the buffered output
    capacity but passes only accepted rows to `ragged_dot`, so 10-25% capacity
    buffers do not charge padded GMM compute.
- Commands:
  - Local validation: `uv run --package marin-levanter python -m py_compile lib/levanter/scripts/bench/bench_moe_dispatch_up_mosaic_gpu.py lib/levanter/scripts/bench/bench_moe_mgpu_block_transport.py`
  - Local checks: `./infra/pre-commit.py --changed-files --fix`
  - Fixed-neighbor H128 correctness: `... --job-name dlwh-6597-moe-fixed-neighbor-h128 -- ... bench_moe_mgpu_block_transport.py --ep-size 8 --rows 32 --hidden 128 --block-rows 16 --block-cols 128 --dtype bf16 --fixed-neighbor --bench-iters 2 --warmup-steps 1`
  - Fixed-neighbor H2560 bulk probe: `... --job-name dlwh-6597-moe-fixed-neighbor-h2560-rows2560 -- ... --rows 2560 --hidden 2560 --block-rows 32 --block-cols 256 --fixed-neighbor --skip-reference-checks --bench-iters 2 --warmup-steps 1`
  - Ring-gather H128 smoke: `... --job-name dlwh-6597-moe-ring-gather-up-h128-nopad-gmm -- ... bench_moe_dispatch_up_mosaic_gpu.py --ep-size 8 --tokens-per-rank 8 --experts-per-rank 4 --top-k 4 --hidden 128 --intermediate 64 --recv-capacity-factor 1.25 --run-ring-gather-dispatch-up --skip-reference-checks --ragged-dot-impl auto --bench-iters 1 --warmup-steps 1`
  - Ring-gather H128 correctness: `... --job-name dlwh-6597-moe-ring-gather-up-h128-correctness -- ... same shape with --run-ring-gather-dispatch-up and reference checks enabled`
  - Ring-gather T/rank=4096 buffer sweep: `... --job-name dlwh-6597-moe-ring-gather-up-t4096-nopad-sweep -- ... for factor in 1.0 1.1 1.25; do ... --tokens-per-rank 4096 --experts-per-rank 32 --top-k 4 --hidden 2560 --intermediate 2560 --recv-capacity-factor $factor --run-ring-gather-dispatch-up --skip-reference-checks --ragged-dot-impl auto --bench-iters 2 --warmup-steps 1; done`
- Result:
  - Fixed-neighbor H128 was correct: max abs error `0`, steady `1.492 ms` for
    `0.062 MiB`.
  - Fixed-neighbor H2560 bulk copy was still `1.486 ms` for `100 MiB`, or
    `70.6 GB/s`. The lowering works, but the launch/synchronization floor makes
    a seven-hop ring unattractive versus built-in collectives.
  - Ring-gather H128 25% capacity smoke lowered and ran at `0.340 ms`, dropped
    `0`.
  - Ring-gather H128 with reference checks enabled matched the W13 oracle: max
    abs error `1.90735e-06`, dropped `0`, steady `0.400 ms`.
  - Ring-gather T/rank=4096, H=I=2560, E=256, topk=4, Grug-truncated weights:
    - 0% buffer (`recv_capacity=16384`): dropped `0`, steady `1.816 ms`.
    - 10% buffer (`recv_capacity=18023`): dropped `0`, steady `1.856 ms`.
    - 25% buffer (`recv_capacity=20480`): dropped `0`, steady `1.861 ms`.
- Interpretation: The fixed-neighbor Pallas transport is a useful lowering
  proof but not a viable ring transport as written. The all-gather/ring-style
  dispatch-up path using built-in collective transport and built-in
  `ragged_dot` reaches the <2 ms target across the realistic 0-25% buffer
  range on the relevant large shape. This is not yet a drop-in production
  subkernel because the benchmark reports the receiver-local W13 layout and
  skips reference row-order checks; the next correctness step is to return the
  metadata needed to compare or combine the ring-gather dispatch-up output.
- Next action: Productionize the ring-gather dispatch-up path behind the
  Mosaic/Grug MoE backend boundary, with an explicit correctness check for the
  receiver-local row metadata or a full combine path.

### 2026-06-24 - Fused Pallas ready-dispatch plus block-ready W13 diagnostic

- Hypothesis: Calling scratch-ready Mosaic GPU dispatch and block-ready Mosaic
  GPU W13/SiLU inside one `jax.jit(shard_map(...))` may expose enough
  producer/consumer structure for scheduling overlap, or at least quantify the
  gap before attempting a deeper single-kernel rewrite.
- Changes:
  - Added `--run-pallas-fused-dispatch-up` to
    `bench_moe_dispatch_up_mosaic_gpu.py`.
  - The mode invokes `dispatch_prepacked_moe_dispatch_up_mosaic_gpu_ready_local`
    or `dispatch_prepacked_moe_dispatch_up_mosaic_gpu_direct_ready_local`,
    immediately feeds the resulting `recv_x` and `ready_block_count` to
    `compute_moe_up_mosaic_gpu_block_ready_local`, and reports W13/reference
    parity plus exact ready-count checks.
- Commands:
  - Local validation: `uv run --package marin-levanter python -m py_compile lib/levanter/scripts/bench/bench_moe_dispatch_up_mosaic_gpu.py`
  - Local checks: `./infra/pre-commit.py --changed-files --fix`
  - H128 correctness: `... --job-name dlwh-6597-moe-pallas-fused-h128 -- ... bench_moe_dispatch_up_mosaic_gpu.py --ep-size 8 --tokens-per-rank 8 --experts-per-rank 4 --top-k 4 --hidden 128 --intermediate 64 --recv-capacity-factor 1.25 --dispatch-copy-mode scratch_ready --dispatch-rows-per-program 4 --w13-impl mosaic_gpu_block_ready --run-pallas-fused-dispatch-up --bench-iters 1 --warmup-steps 1`
  - H2560/T=8 correctness: `... --job-name dlwh-6597-moe-pallas-fused-h2560-t8 -- ... --ep-size 8 --tokens-per-rank 8 --experts-per-rank 32 --top-k 4 --hidden 2560 --intermediate 2560 --recv-capacity-factor 1.25 --dispatch-copy-mode scratch_ready --dispatch-rows-per-program 4 --w13-impl mosaic_gpu_block_ready --run-pallas-fused-dispatch-up --bench-iters 1 --warmup-steps 1`
  - T/rank=4096 perf attempt: `... --job-name dlwh-6597-moe-pallas-fused-t4096-buf100 -- ... --tokens-per-rank 4096 --hidden 2560 --intermediate 2560 --recv-capacity-factor 1.0 --dispatch-copy-mode scratch_ready --dispatch-rows-per-program 16 --run-pallas-fused-dispatch-up --skip-reference-checks --bench-iters 1 --warmup-steps 1`
- Result:
  - H128: ready-count and ready-block-count errors `0`; W13 max abs
    `0.0078125`; steady `1.604 ms`.
  - H2560/T=8: ready-count and ready-block-count errors `0`; W13 max abs
    `0.015625`; steady `1.951 ms`.
  - T/rank=4096 0% buffer: prepack completed in `27.3 s`, then the fused
    Pallas candidate produced no timing or error after roughly 40 minutes in
    compile/run. The job was stopped to avoid burning more H100 time:
    `/dlwh/dlwh-6597-moe-pallas-fused-t4096-buf100`.
- Interpretation: The fused diagnostic is correct on the requested H=I=2560,
  E=256, topk=4 validation shape at T/rank=8 and narrowly under 2 ms there.
  It is not viable for the relevant T/rank=4096 throughput shape in the current
  prepacked scratch-ready form: compilation/codegen does not return in a
  practical window, whereas the ring-gather built-in collective path returns
  `1.816-1.861 ms` across 0-25% buffers. A real overlapped Mosaic path likely
  needs a smaller in-kernel/ring transport schedule or a production
  ring-gather/GMM integration rather than the current giant prepacked scratch
  dispatch.
- Next action: Treat the current prepacked scratch-ready fused path as a
  validation/negative-result harness. For production, integrate the correct
  ring-gather dispatch-up path first, or redesign Mosaic dispatch around a
  streaming fixed-neighbor schedule that avoids the `EP x send_capacity x H`
  scratch/prepack shape.
