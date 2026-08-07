# Results and source versions

## Environment

- GPU model: NVIDIA H100 80GB HBM3
- GPU clock or power policy: cluster default, 700 W power limit; clocks were not pinned
- CUDA toolkit used for CODA and QuACK: CUDA 13.0 runtime through Torch 2.13.0
- CUDA toolkit used to build official FA3: CUDA 12.8 compiler/CCCL with the image's CUDA 12.6 toolkit layout
- NVIDIA driver: 595.71.05
- JAX: 0.10.1 locally; 0.11.0 in the H100 image
- StableHLO portable artifact version: 1.14.1
- CODA commit: `8fa88065e541f6a5b52fb400d94d4be02f18c543`
- CODA QuACK commit: `02c7f69881737731173a6a009aeb6f032e449b61`
- Consumer-prologue QuACK base commit: `84ef91df9bec87c7e4938517234fafb07ef844dd`
- FlashAttention commit: `3fa810570e17bb4354155bdb71d826eca6079208`
- Mixture-of-Kittens commit: `3e1cf43ab93ad040afed52a45ab03cb490ffe4be`
- CUTLASS DSL: 4.6.0
- ThunderKittens commit: `1c3920d993404dd49a6d4c7267ea11d583bd5c68` through the pinned Mixture-of-Kittens submodule

The official FA3 build is BF16, head-dimension-128, forward-only, and includes fixed-length and packed-GQA kernels. The Debian 13/glibc 2.41 image required GCC 13 and NVIDIA's documented CUDA-header exception-specification workaround for `sinpi`, `sinpif`, `cospi`, and `cospif`.

## Attention

Configuration: causal BF16 GQA, `B=1`, `Hq=32`, `Hkv=8`, `D=128`. Causal TFLOP/s uses `2 * B * Hq * S^2 * D` for QK and PV over the causal half.

| Sequence | Backend | Median latency | Causal TFLOP/s | Maximum absolute difference vs FA4 |
|---:|---|---:|---:|---:|
| 2048 | Official FA3 | 0.0672 ms | 511.1 | 0.000977 |
| 2048 | FA4 CuTe SM90 | 0.0708 ms | 485.5 | reference |
| 2048 | Torch SDPA flash | 0.1631 ms | 210.6 | 0.003906 |
| 2048 | JAX XLA tensor algebra | 0.8824 ms | 38.9 | not measured |
| 4096 | Official FA3 | 0.2100 ms | 654.4 | 0.000977 |
| 4096 | FA4 CuTe SM90 | 0.2275 ms | 604.2 | reference |
| 4096 | Torch SDPA flash | 0.5137 ms | 267.5 | 0.003906 |
| 4096 | JAX XLA tensor algebra | 3.0960 ms | 44.4 | not measured |

The official source selects 128x128 query/KV tiles, two stages, packed GQA, RS PV, intra-warpgroup overlap, and a persistent scheduler for this configuration. The compiled kernel reported 168 registers. No sequence-squared score or probability tensor is materialized.

JAX 0.11's explicit XLA attention path is 13.1x slower than FA3 at sequence 2048 and 14.7x slower at sequence 4096. Its cuDNN implementation failed during compilation with `No valid execution plans built` for this GQA configuration, matching Torch's initial cuDNN-plan failure; both framework comparisons therefore use their viable fallback paths.

## RMS scale placement

The compared programs are:

```text
consumer epilogue: WGMMA(A, W), then multiply the FP32 accumulator by inverse-RMS
consumer prologue: multiply A by FP32 inverse-RMS, convert to BF16, then WGMMA
materialized: write BF16(A * inverse-RMS), then launch GEMM
```

Primary shape: `M=2048`, `K=4096`, `N=6144`.

| Variant | Median latency | TFLOP/s | Mean absolute error vs source order | Mean absolute error vs ideal FP32 algebra |
|---|---:|---:|---:|---:|
| Raw Torch GEMM, no scale | 0.130 ms | 789-793 | 19.25 | 19.25 |
| Tuned QuACK FP32 consumer prologue | 0.141 ms | 731.1 | 0 | 0.11510 |
| CODA consumer epilogue | 0.150-0.164 ms | 630-689 | 0.08383 | 0.07128 |
| Materialized BF16 pre-scale plus GEMM | 0.210-0.239 ms | 431-491 | 0 | 0.11508 |

The tuned prologue uses a 128x256x64 cooperative tile, an RS WGMMA mainloop, and `quack_fp32_row_scale.patch`. It carries inverse-RMS in FP32, multiplies the register fragment in FP32, and converts only the transformed A fragment to BF16. Given the same stored BF16 gamma-scaled input, it is bitwise equal to the materialized consumer-boundary pre-scale and is 36% faster in the stable primary run. The complete RMS rewrite can still add a prior BF16 rounding when it stores `u * gamma`; the CPU whole-rewrite test therefore establishes lower error than delayed scaling, not bitwise equality.

CODA is closer to an ideal unrounded FP32 expression because delaying the scale avoids the consumer boundary's extra BF16 input rounding. The prologue is preferable when preserving the exported StableHLO ordering more closely is the contract. This is a semantic-policy choice, not a claim that one placement is numerically superior under every reference.

The best physical prologue schedule is shape-dependent. At `M=4096,K=4096,N=6144`, the 128x256x64 cluster-1 schedule measured 0.2786 ms versus 0.4285 ms for materialization. At `M=2048,K=4096,N=28672`, cluster-N=2 improved the same tile from 0.7147 ms to 0.6504 ms, versus 0.6859 ms for materialization and 0.7032 ms for CODA in the recorded runs.

## SwiGLU

Configuration: `M=2048`, `K=4096`, combined gate/up `N=28672`, BF16 inputs/outputs, FP32 accumulation, weights scaled by `K^-0.5`.

| Variant | Median latency | GEMM TFLOP/s | Mean absolute difference vs BF16 source | Mean absolute difference vs ideal FP32 |
|---|---:|---:|---:|---:|
| Torch materialized GEMM plus SwiGLU | 0.7775 ms | 618.7 | reference | 0.000891 |
| CODA SwiGLU, expanded preactivation saved | 0.7501 ms | 641.3 | 0.000786 | 0.000448 |
| QuACK SwiGLU, expanded preactivation dead | 0.6185 ms | 777.8 | 0.000786 | 0.000448 |

The forward-only plan selects QuACK's dead-preactivation form. It is 20.4% faster than the materialized Torch path and avoids the activation-sized expanded gate/up output. CODA's public wrapper saves that tensor for backward, which is useful as a training oracle but does not satisfy this prototype's forward-only materialization objective.

The complete RMS-to-gate/up boundary was then measured with identical dead-preactivation behavior:

| RMS placement with pairwise SwiGLU | Median latency | GEMM TFLOP/s | Mean absolute difference vs BF16 source |
|---|---:|---:|---:|
| CODA-style delayed consumer epilogue | 0.6430 ms | 748.1 | 0.001440 |
| QuACK FP32 consumer prologue | 0.6509 ms | 739.1 | 0.000931 |
| Materialized BF16 pre-scale, GEMM, and SwiGLU | 0.8723 ms | 551.4 | reference |

Once dead-output behavior is matched, delayed scaling is 1.2% faster. The consumer prologue is bitwise equal to materialized BF16 pre-scaling before the GEMM and is 25.4% faster than the complete materialized path. The default dense plan selects the prologue for source-order fidelity, while retaining the delayed epilogue as an explicit performance alternative.

## QKV and RoPE boundary

Configuration: `M=2048`, `K=4096`, packed QKV `N=6144`, `Hq=32`, `Hkv=8`, `D=128`. Q and K are rotated in the projection epilogue; V passes through. The packed allocation exposes contiguous Q, K, and V segment views accepted by the selected FA3 layout contract.

| RMS placement with QKV/RoPE | Median latency | GEMM TFLOP/s | Mean absolute difference vs materialized boundary |
|---|---:|---:|---:|
| CODA-style delayed consumer epilogue | 0.1357 ms | 759.6 | 0.001311 |
| QuACK FP32 consumer prologue | 0.1467 ms | 702.5 | 0 |
| Materialized BF16 pre-scale plus QKV/RoPE | 0.2169 ms | 475.3 | reference |

Cluster-N=2 is the selected QKV schedule. It improves delayed scaling by 7.4% over cluster-N=1 and changes the prologue by less than 1%. The prologue removes the source-order materialization at a 32.4% latency reduction and exactly matches the materialized kernel output; delayed scaling is another 7.5% faster. Relative to an independent FP64-trigonometric source reference, mean absolute error was 0.000971 for the prologue and 0.001727 for delayed scaling; the remaining prologue error is the RoPE kernel's finite-precision trigonometric path rather than RMS placement.

## Connected dense plan

The semantic planner now composes one block through the following QKV/RoPE boundary into eight execution skeletons:

1. packed QKV GEMM with Q/K RoPE;
2. official FA3 streaming attention;
3. attention output projection with residual, gamma, and RMS partials;
4. small RMS reduction;
5. gate/up GEMM with selectable RMS placement and dead-preactivation pairwise SwiGLU;
6. down projection with residual, gamma, and RMS partials;
7. small RMS reduction; and
8. next QKV GEMM with selectable RMS placement and Q/K RoPE.

The plan contains no standalone residual, RMSNorm, RoPE, SwiGLU, or layout-conversion skeleton and no sequence-squared materialization. Residual streams, gamma-scaled cross-skeleton inputs, reduced SwiGLU output, and Q/K/V at the attention boundary remain materialized because they are mainloop inputs or saved residuals.

The same plan is now recovered from a frozen StableHLO v1.14.1 artifact exported from a parameterized JAX region. The fixture contains 184 operations and ten tensor inputs rather than embedded weights. Recovery accounts for every reachable source operation and validates both QKV partitions, adjacent-pair RoPE equations, causal GQA attention, residual paths, RMS reductions, pairwise SwiGLU, and the following QKV outputs.

The QKV-to-FA3 boundary is executable without a repack. For `B=1`, `S=2048`, and the primary model dimensions, Q/K/V are segment views of one packed `M x 6144` allocation with row stride 6144 and contiguous head dimension. Official FA3 accepts those strides and produces a bitwise-identical result to explicitly contiguous Q/K/V. QKV/RoPE plus FA3 measured 0.1944 ms with segment views versus 0.2367 ms with explicit contiguous copies, so the selected layout contract records last-dimension contiguity rather than falsely requiring fully contiguous separate allocations.

## Full dense region

The hand-composed executable plan now launches all eight recovered skeletons in one Torch 2.13 runtime: QuACK/CODA epilogues and RMS reducers plus the official FA3 extension. The materialized Torch comparison uses the same QKV/RoPE and FA3 boundary, then executes source-ordered normalization and activation materializations. JAX uses the ordinary reference tensor algebra under stock XLA:GPU.

| Sequence | Variant | Median latency | Minimum latency | Speedup vs JAX median |
|---:|---|---:|---:|---:|
| 2048 | Consumer-prologue recovered plan | 1.4800 ms | 1.4184 ms | 1.69x |
| 2048 | Delayed-epilogue recovered plan | 1.4561 ms | 1.3859 ms | 1.72x |
| 2048 | Materialized Torch reference | 1.9614 ms | 1.8963 ms | 1.28x |
| 2048 | Stock JAX/XLA | 2.5010 ms | 2.4899 ms | reference |
| 4096 | Consumer-prologue recovered plan | 3.0563 ms | 2.9192 ms | 2.14x |
| 4096 | Delayed-epilogue recovered plan | 3.0080 ms | 2.8642 ms | 2.17x |
| 4096 | Materialized Torch reference | 4.0236 ms | 3.8281 ms | 1.62x |
| 4096 | Stock JAX/XLA | 6.5257 ms | 6.5167 ms | reference |

At sequence 2048, the delayed plan is 25.8% faster than the materialized Torch path and 41.8% faster than stock JAX/XLA. At sequence 4096 the reductions are 25.2% and 53.9%. The prologue policy is within 1.6% of delayed scaling at both lengths in the median runs. JAX compilation took 1.35 seconds at sequence 2048 and 1.53 seconds at sequence 4096.

The full-region comparison also exposes an important numerical distinction. Given the same stored gamma-scaled input, the prologue exactly matches a materialized BF16 consumer boundary. Across the complete region, however, earlier fused residual and gamma operations and the FP32 preactivation SwiGLU introduce their own rounding changes. At sequence 2048, mean absolute final-QKV differences versus the materialized Torch source were 0.003745 for prologue placement and 0.003363 for delayed placement. Prologue placement is therefore the more faithful local implementation of exported RMS ordering, but it does not guarantee the smallest end-to-end error after all other legal fusions.

Phase profiling at sequence 2048 separates scale transport from the consumer kernels. Repeating each FP32 inverse-RMS row scalar into the current K-tile strip costs 0.0074 ms before gate/up and 0.0069 ms before next QKV. Excluding those copies, gate/up measures 0.6890 ms with prologue scaling and 0.6912 ms with delayed scaling, within the run's clock variation. The next-QKV kernels measure 0.1692 and 0.1559 ms, respectively, so most of that boundary's prologue penalty remains in the A-fragment transform rather than strip preparation.

The residual/RMS producer tile was also swept independently. For `M=2048,K=4096,N=4096`, 128x256x64 cluster-1 was best at 0.1037 ms including the 0.0042 ms reducer. For the down projection, `M=2048,K=14336,N=4096`, the same tile measured 0.3085 ms including reduction, ahead of the tested 128x128 and 256x128 alternatives. Both residual/RMS GEMMs therefore record 128x256x64 cluster-1 in the selected plan.

GPU clocks could not be pinned in the Iris container: the device idled at 345 MHz and boosted up to a reported 1980 MHz, while `nvidia-smi -lgc` was denied. Interleaved timing reduced order bias, but the median-to-minimum spread should be treated as residual clock noise.

## Plan-driven H100 runtime

The runtime now validates and executes the `RegionPlan` produced directly from a primary-shape StableHLO export. It allocates physical row-by-N-tile RMS partials, reduces them to one inverse-RMS scalar per row, allocates one packed QKV buffer at each projection boundary, and exposes Q/K/V as segment views. The backend dispatches the eight selected QuACK/CODA and official-FA3 skeletons by their declared backend and attachment contracts.

At sequence 2048, the plan-driven prologue and delayed variants measured minimum latencies of 1.4199 and 1.3973 ms in the longest interleaved run, with medians of 1.5483 and 1.5313 ms under unpinned-clock variation. An earlier shorter run measured 1.4459 and 1.4189 ms medians. At sequence 4096, minimum latencies were 2.9988 and 3.1103 ms, with 3.1889 and 3.1655 ms medians. These ranges overlap the hand-composed oracle and retain the same prologue-versus-delayed output differences, showing that plan dispatch adds no QKV repack or activation-sized transform.

The measured `rope_posfreq` specialization accepts only canonical Llama base-10000 RoPE tables. The H100 backend validates that contract once before entering the timed hot path; arbitrary dynamic sine/cosine inputs require a table-load epilogue specialization rather than being silently ignored.

## Mixture-of-Kittens follow-on

The pinned Mixture-of-Kittens implementation targets SM100/SM103 rather than H100. Its forward megakernel combines shared and routed expert gate/up GEMMs, SwiGLU, down GEMMs, dispatch/combine communication, readiness events, and a persistent task scheduler. It is used to establish an expert-parallel correctness/performance oracle and to inspect useful physical constraints, not as a complete compiler backend. Generated first-principles plans will be compared on low-priority B200 capacity and reported separately from the H100 dense results.

The official commit now builds and runs on one low-priority four-GPU GB200 tray. The isolated environment uses PyTorch 2.10.0+cu130, CUDA NVCC/CCCL 13.0.88/13.0.85, driver 595.71.05, and SM100 code generation. The pip CUDA toolkit required its library directory on `LIBRARY_PATH` so the host linker could find `libcudadevrt.a` and `libcudart_static.a`; no MoK or ThunderKittens source changes were made.

Official benchmark configuration: 2048 tokens per rank, 384 experts (96 per rank), top-6 routing, hidden size 7168, intermediate size 3072, four GB200 ranks, BF16 forward/backward communication workers 24/28 SMs, MXFP8 36/36 SMs, minibatch 4096, and macrobatch 131072.

| Precision | Forward latency | Forward TFLOP/s | Backward latency | Backward TFLOP/s |
|---|---:|---:|---:|---:|
| BF16 | 3.669 ms | 516.2 | 9.029 ms | 419.5 |
| MXFP8 | 2.521 ms | 751.2 | 7.662 ms | 494.4 |

The official correctness check passed before timing. BF16 output mean/maximum absolute error versus its reference was 0.001723/0.03125; MXFP8 output error was 0.013673/0.121094. Backward checks for inputs, router weights, routed weights, and shared weights also completed within the repository's declared tolerances.

A bounded BF16-forward sweep varied the number of communication SMs and the computation/communication minibatch. With the initial 131072-token macrobatch, the best candidate was 24 communication SMs and minibatch 2048 at 3.6330 ms/521.4 TFLOP/s. A follow-up sweep kept the upstream 32:1 macro-to-minibatch ratio and found 20 communication SMs with minibatch 2048 and macrobatch 65536 at 3.5842 ms/528.4 TFLOP/s. The official end-to-end correctness benchmark with that schedule passed and measured BF16 forward at 3.613 ms/524.2 TFLOP/s and backward at 9.077 ms/417.3 TFLOP/s. Its BF16 numerical errors were unchanged from the default schedule; MXFP8 used its original 36-SM configuration.

The sweep establishes two physical legality constraints for the planner: minibatch size must be positive and divisible by 256, and macrobatch size must be a positive multiple of minibatch size. It also shows that communication-worker allocation is a measured schedule parameter rather than a semantic constant: reducing BF16 forward communication workers from 24 to 20 improved this shape, while the best value varied with batching choices.

The MoK task graph has been extracted as an oracle description exposing its router/top-k boundary, 256-padded expert schedule, symmetric dispatch and combine workspaces, shared and routed gate/up/SwiGLU/down tasks, five readiness-event families, persistent communication and compute worker roles, and pinned revision. It is not the compiler backend. The compiler path will instead derive a distributed plan from a global semantic expert axis using generic route-relation, ownership, segmentation, tile-flow, buffering, and scheduling transforms. The performance target is within 20–30% of the measured BF16 MoK forward oracle.

The ordinary JAX frontend exports a 70-operation StableHLO region using the versioned `chlo.top_k` composite and static expert gathers. Recovery verifies the FP32 top-k softmax, shared and routed SwiGLU algebra, global expert-weight axis, and deterministic weighted merge before producing five semantic operations. The private top-k sort decomposition is deliberately not imported.

The first generic physical plan derives 17 separate stages. It groups routes by destination owner before transport, optionally coalesces payload rows by `(source_token, owner_rank)`, expands metadata on receipt, then segments and pads each receiver-local expert exactly once. At the oracle shape, the guarded 1.25x receiver assignment capacity is 15360 rows and the current 256-aligned padded-local capacity is 39840 rows; overflow selects an exact fallback before contraction. Gate/up physical layouts remain a choice among separate, concatenated, and interleaved forms. Every tile-flow edge records tile shape, storage class, readiness granularity, consumers, and a derived lifetime.

The executable CPU index plane instantiates those contracts without MoK schedule data. It produces stable destination rows and reverse mappings from arbitrary router output, including one optional exchange row per distinct `(source token, destination rank)`. Dispatch, inverse dispatch, and coalesced expansion are exact permutations. Weighted combination restores `(source token, route slot)` and accumulates FP32 values in ascending route-slot order, so execution is deterministic without unordered atomic accumulation. A seeded 384-expert, four-rank, 2048-token, top-6 relation fits the guarded capacity; structured overflow is reported before payload dispatch.

The first generic payload implementation uses JAX ragged all-to-all plus Triton `ragged_dot`, not the fused MoK kernel. The exact route fixture was generated from MoK's per-rank CUDA seeds `1234 + rank`; its saved selected-expert and FP32 router-weight arrays have SHA256 `6ffd9d42c0ae1da109503f3d3a5d6ec992ffdbb84f41b4cc6f0493f35f5c0dff`. On the active JAX/JAXlib 0.11.0 GB200 image, both native ragged-all-to-all runtime paths segfault on first execution even for a tiny four-rank graph. Disabling the one-shot path and enabling XLA's ragged collective decomposer is correct and stable.

At the exact MoK shape, decomposed transport plus Triton segmented GEMMs compiled in 8.858 seconds and measured 9.460 ms median over 50 iterations after 10 warmups, or 200.2 logical TFLOP/s per rank. Output was finite and no assignments were dropped. Replacing only Triton `ragged_dot` with XLA increased compile time to 60.800 seconds and latency to 95.100 ms, so XLA's grouped contraction is excluded from subsequent candidates.

Phase probes measured 0.382 ms for the shared expert, 4.875 ms for routed dispatch plus inverse dispatch, and 4.696 ms for the already-routed padded segmented W13/SwiGLU/W2 computation. Exact routes touch 3.301 destination ranks per token on average, so coalescing H-wide forward payload by `(token, owner rank)` reduces rows from 6.0 to 3.301 per token. Receiver assignment counts were `[12281, 12281, 12349, 12241]`. Tightening capacity from 15360 to the exact safe 12349 rows reduced full latency to 8.445 ms and local segmented compute to 4.183 ms, both 10.7% improvements with zero drops. A 12473-row guard was effectively tied at 8.492/4.184 ms.

The deterministic DeepEP candidate removes its prior unsorted segment accumulation: expert-sorted outputs retain their original `(received token, route slot)` positions, then compact outputs are sorted and found by per-slot search/gather before ascending route-slot FP32 accumulation. No reduction scatter or unordered atomic is used. Before specialization, the 49152-row local assignment domain measured 11.354 ms, 166.8 logical TFLOP/s per rank.

Replacing that domain with one global compact batch reduced full latency to 6.149 ms at the exact 12349-row bound and 6.113 ms with 1% headroom at 12473 rows. Output remained finite and no assignments were dropped. The capacity is `ceil(capacity_factor * local_tokens * top_k)` and overflow is counted exactly through the returned distributed drop count, so the relation planner can reject the candidate and select its exact fallback before payload execution. The selected guarded DeepEP plan is 46.2% faster than uncapped DeepEP, 27.6% faster than the best decomposed-ragged plan, and 1.69 times the 3.613-ms MoK oracle. It remains outside the 1.3-times target.

### Standalone MoK grouped-GEMM primitive probe

The routed BF16 grouped-GEMM implementation below MoK's complete event graph is reusable without invoking dispatch, communication, SwiGLU, combine, or the full persistent forward kernel. The probe includes the pinned upstream header tree at build time, initializes only the two-CTA TMA/tensor-memory pipeline, launches one cluster per 256x256 output tile, and passes null for every cross-task readiness event. No MoK or ThunderKittens source was copied or modified.

The measured input contains 96 local experts with 256 padded rows each, for 24576 physical rows. Hidden and intermediate dimensions are 7168 and 3072. W2 is one grouped contraction; W13 is the sum of separate gate and up launches, matching MoK's two primitive calls.

| Component | Physical shape | Launches | Median latency | Padded-work throughput |
|---|---|---:|---:|---:|
| W2 | `M=24576, K=3072, N=7168, E=96` | 1 | 0.943 ms | 1148 TFLOP/s |
| W13 gate plus up | `M=24576, K=7168, N=3072, E=96` | 2 | 2.036 ms | 1063 TFLOP/s |

The combined projection time is 2.979 ms before SwiGLU. Throughput counts all 256-padded rows and is therefore physical kernel work, not unpadded model FLOPs. A two-expert 256-dimensional correctness probe passed against FP32 Torch with maximum absolute error 0.0149 and no NaNs or infinities. The compiled worker uses 255 registers, five barriers, 224 bytes of static shared memory, and no spills.

The isolated build required a coherent CUDA 13.0 package set: NVCC 13.0.88, CCCL 13.0.85, CUDA CRT 13.0.88, and NVVM 13.0.88. Mixing NVCC 13.0 with CUDA 13.2 headers failed the CCCL compatibility check; mixing it with CRT/NVVM 13.3 emitted PTX 9.3 for a ptxas supporting PTX 9.0. The CUDA 13.0 library directory remains required on `LIBRARY_PATH` for static runtime libraries.

This result isolates grouped compute but is not a component timing extracted from the complete MoK forward. It omits CLC work redistribution and MoK's concurrent task/event schedule, and W13 is timed as two sequential launches. It nevertheless establishes a strong reusable SM100 compute oracle: the generic plan's remaining gap is not evidence that expert GEMM itself must be regenerated from a general-purpose backend. Compact Triton expert compute measures 4.185 ms while raw DeepEP dispatch plus combine measures 1.340 ms; substituting the 2.979-ms standalone projections gives a non-overlapped estimate of approximately 4.92 ms before explicit SwiGLU and packing are measured. The next experiment should connect this primitive to compiler-produced expert-contiguous offsets and then overlap bounded transport with these contractions.

### Generated receiver-local MoK composition

The generic relation planner now drives a Torch/CUDA composition at an already-dispatched DeepEP-shaped boundary. For receiver rank 0, an independent direct scan of the raw route table matched all 6755 coalesced receive tokens, 12281 assignments, and 24576 padded rows. The check covers ascending source-token receive order, source route slots, local expert counts, 256-row padding, route-to-padded-row indices, and the inverse padded-row mapping.

The generated path uses the standalone MoK W13 and W2 mainloops, a BF16x2 inverse-map pack kernel, a BF16-output SwiGLU kernel, and an ascending-route-slot FP32 merge. The exact MoK shape is four ranks, 2048 tokens per rank, 384 global experts, top-6 routing, hidden size 7168, and intermediate size 3072. Timings are 10 warmups and 50 iterations on one GB200 rank.

| Receiver-local stage | Torch median | Generated median |
|---|---:|---:|
| Simulated coalesced receive gather | — | 0.038816 ms |
| 256-padded expert pack | 0.384576 ms | 0.305104 ms |
| W13 gate plus up | — | 1.674016 ms |
| SwiGLU | 0.818992 ms | 0.125232 ms |
| W2 | — | 0.918880 ms |
| Pre-combine route merge | 3.037072 ms | 0.360832 ms |
| Full receiver-local pre-combine composition | 6.747360 ms | 3.454944 ms |

The generated full composition reaches 469.64 logical TFLOP/s and 939.81 padded-work TFLOP/s. It is 1.95 times faster than the matching Torch launch sequence. This comparison excludes official DeepEP dispatch and combine, shared-expert compute, the MoK CLC scheduler, and transport/compute overlap.

The merge kernel does not use atomics. It visits route slots in ascending order and uses explicit `__fmul_rn` followed by `__fadd_rn`. That form is bitwise equal to the Torch reference, with zero maximum and mean absolute error, and repeats bitwise. Allowing contraction to FMA saves 0.002592 ms but is not bitwise equal: maximum absolute error is `2.3841858e-7` and mean absolute error is `1.2035570e-13`. The FMA form remains repeat-bitwise and passes the numerical tolerance. The generated pack and SwiGLU were bitwise equal to their Torch references. The full owner-local merge/shared-add diagnostic was also bitwise equal, repeat-bitwise, and finite.

Official pinned DeepEP combine remains the cross-rank numerical boundary. Its intranode kernel enumerates contributing ranks in ascending rank order, adds each rank's buffer sequentially in FP32, and converts to BF16 without atomic accumulation. The distributed runtime below supplies generated receiver-local output to this combine; the owner-local merge/shared-add diagnostic is not used as a substitute.

### Four-rank generated DeepEP composition

The complete compiler-derived path now runs on four GB200 ranks. DeepEP performs destination-rank-coalesced dispatch and fixed-rank combine. The compiler's `RelationPlan` legalizes DeepEP receive order, groups and pads receiver-local experts, and supplies the inverse mappings used by generated pack and fixed-slot merge kernels. Standalone grouped GEMMs execute W13 and W2, generated CUDA executes SwiGLU, and the generated shared expert overlaps asynchronous dispatch. The runtime does not call the MoK forward kernel or its event graph.

The first 24-SM plan measured `4.2682` ms rank-maximum latency over 10 warmups and 50 iterations. A communication-worker sweep improved this to `4.0148` ms at 80 SMs and turned upward to `4.0395` ms at 96 SMs. Concatenated W13 was then compared with separate gate/up projections at 56 and 80 SMs:

| DeepEP SMs | Gate/up layout | Routed, already dispatched | Sequential | Shared/dispatch overlap |
|---:|---|---:|---:|---:|
| 56 | Concatenated `[E,2I,K]` | 3.4843 ms | 4.0533 ms | 3.9760 ms |
| 56 | Separate `[E,I,K]` | 3.5604 ms | 4.1536 ms | 4.0797 ms |
| 80 | Concatenated `[E,2I,K]` | 3.5182 ms | 4.0085 ms | 4.0305 ms |
| 80 | Separate `[E,I,K]` | 3.5970 ms | 4.0916 ms | 4.1298 ms |

Concatenation reduced overlap latency by `0.1037` ms (`2.54%`) at 56 SMs and `0.0993` ms (`2.40%`) at 80 SMs. A second concatenated 56-SM run measured `3.9910` ms, giving a two-run median of medians of `3.9835` ms. The selected 56-SM concatenated plan is approximately `10.3%` slower than the `3.613`-ms tuned MoK oracle and `34.8%` faster than the `6.113`-ms compact DeepEP baseline.

Every rank in the four A/B runs and the selected-plan repeat matched compiler and DeepEP transport metadata exactly. Outputs were finite, sequential and overlapped paths were bitwise equal, and the overlapped output repeated bitwise. An independent small four-rank source-ordered Torch reference passed with maximum absolute error `0.0001220703125`.

The route fixture has two identities. The original NPZ container SHA256 is `6ffd9d42c0ae1da109503f3d3a5d6ec992ffdbb84f41b4cc6f0493f35f5c0dff`; reserializing the same seeded arrays on a replacement tray produced container SHA256 `c143b12f2879430106d5013aea8e95ef0705ba8daaffa5eeb1ece49559217d38`. The stable tensor-content SHA256 is `f1b5d8b3a53372eca228261b48b7ad9cfe925f1f8083f9cae07f9a24713f6908`. This hash frames each tensor's name, NumPy dtype string, little-endian rank and shape, and C-contiguous bytes in the order `selected_experts`, `combine_weights`. Both serializations produce receiver assignment counts `[12281,12281,12349,12241]`. `scratch/shuttle-generic-results/mok-route-fixture-content-identity.json` records the framing and per-tensor byte hashes.

### Reproducibility snapshot

The annotated tag `shuttle-gb200-moe-v1` seals the source and benchmark record. The generated replay itself uses Shuttle revision `3dd61fad063bae54ac5e337d8f1657264011d6ff`; the tag points to the later archival commit that adds the immutable artifacts and documentation without changing the measured implementation.

The [snapshot manifest](../benchmarks/artifacts/gb200_moe_v1/manifest.json) pins DeepEP, MoK, ThunderKittens, CODA/QuACK, FA3, CUTLASS DSL, CUDA component versions, driver, target architecture, complete four-rank shape, dtypes, seeds, timing protocol, and SHA256 identities for every preserved input and result. The [candidate space](../benchmarks/artifacts/gb200_moe_v1/candidate_space.json) records the bounded search procedure, legal dimensions, unmeasured alternatives, failures, pruning reasons, and selected plan. The [benchmark cache](../benchmarks/artifacts/gb200_moe_v1/benchmark_cache.json) maps candidate fingerprints and timing phases to content-addressed raw runs.

The historical snapshot contains 20 distributed run records. Each retains six 50-sample rank-maximum timing distributions, for 6,000 raw phase samples; the cache indexes 2,000 end-to-end selected/no-overlap samples. It includes the complete 12-to-96 communication-SM sweep, concatenated-versus-separate gate/up comparisons at 56 and 80 SMs, two selected-plan confirmations, and every sequential no-overlap measurement. Those runs predate telemetry capture, so their clocks are explicitly recorded as unknown rather than reconstructed.

The telemetry replay adds per-rank and rank-maximum timing samples, a deliberately coarse activation-materialization phase, exact command lines, GPU UUIDs and topology, power limits, clock policy, observed clocks around each phase, extension/source hashes, deterministic output hashes for every rank, and four saved independent semantic fixtures. The pinned MoK 20-SM, minibatch-2048, macrobatch-65536 oracle is replayed under the same protocol with its raw distribution retained.

| Replay variant | Median rank-maximum latency |
|---|---:|
| Shuttle, concatenated W13 with overlap | 3.9830 ms |
| Shuttle, concatenated W13 without overlap | 4.0649 ms |
| Shuttle, concatenated W13 with coarse materialization | 4.4348 ms |
| Shuttle, separate gate/up with overlap | 4.0690 ms |
| Tuned MoK oracle | 3.5617 ms |

The fresh Shuttle result is 1.118× the MoK oracle. Overlap saves 2.02% against the otherwise matching sequential schedule, concatenated W13 saves 2.12% against separate gate/up, and coarse activation materialization costs 11.34% against the selected plan. Each Shuttle phase retains 50 rank-maximum samples and four per-rank 50-sample distributions; MoK retains 50 rank-maximum and per-rank samples after 100 warmups.

Application-clock controls are deprecated on this GB200 system, so the replay records a cluster-default, unpinned policy rather than claiming locked clocks. Every benchmark telemetry capture reported 1950 MHz SM and 3996 MHz memory clocks on all four GPUs; the pre-benchmark idle snapshot reported 120 MHz SM. The advertised SM maximum is 2062 MHz, the power limit is 1200 W, and sampled draw ranged from 199.83 to 757.32 W. The raw records retain all four GPU UUIDs and the NV18 tray topology.

## Routed sparse-attention generality checkpoint

The first routed-attention slice uses the same binary `RelationPlan` for query-major and KV-major schedules. The index-plane transfer is substantial, but the scheduling transfer is not yet complete enough to call the whole abstraction validated.

| Classification | Actual first-slice changes |
|---|---|
| Reused unchanged from MoE | destination ownership, stable destination grouping, group counts/offsets, padding, source dispatch, coalesced dispatch, capacity checks, and most inverse-map construction |
| Generalized from MoE | ragged `edge_valid`, invalid destination sentinels, inverse-dispatch fill values, compact destination-major edge views, and invalid-edge exclusion from merge/capacity accounting |
| New generic Shuttle machinery | counted/generation-scoped readiness records, bounded-buffer records, and a reusable exact-attention partial-state algebra |
| Sparse-attention-specific semantics | selected-block legality, duplicate rejection, causal/tail masking, GQA head mapping, and stable online-softmax state merge |
| Sparse-attention-specific backend | RelationPlan-to-block-mask legalization for query-major sparse kernels; grouped QK/PV physical work remains backend-specific |

The honest negative result is that MoE's existing worker-pool, buffer-derivation, event, and transport schedule types were expert-specific and did not transfer unchanged. The sparse plan required small generic counted-event and bounded-buffer records. It did not require a `MoBA` or `SparseAttentionRoutePlan` node.

Both CPU schedules match an independent selected-mask reference for causal GQA with uneven relation degree, padded destination groups, and a partial final sequence block. KV-major execution restores maximum, denominator, and weighted-value fields independently, then merges in ascending selected-slot order without atomics; repeated output is bitwise deterministic. The full tile-lifetime suite has 74 passing tests.

At sequence 16384, block size 128, top-k 8, 32 query heads, 8 KV heads, and dimension 128, the synthetic causal relation contains 996 valid edges. Query-major retains one online state on chip in the physical plan. The deliberately coarse two-kernel KV-major candidate would materialize 2.12 GB of FP32 partial state. That cost makes bounded forwarding or a compact-state schedule mandatory before the KV-major orientation can plausibly win.

The first bounded KV-major alternative is an ascending selected-slot wave schedule. Each query has at most one edge in a wave, so grouped KV-major work has exactly one writer for each query state and needs no atomic accumulation. At 16K it replaces the 2,121,400,320-byte edge-partial buffer with one 272,629,760-byte per-query online-state buffer, a 7.78x capacity reduction. The plan derives per-slot/per-KV arrival counts from the same relation and exposes eight wave boundaries.

The first executable H100 fallback uses the pinned SeerAttention Triton kernel. At sequence 2048 it matches an independent source-ordered FP32 selected-block reference with maximum/mean/p99 absolute errors of `0.0078125`, `8.28e-5`, and `0.0009766`. Its 30-sample median is `0.316752` ms. At sequence 16384, 50 raw samples give a `2.388208`-ms median (`2.384032–2.401760` ms), or `111.95` selected-work TFLOP/s. Dense causal Torch GQA SDPA on the same logical Q/K/V measures `6.282496` ms (`6.210304–6.357888` ms).

This is a query-major execution checkpoint, not yet a compact-relation oracle. Every Seer query program scans all causally eligible KV blocks and mask-tests them in the loop. Seer also lacks GQA indexing, so the adapter repeats K/V from 8 to 32 heads once outside the timed region, adding 201,326,592 bytes and 52.05 ms at sequence 16384. The sparse-kernel latency excludes that expansion. The cached relation has SHA256 `b2a57606e303f8af4da0c8002ddea162f86625725696bca7f18b8072a8143427`; the core deterministic relation generator now reproduces that exact Boolean relation and a structural test pins the hash. The output hash is `91972fce5061fde100dd022584692b6fc356e5e3e8fda0b06e77936af1445555`.

The pinned Flash Sparse Attention (FSA) adapter provides an executable KV-major oracle for the same 16K/block-128/top-k-8 relation. It converts only generic `RelationPlan` fields into FSA's int32 `[Hkv,T,topk]` input by repeating each block edge across its query tokens and KV heads. The FSA public call then constructs its own block-to-token orientation; it cannot consume Shuttle's grouped offsets or inverse map. Its 30-sample median is `12.5392` ms (`12.3622–13.0573` ms), or `21.322` selected-work TFLOP/s over 267,361,714,176 QK+PV FLOPs. Eight sampled query blocks match an independent source-ordered FP32 reference with maximum/mean/p99 absolute errors of `0.0207922`, `0.000164022`, and `0.00120181`. Repeated output is bitwise identical and contains no NaN or infinity.

FSA's source-visible partial and statistics buffers account for 111,225,856 bytes, and its internal inverse-index structures account for 20,865,024 bytes. The measured peak allocator increment is 431,091,712 bytes. These buffers are reused serially across query heads; Shuttle's coarse all-edge plan declares 2,121,400,320 bytes because it materializes every query-head edge state simultaneously. This gap motivates the bounded slot-wave candidate rather than invalidating KV-major orientation. The generic relation-plan median is `0.6411` ms and the steady FSA adapter median is `0.6226` ms; four cold adapter samples took `0.93–1.07` seconds and remain in the raw record.

Pristine FSA revision `7ff144fd7ff485dc4220d439f31cc1708b64fef3` fails Triton 3.4 JIT because `reduce_kernel` wraps one pointer expression in an accidental singleton tuple. The measured runs remove only that trailing comma in an ephemeral checkout. Every artifact records the pinned head, dirty status, and exact one-line diff. The failure log, adapter, raw samples, hashes, buffer accounting, and telemetry are preserved in `benchmarks/artifacts/routed_sparse_attention_fsa_h100_v0`.

The bounded slot-wave plan now executes directly from Shuttle's destination-grouped relation. Each Triton program is the sole writer of one query-block/head/query-row FP32 online-state tile; the output is finalized to BF16 after all eight waves. The selected 32-row/four-warp tile measures `0.502016` ms at 2K and `4.017344` ms at 16K over 30 samples. The 16K range is `4.011936–4.027776` ms, or `66.552` selected-work TFLOP/s. It is 1.68 times slower than the Seer query-major smoke but 3.12 times faster than the FSA public-call adapter. The comparison is architectural rather than perfectly matched: Seer scans a dense causal mask and expands GQA outside timing, while FSA reconstructs its private inverse relation and materializes partial state inside timing.

At 16K, eight sampled query blocks have maximum/mean/p99 absolute errors of `0.00783062`, `0.000124260`, and `0.000865310` against the independent source-ordered FP32 reference. There are no NaNs or infinities. Repeated BF16 output is bitwise identical with SHA256 `7fee4b9c61ea72736f203fad5ab212f1f31d9178f750bc967f8c8db2eeb66917`. The input relation remains SHA256 `b2a57606e303f8af4da0c8002ddea162f86625725696bca7f18b8072a8143427`.

The bounded physical candidate set rejected a 16-row/four-warp tile at `0.569072` ms and a 64-row/eight-warp tile at `0.660096` ms on the 2K shape; M32 measured `0.502016` ms. A requested source-order/no-destination-sort ablation measured `4.018880` ms at 16K versus `4.017344` ms for explicit destination sorting, but it is not evidence about cache locality: the canonical relation is already KV-monotone within every selected slot, so both variants execute identical edge arrays and produce the same hash. A non-monotone fixture is required for that ablation.

This result validates deterministic bounded state consumption and direct use of the generic relation plan, not shared KV staging. Destination-grouped edges still launch independent CTAs; grouping can affect cache locality but does not stage one KV block for multiple query CTAs. Eight global wave boundaries and 272.6 MB of FP32 state traffic remain the clearest gap to the query-major baseline. Raw samples, full plan dumps, hashes, telemetry, and every accepted or rejected physical point are preserved in `benchmarks/artifacts/routed_sparse_attention_h100_v0/slot_waves`.

Distributed sparse attention is deliberately deferred. `RelationPlan` ownership, coalescing, and inverse mapping transfer conceptually, but the current DeepEP/MoE adapter moves BF16 routed rows and performs fixed-slot vector merges. It cannot transport structured FP32 attention state or KV-block payloads without a new backend adapter. Building that transport and distributed schedule would be significant infrastructure rather than a small reuse test, so this slice stops at the single-H100 conclusion.

The pinned MIT Block-Sparse-Attention build is blocked by the H100 holder image rather than a measured source incompatibility: driver 595.71.05 is present, but the image has no CUDA toolkit or `nvcc`; PyPI CUDA 12.8 packages provide PTXAS and headers but not the compiler driver. The exact build traceback, source revisions, adapter script, raw distributions, hashes, GPU metadata, and reproduction commands are preserved in `benchmarks/artifacts/routed_sparse_attention_h100_v0`.

## Known limitations

- The plan-driven runtime consumes an in-memory `RegionPlan`; durable JSON serialization and compiled-artifact caching are not implemented yet.
- The complete debug block is recovered from one connected StableHLO fixture; recognition robustness across other JAX/StableHLO canonical forms remains unmeasured.
- The FP32 prologue repeats the row scale for every K tile. The two explicit strip preparations cost about 14 microseconds together at sequence 2048; a reducer that directly emits the physical strip would remove those launches, while a true K-invariant row-vector path would address the remaining mainloop overhead.
- The experimental transform is legal only for tile K=64. A K=128 test compiled but produced grossly incorrect results and must be rejected until its fragment mapping is repaired.
- Historical H100 and first-pass GB200 measurements used unpinned clocks. The GB200 replay now preserves exact observed phase-boundary clocks and power telemetry, but application clocks remain deprecated and unlocked.
- The JAX FA4 wrapper aborts in the ragged-TMA adapter on this toolchain; the physical Torch/CuTe kernel itself works.
- The measured pos/frequency RoPE backend is specialized to canonical base-10000 tables; a general table-load QKV epilogue remains to be implemented.
- The expert-parallel compiler currently models BF16 routed experts. MXFP8 is rejected until its weight and activation scale tensors are represented explicitly in the semantic and physical plans.
- The selected 3.984-ms distributed plan is measured only for one BF16 MoE shape on one four-GB200 tray. MXFP8, larger token counts, profiler evidence for the remaining 0.37-ms oracle gap, and schedules that pipeline routed rows in chunks remain unmeasured.
