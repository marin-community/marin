# Results and source versions

## Synthesis boundary

The dense numbers below are oracle-backed composition checkpoints rather than
end-to-end kernel-synthesis results. Dense execution calls named QuACK/CODA
epilogues and official FA3. The historical distributed-MoE result uses DeepEP
`combine`, which performs reverse movement and fixed-rank accumulation
together. The accepted natural-boundary MoE result starts from runtime router
logits and top-k, generates its receiver `RelationPlan`, uses DeepEP only for
forward payload dispatch, returns payloads with `all_to_all_single`, and runs a
generated rank-ordered Fold and shared Map. The standalone MoK-derived grouped
GEMM is treated as an allowed segmented-contraction skeleton. The routed
slot-wave attention body is
Shuttle-owned rather than a Seer/FSA call, but is still hand-authored
attention-specific Triton rather than generated from generic
Relation/Fold/Contract semantics.

These measurements remain correctness and performance targets. A result is
called a synthesized kernel only when Shuttle emits it from generic semantic
factors and reusable skeletons without invoking a complete workload kernel.
The detailed audit is in
`.agents/projects/tile_lifetime_compiler/synthesis_boundary_audit.md`.

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

### Clean generated dense path

The accepted replacement no longer dispatches the named QuACK/CODA semantic
epilogues or official FA3 used by the historical plan above. Ordinary JAX
StableHLO erases into 36 generic `Map`, `Contract`, `Fold`, and
`DomainRestriction` operations. Scalar/tile AST generation supplies Contract
preparation, finalization, auxiliary RMS partials, RoPE, and SwiGLU around the
generic QuACK/CuTe mainloop. A generated SM90 streaming skeleton supplies the
QK, normalized-exponential state, and PV body.

| Sequence | RMS policy | Shuttle median | Matched oracle | Ratio |
|---:|---|---:|---:|---:|
| 2,048 | source-ordered prologue | 1.705818 ms | 1.523838 ms | 1.119422x |
| 2,048 | delayed epilogue | 1.650502 ms | 1.523838 ms | 1.083122x |
| 4,096 | source-ordered prologue | 3.478322 ms | 3.253411 ms | 1.069131x |
| 4,096 | delayed epilogue | 3.390837 ms | 3.253411 ms | 1.042240x |

These are pooled medians from two independent 30-sample captures; process
order is generated-first in run 1 and oracle-first in run 2. All four
candidates pass the 1.20-times completion ratio, and three pass the 1.10-times
stretch target. They also remain below the conservative completion thresholds
derived from the earlier official-FA3 manual oracle.

Generated component outputs are bitwise equal to their primitive oracles
except for the direct scalar-AST SiLU expression, whose maximum BF16-rounded
difference is 0.125. Replacing `SiLU(left) * right` with `left * right` changes
emitted arithmetic through the same generator. Final raw samples, generated
source, hashes, and dependency lineage are under
`benchmarks/artifacts/dense_clean_synthesis_h100_counterbalanced_v1`; the
earlier `dense_clean_synthesis_h100_20260807` checkpoint retains the focused
component comparison and mutation evidence.

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

### Clean payload return and generated merge

The clean candidate separates payload movement from semantic reduction. DeepEP
dispatch moves routed inputs to expert owners. After generated packing,
standalone grouped W13/W2, generated SwiGLU, and the shared-expert branch,
`all_to_all_single` returns owner partials without reducing them. A generated
CUDA kernel folds owner ranks in ascending order with explicit FP32
multiply/add, adds the shared output, and converts to BF16. It uses no atomics
and does not call DeepEP `combine` or the complete MoK forward event graph.

| Phase | First 30-sample median | Confirmation median |
|---|---:|---:|
| Routed compute after dispatch | 3.555584 ms | 3.536016 ms |
| Generated shared expert | 0.242464 ms | 0.240784 ms |
| Payload return plus generated merge | 0.365168 ms | 0.368576 ms |
| Clean sequential region | 4.229424 ms | 4.175808 ms |
| Clean overlapped region | 4.082608 ms | 4.142576 ms |
| DeepEP-combine control | 4.085024 ms | 4.044240 ms |
| DeepEP combine plus shared bias component | 0.271072 ms | 0.274544 ms |

The first clean run is `1.1463x` the frozen `3.561696`-ms MoK replay. The
confirmation is `1.1631x`; their median of medians is `4.112592` ms, or
`1.1547x` the replay. Both satisfy the `1.2x` supplied-route target. The first
run's clean overlap is 3.47% faster than its matching clean sequential
schedule. Payload-only return plus generated merge costs 0.094096 ms more than
the DeepEP combine component in that run.

All four generated rank outputs are bitwise equal to the DeepEP-combine control
and bitwise stable across repeats. The per-rank semantic fixture SHA256 values
also match across both complete runs. The [clean boundary artifact](../benchmarks/artifacts/gb200_moe_clean_merge_v0/README.md)
contains both raw distributions, stdout and build logs, the exact executed
sources, package and hardware pins, route and semantic fixtures, and validated
checksums. This is a synthesized distributed schedule at a supplied-route
boundary. Router/top-k execution and index-plan construction are not included
in the measured region.

The route fixture has two identities. The original NPZ container SHA256 is `6ffd9d42c0ae1da109503f3d3a5d6ec992ffdbb84f41b4cc6f0493f35f5c0dff`; reserializing the same seeded arrays on a replacement tray produced container SHA256 `c143b12f2879430106d5013aea8e95ef0705ba8daaffa5eeb1ece49559217d38`. The stable tensor-content SHA256 is `f1b5d8b3a53372eca228261b48b7ad9cfe925f1f8083f9cae07f9a24713f6908`. This hash frames each tensor's name, NumPy dtype string, little-endian rank and shape, and C-contiguous bytes in the order `selected_experts`, `combine_weights`. Both serializations produce receiver assignment counts `[12281,12281,12349,12241]`. `scratch/shuttle-generic-results/mok-route-fixture-content-identity.json` records the framing and per-tensor byte hashes.

### Natural StableHLO boundary and generated index plane

The accepted path no longer begins from the route fixture above. An ordinary
JAX MoE StableHLO artifact lowers to generic `Contract`, selection, `Relation`,
`SegmentedContract`, `Map`, and `Fold` operations. At runtime the benchmark
executes the BF16 router Contract, top-k, and FP32 normalized route-weight
Maps/Fold. DeepEP dispatches the resulting payload and relation edges. A
generated fixed-capacity GPU kernel then constructs receiver-local counts,
padded source rows, edge destination rows, and ordered edge weights.

The matched MoK path executes the identical router/top-k/route-weight frontend
before MoK schedule construction and its complete BF16 forward. Two independent
captures counterbalance launch order and contain 30 rank-maximum samples per
implementation:

| Capture | Launch order | Shuttle | MoK | Ratio |
|---|---|---:|---:|---:|
| Run 1 | Shuttle first | 4.126384 ms | 3.645056 ms | 1.132050x |
| Run 2 | Oracle first | 4.140336 ms | 3.642048 ms | 1.136815x |
| Pooled 60 samples | Counterbalanced | 4.137120 ms | 3.645056 ms | 1.134995x |

The pooled result passes the 1.20-times completion gate and misses the
1.10-times stretch target. A prior 60-sample scalar-Fold candidate pooled to
4.364224 ms versus 3.631632 ms, or 1.201725 times, and remains preserved as a
negative result. Vectorizing the generic deterministic route-slot and rank
Folds over BF16 pairs reduced Shuttle latency by 0.227104 ms. Each component
still performs explicit FP32 round-to-nearest multiply and add in fixed slot or
rank order.

All four device-generated relations match the independent relation exactly,
with zero overflow. Repeated Shuttle outputs are bitwise equal. Maximum error
against MoK is `0.0001220703125`; the largest rank mean error is
`2.667012722668005e-06`. The generated relation and Fold source contains no
semantic atomic operation. DeepEP internal readiness counters remain transport
control and do not accumulate semantic values or choose their order. Neither
DeepEP semantic combine nor MoK forward appears in the accepted Shuttle path.

The [natural-boundary artifact](../benchmarks/artifacts/gb200_moe_natural_boundary_v0/README.md)
contains both accepted distributions, every earlier candidate and failed
capture, source snapshots, semantic fixtures, revisions, hardware telemetry,
the DeepEP build patch, and validated checksums.

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

### Natural matched query-major synthesis

A later clean path starts from ordinary JAX rather than a relation fixture. The
matched boundary includes the FP32 router metadata Contract, causal domain
restriction, sorted top-k and RelationPlan construction, selected exact
attention with native GQA, and BF16 output on both Shuttle and the pinned MIT
Block-Sparse-Attention oracle paths. QKV and output projections are excluded
symmetrically.

At S=16,384, block 128, top-8, Hq/Hkv=32/8, and D=128, two counterbalanced
30-sample captures have pooled medians:

| Implementation | Pooled median |
|---|---:|
| Generated Shuttle query-major | 0.617584 ms |
| Matched expert oracle | 1.423632 ms |

The ratio is 0.433809 times. Generated versus oracle maximum/mean differences
are 0.00390625/0.0000652, and both outputs repeat bitwise. The oracle is an
SM80-style implementation compiled for SM90, so the result establishes a
matched buildable secondary control rather than the primary acceptance
denominator.

### Matched FlashMoBA primary oracle

Pinned FlashMoBA `39d9ac043b271d046a2181a9991e99a26b67bca1` exactly
matches the BF16 D=128, 32:8 GQA, causal, block-128, top-8 attention payload
through its precomputed-relation interface. Its native router does not match:
it scores every query token/head against mean-pooled K blocks and forces the
current block, whereas Shuttle routes explicit block metadata once per query
block and shares the relation across heads. The full comparison therefore uses
the common Shuttle router and a generic relation reorientation into
FlashMoBA's KV-column-major sorted query-row lists.

The bounded FlashMoBA physical query-group sweep selected 1024. Two independent
counterbalanced 30-sample captures pool to:

| Measurement | Pooled median |
|---|---:|
| Generated Shuttle full boundary | 0.617200 ms |
| Matched FlashMoBA full boundary | 5.264560 ms |
| FlashMoBA cached-relation payload | 4.894560 ms |
| Common router only | 0.044080 ms |
| Relation reorientation only | 0.211664 ms |

The generated/full ratio is 0.117237 times and closes the exact-expert
1.20-times gate. A semantic fixture with 95 query blocks omitting the current
block passes; generated and FlashMoBA outputs differ by at most 0.00390625 with
mean difference 0.0000651724, and both repeat bitwise.

The denominator is not tight. FlashMoBA retains a general per-token/per-head
row-list interface and its active forward path uses SM80-style MMA plus
`cp.async`, not WGMMA/TMA. Shuttle's generated path is specialized to the
block-shared relation and is Hopper-native. The result should be read as a
successful exact-semantic expert comparison, not as an 8.5-times superiority
claim. The MIT 1.423632-ms result is the tighter secondary H100 control, though
it too is SM80-style. A tight future oracle requires a block-shared WGMMA/TMA
body or a natural workload matching FlashMoBA's native router. Raw evidence is
under `benchmarks/artifacts/sparse_flashmoba_h100_matched_v0`.

The earlier fixed-order captures freeze the 1.424720-ms oracle target and remain
in the artifact. The counterbalanced captures confirm the result without
moving that target. Every warmup and measured pair records its launch order.

The physical KV-major gate is also complete. The generated slot-wave schedule
groups a non-monotone relation by KV block within each selected slot, stages
one KV-head block in 65,536 bytes of dynamic shared memory, and reuses it for a
bounded query group. It replaces the coarse 2.12-GB edge-state design with one
272,629,760-byte query-state buffer and zero per-edge partial-state bytes. The
capacity-two schedule covers 996 edges with 671 tasks, is deterministic, and
differs from query-major by at most 0.015625.

The first KV-major body uses CUDA-core QK/PV and measures 107.879105 ms versus
0.574656 ms query-major in the same process. This closes the structural
relation-orientation test but is not a competitive physical candidate.
Query-major remains selected. Evidence is under
`benchmarks/artifacts/natural_routed_sparse_attention_h100_matched_v0`.

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

## Historical routed sparse-attention Seer delta

The frozen 16K Shuttle KV-major slot-wave schedule measures 4.017344 ms versus
2.388208 ms for the Seer query-major baseline, a 1.682-times ratio and
1.629136-ms delta on the identical 996-edge relation.

The difference is accounted for primarily by materialized online state. The
eight Shuttle waves move at least 4.92 GB through the FP32
`(max, sum-exp, weighted-value)` state lifecycle and reread about 0.91 GB more Q
data than a resident query-major schedule. At 2.5--3.35 TB/s, those bytes imply
about 1.74--2.33 ms, already enough to explain the measured gap. Relation
metadata and launch latency are secondary.

This comparison favors Seer's timed kernel because its 8-to-32-head K/V
expansion is outside the timed region; Shuttle uses native GQA indexing. It
still establishes the correct physical lesson: KV-major ordering without real
shared KV staging does not compensate for spilling online state at every edge.

The next iteration followed this recommendation: it used a non-monotone
relation and generated shared-memory KV staging. That experiment established
legal bounded reuse and deterministic state updates, but its CUDA-core body
measured 107.879105 ms. Query-major remains the selected implementation.

## StatefulScan generality checkpoint

Shuttle now represents ordered matrix-state programs with one generic
`StatefulScan`: stable logical axes, typed persistent state, Map/Contract/Fold
body primitives, an explicit finite-precision contract, and an optional chunk
algebra. Scalar-decay Gated DeltaNet and per-key-channel Kimi Delta Attention
both use this record. Neither adds an architecture-specific semantic node.

For both programs, a token update is an affine state transform
`S' = P S + H`. Exact full summaries compose as
`(P2 P1, P2 H1 + H2)`, and prefix summaries emit every token result. Independent
NumPy recurrent and exact-affine chunk executors agree for nonzero state, tail
chunks, multiple chunk sizes, and distinct decay regimes. KDA changes the
transition from scalar decay to a noncommuting diagonal-plus-rank-one form; it
requires richer physical factors but not a new semantic primitive. The compact
factored representation is not closed under unrestricted tree composition, so
the legal physical form retains an ordered inter-chunk state scan.

The first H100 backend experiment uses the Qwen3-Next core shape: BF16 Q/K/V,
FP32 `[B,32,128,128]` state, 16 Q/K heads, 32 value heads, dimensions 128, and
chunk size 64. The backend is pinned FLA revision
`9c8e42e762fce087c27b673af4922795d9edb85e`.

| Sequence length | FLA recurrent | FLA chunkwise | Measured winner |
|---:|---:|---:|---|
| 64 | 0.084960 ms | 0.515104 ms | recurrent |
| 256 | 0.321792 ms | 0.532176 ms | recurrent |
| 2048 | 3.940768 ms | 0.510624 ms | chunkwise |
| 8192 | not measured | 0.703536 ms | chunkwise |

Every timing record retains 50 CUDA-event samples. One-token recurrent medians
are 0.073168, 0.070048, and 0.073728 ms at batches 1, 4, and 16. The matched
measurements establish a genuine execution-form crossover between lengths 256
and 2048 rather than two disjoint decode/prefill demonstrations.

At length 64, both forms are finite and bitwise deterministic across repeats.
Against Shuttle's independent source-ordered FP32 recurrence, recurrent maximum
output/final-state absolute errors are `2.441e-4` and `5.364e-7`; chunkwise
errors are `4.427e-4` and `5.543e-3`. The larger state deviation confirms that
the chunk form should carry `bounded_reassociation`, not `source_ordered`, as
its numerical contract.

FlashQLA revision `050c6bbee9e03efbbfe41063fe4e33742c4a87cb`
installed and passed its API-signature test, but its TileLang kernel JIT could
not use the holder's split CUDA package set: `crt/host_config.h` was absent and
there was no system toolkit. The exact failure is preserved without changing
the pin. The successful FLA results used one H100 80GB HBM3, driver 595.71.05,
Torch 2.8.0+cu128, CUDA runtime 12.8, Triton 3.4.0, and a 700 W power limit.
Application clocks were unpinned.

### Generated ordered factored chunks

Shuttle now derives the ordered chunk form directly from the recovered affine
factors. A masked triangular solve produces diagonal, low-rank, additive,
transformed-read, and local-output factors for each chunk. The GPU skeleton
then applies these factors in source chunk order while retaining an FP32 state
value block. This path contains no FLA, FlashQLA, GDN, or KDA kernel call and
uses the explicit `bounded_reassociation` numerical contract.

At B1, T2048, H32, K=V=128, scalar rank one, the selected C16/BV32 candidate
measures 0.665568 ms preparation, 0.340032 ms execution, and 0.984496 ms
combined. The complete path is 1.928x the pinned FLA chunk oracle and therefore
does not meet the 1.2x target. The isolated ordered execution is 0.666x the
oracle. The 84,410,368-byte summary materialization and its preparation are the
specific performance deficit.

The primary output and state repeat bitwise, with maximum absolute errors of
`4.883e-4` and `2.840e-4` against Shuttle's generated source-ordered recurrent
skeleton. A per-key-diagonal, rank-two mutation also executes deterministically.
All repeated samples and candidate ablations are stored in
`benchmarks/artifacts/stateful_scan_generated_chunk_h100`.

### Matched generated affine chunk pipeline

The next implementation replaces the C16 materialization-heavy path with a
generic chunk-64 affine pipeline and a four-by-four block triangular inverse
over 16-wide subblocks. It still derives its factors from the ordinary JAX
`stablehlo.while` recurrence and contains no FLA/GDN/KDA execution call.

The first comparison to the historical 0.510624-ms FLA record was not accepted
under the updated benchmark-boundary policy. A same-process audit feeds
identical BF16 Q/K/V and FP32 log-decay, beta, and initial state to both
implementations, disables Q/K normalization, sets query scale to one, and
alternates launch order over 50 samples.

| Implementation | Median | Minimum | Maximum |
|---|---:|---:|---:|
| Generated Shuttle | 0.466752 ms | 0.457216 ms | 0.471424 ms |
| Pinned FLA `9c8e42e` | 0.420528 ms | 0.395712 ms | 0.459552 ms |

The ratio is 1.1099 times, within the 1.2-times target. Maximum/mean absolute
output error is `4.8828125e-4`/`5.270477e-5`; final-state error is
`3.154259e-4`/`4.448347e-5`. Both generated outputs repeat bitwise, as do all
scalar/per-key and rank-one/rank-two mutation cases.

Two independent confirmation captures then counterbalanced every warmup and
measured pair and reversed the initial implementation order. Pooled across 100
samples per implementation, Shuttle measures 0.465824 ms and FLA measures
0.424304 ms, a 1.097854-times ratio. This is the accepted performance
denominator and passes both the 1.20-times completion target and the reported
1.10-times stretch target. The original single-capture record above remains as
superseded evidence.

Profiling made the last optimization mechanical: the factor transform spent
68.99% of preparation GPU time recomputing the 64-token diagonal prefix for
four K=32 tiles. A K=64 tile reduced preparation from 0.407440 to 0.281424 ms
without changing recovered semantics or mutation behavior. Complete evidence
is under `benchmarks/artifacts/stateful_scan_affine_pipeline_h100_v0`.

The natural StableHLO compiler emits and validates a machine-readable
semantic-erasure report before enumerating recurrent or chunkwise candidates.
`stablehlo.while` and its tensor-expression body lower to generic `Scan`,
`Map`, and `Contract`; scheduling keys contain only structural properties.
Named or stale keys are rejected in tests, and the exact report for the
measured artifact reproduces from its stored StableHLO. Together with the
matched performance, mutation, correctness, and determinism evidence, this
closes the current StatefulScan core acceptance row.

This oracle checkpoint validates semantic recovery, recurrent/chunk
equivalence, numerical-policy tracking, and the existence of a shape-dependent
performance crossover. FLA and FlashQLA remain oracle-only. Their measurements
are under `benchmarks/artifacts/stateful_scan_h100_v0`.

### Generated affine recurrent skeleton

Shuttle now linearizes generic tensor expressions with respect to prior state,
rejects nonlinear state dependence, and classifies diagonal and
diagonal-plus-bounded-rank transitions. The analyzer is not a GDN/KDA pattern:
the focused mutation suite varies scalar versus per-key diagonal decay, gate
expressions `exp`, `sigmoid`, and clamped `softplus`, update ranks 1/2/4, and a
post-update diagonal transform while retaining one recovered factor family.

That recovery instantiates a generic Triton recurrent skeleton with BF16
factors/output and FP32 state. The H100 environment contained neither FLA nor
FlashQLA. At `B1,T64,H32,K=V=128`, 50-sample results are:

| Recovered form | Physical choice | Median |
|---|---:|---:|
| Scalar diagonal, rank 1 | `block_v=8` | 0.157120 ms |
| Scalar diagonal, rank 1 | `block_v=16` | 0.149424 ms |
| Scalar diagonal, rank 1 | `block_v=32` | 0.138544 ms |
| Per-key diagonal, rank 1 | `block_v=32` | 0.138000 ms |
| Scalar diagonal, simultaneous rank 2 | `block_v=32` | 0.183376 ms |

All cases are finite and bitwise deterministic. Maximum BF16 output error is
`2.441e-4`; maximum FP32 final-state error is `1.863e-8`. The rank-two case
computes every residual against the same decayed state and applies the summed
correction, rather than silently changing the recurrence to sequential rank-one
updates.

This was the first clean StatefulScan kernel-synthesis result: the executable
path contains no complete architecture kernel and survives nearby recurrence
mutations. The later matched chunk-pipeline result above connects the natural
`stablehlo.while` importer and generated ordered chunk path. Producer maps such
as Q/K normalization and gate formation remain outside the frozen core
boundary. Generic exact factored chunk summaries match the recurrent executor
for scalar/per-key diagonals, rank 1/3, tail chunks, and multiple chunk sizes.
Raw generated-kernel distributions, deterministic hashes, source hashes,
environment, and checksums are under
`benchmarks/artifacts/stateful_scan_generated_h100`.
