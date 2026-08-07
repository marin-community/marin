# Background Research Brief: Tile-Lifetime Transformer Compiler

Date: 2026-08-05

Effort: medium

Stop rule: stop after the major adjacent compiler systems, the named kernel references, the local GPU integration path, and one adversarial numerical check all converge on a concrete first experiment.

## TL;DR

The proposed compiler experiment is worth running, but the current specification combines semantic recovery, floating-point legality, whole-region planning, H100 kernel construction, runtime integration, and autotuning in one definition of done. Recent systems already cover substantial parts of the broad claim: Nautilus compiles high-level attention into FlashAttention-3-class kernels; Mirage searches across kernel and graph rewrites; Welder plans inter-operator tile reuse; Flashlight compiles attention programs; and RedFuser formalizes cascaded-reduction fusion. The defensible experiment is narrower: recover CODA-style cross-GEMM transformations from ordinary StableHLO, compose them with prevalidated GEMM and attention skeletons, and explain the numerical, layout, and materialization decisions.

The largest specification defect is the use of “exact semantics” for the delayed RMSNorm rewrite. Moving an inverse-RMS row scale through a BF16 GEMM changes the location of BF16 rounding. CODA treats this as a numerical reordering and validates the result against FP32; it is not bitwise StableHLO equivalence. A local BF16 experiment changed every output element, with maximum absolute differences of roughly 0.0063–0.0072 across five seeds. The IR needs an explicit numerical contract before implementing this rewrite.

Marin already has a useful JAX/CuTe integration seam through `cutlass.jax.cutlass_call` and pins CUDA 13, CUTLASS DSL 4.5, and FlashAttention 4 packages. The first backend should use that seam. CODA and ThunderKittens should be performance and semantic oracles while their toolchain constraints are evaluated.

Project decision, 2026-08-05: licensing is not an implementation gate for the research prototype. License review is deferred until code is distributed or promoted beyond the experiment branch.

## Research question

Can a standalone compiler recover cross-operator Transformer rewrites from exported StableHLO and choose a small set of named H100 skeletons, layouts, and materialization boundaries while staying close to the best manually assembled implementation?

The experiment should distinguish two claims:

1. Semantic claim: the compiler can prove or explicitly qualify each rewrite from ordinary tensor semantics.
2. Performance claim: a bounded set of prevalidated skeletons is sufficient to approach the best library or manual implementation for the selected region and shapes.

Generating an arbitrary FlashAttention schedule from a tensor graph is not required to test either claim.

## Current Marin context

Marin already pins the relevant CUDA 13 stack in [`lib/levanter/pyproject.toml`](../../../../lib/levanter/pyproject.toml): JAX 0.10.1, CUTLASS DSL 4.5.x, FlashAttention 4 beta, JAX-Triton 0.3.1, and Triton 3.6.x. The existing Grug attention path imports `cutlass.cute` and `cutlass.jax`, constructs a specialized kernel launcher, and calls it through `cutlass_call` in [`_fa4_cute_backend.py`](../../../../lib/levanter/src/levanter/grug/attention/_fa4_cute_backend.py). It already avoids a `[B, S, S]` materialized mask for dynamic segmented causal attention.

This suggests a smaller integration path than the specification's new allocator, launcher, CUDA Graph runtime, and Python callable. A compiler-generated or instantiated CuTe kernel can first enter JAX through the existing custom-call boundary. Buffer assignment and repeated-execution concerns can stay with JAX/XLA until measurements show that the boundary itself is limiting.

The existing local kernel is an adapted attention implementation with explicit upstream provenance, but it is not automatically the H100 performance oracle for this project. Official FlashAttention, cuDNN SDPA, and the current CUTLASS/FlashAttention packages still need component benchmarks on the target H100.

An earlier Marin GPU kernel experiment is a useful warning. A native GB10 Pallas fused-cross-entropy forward measured 0.10–0.13x the XLA implementation and was less numerically aligned; the hybrid XLA-forward/custom-backward path remained the default. See [`fused_cross_entropy_gpu.md`](../fused_cross_entropy_gpu.md). Structural fusion is not evidence of a performance win. Every custom skeleton needs an explicit fallback to the best measured library or XLA path.

## External prior art

### Direct execution references

[CODA](https://arxiv.org/html/2605.19269) gives the clearest reference for a fixed H100 GEMM mainloop plus composable epilogues. Its delayed RMSNorm construction, pairwise transformations, partial reductions, and Transformer compositions closely match the proposed non-attention skeleton. CODA also states that delayed scaling changes numerical order and evaluates BF16 results against an FP32 reference. The [implementation](https://github.com/HanGuo97/coda-kernels) is a useful executable oracle. Its public repository did not expose an obvious license file during this review, but license review is not an implementation gate for this research branch; preserve exact source provenance for any copied code so the issue is tractable before distribution.

[FlashAttention](https://arxiv.org/abs/2205.14135) supplies the online-softmax algorithm. [FlashAttention-3](https://arxiv.org/html/2407.08608) supplies the Hopper scheduling reference: TMA/WGMMA overlap, warp specialization, and interleaving softmax with matrix operations. The [official repository](https://github.com/Dao-AILab/flash-attention) should be benchmarked directly. [cuDNN SDPA](https://docs.nvidia.com/deeplearning/cudnn/latest/operations/Attention.html) is also a required baseline because it supports causal attention, grouped and multi-query attention, and multiple layouts on NVIDIA GPUs.

[ThunderKittens](https://github.com/HazyResearch/ThunderKittens) is a useful compact implementation and abstraction reference. Supporting it as a second generated backend in the first prototype would multiply toolchain, layout, and tuning work without testing a new compiler hypothesis. Keep it as a component comparison until the CuTe path works.

### Compiler overlap

[Nautilus](https://arxiv.org/html/2604.14825) is the closest attention compiler found. It lowers math-like attention descriptions to FlashAttention-3-class kernels, supports causal and grouped-query configurations, and tunes physical schedules on Hopper and Blackwell through backends including Tawa and TileLang. This removes “automatic recovery of tiled attention” as a clean novelty claim. The proposed project still differs by importing StableHLO and planning cross-attention Transformer regions with CODA-style algebraic rewrites.

[Mirage](https://arxiv.org/html/2405.05751) searches graph and kernel transformations and reports discovery of FlashAttention/FlashDecoding, RMSNorm-plus-matmul, GatedMLP, and QK-normalization-plus-attention patterns. Its probabilistic equivalence verifier applies only to a restricted algebraic fragment, and searches may take minutes to hours. A constrained, explainable skeleton planner can still contribute, but the proposal should compare its transformation coverage and compile cost against Mirage rather than describe graph-to-tiled recovery as unexplored.

[Welder](https://www.usenix.org/conference/osdi23/presentation/shi) models tensor programs as a tile graph and jointly considers inter- and intra-operator data reuse. Its traffic model and tile-level materialization choices are direct prior art for the tile-lifetime plan and cost model.

[Flashlight](https://proceedings.mlsys.org/paper_files/paper/2026/hash/bc52716d13d2d72ea0f335667d86c0f8-Abstract-Conference.html) compiles attention programs into FlashAttention-style kernels from PyTorch-facing descriptions. [RedFuser](https://arxiv.org/abs/2603.10026) formalizes fusion of cascaded reductions such as softmax followed by a contraction. [Neptune](https://arxiv.org/abs/2510.08726) is related reduction-fusion work. These systems should be cited in the semantic-attention and legality sections.

[Tawa](https://arxiv.org/html/2510.14719) starts from an already tiled Triton program and partitions work into producer and consumer warps using asynchronous references. It is relevant to future schedule derivation, but it does not solve StableHLO semantic recovery or cross-GEMM algebraic rewriting.

### Competing normalization placements

Meta and PyTorch's recent [normalization fusion work](https://pytorch.org/blog/towards-free-normalization-fusing-normalization-into-gemm-and-attention-kernels/) demonstrates several placements beyond CODA's global partial-statistic buffer: lazy pre-normalization, redundant prologue work, multi-CTA cluster reductions, and normalization fused with attention. The planner should eventually enumerate these as named alternatives. The first implementation should keep the CODA partial-buffer form and record the omitted alternatives in the plan report.

### Frontend and baseline constraints

[StableHLO](https://openxla.org/stablehlo/spec) defines operation semantics, including conversion and reduction behavior. [`jax.export`](https://docs.jax.dev/en/latest/export/export.html) adds versioned serialization and calling-convention metadata around StableHLO. Frozen fixtures should come from `jax.export`, not only `lower(...).compiler_ir()`, if compatibility across JAX versions is part of acceptance.

[XLA:GPU](https://openxla.org/xla/gpu_architecture) already performs fusion, buffer assignment, layout assignment, library selection, and LLVM/Triton code generation. [Transformer Engine](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/performance_considerations/performance_considerations.html) provides fused normalization and linear-layer components. The baseline matrix should include stock XLA, cuDNN attention, and Transformer Engine components in addition to CODA and official FlashAttention.

## Adversarial checks

### Delayed RMS scaling is not exact StableHLO equivalence

For real-valued matrices,

```text
((u * gamma) * r) @ W = ((u * gamma) @ W) * r
```

when `r` is constant along the GEMM reduction dimension. With BF16 activation storage, the two programs generally compute different values:

```text
bf16((u * gamma) * r) @ bf16(W)
bf16(u * gamma) @ bf16(W), followed by FP32/BF16 row scaling
```

A local JAX/Numpy-style BF16 experiment used `M=32`, `K=N=128`, FP32 accumulation, and five random seeds. Every output element differed. Maximum absolute error ranged from about 0.0063 to 0.0072, mean absolute error from about 0.00127 to 0.00135, and p99 absolute error from about 0.00425 to 0.00451.

The semantic IR should classify transformations as:

- `bitwise_exact`: preserves the declared finite-precision program.
- `algebraically_exact`: equal over reals but reorders rounding.
- `numerically_approved`: accepted under an explicit dtype- and shape-specific error policy.

The delayed RMS rewrite belongs in the second class and needs approval under the third before code generation. Rewrite reports should name the moved rounding point.

### Exported StableHLO needs more normalization than the current list

A local lowering of a small JAX `GEMM -> residual -> RMSNorm -> GEMM` function produced repeated pure subgraphs, explicit broadcast chains, and conversion operations. The second dot also became FP32 when the reference function did not explicitly cast the normalized activation back to BF16. The importer needs value numbering or common-subexpression elimination, broadcast normalization, and precision/cast normalization. Fixture authors must make the intended BF16 boundaries explicit.

### Shape and execution mode are underspecified

The specification lists hidden sizes and sequence lengths but does not define the projection `M` dimension independently. GEMM performance depends on total tokens, batching, padding, and whether the region is prefill, training forward, or decode. Weight prepacking and whether weights are static are also unspecified. A first benchmark should target inference prefill with static BF16 weights and fixed token buckets. Training forward requires saved-value and backward contracts that the current project excludes.

## Evidence map

| Claim | Evidence | Caveat | Confidence | Action |
| --- | --- | --- | --- | --- |
| CODA-style cross-GEMM rewrites are implementable with a fixed GEMM mainloop. | CODA paper and repository. | Published shapes and toolchain may not match Marin's CUDA 13/CUTLASS 4.5 stack. | High | Reproduce one CODA kernel before building the importer. |
| Delayed RMS scaling is not bitwise equivalent to the StableHLO graph. | CODA's numerical discussion plus the local five-seed BF16 check. | Error acceptability is model- and shape-dependent. | High | Add numerical contracts to the IR and tests. |
| Attention schedule generation is no longer an uncontested novelty. | Nautilus, Mirage, Flashlight, RedFuser, and Neptune. | Public implementations and supported frontends vary. | High | Frame attention as recognition and skeleton selection in v0. |
| Marin can integrate CuTe kernels through JAX without a new runtime. | Existing `cutlass.jax.cutlass_call` attention backend. | GEMM epilogues may require additional output/alias support. | High | Reuse the JAX/CuTe boundary for the first executable plan. |
| A fused custom kernel may lose to XLA or a library. | Marin GB10 Pallas measurements; XLA/cuDNN/Transformer Engine capabilities. | Different GPU and operation. | Medium | Make fallback selection a structural plan outcome. |
| Arbitrary layout search is unnecessary for the first test. | Fixed layout contracts in CODA/FA3/CuTe and the bounded-skeleton hypothesis. | May leave performance on the table. | Medium | Select among named backend contracts, then expand from profiler evidence. |

## Recommended first experiment

Use a fixed inference-prefill region containing one Llama block plus the next QKV projection. Freeze two exported StableHLO fixtures: a debug shape and one H100 shape. Use static BF16 weights, FP32 reductions/accumulation where declared, and explicit BF16 cast boundaries.

Implement only these compiler stages:

1. Import the required StableHLO subset with provenance, value numbering, broadcast normalization, and cast normalization.
2. Recover `Linear`, `ResidualAdd`, `RMSNorm`, `RoPE`, `SwiGLU`, and semantic attention.
3. Emit a non-executable structural plan using named CuTe/attention layout contracts.
4. Implement the residual/RMSNorm/GEMM rewrite under an explicit numerical policy and CPU/JAX differential tests.
5. Execute attention through the best measured official or existing JAX/CuTe skeleton.
6. Generate one CODA-style GEMM epilogue through the existing JAX/CuTe call boundary.

The go/no-go result is whether the compiler can recover the required full-region plan and whether one generated cross-GEMM rewrite is within 10% of the corresponding manually assembled component without increasing end-to-end region latency. Failure still identifies whether semantic recovery, rounding, layout conversion, or backend quality is limiting.

Do not implement a standalone runtime, CUDA Graph capture, arbitrary attention code generation, multiple GPU backends, or a learned/analytical cost model in this experiment.

## Ranked experiment queue

1. **Backend reproduction.** Pin exact revisions and reproduce one CODA GEMM epilogue plus official FlashAttention/cuDNN attention on the target H100. Record compiler versions and clocks.
2. **Numerical contract.** Measure delayed RMS scaling against the original BF16 program and an FP32 reference over the debug and primary shapes. Set explicit max/percentile tolerances or disable the rewrite.
3. **Importer fixture audit.** Export the canonical region with `jax.export`, inspect casts and duplicated subgraphs, and make a golden semantic graph. Add negative recognizer fixtures for wrong axes, extra consumers, and non-row scales.
4. **Structural full-region plan.** Recover the expected sequence of QKV GEMM, attention, output GEMM, reductions, MLP GEMMs, and next QKV projection. Count activation-sized and sequence-squared materializations before writing GPU code.
5. **One executable cross-boundary rewrite.** Generate residual/gamma/RMS partials in one GEMM epilogue, reduce the partials, and delay the scale into the next GEMM epilogue through `cutlass_call`.
6. **Layout boundary measurement.** Compare named QKV-to-attention and attention-to-output layouts. Add layout alternatives only when the measured conversion or store cost is material.
7. **Bounded candidate tuning.** Empirically tune the top few tile configurations. Use backend-reported registers/shared memory and measured latency; avoid a detailed analytical occupancy model until there are enough observations to validate it.

## Negative and inconclusive leads

- No prior Marin project was found that implements CODA-style StableHLO recovery or a tile-lifetime IR. The reusable local work is the JAX/CuTe integration and GPU benchmarking experience.
- Tawa does not start from a tensor graph. It cannot replace semantic recovery or the region planner.
- ThunderKittens provides useful primitives, but adding it beside CuTe in v0 does not isolate a compiler hypothesis.
- The public Nautilus paper establishes substantial overlap, but this review did not find a public code repository. It can be a design and performance comparator, not yet a reproducible implementation oracle.
- CODA's public repository did not show an obvious license file. License work is explicitly deferred for this research prototype.
- The local BF16 RMS experiment establishes non-identity, not model-level acceptability. End-to-end error and perplexity remain unmeasured.

## Open decisions

- Is the first region inference prefill or training forward? This brief recommends inference prefill.
- Are weights static and prepacked outside measured latency? This brief recommends yes, with packing cost reported separately.
- Does “exact” mean finite-precision identity or real-algebra equality within a declared error budget? The compiler needs both concepts.
- Which H100 host is the reproducible target, and can its clocks/power policy be fixed?
- CODA license review is deferred until distribution or production promotion.
- Is the 10% oracle gap a stretch target or a binary completion criterion? It should become a stretch target until Milestone 0 reproduces the oracle.

## Source ledger

### Internal

- [`lib/levanter/pyproject.toml`](../../../../lib/levanter/pyproject.toml): pinned JAX/CUDA/CUTLASS/FlashAttention stack.
- [`_fa4_cute_backend.py`](../../../../lib/levanter/src/levanter/grug/attention/_fa4_cute_backend.py): existing JAX/CuTe custom-call boundary and non-quadratic mask path.
- [`_fa4_cute_kernels.py`](../../../../lib/levanter/src/levanter/grug/attention/_fa4_cute_kernels.py): adapted attention code and provenance practice.
- [`fused_cross_entropy_gpu.md`](../fused_cross_entropy_gpu.md): measured custom-kernel regression and fallback lesson.
- Local StableHLO lowering and five-seed BF16 delayed-RMS experiments performed during this review; commands were read-only and produced no repository artifacts.

### External

- [CODA paper](https://arxiv.org/html/2605.19269) and [implementation](https://github.com/HanGuo97/coda-kernels).
- [FlashAttention](https://arxiv.org/abs/2205.14135), [FlashAttention-3](https://arxiv.org/html/2407.08608), and the [official implementation](https://github.com/Dao-AILab/flash-attention).
- [ThunderKittens](https://github.com/HazyResearch/ThunderKittens).
- [CUTLASS documentation](https://docs.nvidia.com/cutlass/latest/) and [implementation](https://github.com/NVIDIA/cutlass).
- [Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/).
- [StableHLO specification](https://openxla.org/stablehlo/spec) and [`jax.export`](https://docs.jax.dev/en/latest/export/export.html).
- [XLA:GPU architecture](https://openxla.org/xla/gpu_architecture).
- [Tawa](https://arxiv.org/html/2510.14719).
- [Nautilus](https://arxiv.org/html/2604.14825).
- [Mirage](https://arxiv.org/html/2405.05751).
- [Welder](https://www.usenix.org/conference/osdi23/presentation/shi).
- [Flashlight](https://proceedings.mlsys.org/paper_files/paper/2026/hash/bc52716d13d2d72ea0f335667d86c0f8-Abstract-Conference.html).
- [RedFuser](https://arxiv.org/abs/2603.10026) and [Neptune](https://arxiv.org/abs/2510.08726).
- [Towards Free Normalization](https://pytorch.org/blog/towards-free-normalization-fusing-normalization-into-gemm-and-attention-kernels/).
- [cuDNN attention operations](https://docs.nvidia.com/deeplearning/cudnn/latest/operations/Attention.html) and [cuDNN Graph API](https://docs.nvidia.com/deeplearning/cudnn/latest/developer/graph-api.html).
- [Transformer Engine performance considerations](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/performance_considerations/performance_considerations.html).

## Handoff

The evidence is sufficient to choose the first experiment. More literature search is unlikely to change the immediate sequence: establish the backend oracle, define floating-point legality, freeze real JAX exports, recover a structural plan, and execute one cross-GEMM rewrite through the existing CuTe/JAX boundary.
