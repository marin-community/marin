---
topic: tile-lifetime-compiler
description: Compile StableHLO Transformer regions into compositions of expert tiled skeletons.
author: dlwh
---

# Tile-Lifetime Compiler: Task Logbook

## Scope

- Goal: recover cross-skeleton algebraic rewrites, layouts, and materialization boundaries from ordinary Transformer graphs.
- Primary metrics: structurally correct plans, declared numerical legality, activation materializations, and latency versus manually composed H100 skeletons.
- Constraints: H100 forward prototype, bounded expert skeleton family, no dependency on unreleased Nautilus code.
- Coordinating issue/PR: none; work is local until publication is requested.
- Experiment prefix: `TLTC`.

## Current TL;DR

- Dense StableHLO recovery executes through CODA/QuACK and official FA3 on H100. The first generated GB200 receiver-local MoE composition now maps the generic relation into 256-padded MoK grouped GEMMs, generated SwiGLU, and deterministic pre-combine merge. It measures 3.455 ms versus 6.747 ms for the matching Torch sequence. Official DeepEP dispatch/combine, generated shared-expert compute, and rank-maximum overlap measurements are in progress.

## Hypothesis Queue

### Active

- `TLTC-005`: the semantic recognizer can tolerate ordinary JAX broadcast, multiply-order, and projection variations while rejecting near-misses with structured diagnostics. Next test: add permuted and illegal StableHLO fixtures.
- `TLTC-002`: named layout contracts around fixed CuTe and attention skeletons are sufficient for the first full-region planner. Next test: inventory the layouts required by the local JAX/CuTe attention boundary.
- `TLTC-003`: a direct Torch/CuTe or CODA boundary can execute expert Hopper kernels while the JAX adapter is repaired separately. Next test: benchmark official FA3/FA4 and CODA RMS on the reserved H100.
- `TLTC-007`: consumer-prologue RMS scaling should track source BF16 ordering more closely than CODA's delayed epilogue scale, but cost more because the input transform disrupts the TMA/WGMMA mainloop. Next test: compare CODA, a fused prologue experiment, and materialized pre-scaling on primary shapes.

### Blocked

- The checked-in JAX FA4 THD wrapper aborts in `quack.copy_utils.create_ragged_tensor_for_tma` for all tested package tuples. The same physical Hopper kernel works through Torch, so this blocks JAX integration but not component benchmarking.

### Falsified / Dead End

- None.

### Promoted

- `TLTC-001`: the semantic builder, legality rule, materialized fallback, numerical classification, and structured plan pass seven CPU tests. Evidence: entry `2026-08-05 22:35 - TLTC-001 first vertical slice`.
- `TLTC-004`: the frozen StableHLO fixture compiles end-to-end into the delayed-RMS plan. Evidence: entry `2026-08-05 22:55 - TLTC-004 StableHLO semantic recovery`.
- `TLTC-006`: ordinary JAX causal GQA StableHLO compiles into an FA3-style streaming skeleton, and a combined module recovers both CODA and FA3 plans. Evidence: entry `2026-08-06 - TLTC-006 CODA plus FA3 structural baseline`.
- `TLTC-031`: the generic relation drives generated receiver-local packing, MoK W13/SwiGLU/W2, and deterministic pre-combine merge at the exact oracle shape. Evidence: entry `2026-08-06 - TLTC-031 generated receiver-local MoK composition`.

## Decision Log

- 2026-08-05: proceed without waiting for Nautilus source. Treat Nautilus as an architectural reference and keep v0 centered on cross-skeleton region planning.
- 2026-08-05: distinguish bitwise equivalence, real-algebra equivalence with reordered rounding, and numerically approved rewrites.
- 2026-08-05: keep GitHub publication out of scope until explicitly requested.
- 2026-08-05: defer license work for the research prototype at the user's direction.
- 2026-08-05: place the compiler in an isolated `lib/tile_lifetime` workspace package; do not make Levanter depend on it.
- 2026-08-06: model inverse-RMS placement explicitly. Keep CODA's consumer-epilogue scale as the performance-oriented plan and add a consumer-prologue plan for source-like BF16 ordering.
- 2026-08-06: use CODA commit `8fa88065e541f6a5b52fb400d94d4be02f18c543` with QuACK commit `02c7f69881737731173a6a009aeb6f032e449b61` for the first executable RMS oracle. CODA `v0.2` lacks the scale entry point and current main is between QuACK epilogue APIs.
- 2026-08-06: use explicit FP32 round-to-nearest multiplication and addition for deterministic routed merge. FMA remains a measured numerical alternative but is not the selected exact-order path.

## Entry Log

### 2026-08-05 22:00 - TLTC-001 start

- Hypothesis: deterministic legality rules can recover the first cross-GEMM transformation and produce a useful plan before any GPU backend is implemented.
- Commit Hash: uncommitted branch `prototype/tile-lifetime-compiler`.
- Command: pending implementation.
- Config: CPU-only, small synthetic semantic graph, BF16 inputs with FP32 accumulation policy.
- Result: prior-art and local-code research completed; implementation started.
- Interpretation: the first slice should test the compiler's distinguishing region-planning claim and encode the delayed-RMS numerical caveat.
- Next action: implement semantic IR, plan format, recovery pass, and behavior-focused tests.

### 2026-08-05 22:35 - TLTC-001 first vertical slice

- Hypothesis: deterministic legality rules can recover the delayed-RMS region without general graph superoptimization.
- Commit Hash: uncommitted branch `prototype/tile-lifetime-compiler`.
- Command: `uv run --frozen --package marin-tile-lifetime --group test pytest lib/tile_lifetime/tests`.
- Config: CPU, JAX 0.10.1, frozen StableHLO v1.14.1 portable fixture with `M=2`, `K=4`, intermediate `N=3`, and output `P=5`.
- Result: seven tests passed in 0.95 seconds. The importer recovered two `dot_general` operations, reduction dimension 1 with an add reducer, tensor shapes/dtypes, dataflow, and source provenance. The planner emitted GEMM, RMS reduction, and GEMM skeletons; strict numerical policy and illegal graph variants selected a materialized fallback.
- Command: `uv run --frozen --package marin-core --group lint pyrefly check lib/tile_lifetime/src`.
- Result: zero type errors.
- Command: `./infra/pre-commit.py <changed compiler files>`.
- Result: Ruff, Black, license headers, syntax, TOML, whitespace, and conflict checks passed.
- Interpretation: `TLTC-001` is supported for the semantic graph. StableHLO import also works, but the importer-to-semantic-recovery bridge is still missing.
- Negative result: unconstrained `uv lock` rewrote 1,400+ unrelated lines because the repository lock was resolved by a newer uv version. The churn was discarded; the new workspace package has a 27-line lock entry and passes `uv run --frozen`.
- Next action: recover `Linear -> ResidualAdd -> RMSNorm -> Linear` directly from the frozen StableHLO fixture and compile it into the same structured plan.

### 2026-08-05 22:55 - TLTC-004 StableHLO semantic recovery

- Hypothesis: convert and broadcast chains in an ordinary JAX export can be reduced to the semantic residual/RMSNorm/GEMM region with deterministic rules.
- Commit Hash: uncommitted branch `prototype/tile-lifetime-compiler`.
- Command: `uv run --frozen --package marin-tile-lifetime --group test pytest lib/tile_lifetime/tests`.
- Config: CPU, JAX 0.10.1, StableHLO v1.14.1 portable fixture, BF16 dots and residual, FP32 RMS reduction, explicit BF16 conversion before the consumer GEMM.
- Result: eight tests passed in 0.82 seconds. The end-to-end test imports the fixture, validates both matrix contractions, recognizes the additive sum-of-squares reduction, checks the mean divisor and epsilon, traces gamma and inverse-RMS broadcasts, recovers semantic operations with provenance, and selects the three-skeleton delayed-scale plan.
- Command: `uv run --frozen --package marin-core --group lint pyrefly check lib/tile_lifetime/src`.
- Result: zero type errors.
- Interpretation: the first complete compiler path works for one deliberately frozen JAX form. Robustness across equivalent forms is unmeasured.
- Next action: generate equivalent and illegal fixture variants, then add structural diagnostics before expanding to RoPE, SwiGLU, or attention.

### 2026-08-06 - TLTC-006 CODA plus FA3 structural baseline

- Hypothesis: an ordinary JAX attention graph can be recognized by axis roles and dataflow, then lowered to a fixed expert streaming skeleton without materializing score or probability tensors.
- Commit Hash: uncommitted branch `prototype/tile-lifetime-compiler`.
- Config: CPU, JAX 0.10.1, StableHLO v1.14.1; causal GQA with `B=1`, `S=5`, `Hq=6`, `Hkv=2`, and `D=64`; BF16 Q/K/V with FP32 QK/PV accumulation and softmax state.
- Result: the importer handles reshape, transpose, iota, signed comparison, select, maximum, exponential, subtract, and both reduction forms. Recovery proves contiguous GQA head replication, QK/PV contraction axes, exact negative-infinity causal masking, stable softmax dataflow, and absence of external score/probability consumers.
- Result: the selected attention plan records an H100 template with 128x128 Q/KV blocks, three pipeline stages, one producer worker group, two consumer worker groups, and FP32 online maximum/sum/output state. Logical score and probability tensors are `internal_attention_state`, not activation materializations.
- Result: a second frozen StableHLO module contains both the residual/RMSNorm/GEMM graph and the causal-GQA graph. Its plan is `GEMM -> RMS reduction -> GEMM -> streaming attention`, with both rewrite explanations applied.
- Command: `uv run --frozen --package marin-tile-lifetime --group test pytest lib/tile_lifetime/tests`.
- Result: thirteen tests passed locally.
- Interpretation: the StableHLO-to-program structural baseline is complete. It demonstrates semantic recovery and skeleton selection, not CUDA generation, layout co-planning between adjacent skeletons, or measured H100 performance.
- Next action: connect QKV/RoPE projection output to the attention skeleton through named layout contracts, then reproduce one CODA/CuTe and one FA3 kernel on H100 before expanding recognition breadth.

### 2026-08-06 - TLTC-007 H100 bring-up and RMS alternatives

- Hypothesis: physical Hopper kernels are usable independently of the current JAX integration, and the planner can expose both RMS scale placements before backend work is complete.
- Commit Hash: uncommitted branch `prototype/tile-lifetime-compiler`; clean H100 control worktree at Marin `121b71af64`.
- Config: Iris job `/dlwh/dev-gpu-dlwh-tile-lifetime-h100`, one node with eight NVIDIA H100 80GB HBM3 GPUs. Experiments use GPU 0.
- Source revisions: FlashAttention `3fa810570e17bb4354155bdb71d826eca6079208`; CODA main inspected at `8c7c4d5f109e03fc2444f5854dad40d6f227c605`; executable CODA candidate `8fa88065e541f6a5b52fb400d94d4be02f18c543`; QuACK `02c7f69881737731173a6a009aeb6f032e449b61`.
- Result: the checked-in `gpu_fa4_thd_attention` path aborts natively in `cutlass.cute.tensor.make_tensor` through `quack.copy_utils.create_ragged_tensor_for_tma`. JAX 0.11.0 with FlashAttention 4 beta 16, 19, and 25 all fail at the same boundary.
- Result: `flash_attn.cute.flash_attn_func` succeeds on the same H100 for causal BF16 GQA with `B=1`, `S=256`, `Hq=32`, `Hkv=8`, and `D=128`. This isolates the failure to the JAX ragged-TMA adapter rather than the physical Hopper kernel.
- Result: the planner now represents `gemm_prologue` attachments and emits `scale_rms_in_consumer_gemm_prologue` separately from `delay_rms_row_scale_through_gemm`. A seeded CPU check has mean absolute error 0.00642 for prologue scaling versus 0.00910 for epilogue scaling against the BF16 source-ordered result.
- Command: `uv run --frozen --package marin-tile-lifetime --group test pytest -q lib/tile_lifetime/tests`.
- Result: fifteen tests passed. Ruff and Pyrefly report no errors.
- Negative result: the official FA3 source extension has no Torch 2.11/Python 3.12 prebuilt wheel. The runtime image lacks a system CUDA toolkit; a minimal source build needs FA3's CUDA 12.6 compiler, CUDA 12.8 runtime/CCCL headers, and GCC 13 rather than the image's GCC 14.
- Negative result: CODA `v0.2` does not expose the RMS scale epilogue used in the paper, while current main mixes pre- and post-reorganization QuACK imports. Commit `8fa8806` imports `gemm_rmsnorm` successfully with the older QuACK commit and is the current oracle candidate.
- Next action: finish the minimal official FA3 build with GCC 13, run the attention benchmark, then compile and time CODA `gemm_rmsnorm` against materialized BF16 pre-scaling.

### 2026-08-06 - TLTC-008 executable H100 component oracles

- Hypothesis: official expert kernels provide a strong enough performance floor that the compiler can focus on recovering skeleton choice and region boundaries instead of regenerating their mainloops.
- Commit Hash: uncommitted branch `prototype/tile-lifetime-compiler`; clean H100 control worktree at Marin `121b71af64`.
- Config: NVIDIA H100 80GB HBM3, BF16 inputs and outputs, FP32 accumulation. Attention uses `B=1`, `Hq=32`, `Hkv=8`, `D=128`, causal GQA. CODA uses `M=2048`, `K=4096`, `N=6144`.
- Source revisions: official FlashAttention `3fa810570e17bb4354155bdb71d826eca6079208`; executable CODA `8fa88065e541f6a5b52fb400d94d4be02f18c543`; QuACK `02c7f69881737731173a6a009aeb6f032e449b61`; CUTLASS DSL 4.6.0.
- Build result: the official FA3 Hopper extension built from source for BF16, head dimension 128, forward-only fixed-length and packed-GQA kernels. The Debian 13/glibc 2.41 host required GCC 13, CUDA 12.8 compiler/CCCL headers, and NVIDIA's documented `sinpi`/`cospi` exception-specification workaround in CUDA's `crt/math_functions.h`.
- Command: `CUDA_VISIBLE_DEVICES=0 /app/.venv/bin/python /tmp/h100_attention_torch.py --sequences 2048 4096 --warmups 5 --repeats 7 --iterations 20`.
- Result at sequence 2048: official FA3 median 0.0672 ms (511.1 causal TFLOP/s), FA4 CuTe SM90 0.0708 ms (485.5 TFLOP/s), Torch SDPA flash backend 0.1631 ms (210.6 TFLOP/s). FA3 versus FA4 maximum absolute difference was 0.000977.
- Result at sequence 4096: official FA3 median 0.2100 ms (654.4 causal TFLOP/s), FA4 CuTe SM90 0.2275 ms (604.2 TFLOP/s), Torch SDPA flash backend 0.5137 ms (267.5 TFLOP/s). FA3 versus FA4 maximum absolute difference was 0.000977.
- Command: `/tmp/coda-venv/bin/python /tmp/h100_rms_scale.py --m 2048 --k 4096 --n 6144 --warmups 3 --repeats 5 --iterations 10`.
- Result: raw Torch GEMM 0.1299 ms (793.5 TFLOP/s); CODA delayed consumer-epilogue scale 0.1617 ms (637.6 TFLOP/s); materialized BF16 pre-scale plus GEMM 0.2380 ms (433.1 TFLOP/s); Torch post-scale 0.2668 ms (386.3 TFLOP/s). Relative to the strict BF16 source ordering, CODA had maximum absolute error 2.0 and mean absolute error 0.083834, while the materialized pre-scale was exact.
- Interpretation: FA3 is 2.4x faster than stock Torch SDPA at sequence 2048 and 2.45x at sequence 4096 for this configuration. CODA's delayed scale is about 24.5% slower than raw GEMM but about 32% faster than the strict materialized pre-scale path. The requested fused consumer-prologue variant is not yet represented by the materialized baseline and remains the critical implementation experiment.
- Next action: reproduce FA3's actual physical template fields in the selected plan, benchmark the remaining projection shapes, and implement row scaling after the consumer TMA wait but before WGMMA.

### 2026-08-06 - TLTC-009 fused consumer-prologue RMS scaling

- Hypothesis: applying inverse-RMS to the BF16 A fragment after TMA and before WGMMA can preserve StableHLO source ordering without materializing the normalized activation, and a wider cooperative tile can hide most of the transform cost.
- Commit Hash: uncommitted branch `prototype/tile-lifetime-compiler`; QuACK transform prototype based on `84ef91df9bec87c7e4938517234fafb07ef844dd` plus `backends/h100/quack_fp32_row_scale.patch`.
- Config: H100 80GB HBM3; Torch 2.13.0+cu130; CUTLASS DSL 4.6.0; BF16 A/B/output, FP32 accumulator and inverse-RMS; `M=2048`, `K=4096`, `N=6144`.
- Implementation: QuACK's SM90 RS mainloop loads the A fragment from shared memory into registers, multiplies it by an auxiliary row-scale strip, converts the transformed fragment to BF16, and feeds it to WGMMA. The first implementation repeats one scale per row for each K tile. The FP32 extension keeps the scale and multiply in FP32 before the BF16 WGMMA-input conversion.
- Negative result: the existing BF16 strip at tile 128x128 ran in 0.1900 ms but quantized inverse-RMS to BF16. It had maximum/mean source-order error 2.0/0.132473, worse than CODA's delayed-scale mean error 0.083834.
- Tile sweep result for the FP32 strip at `M=2048,K=4096,N=6144`: 128x128 cooperative 0.1941 ms; 128x208 ping-pong 0.2756 ms; 128x256 cooperative 0.1421 ms; 256x128 cooperative 0.1642 ms; 128x128 cluster-N=2 0.1833 ms. Tile 192x128 is illegal because 2048 is not divisible by 192 in the current host transform.
- Command: `PYTHONPATH=/tmp/quack-prologue CUDA_VISIBLE_DEVICES=0 /tmp/coda-venv/bin/python /tmp/h100_rms_prologue.py --shapes 2048x4096x6144 --tile-m 128 --tile-n 256 --warmups 5 --repeats 7 --iterations 20`.
- Tuned result: FP32 consumer prologue 0.1410 ms (731.1 TFLOP/s), BF16 consumer prologue 0.1417 ms, raw Torch GEMM 0.1300 ms, and materialized source-ordered pre-scale plus GEMM 0.2204 ms. The FP32 prologue is bitwise equal to the materialized source-order output and is 36% faster than materialization.
- CODA comparison: a same-host rerun of CODA's consumer epilogue ranged from 0.1496 to 0.1636 ms across runs. Its maximum/mean error versus source order was 2.0/0.083834. The tuned FP32 prologue was faster in these runs while eliminating the source-order error.
- Numerical interpretation: CODA's delayed scale is closer to ideal FP32 real algebra (mean error 0.071284) than source-ordered BF16 pre-scaling (0.115098), because it avoids an extra BF16 input rounding. The FP32 prologue exactly preserves the actual source program when that program converts the normalized activation to BF16 before GEMM. Therefore `numerically less problematic` means semantic fidelity here, not universally lower error against an unrounded real-number expression.
- Shape result at `M=4096,K=4096,N=6144`: tuned FP32 prologue 0.2786 ms versus materialized 0.4285 ms, both bitwise equal.
- Shape result at `M=2048,K=4096,N=28672`: tuned FP32 prologue 0.7147 ms versus materialized 0.6918 ms. For this compute-dominated gate/up shape, the current repeated-strip prologue loses by 3.3%; CODA's delayed epilogue previously measured 0.7032 ms.
- Interpretation: the requested consumer-prologue structure is recovered and executable. It is a strong choice for the primary `N=6144` projection but not universally best. The planner needs shape-specific alternatives, and a true K-invariant row-vector prologue may remove repeated scale loads for the wide gate/up case.
- Next action: replace the repeated K-tile strip with a once-per-output-tile row-vector load, then encode both measured alternatives in the cost model while expanding the recovered region to QKV/RoPE and SwiGLU.

### 2026-08-06 - TLTC-010 stock JAX attention baseline

- Hypothesis: the stock JAX/XLA tensor-algebra path provides a useful framework baseline even when cuDNN cannot build a GQA execution plan.
- Config: JAX 0.11.0, H100 80GB, causal BF16 GQA, `B=1`, `Hq=32`, `Hkv=8`, `D=128`.
- Command: `CUDA_VISIBLE_DEVICES=0 /app/.venv/bin/python /tmp/h100_attention.py --sequence 2048 --warmups 3 --repeats 5 --iterations 10`.
- Result: JAX XLA median 0.8824 ms with 1.04 seconds compilation. Official FA3 previously measured 0.0672 ms, a 13.1x speedup.
- Command: same benchmark with sequence 4096.
- Result: JAX XLA median 3.0960 ms with 1.09 seconds compilation. Official FA3 previously measured 0.2100 ms, a 14.7x speedup.
- Negative result: `jax.nn.dot_product_attention(..., implementation="cudnn")` fails during compilation with `No valid execution plans built` for this GQA shape. Torch's cuDNN SDPA path failed similarly, so the viable Torch comparison explicitly selects its flash backend.
- Interpretation: the StableHLO attention recovery is materially useful relative to ordinary XLA tensor algebra; the official FA3 skeleton is also about 2.4x faster than Torch's already fused flash fallback.
- Next action: connect the QKV/RoPE producer layout to the measured FA3 input contract.

### 2026-08-06 - TLTC-011 wide-projection prologue tuning

- Hypothesis: multicast clustering can recover the prologue's wide gate/up regression by reducing redundant A-side work across output-N tiles.
- Config: H100 80GB; `M=2048`, `K=4096`, `N=28672`; FP32 row scale; cooperative QuACK RS mainloop.
- Sweep: 64x256x64 cluster 1x1, 64x512x64, 128x512x64, 128x256x64 cluster-N=2, 256x128x64 cluster-N=2, and 128x256x128.
- Result: 128x256x64 with cluster-N=2 won at 0.6504 ms (739.6 TFLOP/s), versus 0.6859 ms for materialized BF16 pre-scaling and the earlier 0.7032 ms CODA consumer-epilogue result. It was bitwise equal to the materialized consumer-boundary ordering.
- Negative result: N=512 configurations pass the QuACK constructor but fail CUTLASS WGMMA legalization because the selected MMA op requires N <= 256. The planner must reject them before compilation.
- Negative result: tile K=128 produced maximum source-order error above 403, indicating that the current A-transform fragment mapping is only valid for K=64 despite accepting K=128. K=128 is now outside the legal transform family.
- Result: cluster-N=2 did not change the `M=4096,K=4096,N=6144` result materially (0.2786 ms). The selected heuristic uses cluster-N=2 only for output widths at least 16384.
- Interpretation: the source-ordered FP32 prologue now beats materialization and the recorded CODA epilogue at all three measured projection shapes. It still requires shape-specific tile/cluster selection and stricter backend legality checks.
- Next action: recover the connected QKV/RoPE/FA3 and gate/up/SwiGLU/down structure, then benchmark region boundaries.

### 2026-08-06 - TLTC-012 QKV/RoPE and SwiGLU structural recovery

- Hypothesis: producer epilogue attachments and explicit boundary layouts are sufficient to connect the measured CODA and FA3 skeletons without generic CUDA synthesis.
- Result: a semantic QKV projection followed by adjacent-pair RoPE and exact causal GQA lowers to one CODA QKV GEMM plus the selected official FA3 skeleton. The QKV epilogue partitions accumulator regions, rotates Q/K, and stores separate contiguous BSHD Q/K/V. Unrotated Q/K are epilogue-only; there is no layout-conversion skeleton or sequence-squared materialization.
- Result: separate gate/up projections or one adjacent-pair combined projection lower to one CODA GEMM with pairwise SwiGLU in its epilogue, followed by the down-projection GEMM. Expanded gate/up values are epilogue-only; only the dimension-reduced activation is stored.
- Numerical policy: both attachments are algebraically exact over real arithmetic but change rounding by consuming FP32 accumulators before a BF16 store. Bitwise policy retains the materialized source boundaries.
- Command: `uv run --frozen --package marin-tile-lifetime --group test pytest -q lib/tile_lifetime/tests`.
- Result: 25 tests passed; Pyrefly reported zero errors; scoped pre-commit passed.
- Interpretation: the compiler now recovers all major individual dense-block structures. The remaining structural gap is composing them across residual/RMS boundaries in one connected graph, followed by StableHLO recovery of that connected graph.
- Next action: emit the full connected dense-region plan and compare its boundary layouts/materializations with the hand-composed CODA-plus-FA3 oracle.

### 2026-08-06 - TLTC-013 executable SwiGLU oracle

- Hypothesis: forward-only dead-output elimination is necessary to realize the planner's no-expanded-preactivation claim.
- Config: H100 80GB; `M=2048`, `K=4096`, combined gate/up `N=28672`; BF16 input/output, FP32 accumulation, weight standard deviation scaled by `K^-0.5`.
- Command: `CUDA_VISIBLE_DEVICES=0 /tmp/coda-venv/bin/python /tmp/h100_swiglu.py --shapes 2048x4096x28672 --warmups 3 --repeats 5 --iterations 5`.
- Result: Torch materialized GEMM plus SwiGLU 0.7775 ms; CODA SwiGLU with expanded preactivation saved 0.7501 ms; QuACK SwiGLU with `store_preact=False` 0.6185 ms.
- Numerical result: both fused variants have maximum/mean difference 0.0625/0.000786 versus the BF16-materialized source and 0.031226/0.000448 versus ideal FP32. The materialized source has mean ideal-FP32 error 0.000891.
- Interpretation: the QuACK dead-output form is 20.4% faster than the materialized path and is the correct forward-only physical skeleton. CODA's public wrapper saves the full expanded preactivation for backward and therefore cannot directly implement the plan's materialization contract.
- Next action: mark the selected gate/up skeleton as `quack_sm90_swiglu_dead_preact` and integrate it into the connected dense-region plan.

### 2026-08-06 - TLTC-014 composed RMS and SwiGLU alternatives

- Hypothesis: composing RMS placement with dead-preactivation pairwise SwiGLU will expose the true forward-only tradeoff, without CODA's training-only preactivation store confounding the comparison.
- Config: H100 80GB; `M=2048`, `K=4096`, combined gate/up `N=28672`; 128x256x64 cooperative tile, cluster-N=2; BF16 A/B/output, FP32 inverse RMS and accumulation.
- Result: FP32 consumer-prologue scaling plus pairwise SwiGLU measured 0.6509 ms (739.1 TFLOP/s), CODA-style delayed consumer-epilogue scaling plus the same dead-output SwiGLU measured 0.6430 ms (748.1 TFLOP/s), and source-ordered materialized BF16 scaling plus GEMM/SwiGLU measured 0.8723 ms.
- Numerical result: prologue and delayed variants had mean absolute difference 0.000931 and 0.001440, respectively, versus the materialized BF16 source. The prologue is exact at the consumer input boundary; fused SwiGLU still consumes FP32 GEMM accumulators before the source's BF16 preactivation store.
- Interpretation: dead-output elimination matters more than RMS placement at this shape. Delayed scaling wins by 1.2%; the prologue wins on source-order fidelity and is 25.4% faster than materialization. Both should remain explicit planner choices.
- Next action: measure the same placement alternatives when the consumer epilogue is packed QKV/RoPE.

### 2026-08-06 - TLTC-015 connected dense semantic plan

- Hypothesis: the individually recovered expert boundaries can compose without introducing a generic transform skeleton or an accidental activation-sized materialization.
- Result: one connected semantic graph from QKV/RoPE through the following QKV/RoPE boundary lowers to eight skeletons: QKV/RoPE, FA3 attention, output projection with residual/RMS partials, RMS reducer, gate/up with RMS placement and pairwise SwiGLU, down projection with residual/RMS partials, RMS reducer, and next QKV/RoPE.
- Result: residual, RoPE, SwiGLU, and normalization work is attached to the relevant tile lifetimes; logical flatten/BSH views are aliases. No transform skeleton and no sequence-squared score/probability materialization remains.
- Result: both consumer-prologue and delayed-epilogue RMS policies produce structurally valid dense plans. The prologue plan selects measured 128x256x64 QuACK configurations, using cluster-N=2 only for wide gate/up outputs.
- Command: `uv run --frozen --package marin-tile-lifetime --group test pytest -q lib/tile_lifetime/tests`.
- Result: 28 tests passed before adding the placement-policy test; Pyrefly and scoped pre-commit passed.
- Next action: recover the same connected graph from one frozen StableHLO export.

### 2026-08-06 - TLTC-016 composed RMS and packed QKV/RoPE alternatives

- Hypothesis: the FP32 A-fragment transform can compose with QuACK's packed QKV/RoPE epilogue without harming its physical mainloop or attention-boundary layout.
- Config: H100 80GB; `M=2048`, `K=4096`, packed QKV `N=6144`; `Hq=32`, `Hkv=8`, `D=128`; 128x256x64 tile, cluster-N=1.
- Result: the longer cluster-N=1 run measured 0.1476 ms for FP32 prologue scaling, 0.1466 ms for CODA-style delayed scaling, and 0.2263 ms for materialized scaling. Cluster-N=2 improved these to 0.1467 ms (702.5 TFLOP/s), 0.1357 ms (759.6 TFLOP/s), and 0.2169 ms, respectively.
- Numerical result: the prologue was bitwise equal to the materialized kernel. Delayed scaling had mean absolute difference 0.001311 versus that boundary. Against an independent FP64-trigonometric source reference, prologue/materialized mean error was 0.000971 and delayed mean error was 0.001727.
- Interpretation: cluster-N=2 is performance-neutral for the prologue but improves delayed scaling by 7.4%. The prologue removes 32.4% of materialized-path latency and better preserves source ordering; delayed scaling is another 7.5% faster. The planner selects cluster-N=2 for primary-width packed QKV.
- Next action: carry this exact backend contract into connected StableHLO recovery and the region runtime.

### 2026-08-06 - TLTC-017 connected StableHLO dense recovery

- Hypothesis: a parameterized ordinary JAX dense region can be frozen compactly and recovered into the same eight-skeleton plan without frontend annotations or embedded weight constants.
- Config: debug `B=1`, `S=4`, `H=128`, intermediate 256, `Hq=2`, `Hkv=1`, `D=64`; StableHLO portable artifact version 1.14.1.
- Result: the exporter keeps weights, RMS gammas, and RoPE tables as ten function inputs; the base64 fixture is about 9.6 KB. The imported graph contains 184 operations.
- Result: recovery validates FP32 dot accumulation followed by BF16 conversion, contiguous QKV slicing, exact adjacent-pair RoPE, causal GQA softmax, both RMS reductions, exact pairwise SwiGLU, both residual connections, and the next QKV outputs. Every source operation is accounted for and semantic operations retain source provenance.
- Result: the public StableHLO entry point produces the same eight-skeleton dense plan, with no transform skeleton and no sequence-squared materialization. Both RMS placement policies are exposed through the public entry point.
- Command: `uv run --frozen --package marin-tile-lifetime --group test pytest -q lib/tile_lifetime/tests`.
- Result: 33 tests passed for the initial path; Pyrefly and scoped pre-commit passed.
- Next action: validate the physical QKV output layout directly against official FA3.

### 2026-08-06 - TLTC-018 packed QKV segment views into FA3

- Hypothesis: official FA3 requires only a contiguous head dimension and can consume Q/K/V segment views from one packed QKV allocation without separate copies.
- Config: H100 80GB; `B=1`, `S=2048`, `H=4096`, packed QKV width 6144, `Hq=32`, `Hkv=8`, `D=128`; QuACK QKV/RoPE cluster-N=2 followed by official FA3.
- Result: packed Q/K/V strides were `(12582912, 6144, 128, 1)`. FA3 accepted them and produced a bitwise-identical result to explicitly contiguous Q/K/V.
- Result: packed QKV/RoPE plus FA3 measured 0.1944 ms; adding three explicit contiguous copies increased the boundary to 0.2367 ms.
- Interpretation: the direct boundary saves 17.9% and requires no layout-conversion skeleton. The plan's layout contract is now `fa3_bshd_last_dimension_contiguous`, which accurately includes packed segment views.
- Next action: assemble the output projection, RMS partial reducer, gate/up, down, and next QKV into the same executable runtime.

### 2026-08-06 - TLTC-019 complete executable dense oracle

- Hypothesis: the eight recovered skeletons can coexist in one runtime and retain their isolated performance when chained across real data dependencies.
- Runtime: Torch 2.13.0+cu130 and CUTLASS DSL 4.6.0 from the CODA environment, QuACK source plus the FP32 A-transform patch, and the official FA3 ABI extension copied from the Torch 2.11 build environment. The FA3 extension registered and executed successfully under Torch 2.13.
- Config: primary Llama shape, `B=1`, `H=4096`, `I=14336`, `Hq=32`, `Hkv=8`, `D=128`; sequence lengths 2048 and 4096.
- Result at sequence 2048: consumer prologue 1.4800 ms (minimum 1.4184), delayed epilogue 1.4561 ms (minimum 1.3859), materialized Torch 1.9614 ms, and stock JAX/XLA 2.5010 ms. The delayed plan is 25.8% faster than materialized Torch and 41.8% faster than JAX.
- Result at sequence 4096: consumer prologue 3.0563 ms (minimum 2.9192), delayed epilogue 3.0080 ms (minimum 2.8642), materialized Torch 4.0236 ms, and stock JAX/XLA 6.5257 ms. The delayed plan is 25.2% faster than materialized Torch and 53.9% faster than JAX.
- Numerical result at sequence 2048: prologue versus delayed mean absolute difference was 0.001599 at the second residual and 0.002431 at the following QKV/RoPE output. Versus the materialized Torch source, final-QKV mean differences were 0.003745 for prologue and 0.003363 for delayed.
- Interpretation: prologue placement is exact at its local BF16 consumer boundary, but other fused rounding changes mean it does not necessarily minimize final-region error. Placement should remain a declared local numerical policy backed by end-to-end differential tests.
- Measurement caveat: clocks could not be pinned; the container lacked permission for `nvidia-smi -lgc`. Interleaved variant timing reduced drift, but median/minimum spreads remain visible.
- Next action: make a runtime consume `RegionPlan` directly, profile the two residual/RMS GEMMs and reducers, and tune their tile/cluster choices before carrying the architecture into Mixture-of-Kittens MoE.

### 2026-08-06 - TLTC-020 residual/RMS and prologue phase profiling

- Hypothesis: the remaining consumer-prologue latency is primarily the repeated FP32 inverse-RMS strip preparation, and the residual/RMS producers may prefer a different physical tile from the consumer projections.
- Config: H100 80GB HBM3; primary `B=1,S=2048,H=4096,I=14336,Hq=32,Hkv=8,D=128`; interleaved CUDA-event measurements; clocks not pinned.
- Result: scale-strip preparation measured 0.0074 ms before gate/up and 0.0069 ms before next QKV. Excluding the copies, gate/up prologue and delayed kernels measured 0.6890 and 0.6912 ms. Next-QKV prologue and delayed kernels measured 0.1692 and 0.1559 ms.
- Result: the output projection sweep selected 128x256x64 cluster-1 at 0.1037 ms including its 0.0042 ms reducer. The down-projection sweep selected the same schedule at 0.3085 ms including reduction; tested 128x128 and 256x128 schedules ranged from 0.3230 to 0.4561 ms and 0.3271 to 0.3373 ms, respectively.
- Interpretation: the explicit strip copies explain about 14 microseconds per region, but not the next-QKV transform gap. Both residual/RMS producers should record 128x256x64 cluster-1. A fused reducer-to-strip output can remove two launches; a K-invariant scale path is the deeper optimization.
- Next action: execute the selected `RegionPlan` through a validated backend dispatch contract, with packed-QKV aliasing and physical partial-buffer shapes represented explicitly.

### 2026-08-06 - TLTC-021 plan-driven H100 execution

- Hypothesis: a validated runtime can consume the selected StableHLO-derived plan without adding layout copies, incorrect partial-buffer allocations, or measurable GPU work beyond the eight expert skeletons.
- Corrections: RMS partial records now have physical `(rows, ceil(hidden / 256))` shape and reducers emit `(rows,)`; QKV projection outputs are packed materializations and rotated Q/K/V are alias views of those allocations; H64 and H128 FA3 schedules and shape-dependent cluster-N choices are validated.
- Result: the CPU-testable runtime dispatches all eight skeletons in dependency order for both RMS placements and rejects unsupported backends, layouts, attachments, resources, bindings, and mixed placement families. The suite has 40 passing tests.
- H100 result: a primary-shape StableHLO export of 7.4 KB compiles to eight skeletons and executes through the runtime. At sequence 2048, an interleaved nine-repeat run measured prologue 1.5483 ms median/1.4199 minimum and delayed 1.5313/1.3973; an earlier shorter run measured 1.4459/1.4189 medians. At sequence 4096 the variants measured 3.1889/2.9988 and 3.1655/3.1103. Unpinned clocks dominate the median spread.
- Numerical result: at sequence 2048, prologue versus delayed mean absolute difference is 0.001601 at `x2` and 0.002429 at packed next QKV, matching the hand-composed oracle with BF16 gammas. At sequence 4096 the corresponding means are 0.001635 and 0.002482.
- Backend contract: `rope_posfreq` validates canonical base-10000 sine/cosine tables once before timed execution. General dynamic RoPE tables remain unsupported rather than being silently ignored.
- Interpretation: dense structure is recovered from ordinary StableHLO through executable H100 dispatch. Remaining dense work is refinement—durable serialization, cache/capture, and K-invariant prologue scale transport—not a structural blocker for beginning MoK recovery.
- Next action: use low-priority B200 capacity to reproduce the pinned Mixture-of-Kittens oracle and define the first expert-parallel skeleton around its router schedule, shared/routed expert tasks, readiness events, and dispatch/combine buffers.

### 2026-08-06 - TLTC-022 official Mixture-of-Kittens GB200 oracle

- Hypothesis: the official pinned MoK implementation can provide an executable SM100 structural and performance oracle on the authorized low-priority Blackwell pool.
- Source: Mixture-of-Kittens `3e1cf43ab93ad040afed52a45ab03cb490ffe4be`; ThunderKittens submodule `1c3920d993404dd49a6d4c7267ea11d583bd5c68`.
- Environment: one batch-priority 4xGB200 tray, compute capability 10.0, driver 595.71.05, PyTorch 2.10.0+cu130, NVCC 13.0.88, CCCL 13.0.85, Python 3.12.13.
- Build result: the full SM100 translation unit compiled. Reported forward worker variants use 255 registers, five barriers, 592 bytes static shared memory, and no spills. The pip toolkit required its CUDA library directory on `LIBRARY_PATH` for `libcudadevrt.a` and `libcudart_static.a`; upstream source remained unmodified.
- Benchmark config: four ranks, 2048 tokens/rank, 384 total experts, 96 local experts/rank, top-6, `H=7168`, `I=3072`, BF16 communication SMs 24/28 forward/backward, MXFP8 36/36, minibatch 4096, macrobatch 131072.
- Result: BF16 forward 3.669 ms/516.2 TFLOP/s and backward 9.029 ms/419.5 TFLOP/s. MXFP8 forward 2.521 ms/751.2 TFLOP/s and backward 7.662 ms/494.4 TFLOP/s.
- Correctness: official checks passed. BF16 output mean/max absolute error was 0.001723/0.03125; MXFP8 was 0.013673/0.121094. All reported gradient checks completed within upstream tolerances.
- Interpretation: MoK is usable infrastructure on the available GB200 pool. The next task is no longer environment reproduction; it is recovering the scheduler, shared/routed expert task graph, and communication contracts from an ordinary MoE graph.
- Next action: add a backend-independent expert-parallel skeleton and structural tests, then bind its physical SM100 configuration to the measured official oracle.

### 2026-08-06 - TLTC-023 Mixture-of-Kittens forward schedule sweep

- Hypothesis: MoK communication-worker allocation and minibatch size should be explicit tunable plan fields, and a small bounded sweep can improve on the published default for the measured four-rank shape.
- Broad sweep: with macrobatch 131072, communication SMs `{8,16,24,32,40,48}` and minibatches `{2048,4096,8192}` selected 24 SMs/minibatch 2048 at 3.6330 ms/521.4 TFLOP/s.
- Fine sweep: preserving the upstream 32:1 macro-to-minibatch ratio, communication SMs `{20,24,28}` and minibatches `{2048,2560,3072,4096}` selected 20 SMs/minibatch 2048/macrobatch 65536 at 3.5842 ms/528.4 TFLOP/s.
- Correctness replay: the official benchmark passed with the winning BF16 forward schedule and measured 3.613 ms/524.2 TFLOP/s forward and 9.077 ms/417.3 TFLOP/s backward. BF16 output mean/max absolute error remained 0.001723/0.03125, and all official gradient checks passed.
- Legality learned from the failed first fine-sweep attempt: minibatch must be positive and divisible by 256; macrobatch must be a positive exact multiple of minibatch. These constraints belong in structural validation before backend launch.
- Interpretation: communication SM count, minibatch, and macrobatch are physical expert-parallel schedule fields. The best communication allocation is coupled to batching and should be selected empirically from a bounded legal set.
- Next action: incorporate these contracts into the backend-independent MoK skeleton and use 20/2048/65536 as the measured BF16-forward SM100 candidate for this shape.

### 2026-08-06 - TLTC-024 explicit MoK structural plan

- Hypothesis: the official MoK megakernel can be represented as one physical skeleton without becoming opaque if its schedule arrays, buffers, readiness events, task roles, worker roles, tiles, and backend revision remain explicit and independently validated.
- Semantic region: router projection/top-k, one shared gated MLP, top-k routed gated MLPs, and weighted shared-plus-routed combination.
- Plan result: one expert-parallel skeleton contains the 256-padded capacity-bounded schedule, symmetric dispatch/combine workspaces, shared and routed gate/up/SwiGLU/down buffers, five forward readiness-event families, ten persistent task roles, and separate communication/GEMM producer/GEMM consumer workers.
- Measured physical candidate: four ranks, 384/96 global/local experts, top-6, `H=7168`, `I=3072`, 2048 local tokens, 20 communication SMs, minibatch 2048, macrobatch 65536, and pinned MoK revision `3e1cf43ab93ad040afed52a45ab03cb490ffe4be`.
- Legality: structured diagnostics aggregate unsupported rank partitions, tile dimensions, batch divisibility, worker counts, numerical policy, and dtypes. All shared and routed weights are checked. MXFP8 is rejected because explicit FP8 scale-tensor semantics are not modeled yet.
- Validation: five MoK behavior tests pass; the complete package suite passes with 45 tests and Pyrefly reports no source/test errors.
- Interpretation: the task/event structure has been recovered in a backend-independent form, but its input is still a hand-built semantic graph and its backend binding is not executable. The skeleton is therefore an honest intermediate checkpoint, not yet a compiler-generated MoK comparison.
- Next action: recover this semantic region from ordinary JAX StableHLO and add a validated dispatch boundary for the pinned SM100/SM103 implementation.

### 2026-08-06 - TLTC-025 first-principles MoE synthesis scope correction

- Goal correction: Mixture-of-Kittens is only a correctness/performance oracle. The compiler must not select its complete megakernel or encode the recovered MoK task graph as the compilation result.
- Required path: ordinary global routed-MoE graph -> route relation -> expert ownership -> segment/group and padding -> forward exchange -> shared dense and routed segmented contractions -> SwiGLU -> routed down contraction -> reverse exchange -> weighted scatter reduction -> shared add.
- Physical lowering: derive tile-flow edges, layouts, readiness granularity, fan-out/fan-in, buffer capacities/lifetimes, worker pools, minibatches, macrobatches, and pipeline depths through generic transformations.
- Semantic correction: ordinary graph expert indices and routed weights live on a global expert axis. Rank-local expert weights are a physical partition derived by the planner, not an assumption embedded in the semantic operation.
- Oracle status: the extracted MoK skeleton remains useful for checking stage coverage and measured constraints, including 256-row padding and the tuned 20-SM/2048-minibatch/65536-macrobatch candidate, but it is no longer on the compiler execution path.
- Performance criterion: generated BF16 forward execution should come within 20–30% of the pinned four-GB200 MoK oracle at the same shape.
- Immediate actions: finish honest JAX StableHLO recovery to a global semantic graph, add generic relation/segmented-contraction/tile-flow plans, then implement and tune a distributed backend from those plans.

### 2026-08-06 - TLTC-026 ordinary MoE recovery and generic EP plan

- Frontend: an ordinary JAX shared-plus-routed gated MLP exports to a frozen 70-operation StableHLO fixture. The importer preserves the version-1 `chlo.top_k` composite, three static expert gathers, reduction/broadcast semantics, and source provenance.
- Semantic result: `Linear -> TopKRouter -> SharedExpertMLP -> RoutedExpertMLP -> WeightedExpertCombine`, with routed weights explicitly on the same global expert axis as router logits. No MoK operation or local-weight assumption appears in recovery.
- Generic lowering: 17 atomic stages cover route relation, ownership, gate/up layout legalization, owner grouping, exchange projection, forward exchange, receiver expansion, local-expert segmentation/padding, shared/routed W13, separate SwiGLU maps, shared/routed W2, reverse exchange, deterministic weighted reduction, and shared add.
- Alternatives: payload transport may use one row per assignment or coalesce by `(source_token, owner_rank)`; gate/up weights may remain separate or become concatenated/interleaved physical views. Candidate primitives are ragged all-to-all or DeepEP transport plus `ragged_dot`, not a fused MoK call.
- Capacity correction: expert padding occurs after receiver-local segmentation, not on every sender. Default guarded capacity is 15360 received assignments and 39840 padded-local rows at the oracle shape, down from the rejected approximately 147k-row construction. Overflow takes an exact fallback before contraction.
- Numerical decision: route-slot order may change, but merge order must remain deterministic. Unordered atomic accumulation is disallowed.
- Validation: 54 package tests pass, Pyrefly reports no source/test errors, and scoped pre-commit passes.
- Next action: execute the generic four-GB200 ragged-all-to-all and DeepEP baselines using the exact pinned MoK routing fixture, then implement the reusable runtime relation/index plane and use measured phase costs to choose the first generated schedule.

### 2026-08-06 - TLTC-027 reusable relation/index plane

- Hypothesis: one generic runtime relation structure can drive forward dispatch, optional payload coalescing, inverse dispatch, and deterministic weighted merge without importing MoK's schedule builder.
- Result: `RelationPlan` records source item and route slot, global and owner-local destination, stable padded destination rows, per-group counts/offsets, router weights, validity/padding, exchange rows, and both route-to-destination and destination-to-route mappings.
- Result: assignment dispatch and inverse dispatch are exact permutations. Coalesced transport sends one row per distinct `(source item, destination rank)` and expands it through relation metadata on receipt.
- Numerical contract: weighted merge restores source-item/route-slot order and accumulates FP32 values in ascending route-slot order. Route storage order may change, but unordered atomic accumulation is not permitted.
- Capacity contract: receiver assignment and padded-row overflow are diagnosed before payload movement. The seeded oracle-size 384-expert, four-rank, 2048-token, top-6 relation fits the generic plan's guarded capacity.
- Validation: 61 package tests pass and Pyrefly reports zero errors.
- Next action: benchmark exact MoK-seeded routes through the generic four-GB200 payload implementations and isolate transport, segmented-contraction, materialization, and capacity costs.

### 2026-08-06 - TLTC-028 first generic four-GB200 payload baseline

- Fixture: exact selected experts and FP32 router weights from MoK's per-rank CUDA seeds `1234 + rank`; saved NPZ SHA256 `6ffd9d42c0ae1da109503f3d3a5d6ec992ffdbb84f41b4cc6f0493f35f5c0dff`.
- Runtime finding: native JAX/JAXlib 0.11 ragged-all-to-all one-shot and fallback paths segfault on first execution, including a tiny four-rank graph. XLA decomposition of the ragged collective executes correctly. The structured investigation is in `docs/debug-log-shuttle-generic-ep.md`.
- Baseline: decomposed transport plus Triton `ragged_dot` compiled in 8.858 seconds and measured 9.460 ms median over 10 warmups and 50 iterations, or 200.2 logical TFLOP/s per rank. Output was finite and no assignments were dropped. XLA `ragged_dot` measured 95.100 ms and is rejected.
- Phase result: shared expert 0.382 ms; routed path 9.289 ms; dispatch/inverse-dispatch identity round trip 4.875 ms; already-routed padded W13/SwiGLU/W2 4.696 ms. Transport and grouped compute are both first-order costs.
- Capacity result: exact receiver counts were `[12281, 12281, 12349, 12241]`. Reducing capacity from 15360 to 12349 rows improved full latency to 8.445 ms and local compute to 4.183 ms with zero drops. A 1% guard above the exact bound was effectively tied.
- Relation result: exact routes touch 3.301 destination ranks per token, so `(token, owner rank)` projection can remove about 45% of H-wide forward rows while preserving route-slot identity for deterministic combine.
- DeepEP result: after replacing unordered segment accumulation with inverse permutation and ascending-route-slot FP32 reduction, the un-specialized 49152-assignment path measured 11.354 ms with zero drops. Its four-times-overprovisioned local assignment domain is the next transport candidate to fix.
- Oracle comparison: best generic 8.445 ms is 2.34 times the tuned 3.613-ms MoK oracle and remains outside the 1.3-times target.
- Next action: cap DeepEP's assignment domain with explicit overflow handling, benchmark the coalesced transport plan, evaluate MoK's reusable grouped-GEMM primitive below its full event graph, then overlap bounded dispatch and compute chunks.

### 2026-08-06 - TLTC-029 compact deterministic DeepEP plan

- Hypothesis: DeepEP's token-owner-coalesced transport is obscured by a 49152-row local assignment domain, four times the nominal 12288 assignments per rank. Compacting valid local assignments once should preserve transport savings and reduce grouped-GEMM launch/work domains.
- Rejected candidate: splitting received tokens into four smaller batches repeated all 96 expert launch grids and regressed routed latency by 32.5%, from 11.195 to 14.832 ms. Token batching was removed.
- Selected construction: one global `top_k(capacity)` compacts assignments in expert order and retains each original `(received token, route slot)` position. W13/SwiGLU/W2 operate on the compact domain. Collapse sorts compact positions, uses per-slot search/gather, and accumulates ascending route slots in FP32 without scatter reduction or atomics.
- Capacity contract: `ceil(capacity_factor * local_tokens * top_k)` rows, with exact overflow count returned through the distributed drop count. The compiler's relation precheck selects an exact fallback before an overflowing candidate executes.
- GB200 result: exact safe capacity 12349, 10 warmups/50 iterations, 6.149 ms median and 308.1 logical TFLOP/s per rank, finite output, zero drops. This is 45.8% faster than uncapped DeepEP, 27.2% faster than the best decomposed-ragged plan, and 1.70 times tuned MoK.
- Validation: the full local Grug MoE test file passes with 15 tests and six GPU-dependent skips; scoped lint, format, type, and repository checks pass.
- Next action: split capped DeepEP into transport/merge and compact contraction costs, benchmark the extracted standalone MoK/TK grouped-GEMM primitive, and introduce bounded overlap using the faster components.

### 2026-08-06 - TLTC-030 standalone Blackwell expert mainloop

- Hypothesis: a high-quality expert mainloop below MoK's complete event graph can close the compact Triton compute gap without making MoK itself the compiler lowering.
- Extraction: `expert_grouped_gemm_kernel<false>()` runs as a standalone two-CTA cluster with its TMA, TCGEN05, tensor-memory, and semaphore pipeline intact. Dispatch, SwiGLU, combine, reuse events, and the CLC persistent scheduler are absent; every cross-task event pointer is null.
- Build: pinned MoK `3e1cf43...`, ThunderKittens `1c3920d...`, NVCC/CRT/NVVM 13.0.88, CCCL 13.0.85, SM100a. The kernel uses 255 registers, five barriers, 224 bytes static shared memory, and no spills.
- Correctness: the quick two-expert 256-shape W2/W13 checks pass against Torch; maximum absolute error is 0.0149 with no NaN or infinity.
- Performance: 96 experts and 24576 total 256-padded rows. W13 is two launches totaling 2.036 ms and 1063 padded-work TFLOP/s. W2 is 0.943 ms and 1148 padded-work TFLOP/s.
- Comparison: compact Triton W13/SwiGLU/W2 is 4.185 ms and raw DeepEP dispatch/combine identity is 1.340 ms. Replacing the compact contractions with the standalone projections predicts approximately 4.92 ms before packing, explicit SwiGLU, and overlap are measured.
- Next action: compose DeepEP transport, generic 256-padding, standalone W13, explicit SwiGLU, standalone W2, deterministic return/merge, and shared expert in one Torch/CUDA physical runtime; then pipeline bounded row chunks to overlap transport and compute.

### 2026-08-06 - TLTC-032 four-rank generated physical runtime

- Hypothesis: official DeepEP transport can be legalized into the compiler's relation order, then composed with standalone generated expert stages without importing the MoK forward/event graph; overlapping the shared expert with asynchronous dispatch should close most of the remaining gap.
- Source: DeepEP `7febc6e25660af0f54d95dd781ecdcd62265ecca`; MoK `3e1cf43ab93ad040afed52a45ab03cb490ffe4be`; ThunderKittens `1c3920d993404dd49a6d4c7267ea11d583bd5c68`; generated extension `/tmp/mok-gmm-probe-explicit-merge-cu130/_mok_gmm_probe.cpython-312-aarch64-linux-gnu.so`.
- Config: four GB200 ranks, 2048 tokens/rank, 384 global and 96 local experts/rank, top-6, hidden 7168, intermediate 3072, BF16 payloads, exact route fixture SHA256 `6ffd9d42c0ae1da109503f3d3a5d6ec992ffdbb84f41b4cc6f0493f35f5c0dff`, DeepEP 24 communication SMs.
- Construction: `build_relation_plan` produces coalesced receiver, expert grouping, and 256-padding metadata. A legalization maps DeepEP's source-rank prefixes plus `recv_src_idx` to global source tokens. The runtime uses generated padded pack, standalone W13, generated SwiGLU, standalone W2, fixed route-slot explicit FP32 multiply/add without atomics, official fixed-rank DeepEP combine, and the generated shared expert as combine bias.
- Mapping result: every rank exactly matched received payloads, local expert IDs, valid route weights, expert counts, and compiler receiver/assignment/padded-row metadata. Receiver rows were `[6755, 6810, 6766, 6713]`; assignments were `[12281, 12281, 12349, 12241]`; every rank had 24576 padded rows.
- Numerical result: sequential and overlapped exact-shape outputs were bitwise equal, repeat-deterministic, and finite on all ranks. An independent small four-rank source-ordered Torch MoE reference (`T=256`, `E=8`, top-2, `H=I=256`) passed on every rank with maximum absolute error `0.00012207` and mean absolute error approximately `4.7e-6`. The reference is numerically independent of schedule parity and explicitly permits the physical path's per-owner BF16 collapse before fixed-rank combine.
- Command: `CUDA_HOME=/tmp/mok-route-env/lib/python3.12/site-packages/nvidia/cu13 PYTHONPATH=/tmp/deepep-torch-intranode-build/lib:/tmp/DeepEP-torch-intranode:/tmp/tile_lifetime_runtime LIBRARY_PATH=/tmp/deepep-torch-link:$CUDA_HOME/lib LD_LIBRARY_PATH=/tmp/deepep-torch-link:$CUDA_HOME/lib:/tmp/mok-route-env/lib/python3.12/site-packages/torch/lib PATH=/tmp/mok-route-env/bin:$CUDA_HOME/bin:$PATH OMP_NUM_THREADS=1 /tmp/mok-route-env/bin/torchrun --standalone --nproc-per-node=4 lib/tile_lifetime/benchmarks/backends/gb200_deepep_mok_distributed.py --route-fixture /tmp/mok_routes_t2048_e384_k6_seed1234_torch2.10.npz --probe-extension /tmp/mok-gmm-probe-explicit-merge-cu130/_mok_gmm_probe.cpython-312-aarch64-linux-gnu.so --deepep-root /tmp/DeepEP --warmup 10 --iterations 50 --json-output /tmp/deepep-mok-distributed-rankmax-reference.json`.
- Rank-max medians: 4.4782 ms sequential, 4.2682 ms with shared expert overlapped with async dispatch, 3.4775 ms already-dispatched routed path, 0.2948 ms shared expert, 0.4477 ms combine with shared bias, and 0.8201 ms dispatch plus identity combine.
- Interpretation: bounded overlap improves end-to-end latency by 4.69%. The 4.2682-ms generated path is 18.1% slower than the tuned 3.613-ms MoK oracle, inside the first-principles 20–30% target, and 30.2% faster than the prior 6.113-ms compact DeepEP plan. It uses no MoK forward call or event graph.
- Artifact: `scratch/shuttle-generic-results/deepep-mok-distributed-rankmax-reference.json`.
- Next action: sweep the bounded DeepEP communication-worker count, then evaluate concatenated W13 and coarser pipeline overlap without weakening the deterministic merge contract.

### 2026-08-06 - TLTC-033 DeepEP worker sweep and concatenated W13

- Hypothesis: the generated runtime remains transport-limited enough that more DeepEP communication workers improve the overlapped plan, while concatenating gate/up can reduce local launch overhead.
- DeepEP sweep: communication SMs `{12,16,20,24,28,32,36,40,48}`, each with 10 warmups and 50 rank-max iterations. Overlapped medians were `4.9008`, `4.5903`, `4.3961`, `4.2645`, `4.1727`, `4.1289`, `4.1171`, `4.0878`, and `4.0315` ms respectively. Dispatch-plus-identity-combine declined from 1.4572 ms at 12 SMs to 0.5035 ms at 48 SMs.
- Correctness: every sweep point retained exact compiler-to-DeepEP metadata/payload correspondence, bitwise sequential/overlap parity, repeat determinism, finite output, and the independent small semantic reference.
- Selection: 48 communication SMs is the measured best of the bounded set. Its 4.0315-ms rank-max median is 3.39% faster than 28 SMs, 11.6% slower than the 3.613-ms MoK oracle, and 34.1% faster than the prior 6.113-ms compact DeepEP plan. Latency remained monotonic through the bounded upper limit, so a later sweep may test whether additional communication workers eventually trade off against compute occupancy.
- Artifacts: `scratch/shuttle-generic-results/deepep-mok-distributed-sms12.json` through `deepep-mok-distributed-sms48.json` for the measured candidate set.
- Concatenated W13 result: one `[E,2I,H]` grouped GEMM plus the generated row-halves SwiGLU path was bitwise equal to the separate gate/up construction and repeat-deterministic. The initial already-dispatched pre-combine median improved from 3.4518 to 3.3816 ms (2.03%); an alternating-order measurement is queued because isolated component timings drifted during the first sweep.
- Concatenated artifact: `scratch/shuttle-generic-results/deepep-mok-local-concat-w13.json`.
- Next action: validate concatenated W13 with alternating sample order, bind the selected 48-SM schedule into the physical candidate, and profile the remaining approximately 0.42-ms gap to the MoK oracle.

### 2026-08-06 - TLTC-031 generated receiver-local MoK composition

- Hypothesis: compiler-produced token-owner relation metadata can drive MoK's standalone grouped-GEMM mainloop without copying its task graph, and generated pack/SwiGLU/merge kernels can remove the Torch materialization overhead around it.
- Commit Hash: uncommitted branch `prototype/tile-lifetime-compiler`.
- Config: one GB200 rank at an already-dispatched receive boundary, owner rank 0, four-rank global route fixture, 2048 tokens per rank, 384/96 global/local experts, top-6, `H=7168`, `I=3072`, BF16 activations/weights/output, FP32 routed merge, 256-row expert padding, 10 warmups, and 50 iterations. Pinned MoK `3e1cf43...`, ThunderKittens `1c3920d...`, and DeepEP `7febc6e...`.
- Command:

  ```bash
  export CUDA_HOME=/tmp/mok-route-env/lib/python3.12/site-packages/nvidia/cu13
  export PATH=/tmp/mok-route-env/bin:$CUDA_HOME/bin:$PATH
  export PYTHONPATH=/tmp/tile_lifetime_runtime
  export LIBRARY_PATH=/tmp/deepep-torch-link:$CUDA_HOME/lib:${LIBRARY_PATH:-}
  export LD_LIBRARY_PATH=/tmp/deepep-torch-link:$CUDA_HOME/lib:/tmp/mok-route-env/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}
  cd /app
  /tmp/mok-route-env/bin/python lib/tile_lifetime/benchmarks/backends/gb200_deepep_mok_local.py \
    --mok-root /tmp/mixture-of-kittens-fixture-source \
    --deepep-root /tmp/DeepEP \
    --route-fixture /tmp/mok_routes_t2048_e384_k6_seed1234_torch2.10.npz \
    --build-dir /tmp/mok-gmm-probe-explicit-merge-cu130 \
    --nvcc $CUDA_HOME/bin/nvcc \
    --owner-rank 0 \
    --device cuda:0 \
    --warmup 10 \
    --iterations 50 \
    --json-output /tmp/deepep-mok-local-explicit.json \
    2>&1 | tee /tmp/deepep-mok-local-explicit.log
  ```

- Import setup: `/tmp/tile_lifetime_runtime` contains the exact checked-in `relation.py`, `expert_parallel_plan.py`, `ir.py`, and `plan.py` under a Torch-only package. It avoids importing JAX in the Torch benchmark and does not duplicate relation construction.
- Relation result: an independent direct source-token/route-slot scan exactly matched 6755 coalesced receive tokens, 12281 local assignments, 96 expert counts, 24576 padded rows, every route-to-padded-row index, and the inverse padded-row map.
- Stage medians:

  | Stage | Torch | Generated |
  |---|---:|---:|
  | Coalesced gather simulation | — | 0.038816 ms |
  | Padded pack | 0.384576 ms | 0.305104 ms |
  | W13 | — | 1.674016 ms |
  | SwiGLU | 0.818992 ms | 0.125232 ms |
  | W2 | — | 0.918880 ms |
  | Pre-combine merge | 3.037072 ms | 0.360832 ms |
  | Full pre-combine composition | 6.747360 ms | 3.454944 ms |

- Performance: the generated composition is 1.95 times faster than the matching Torch sequence and reaches 469.64 logical TFLOP/s or 939.81 padded-work TFLOP/s. The FMA merge measured 0.358240 ms, 0.002592 ms faster than explicit round-to-nearest multiply/add.
- Numerical result: generated pack and SwiGLU are bitwise equal to Torch. Explicit `__fmul_rn` plus `__fadd_rn` merge is bitwise equal with zero maximum/mean error and repeats bitwise. FMA is deterministic and allclose but not bitwise, with maximum absolute error `2.3841858e-7` and mean absolute error `1.2035570e-13`. The full owner-local merge/shared-add diagnostic is bitwise equal, repeat-bitwise, and finite.
- Scope: the primary timing excludes official DeepEP dispatch/combine, shared-expert compute, MoK CLC scheduling, and overlap. The owner-local fused merge/shared-add diagnostic measured 0.123424 ms but is not a replacement for cross-rank combine.
- Artifact: `scratch/shuttle-generic-results/deepep-mok-local-explicit.json`; remote JSON/log are `/tmp/deepep-mok-local-explicit.json` and `/tmp/deepep-mok-local-explicit.log`.
- Interpretation: the prior merge mismatch was contraction rounding, not a route or padding mapping error. The selected fixed-order explicit-rounding path preserves the Torch accumulation order without atomics. Generic relation metadata is sufficient to feed the standalone expert mainloop at this boundary.
- Next action: run official pinned DeepEP dispatch/combine around this path, compute the shared expert with the same grouped-GEMM and generated SwiGLU entrypoints, and compare sequential with legal asynchronous overlap using rank-maximum latency.

### 2026-08-06 - TLTC-034 final distributed MoE layout and worker selection

- Result: the compiler-derived four-rank runtime composes official DeepEP transport with relation-driven packing, standalone grouped GEMMs, generated SwiGLU, deterministic fixed-slot merge, and generated shared-expert work. It does not call the MoK forward kernel or copy its event graph.
- Worker sweep: rank-maximum overlap latency fell from `4.9008` ms at 12 DeepEP SMs to `4.0148` ms at 80 SMs, then rose to `4.0395` ms at 96 SMs. The measured sweep was `{12,16,20,24,28,32,36,40,48,56,64,80,96}` SMs. The turn at 96 closed this tuning dimension.
- Layout A/B: concatenated `[E,2I,K]` W13 beat separate `[E,I,K]` gate/up at both confirmation points. At 56 SMs, rank-maximum overlap medians were `3.9760` versus `4.0797` ms, a `0.1037`-ms or `2.54%` reduction. At 80 SMs they were `4.0305` versus `4.1298` ms, a `0.0993`-ms or `2.40%` reduction. A second concatenated 56-SM run measured `3.9910` ms; the two-run median of medians is `3.9835` ms.
- Selection: concatenated W13 with 56 DeepEP SMs is the final measured plan. Its two confirmation runs are `10.0%` and `10.5%` slower than the `3.613`-ms tuned MoK oracle, and about `34.8%` faster than the `6.113`-ms compact DeepEP baseline.
- Correctness: all ranks in every 56/80 A/B and repeat run had exact transport/relation mappings, finite outputs, bitwise sequential/overlap equality, repeat-bitwise equality, and a passing independent small four-rank semantic reference. The reference maximum absolute error was `0.0001220703125`.
- Fixture identity: the original NPZ container SHA256 is `6ffd9d42c0ae1da109503f3d3a5d6ec992ffdbb84f41b4cc6f0493f35f5c0dff`. The replacement tray reserialized the same Torch-2.10/CUDA-seeded tensors, yielding container SHA256 `c143b12f2879430106d5013aea8e95ef0705ba8daaffa5eeb1ece49559217d38`. A container-independent hash over tensor name, dtype, shape, and C-order bytes is `f1b5d8b3a53372eca228261b48b7ad9cfe925f1f8083f9cae07f9a24713f6908`. Receiver assignment counts remained `[12281,12281,12349,12241]`.
- Artifacts: `scratch/shuttle-generic-results/deepep-mok-distributed-{concat,separate}-sms{56,80}.json`, `deepep-mok-distributed-concat-sms56-repeat.json`, and `mok_routes_t2048_e384_k6_seed1234_torch2.10-reserialized.npz`.
- Infrastructure: the first low-priority holder expired after its results had been copied locally. The replacement `/dlwh/dev-gpu-shuttle-generic-final` reproduced the environment and final measurements with no result loss. It was released after the A/B; Iris reports `killed` with reason `Terminated by user`, and its Kubernetes pod is gone.
- Validation: scoped pre-commit and Pyrefly checks pass for the DeepEP build helper and distributed benchmark. The package suite remains at 61 passing tests.
- Next action: profile the selected 56-SM concatenated plan against the MoK oracle before changing the communication/computation pipeline. The remaining median gap is approximately `0.37` ms.
