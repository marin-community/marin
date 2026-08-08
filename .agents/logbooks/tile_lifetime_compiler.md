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

- Dense and distributed-MoE checkpoints are frozen. The active experiment is a clean MiniMax Sparse Attention (MSA) proof on one GB200: ordinary JAX/StableHLO must recover index projections, causal block routing, a generic Relation, and exact selected attention; the device path must synthesize its Contract/Map/Fold/DomainRestriction body rather than call the official MSA kernel. The official SM100 implementation is the matched oracle and a source of low-level physical machinery only.

## Hypothesis Queue

### Active

- `TLTC-MSA-001`: MSA's native token-and-GQA-group route can lower to the existing generic relation and normalized-exponential state machinery, then drive a Shuttle-owned SM100 KV-major skeleton within 1.20 times the pinned official oracle. Next test: close the exact payload boundary on one GB200 and compile a natural StableHLO fixture into the same relation.
- `TLTC-RSA-003`: query-major and KV-major performance cross over as relation degree, KV reuse, padding, and partial-state traffic change. Next test: use a deliberately non-monotone relation and implement actual cross-query KV staging; current grouped CTAs do not share staged K/V.
- `TLTC-005`: the semantic recognizer can tolerate ordinary JAX broadcast, multiply-order, and projection variations while rejecting near-misses with structured diagnostics. Next test: add permuted and illegal StableHLO fixtures.
- `TLTC-002`: named layout contracts around fixed CuTe and attention skeletons are sufficient for the first full-region planner. Next test: inventory the layouts required by the local JAX/CuTe attention boundary.
- `TLTC-003`: a direct Torch/CuTe or CODA boundary can execute expert Hopper kernels while the JAX adapter is repaired separately. Next test: benchmark official FA3/FA4 and CODA RMS on the reserved H100.
- `TLTC-007`: consumer-prologue RMS scaling should track source BF16 ordering more closely than CODA's delayed epilogue scale, but cost more because the input transform disrupts the TMA/WGMMA mainloop. Next test: compare CODA, a fused prologue experiment, and materialized pre-scaling on primary shapes.

### Blocked

- The checked-in JAX FA4 THD wrapper aborts in `quack.copy_utils.create_ragged_tensor_for_tma` for all tested package tuples. The same physical Hopper kernel works through Torch, so this blocks JAX integration but not component benchmarking.

### Falsified / Dead End

- `TLTC-RSA-004A`: source-order traversal isolates the benefit of per-wave KV sorting on the canonical fixture. It does not: source order is already KV-monotone in every slot, so the ablation executes identical edge arrays.
- `TLTC-RSA-004B`: increasing the query tile from M32/four warps to M64/eight warps improves the first Triton kernel. At 2K it regresses from 0.502016 to 0.660096 ms.

### Promoted

- `TLTC-RSA-001`: executable FP32 attention partial state and stable slot-order merge match an independently materialized selected-mask reference for causal GQA, uneven relation degree, padding, and a sequence tail.
- `TLTC-RSA-002`: `RelationPlan` now drives both source traversal and compact destination-major offsets/sorted source identities. The full package suite passes without changing existing MoE behavior.
- `TLTC-RSA-004`: an executable deterministic slot-wave plan consumes the generic relation directly, updates one FP32 online state per query without atomics or edge partials, and measures 4.017344 ms at 16K.
- `TLTC-035`: the dense and distributed-MoE checkpoint is preserved with raw distributions, correctness fixtures, hashes, candidate records, pins, and hardware telemetry at annotated local tag `shuttle-gb200-moe-v1`.
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

### 2026-08-08 - TLTC-MSA-001 native routed sparse-attention start

- Hypothesis: the existing generic `RelationPlan`, orientation, readiness, bounded-buffer, `Contract`, `DomainRestriction`, and normalized-exponential `Fold` machinery can express MSA's native `(query token, KV group) -> selected KV block` computation. The only new workload-specific code should be frontend recovery and SM100 physical legalization; online-state, mask, merge, and routing semantics must remain Shuttle-owned.
- Baseline: branch `research/shuttle-clean-helper-boundaries` at commit `dd3bb84759`.
- Oracle revisions: MiniMax Sparse Attention `80434d7f67877c6570ca19cac444b84bc9855dac`; its CUTLASS submodule `eb61c911471867a5fd2466bfd8f29306cea6ebf8`.
- Primary configuration: one low-priority B200/GB200 GPU, BF16 Q/K/V and FP32 online state, batch 1, sequence 65536, 64 query heads, 4 KV heads, head dimension 128, KV block 128, top-k 16, causal. Sequence 16384 is the bring-up configuration.
- Natural semantics: index-Q/index-K `Contract`s; causal token-score `Contract`; per-block maximum `Fold`; deterministic per-GQA-group top-k `Selection` with the local block forced; generic `Relation`; selected exact QK `Contract`; score-scale `Map`; `DomainRestriction`; normalized-exponential `Fold`; and PV `Contract`.
- Measurement boundary A, payload: identical precomputed `q2k_indices`; q2k-to-k2q orientation/schedule preparation; sparse attention body. Boundary B, full route: index projections and scores; block-max; top-k; relation construction; and sparse attention. Routing/index planning is never silently excluded from the full boundary.
- Acceptance: deterministic numerical correctness against the natural semantic reference; one semantic mutation through the same generator; no opaque official MSA/FlashAttention semantic call; generated Shuttle latency no more than 1.20 times the pinned official MSA implementation for one matched primary configuration.
- Allowed lineage: retain generic SM100 tensor-core mainloops, TMA/copy/layout machinery, barriers, and bounded pipeline templates derived from MSA/CUTLASS. Replace or generalize MSA-owned softmax-state, attention-mask, route-combine, and workload scheduling semantics behind Shuttle `Fold`, `DomainRestriction`, and `RelationPlan` interfaces.
- Candidate set: query-major sparse fold and KV-major right-resource reuse; a bounded set of tile/pipeline/worker configurations derived from hardware legality. Do not hard-code the official MSA schedule as the answer.
- Parallel work: one GB200 oracle reproduction, one matched benchmark adapter, and one natural JAX/StableHLO recovery path. Root owns the clean SM100 synthesis boundary and final matched evaluation.
- Next action: reproduce the exact pinned oracle and semantic interface, freeze a debug fixture, then extract only the generic SM100 physical mechanisms needed by the generated KV-major candidate.

### 2026-08-08 - TLTC-MSA-002 full-route synthesis boundary

- Source check: MSA paper Section 3.1 defines two index projections from hidden state, a causal token-score Contract scaled by `1/sqrt(d_idx)`, a maximum Fold over 128-token KV blocks, per-query/per-GQA-group top-16 selection with the local block forced, and exact selected attention. Section 4 implements the route with a dense max-score pass and exp-free top-k, then uses KV-outer sparse attention with a two-phase partial-state combine.
- Scope correction: the primary full-route acceptance path must generate or instantiate generic index `Contract -> Map -> DomainRestriction -> Fold(max) -> Selection` machinery. Calling the official MSA dense proxy or top-k implementation in Shuttle would be the same semantic-boundary violation as calling its sparse-attention body.
- Physical-state derivation: MSA materializes normalized per-block output and log-sum-exp rather than separate `(max, sum, weighted value)` fields. Shuttle now models this as a generic coordinate change of the normalized-exponential Fold state and tests that log-normalizer-weighted merging reproduces the canonical state merge.
- Backend-neutral progress: `sm100_routed_lowering.py` validates a flattened `(query token, KV group) -> (KV group, block)` relation, GQA mapping, BF16 D=128 legality, selected counts, group preservation, both relation orientations, explicit pipeline fields, and compact partial-state representation. A causal-to-unrestricted mutation changes semantics through the same schedule lowering.
- Next action: add the equivalent backend-neutral lowering for the index Contract/block-max/Selection, then connect both legalizations to Shuttle-owned SM100 physical emitters.

### 2026-08-06 - TLTC-RSA-001 routed sparse-attention start

- Hypothesis: the relation/index machinery recovered for MoE is generic enough to drive query-block to KV-block traversal in both query-major and KV-major orientations; only the grouped QK/softmax/PV body and structured merge should be attention-specific.
- Baseline: annotated local tag `shuttle-gb200-moe-v1`, peeled commit `9ba3888cb0f91e2cf54f2a182927f13e769be2c6`.
- Branch: `research/shuttle-routed-sparse-attention`.
- Scope: CPU semantics and relation reuse first, then one single-GPU BF16 backend; no major XLA integration and no distributed implementation unless transport reuse proves small.
- Research sources: MoBA semantics first; FlashMoBA and FlashMLA as physical oracles; Quest for index-plane/payload-plane separation; FlashAttention for exact online-state algebra; NSA and SeerAttention as overfitting checks.
- Durable brief: `.agents/projects/tile_lifetime_compiler/routed_sparse_attention_brief.md`.
- Implementation plan: `.agents/projects/tile_lifetime_compiler/routed_sparse_attention_plan.md`.
- Coordinating issue: none; this remains local until publication is requested.
- Next action: implement an independent masked reference, structured partial state, query-major fold, and KV-major `RelationPlan` adapter with deterministic source-slot merge.

### 2026-08-06 - TLTC-RSA-002 relation reuse and two executable orientations

- Hypothesis: one binary relation plus exact online-softmax state is sufficient to execute both query-major and pure KV-major selected-block attention without a workload-specific route-plan type.
- Commit Hash: uncommitted branch `research/shuttle-routed-sparse-attention`, based on `shuttle-gb200-moe-v1` peeled commit `9ba3888cb0f91e2cf54f2a182927f13e769be2c6`.
- Config: CPU debug workload with sequence length 10 padded into three 4-token Q/KV blocks, four query heads, two KV heads, query/key dimension 8, value dimension 5, causal masking, source degrees `[1,2,3]`, destination degrees `[3,2,1]`, and destination padding quantum 2.
- Result: the generic relation gained per-edge validity, `-1` invalid sentinels, invalid-slot inverse fill, valid-edge capacity accounting, and compact destination-major offsets/source identities/source slots. Existing all-valid MoE behavior remains unchanged.
- Result: query-major keeps a query's FP32 `(max,sum,value)` state resident while visiting selected blocks. KV-major groups edges by KV block, produces one partial per valid edge, inverse-routes the three state fields independently, and merges in stable query-block/slot order without atomics.
- Result: both orientations match the independently materialized selected-mask reference at `rtol=2e-6`, `atol=2e-6`; repeated KV-major output is bitwise deterministic; padded query-tail outputs are zero; duplicate selected blocks and empty fold domains are rejected.
- Structural result: both physical candidates are emitted from the same relation with derived source/destination arrival counts, bounded buffers, worker roles, kernel regions, and byte estimates. The query-major plan materializes zero partial-state bytes. The initial two-kernel KV-major baseline explicitly materializes one FP32 state per valid edge. Neither materializes sequence-squared scores or probabilities.
- Primary-shape estimate: at sequence 16384, Q/KV block 128, 32/8 heads, dimension 128, and top-k 8, the deterministic fixture contains 996 valid edges. The coarse pure-KV-major plan would materialize 2,121,400,320 bytes of FP32 partial state, versus zero partial-state bytes for query-major. This is an analytical warning and pruning signal, not a measured rejection of KV-major reuse.
- Command: `uv run --frozen --package marin-tile-lifetime --group test pytest -q lib/tile_lifetime/tests`.
- Result: the complete package suite passes 74 tests, including routed-attention and slot-wave behavior tests.
- Command: `uv run --frozen --package marin-core --group lint pyrefly check lib/tile_lifetime/src`.
- Result: zero type errors. Scoped repository pre-commit checks passed.
- Generality accounting: relation ownership, grouping, padding, dispatch, inverse mapping, coalescing, and capacity were reused. Ragged validity and inverse fill generalized MoE machinery. Exact attention state and counted readiness fields are new generic machinery. Selected-block legality, causal/tail masking, and GQA mapping are workload-specific. Existing MoE worker-pool and event derivation did not transfer unchanged and must not be reported as reused.
- Background research: H100 first. Use Block-Sparse-Attention as query-major standard-attention reference, Flash Sparse Attention for pure KV-major partial reduction, and FlashMoBA's precomputed-pattern API as a hybrid performance oracle. FlashMLA remains a GB200 index-plane/roofline control because its MLA shapes do not match standard GQA.
- Next action: compile and smoke the pinned H100 oracles on one prerecorded 2K relation, then benchmark the same cached relation at sequence 16K.

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

### 2026-08-06 - TLTC-035 reproducible dense/MoE checkpoint

- Result: preserved the dense and distributed-MoE proof of life as a content-addressed snapshot and prepared annotated tag `shuttle-gb200-moe-v1`. The measured replay implementation is Shuttle `3dd61fad063bae54ac5e337d8f1657264011d6ff`; the tag points to the archival commit containing the raw results and documentation.
- Replay config: one batch-priority four-GB200 tray; 2048 tokens/rank, 384/96 global/local experts, top-6, `H=7168`, `I=3072`, BF16 with deterministic fixed-slot FP32 merge, 56 DeepEP SMs, and 256-row padding. MoK used 20 communication SMs, minibatch 2048, and macrobatch 65536.
- Replay results: selected concatenated/overlap 3.982976 ms, no-overlap 4.064928 ms, coarse materialization 4.434816 ms, separate gate/up overlap 4.069040 ms, and MoK 3.561696 ms/531.792 TFLOP/s. Shuttle is 1.1183 times the replay oracle.
- Revisions: MoK `3e1cf43ab93ad040afed52a45ab03cb490ffe4be`, ThunderKittens `1c3920d993404dd49a6d4c7267ea11d583bd5c68`, DeepEP `7febc6e25660af0f54d95dd781ecdcd62265ecca`, Torch 2.10.0+cu130, NVCC/PTXAS 13.0.88, CUDA runtime package 13.0.96, NCCL 2.28.9, and driver 595.71.05. The snapshot manifest also pins CODA/QuACK and FA3.
- Telemetry: application clocks were deprecated and unpinned. Every benchmark-boundary capture reported 1950 MHz SM and 3996 MHz memory clocks on all four GPUs; pre-benchmark idle SM clock was 120 MHz. Power limit was 1200 W and sampled draw ranged from 199.83 to 757.32 W.
- Correctness: every full-scale schedule variant and repeat was bitwise equal per rank. Eight independent source-ordered semantic fixtures passed with maximum absolute error `0.0001220703125`; all output and fixture hashes are retained.
- Selection evidence: the selected fingerprint is computed from the preserved schema-1 search records rather than hard-coded. The cache contains the 12-to-96 worker sweep, no-overlap phases, separate/concatenated A/B runs, confirmation runs, and the schema-2 coarse-materialization replay. Schema-2 is an out-of-sample confirmation and does not participate in selection.
- Validation: snapshot integrity tests verify every artifact digest, raw sample count, selected fingerprint, route content identity, output hashes, semantic fixture contents, and observed clock/power fields.
- Next action: test whether `RelationPlan`, its two orientations, task derivation, buffer lifetimes, and readiness machinery transfer unchanged to a MoBA-like routed sparse-attention workload.

### 2026-08-06 - TLTC-036 routed sparse-attention semantic and schedule slice

- Result: one generic ragged `RelationPlan` now drives exact query-major and KV-major selected-block attention. The KV-major path groups by KV block, computes structured partials, inverse-routes all three state fields, and merges in stable query-block/selected-slot order without atomics.
- State algebra: `AttentionPartial(max, sum_exp, weighted_value)` lives in the shared attention module rather than the routed-attention adapter. Its merge rescales both inputs to a common maximum, is associative over exact arithmetic, and retains explicit FP32 state.
- Structural plan: both orientations emit task roles, worker roles, counted readiness, bounded buffers, kernel boundaries, and explicit materialization bytes. At sequence 16384, block 128, and top-k 8, the deterministic coarse KV-major candidate exposes approximately 2.12 GB of partial-state materialization; query-major retains state internally.
- Backend boundary: `h100_routed_sparse_attention.py` converts the compiler relation into the pinned MIT Block-Sparse-Attention mask interface, checks selected query blocks against an independent exact reference, saves raw timing samples and output/relation hashes, and records both candidate dumps and complete GPU/toolchain metadata. This is the first executable query-major adapter; the generated KV-major candidate remains structural until an FSA-compatible physical adapter runs.
- Validation: all 74 tile-lifetime tests pass; source Pyrefly reports zero errors; all scoped repository checks pass.
- Infrastructure: the H100 holder had driver 595.71.05 but no system CUDA toolkit or PyTorch. Torch 2.7.1+cu128 and PyPI CUDA compiler components provided runtime headers/PTXAS but not the `nvcc` driver required by Block-Sparse-Attention. The exact pre-compilation failure is preserved. The holder was released after the fallback measurements.
- Index-plane follow-up: the first H100 artifact reported 4.558 ms for synthetic routing plus relation construction, which is slower than the sparse kernel itself. Replacing per-edge Python row assignment and exchange dictionaries with stable vectorized grouping reduced the local compiler's 16K relation-plan construction to 0.331 ms median; deterministic synthetic routing itself takes 0.536 ms (0.869 ms combined). Future GPU artifacts record these phases separately; neither is hidden inside kernel timing.
- H100 fallback: pinned SeerAttention `aba03e3...` at sequence 2048 measured 0.316752 ms and matched an independent source-ordered selected-block reference with maximum/mean/p99 errors `0.0078125`, `8.28e-5`, and `0.0009766`. At sequence 16384, 50 samples measured 2.388208 ms median/111.95 selected-work TFLOP/s versus 6.282496 ms for dense Torch causal GQA SDPA.
- Limitation: Seer scans every causal KV block and mask-tests in-loop, so it does not validate compact selected-edge traversal. Its oracle adapter also expands K/V from 8 to 32 heads outside timing, adding 201,326,592 bytes and 52.05 ms at 16K. Raw distributions, hashes, pins, telemetry, script, and BSA build log are under `benchmarks/artifacts/routed_sparse_attention_h100_v0`.
- Bounded KV-major candidate: process selected slots in ascending waves, grouping each wave by KV block. Each query appears at most once per wave, giving a unique state writer and deterministic FP32 online updates without atomics. At 16K this replaces 2,121,400,320 bytes of per-edge partial state with a 272,629,760-byte per-query online-state buffer (7.78x less capacity) and derives all per-slot/per-KV arrival counts from the relation.
- Next action: make the KV-major candidate executable through a compact-edge primitive, beginning with FSA on a CUDA-devel H100 environment or a narrowly extracted Triton/CUDA grouped-QK/PV body. Prune the current 2.12-GB partial-state boundary by bounded forwarding rather than treating it as the intended schedule.

### 2026-08-07 - TLTC-036 FSA KV-major oracle adapter

- Hypothesis: Shuttle's exact block-shared `RelationPlan` can drive FSA's KV-major selected-attention oracle through a thin index adapter, exposing the backend gap without introducing an attention-specific relation type.
- Commit Hash: uncommitted branch `research/shuttle-routed-sparse-attention`, based on `9ba3888cb0f91e2cf54f2a182927f13e769be2c6`.
- Source: FSA `7ff144fd7ff485dc4220d439f31cc1708b64fef3`; PyTorch 2.8.0+cu128; Triton 3.4.0; driver 595.71.05; one H100 80GB HBM3 under cluster-default unpinned clocks.
- Adapter: `h100_fsa_kv_major.py` reconstructs FSA int32 `[Hkv,T,topk]` indices from generic `RelationPlan` source, slot, destination, and validity fields. It repeats block-shared edges across query tokens and KV heads. FSA privately reconstructs the block-to-token orientation inside every timed public call and cannot accept Shuttle's grouped offsets or inverse map.
- Pristine-source failure: Triton rejects `lse_ptrs = (lse_ptr + pid_q_j * stride_lse_n,)` because `tl.load` receives a singleton tuple. The executable checkout removes only the trailing comma. The pinned head, dirty status, complete diff, and pristine traceback are preserved; the patch changes neither pointer arithmetic nor schedule/math.
- Config: BF16 causal self-attention, sequence 16384, Q/KV block 128, top-k 8, 32 query heads, 8 KV heads, head/value dimension 128, 996 relation edges, 10 warmups, 30 timed calls, and eight sampled query blocks for independent FP32 correctness.
- Command: `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/tmp/shuttle-fsa/lib/tile_lifetime/src:/tmp/shuttle-fsa/lib/tile_lifetime/benchmarks python /tmp/shuttle-fsa/lib/tile_lifetime/benchmarks/h100_fsa_kv_major.py --sequence-length 16384 --block-size 128 --selected-blocks 8 --query-heads 32 --key-value-heads 8 --head-dimension 128 --warmups 10 --repeats 30 --iterations 1 --planning-repeats 20 --correctness-blocks 8 --fsa-root /tmp/shuttle-fsa/vendor/fsa --shuttle-revision 9ba3888cb0f91e2cf54f2a182927f13e769be2c6 --json-output /tmp/shuttle-fsa/fsa_16k_b128_k8.json`.
- Relation identity: raw Boolean mask SHA256 `b2a57606e303f8af4da0c8002ddea162f86625725696bca7f18b8072a8143427`, identical to the Seer query-major artifact. Expanded FSA index SHA256 is `2c919c9ae176dc4b4e7c76bc464f835f21693ae12c518ad60a347fa97daf0deb`.
- Performance: the combined FSA public call measures 12.5392 ms median over 30 samples (`12.3622–13.0573` ms), or 21.322 selected-work TFLOP/s for 267,361,714,176 QK+PV FLOPs. The timing includes FSA's relation inversion, allocations, QK/PV partial work, and reduction; those phases are not separately exposed.
- Correctness: maximum/mean/p99 absolute error against the sampled source-ordered FP32 reference is `0.0207922`/`0.000164022`/`0.00120181`; allclose at 0.03 passes; output has no NaN or infinity. First and final timed outputs are bitwise identical with SHA256 `0d711cf008f91f857a2241737ebb122b2336c5c1d128e885f2db3b6b47ae53f5`.
- Planning: generic relation-plan median is 0.6411 ms. FSA index expansion median is 0.6226 ms, but its first four raw samples are `0.93–1.07` seconds; raw samples remain in the artifact.
- Memory: source-visible FSA partial/statistics buffers account for 111,225,856 bytes and internal inverse indices for 20,865,024 bytes; measured peak allocator increment is 431,091,712 bytes. The coarse Shuttle plan declares 2,121,400,320 bytes by materializing all query-head edge states. FSA reuses one-head buffers serially.
- Artifact: `lib/tile_lifetime/benchmarks/artifacts/routed_sparse_attention_fsa_h100_v0` contains the 2K correctness smoke, 16K result, pristine failure log, source patch, raw distributions, hashes, buffer accounting, and GPU telemetry.
- Interpretation: the generic relation is sufficient at the semantic adapter boundary. The public FSA API does not validate direct reuse of Shuttle's destination-major index plane. Bounded state consumption is required before the generated KV-major candidate is competitive in memory.
- Next action: compare the executable ascending-slot-wave kernel with FSA and Seer on the same 2K and 16K relations, then decide whether the next backend seam should accept compact destination groups or bounded slot waves.

### 2026-08-07 - TLTC-RSA-004 executable bounded slot waves

- Hypothesis: selected-slot waves can consume the generic destination-grouped relation with one deterministic FP32 state writer per query tile, avoiding atomics and all-edge partial materialization while remaining competitive enough to expose the next physical bottleneck.
- Commit Hash: uncommitted implementation on branch `research/shuttle-routed-sparse-attention`, based on `9ba3888cb0f91e2cf54f2a182927f13e769be2c6`. The base commit does not contain the benchmark. The artifact manifest pins exact per-stage hashes for all three executed scripts and every raw JSON.
- Command: `PYTHONPATH=/app/shuttle_slot_wave/lib/tile_lifetime/src:/app/shuttle_slot_wave/lib/tile_lifetime/benchmarks python lib/tile_lifetime/benchmarks/h100_kv_major_slot_waves.py --gpu --sequence-length 16384 --block-size 128 --selected-blocks 8 --query-tile-size 32 --warmups 10 --repeats 30 --correctness-blocks 8 --json-output artifacts/slot-wave-16k-b128-k8-final.json`.
- Config: BF16 causal GQA, sequence 2K and 16K, block 128, top-k 8, 32 query heads, 8 KV heads, dimension 128, one H100 80GB HBM3, driver 595.71.05, PyTorch 2.8.0+cu128, Triton 3.4.0, cluster-default unpinned clocks. The 16K relation has 996 edges and Boolean SHA256 `b2a57606e303f8af4da0c8002ddea162f86625725696bca7f18b8072a8143427`.
- Construction: launch one wave per selected slot. Sort each wave by KV block, then assign each edge/head/query-row tile to one Triton program. The program updates global FP32 maximum, denominator, and weighted-value state. No atomic or per-edge partial is used; one BF16 finalize follows eight waves.
- Result: selected M32/four-warps measures 0.502016 ms at 2K and 4.017344 ms at 16K. The 16K 30-sample range is 4.011936–4.027776 ms, or 66.552 selected-work TFLOP/s. It is 1.68 times the 2.388208-ms Seer query-major smoke and 0.32 times the 12.5392-ms FSA public-call adapter.
- Candidate selection: M16/four-warps measures 0.569072 ms at 2K; M32/four-warps measures 0.502016 ms; M64/eight-warps measures 0.660096 ms. M32 is selected. M64 is correct and deterministic but 31.5% slower than M32.
- Correctness: eight sampled 16K query blocks have maximum/mean/p99 absolute error `0.00783062`/`0.000124260`/`0.000865310` against the independent source-ordered FP32 reference. There are no NaNs or infinities. Repeated BF16 output is bitwise identical with SHA256 `7fee4b9c61ea72736f203fad5ab212f1f31d9178f750bc967f8c8db2eeb66917`.
- Memory: the schedule materializes 272,629,760 bytes of global per-query online state, 7.78 times less than the coarse 2,121,400,320-byte all-edge plan. It materializes zero edge partials and zero sequence-squared scores.
- Negative result: source-order/no-sort measures 4.018880 ms versus 4.017344 ms for KV-major sorting, but the fixture is already KV-monotone within every selected slot. Both schedules have identical edge arrays and output hashes, so this is a no-op rather than a cache-locality result.
- Interpretation: bounded deterministic state consumption transfers cleanly. Physical KV reuse does not yet transfer: destination-grouped edges remain independent CTAs and never stage one KV block for multiple query CTAs. Eight wave boundaries and global online-state traffic are the likely gap to query-major execution.
- Distributed decision: defer distributed sparse attention. Generic ownership/coalescing concepts transfer, but the DeepEP/MoE adapter cannot transport structured FP32 attention state or KV-block payloads without a new backend adapter. That would be significant new infrastructure rather than a small reuse test.
- Validation: all 74 tile-lifetime tests pass.
- Artifact: `lib/tile_lifetime/benchmarks/artifacts/routed_sparse_attention_h100_v0/slot_waves` contains raw distributions, outputs/input hashes, exact source/result digests, source evolution, telemetry, complete plan dumps, and all candidate points.
- Next action: use a deliberately non-monotone relation to measure grouping, then add actual shared KV staging or cluster-level reuse. Do not infer a sorting benefit from the canonical fixture.
### 2026-08-07 - TLTC-040 StatefulScan experiment begins

- Routed sparse attention is frozen at commit `fae336fd48143fb70a9be3257ac45223a710d675`; the new branch is `research/shuttle-stateful-scan`.
- Goal: represent Gated DeltaNet as a generic ordered state program and recover both recurrent decode and chunkwise prefill/training candidates without a GDN-specific semantic node.
- Existing asset: Marin already contains independent JAX recurrent and chunkwise GDN implementations with parity tests against Hugging Face. They will be used at the backend/reference boundary; the Shuttle semantic IR remains dependency-free.
- Sparse-attention carryover: first explain the 2.388-ms Seer versus 4.017-ms KV-major gap through state traffic, fusion, staging, and resource evidence. Do not do another tile-size sweep.
- Plan: add a minimal `StatefulScan` record and source-order NumPy executor, then add the weakest useful `ChunkAlgebra` contract and benchmark both physical forms on H100.
- Next action: finish primary-source and backend foraging, preserve the Seer delta analysis, and implement the ordered recurrence plus behavioral tests.

### 2026-08-07 - TLTC-041 Seer delta accounted for

- Frozen comparison: identical 996-edge 16K relation and 267.362-GFLOP selected work; Seer is 2.388208 ms and Shuttle slot waves are 4.017344 ms, a 1.629136-ms gap.
- The generated schedule reads and writes FP32 `(max, sum-exp, weighted-value)` state per edge. Its minimum state lifecycle is 4.92 GB; edge-wise Q traversal adds roughly 0.91 GB beyond a single query-major read.
- At 2.5--3.35 TB/s those bytes predict 1.74--2.33 ms. This is sufficient to explain the observed gap; relation metadata and eager launch overhead are smaller terms.
- Seer uses query-major M128 work units and can keep Q/state resident. Shuttle uses eight M32 waves and spills state between them. Neither result demonstrates explicit shared-memory reuse of K/V across queries.
- Caveat: Seer's 8-to-32-head K/V expansion is excluded from its timed kernel; Shuttle performs native GQA indexing.
- Decision: stop sparse tile tuning. A future iteration must use a non-monotone relation plus real cluster/shared-memory KV staging and a longer fused state lifetime.

### 2026-08-07 - TLTC-042 generic StatefulScan semantic slice

- Added a dependency-free ordered-state representation with stable logical axes, typed state/input/output values, Map/Contract/Fold body primitives, explicit numerical contracts, and optional chunk algebra.
- Added exact full-affine `(P,H)` composition: `(P2,H2) after (P1,H1) = (P2 P1, P2 H1 + H2)`. GDN and KDA use the same semantic record and candidate skeleton types.
- GDN recovery uses scalar head decay; KDA uses per-key-channel diagonal decay. Both source-order recurrences and exact affine chunk executions pass independent NumPy comparisons, nonzero-state continuation, tail chunks, and multiple decay regimes.
- The efficient physical summary is factored affine. GDN uses scalar-decay WY/UT factors; KDA needs diagonal-plus-low-rank factors. Compact factors are not closed under unrestricted tree composition, so the physical candidates retain an ordered inter-chunk scan.
- Local validation: 89 tile-lifetime tests pass before the added decay-regime cases; Levanter's 13 JAX GDN kernel tests also pass; Pyrefly reports zero errors.
- Frontend probe: ordinary JAX `lax.scan` lowers to `stablehlo.while` with private called recurrence bodies. The current flat StableHLO importer rejects this form. Structured while/private-function import is the next semantic-recovery requirement after physical benchmarking.
- Next action: benchmark FLA fused recurrent decode and FLA/FlashQLA chunk prefill on the pinned Qwen3-Next shape, preserving raw distributions and numerical error fields.

### 2026-08-07 - TLTC-043 StatefulScan H100 execution-form crossover

- Hypothesis: one generic ordered-state program can select recurrent decode or
  factored chunkwise execution based on measured shape behavior, without a
  Gated DeltaNet semantic kernel node.
- Hardware: one H100 80GB HBM3 from an eight-GPU low-priority holder, driver
  595.71.05, Torch 2.8.0+cu128, CUDA runtime 12.8, Triton 3.4.0, Python 3.12.13,
  700 W power limit, and cluster-default unpinned clocks.
- Workload: Qwen3-Next core with 16 Q/K heads, 32 value/state heads, K=V=128,
  BF16 inputs/output, FP32 persistent state, chunk 64, seed 1234, random decay
  in `[-0.1,0]`, and random beta.
- Backend: FLA `9c8e42e762fce087c27b673af4922795d9edb85e`.
  FlashQLA `050c6bbee9e03efbbfe41063fe4e33742c4a87cb` imported and passed its API test,
  but kernel JIT could not find `crt/host_config.h` in the split CUDA package
  set. The pin was not changed.
- Matched results: recurrent/chunk medians are 0.084960/0.515104 ms at T=64,
  0.321792/0.532176 ms at T=256, and 3.940768/0.510624 ms at T=2048. Chunk T=8192
  is 0.703536 ms. Each record retains 50 raw samples.
- Decode: recurrent T=1 medians are 0.073168, 0.070048, and 0.073728 ms for
  batches 1, 4, and 16.
- Correctness: recurrent output/state maximum absolute errors are `2.441e-4`
  and `5.364e-7`; chunkwise errors are `4.427e-4` and `5.543e-3`. Both outputs
  and final states repeat bitwise. The chunk candidate retains a
  `bounded_reassociation` contract.
- Interpretation: the execution-form crossover is a real compiler choice.
  This validates backend binding and selection, not synthesis of FLA's WY
  kernel. KDA fits the same semantic abstraction with diagonal-plus-low-rank
  factors and an ordered inter-chunk scan.
- Frontend blocker: ordinary JAX scan exports structured `stablehlo.while` and
  a private called body; narrow region/function import is still required.
- Artifacts: `lib/tile_lifetime/benchmarks/artifacts/stateful_scan_h100_v0`
  contains raw distributions, candidate dumps, hashes, pins, correctness, and
  exact failure logs. The benchmark script SHA256 is
  `de4b8746b4b8cdeabff254a037f9584fff4214a014f268d939a001c11ca5b36d`.
- Infrastructure: holder `/dlwh/dev-gpu-dlwh-stateful-gdn-i-20260807` was
  released; Iris reports `killed`, its pod is gone, and `dev_gpu status` shows
  no active session.
- Next action: test FSDP placement transitions, streamed communication, and
  tile lifetimes before serious XLA/Shardy integration.

### 2026-08-07 09:12 PDT - TLTC-044 generated affine recurrent skeleton

- Hypothesis: state-affine tensor analysis can instantiate one recurrent
  physical skeleton that remains valid when the decay domain, gate expression,
  or bounded update rank changes, without recognizing or calling a named
  GDN/KDA kernel.
- Commit Hash: uncommitted research state based on
  `fae336fd48143fb70a9be3257ac45223a710d675`. Executed source identity is pinned
  by SHA-256 in the artifact manifest.
- Local command:
  `uv run --project lib/tile_lifetime pytest -q` followed by
  `uv run --all-packages pyrefly check lib/tile_lifetime/src`.
- Local result: 1178 tests passed, 4 skipped, 41 deselected, and 5 expected
  failures; Pyrefly reports zero errors. The focused StatefulScan suite has 55
  passing tests, including 18 scalar/per-key, gate-expression, and rank
  recovery combinations plus 12 exact factored-chunk variants.
- H100 command: the full invocation and environment setup are preserved in
  `lib/tile_lifetime/benchmarks/artifacts/stateful_scan_generated_h100/README.md`.
  Representative production arguments are `B1,T64,H32,K=V128,R1`, scalar
  exponential decay, `block_v=32`, 10 warmups, and 50 samples.
- Environment: NVIDIA H100 80GB HBM3, driver 595.71.05, CUDA runtime 12.8,
  Torch 2.8.0+cu128, Triton 3.4.0, Python 3.12.13, 700 W, unpinned application
  clocks. FLA and FlashQLA were absent and asserted unimportable.
- Result: scalar-rank-1 medians for `block_v` 8/16/32 are
  0.157120/0.149424/0.138544 ms. Per-key-rank-1 is 0.138000 ms and simultaneous
  scalar-rank-2 is 0.183376 ms at `block_v=32`.
- Correctness: all outputs/states are finite and repeat bitwise. Maximum BF16
  output error is `2.441e-4`; maximum FP32 state error is `1.863e-8`. The rank-2
  implementation computes all residuals from one decayed state before applying
  their summed correction.
- Interpretation: this is a synthesized recurrent core, not oracle-backed
  execution. Generic expression recovery supplies physical factors to one
  generated Triton skeleton; no complete architecture kernel is called. The
  result excludes producer-map preparation and does not yet cover the generated
  ordered-chunk path or full GDN/KDA layer.
- Artifact:
  `lib/tile_lifetime/benchmarks/artifacts/stateful_scan_generated_h100`
  contains 20/50-sample distributions, deterministic hashes, source hashes,
  environment, commands, manifest, and validated checksums.
- Infrastructure: holder `/dlwh/generated-affine-scan-20260807` was released;
  Iris reports `killed`, no matching pod remains, and no active dev-GPU session
  exists.
- Decision: replace the stale TLTC-043 FSDP next action. Generate the ordered
  factored-chunk physical path and include factor-preparation/materialization
  costs before moving to FSDP.

### 2026-08-07 10:45 PDT - TLTC-045 natural StableHLO StatefulScan recovery

- Added a structured StableHLO importer separate from the flat feed-forward
  path. It preserves `stablehlo.while` condition/body regions, uses
  function-local stable value identifiers, and imports only private functions
  transitively reached through `func.call`.
- Added an ordinary JAX `lax.scan` reference whose source is only tensor/state
  math: decay, two contractions, bounded-rank correction, state update, and
  readout. The exported program contains the normal JAX while, dynamic-index,
  recurrence-call, and dynamic-update structure.
- Recovery identifies the carried state and emitted scan value from dataflow,
  resolves source input names, derives logical-axis equivalence from
  StableHLO broadcast/transpose/dot dimension numbers, rebuilds generic tensor
  expressions, and invokes the existing affine-state analysis.
- Mutation evidence: scalar decay with rank one and per-key decay with rank two
  recover one `DIAGONAL_PLUS_LOW_RANK` family with maximum ranks one and two.
  They produce the same recurrent/chunk candidate generator without a GDN/KDA
  symbol match.
- Numerical evidence: random BF16 rank-two source inputs execute through the
  generated recurrent factor skeleton and match the ordinary JAX scan output
  and final state within `2e-5` absolute/relative tolerance.
- Validation command:
  `uv run --frozen --package marin-tile-lifetime --group test pytest lib/tile_lifetime/tests/test_stateful_scan.py -q`.
  Result: 62 passed. Targeted Pyrefly reports zero errors.

### 2026-08-07 10:45 PDT - TLTC-046 generated chunk gap isolated

- Generated chunk execution at `B1,T2048,H32,K=V=128`, scalar rank one,
  chunk 16 and value block 32 measures 0.340032 ms, versus the pinned FLA chunk
  oracle at 0.510624 ms.
- Compiler-owned BF16 summary preparation measures 0.665568 ms. Combined
  latency is 0.984496 ms, or 1.928 times FLA, so this path does not meet the
  1.2-times goal.
- The materialized generic summary is 84,410,368 bytes. Preparation and that
  materialization, not the ordered scan executor, are the measured blocker.
- Primary output/final-state maximum errors are `4.883e-4`/`2.840e-4`; a
  per-key rank-two mutation is also deterministic with `4.883e-4`/`2.668e-4`
  maximum errors.
- Artifact:
  `lib/tile_lifetime/benchmarks/artifacts/stateful_scan_generated_chunk_h100`
  contains 50-sample distributions, deterministic hashes, source pins, and
  validated checksums. The H100 holder was released and verified killed.
- Decision: fuse summary preparation with consumption or forward a smaller
  summary. Do not spend another iteration on chunk-size/value-block tuning.

### 2026-08-07 10:49 PDT - TLTC-047 clean distributed MoE return and merge

- Hypothesis: replacing DeepEP's combined reverse-transport/reduction boundary
  with payload-only return plus a generated rank-ordered fold will remain
  within 1.2 times the pinned MoK oracle.
- Commit Hash: executed dirty source based on
  `4fba36752bdbfd28ad9a0ea8dee121bb382b21c9`; the exact benchmark and probe
  sources are preserved with SHA256
  `662cc027286f8363dc08fe813d54e4842748c97ba4d86e989b7d48593e93030f`
  and `d76a0878d9e23d04230610db2a696e98bf11d7231b485605271731a4713b33c5`.
- Command: `/tmp/mok-route-env/bin/torchrun --standalone --nproc-per-node=4
  lib/tile_lifetime/benchmarks/backends/gb200_deepep_mok_distributed.py
  --route-fixture
  /tmp/mok_routes_t2048_e384_k6_seed1234_torch2.10-reserialized.npz
  --probe-extension
  /tmp/mok-gmm-probe-clean-merge-cu130/_mok_gmm_probe.cpython-312-aarch64-linux-gnu.so
  --mok-root /tmp/mixture-of-kittens-fixture-source --deepep-root
  /tmp/DeepEP --shuttle-revision
  4fba36752bdbfd28ad9a0ea8dee121bb382b21c9+clean-merge-dirty
  --clock-policy cluster_default_unpinned --deepep-sms 56
  --gate-up-layout concatenated_e_2i_k --warmup 10 --iterations 30`.
- Config: four GB200 ranks; 2,048 tokens/rank; 384 global and 96 local
  experts; top-6; hidden 7,168; intermediate 3,072; BF16; concatenated W13;
  DeepEP dispatch with 56 communication SMs; payload-only
  `all_to_all_single` return; generated ascending-owner FP32 merge plus shared
  add; no atomics. Driver 595.71.05, Torch 2.10.0+cu130, CUDA toolkit 13.0.2,
  NCCL 2.28.9, 1,200-W limit, cluster-default clocks observed at 1,950-MHz SM
  and 3,996-MHz memory during benchmark boundaries.
- Result: the first clean overlap median is 4.082608 ms over 30 samples,
  1.1463 times the frozen 3.561696-ms MoK replay. The confirmation is
  4.142576 ms and 1.1631 times the replay. Their median of medians is
  4.112592 ms, or 1.1547 times the replay. Both meet the 1.2-times target.
- Phase result: first-run routed compute after dispatch is 3.555584 ms,
  generated shared expert is 0.242464 ms, payload return plus generated merge
  is 0.365168 ms, and the matching clean sequential region is 4.229424 ms.
  The DeepEP combine component is 0.271072 ms, making the clean return/merge
  boundary 0.094096 ms slower in isolation.
- Correctness: generated output is bitwise equal to the historical
  DeepEP-combine control on every rank, repeats bitwise, remains finite, and
  uses no atomic accumulation. All four semantic fixture SHA256 values repeat
  exactly across the two complete runs.
- Interpretation: this removes the semantic-reduction contamination from
  DeepEP `combine`. The result is a synthesized distributed schedule at a
  supplied-route boundary. The MoK-derived grouped GEMM is an allowed
  segmented-contraction skeleton. Router/top-k and index-plan construction
  remain outside timing.
- Artifact:
  `lib/tile_lifetime/benchmarks/artifacts/gb200_moe_clean_merge_v0` contains
  both raw distributions, logs, exact executed sources, package/hardware pins,
  route and semantic fixtures, manifest, and validated SHA256 checksums. The
  original copied bundle is under
  `scratch/shuttle-generic-results/clean-merge-20260807`.
- Infrastructure: holder `/dlwh/dev-gpu-dlwh-shuttle-clean-merge` was released;
  Iris reports `killed`, `dev_gpu status` has no active session, and the pod is
  gone.

### 2026-08-07 12:54 PDT - TLTC-048 natural-source compiler paths connected

- Added `compile_stablehlo_expert_parallel_region`, connecting the frozen
  ordinary JAX MoE StableHLO fixture through semantic recovery to the generic
  expert-parallel plan. The public test observes payload-only forward/reverse
  transports and the generated deterministic merge rather than testing a
  private helper.
- Added `compile_routed_streaming_attention_candidates`, which validates
  relation block extents, head mappings, contraction dimensions, and tile
  contracts before deriving query-major and KV-major schedules from a generic
  `StreamingAttentionProgram`.
- Mutation coverage applies both causal-only and causal-plus-tanh-softcap score
  maps through the same relation scheduler. Neither path dispatches on a
  workload kernel name.
- Validation: all 161 package tests pass; targeted Pyrefly for the new MoE and
  relation-planning paths reports zero errors; `git diff --check` passes.
- The package-wide Pyrefly check initially found 67 errors only in the
  concurrently active optional SM90 CuTe extraction: missing local
  CUDA/CUTLASS/QuACK/FA3 modules and copied upstream nullable annotations. The
  extraction was moved from the typed core package into `backends/h100`; the
  complete source-and-test Pyrefly check now reports zero errors.
- Renamed the older opaque MoK compiler entry point and plan node to
  `compile_mok_oracle_region` and `OpaqueMoKOracleSkeleton`. The clean generic
  expert-parallel path remains `compile_stablehlo_expert_parallel_region`;
  package users can no longer mistake the complete MoK call contract for a
  synthesis result.

### 2026-08-07 - TLTC-049 clean-synthesis acceptance frozen

- Added the canonical acceptance specification at
  `.agents/projects/tile_lifetime_compiler/clean_synthesis_acceptance.md`.
- A proof point now requires all six gates together: ordinary high-level
  frontend, named-semantic erasure into generic algebra, generic physical code
  generation, no opaque semantic kernel, numerical correctness, and latency no
  greater than 1.2 times an equivalent expert oracle.
- Clarified the dense RMS result: its generated `SCALE_ROW -> BF16 CONVERT ->
  GEMM` A-fragment path is primitive-driven and clean at the physical layer,
  but `_find_rms_region` and `_rms_plan` remain a named macro rewrite. Dense
  therefore still fails the specification's semantic-erasure gate.
- Revalidated the active package while GPU work continued: 166 tests pass,
  Pyrefly reports zero source/test errors, and the scoped diff check passes.

### 2026-08-07 - TLTC-050 acceptance targets made boundary-exact

- Made semantic-name erasure a machine-checked stage boundary with a required
  lowering report and a structural ban on workload-specific schedule dispatch
  keys.
- Required a benchmark-boundary manifest and symmetric inclusion of frontend,
  indexing, layout conversion, transport, and postprocessing work.
- Reclassified MoK's 3.613-ms result and Seer's 2.388-ms result as provisional
  post-routing/kernel checkpoints. Final acceptance uses newly measured
  natural-program oracle boundaries with identical included work.
- Added helper-level lineage requirements for FlashAttention-derived code.
  Removing the public entry point is insufficient if an internal softmax,
  attention-state, or mask helper still hides the semantic Fold/Map/domain
  program.
- Added the matched benchmark boundary as a separate required column in the
  four-workload acceptance matrix.

### 2026-08-07 - TLTC-050 generated routed SM90 streaming backend

- Added a query-major SM90 emitter that consumes
  `RoutedStreamingAttentionCompilation`, derives compact block lists from the
  generic `RelationPlan`, and instantiates the compiler-owned
  QK/online-state/PV skeleton. It does not call Seer, FSA, or another named
  sparse-attention entry point.
- The physical skeleton uses TMA global-to-shared K/V movement and a
  three-stage circular shared-memory pipeline. It does not yet implement TMA
  multicast or a physical KV-major cluster schedule.
- At causal BF16 GQA `S=16384`, `Hq/Hkv=32/8`, `D=128`, block 128, top-8, a
  deliberately non-monotone 996-edge relation measures 0.491984 ms over 30
  samples. It is deterministic and has sampled FP32-reference maximum/mean
  absolute error 0.008165/0.000179.
- The same generated path on the historical Seer relation measures 0.492512
  ms versus the preserved Seer baseline at 2.388208 ms, 4.85 times faster.
  The main known reasons are selected-edge rather than dense-causal-block
  traversal and native GQA rather than K/V head expansion.
- A mutation changes the relation, scale from `2^-3.5` to 0.125, and adds tanh
  softcap 16 through the same generator. It measures 0.618176 ms, remains
  deterministic, and has 0.007996 maximum sampled-reference error.
- Backend-neutral query-major and KV-major executions agree for a deliberately
  non-monotone relation. Only query-major is physically executable in this
  checkpoint; the current unbounded KV-major plan would materialize about
  2.12 GB of partial state at 16K.
- Artifact:
  `lib/tile_lifetime/benchmarks/artifacts/generated_routed_streaming_attention_h100_v0`
  contains all raw samples, hashes, plans, generality accounting, and verified
  source/data manifests. The dense generated SM90 artifact now also has a
  verified manifest.
- Acceptance status is partial: physical generation, mutation, numerical,
  determinism, and performance gates pass, but the benchmark still constructs
  the generic relation and streaming program directly. Natural MoBA-like JAX
  frontend recovery and named-semantic erasure remain open.

### 2026-08-07 - TLTC-052 matched StatefulScan target passed

- Replaced the naive 64-row triangular solve with a generic four-by-four block
  inverse over 16-wide tensor-core-friendly subblocks. The complete generated
  path remains `AffineIntraChunkPrepare -> AffineStateScan -> AffineReadout`
  and contains no FLA/GDN/KDA call.
- The first blocked result measured 0.579872 ms against the historical
  0.510624-ms FLA record, apparently 1.136 times the oracle. The stricter
  acceptance audit rejected that ratio because the denominator came from a
  different process and input boundary.
- Added a matched delta-rule benchmark mode. Pinned FLA and Shuttle receive the
  identical BF16 Q/K/V, FP32 log-decay/beta/initial-state boundary, with Q/K
  normalization disabled and query scale one. FLA is loaded only by the oracle
  harness.
- The first interleaved matched run exposed 0.595696 ms generated versus
  0.434368 ms FLA, or 1.371 times. Profiling assigned 68.99% of preparation
  GPU time to `_affine_transform_factors`, which recomputed the 64-token
  diagonal prefix for four K=32 tiles.
- Increasing the generic preparation K tile to 64 reduced preparation from
  0.407440 to 0.281424 ms. The final 50-sample interleaved result is 0.466752
  ms generated versus 0.420528 ms FLA, or **1.1099 times**, passing the matched
  1.2-times target.
- Output maximum/mean error versus FLA is `4.8828125e-4`/`5.270477e-5`; final
  state error is `3.154259e-4`/`4.448347e-5`. Scalar/per-key decay crossed with
  rank-one/rank-two mutations remains finite and bitwise deterministic.
- Raw distributions, StableHLO, source hashes, hardware telemetry, mutation
  records, manifest, and validated checksums are under
  `benchmarks/artifacts/stateful_scan_affine_pipeline_h100_v0`.
- The one-H100 batch-priority holder was released and its temporary fresh-Iris
  worktree was removed. Formal integration with the shared name-erasure report
  remains the only open StatefulScan acceptance bookkeeping item.

### 2026-08-07 - TLTC-053 natural routed-attention frontend

- Added an ordinary JAX selected-attention source with a metadata Contract,
  causal block-domain predicate, top-k, selected K/V gathers, QK, normalized
  exponential, and PV. The accepted compiler test uses a frozen StableHLO
  fixture rather than a manually assembled relation or streaming program.
- Recovery erases the source into generic `RelationSelectionProgram`,
  `RelationPlan`, `Contract`, `Map`, `DomainRestriction`, and `Fold` structure.
  The shared `SemanticErasureReport` validator runs before physical candidate
  enumeration and reports no named scheduling keys.
- Runtime metadata mutation changes the selected relation without changing the
  generated streaming body or the query-major/KV-major candidate family. The
  generic online reference matches the natural JAX source within 0.016 maximum
  and 0.002 mean absolute error.
- Added `h100_natural_routed_streaming_attention.py`, which includes the router
  Contract and GPU top-k/index forwarding in the timed generated path. The
  benchmark-boundary manifest explicitly rejects the old 2.388208-ms Seer
  number as unmatched.
- Existing 0.491984-ms generated SM90 measurements remain backend-only until
  the new harness and an equivalent expert boundary are measured. The complete
  package passes 176 tests, the source/test type check reports zero errors, and
  touched Python files pass Ruff.

### 2026-08-07 - TLTC-053 clean-synthesis targets revised

- Kept 1.20 times a matched natural expert boundary as the completion gate for
  every workload and added 1.10 times as a reported stretch target. Stretch is
  not required to complete the milestone.
- Dense must pass both established primary sequence lengths rather than one
  favorable shape. The completion thresholds are 1.7472 ms at sequence 2,048
  and 3.6096 ms at sequence 4,096; the stretch thresholds are 1.6016 and
  3.3088 ms.
- Replaced the rounded MoK component checkpoint with the frozen supplied-route
  replay: 3.561696 ms, giving provisional completion/stretch checkpoints of
  4.274035/3.917866 ms. Final MoE acceptance still requires a newly measured
  matched natural-program boundary.
- Left sparse attention without a final absolute target because the current
  Shuttle and Seer timings do not execute equivalent work. Its target freezes
  only after both paths share the natural router, relation, GQA representation,
  and output boundary.
- Froze the matched StatefulScan oracle at 0.420528 ms, with completion/stretch
  targets of 0.504634/0.462581 ms. The generated 0.466752-ms result passes the
  completion gate at 1.1099 times and narrowly misses stretch.

### 2026-08-07 - TLTC-054 StatefulScan name-erasure gate closed

- Added a shared `SemanticErasureReport` to the natural
  `stablehlo.while` compiler path. The report lowers the source to generic
  `Scan`, `Map`, and `Contract` structure and derives scheduling keys from
  ordered extent, state rank, primitive arity, affine transition structure,
  and numerical policy.
- The validator runs before physical candidate enumeration and rejects both
  workload-named scheduling keys and reports that do not match the recovered
  generic program.
- The focused StatefulScan suite passes 64 tests, including scalar/per-key and
  rank-one/rank-two natural-source mutations through the same report and
  generator.
- Preserved the exact T=2048 erasure report in
  `benchmarks/artifacts/stateful_scan_affine_pipeline_h100_v0`; it reproduces
  from the stored StableHLO and its checksum validates. The artifact status now
  records all StatefulScan acceptance gates as passing.

### 2026-08-07 - TLTC-055 dense clean-synthesis path passes both shapes

- Replaced the named RMS macro scheduling path with semantic erasure from the
  ordinary JAX StableHLO fixture into 36 generic `Map`, `Contract`, `Fold`, and
  `DomainRestriction` operations. The shared validator runs before the generic
  eight-skeleton planner enumerates candidates.
- Generated Contract scalar/tile ASTs now emit residual/RMS partials,
  source-ordered row scaling, delayed scaling, RoPE, and SwiGLU arithmetic
  directly around the generic QuACK/CuTe mainloop. The measured attention path
  uses Shuttle's generated SM90 QK/online-Fold/PV skeleton, not official FA3.
- At S=2,048, prologue/delayed policies measure 1.6872/1.6339 ms versus the
  1.4561-ms oracle, or 1.159/1.122 times. At S=4,096 they measure
  3.4148/3.3848 ms versus 3.0080 ms, or 1.135/1.125 times. Every candidate
  passes the 1.20-times ratio at both required shapes.
- The same generator changes pairwise SwiGLU to pairwise product from a
  semantic AST mutation. Raw distributions, generated sources, dependency
  lineage, and hashes are under
  `benchmarks/artifacts/dense_clean_synthesis_h100_20260807`.
- The captures predate the newly frozen two-independent-run policy. One
  counterbalanced confirmation per shape remains before final statistical
  acceptance; older named-hook/FA3 runs cannot serve as that confirmation.

### 2026-08-07 - TLTC-056 natural routed-attention matched target frozen

- Matched ordinary-JAX boundaries now include router metadata contraction,
  causal restriction, sorted top-k and index forwarding, selected exact
  attention with native GQA, and BF16 output on both Shuttle and pinned MIT
  Block-Sparse-Attention paths.
- Two independent 30-sample captures have pooled medians 0.584304 ms for
  generated query-major Shuttle and 1.424720 ms for the oracle, or 0.410118
  times. The target is frozen at 1.709664 ms for completion and 1.567192 ms for
  stretch. Output maximum/mean difference is 0.00390625/0.0000652, and both
  outputs repeat bitwise.
- The oracle is an SM80-style implementation compiled for SM90; it is the
  strongest matching buildable oracle currently wired to the boundary, not a
  Hopper state-of-the-art claim.
- Both captures measured generated execution before the oracle and therefore
  predate the new counterbalanced-launch rule. Preserve their denominator as
  the frozen target, but require a counterbalanced confirmation before final
  statistical acceptance.
- Proof C remains structurally open: the KV-major candidate must execute with
  bounded right-resource K/V staging and deterministic online-state return and
  merge. The structural 2.12-GB edge-state materialization does not pass.

### 2026-08-07 16:52 PDT - TLTC-057 natural MoE boundary accepted

- Hypothesis: vectorizing the generic deterministic route-slot and rank Folds
  over BF16 pairs will move the matched natural-source MoE boundary below the
  1.20-times completion target without changing source order or introducing
  semantic atomics.
- Commit hash: Shuttle base
  `4fba36752bdbfd28ad9a0ea8dee121bb382b21c9`; measured source remained a
  dirty prototype and is snapshotted in the artifact.
- Command: four-rank `torchrun` of
  `benchmarks/backends/gb200_deepep_mok_distributed.py` with
  `--routing-source natural_stablehlo`, the primary T2048/H7168/I3072/E384/K6
  StableHLO fixture, `--deepep-sms 56`, concatenated W13, 10 warmups, 30
  samples, and launch orders `shuttle_first` then `oracle_first`. Exact argv
  and environment are stored in both accepted JSON records.
- Config: 4 x GB200, BF16, cluster-default unpinned clocks, observed 1950 MHz
  SM and 3996 MHz memory, 1200 W power limit. MoK is
  `3e1cf43ab93ad040afed52a45ab03cb490ffe4be`; DeepEP is
  `7febc6e25660af0f54d95dd781ecdcd62265ecca`; ThunderKittens is
  `1c3920d993404dd49a6d4c7267ea11d583bd5c68`; CUDA is 13.0.88 and driver is
  595.71.05.
- Result (`replicated`): the two runs measure 4.126384/4.140336 ms for Shuttle
  and 3.645056/3.642048 ms for matched MoK. Pooled medians over 60 samples are
  4.137120 and 3.645056 ms, or 1.134995 times. A preserved scalar-Fold pair
  pooled to 1.201725 times. The BF16x2 change reduces Shuttle latency by
  0.227104 ms.
- Correctness: all eight accepted per-rank relation checks are exact with zero
  overflow. Generated outputs repeat bitwise. Maximum/mean errors against MoK
  are `0.0001220703125` and at most `2.667012722668005e-06`. The generated
  relation and Fold source has zero semantic atomic operations.
- Interpretation: distributed MoE passes the clean-synthesis completion gate
  at the natural JAX/StableHLO boundary. The accepted Shuttle path does not use
  MoK forward or DeepEP semantic combine. It misses the 1.10-times stretch
  target by 0.034995 ratio points and trails MoK by 0.492064 ms.
- Artifact:
  `lib/tile_lifetime/benchmarks/artifacts/gb200_moe_natural_boundary_v0`.
- Next action: preserve this result; do not spend another MoE tuning iteration
  before the open dense confirmation and sparse KV-major structural proof.

### 2026-08-07 17:00 PDT - TLTC-058 dense counterbalanced boundary accepted

- Added raw-sample JSON output to the hand-composed dense oracle harness and a
  selectable FlashAttention-4 CuTe expert body. Named QuACK/CODA and CuTe
  attention remain oracle-only; the generated path continues to use erased
  algebra, generated Contract ASTs, and generated SM90 streaming attention.
- Config: one H100 80GB, driver 595.71.05, 700 W, Torch
  2.14.0.dev20260807+cu130, CUTLASS DSL 4.6.1, FlashAttention-4 4.0.0b16,
  QuACK `84ef91df9bec87c7e4938517234fafb07ef844dd` plus the recorded FP32 row-scale
  patch. Application clocks were unpinned.
- Protocol: two independent captures at each of S=2,048 and S=4,096, ten
  warmups, 30 samples, ten region iterations per sample. Run 1 launches the
  generated process before the oracle process; run 2 reverses the order.
- Pooled S=2,048 medians are 1.705818 ms source-ordered, 1.650502 ms delayed,
  and 1.523838 ms oracle, or 1.119422/1.083122 times.
- Pooled S=4,096 medians are 3.478322 ms source-ordered, 3.390837 ms delayed,
  and 3.253411 ms oracle, or 1.069131/1.042240 times.
- Both policies pass completion at both required shapes. Delayed scaling passes
  the 1.10-times stretch target at both; source-ordered scaling passes stretch
  at 4,096 and misses at 2,048. Every generated output hash repeats within and
  across captures.
- Artifact:
  `lib/tile_lifetime/benchmarks/artifacts/dense_clean_synthesis_h100_counterbalanced_v1`.
  Every file is covered by a verified portable checksum manifest.
- The H100 holder was released immediately after copying the evidence; the
  temporary current-main Iris worktree used only for reservation was removed.

### 2026-08-07 18:12 PDT - TLTC-059 sparse counterbalance and bounded KV-major

- Repeated the natural sparse-attention boundary with two independent
  30-sample paired captures. Every pair alternates generated→oracle and
  oracle→generated, and the JSON preserves every warmup and sample order.
- Pooled medians are 0.617584 ms for generated query-major Shuttle and
  1.423632 ms for pinned MIT Block-Sparse-Attention, or 0.433809 times. The
  1.424720-ms frozen oracle target was not moved. Both paths repeat bitwise;
  maximum/mean output differences remain 0.00390625/0.0000652.
- Added a generic `BoundedKVReusePlan`. Each selected-slot wave groups by the
  right resource, splits incident consumers into fixed-capacity tasks, and
  proves one query-state writer per wave. Readiness counts now derive from
  physical tasks rather than relation edges.
- Generated and executed a CUDA H100 structural skeleton. One CTA stages a
  KV-head block into dynamic shared memory and reuses it for a bounded query
  group, performs the QK/normalized-exponential/PV Fold, and writes directly to
  the source-owned global state without atomics or edge partials.
- At S=16,384, block 128, top-8, Hq/Hkv=32/8, D=128, capacity two covers 996
  non-monotone edges with 671 tasks. It uses 65,536 bytes of shared K/V per CTA,
  272,629,760 bytes of global state, and zero edge-partial bytes. Output is
  deterministic with 0.015625/0.00006397 maximum/mean difference from
  query-major.
- The first structural kernel measures 107.879105 ms versus 0.574656 ms
  query-major. Capacity one uses the same generated source, creates 996 tasks,
  produces bitwise-identical output, and measures 103.355042 ms. The current
  gap is CUDA-core QK/PV, global state traffic, and the lack of TMA/WGMMA or
  cluster multicast, not relation planning or correctness.
- Artifact:
  `lib/tile_lifetime/benchmarks/artifacts/natural_routed_sparse_attention_h100_matched_v0`.
  The H100 holder remains live temporarily for a requested StatefulScan
  confirmation capture.

### 2026-08-07 18:22 PDT - TLTC-060 clean-synthesis milestone audit

- Repeated the matched StatefulScan boundary in two independent captures on
  one H100 with pinned FLA `9c8e42e`, Torch 2.8.0+cu128, and Triton 3.4.0.
  Each run contains ten warmup pairs and 50 measured pairs; launch order
  alternates within the run and the initial implementation reverses between
  runs.
- Pooled medians are 0.465824 ms for generated Shuttle and 0.424304 ms for
  FLA, or 1.097854 times. This passes both the 1.20-times completion target and
  the 1.10-times stretch target. Output/final-state maximum errors remain
  `4.883e-4`/`3.154e-4`, and every generated mutation repeats bitwise.
- The accepted StableHLO hash is
  `417852499eed3f1dcc4b270d73c3922b1c0a5e5071951c78879319d41a65730a`;
  the prior single-capture export remains labeled as legacy evidence.
- Final package validation passed 179 tests, Pyrefly with zero errors, and
  Ruff. Dense, natural MoE, routed sparse attention, and StatefulScan artifact
  checksum manifests all verify.
- The final acceptance matrix records worst required-shape dense ratio
  1.119422 times, natural MoE 1.134995 times, routed sparse attention 0.433809
  times, and StatefulScan 1.097854 times. All four workloads satisfy natural
  frontend, name erasure, generic physical generation, mutation, correctness,
  matched-boundary, and 1.20-times gates.
- Released the last H100 holder at 18:22 PDT and removed its temporary Iris
  worktree.

### 2026-08-07 19:10 PDT - TLTC-061 sparse Hopper-oracle audit

#### Background research brief

- Effort: medium. Stop rule: inspect the current pinned repositories, public
  interfaces, semantic references, tests, benchmark boundaries, active kernel
  traits, and relevant negative evidence until the oracle ordering is
  decisive. FlashMoBA, MIT Block-Sparse-Attention, FlashMLA sparse prefill, and
  FSA/NSA all converged without requiring a broader literature sweep.
- Question: which current expert implementation can provide an exactly matched
  H100 acceptance denominator for Shuttle's natural BF16 causal GQA workload
  at S=16,384, block 128, top-8, Hq/Hkv=32/8, and D=128?
- Current Shuttle boundary: FP32 block-metadata Contract, causal block-domain
  restriction, sorted top-k shared by every token and head in one query block,
  exact selected attention, and BF16 output. QKV and output projections are
  outside both paths.

#### Evidence map

- FlashMoBA at
  [`39d9ac043b271d046a2181a9991e99a26b67bca1`](https://github.com/mit-han-lab/flash-moba/tree/39d9ac043b271d046a2181a9991e99a26b67bca1)
  is the primary payload oracle. Its precomputed-relation interface exactly
  supports BF16, D=128, native 32:8 GQA, causal token masking, block 128, top-8,
  scale, and packed BF16 output. The interface is documented in the
  [pinned implementation](https://github.com/mit-han-lab/flash-moba/blob/39d9ac043b271d046a2181a9991e99a26b67bca1/flash_moba/flash_moba_interface.py#L578-L685).
  Shuttle can reorient each `query block -> KV block` edge into FlashMoBA's
  per-head, KV-column-major sorted query-row lists without changing the
  selected set.
- FlashMoBA's complete wrapper is not an exact natural-program oracle. It
  scores every query token/head against mean-pooled actual K blocks, always
  includes the current causal block, and can select a different relation per
  head. Shuttle scores explicit block metadata once per query block, shares the
  relation across heads, and does not force the current block after the first
  top-k-saturated blocks. The per-head behavior is confirmed in
  [FlashMoBA issue 8](https://github.com/mit-han-lab/flash-moba/issues/8).
  Therefore the fair full boundary is the common Shuttle router plus a generic
  relation reorientation plus `flash_moba_attn_varlen_func`; the native
  FlashMoBA top-k is diagnostic only.
- FlashMoBA's published H100 plot is not a reusable number. The published
  workload is batch 2, FP16 MHA with 16 heads, and its own token/head router;
  Shuttle needs a local BF16 GQA run. The physical kernel is current and
  H100-measured but remains FA2/SM80-style MMA plus `cp.async`, not a
  WGMMA/TMA Hopper-native body.
- MIT Block-Sparse-Attention current HEAD/tag v0.0.2 is exactly
  [`49d6c39e4dc0303442cda3bb758b3925d4399c49`](https://github.com/mit-han-lab/Block-Sparse-Attention/tree/49d6c39e4dc0303442cda3bb758b3925d4399c49),
  with CUTLASS `a75b4ac483166189a45290783cb0a18af5ff0ea5`. This is already the
  revision used for Shuttle's local 1.423632-ms H100 measurement. The December
  update added SM90/SM100 build compatibility, but the active traits still use
  [SM80 MMA and cp.async](https://github.com/mit-han-lab/Block-Sparse-Attention/blob/49d6c39e4dc0303442cda3bb758b3925d4399c49/csrc/block_sparse_attn/src/kernel_traits.h#L15-L34).
  It remains an exact semantic and local-H100 secondary reference, not the
  strong Hopper performance gate. Published figures remain A100-only.
- FlashMLA sparse prefill at
  [`15f13e5030374295491c5ce31b02d7e63a7772c6`](https://github.com/deepseek-ai/FlashMLA/tree/15f13e5030374295491c5ce31b02d7e63a7772c6)
  is rejected for this row. Its sparse SM90 path requires MLA/MQA semantics
  with Hkv=1, Dqk=512/576, Dv=512, and shared latent K/V, not ordinary 32:8
  GQA D=128. It may become a separate DSA/MLA control.
- Full FSA at
  [`7ff144fd7ff485dc4220d439f31cc1708b64fef3`](https://github.com/Relaxed-System-Lab/Flash-Sparse-Attention/tree/7ff144fd7ff485dc4220d439f31cc1708b64fef3)
  is NSA semantics: compressed routing, selected attention, an independent
  sliding-window branch, learned gated merge, and projections. It must be a
  separate workload. Its selected-attention subkernel is structurally
  adaptable but was already measured at 12.539 ms for 16K and should remain a
  secondary control.

#### Ranked experiments

1. Build the pinned FlashMoBA SM90 target and validate a relation that omits
   the current KV block against the same exact sampled semantic reference.
2. Measure payload-only `flash_moba_attn_varlen_func` with a cached identical
   relation. Sweep physical query grouping `{128,256,512,768,1024}` while
   keeping logical block size 128.
3. Measure the matched whole boundary: identical natural router, top-k,
   generic relation reorientation, and attention. Record relation conversion
   separately and counterbalance two independent 30-sample captures.
4. Keep the existing current-revision MIT result as a secondary control. Do
   not rerun it as though a newer Hopper-native implementation existed.
5. Do not spend H100 time on FlashMLA or whole FSA for this semantic row.

#### Falsifiers and handoff

- FlashMoBA is disqualified from the current semantic row if a precomputed
  relation that omits the current block produces incorrect output, or if its
  causal/GQA precision boundary differs from the natural reference.
- Payload-only speed cannot alone close Shuttle's natural-program gate. The
  matched router/index boundary must also be reported; if generic relation
  reorientation dominates, that is a Shuttle index-plane optimization target,
  not permission to substitute FlashMoBA's different router.
- The prior TLTC-060 release statement was operationally stale: the
  `dlwh-shuttle-affine-scan` one-H100 batch reservation remained alive. It is
  being reused for this audit and will be explicitly released after the run.

### 2026-08-07 20:40 PDT - TLTC-062 matched FlashMoBA H100 oracle

- Built pinned FlashMoBA
  `39d9ac043b271d046a2181a9991e99a26b67bca1` with CUTLASS
  `a2439551c765c5393aebe557ee75d3a0412d2211` on one H100. The build-only
  specialization disables backward and unused forward variants; it does not
  change the forward algorithm.
- Closed the semantic boundary before timing. The exact payload interface
  supports BF16, D=128, block 128, top-8, causal masking, and native 32:8 GQA.
  FlashMoBA's native per-token/per-head K-derived router is not matched, so the
  full comparison uses the identical natural Shuttle metadata router plus a
  generic RelationPlan reorientation into FlashMoBA's destination-major row
  lists. The primary fixture has 95 query blocks that omit the current block.
- Swept FlashMoBA physical query grouping `{128,256,512,768,1024}` while
  retaining Shuttle's logical block size 128. Group 1024 gave the best
  FlashMoBA full-boundary median and was frozen before confirmation.
- Two independent counterbalanced captures contain 30 samples per
  implementation. Pooled medians are 0.617200 ms for generated Shuttle,
  5.264560 ms for the matched FlashMoBA full boundary, and 4.894560 ms for
  FlashMoBA cached-relation payload. The common router and relation
  reorientation separately measure 0.044080 and 0.211664 ms. Shuttle/full is
  0.117237 times.
- Generated and FlashMoBA outputs differ by at most 0.00390625 with mean
  absolute difference 0.0000651724. Both repeat bitwise across both captures;
  relation and output hashes are stable.
- Interpretation: the exact-expert 1.20-times gate is closed, but FlashMoBA is
  a loose physical denominator for block-shared semantics. It retains a more
  general per-token/per-head row-list interface and uses SM80-style MMA plus
  `cp.async`; Shuttle is specialized to the shared relation and Hopper-native.
  The current MIT 1.423632-ms measurement is a tighter secondary local H100
  control, though it is also SM80-style. Do not claim an 8.5-times advantage
  over the best possible expert implementation.
- FlashMLA remains excluded because its sparse prefill uses MLA/MQA dimensions
  rather than ordinary 32:8 GQA D=128. Whole FSA remains an NSA-semantics
  workload, not an interchangeable MoBA oracle.
- Artifact:
  `lib/tile_lifetime/benchmarks/artifacts/sparse_flashmoba_h100_matched_v0`.
  It freezes raw distributions, the candidate sweep, semantic and deterministic
  hashes, source/toolchain pins, build logs, hardware telemetry, exact benchmark
  source, and checksums.
- Next action: release the reused H100 and push the checkpoint. If sparse
  attention is revisited, build a tight block-shared WGMMA/TMA oracle or change
  the natural workload to match FlashMoBA's native router; do not spend the next
  iteration on more tile-size tuning against this loose denominator.

### 2026-08-08 06:35 PDT - TLTC-063 clean helper-boundary repair

- An adversarial source audit rejected the dense and MoE clean-synthesis rows
  despite their preserved performance evidence. Dense still imported
  FlashAttention's semantic `Softmax` and `AttentionMask` helpers. MoE called
  handwritten CUDA SwiGLU and ordered-merge bodies. A second audit also found
  that the full dense planner assigned semantic roles through an exact
  36-operation positional tuple.
- Replaced the positional dense recovery with generic producer/consumer
  dataflow discovery. Reversing the complete erased Flow operation order now
  recovers the same eight-skeleton plan.
- Replaced FA-owned attention semantics with Shuttle-owned normalized-exp Fold,
  score-Map, and `DomainRestriction` physical helpers. The retained FA-derived
  dependencies are lower-level CuTe layouts, copies, reductions, pipeline
  records, barriers, sequence metadata, packed-GQA indexing, and scheduling.
  H100 lowering now carries score-scale, softcap, causality, and scalar output
  finalization mutations through one physical skeleton.
- Added backend-neutral MoE `MapFoldSemantics` to the recovered plan and a CUDA
  scalar-expression generator. The generic CUDA loop skeletons now invoke
  generated pair-Map, Fold-contribution, Fold-update, and post-Fold functions.
  The build verifies the generated include, and the natural StableHLO runtime
  rejects an extension whose exported IR digest differs from the recovered
  plan. Mutation tests change Map and Fold arithmetic without editing CUDA.
- Tightened the natural MoE runtime connection: Relation/top-k/expert shapes
  and scalar semantics come from the recovered plan. The small router adapter
  still launches generic `torch.mm`, `torch.topk`, and `torch.softmax`; do not
  describe that adapter as generated router code.
- CPU result: all 190 `lib/tile_lifetime/tests` tests pass. Device compilation,
  correctness, and performance replay remain required before restoring dense,
  MoE, or sparse-attention acceptance.
- Working branch: `research/shuttle-clean-helper-boundaries`.

### 2026-08-08 06:46 PDT - TLTC-064 GB200 clean Map/Fold replay

- Pushed the source-boundary snapshot as `31f600f228` on
  `research/shuttle-clean-helper-boundaries`.
- Rebuilt the generated Map/Fold extension on four GB200s with pinned MoK,
  ThunderKittens, DeepEP, CUDA, driver, and Torch revisions. The recovered
  program, generated include, and loaded binary agree on digest
  `3048c6b922de317e556ff4e1a6fe9c81a22bfc9ba4d6582d0245fbf275f81fba`.
- Correctness passes on all ranks: exact relation and payload mappings, zero
  overflow, bitwise deterministic repeats, and Shuttle/MoK maximum absolute
  error `0.0001220703125`.
- Counterbalanced captures measure 4.147536/3.630992 ms with Shuttle first and
  4.144560/3.711088 ms with MoK first. The pooled 60-sample rank-maximum
  medians are 4.147536 ms Shuttle and 3.647136 ms MoK, or `1.137204×`.
- The accepted generated call graph contains no external semantic kernel.
  DeepEP is forward payload transport, reverse transport is
  `all_to_all_single`, and complete MoK forward is oracle-only.
- One batch-priority GB200 pod was preempted before smoke execution. The full
  environment was rebuilt on its replacement, both captures completed, raw
  evidence was copied, and the reservation was released.
- A separate low-priority H100 request never left `SchedulingGated` during its
  bounded wait. It consumed no GPU time and was removed, so Dense and Sparse
  remain device-pending.
- Sealed evidence:
  `lib/tile_lifetime/benchmarks/artifacts/gb200_moe_clean_map_fold_v1`.

### 2026-08-08 08:55 PDT - TLTC-MSA-003 pinned SM100 oracle

- Pinned MiniMax Sparse Attention at
  `80434d7f67877c6570ca19cac444b84bc9855dac` with its CUTLASS gitlink
  `eb61c911471867a5fd2466bfd8f29306cea6ebf8`. The compatible Python stack is
  CUDA/NVCC 13.0.88, Torch 2.10.0+cu130, CUTLASS DSL 4.4.1, and QuACK 0.2.10;
  CUTLASS DSL 4.6.2 is incompatible with the pinned source.
- Reproduced the official BF16 sparse QK/normalized-exp/PV/combine path on one
  1200-W GB200 (driver 595.71.05, unpinned clocks). Upstream
  S=16384,Hq/Hkv=64/4,D=128,top-k=16 measures 2.7211 ms (0.0146-ms standard
  deviation, 404.07 TFLOP/s).
- Established the matched primary boundary at K=16384,Q=256,Hq/Hkv=32/8,
  D=128, block 128, top-k=16. Official MSA takes 0.472528 ms pooled median for
  q2k-to-k2q scheduling plus sparse attention and deterministic combine. The
  full common FP32 proxy Contract, block-max Fold, top-k Selection, relation
  conversion, and sparse payload takes 0.586480 ms pooled median. The clean
  Shuttle completion threshold is therefore 0.703776 ms for this full path.
- Two counterbalanced captures contain 30 samples per ordering. Route score
  multisets match exactly. Output cosine is 0.999995--0.999997, maximum error
  is 0.0005--0.0020, relation/output hashes are stable, and repeats are bitwise
  deterministic. Small filtered-CTA and large 4096-tile lookback cases pass.
- Sealed local evidence is under `scratch/msa-sm100-oracle-80434d7f/`; raw JSON
  checksums begin `245946d7` and `a47efa88`. Preserve the full provenance and
  raw distributions before releasing the held GB200.
- The first natural frontend used symmetric Q/K length. It must be generalized
  to asymmetric Q=256,K=16384 sparse prefill with bottom-right causal position
  mapping before the full-route number can be claimed as an exact natural JAX
  acceptance boundary.

### 2026-08-08 - TLTC-MSA-004 clean 16K synthesis checkpoint

- Generalized the natural frontend to the full symmetric 16K MSA workload and
  lowered it through generic index-projection Contracts, score Contract,
  block-maximum Fold, Selection, RelationPlan, causal DomainRestriction,
  normalized-exponential Fold, and QK/PV Contracts.
- The accepted generated path calls no public MSA score, attention, or combine
  operation. It keeps low-level CuTe layout, copy, MMA, and pipeline templates
  while Shuttle owns the semantic body, relation scheduling, and deterministic
  partial-state merge.
- Isolated raw-10 GB200 medians are 0.637888/0.707600 ms for generated/oracle
  score-Fold-selection, 0.785760/0.837360 ms with natural index projections,
  and 4.431920/3.234160 ms for the full natural boundary. The full ratio is
  `1.37035x`, so the written `1.20x` performance gate remains open.
- Generated and oracle selectors produce the same deterministic route hash.
  Both differ from the materialized reference in 61,446 slots: all but one
  affected row has an underfilled finite causal domain and the last has an
  exactly tied cutoff. Natural maximum/mean output differences are
  0.0536499/0.0000687, so the current 0.01 maximum numerical gate also remains
  open under the real-algebra route policy.
- On the exact official relation, generated payload differs from official MSA
  by at most 0.0009765625 and measures 3.702272/2.644624 ms (`1.39992x`). The
  generic deterministic merge costs 1.831552 ms. A BF16x2 candidate regressed
  to 3.829760 ms and was removed rather than retained as dead code.
- Found and fixed an omitted causal DomainRestriction in the first 16K full
  lowering. The invalid pre-fix run is preserved and labeled rather than used.
- Chose to stop workload-specific merge tuning. The result is a clean generic
  infrastructure checkpoint, not a completed performance row.
- Sealed raw distributions, commands, audits, negative results, and checksums
  under `lib/tile_lifetime/benchmarks/artifacts/msa_clean_sm100_v0` and released
  the GB200 reservation.
