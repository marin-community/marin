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

- The active path replaces generic forward and reverse regions in a natural
  one-layer Grug training step after JAX-owned AD. The accepted ten-call H100
  boundary owns routed forward, two input-adjoint Contracts, a shared Contract
  with two scalar Maps, a deterministic source Fold, two expert-weight
  Contracts, streaming-attention reverse, and two row Folds at `1.178695x` XLA.
  A twelve-call extension also owns the generic weighted RelationProgram
  reverse and passes correctness/determinism, but remains unaccepted. Generic
  demand-driven row narrowing reduces its ratio from `1.247988x` to
  `1.241446x`, insufficient to recover the `1.20x` gate. The next bounded
  candidate is generic Contract-plus-nested-Fold composition, not further
  narrowing tuning. Generic RMS reverse is correct on H100 but remains about
  `1.48x` XLA, so Fold decomposition remains a performance task. H100, B200,
  and GB200 evidence remain separate. The available secondary Blackwell cluster
  provides B200 portability evidence, not GB200 acceptance evidence.

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
- 2026-08-09: treat H100, B200, and GB200 as separate evidence classes. B200 is useful for SM100 portability and single-GPU experiments, but it does not substitute for GB200 measurements of the four-GPU MoE boundary.

## Entry Log

### 2026-08-08 - TLTC-MSA-001 native routed sparse-attention start

- Hypothesis: the existing generic `RelationPlan`, orientation, readiness, bounded-buffer, `Contract`, `DomainRestriction`, and normalized-exponential `Fold` machinery can express MSA's native `(query token, KV group) -> selected KV block` computation. The only new workload-specific code should be frontend recovery and SM100 physical legalization; online-state, mask, merge, and routing semantics must remain Shuttle-owned.
- Baseline: branch `research/shuttle-clean-helper-boundaries` at commit `dd3bb84759`.
- Oracle revisions: MiniMax Sparse Attention `80434d7f67877c6570ca19cac444b84bc9855dac`; its CUTLASS submodule `eb61c911471867a5fd2466bfd8f29306cea6ebf8`.
- Primary configuration: one low-priority GB200 GPU, BF16 Q/K/V and FP32 online state, batch 1, sequence 65536, 64 query heads, 4 KV heads, head dimension 128, KV block 128, top-k 16, causal. Sequence 16384 is the bring-up configuration. A B200 replay is a separate SM100 portability result, not GB200 acceptance evidence.
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
- Next action: use low-priority four-GB200 capacity to reproduce the pinned Mixture-of-Kittens oracle and define the first expert-parallel skeleton around its router schedule, shared/routed expert tasks, readiness events, and dispatch/combine buffers. B200 remains a separate SM100 portability target.

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

### 2026-08-08 - TLTC-MSA-005 naive semantic baseline and delta

- Measured a direct eager selected-attention reference at the full 16K MSA
  shape on a low-priority GB200. The exact payload boundary excludes route
  construction and includes selected K/V gather, materialized FP32 QK and
  softmax, FP32 PV, and BF16 cast.
- One warmup and three repetitions measure 220.386566, 220.194427, and
  220.188004 ms; the median is 220.194427 ms with a 2.708-GiB peak allocated
  delta. Relation/output hashes match the preserved materialized reference and
  repeats are bitwise identical.
- The naive payload is 59.48 times slower than Shuttle's 3.702272-ms generated
  payload and 83.26 times slower than MSA's 2.644624-ms payload. It executes
  256 eager group/chunk bodies and lacks tile residency, online Fold state,
  fusion, and producer-consumer pipelining. These are scale comparisons: the
  naive run uses the materialized-reference relation rather than the shared
  generated/MSA relation at the documented underfilled/tied rows.
- The replacement pod resolved Torch 2.13.0+cu130 instead of pinned 2.10.0, so
  this result is preserved as a semantic scale reference and is excluded from
  acceptance ratios. Raw data, harness, telemetry, and checksums are stored in
  `benchmarks/artifacts/msa_clean_sm100_v0`.
- Source inspection attributes the generated-versus-MSA payload gap primarily
  to final merge: scalar row-warp loads and serial per-row weights versus MSA's
  staged, vectorized, compile-time-tiled combine. The estimated official
  0.773904-ms combine remains explicitly inferred; no pinned combine-only
  measurement was obtained.

### 2026-08-08 - TLTC-MSA-006 generic tiled Fold checkpoint

- Added a backend-neutral tiled Fold-finalization program with explicit
  partial/row/feature axes, dense or indexed addressing, physical feature
  layout, validity, scalar reduction, vector contribution/update/finalize
  ASTs, and source-ordered versus deterministic-tree numerical policies.
- One SM100 emitter now generates the same CTA, 128-bit copy, four-stage
  shared-memory staging, row-warp reduction, vector accumulation, and BF16
  store loop for two bindings. Attention uses 16 dense STG.128 partials with
  normalized-exponential state merge. MoE-style deterministic merge uses six
  indexed contiguous partials with arbitrary non-prefix validity and explicit
  FP32 `mul` followed by `add`.
- Projected Selection now represents score ordering, lower-index tie breaking,
  and underfilled rows explicitly. The physical SM100 selector replaces
  causally invalid slots with `-1` before RelationPlan construction.
- The package suite passes 218 tests; 20 focused Fold/emitter tests and four
  projected-selection backend tests also pass. Commit `932db2193f` is the
  pushed CPU checkpoint.
- Both bindings compiled unchanged on GB200. The indexed non-attention binding
  matches its source-ordered reference exactly, repeats bitwise, handles
  arbitrary non-prefix validity, and measures 0.018128 ms median. The small
  attention binding is finite and deterministic with maximum/mean error
  0.00134033/0.000149893.
- At `Q=K=16384,Hq/Hkv=64/4,D=128,top-k=16`, two isolated captures pool to
  3.823488 ms for generated Shuttle and 3.191376 ms for pinned MSA, a
  1.198069-times ratio. This improves the prior generated natural path by
  13.7%, clears the explicit 3.88-ms objective, and lies at the 1.20-times
  clean-synthesis gate.
- Underfilled causal slots are now explicitly `-1` and excluded from the
  RelationPlan. Reference mismatches fall from 61,446 to six, all caused by one
  exact zero-margin cutoff tie. Routes and outputs repeat bitwise, but the
  selected-set tie yields maximum/mean output error 0.0536499/6.8702e-5.
  Exact source-order tie selection therefore remains open; the measured path
  retains the declared `real_algebra_equivalent` selection policy.
- Raw samples, reuse fixtures, device pins, failure caveats, and interpretation
  are frozen under
  `benchmarks/artifacts/msa_generic_tiled_fold_sm100_v1`.

### 2026-08-08 - TLTC-TRAIN-001 clean Grug train-step boundary and reverse mode

- A natural one-layer Grug MoE train step lowers through ordinary JAX
  `value_and_grad` and an optimizer update with zero StableHLO custom calls when
  the fixture selects reference attention, scatter MoE, and XLA ragged
  contraction. The SGD/AdamW modules contain 329,403/466,782 characters and 82
  `dot_general` operations each.
- Added a reproducible StableHLO exporter and frozen both modules under
  `benchmarks/artifacts/grug_moe_train_step_stablehlo_v0`. This demonstrates
  that Grug itself does not need an opaque-kernel rewrite before Shuttle can
  analyze training math.
- Added generic reverse-mode construction for scalar Map ASTs, multilinear
  Contract adjoints, sum-Fold broadcast adjoints, broadcast-Map reductions, and
  fanout accumulation. A SwiGLU-to-tanh semantic mutation changes the generated
  VJP without editing a backend.
- An RMSNorm-GEMM program differentiates into two transposed Contracts plus
  generated Map/Fold work. Independent formulas for `dx`, `dgamma`, and `dW`
  agree within 2e-5. The package suite passes 220 tests and Pyrefly reports no
  errors.
- Next experiment: pin JAX/JAXLIB 0.11 in an isolated H100 environment, capture
  the post-SPMD train-step `HloModuleProto` through the pre-scheduler extension,
  recover one forward/backward contraction region without rewriting it, then
  replace the linear-SwiGLU family through a generic XLA FFI region call.

### 2026-08-08 - TLTC-TRAIN-002 generated pair-Map gradient checkpoint

- Natural two-projection plus scalar-Map algebra now lowers into a combined
  interleaved Contract, generated pair-Map finalization, scalar-AST VJP, and
  generic dX/dW Contracts. No linear-SwiGLU backward kernel is selected.
- Saved-preactivation and recompute-preactivation are explicit physical
  candidates. The H100 harness records forward-only, saved-backward,
  recompute-backward, and complete forward/backward boundaries against
  preallocated CODA components.
- Changing `SiLU(left) * right` to `tanh(left) * right` changes the generated
  forward and reverse expressions while retaining the same physical skeletons.
- The package suite passes 226 tests. Commit `5fc7ab715b` is pushed; the matched
  H100 result is pending a batch-priority allocation.

### 2026-08-08 - TLTC-XLA-001 Grug pre-scheduler insertion point

- JAX/JAXLIB 0.11.0 invokes a `PRE_SCHEDULER` HLO transformation once while
  compiling the natural one-layer Grug MoE train step. The callback input
  retains 82 dots, 67 reductions, 8 scatters, and 4 sorts.
- Frontend StableHLO contains zero custom calls. XLA introduces one generic
  `TopK` custom call before the callback; no FA, MoK, DeepEP, or recurrent
  semantic kernel is present.
- The callback returned `None`, so this is a real inspection/no-op insertion
  point, not a region replacement. A disposable unary replacement confirmed
  that returning a modified serialized proto changes the executable, but the
  Python binding cannot construct a generic FFI custom call.
- The exact CPU proto, compressed text, frontend StableHLO, census, JAX release
  pin, and reproduction command are frozen under
  `benchmarks/artifacts/grug_moe_train_step_pre_scheduler_jax011_v0` in commit
  `6812303ead`.
- A nontrivial GPU SPMD capture and a narrow C++ bridge for
  `shuttle.execute_region_v1` remain open.

### 2026-08-08 - TLTC-MSA-007 128-feature Fold candidate

- Added one bounded generic physical candidate for the existing
  `TiledFoldFinalizeProgram`. A 128-feature tile assigns four BF16 values to
  each lane, so D=128 needs one output block instead of two.
- Two shared-memory banks issue the next four-partial `cp.async` group before
  evaluating the current generated contribution/update AST, then wait before
  swapping banks. The semantic Fold, partial order, and AST generator are
  unchanged.
- The same emitter path covers normalized-exponential attention state and
  indexed deterministic weighted sums. Focused numerical and emitted-source
  tests pass. CUDA compilation and GB200 combine-only/full-path timing remain
  pending; the candidate does not replace the accepted 1.198069-times result
  until measured.

### 2026-08-08 - TLTC-TRAIN-003 generic row-normalization adjoint

- Added ordinary Map/Fold/Contract source algebra and a generated compact
  reverse program for row second-moment normalization followed by a Contract.
  The generated plan exposes Maps, Folds, dX/dW Contracts, saved values, and
  statistic recomputation rather than selecting a named normalization kernel.
- Save-standardized, save-input-and-inverse, and recompute-statistic policies
  are explicit candidates. Source-ordered Contract preparation and delayed
  real-algebra-equivalent output scaling remain separate numerical choices.
- A centered-second-moment mutation adds the mean Fold and the extra backward
  mean-subtraction Fold through the same compiler path, demonstrating that the
  algebra covers LayerNorm as well as RMSNorm. Nine independent NumPy tests
  cover both statistics, all save policies, and a nondefault epsilon.
- This is an inspectable generated physical plan, not yet an H100 performance
  result. Generic executable Map/Fold emission and matched CODA backward timing
  are the next boundary.

### 2026-08-08 - TLTC-XLA-002 post-SPMD Contract/Map recovery

- Added a read-only parser for the frozen JAX 0.11 callback HLO, an ordinary
  fusion-body inliner, and explicit convert/bitcast/copy boundary tracking.
  The matcher uses only opcodes, shapes, and data dependencies.
- The frozen Grug train step contains 569 computations and 2,913 inlined
  logical nodes. The pass recovers 82 Contracts and two shared-input Contract
  pairs feeding exp/divide/multiply scalar Maps and one downstream Contract
  each, without reading metadata, stack frames, model names, or instruction
  names.
- The recovered Maps contain 16--18 BF16 round-trip conversion edges. This
  makes the numerical blocker explicit: direct CODA-style fusion is not
  source-order legal unless the cast graph is preserved or the compilation
  policy allows rounding reorder.
- A tanh-to-exponential mutation reuses the matcher. Reverse dX/dW Contracts
  are visible, but post-SPMD structure alone does not assign cotangent and
  saved-value roles; the analyzer reports that as open instead of guessing.

### 2026-08-08 - TLTC-MSA-008 direct Fold ablation harness

- The generic tiled Fold emitter now retains an explicit one-buffer schedule
  as the no-overlap control. The bounded physical candidate set is 64 features
  with one buffer, 64 features with ping-pong buffers, and 128 features with
  ping-pong buffers.
- A combine-only GB200 harness compiles and preallocates every candidate before
  measurement, rotates execution order for every warmup and sample, and uses a
  preallocated output. It records raw samples, repeat order, correctness,
  determinism, source hashes, and the canonical generic Fold program.
- All candidates use an identical normalized-exponential state-merge AST; the
  indexed deterministic weighted-sum mutation remains available through the
  same emitter. The package suite passes 246 tests.
- CUDA compilation and timing remain pending. The accepted full MSA result is
  unchanged until the direct ablation and complete natural boundary are both
  measured on GB200.

### 2026-08-08 - TLTC-TRAIN-004 executable row-normalization Folds

- Added one rank-two CUDA Map/Fold generator with row- and column-reduction
  layouts, element and reduced outputs, scalar contribution/finalization ASTs,
  and explicit source-ordered versus deterministic-tree policies.
- The same generator lowers RMSNorm backward and a centered LayerNorm mutation.
  RMSNorm emits a row correlation Fold plus final dX Map and a column dgamma
  Fold; LayerNorm adds the local-sum Fold and backward centering without a
  workload-named kernel.
- Independent NumPy execution agrees for both forms. A matched H100/GB200
  harness compiles the emitted CUDA and compares the identical algebra against
  `torch.compile`; hardware compilation and performance remain pending.
- Commit `5509e44448` is pushed.

### 2026-08-08 - TLTC-TRAIN-005 generated streaming-attention backward

- The forward normalized-exponential Fold now exposes saved output, row
  maximum, and row sum-exp state. A generic reverse derivation emits QK
  recomputation, probability reconstruction, dV and dP Contracts, score-Map
  VJP, and dQ/dK Contracts.
- The generated physical schedule uses deterministic query-major dQ and
  key-major dK/dV loops with no atomics and no score/probability
  materialization. It supports BF16, head dimensions 64/128, GQA, causality,
  scale Maps, and a tanh-softcap mutation through the same generator.
- JAX autodiff parity passes for causal scaling and for the noncausal softcap
  mutation. H100/GB200 compile smoke and matched performance remain pending.
- Commit `2f45798ff4` is pushed.

### 2026-08-08 - TLTC-XLA-003 natural Grug forward/backward replacement

- The JAX 0.11 pre-scheduler callback now replaces a structurally recovered
  pair-Contract/Map/Contract forward region inside the natural one-layer Grug
  train step. The generated scalar body preserves all 16 BF16/F32 cast
  boundaries and all 58 result leaves are bitwise identical.
- Starting from the second recovered Contract pair, generic entry-region growth
  follows only pointwise/wrapper users, treats additional saved/cotangent values
  as inputs, and treats every externally consumed value as an output. On Grug
  this forms a nine-instruction backward region with four inputs and three live
  outputs feeding five downstream Contracts.
- A generated tuple-result body replaces that backward region through one
  custom call and three tuple projections. All 58 leaves remain bitwise
  identical, including the two baseline NaN payloads, and the handler executes
  exactly once.
- This is disposable insertion evidence: HLO text mutation and the removed
  legacy CPU ABI are not the production bridge. Typed connected-region
  replacement, supported multi-result FFI, sharding/alias/effect transfer, and
  GPU lowering from the same AST remain open.
- Reproducible compressed modules and generated handlers are frozen under the
  three `xla_*custom_call_smoke_jax011_v0` artifact directories. Commit
  `023755efca` is pushed; the package suite passes 256 tests.

### 2026-08-08 - TLTC-MSA-009 measured Fold schedules

- On one GB200, 100-sample combine-only measurements give 0.015728 ms for the
  64-feature one-buffer control, 0.014896 ms for 64-feature ping-pong, and
  0.013824 ms for 128-feature ping-pong. All three share one semantic hash and
  deterministic output hash.
- A single generated-only full natural confirmation with 128-feature
  ping-pong measures 3.310736 ms versus the preserved 3.831632-ms 64-feature
  result. Selection and sparse core timing are unchanged, localizing the gain
  to the Fold/finalization boundary.
- The one-sided confirmation is diagnostic rather than a new counterbalanced
  acceptance capture. The prior pooled 1.198069x generated/oracle comparison
  remains the strict accepted result. Exact source-order top-k tie behavior is
  still open.

### 2026-08-08 - TLTC-TRAIN-006 first GB200 gradient measurements

- Generic RMSNorm backward measures 0.087160 ms versus 0.054582 ms for the
  identical `torch.compile` algebra (1.597x). The centered LayerNorm mutation
  measures 0.088042 versus 0.053051 ms (1.660x). Both are deterministic and
  correct but fail the 1.20x performance gate.
- The measured column Fold used one block per feature and therefore loaded
  row-major elements with a hidden-size stride. Commit `82bedcd6b4` adds a
  generic 32-feature schedule whose warps load contiguous columns while eight
  row lanes reduce each feature. A new measurement is pending.
- The first physical streaming-attention backward compiles and measures
  0.131408 ms versus 0.181536 ms for SDPA, but dQ/dK/dV maximum errors are
  0.762/0.844/0.766. This is a correctness failure, not acceptance.
- Diagnosis found that the backward harness saved forward state from a causal
  M=32,N=64 kernel even though its diagonal splitting is legal only when M is
  a multiple of N. The schedule double-counted/unmasked keys before backward.
  Commit `14f2bbc3f9` uses the matched legal 32x32 forward state, rejects
  illegal causal tile pairs, and requires numerical gates in acceptance. A
  corrected GB200/H100 replay is pending.

### 2026-08-08 - TLTC-XLA-004 supported multi-result FFI

- The natural Grug backward-region proof now emits an XLA typed-FFI handler
  with four typed inputs and three typed results. One pre-scheduler tuple call
  plus three tuple projections executes through FFI API version 1; all 58
  train-step leaves remain bitwise identical.
- This removes the obsolete legacy tuple ABI from the backward proof. Text HLO
  mutation, sharding/alias/effect transfer, and a GPU body generated from the
  same multi-output AST remain open. Commit `1ff3e6b77e` is pushed.

### 2026-08-08 - TLTC-XLA-005 routed train-step ownership boundary

- Hypothesis: physical XLA HLO retains enough name-free structure to recover
  the main routed forward and backward program after fusion and padding, while
  leaving placement collectives external.
- Commit Hash: `bb363f56ac`.
- Command: `uv run --frozen --package marin-tile-lifetime python
  lib/tile_lifetime/benchmarks/analyze_xla_relation_program_hlo.py
  lib/tile_lifetime/benchmarks/artifacts/grug_moe_train_step_pre_scheduler_jax011_v0/pre-scheduler-hlo.txt.gz
  --output
  lib/tile_lifetime/benchmarks/artifacts/grug_moe_train_step_pre_scheduler_jax011_v0/relation-program-recovery.json`.
- Result: a metadata-independent pass recovers two equivalent 8-row, 2-slot,
  16-edge, 4-segment `RelationPlan`s; the executed and rematerialized segmented
  Contract/Map/Contract forward chains; one segmented input-gradient
  Contract/Map-adjoint/Contract chain; two additive source scatter Folds; and
  two group-batched weight-gradient Contracts. The associated all-reduce stays
  visible as an external boundary.
- Numerical boundary: every BF16/F32 conversion along the recovered Maps and
  the scatter reducer's BF16 round trip are retained in the report. The
  executed forward Fold includes an explicit multiply before additive scatter.
  The physical HLO does not prove source-ordered GPU scatter updates, so the
  replacement must choose that policy explicitly rather than infer it.
- Interpretation: the train-step HLO supports a concrete Shuttle ownership
  boundary spanning routing metadata, routed expert forward, routed input
  gradient, and routed weight gradients. The pass does not use model/source
  names or select an opaque MoE kernel.
- Next action: share the existing HLO scalar-AST importer with this routed Map,
  then replace the recovered forward and backward regions with the generic GMM,
  generated Map/Fold, and RelationPlan runtime. Preserve the all-reduce as an
  external placement transition until Shuttle owns collectives.

### 2026-08-08 - TLTC-TRAIN-007 corrected GB200 gradient schedules

- The coalesced 32-column-group Fold closes both row-normalization backward
  performance gaps at rows=2048 and hidden=4096 on one GB200. RMSNorm measures
  0.039704 ms versus 0.051835 ms for the matched `torch.compile` algebra
  (0.766x); centered LayerNorm measures 0.041109 versus 0.050499 ms (0.814x).
  These are 2.20x and 2.14x faster than the original generated schedules.
- Both paths are deterministic. Maximum dX error is 1.90735e-6 and maximum
  feature-scale-gradient error is 7.62939e-5. The same generic Fold/Map/Contract
  generator handles RMS and the centered LayerNorm mutation.
- Correcting the causal forward-state tile to 32x32 also fixes streaming
  backward numerics. At S=128, forward/dQ/dK/dV gates and deterministic hashes
  pass and the generated path measures 0.080550 ms versus 0.143165 ms for SDPA.
- At the primary S=2048 shape, correctness remains good but performance does
  not: 2.200320 ms generated versus 0.155450 ms SDPA, or 14.155x slower. The
  current generic Triton schedule repeatedly visits query heads and tiles in
  the key-major dK/dV path and lacks an expert shared-memory/TMA/WGMMA pipeline.
  This is now a physical-schedule problem, not a semantic-recovery problem.
- Raw counterbalanced samples, errors, hashes, telemetry, commands, and failure
  diagnostics are preserved under
  `benchmarks/artifacts/generated_gradient_skeletons_gb200_v1`.

### 2026-08-08 - TLTC-XLA-006 GPU FFI and routed scalar ownership

- Commits `36e2a52aad` and `d7ad815f48` generate a CUDA typed-FFI handler for
  the recovered Grug multi-output reverse Map. The latest body uses two generic
  pedantic-FP32 cuBLAS Contracts followed by a generated source-ordered Map;
  it contains no workload-specific semantic kernel.
- The exact `36e2a52aad` GB200 execution attempt failed before CUDA compilation:
  the GPU pre-scheduler module contained no region matching the CPU recovery
  boundary. This invalidates the assumption that the frozen CPU HLO selector
  transfers unchanged to GPU. The GPU HLO must be captured and the region
  recovery generalized rather than patched by names.
- Commits `d69753c2c1` and `fa2f707441` import both routed forward Maps and the
  two concatenated input-adjoint Map outputs into one cast-aware scalar AST
  family. The reports retain affine slice offsets and every BF16/F32 conversion
  (30 converts forward; 86 and 30 for the adjoint outputs), generate generic
  CUDA scalar functions, and pass mutations that alter only the affected
  semantic digest/source.
- The next routed execution boundary is now explicit: runtime `RelationPlan`,
  generic segmented Contracts, generated scalar Maps, deterministic generated
  scatter Folds, and group-batched weight-gradient Contracts, with all-reduce
  left external. GPU execution and GMM attachment remain open.

### 2026-08-08 22:12 PDT - TLTC-TRAIN-008 JAX-owned RMS reverse recovery

- Hypothesis: JAX can own model-level AD while Shuttle recovers and fuses the
  resulting normalization reverse algebra without a normalization-backward
  recognizer or compiler-owned VJP in the accepted path.
- Commit Hash: `a152a7eb40af7cb475893642fb8dc3a9b78093ac`.
- Command: `uv run --frozen --package marin-tile-lifetime --group test pytest
  lib/tile_lifetime/tests/test_stablehlo_row_normalization_backward.py
  lib/tile_lifetime/tests/test_row_normalization_training.py -q`.
- Config: ordinary BF16 JAX normalization followed by `jax.vjp` and
  `jax.export`; arbitrary importer input names; RMS and centered-normalization
  mutation; generated row and feature-axis Fold programs.
- Result: 13 tests pass. The RMS reverse graph imports as generic StableHLO
  Map/Fold algebra, and name-independent role/axis recovery produces the same
  generic executable Fold family previously reached through Shuttle-owned AD.
  The centered mutation adds the expected mean and centering Folds through the
  same path. Generated CUDA source contains neither RMS nor LayerNorm names.
- Interpretation: the accepted architecture no longer needs Shuttle-owned AD
  for this component. The current CUDA benchmark still uses Torch only to
  compile/load generated source and provide a matched timing oracle; this is
  parity scaffolding rather than the intended final JAX runtime path.
- Next action: replay the exact commit on H100/GB200, then connect the recovered
  program to the JAX typed-FFI replacement boundary so the final execution path
  is Torch-free by default.

### 2026-08-08 23:10 PDT - TLTC-XLA-007 generated routed Fold bodies

- Hypothesis: the source-keyed scatter Folds in the natural Grug reverse HLO
  can be recovered as generic scalar contribution and update programs rather
  than retained as named MoE merge logic.
- Commit Hash: `b5d4f81e6a`.
- Command: `uv run --frozen --package marin-tile-lifetime --group test pytest
  lib/tile_lifetime/tests/test_xla_relation_program_recovery.py -q`.
- Result: six focused tests pass. The forward Fold imports the routed Contract
  result plus a separately formed route-weight value; the input-adjoint Fold
  imports only the routed cotangent. Both reducers import as FP32 add followed
  by the original BF16/F32 round trip and generate standalone CUDA scalar
  bodies. A multiply-to-add contribution mutation changes only the
  contribution digest/source, leaving the reducer unchanged.
- Interpretation: every scalar Map/Fold body in the recovered routed
  forward/input-adjoint boundary is now compiler-owned. The remaining boundary
  is physical: attach these bodies to generic segmented Contracts and a
  deterministic source-ordered scatter executor.
- Next action: use the recovered RelationPlan and generated Fold bodies to
  replace the executed routed forward/input-adjoint region; retain the
  all-reduce as an external placement transition for this checkpoint.

### 2026-08-09 00:25 PDT - TLTC-XLA-008 Torch-free JAX Fold registration

- Hypothesis: the generic axis-Fold family recovered from a JAX-owned reverse
  graph can execute through JAX without using Torch as either the accepted
  runtime or the owner of automatic differentiation.
- Commit Hash: `a5b39dc65e`.
- Result: one generated XLA typed-FFI handler now registers multiple generic
  CUDA axis Folds with JAX. The benchmark path begins with ordinary JAX,
  obtains the reverse graph through `jax.vjp`, recovers the Fold programs from
  StableHLO, compiles their generated CUDA, and calls them through `jax.ffi`.
  Source audits and focused tests pass; generated runtime source contains no
  Torch dependency.
- Caveat: H100/GB200 compilation, correctness, and performance remain pending.
  This checkpoint establishes the intended ownership and runtime boundary, not
  a measured acceptance result.

### 2026-08-09 00:52 PDT - TLTC-EVENT-002 runtime and phased Event inputs

- Hypothesis: readiness state can be derived mechanically from generic task
  relations, including runtime RelationPlan indegrees and repeated phased
  Contract/Fold dataflow, without workload-specific event annotations.
- Commit Hash: `1e0512923d`.
- Result: `EventTensorRuntimeInputs` carries count and notify/trigger CSR
  tables; empty runtime groups are initially ready. Explicit physical event
  slots and generations represent circular reuse. A generic phased
  Contract-to-Fold-to-Contract-to-finalize graph derives its dependencies from
  task relations. Twenty-three focused CPU/reference tests pass.
- Caveat: this is the static/reference checkpoint. Device-side notify, wait,
  visibility, and generation validation for segmented and attention-like
  schedules is being measured separately.

### 2026-08-09 01:43 PDT - TLTC-TRAIN-009 rejected partitioned key/value reverse Fold

- Hypothesis: the primary streaming-attention reverse gap is partly caused by
  insufficient parallelism in the deterministic key/value-gradient Fold. A
  bounded query-domain partition can expose more tasks without changing the
  recovered Contract/Map/Fold semantics or using atomic accumulation.
- Commit Hash: `ace2636514`.
- Result: on one GB200 at S=2048 with a 32x32 tile, the one-partition path
  measures 0.864582 ms versus 0.148534 ms for matched SDPA (5.821x). Four
  partitions measure 0.854435 ms versus 0.154432 ms (5.533x). Correctness and
  deterministic hashes pass for both.
- Interpretation: 1.17% generated latency improvement does not justify 64 MiB
  of FP32 partials, four times as many key/value tasks, and 512 finalizers. The
  experiment was stopped before the two/eight-partition variants. Commit
  `0f28376aea` removes the candidate. The remaining gap is in the physical
  reverse QK/PV pipeline and data reuse, not missing Fold task parallelism.

### 2026-08-09 02:02 PDT - TLTC-XLA-009 Torch-free JAX Fold execution

- Hypothesis: JAX can own AD and invoke Shuttle-generated generic Fold kernels
  directly, without Torch in the accepted runtime, while remaining competitive
  with the same explicit algebra compiled by XLA.
- Commit Hash: `1e0512923d`.
- Withdrawn performance result: the JAX typed-FFI path measured 0.0610993 ms
  versus 0.0669201 ms for the nominal XLA baseline, but the XLA function closed
  over benchmark arrays and was constant-folded. The `0.91302x` ratio is
  invalid. The 312 handler executions, repeated hashes, and correctness metrics
  remain valid. GB200 requires a corrected runtime-input replay.
  Against the matched FP32 algebra, maximum errors are 9.537e-7 for dX and
  2.289e-5 for the feature Fold.
- Numerical boundary: after casting both results to the natural BF16 VJP
  outputs, the differences are materially larger (maximum 0.0625 for dX and
  1.0 for the feature Fold). This is a deterministic-tree result; source-order
  equivalence to XLA's selected reduction tree is not established and must not
  be claimed. The benchmark now records that contract explicitly.

### 2026-08-09 01:31 PDT - TLTC-TRAIN-010 packed mapped heads in dQ

- Hypothesis: deriving the GQA packed-row domain from the QK Contract index map
  for the query-gradient traversal will reduce physical Contract and K/V-load
  duplication without partial gradients or atomics.
- Commit Hash: `f0f2aa6b73395d360d69a0ce4dd74add86d022ca`.
- Config: one H100 80GB, causal BF16, batch 1, sequence 2,048, 32 query heads,
  eight K/V heads, dimension 128, 32-by-32 tile, eight warps, three stages;
  Torch 2.8.0+cu128 and Triton 3.4.0. Each path used 30 counterbalanced samples
  with five iterations per sample.
- Result: scalar-head dQ measures 1.297498 ms versus 0.464624 ms for SDPA.
  Packed dQ measures 0.584992 ms versus 0.465133 ms for SDPA. Packing reduces
  generated latency by 54.91%; the packed result is 1.257688x SDPA and remains
  outside the 1.20 gate. Both paths pass correctness and produce the same
  deterministic output hash.
- Resources: packed dQ uses 114,688 bytes of shared memory versus 45,568 bytes
  for scalar-head dQ. Register and spill counts were unavailable. Static
  physical reverse Contract invocations fall from 266,240 to 116,480.
- Interpretation: mapped-head operand reuse is a useful generic schedule
  transformation. The next gap is the two-traversal reverse pipeline: dQ and
  dK/dV still recompute QK, probability, dP, and dS separately and do not use a
  TMA/WGMMA producer-consumer schedule.
- Artifact:
  `lib/tile_lifetime/benchmarks/artifacts/streaming_attention_backward_packed_h100/`.

### 2026-08-09 01:38 PDT - TLTC-XLA-010 Grug GPU Contract+Map replacement

- Hypothesis: a generic Contract+Map region recovered from the natural Grug
  reverse HLO can be replaced and executed through XLA typed FFI without a
  model-name recognizer or opaque workload kernel.
- Commit Hash: source under test `2ed4741e13`; result checkpoint recorded by the
  following commit.
- Command: `xla_grug_backward_multi_output_gpu_custom_call_smoke.py
  --cuda-architecture sm_100a --warmup 4 --repeats 30` on one low-priority
  GB200. The exact command and environment are in
  `benchmarks/artifacts/grug_contract_map_gpu_gb200_v0/provenance.md`.
- Config: ordinary one-layer Grug training step; JAX/JAXlib 0.11.0; coherent
  NVCC/CRT/NVVM/NVRTC/NvJitLink 13.0.88; CUDA runtime 13.0.96; cuBLAS
  13.0.2.14; driver 595.71.05; cluster-default unpinned clocks.
- Result: the transformed executable contains one custom call and executes it
  35 times. The initial 53-leaf comparison is bitwise exact. Thirty
  counterbalanced medians are 0.552480 ms baseline and 0.563937 ms transformed,
  a `1.020737x` ratio.
- Determinism: repeated full-step hashes have four variants on both baseline
  and transformed paths. The initial paired result is exact, but whole-step
  bitwise determinism is not established.
- Environment finding: mixed CUDA 13.3 NVVM with 13.0 PTXAS fails at transformed
  compilation, and pip's versioned-only `libcublas.so.13` defeats `-lcublas`
  even when the correct search path is present. The checkpoint resolves
  absolute shared-library paths and uses `-cudart=none`; disposable symlinks
  were used only for the measured run.
- Interpretation: the natural JAX-to-HLO-to-generic-region-to-generated-FFI
  execution chain now works on GB200. The result is an integration proof around
  a generic cuBLAS Contract, not yet a competitive fused tile mainloop.
- Next action: replace the generic cuBLAS call plus scalar Map with Shuttle's
  reusable tiled Contract skeleton, then apply the same post-SPMD replacement
  mechanism to the larger routed forward/input-adjoint regions.

### 2026-08-09 - TLTC-TRAIN-007 JAX-owned RMS reverse Fold on H100

- A low-priority H100 replay executes the ordinary `jax.vjp` StableHLO through
  the generated typed-FFI row/column Fold path with no Torch dependency.
- Exact versioned `libcudart.so.13` linking now uses host-linker forwarding and
  an embedded rpath. Both CUDA-library symlink scans were empty, and all manual
  CUDA library-path environment variables were unset. The original NVCC
  positional-input failure is retained in the artifact.
- The generated result is bitwise deterministic. Against matched explicit FP32
  algebra, `dx` maximum/mean error is `9.537e-7`/`9.838e-10` and feature-scale
  cotangent maximum/mean error is `2.289e-5`/`3.701e-6`.
- Withdrawn performance result: the nominal XLA baseline was constant-folded
  because it closed over benchmark arrays. The `1.955629x` ratio is invalid.
  This entry retains only the H100 execution, correctness, and determinism
  evidence; corrected profiling is recorded below.
- The natural BF16 JAX VJP is separately reported as an order/cast diagnostic.
  Source-ordered BF16 equivalence is not established.

### 2026-08-09 - TLTC-TRAIN-011 corrected RMS reverse component profile

- HLO inspection showed that the prior JAX/XLA benchmark closed over its random
  arrays. XLA compiled constants and copies, invalidating the H100 `1.955629x`
  and GB200 `0.913018x` ratios. Their raw samples remain preserved; only their
  correctness and determinism evidence is retained.
- Revision `9de6770953` makes all generated and matched-XLA arrays runtime
  arguments. On one H100, 30 counterbalanced samples with 100 iterations each
  measure 0.072270 ms generated and 0.072500 ms XLA (`0.996827x`) for the full
  reverse.
- Isolated generated/XLA measurements are 0.041130/0.053499 ms (`0.768795x`)
  for the input-cotangent row Fold and 0.031282/0.033126 ms (`0.944353x`) for
  the feature-scale column Fold. Separate generated components are bitwise
  identical to full generated outputs and repeat hashes are stable.
- Corrected optimized HLO contains parameters. Full XLA emits a Triton row
  reduction followed by one input fusion that produces dX and performs the
  feature-scale reduction. This identifies generic multi-output Map/Fold fusion
  as a useful follow-up, but the current generated full path already matches
  XLA on H100.

### 2026-08-09 - TLTC-TRAIN-012 corrected GB200 RMS reverse replay

- Revision `07bbabb184` was replayed on one NVIDIA GB200 from the four-GPU
  GB200 pool. No B200 device or B200 result is involved. The measured Shuttle
  source and newer allocation-control client are pinned separately in the
  artifact provenance.
- Two independent captures each use 30 counterbalanced samples and 100
  iterations per sample. Primary full generated/XLA medians are
  0.101149/0.114715 ms (`0.881737x`); confirmation medians are
  0.112671/0.125017 ms (`0.901247x`). Pooled 60-sample medians are
  0.107179/0.121351 ms (`0.883215x`).
- Primary generated/XLA component medians are 0.093349/0.104646 ms
  (`0.892046x`) for the input-cotangent row Fold and 0.097220/0.101979 ms
  (`0.953333x`) for the feature-scale column Fold. Both components remain
  below XLA in the confirmation capture.
- Separate component outputs are bitwise identical to the full generated
  outputs inside each capture, repeated executions have stable hashes, and the
  generated path matches explicit FP32 algebra within `9.537e-7` maximum dX
  error and `3.052e-5` maximum feature-scale-cotangent error.
- Corrected optimized HLO contains runtime tensor parameters. A two-process
  diagnostic found that the upstream XLA-produced FP32 inverse-scale buffer is
  not bitwise identical across fresh processes, while the random BF16 inputs
  and standardized BF16 activation are identical. Determinism claims are
  therefore scoped to identical runtime buffers within a capture.
- Interpretation: the generic two-kernel row/column Fold reverse beats matched
  XLA on this GB200 configuration. A workload-specific fused RMS backward body
  is not justified by the current H100 or GB200 measurements.
- Artifact:
  `lib/tile_lifetime/benchmarks/artifacts/jax_row_normalization_backward_gb200_components_corrected_v1/`.

### 2026-08-09 - TLTC-EVENT-013 Torch-free Event Tensor replay on GB200

- Hypothesis: one generic `EventTensorPlan` lowering can execute runtime
  `RelationPlan` readiness and phased Contract/Fold readiness through JAX typed
  FFI without Torch or workload-specific event wiring.
- Commit Hash: `380f715d74`; measured source `1a04930ecd`.
- Command: `h100_jax_event_tensor_ffi.py --architecture sm_100a
  --expected-gpu-substring GB200 --allocation-gpus 1 --allocation-cpu 1
  --allocation-memory 32GB --allocation-disk 50GB --allocation-priority batch`.
  The complete argv and environment are stored in `summary.json`.
- Config: one NVIDIA GB200, driver 595.71.05, JAX 0.10.1 with CUDA 13,
  NVCC/PTXAS 13.3.73, 30 counterbalanced samples and 100 invocations per
  sample. The preceding H100 request timed out before admission and consumed no
  GPU time. No B200 result is involved.
- Result: runtime primary/mutation are bitwise source-order matches and measure
  0.061314/0.061152 ms. Phased primary/mutation maximum errors are 8.941e-8 and
  1.192e-7 and measure 0.169697/0.146477 ms. All four paths are bitwise
  deterministic over five repeated executions. Optimized HLO retains runtime
  parameters, one typed-FFI target, and no constant/copy substitution.
- Interpretation: the generic JAX-to-CUDA readiness boundary works on SM100.
  The phased payload is a scalar reference Contract/Fold pipeline. It does not
  establish tensor-core attention performance or complete circular-buffer
  slot/reuse derivation.
- Artifact:
  `lib/tile_lifetime/benchmarks/artifacts/event_tensor_jax_ffi_gb200_v0/`.
- Next action: connect buffer-slot assignment and last-consumer reuse edges to
  a real generated attention or routed-MoE task graph after the current Grug
  integration work.

### 2026-08-09 - TLTC-XLA-014 routed Grug forward replacement on GB200

- Hypothesis: Shuttle can recover and replace the natural routed
  Contract→Map→Contract→Fold region from actual GPU PRE_SCHEDULER HLO using
  only generic Contracts, generated scalar ASTs, and a deterministic Fold.
- Commit Hash: `9cac1dd40b`.
- Command: `xla_grug_routed_forward_gpu_custom_call.py --architecture sm_100a
  --warmup 4 --repeats 30 --artifact-directory <artifact> --output
  <artifact>/summary.json`, using the pinned NVCC 13.0.88 path recorded in the
  artifact.
- Config: ordinary one-layer Grug train step with JAX-owned differentiation;
  one NVIDIA GB200, compute capability 10.0, driver 595.71.05, JAX/JAXlib/CUDA
  plugin 0.11.0, cuBLAS 13.4.1.1, batch priority, one CPU. No B200 result is
  involved.
- Result: one region is replaced and the handler executes 35 times. All seven
  operands have runtime parameter ancestry. The transformed full train step
  measures 0.991105 ms versus 0.851664 ms for XLA, a 1.163727x ratio within the
  1.20 proof target. Maximum/mean output errors are 2.328e-10/4.183e-15, and 52
  of 53 result leaves are bitwise equal in the direct comparison.
- Numerical/scheduling contract: BF16 Contract and Map/Fold storage with FP32
  Contract accumulation, generated source BF16 rounding, a fixed
  destination-major Fold traversal, one writer per source-feature, and no
  atomic accumulation. Whole-step hashes vary because other XLA-owned
  reductions remain nondeterministic.
- Interpretation: the natural JAX→GPU HLO→generic region→generated typed-FFI
  chain now works for a routed forward slice. The first real GPU HLO differed
  from the CPU-derived fixture; structural recovery was generalized to the
  unfused BF16 dot/Map/scatter form before code generation.
- Artifact:
  `lib/tile_lifetime/benchmarks/artifacts/xla_grug_routed_forward_gpu_gb200_v0/`.
- Next action: recover and replace the input-adjoint
  Contract→reverse-Map→Contract→source-Fold region, followed by the routed
  weight-gradient Contracts. Keep collectives external until those local
  regions execute correctly.

### 2026-08-09 - TLTC-TRAIN-015 recovered attention reverse on H100

- Hypothesis: the JAX-owned VJP recovered into generic
  Contract/Fold/DomainRestriction algebra can drive the existing deterministic
  streaming reverse skeleton within 1.20x of a matched H100 expert backend.
- Commit Hash: artifact source `9cac1dd40b`; integrated artifact `0823671efb`.
- Command: `h100_generated_streaming_attention_backward.py --semantic-source
  jax_vjp_hlo_recovery --sequence 2048 --mutation causal --block-m 32
  --block-n 32 --num-warps 8 --num-stages 3 --warmups 5 --repeats 30
  --iterations 5 --profile-components`. The cuDNN SDPA backend was disabled
  after it failed to construct an oracle plan; Torch flash SDPA supplied the
  matched oracle.
- Config: one NVIDIA H100 80GB HBM3, driver 595.71.05, CUDA 12.8, Torch
  2.11.0+cu128, Triton 3.6.0, JAX 0.11.0, causal BF16 GQA, batch 1, 32 query
  heads, eight K/V heads, dimension 128, sequence 2,048. Thirty
  counterbalanced samples contain five iterations each.
- Result: generated/oracle medians are 0.549139/0.462077 ms, a 1.188415x ratio
  inside the 1.20 proof target. Component medians are 0.044742 ms for the
  output-dot Fold, 0.157680 ms for dQ, and 0.356966 ms for dK/dV. Maximum errors
  are 0.003906 for the forward output, 0.015625 for dQ, 0.03125 for dK, and
  0.0625 for dV. Repeated generated outputs have one stable hash.
- Synthesis boundary: ordinary JAX owns AD; StableHLO recovery identifies the
  generic reverse algebra; a guard verifies that the physical score-Map VJP is
  the derivative of the recovered forward scalar AST. The physical harness is
  still Torch/Triton and is not the final Torch-free JAX runtime.
- Interpretation: dK/dV accounts for most generated time. The separate
  output-dot Fold is 8.1% of generated latency and is a useful generic
  attachment candidate: compute/store it in the dQ owner traversal, then let
  dK/dV consume the result.
- Artifact:
  `lib/tile_lifetime/benchmarks/artifacts/streaming_attention_backward_h100_replay_9cac1dd40b/`.
- Next action: reproduce the same boundary on GB200, then test the output-dot
  Fold attachment before pursuing a broader physical reverse-pipeline rewrite.

### 2026-08-09 - TLTC-TRAIN-016 recovered attention reverse on GB200

- Hypothesis: the H100 result transfers to the same generic streaming reverse
  schedule on SM100 without changing recovered semantics or physical tile
  parameters.
- Commit Hash: artifact source `9cac1dd40b`; integrated artifact `18f82b2342`.
- Command/config: identical to TLTC-TRAIN-015, using one NVIDIA GB200,
  compute capability 10.0, driver 595.71.05, 1,200 W power limit, CUDA 12.8,
  Torch 2.11.0+cu128, Triton 3.6.0, and JAX 0.11.0. Thirty counterbalanced
  samples contain five iterations each.
- Allocation correction: one nominal GB200 request exposed the same H100 UUID
  and compute capability 9.0 as TLTC-TRAIN-015. It was released before any
  environment install or benchmark. The accepted replay used the proven GB200
  pool and verified the model and compute capability before setup. No B200
  result is involved.
- Result: generated/oracle medians are 0.484029/0.409293 ms, a 1.182598x ratio
  inside the 1.20 proof target. Component medians are 0.040477 ms for the
  output-dot Fold, 0.136282 ms for dQ, and 0.314320 ms for dK/dV. Correctness
  passes and repeated generated outputs have one stable hash.
- Interpretation: H100 and GB200 agree on the physical bottleneck. dK/dV is
  64.9% of generated latency, and the standalone output-dot Fold is 8.4%.
  Attaching that Fold to the dQ owner traversal is the next bounded generic
  optimization; a larger reverse-pipeline rewrite is unnecessary unless the
  attachment fails to close the remaining gap.
- Artifacts:
  `lib/tile_lifetime/benchmarks/artifacts/streaming_attention_backward_gb200_replay_9cac1dd40b/`
  and the rejected-device record under
  `streaming_attention_backward_gb200_replay_blocked_9cac1dd40b/`.

### 2026-08-09 - TLTC-XLA-017 routed Grug input-adjoint replacement on GB200

- Hypothesis: Shuttle can recover a multi-output routed reverse region from the
  ordinary differentiated Grug train step and replace it with generated generic
  Contract/Map/Fold execution without owning model-level AD.
- Commit Hash: code checkpoint `2145f8eadf`; integrated artifact checkpoint
  `41af1a047b`.
- Config: one actual NVIDIA GB200, compute capability 10.0, driver 595.71.05,
  JAX/JAXlib/CUDA plugin 0.11.0, CUDA compiler 13.0.88, batch priority, one CPU,
  30 counterbalanced full-step sample pairs. No B200 result is involved.
- Recovered region: BF16 Contract with FP32 accumulation, generated two-output
  reverse scalar Map, second BF16 Contract, and a deterministic source Fold.
  The custom call returns both the reverse-Map physical buffer consumed by the
  still-XLA-owned grouped weight adjoint and the source adjoint. JAX owns the
  differentiation; the generated implementation contains no atomics or
  workload-name dispatch.
- Result: stock XLA measures 0.904177 ms and the transformed full train step
  measures 0.837056 ms, a 0.925766x ratio. The complete 53-leaf output has
  2.328e-10 maximum and 1.308e-14 mean absolute error; 51 leaves are bitwise
  equal in the direct comparison. There is one transformed custom call and 35
  observed handler executions. Every dynamic operand has runtime parameter
  ancestry; only the Fold-zero initial value is static.
- Determinism: the generated full-step output has one repeated hash versus nine
  for the stock path. This is whole-step evidence, not a claim about unrelated
  XLA reductions. The generated source Fold is deterministic by construction:
  one writer per source-feature visits compact edges in fixed destination-major
  order.
- Interpretation: the routed forward and input-adjoint slices now execute from
  natural JAX through post-SPMD HLO replacement. The next local ownership gap is
  the routed expert weight-gradient Contracts. Relation construction and
  collectives remain external until the local forward/backward regions are
  complete.
- Artifact:
  `lib/tile_lifetime/benchmarks/artifacts/xla_grug_routed_input_adjoint_gpu_gb200_v0/`.

### 2026-08-09 - TLTC-TRAIN-018 generic owner-Fold attachment on H100 and GB200

- Hypothesis: the visible output-times-output-cotangent Map and feature Fold can
  attach to the dQ query-owner traversal when both inputs are complete along the
  reduction axis, eliminating one launch and one redundant cotangent read.
- Commit Hash: implementation `9e73953092`; integrated artifacts
  `1b8b74bfe3`.
- Synthesis boundary: a workload-name-free `FoldAttachment` carries explicit
  `OwnerTileAvailability` evidence. Planning and emission reject partial
  reduction-axis tiles. The emitter revalidates the recovered Map/Fold and
  derives its physical head dimension from the proven complete feature axis.
  The row scalar remains materialized because the dK/dV traversal consumes it.
- H100 result: generated/flash-SDPA medians are 0.527968/0.462000 ms, a
  1.142788x ratio. The fused dQ plus output-dot stage measures 0.173603 ms and
  dK/dV measures 0.356864 ms. Relative to the sealed prior result, the full
  reverse improves 3.86%, and the affected stages improve 14.24%.
- GB200 result: generated/flash-SDPA medians are 0.455238/0.407830 ms, a
  1.116244x ratio. The fused dQ plus output-dot stage measures 0.140496 ms and
  dK/dV measures 0.314048 ms. Relative to the sealed prior result, the full
  reverse improves 5.95%, and the affected stages improve 20.52%. The accepted
  device was an actual NVIDIA GB200 at compute capability 10.0; no B200 result
  is involved.
- Correctness: H100 dQ/dK/dV maximum errors are
  0.015625/0.03125/0.0625; GB200 errors are 0.015625/0.03125/0.03125. Both
  generated paths have one stable repeated-output hash.
- Interpretation: this bounded generic placement closes most of the previous
  launch/read overhead without changing the reverse pipeline. dK/dV remains the
  dominant physical gap; further changes should target generic ownership and
  accumulation structure rather than another named attention optimization.
- Artifacts:
  `lib/tile_lifetime/benchmarks/artifacts/streaming_attention_backward_output_dot_fold_h100_72397500a2/`
  and
  `lib/tile_lifetime/benchmarks/artifacts/streaming_attention_backward_output_dot_fold_gb200_72397500a2/`.

### 2026-08-09 - TLTC-XLA-019 routed Grug weight-gradient Contracts on GB200

- Hypothesis: the two routed expert weight adjoints in the natural differentiated
  Grug HLO can use one generic group-batched Contract generator instantiated at
  two shapes, while XLA retains the placement collectives.
- Commit Hash: implementation `375d6ede60`; integrated artifact `577801269e`;
  ignored telemetry correction `1615367599`.
- Recovered boundaries: `[4,512,32] x [4,512,64] -> [4,32,64]` and
  `[4,512,32] x [4,512,32] -> [4,32,32]`. Both lower through the same generic
  `(E,K,M) x (E,K,N) -> (E,M,N)` strided-batched Contract interface. A shape
  mutation changes the generated right-feature extent without editing physical
  source. `psum.52` and `psum.53` consume the custom-call results directly and
  remain outside Shuttle.
- Numerical contract: BF16 operands, FP32 accumulation, one round-to-nearest-even
  BF16 output conversion, and `ALLOW_ROUNDING_REORDER`. `BITWISE_EXACT` is
  rejected because the source HLO does not specify a bitwise dot reduction
  tree. Each output element has one Contract owner; there are no atomics or
  output aliases.
- Result: the primary telemetry replay measures 0.774129 ms for the transformed
  full train step and 0.725728 ms for stock XLA, a 1.066692x ratio. An
  independent preceding 30-pair capture measures 0.744593/0.681649 ms, or
  1.092341x. Both use one actual NVIDIA GB200 at compute capability 10.0; no
  B200 result is involved.
- Correctness/audit: maximum and mean absolute errors are 3.725e-9 and
  1.188e-12; 49 of 53 result leaves are bitwise equal. Both generated handlers
  execute 35 times, the transformed HLO has exactly two targets, every operand
  has runtime parameter ancestry, and copy/transpose counts are unchanged.
- Interpretation: routed forward, input-adjoint, and weight-adjoint slices now
  each have clean natural-JAX-to-HLO generated execution proofs. They remain
  separate transformed runs. The next compiler integration step is to compose
  these replacements in one train step before taking ownership of placement
  collectives.
- Artifact:
  `lib/tile_lifetime/benchmarks/artifacts/xla_grug_routed_weight_gradient_gpu_gb200_v0/`.

### 2026-08-09 - TLTC-EVENT-020 Event Tensor workload linkage on GB200

- Hypothesis: schedule-level Event Tensor plans derived from generic task
  dependences can drive real runtime-relation and streaming Contract/Fold
  payloads on GPU without introducing MoE- or attention-specific event
  construction.
- Commit Hash: implementation checkpoint `e59c81d836`; hardened replay artifact
  checkpoint `2a915cd2dd`.
- Runtime relation path: runtime `RelationPlan` CSR counts, offsets, and source
  rows determine a ragged segmented-Contract task domain. The first physical
  lowering erases readiness to verified in-task program order.
- Streaming path: generic QK Contract, normalized-exponential Fold, and PV
  Contract task families use a bounded shared K/V buffer. The compiler derives
  the last PV consumer, circular-slot reuse edges, slot generations, and the
  physical CTA acquire/release barriers. Same-owner dependences erase only
  after a covering-order audit.
- Result: on one actual NVIDIA GB200 at compute capability 10.0, segmented
  primary/mutation medians are 0.073328/0.072800 ms and streaming
  primary/mutation medians are 0.074672/0.074224 ms. All four generated JAX
  typed-FFI paths are correct and bitwise deterministic across repeated
  execution. The allocation used one CPU, 32 GB host memory, and batch
  priority; no B200 result is involved.
- Scope: these are deliberately small FP32 generated payload kernels. They
  validate real tensor/CSR execution, physical Event Tensor realization,
  bounded-buffer generations, mutation, and the Torch-free JAX boundary. They
  are not grouped-GEMM or tensor-core attention throughput results.
- Verification: all six artifact hashes pass from a detached checkout, 43
  focused Event Tensor tests pass, and the complete tile-lifetime suite passes
  363 tests.
- Artifact:
  `lib/tile_lifetime/benchmarks/artifacts/event_tensor_workload_linkage_gb200_v0/`.

### 2026-08-09 - TLTC-XLA-021 combined routed Grug train step on GB200

- Hypothesis: the independently validated routed forward, input-adjoint, and
  two expert weight-gradient replacements can coexist in one natural
  JAX-differentiated Grug train step without changing their generic physical
  bodies or absorbing placement collectives.
- Commit Hash: composed transformation checkpoint `6d5afa01fd`; integrated
  replay artifact checkpoint `ad6ae00733`.
- Boundary: one GPU `PRE_SCHEDULER` transformation inserts four independent
  generated typed-FFI calls: routed forward Contract/Map/Contract/source Fold,
  routed input adjoint Contract/reverse Map/Contract/source Fold, and two
  instances of the generic group-batched weight-gradient Contract. This is not
  a routed-training megakernel. JAX owns differentiation; XLA retains relation
  construction, surrounding rematerialization, and placement collectives.
- Result: on one actual NVIDIA GB200 at compute capability 10.0, 30 paired
  counterbalanced whole-step samples measure 0.654897 ms for the combined
  generated path and 0.554336 ms for stock XLA, a 1.181407x ratio. No B200
  result is involved.
- Correctness/audit: maximum and mean absolute errors are 3.725e-9 and
  1.199e-12; 49 of 53 leaves are bitwise equal and the generated full-step
  hash is stable across all retained executions. Each target occurs once in
  post-roundtrip HLO and each handler executes 35 times. The input-adjoint
  auxiliary remains a direct operand of the first weight Contract; both weight
  results feed one direct external `psum`; copies remain 0 -> 0 and transposes
  change 51 -> 50. Generated sources contain no `atomicAdd`.
- Numerical policy: forward and input-adjoint regions are source ordered. The
  weight Contracts use BF16 operands, FP32 accumulation, BF16 RNE output, and
  `ALLOW_ROUNDING_REORDER`; the only static operand is a Fold identity.
- Verification: all 21 artifact hashes pass, 46 focused recovery/replacement
  tests pass, and the complete tile-lifetime suite passes 369 tests.
- Artifact:
  `lib/tile_lifetime/benchmarks/artifacts/xla_grug_routed_combined_gpu_gb200_v0/`.

### 2026-08-09 - TLTC-DIST-022 generic collective-completion recovery

- Hypothesis: the placement collectives left outside the combined routed train
  step can first become generic Shuttle plans without replacing XLA transport
  or introducing a workload-specific distributed lowering.
- Commit Hash: `1cdffc27b1`.
- Result: both direct expert weight-gradient `psum` consumers recover from
  post-SPMD HLO as a placement-partial input, a generic completion Fold, and a
  placement transition. Plans retain producer/result wiring, tensor shape and
  dtype, reducer, explicit reassociation policy, replica groups, global versus
  local device-ID semantics, and channel ID. They contain no NCCL or XLA
  implementation choice.
- Generality: mutating replica groups changes only the transport domain;
  mutating the reducer changes only Fold semantics. Reducer dtype mismatches
  are rejected. Recovery uses HLO structure and reducer computations rather
  than metadata or model names.
- Scope: this is the structural start of collective ownership. XLA continues
  to execute the all-reduces. The next steps are candidate transport selection,
  Event Tensor completion/visibility linkage, and an executable typed-FFI or
  existing-runtime lowering only after those contracts are explicit.
- Verification: four collective mutation/recovery tests and all six combined
  routed-training structural tests pass; scoped pre-commit checks pass.

### 2026-08-09 - TLTC-XLA-023 Torch-free JAX attention reverse execution

- Hypothesis: the generic Contract/Fold/Map/DomainRestriction reverse recovered
  from ordinary JAX VJP StableHLO can execute through JAX typed FFI without a
  Torch or Triton runtime while remaining within 1.20x of Flash-SDPA at the
  same state-recompute boundary.
- Commit Hash: implementation and benchmark source
  `e96c1cfba79b47b0b5a158c08fb969545e6d2726`.
- Boundary: BF16 Q/K/V and output cotangent to Q/K/V cotangents. The natural
  input does not expose forward output or log-sum-exp, so both Shuttle and the
  expert oracle recompute that state. The saved-state result remains a
  separately labeled lower boundary rather than the denominator.
- Physical execution: Triton 3.6 is an AOT build compiler only. Three emitted
  C launchers embed independently compiled forward, dQ, and dK/dV CUBINs and
  link with one generated XLA typed-FFI handler. The handler and DSO contain no
  Torch or Triton runtime dependency.
- H100 result: Shuttle/Flash-SDPA medians are 0.679229/0.634584 ms across 30
  counterbalanced samples with five iterations each, a 1.070353x ratio.
- GB200 result: a fresh `sm_100a` AOT build on an actual NVIDIA GB200 measures
  0.580810/0.555089 ms, a 1.046338x ratio. No H100 binary was reused and no
  B200 result is involved.
- Correctness: both generated paths are bitwise deterministic. Generated
  dQ/dK/dV maximum absolute errors are 0.03125 on both devices, with mean
  errors at most 0.000146. The expert output independently matches the natural
  JAX semantic VJP.
- Audit: the generated runtime imports neither Torch nor Triton before the
  benchmark-only oracle loads. The artifacts preserve raw samples, StableHLO
  and semantic fingerprints, generated sources and binary hashes, commands,
  hardware class, clocks, driver, and toolchain versions.
- Artifacts:
  `lib/tile_lifetime/benchmarks/artifacts/jax_streaming_attention_backward_ffi_h100_v0/`
  and
  `lib/tile_lifetime/benchmarks/artifacts/jax_streaming_attention_backward_ffi_gb200_v0/`.

### 2026-08-09 - TLTC-EVENT-024 SM90 streaming synchronization attachment

- Hypothesis: the synchronization configuration of the existing generic SM90
  streaming Contract/Fold skeleton can be derived from exact task dependences
  and bounded buffer lifetimes without changing its tensor-core payload.
- Commit Hash: `915fee0b7a`.
- Result: separate Q, K, and V task families now derive full-event edges,
  last-consumer reuse edges, physical slots, and phased generations. The Event
  Tensor attachment supplies one Q stage, the selected K/V pipeline depth, two
  full/empty barriers per stage, transfer/matrix worker counts, scheduler-ring
  arrival participants, and Q/K/V transaction bytes to the CuTe skeleton. The
  backend checks every supplied quantity against its tiled-MMA and layout
  construction.
- Mutations: changing pipeline depth changes K/V slots, generations, barrier
  storage, and the plan fingerprint. Changing the query tile from 128 to 64
  changes the matrix decomposition from two warpgroups to one and erases the
  pairwise scheduler event by program order.
- Validation: 26 focused Event Tensor/streaming tests pass. An H100 replay was
  attempted, but both the canonical and Event Tensor sources fail identically
  in the same Shuttle normalized-exponential CuTe dominance check before any
  samples. This is a compile-blocker record, not a performance result.
- Grouped-GEMM gap: the generic MoK-derived GMM exposes semaphore storage but
  not the producer/consumer ownership needed to derive its arrival counts. The
  work stops at that primitive-interface gap rather than copying literal
  counts into a generated header.
- Artifact:
  `lib/tile_lifetime/benchmarks/artifacts/event_tensor_sm90_compile_blocker_h100_v0/`.

### 2026-08-09 - TLTC-DIST-025 collective Event Tensor readiness

- Hypothesis: a recovered placement-changing completion Fold can induce tiled
  readiness and visibility from replica-group structure without selecting a
  collective transport backend or introducing routed-workload logic.
- Commit Hash: `4a3c8e86d6` (integrated from `51c941d461`).
- Result: every completion lowers to generic partial-value tiles, placement-
  transition completion tiles, and collective Fold tiles. Replica membership
  mechanically determines Event Tensor indegrees. Partial-to-transport edges
  require device visibility; transport-to-Fold edges require system-scoped
  release/acquire visibility.
- Mutations: changing replica groups alters the task domain and readiness
  counts without changing Fold semantics. Changing sum to maximum alters only
  the semantic Fold; it produces the same task/event program.
- Scope: this remains a structural schedule candidate. XLA still owns the
  executable collective transport, and no NCCL or device-side transport choice
  appears in the generic plan.
- Verification: 33 focused collective/Event Tensor tests and all 387
  tile-lifetime tests pass; scoped pre-commit checks pass.

### 2026-08-09 - TLTC-EVENT-026 H100 Event Tensor streaming replay

- Hypothesis: once the generic normalized-exp Fold carries its register state
  correctly across CuTe child regions, Event Tensor-derived pipeline and worker
  parameters can drive the real SM90 TMA/WGMMA skeleton without changing its
  result or materially changing latency.
- Commit Hash: Fold-state fix `ad1a0c3192`; replay artifact `ebc03388a5`.
- Debug result: aliasing state only in finalization did not repair MLIR
  dominance. The defining child region was the per-row Fold update. Binding
  row max, row sum, scale, and architecture before that update loop removed
  the verifier failure.
- Result: two counterbalanced 10-sample H100 captures measure 0.080272 ms for
  the repaired pre-Event source and 0.080352 ms for the Event Tensor source, a
  1.000997x ratio. Both paths have maximum/mean sampled error
  0.015625/1.14395e-4 and the same bitwise-stable output hash.
- Hardware: one actual H100, two host CPUs, 32 GB host memory, batch priority.
  No B200 or GB200 result is involved, and the allocation was released.
- Artifacts:
  `lib/tile_lifetime/benchmarks/artifacts/event_tensor_sm90_fold_alias_replay_h100_v1/`
  and
  `lib/tile_lifetime/benchmarks/artifacts/event_tensor_sm90_fold_state_replay_h100_v1/`.

### 2026-08-09 - TLTC-XLA-027 natural attention-reverse HLO replacement

- Hypothesis: the natural four-input/three-output JAX VJP can be replaced by
  the generated recompute typed FFI using only physical Contract, Fold, and
  DomainRestriction provenance, without parameter names or a model recognizer.
- Commit Hash: `903bd9b35a` (integrated from `b4d2f9cc29`).
- Result: a whole-entry replacement plan derives Q/K/V/output-cotangent and
  Q/K/V-cotangent roles from physical dataflow, validates the score scale and
  causal restriction, inserts canonical-layout copies around the FFI, and
  preserves the entry result layouts. Saved-state substitution and ambiguous
  or mismatched graphs fail closed.
- Scope: this checkpoint uses live natural JAX frontend HLO on CPU. A CUDA
  PRE_SCHEDULER callback smoke exists, but the minimal H100 allocation did not
  schedule. There is no claim yet that optimized GPU post-SPMD HLO proves or
  executes the same whole-entry replacement.
- Verification: 15 focused recovery, rewrite, typed-FFI, and StableHLO tests
  and all 392 tile-lifetime tests pass; scoped pre-commit checks pass.

### 2026-08-09 - TLTC-DIST-028 executable JAX collective completion

- Hypothesis: a recovered collective Fold and its Event Tensor completion can
  execute through a JAX-owned transport boundary without Shuttle taking over
  model AD or introducing a communication custom call.
- Commit Hash: `f7a02b9f35`.
- Result: the adapter binds the recovered reducer, numerical policy, replica
  groups, and system-scoped completion visibility to a JAX named-axis
  collective. Global device IDs map explicitly to JAX axis indices. Local
  replica-ID semantics, product reduction, and bitwise-fixed reduction trees
  fail closed.
- Four-device CPU replay: full-group sum, two-group maximum mutation, and the
  JAX-generated sum gradient have zero error against direct references. The
  forward StableHLO has one all-reduce and no custom call; the differentiated
  StableHLO has two all-reduces. Repeated output is deterministic.
- Scope: the JAX array result is the device completion dependency, not a
  host-observed event. Transport is whole-value, so it legally coarsens tiled
  readiness. GPU and multi-host transport remain untested; XLA still selects
  the physical communication implementation.
- Verification: 13 focused tests and all 397 tile-lifetime tests pass. The
  checksum manifest seals the replay results and JAX/JAXlib/host environment.
- Artifact:
  `lib/tile_lifetime/benchmarks/artifacts/jax_collective_completion_cpu_v0/`.

### 2026-08-09 - TLTC-XLA-028 live H100 attention-reverse replacement

- Hypothesis: the generic whole-entry replacement can match live optimized GPU
  HLO at XLA `PRE_SCHEDULER` and execute the generated typed-FFI handler, rather
  than only reproducing the boundary through a direct benchmark wrapper.
- Replay revision: `ab6c6493f1` on one H100 with two host CPUs, 32 GB host
  memory, and batch priority. The allocation was released after artifact copy.
- Replacement proof: the callback saw one natural JAX differentiated module,
  recovered five Contracts, four additive Folds, normalized-exponential state,
  and a causal DomainRestriction, then emitted one typed-FFI custom call. The
  handler executed 160 times. Exact source StableHLO and pre/post HLO modules
  are preserved.
- Measurement: 30 counterbalanced samples with five executions each measure
  4.466521 ms for stock natural JAX/XLA and 0.880135 ms for the transformed
  executable. Both are bitwise stable. dQ/dK/dV maximum absolute errors are
  0.03125 and mean errors are at most 0.000146.
- Acceptance caveat: this closes the live XLA replacement proof, not the expert
  performance gate. The closest preserved direct H100 results are 0.679229 ms
  for Shuttle typed FFI and 0.634584 ms for the expert recompute oracle, so the
  integrated replay is approximately 1.387x the expert number across different
  captures/toolchains.
- Layout gap: inputs require no copies and dQ is root-layout native. dK and dV
  require two output copies over 8 MiB of payload, or 16 MiB nominal
  read-plus-write traffic. The next bounded experiment is physical-layout-
  native FFI output binding and generated dK/dV strides so XLA can erase both
  copies.
- Build/runtime boundary: Torch is an incidental top-level import in the copied
  AOT input module and is build-only. Runtime imports contain neither Torch nor
  Triton, and the DSO links only CUDA and system libraries.
- Negative result: S=64 executes the replacement but fails the primary BF16
  bound with dV maximum absolute error 0.0625. It remains unaccepted rather
  than being relabeled a compiler failure.
- Artifacts:
  `lib/tile_lifetime/benchmarks/artifacts/xla_streaming_attention_backward_pre_scheduler_h100_v0/`
  and
  `lib/tile_lifetime/benchmarks/artifacts/xla_streaming_attention_backward_pre_scheduler_h100_s64_negative_v0/`.

### 2026-08-09 - TLTC-XLA-029 physical-layout-native reverse outputs

- Hypothesis: binding the generated dK/dV stores to the physical layouts of the
  exact live HLO results can erase both PRE_SCHEDULER output copies and close
  the integrated attention-reverse gap.
- Replay revision: `c6a4244052` on one H100 with two host CPUs, 32 GB host
  memory, and batch priority. The allocation was released after artifact copy.
- Structural result: Shuttle derives each output's minor-to-major permutation
  from the captured HLO, specializes the typed-FFI result layouts and generated
  strides, and emits dQ, dK, and dV directly in their requested layouts. The
  transformed HLO contains zero Shuttle boundary copies, down from two. Invalid,
  duplicate, incomplete, or absent output layouts fail closed.
- Measurement: 30 balanced samples with five executions each measure
  4.433139 ms for stock JAX/XLA, 0.838209 ms for direct layout-native typed
  FFI, and 0.829328 ms for the integrated replacement. Direct and integrated
  outputs are bitwise identical and deterministic. Maximum absolute errors are
  0.03125; mean errors are at most 0.000146.
- Performance conclusion: eliminating 16 MiB nominal copy traffic improves the
  prior integrated result by 5.8%, but remains approximately 1.307x the prior
  0.634584 ms matched expert measurement. The physical-layout-native plan is a
  candidate rather than a mandatory legalization rule.
- Bounded negative: a generic layout-selected block-pointer store remains
  correct and copy-free but measures 0.868246 ms, 4.7% slower than explicit
  arbitrary-stride stores. Its source and raw distribution are preserved and
  the candidate was reverted from the active emitter.
- Runtime boundary: the replay imports neither Torch nor Triton. Both remain
  build-only AOT dependencies; the generated DSO links no Torch or Triton
  library.
- Artifact:
  `lib/tile_lifetime/benchmarks/artifacts/xla_streaming_attention_backward_physical_layout_h100_v0/`.

### 2026-08-09 - TLTC-XLA-030 region-local Grug ownership audit

- The natural one-layer Grug post-SPMD artifact contains 68 physical Contract
  instructions. The current composed Shuttle transform proves four routed
  regions and one local attention-reverse region without metadata or model
  names: routed forward, routed input adjoint, two group-batched expert-weight
  Contracts, and normalized-exponential attention reverse.
- The routed input-adjoint call still consumes `dot.66`, a rematerialized first
  expert Contract, as the auxiliary input to its generated reverse Map. That
  value also feeds the independently generated forward-Map activation used by
  the second expert-weight Contract, so simply moving it into the existing call
  would duplicate rather than eliminate the Contract.
- The next clean ownership candidate is therefore a generic multi-output
  rematerialization plan: one Contract produces both the forward scalar-Map
  value and reverse scalar-Map value, which then feed the input-adjoint and two
  weight-gradient Contracts. The candidate must be recovered from shared
  Contract/Map dataflow, preserve both live outputs, and remain valid when the
  scalar Map changes. It must not be keyed on MoE, SwiGLU, or source metadata.
- Separate structural recovery found three unambiguous row Fold to full-shape
  BF16 final-Map reverse regions in the same entry. Their exact HLO has an FP32
  primal/inverse-scale boundary and a final BF16 cast, so the existing standalone
  four-buffer RMS helper cannot be substituted blindly. The integration must
  derive and generate the exact Map/Fold boundary while retaining an explicit
  deterministic-tree numerical policy.
- Hardware taxonomy remains strict: the preserved artifact reports an actual
  NVIDIA GB200. Secondary-cluster allocations, if used later, are B200
  portability data and must not be cited as GB200 evidence.

### 2026-08-09 - TLTC-XLA-031 collective replay and anonymous row Folds

- A two-device H100 replay now executes the JAX-owned collective boundary with
  EventTensor-derived completion plans. BF16 full-group sum, grouped maximum,
  and the JAX-owned gradient have zero maximum absolute error and bitwise-stable
  repeated hashes. StableHLO retains ordinary `all-reduce` operations and zero
  custom calls; Shuttle owns completion/readiness planning, while JAX/XLA owns
  AD and transport.
- The replay used two physically verified NVIDIA H100 80GB HBM3 devices. It is
  not B200 or GB200 evidence. Raw results and StableHLO are preserved under
  `lib/tile_lifetime/benchmarks/artifacts/jax_collective_completion_h100_v0/`.
- Generic physical-HLO recovery now finds and rewrites anonymous entry-local
  sum-Fold plus final-Map regions. The selected Grug composition adds two 8x32
  BF16 row-axis calls without using RMS, model, or metadata names. Metadata
  renaming leaves semantic fingerprints unchanged, and sequential replacement
  adds no copy or transpose adapter.
- Parallel reduction is admitted only under the explicit
  `ALLOW_ROUNDING_REORDER` policy; `BITWISE_EXACT` fails closed. The generated
  boundary consumes the exact FP32 values and retains the final BF16 cast.
- The routed, attention-reverse, collective, and axis-Fold focused suite passes
  35 tests. Changed-file lint, formatting, and type checks pass at canonical
  revision `4c74101ce9`.

### 2026-08-09 - TLTC-EVENT-008 grouped-Contract synchronization ABI

- Hypothesis: the remaining grouped-GEMM event boundary can be expressed as a
  generic synchronization descriptor and derived Event Tensor task graph
  without copying MoK's MoE event graph or treating CUDA barrier counts as the
  semantic dependency relation.
- Result: a generic grouped-Contract descriptor now exposes cooperative task
  owners, bounded operand stages, producer/consumer cardinalities, and release
  points. Mechanical task-relation derivation emits a fingerprinted SM100 ABI.
  The wrapper statically checks the selected two-CTA cluster, six stages, and
  BF16 tile bytes against that ABI.
- Important distinction: the logical operand-ready indegree is two cooperative
  transfer tasks. Its TMA realization uses transaction completion plus 65,536
  expected bytes and a physical arrival-count argument of one. Both facts are
  represented; the physical encoding is not mistaken for the logical count.
- Mutation evidence: changing cluster cardinality from two to four changes the
  event domains and counts; changing stages from two to three changes buffer
  capacity, slots, generations, and the schedule fingerprint. Backend and
  generated-include drift fail closed.
- GPU proof: revision `30c0ba6bfc` built for SM100a on a physical NVIDIA GB200,
  driver 595.71.05, with Torch 2.10.0+cu130 and NVCC 13.0.88. The runtime ABI
  fingerprint matched. W2 correctness passed with maximum absolute error
  0.0148849, mean absolute error 0.00112154, and no NaNs or infinities. PTXAS
  reported 255 registers, five barriers, 224 bytes of static shared memory, and
  no spills. The allocation was released after the proof.
- Exact boundary: EventTensor owns the generated synchronization ABI and counts
  at the wrapper boundary. The external generic grouped-GEMM primitive still
  owns internal barrier arrival/wait instruction placement, phase advancement,
  TMA issue, and accumulator release. MXFP8 scale-pipeline synchronization is
  not covered.
- Artifact:
  `lib/tile_lifetime/benchmarks/artifacts/event_tensor_grouped_contract_sm100_gb200_v0/`.

### 2026-08-09 - TLTC-XLA-032 shared Contract and exact target audit

- The natural Grug replacement now has a shared-map composition. One recovered
  Contract emits two generated scalar-Map outputs and leaves only the
  nonoverlapping input-adjoint remainder in XLA. The scalar Maps are generated
  from recovered expressions and do not dispatch on MoE or activation names.
- The transformed-HLO audit parses exact `custom_call_target` attributes. Every
  selected target must occur exactly once before correctness, warmup, or timing.
  Generated adapter names containing target text no longer inflate the count.
- Post-execution failures preserve raw timing samples, hashes, handler counts,
  and numerical comparison in an explicitly unaccepted result.
- Canonical revisions: `fdd8380cf0` for shared-map composition and `9798ebd794`
  for the exact target audit.

### 2026-08-09 - TLTC-XLA-033 H100 shared-map replay rejected by determinism

- A physical H100 replay executed all seven selected handlers 35 times. Each
  exact custom-call target occurred once in the transformed HLO.
- Ordered-FP correctness passed against XLA across 53 leaves. Thirty-eight leaves
  were bitwise equal; maximum absolute error was `9.760261e-7` and mean absolute
  error was `7.976652e-11`.
- XLA produced one output hash in 30 runs. Shuttle produced the dominant hash in
  27 runs and an alternate hash in 3 runs. The determinism guard rejected the
  run.
- Median latency was `0.528433 ms` for XLA and `0.603667 ms` for Shuttle, or
  `1.142374x`. The timing is diagnostic because determinism failed.
- The benchmark now records per-leaf hashes so the next replay can identify the
  varying state or metric before a component-level rerun.
- Artifact:
  `lib/tile_lifetime/benchmarks/artifacts/xla_grug_shared_map_h100_unaccepted_v0/`.

### 2026-08-09 - TLTC-XLA-034 generated RMS reverse H100 replay

- Ordinary JAX defines an uncentered row normalization and owns its VJP. Shuttle
  recovers the exported StableHLO as a three-stage generic AxisFoldPipeline with
  FP32 partial state and BF16 input/output boundaries.
- The generated typed-FFI path passed the post-roundtrip audit with one custom
  call, no copy or transpose adapters, two expected roots, and dead internal
  source instructions. Both output hashes were deterministic.
- Input-cotangent maximum and mean absolute errors were `0.0078125` and
  `1.9742053e-8`. Feature-scale-cotangent errors were `0.00390625` and
  `9.536743e-7`.
- On a physical H100, the generated median was `0.102763 ms` and matched XLA was
  `0.067930 ms`, or `1.512778x`. Correctness passes; performance does not meet the
  `1.20x` target.
- Artifact:
  `lib/tile_lifetime/benchmarks/artifacts/row_normalization_backward_h100_fdd838_v0/`.

### 2026-08-09 - TLTC-XLA-035 deterministic input-adjoint remainder

- The shared-map composition now generates two generic rank-two BF16 Contracts
  for the remaining input-adjoint matrix products and one generic
  source-indexed Fold for the reverse route merge.
- The Fold traverses route edges in source order, assigns one thread to each
  output element, and contains no atomics. Contribution and reducer arithmetic
  come from scalar ASTs; reducer and shape mutations regenerate the source.
- XLA retains only slice, transpose, and reshape views plus relation/index and
  placement operations. No input-adjoint arithmetic remains outside Shuttle.
- A static audit of the failed H100 replay found two colliding BF16 updates per
  source row in the former XLA `%scatter-add.42`. This is the leading explanation
  for the 3-of-30 alternate hashes. The deterministic Fold requires a bounded
  H100 replay before the diagnosis is confirmed.

### 2026-08-09 - TLTC-XLA-036 accepted ten-call H100 replay

- Revision `992a7467da` replaced the residual BF16 scatter with the generated
  deterministic source Fold and added both generic input-adjoint Contracts.
- All ten exact custom-call targets occurred once in transformed HLO and every
  handler executed 35 times. The evidence file was written after target,
  handler, correctness, and determinism guards and before summary assembly.
- Both XLA and Shuttle produced one output hash across all 30 measured
  repetitions. Ordered-FP comparison covered 53 leaves: 38 were bitwise equal,
  maximum absolute error was `9.760261e-7`, and mean absolute error was
  `7.977959e-11`.
- Median latency was `0.585042 ms` for XLA and `0.689586 ms` for Shuttle, or
  `1.178695x`. This satisfies the bounded `1.20x` training-region target.
- The run used one physical H100, requested one CPU, incurred the platform's
  four-CPU minimum, and made no retry. The allocation was explicitly released.
- Artifact:
  `lib/tile_lifetime/benchmarks/artifacts/xla_grug_shared_map_h100_992a7467_v0/`.

### 2026-08-09 - TLTC-XLA-037 remaining Grug arithmetic ownership

- A live-entry audit of the exact ten-call transformed HLO counts 46,145,536
  dot FLOPs inside Shuttle and 8,720,384 dot FLOPs remaining in XLA, or 84.1%
  static ownership for this padded fixture.
- The largest remaining region is the weighted RelationProgram reverse:
  `%dot.69` plus its per-edge hidden Fold, inverse relation, and router VJP. It
  accounts for 48.1% of remaining dot work and computes 512 padded rows although
  only the first 16 survive.
- The next generic ownership targets are normalized-exponential loss Contracts
  and Fold (26.3% of remaining dot work), followed by repeated GatedNorm/RMS
  training regions (21.0%). These three families cover 95.5% of the residual dot
  work.
- JAX continues to own AD and the natural frontend. Collectives, relation-index
  construction, views, and non-bottleneck runtime plumbing remain outside the
  current arithmetic replacement boundary.

### 2026-08-09 - TLTC-XLA-038 RMS component profile is non-attributable

- The exact-source H100 replay reproduced the full generated RMS reverse gap at
  `0.104100 ms` versus `0.070481 ms` XLA, or `1.476984x`. Correctness and
  deterministic-output checks passed.
- The separately generated input and feature components are alternate Fold
  programs, not the K1 and K2 kernels from the full pipeline. They change input
  algebra, dtypes, output dtypes, and the K0 scratch interface. Their outputs
  differ from the full generated path by maxima `0.0312643` and `0.512909`.
- Their isolated timings therefore cannot attribute the full latency gap or
  justify a schedule change. This is preserved as a negative benchmark result.
- Two bounded profiler attempts did not yield per-kernel timings. The first
  stopped before GPU execution because the synced package lacked Git metadata.
  The corrected attempt executed exactly one unchanged full handler inside a
  CUDA profiler range, with correct deterministic hashes, but `nsys stats`
  rejected its stale SQLite export before the report was transferred. The
  terminal pod could not be copied from. A forced-export script is prepared but
  remains unlaunched; no timing claim is made from either attempt.
- Artifact:
  `lib/tile_lifetime/benchmarks/artifacts/jax_row_normalization_backward_h100_components_non_attributable_fdd838/`.

### 2026-08-09 - TLTC-XLA-039 twelve-call weighted reverse replay

- The natural Grug composition now generates a weighted RelationProgram reverse
  as a generic rank-two Contract followed by scalar edge Map, hidden Fold, and
  deterministic source-slot Fold. The placement all-reduce and router VJP remain
  explicit XLA boundaries.
- On a physical H100, all twelve exact targets occurred once and all handlers
  executed 35 times. The generated path passed ordered-FP correctness and
  produced one output hash across 30 samples.
- The result is unaccepted for performance: generated median was `0.658288 ms`
  versus `0.527480 ms` XLA, or `1.247988x`, above the `1.20x` ceiling.
- XLA itself produced two hashes. Only the MLP GatedNorm down-weight leaf varied;
  its alternate hash appeared in the final sample, alongside a `1.292757 ms`
  baseline latency. The generated result remained stable in that sample.
- The next bounded optimization is demand-driven Contract-domain narrowing. The
  current generated Contract recomputes 512 padded rows although its Fold
  consumes only 16 logical relation edges. Domain narrowing must be derived from
  the consumer slice/index relation, not MoE identity.
- Artifact:
  `lib/tile_lifetime/benchmarks/artifacts/xla_grug_shared_map_h100_unaccepted_2732ef51_v0/`.

### 2026-08-09 - TLTC-XLA-040 migration checkpoint

- Commit `1ee45b825d` derives a rank-two Contract's demanded row domain from its
  single-consumer contiguous slice/view chain. It moves the weighted reverse
  custom-call site from `[512,32]` to `[16,32]` while retaining the full operand
  ABI and a separate generated Fold.
- The generic transformation rejects noncontiguous slices and competing
  consumers, supports nonzero row offsets, and has no MoE, Grug, or instruction
  name dispatch.
- Static work falls from 4,194,304 to 131,072 Contract FLOPs and estimated ideal
  BF16 traffic falls from 172,032 to 13,312 bytes. These estimates are not GPU
  measurements.
- Canonical verification passed 26 focused tests, Pyrefly, scoped pre-commit,
  and `git diff --check`. No GPU replay was launched before migration and no GPU
  allocation remains active.
- Resume with one bounded physical-H100 replay of the twelve-call natural Grug
  harness. If it remains above `1.20x`, evaluate generic Contract-plus-nested-
  Fold composition rather than adding workload-specific reverse code.

### 2026-08-09 14:15 PDT - TLTC-XLA-041 narrowed twelve-call H100 replay

- Hypothesis: deriving the weighted reverse Contract's demanded row domain from
  its sole contiguous-slice consumer will remove enough padded Contract work to
  bring the twelve-call natural Grug boundary back under `1.20x` XLA.
- Commit Hash: executed Shuttle revision `da49b94c359104690c2b8f98192300605cbd292e`,
  containing demand-narrowing checkpoint `1ee45b825d`.
- Command: `xla_grug_routed_combined_gpu_custom_call.py --architecture sm_90a
  --composition-mode shared_map_xla_remainder --warmup 4 --repeats 30`, with
  the exact NVCC 13.2.78 and repository paths preserved in the artifact README.
- Config: one physical NVIDIA H100 80GB HBM3, compute capability 9.0, driver
  595.71.05, 700 W power limit, JAX/JAXLIB 0.11.0, Torch 2.11.0+cu128, Triton
  3.6.0, and NVCC 13.2.78. One CPU was requested and normalized to four. The
  benchmark was invoked once with no replay retry.
- Result: all twelve exact targets occurred once, every handler executed 35
  times, 30 counterbalanced pairs were retained, both paths produced one stable
  output hash, and ordered-FP correctness passed with maximum absolute error
  `9.760261e-7` and mean absolute error `7.977502e-11`.
- Result: generated median was `0.647562 ms` versus `0.521619 ms` XLA, or
  `1.241446x`. This remains above `1.20x` and is therefore unaccepted.
- Interpretation: row narrowing improves generated latency by `1.63%` and the
  ratio by `0.006542` absolute versus the prior `1.247988x` replay, but removes
  only `0.004866 ms` from the whole-step gap. Static padded-FLOP elimination is
  real but is not the dominant remaining fixed-call cost at this shape.
- Next action: if this boundary is pursued, evaluate exactly one generic
  Contract-plus-nested-Fold composition candidate rather than tuning the
  narrowed Contract or adding workload-specific MoE code.
- Artifact:
  `lib/tile_lifetime/benchmarks/artifacts/xla_grug_shared_map_h100_narrowed_unaccepted_da49b94c_v0/`.
- Infrastructure: holder `/dlwh/dev-gpu-codex-h100-narrowed-da49` was released;
  local session status and pod lookup both verified no active allocation.

### 2026-08-09 14:32 PDT - TLTC-XLA-042 normalized-exp Contract training semantics

- The next natural Grug ownership target is expressed without a loss or model
  dispatch key as Contract, score Map, Fold-domain restriction, maximum and sum
  Folds, indexed selection, and a final Map. JAX-provided output and saved-state
  cotangents feed a generated score reverse Map followed by generic input and
  operand reverse Contracts.
- The scalar algebra now includes generic `log` evaluation, differentiation,
  CUDA rendering, and CuTe/QuACK rendering. The score Map may be identity or a
  finite tanh soft cap; the same Contract/Fold family and reverse construction
  handle both mutations.
- The materialized CPU reference matches an independent normalized-exp
  forward/reverse calculation with a restricted padded Fold domain, invalid
  rows, indexed selection, nonzero output cotangent, and nonzero saved-state
  cotangent. Invalid or restricted selected coordinates fail closed.
- This is semantic and source-generation groundwork only. Natural post-SPMD HLO
  recovery, exact replacement/liveness audit, GPU generation, and performance
  remain open. The bounded implementation plan is
  `.agents/projects/tile_lifetime_compiler/normalized_exp_contract_training_plan.md`.
- Verification: 31 focused tests pass; Pyrefly reports zero errors; scoped
  pre-commit and `git diff --check` pass.

### 2026-08-09 15:02 PDT - TLTC-XLA-043 natural normalized-exp reverse recovery

- The natural Grug post-SPMD module now structurally recovers the output-head
  reverse as one score Contract, saved normalized-exponential state, Fold-domain
  validity, indexed selection, a row cotangent, and input/operand reverse
  Contracts. Recovery uses entry dataflow, opcodes, shapes, and dot dimension
  relations; frontend metadata and instruction spelling do not participate.
- Indexed selection is reduced to its compact inputs: one selected coordinate
  and validity bit per row. The generated boundary does not need the materialized
  one-hot tensor. The Fold restriction similarly consumes the one-dimensional
  validity domain rather than its broadcast materialization.
- A local two-output typed-FFI replacement is planned at the score-cotangent
  boundary. It rewires the natural input- and operand-cotangent consumers while
  leaving placement collectives outside the generated region. The old reverse
  dataflow becomes dead after ordinary XLA cleanup.
- The current work is recovery and exact HLO boundary formation only. A physical
  generated CUDA body, JAX registration, H100 execution, and performance remain
  open.
- Verification: 13 focused normalized-exp/autodiff tests pass; Pyrefly reports
  zero errors; scoped pre-commit and `git diff --check` pass.

### 2026-08-09 15:24 PDT - TLTC-XLA-044 generated normalized-exp reverse body

- The recovered natural Grug output-head reverse now lowers to a bounded generic
  one-CTA Contract/Map/Fold/Contract family. It generates the score Contract,
  score Map, Fold-domain restriction, normalized-exponential update, indexed
  selected-coordinate correction, and both reverse Contracts without calling a
  named loss, softmax, or attention kernel.
- Numerical boundaries are explicit: the score Contract rounds to BF16 before
  the generated score Map, the score cotangent rounds to BF16 before both reverse
  Contracts, and all Contract reductions use deterministic source-ordered FP32
  accumulation. The compact selection ABI carries indices and row validity
  rather than a materialized one-hot tensor.
- An identity score Map has a constant generated derivative. A tanh soft-cap
  mutation changes the generated scalar Map and derivative while retaining the
  same physical Contract/Fold family, shapes, and handler interface.
- A Torch-free JAX typed-FFI registration/call boundary validates all seven
  physical input shapes and dtypes before dispatch. GPU compilation, numerical
  execution, integration into the full natural Grug harness, and performance
  remain unmeasured.
- Verification: 17 focused codegen, FFI-boundary, HLO-recovery, and semantic
  tests pass; Pyrefly reports zero errors; scoped pre-commit passes.
### 2026-08-09 15:36 PDT - TLTC-XLA-045 fused weighted reverse bootstrap failure

- Commit `68c7770410` integrates a bounded one-CTA Contract–scalar Map–hidden
  Fold–source Fold as an explicit eleven-call natural Grug composition. The
  generated source preserves an FP32 Contract accumulator followed by an RNE
  BF16 shared-memory boundary, ordered FP32 inner and outer Folds, one owner per
  source, and no atomics.
- The only attempted physical-H100 invocation did not reach HLO recovery or
  generated-handler compilation. The repository-pinned JAX/JAXLIB 0.10.1 CUDA
  environment did not expose the private `jaxlib._hlo` module required by the
  benchmark harness and raised `ModuleNotFoundError` before the correctness
  call or any warmup.
- No retry was made. There are no timings, correctness results, or performance
  acceptance claim for the fused candidate. A future physical replay requires
  explicit authorization after pinning a compatible JAX HLO API or reviewing a
  harness migration.
- The device was an NVIDIA H100 80GB HBM3 with driver 595.71.05 and a 700 W
  power limit. The batch-priority one-GPU/four-CPU allocation was released, and
  pod lookup verified it inactive.
- Artifact:
  `lib/tile_lifetime/benchmarks/artifacts/xla_grug_shared_map_h100_fused_weighted_reverse_bootstrap_failure_68c777_v0/`.

### 2026-08-09 - TLTC-XLA-043 JAX HLO rewrite compatibility boundary

- Commit `274e1b9238` removes the combined Grug harness's direct import of
  `jaxlib._hlo`. Serialized callback protos now roundtrip through the HLO module
  type obtained from public `lower(...).compiler_ir(dialect="hlo")`.
- JAX exposes no public HLO text parser. The remaining parser dependency is
  isolated behind one runtime adapter that prefers
  `jaxlib._jax.hlo_module_from_text` and retains `jaxlib._hlo` only as a
  compatibility fallback. Rewritten text cannot be returned to the
  pre-scheduler callback without this parser or a future public equivalent.
- A CPU preflight on the repository-pinned JAX/JAXLIB 0.10.1 proves the public
  proto roundtrip and `_jax` text parser work, then rejects the runtime because
  `jax.extend.xla` and its HLO transformation registry are absent. This now
  fails before GPU inspection or allocation rather than inside a physical run.
- The next physical replay requires a matched JAX/JAXLIB build with the public
  compiler-IR proto path, a compatible text parser, and
  `jax.extend.xla.register_hlo_module_transformation` plus its clear operation.
  The prior successful Grug replay used JAX/JAXLIB 0.11.0. No GPU was launched
  for this compatibility fix.
### 2026-08-09 15:44 PDT - TLTC-XLA-047 same-domain RMS Fold coalescing

- A bounded schedule comparison compiled the same ordinary JAX RMS backward VJP
  and StableHLO graph into separate and same-domain-coalesced generic AxisFold
  pipelines. Semantic fingerprints, BF16 inputs, and the
  `allow_rounding_reorder` numerical policy were identical.
- Thirty H100 samples covered all six execution-order permutations five times.
  Separate stages measured `0.096221 ms`; coalesced row stages measured
  `0.089681 ms`; matched XLA measured `0.061414 ms`.
- Coalescing improved generated median latency by `6.797%` and reduced the
  generated launch count from three to two. One-call Nsight profiling measured
  `89.856 us` of separate-stage kernel time and `83.616 us` of coalesced kernel
  time, a `6.944%` reduction.
- Both generated schedules produced the same deterministic hashes. Maximum
  absolute error versus natural JAX was `0.0078125` for the input cotangent and
  `0.00390625` for the feature-scale cotangent.
- This schedule choice helps, but does not close the component gap: coalesced
  execution remains `1.460268x` matched XLA. No additional tuning was run and no
  acceptance claim is made.
- The raw Nsight report was copied before stats conversion. Its secure local
  copy is withheld from Git because Nsight embedded secret-bearing environment
  records; the publishable artifact retains its checksum and derived launch
  tables. Artifact:
  `lib/tile_lifetime/benchmarks/artifacts/jax_row_normalization_backward_h100_coalesced_v0/`.

### 2026-08-09 - TLTC-XLA-048 compact normalized-exp forward

- Commit `554e4ecc65` recovers the padded output-head forward from the natural
  Grug post-SPMD HLO as a compact score Contract, Fold-domain restriction,
  normalized-exponential max/sum Folds, indexed selection, and two observable
  outputs: per-row loss and saved log-normalizer state.
- The compact operands are proven to share their unpadded primal bases with the
  previously recovered reverse. The replacement consumes `[8,32]` and
  `[32,128]` BF16 operands instead of executing XLA's zero-padded
  `[128,64] @ [64,128]` physical Contract.
- A bounded generated one-CTA family emits the Contract, scalar score Map,
  source-ordered FP32 max/sum Folds, validity restriction, and selected-index
  finalization. The score Contract has an explicit RNE BF16 boundary. A tanh
  score-map mutation regenerates through the same physical family.
- Forward and reverse replacement compose as two generic typed-FFI calls. The
  reverse consumes the generated forward's saved state directly; the old
  loss/state consumers are rewired and the padded subgraph is left dead for
  ordinary XLA cleanup.
- GPU compilation, numerical execution, and performance are unmeasured.
  Twenty-one focused forward, reverse, FFI-boundary, and independent semantic
  tests pass; scoped pre-commit and Pyrefly pass.

### 2026-08-09 - TLTC-XLA-049 normalized-exp reverse physical proof

- A CPU-only preflight compiled, linked, and loaded the generated identity and
  tanh-softcap normalized-exp reverse families for `sm_90a`. Both typed-FFI
  symbols resolved. The mutation changed generated Map/derivative source while
  retaining the same generic physical extents; it was not run on a GPU.
- The authorized one-H100 run executed the identity family through Torch-free
  JAX typed FFI at shape `[8,32] @ [32,128]`. The fixture includes 29 restricted
  Fold positions, three invalid rows, and eight nontrivial selected indices.
- Generated input and operand cotangents are bitwise identical to both the
  matched explicit JAX reverse and an independent natural JAX objective
  differentiated with `jax.vjp`. Three generated runs have identical hashes,
  and the handler count is exactly `30,013`.
- Thirty counterbalanced samples measured `0.029879 ms` for generated FFI and
  `0.037568 ms` for matched JAX, or `0.795326x`. Compilation, saved-state
  forward work, and input generation are outside both boundaries.
- This is a small one-CTA component proof, not full-Grug or attention-backward
  acceptance. No schedule tuning or second GPU invocation was run. Artifact:
  `lib/tile_lifetime/benchmarks/artifacts/jax_normalized_exp_contract_reverse_h100_v0/`.

### 2026-08-09 - TLTC-XLA-050 natural Grug normalized-exp ownership

- A new explicit `shared_map_fused_reverses` composition retains the accepted
  fused weighted RelationProgram reverse and adds two generated normalized-exp
  targets. The compact forward is inserted before the reverse, and recovery of
  the reverse is rerun on that transformed graph so its saved-state operand is
  exactly `shuttle.generated.normalized_exp_contract_forward.output.1`.
- The transformed module contains thirteen generated calls, one occurrence of
  each selected target, and no live old normalized-exp forward or reverse
  arithmetic outside shared boundary inputs. The forward audit records 69 dead
  old instructions. The two reverse outputs retain their original placement
  paths through `reshape.198 -> psum.48` and `slice.87 -> psum.49`.
- Forward and reverse use the same generated scalar score Map. A tanh soft-cap
  mutation changes both generated sources and makes the reverse generator emit
  its derivative without changing either typed-FFI handler family. The sources
  contain no Torch or Triton reference. Runtime compilation still audits linked
  libraries with `ldd`, rejects Torch/Triton linkage, counts both handlers, and
  rejects semantic atomic accumulation.
- The existing eleven-call `shared_map_fused_weighted_reverse` composition is
  unchanged and remains independently selectable. Fifty-four focused recovery,
  code-generation, public compiler-IR preflight, FFI-boundary, and composition
  tests pass; Pyrefly reports zero errors. No GPU was launched, so numerical
  execution and latency remain unmeasured.
- A physical replay must first pass `require_hlo_rewrite_runtime()` on CPU. It
  requires matched JAX/JAXLIB exposing public compiler-IR proto roundtrip,
  `jax.extend.xla` transformation registration/clear APIs, and a compatible HLO
  text parser; the last working environment used JAX/JAXLIB 0.11.0. On an H100
  CUDA-13 checkout, run:

  ```bash
  PYTHONPATH="$PWD/lib/tile_lifetime/src:$PWD" /app/.venv/bin/python \
    lib/tile_lifetime/benchmarks/xla_grug_routed_combined_gpu_custom_call.py \
    --nvcc /app/.venv/lib/python3.12/site-packages/nvidia/cu13/bin/nvcc \
    --architecture sm_90a \
    --repository "$PWD" \
    --artifact-directory /tmp/shuttle-normalized-exp-grug-raw \
    --output /tmp/shuttle-normalized-exp-grug-raw/summary.json \
    --composition-mode shared_map_fused_reverses \
    --warmup 4 \
    --repeats 30
  ```

### 2026-08-09 - TLTC-XLA-051 normalized-exp forward physical proof

- A compile/link/load-only preflight built both identity and tanh-softcap score
  Maps for the generated compact normalized-exp forward. Both `sm_90a` typed-FFI
  symbols resolved before the generated GPU invocation. The mutation retained
  the same generic physical extents and was not run on the GPU.
- The single authorized H100 invocation executed `[8,32] @ [32,128]` through
  Torch-free JAX typed FFI. The fixture contains 29 restricted Fold positions,
  two invalid rows, and nontrivial selected indices including an out-of-range
  sentinel and a restricted coordinate.
- Maximum error versus matched explicit JAX and an independent natural JAX
  formulation is `4.76837158203125e-07` for both loss and saved state. Three
  generated executions have identical hashes, and the handler count is exactly
  `30,013`. Removing the explicit BF16 score boundary changes output hashes.
- Thirty counterbalanced samples measured `0.055004 ms` for generated FFI and
  `0.058033 ms` for matched JAX, or `0.947791x`. The raw samples cross two clock
  regimes; the median paired ratio is `0.900688x`. No tuning or second physical
  invocation was run.
- This is a bounded component result, not a full-Grug or attention-forward
  acceptance result. The batch-priority one-H100/four-CPU allocation was
  released, and both session status and pod lookup verified it absent. Artifact:
  `lib/tile_lifetime/benchmarks/artifacts/jax_normalized_exp_contract_forward_h100_v0/`.

### 2026-08-09 - TLTC-XLA-052 thirteen-call natural Grug replay

- Hypothesis: the `shared_map_fused_reverses` composition can own the compact
  normalized-exp forward/reverse in addition to the routed, weighted,
  attention-reverse, and axis-Fold regions while preserving the natural JAX
  train step and remaining within the `1.20x` whole-step target.
- Measured source: `e34116793d`; holder source: current Iris
  `eafa4d49f7`. A local and in-pod `require_hlo_rewrite_runtime()` preflight
  passed on JAX/JAXLIB 0.11.0 with public compiler-IR proto roundtrip,
  `jaxlib._hlo.hlo_module_from_text`, and `jax.extend.xla` transformation
  registration/clear. The benchmark command is preserved in the artifact.
- Config: one NVIDIA H100 80GB HBM3, compute capability 9.0, driver 595.71.05,
  700 W power limit, batch priority, one requested CPU, 32 GB memory, 50 GB
  disk, four warmups, and 30 counterbalanced samples. Clocks were not pinned.
  The benchmark was invoked once with no retries or schedule tuning.
- Structural result: all thirteen selected targets occur exactly once, and
  every handler executes exactly 35 times. The generated normalized-exp
  reverse consumes the generated forward saved state; audits report the old
  normalized-exp forward/reverse arithmetic dead. Six placement all-reduces
  remain external. Runtime source/linkage is Torch/Triton-free and contains no
  semantic atomic accumulation.
- Correctness: all 53 output leaves match the ordered-FP policy with maximum
  absolute error `9.760260582e-7`, mean absolute error `7.977541458e-11`, and
  38 bitwise-equal leaves. Generated and stock XLA paths each retain one stable
  whole-tree hash over all samples.
- Performance: generated median `0.731302 ms`; stock XLA median `0.591416 ms`;
  ratio `1.236525x`; absolute gap `0.139885 ms`. The structural and numerical
  ownership proof passes, but the performance result is unaccepted because it
  exceeds `1.20x`.
- Interpretation: the remaining result is a connected-region launch/schedule
  problem rather than an ownership or semantic-boundary failure. The simple
  gap-per-call quotient is `10.760 us` across thirteen calls, but it is not a
  causal attribution. Do not tune the normalized-exp components in isolation;
  the next bounded experiment should reduce generic region boundaries or
  attach compatible stages while retaining the exact generated semantics.
- Allocation cleanup: earlier holder-submission attempts failed before job
  creation because of a large dirty bundle and stale Iris client protocol, so
  they consumed no GPU time. The successful one-H100 holder was released after
  artifact copy; Iris reports it killed and Kubernetes reports no pod.
- Artifact:
  `lib/tile_lifetime/benchmarks/artifacts/xla_grug_shared_map_h100_fused_reverses_unaccepted_e3411679_v0/`.

### 2026-08-09 15:19 PDT - TLTC-XLA-053 low-rank gated-product training recovery

- Commit `dae11b30e4` structurally recovers six forward/rematerialized
  low-rank gated-product realizations and four JAX-owned reverse families from
  the pinned natural post-SPMD Grug HLO. Recovery uses rank-two Contract
  dimensions, scalar dataflow, root liveness, layout-stripped value identity,
  and shared parameter origins. Removing frontend metadata produces identical
  plans.
- The repeated family contains 12 forward/rematerialized, eight input-adjoint,
  and eight weight-adjoint Contracts. After applying the compact
  normalized-exponential forward and reverse replacements, these 28 Contracts
  account for `1,835,008` of `2,232,320` live dot FLOPs, or `82.2%`, in the
  pinned small padded fixture.
- Every forward and reverse scalar Map is a source-ordered imported AST. All six
  hidden Maps share one digest, all six output Maps share one digest, and each
  of the three reverse Maps shares one digest across four families. A tanh
  hidden-Map mutation regenerates through the same scalar generator without
  changing either Contract.
- BF16 Contract and Map boundaries, BF16 weight-adjoint outputs before the FP32
  optimizer conversion, and upstream placement all-reduces are explicit. JAX
  remains responsible for AD.
- This is static recovery evidence. No HLO replacement, GPU execution,
  launch-count reduction, or latency result is claimed. The design and exact
  instance audit are in
  `.agents/projects/tile_lifetime_compiler/grug_low_rank_gated_product_training.md`.
- Verification: four focused tests pass; Pyrefly reports zero errors; scoped
  pre-commit and `git diff --check` pass.

### 2026-08-09 - TLTC-XLA-054 thirteen-call H100 attribution trace

- Profiled the exact canonical `ac34883d03` thirteen-call
  `shared_map_fused_reverses` path and stock XLA once in the same process on
  one batch-priority H100. JAX/JAXLIB 0.11.0 and the public HLO rewrite runtime
  passed preflight locally and in the allocation. This was attribution only:
  no tuning, candidate sweep, or implementation change was made.
- The profiler perturbs this sub-millisecond program, so the sealed 30-sample
  result remains performance truth: Shuttle `0.731302 ms`, stock XLA
  `0.591416 ms`, and a `0.139885 ms` gap. The traced process retained all 13
  exact targets, 53 matching leaves, maximum absolute error
  `9.760260582e-7`, and 38 bitwise-equal leaves.
- In the final warmed trace, stock XLA occupies one `261.151 us` CUDA graph.
  Shuttle spans `440.606 us`: nine graph segments (`228.863 us`), 28 direct
  kernels (`166.335 us`), two 1-KiB D2D copies (`2.112 us`), and `44.224 us`
  of inter-segment/unattributed gap. Generated direct work comprises seven GEMM
  primitive launches (`97.280 us`) and 14 named generated kernels
  (`60.479 us`); residual XLA direct kernels add `8.576 us`.
- Final optimized HLO contains the same nine `copy` instructions in both paths.
  Stock XLA has 62 dots, 252 fusions, six cuBLAS calls, and no Shuttle calls;
  Shuttle has 49 dots, 239 fusions, three cuBLAS calls, and 13 Shuttle calls.
  The gap is therefore primarily generated direct physical work plus command
  graph fragmentation, not copy/layout conversion.
- The smallest next experiment is to make one adjacent generic
  Contract/Map/Fold cluster command-buffer-compatible as a single region. If
  that does not close the gap, attach the generated GEMM-heavy reverse Maps and
  Folds to generic Contract preparation/finalization. Do not target the D2D
  copies first.
- Raw Nsight reports remain in a mode-0700 local directory and are represented
  in the repository only by SHA-256 checksums. Sanitized timelines, attribution
  totals, exact commands, final optimized HLO, correctness evidence, and
  allocation-release evidence are in
  `lib/tile_lifetime/benchmarks/artifacts/xla_grug_13call_h100_profile_ac34883d_v0/`.
- The one-H100 allocation was released; both the local session and Kubernetes
  pod lookup verify it absent.

### 2026-08-09 15:48 PDT - TLTC-XLA-055 exact Contract/Map replacement boundaries

- Commit `db82fba39d` forms ten exact generic typed-FFI boundaries from the
  recovered low-rank Contract/Map structure in the frozen thirteen-call natural
  Grug HLO: six forward/rematerialization calls and four JAX-owned reverse
  calls. No model name or HLO metadata affects boundary formation.
- Every forward call has four BF16 inputs. Output arities are `5,1,1,5,5,5`:
  forward-only realizations return the final value, while reverse-bearing
  realizations also return the exact BF16 saved values already consumed by JAX
  reverse. Every reverse call has nine BF16 inputs and returns the input,
  down-weight, and up-weight adjoints.
- The replacement removes 28 live dots and `1,835,008` static dot FLOPs from
  the old HLO region. All replaced scalar arithmetic is absent or dead. The
  transformed module contains 23 calls: the existing thirteen plus the ten new
  structural placeholders.
- All ten placement all-reduces retain their original names and remain outside
  the generated boundaries. The audit traces each reverse cotangent input to
  its upstream collectives and verifies the generated outputs retain their
  exact external users. JAX still owns AD, save/rematerialization, and placement.
- Replacing the hidden scalar Map with tanh changes the semantic digest and
  generated scalar source without changing the boundary family, target, ABI,
  or Contract shapes.
- Verification:
  `uv run --frozen --package marin-tile-lifetime --group test pytest -q
  lib/tile_lifetime/tests/test_xla_low_rank_gated_product.py` passes all eight
  tests. Scoped pre-commit, including Pyrefly, and `git diff --check` pass.
- This is a structural typed-FFI checkpoint. No GPU body was generated or run,
  and it makes no latency or launch-count claim.

### 2026-08-09 - TLTC-XLA-056 generated gated-product Grug composition

- Added a selectable `shared_map_fused_reverses_and_gated_products` natural
  Grug composition. It applies the existing thirteen replacements, then derives
  and replaces six forward/rematerialization plus four JAX-owned reverse
  low-rank Contract/Map boundaries from the rewritten HLO.
- The ten logical boundaries normalize to one shape/AST physical family. Six
  calls reuse one generated forward target and four calls reuse one generated
  reverse target, so the final HLO has 23 calls but only two additional handler
  implementations. Target accounting now accepts and verifies exact
  multiplicities instead of requiring one target per call.
- The physical forward ABI is three rank-2 BF16 inputs and four rank-2 BF16
  outputs. The reverse ABI is seven rank-2 BF16 inputs and three outputs. XLA
  retains the redundant rank-3 views outside typed FFI, and generated BF16 save
  values connect each relevant forward realization directly to its JAX-owned
  reverse.
- Both recovered weight adjoints use `{0,1}` physical layout. The generic
  Contract program now records output minor-to-major layouts, generated CUDA
  writes each logical result at the corresponding physical offset, and the JAX
  component wrapper requests the same layouts.
- Static auditing removes all 28 old Contracts and `1,835,008` old dot FLOPs,
  finds no live old scalar arithmetic, preserves all ten placement all-reduces,
  and preserves every external logical output user. The generated source is
  source-ordered, atomic-free, and Torch-free. A tanh hidden-Map mutation reuses
  the same targets and physical ABI while changing both semantic and source
  digests.
- This is a CPU/static integration checkpoint. It does not claim successful
  whole-step GPU compilation, execution, correctness, or latency yet.

### 2026-08-09 16:10 PDT - TLTC-XLA-057 command-buffer replay bootstrap failure

- Revision `e5ec17f21c` marks only the generated normalized-exponential
  forward/reverse pair command-buffer compatible. Local 17-test and in-pod
  audits confirm both sources are free of scratch/runtime allocation, lazy
  handles, autotuning, status queries, and synchronization. JAX, JAXLIB, the
  CUDA plugin, and PJRT were all 0.11.0 on one physical H100.
- The only authorized benchmark invocation failed before warmup or timing. The
  fresh environment did not contain Triton, so the existing generated
  attention-backward path failed at `python -m triton.tools.compile`. No timed
  samples, correctness/determinism result, target-multiplicity audit, or
  capture-aware handler counts exist. The candidate remains unaccepted and
  unmeasured; it was not retried and no profiler was invoked.
- Before another allocation, require a CPU compile preflight of the complete
  composition: `triton.tools.compile` and the pinned Torch/Triton dependency
  set for the FA4-derived AOT path must import and execute a minimal compile in
  the exact environment. Checking only the newly eligible handlers was too
  narrow.
- The one-H100, one-CPU, 32-GB, 50-GB batch allocation was released after the
  failure artifact was copied. Iris reports the job killed by explicit
  release; the local session and Kubernetes pod are absent.
- Artifact:
  `lib/tile_lifetime/benchmarks/artifacts/xla_grug_command_buffer_h100_unaccepted_e5ec17f2_v0/`.

### 2026-08-09 - TLTC-XLA-058 H100 component replay blocked before execution

- Requested one bounded batch-priority H100 replay of the standalone generated
  two-Contract scalar-Map training component at source revision `bcafcc5ab1`.
  The request was one H100, one CPU, 32 GB host memory, and 50 GB disk. The
  benchmark protocol remained fixed at 30 counterbalanced samples with 1,000
  iterations per sample; no candidate tuning or replay retry was authorized.
- The canonical workspace was too large for the 25 MB Iris control-plane
  bundle, and its Iris client predated the cluster minimum. Neither pre-submit
  rejection created a job. A clean narrow holder using separately pinned Iris
  revision `eafa4d49f7` reached the controller, but its one admitted job failed
  during setup because the narrow bootstrap project did not define the `dev`
  dependency group expected by the task image.
- The benchmark process never started. Compile/link/load preflight, H100
  execution, natural-JAX and ordered-CPU comparisons, column-major weight-
  adjoint validation, determinism, handler counts, and timing distributions are
  all unmeasured. This is an infrastructure result, not a component failure.
- The controller reports the one matching job terminal failed and no active
  matching job. An explicit terminate was issued; the exact task-label pod was
  absent after cleanup, and no local holder session state remained.
- A future authorized run requires a clean detached measured-source checkout
  plus a current separate Iris holder whose sub-25-MB bootstrap project defines
  the expected dependency groups (or an equivalent small holder setup). The
  unmeasured artifact and release proof are under
  `lib/tile_lifetime/benchmarks/artifacts/generated_contract_map_chain_h100_unmeasured_bcafcc5a_v0/`.

### 2026-08-09 - TLTC-XLA-059 fail before allocation on incomplete build environments

- The combined natural-Grug benchmark now has a CPU-only dependency preflight.
  It checks the public compiler-IR roundtrip, compatible HLO text parser,
  `jax.extend.xla` transformation hooks, installed Torch/Triton distributions,
  and imports of `triton.tools.compile` and `triton.language` before device
  access.
- The build-time Torch/Triton audit is recorded separately from the generated
  runtime dependency audit. Accepted generated DSOs remain required to link no
  Torch or Triton runtime dependency.
- The local macOS environment correctly fails the new preflight because its JAX
  build lacks `jax.extend.xla`. A future H100 submission must pass the same CLI
  preflight in its exact task environment before allocation or execution.

### 2026-08-09 - TLTC-XLA-060 Contract/Map H100 reverse correctness failure

- Replayed the standalone generated two-Contract scalar-Map training component
  exactly once on one batch-priority H100 at source revision `d9e8990e87`.
  The measured harness, generator, wrapper, recovery files, and frozen HLO
  fixture are byte-identical to the preceding `bcafcc5ab1` checkpoint.
- The environment used JAX, JAXLIB, CUDA plugin, and PJRT 0.11.0 with NVCC
  13.3.73. Compile/link/load preflight passed. Fresh forward and reverse handler
  counts were both zero; the source audit found two generated kernels, no
  atomics, explicit BF16 Contract boundaries, and no opaque semantic dependency.
- GPU execution failed before timing. Generated `input_adjoint` differs from
  natural `jax.vjp` by maximum absolute error `0.47265625` and mean absolute
  error `0.1155403256`, exceeding the fixed `0.0078125` and `0.0005` limits.
  The process aborted before ordered-CPU input-adjoint parity, dW parity,
  determinism, final handler counts, or the 30-sample counterbalanced loop.
- The generated source contains the recovered `{0,1}` physical dW store layouts,
  but this run does not validate their output values because the earlier
  input-adjoint guard terminated the process. No component latency or whole-
  Grug claim is attached.
- No retry or tuning followed. The holder was explicitly terminated after
  copying evidence; the controller has no active matching job, the task-label
  pod is absent, and the local session state is absent. Raw logs, generated
  source, toolchain, invocation, source identity, preflight, and release proof
  are under
  `lib/tile_lifetime/benchmarks/artifacts/generated_contract_map_chain_h100_correctness_failure_d9e8990e_v0/`.

### 2026-08-09 - TLTC-XLA-061 command-buffer task environment remained too broad

- After the prior Contract/Map holder reported release, allocated one bounded
  batch-priority H100 holder for the command-buffer replay at source revision
  `2bfe584438`. The holder used one H100, one requested CPU, 32 GB memory, and
  50 GB disk. Its bootstrap project defined the required empty `dev` group and
  produced a 468-byte Iris bundle. The detached 41-MB source archive was
  verified in the pod at SHA-256
  `c39f7f878d19b8d1af60df8674dcc77effd045a42a2361de6c7eacf3b6f59cd1`.
- The attempt to construct the exact task environment selected the broad
  `marin-core[gpu]` dependency set. Resolution reached the unrelated Levanter
  serve/vLLM dependency and failed in vLLM's build backend because `CUDA_HOME`
  was unset. This happened before `--dependency-preflight-only`; JAX did not
  initialize a backend and no CUDA or benchmark command executed.
- The benchmark has zero invocations, warmups, timed samples, correctness or
  determinism checks, target audits, handler counts, and performance results.
  This is an environment-selection failure, not generated-program evidence.
- A new fixed-layout Contract/Map replay became the scheduling gate while the
  holder was active, so the holder was released immediately. Iris reports
  `JOB_STATE_KILLED`, exit 0, and no active matching job; the local state and
  exact task-label pod are absent.
- The next command-buffer attempt must start from canonical `239372d31d` or
  later and install the narrow natural-Grug runtime plus the pinned JAX 0.11 and
  Torch/Triton AOT stack without the serve/vLLM extra. It requires new explicit
  authorization after the fixed-layout component run releases. Artifact:
  `lib/tile_lifetime/benchmarks/artifacts/xla_grug_command_buffer_h100_unmeasured_2bfe5844_v0/`.
### 2026-08-09 - TLTC-XLA-062 fixed-layout Contract/Map H100 component

- Revision `239372d31d` converts recovered minor-to-major output layouts to the
  major-to-minor order expected by JAX typed FFI. Eighteen focused tests and
  changed-files pre-commit passed before allocation, and the transferred source
  archive contains the conversion for dX, dW0, and dW1.
- One fixed-protocol batch-priority H100 replay used JAX, JAXLIB, CUDA plugin,
  and PJRT 0.11.0 with NVCC 13.3.73. Compile/link/load preflight passed with
  fresh handler counts at zero, two generated kernels, no atomics, explicit
  BF16 Contract boundaries, and no opaque semantic dependency.
- All ordered-CPU outputs are bitwise equal. Natural-JAX maximum errors are
  `0.0009765625` for dX, `0.0001220703` for dW0, and `0.0001831055` for dW1;
  output is bitwise equal. Three repeated generated executions have identical
  hashes. Forward and reverse handler counts are both exactly `30,013`.
- Thirty counterbalanced samples with 1,000 iterations each measure
  `0.05889887 ms` for the generated two-call path and `0.03779946 ms` for
  matched natural JAX forward plus VJP, a `1.558193x` ratio. The standalone
  component is correct but misses the 1.20 performance target and remains
  unaccepted. It is a one-CTA scalar implementation, not a whole-Grug result.
- No tuning, retry, or profiler invocation followed. The holder was explicitly
  terminated; the controller has no active matching job, the task-label pod is
  absent, and the local session state is absent. Raw samples, generated source,
  StableHLO, optimized HLO, environment, invocation, and release proof are under
  `lib/tile_lifetime/benchmarks/artifacts/generated_contract_map_chain_h100_fixed_layout_239372d3_v0/`.

### 2026-08-09 - TLTC-XLA-063 natural-Grug Contract/Map ABI audit

- The `shared_map_fused_reverses` HLO contains 13 existing calls. Generic
  low-rank recovery adds six forward/rematerialization calls and four JAX-owned
  reverse calls, for 23 calls total. The ten new calls share one shape/AST
  family with target multiplicities 6 and 4 and remove 28 old Contracts.
- The CUDA generator now exposes one physical ABI used by direct-HLO emission,
  source auditing, and the JAX wrapper. Forward calls take three row-major
  rank-two BF16 buffers and return four row-major buffers. Reverse calls take
  seven row-major buffers and return row-major dX plus two recovered
  column-major dW buffers.
- Direct HLO retains XLA's minor-to-major spelling: dX is `{1,0}` and both dW
  outputs are `{0,1}`. The standalone JAX wrapper reverses these only when
  passing `output_layouts` to `jax.ffi.ffi_call`, yielding `(0,1)` for dX and
  `(1,0)` for each dW.
- Rank-three `[2,4,*]` logical outputs and cotangents are accepted only as
  contiguous `{2,1,0}` views of the rank-two CUDA buffers. The pre-allocation
  audit rejects mismatched operand constraints, dX layout, either dW layout,
  and noncontiguous logical views.
- Thirty-five focused CPU/static tests, including the capture-safe handler
  checks, and all 559 package tests pass. No GPU allocation or performance
  claim was made.
