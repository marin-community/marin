# Independent clean-synthesis acceptance audit

Date: 2026-08-08

## Resolution in `research/shuttle-clean-helper-boundaries`

The implementation branch addresses the three physical-lineage failures found
below, but the new CUDA/CuTe sources still require device compilation and
benchmark replay before the affected workload rows can be accepted again.

- Dense full-region recovery now follows generic producer/consumer dataflow
  instead of requiring 36 operations in fixed tuple positions. A public test
  reverses the complete erased operation list and recovers the same plan.
- The H100 streaming skeleton no longer imports FlashAttention's `Softmax`,
  `AttentionMask`, or score-mod helper. Shuttle-owned generic normalized-exp
  Fold and `DomainRestriction` implementations retain the extracted physical
  TMA/WGMMA pipeline. Score scale and Fold-finalization mutations lower through
  the same physical skeleton.
- The MoE CUDA extension now contains generic adjacent-pair Map and ordered-Fold
  loop skeletons. Scalar arithmetic is emitted from `ScalarExpression` ASTs
  carried by the recovered `ExpertParallelPlan`. Build-time include validation
  and a runtime digest check reject plan/body drift, and mutation tests change
  emitted CUDA without editing the loop skeleton.

The MoE frontend-to-runtime link is improved but remains narrower than a full
generic tensor-program executor. The recovered plan supplies relation, top-k,
expert, and scalar-body semantics, while a small runtime adapter still launches
generic `torch.mm`, `torch.topk`, and `torch.softmax` primitives. This is not an
opaque workload kernel, but it should not be described as generated router code.

The four-GB200 replay at Shuttle revision `31f600f228` now passes. Two
counterbalanced 30-sample captures produce a pooled Shuttle/MoK ratio of
`1.137204×`; outputs are bitwise deterministic, maximum absolute error is
`0.0001220703125`, the compiled scalar-program digest matches the recovered
plan, and the accepted call graph contains no external semantic kernel. The
sealed evidence is under
`lib/tile_lifetime/benchmarks/artifacts/gb200_moe_clean_map_fold_v1`.

The low-priority H100 replay did not acquire capacity within its bounded
15-minute request. The pod remained `SchedulingGated`, consumed no GPU time,
and was deleted. No H100 device claim follows from the source-only tests.

The accepted status after these replays is:

- StatefulScan: accepted from the prior checkpoint.
- Dense: source-lineage fixes implemented; H100 device verification pending.
- MoE: clean Map/Fold boundary accepted on four GB200s, with the narrower
  generic-Torch router-executor caveat stated below.
- Sparse attention: source-lineage fix shared with dense; matched Hopper
  device verification remains pending.

This audit applies the boundary in
[`clean_synthesis_acceptance.md`](clean_synthesis_acceptance.md) to the current
accepted dense-attention and distributed-MoE paths. It distinguishes a useful
generated physical mechanism from first-principles discovery of when to use
that mechanism.

## Executive result

The current headline dense and MoE results do not yet satisfy the strict clean-
synthesis criterion.

- The dense GEMM consumer prologue is a real generated tile transform. It is
  not an opaque CODA call. However, the full dense planner selects that
  transform through a fixed 36-operation positional template, so the derivation
  is still a region-sized pattern match.
- The accepted dense attention path directly imports FlashAttention's
  `Softmax` and `AttentionMask` semantic helpers and instantiates a mostly
  extracted FlashAttention SM90 body. It therefore does not yet generate the
  normalized-exponential fold and domain predicate from Shuttle IR.
- The natural StableHLO MoE artifact is compiled only to validate shape and
  schedule metadata. Runtime routing is separately reimplemented with PyTorch
  `mm`, `topk`, and `softmax`.
- The MoE execution path calls handwritten CUDA SwiGLU and ordered-merge
  kernels. The generic `Map` and `Fold` plans exist, but they are not the source
  from which those CUDA bodies are emitted.

The component performance measurements remain useful. They demonstrate that
the generated GEMM prologue works, that the relation schedule is viable, and
that the selected physical decompositions are competitive. They should be
described as generated-mechanism or generated-schedule evidence until the
lineage gaps below are closed.

## Dense prologue: exact classification

### What is clean

`gemm_program.compile_gemm_program` lowers a `scale_row` preparation attachment
to a generic `TileOp` and inserts the BF16 conversion before the GEMM mainloop.
The relevant implementation is
`lib/tile_lifetime/src/tile_lifetime/gemm_program.py:78-128`.

`quack_gemm_codegen.generate_quack_gemm` then emits an A-fragment transform
from that tile program. The scale is rendered as `activation * column_operand`
in `lib/tile_lifetime/src/tile_lifetime/quack_gemm_codegen.py:107-150`.
The emitted body is derived from the tile program and uses QuACK as a generic
physical GEMM/operand-transform substrate. This part meets the intended
skeleton boundary.

### What is still a mega-pattern match

The accepted full-region planner first requires exactly 36 erased operations
and a hard-coded positional type sequence:

- `lib/tile_lifetime/src/tile_lifetime/dense_flow_planner.py:93-138` checks the
  exact operation count and positional types.
- `lib/tile_lifetime/src/tile_lifetime/dense_flow_planner.py:139-176` assigns
  fixed semantic roles such as `first_square`, `first_fold`, `gate_contract`,
  and `second_row_finalize` by tuple position.
- `lib/tile_lifetime/src/tile_lifetime/dense_flow_planner.py:311-407` directly
  constructs the eight expected skeletons and calls `_row_scale_preparation`
  at the two preselected consumer sites.
- `lib/tile_lifetime/src/tile_lifetime/dense_flow_planner.py:621-653` turns the
  requested placement enum into a prologue or epilogue attachment. It does not
  discover arbitrary row-scalar-to-contract opportunities through a reusable
  legality pass.

Thus the honest statement is:

> Shuttle has a generic, generated source-ordered GEMM prologue mechanism and
> has measured it successfully, but the current complete dense path reaches it
> through a whole-region template rather than a general Map/Fold/Contract
> rewrite search.

The present semantic-erasure validator does not detect this distinction. It
checks generic names and scheduling-key strings
(`semantic_erasure.py:145-195`), but it does not reject a fixed positional
template or audit physical source lineage.

## Attention lineage finding

The accepted H100 emitter lowers a `StreamingAttentionProgram`, but its
physical body still owns the attention semantics through FlashAttention code:

- `lib/tile_lifetime/backends/h100/cute_streaming_base.py:34-35` imports
  `AttentionMask`, `Softmax`, and `apply_score_mod_inner` from FlashAttention.
- `lib/tile_lifetime/backends/h100/cute_streaming_base.py:913-918` constructs
  the FlashAttention `Softmax` state object.
- `lib/tile_lifetime/backends/h100/cute_streaming_base.py:1003-1021` constructs
  `AttentionMask` and binds its attention-specific masking API.
- `lib/tile_lifetime/backends/h100/cute_streaming_sm90.py:31-32` repeats the
  direct semantic imports.
- `lib/tile_lifetime/backends/h100/cute_streaming_sm90.py:571-578` specializes
  the FlashAttention mask class.
- `lib/tile_lifetime/backends/h100/cute_streaming_sm90.py:1372-1410` calls
  `softmax.online_softmax`, `softmax.rescale_O`, and the PV mainloop from the
  extracted fixed body.
- `lib/tile_lifetime/backends/h100/cute_streaming_emitter.py:99-124` instantiates
  that class as the executable selected for a generic program.

`h100_streaming_lowering.py:74-124` does recover scale, causality, and optional
softcap from scalar dataflow. That is useful frontend legalization, but it
ultimately reduces causality to `is_causal=True` and delegates the fold and
predicate implementation to named FlashAttention helpers. A small change to
the reduction state or domain predicate cannot currently flow through a
generic fold/predicate emitter.

Allowed extraction should stop below that boundary: WGMMA/TMA operations,
layouts, barriers, pipeline state, and neutral tile copies are reusable
physical skeleton machinery. The max/sum-exp/output recurrence and the domain
predicate must be emitted from Shuttle's `Fold` and `DomainRestriction`.

## MoE lineage finding

### The natural StableHLO plan is validation-only at runtime

`_compile_natural_stablehlo_plan` states that it validates the benchmark shape
and returns only a diagnostic dictionary:

- `lib/tile_lifetime/benchmarks/backends/gb200_deepep_mok_distributed.py:213-243`.

The runtime router is a separate handwritten PyTorch implementation:

- `NaturalRouterRuntime` executes `torch.mm`, `torch.topk`, and
  `torch.softmax` at
  `lib/tile_lifetime/benchmarks/backends/gb200_deepep_mok_distributed.py:176-210`.
- `main` creates that router independently after compiling the StableHLO plan
  and feeds its tensors to the physical runtime at
  `lib/tile_lifetime/benchmarks/backends/gb200_deepep_mok_distributed.py:995-1049`.

The recovered StableHLO semantics therefore validate that the separately coded
router has a compatible shape; they do not instantiate the runtime router.
The existing public StableHLO tests are structural only
(`tests/test_stablehlo_moe.py:69-134`). They do not execute values through the
recovered program.

### The Map and Fold bodies are handwritten

The physical runtime calls named extension functions directly:

- routed/shared SwiGLU calls are at
  `benchmarks/backends/gb200_deepep_mok_distributed.py:493-530`;
- route merge and rank/shared merge calls are at the same file's lines
  `504-509` and `540-553`.

Their CUDA arithmetic is written by hand:

- `backends/sm100/mok_gmm_probe/mok_gmm_probe.cu:160-243` contains fixed SiLU
  sigmoid arithmetic and two named SwiGLU launchers;
- `backends/sm100/mok_gmm_probe/mok_gmm_probe.cu:450-560` contains the fixed
  route-slot weighted reduction;
- `backends/sm100/mok_gmm_probe/mok_gmm_probe.cu:653-725` contains the fixed
  rank-order plus shared-output reduction.

`compile_ordered_relation_fold` does build a generic tile program
(`src/tile_lifetime/relation.py:40-106`), and plan tests inspect its primitive
sequence (`tests/test_expert_parallel.py:122-135`). No source generator consumes
that tile program to produce the called merge kernel. The plan and executable
body can therefore drift independently.

DeepEP dispatch remains an allowed payload-transport primitive. The standalone
grouped GEMM remains an allowed segmented-contraction primitive. The findings
above concern router, map, and semantic merge lineage, not those two physical
primitives.

## Required machine-checkable gates

The following checks should be public tests, not benchmark-report assertions.
They are intentionally phrased so the current paths fail and generic helpers
can pass.

### 1. Physical dependency manifest and call-graph audit

Every accepted backend artifact should carry a machine-readable dependency
manifest with entries:

```text
symbol
origin file and revision
category = physical_primitive | generated_semantic_body | oracle_only
generic IR digest, when category = generated_semantic_body
```

The accepted executable's transitive call graph may contain only
`physical_primitive` and `generated_semantic_body`. Its generated semantic-body
digest must equal the digest of the selected Shuttle IR.

Portable AST/source assertions should additionally reject, in the accepted
attention closure:

```text
from flash_attn... import Softmax
from flash_attn... import AttentionMask
Softmax.create(...)
AttentionMask(...)
```

They should reject, in the accepted MoE closure, direct calls to:

```text
swiglu_bf16_out
swiglu_row_halves_bf16_out
fixed_route_merge_out
fixed_rank_merge_shared_out
```

unless those symbols are generated artifacts whose manifest digest matches the
corresponding generic `Map` or `Fold` program. Renaming or copying a fixed body
must not satisfy the manifest.

Suggested tests:

```text
test_accepted_attention_call_graph_contains_no_oracle_semantic_helper
test_accepted_moe_call_graph_contains_no_handwritten_map_or_fold_body
test_generated_semantic_body_digest_matches_selected_ir
```

### 2. Dense rewrite perturbation tests

The row-scale rewrite must be found from dataflow and properties, not tuple
position. Starting from natural math, each of these must compile to the same
generic prologue mechanism:

- topologically reorder independent operations;
- insert an identity/view on the RMS row scalar or contract input;
- place an unrelated Map elsewhere in the region;
- use the same row-scalar structure before a standalone Contract outside the
  canonical two-block template.

The compiled prologue must disappear when the scalar varies over the
contraction-reduction axis or has an intervening observable consumer.

Suggested tests:

```text
test_row_scalar_contract_preparation_is_independent_of_region_position
test_row_scalar_contract_preparation_survives_identity_and_topological_reorder
test_row_scalar_contract_preparation_rejects_reduction_axis_varying_scale
test_row_scalar_contract_preparation_rejects_observable_normalized_value
```

The first two fail the current exact-36-operation planner.

### 3. Attention semantic mutation tests

Compile at least two closely related natural programs through one skeleton:

- causal `key.position <= query.position` and a noncausal or sliding-window
  domain predicate;
- ordinary normalized exponential and a small supported change to its fold
  state/update/finalize algebra;
- a score Map mutation such as diagonal bias or softcap.

The generated semantic source digest must change when the predicate or fold AST
changes, while the physical QK/PV/pipeline skeleton identity stays fixed. No
backend `if attention-kind` dispatch is allowed.

Suggested tests:

```text
test_domain_restriction_mutation_changes_generated_predicate_only
test_normalized_fold_mutation_changes_generated_fold_body_only
test_attention_semantic_mutations_reuse_one_physical_skeleton
```

An import-poison test may supplement, but not replace, the lineage manifest:
make FlashAttention's `Softmax` and `AttentionMask` unavailable and prove the
accepted emitter still compiles. The current emitter fails this test.

### 4. Natural StableHLO MoE runtime-lineage test

The public compiler result should expose an executable or compiled subprogram
for the recovered router `Contract`, selection, and normalized-weight `Fold`.
The benchmark must call that executable, not a separately authored router.

Use a semantic mutation whose output is visible at runtime, for example a
different normalization expression over selected logits. Compile both frozen
StableHLO fixtures, run identical tensors, and assert:

- the recovered runtime outputs match direct JAX outputs;
- the outputs differ between the two semantic programs as expected;
- the same generic selection/Fold code generator and RelationPlan builder are
  used;
- deleting or poisoning `NaturalRouterRuntime` does not affect the accepted
  path.

Suggested tests:

```text
test_natural_stablehlo_router_values_feed_runtime_relation_plan
test_router_normalization_mutation_changes_runtime_without_backend_change
test_accepted_moe_path_does_not_instantiate_benchmark_router_reimplementation
```

### 5. MoE Map and merge mutation tests

Generate the activation and merge bodies from generic scalar/tile ASTs. Then
test:

- `silu(gate) * up` versus a nearby pointwise expression through the same Map
  generator;
- two and six route slots through one Fold generator;
- varying partition/rank count through one ordered-Fold generator;
- base-plus-partials and partials-only folds;
- deterministic ascending edge/partition order with no semantic atomics.

For each mutation, assert that the generated source digest follows the IR
digest and that no named workload kernel is selected.

Suggested tests:

```text
test_segmented_map_expression_mutation_changes_generated_cuda
test_ordered_relation_fold_generates_two_and_six_slot_bodies
test_ordered_partition_fold_generates_variable_partition_count
test_generated_relation_fold_is_deterministic_and_atomic_free
```

The current plan-only two/six-slot assertions do not meet this requirement.

## Minimal implementation corrections

1. Retain the current QuACK prologue emitter. Replace the 36-position RMS
   placement logic with a connected-subgraph rewrite over generic
   Map/Fold/Contract values and legality properties.
2. Extract only the neutral SM90 attention pipeline/mainloop machinery. Add a
   generated predicate callback from `DomainRestriction` and a generated
   online-state callback from `Fold`; remove `Softmax` and `AttentionMask` from
   the accepted dependency closure.
3. Make recovered MoE router algebra executable and feed those exact runtime
   tensors into RelationPlan/DeepEP.
4. Add one scalar-Map CUDA emitter for segmented/dense tensors and use it for
   SwiGLU-like expressions.
5. Add one ordered-Fold CUDA emitter and use it for both route-slot and
   returned-partition merges. Keep DeepEP as transport and grouped GEMM as the
   segmented contraction.
6. Extend the semantic-erasure boundary with a physical-lineage manifest. Name
   erasure alone cannot distinguish generated semantics from a renamed or
   copied expert semantic body.

## Acceptance accounting after this audit

| Evidence | Strict status | What remains valid |
|---|---|---|
| Dense GEMM source-ordered prologue component | mechanism passes; discovery fails | generated A-fragment transform and measured cost/performance |
| Dense whole-region headline | fails | region decomposition and layout experiment; GEMM mechanisms |
| Dense/routed attention | fails | generic semantic IR and useful SM90 physical oracle/extraction |
| Distributed MoE | fails | RelationPlan/schedule experiment, DeepEP transport, grouped-GEMM decomposition |
| MoK and FlashAttention numbers | oracle only | performance targets and differential correctness references |

The checkpoint should not be discarded. The next work is narrower than
rebuilding the kernels from scalar CUDA: preserve the physical skeletons, but
replace semantic helper imports and handwritten semantic bodies with callbacks
or source generated from generic Shuttle programs.
