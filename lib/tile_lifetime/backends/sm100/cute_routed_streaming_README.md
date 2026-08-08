# SM100 relation-driven streaming skeleton

This backend is the physical target for Shuttle's generic
`Relation -> Contract -> Map -> Fold -> Contract` lowering on Blackwell. The
accepted path must compile `StreamingAttentionProgram`, `RelationPlan`, and an
explicit `SM100RoutedSchedule`; it must not accept an MSA, MoBA, NSA, or
attention operator identity.

## Oracle and extraction source

The initial physical design is derived from MiniMax Sparse Attention at commit
`80434d7f67877c6570ca19cac444b84bc9855dac`, with its CUTLASS submodule pinned
to `eb61c911471867a5fd2466bfd8f29306cea6ebf8`.

Relevant source hashes at that revision are:

| Source | SHA256 |
| --- | --- |
| `src/sm100/fwd/atten_fwd.py` | `69b615bcbeaacd1fd87446870c3dbd5e65300549590f5477701b1cd51dc65510` |
| `src/sm100/fwd/combine.py` | `9d1c3eecbc822f512ea148ed400230018b94813b6864c200daf30f122a00d752` |
| `src/common/softmax.py` | `d6756e56a74c638eaeae6ca840d34db2863f7d0836f84afbbce2b4a8015caaeb` |
| `src/common/mask.py` | `2fbef9d57a60398a65c11e5789225dc15a85d007633f0887c5ee8e3f11e85cb0` |
| `src/sm100/prepare_k2q_csr.py` | `429ab94135902eec524ec8a0db31857aa51b067c525330b2009412384601199f` |
| `src/sm100/prepare_scheduler.py` | `55e49e022a9439b21be5b33f9975413332375a6ce37836c6ad13e58d204470ac` |

The official executable remains an oracle-only dependency.

## Retained physical mechanisms

The first right-major template may retain or adapt these generic low-level
mechanisms:

- SM100 `tcgen05` QK and PV matrix operations;
- TMA Q/K/V movement and gather descriptors;
- packed-GQA layouts and index arithmetic;
- TMEM score/probability/output rings;
- bounded producer/consumer pipelines and barriers;
- shared-memory and register layouts;
- a right-resource scheduler over generic CSR relation metadata;
- vectorized stores and physical partial-state layouts.

These mechanisms are legal because their interfaces describe tensors,
relations, tiles, buffers, and readiness—not a named model.

## Shuttle-owned semantics

The accepted emitter must replace or generalize these parts rather than import
them as semantic helpers:

- `SoftmaxSm100`: generated normalized-exponential `Fold` update and finalize;
- `AttentionMask`: generated `DomainRestriction` predicate;
- score scaling and other score transforms: generated scalar `Map`;
- `SparseAttentionForwardCombine`: generic merge of
  `(row_max, row_sum_exp, weighted_value_accumulator)` partial states;
- q2k-to-k2q conversion and task derivation: generic `RelationPlan`
  orientation and bounded scheduling.

MSA's physical kernel computes one selected right block per partial and merges
partials later. Shuttle may select that decomposition, a left-major resident
state, or another legal bounded candidate. The compiler must benchmark the
finite candidate set instead of hard-coding the oracle decomposition as the
answer.

## Mutation gate

The same emitter must accept at least one semantic change without edits to the
physical CuTe skeleton. The first mutation should replace the causal predicate
with a generic window or prerecorded-validity predicate. A score-map or output
finalization mutation is a second useful check.

The generated-code audit fails if the accepted call graph reaches the official
MSA public interface or imports its `SoftmaxSm100`, `AttentionMask`, or semantic
combine implementation.
