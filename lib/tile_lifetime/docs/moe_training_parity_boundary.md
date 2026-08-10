# Distributed MoE training parity boundary

The next distributed MoE result must not combine two already-valid but
different claims: Shuttle's four-rank primary-shape forward result and its
single-device differentiated Grug result. The former has the correct
communication and workload scale but no generated backward. The latter has
JAX-owned AD and generated local reverse arithmetic but uses a tiny physical
fixture and no expert-parallel transport.

## Matched semantic program

The natural source is an ordinary JAX program:

```text
router logits = x @ router weight
selected logits, selected experts = top-k(router logits)
route weights = normalized exponential(selected logits)
shared output = shared W13 -> pair Map -> shared W2
routed output = Relation(selected experts)
              -> SegmentedContract W13
              -> pair Map
              -> SegmentedContract W2
output = shared output + source-ordered weighted Fold(routed output)
```

JAX owns reverse-mode differentiation. The exported boundary deliberately
exposes the cotangent of the normalized route weights because the pinned MoK
backward returns the same semantic value. The remainder of the router VJP is a
generic JAX Map/Fold/Contract program shared by both measured paths; it is not
part of MoK's expert kernel.

The primary BF16 shape is fixed at 2,048 tokens per rank, hidden size 7,168,
intermediate size 3,072, 384 global experts, 96 experts per rank, top-6, and
four GB200 ranks.

## Derived Shuttle reverse

`derive_expert_parallel_training_plan` mechanically adds these generic stages
to the existing forward plan:

1. map the output cotangent and route weight to one edge cotangent;
2. transport edge cotangents to expert owners without reduction;
3. form segmented W2 input- and weight-adjoint Contracts;
4. generate the pair-Map VJP from the recovered forward scalar expression;
5. form segmented W13 input- and weight-adjoint Contracts;
6. Fold the routed output and output cotangent over the feature axis to obtain
   route-weight cotangents;
7. return input and route-weight cotangents as payloads;
8. Fold input cotangents in fixed source-slot order without atomics;
9. let JAX propagate route-weight cotangents through normalized selection and
   the router Contract;
10. add routed, shared, and router input cotangents with a generated Map.

Changing the pair Map changes its two generated scalar VJPs without changing
the stage family or transport plan.

## Allowed implementation boundary

The generated path may use generic grouped/ragged Contract mainloops and
generic payload transport. DeepEP dispatch is acceptable only as a payload
permutation. Reverse transport must likewise return payload and leave source
Folds visible to Shuttle. MoK forward and backward are oracle-only. DeepEP or
MoK semantic combine, route-weight reduction, pair activation, or complete
backward calls are excluded.

The full replay must compare:

- forward output;
- input cotangent;
- post-selection route-weight cotangent;
- shared and routed gate/up/down weight cotangents;
- router weight cotangent after the common JAX-owned router VJP.

It must preserve maximum and mean errors, deterministic hashes where the
source-order policy requires them, raw counterbalanced distributions, exact
source and toolchain revisions, all transport boundaries, and every candidate
considered.

## Current ownership and blockers

Shuttle now derives one four-rank backward ABI directly from `RelationPlan` and
executes its complete routed algebra in a deterministic CPU interpreter. The
same relation metadata drives output-cotangent dispatch, segmented W2 and W13
input/weight Contracts, generated pair-Map VJP, route-weight feature Fold,
inverse payload return, and the fixed source-slot input Fold. An uneven
four-rank fixture, including empty experts, matches the natural JAX VJP and is
bitwise stable across repeated interpreter executions.

The primary 4×GB200 shape has also passed a non-allocating buffer audit. It
covers all 49,152 route edges and derives rank-local BF16 activation and weight
gradient buffers plus FP32 route-cotangent buffers. Output weight gradients are
BF16, matching the natural JAX and MoK boundary, while their Contract
accumulators remain FP32 physical state.

The SM100 probe now contains generic CUDA loop skeletons for the remaining
non-Contract stages: route-weighted output-cotangent packing, pair-Map VJP,
route-weight feature Fold, and deterministic source Fold. Their scalar bodies
are generated from the erased pair expression and generic multiply/add Fold
algebra; changing the pair expression changes the generated program
fingerprint without editing the loop skeleton.

This is not yet a distributed GPU result. The CUDA extension still needs a
one-device build/correctness smoke, the transposed-weight layouts for W2 and
W13 input-adjoint Contracts need an explicit ABI check, and the BF16
weight-adjoint Contract family must be connected to the four-rank runtime.
External payload transport and JAX-owned router VJP then need a small
multi-process correctness gate before a primary-shape replay.

Therefore no four-rank GB200 replay is authorized by this checkpoint. A replay is
allowed only after a source audit proves that the generated executor contains
no semantic expert kernel, small multi-rank correctness covers every returned
cotangent, the exact pinned MoK backward passes independently, and build/runtime
preflight succeeds before allocation.

The first authorized one-GB200 compile/correctness smoke did not reach holder
submission. The corrected low-resource command failed because the local
workspace bundle exceeded the controller client's 25 MB limit. No device was
allocated or accessed, and the no-retry policy preserved this as a rejected
bootstrap artifact. The audit also records that the current SM100 probe adapter
is Torch-bound even though the scalar generator and CPU reference are
Torch-free. Both the bootstrap and physical-adapter boundary must be corrected
before the next GPU gate is proposed.
