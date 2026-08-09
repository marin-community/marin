# Normalized-exp Contract training ownership

## Objective

Recover the output projection and its JAX-owned reverse from the natural Grug
post-SPMD HLO without recognizing cross entropy, Grug, or a model-specific
kernel.  Lower the region through generic Contract, Map, Fold, domain, and
indexed-relation structure, then generate one bounded streaming physical
family.

The source pattern is mathematically:

```text
scores[row, fold] = Contract(x[row, hidden], weight[hidden, fold])
scores = score_map(scores)
lse[row] = log(sum(exp(scores[row, fold])))
selected[row] = scores[row, selected_fold[row]]
output[row] = lse[row] - selected[row]
```

JAX supplies output cotangents.  The recovered reverse is:

```text
dscore = exp(scores - lse) * (doutput + dlse)
dscore -= indicator(fold == selected_fold) * doutput
dscore = score_map_vjp(dscore)
dx = Contract(dscore, transpose(weight))
dweight = Contract(transpose(x), dscore)
```

The compiler may tile the fold domain and merge partial normalized-exp state.
The operation name that produced the graph must not select the implementation.

## Semantic boundary

Introduce only the generic structure needed by this graph:

```python
@dataclass(frozen=True)
class IndexedRelation:
    source_axes: tuple[TensorAxis, ...]
    destination_axis: TensorAxis
    destination_indices: ProgramValue
    validity: ProgramValue | None

@dataclass(frozen=True)
class NormalizedExpContractTrainingProgram:
    score_contract: ContractPrimitive
    score_map: MapPrimitive
    fold_axis: TensorAxis
    selection: IndexedRelation
    output_map: MapPrimitive
    output_cotangent: ProgramValue
    state_cotangent: ProgramValue
    score_reverse_map: MapPrimitive
    input_reverse_contract: ContractPrimitive
    operand_reverse_contract: ContractPrimitive
    numerical_policy: NumericalPolicy
```

`NormalizedExpContractTrainingProgram` is a composition record, not a physical
primitive or dispatch key.  The physical selector must depend on its component
algebra, layouts, extents, and numerical policy.

The normalized-exp state remains generic:

```text
state = (maximum, sum_exp)
update(state, score_tile)
merge(left_state, right_state)
finalize(state) = log(sum_exp) + maximum
```

An indexed relation emits the selected score independently of the Fold.  It is
valid for any row-to-fold-coordinate selection, not only labels.

## Recovery from natural HLO

Start from the frozen natural Grug pre-scheduler fixture, then add a natural JAX
micro-fixture for mutation tests.  Recovery must prove:

1. one rank-two Contract produces the score domain;
2. maximum, shifted exponential, sum, and final log/add form a normalized-exp
   Fold over the same logical axis;
3. one gather/indexed selection reads the same score producer and is subtracted
   in the final Map;
4. the JAX reverse recomputes or consumes the same score relation and saved
   Fold state;
5. the reverse scalar Map uses only recovered score, Fold state, cotangents,
   selection relation, validity, and optional score-Map derivative;
6. its two reverse Contracts use the exact primal operands and recovered
   dscore;
7. padding and validity are DomainRestriction/index predicates rather than
   workload names;
8. all claimed internal values become dead after replacement, while placement
   collectives remain explicit.

Metadata and instruction names are diagnostics only.  Renaming them must leave
the semantic fingerprint unchanged.

## Physical candidate

Generate a resident-row, streamed-fold skeleton:

```text
resident x row tile
for each weight fold tile:
    score Contract
    generated score Map
    normalized-exp state update
    indexed selected-score update
finalize lse and output

reverse, for each weight fold tile:
    score Contract or legal saved score tile
    generated dscore Map
    input-gradient Contract and deterministic partial Fold
    weight-gradient Contract and direct tile store
```

The initial candidate set is deliberately bounded:

- fold tile chosen from the generic Contract skeleton's legal tile set;
- save Fold state, recompute score tiles in reverse;
- one deterministic input-gradient Fold order;
- separate forward and reverse kernel boundaries unless direct measurement
  justifies a generic combined boundary.

Reuse expert-quality generic matrix mainloops.  Do not call a fused loss kernel,
named attention kernel, or Levanter's workload implementation.

## Numerical contract

Record explicitly:

- input and weight dtype;
- Contract accumulator dtype;
- Contract output cast boundary;
- score-Map evaluation dtype;
- normalized-exp state dtype;
- deterministic partial-state merge order;
- dscore cast before reverse Contracts;
- input- and weight-gradient accumulation order.

The natural Grug path currently exposes BF16 Contract outputs with FP32
normalized-exp state.  A bitwise claim is not appropriate for a changed matrix
reduction tree; use the narrowest ordered-FP/reassociation policy supported by
the exported program and test against it.

## Mutation tests

The same recovery and generator must handle:

1. identity score Map versus finite tanh soft-cap;
2. a different row-to-fold indexed selection;
3. a smaller valid fold extent inside the same padded physical extent;
4. nonzero state cotangent as well as output cotangent;
5. fold-tile changes without semantic-source edits.

Changing any of these must change the semantic/source fingerprint as
appropriate without selecting handwritten source.

## Acceptance sequence

1. CPU reference program and structured-state merge equivalence.
2. Natural JAX StableHLO and post-SPMD HLO recovery tests.
3. Exact replacement/liveness audit on the frozen Grug fixture.
4. Direct H100 typed-FFI correctness and deterministic replay.
5. Matched component comparison against XLA's natural region.
6. Integrate into the complete Grug training replacement and replay the whole
   boundary.
7. Confirm the accepted generated path on a physical GB200; B200 remains only
   portability evidence.

The component is accepted only if its generated path is mutation-general,
Torch-free at runtime, contains no opaque semantic kernel, and is within
`1.20x` of the matched natural expert/XLA implementation for at least one
representative configuration.

