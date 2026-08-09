# Grug relation-program execution plan

## Objective

Extend the proven XLA `PRE_SCHEDULER` typed-FFI replacement from one small
Contract+Map island to one complete routed compute region recovered from the
ordinary differentiated Grug program.

JAX continues to own autodiff. XLA continues to expose collectives and may
initially perform top-k selection. Shuttle must derive the replaced region from
generic `RelationPlan`, `SegmentedContract`, `Map`, and `Fold` structure. The
generated execution path may use generic dense/grouped contraction primitives,
but not a named MoE forward/backward kernel.

## Current evidence

The frozen post-SPMD HLO structurally recovers:

- two runtime relation plans;
- forward, recompute, and input-gradient segmented Contract chains;
- generated cast-aware scalar Maps for each chain;
- two source-keyed Fold merges;
- two routed weight-gradient Contracts; and
- explicit external all-reduces.

The first live replacement already proves the JAX/XLA integration mechanism on
GB200. It replaces one Contract plus one generated Map, is initially bitwise
exact across 53 leaves, and measures 1.021x the baseline one-layer step.

## Next bounded region

Start with the routed forward chain and its source-keyed Fold:

```text
physical routed input
  -> first Contract
  -> generated pairwise Map
  -> second Contract
  -> generated weighted contribution Map
  -> deterministic source-keyed Fold
```

Keep top-k, the existing physical relation permutation, and collectives outside
the first replacement. This isolates compute-body ownership before expanding
the boundary to Shuttle-owned relation construction.

The replacement boundary must be formed from HLO dataflow, not instruction
names or frontend metadata. The existing `SegmentedContractChainRecord` and
`FoldRecord` are the source of truth.

## Implementation sequence

1. Add a region record containing exact internal instructions, boundary inputs,
   boundary outputs, contract dimension/index maps, generated Map programs,
   Fold program, and numerical contract.
2. Verify the region is convex/topologically insertable. If the maximal region
   is non-convex, lower the largest compute prefix and expose required auxiliary
   outputs, as the existing Contract+Map proof does.
3. Add a generic two-Contract typed-FFI generator. The first correctness path
   may use cuBLAS Contracts plus generated scalar CUDA, but the interface must
   remain `Contract -> Map -> Contract -> Fold`.
4. Generate the contribution and reducer bodies from recovered scalar ASTs.
   Preserve the source-order BF16 conversions already present in the HLO.
5. Replace the region in the natural Grug module at `PRE_SCHEDULER`, compile,
   execute, and compare the complete result tree.
6. Add mutations for the pairwise Map and Fold contribution. The same generator
   must emit changed code without a workload switch.
7. Benchmark a counterbalanced full-step boundary. Record the handler count,
   raw samples, output hashes, and all remaining external routed work.
8. Only after this passes, expand the boundary upward to Shuttle-owned
   `RelationPlan` construction and downward to the input-gradient and weight-
   gradient regions. Leave all-reduces explicit until compute ownership is
   stable.

## Acceptance

The bounded forward region passes when:

- it is recovered from natural differentiated Grug HLO;
- named MoE semantics are absent from region formation and code generation;
- both Contracts, the intervening Map, and the source Fold execute inside the
  generated call;
- changing the Map or Fold AST changes generated code without editing CUDA;
- the complete train-step result obeys the declared numerical contract; and
- the matched whole-step latency remains within 1.20x baseline.

This is an incremental ownership proof. It does not yet claim that routing,
backward, weight gradients, or collectives are Shuttle-owned.

## Bounded region-formation result

The frozen forward chain forms one convex, topologically insertable physical
entry region. It contains six entry instructions spanning the first Contract,
the scalar and layout Map path, the second Contract, the weighted contribution
Map, and the source Fold. All seven boundary inputs are available before the
first region instruction. The only boundary output is the Fold result, so this
case does not require an auxiliary-output split.

The typed-FFI plan now recovers and verifies the intervening segmented layout
from the physical HLO rather than from an assumed expert layout. For compact
destination-major edge row `r`, logical feature `f`, and destination segment
`s`, the recovered relation is:

```text
physical_row = r
physical_k = f * segment_count + s
valid = exclusive_prefix_count[s] <= r < inclusive_prefix_count[s]
value = logical[r, f] if valid else 0
```

The segment dimension is therefore interleaved inside the physical Contract K
axis. This is not the superficially plausible segment-major layout
`s * feature_extent + f`. Recovery proves the interleaving from the Map-side
`broadcast -> select -> transpose -> copy -> bitcast` and separately verifies
that the weight-side `transpose -> copy -> bitcast` uses the identical K map.

The source Fold inverse is also explicit. The stable destination sort carries
the original flattened source-route position, so for destination-major row
`r`:

```text
route = stable_permutation[r]
source_item = route // route_slots
route_slot = route % route_slots
```

The verifier evaluates the actual HLO gather/index path for several legal
runtime permutations before accepting this relation. Destination prefix ends
and the stable permutation remain runtime inputs to eventual generated GPU
execution; benchmark callables must not close over routing fixtures in a form
that XLA could constant-fold.

All four required relations are now present, so region planning is `READY` for
generic typed-FFI code generation. The current checkpoint deliberately stops
before CUDA emission. A physical-row-capacity mutation reuses the same recovery
logic, while a mismatched weight flattening produces a structured rejection.
Map and Fold mutations retain the same convex boundary and Contract index maps
while changing only their generated scalar bodies.
