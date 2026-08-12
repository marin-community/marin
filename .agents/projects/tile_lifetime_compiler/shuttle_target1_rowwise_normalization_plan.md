# Shuttle Target 1 rowwise normalization plan

Status: design audit only. This note does not change the evaluation scorecard,
declare a representative shape, pin an accepted oracle artifact, or promote any
Target 1 cell.

## Decision

Use one anonymous rowwise reduction-and-scale program at `R=2048` rows and
`H=4096` features as the first Target 1 implementation slice. The source is an
ordinary JAX function with JAX-owned VJP. Fixture IDs and compiler inputs use
structural names such as `row_fold_scale_bf16_r2048_h4096`; `rmsnorm` appears
only in the evaluation target and oracle description.

The current scorecard remains accurate: Target 1 is blocked by the missing
ordinary-JAX GPU path, undeclared representative shape, and unpinned oracle
artifact. Historical `tile_lifetime` row-normalization results inform the
shape and physical decomposition, but cannot accept a Shuttle cell.

## Proposed evaluation contract

### Shape and source

The proposed contiguous row-major public tensors are:

| Value | Shape | Dtype | Role |
| --- | --- | --- | --- |
| `x` | `[2048, 4096]` | BF16 | primal rows |
| `gamma` | `[4096]` | BF16 | feature scale |
| `dy` | `[2048, 4096]` | BF16 | output cotangent |
| `y` | `[2048, 4096]` | BF16 | forward output |
| `dx` | `[2048, 4096]` | BF16 | input cotangent |
| `dgamma` | `[4096]` | BF16 | scale cotangent |

`epsilon` is the compile-time FP32 scalar `1e-5`. Test data uses JAX seed
`20260809`, split into independent standard-normal BF16 `x`, `gamma`, and `dy`
arrays. The source function is:

```python
def row_fold_scale(x, gamma):
    local = x.astype(jnp.float32)
    inverse = jax.lax.rsqrt(
        jnp.mean(local * local, axis=-1, keepdims=True) + 1e-5
    )
    return (local * inverse * gamma.astype(jnp.float32)).astype(jnp.bfloat16)
```

The three public boundaries are exact:

- `forward(x, gamma) -> y` calls `row_fold_scale`.
- `backward(x, gamma, dy) -> (dx, dgamma)` calls
  `jax.vjp(row_fold_scale, x, gamma)[1](dy)`. It is a recompute boundary.
- `composed_forward_backward(x, gamma, dy) -> (y, dx, dgamma)` calls one
  `jax.vjp`, returns its primal output, and applies the pullback.

The warmed ordinary-JAX baseline uses these functions without Shuttle compiler
options. No custom VJP or hand-written adjoint is part of the reference.

### Independent oracle

The proposed expert interface is NVIDIA Transformer Engine 2.17.0's C API:
`nvte_rmsnorm_fwd` and `nvte_rmsnorm_bwd`, with
`zero_centered_gamma=false`. The API accepts `[N,H]` input, `[H]` gamma, and
produces `[N,H]` output plus the `[N]` reciprocal RMS needed by backward. The
upstream interface is documented in
[normalization.h](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/c/normalization.html).

The oracle adapter uses the same BF16 public tensors and `epsilon=1e-5`:

- forward calls `nvte_rmsnorm_fwd`; reciprocal RMS is internal scratch;
- backward calls forward to recompute reciprocal RMS, then calls
  `nvte_rmsnorm_bwd`; the forward kernel is inside the timed boundary;
- composed calls forward once and reuses reciprocal RMS in backward.

Workspace queries, allocation, compilation, and warmup occur before timing.
Workspace reuse is allowed. Kernel launches, reciprocal-RMS writes, and all
per-invocation adapters remain inside the boundary. An oracle artifact must pin
the Transformer Engine source revision, shared-library SHA-256, CUDA/cuDNN
versions, workspace shape and dtype, and whether the cuDNN normalization backend
is enabled. Target 1 retains `oracle_not_pinned` until that artifact exists and
the BF16 output and gradient dtypes are verified on both hardware classes.

### The 12 required cells

Each row below is evaluated on both `h100` and `gb200_or_b200`, producing 12
scorecard cells.

| Boundary | Policy | Ordinary-JAX reference | Oracle | Shuttle path |
| --- | --- | --- | --- | --- |
| forward | `source_ordered` | `forward` | TE forward | same JAX source through `SOURCE_ORDERED` |
| forward | `fast` | `forward` | TE forward | same JAX source through `FAST` |
| backward | `source_ordered` | recompute `backward` | TE forward + backward | same JAX VJP through `SOURCE_ORDERED` |
| backward | `fast` | recompute `backward` | TE forward + backward | same JAX VJP through `FAST` |
| composed forward/backward | `source_ordered` | `composed_forward_backward` | TE forward + backward | same composed JAX source through `SOURCE_ORDERED` |
| composed forward/backward | `fast` | `composed_forward_backward` | TE forward + backward | same composed JAX source through `FAST` |

`SOURCE_ORDERED` preserves the BF16-to-FP32 conversions, FP32 accumulator and
epsilon, gamma placement, final FP32-to-BF16 rounding boundary, and every
reduction-order requirement expressed by StableHLO. Conversion derives the
Fold's order contract from source semantics; it does not grant blanket
reassociation freedom. `FAST` may select a deterministic tree reduction,
fuse the Fold with dependent Maps, and use approximate reciprocal square root
only if the pinned oracle artifact and numerical policy declare comparable
freedom. It may not move the final BF16 cast, call the oracle, or select a named
normalization kernel.

Before timing, the harness must declare per-output maximum and mean absolute,
relative, and BF16 ULP limits, plus repeatability limits. The limits are not
specified here because no accepted oracle artifact exists. They must follow the
scorecard rule `Shuttle error <= max(oracle error, declared dtype floor)` and
must be frozen before the first performance run.

## StableHLO inventory at the canonical audit revision

JAX and JAXlib 0.10.1 at Marin commit
`28a38e925aa5d57c94e66f8e772c3b5d40ce0c8a` emit the following top-level
operation counts for the exact source above. Every `reduce` has a scalar
`stablehlo.add` combiner in its region; those combiner operations are not
included in the top-level `add` count.

| Operation | Forward | Backward | Composed |
| --- | ---: | ---: | ---: |
| `convert` | 3 | 5 | 6 |
| `multiply` | 3 | 10 | 11 |
| `reduce` | 1 | 5 | 5 |
| `broadcast_in_dim` | 6 | 10 | 11 |
| `constant` | 3 | 9 | 9 |
| `divide` | 1 | 3 | 3 |
| `add` | 1 | 3 | 3 |
| `rsqrt` | 1 | 1 | 1 |
| `reshape` | 0 | 2 | 2 |

The forward graph is one FP32 row Fold followed by scalar Maps and a final BF16
cast. The reverse graph contains the recomputed row sum-square Fold, the row
correlation Fold used by `dx`, the feature-column Fold used by `dgamma`, and
two singleton-dimension cleanup Folds emitted by JAX's transpose rules.

## Gap against current Shuttle MLIR

`ShuttleOps.td` already defines generic `shuttle.map`, `shuttle.fold`, and
`shuttle.contract`. The current pass implementation does not convert this
program:

| Layer | Implemented at `28a38e925a` | Missing for Target 1 |
| --- | --- | --- |
| Structural selection | Ranked F32 `dot_general`, `tanh`, `add`, `multiply`, and `transpose` | BF16/F32 `convert`; region-bearing `reduce`; constants; broadcasts; divide; rsqrt; reshape |
| Algebra conversion | Identity/transpose Map indexing, scalar tanh/add/multiply, F32 Contract | StableHLO reduction to Fold; scalar constant capture; projected broadcast/reshape maps; scalar divide/rsqrt; BF16 cast boundaries |
| Algebra verification | Typed Map/Fold/Contract source references and Fold ordering field | Derivation of the exact reduction-order contract from StableHLO and per-operation provenance for reducer regions |
| Canonicalization | Empty pass | Policy-checked Fold/Map fusion and materialization choices |
| Source lowering | Map tanh/add/multiply/transpose and F32 Contract back to StableHLO | Fold, casts, broadcasts/reshape, constants, divide, rsqrt |
| Physical lowering | None; the current pipeline reconstructs StableHLO for XLA | task/buffer/lifetime plan, row and column Fold schedules, generated GPU code, and ordinary-JAX dispatch to that code |

The old `tile_lifetime.experimental_stablehlo_row_normalization_backward` path
has useful generic row/column Fold code and corrected H100 and GB200 timing
artifacts. It imports StableHLO into Python objects and dispatches typed FFI
outside the current Shuttle MLIR path. It is reference-only for this work.

## Smallest test-first sequence

The Reduce/Fold provenance, ordering, initializer, lowering, and test contract
for steps 2 and 3 is specified in
[`shuttle_target1_fold_conversion_design.md`](shuttle_target1_fold_conversion_design.md).

1. Add audited ordinary-JAX forward, backward, and composed StableHLO fixtures
   for `R=2048,H=4096`, plus one small shape mutation. Regeneration must pin
   JAX/JAXlib/XLA identities and fail on normalized-fingerprint drift. Pin the
   current failure first: structural-region formation rejects the
   region-bearing `stablehlo.reduce` before it constructs a coverage manifest.
2. Teach structural-region formation to traverse a `stablehlo.reduce` without
   losing its nested provenance. The Fold's `source` represents the reduce
   result. Each converted scalar combiner operation retains its own annotated
   source-result reference, and `shuttle.yield` retains the nested source
   terminator's operation reference. Extend coverage verification to audit
   these nested references before stripping them. A mutation that drops the
   combiner `add` or reducer terminator provenance must fail. Add
   lossless Map support for BF16/F32 converts, constants,
   `broadcast_in_dim`, reshape, divide, and rsqrt. Test the public outputs of
   small Map-only CPU fixtures and require the Target 1 reduction to remain an
   explicit exclusion; do not assert helper calls or prose.
3. Convert `stablehlo.reduce` with its scalar add region to `shuttle.fold`,
   including initializer, accumulator dtype, reduction dimensions, source
   reference, and source-permitted order freedom. Add source-ordered Fold
   lowering back to StableHLO. The forward fixture must then have total source
   coverage and exact CPU parity.
4. Extend the same conversion to JAX-owned backward and composed fixtures.
   Tests compare `y`, `dx`, and `dgamma` against ordinary JAX at the public
   boundary and require every source result to be selected or explicitly
   excluded. A nearby scalar or shape mutation must use the same passes.
5. Implement policy legality before optimization. `SOURCE_ORDERED` initially
   round-trips the lossless algebra. `FAST` may add only a generic Fold/Map
   fusion with an explicit legality predicate. Tests must reject a rewrite that
   moves the final BF16 cast or changes an undeclared reduction order.
6. Add a generic physical plan for one row Fold plus dependent Maps. Derive
   schedules from affine maps, reduction dimensions, dtypes, and policy. The
   plan and generated source must contain no target or workload name. Test a
   scalar-body mutation and a reduction-axis mutation through the same
   generator.
7. Add the reverse physical plan in three correctness-first stages: row
   sum-square and reciprocal scratch, row correlation plus `dx`, then the
   feature-column `dgamma` Fold. Only after parity passes should the planner
   consider same-domain Fold coalescing or emitting deterministic column
   partials from the final Map.
8. Connect these generated plans through the existing JAX/XLA registry adapter.
   The acceptance harness must observe ordinary JAX input StableHLO, Shuttle
   algebra coverage, transformed MLIR, generated source identity, handler
   execution, and dead source-region replacement. CPU and local MLIR tests
   precede any H100 or GB200 measurement.

No Target 1 status changes until the representative shape, oracle artifact,
numerical limits, ordinary-JAX generated path, and both hardware results are
recorded in a new scorecard revision.

## Evidence ledger

- Authoritative goal:
  `/Users/dlwh/.codex/attachments/c5f3fc0d-2981-4a61-b4d8-4cb6e4433531/goal-objective.md`.
- The conceptual Shuttle note supplied for this audit is absent from canonical
  commit `28a38e925a`; it informed this draft but is not a checked-in source.
- Current scorecard:
  `.agents/projects/tile_lifetime_compiler/shuttle_evaluation_manifest_v1.json`.
- Current dialect and passes: `lib/shuttle/mlir/include/shuttle/IR/ShuttleOps.td`
  and `lib/shuttle/mlir/lib/Transforms/Passes.cc`.
- Historical reference-only implementation:
  `lib/tile_lifetime/src/tile_lifetime/experimental_stablehlo_row_normalization_backward.py`
  and `lib/tile_lifetime/src/tile_lifetime/row_normalization_training.py`.
- Corrected historical measurements:
  `lib/tile_lifetime/benchmarks/artifacts/jax_row_normalization_backward_h100_components_corrected_v1/`
  and
  `lib/tile_lifetime/benchmarks/artifacts/jax_row_normalization_backward_gb200_components_corrected_v1/`.

Background-research effort: low. The stop rule was reached when the exact JAX
operation census, current MLIR gaps, historical physical decomposition, and an
independent expert oracle interface agreed on the same public boundaries. No
external performance result or historical prototype is used as acceptance
evidence.
