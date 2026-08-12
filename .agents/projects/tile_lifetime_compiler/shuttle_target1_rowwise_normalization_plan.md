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

The proposed expert interface is NVIDIA Transformer Engine package 2.17.0's C
API from source tag `v2.17` at commit
`2e559f062497bef768dfbe9d7e45548fadeca80a`. NVIDIA names the package version
`2.17.0` in its pinned
[version file](https://github.com/NVIDIA/TransformerEngine/blob/2e559f062497bef768dfbe9d7e45548fadeca80a/build_tools/VERSION.txt);
the corresponding source tag is `v2.17`, not `v2.17.0`. The adapter calls
[`nvte_rmsnorm_fwd` and `nvte_rmsnorm_bwd`](https://github.com/NVIDIA/TransformerEngine/blob/2e559f062497bef768dfbe9d7e45548fadeca80a/transformer_engine/common/include/transformer_engine/normalization.h#L85-L141)
with `zero_centered_gamma=false`.

The proposed all-BF16 public boundary needs no dtype adapter. Backward requires
`dx.dtype == x.dtype`, `dy.dtype == gamma.dtype`, and
`dgamma.dtype == gamma.dtype`; `rsigma` is FP32. Transformer Engine registers
H=4096 BF16 forward and backward kernels with FP32 compute. These constraints
are enforced in the pinned
[RMSNorm implementation](https://github.com/NVIDIA/TransformerEngine/blob/2e559f062497bef768dfbe9d7e45548fadeca80a/transformer_engine/common/normalization/rmsnorm/rmsnorm_api.cpp#L113-L173),
and the BF16 kernel combination appears in the pinned
[forward](https://github.com/NVIDIA/TransformerEngine/blob/2e559f062497bef768dfbe9d7e45548fadeca80a/transformer_engine/common/normalization/rmsnorm/rmsnorm_fwd_cuda_kernel.cu#L150-L155)
and
[backward](https://github.com/NVIDIA/TransformerEngine/blob/2e559f062497bef768dfbe9d7e45548fadeca80a/transformer_engine/common/normalization/rmsnorm/rmsnorm_bwd_semi_cuda_kernel.cu#L170-L172)
registries.

The oracle adapter uses the same BF16 public tensors and `epsilon=1e-5`:

- forward writes public BF16 `y` and adapter-private FP32 `rsigma` of shape
  `[2048]`;
- backward calls forward to recompute `rsigma`, then calls
  `nvte_rmsnorm_bwd`. The public forward API also requires an output `z`, so
  this boundary computes and writes a full throwaway BF16 `z` inside timing;
- composed calls forward once, returns its `z` as public `y`, and reuses the
  same `rsigma` in backward.

`rsigma` is saved forward state, not workspace. Calling either API with an
empty workspace tensor performs a query without executing the operation. The
query returns the exact one-dimensional workspace shape and byte dtype required
by that backend and device. Adapter and library compilation, plan construction,
workspace queries, allocation, and warmup occur before timing. Raw workspace
storage may be shared between forward and backward when it is large enough, but
the `NVTETensor` metadata for each call must expose that call's exact queried
shape. The implementation checks shape equality rather than capacity. Kernel
launches, `z` and `rsigma` writes, and per-invocation adapters remain inside the
boundary. The pinned
[workspace implementation](https://github.com/NVIDIA/TransformerEngine/blob/2e559f062497bef768dfbe9d7e45548fadeca80a/transformer_engine/common/normalization/rmsnorm/rmsnorm_api.cpp#L23-L173)
defines this query protocol.

Forward and backward select the cuDNN normalization backend independently via
`NVTE_NORM_FWD_USE_CUDNN` and `NVTE_NORM_BWD_USE_CUDNN`, or
`nvte_enable_cudnn_norm_fwd` and `nvte_enable_cudnn_norm_bwd`. An artifact must
record both effective settings. The controls are defined in the pinned
[normalization backend source](https://github.com/NVIDIA/TransformerEngine/blob/2e559f062497bef768dfbe9d7e45548fadeca80a/transformer_engine/common/normalization/common.cpp#L546-L576).

The minimum oracle artifact identity is:

- Marin evaluation-harness revision and oracle-adapter digest, plus the JAX,
  JAXlib, CUDA plugin, PJRT, and XLA build identities used by the ordinary-JAX
  reference;
- Transformer Engine distribution and version, source tag and commit, recursive
  submodule commits, wheel or source-build identity, build flags, compiler and
  target architectures;
- resolved `libtransformer_engine.so` path, SHA-256, ELF build ID and SONAME,
  plus its resolved shared-library dependencies;
- CUDA toolkit, NVCC, CUDA driver and runtime, cuDNN compile-time and runtime
  versions, and the resolved cuDNN library identity;
- GPU model, UUID, compute capability, physical SM count, and the exact
  `multiprocessorCount` passed to each API;
- all tensor layouts, scaling modes, shapes, and dtypes; `epsilon`,
  `zero_centered_gamma`, stream policy, both backend controls, and the separately
  queried forward and backward workspace shapes, dtypes, and byte counts;
- warmup, synchronization and timing methods, including the complete launch and
  adapter sequence for each public boundary.

Transformer Engine requires CUDA 12.1 or newer and CUDA 12.8 or newer for
Blackwell, but artifacts record exact installed versions rather than these
minimums. NVIDIA documents these constraints in the pinned
[installation guide](https://github.com/NVIDIA/TransformerEngine/blob/2e559f062497bef768dfbe9d7e45548fadeca80a/docs/installation.rst#L10-L18).
Target 1 retains `oracle_not_pinned` until conforming artifacts exist on both
hardware classes and their numerical and performance gates pass.

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
