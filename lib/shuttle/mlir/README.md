# Native Shuttle MLIR scaffold

This directory is the first native compiler scaffold for Shuttle. It defines
the generic `shuttle` algebra dialect, structural source references, numerical
policy, pass declarations, verification passes, and `shuttle-opt`. It does not
yet implement structural StableHLO selection, StableHLO-to-Shuttle conversion,
algebra canonicalization, or lowering back to StableHLO. Those passes fail
closed when invoked, so this scaffold cannot be mistaken for the end-to-end
compiler proof.

The native code builds inside the dependency graph of the XLA revision pinned
by JAX/JAXlib 0.10.1:

```text
9b635916ecc6df6efee62d8e4b0c7ef87ef84d69
```

Build from a checkout of that exact XLA commit, with this directory registered
as a local external repository:

```starlark
# Add to the XLA checkout's WORKSPACE for a local developer build.
local_repository(
    name = "shuttle_mlir",
    path = "/absolute/path/to/marin/lib/shuttle/mlir",
)
```

Then run:

```bash
bazel build @shuttle_mlir//:shuttle-opt
bazel test @shuttle_mlir//:mlir_tests
```

Building from XLA rather than an installed `_jax.so` is intentional. MLIR's C++
ABI and the private LLVM/StableHLO revisions embedded in a jaxlib wheel are not
a supported extension interface. The root Marin `uv` workflow does not build
LLVM or these targets.

Current implemented behavior is deliberately narrow:

- `shuttle-annotate-source` attaches rename-invariant
  `#shuttle.source_ref<function, block, operation, result>` attributes.
- `shuttle-verify-source-coverage` checks exact declared-to-represented source
  coverage inside each extant `shuttle.region`, allowing one source result to
  feed multiple algebra operations. This is only a pre-lowering check.
- `shuttle-verify-semantic-erasure` rejects a source operation still carrying
  `shuttle.selected`.
- `shuttle-verify-no-shuttle-ops` rejects all remaining Shuttle operations and
  attributes before HLO export.

The export verifier keys operation rejection on the operation-name namespace,
so it also covers opaque `shuttle.*` operations in a context where the Shuttle
dialect is not registered. `shuttle-opt` always registers the dialect, and
pinned MLIR rejects unknown operations in a registered dialect during parsing;
that opaque-operation path therefore remains unexercised until a dedicated
native test harness can construct the alternate context.

The remaining declared passes emit an error and fail. No Python StableHLO
parser or textual HLO transformation participates in this native target.
Post-lowering provenance manifests and removal of `shuttle.source_refs` from
untouched source operations are not implemented. Consequently the current
coverage pass does not prove coverage after regions have been lowered.

`shuttle.map` and `shuttle.fold` admit only scalar, region-free operations with
proven no memory effects in their bodies. Fold inputs are positive-rank tensors
whose element types equal the scalar numeric accumulator types. Initializers,
combiner arguments and yields, and result elements use the same accumulator
types; an output cast must be a separate `shuttle.map`.

Map input indexing maps may project dimensions to express broadcast. Result
maps are full domain permutations: projection would imply duplicate writes,
for which Map deliberately has no semantics.

The first `shuttle.contract` surface is intentionally closed to two-input,
one-result `dot_general` with f32 operands, accumulator, and result. Its
indexing maps are symbol-free projected permutations of direct domain
dimensions with consistent static extents. Inputs and results are ranked
tensors, and precision and iterator values are closed. Wider or narrower
element-type contracts require a separate StableHLO-lowering proof before they
can enter this matrix.
