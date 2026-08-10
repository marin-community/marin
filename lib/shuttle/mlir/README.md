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
  coverage, allowing one source result to feed multiple algebra operations.
- `shuttle-verify-semantic-erasure` rejects a source operation still carrying
  `shuttle.selected`.
- `shuttle-verify-no-shuttle-ops` rejects all remaining Shuttle operations and
  attributes before HLO export.

The remaining declared passes emit an error and fail. No Python StableHLO
parser or textual HLO transformation participates in this native target.

`shuttle.fold` carries initializer operands explicitly. Its verifier checks the
combiner's source arguments against input element types and its initializer,
accumulator arguments, yields, and result elements against declared
accumulator types. An output cast must be a separate `shuttle.map`.
`shuttle.contract` keeps accumulator types distinct from its explicit result
element types as part of the selected contraction algorithm.
