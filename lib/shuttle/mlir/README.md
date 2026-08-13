# Native Shuttle MLIR slice

This directory implements the bounded native compiler-unit slice for generic
f32/BF16 Contract and scalar Map graphs plus one-input f32 add Reduce graphs.
`shuttle-opt` partitions typed StableHLO, converts selected regions to Shuttle
algebra, checks total source coverage, lowers from the authoritative algebra,
and removes all Shuttle semantics before StableHLO-to-HLO conversion. The
native compiler-option parser and pinned XLA/JAX registration overlays live
here. The sealed ABI 1
[`jaxacceptance6`](artifacts/native-preflight-20260810-jaxacceptance6/README.md)
artifact records the pinned CPU ordinary-JAX build, observer contract, and
persistent-cache populate/reuse protocol. GPU PJRT linkage, GPU execution, and
performance remain separate acceptance work.

The native code builds inside the dependency graph of the XLA revision pinned
by JAX/JAXlib 0.10.1:

```text
9b635916ecc6df6efee62d8e4b0c7ef87ef84d69
```

Build from a checkout of that exact XLA commit with the patches under
`xla_patch` applied and this directory registered as a local external
repository:

```starlark
# Add to the XLA checkout's WORKSPACE for a local developer build.
local_repository(
    name = "shuttle_mlir",
    path = "/absolute/path/to/marin/lib/shuttle/mlir",
)
```

Then run:

```bash
bazel build @shuttle_mlir//:shuttle_ops_inc_gen
bazel build @shuttle_mlir//:ShuttleDialect
bazel build @shuttle_mlir//:shuttle-opt
bazel build @shuttle_mlir//:ShuttleXlaRegistration
bazel build @shuttle_mlir//:ShuttleXlaRegistryAdapter
bazel build @shuttle_mlir//:mlir_tests
bazel test @shuttle_mlir//:mlir_tests
bazel test @shuttle_mlir//:pipeline_observer_test
bazel test @shuttle_mlir//:observer_test_bridge_test
bazel test @shuttle_mlir//:xla_registration_test
bazel test @shuttle_mlir//:xla_registry_adapter_test
```

The generated-operations target is the fast preflight for ODS and TableGen
compatibility with the pinned MLIR revision. The dialect library is the next
preflight: it compiles the generated operation declarations and definitions
without building the pass driver or linking `shuttle-opt`. Run both before
scheduling the larger cluster build.

Building `mlir_tests` is the analysis and executable preflight for the lit
suite. It preserves all fixture targets without running them; the following
`bazel test` command executes the fixtures.

Building from XLA rather than an installed `_jax.so` is intentional. MLIR's C++
ABI and the private LLVM/StableHLO revisions embedded in a jaxlib wheel are not
a supported extension interface. The root Marin `uv` workflow does not build
LLVM or these targets.

Current implemented behavior is deliberately narrow:

- `shuttle-annotate-source` attaches rename-invariant
  `#shuttle.source_ref<function, block, operation, result>` attributes.
- `shuttle-form-structural-regions` derives maximal contiguous supported-pure
  intervals and deterministic SSA weakly connected components.
- `shuttle-convert-stablehlo-to-algebra` produces generic `shuttle.contract`
  `shuttle.map`, and `shuttle.fold` operations. BF16/F32 conversions become typed
  `shuttle.scalar_convert` operations carrying exact or round-to-nearest-even
  provenance. No workload name or fixture selector is used.
- `shuttle-verify-source-coverage` checks exact selected and excluded result
  coverage, excluded-operation fingerprints and operand anchors, zero-result
  operations, function returns, policy identity, and region membership before
  and after lowering.
- `shuttle-verify-semantic-erasure` rejects a source operation still carrying
  `shuttle.selected`.
- `shuttle-lower-algebra-to-stablehlo` derives Map lowering from the scalar body
  and indexing maps, and Contract lowering from indexing maps, iterator kinds,
  precision, and algorithm. It does not replay source operation attributes.
- `shuttle-verify-no-shuttle-ops` rejects all remaining Shuttle operations and
  attributes before HLO export.
- The native observer registry emits immutable algebra-coverage,
  lowered-coverage, final-erasure, and terminal-failure records keyed by a
  process-unique invocation ID. Records include policy and tuning digests,
  pre-strip manifests and unsupported-island fingerprints, and the post-strip
  StableHLO fingerprint and erasure result. Coverage manifest version 2
  recursively accounts for nested source results and terminators.
- `subscribeShuttlePipelineObserver` returns a move-only scoped subscription.
  Its destructor removes the observer from future invocations and waits for
  invocations that captured it. An observer callback must not destroy or
  replace any subscription captured by its current invocation; release builds
  terminate with `kShuttleObserverReentrantTeardownDiagnostic` instead of
  waiting on the current invocation. Teardown from another thread waits for
  captured invocations to finish.
- Observer subscriptions are separate from `ShuttlePipelineOptions` and
  `shuttlePipelineIdentity`; installing an observer does not change the
  compiled module or semantic cache identity.
- `ShuttleXlaRegistration` parses the exact canonical Python compiler-options
  schema and invokes the shared production builder. Numerical policy, schema
  version, pipeline ABI version, and the complete tuning object are part of
  `canonicalOptions`; observer policy identity hashes that full cache key.
  Recursive Reduce/Fold conversion is pipeline ABI version 3.
- `ShuttleXlaRegistryAdapter` is an `alwayslink` translation unit that
  automatically registers the keyed `shuttle` callback in XLA's generic
  registry. The separate pinned JAX patch links it at final CPU `_jax`
  composition. The `jaxacceptance6` artifact proves that CPU composition path;
  dynamically loaded GPU PJRT plugins still require their own linkage and
  registration proof. Pipeline ABI 8 extends the opt-in
  `cpu_executable_bundle` mode to the exact `7x13` forward, JAX-owned backward,
  and composed Host boundaries under `SOURCE_ORDERED` and identity `FAST`.
  Both policies emit equal instruction bytes but retain distinct closed bundle
  roots and cache identities. The mode transports one verified stripped bundle
  through the fixed typed-FFI target `shuttle.cpu.executable_bundle.v2`; other
  shapes and accelerator execution remain unsupported.

The export verifier keys operation rejection on the operation-name namespace,
so it also covers opaque `shuttle.*` operations in a context where the Shuttle
dialect is not registered. `shuttle-opt` always registers the dialect, and
pinned MLIR rejects unknown operations in a registered dialect during parsing;
that opaque-operation path therefore remains unexercised until a dedicated
native test harness can construct the alternate context.

The six f32 fixtures and four anonymous BF16 primal+VJP fixtures under
`test/Inputs` are generated by ordinary JAX and record JAX, jaxlib, XLA
revision, shapes, dtypes, raw SHA-256, and parsed normalized SHA-256. The four
BF16 graphs are a compiler-unit corpus, not rebuilt-jaxlib or GPU acceptance
evidence. Regenerate and run the local CPU bitwise check with JAX/JAXlib
0.10.1:

```bash
uv run python lib/shuttle/mlir/test/Inputs/regenerate-jax-fixtures.py \
  --normalizer /path/to/shuttle-test-opt
uv run python lib/shuttle/mlir/test/Inputs/run-cpu-parity.py \
  --shuttle-opt /path/to/shuttle-opt
```

The first command is a check by default; pass `--write` only when intentionally
refreshing reviewed fixtures. No Python StableHLO parser or textual HLO
transformation participates in the compiler pipeline.

`shuttle.map` and `shuttle.fold` admit only scalar, region-free operations with
proven no memory effects in their bodies. Fold inputs are positive-rank tensors
whose element types equal the scalar numeric accumulator types. Initializers,
which remain rank-zero tensors at the Fold boundary, have element types equal
to the accumulator types. Combiner arguments and yields are scalars, and result
elements use the same accumulator types; an output cast must be a separate
`shuttle.map`.

`order_free = true` carries StableHLO Reduce's exact freedom: binary-tree
association and initializer multiplicity and placement are implementation
defined, while data leaves remain in ascending lexicographic order. It does
not allow leaf permutation and does not infer associativity or commutativity.
`order_free = false` requires one initializer followed by lexicographically
ordered accumulator updates, which StableHLO Reduce cannot represent; source
lowering rejects it.

Map input indexing maps may project dimensions to express broadcast. Result
maps are full domain permutations: projection would imply duplicate writes,
for which Map deliberately has no semantics.

`shuttle.yield` is an MLIR terminator, and the parent operations use
single-block implicit `shuttle.yield` terminators. Their region verifiers also
require the final operation to be `shuttle.yield`. It is not `ReturnLike`:
that MLIR trait also models the region-branch terminator interface, while the
current Shuttle parents do not model region-branch control flow.

The first `shuttle.contract` surface is intentionally closed to two-input,
one-result `dot_general` with matching BF16 or f32 operands and an f32
accumulator and result. BF16 output rounding is a following
`shuttle.scalar_convert`, never implicit Contract behavior. Its
indexing maps are symbol-free projected permutations of direct domain
dimensions with consistent static extents. Inputs and results are ranked
tensors, and precision and iterator values are closed. Other element-type or
accumulator contracts require a separate StableHLO-lowering proof before they
can enter this matrix.
