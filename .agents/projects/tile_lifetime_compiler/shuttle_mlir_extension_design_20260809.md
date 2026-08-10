# Shuttle MLIR extension design

## Decision

Shuttle is a native MLIR compiler extension in the ordinary JAX/XLA
compilation path. JAX owns tracing and automatic differentiation. Shuttle sees
the resulting StableHLO module while it is still typed MLIR, selects structural
regions, converts them to generic Shuttle algebra, applies numerical-policy and
schedule rewrites, and lowers every Shuttle operation before XLA converts the
module to HLO.

Python StableHLO parsers, textual HLO rewriters, whole-pattern recognizers, and
the JAX 0.11 serialized-HLO callback remain experimental bridges. They are not
production frontends and cannot establish architectural acceptance.

## XLA integration seam

The pinned JAX 0.10.1 XLA revision is
`9b635916ecc6df6efee62d8e4b0c7ef87ef84d69`. Both ordinary CPU JIT and the GPU
StreamExecutor path call `xla::MlirToXlaComputation`. The Shuttle hook belongs
in `xla/pjrt/mlir_to_hlo.cc`, after Shardy export and CHLO/StableHLO cleanup and
immediately before `ConvertStablehloToHloWithOptions`.

The hook is transactional:

1. clone the live MLIR module;
2. run registered StableHLO module transforms on the clone;
3. verify selected-source coverage, boundary completeness, and semantic
   erasure;
4. verify no Shuttle dialect operation or attribute remains; and
5. replace the original module only after all checks succeed.

An enabled transform failure aborts compilation. It never silently falls back.
XLA should own a small generic StableHLO transform registry rather than call
Shuttle by name. A Shuttle-enabled jaxlib links a registration translation
unit.

## Public options and cache identity

The public Python API is:

```python
jax.jit(
    function,
    compiler_options=shuttle.compiler_options(
        shuttle.Options(
            numerics=shuttle.Numerics.SOURCE_ORDERED,
            tuning=shuttle.Tuning(...),
        )
    ),
)
```

Current JAX already forwards top-level compiler options into
`CompileOptions.env_option_overrides` and includes them in compilation-cache
identity. The Shuttle-enabled XLA build adds two recognized fields:

- `xla_shuttle_enable`; and
- `xla_shuttle_options`, a canonical closed-schema JSON value.

Unknown fields and workload labels are rejected. `SOURCE_ORDERED` and `FAST`
always have distinct option and cache identities, even while they select the
same physical implementation.

## First dialect slice

The first native slice contains only the operations required to prove a generic
two-Contract scalar-Map forward and its JAX-generated VJP:

- `shuttle.region`, with explicit SSA boundary and numerical policy;
- `shuttle.contract`, with normalized indexing maps, iterator kinds,
  accumulator/result types, precision, and algorithm attributes;
- `shuttle.map`, with an explicit scalar body, casts, broadcasts, and index
  maps;
- `shuttle.fold`, with reduction dimensions, initializer, scalar combiner,
  accumulator type, and source order-freedom;
- `shuttle.yield`;
- `#shuttle.policy<source_ordered|fast>`; and
- structural source references based on function, block, operation, and result
  ordinals.

Scan, Relation, DomainRestriction, Transport, task, event, and physical
dialects are added only when their verifiers and lowerings are ready.

## Pass pipeline

```text
shuttle-annotate-source
shuttle-form-structural-regions
shuttle-convert-stablehlo-to-algebra
shuttle-verify-source-coverage
shuttle-verify-semantic-erasure
shuttle-canonicalize
shuttle-lower-algebra-to-stablehlo
shuttle-verify-source-coverage
shuttle-verify-no-shuttle-ops
```

Maximal convex connected components of supported pure operations are candidate
regions. Unsupported, effectful, alias-sensitive, collective, or control-flow
operations are hard boundaries. Selection depends on operation structure,
types, indexing maps, effects, and shapes, never module or function names.

Selected supported StableHLO operations become dynamically illegal during
partial conversion. Every selected source operation must map to one or more
provenance-bearing Shuttle operations. Unknown operations outside selected
regions remain untouched.

`SOURCE_ORDERED` preserves source-visible casts, accumulator types, and any
prescribed order. It does not promise ordering stronger than StableHLO.
`FAST` may enable reassociation only when a typed numerical contract permits
it.

## Source and build layout

The Python package stays under `lib/shuttle/src/shuttle`. Native code lives
under `lib/shuttle/mlir` and builds with Bazel against the XLA/LLVM/StableHLO
revision pinned by the target jaxlib. Routine `uv sync` must not build LLVM.

The native build produces:

- `libShuttleDialect`;
- `libShuttlePasses`;
- `libShuttleXlaRegistration`; and
- `shuttle-opt`.

`shuttle-opt` and the Shuttle-enabled jaxlib link the same pass library. There
is no Python reimplementation of the compiler pipeline. A separately linked
extension against an arbitrary installed `_jax.so` is not supported because
the MLIR C++ ABI and headers are not a wheel-level interface.

## Proof sequence

The first standalone proof exports ordinary JAX forward and JAX-owned VJP
StableHLO for `tanh(x @ w0) @ w1`, runs the native pass library through
`shuttle-opt`, and preserves the algebra-stage Shuttle MLIR and fully lowered
StableHLO. It proves exact source coverage, cast and dot-precision
preservation, rename-invariant semantic fingerprints, hard unsupported-op
boundaries, distinct numerical-policy digests, and complete Shuttle erasure.

This offline proof is real compiler evidence but not ordinary-JAX integration.
The architectural proof requires a Shuttle-enabled jaxlib and the same pass
library running through `jax.jit(..., compiler_options=...)` for both forward
and JAX-generated backward. GPU scheduling, NVGPU/NVVM lowering, collective
transport, and performance acceptance follow only after the CPU spine is
green.

## Prototype retirement rule

`shuttle` current code must not import `shuttle.experimental` or
`tile_lifetime`. `tile_lifetime` may depend one-way on `shuttle` while legacy
code is retired. A Python prototype is deleted once native MLIR conversion and
verifier tests cover its useful semantics. Preserved benchmark artifacts remain
immutable historical evidence.
