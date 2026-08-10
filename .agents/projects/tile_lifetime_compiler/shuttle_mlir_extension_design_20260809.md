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
        numerics=shuttle.Numerics.SOURCE_ORDERED,
        tuning=shuttle.Tuning(...),
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
shuttle-verify-source-coverage{stage=algebra}
shuttle-verify-semantic-erasure
shuttle-canonicalize
shuttle-lower-algebra-to-stablehlo
shuttle-verify-source-coverage{stage=lowered}
shuttle-strip-source-provenance
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

## First ordinary-JAX vertical slice checkpoint

The bounded proof fixture is an f32 two-Contract scalar-Map program:

```python
def reference_function(x, w0, w1):
    return jnp.tanh(x @ w0) @ w1
```

The fixed shapes are:

| Value | Shape |
| --- | --- |
| `x` | `[2, 3]` |
| `w0` | `[3, 4]` |
| `w1` | `[4, 5]` |
| output and output cotangent | `[2, 5]` |
| `dx` | `[2, 3]` |
| `dw0` | `[3, 4]` |
| `dw1` | `[4, 5]` |

These dimensions bound the first compiler proof. They are test data, not
selection criteria or a semantic specialization. Region formation may inspect
ranked types, static extents, indexing maps, and effects. It must not inspect
module names, function names, parameter names, or a workload label.

JAX owns the pullback:

```python
def reference_vjp(x, w0, w1, output_cotangent):
    _, pullback = jax.vjp(reference_function, x, w0, w1)
    return pullback(output_cotangent)
```

The input StableHLO fixtures are exported from ordinary JAX 0.10.1 at the
pinned shapes. They are checked in verbatim with the JAX version and export
command. A hand-authored equivalent is insufficient for this gate.

### Selected regions

The forward module has one selected contiguous region:

```text
dot_general(x, w0) -> tanh -> dot_general(_, w1)
```

Both `dot_general` operations convert to `shuttle.contract`. The `tanh`
converts to `shuttle.map`. The region returns the `[2, 5]` result.

The pinned JAX VJP contains this operation order:

```text
dot_general(x, w0)
tanh
constant(1.0)
broadcast_in_dim
subtract
dot_general(output_cotangent, tanh_result)
transpose
dot_general(output_cotangent, w1)
multiply
multiply
add
dot_general(_, x)
transpose
dot_general(_, w0)
```

The first slice deliberately does not convert `constant`,
`broadcast_in_dim`, or `subtract`. They form a hard boundary and remain
StableHLO. The supported operations form three contiguous regions in program
order:

1. the primal `dot_general -> tanh` prefix, exporting the activation;
2. `dot_general -> transpose`, producing `dw1`; and
3. `dot_general -> multiply -> multiply -> add`, followed by the two
   `dot_general` branches and the `transpose` that produce `dx` and `dw0`.

The third region accepts the untouched subtraction result as an external
operand. Region boundaries expose all live-ins and live-outs as explicit SSA
values. A selected region may have several results and external users. Region
formation fails if an effect, alias obligation, control edge, or unmodeled
external user prevents a complete boundary.

`dot_general` converts to `shuttle.contract`. `tanh`, `multiply`, `add`, and
`transpose` convert to `shuttle.map` with explicit scalar bodies and indexing
maps. Static `reshape` and `convert` are part of the same structural Map
conversion surface when a JAX version introduces them, but the frozen f32
fixtures do not depend on either operation. This slice does not require
`shuttle.fold`.

The unsupported island must survive conversion and lowering with the same
operation names, types, attributes, and operand relationships after ignoring
SSA names and transient Shuttle provenance. Its input may be the lowered
result of the first selected region. Shuttle must not rebuild, absorb, or
constant-fold the island as part of this proof.

### Coverage manifest and provenance lifetime

`shuttle-annotate-source` assigns a structural source reference to every
source result. `shuttle-form-structural-regions` then writes a transient
module-level coverage manifest. The manifest records:

- the complete source-result reference set;
- the selected reference set, grouped by region;
- the excluded reference set and a closed structural reason such as
  `unsupported_operation`, `effect_boundary`, `control_boundary`, or
  `external_user_boundary`;
- the numerical policy and canonical tuning digest used for selection; and
- a format version.

Selected and excluded references are disjoint and their union equals the
complete source-result set. Every selected reference belongs to exactly one
region. Exclusion reasons describe structural facts; they cannot contain
module, function, or workload names. The manifest is compiler state, not a
public StableHLO attribute.

At the algebra stage, `shuttle-verify-source-coverage{stage=algebra}` requires
the selected set to equal the union of provenance on the selected regions'
Shuttle operations. It also verifies each region's explicit live-in/live-out
boundary. Deleting a selected source operation without provenance-bearing
algebra fails the pass.

Lowering transfers each Shuttle operation's source references to the resulting
StableHLO operations. One source reference may appear on several lowered
operations, but every lowered reference must occur in the selected set. The
lowered coverage check requires:

- selected manifest references equal the selected references represented by
  lowered StableHLO;
- excluded manifest references remain represented by untouched StableHLO;
- no reference appears in both classes; and
- the union of represented selected and excluded references equals the
  manifest's complete set.

This equality makes the second coverage check meaningful after all
`shuttle.region` operations have disappeared. It also catches a lowering that
drops selected provenance or accidentally consumes an unsupported operation.

`shuttle-strip-source-provenance` runs only after lowered coverage succeeds. It
removes the coverage manifest, source-reference attributes, source locations
owned by Shuttle, and temporary selection markers. It does not erase Shuttle
algebra operations or conceal an incomplete conversion.
`shuttle-verify-no-shuttle-ops` then rejects any remaining `shuttle.*`
operation, attribute, location, or option payload before StableHLO-to-HLO
conversion.

### One native pipeline builder

The pass library exposes one builder for the production sequence:

```cpp
void buildShuttleStablehloPipeline(
    mlir::OpPassManager &manager,
    const ShuttlePipelineOptions &options);
```

`ShuttlePipelineOptions` is constructed from the validated canonical compiler
options. It contains numerical policy and tuning data but no workload label.
The builder fixes pass order and verifier placement. Callers cannot assemble a
shorter production pipeline that omits coverage, provenance stripping, or
final erasure.

`shuttle-opt` registers a command-line pipeline that calls this builder. The
XLA StableHLO transform registration calls the same builder from the same pass
library. Individual passes remain available for focused MLIR tests, but neither
the offline proof nor the jaxlib integration test manually reconstructs the
production sequence.

`SOURCE_ORDERED` and `FAST` are distinct `ShuttlePipelineOptions` values,
coverage-manifest policy fields, semantic-policy digests, and JAX compilation
cache keys. The first slice may lower them to identical StableHLO. Identical
physical lowering does not permit the compiler to merge their cache or policy
identities.

### Offline `shuttle-opt` gate

The native offline gate runs before building a patched jaxlib. It requires all
of the following on the frozen forward and JAX VJP StableHLO fixtures:

1. Run the shared production pipeline through `shuttle-opt` under both
   `SOURCE_ORDERED` and `FAST`.
2. Preserve an IR dump after StableHLO-to-Shuttle conversion and a fully
   lowered StableHLO output for each module and policy.
3. Check one forward region and the three VJP regions listed above, with only
   generic `shuttle.contract` and `shuttle.map` operations.
4. Check exact algebra-stage and lowered-stage manifest equality. A mutation
   that removes one selected reference, duplicates a selected reference into
   the excluded class, or absorbs one unsupported reference must fail.
5. Check the VJP `constant -> broadcast_in_dim -> subtract` island is
   structurally unchanged after ignoring SSA names and temporary provenance.
6. Check the lowered modules parse and verify as StableHLO and contain no
   Shuttle operation, attribute, location, manifest, or selection marker.
7. Execute the original and lowered modules on CPU with fixed nonzero inputs.
   Compare the forward result and all three cotangents. `SOURCE_ORDERED` must
   match the original bitwise for this initially identity-shaped lowering;
   `FAST` uses the same requirement until it enables a declared rewrite.
8. Check `SOURCE_ORDERED` and `FAST` produce distinct canonical option and
   semantic-policy digests even if their lowered StableHLO fingerprints are
   equal.
9. Rename the module and function in each fixture and check that region
   selection, algebra fingerprint, and lowered structure are unchanged.

This gate proves the native dialect, conversion, coverage, and lowering without
a Python compiler path. It does not prove that ordinary `jax.jit` invokes the
pipeline.

### Patched-jaxlib ordinary-JAX gate

The integration gate uses jaxlib 0.10.1, XLA revision
`9b635916ecc6df6efee62d8e4b0c7ef87ef84d69`, the reviewed StableHLO transform
hook, and the same Shuttle pass library that built `shuttle-opt`. The test calls
only:

```python
jax.jit(
    reference_function,
    compiler_options=shuttle.compiler_options(
        numerics=numerics,
        tuning=tuning,
    ),
)
```

and an ordinary `jax.jit` of `reference_vjp` with the same compiler options.
No custom primitive, typed FFI call, serialized-HLO callback, Python StableHLO
parser, textual-HLO rewrite, or precomputed workload plan may participate.

The CPU integration test must:

1. execute forward and JAX-owned VJP under `SOURCE_ORDERED` and `FAST`;
2. compare the output, `dx`, `dw0`, and `dw1` with Shuttle disabled, using the
   same numerical requirements as the offline gate;
3. obtain transform diagnostics from the native registration and check the
   forward/VJP region counts and complete coverage manifests;
4. check the unsupported VJP island in the pre-HLO transformed module;
5. check the module handed to StableHLO-to-HLO has no Shuttle semantics or
   provenance;
6. compile the same callable and shapes twice under one policy and observe one
   cache identity, then compile under the other policy and observe a distinct
   cache identity and native transform invocation; and
7. fail compilation, without fallback, when the enabled Shuttle transform is
   absent or any coverage, lowering, or erasure verifier fails.

Passing the offline gate alone is a compiler-unit milestone. Passing this
patched-jaxlib gate establishes the first ordinary-JAX forward-and-training
vertical slice. It makes no GPU, scheduling, kernel-performance, collective,
or transport claim.

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
