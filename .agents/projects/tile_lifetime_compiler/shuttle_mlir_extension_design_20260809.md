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

The canonical JSON contains `schema_version = 1` and
`pipeline_abi_version = 1`. The schema version changes when the wire shape
changes. The pipeline ABI version changes when existing fields acquire new
compiler semantics, ensuring that such changes cannot reuse an executable
compiled under the earlier semantics.

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

Candidate regions come from maximal contiguous supported-pure intervals in
each source block, followed by deterministic SSA weakly connected components
within each interval. Unsupported, nonlocal, effectful, alias-sensitive,
collective, or control-flow operations are lexical boundaries. Selection
depends on operation structure, types, indexing maps, effects, and shapes,
never module or function names.

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

Region partitioning is deterministic and generic. For each source block, the
formation pass:

1. classifies each operation from its registered conversion, types, indexing
   maps, regions, effects, and control or alias obligations;
2. splits the block into maximal contiguous intervals of supported pure
   operations, with every unsupported or nonlocal operation ending an
   interval;
3. builds an undirected graph within each interval whose vertices are
   operations and whose edges are direct SSA producer-consumer relationships;
4. partitions that graph into weakly connected components; and
5. orders components by their minimum source operation ordinal, breaking any
   remaining tie by the lexicographic sequence of member ordinals.

Each weakly connected component is one candidate region. Its member operations
retain source order. Simultaneously selected disconnected components are
materialized in the deterministic component order after boundary verification;
their purity and lack of SSA edges make this reordering semantics-preserving.
An unsupported operation is never crossed when constructing an interval. A
shared external operand does not connect two components.

No pass option, fixture metadata, or test assertion supplies a target region
count. Tests run the partition rule, compare the exact normalized membership,
and treat the count only as a derived result. A component that fails boundary
verification is classified as excluded with its structural reason; it is not
silently split or dropped.

Applying this rule to the forward module produces one region:

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
StableHLO. The first two supported operations form the first interval. The
remaining supported suffix is a second interval containing two weakly
connected components. The derived VJP regions, in deterministic order, are:

1. the primal `dot_general -> tanh` prefix, exporting the activation;
2. `dot_general -> transpose`, producing `dw1`; and
3. `dot_general -> multiply -> multiply -> add`, followed by the two
   `dot_general` branches and the `transpose` that produce `dx` and `dw0`.

The second and third regions occupy one contiguous supported interval but are
disconnected: the `dw1` branch consumes the exported activation, while the
`dx`/`dw0` branch consumes the unsupported subtraction result. This derivation,
not an expected count of three, defines the result.

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
source result and a structural operation reference to every zero-result
operation. A value anchor is either a function argument reference or a source
result reference. `shuttle-form-structural-regions` then writes a transient
module-level coverage manifest. The manifest records:

- the complete source-result reference set;
- the selected reference set, grouped by region;
- the excluded reference set and a closed structural reason such as
  `unsupported_operation`, `effect_boundary`, `control_boundary`, or
  `external_user_boundary`;
- every zero-result operation's operation reference, normalized operation
  fingerprint, classification, and ordered operand value anchors;
- every terminator operand as a tuple of terminator operation reference,
  operand ordinal, and source value anchor;
- every function result as a tuple of function ordinal, result ordinal, and
  the corresponding terminator-operand anchor;
- the numerical policy and canonical tuning digest used for selection; and
- a format version.

Selected and excluded references are disjoint and their union equals the
complete source-result set. Every selected reference belongs to exactly one
region. Exclusion reasons describe structural facts; they cannot contain
module, function, or workload names. The manifest is compiler state, not a
public StableHLO attribute.

Zero-result operations are part of the source audit even though they contribute
no result reference. In the first slice, terminators and any other zero-result
operations remain outside selected regions. Their records prevent result-only
coverage from overlooking a deleted, replaced, or rewired terminator. A future
conversion that selects a zero-result operation needs an explicit algebra
effect model and a new manifest classification; it cannot reuse result
coverage as proof.

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
  manifest's complete result set;
- every zero-result operation record still has the same normalized operation
  fingerprint and ordered operand-anchor relationship; and
- every terminator operand and function result resolves to the same source
  value anchor through the lowered provenance relation.

This equality makes the second coverage check meaningful after all
`shuttle.region` operations have disappeared. It also catches a lowering that
drops selected provenance or accidentally consumes an unsupported operation.
It also catches a type-correct return rewire that leaves the union of source
references unchanged.

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
    const ShuttlePipelineOptions &options,
    std::shared_ptr<const ShuttlePipelineObserver> observer = {});
```

`ShuttlePipelineOptions` is constructed from the validated canonical compiler
options. It contains numerical policy and tuning data but no workload label.
The optional observer is a separate native instrumentation channel. It is not
a field of `ShuttlePipelineOptions`, canonical compiler JSON, policy digest, or
JAX/XLA compilation cache identity.

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

### Native test observer

Offline and patched-jaxlib tests use native observer events instead of parsing
logs or adding diagnostic fields to compiler options. Each pipeline execution
receives a process-unique `invocation_id` from an atomic counter. The ID is only
for correlating observer events; it is absent from the IR, policy digest, and
cache identity.

The observer receives immutable snapshots at three points:

1. after algebra coverage verification: normalized region membership, the
   complete coverage manifest, the unsupported-island fingerprint, numerical
   policy digest, and invocation ID;
2. after lowered coverage verification and immediately before provenance
   stripping: the lowered manifest relation, terminator/function-result
   anchors, unsupported-island fingerprint, policy digest, and invocation ID;
3. after provenance stripping and final erasure verification: the normalized
   StableHLO fingerprint, an explicit no-Shuttle-semantics result, policy
   digest, and invocation ID.

Region membership is an ordered list of structural source references derived
by the partition rule. The unsupported-island fingerprint includes normalized
operation names, types, attributes, and operand-anchor relationships. The
post-strip event cannot contain the manifest or source references; it records
their absence and the final normalized module fingerprint.

Observer callbacks are non-mutating and cannot change pass success. A native
test registry stores observer subscriptions as reference-counted objects. It
uses a mutex to install or remove subscriptions and takes an immutable
subscription snapshot at invocation start. Each test observer stores events in
a mutex-protected map keyed by invocation ID; there is no process-global "last
compilation" record. RAII scope removal waits until every captured observer
reference is released, so a compilation on an XLA worker thread cannot call a
destroyed observer. Failed pipelines emit a terminal failure event and never a
successful post-erasure event.

Tests may install or remove the observer without changing the compiled
executable's cache key. A cache hit does not run the transform and therefore
does not emit a new pipeline invocation. Tests correlate expected misses with
observer IDs and verify hits separately through the isolated cache protocol
below.

### Offline `shuttle-opt` gate

The native offline gate runs before building a patched jaxlib. It requires all
frozen StableHLO fixtures to carry a source audit containing the exact export
command, JAX and jaxlib versions, pinned XLA revision, input shapes and dtypes,
raw StableHLO SHA-256, and normalized StableHLO SHA-256. The test regenerates
each fixture with JAX 0.10.1 and compares parsed, normalized MLIR. Copying an
expected operation list into a hand-authored fixture does not pass the audit.

Normalized comparisons parse MLIR and assign canonical module symbols,
function symbols, block ordinals, and SSA value numbers by structural traversal.
They preserve operation names, types, non-symbol attributes, operand edges,
result order, structural source references, and manifest classifications. They
remove only symbol spelling, SSA spelling, nonsemantic location spelling, and
observer invocation IDs. Rename tests compare normalized structures, not text
after search-and-replace.

The gate requires all of the following:

1. Run the shared production pipeline through `shuttle-opt` under both
   `SOURCE_ORDERED` and `FAST`.
2. Preserve an IR dump after StableHLO-to-Shuttle conversion and a fully
   lowered StableHLO output for each module and policy.
3. Check one forward region and the three VJP regions listed above, with only
   generic `shuttle.contract` and `shuttle.map` operations. Derive these
   memberships from the partition rule; do not request counts of one or three.
4. Check exact algebra-stage and lowered-stage manifest equality. A mutation
   that removes one selected reference, duplicates a selected reference into
   the excluded class, or absorbs one unsupported reference must fail.
   Rewire a function return to a different same-type live value while leaving
   the represented source-reference union unchanged; the terminator and
   function-result anchor checks must fail.
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
   selection, normalized region membership, algebra fingerprint, and lowered
   structure are unchanged.
10. Repeat the forward and VJP proof at `x=[3,2]`, `w0=[2,6]`, `w1=[6,4]`, and
    output cotangent `[3,4]`, all f32. The partition and generic operation kinds
    must match after substituting extents; no fixed primary-fixture extent may
    appear in selection or conversion code.
11. Export and compile an unrelated Map-only graph,
    `transpose((a * b) + c)` for `[2,3]` f32 operands, and an unrelated
    Contract-only graph, `a @ b` for `[3,2] @ [2,4]` f32 operands. The same
    partition, conversion, coverage, lowering, and erasure pipeline must accept
    them without a new provider, workload label, or fixture switch.
12. Regenerate all primary and genericity fixtures from their audited JAX
    sources in the test environment and fail on any unexplained normalized IR
    drift. A JAX upgrade must update the source audit and receive review before
    changing expected partition membership.

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

Cache testing runs in a fresh process with a unique empty persistent-cache
directory and no inherited global JAX cache directory. It sets
`jax_persistent_cache_min_compile_time_secs=0` and
`jax_persistent_cache_min_entry_size_bytes=0` so the bounded CPU fixture is not
discarded by cache admission thresholds. The test creates one Python
`reference_function` object, then constructs separate
`source_ordered_jit` and `fast_jit` wrappers around that same object because
`compiler_options` are immutable properties of a jitted wrapper. The same rule
applies to the VJP callable. "Compile twice under one policy" means call the
same jitted wrapper object twice with identical shapes, not construct two
nominally equivalent wrappers.

The first process calls each policy wrapper twice. Each wrapper must produce
one native observer invocation and one persistent executable entry. The two
policy wrappers for a callable must report distinct cache lookup keys and
distinct policy digests. A second fresh process points at the same isolated
directory, reconstructs the equivalent wrappers, and requires persistent-cache
hits with no native pipeline invocation. Forward and VJP cache records are
audited separately. Helper compilations are forbidden in this process so cache
entry counts and lookup keys remain attributable to these four wrappers.

The CPU integration test must:

1. execute forward and JAX-owned VJP under `SOURCE_ORDERED` and `FAST`;
2. compare the output, `dx`, `dw0`, and `dw1` with Shuttle disabled, using the
   same numerical requirements as the offline gate;
3. use the concurrency-safe native observer to check normalized forward/VJP
   region membership, complete coverage manifests, policy digests, unsupported
   island fingerprints, invocation IDs, and successful post-strip erasure;
4. check the unsupported VJP island in the pre-strip transformed module;
5. check the module handed to StableHLO-to-HLO has no Shuttle semantics or
   provenance;
6. run the isolated two-process cache protocol above, including repeated calls
   to the same jitted object, separate policy wrappers around the same Python
   callable, distinct persistent keys, and zero transform invocations on
   persistent hits;
7. fail compilation, without fallback, when the enabled Shuttle transform is
   absent or any coverage, lowering, or erasure verifier fails; and
8. compile forward and VJP wrappers for both policies concurrently in a
   separate cache-disabled test, then check unique invocation IDs, ordered
   per-invocation observer phases, correct policy/manifest attribution, and no
   callbacks after observer scope teardown.

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
