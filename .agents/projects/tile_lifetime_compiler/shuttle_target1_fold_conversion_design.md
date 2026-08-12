# Shuttle Target 1 Fold conversion design

Status: implementation contract for Target 1 Steps 2 and 3. This note was
audited against Marin commit `ffb3ff8e1b`. It does not change the evaluation
scorecard or claim CPU, rebuilt-jaxlib, or GPU acceptance.

## Scope

Convert one structurally supported `stablehlo.reduce` and its nested scalar add
body into one `shuttle.fold`, then reconstruct the same StableHLO reduction.
The first conversion slice is one input, one rank-0 initializer, one result,
one or more static reduction dimensions, and an FP32 add combiner. Multi-result
reductions, promoted input elements, non-add combiners, and captured reducer
values remain explicit exclusions.

Step 2 makes source accounting recursive before Reduce selection is enabled.
Step 3 enables the closed Reduce subset and requires exact structural
round-trip plus CPU parity. The remaining Target 1 Map gaps, including
non-scalar broadcasts, reshape, and `rsqrt`, stay in the parent rowwise
normalization plan.

## StableHLO order contract

StableHLO Reduce does not prescribe a sequential floating-point reduction.
Its result uses an implementation-defined full binary tree. The tree's
in-order data leaves retain ascending lexicographic input order, while an
implementation-defined number of initializer leaves may appear at
implementation-defined positions.

An imported StableHLO Reduce therefore has `order_free = true` under this exact
Shuttle meaning:

- binary-tree association is free;
- initializer multiplicity and placement are free;
- logical input leaves may not be permuted; and
- no additional algebraic identity, commutativity, or reassociation claim is
  inferred from the combiner operation.

Both `SOURCE_ORDERED` and `FAST` import the same `order_free = true`. The flag
records source-granted freedom, not policy-selected freedom. In particular,
floating-point non-associativity does not make the flag false, and the
commutative trait on add does not permit leaf permutation.

`order_free = false` denotes the stricter Shuttle Fold form whose accumulator
updates follow ascending lexicographic reduction-index order with one initial
value before the first update. StableHLO Reduce cannot encode that guarantee.
Source-ordered StableHLO lowering must reject such a Fold. A future physical
lowering may implement it directly, but it must not round-trip it through
StableHLO.

The `order_free` name and Boolean storage remain unchanged. The verifier and
README must state the definition above so a scheduler cannot interpret `true`
as arbitrary permutation.

## Rank-0 initializer normalization

StableHLO `init_values` are rank-0 tensors. `shuttle.fold` currently verifies
scalar initializers, which cannot retain the source SSA value without adding an
unattributed extract and reconstruction pair.

Fold initializers must remain ranked, rank-0 tensors at the operation boundary.
For each result `i`:

- `accumulator_types[i]` is a scalar type;
- `initializers[i]` is `tensor<accumulator_types[i]>`;
- Fold combiner arguments and `shuttle.yield` values are scalar accumulator
  values; and
- the result tensor element type is the accumulator type.

The first converter requires the input element, initializer element, combiner
argument, combiner result, and Reduce result element to be FP32. StableHLO's
more general promotable-input rule remains outside the slice. Lowering reuses
the original rank-0 initializer operand and preserves the dimensions array in
its original order.

This changes the verified meaning of existing Fold IR. Hand-authored scalar
initializer tests must move to rank-0 tensors; no compatibility form is kept.

## Recursive source model and manifest v2

`shuttle-annotate-source` already descends into nested regions. The remaining
passes currently enumerate only function-body operations. Manifest v2 uses one
deterministic preorder over every source operation and nested block.

Every source operation receives `shuttle.operation_ref`. Result-producing
operations also retain `shuttle.source_refs`. A nested block argument anchor is
the tuple of its owning operation reference, region ordinal, block ordinal, and
argument ordinal. This makes reducer operands auditable without assigning
source results to block arguments.

The Reduce operation reference is owner-anchor provenance, independent of its
result source reference. Conversion copies it to the Fold as the transient
`shuttle.operation_ref` attribute. Coverage verifies that the owner reference is
unique and that every Fold combiner block-argument anchor resolves through it.

For one selected Reduce, provenance is represented as follows:

- `shuttle.fold.source` is the Reduce result's source reference;
- `shuttle.fold` carries the Reduce operation reference used to anchor its
  combiner block arguments;
- the scalar `arith.addf` in the Fold combiner carries the source reference of
  the nested `stablehlo.add` result; and
- the combiner `shuttle.yield` carries the operation reference of the nested
  `stablehlo.return`.

The enclosing `shuttle.region.source_refs` and its manifest group contain both
the Reduce result and nested add result, in source preorder. Generated region
and Map terminators have no source operation reference and do not enter the
manifest.

Manifest v2 records all nested source results and all source zero-result
operations. It verifies its `version` field exactly. Unsupported region-bearing
operations exclude the parent result with `unsupported_operation` and nested
results with the closed reason `enclosing_region_excluded`. This keeps the
complete/selected/excluded partition total before Reduce selection is enabled.

Coverage verification is stage-aware:

- at the source stage, the nested `stablehlo.add` and `stablehlo.return` own
  their annotated references;
- at the algebra stage, the Fold, scalar `arith.addf`, and combiner
  `shuttle.yield` own those same references; and
- at the lowered stage, the recreated Reduce, `stablehlo.add`, and
  `stablehlo.return` own them again.

Exactly one operation represents each selected source result and each source
terminator reference at every stage. The verifier rejects a surviving selected
StableHLO operation, a missing or duplicate combiner source, a missing or
duplicate terminator reference, a changed terminator operand anchor, and an
excluded nested-operation mutation.

The normalized StableHLO fingerprint must recurse into regions, block argument
types, nested operations, and terminator operands. The region-free hash format
should remain unchanged so the existing ten Contract/Map fixture hashes do not
need an unrelated refresh.

## Selection and conversion

Add a closed `isSupportedStablehloReduce` predicate in
`lib/shuttle/mlir/lib/Transforms/Passes.cc`. It accepts only:

- one ranked FP32 input, one rank-0 FP32 initializer, and one ranked FP32
  result;
- a nonempty, unique, in-range dimensions array;
- one reducer block with two rank-0 FP32 block arguments;
- exactly one default-semantics `stablehlo.add`; and
- one `stablehlo.return` that returns the add result directly.

Selection treats the Reduce as one top-level candidate node. Its nested add
source belongs to the same candidate but is never a separate structural
region. Component connectivity remains based on the top-level Reduce operands
and results. Materialization moves the Reduce and its body together and marks
both result-producing source operations as selected. It retains the Reduce
operation reference because nested source anchors depend on it.

`convertStablehloReduce` builds one Fold with the source input and rank-0
initializer, original dimensions, scalar FP32 accumulator type,
`order_free = true`, and the Reduce result source. It replaces the reducer's
rank-0 tensor arguments with scalar Fold combiner arguments, converts the
nested add to `arith.addf`, converts `stablehlo.return` to `shuttle.yield`, and
transfers the Reduce operation reference plus nested add and return provenance.
No operation or fixture name participates in selection.

## Lossless lowering

Fold lowering is a dedicated region-aware path rather than an `OperationState`
leaf helper. It must:

1. reject `order_free = false`;
2. validate the same closed scalar add body accepted by conversion;
3. create `stablehlo.reduce` with the original ranked input, rank-0 initializer,
   result type, and dimensions array;
4. create rank-0 tensor reducer block arguments;
5. rebuild `stablehlo.add` and `stablehlo.return` in the same operand order;
6. restore the Reduce operation and result references, add-result reference,
   and terminator operation reference; and
7. let lowered-stage coverage verification run before provenance stripping.

The source-ordered and fast pipelines initially use this same lossless
lowering. Policy-specific Fold fusion or physical scheduling is a later step.

## Observable versions

Recursive coverage changes the manifest wire contract from version 1 to
version 2. The verifier must reject either a missing version or any value other
than 2.

Reduce selection, Fold ordering semantics, rank-0 initializer verification,
and recursive normalized fingerprints change observable compiler behavior for
unchanged JSON options. Bump `pipeline_abi_version` from 2 to 3 in:

- `lib/shuttle/src/shuttle/options.py`;
- `lib/shuttle/mlir/include/shuttle/Transforms/Observer.h`;
- `lib/shuttle/mlir/lib/Transforms/Passes.cc`;
- `lib/shuttle/mlir/lib/Transforms/XlaRegistration.cc`; and
- the option, observer, registration, and adapter tests.

The JSON schema remains version 1 because its field shape does not change.
Pipeline ABI 2 becomes an explicit rejection case, and option/cache digest
expectations must be recomputed.

## Behavior-first test sequence

1. Add an anonymous small Reduce fixture and pin the current structural-region
   failure. After Step 2, require manifest v2 to include the Reduce result,
   nested add result, nested return, and function return while the Reduce
   subtree remains explicitly excluded.
2. Add test-only mutations that remove the combiner add source, remove the Fold
   owner operation reference, remove the reducer terminator operation
   reference, duplicate any of those references, and rewire the reducer
   terminator. Each mutation must fail the public coverage verifier.
3. Enable the closed Reduce predicate and require one Fold with a rank-0 FP32
   initializer, exact dimensions, FP32 accumulator, `order_free = true`, add
   source reference, and terminator operation reference. Require
   `excluded = []`.
4. Run the full source-ordered and fast pipelines and compare recursive
   normalized hashes with the original StableHLO. Mutate the reduction axis and
   input shape; both variants must use the same conversion and lowering.
5. Execute original and round-tripped modules on CPU with fixed inputs. Compare
   public results bitwise for the small FP32 fixture. The full Target 1 forward
   fixture follows after its remaining Map operations are supported.
6. Add negative tests for `order_free = false` lowering, scalar initializers,
   initializer/accumulator dtype mismatch, duplicate or out-of-range
   dimensions, extra reducer operations, non-add combiners, promoted inputs,
   and multi-result Reduce.
7. Re-run the ten existing Contract/Map raw and hook-boundary hashes, the full
   Shuttle MLIR/C++ gate, Python option tests, and repository precommit.

No GPU test or rebuilt jaxlib is required for Steps 2 and 3. Rebuilt-jaxlib CPU
acceptance is required before claiming the ordinary-JAX callback path, and GPU
acceptance remains gated on the later physical Fold lowering and Target 1
oracle contract.
