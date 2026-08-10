# Centered affine Contract preparation

This checkpoint closes the generated-code boundary for the source-ordered
centered affine preparation at the level of a generic physical specification.
It does not claim a tensor-core implementation or a latency result.

The frontend and region planner erase centered affine normalization into two
row Folds followed by this consumer preparation:

```text
FP32(source) - row_center
  -> * row_scale
  -> * feature_scale
  -> + feature_bias
  -> BF16 round-to-nearest-even
  -> Contract mainloop
```

The three ordinary binary operations now carry an explicit right-operand
delivery domain. `row` means one FP32 value per M coordinate and `feature`
means one FP32 value per Contract K coordinate. `scale_row` retains its
existing row-scalar contract. These delivery domains are schedule data; the
generator does not inspect normalization or model names.

`cuda_prepared_contract_codegen.py` is the bounded executable specification.
It emits one generic CUDA kernel with separate M-row and K-feature inputs,
performs every preparation operation with explicit FP32 round-to-nearest
arithmetic, converts the prepared value with `__float2bfloat16_rn`, and only
then enters the ordered Contract accumulation. The scalar loop is intended for
correctness and code-ownership validation, not throughput.

The same generator accepts a mutation that changes the final feature-bias add
to subtraction. The operand family and loop structure remain unchanged while
the semantic and source digests change. Epsilon mutations remain upstream in
the generic Fold finalizer and feed a different row-scale value into this same
physical family.

## QuACK boundary

The generated QuACK A-transform source now represents both delivery domains:

- `colvec_ktile_fp32` for an M-row value repeated across one K tile;
- `kvec_mtile_fp32` for a K-feature value repeated across one M tile.

The pinned patch applies cleanly to QuACK revision
`84ef91df9bec87c7e4938517234fafb07ef844dd`. It generalizes the strip staging
dtype and adds the K-feature kind. The K-feature host view at that revision is
physically repeated per M tile; eliminating that small repetition is a later
backend optimization rather than a semantic requirement.

That revision still has exactly one auxiliary A-side TMA slot. Its
`TransformAOperand` contains only `(A, sf)`, the host adapter requires exactly
one auxiliary view, and `TransformAValue` rejects more than one auxiliary
implementation. Centered affine preparation needs four independent values:
row center, row scale, feature scale, and feature bias. Generated source is
therefore syntactically complete but reports
`executable_with_single_auxiliary_transform_backend = False`; the measured
H100 runtime continues to fail closed instead of silently materializing or
dropping an operand.

A reusable high-throughput implementation needs a multi-auxiliary A-transform
extension that owns:

1. a tuple of auxiliary TMA tensors and per-stage shared-memory layouts;
2. descriptor, barrier-byte, and pipeline accounting for every tuple member;
3. a stable mapping from each transform argument to its staged register view;
4. matching runtime and trace-time host views;
5. a cache key containing the operand delivery kinds and dtypes.

This is not specific to centered normalization. The same extension supports
any generated Contract preparation composed from multiple row- and
feature-broadcast values.

## Delayed algebraic candidate

The delayed alternative remains a separate real-algebra-equivalent plan:

```text
row_scale * Contract(source * feature_scale, W)
  - (row_center * row_scale) * Contract(feature_scale, W)
  + Contract(feature_bias, W)
```

It retains two parameter-vector projections and changes the BF16 cast and
reduction order. It is not labeled source-ordered and is legal only under the
rounding-reassociation policy. The new preparation generator does not rewrite
or collapse this candidate.

## Source lineage and status

The bounded CUDA source is generated entirely from `GemmProgram` tile
operations and generic operand-delivery attributes. It calls no external
semantic kernel and contains no normalization, Transformer, or workload name.

The prospective tensor-core path derives from the pinned QuACK A-transform
and GEMM mainloop. It retains generic WGMMA, TMA, strip-layout, and pipeline
machinery. No named normalization callback or model control flow is retained.
The existing QuACK patch and the new delivery kind are insufficient by
themselves because the multi-auxiliary transport described above is still
missing.

CPU tests compare the generated preparation boundary and the complete bounded
Contract exactly with natural JAX BF16 semantics at the fixed test shape; both
maximum and mean absolute differences are zero. Static audits verify one
generated kernel, explicit FP32 preparation, BF16 round-to-nearest-even before
the mainloop, ordered FP32 accumulation, no atomics, and no opaque semantic
dependency. GPU correctness and performance remain unmeasured.
