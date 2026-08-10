# Attached partitioned Contract Map

## Boundary

The post-gated natural Grug HLO contains one forward projection whose 68-wide
output is demanded as contiguous widths `[32, 32, 4]`. The first two partitions
are used exclusively by a pointwise scalar chain. The third partition remains
independent and feeds routing logic.

Shuttle now recovers the scalar chain directly from the inlined HLO dataflow:

```text
gate * (1 / (1 + exp(-gate))) * up
```

Every intermediate and operation in this expression is BF16 in the source HLO.
The physical program therefore requires FP32 Contract accumulation followed by
BF16 round-to-nearest-even conversion of the two accumulator partitions before
the generated source-ordered scalar body runs. The scalar result is BF16. The
4-wide passthrough is independently rounded and stored as BF16.

The exact replacement has four physical inputs:

```text
activation
32-wide first weight partition
32-wide second weight partition
4-wide passthrough weight partition
```

and two outputs:

```text
generated 32-wide scalar-Map result
independent 4-wide partition
```

It removes the logical weight concatenation, 68-wide dot result, two 32-wide
slice views, and the standalone scalar-Map chain. It preserves the existing
consumer of the generated 32-wide value, the existing router consumer, all ten
collectives, and all physical result layouts.

The rematerialized/backward projection is not absorbed. Its first two
partitions have additional adjoint consumers, so emitting only the fused Map
result would drop live values. This rejection follows liveness, not instruction
names or model metadata.

## Generic physical program

`PartitionedGemmProgram` extends the existing fixed-mainloop idea with:

- contiguous logical accumulator partitions;
- scalar finalizations that reference one or more equal-width partitions;
- independently stored passthrough partitions;
- explicit accumulator, partition-boundary, and output dtypes;
- generated cast-aware scalar bodies.

All partitions share one K reduction and one GEMM mainloop. Splitting the three
partitions into separate launches is not the intended implementation.

The program is mutation-general. Replacing the natural SiLU chain with
`tanh(gate) * up` changes only the imported scalar AST and generated scalar
source. The partition descriptor, physical template, layouts, and replacement
algorithm remain unchanged.

## QuACK boundary

The current QuACK adapter supports one ordinary accumulator output and an
`acc_pair` mode that combines two equal halves. It cannot express an arbitrary
partitioned accumulator that combines two 32-wide partitions while separately
storing a 4-wide tail. Treating the router tail as part of `acc_pair` would be a
workload-specific encoding and would give the scalar Map the wrong domain.

The required generic QuACK/CuTe extension is a partition-aware epilogue adapter:

1. retain the existing shared-reduction mainloop;
2. expose accumulator fragments with logical N-range identities;
3. round scalar-Map source partitions to their declared boundary dtype;
4. invoke the generated scalar body over aligned coordinates;
5. store scalar results and passthrough partitions directly in their declared
   output layouts.

This is a bounded backend experiment. The checked-in typed-FFI replacement is
structurally valid but has no registered physical target. It is not an accepted
compute path and carries no performance or GPU-correctness claim.

### Pinned interface inspection

The bounded adapter targets QuACK revision
`84ef91df9bec87c7e4938517234fafb07ef844dd`. At that revision, the reusable
pieces and missing pieces are precise:

- `quack/epilogue/frontend.py` supports elementwise accumulator visits and
  `acc_pair`. `acc_pair` groups adjacent N lanes and halves every output. It
  cannot pair corresponding coordinates from two contiguous 32-wide ranges or
  preserve an independent 4-wide range.
- `quack/epilogue/visit.py::epi_visit_subtile` is the smallest place to add a
  coordinate-aware accumulator-partition view. The view must use the fragment's
  logical coordinate tensor; CuTe register layout is thread-distributed, so a
  Python slice of the register storage is not a semantic N slice.
- `quack/epilogue/ops.py::TileStore` already owns conversion, register-to-shared
  copy, and TMA store setup. It needs a generic input N-range to output-local
  coordinate map instead of the existing Boolean `gated` half-width special
  case.
- `quack/gemm_base.py` already sequences multiple auxiliary output fragments
  through conversion, shared memory, and TMA stores. It does not need a
  workload-specific branch.

The natural inputs are three distinct HLO parameters. They are not aliases or
views of one packed weight allocation:

```text
f32[32,32] parameter(21) -> bf16 -> transpose -> N [0:32]
f32[32,32] parameter(22) -> bf16 -> transpose -> N [32:64]
f32[32,4]  parameter(17) -> bf16 -> transpose -> N [64:68]
```

Consequently, epilogue partitioning alone is insufficient. The mainloop's B
producer must accept a static segmented tensor source: one logical `(N,K)`
domain backed by several independent tensors. Each TMA stage composes the
loads for all source intervals intersecting its N tile and accounts for their
combined transaction bytes in the existing pipeline barrier. This remains a
generic tensor-source adapter; it does not encode gate, up, routing, or a model
name. QuACK's later `concat_layout` support only interleaves one physical
tensor and therefore does not remove this requirement.

`plan_quack_partitioned_gemm_adapter` records the reusable physical contract:

1. static segmented RHS source intervals;
2. FP32 logical accumulator partition views;
3. BF16 round-to-nearest-even boundaries before the generated scalar AST;
4. direct scalar and passthrough stores with output-local coordinates;
5. the exact QuACK implementation hooks above.

The smallest implementation sequence is:

1. add a segmented RHS tensor descriptor and compose its intersecting TMA
   loads into the existing shared B stage;
2. add an accumulator-partition descriptor to `EpiMod` and its semantic cache
   key;
3. form dense coordinate-aligned partition fragments in
   `epi_visit_subtile`;
4. generalize `TileStore` from `gated` half-width output to an explicit N-range
   domain map;
5. keep the existing multi-output store driver unchanged;
6. compile and test SiLU and tanh scalar mutations through the same adapter,
   then measure against the natural concatenated GEMM.

Until those steps execute on GPU, the adapter plan is an implementation
contract only. It does not establish physical ownership or satisfy a
performance acceptance row.

The follow-up [source-lineage audit](quack_partitioned_mainloop_lineage.md)
also checks the locked `quack-kernels==0.5.0` package and narrows the first
tiled implementation to one shared A stage plus coordinate-aligned WGMMA
groups for each RHS segment. The newer package does not remove the segmented
operand limitation.
