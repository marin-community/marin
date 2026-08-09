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
