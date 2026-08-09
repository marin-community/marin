# Demand-sliced Contract boundary

## Result

After the generated gated-product replacements, the natural Grug HLO retains
24 live Contracts and 397,312 logical Contract FLOPs. Six Contracts use one
concatenated noncontracting operand and immediately slice the corresponding
output dimension at the same boundaries. They account for 205,824 FLOPs, or
51.804% of the remaining Contract work.

The generic recovery represents each one as a static multi-result Contract ABI.
The six call sites normalize to four physical shape/layout families:

- two forward/rematerialized `[2,4,32] x [66,32]` calls with four outputs;
- two forward/rematerialized `[2,4,32] x [68,32]` calls with three outputs;
- one `[8,32] x [66,8]` weight-adjoint call with four outputs;
- one `[8,32] x [68,8]` weight-adjoint call with three outputs.

The HLO replacement removes the HLO concatenation, Contract, and slices,
then restores each slice name as a typed-FFI tuple result. Physical layouts,
external users, and all ten placement collectives remain unchanged.

This is an ABI and planning checkpoint, not a compute-ownership result. The
slices are views, and replacing the existing Contract with an opaque call does
not by itself eliminate meaningful work.

## Physical candidates

The preferred backend is one shared-reduction mainloop with:

- partition-aware loads from the logical concatenated operand;
- one unchanged reduction schedule per output element;
- direct stores into each typed-FFI result layout;
- no concatenated operand or result materialization.

This should be implemented as one generic multi-output Contract template.
Current static output arities require four typed-FFI registrations, but they are
instances of the same template rather than workload-specific kernels.

Issuing one Contract kernel per partition is algebraically legal because the
partition axis is noncontracting. It is retained only as a fallback candidate.
Three or four launches, including narrow 2-wide and 4-wide partitions, are
unlikely to improve on the existing concatenated GEMM.

An efficient CUDA body is deliberately deferred. The bounded backend experiment
should compare a partition-aware CuTe/QuACK mainloop with the existing
concatenated XLA GEMM. A wrapper around the same opaque GEMM is not sufficient.

## Attachment opportunities

The downstream dataflow identifies where the boundary can remove real work.

The `[32,32,4]` output families expose a cross-partition local Map over the two
32-wide partitions before the next Contract. Attaching that generated Map to
the producer emits the fused hidden value directly while retaining the 4-wide
router result. This removes the two activation-sized gate/up intermediates and
is the smallest useful physical experiment.

The `[32,16,16,2]` families expose per-row sum-of-squares Folds for two projected
partitions before downstream attention Contracts. A later candidate can emit
the Fold partials from the projection and apply the row scale during attention
input preparation. This is the normalization analogue of the existing generic
RMS preparation/finalization choice.

The two weight-adjoint families expose convert, scale, and update Maps between
each output partition and the train-step root, with no intervening collective.
Those Maps could consume accumulator fragments directly, but doing so expands
the boundary to optimizer-state reads and parameter updates. They should follow
the smaller projection-plus-pair-Map experiment.

## Numerical contract

Partitioning does not change the logical K reduction. A source-order claim still
requires the backend to preserve the same per-output reduction schedule and BF16
result boundary. Alternative tensor-core tile schedules should use the existing
ordered or reassociation numerical policies rather than infer equivalence from
the partition proof alone.
