# Target 1 abstract SIMT32 schedule contract

Status: local, opt-in compiler planning slice. This is not executable code
generation, a physical device ABI, or an acceptance claim.

## Boundary

The materialization plan fixes exact task dependencies, logical tensor
materializations, and lifetimes. It does not provide buffer addresses, strides,
offsets, alignment, aliasing or reuse, memory spaces, device placement, launch
arguments, barriers, visibility, runtime ownership, or XLA buffer binding.
Those omissions make NVIDIA, H100, GB200, CUDA, NVVM, or any other physical
device target unjustified at this stage.

This slice therefore names only an abstract `simt32` capability profile:

- subgroup width 32;
- at most 256 candidate workgroup threads;
- scalar, flattened elementwise, and one-workgroup-per-row Fold task families;
- a bounded FP32 scratch-resource envelope for the Fold candidate.

The profile is an explicit opt-in planner choice. It is not inferred from a
workload, operation name, host, or device. The passes remain outside the
observed production StableHLO pipeline, so pipeline ABI 5, cache identity,
observer output, and exported StableHLO remain unchanged.

## Logical indexing and schedule candidate

`shuttle.schedule_buffer` records only a logical tensor iteration contract.
Rank-zero tensors use scalar indexing. Positive-rank tensors use lexicographic
identity-dimension iteration (`[0]`, `[0,1]`, and so on). This is not a memory
layout: absence of a RankedTensorType encoding does not establish physical
row-major storage, strides, alignment, address space, or aliasing.

Each materialization task binds one schedule task by structural ordinal. Scalar
Maps have one logical scalar instance. Other Maps flatten their positive static
domain into disjoint, contiguous tiles of at most 256 logical elements. The row
Fold candidate assigns one grid item to each row and partitions the feature
axis into contiguous chunks of at most 256 elements. The final chunk may be
partial. These formulas prove complete in-bounds domain coverage without
overlap.

Each schedule buffer also copies the exact task-ordinal lifetime interval from
the materialization plan. This retains lifetime information for a later
allocator without claiming an address, storage reuse, or alias decision.

The Fold's thread count, subgroup width, serial chunk count, and scratch bytes
are a resource envelope candidate, not an executable algorithm. In particular,
the plan does not specify lane loads, partial-accumulator placement, barriers,
shuffle operations, or how serial chunks are merged. The imported Fold's exact
constraint is `tree_association_free_leaf_order_fixed`: tree association and
initializer placement are free, while leaf order is fixed. Both
`source_ordered` and the post-conversion `fast` mutation retain that constraint;
policy alone cannot strengthen it to leaf-order freedom. A future verified
algebra rewrite would need to change the Fold semantics before scheduling. The
verifier binds this requirement but does not claim a numerical realization
exists.

## Closed verification

The dialect verifier independently recomputes the abstract profile geometry,
logical iteration order, resource bounds, reduction axis and Fold order, and
the schedule fingerprint. The source verifier additionally requires exactly
one materialization plan re-derived from the surviving algebra and one schedule
plan, then compares every buffer type and every task kind, domain, input/output
buffer vector, dependency vector, and semantic fingerprint to that source.

The fingerprint includes schema, target profile, numerical policy, source-plan
fingerprint, and every buffer and task attribute. No symbol, workload name, or
source operation name is serialized or used for dispatch.

## Behavior gates and missing consumer

Both frozen forward shapes emit 21 schedule buffers and 19 schedule tasks.
`2048x4096` produces a row-Fold candidate with grid `[2048]`, tile `[1,256]`,
16 feature chunks, 256 threads, and a 1024-byte scratch envelope. `7x13`
produces grid `[7]`, tile `[1,13]`, one chunk, 32 threads, and 128 bytes.
Rank-zero scalar Maps remain explicit. Symbol renaming preserves the schedule
fingerprint; changing the already-converted Region policy to `fast` changes the
plan policy and fingerprint but retains the Fold-derived reduction constraint.
That FAST gate tests schedule binding only, not full FAST conversion,
reassociation freedom, or numerical execution.

Mutation gates reject logical iteration, reduction axis/order, tile, resource
profile, dependency, source-task, buffer type, unknown-attribute, and
multiple-plan corruption. Axis-zero and BF16-accumulator Folds remain outside
the source materialization boundary.

No static GPU-code or runtime gate is meaningful yet. The next consumer must
choose and verify a concrete Fold algorithm plus physical buffer ABI, address
spaces, layout/strides/alignment, synchronization and visibility, launch ABI,
device placement, and XLA/runtime binding before device IR can be emitted.
