# Routed relation Event Tensor GPU plan

## Current boundary

The dense H100 streaming attachment and the routed GB200 grouped-Contract
proof establish different halves of the required boundary.

- On H100, Shuttle derives the dense streaming Contract/Fold task graph,
  circular-buffer generations, transaction bytes, and outer readiness. CuTe
  owns the physical `mbarrier` sites inside the TMA/WGMMA skeleton. The matched
  replay was deterministic and measured 1.001x the body before attachment.
- On GB200, a Torch-free JAX typed-FFI path consumes runtime `RelationPlan`
  tables, derives segmented readiness, groups and pads rows, and launches a
  generic SM100 grouped Contract on the same stream. Uneven occupancy, an empty
  segment, and a relation mutation execute without changing the generic
  program or grouped-Contract fingerprints.
- The existing clean routed streaming emitter owns normalized-exponential Fold
  semantics, domain restrictions, and deterministic generated partial-state
  finalization. Its tcgen05/TMA physical extraction and executable runtime are
  still Torch-hosted.

The missing proof is therefore not another dense attachment. It is one routed
relation whose mechanically derived right-resource schedule controls a real
SM100 Contract/Fold body and generated partial-state finalizer.

## Generic static linkage

`RightResourceFoldEventSchedule` is derived after semantic lowering. Its input
is only:

- a generic `RelationPlan`;
- a route-slot-to-partition map;
- a maximum edge count per grouped task;
- right-resource payload bytes;
- bounded resource-buffer depth;
- generic grouped-body and Fold-finalizer task names.

The derivation groups relation edges by `(partition, right resource)`, splits
each group into bounded tasks, and produces:

```text
right_resource_stage[task]
  -> grouped_contract_fold_body[task]
  -> partial_state_fold_finalize[left, partition]
```

It also derives circular resource slots, generations, and slot-reuse edges.
The module contains no query, key, value, attention, MoE, or expert role
identifiers. The routed-attention adapter is the only layer that maps semantic
GQA route slots and the two physical right-side operands onto this descriptor.

The current physical binding is intentionally conservative:

- resource-ready and resource-reuse events are physical, with barrier choice
  delegated to the backend primitive;
- grouped-body to Fold-finalizer readiness erases to verified same-device-
  stream order;
- the external tcgen05/TMA primitive continues to own internal `mbarrier`
  allocation and phase advancement.

Program and runtime fingerprints are separate. A relation permutation that
preserves task extents retains the program fingerprint and changes the runtime
fingerprint. Pipeline depth or task-capacity mutations change the program
fingerprint.

## One bounded GB200 run

Do not tune this experiment. Request one low-priority physical GB200 with the
minimum available CPU allocation. Do not substitute a B200.

Use the existing primary routed-streaming shape:

```text
batch: 1
query length: 16384
key/value length: 16384
query heads: 64
key/value heads: 4
feature dimension: 128
right-resource block: 128
selected right resources per left/partition: 16
dtype: BF16
```

Construct a deterministic nonmonotone relation that leaves one right resource
empty. The mutation applies a bijection to the active right-resource IDs, so
the empty resource, occupancies, task count, and physical program remain
unchanged while relation tables and the runtime fingerprint change.

Run exactly these two cases:

1. baseline nonmonotone relation;
2. relation permutation mutation.

For each case retain two warmups and ten measured samples. This is a physical
linkage experiment, not a performance, overlap, or tile-tuning claim.

Record:

- exact GPU, driver, CUDA, JAX, CuTe/CUTLASS, MSA-lineage, and Shuttle
  revisions;
- generated HLO/custom-call targets or equivalent exact handlers;
- Event Tensor program/runtime fingerprints;
- grouped task count, resource-buffer slots/generations, event counts, and
  realization audit;
- generated physical and partial-merge source hashes;
- external dynamic dependencies and confirmation that no complete attention
  entry point is called;
- maximum/mean error against the generic semantic reference;
- repeated-output hashes and deterministic equality;
- raw latency samples.

The expected call boundary is:

```text
runtime RelationPlan tables
  -> generic right-major grouping and bounded task schedule
  -> primitive-owned staged-resource completion
  -> generic SM100 Contract/Fold body
  -> same-stream Event Tensor completion
  -> generated deterministic Fold finalizer
```

## Allocation gate

The static schedule, verifier, mutation, emitter attachment, and CPU tests are
green. A Torch-free JAX boundary now exists before device execution:

- the generic `RelationPlan` and `RightResourceFoldEventSchedule` derive the
  right-major CSR, bounded work list, split ownership, and count tables;
- those tables become JAX operands rather than Torch tensors;
- the extracted physical class is passed to `cutlass.jax.cutlass_call` without
  an opaque routed-attention entry point;
- the compiler-generated Fold finalizer is wrapped as a command-buffer-safe
  JAX typed-FFI handler;
- only the routed semantic adapter names query/key/value roles. The schedule
  and table construction remain generic left/right-resource machinery.

Before allocating a GPU, run
`preflight_jax_right_resource_runtime.py` on a dependency-matched Linux host.
The preflight compiles and registers the Fold finalizer, imports the extracted
CuTe physical source, constructs the `cutlass_call`, verifies that Torch was
not imported, and records source plus EventTensor fingerprints. It does not
execute a device kernel. A GB200 allocation remains blocked until that
dependency-only preflight is preserved and green.
