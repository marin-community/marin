# Event Tensor workload linkage

## Purpose

This checkpoint connects mechanically derived Event Tensor plans to real tensor
payload execution. It remains schedule-level: tensor semantics first lower to
tasks and exact task dependences, then eventization and physical realization
are chosen.

## Generic buffer and reuse model

`BoundedBufferPlan` contains:

- one producer task family whose coordinates identify logical buffer items;
- exact item-to-reader relations;
- finite physical capacity;
- one physical slot and generation for each logical item;
- a mechanically derived unique last consumer per item;
- last-consumer-to-next-producer dependences for items sharing a slot.

The last consumer is the unique maximal reader under the scheduled task graph.
The planner rejects an item with no reader, incomparable terminal readers,
duplicate `(slot, generation)` identities, or non-contiguous generations within
one reused slot. The workload does not annotate a last-consumer task.

For the streaming Contract/Fold graph, the exact dependences are:

```text
key_value_stage[row, partition] -> qk_contract[row, partition]
key_value_stage[row, partition] -> pv_contract[row, partition]
qk_contract[row, partition]     -> fold_update[row, partition]
fold_update[row, partition]     -> pv_contract[row, partition]
pv_contract[row, partition]     -> fold_finalize[row]
```

The finite K/V buffer maps partition `p` to slot `p % depth` and generation
`p // depth`, independently per row tile. Reachability proves that PV is the
last K/V consumer. Reuse therefore adds:

```text
pv_contract[row, p] -> key_value_stage[row, p + depth]
```

## Event realization audit

Every `EventTensorPlan` is classified as one of:

- `erased_program_order`: a covering task order proves readiness;
- `erased_stream_order`: a covering stream order proves readiness and
  visibility;
- `physical`: a barrier, semaphore, or other synchronization mechanism remains.

An erased realization carries the relation that supposedly orders the tasks.
The verifier requires that relation to cover every scheduled edge, including
legal false dependencies introduced by event coarsening. This makes erasure an
audited lowering result rather than an informal optimization note.

## Runtime RelationPlan / SegmentedContract lowering

Typed FFI inputs:

```text
source         f32[source_item, K]
weight         f32[segment, K, N]
event_counts   s32[segment]
event_offsets  s32[segment + 1]
edge_sources   s32[edge]
```

Typed FFI output:

```text
output         f32[edge, N]
```

The generated CUDA launches one CTA per runtime segment. The RelationPlan CSR
counts and offsets determine the ragged task domain. Each edge reads a real
source row and contracts it with the corresponding segment weight. This is not
a scalar counter validation.

The relation-edge event erases to program order in this first lowering: payload
gather and Contract execute in the same task body after the JAX stream has made
the FFI operands visible. A future asynchronous transport can preserve the same
logical event and select a physical device/system readiness primitive instead.

## Streaming Contract/Fold lowering

Typed FFI inputs:

```text
query         f32[row_tile, query_tile, K]
key           f32[row_tile, partition, kv_tile, K]
value         f32[row_tile, partition, kv_tile, V]
domain_valid  s32[row_tile, query_tile, partition, kv_tile]
```

Typed FFI output:

```text
output        f32[row_tile, query_tile, V]
```

The first physical body is intentionally a correctness/reference CUDA
skeleton, not a tensor-core performance kernel. It executes real QK and PV
contractions and the exact online normalized-exponential Fold. A finite
multi-slot shared K/V buffer carries explicit generation state.

Realization choices:

- `key_value_stage -> QK` and `key_value_stage -> PV` remain physical CTA
  acquire barriers;
- `QK -> Fold` and `Fold -> PV` erase to source order within one query-row
  owner;
- `PV -> finalize` erases to completion of that owner's partition loop;
- `PV(last consumer) -> next key_value_stage` remains a physical CTA release
  barrier plus slot-generation advance.

Generated source contains the shared stage, barriers, slot-generation check,
online max/sum/weighted-state update, and final division. Changing pipeline
depth or Fold partition count changes both the plan fingerprint and compiled
constants/source without a workload-name dispatch.

## Current limits and SM100 attachment

The payload kernels are deliberately simple FP32 CUDA. They validate the
EventTensorPlan-to-execution boundary and Torch-free JAX typed FFI, but do not
claim useful GEMM or attention throughput.

An SM100 emitter can retain the same audit while replacing physical bodies:

- segmented Contract: generic grouped/ragged tcgen05 mainloop, with a transport
  completion event instead of erased in-task gather when DeepEP is asynchronous;
- streaming Contract/Fold: TMA K/V producer, tcgen05 QK/PV consumers, mbarrier
  acquire/release, and the same derived circular-slot generations;
- same-owner edges remain erased rather than allocated as global counters.

This boundary keeps event semantics independent of mbarrier, semaphores,
programmatic dependent launch, queues, and stream order.
