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

For the generic resident/streamed Contract/Fold graph, the exact dependences
are:

```text
first_streamed_input_stage[row, partition]  -> first_contract[row, partition]
second_streamed_input_stage[row, partition] -> second_contract[row, partition]
resident_input_stage[row]                   -> first_contract[row, partition]
first_contract[row, partition]              -> fold_update[row, partition]
fold_update[row, partition]                 -> second_contract[row, partition]
second_contract[row, partition]             -> fold_finalize[row]
```

Each finite streamed-input buffer maps partition `p` to slot `p % depth` and
generation `p // depth`, independently per row tile. Reachability proves that
each Contract is the last consumer of its corresponding streamed input. Reuse
therefore adds:

```text
first_contract[row, p]  -> first_streamed_input_stage[row, p + depth]
second_contract[row, p] -> second_streamed_input_stage[row, p + depth]
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

The schedule consumes a backend-neutral descriptor containing one resident
input, two independently buffered streamed inputs, two Contracts, and one
ordered Fold. It does not name attention operands or select a workload by name.

### Attention adapter

The current typed-FFI example is an attention adapter. This is the only layer
in this path that binds the generic resident, first-streamed, and
second-streamed inputs to Q, K, and V, respectively.

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

After that adapter binding, the generic realization choices are:

- each `streamed_input_stage -> Contract` edge remains a physical CTA acquire
  barrier;
- `first Contract -> Fold` and `Fold -> second Contract` erase to source order
  within one resident-row owner;
- `second Contract -> finalize` erases to completion of that owner's partition
  loop;
- each `Contract(last consumer) -> next streamed_input_stage` edge remains a
  physical CTA release barrier plus slot-generation advance.

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
- streaming Contract/Fold: TMA streamed-input producer, tcgen05 Contract
  consumers, mbarrier acquire/release, and the same derived circular-slot
  generations;
- same-owner edges remain erased rather than allocated as global counters.

This boundary keeps event semantics independent of mbarrier, semaphores,
programmatic dependent launch, queues, and stream order.

## GB200 replay

The Torch-free typed-FFI path was replayed on one low-priority GB200 with one
host CPU, JAX 0.10.1, CUDA/NVCC 13.3.73, driver 595.71.05, and `sm_100a`.
Each case retained 30 samples after 10 warmups:

| Case | Median | Maximum absolute error | Deterministic |
| --- | ---: | ---: | --- |
| Segmented Contract | 0.073328 ms | 0 | yes |
| Relation mutation | 0.072800 ms | 0 | yes |
| Streaming Contract/Fold | 0.074672 ms | 2.384e-7 | yes |
| Pipeline-depth/partition mutation | 0.074224 ms | 1.192e-7 | yes |

The purpose of these measurements is proof of physical execution and mutation,
not kernel throughput. The bodies are deliberately small FP32 generated payload
kernels. They validate real tensor/CSR consumption and physical event
realization, not expert grouped-GEMM or tensor-core attention throughput. Raw
distributions, generated CUDA, HLO records, hashes, the
event-realization audit, and exact environment are preserved in
`benchmarks/artifacts/event_tensor_workload_linkage_gb200_v0/`.
