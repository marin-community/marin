# Shuttle ownership of routed transport

Date: 2026-08-09

## Decision

Shuttle should own the relation metadata, transport schedule, readiness, and
the ABI presented to JAX. JAX should continue to own automatic
differentiation. A physical backend may provide payload permutation or remote
copy, but it may not multiply route weights, combine returned edges, or hide a
named routed-compute program.

The first clean ABI is exact-edge transport:

```text
RelationPlan
  -> transport metadata plane
  -> source-to-destination payload permutation
  -> destination computation
  -> destination-to-source [source item, relation slot] payload permutation
  -> generated source-ordered Fold
```

The same topology is instantiated for the cotangent program JAX derives:
output cotangent dispatches to destination work; input and edge-attribute
cotangents return with exact relation-edge identity. Transport does not define
the cotangent algebra.

This interface follows the useful communication boundary visible inside MoK,
but does not call MoK or encode an expert, activation, or model name. It also
admits a DeepEP dispatch backend for the outbound leg. Public DeepEP combine
is not an accepted backend for the return leg because its interface performs a
semantic reduction.

## Evidence map

### Existing Shuttle structure

- `relation.py` already provides stable flat relation-edge identity,
  destination-major padded rows, valid masks, group counts/capacities/offsets,
  inverse source-slot mapping, and a coalesced `(source item, destination rank)`
  projection. Fixed-capacity construction rejects overflow.
- `event_dataflow.py` already derives counted readiness from exact task
  dependence. Zero-indegree consumers are initially ready; phased storage
  bindings carry generation identity; system visibility represents
  release-on-notify and acquire-before-consumer.
- `event_dataflow_adapters.py` already derives segmented-Contract and
  collective-completion readiness without workload dispatch.
- `distributed_expert_jax_module.py` already lets JAX own AD across explicit
  payload collective boundaries. Replacing those boundaries with a
  Shuttle-owned transport ABI is the appropriate integration seam.
- `gb200_deepep_mok_distributed.py` previously demonstrated the correct
  semantic split: DeepEP dispatch for outbound payload, a return placement
  transition, and a compiler-owned ordered Fold. The remaining problem was
  that the placement transition was external rather than a Shuttle plan.

### DeepEP

The repository's training example makes the semantic boundary explicit:

- dispatch returns received payload, received route metadata, a handle, and an
  overlap event;
- combine returns a reduced value at the original rank;
- dispatch backward is implemented by combine and combine backward by
  dispatch;
- `EventOverlap.current_stream_wait()` exposes coarse communication completion.

The local pinned JAX wrapper at DeepEP commit
`7febc6e25660af0f54d95dd781ecdcd62265ecca` confirms the same boundary.
`deepep_dispatch_intranode` is a transport/permutation candidate.
`deepep_combine_intranode` invokes upstream combine with route metadata and
produces combined values, so it is an opaque semantic Fold for the clean path.
The cached dispatch variant is payload-only but does not by itself expose the
arbitrary inverse exact-edge return ABI required here.

DeepEP V2 at audited commit
`01dc3aaac82068020353dce2c302e38153c0bfaa` unifies dispatch/combine around
`ElasticBuffer` and improves the performance/configuration surface, but does
not change this semantic distinction.

### Mixture of Kittens

Pinned commit: `3e1cf43ab93ad040afed52a45ab03cb490ffe4be`.

The internal mechanics provide a useful transport reference even though the
published megakernel entry point is not acceptable as a clean Shuttle backend:

- schedule rows retain `peer_rank` and `peer_token_idx = source * topk + slot`;
- dispatch receiver-pulls each exact source edge from symmetric memory;
- return direct-pushes each exact per-edge result into a source-owned
  `[source, slot]` buffer without reducing it;
- route-weighted forward merge and input-cotangent merge occur later in
  epilogue logic and are therefore separable from transport;
- route cotangents also return into exact source-slot locations;
- communication CTAs, macrobatches, readiness counters, and buffer phases
  overlap dispatch, computation, and return;
- release/acquire system counters protect remote visibility, while CTA-local
  stages use shared barriers and phase bits.

The parts worth abstracting are exact edge addressing, pull/push placement
transitions, bounded buffers, and readiness. Row scaling, weighted merge, and
the fused model program must not cross the generic transport interface.

## Prototype data model

`relation_transport.py` derives:

- a metadata plane with exact edge/source/slot/rank identity;
- destination group logical counts and physical capacities;
- rank-pair logical counts and per-rank physical offsets;
- deterministic destination-row and inverse source-slot mappings;
- the existing coalesced outbound projection as an optional backend view;
- four transport legs: primal dispatch/return and cotangent dispatch/return;
- an EventTensor readiness plan for each leg with system visibility and a
  distinct generation;
- overlap hooks at completed-edge and destination-segment/source-item
  granularity;
- candidate mechanisms without selecting one in semantics.

Fixed capacity retains padded physical rows but notifications count only valid
logical edges. Empty destination segments consequently begin ready. Returned
payload always has shape `[source item, relation slot, ...]`; a generated Fold
is a separate consumer.

The CPU interpreter performs only gather/scatter. Mutation tests change route
weights without changing transport, vary relation arity, cover empty
destinations, and exercise primal and cotangent traffic.

## Negative results and risks

- Public DeepEP combine is not transport-only. Using it would recreate the
  synthesis-boundary violation the clean milestone is intended to remove.
- Coalescing several edges from one source item to one destination rank is
  safe for outbound source payload, but it is not a general return
  representation: destination computation and edge cotangents require exact
  edge identity.
- A generic all-to-all can realize this ABI but may serialize communication and
  computation. It is a correctness backend, not the intended parity endpoint.
- A DeepEP transport-only inverse API is not exposed by the audited public API.
  Achieving parity may require a small lower-level DeepEP adapter or a
  MoK-inspired symmetric-memory push backend.
- The reference EventTensor currently marks readiness per completed physical
  row. A GPU backend will likely coarsen notifications to tiles or
  macrobatches. That must remain a schedule quotient that covers the exact
  dependence, not a change to relation semantics.

## Ranked next experiments

1. Replace the three JAX payload-boundary descriptors in the distributed
   training plan with this four-leg Shuttle ABI while leaving the current JAX
   collective execution underneath. This checks frontend/AD ownership before
   changing the GPU transport.
2. Implement a typed-FFI exact-edge symmetric-memory backend: receiver-pull for
   dispatch and direct-push to source-slot buffers for return. Lower the
   EventTensor system-visibility contract to the backend counters/events.
3. Add a DeepEP V2 outbound candidate using the coalesced metadata projection,
   followed by generic expansion into exact destination edges. Do not use
   public combine.
4. Search event granularity, communication worker count, macrobatch depth, and
   buffer generations. Compare exact-edge and coalesced outbound traffic under
   identical semantic boundaries.
5. Replay forward and backward on GB200/B200 against the pinned MoK oracle,
   including router, metadata construction, dispatch, computation, return,
   generated merge, and readiness overhead.

## Source ledger

Accessed 2026-08-09.

- DeepEP pinned README/API:
  <https://github.com/deepseek-ai/DeepEP/blob/7febc6e25660af0f54d95dd781ecdcd62265ecca/README.md>
- DeepEP V2 README/API:
  <https://github.com/deepseek-ai/DeepEP/blob/01dc3aaac82068020353dce2c302e38153c0bfaa/README.md>
- Shuttle pinned DeepEP wrapper:
  `lib/levanter/src/levanter/kernels/deepep/transport_ffi.py`
- MoK scheduler:
  <https://github.com/cursor/mixture-of-kittens/blob/3e1cf43ab93ad040afed52a45ab03cb490ffe4be/csrc/scheduler.cuh>
- MoK megakernel transport and epilogues:
  <https://github.com/cursor/mixture-of-kittens/blob/3e1cf43ab93ad040afed52a45ab03cb490ffe4be/csrc/mok_megakernel.cuh>
- MoK Python workspace and schedule construction:
  <https://github.com/cursor/mixture-of-kittens/blob/3e1cf43ab93ad040afed52a45ab03cb490ffe4be/mok/functional.py>
- Event Tensor paper, used for the readiness/dataflow boundary:
  <https://arxiv.org/abs/2604.13327>
- Shuttle relation/index plane: `lib/tile_lifetime/src/tile_lifetime/relation.py`
- Shuttle readiness algebra: `lib/tile_lifetime/src/tile_lifetime/event_dataflow.py`
- Shuttle relation/collective adapters:
  `lib/tile_lifetime/src/tile_lifetime/event_dataflow_adapters.py`
- Existing JAX distributed training boundary:
  `lib/tile_lifetime/src/tile_lifetime/distributed_expert_jax_module.py`
- Existing clean distributed benchmark:
  `lib/tile_lifetime/benchmarks/backends/gb200_deepep_mok_distributed.py`
