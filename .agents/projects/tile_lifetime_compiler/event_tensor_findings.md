# Event Tensor prototype findings

## Result

The CPU interpreter and bounded H100 prototype support the proposed lowering boundary:

```text
semantic algebra
→ task decomposition
→ exact indexed dependence
→ EventTensorPlan factorization
→ static or dynamic imperative policy
```

One derivation, verifier, and interpreter handles split Fold, runtime relation segmentation, and a tiled Contract-to-placement-change graph. Event coarsening is a schedule projection and produces a verified superset of the exact dependency relation.

The H100 proof lowers runtime `RelationPlan` indegrees to counted readiness and
lowers a generic Contract/Fold/Contract task graph to a generation-tagged CTA
pipeline. Runtime relation mutations and phased shape mutations reuse the same
generated bodies. This validates the physical attachment boundary, not a
production MoE or attention kernel.

The follow-up structural adapter now derives QK Contract, normalized-exponential
partial Fold, PV Contract, and finalization task families from an actual generic
`StreamingAttentionProgram`. It computes row-tile and Fold-partition domains
from the program's logical axes and tile schedule. Changing the Fold domain
changes event counts without a workload dispatch. Runtime segment readiness
also consumes the same executable `RelationPlan` type used by the MoE path.

This is not yet an end-to-end attention/MoE Event Tensor integration. The
accurate streaming graph has one task per row tile and K/V Fold partition,
whereas the bounded CUDA validation template currently assumes that its
pipeline-slot count is also the complete Fold-partition count and adds a
conservative finalize-before-next-generation reuse edge. Connecting the exact
graph to circular storage needs a separate buffer-slot assignment and reuse
dependence derivation. Until that exists, the branch proves structural reuse
and a physical attachment point, not replacement of current attention or MoE
readiness logic.

The Torch-free JAX typed-FFI replay also executes on one NVIDIA GB200 at revision
`1a04930ecd`. Runtime relation inputs and all phased Q/K/V inputs remain
parameters in optimized HLO; each executable contains one typed-FFI target and
no constant or copy lines. Runtime primary and mutation are bitwise equal to
their source-ordered references. Phased primary and mutation have maximum
absolute errors `8.9407e-8` and `1.1921e-7`; all four paths are bitwise
deterministic over five repeats. The 30-sample medians are 0.061314 ms and
0.061152 ms for runtime primary/mutation, and 0.169697 ms and 0.146477 ms for
phased primary/mutation.

These GB200 timings measure a scalar physical reference pipeline. The
`StreamingAttentionProgram` adapter is structural and is not physically linked
to the measured payload. This result validates JAX-owned runtime inputs,
EventTensorPlan readiness, generations, and SM100 lowering. It does not validate
tensor-core QK/PV execution or full attention performance.

## Acceptance answers

### 1. Is Event Tensor a necessary first-class IR object?

It is useful as a first-class **schedule plan**, but the durable source of truth should be the exact symbolic task-dependence relation. Event Tensor can remain a derived view until a physical pass needs event-domain shape, counts, scope, generation, or static/dynamic scheduling metadata. It should not be a semantic tensor primitive.

### 2. Is producer → event → consumer sufficient?

It is sufficient for the three current acyclic fan-in/fan-out cases. Multiple incoming dependences are represented by multiple event plans, and a consumer becomes runnable when all associated events are ready. The factorization does not by itself represent resource capacity or backpressure.

### 3. What is needed for pipelines, circular buffers, and persistent schedules?

They require bounded buffer capacity, reuse events tied to last-consumer completion, phase/generation progression, and a proof that blocked consumers do not occupy every worker needed by producers. Transaction byte counts may be needed for TMA. Those belong beside Event Tensor in schedule IR, not inside the event relation.

### 4. Can RelationPlan and EventTensorPlan share relation representation?

Conceptually yes. Both need finite or symbolic index relations and alternative orientations. Today `RelationPlan` stores specialized NumPy arrays while `TaskRelation` stores concrete coordinate pairs. The prototype uses a narrow adapter. A later cleanup should extract a shared indexed-relation interface without replacing the payload/index-plane arrays that make `RelationPlan` executable.

### 5. Is event coarsening relation/domain quotienting?

Yes for this prototype. A projection maps consumer coordinates to a quotient event domain. Notify edges become the union of producer preimages, and composing notify with trigger adds the expected cross-product false dependencies. More complex coarsening may need affine/symbolic projections rather than enumerated coordinate maps.

### 6. What does an SM100 emitter need?

It needs placement, scope, visibility, generation, counts, fan-in/fan-out, buffer address space/reuse, static versus dynamic policy, and whether completion means task or asynchronous transaction completion. With those inputs it can choose event erasure, CTA/cluster `mbarrier`, a scoped semaphore, queue-on-zero, PDL, or a kernel boundary.

### 7. What new optimization does the abstraction expose?

It exposes event granularity explicitly. The compiler can trade counter/atomic/storage overhead against false-dependency critical path without changing semantics. It also makes static versus dynamic scheduling a lowering of one plan rather than a workload-specific rewrite.

### 8. Where can it replace handwritten readiness now?

- Routed attention's `source_degree`, `destination_degree`, and slot-wave arrival tuples can be derived from task relations.
- MoE tile-flow readiness and segment occupancies can derive event counts from `RelationPlan` rather than constructing named events.
- Split row/column Folds and deterministic route merges can derive partial-to-finalizer readiness mechanically.

Replacing those records is not part of this bounded branch because existing accepted paths must not depend on an exploratory abstraction.

## Limits

- Relations are concretely enumerated, not symbolic affine/runtime expressions.
- The interpreter validates legal ordering, not performance or GPU resource occupancy.
- Static scheduling is a deterministic priority policy, not an optimized per-SM queue.
- Dynamic scheduling is a reference FIFO, not a concurrent MPMC queue.
- Deadlock verification proves acyclicity of the concrete scheduled task graph but does not reason about finite worker/resource deadlock.
- CTA memory visibility is legalized through `__threadfence_block` plus atomics in the bounded emitters. Cluster, device, and system scopes remain unimplemented.
- No official ETC code was available for implementation-level comparison.

The earlier Torch extension used to compile the H100 probes is prototype-only.
The current replay uses JAX typed FFI without Torch. The algebra and CTA lowering
survived the intended tests without workload-specific event construction;
high-performance persistent scheduling remains separate work.

## Paper lineage

The implementation used Jin et al., arXiv:2604.13327v2 (2026-04-21), and the
MLSys 2026 slides as design references. Shuttle adopted tensor-shaped event
domains, producer-notify/consumer-trigger relations, runtime-dependent counts,
and the ability to lower one plan with static or dynamic policy.

Shuttle did not copy ETC's explicitly annotated tiled frontend, centralized
dynamic queue, round-robin static scheduler, persistent megakernel runtime, or
workload schedules. Shuttle derives exact task dependences from generic task
decomposition, verifies coverage, makes event granularity a coarsening choice,
and keeps memory visibility and generation policy explicit.
