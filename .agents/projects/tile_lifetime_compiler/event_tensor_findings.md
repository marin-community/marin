# Event Tensor prototype findings

## Result

The bounded CPU prototype supports the proposed lowering boundary:

```text
semantic algebra
→ task decomposition
→ exact indexed dependence
→ EventTensorPlan factorization
→ static or dynamic imperative policy
```

One derivation, verifier, and interpreter handles split Fold, runtime relation segmentation, and a tiled Contract-to-placement-change graph. Event coarsening is a schedule projection and produces a verified superset of the exact dependency relation.

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
- Memory visibility is verified by scope and release/acquire requirements but is not yet legalized to an instruction sequence.
- No official ETC code was available for implementation-level comparison.

These limits support stopping here. The algebra survived the intended tests without requiring a GPU runtime or workload-specific event construction.
