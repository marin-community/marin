# Shuttle Event Tensor Algebra and Prototype Plan

## Goal

Add an Event Tensor abstraction to Shuttle that represents the imperative/readiness realization of a tiled data-dependence graph.

The key design principle is:

Event Tensors are not semantic tensor operations. They are derived schedule objects.

Map, Contract, Fold, Scan, Relation, RelationPlan, SegmentedContract, DomainRestriction, and Transport describe what computation means.

Event Tensors describe when tiled pieces of that computation are ready to execute after Shuttle has chosen a decomposition.

The implementation should make it possible to derive event synchronization automatically from ordinary Shuttle programs rather than requiring workload-specific event annotations.

This work is exploratory and should not block current Grug training integration.

---

## 1. Conceptual model

After semantic lowering and schedule decomposition, suppose Shuttle has:

- a family of producer tasks \(P\);
- a family of consumer tasks \(C\);
- a required data-dependence relation \(D \subseteq P \times C\).

An Event Tensor introduces an intermediate readiness domain \(E\):

\[
P \xrightarrow{N} E \xrightarrow{T} C.
\]

where:

- \(N\) is the notify relation from producer tasks to event coordinates;
- \(T\) is the trigger/wait relation from event coordinates to consumer tasks;
- each event coordinate has an initial count equal to the number of required producer completions represented by that event.

An event coordinate becomes ready when all of its required notifications have arrived.

The simplest case is:

```text
partial[i, j]
    │
    │ notify_relation: (i,j) -> i
    ▼
event[i], initial_count = number of j
    │
    │ trigger_relation: i -> i
    ▼
finalize[i]
```

The Event Tensor is therefore a factorization of task dataflow, not an independently meaningful tensor computation.

---

## 2. Placement in the Shuttle lowering stack

The intended pipeline is:

```text
ordinary tensor program
    ↓
Map / Contract / Fold / Scan / Relation / ...
    ↓
semantic rewrites
    ↓
materialization + decomposition choices
    ↓
TaskFamily graph
    ↓
exact task-dependence relations
    ↓
EventTensorPlan synthesis
    ↓
event coarsening / scheduling choices
    ↓
static, dynamic, or hybrid task schedule
    ↓
physical synchronization primitives
```

Do not add `EventTensorPrimitive` to `TensorProgram` alongside `ContractPrimitive` or `FoldPrimitive`.

Prefer a schedule-level module, tentatively:

```text
tile_lifetime.event_dataflow
```

or another name consistent with the existing schedule IR.

---

## 3. Core IR

Introduce generic schedule-level concepts roughly equivalent to:

```python
@dataclass(frozen=True)
class TaskFamily:
    name: str
    axes: tuple[TaskAxis, ...]
    placement: Placement | None
    # Existing/appropriate Shuttle task metadata.


@dataclass(frozen=True)
class TaskRelation:
    source: TaskFamily
    target: TaskFamily
    # Symbolic/index relation mapping source coordinates to target coordinates.
    relation: IndexRelation


@dataclass(frozen=True)
class EventDomain:
    name: str
    axes: tuple[TaskAxis, ...]


@dataclass(frozen=True)
class EventTensorPlan:
    domain: EventDomain
    # Which producer completions contribute to each event.
    notify_relation: TaskRelation
    # Which consumers become eligible after each event is ready.
    trigger_relation: TaskRelation
    # Usually derived as notify indegree, but represented explicitly so
    # dynamic/runtime relations are possible.
    initial_count: EventCountExpression
    memory_scope: EventMemoryScope
    generation_policy: EventGenerationPolicy
    visibility: MemoryVisibility
    # Scheduling/lowering preferences, not semantics.
    scheduling_mode: EventSchedulingMode | None = None
```

Names are provisional. Fit them to the existing IR rather than creating parallel abstractions unnecessarily.

The important semantic contents are:

- event domain;
- producer -> event relation;
- event -> consumer relation;
- initial readiness count;
- memory visibility requirement;
- generation/epoch identity.

---

## 4. Exact dependence relation is fundamental

Do not make Event Tensors the source of truth for dependencies.

The compiler should first derive the exact required relation:

\[
D_{\text{required}} \subseteq P \times C.
\]

The Event Tensor plan is one implementation of that relation.

This distinction is important because eventization itself is an optimization/search choice.

For example:

```text
A0 A1 A2 A3 -> B0
A4 A5 A6 A7 -> B1
```

may use two independent events:

```text
E0(count=4) -> B0
E1(count=4) -> B1
```

or one coarser event:

```text
E(count=8) -> B0,B1
```

The latter adds false dependencies but remains correct.

Represent this explicitly as:

\[
D_{\text{required}} \subseteq D_{\text{scheduled}}.
\]

A legal schedule may add dependencies but may never omit a required dependency.

This gives Shuttle a future optimization dimension:

event granularity versus synchronization overhead versus exposed concurrency.

The initial prototype does not need to search this space aggressively, but the representation must not make event granularity synonymous with semantics.

---

## 5. Readiness semantics

For event coordinate \(e\):

```text
remaining[e] = initial_count[e]
```

Each completion related through `notify_relation` contributes exactly one logical notification:

```text
notify(e):
    remaining[e] -= 1
```

The event is ready iff:

```text
remaining[e] == 0
```

Consumers attached through `trigger_relation` may execute only once the event is ready.

The compiler/runtime must reject:

- negative event counts;
- duplicate logical notifications unless explicitly represented by the relation;
- consumers that can execute without all required dependencies;
- cyclic schedules that cannot establish an initial runnable frontier.

Dynamic relations may determine counts at runtime.

For example, MoE expert readiness can derive its initial count from the runtime `RelationPlan` indegree.

---

## 6. Event tensors are virtual

Do not assume that an Event Tensor lowers to a global integer tensor.

An `EventTensorPlan` represents logical readiness.

Physical lowering may erase the Event Tensor entirely or realize it through different mechanisms:

```text
same sequential task
    -> program order / no synchronization
same warp
    -> implicit warp ordering where legal
same CTA
    -> barrier / mbarrier / shared state
producer and consumer warp groups
    -> shared-memory mbarrier
cluster
    -> cluster-visible synchronization
device-global task graph
    -> semaphore / atomic counter
dynamic persistent scheduler
    -> counter reaching zero + runnable queue
separate kernels
    -> stream dependency / programmatic dependent launch
remote placement
    -> transport completion + remote visibility event
```

The physical synchronization mechanism should be a lowering/search choice.

Do not embed CUDA-specific barrier concepts into the logical Event Tensor algebra.

---

## 7. Memory visibility

Readiness alone is insufficient.

A producer becoming “complete” must establish the required visibility of the values consumed downstream.

Represent a memory contract on the event edge, approximately:

```python
MemoryVisibility {
    scope
    release_on_notify
    acquire_before_consumer
}
```

The initial implementation may support only a small number of scopes, but the logical model should distinguish:

- CTA;
- cluster;
- device;
- system / transport.

A compiler verification pass should ensure that the selected physical synchronization primitive satisfies the required visibility.

Avoid implementing an unnecessarily elaborate C++ memory model. The purpose is to avoid treating a counter transition as sufficient when the produced data is not yet guaranteed visible.

---

## 8. Generations / epochs

Persistent schedules reuse event storage.

Therefore event identity should conceptually include a generation:

```text
(event_coordinate, generation)
```

A notification from generation \(g+1\) must not accidentally satisfy a consumer waiting on generation \(g\).

Provide an explicit `EventGenerationPolicy`, even if the first prototype simply allocates/reset events between invocations.

This should make future persistent/streaming schedules safe without changing Event Tensor semantics.

---

## 9. RelationPlan interaction

`RelationPlan` and `EventTensorPlan` must remain separate concepts.

Use:

```text
RelationPlan = who is connected to whom.

EventTensorPlan = which connected work is ready yet.
```

For dynamic MoE:

```text
tokens
  ↓ top-k
Relation
  ↓
RelationPlan(token, expert)
  ↓
group / dispatch tasks
  ↓
EventTensorPlan(expert)
  ↓
SegmentedContract expert tasks
```

The event initial count may be derived from:

```text
indegree(RelationPlan, expert)
```

and the trigger relation may use the existing expert offsets/indptr to identify which expert tiles become runnable.

Do not introduce an MoE-specific event object.

---

## 10. Fold interaction

Fold is an especially important source of Event Tensor structure.

If Shuttle decomposes:

```text
Fold(input, reduction_axis)
```

into partial producers and a finalizer, then the decomposition automatically induces a dependence relation.

Example:

```text
PartialFold[row, partition]
        ↓
Event[row]
        ↓
FoldFinalize[row]
```

The event count is the number of valid partials for that row.

This should work unchanged for:

- split-K reduction;
- normalized-exponential partial states;
- deterministic routed merge;
- gradient reductions;
- arbitrary mergeable Fold state.

The Event Tensor layer must not know that the Fold is “attention” or “MoE”.

---

## 11. Static versus dynamic scheduling

The Event Tensor semantics should not decide how workers are assigned.

Support at least two conceptual lowerings:

### Static

Tasks have predetermined worker/order assignments.

Consumers wait until their event becomes ready.

```text
wait(event)
execute task
notify(children)
```

### Dynamic

Tasks whose dependencies reach zero become runnable.

```text
notify(event)
if event became ready:
    enqueue(triggered_tasks)
```

The same `EventTensorPlan` should admit both lowerings where legal.

The initial prototype only needs a reference implementation of each; high-performance persistent scheduling is explicitly out of scope.

Eventually this becomes another search choice:

```text
static schedule
dynamic schedule
hybrid schedule
```

---

## 12. Required prototype cases

Implement and test Event Tensor derivation on exactly three initial cases.

### A. Split reduction / Fold

Construct a generic Fold decomposition with several partial producers per output.

Derive:

```text
partial tasks
→ EventTensor
→ finalizer tasks
```

Requirements:

- no workload names;
- exact counts derived mechanically;
- arbitrary reduction partition count;
- test event coarsening adds only legal false dependencies;
- execute a CPU/reference event interpreter and match direct Fold semantics.

### B. Runtime MoE relation

Start from a generic runtime `RelationPlan`.

Derive readiness for expert/segment work based on runtime relation indegrees.

Requirements:

- varying expert occupancy;
- empty experts;
- uneven counts;
- arbitrary route slots;
- deterministic behavior;
- no MoE-specific branch in Event Tensor construction.

The test should mutate the `RelationPlan` and observe the `EventTensorPlan`/counts change without changing event-generation code.

### C. Producer -> reduction/communication-style tiled graph

Use either:

- tiled GEMM -> ReduceScatter-like consumer structure; or
- another existing Shuttle tiled producer/consumer graph with multiple producers and nontrivial consumers.

The purpose is to ensure the abstraction is not secretly specialized to Fold or dynamic routing.

Prefer an existing Shuttle task representation rather than inventing a fake workload if one is readily available.

---

## 13. Reference interpreter

Build a tiny backend-neutral Event Tensor interpreter for testing.

It should:

1. instantiate event counts;
2. maintain runnable tasks;
3. execute tasks in a deterministic reference order;
4. apply notifications;
5. trigger consumers;
6. detect deadlock;
7. record an execution trace.

This interpreter is for semantic validation, not performance.

Useful invariants:

- every executed consumer was ready;
- every required producer executed first;
- every logical notify occurred exactly once;
- every event reached zero exactly once;
- no event became negative;
- all reachable tasks completed.

Allow randomized legal task ordering in tests to validate schedule independence where the computation’s numerical contract permits it.

---

## 14. Event synthesis verifier

Add a verification pass checking:

- required dependencies are covered;
- no dependency is removed;
- event counts equal represented producer indegrees;
- event/trigger domains are consistent;
- generation use is consistent;
- memory scope is sufficient;
- no impossible cycle/deadlock exists, where statically provable.

It is acceptable for the first deadlock analysis to be conservative.

Do not build a general distributed deadlock theorem prover.

---

## 15. Search-space hooks

Represent, but do not yet deeply optimize:

- event granularity;
- event-domain coarsening;
- static vs dynamic scheduling;
- event storage scope;
- worker ownership;
- queue/no-queue;
- physical synchronization primitive.

Candidate transformations should be expressible independently from workload semantics.

A future cost model can consider:

- number of event operations;
- number of atomic operations;
- false-dependency critical-path cost;
- event storage;
- queue overhead;
- available concurrency;
- producer/consumer placement.

For now, provide deterministic heuristic choices plus enough metadata to enumerate alternatives later.

---

## 16. Relationship to current TaskFamily / Buffer / Event work

Before adding new structures, inspect the existing Shuttle schedule/runtime IR for concepts corresponding to:

- `TaskFamily`;
- `Buffer`;
- `Event`;
- readiness/dependencies.

Reuse or extend existing representations where possible.

Do not create a second parallel scheduling system.

The desired conceptual change is that dependencies become symbolic tensor/index relations over task families, from which imperative readiness state can be derived.

If the existing `Event` concept is scalar, consider making `EventTensorPlan` a symbolic family of those events rather than replacing it.

---

## 17. Non-goals

Do not:

- integrate Event Tensors into `TensorProgram` semantic algebra;
- add workload names such as attention/MoE/GDN to event lowering;
- reproduce the Event Tensor Compiler runtime wholesale;
- implement a full persistent-kernel scheduler;
- build cross-node/distributed event transport;
- interfere with current XLA typed-FFI Grug integration;
- optimize production GPU event scheduling yet;
- introduce a new MLIR dialect;
- require Event Tensors for existing accepted Shuttle paths.

This phase is about validating the algebra and lowering boundary.

---

## 18. Deliverables

Produce:

1. a short design document describing the exact data model and invariants;
2. generic schedule-level Event Tensor IR;
3. mechanical `TaskDependence -> EventTensorPlan` derivation;
4. reference interpreter;
5. verifier;
6. split-Fold example/tests;
7. runtime-`RelationPlan` example/tests;
8. third non-Fold/non-MoE tiled-dataflow example;
9. at least one event-coarsening transformation;
10. static and dynamic reference scheduling policies;
11. mutation tests demonstrating workload-independent generation;
12. a follow-up document identifying where an SM100 lowering could attach without implementing it yet.

---

## 19. Acceptance criteria

Call the prototype successful if all of the following hold.

### Generality

The same Event Tensor construction code handles all three prototype families.

No workload-specific dispatch keys affect construction.

### Derivation

Events are mechanically derived from a tiled task dependence relation. Tests do not manually specify the expected runtime event wiring except as an oracle.

### Correctness

The reference interpreter reproduces the semantic/reference outputs.

Dependency-verification invariants pass.

Dynamic `RelationPlan` mutations correctly alter event counts and triggers.

### Coarsening

At least one finer and one coarser legal eventization of the same dependency graph execute correctly, demonstrating that event granularity is a schedule choice rather than semantics.

### Imperative correspondence

The plan can be lowered conceptually to:

```text
initialize count
wait/trigger
notify/decrement
```

without introducing workload-specific information.

### Clean boundary

The resulting architecture maintains:

```text
semantic algebra
    ↓
task decomposition
    ↓
dependency algebra
    ↓
EventTensorPlan
    ↓
imperative schedule/runtime
```

rather than allowing runtime scheduling concepts to leak upward into tensor semantics.

---

## 20. Questions to answer in the report

At completion, explicitly answer:

1. Is an Event Tensor genuinely necessary as a first-class IR object, or can it remain a derived view over symbolic task-dependence relations?
2. Is producer -> event -> consumer factorization sufficient for the Shuttle schedules currently implemented?
3. What additional structure, if any, is required for pipeline stages, circular buffers, or repeated persistent schedules?
4. Can `RelationPlan` and `EventTensorPlan` share the same underlying symbolic relation representation?
5. Can event coarsening be expressed simply as relation/domain quotienting or projection?
6. What information would an SM100 emitter require to choose between mbarrier, semaphore, PDL, queue-triggering, and no explicit event?
7. Does the abstraction expose any useful optimization not naturally expressible in the current Shuttle scheduler?
8. Are there places where current handwritten readiness logic can already be replaced by this generic derivation?

---

## 21. Recommended implementation order

Keep this bounded.

```text
inspect existing scheduling IR
    ↓
formalize TaskRelation
    ↓
derive exact dependence graph
    ↓
EventTensorPlan + verifier
    ↓
reference interpreter
    ↓
split-Fold
    ↓
RelationPlan / segmented work
    ↓
third generic example
    ↓
event coarsening
    ↓
static vs dynamic reference lowering
    ↓
write findings; stop
```

Do not proceed to high-performance GPU implementation unless the abstraction survives these tests cleanly.

The expected conceptual result is:

Event Tensors are Shuttle’s imperative normal form for scheduled tensor dataflow: derived from generic task-dependence relations after semantic decomposition, and subsequently lowered into the weakest physical synchronization sufficient for the chosen schedule.

A stronger version, if supported by the prototype, is:

The exact data-dependence relation is semantic to the schedule; its factorization into event coordinates, granularity, storage, and static/dynamic execution policy are optimization choices.
