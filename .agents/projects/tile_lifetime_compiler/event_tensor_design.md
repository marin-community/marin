# Shuttle Event Tensor prototype design

## TL;DR

The exact indexed task dependence is the source of truth. `EventTensorPlan` is a schedule-level factorization of that relation into producer notifications and consumer triggers. It is derived after semantic decomposition, may legally add false dependencies through coarsening, and lowers to static waits or dynamic queue triggers without entering `TensorProgram`.

## Data model

`event_dataflow.py` adds five layers:

1. `TaskAxis` and `TaskFamily` define a concrete rectangular task grid.
2. `TaskRelation` records a finite coordinate relation between grids.
3. `TaskDependence` marks one task-to-task relation as exact and adds its visibility contract.
4. `EventFactorization` maps each consumer coordinate to one virtual `EventDomain` coordinate.
5. `EventTensorPlan` derives producer-to-event notifications, event-to-consumer triggers, and event counts.

For an exact relation (D \subseteq P \times C), event factorization chooses a map (q: C \rightarrow E). The derivation is:

```text
notify(p, e) iff exists c: D(p, c) and q(c) = e
trigger(e, c) iff q(c) = e
initial_count(e) = |{p: notify(p, e)}|
```

The scheduled relation is the composition `notify ; trigger`. The verifier requires:

```text
D_required ⊆ D_scheduled
```

The default identity factorization has one event per consumer and introduces no false edges. `coarsen_event_tensor_plan` applies a projection/quotient to the consumer coordinates. Producers that were required by any consumer in a coarse group notify that group's event once. Every consumer in the group consequently waits for the union of producers.

## Invariants

- A task relation contains no duplicate logical edge.
- Every relation coordinate lies in its source or target domain.
- Every consumer maps to exactly one event for each exact dependence.
- Initial counts equal notify-relation indegrees.
- The composed scheduled relation covers every required edge.
- Event scope is at least as wide as the dependence's visibility scope.
- Required release-on-notify and acquire-before-consumer contracts survive planning.
- Program-level event plans cover every exact dependence exactly once.
- The concrete scheduled task graph is acyclic before reference execution.
- A task executes at most once, each notify decrements a positive count, and each event reaches zero once.
- Event identity includes the execution generation.

## Interpreter and imperative lowering

`execute_event_dataflow` instantiates counts, marks zero-count events ready, schedules source tasks, applies notifications, and enables consumers only after all their incoming events are ready. It records a stable trace and rejects deadlock, duplicate/excess notification, negative counts, or incomplete execution.

The static reference policy chooses the first ready task in a fixed task-grid order. The dynamic policy uses readiness-triggered FIFO enqueueing. A seeded random legal order is available for schedule-independence tests. These policies validate semantics; they are not performance models.

`lower_event_tensor_plan` exposes the backend-neutral imperative form:

```text
static:  initialize + wait + notify
dynamic: initialize + notify + trigger_enqueue
```

It does not select `mbarrier`, atomic semaphore, task queue, PDL, or a kernel boundary.

## Interaction with existing Shuttle IR

The prototype does not change `TensorProgram` or semantic `Map`/`Contract`/`Fold`/`Scan`/`Relation` records.

Existing `ReadinessEvent` is a useful scalar diagnostic and physical-plan record, but it lacks indexed relations and cannot verify derived counts. Existing `TileFlowEdge` identifies tiled values, consumers, layouts, and readiness granularity but not the exact coordinate relation. A future integration should derive `TaskDependence` while lowering tile flows, then optionally render an `EventTensorPlan` as scalar `ReadinessEvent` records for legacy plan dumps.

The prototype deliberately uses a small `TaskFamily` because no backend-neutral indexed task-grid type exists. It should become the shared task coordinate vocabulary if subsequent experiments need it; it should not coexist indefinitely with another generic task IR.

## Prototype families

- Split Fold: partial task `[row, partition]` to finalizer `[row]`.
- Runtime relation: valid relation-edge task `[edge]` to destination segment `[destination]`; counts come from the runtime `RelationPlan` occupancy, including zero-count destinations.
- Tiled Contract to placement change: producer `[output_tile, destination, partial]` to communication-style consumer `[output_tile, destination]` with system-scope visibility.

All three pass through `derive_event_tensor_plan`; there is no workload dispatch in event construction.
