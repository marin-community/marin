# Iris resource model contract

**Status:** Implemented contract for the resource-model pull request
**Parent:** [#7560](https://github.com/marin-community/marin/issues/7560)

This document records the contract implemented in this branch. The Python
records and protobuf definitions remain authoritative for individual fields.

## Scope

Iris exposes Job, Task, Attempt, Node, Slice, and Endpoint resources. Backend is
placement and source metadata. Worker remains an implementation term for the
RPC execution daemon and its persisted registration.

The public operations in this change are:

| Resource | Reads | Mutations |
| --- | --- | --- |
| Job | list, describe, spec, logs, activity | submit, cancel |
| Task | list, describe, logs, activity | retry |
| Attempt | describe, logs, activity, exec, profile | terminate |
| Node | list, describe | none |
| Slice | list, describe | none |
| Endpoint | list, describe | mint |

`activity` is the public history term. Manual Slice deletion and a generic
polymorphic resource API are outside this change.

## Resource records

`iris.cluster.resources` contains frozen records for public requests and
responses. These records are the in-process Job and Task model used by the
controller, client conversion, CLI, dashboard RPC, and federation admission.
Mutable protobuf messages are decoded at a process boundary and are not passed
through controller business logic.

A `ResourceKey` identifies one logical resource by authority cluster, kind, and
human-readable ID. Job and Task identities add an incarnation UID. Attempt
identity adds an Attempt number and UID. Node and Slice identities add backend
and provider incarnation coordinates. Mutating or live-provider operations use
the exact identity; a stale incarnation fails before a backend call.

List methods return `Page[T]`. Page tokens are bound to the normalized query.
Federated or provider-backed pages also carry bounded `ResourceSourceStatus`
records. A failed source is reported as unavailable while healthy source rows
remain visible.

## Reads

`Controller` owns resource reads over one controller snapshot and the
configured backend set.

- Job and Task reads use the active controller tables and typed records.
- Attempt detail separates lifecycle facts in `AttemptSummary` from provider
  coordinates in `AttemptRuntimeObject`.
- Node reads merge persisted RPC Worker registrations with provider observations.
- Slice reads project existing autoscaler and provider state.
- Endpoint reads retain the existing Endpoint service and authorization rules.
- Logs and activity authorize the resource first, then query finelog or durable
  action receipts.

`AttemptRuntimeObject` is a read-model record in this change. It is assembled
from the current Attempt storage and provider observations. This pull request
does not add an `attempt_runtime_objects` table or a second runtime persistence
path.

## Actions

Job cancel, Task retry, and Attempt terminate return an `ActionReceipt`.
Acceptance exact-compares the target and writes the workload transition and
receipt in the controller's normal transaction. The receipt stores the
principal, action kind, idempotency key, payload hash, exact target, result, and
timestamps.

Repeating the same principal, action kind, and idempotency key with the same
payload returns the existing receipt. Reusing the key with a different payload
fails. A stale Job, Task, or Attempt UID fails before a provider action.

The action meanings are:

- Job cancel applies the existing cancellation behavior and federation cancel
  path.
- Task retry preempts the exact current Attempt and uses normal retry policy. It
  does not reopen an arbitrary terminal Task.
- Attempt terminate targets the exact current Attempt and disables automatic
  retry for that operation.

Provider cleanup remains part of normal reconciliation. Activity reads merge
receipts with the existing task event stream; no second event store is added.

## Internal ownership

The implementation uses these boundaries:

```text
iris/cluster/resources/                    frozen public records
iris/cluster/controller/
  controller.py                            canonical resource application API
  runtime.py                               process lifecycle and control loop
  jobs.py                                  resource-native Job admission
  api/
    resource_service.py                    canonical ResourceService adapter
  legacy/
    controller_service.py                  old ControllerService adapter
    codec.py                               old request/response translation
  persistence/
    database.py                            engine and transaction lifecycle
    schema.py                              active SQLAlchemy schema
    reads.py                               shared typed reads
    writes.py                              shared transactional writes
    json_codec.py                          persisted native-record JSON
    federation.py                          FederationStore DB implementation
    attempt_counts.py                      SQL attempt-count expressions
    action.py                              action receipt persistence
    migrations/                            ordered schema deltas
    projections/                           write-through/read projections
iris/cluster/controller/backend.py         existing TaskBackend contract
```

The concrete names may be split further by noun, but ownership does not move:
generated Job and Task protobuf values are confined to API, legacy, worker,
federation, and provider transport adapters; SQLAlchemy and controller table
definitions are confined to `controller/persistence/`. Historical
controller/VM operational status return types on `TaskBackend` remain explicit
tracked debt rather than part of the resource model.

The change does not introduce a second backend protocol or resolver package.
Existing RPC and Kubernetes backends continue to implement `TaskBackend`.
Resource Node and Slice views adapt their existing registrations and provider
observations without moving backend scheduling or autoscaling ownership.

There is no universal Resource base class, repository class, generic CRUD
layer, or table containing every resource kind.

## Wire boundaries

`iris/rpc/resource.proto` is the resource RPC surface. Client conversion turns
its mutable, presence-sensitive protobuf messages into frozen Python records.
The dashboard consumes the resource RPC responses rather than reconstructing
resource state.

The old Job and Task methods on `ControllerService` are a one-way network
boundary, not a supported second product version. `ResourceService` is the
default and only first-party resource API. `controller/legacy/` translates old
requests and responses immediately and delegates to the same `Controller` used
by `controller/api/`. It does not query controller tables or contact a backend
to implement resource behavior.

Federation retains its authenticated peer and handoff protocol. Received Jobs
are decoded into resource `JobSpec` records before admission. This change does
not add a second federation transport or require simultaneous fleet cutover.

## Persistence and migration

The controller keeps one active schema in `controller/persistence/schema.py`
and one ordered migration chain in `controller/persistence/migrations/`.
Existing Job, Task,
Attempt, Worker, Endpoint, federation, and autoscaler tables remain in place.

`0051_action_receipts.py` is the only new physical schema migration. It
additively creates `action_receipts` and its indexes. It does not copy or rewrite
existing noun rows. A fresh database materializes the current SQLAlchemy schema
and records the migration as applied. A pre-0051 database executes the
idempotent migration, retains its existing Job and Task rows, and can create and
reopen action receipts.

This release has no constructor-selected schema, schema fingerprint parser,
repair-on-open pass, dual writes, or one-shot migration framework. The empty
attached auth database remains because historical migrations still reference
it. Removing that file and rolling up migration history is separate work.

Release uses the normal controller checkpoint and ordered migration path.
Ordinary defects roll forward. Disaster recovery may restore the checkpoint
and image selected by the existing operator procedure; this resource change
does not add a rollout state machine.

## Test contract

Deterministic journeys cover behavior crossing persistence, scheduling,
reconciliation, restart, and federation boundaries. Resource journeys cover:

- global Job and Task reads;
- exact Attempt history and replacement;
- partial backend status and recovery;
- Job cancellation, Task retry, and Attempt termination;
- duplicate action requests across controller restart;
- scheduling and execution deadlines across controller restart;
- federation handoff, outage, cached-read status, and redelivery;
- stale Attempt rejection before an execution-provider call; and
- Endpoint and activity/log reads through public boundaries.

Narrow tests remain for migration ordering, fresh and upgraded database
behavior, query budgets, protobuf conversion, authorization, and provider
adapters. Tests assert public records, persisted effects, or external fake
observations. They do not pin private helper names or dispatch order.

## Out of scope

- Physical schema or migration-history rollup.
- An Attempt runtime sidecar table.
- Splitting or replacing the existing backend contract.
- Replacing the scheduler, Kueue, autoscaler, finelog, or federation transport.
- Old-image dual writes, repair-on-open, CLI aliases, or internal compatibility
  models.
- Persisting Kubernetes Node heartbeats as synthetic RPC Worker rows.
- Time-series utilization in resource records; measurements remain in finelog.
- A generic repository interface or dependency-injection framework. The
  persistence package is the concrete SQLite implementation.
- The future client-level abstraction sweep beyond the already-native resource
  client surface.
