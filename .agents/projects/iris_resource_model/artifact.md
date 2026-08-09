# Iris resource model: implemented shape

The package and protobuf boundaries are mapped in
[architecture.md](architecture.md), including the current checkpoint and the
target that confines generated messages to their owning transports.

## Decision

Land the resource model in one forward-only pull request, using the original
design stages as review groups. Resource records are the sole in-process Job and
Task contract. `ResourceService` is the only first-party resource API. The
retained `ControllerService` Job and Task protobufs form a one-way old-wire
adapter that decodes into resources at the boundary.

This branch does not carry forward the abandoned rollout machinery: no dual
writes, repair-on-open, schema selector, parity telemetry, old-image reader, or
rollout ancestry protocol.

## Operator surface

Iris exposes Job, Task, Attempt, Node, Slice, and Endpoint resources. Backend is
placement and source metadata. Worker remains an implementation name for the
RPC execution daemon.

```text
submit Job
  -> inspect global Tasks
  -> inspect one exact Attempt
  -> read logs and activity
  -> cancel Job | retry Task | terminate Attempt
```

The typed Python client, CLI, dashboard, operational callers, resource RPC,
federation admission, and controller use the same resource records. Global
provider-backed reads include source status, so unavailable data is
distinguishable from an empty result.

## Durable actions

Job cancel, Task retry, and Attempt terminate write an action receipt with the
workload transition. The action exact-compares the requested incarnation before
mutation. A duplicate principal, action kind, and idempotency key returns the
same receipt; a different payload conflicts.

- Job cancel reuses controller cancellation and federation redrive.
- Task retry preempts the exact current Attempt under normal retry policy.
- Attempt terminate targets the exact current Attempt and prevents retry for
  that operation.

Activity joins receipts with the existing task event stream. Provider cleanup
continues through normal reconciliation.

## Production ownership

```text
iris/cluster/resources/                    frozen public records
iris/cluster/controller/resources/jobs.py  typed Job admission
iris/cluster/controller/resources/facade.py
                                            reads and actions
iris/cluster/controller/resources/rpc.py   resource protobuf translation
iris/cluster/controller/resources/legacy_rpc.py
                                            old RPC boundary translation
iris/cluster/controller/persistence/action.py
                                            action receipt persistence
iris/cluster/controller/schema.py          active schema
iris/cluster/controller/reads.py           existing shared reads
iris/cluster/controller/writes.py          existing shared writes
iris/cluster/controller/backend.py         existing backend contract
```

The branch deletes the unused alternative backend protocol/resolver and the
parallel persistence schema/migration package. It keeps the active controller
schema and existing backend implementations. Node and Slice views project
current Worker, autoscaler, and provider observations.

`AttemptRuntimeObject` separates provider coordinates in the public read model
from `AttemptSummary` lifecycle facts. It is assembled from current storage and
observations; this branch does not add an `attempt_runtime_objects` table.

## Persistence

`0051_action_receipts.py` is an ordinary additive controller migration. It
creates `action_receipts` and three indexes. Existing resource rows are neither
copied nor rewritten.

Fresh databases create the current `schema.py` metadata and mark all deltas
applied. Existing databases execute pending deltas in lexical order. The
pre-0051 behavior proof starts with populated Job and Task rows, removes the
0051 table and ledger entry, reopens the controller, reads the same resources,
creates a receipt, and verifies the duplicate request after another reopen.

The empty auth database remains attached because historical migrations refer to
it. Removing it belongs in a later migration rollup. This PR does not introduce
a parallel schema catalog or one-shot upgrade framework.

## Release

Controllers use their current ordered-migration startup and checkpoint
procedure. Federation messages are translated to resource records at the
receiving boundary, so controllers can canary independently. Ordinary defects
roll forward. Disaster restore retains the existing possibility of losing
writes accepted after the selected checkpoint.

## Evidence expected before merge

- Resource journeys cover exact replacement, timeout continuity, partial
  outage, restart, federation, actions, activity, and endpoints.
- A populated pre-0051 upgrade preserves Job and Task reads and action
  idempotency.
- Resource pages have fixed query and bind budgets; retired unpaged adapters
  consume them in bounded batches.
- CLI, resource RPC, legacy RPC adapter, Python conversion, and dashboard
  behavior are covered at their public boundaries.
- Full safe Iris tests, Pyrefly, precommit, dashboard typecheck, and deterministic
  replay are green.

## Deferred work

- Auth database and historical migration rollup.
- Attempt runtime sidecar persistence.
- Further backend or Kubernetes adapter decomposition.
- Slice deletion and time-series Node utilization.
