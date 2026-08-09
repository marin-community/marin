# Iris resource model in one forward-only release

**Status:** Implemented design
**Parent:** [#7560](https://github.com/marin-community/marin/issues/7560)

## Summary

Iris now uses typed resource records as its in-process Job and Task contract.
`Controller` owns public reads and actions, `JobResources` owns typed Job
admission, and boundary adapters translate protobuf messages. The existing
controller schema, ordered migrations, `TaskBackend` contract, scheduler,
autoscaler, and federation transport remain in place.

The original five-stage design still orders review: persistence and exact
identity, unified reads, provider views, durable actions, then old internal-path
removal. These stages land in one pull request. They are not separate production
schema epochs or a compatibility rollout.

## Problem

Iris previously exposed different Job and Task shapes through controller RPC,
the Python client, CLI, dashboard, and federation code. Provider-specific reads
also forced callers to know whether execution used RPC Workers or Kubernetes.
Mutation names such as kick, stop, retry, and terminate did not state one exact
target or return a durable result.

The testing cleanup in PR #7842 established deterministic cross-component
journeys. This change uses those journeys to move the product surface without a
dual runtime model.

## Public model

The public nouns are Job, Task, Attempt, Node, Slice, and Endpoint. Backend is a
placement and source coordinate. Worker remains the RPC daemon's internal name.

The common path is:

```text
submit Job -> inspect Tasks -> inspect exact Attempt -> read logs/activity
           -> cancel Job | retry Task | terminate Attempt
```

Resource identities separate a logical key from an exact incarnation UID. Live
or mutating operations exact-compare that identity before contacting a backend.
List responses carry per-source status so one unavailable provider does not
appear to be an empty fleet.

Job cancel, Task retry, and Attempt terminate return durable receipts.
Idempotency keys recover a committed result after a lost response. Existing
controller cancellation, retry policy, reconciliation, and federation redrive
remain the mechanisms that apply those operations.

## Internal design

Frozen records in `iris.cluster.resources` are the only in-process resource
contract. The package layout makes the dependency direction visible rather than
relying on conventions around a file named `facade.py`:

- `controller/controller.py` contains the canonical resource-oriented
  application controller.
- `controller/runtime.py` contains the daemon lifecycle, control loop, backend
  coordination, and service composition formerly called `Controller`.
- `controller/api/` translates the canonical resource protobuf at entry and
  exit.
- `controller/legacy/` implements the old `ControllerService` as a one-way
  adapter over the canonical controller. It owns no resource behavior or SQL.
- `controller/persistence/` owns SQLAlchemy, schema, migrations, transactions,
  queries, projections, and persisted JSON codecs.

The application controller consumes typed persistence results and delegates
execution work to the runtime. Neither it nor its Job, Task, Attempt, Endpoint,
Node, Slice, or federation policy modules import SQLAlchemy or generated
protobuf modules. The runtime may use persistence repositories for control-loop
snapshots and commits, but backends remain database-free.

`ResourceService` is the default and only first-party Job, Task, Attempt, Node,
Slice, and Endpoint API. The old Job and Task protobufs are not a supported
second version: the legacy service decodes into resources immediately and
encodes only its response. Active worker and federation transports retain their
own wire adapters without exporting those messages into the product model.

Persistence remains one concrete SQLAlchemy/SQLite implementation, not a
generic repository framework. Modules are noun-owned where useful, and
transactional commands may enforce storage invariants atomically. Pure policy
does not live beside SQL merely because a transaction consumes its result.

`TaskBackend` remains in `controller/backend.py`. RPC and Kubernetes adapters
keep their existing scheduling, reconciliation, and autoscaling boundaries.
Node and Slice resources adapt existing Worker, scaling-group, autoscaler, and
provider observations. The PR does not introduce a second backend protocol or
split the Kubernetes adapter merely to match a proposed file tree.

Attempt lifecycle and runtime detail are separate public records. Runtime detail
is assembled from the current Attempt storage and provider observation path.
There is no new Attempt runtime sidecar table in this change.

## Persistence and release

`0051_action_receipts.py` additively creates the action table and indexes. It
does not rewrite Job, Task, Attempt, Worker, Endpoint, or federation rows. Fresh
databases materialize the current SQLAlchemy schema. Existing databases run the
same ordered migration chain and retain their noun data.

The abandoned implementation introduced a parallel schema catalog, fingerprint
attestation, repair-on-open, dual writes, and old-image readers. None are needed
for an additive table. A later migration rollup may remove the empty auth
database and consolidate historical migrations; it is not part of this PR.

Controllers can canary independently because the retained federation wire is
translated at the boundary. Release uses the existing checkpoint and migration
procedure. Ordinary defects roll forward. This change accepts the existing
disaster-recovery tradeoff: restoring a checkpoint can lose writes accepted
after that checkpoint.

## Review sequence

1. Use resource identities and typed admission on the active persistence model.
2. Expose global Tasks, exact Attempts, source status, Nodes, and Slices.
3. Adapt provider observations without changing backend ownership.
4. Add durable cancel, retry, and terminate receipts.
5. Move first-party callers and delete superseded internal paths.

Each group has journey coverage before the corresponding old internal path is
removed.
Mechanical protobuf generation stays separate from semantic review where
possible.

## Acceptance

- CLI, Python client, dashboard, resource RPC, and federation use the same
  resource records and exact-target semantics.
- Old Job and Task protobuf shapes appear only in generated/client transport or
  the explicit legacy RPC adapter.
- `controller/controller.py` is the resource application surface;
  `controller/runtime.py` is the process/control-loop implementation.
- Hand-written SQLAlchemy imports under `controller/` appear only in
  `controller/persistence/`.
- The legacy service depends on the canonical controller and never reaches
  persistence or a backend directly for resource operations.
- A populated pre-0051 database retains public Job and Task reads and can create
  and reopen an idempotent action receipt after migration.
- Resource list pages retain fixed statement and SQLite bind budgets. Retired
  unpaged list RPCs drain those pages in bounded batches, so their work scales
  with the collection they must return.
- Partial provider failures remain visible without erasing healthy results.
- Deterministic journeys cover restart, replacement, federation, and action
  behavior.
- Full safe Iris tests, Pyrefly, precommit, dashboard typecheck, and replay pass
  before merge.

## Deferred work

- Physical schema and migration-history rollup, including auth database removal.
- Attempt runtime sidecar persistence if current columns become insufficient.
- Further `TaskBackend` or Kubernetes adapter decomposition based on a concrete
  ownership or testing need.
- Slice deletion and bounded utilization history in Node detail.
