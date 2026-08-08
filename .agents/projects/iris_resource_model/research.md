# Iris resource model research

## Question

How should Iris adopt the resource model from the repo-shared design in one
reviewable pull request, now that deterministic journey coverage is available
and backward compatibility is not a requirement?

## Sources reviewed

- Repo-shared `iris-resource-unification-design`, revision 2 (document revision
  6), including its five semantic stages and old-image rollout contract.
- Branch artifacts `iris-journey-testing-catalog` and
  `iris-test-cleanup-plan`.
- [Operator vocabulary issue #7560](https://github.com/marin-community/marin/issues/7560).
- [Operator vocabulary PR #7546](https://github.com/marin-community/marin/pull/7546),
  [rollout hardening PR #7583](https://github.com/marin-community/marin/pull/7583),
  [task history PR #7661](https://github.com/marin-community/marin/pull/7661),
  and [journey cleanup PR #7842](https://github.com/marin-community/marin/pull/7842).
- Current Iris controller, persistence, backend, CLI, protobuf, migration, and
  test layout at commit `2804493ff9978343b0c68b799cdbcc8b25aad800`.

## Findings

### The resource model remains useful

The original nouns still describe the system better than its current storage
and API vocabulary:

- `Job` owns user intent.
- `Task` is one schedulable unit within a Job.
- `Attempt` is one execution incarnation of a Task.
- `Node` is backend-neutral capacity; `Worker` is an implementation term.
- `Slice` is a capacity allocation whose membership and lifecycle must be
  explicit.
- `Endpoint` is an addressable service attached to execution.
- `Backend` is a placement and execution boundary, not a public resource base
  class.

`ResourceKey(cluster_id, kind, resource_id)` is useful at API boundaries, but a
universal resource table, repository, or generic CRUD API is not. Exact UIDs
are justified for incarnations and durable mutation targets, not as decoration
on every internal object.

### The old rollout strategy is now the wrong constraint

The five-PR design assumed old and new controller images had to alternate over
an expanded database. That forced dual writes, legacy mirrors, old-image repair,
parity telemetry, historical-schema replay, and a large rollout attestation
surface. Those mechanisms dominated the earlier Stage 1 diff without improving
the final resource experience.

The current direction explicitly permits a forward-only release and accepts
losing submissions during a short disaster rollback. Therefore the mega PR
should have:

- no old-image compatibility writes;
- no compatibility RPC or CLI aliases;
- no dual-read or dual-write interval;
- no telemetry waiting period before contraction;
- one exact current-schema-to-resource-schema migration;
- a pre-migration checkpoint and quiesced controller;
- restore of that checkpoint plus the old image as disaster recovery, with the
  possible loss window stated plainly;
- roll-forward as the normal response after schema activation.

### Current code is split by mechanism rather than resource

The controller still concentrates behavior in large files: service reads and
writes, controller reconciliation, a wide persistence schema, and a large K8s
task backend. The database retains `workers`, inline attempt/provider fields,
and denormalized slice/capacity data. The protobuf and CLI expose both old and
new vocabulary (`Worker`, per-Job task listing, `kick`, `stop`, `kill`, and
provider-specific status surfaces).

Some target behavior already exists and should be retained rather than rebuilt:

- Task and Attempt describe commands;
- durable task activity records;
- authenticated federation and its redelivery model;
- checkpoint and rollout primitives;
- exact attempt identity in execution paths;
- the new synchronous controller tick and 49 deterministic journey cases.

### The testing foundation changes how the work can be staged

PR #7842 closed the catalog for 2,496 collected test families, removed 11,867
lines while adding the journey framework, and established 49 concise journeys.
The resource PR can now stage work by behavior:

1. add or update the journey that describes the target contract;
2. change one coherent resource slice;
3. retain narrow protocol, migration, authorization, query-plan, and provider
   tests where journeys are the wrong level;
4. remove the superseded API, schema, or implementation in the same commit
   group.

This is stronger than retaining temporary compatibility paths merely so old
tests remain green.

## Design constraints inferred from the evidence

- One pull request, with independently understandable commit groups matching
  the original five semantic stages.
- Every commit group ends in user-visible journey coverage and a green safe
  Iris suite.
- Final vocabulary is introduced directly; all call sites change together.
- Persistence APIs are typed and noun-owned. Avoid a generic repository layer.
- Backends do not own controller database access.
- Reads report partial source failure explicitly rather than returning an empty
  result.
- Mutations target exact incarnations and have durable, idempotent receipts.
- Activity uses the existing task-event stream plus action receipts; do not add
  a second event store.
- Fresh and upgraded databases must have the same final schema fingerprint.
- The migration supports one declared source fingerprint and rejects unknown
  or partially migrated databases without mutation.

## Risks to resolve in design review

- A single schema cutover makes the first commit group large even when its
  behavior is intentionally unchanged.
- The final public surface spans CLI, Python SDK, dashboard, federation, and
  providers; leaving one out would preserve the vocabulary split.
- Durable actions add real product behavior, not only cleanup. Their supported
  set must be bounded before implementation.
- A migration rollup can remove the obsolete auth sidecar and old migration
  history, but the minimum supported source schema must be explicit.
- Slice deletion and automatic capacity policy are less mature than the core
  Job/Task/Attempt experience and may not belong in the same acceptance bar.

## Decisions after review

- Use `activity` for the public history verb and remove `task events`.
- Implement Job cancel, Task retry, and Attempt terminate. Manual Slice deletion
  is outside the PR.
- Accept only the exact merge-base schema fingerprint during upgrade.
- Cut independent controllers over separately. A connected federation component
  must quiesce and cut over together because the design has no compatibility
  wire bridge.
