# Iris resource model in one forward-only release

**Status:** Draft for design review
**Parent:** [#7560](https://github.com/marin-community/marin/issues/7560)
**Source design:** repo-shared `iris-resource-unification-design`, revision 2

## Summary

Move Iris to one coherent resource model in a single pull request, reviewed as
five ordered commit groups. Keep the nouns, typed boundaries, normalized
persistence, backend separation, and durable actions from the original design.
Remove its multi-image compatibility rollout. The release performs one
fail-closed migration from the exact current schema to the final schema, and
all callers move to the new contract together.

The result should make the common operator path obvious:

```text
submit Job -> inspect Tasks -> inspect exact Attempt -> read activity/logs
           -> cancel Job | retry Task | terminate Attempt
```

The same nouns and semantics must appear in the CLI, Python client, dashboard,
controller services, federation protocol, and persistence modules.

## Background

[PR #7842](https://github.com/marin-community/marin/pull/7842) replaced the Iris
chaos suite with 49 deterministic journeys and closed the disposition catalog
for 2,496 test families. The controller still concentrates unrelated resource
work in [service.py](https://github.com/marin-community/marin/blob/2804493ff9978343b0c68b799cdbcc8b25aad800/lib/iris/src/iris/cluster/controller/service.py),
[reads.py](https://github.com/marin-community/marin/blob/2804493ff9978343b0c68b799cdbcc8b25aad800/lib/iris/src/iris/cluster/controller/reads.py),
and [writes.py](https://github.com/marin-community/marin/blob/2804493ff9978343b0c68b799cdbcc8b25aad800/lib/iris/src/iris/cluster/controller/writes.py),
while the public protocol retains per-Job Tasks, `KickTasks`, and Worker reads in
[controller.proto](https://github.com/marin-community/marin/blob/2804493ff9978343b0c68b799cdbcc8b25aad800/lib/iris/src/iris/rpc/controller.proto#L261).
[research.md](research.md) records the design history and repository survey.

## Challenges and costs

- The exact schema cutover changes every persistence caller in one commit group.
  Unknown or ambiguous retained rows must stop the migration before mutation.
- CLI, Python client, dashboard, federation, RPC, K8s, and local RPC execution
  must change together or the vocabulary remains split.
- Disaster rollback restores the pre-migration checkpoint and can lose writes
  accepted after it. Each controller therefore needs a short write outage.
- Action receipts add product behavior and retained state. The PR is limited to
  three actions so one single-writer model can be tested end to end.

## Public contract

The public nouns are `Job`, `Task`, `Attempt`, `Node`, `Slice`, and `Endpoint`.
`Backend` identifies placement and source status. `Worker` remains private to
the controller where it still describes a process allocation.

The initial verb matrix is deliberately small:

| Noun | Reads | Mutations |
| --- | --- | --- |
| Job | list, describe, spec, logs, activity | submit, cancel |
| Task | list, describe, logs, activity | retry |
| Attempt | describe, logs, activity, exec, profile | terminate |
| Node | list, describe | none |
| Slice | list, describe | none |
| Endpoint | list, describe | mint |

Reads return typed records with source metadata: source ID, backend ID,
observation time, freshness, and a bounded error. Global Tasks and Nodes are
first-class; callers do not need to enumerate Jobs or know whether capacity is
K8s or RPC. A failed backend produces a partial result with source status, not
an apparently empty fleet.

Mutations are asynchronous and exact. `job cancel`, `task retry`, and
`attempt terminate` return a durable action receipt. Repeating the same request
is idempotent. A receipt records the requested target incarnation, accepted
transition, progress, and terminal outcome. It is applied by the controller's
normal single-writer reconciliation path rather than by a second imperative
backend path.

Task retry is deliberately bounded: it preempts the exact current active
Attempt, consumes normal preemption retry budget, and makes the Task
retry-eligible. It does not reopen a terminal failed Task or guarantee immediate
replacement placement.

Old names such as `worker status`, `job kick`, `job stop`, and `job kill` are
deleted with their call sites. There are no aliases or deprecation shims.

## Internal shape

Use typed resource models in `iris.cluster.resources` and resource services in
`iris.cluster.controller.resources`. Persistence is owned by noun modules such
as `persistence/job.py`, `task.py`, `attempt.py`, `node.py`, `slice.py`, and
`action.py`. Cross-noun scheduling and control queries live in explicitly named
use-case modules. Do not introduce a universal `Resource` base class, generic
repository, generic `List(kind=...)`, or a table that stores every noun.

The final schema directly represents the model:

- Jobs and Tasks have exact UIDs in addition to their human-readable IDs.
- Attempts retain exact attempt UIDs; runtime/provider observations move to
  `attempt_runtime_objects` and do not mutate Attempt lifecycle facts.
- RPC and K8s execution project into a shared Node read model while provider
  detail remains typed and source-specific.
- Slice membership and capacity are normalized rather than encoded in Worker
  rows.
- `action_receipts` durably represents accepted control operations.
- The obsolete auth database and its migrations are absent from fresh and
  upgraded resource-schema databases.

Backends consume typed requests and return typed observations. They do not read
or write the controller database. Split K8s lifecycle, observation, logs/exec,
and capacity code so each adapter has one responsibility. RPC and K8s share the
resource contract, not an inheritance hierarchy.

## Ordered implementation plan

The mega PR follows the original stages as commit groups. Mechanical moves stay
separate from semantic changes, and every group begins with journey expectations
and ends by deleting the path it replaces.

### 1. Final persistence and exact routing

Introduce typed identities and persistence modules, then perform the one-time
schema cutover. The upgrader accepts one fingerprinted current schema, builds
the final schema transactionally, copies and validates all nouns, and swaps only
after postconditions pass. Fresh and upgraded fingerprints must match. Unknown,
partial, or ambiguous source data fails without mutation.

Move runtime observations to the Attempt sidecar with no legacy mirrors. Make
backend resolution exact at every Task and Attempt entry point. Complete this
group with reopen, crash, malformed-source, routing-conflict, query-budget, and
fresh-versus-upgraded journeys/tests.

### 2. Unified execution reads

Expose global Task and Node lists, exact Attempt resolution, provider-neutral
describe/logs/exec/profile, and explicit per-source status. Update CLI, SDK,
dashboard, and federation together. Remove public Worker and provider-specific
read paths. Journeys cover partial backend outage, authorization, stale
observations, Task replacement, and federation ownership.

### 3. Backend and capacity boundaries

Remove database access from execution backends. Split the K8s omnibus adapter
and project K8s/RPC observations into the shared Node/Slice/capacity model.
Normalize Slice membership and scaling-group capacity. Journeys prove the same
resource behavior for local RPC, K8s, and federated execution, including
degraded and recovering sources.

### 4. Durable control and activity

Implement action receipts and route Job cancel, Task retry, and Attempt
terminate through the controller writer. Extend the existing task activity
stream with receipt transitions rather than adding another event store. Journeys
cover duplicate requests, controller restart, target replacement, provider
failure, federation redelivery, and eventual completion. Slice deletion remains
an infrequent provider operation outside this resource contract.

### 5. Contract and code contraction

Delete the old RPCs, protobuf fields, CLI commands, schema/query bundles,
Worker-facing dashboard surfaces, dead rollout compatibility, and historical
auth migration path. Regenerate clients and update OPS/docs. Run the complete
journey corpus against fresh and upgraded databases and enforce query and bind
budgets on the public operations.

## Release model

This is a forward-only schema release, not an expand/contract deployment.

1. Take a checkpoint and run the complete read-only migration preflight against
   its exact merge-base schema and retained rows.
2. Stop acceptance of new writes on that controller and stop its old process.
3. Start the new image, run the atomic migration, and execute a small health and
   resource smoke suite before reopening ingress.
4. Canary the same artifact on representative RPC and K8s controllers. Quiesce
   and cut over every controller in one connected federation component together;
   independent components roll separately.
5. Roll forward for ordinary defects. Disaster rollback restores the
   pre-migration checkpoint and old image, and may lose submissions accepted
   after that checkpoint.

The existing rollout record only needs to identify the immutable image, source,
and schema epoch. This design does not add dual writes, repair-on-open,
old-image readers, parity telemetry, or an ancestry protocol.

## Testing

- The 49 existing journeys remain green and resource journeys are added before
  each group.
- CLI, SDK, dashboard, federation, and controller use the same nouns and exact
  target semantics.
- Fresh and upgraded database fingerprints are identical.
- Public reads have fixed statement budgets and bounded bind sizes at high
  cardinality.
- Partial backend failures remain visible and do not erase healthy results.
- Accepted mutations survive controller restart and are idempotent.
- No production code writes both old and new schema representations.
- No auth sidecar, compatibility alias, or public Worker API remains.

## Settled decisions

- `activity` replaces the public `task events` spelling.
- The mutation set is Job cancel, Task retry, and Attempt terminate. Slice
  deletion is not part of this PR.
- The upgrader accepts one exact merge-base schema fingerprint.
- Independent controllers cut over separately. Controllers joined by federation
  cut over as one quiesced connected component because no old/new wire bridge
  exists. There is no fleet-wide write outage unless the federation graph spans
  the fleet.

## Open questions

- Should Node describe return bounded utilization history, or link to the
  existing stats views and keep the resource response structural? The proposal
  keeps time-series data in stats.
- Is 30 days the right retention for completed action receipts? Nonterminal
  receipts are never age-pruned.
