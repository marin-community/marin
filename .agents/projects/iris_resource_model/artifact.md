# Iris resource model: one-PR target

## Decision

Land the Iris resource model in one forward-only pull request after the journey
testing cleanup in PR #7842. Use the five stages from the resource-unification
design as ordered review commits. Do not support old and new controller images
against the same migrated database.

This removes the mechanisms that dominated the abandoned Stage 1 attempt:
dual writes, legacy runtime mirrors, repair-on-open, compatibility RPCs, parity
telemetry, old-image schema readers, and rollout ancestry protocols.

## Operator experience

Iris exposes Job, Task, Attempt, Node, Slice, and Endpoint as its resource nouns.
Backend identifies placement and source status. Worker remains the private name
of the RPC daemon.

```text
submit Job
  -> inspect global Tasks
  -> inspect one exact Attempt
  -> read activity and logs
  -> cancel Job | retry Task | terminate Attempt
```

The CLI, Python client, dashboard, controller RPC, and federation protocol use
the same nouns and target semantics.

The public command families are:

- `iris job run|list|describe|spec|logs|activity|cancel|wait`
- `iris task list|describe|logs|activity|retry`
- `iris attempt describe|logs|activity|exec|profile|terminate`
- `iris node list|describe`
- `iris slice list|describe`
- `iris endpoint list|describe|mint`

`activity` replaces `task events`. `job stop`, `job kill`, `job kick`, public
Worker commands, and standalone Kubernetes-cluster status commands are removed.
Manual Slice deletion is outside this PR because it is rarely used and does not
need to delay the core resource contract.

Global Task and Node lists return partial results with one source status per
backend. A failed backend is visible as unavailable and does not turn the whole
fleet into an empty result. Task and Attempt describe, logs, exec, and profile
share one exact target resolver.

## Durable mutations

Job cancel, Task retry, and Attempt terminate return action receipts. An action
is accepted only after the controller's single writer exact-compares the target,
commits the workload transition, and inserts the receipt in one transaction.
Provider teardown remains part of normal reconcile.

- Job cancel reuses the existing cancellation and federation-redrive behavior.
- Task retry preempts the exact current active Attempt, consumes normal
  preemption retry budget, and makes the Task retry-eligible. It does not reopen
  a terminal failed Task or guarantee a new Attempt.
- Attempt terminate makes the exact Attempt and Task operator-terminated and
  disables automatic retry.

Idempotency keys recover a committed action after a lost RPC response. Provider
timeout leaves a receipt verifying. Replacement UIDs are never contacted.
Federated actions retain the authority action ID and deduplicate at the peer.

Activity merges the existing `iris.task_event` stream with current receipt
outcomes behind controller authorization. It does not add a second event store
or claim a total order across finelog and SQLite.

## Internal experience

The final tree is organized by nouns and use cases:

```text
iris/cluster/resources/                 typed public models and protocols
iris/cluster/controller/resources/      auth and response/action services
iris/cluster/controller/persistence/
  schema/                               noun-specific SQLAlchemy tables
  job.py task.py attempt.py endpoint.py noun persistence
  node.py slice.py action.py            execution/action persistence
  scheduling.py control.py              deliberate cross-noun hot queries
iris/cluster/backends/k8s/              backend, manifests, node, task,
                                        telemetry, and garbage collection
iris/cluster/backends/rpc/              cohesive backend plus Node/Slice views
```

Persistence uses typed top-level functions over a transaction. It does not add
repository classes, a universal Resource base, generic CRUD, or a polymorphic
resource table. Backends consume typed requests and return observations; they
do not import controller persistence or the controller database.

`schema.py`, `reads.py`, `writes.py`, `backend_store.py`, and the 3,158-line K8s
`tasks.py` disappear after their callers move. Dashboard Fleet/Worker and
provider-specific Node branches become one Nodes inventory. The worker daemon
and its local diagnostic dashboard keep their implementation names.

## Final database

The resource schema contains:

- `jobs` and `job_specs`, with exact Job UIDs and explicit authority,
  execution-cluster, and backend coordinates;
- `tasks`, with exact Task UIDs and the exact current Attempt/Node hot-path
  mirrors;
- `attempts` plus `attempt_runtime_objects`, so provider observations do not
  mutate Attempt lifecycle facts;
- `endpoints`, with exact registration IDs and explicit owner coordinates;
- `rpc_nodes`, `rpc_node_details`, `node_capacity`, and `node_attributes`;
- backend-qualified `scaling_groups`, `slices`, and observed `slice_members`;
  and
- `action_receipts` for the three public mutations.

Live Kubernetes Nodes remain cached resources and do not get fake `rpc_nodes`
rows. Unknown Slice membership differs from observed empty membership. The
database contains one SQLite file: the obsolete empty auth sidecar, its
attachment, backup path, and historical auth migrations are removed.

## Migration and release

The upgrader accepts only the exact schema and migration-manifest fingerprints
at the implementation PR's merge base. A fresh database receives the final
schema directly. An existing database is validated in bounded read-only batches
before any DDL. Unknown backends, ambiguous federation coordinates, malformed
relationships, or invalid runtime evidence fail without changing the source
schema.

One SQLite transaction copies validated data into the final tables, checks
postconditions, and contracts the old schema. Fresh and upgraded databases must
have the same normalized DDL/index/foreign-key fingerprint. Historical
migrations are replaced by one v2 schema epoch; this is not a general migration
rollup framework.

Independent controllers cut over separately. Controllers connected by
federation are one cutover unit because there is no old/new wire bridge:

1. Take a checkpoint and run the complete read-only schema and retained-row
   migration preflight.
2. Stop writes and the old controller for that cluster.
3. Start the new image, migrate, and run resource health smoke tests.
4. Canary the same artifact on representative RPC and K8s controllers.
5. Quiesce and migrate every controller in one connected federation component,
   then continue component by component and roll forward for ordinary defects.

Disaster rollback restores the pre-migration checkpoint and old image. It can
lose submissions accepted after that checkpoint. The release does not add a
second rollout state machine to preserve those writes.

## Review sequence

1. Final persistence and exact backend routing.
2. Global Tasks, unified Nodes, and exact execution reads.
3. DB-free backend boundaries and normalized Node/Slice capacity.
4. Durable cancel/retry/terminate plus authorized Activity.
5. API, CLI, dashboard, migration, and physical-schema contraction.

Every group starts with journey expectations, changes one coherent resource
slice, deletes the path it replaces, and leaves the safe Iris suite green.
Mechanical moves and generated protobufs remain separate commits inside the
group.

## Acceptance

- The 49 journeys from PR #7842 stay green; resource journeys cover exact
  replacement, partial outage, federation, restart, and fresh/upgraded storage.
- CLI, Python, dashboard, RPC, and federation expose the same nouns and actions.
- Fresh and upgraded schema fingerprints match.
- Public list operations have fixed statement budgets and bounded SQLite binds.
- Accepted actions survive restart and are idempotent.
- No production dual write, compatibility alias, auth sidecar, or public Worker
  API remains.
- Full safe Iris tests, Pyrefly, precommit, dashboard typecheck/build, and stable
  replay pass before merge.

## Review questions

- Keep Node describe structural and link to existing stats, or embed bounded
  utilization history? The current contract keeps time series in stats.
- Retain completed action receipts for 30 days? Nonterminal receipts are never
  age-pruned.
