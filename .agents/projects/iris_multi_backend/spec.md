# Multi-Backend — Spec

Concrete contracts for [`design.md`](./design.md). This pins the public surface reviewers are agreeing
to: the RPC service, the config schema, the DB migration, the file layout, auth identities, and error
dispositions. It is **not** an implementation plan — no pseudocode, no file-by-file steps.

## 0. Identity model (read first — everything keys on this)

- **`attempt_uid` is fresh per attempt.** A re-placement (after preemption/failure/partition) is a
  **new attempt with a new `attempt_uid`**, never a reused one. `attempt_uid` is already `UNIQUE` in the
  schema and is the primary CAS/idempotency key.
- **`desired_generation` is the attempt ordinal** for its task (i.e. the existing per-task `attempt_id`
  counter, not a parallel counter). It increases monotonically as a task is re-placed, and orders
  desired-state edits so a `removal` can't be undone by a late `upsert`.
- **A late observation is dropped** when its `(attempt_uid, desired_generation)` is no longer the task's
  current attempt — see CAS in §1.

So phrasings like "the root re-places the task" mean *a new attempt at the next generation*, never
"reuse the same uid."

## 1. The wire: `RemoteAgentService`

New proto at `lib/iris/src/iris/rpc/remote_agent.proto`. The agent is the **client** (dials the root);
the root is the **server**. **One RPC.** Interactive ops piggyback on it (the agent only dials out, so it
cannot host server-side interactive RPCs — see §1.1).

```proto
service RemoteAgentService {            // root = SERVER; the agent is the CLIENT (dials home)
  rpc Poll(PollRequest) returns (PollResponse);   // the ONLY call: reconcile + interactive piggyback
}

message PollRequest {                         // observed state goes UP
  string  backend_id = 1;
  uint64  root_epoch_seen = 2;                // highest leader epoch the agent has seen
  uint64  last_sync_id = 3;                   // 0 = fresh / cache lost -> root forces a full snapshot
  Capabilities caps = 4;                      // present on first poll / re-register
  repeated AttemptObservation observations = 5;  // changed + still-unacked (state, chosen worker, exit)
  WorkerHealth health = 6;                    // rolled-up per-worker health
  CapacitySummary capacity = 7;               // see §3.1
  repeated CommandResult command_results = 8; // interactive results for earlier pending_commands (§1.1)
}

message PollResponse {                        // desired state comes DOWN
  uint64  root_epoch = 1;                     // current leader epoch
  uint64  new_sync_id = 2;                    // version of the desired set in this response
  bool    snapshot = 3;                       // true = full set; false = delta from last_sync_id
  google.protobuf.Duration lease_duration = 4;   // renews ALL desired entries (the Poll IS the renewal)
  repeated DesiredAttempt   upserts = 5;      // run-intent: uid, generation, spec(once), constraints
  repeated string           removals = 6;     // attempt_uids; a "fence" = absence: kill what's removed
  repeated DesiredCapacity  autoscale = 7;    // target slice counts per scale group (level-triggered)
  repeated AckObservation   acks = 8;         // APPLIED / STALE_DISCARDED / RETRY_LATER
  repeated InteractiveCommand pending_commands = 9;  // interactive ops to run now (§1.1)
}

message DesiredAttempt {
  string  attempt_uid = 1;
  uint64  desired_generation = 2;             // the attempt ordinal (§0)
  AttemptSpec spec = 3;                       // sent on first appearance; cached after (sync_id deltas)
  repeated Constraint constraints = 4;        // local task->worker placement constraints
  // NOTE: no per-attempt root_epoch / lease_duration — the response-level fields renew all entries.
}

message AttemptObservation {
  string  attempt_uid = 1;
  uint64  acted_root_epoch = 2;               // epoch the agent acted under (root rejects non-current)
  uint64  desired_generation = 3;
  AttemptState state = 4;                     // RUNNING / SUCCEEDED / FAILED / KILLED / PREEMPTED / ...
  string  observed_worker = 5;               // reported, display-only (root never treats as authority)
  int32   exit_code = 6;
  string  message = 7;
}

enum AckDisposition { APPLIED = 0; STALE_DISCARDED = 1; RETRY_LATER = 2; }
message AckObservation { string attempt_uid = 1; AckDisposition disposition = 2; }
```

**Contract notes.**

- **Fence = absence.** A reroute/cancel/preempt drops the attempt from `upserts` and lists it in
  `removals`; the agent kills what runs but is no longer desired (the worker's existing zombie-kill, one
  level up). There is no imperative `Stop`.
- **`sync_id` delta.** The root keeps a per-backend monotonic `sync_id` scoped to `(backend_id,
  root_epoch)` plus a short change-log. Delta when the agent's `last_sync_id` is diffable; otherwise
  `snapshot = true` (fresh boot, change-log rolled past, or prior epoch). The agent keeps **at most one
  in-flight `Poll` per backend** so deltas apply in order.
- **One lease field.** Only `PollResponse.lease_duration` exists; applying any response renews **all**
  currently-desired entries to `monotonic_now() + lease_duration`. There is no per-attempt lease.
- **Lease renewal is response-applied + root-conservative.** The agent renews only *after applying* a
  response; the root assumes a lease *was* renewed the moment it *sends* a response that could renew it,
  and starts its reuse clock then. A lost response ⇒ the agent does not renew (ages → self-fence) while
  the root over-waits — never under-waits.
- **Skew-safe reuse invariant.** The root re-places a task (new attempt, §0) only after the prior
  attempt's `send_time + lease_duration + max_skew + transport_grace + kill_grace`, guaranteeing the
  agent self-fences the old runner before the new attempt can launch.
- **CAS apply lives in the controller, not the backend.** `RemoteTaskBackend` is DB-less (§2);
  `reconcile` returns the agent's observations as a `ReconcileResult`, and the controller's existing
  commit path applies each only if `(attempt_uid, desired_generation)` is still the task's current
  attempt under the current `root_epoch` — else `STALE_DISCARDED`. The resulting per-observation
  dispositions ride back to the agent as `acks` on its next `PollResponse`. The agent GCs a terminal
  substrate object on `APPLIED` **or** `STALE_DISCARDED` (no GC-starvation), retains on `RETRY_LATER`.

### 1.1 Interactive ops (exec / profile / process-status)

The agent dials out, so it cannot receive server-side interactive RPCs. The **root-side surface is the
existing `TaskBackend` methods** — `get_process_status` / `profile_task` / `exec_in_container`
([`backend.py:409-433`](https://github.com/marin-community/marin/blob/1013be215490cce01d095518ba3c07bbe0de0a7f/lib/iris/src/iris/cluster/controller/backend.py#L409)).
`RemoteTaskBackend` implements them by tunneling over the `Poll` channel:

```proto
message InteractiveCommand {            // root -> agent, in PollResponse.pending_commands
  string command_id = 1;
  TaskTarget target = 2;
  string origin_user = 3;              // carried to the worker for audit
  oneof op {
    worker.ExecInContainerRequest exec = 4;
    job.ProfileTaskRequest        profile = 5;
    job.GetProcessStatusRequest   status = 6;
  }
}
message CommandResult {                 // agent -> root, in PollRequest.command_results
  string command_id = 1;
  string error = 2;                    // set iff the op failed / ProviderUnsupportedError
  oneof result {
    worker.ExecInContainerResponse exec = 3;
    job.ProfileTaskResponse        profile = 4;
    job.GetProcessStatusResponse   status = 5;
  }
}
```

The controller-side call blocks until the matching `CommandResult` returns on a subsequent `Poll`. The
agent **fast-follows** — re-Polls immediately when a `command_result` is pending — so interactive latency
is ≈**0.5× Poll cadence** (≈0.5 s mean / 1 s p95 at a 1 s cadence; spike S4 validated this on a loopback
Connect prototype; IAP adds ~30–100 ms RTT). A **held stream** (server-push + unary report, ~2 ms) is an
opt-in escalation for trusted in-VPC latency-sensitive backends; escalate to long-poll on the single
`Poll` RPC before ever adding a second RPC. **PTY / interactive-TTY exec is out of scope** (needs real
bidi). Bulk logs are *not* tunneled: they stay in each backend's finelog, proxied per `backend_id` (no
cross-region shipping).

### 1.2 Lease timing & self-fence (spike S3)

The `lease_duration` (§1) must be a **dedicated short lease**, not the worker's 600 s
`heartbeat_timeout` (`worker.py:83`) — that 600 s term alone makes post-partition re-placement ~10 min.
Spike S3 measured the other terms of the reuse invariant: `kill_grace` ≈ the task-monitor poll interval
(**~5 s**), `transport_grace` = `RECONCILE_RPC_TIMEOUT` (**3 s**), `max_skew` negligible (<30 ms, since
leases are monotonic-duration — only rate drift counts). A short lease therefore targets **~20–30 s**
re-placement with an **~8–9 s** floor.

- **Worker-daemon backends self-fence via the lease** (the worker self-terminates on a stale lease). The
  worker is the *sole* authority over its own process during a partition, so this is required.
- **k8s backends need NO pod self-fence.** The apiserver is a durable, independently-reachable authority
  and the agent is recoverable, so any live agent reconciles an undesired pod away (poll-and-delete —
  today's single-cluster model). Pods are named deterministically by `attempt_uid`, so reconcile is
  **idempotent across agents** (create-if-absent / delete-if-undesired converge; two agents or a
  restarted agent never fight or double-create) — this is what makes multiple agents on one cluster safe.
  The two-phase reroute (remove from the old backend first; add to the new only after observed-drain or
  the lease horizon) waits for the observed pod-drain; in the rare full-isolation case it falls back to
  the lease horizon, leaving a **bounded, benign** double-run window
  — benign because re-placement is a *fresh* attempt with a fresh `attempt_uid` writing to a fresh
  `…/attempt=<uid>/` path, so the two pods never corrupt each other and the publish-gate picks the
  winner (identical to single-cluster k8s today). A lease sidecar that kills the pod on expiry survives
  only as an **opt-in hard-fence** for jobs with *external* side effects that cannot tolerate even brief
  overlap. **Validation:** a local `kind` cluster (not gated infra) measures pod create/delete latency
  and verifies the reconcile-delete + idempotent multi-agent path.

## 2. Root-side adapter & agent

- **`lib/iris/src/iris/cluster/controller/backends/remote.py` — `RemoteTaskBackend`** (the only new
  root-side backend). Implements the existing **DB-less** `TaskBackend` Protocol
  ([`backend.py:340`](https://github.com/marin-community/marin/blob/1013be215490cce01d095518ba3c07bbe0de0a7f/lib/iris/src/iris/cluster/controller/backend.py#L340))
  and tunnels to one agent over `Poll`. **The root has one remote backend type regardless of substrate.**
  It holds **no DB handle and no substrate credentials** — it caches the agent's latest Poll and returns
  plain data, exactly like the in-process backends. The surface mirrors the Protocol:

  ```python
  class RemoteTaskBackend:
      """Root-side, DB-less TaskBackend tunneling to one cluster agent over RemoteAgentService.

      schedule() runs the root meta-scheduler (task->backend) over the agent's last-reported
      CapacitySummary. reconcile() drains the agent's latest Poll into a ReconcileResult (observations
      + health_events); the CONTROLLER's commit path applies them under CAS and the dispositions ride
      back as acks on the next Poll. Holds no DB handle and no substrate credentials.
      """
      name: str
      capabilities: ClassVar[frozenset[BackendCapability]]   # {CLUSTER_VIEW} — root never places workers
      autoscaler: Autoscaler | None                          # None; capacity rides over Poll, not Iris autoscaler

      def schedule(self, snapshot: ScheduleInput) -> ScheduleResult: ...
      def reconcile(self, snapshot: ControlSnapshot) -> ReconcileResult: ...
      def autoscale(self, snapshot: ControlSnapshot,
                    residual_demand: list[DemandEntry], dead_workers: list[WorkerId]) -> AutoscaleResult: ...
      def attach_autoscaler(self, autoscaler: Autoscaler) -> None: ...   # not called (no IRIS_AUTOSCALER cap)
      def get_process_status(self, target, request) -> GetProcessStatusResponse: ...  # tunnels via §1.1
      def profile_task(self, target, request, timeout_ms) -> ProfileTaskResponse: ... # tunnels via §1.1
      def exec_in_container(self, target, request, timeout_seconds=60) -> ExecInContainerResponse: ...
      def close(self) -> None: ...
  ```

  `RemoteTaskBackend` carries `capabilities = {CLUSTER_VIEW}`: from the root's view a remote backend
  places its own workers (the agent does), exactly like k8s today — so the controller drives it through
  the existing `CLUSTER_VIEW` path with no new control-loop branch.

- **`lib/iris/src/iris/cluster/agent/` (new package) — hosts a real backend + the Poll loop + cache.**
  - `agent/loop.py` — the `Poll` client; dials the root, drives the reconcile loop on its own cadence,
    executes `pending_commands` and returns `command_results`.
  - `agent/cache.py` — the in-memory recoverable worker DB (roster, health, slices, attempt→worker
    binding, allocated ports, local placement). In-memory SQLite is the default.
  - The agent instantiates the **unchanged** `RpcTaskBackend` **or** `K8sTaskProvider` and drives its
    existing `schedule(local)` + `reconcile` + `autoscale`, calling the same `WorkerInfraProvider`.

**Two RPC layers (do not conflate).** (1) **agent ↔ root** = `RemoteAgentService.Poll` (new). (2)
**worker daemon ↔ agent** = the existing worker `Reconcile`
([`worker.proto:154`](https://github.com/marin-community/marin/blob/1013be215490cce01d095518ba3c07bbe0de0a7f/lib/iris/src/iris/rpc/worker.proto#L154))
/ kube-apiserver (intra-cluster, unchanged). A GCP worker daemon lives at layer (2).

## 3. Prerequisite refactor — file layout

Regroup `cluster/backends/` by concern into three top-level trees. The protocols
(`WorkerInfraProvider` / `ControllerProvider`,
[`protocols.py`](https://github.com/marin-community/marin/blob/1013be215490cce01d095518ba3c07bbe0de0a7f/lib/iris/src/iris/cluster/backends/protocols.py#L21))
+ `factory.create_provider_bundle()` stay as the seam.

| Tree | Holds | Imported by the agent? |
|---|---|---|
| `cluster/backends/` | pure `TaskBackend`: `rpc/` (`RpcTaskBackend`), `k8s/tasks.py` (`K8sTaskProvider`) | yes |
| `cluster/platforms/` (new) | runtime substrate drivers: `WorkerInfraProvider` impls (`gcp`/`manual`/`local` `workers.py`), cloud-API wrappers (`GcpService`, `CloudK8sService`), worker bootstrap, `resolve_image`, remote-exec health | yes |
| `cluster/setup/` (new) | admin bring-up/ops: `ControllerProvider` impls, `vm_lifecycle.py`, controller bootstrap, image build/push, IAM (`setup_iam.py`) | **never** |

**Invariant: the agent depends only on `backends/` + `platforms/`, never `setup/`** — a compile-time
fact, not a convention.

Migration order (each ships independently; single-cluster unaffected):

1. Move k8s priority-class constants out of `k8s/tasks.py` (backend) into `k8s/types.py` — the one real
   layering leak today (admin `k8s/controller.py` imports them from the backend).
2. Split `bootstrap.py` into `worker_bootstrap` (→ `platforms/`) + `controller_bootstrap` (→ `setup/`);
   lift `resolve_image` into a `platforms/` utility.
3. Introduce `cluster/setup/` and move `ControllerProvider` impls + `vm_lifecycle.py`; the
   `WorkerInfraProvider` side + cloud services become `cluster/platforms/`. **The structural unlock.**
4. (Optional) finish the per-platform regroup under `cluster/platforms/<platform>` +
   `cluster/setup/<platform>`.

### 3.1 Meta-scheduler match contract

The root matches a task's `Constraint`s against the union of each backend's static `attributes` and its
last-reported `CapacitySummary`, reusing the existing `Constraint{key, op, value(s), mode}` +
`ConstraintIndex` matcher (`scheduling/scheduler.py`).

- **Attribute normalization.** `BackendConfig.attributes` values are **comma-split into a set** at load
  (`device-variant: "v5e-4,v5p-8"` → `{v5e-4, v5p-8}`). The existing `ConstraintIndex` is **scalar (one
  value per key)**, so matching a constraint against a set-valued backend attribute needs a
  **set-membership extension** to the matcher (spike S2) — "reuse the existing matcher" is not literal.
  Normalization happens once at config load, not per-match.
- **`--backend X` is a routing directive, not a worker constraint.** It pins task→backend, then **must be
  stripped from the constraints handed to the agent's local scheduler**: agents don't advertise a
  `backend` attribute on workers, so a leftover `backend=X` constraint matches no worker and the task
  starves (a real bug surfaced in S2).
- **`allow_policy` filters first.** Backends the requesting user can't access are removed *before*
  matching (see §6).
- **`CapacitySummary` (validated against the real `Scheduler` in spike S2 — 512 tasks / 5 backends;
  two-level placed 512/512, 0 starved, vs. one global scheduler, at +0.15 tick mean wait / −0.6 pts
  util):**

  ```python
  class CapacitySummary(BaseModel):
      # free, dynamic capacity reported each Poll — config the root already holds is NOT repeated here:
      allocatable: dict[str, int]            # free worker slots per device variant, now
      max_free_cpu_millicores: int           # largest single-worker free CPU bin (a free *count* can't bin-fit)
      max_free_memory_bytes: int             # largest single-worker free RAM bin
      largest_gang: dict[str, int]           # biggest coschedulable gang placeable NOW, per variant (balance)
      stale_ms: int                          # age of this summary (root discounts stale capacity)
      backoff: dict[str, int]                # per-group quota-exceeded / cooldown until-ms
  ```

  S2 pruned two strawman fields and split one: **`static` dropped** (it only mirrors
  `BackendConfig.attributes`, which the root already has); **`pending_leases` dropped** (the root is
  authoritative over task→backend + leases, so its own ledger subsumes it — the real double-count guard is
  an **in-tick decrement** in root-side bookkeeping, *not* a summary field; omitting it halved utilization
  83%→50% in the BURST scenario); **`largest_gang` split** into a static config-derived `max_gang` per
  variant (prevents permanent gang starvation; lives in config, not the summary) and the dynamic
  `largest_gang` above (latency/balance only). Each capacity field is **per-variant**, not a single int.

  A task matching **no** backend statically → `UNSCHEDULABLE` with a reason; matching but with no live
  capacity → stays `PENDING` and rides the autoscaler.

## 4. Config schema

A new `backends:` map on the pydantic `IrisClusterConfig`
([`config.py:564`](https://github.com/marin-community/marin/blob/1013be215490cce01d095518ba3c07bbe0de0a7f/lib/iris/src/iris/cluster/config.py#L564),
`extra="forbid"`). No proto change.

```python
class BackendConfig(BaseModel):
    kind: Literal["worker_daemon", "k8s"]          # the TaskBackend type
    transport: Literal["remote", "in_process"] = "in_process"
    attributes: dict[str, str] = {}                # comma-split to a set at load (§3.1)
    allow_policy: AllowPolicy = AllowPolicy(users=["*"])
    worker_provider: WorkerProviderConfig | None = None   # gcp/manual, for kind=worker_daemon
    kubernetes_provider: KubernetesProviderConfig | None = None  # for kind=k8s
    scale_groups: dict[str, ScaleGroupConfig] = {}        # this backend's groups

# on IrisClusterConfig:
#   backends: dict[str, BackendConfig] | None = None
```

```yaml
# single-cluster (today) — unchanged; an absent `backends:` = one implicit in-process backend
name: marin
platform: { label_prefix: marin, gcp: { project_id: hai-gcp-models } }
scale_groups: { tpu_v5e_4-us-west4-a: { … } }

# multi-backend
backends:
  gcp-tpu-west:
    kind: worker_daemon
    transport: remote
    attributes: { provider: gcp, region: us-west4, device-variant: "v5e-4,v5p-8" }
    allow_policy: { users: ["*"] }
    worker_provider: { gcp: { project_id: hai-gcp-models } }
    scale_groups: { tpu_v5e_4-us-west4-a: { … } }
  cw-east-h100:
    kind: k8s
    transport: remote
    attributes: { provider: coreweave, region: us-east-02e, device-variant: h100 }
    allow_policy: { users: ["alice", "bob"] }
    kubernetes_provider: { namespace: iris, kueue: { … } }
```

**Validation rules.**

- Mixing `backends:` with the legacy top-level provider fields (`scale_groups` / `worker_provider` /
  `kubernetes_provider`) is **rejected** — the top-level form is accepted only as the implicit
  single-backend case.
- **At most one** `transport: in_process` backend (the root can co-locate one backend in-process); it
  may coexist with any number of `remote` backends. More than one `in_process` is a config error.
- The root reads `backends:` at startup (`composer.make_backend` per entry,
  [`composer.py:158`](https://github.com/marin-community/marin/blob/1013be215490cce01d095518ba3c07bbe0de0a7f/lib/iris/src/iris/cluster/composer.py#L158))
  and builds a `RemoteTaskBackend` or an in-process backend for each.

## 5. Persisted shapes — root DB migration

A new idempotent migration (`controller/migrations/00NN_*.py`) adds to the **root** (authoritative) DB:

- `jobs.backend_id`, `tasks.backend_id` — the backend the job is **pinned** to; inherited by descendants.
  Backfilled to the implicit backend id (below) for existing rows; **immutable for the job's lifetime
  except an operator `drain`/`remove` of the whole backend** (a rare, authoritative re-pin). Preemption/
  failure re-placement stays on the pinned backend (queue until capacity); a job never splits across
  backends (out-of-scope: cross-backend gangs).
- `attempts.root_epoch`, `attempts.desired_generation`, `attempts.lease_deadline_ms` — **per-attempt**
  fencing fields (the lease guards a specific running attempt; `desired_generation` = the attempt ordinal,
  §0).
- `attempts.observed_worker` (nullable) — last-reported worker, **display only**.
- `backends` table: `backend_id PK, kind, status, last_seen_ms, attributes_json, allow_policy_json`,
  where `status` is a `StrEnum BackendStatus { ACTIVE, DRAINING, REMOVED }` tied to the §6 CLI lifecycle.
- `controller_state.root_epoch` — the monotonic leadership token.
- **No `workers` table for worker-daemon backends** — the idle-worker inventory is recoverable cache,
  owned by the agent.
- `worker_policy` table (**sparse** — only cordoned/draining workers): `(backend_id, worker_id) PK,
  draining bool, reason, set_at_ms`. Operator cordon/drain is an authoritative *decision*, not derivable
  from the substrate, so it lives in the root (low-cardinality) and rides down to the agent in the Poll
  desired-state; the agent's local scheduler skips draining workers, and **drain** = cordon + re-place
  running attempts via the normal path. This is a sparse *overlay*, not the worker roster — the root
  still holds no full roster. k8s may instead use a native node cordon/taint (the agent projects the
  flag onto the node); a worker-daemon may additionally persist the flag on local disk and re-report it
  on registration (then it is also recoverable by construction).

**Implicit backend id.** When `backends:` is absent, the single in-process backend is stamped with a
reserved id derived from `config.name` (e.g. the cluster name); the migration backfills all existing
`jobs`/`tasks` rows to it so the "single-cluster unchanged" path has a non-null `backend_id`.

**Agent-local cache (NOT a root migration).** Worker roster, per-worker health, slice inventory, idle
inventory, attempt→worker binding, and **allocated host ports** live in an agent-local, in-memory store
rebuilt at startup from re-registration + `list_all_slices()` + the root's Poll response. No
outbox/inbox/event-log/cursor tables. Spike S1 confirmed roster/slices/bindings/ports rebuild exactly
(health resets — the allowed diff), with two boundary corrections:

- **Service endpoints stay root-authoritative — never demote them to the agent cache.** They are *not* a
  pure function of the three sources (a long-running task does not re-announce its endpoint across an
  agent restart), so they remain in the root DB; `endpoints_table`
  ([`schema.py:448`](https://github.com/marin-community/marin/blob/1013be215490cce01d095518ba3c07bbe0de0a7f/lib/iris/src/iris/cluster/controller/schema.py#L448))
  is reframed as a lease in [#6722](https://github.com/marin-community/marin/issues/6722) so endpoint
  liveness is self-healing rather than tied to task-row lifetime.
- **The attempt→worker binding recovers from the substrate object's stamped labels** (surfaced via the
  worker's reconcile/discovery), **not from the `Register` handshake** — `RegisterRequest` carries
  identity only. This is why `attempt_uid` (+ ports) must be stamped on the substrate (below); it holds
  for the k8s path too (pod labels listed via the API).

**Substrate stamping (the recoverability footprint).** Every substrate object carries full identity: k8s
pods get label `iris.attempt_uid` + annotations `iris.full_task_id` / `iris.root_epoch` /
`iris.desired_generation` / `iris.lease_deadline` **and the attempt's allocated host ports**;
worker-daemon containers get the same in Docker labels. CAS keys on `attempt_uid` (already `UNIQUE`),
never on a sanitized label. `TaskAttempt.adopt()`
([`task_attempt.py:303`](https://github.com/marin-community/marin/blob/1013be215490cce01d095518ba3c07bbe0de0a7f/lib/iris/src/iris/cluster/worker/task_attempt.py#L303))
must restore ports from the stamp into `PortAllocator` before scheduling new work on that worker. This
port recovery is **mandatory** — spike S1 reproduced the loss against the real classes (a live
double-allocation bug on *any* worker restart, tracked as
[#6721](https://github.com/marin-community/marin/issues/6721)) — and takes three concrete deltas:
(1) add a `ports` field to `DiscoveredContainer`
([`runtime/types.py:271`](https://github.com/marin-community/marin/blob/1013be215490cce01d095518ba3c07bbe0de0a7f/lib/iris/src/iris/cluster/runtime/types.py#L271)),
populated from the stamped label/annotation in discovery; (2) add `PortAllocator.reserve(ports)`
([`port_allocator.py:13`](https://github.com/marin-community/marin/blob/1013be215490cce01d095518ba3c07bbe0de0a7f/lib/iris/src/iris/cluster/worker/port_allocator.py#L13)
has only `allocate`/`release` today); (3) have `adopt()` restore `attempt.ports` and call `reserve()`.

## 6. Auth & access control

- **Agent → root identity:** a backend-scoped `system:controller` role minted at onboarding, parallel to
  `system:worker` (`_create_worker_jwt`,
  [`auth.py:462`](https://github.com/marin-community/marin/blob/1013be215490cce01d095518ba3c07bbe0de0a7f/lib/iris/src/iris/cluster/controller/auth.py#L462);
  add the mint in `create_controller_auth`,
  [`auth.py:322`](https://github.com/marin-community/marin/blob/1013be215490cce01d095518ba3c07bbe0de0a7f/lib/iris/src/iris/cluster/controller/auth.py#L322)).
  The role is named for the controller-tier **authority** the agent carries over its one cluster, not the
  component. Claims carry `backend_id` + the allowed RPC set; `RemoteAgentService` binds every call's
  subject to its `backend_id`; the role is **default-deny outside `RemoteAgentService`**; loopback-trust
  is disabled on this path. IAP service-account ID tokens bootstrap agents behind IAP. Revocable via the
  existing `api_keys` set.
- **Per-backend access control.** `backends.allow_policy_json` gates which users may route to a backend:
  - An **explicit** `--backend X` the user can't access → rejected at admission (permission-denied).
  - For an **un-pinned** job, disallowed backends are **filtered out before matching** (§3.1); if no
    accessible backend matches, the job is `UNSCHEDULABLE` with a reason (never a leaked permission error
    naming a backend the user can't see).

## 7. CLI

- `iris agent serve --backend <id> --controller iris.oa.dev --config <cluster.yaml>` — starts an agent
  near its cluster (k8s: an in-cluster Deployment; worker-daemon: a small VM in the cluster's project).
- `iris backend add <id> --kind … --region …` (register + mint identity); `iris backend list/status`
  (health, epoch, lease margin, last-Poll); `iris backend drain <id>` (status → `DRAINING`: stop routing,
  drain running, autoscale → 0); `iris backend remove <id>` (status → `REMOVED`, after drained). These
  extend, not replace, today's `iris cluster {start,stop,create-slice,status,dashboard}`.

## 8. Errors / dispositions

- `AckDisposition.STALE_DISCARDED` — CAS mismatch (superseded attempt); the agent GCs the terminal
  object anyway.
- `AckDisposition.RETRY_LATER` — root could not commit this tick; the agent retains the object and
  re-reports.
- Routing: no static match → `UNSCHEDULABLE` (with reason); match but no live capacity → `PENDING` +
  autoscaler.
- Preemption vs app-failure keeps the existing `KILLED`-vs-`WORKER_FAILED` budget split; re-placement is
  a **fresh attempt at the next generation on the pinned backend** (never live migration, never
  cross-backend except operator drain).
- `ProviderUnsupportedError` from an interactive op surfaces as `CommandResult.error` (§1.1).

## 9. Out of scope (don't push back on these)

- A single job spanning backends (no cross-backend gangs). A job runs entirely within one backend.
- Live migration of a *running* task; cross-backend re-placement of a preempted task (drain is the only
  cross-backend move, and it re-runs the job's tasks fresh).
- Cross-backend global DRF fairness in v1 (per-user budgets stay global; per-backend fairness later).
- Cross-backend preemption.
- Replacing Kueue inside a k8s backend.
- Backwards compatibility for the config: single-cluster uses the unchanged top-level form; there is no
  migration of persisted config.

## 10. File summary

| Path | New? | What |
|---|---|---|
| `lib/iris/src/iris/rpc/remote_agent.proto` | new | `RemoteAgentService.Poll` + messages (§1, §1.1) |
| `lib/iris/src/iris/cluster/controller/backends/remote.py` | new | `RemoteTaskBackend` (DB-less adapter, §2) |
| `lib/iris/src/iris/cluster/agent/` | new | agent package: `loop.py`, `cache.py` (§2) |
| `lib/iris/src/iris/cluster/platforms/` | new (moved) | runtime substrate drivers (§3) |
| `lib/iris/src/iris/cluster/setup/` | new (moved) | admin bring-up / ops (§3) |
| `lib/iris/src/iris/cluster/config.py` | edit | `BackendConfig` + `backends:` map (§4) |
| `lib/iris/src/iris/cluster/controller/migrations/00NN_*.py` | new | root DB migration (§5) |
| `lib/iris/src/iris/cluster/controller/auth.py` | edit | `system:controller` mint (§6) |
| `lib/iris/src/iris/cluster/composer.py` | edit | build `RemoteTaskBackend` per `backends:` entry |
