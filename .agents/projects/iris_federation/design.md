# Iris federation: remote clusters as peers (Model D, Track 2)

> **Builds on the DECIDED architecture.** [`iris_backend_contract/delegation_model.md`](../iris_backend_contract/delegation_model.md)
> settled **Model D (2026-07-01)**: a **backend** is a *local execution substrate* that shares the
> controller's one job DAG; a **remote Iris cluster is NOT a backend** — it is a **federation peer**
> that owns its own DAG + backends, and whole root jobs are *handed off* to it. That doc decided the
> *cut*; it deferred the federation project itself as "Track 2 — greenfield, later." **This doc is
> Track 2**: the concrete design for how a user's program flows through the multi-cluster world, and
> where every piece hooks into today's code. It answers five questions the goal poses — child-job
> tracking, parent↔child sync, log forwarding from siloed clusters, visualization, and the backend
> representation — and pins the backend-vs-peer contract so the codebase stays clean and a
> single-cluster user sees *no change at all*.

## 1. The mental model: two downstreams, distinguished by ownership

An Iris controller has exactly two kinds of downstream. They are **not** unified — Model D's whole
point is that forcing them into one abstraction is what created the DAG-ownership confusion.

| | **Backend** (owned) | **Peer / sub-controller** (fire-and-forget) |
|---|---|---|
| What it is | a `TaskBackend` impl in the controller process (`controller/backend.py:455`) | a *full remote Iris* with its own DAG, DB, backends, autoscaler, finelog |
| DAG ownership | **shares** this controller's DAG — its tasks are rows in `jobs`/`tasks`/`task_attempts` with `backend_id` set | **owns its own** DAG; the parent holds only a *handle*, never remote task rows |
| Data movement | in-memory, same SQLite; the backend authors *effects*, the controller folds them (`backend.py:533`, `ReconcileResult.effects` only) | **RPC only**, across a cluster boundary; the parent is a *passive cache* of the peer's reported status |
| Scheduling | the controller's meta-scheduler places every task (`meta_scheduler.route_jobs_to_backends`) | the **peer** schedules; the parent is not in the loop after handoff |
| Budget | trivial local read (controller sees every task) | admission at submit + a spend report pulled back (distributed) |
| Failure blast radius | shares the control loop — a hung backend can stall the tick | **fully isolated** — a dead peer only staleness-freezes its own handles |
| Contract surface | four control methods: `schedule` / `reconcile` / `autoscale` / `status` | two federation ops: **hand off a root job**, **pull status/spend** (+ proxied cancel/logs/exec) |
| Reachability | flat L3 to its workers (`endpoint_proxy` research: "Iris assumes flat L3 within a VPC/k8s") | **one authenticated hop to the peer *controller* only**; the peer's workers are never touched |

**The one-sentence contract.** A *backend* is something this controller **drives** (owns the DAG,
schedules, folds effects, moves data in memory). A *peer* is something this controller **delegates
to** (hands off a whole job tree, then caches what the peer reports). Everything below follows from
that split.

## 2. The full flow of a user's program

One entrypoint (`iris.oa.dev`), transparent fan-out. Trace of a run whose constraints only a remote
cluster can satisfy (e.g. `cluster=cw-us-east`, or an accelerator only CW has):

1. **Submit.** User runs an executor / `iris submit` against the parent. The client builds a
   `LaunchJobRequest` and calls `ControllerService.LaunchJob` (`remote_client.py:205`) — *exactly as
   today, unchanged*. One controller URL, one connection (`RemoteClusterClient`, `remote_client.py:71`).
2. **Route.** The parent's meta-scheduler evaluates the job's constraints. Today it matches them
   against local **backends** (`route_jobs_to_backends`, `meta_scheduler.py:93`). Federation adds a
   second target kind: if the constraints match a **peer** (and not a local backend), the job is a
   *federation candidate*. An explicit `cluster=<peer>` pin forces it.
3. **Hand off.** The parent creates a **local job row** for the federated root with
   `child_cluster=<peer>` set and **no tasks materialized** (§4.1), then synchronously calls the
   peer's `LaunchJob` with an idempotency key, recording the returned `remote_job_id` on the handle
   (§4.2, §5). It returns the *parent's* job id to the user immediately.
4. **Peer runs the whole tree.** The peer materializes the job in **its own** DAG and schedules it on
   **its own** backends. Crucially, the peer injects **its own** `IRIS_CONTROLLER_ADDRESS` into every
   task's env (`runtime/env.py:150`). So when the running program spawns a child job, `get_iris_ctx()`
   connects the in-task client back to the **peer** (`client.py:1190`), and the child's `LaunchJob`
   lands in the **peer's** DAG. **The subtree stays on the peer by construction** — the parent never
   sees the children. This is Model D's "self-enforcing whole-tree lock," and it is *already true* of
   the current code: it is simply what `IRIS_CONTROLLER_ADDRESS` does.
5. **Status.** User polls the parent (`iris status <job_id>` / dashboard). The parent serves the
   **cached** handle status — phase, task-state counts, spend — refreshed by a pull-sync loop (§5),
   annotated `cluster: cw-us-east`.
6. **Logs.** User asks for logs. The parent **proxies** the finelog query to the peer controller,
   which serves it from the peer's local finelog. No log bytes are shipped cross-region (§6).
7. **Exec / profile.** On-demand RPCs against a federated job proxy through the peer controller,
   keyed by the handle, not by `task.backend_id` (§7).
8. **Cancel.** User cancels on the parent; the parent routes a versioned cancel intent to the peer;
   the peer tears down its subtree.
9. **Terminal.** The peer's job finishes; the sync loop caches the final status + final spend and
   **stops syncing**. The handle is now the permanent passive cache — the parent can show the run
   forever without the peer being up.

The user never learns there were two clusters unless they look at the `cluster:` annotation.

## 3. Where this touches the code (nothing, until you add a peer)

The federation machinery is **additive** and **inert on a single-cluster deployment**:

- `jobs.child_cluster` is `NULL`/`""` for every local job (§4.1), so `GetJobStatus`, the scheduler,
  and the dashboard behave byte-identically.
- With no `peers:` configured, the router never produces a federation candidate; `route_jobs_to_backends`
  is unchanged.
- The federation sync loop, the peer proxy, and the `ListPeers` RPC simply have nothing to iterate.

This mirrors the multi-backend dashboard's proven principle
([`iris_multi_backend_dashboard/design.md`](../iris_multi_backend_dashboard/design.md)):
*"single-backend clusters look exactly as today; every multi-* affordance is gated on
`count > 1`."* Federation is the same gate, one level up.

## 4. Q1 — How the parent tracks child-cluster jobs

**Recommendation: yes, add `jobs.child_cluster: str` (a new nullable/`""`-default column), *plus* a
`federated_jobs` sidecar table for the mutable handle state.** The column is the *ownership
discriminator*; the sidecar is the *passive cache*.

### 4.1 `jobs.child_cluster` — the discriminator, a sibling of `backend_id`

`backend_id` (`schema.py:273`) already answers "*which local substrate owns this?*". `child_cluster`
answers the orthogonal "*is this owned locally, or handed off to a peer, and which one?*":

- `child_cluster == ""` → **local job.** Routed to a local backend via `backend_id` as today.
- `child_cluster == "<peer>"` → **federated job.** Handed off; `backend_id` stays `""` (there is no
  local backend). The two columns are mutually exclusive by construction.

**Asymmetry vs `backend_id` (important):** `backend_id` lives on `jobs` **and** `tasks` **and**
`task_attempts` (`schema.py:273/350/401`) because each *local task* carries it. `child_cluster` lives
on **`jobs` only** — a federated root has **no local task rows** (Model D: "never mirror remote task
rows locally"). This asymmetry is the physical embodiment of whole-tree handoff: a peer job is a
single job row with a pointer, not a task subtree.

Why a column and not just a sentinel `backend_id`? Because it is the exact dimension the dashboard
scope selector and `JobQuery` filter on server-side (`JobQuery.backend_id` today → add
`JobQuery.child_cluster`), and routing/GetJobStatus branch on it. Making it a first-class column —
never an overloaded empty string — is the same call the multi-backend design made for `backend_id`
("never an overloaded sentinel").

### 4.2 `federated_jobs` — the sidecar holding the cache/sync state

The hot `jobs` table stays lean (one discriminator column); all federation-only, *frequently
rewritten* state lives in a sidecar keyed by `job_id`. This matches the repo's partition philosophy
(controller-owned `jobs` vs backend-owned `workers` are already separate tables) and isolates the
sync loop's write churn from the scheduler's transactions:

```python
federated_jobs_table = Table(
    "federated_jobs", metadata,
    Column("job_id", JobNameType, ForeignKey("jobs.job_id", ondelete="CASCADE"), primary_key=True),
    Column("peer_id", String, nullable=False),                # == jobs.child_cluster (denormalized for standalone sync reads)
    Column("remote_job_id", String, nullable=False),          # the peer's own JobName; the key for every peer RPC
    Column("idempotency_key", String, nullable=False),        # request digest — exactly-once handoff + safe retry
    Column("owner_principal", String, nullable=False),        # auth identity to assert to the peer on delegated RPCs
    Column("handoff_state", Integer, nullable=False),         # PENDING_HANDOFF | HANDED_OFF | HANDOFF_FAILED
    Column("cached_status_json", String, nullable=False, server_default="{}"),  # last JobStatus summary + task_state_counts
    Column("spend_snapshot_micros", Integer, nullable=False, server_default="0"),
    Column("sync_revision", Integer, nullable=False, server_default="0"),        # cursor for incremental delta pull
    Column("cancel_intent_version", Integer, nullable=False, server_default="0"),
    Column("last_sync_ms", TimestampMsType),
    Column("terminal_error", String),
)
```

These are exactly the handle fields the delegation-model doc enumerated
(`delegation_model.md:273-280`), now typed against the schema. **Never** add a `tasks` row for a
federated job — `cached_status_json` carries the peer's task-state counts; that is the whole "passive
cache."

### 4.3 Insertion points (grounded)

| Concern | Where | Change |
|---|---|---|
| Schema | `schema.py:255` (jobs), new `federated_jobs_table` | add `child_cluster` col + sidecar table |
| Migration | new `migrations/0034_federation.py` | `ALTER TABLE jobs ADD COLUMN child_cluster` + `CREATE TABLE federated_jobs`; template is `0033_backend_id.py` verbatim (idempotent `_has_column` guard, `""` backfill) |
| Route | `controller.py:1011` `_route_pending` → `meta_scheduler.route_jobs_to_backends` | add peer as a target kind; a federation candidate stamps `child_cluster` instead of `backend_id` |
| Write (handoff) | new `FederationManager.submit` invoked from `launch_job` (`service.py:1097`) after routing, *or* stamped at the routing commit next to `writes.stamp_backend` (`writes.py:208`, `controller.py:1172`) | insert `jobs.child_cluster` + `federated_jobs` row, call peer `LaunchJob` |
| Read | `reads.get_job_detail` (`reads.py:495`, already selects `jobs.backend_id` at `:512`) | also select `child_cluster`; left-join `federated_jobs` for federated ids |
| Proto | `job.proto:328` `JobStatus`, next free field **35** | `string child_cluster = 35;` (sibling of `backend_id = 34`) |
| Response | `get_job_status` (`service.py:1443`), the federated branch | for `child_cluster != ""`, serve counts from `cached_status_json` instead of a local `task_summaries_for_jobs` read (`service.py:1460`) |
| Dashboard | §8 | one `cluster:` column + scope, cloned from the `backend_id` UI |

## 5. Q2 — Synchronization between parent and child

Two distinct channels, deliberately different in shape.

### 5.1 Submission = **synchronous handoff**, not a queue

The parent (a) durably writes the `jobs`+`federated_jobs` handle in one local transaction, then (b)
**synchronously** calls the peer's `LaunchJob` with the `idempotency_key`. On success it records
`remote_job_id` and flips `handoff_state → HANDED_OFF`. The user's client blocks only for one RPC —
the same interactive contract as a local submit (which is one SQLite txn + recheck,
`service.py:1407`).

*Why synchronous, not a general queue:* submission is user-facing; the client wants a job id back.
Durable-handle-first + idempotent-retry gives resilience **without** a queue's complexity:

- If the peer is **unreachable**, the handle persists in `PENDING_HANDOFF`; a background retry re-sends
  with the *same* `idempotency_key` (the peer dedups). The parent surfaces the job as
  `pending (handing off to <peer>)` — never lost, never double-submitted.
- If the parent **crashes mid-handoff**, recovery re-drives `PENDING_HANDOFF` handles on boot; the
  idempotency key makes the re-send safe.

This is the "exactly-once root handoff" the delegation doc flagged (`delegation_model.md:346`), scoped
to one durable state machine. A true queue is unnecessary because there is exactly one hop and the
handle *is* the durable record.

### 5.2 Status = **pull-based incremental sync**, parent-driven

A `FederationManager` sync loop (one background thread; peers iterated per tick, like the pruner loop)
**pulls** status for its *active* handles. Direction and shape:

- **Pull, not push.** The parent pulls; the **peer is a vanilla Iris that does not know it is
  federated** — it just answers `GetJobStatus`/`GetJobState`. This keeps *all* federation logic on the
  parent and requires **zero peer-side code** to start (a peer is any Iris with an ingress). Push
  (peer notifies parent on terminal transition) is a later latency optimization, not the baseline.
- **Incremental, batched.** Baseline: one batched RPC per peer carrying the active `remote_job_id`s →
  their `JobStatus` summaries (`GetJobState` is already the lightweight batch-state RPC the dashboard
  uses). Upgrade: a `since_revision` cursor (`federated_jobs.sync_revision`) so the peer returns only
  jobs changed since the last pull — avoids re-shipping stable summaries. On **reconnect / parent
  restart**, a full re-list reconciles the active set (bulk), then steady-state goes back to delta.
- **Cadence, adaptive.** Active handles sync on a short interval (a few seconds, matching the
  dashboard's own 5–10s job refresh); idle/near-terminal back off; **terminal handles stop syncing
  entirely** — the final `cached_status_json` is the permanent passive cache. This bounds cross-cluster
  RPC to live work only.
- **Spend / budget.** Admission at submit: the parent enforces the per-user global cap *before*
  handoff (it knows the job's max-band, `service.py:1161`), aggregating local spend + cached federated
  `spend_snapshot`. During the run the peer's spend rides back on the same status pull. Overspend bound
  = one sync interval of peer spend — acceptable; a reservation protocol is a later hardening
  (`delegation_model.md:419`).

### 5.3 Cancel / preemption = versioned intent, routed

Cancel on the parent bumps `cancel_intent_version` on the handle and routes an idempotent
`CancelJob(remote_job_id)` to the peer; the sync loop confirms the peer reached a terminal state.
Versioning makes a retried/late cancel a no-op. This replaces today's single-transaction subtree kill
(`ops/job.py:284`) *only at the federation boundary* — local cancel is unchanged.

## 6. Q3 — Log forwarding from internet-siloed child clusters

**The key realization: the child *controller* is reachable; only the child *workers* are siloed.**
That is the normal Iris topology already — controllers have an ingress (IAP/SSH-tunnel, how the CLI
reaches them today), workers are internal and never publicly reachable. So the child controller is the
**sole ingress/egress for its whole cluster**, and federation needs exactly **one** reachable address
per peer: the peer controller.

### 6.1 Nothing changes inside the child cluster

Each Iris controller **already runs its own finelog server** (`/system/log-server`;
`log_stack.py`, `main.py:98`). The child's siloed workers ship logs to the **child's** finelog
*intra-cluster* — the k8s `logship` sidecar resolves the finelog address from the *child* controller's
endpoint registry and pushes directly (`logship.py:315`), exactly as today. Log **keys** are already
cluster-agnostic (`/user/<job>/<task>:<attempt>`, `/system/worker/<id>`; `log_keys.py`), so no key
scheme changes. **No worker ever talks to the parent, and no log bytes cross the cluster boundary on
the write path.**

### 6.2 Reads = query-time proxy through the peer controller (no shipping)

The parent does **not** ingest child logs. When a user reads logs for a federated job, the parent
**proxies the finelog query** to the peer controller, which answers from its local finelog. This is a
one-hop generalization of the mechanism the dashboard *already uses*: today the browser reads logs via
the controller's `EndpointProxy` at `/proxy/system.log-server/finelog.logging.LogService/FetchLogs`
(`dashboard.py:298`, `endpoint_proxy.py`). Federation adds a **peer proxy** route —
`parent/proxy/peer/<peer>/system.log-server/...` — that forwards to the peer controller's *own*
`EndpointProxy`, which resolves `/system/log-server` internally. Same httpx/Starlette reverse-proxy
shape, one extra authenticated hop parent→peer-controller.

- **Key translation.** The parent's job id ≠ the peer's job id. The proxy rewrites the `FetchLogs`
  `source` from the parent `JobName` to the handle's `remote_job_id` before forwarding (a
  `federated_jobs` lookup). The peer answers in its own namespace; the parent relabels on the way back.
- **Cost.** Query-time proxy honors the repo's cross-region bandwidth rule (`AGENTS.md`: "never read
  or write large amounts of data across GCS regions") — logs move only when a human is looking, and
  only the lines requested. Bulk log shipping to a central store is explicitly rejected as the default.
- **Auth.** finelog itself has **no auth** (`app.rs:100`, confirmed) — which is *fine* because it is
  never exposed cross-cluster. The only cross-cluster surface is the peer *controller's* authenticated
  `EndpointProxy`; the parent authenticates to the peer over the peer link (§9), and the peer's finelog
  stays private behind it. This closes the "finelog can't be safely exposed publicly" gap the log
  research flagged.

### 6.3 The fully-siloed-controller case (deferred)

If a child *controller* itself cannot accept inbound connections (behind NAT, no ingress), the peer
link must be **reverse-dialed**: the child controller establishes an outbound connection to the parent
and the parent multiplexes RPCs/proxying over it. This is the unbuilt "relay+agent / `RemoteAgent`
transport" (the `BackendConfig.transport="remote"` seam that `validate_config` currently rejects,
`config.py:864`). The `FederationPeer` abstraction (§9) holds *a connection* regardless of who dialed,
so this is a transport swap under a stable interface — a later PR, not a baseline requirement. The
baseline assumes a reachable peer controller (the common case).

## 7. Q7 — On-demand RPCs (exec / profile / process status) against a federated job

Today these are **controller-mediated** — the client hits the controller, which resolves
`task→worker→address` and forwards to the worker (`service.py:2136/2348/2541`; no client→worker path
exists). For a federated job the target task lives in the *peer's* DAG, so the parent cannot resolve
it. **Recommendation: proxy through the peer controller** (not redirect the client):

- The parent's on-demand handler checks the target job's `child_cluster`. If set, it forwards the RPC
  to the peer controller (keyed by the handle's `remote_job_id`), which does its *own* normal
  `task→worker` resolution internally and returns the result. Uniform auth (the parent stays the trust
  boundary), one endpoint for the user, no client reach/creds into the peer.
- Redirect (hand the client the peer address) is the alternative — parent off the hot path, but the
  client needs direct reach + peer auth, which siloed peers may deny. Proxy is the safer baseline; this
  is the open "proxy vs redirect" question from the spec (`spec.md:216`), resolved toward **proxy** for
  the siloed threat model.

## 8. Q4 — Visualization

**Reuse the existing jobs display; add `cluster` as a scope dimension — the exact parallel of the
`backend_id` UI that already ships.** The multi-backend dashboard already built every piece we need
(the dashboard survey confirmed the full plumbing):

- **Jobs list**: add a **Cluster** column, gated `multiCluster && !clusterScope`, cloned from the
  Backend column (`JobsTab.vue:534-637`). A federated row renders `cluster: cw-us-east`; clicking it
  sets `?cluster=`. Local jobs show nothing (empty `child_cluster`, column hidden by the gate).
- **Job/Task detail**: a `Cluster` `InfoRow`, gated `multiCluster && job.childCluster`, cloned from the
  Backend `InfoRow` (`JobDetail.vue:960`). The federated job detail shows the **cached** counts +
  spend, plus a **"view on cw-us-east ↗"** deep-link (the peer's own dashboard, `ClusterManifest.dashboard_url`)
  and **proxied** log/exec panels (§6, §7) that Just Work through the peer proxy.
- **Scope selector**: extend `BackendScope.vue` / `useBackends.ts` to an `All ▾` selector over
  *local backends + peers*, writing `?cluster=`/`?backend=`. Server-side filter via `JobQuery.child_cluster`.
- **Proto**: `JobStatus.child_cluster = 35` (§4.3), stamped in `get_job_status` next to
  `backend_id` (`service.py:1490`); TS mirror in `types/rpc.ts`.

For the federated job's **tasks**, the parent has no task rows — the task table renders the cached
per-state counts as a summary (the segmented progress bar the jobs list already draws from
`task_state_counts`), with a "detailed tasks live on cw-us-east ↗" link rather than a mirrored task
table. This is the visual expression of "passive cache, not a DAG subtree."

## 9. Q5 — The backends representation, and the parallel peers representation

**Backends are unchanged.** A backend is a local `TaskBackend` (`schedule`/`reconcile`/`autoscale`/
`status`) surfaced by `ListBackends → BackendSummary` (`controller.proto:588`, `service.py:2830`) and
the **Backends** tab. In-flight **PR #6773** generalizes the k8s-special-cased `get_cluster_status()`
into a uniform `TaskBackend.status()` returning a `BackendStatus` oneof (`kubernetes | worker`) — the
right shape, and orthogonal to federation. A peer is **not** a `BackendStatus` variant; it is a
separate concept with a separate surface:

```proto
message PeerSummary {
  string peer_id = 1;
  string controller_address = 2;
  string dashboard_url = 3;
  bool reachable = 4;              // last peer-link probe
  int64 last_sync_ms = 5;
  int32 active_federated_jobs = 6;
  int64 aggregate_spend_micros = 7;
  repeated string advertised_regions = 8;      // what the parent's router matches on
  map<string, StringList> advertised_attributes = 9;  // accelerators the peer offers
}
rpc ListPeers(ListPeersRequest) returns (ListPeersResponse);  // peers[], parallel to ListBackends
```

The dashboard grows a **Clusters** (peers) overview beside **Backends** — one card per peer
(reachability dot, last-sync, active-handle count, aggregate spend, advertised capability chips,
deep-link to the peer dashboard). Two lists because ownership differs: **Backends = what I run;
Clusters = what I delegate to.** Presenting them as one table would releak the exact
backend-vs-peer confusion Model D resolved.

### 9.1 The peer registry and the router's second target kind

- **Config.** Peers are declared in cluster config — a new `peers:` section (peer id →
  controller address + auth + advertised capabilities), the gap the client survey found: today
  `rigging.ClusterManifest` models *one* cluster's identity/auth and **no manifest lists sibling
  controllers** (`cluster_manifest.py`). Federation fills it with a *manifest of peers*. A peer's
  advertised regions/accelerators feed the parent's routing index.
- **Router.** `route_jobs_to_backends` gains a peer arm: build the constraint index over
  *backends ∪ peers*; a job matching only a peer becomes a federation candidate; an explicit
  `cluster=<peer>` constraint forces it; ambiguity (matches both) resolves by policy (prefer local, or
  require an explicit pin). This reuses the existing constraint-matcher and the `region ANY/PINNED`
  machinery — peers are just another set of match targets.
- **Connection.** `RemoteClusterClient` already encapsulates "one connection to one controller"
  (`remote_client.py:71`); a `FederationPeer` holds one per peer, keyed by peer id — the natural place
  the reverse-dial transport (§6.3) plugs in later.

## 10. Auth / trust across the peer link

The parent authenticates the user as today (`service.py:1152`), then **delegates** to the peer: the
peer trusts the parent as a principal and the parent asserts the end-user identity on each handoff /
proxied RPC (`federated_jobs.owner_principal`). This reuses the rigging `server_auth` verifier +
`credentials_for` provider work already carved out for cross-service auth (the
[cluster-admin-unification](../../MEMORY.md) auth split). One authenticated connection per peer; the
peer applies its own RBAC to the asserted identity. Peer trust config lives with the `peers:` registry.

## 11. Rollout — clean, and invisible to single-cluster users

Federation is **Track 2**, independent of the local backend hygiene (**Track 1**: P5 WorkerJobService,
P7 published status, P8 autoscaler-single-writer — see `iris_backend_contract/spec.md`). They share
only the router (which grows the peer arm). Suggested sequence, each PR inert until the next:

1. **Schema + handle.** `jobs.child_cluster` + `federated_jobs` + migration 0034; `JobStatus.child_cluster`;
   `get_job_status` federated branch reading the cache. *No behavior yet* — no peer can be configured, so
   every job is local and the column is always `""`. Pure additive, single-cluster byte-identical.
2. **Peer registry + router arm.** `peers:` config, `FederationPeer`, the `route_jobs_to_backends` peer
   target, `ListPeers`/`PeerSummary`. Still inert with zero peers configured.
3. **Handoff + sync.** `FederationManager`: synchronous durable handoff (§5.1) + pull sync loop (§5.2)
   + versioned cancel (§5.3) + budget admission. First end-to-end federated job (parent + one peer,
   both plain Iris).
4. **Proxies.** Peer log proxy (§6) + on-demand RPC proxy (§7), generalizing `EndpointProxy`.
5. **Dashboard.** `cluster` scope + column + detail row + Clusters/peers overview (§8), cloned from the
   multi-backend UI. Screenshot smoke: single-cluster screenshots must be **unchanged**.
6. **(Later) reverse-dial transport** for fully-siloed peer controllers (§6.3), and push-on-terminal
   as a sync latency optimization.

The contract that keeps the codebase clean: **the backend seam never learns federation exists** (no
`RemoteTaskBackend`, no remote-safe store, no per-backend DB split — all deleted by Model D), and
**federation never touches the DAG fold** (it holds handles, not tasks). Two honest seams, each simple,
meeting only at the router.

## 12. Open questions

1. **Ambiguous routing policy.** When a job's constraints match *both* a local backend and a peer, is
   the default "prefer local" or "require an explicit `cluster=` pin"? (Leaning prefer-local; peers are
   for capacity a local backend lacks.)
2. **Budget admission strength.** Report-and-throttle (baseline, one-interval overspend) vs a
   grant/reservation protocol before handoff. Start with report-and-throttle; revisit under real
   multi-tenant peer load.
3. **Sync cadence & scale.** Concrete active-handle poll interval and back-off curve; batch size per
   `GetJobState` pull; when the `since_revision` delta becomes necessary vs the full batched pull.
4. **Peer capability freshness.** Peers advertise regions/accelerators to the parent's router — pushed
   in config, or pulled/refreshed live (a peer that loses a pool should stop attracting routes)?
5. **Handle GC / retention.** Terminal handles are the permanent cache — do they age out (and lose the
   run's history), or persist indefinitely like a completed local job?
6. **Reverse-dial priority.** Is any real target cluster a fully-siloed *controller* (needing §6.3
   now), or do all near-term peers have a reachable controller ingress (baseline suffices)?
