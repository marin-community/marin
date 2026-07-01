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
> single-cluster user's workflow is unchanged. **Revised 2026-07-01 after review** — see §12 for the
> decisions folded in (task-level cache, global finelog, bulk delta-sync, always-on combined UI).

## 1. The mental model: two downstreams, distinguished by ownership

An Iris controller has exactly two kinds of downstream. They are **not** unified — Model D's whole
point is that forcing them into one abstraction is what created the DAG-ownership confusion.

| | **Backend** (owned) | **Peer / sub-controller** (fire-and-forget) |
|---|---|---|
| What it is | a `TaskBackend` impl in the controller process (`controller/backend.py:455`) | a *full remote Iris* with its own DAG, DB, backends, autoscaler, finelog |
| DAG ownership | **shares** this controller's DAG — its tasks are rows in `jobs`/`tasks`/`task_attempts` with `backend_id` set, folded by the scheduler | **owns its own** DAG; the parent holds a *handle* + a **read-only cached projection** of the peer's jobs/tasks that the scheduler/fold **never read** |
| Data movement | in-memory, same SQLite; the backend authors *effects*, the controller folds them (`backend.py:533`, `ReconcileResult.effects` only) | **RPC only**, across a cluster boundary; the parent is a *passive cache* refreshed by a delta-sync protocol |
| Scheduling | the controller's meta-scheduler places every task (`meta_scheduler.route_jobs_to_backends`) | the **peer** schedules; the parent is not in the loop after handoff |
| Budget | trivial local read (controller sees every task) | admission at submit + a spend report ridden back on the sync (distributed) |
| Failure blast radius | shares the control loop — a hung backend can stall the tick | **fully isolated** — a dead peer only staleness-freezes its own handles |
| Contract surface | four control methods: `schedule` / `reconcile` / `autoscale` / `status` | two federation ops: **hand off a root job**, **bulk delta-sync status/spend** (+ routed cancel, proxied exec; logs go to a shared finelog) |
| Reachability | flat L3 to its workers (`endpoint_proxy` research: "Iris assumes flat L3 within a VPC/k8s") | **one authenticated hop to the peer *controller* only** (it relays logs out + fronts exec in); the peer's workers are never touched by the parent |

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
   second target kind: if the constraints match a **peer** — and no local backend can *currently*
   schedule them — the job is a *federation candidate*. **Prefer-local is the rule** (decided): a job
   a local backend can run stays local; a peer is only chosen for capacity/hardware the local
   backends lack. An explicit `cluster=<peer>` pin forces a peer.
3. **Hand off.** The parent creates a **local job row** for the federated root with
   `child_cluster=<peer>` set and **no authoritative task rows** (§4.1), then synchronously calls the
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
   **cached projection** — job phase, spend, and **per-task state** mirrored into a read-only
   `federated_tasks` table (§4.2) — so the *same* job/task views a local job gets render natively,
   annotated `cluster: cw-us-east`. The projection is refreshed by a bulk delta-sync (§5.2).
6. **Logs.** User asks for logs. They resolve from the **shared global finelog** (§6): the peer
   controller relays its (siloed) workers' log batches out to the one finelog service, and reads —
   from the parent, the peer, or any client — hit that single store. No per-peer log proxy.
7. **Exec / profile.** On-demand RPCs against a federated job proxy through the peer controller,
   keyed by the handle, not by `task.backend_id` (§7).
8. **Cancel.** User cancels on the parent; the parent routes a versioned cancel intent to the peer;
   the peer tears down its subtree.
9. **Terminal.** The peer's job finishes; the sync caches the final status + spend and stops active
   polling. The parent's cache **mirrors the peer's retention**: when the peer eventually prunes the
   finished job, the next sync carries a tombstone and the parent drops its projection too (§5.3) —
   the parent shows exactly what the peer still has, no more, no separate GC policy.

The user never learns there were two clusters unless they look at the `cluster:` annotation.

## 3. Where this touches the code (nothing, until you add a peer)

The federation machinery is **additive** and **inert on a single-cluster deployment**:

- `jobs.child_cluster` is `NULL`/`""` for every local job (§4.1), so `GetJobStatus`, the scheduler,
  and the routing path behave byte-identically.
- With no `peers:` configured, the router never produces a federation candidate; `route_jobs_to_backends`
  is unchanged.
- The federation sync loop, the peer relay, and the `ListPeers` RPC simply have nothing to iterate.

The *behavior* is inert, but the *UI is not gated* on a peer count (per review — gating adds
complexity for little payoff): the dashboard always renders one combined **execution-targets** view
(backends + peers) and a `cluster` annotation that is simply empty for local jobs (§8). A
single-cluster deployment therefore sees the same tab with one card in it, not a different UI.

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
`task_attempts` (`schema.py:273/350/401`) because each *local task* carries its owning backend as a
routing pin. `child_cluster` lives on **`jobs` only** — it is a whole-*tree* handoff property, not a
per-task one; the peer owns the tasks. The parent keeps a *read-only cached projection* of the peer's
tasks (in a separate `federated_tasks` table, §4.2), but those are **not** authoritative `tasks` rows:
the scheduler and the DAG fold never read them. This is the precise reading of Model D's "never mirror
remote task rows" — the rule forbids mirroring rows *into the DAG the fold operates on*; a read-only
display projection the fold never touches is safe and is what makes the parent a real cache.

Why a column and not just a sentinel `backend_id`? Because it is the exact dimension the dashboard
scope selector and `JobQuery` filter on server-side (`JobQuery.backend_id` today → add
`JobQuery.child_cluster`), and routing/GetJobStatus branch on it. Making it a first-class column —
never an overloaded empty string — is the same call the multi-backend design made for `backend_id`
("never an overloaded sentinel").

### 4.2 `federated_jobs` (handle) + `federated_tasks` (read-only task projection)

Two federation-only tables, keyed by `job_id`, holding all the *frequently rewritten* sync state
apart from the hot `jobs` table (matching the repo's partition philosophy — controller-owned `jobs`
vs backend-owned `workers` are already separate tables — and isolating sync-write churn from the
scheduler's transactions). **`federated_jobs`** is the handle + job-level summary; **`federated_tasks`**
is the read-only per-task cache that lets the parent render a real task table.

```python
federated_jobs_table = Table(
    "federated_jobs", metadata,
    Column("job_id", JobNameType, ForeignKey("jobs.job_id", ondelete="CASCADE"), primary_key=True),
    Column("peer_id", String, nullable=False),                # == jobs.child_cluster (denormalized for standalone sync reads)
    Column("remote_job_id", String, nullable=False),          # the peer's own JobName; the key for every peer RPC
    Column("idempotency_key", String, nullable=False),        # request digest — exactly-once handoff + safe retry
    Column("owner_principal", String, nullable=False),        # auth identity to assert to the peer on delegated RPCs
    Column("handoff_state", Integer, nullable=False),         # PENDING_HANDOFF | HANDED_OFF | HANDOFF_FAILED
    # Job-level summary (small, fixed size — counts, not per-task):
    Column("job_state", Integer, nullable=False),
    Column("task_count", Integer, nullable=False, server_default="0"),
    Column("completed_count", Integer, nullable=False, server_default="0"),
    Column("failure_count", Integer, nullable=False, server_default="0"),
    Column("spend_snapshot_micros", Integer, nullable=False, server_default="0"),
    Column("sync_cursor", String, nullable=False, server_default=""),           # peer's delta watermark (§5.2)
    Column("cancel_intent_version", Integer, nullable=False, server_default="0"),
    Column("last_sync_ms", TimestampMsType),
    Column("terminal_error", String),
)

# Read-only projection. Written ONLY by the sync loop; the scheduler and the DAG fold never read it.
federated_tasks_table = Table(
    "federated_tasks", metadata,
    Column("job_id", JobNameType, ForeignKey("federated_jobs.job_id", ondelete="CASCADE"), nullable=False),
    Column("task_index", Integer, nullable=False),
    Column("state", Integer, nullable=False),
    Column("worker_label", String, nullable=False, server_default=""),   # opaque peer-side worker name, for display only
    Column("exit_code", Integer),
    Column("error", String),
    Column("started_at_ms", TimestampMsType),
    Column("finished_at_ms", TimestampMsType),
    PrimaryKeyConstraint("job_id", "task_index"),
)
```

`federated_jobs` carries the handle fields the delegation-model doc enumerated
(`delegation_model.md:273-280`); the job summary is inherently small (state + four counts, **not**
per-task, so it does not grow with task count). Per-task detail lives in `federated_tasks` rows, which
scale exactly like the local `tasks` table (SQLite handles millions of rows), and the sync only writes
the tasks that *changed* (§5.2), so a 5 000-task job costs one row-set once and then only deltas.

**Why a task projection at all — the compare/contrast the review asked for.** The first draft cached
only job-level counts and deep-linked to the peer for tasks. That is minimal but a usability cliff: a
federated job's detail page can't show the per-task table, worker, exit code, or timing that every
local job shows. The alternative is to mirror the peer's task state into the parent DB:

| | A — summary only (first draft) | **B — cached task projection (chosen)** |
|---|---|---|
| Parent state | job counts on `federated_jobs` | + per-task rows in read-only `federated_tasks` |
| Job/task detail UX | segmented bar + "view on peer ↗" | native task table & task-detail, identical to a local job |
| Sync payload | job summary per job | job summary + *changed* task rows (delta-bounded, §5.2) |
| Model D safety | trivially safe | safe **because** the scheduler/fold never read `federated_tasks` — enforced by table separation, not convention |
| Cost | minimal | O(changed tasks) sync + local rows; bounded by the delta protocol |

**B is chosen for usability.** The Model D boundary is preserved by *where* the rows live: a separate
projection table the fold never queries, not the authoritative `tasks` table. Attempt-level drill-down
(the per-attempt history) is not mirrored — it stays a deep-link to the peer, keeping the projection
to one row per task.

### 4.3 Insertion points (grounded)

| Concern | Where | Change |
|---|---|---|
| Schema | `schema.py:255` (jobs), new `federated_jobs_table` + `federated_tasks_table` | add `child_cluster` col + the two projection tables |
| Migration | new `migrations/0034_federation.py` | `ALTER TABLE jobs ADD COLUMN child_cluster` + `CREATE TABLE federated_jobs` + `federated_tasks`; template is `0033_backend_id.py` verbatim (idempotent `_has_column` guard, `""` backfill) |
| Route | `controller.py:1011` `_route_pending` → `meta_scheduler.route_jobs_to_backends` | add peer as a target kind; a federation candidate stamps `child_cluster` instead of `backend_id` |
| Write (handoff) | new `federation.FederationManager.submit` invoked from `launch_job` (`service.py:1097`) after routing, *or* stamped at the routing commit next to `writes.stamp_backend` (`writes.py:208`, `controller.py:1172`) | insert `jobs.child_cluster` + `federated_jobs` row, call peer `LaunchJob` |
| Sync write | `FederationManager` sync loop (§5.2) | upsert `federated_jobs` summary + changed `federated_tasks` rows; apply tombstones |
| Read (job) | `reads.get_job_detail` (`reads.py:495`, already selects `jobs.backend_id` at `:512`) | also select `child_cluster`; left-join `federated_jobs` for federated ids |
| Read (tasks) | `get_job_status`/`list_tasks` (`service.py:1443`, `task_summaries_for_jobs` `:1460`) | for `child_cluster != ""`, read the summary from `federated_jobs` and the task list from `federated_tasks` instead of `tasks` |
| Proto | `job.proto:328` `JobStatus`, next free field **35** | `string child_cluster = 35;` (sibling of `backend_id = 34`) |
| Dashboard | §8 | always-on `cluster:` annotation + combined execution-targets view |

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

### 5.2 Status = **one bulk delta-sync RPC per peer** (`FederationSync`)

A `FederationManager` sync loop (one background thread; peers iterated per tick, like the pruner loop)
refreshes the projection with **one RPC per peer per tick**, not one RPC per job. The review's ask
was a bulk-update protocol that doesn't sprawl into cruft — so federation adds exactly **one**
purpose-built endpoint the peer serves and the parent calls:

```proto
// Served by every peer (a small, self-contained federation surface).
message FederationSyncRequest {
  string requester_id = 1;   // the federating cluster's id (auth + which handoffs to report)
  string cursor       = 2;   // opaque per-(peer,requester) watermark; "" on first call / after restart
}
message FederationJobDelta {
  string remote_job_id = 1;
  JobStatus summary    = 2;              // job state + counts + spend (reuses the existing message)
  repeated TaskStatus changed_tasks = 3; // only tasks whose state changed since the cursor
  bool tombstone       = 4;              // peer has pruned this job → parent drops its projection (§5.4)
}
message FederationSyncResponse {
  repeated FederationJobDelta deltas = 1;
  string next_cursor = 2;    // persisted to federated_jobs.sync_cursor; passed back next tick
}
rpc FederationSync(FederationSyncRequest) returns (FederationSyncResponse);
```

- **Bulk delta, cursor-driven.** The peer returns *only what changed* since `cursor` across **all**
  jobs this parent handed it — job summaries and the specific tasks that transitioned — plus a
  `next_cursor`. The parent applies the batch in one transaction (upsert `federated_jobs` +
  changed `federated_tasks`), advances the cursor, done. Steady state is one small RPC per peer;
  the payload is bounded by *churn*, not by job or task count. On **first call / after a parent
  restart** the cursor is empty and the peer replays the full active set (bulk reconcile), then
  reverts to deltas — so restart recovery and steady state are the *same* code path, just a different
  starting cursor. This is the whole protocol; there is no second per-job status RPC to maintain.
- **Peer-side cost is contained.** The peer computes the delta from a monotonic per-job revision it
  already needs for its own change-tracking (a watermark bumped on every job/task transition); it does
  **not** grow federation state per requester beyond the cursor. This is the one piece of genuinely
  new peer-side code, and it is deliberately one RPC + one watermark — not a mirror of the parent's DAG.
- **Transport-agnostic (per review).** `FederationSync` is a request/response over the peer link; it
  works **identically** whether the parent dials the peer (baseline) or the peer dials the parent and
  the parent calls back over that reverse channel (§6.3). The delta protocol is designed once and is
  independent of who opened the socket — so reverse-dial can be deferred without redesigning sync.
- **Cadence, adaptive.** Active peers sync on a short interval (a few seconds, matching the dashboard's
  5–10s job refresh); a peer with no active handoffs backs off. There is no "terminal handles sync
  forever" — a job goes quiet the moment it is terminal (no more deltas) and is dropped when the peer
  tombstones it (§5.4).
- **Spend / budget.** Admission at submit: the parent enforces the per-user global cap *before* handoff
  (it knows the job's max-band, `service.py:1161`), aggregating local spend + cached federated spend.
  During the run each job's spend rides back inside its `FederationJobDelta.summary`. Overspend bound =
  one sync interval of peer spend — acceptable; a reservation protocol is a later hardening
  (`delegation_model.md:419`).

### 5.3 Cancel / preemption = versioned intent, routed

Cancel on the parent bumps `cancel_intent_version` on the handle and routes an idempotent
`CancelJob(remote_job_id)` to the peer; the next `FederationSync` confirms the peer reached a terminal
state. Versioning makes a retried/late cancel a no-op. This replaces today's single-transaction subtree
kill (`ops/job.py:284`) *only at the federation boundary* — local cancel is unchanged.

### 5.4 Retention = the parent mirrors the peer (no separate GC)

The parent's projection is a cache of what the peer *still holds*, so its lifetime is the peer's
lifetime — the parent needs **no retention policy of its own** (per review). When the peer's normal
job pruning removes a finished job, the next `FederationSync` carries a `tombstone` for it; the parent
deletes the `federated_jobs`/`federated_tasks`/`jobs` rows for that handle in the same batch. Want
longer history on the parent? Lengthen the *peer's* retention — the parent follows. This reuses the
peer's existing pruning (`pruner.py`) as the single source of truth for job lifetime and avoids two
divergent GC clocks.

## 6. Q3 — Logs: one shared global finelog, fed by peer-controller relays

**Decision (per review): a single *global* finelog service, not per-cluster logs behind a query
proxy.** Every cluster's logs land in one store, so reads are uniform — from the parent, a peer, or a
laptop — and work even when a peer controller is down or slow. The enabling fact is the same as
elsewhere: the child *controller* has egress (it reaches the global service); only the child *workers*
are siloed. So the child controller is the **egress relay** for its silo's logs.

### 6.1 Ingest inside the child is unchanged; the controller relays it out

The child's siloed workers ship logs to the **child controller's** finelog ingest *intra-cluster* —
the k8s `logship` sidecar resolves `/system/log-server` from the *child* controller's endpoint
registry and pushes directly (`logship.py:315`), exactly as today (no worker ever needs to reach the
global service). The change is one level up: the child controller's `LogStack` (`log_stack.py`) is
configured as a **store-and-forward relay** — instead of (or in addition to) storing locally, it
forwards each batch to the shared global finelog over its egress. Log **keys** are already
cluster-agnostic (`/user/<job>/<task>:<attempt>`, `/system/worker/<id>`; `log_keys.py`), so many
clusters multiplex into one store with **no key remapping** — the parent's `remote_job_id` *is* the
key the peer already wrote under. A local buffer at the child absorbs transient egress blips and
retries; the global store is the single durable, queryable surface. finelog's existing remote-archive
tiering (GCS) bounds hot storage in the global service.

### 6.2 Reads hit the one global finelog

There is no per-peer log proxy. The parent (and every client) queries the shared finelog exactly as a
single-cluster controller queries its own today — via `FetchLogs`/`StatsService.Query` through the
`EndpointProxy` (`dashboard.py:298`). Because the store is global and keys are cluster-agnostic, a
federated job's logs, a local job's logs, and even a **cross-cluster** query ("all failures for user
X across every cluster") are one query against one store. The parent still translates its own job id →
the handle's `remote_job_id` when building the `FetchLogs` source (a `federated_jobs` lookup), because
that is the key the peer wrote under.

### 6.3 What the global store requires (honest costs)

- **Auth — a real addition.** A globally shared finelog receives pushes from *many* controllers across
  the internet, so it can no longer rely on being private behind one controller's `EndpointProxy`
  (finelog has **no auth** today, `app.rs:100`). The global finelog must be fronted by an authenticated
  ingress — reuse the rigging `server_auth` verifier so each relaying controller authenticates its
  pushes, or co-locate it with a controller and reach it only through that controller's authed proxy.
  This is net-new work the per-cluster model avoided; it is the price of a single store.
- **Cross-region egress.** Relaying every cluster's logs to one region is real bandwidth (the
  `AGENTS.md` cost concern), unlike a query-time proxy that moved bytes only on demand. It is the
  deliberate trade for a uniform, always-available, cross-cluster-queryable log surface. finelog
  already batches/compresses and tiers cold segments to GCS, which bounds the steady-state cost; if a
  specific peer's volume is prohibitive, its relay can be configured to tier locally and forward only
  on query (a per-peer fallback), but the default is relay-to-global.

### 6.4 The fully-siloed-controller case (deferred)

If a child *controller* itself cannot make outbound connections (behind NAT, no egress), its relay
can't reach the global finelog and its exec/status link can't be dialed either. Then the peer link is
**reverse-dialed**: the child controller opens one outbound connection to the parent and everything
(the `FederationSync` calls of §5.2, exec proxying of §7, and log relay) is multiplexed over it. This
is the unbuilt "relay+agent / `RemoteAgent` transport" (the `BackendConfig.transport="remote"` seam
`validate_config` currently rejects, `config.py:864`). The `FederationPeer` abstraction (§9) holds *a
connection* regardless of who dialed, and — importantly — the `FederationSync` delta protocol is
identical either way (§5.2), so reverse-dial is a transport swap under a stable interface, a later PR
rather than a baseline requirement.

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

**Reuse the existing jobs display; surface `cluster` as an annotation — always on, no multi-cluster
gate (per review).** The first draft gated every affordance on `count > 1`; the review's call is that
the gate buys too little for its complexity, so federation renders the same UI whether there are zero
peers or ten — a single-cluster deployment just sees one execution target and empty `cluster:` cells.

- **Jobs list**: a **Cluster** column, always rendered (`JobsTab.vue:534-637`), showing
  `cluster: cw-us-east` for a federated job and blank for a local one; clicking it filters `?cluster=`.
- **Job/Task detail**: a `Cluster` `InfoRow` (`JobDetail.vue:960`) and — because the parent now holds a
  read-only task projection (§4.2) — the **native task table and task-detail pages render exactly as
  they do for a local job** (state, worker label, exit code, timing), read from `federated_tasks`
  instead of `tasks`. Attempt-level drill-down deep-links to the peer (`ClusterManifest.dashboard_url`),
  the one place the projection stops.
- **Scope selector**: extend `useBackends.ts`/`BackendScope.vue` to an `All ▾` selector over *backends +
  peers* (one combined list, below), writing `?cluster=`/`?backend=`; server-side filter via
  `JobQuery.child_cluster`.
- **Proto**: `JobStatus.child_cluster = 35` (§4.3), stamped in `get_job_status` next to `backend_id`
  (`service.py:1490`); TS mirror in `types/rpc.ts`.

### 8.1 One combined "execution targets" view (backends + peers)

The review is explicit that **users don't care about the backend-vs-peer distinction** in the UI, so
the dashboard presents **one** overview tab listing local backends *and* peers together — a single
grid of cards, each a place work can run — rather than a separate Backends tab and Clusters tab. The
ownership distinction stays where it belongs: **in the code and the RPCs** (`ListBackends` vs
`ListPeers`, §9), which the tab merges for display. A card is a backend or a peer; both show a health
dot, kind, capabilities/devices, and running·pending counts, so the surface is uniform even though the
underlying contract is not.

## 9. Q5 — The backends representation, and the parallel peers representation

**Backends are unchanged.** A backend is a local `TaskBackend` (`schedule`/`reconcile`/`autoscale`/
`status`) surfaced by `ListBackends → BackendSummary` (`controller.proto:588`, `service.py:2830`) and
the **Backends** tab. In-flight **PR #6773** generalizes the k8s-special-cased `get_cluster_status()`
into a uniform `TaskBackend.status()` returning a `BackendStatus` oneof (`kubernetes | worker`) — the
right shape, and orthogonal to federation. A peer is **not** a `BackendStatus` variant; in the *code*
it is a separate concept with a separate RPC (even though the *UI* merges the two into one execution-
targets view, §8.1 — the ownership distinction is load-bearing for the contract, not for the pixels):

```proto
message PeerSummary {
  string peer_id = 1;
  string controller_address = 2;
  string dashboard_url = 3;
  bool reachable = 4;              // last peer-link probe
  int64 last_sync_ms = 5;
  int32 active_federated_jobs = 6;
  int64 aggregate_spend_micros = 7;
  // Dynamically advertised, live (see 9.1) — not a static device list. Values are
  // availability markers like "available:H100", so the parent routes only to peers
  // that can schedule the request *now*, not merely peers that own the hardware.
  repeated string advertised_capabilities = 8;
}
rpc ListPeers(ListPeersRequest) returns (ListPeersResponse);  // peers[], parallel to ListBackends
```

`ListBackends` and `ListPeers` stay distinct RPCs because ownership differs — **Backends = what I run;
Peers = what I delegate to** — and the dashboard's combined tab (§8.1) is a *display* merge of the two,
not a merge of the concepts.

### 9.1 The federation module, the peer registry, and the router's second target kind

- **Module (per review).** All of this lives in one **`iris.cluster.federation`** package —
  `FederationManager` (handoff + `FederationSync` loop + cancel), the `FederationPeer` connection
  abstraction, the peer registry, and `PeerSummary`/`ListPeers`. Federation is a self-contained module
  the controller composes in, not logic smeared across the scheduler and service.
- **Config.** Peers are declared in cluster config — a new `peers:` section (peer id → controller
  address + auth), the gap the client survey found: today `rigging.ClusterManifest` models *one*
  cluster's identity/auth and **no manifest lists sibling controllers** (`cluster_manifest.py`).
  Federation fills it with a *manifest of peers*. Config declares *identity + reachability + trust*
  only — **not** capabilities, which are dynamic:
- **Dynamic capability advertisement (per review).** A peer advertises what it can *currently
  schedule*, not a static device list — availability markers like `available:H100` (mirroring the
  autoscaler's existing `availability_status` and the `region ANY/PINNED` markers). The parent refreshes
  these live, piggybacked on the `FederationSync` response (or a lightweight capability heartbeat when a
  peer has no active jobs), so a peer that loses a pool stops attracting routes within one interval — no
  config edit, no stale routing to a peer that can't actually run the work.
- **Router.** `route_jobs_to_backends` gains a peer arm: build the constraint index over
  *backends ∪ peers* using the live advertised capabilities; a job that a local backend can currently
  schedule stays local (**prefer-local, decided** — a peer is only for capacity/hardware the local
  backends lack); a job only a peer can satisfy becomes a federation candidate; an explicit
  `cluster=<peer>` constraint forces a peer. This reuses the existing constraint-matcher and the
  `region ANY/PINNED` machinery — peers are just another set of match targets.
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

1. **Schema + projection.** `jobs.child_cluster` + `federated_jobs` + `federated_tasks` + migration
   0034; `JobStatus.child_cluster`; `get_job_status`/`list_tasks` federated branch reading the
   projection. *No behavior yet* — no peer can be configured, so every job is local and the column is
   always `""`. Pure additive, single-cluster byte-identical.
2. **Federation module + router arm.** The `iris.cluster.federation` package (§9.1): `peers:` config,
   `FederationPeer`, dynamic capability advertisement, the `route_jobs_to_backends` prefer-local peer
   target, `ListPeers`/`PeerSummary`. Still inert with zero peers configured.
3. **Handoff + `FederationSync`.** `FederationManager`: synchronous durable handoff (§5.1) + the bulk
   delta-sync loop and its peer-served `FederationSync` endpoint (§5.2) + tombstone retention (§5.4) +
   versioned cancel (§5.3) + budget admission. First end-to-end federated job (parent + one peer).
4. **Global finelog + exec proxy.** Stand up the shared finelog with its authenticated ingress (§6.3);
   configure the child controller's `LogStack` as a relay (§6.1); proxy on-demand exec/profile through
   the peer controller (§7).
5. **Dashboard.** Always-on `cluster` column + `Cluster` detail row + **native federated task views**
   off the projection + the combined execution-targets tab (§8, §8.1). Screenshot smoke updates to
   include the always-on tab (a single-cluster cluster shows one card, not a hidden tab).
6. **(Later) reverse-dial transport** for fully-egress-blocked peer controllers (§6.3) — a transport
   swap under `FederationPeer`, with the `FederationSync` protocol unchanged.

The contract that keeps the codebase clean: **the backend seam never learns federation exists** (no
`RemoteTaskBackend`, no remote-safe store, no per-backend DB split — all deleted by Model D), and
**federation never touches the DAG fold** (it holds handles, not tasks). Two honest seams, each simple,
meeting only at the router.

## 12. Decisions folded in from review (2026-07-01)

- **Task-level cache, not summary-only.** Mirror the peer's tasks into a read-only `federated_tasks`
  projection so the parent renders native task views; the Model D boundary holds because the fold never
  reads it (§4.2). *(Comments: federated-tasks table, blob size, usability.)*
- **Global finelog, not a query-time proxy.** One shared store fed by peer-controller relays; uniform
  and cross-cluster-queryable, at the cost of an auth front + cross-region egress (§6). *(Comment: global finelog.)*
- **Bulk `FederationSync` delta protocol,** transport-agnostic so reverse-dial can defer (§5.2). *(Comments: bulk update, delta-in-either-case.)*
- **Retention mirrors the peer** via sync tombstones — no separate parent GC (§5.4). *(Comment: pruning handles it.)*
- **Always-on, combined execution-targets UI** — no `multiCluster` gate; backends + peers in one tab,
  the split kept in code only (§8, §8.1). *(Comment: gating complexity / users don't care.)*
- **Prefer-local routing** (§2, §9.1). *(Comment: prefer-local.)*
- **Dynamic `available:X` capability advertisement** on the sync channel (§9.1). *(Comment: dynamic availability.)*
- **`iris.cluster.federation` module** owns the abstractions (§9.1). *(Comment: peering/federation module.)*

## 13. Open questions

1. **Budget admission strength.** Report-and-throttle (baseline, one-interval overspend) vs a
   grant/reservation protocol before handoff. Start with report-and-throttle; revisit under real
   multi-tenant peer load.
2. **Global-finelog auth mechanism.** Front the shared store with a rigging `server_auth` ingress that
   each relaying controller authenticates to, or co-locate it with one controller and reach it only
   through that controller's authed proxy? (§6.3 — the one genuinely new security surface.)
3. **`FederationSync` watermark mechanics.** What exactly is the peer-side cursor — a monotonic
   per-job revision counter, a logical clock, or a `(updated_at, job_id)` pair — and how does the peer
   index it cheaply enough to compute deltas without scanning all jobs each tick?
4. **Sync cadence & scale.** Concrete active-peer interval and back-off curve; expected delta size at,
   say, 10 peers × thousands of live tasks; when a peer with zero active jobs drops to heartbeat-only.
5. **Reverse-dial priority.** Is any near-term target cluster fully egress-blocked at the *controller*
   (needing §6.4 now), or do all near-term peers have controller egress (baseline suffices)?
