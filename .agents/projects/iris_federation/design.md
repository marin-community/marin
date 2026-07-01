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
> single-cluster user's workflow is unchanged. **Revised 2026-07-01 through two review rounds** — see
> §12 for the maintainer's direction (task-level cache, global finelog, bulk delta-sync, always-on
> combined UI) and §12.1 for the codex hardening pass (submit-time federation, deterministic handoff
> id, durable sync changelog, `jobs.state` as source of truth, pruner exclusion, namespaced logs).

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
2. **Decide at submit.** Federation is decided **inside `launch_job`, before tasks are materialized**
   (codex: `ops.job.submit` unconditionally inserts local task rows, `ops/job.py:262`). The parent
   asks: is any local backend *feasible* for the job's constraints (the feasibility check already in
   `launch_job`, `service.py:1366`)? If yes → **local** (normal submit; the tick schedules it,
   possibly waiting for capacity). **Prefer-local is the rule** (decided). Only if no local backend
   can satisfy the constraints — or an explicit `cluster=<peer>` pin — does peer routing run, a
   *separate* layer over live peer capability (§9.1).
3. **Hand off.** For a federated job the parent takes a distinct `submit_federated_handle` path: it
   persists the `jobs` row (with `child_cluster=<peer>`) + `job_config` + the `federated_jobs` handle
   with **no task rows** (so `_route_pending` never sees it, §4.3), computes a **deterministic
   globally unique `remote_job_id`** (§5.1), then synchronously calls the peer's `LaunchJob`. It
   returns the *parent's* job id to the user immediately.
4. **Peer runs the whole tree.** The peer materializes the job in **its own** DAG and schedules it on
   **its own** backends. Crucially, the peer injects **its own** `IRIS_CONTROLLER_ADDRESS` into every
   task's env (`runtime/env.py:150`). So when the running program spawns a child job *via the default
   in-task client*, `get_iris_ctx()` connects back to the **peer** (`client.py:1190`), and the child's
   `LaunchJob` lands in the **peer's** DAG. **The subtree stays on the peer by construction** for the
   common path. (This guarantee is scoped to the default in-task client — a program that hand-builds an
   `IrisClient.remote(...)` to the parent, `client.py:524`, or an out-of-band submit of a child with a
   federated parent id, escapes it; the parent handles that server-side: route such a child to the
   parent's peer via the handle, or reject it — §9.2.)
5. **Status.** User polls the parent (`iris status <job_id>` / dashboard). The parent serves the
   **cached rows** — the sync mirrors the peer's job/task state into the parent's own `jobs`/`tasks`
   rows (`child_cluster`-stamped, §4.2) — so the *same* job/task views a local job gets render
   natively, annotated `cluster: cw-us-east`. Refreshed by a bulk delta-sync (§5.2).
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

**Recommendation: add `child_cluster: str` to *both* `jobs` and `tasks` (a new `""`-default column,
exactly parallel to `backend_id`), and keep the federated rows in the *main* tables so every read
just works.** `federated_jobs`/`federated_tasks` become thin **join sidecars** for federation-only
metadata — not parallel copies of the rows. (This supersedes the v3 separate-`federated_tasks`
projection — see §4.2 for why the review's unified-table instinct is the better call.)

### 4.1 `child_cluster` — the discriminator, symmetric with `backend_id`

`backend_id` (`schema.py:273/350`) already answers "*which local substrate owns this?*". `child_cluster`
answers the orthogonal "*is this owned locally, or handed off to a peer, and which one?*", and lives in
exactly the same places:

- `child_cluster == ""` → **local.** Routed to a local backend via `backend_id` as today.
- `child_cluster == "<peer>"` → **federated.** `backend_id` stays `""` (no local backend). Mutually
  exclusive by construction, on both `jobs` and `tasks`.

**A federated job's tasks live in the normal `tasks` table**, with `child_cluster` set — they are not
held off in a parallel projection. The payoff (per review): every existing read — `ListTasks`,
`GetTaskStatus`, the job's task-count `GROUP BY`, the dashboard task table, `JobQuery`/`WorkerQuery`
filters — *just works* with no federated branch, exactly as they already work across `backend_id`.

**How this stays inside Model D — the fold is excluded structurally, not the rows.** Model D says
"never mirror remote task rows"; the operative danger is the *fold/scheduler operating on* those rows,
not their physical presence in a table. So the exclusion is enforced in **one** place: the scheduler,
reconcile, budget, and routing read a **`local_tasks` view** = `tasks WHERE child_cluster = ''` (and
`_route_pending`'s unrouted scan, the active-task budget count, and the cancel/timeout scans read it),
so federated rows are *structurally* invisible to scheduling. That is a stronger guarantee than the v3
"separate table + hope every read remembers to union it in": here the risky direction (a scheduler
query seeing a federated row) is opt-*out*, so a newly-written scheduler query is safe by default and
must go out of its way (raw `tasks`) to see a federated row. Reads use the raw `tasks` table and get
federation for free. (This is the same lesson the backend-contract work drew from worker state: a
boundary held *by convention across N call sites* leaks; a boundary enforced *in one view/store* holds.)

### 4.2 The rows live in `jobs`/`tasks`; `federated_jobs`/`federated_tasks` are thin join sidecars

`jobs` and `tasks` hold the federated rows (with `child_cluster` set); the sync mirrors the peer's
state into them. The `federated_*` tables are **joins for federation-only metadata**, not parallel
copies — the same shape as `jobs` ⋈ `job_config` today. `federation_sync_state` holds the delta cursor.

**`jobs.state`/`tasks.state` are the single source of truth (codex).** List/status/filter/count paths
read `jobs.state`/`started_at_ms` and the `tasks`-`GROUP BY` (`reads.py:256/292`, `service.py:1480`);
a duplicate mutable state would diverge. So the sync writes the peer's job/task state **into the `jobs`
and `tasks` rows transactionally**, and the sidecars carry only what those tables can't express.

```python
# jobs   += Column("child_cluster", String, nullable=False, server_default="")   # discriminator (§4.1)
# tasks  += Column("child_cluster", String, nullable=False, server_default="")   # discriminator (§4.1)

# Scheduler/reconcile/budget/route read THIS, never raw `tasks` — structural fold-exclusion (§4.1).
local_tasks_view = "CREATE VIEW local_tasks AS SELECT * FROM tasks WHERE child_cluster = ''"

federated_jobs_table = Table(   # job-level handle + sync metadata; job STATE lives in `jobs`
    "federated_jobs", metadata,
    Column("job_id", JobNameType, ForeignKey("jobs.job_id", ondelete="CASCADE"), primary_key=True),
    Column("peer_id", String, nullable=False),                # == jobs.child_cluster
    Column("remote_job_id", String, nullable=False),          # DETERMINISTIC, globally unique: "<parent_cluster>/<job_id>" (§5.1)
    Column("owner_principal", String, nullable=False),        # auth identity asserted to the peer on delegated RPCs
    Column("handoff_state", Integer, nullable=False),         # PENDING_HANDOFF | HANDED_OFF | HANDOFF_FAILED
    Column("spend_snapshot_micros", Integer, nullable=False, server_default="0"),
    Column("cancel_intent_version", Integer, nullable=False, server_default="0"),
    Column("last_sync_ms", TimestampMsType),
    Column("terminal_error", String),
)

# Per-(peer,requester) delta cursor — NOT per-job (codex). One row per peer this controller federates to.
federation_sync_state_table = Table(
    "federation_sync_state", metadata,
    Column("peer_id", String, primary_key=True),
    Column("cursor", String, nullable=False, server_default=""),   # opaque monotonic watermark into the peer's changelog (§5.2)
    Column("last_full_resync_ms", TimestampMsType),
)

# OPTIONAL per-task join: ONLY the federation extras that don't fit `tasks` (e.g. the opaque peer-side
# worker name for display, since there is no local worker row). Add a column here only when a field the
# UI shows has no home in `tasks`; task STATE / timing / exit / counts live in the `tasks` row itself.
federated_tasks_table = Table(
    "federated_tasks", metadata,
    Column("task_id", TaskNameType, ForeignKey("tasks.task_id", ondelete="CASCADE"), primary_key=True),
    Column("peer_worker_label", String, nullable=False, server_default=""),   # opaque peer worker name, display only
)
```

Attempt-level history (`task_attempts`) is **deliberately not mirrored** — the task-detail page shows
the current attempt from the `tasks` row and deep-links to the peer for the full attempt list; a
conscious *reduced* path (the `tasks` row still carries `current_attempt_id`, exit, timing, counts, so
the task *list* and detail header render natively).

**Why unified tables + a view, not a separate projection (the review's question).** v3 put federated
tasks in a parallel `federated_tasks` projection so the fold physically couldn't see them — but that
forces *every read* (`ListTasks`, `GetTaskStatus`, counts, dashboard) to branch on `child_cluster` and
union the projection in. The review's instinct is better: keep the rows in `tasks` so reads just work,
and move the guarantee to the *write/scheduler* side:

| | v3 — separate `federated_tasks` projection | **chosen — `tasks` + `child_cluster` + `local_tasks` view** |
|---|---|---|
| Read paths (list/detail/counts/filters) | **branch** on `child_cluster`, union the projection | **unchanged** — read raw `tasks`, works like `backend_id` |
| Fold/scheduler safety | structural (rows not in `tasks`) | structural (scheduler reads `local_tasks`; federated rows opt-*out*) |
| Where the boundary lives | in every *read* | in **one** *view* + the handful of scheduler task-scans |
| New-code failure mode | a new read forgets to union → missing federated data (display bug) | a new scheduler query hits raw `tasks` → sees federated rows (caught by the view convention + review) |
| Net | reads pay forever, fold trivially safe | reads free, fold safe via one view |

**Chosen: unified.** The cost moves from "every read branches" to "the ~4 scheduler task-scans read
`local_tasks`" (`_route_pending`, `build_scheduling_context`, the active-task budget count, the
cancel/timeout scans) — fewer sites, all on the write/schedule side where the discipline already
exists (they already filter by `backend_id`). Symmetric with jobs (which already live in `jobs` with
`child_cluster` + the `federated_jobs` join), so tasks and jobs are modeled the same way.

### 4.3 Insertion points (grounded)

| Concern | Where | Change |
|---|---|---|
| Schema | `schema.py:255/326` (jobs, tasks) | add `child_cluster` col to **both** `jobs` and `tasks`; new `federated_jobs` + `federation_sync_state` + (thin) `federated_tasks`; `local_tasks` view |
| Migration | new `migrations/0034_federation.py` | `ALTER TABLE jobs/tasks ADD COLUMN child_cluster` + `CREATE TABLE` the sidecars + `CREATE VIEW local_tasks`; template is `0033_backend_id.py` verbatim (idempotent guard, `""` backfill) |
| **Scheduler exclusion** | `_route_pending` (`controller.py:991`), `build_scheduling_context`, active-task budget count (`reads.py:1181`, `budget.py:43`), cancel/timeout scans | read **`local_tasks`** (or add `AND child_cluster = ''`) so federated task rows are never routed/scheduled/counted — the one load-bearing change (§4.1) |
| **Decide (submit)** | `launch_job` (`service.py:1097`), **before** `ops.job.submit` materializes tasks (`ops/job.py:208/262`) | federation is decided **at submit** — a separate `submit_federated_handle` path persists `jobs`(+`child_cluster`)+`job_config`+`federated_jobs` with **no tasks** (peer decides the tasks); the sync later inserts federated `tasks` rows |
| Handoff | `federation.FederationManager.submit` | synchronous peer `LaunchJob` with the deterministic `remote_job_id` (§5.1); flip `handoff_state` |
| Sync write | `FederationManager` sync loop (§5.2) | mirror peer job state into `jobs`, peer task state into `tasks`(+`child_cluster`), extras into `federated_tasks`; apply tombstones; advance `federation_sync_state.cursor` |
| Read (job/tasks) | `get_job_detail`/`list_tasks`/`get_task_status` (`reads.py:495`, `service.py:417/508/1697`) | **unchanged** — read raw `jobs`/`tasks`; `child_cluster` and mirrored state come for free; left-join `federated_jobs`/`federated_tasks` only where a handle/extra field is shown |
| Prune | `pruner.py:52`, `reads.py:455` terminal-job prune | **exclude `child_cluster != ""`** — tombstones are the only deletion path for federated rows (§5.4) |
| Proto | `job.proto:328/234` | `JobStatus.child_cluster = 35`; `TaskStatus.child_cluster = 25` (siblings of `backend_id`) |
| Dashboard | §8 | always-on `cluster:` annotation + combined execution-targets view |

## 5. Q2 — Synchronization between parent and child

Two distinct channels, deliberately different in shape.

### 5.1 Submission = **synchronous handoff**, not a queue

**Idempotency comes from a deterministic remote id, not a new protocol (codex).** Today
`LaunchJobRequest` has no idempotency-key field and duplicate handling is same-job-name *policy*, not
request-digest dedup (`controller.proto:80`, `service.py:1407`). Rather than add a new idempotency
mechanism, the parent derives the `remote_job_id` **deterministically and globally uniquely** before
it calls the peer — `"<parent_cluster_id>/<parent_job_id>"` — and hands the job to the peer *under
that name*. Re-sending the same handoff is then a same-name submit, which the peer's **existing** KEEP
policy makes a no-op (`service.py:1254-1297`). The handle stores `remote_job_id` at creation time (so
the column is non-null from the start), and the peer's global uniqueness also fixes cross-cluster log-key
collisions (§6).

The parent (a) durably writes the `jobs`+`federated_jobs` handle (with the deterministic `remote_job_id`,
`handoff_state=PENDING_HANDOFF`) in one local transaction, then (b) **synchronously** calls the peer's
`LaunchJob`. On ack it flips `handoff_state → HANDED_OFF`. The user's client blocks only for one RPC —
the same interactive contract as a local submit.

*Why synchronous, not a general queue:* submission is user-facing; the client wants a job id back.
Durable-handle-first + deterministic-id retry gives resilience **without** a queue's complexity:

- If the peer is **unreachable**, the handle persists in `PENDING_HANDOFF`; a background retry re-sends
  under the same `remote_job_id` (the peer's KEEP policy dedups). The parent surfaces the job as
  `pending (handing off to <peer>)` — never lost, never double-submitted.
- If the parent **crashes mid-handoff**, recovery re-drives `PENDING_HANDOFF` handles on boot; the
  deterministic id makes the re-send a no-op if the peer already has it.

This is the "exactly-once root handoff" the delegation doc flagged (`delegation_model.md:346`), reduced
to "a deterministic name + the peer's existing same-name policy" — no new idempotency machinery.

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
  repeated TaskStatus changed_tasks = 3; // FULL row for any task whose display fields changed since the cursor
  bool tombstone       = 4;              // peer has pruned this job → parent drops its projection (§5.4)
}
message FederationSyncResponse {
  repeated FederationJobDelta deltas = 1;
  string next_cursor  = 2;   // persisted to federation_sync_state.cursor; passed back next tick
  bool   cursor_stale = 3;   // cursor older than the changelog floor → parent must full-resync (below)
}
rpc FederationSync(FederationSyncRequest) returns (FederationSyncResponse);
```

- **The cursor is per-(peer, requester), and the peer keeps a durable changelog (codex).** The cursor
  lives in `federation_sync_state.cursor` (one row per peer), **not** per job. On the peer side it
  indexes a durable **federation changelog** — a monotonic sequence over job/task transitions **and**
  tombstones, retained past the maximum tolerated parent outage. The parent applies each batch in one
  transaction (mirror peer state into the `jobs` + `tasks`(+`child_cluster`) rows, upsert any
  `federated_tasks` extras, apply tombstones, advance the cursor). A "changed task" is any task whose
  *display fields* changed — worker, exit, timing, counts, not just state — so each delta carries the
  **full row**, not a state-only diff.
- **Full-resync is authoritative set-replacement, so a missed tombstone can't leak (codex).** If the
  parent's cursor is older than the changelog floor (long outage) the peer sets `cursor_stale`; the
  parent then does a full resync — fetch the peer's entire active set for this requester and **delete
  any local federated handle not in that set**. This set-difference is what reclaims a job the parent
  never saw tombstoned. An empty cursor (first contact) is the same path. So restart/gap recovery and
  steady state converge, but recovery is a *set replace*, not just an additive replay.
- **Transport-agnostic (per review).** `FederationSync` is one request/response over the peer link;
  it works **identically** whether the parent dials the peer (baseline) or the peer dials the parent
  and the parent calls back over that reverse channel (§6.4) — so reverse-dial defers without a redesign.
- **Cadence, adaptive.** Active peers sync on a short interval (a few seconds); a peer with no active
  handoffs drops to a slow capability heartbeat. A terminal job simply stops producing deltas; it is
  removed only by its eventual tombstone (§5.4).
- **Spend / budget (codex — needs its own path).** The current submit cap counts active *local task
  rows* (`service.py:1207`, `reads.py:1181`) and the scheduler sums active local tasks (`budget.py:43`);
  a zero-local-task federated root bypasses **both**. Federated admission is therefore an explicit
  transaction: `local_active_spend + Σ cached_federated_spend (+ optional reservation)` against the
  user cap, evaluated in the `submit_federated_handle` path. Spend then rides back in each
  `FederationJobDelta.summary`. **Report-and-throttle is explicitly weaker** than local enforcement
  (overspend bound = one sync interval); a grant/reservation protocol is the hardening
  (`delegation_model.md:419`).

### 5.3 Cancel / preemption = versioned intent, routed, with defined races

Cancel on the parent bumps `cancel_intent_version` on the handle and routes an idempotent
`CancelJob(remote_job_id)` to the peer; the next `FederationSync` confirms the peer reached a terminal
state. Versioning makes a retried/late cancel a no-op. This replaces today's single-transaction subtree
kill (`ops/job.py:284`) *only at the federation boundary* — local cancel is unchanged. Because the
`remote_job_id` is deterministic and set at handle creation (§5.1), the "cancel before we learned the
remote id" race **does not exist**. The remaining races have defined outcomes (codex):

- **Cancel while `PENDING_HANDOFF`** (peer may not have the job yet): mark the handle cancelled locally
  and *skip or supersede* the handoff; if the handoff already reached the peer, the deterministic id
  means the follow-up `CancelJob(remote_job_id)` still targets the right job.
- **Peer returns `NOT_FOUND`** (job already terminal-and-pruned, i.e. tombstoned): treat the cancel as
  **satisfied**, not an error — the job is already gone.
- **Cancel a job the sync last saw as running but the peer has since finished**: idempotent; the next
  sync reconciles the terminal state.

### 5.4 Retention = the parent mirrors the peer (no separate GC)

The parent's projection is a cache of what the peer *still holds*, so its lifetime is the peer's
lifetime — the parent needs **no retention policy of its own** (per review). When the peer's normal
job pruning removes a finished job, the next `FederationSync` carries a `tombstone` for it (or a
full-resync set-difference reclaims it after a long outage, §5.2); the parent deletes that handle's
`jobs` + `tasks` rows (both `child_cluster`-stamped) and its `federated_*` sidecar rows. Want longer
history on the parent? Lengthen the *peer's* retention — the parent follows.

**This requires excluding federated rows from the local pruner (codex).** The current pruner deletes
*any* terminal `jobs` row (and its tasks) older than local retention (`pruner.py:52`, `reads.py:455`) —
and because the sync mirrors the peer's terminal state into `jobs`/`tasks`, federated rows would look
prunable and the two GC clocks would fight. So the local terminal-job pruner must **skip
`child_cluster != ""`**; tombstone application (or full-resync set-difference) is the *only* deletion
path for federated rows. This is the single change that makes "the parent mirrors the peer" actually
hold.

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
global service). The change is one level up: the child controller runs a **new durable relay** that
forwards each batch to the shared global finelog over its egress.

**The relay is net-new, not "configure the existing `LogStack`" (codex).** Today `LogStack` is just a
client/server holder (`log_stack.py:33`) and a finelog `Table` is a *bounded in-memory queue that
drops oldest rows on overflow and drops non-retryable failures* (`log_client.py:190/334/421`) — nowhere
near store-and-forward. So the relay is a real component: a durable spool with per-batch ids,
retry-dedup, backpressure, and loss metrics, so a child-region egress stall doesn't silently drop logs.

**Keys must be cluster-namespaced (codex — the v2 "no remapping" claim was wrong).** Iris log keys are
just `/user/<job>/<task>:<attempt>` (`log_keys.py:52`) and finelog keys are opaque strings
(`finelog/types.py`), so two clusters running the same user/job path would **collide** in one store.
Two things prevent it, and they compose: (a) the relay stamps a **cluster id** (a per-cluster
namespace/key-prefix) on everything it forwards; and (b) a federated job's logs are already written
under the **globally unique** `remote_job_id` (`<parent_cluster>/<job_id>`, §5.1). So the parent's log
query targets `<peer_cluster>` + `remote_job_id` — a real (small) remapping, not "none."

### 6.2 Reads hit the one global finelog

There is no per-peer log proxy. The parent (and every client) queries the shared finelog exactly as a
single-cluster controller queries its own today — via `FetchLogs`/`StatsService.Query` through the
`EndpointProxy` (`dashboard.py:298`). Because the store is global and keys are cluster-namespaced, a
federated job's logs, a local job's logs, and even a **cross-cluster** query ("all failures for user X
across every cluster") are one query against one store. The parent translates its own job id → the
`<peer_cluster>` namespace + `remote_job_id` when building the `FetchLogs` source (a `federated_jobs`
lookup), because that is where the peer's relay wrote it (§6.1).

### 6.3 What the global store requires (honest costs)

- **Auth is mandatory, and net-new.** A globally shared finelog receives pushes from *many* controllers
  across the internet, so it can no longer rely on being private behind one controller's `EndpointProxy`
  (finelog has **no server auth** today — `lib/finelog/AGENTS.md:29`, `app.rs:100`). The global finelog
  must be fronted by an authenticated ingress — reuse the rigging `server_auth` verifier so each relaying
  controller authenticates its pushes, or co-locate it with a controller and reach it only through that
  controller's authed proxy. This is the one genuinely new security surface the design adds.
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

- The parent's on-demand handler branches on the target job's `child_cluster` *before* the local
  `task→worker→backend` resolution (`service.py:2222/2556`). If set, it forwards the RPC to the peer
  controller (keyed by the handle's `remote_job_id`), which does its *own* `task→worker` resolution and
  returns the result. Uniform auth (the parent stays the trust boundary), one endpoint for the user, no
  client reach/creds into the peer.
- Redirect (hand the client the peer address) is the alternative — parent off the hot path, but the
  client needs direct reach + peer auth, which siloed peers may deny. Proxy is the safer baseline; this
  is the open "proxy vs redirect" question from the spec (`spec.md:216`), resolved toward **proxy** for
  the siloed threat model.
- **Races have defined outcomes (codex).** The parent's mirrored `tasks` rows are last-sync stale, so
  an exec/profile may target a task the peer has since moved or finished — the *peer* is authoritative
  and returns its live answer or a `NOT_FOUND`, which the parent surfaces verbatim (it does not guess
  from the cached row). A federated job whose handle is tombstoned resolves exec/profile to `NOT_FOUND`
  immediately, without a peer round-trip.

## 8. Q4 — Visualization

**Reuse the existing jobs display; surface `cluster` as an annotation — always on, no multi-cluster
gate (per review).** The first draft gated every affordance on `count > 1`; the review's call is that
the gate buys too little for its complexity, so federation renders the same UI whether there are zero
peers or ten — a single-cluster deployment just sees one execution target and empty `cluster:` cells.

- **Jobs list**: a **Cluster** column, always rendered (`JobsTab.vue:534-637`), showing
  `cluster: cw-us-east` for a federated job and blank for a local one; clicking it filters `?cluster=`.
- **Job/Task detail**: a `Cluster` `InfoRow` (`JobDetail.vue:960`) and — because federated tasks live in
  the normal `tasks` table (§4.2) — the **native task table and task-detail pages render with no
  federated branch at all** (state, worker, exit code, timing come straight from the `tasks` row).
  Attempt-level drill-down deep-links to the peer (`ClusterManifest.dashboard_url`), the one place the
  mirror stops.
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
- **Peer routing is a *separate layer*, not folded into `route_jobs_to_backends` (codex).** The
  existing meta-scheduler index is built **once at startup** from immutable backend attributes and
  picks a match lexicographically with no prefer-local phase (`controller.py:328`,
  `meta_scheduler.py:93`) — it is the wrong home for *dynamic* peer capabilities. So federation adds a
  distinct step, run at **submit** (§2, before task materialization): (1) is any local backend
  *feasible*? → local (prefer-local, decided); (2) else match the job against **live** peer capability
  snapshots; (3) an explicit `cluster=<peer>` forces a peer. The peer step **tolerates rejection** — a
  peer whose live capacity vanished between the snapshot and the handoff can reject, and the layer
  requeues / tries another peer / falls back to local-and-wait, rather than wedging the job.
- **Connection.** `RemoteClusterClient` already encapsulates "one connection to one controller"
  (`remote_client.py:71`); a `FederationPeer` holds one per peer, keyed by peer id — the natural place
  the reverse-dial transport (§6.4) plugs in later.

### 9.2 Out-of-band child submits (closing the subtree-lock gap)

The subtree lock (§2, step 4) is self-enforcing only for the *default in-task client*. Two escapes exist
(codex): a program that hand-builds an `IrisClient.remote(...)` pointed at the parent
(`client.py:524`), and an out-of-band client that submits a job whose *parent id* is a federated job,
directly to the parent. The parent closes both **server-side** in `launch_job`: when a submitted job's
parent id resolves to a `child_cluster != ""` handle, the parent does not materialize it locally — it
**forwards the submit to that parent's peer** via the handle (the child belongs on the peer with the
rest of the tree), or rejects it if forwarding is disallowed. So the lock is enforced by the parent's
submit path, not merely by `IRIS_CONTROLLER_ADDRESS` convention.

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

1. **Schema.** `child_cluster` on `jobs` **and** `tasks` + `federated_jobs` + `federation_sync_state` +
   (thin) `federated_tasks` + the `local_tasks` view + migration 0034; `JobStatus`/`TaskStatus.child_cluster`;
   point the scheduler task-scans at `local_tasks`. *No behavior yet* — no peer can be configured, so
   every row is local and the column is always `""`. Pure additive, single-cluster byte-identical.
2. **Federation module + submit-time routing.** The `iris.cluster.federation` package (§9.1): `peers:`
   config, `FederationPeer`, dynamic capability advertisement, the **separate** submit-time prefer-local
   peer-routing layer (§9.1, not folded into `route_jobs_to_backends`), `ListPeers`/`PeerSummary`. Still
   inert with zero peers configured.
3. **Handoff + `FederationSync`.** `FederationManager`: the `submit_federated_handle` path with a
   deterministic `remote_job_id` (§5.1) + the bulk delta-sync loop and its peer-served `FederationSync`
   endpoint incl. the durable changelog + full-resync (§5.2) + tombstone retention & pruner exclusion
   (§5.4) + versioned cancel with race semantics (§5.3) + the federated budget-admission path. First
   end-to-end federated job (parent + one peer).
4. **Global finelog + exec proxy.** Stand up the shared finelog with its mandatory authenticated ingress
   (§6.3); build the child controller's **durable log relay** (§6.1, a new spool — not a `LogStack`
   flag) with cluster-namespaced keys; proxy on-demand exec/profile through the peer controller (§7).
5. **Dashboard.** Always-on `cluster` column + `Cluster` detail row + **native federated task views**
   off the projection + the combined execution-targets tab (§8, §8.1). Screenshot smoke updates to
   include the always-on tab (a single-cluster cluster shows one card, not a hidden tab).
6. **(Later) reverse-dial transport** for fully-egress-blocked peer controllers (§6.4) — a transport
   swap under `FederationPeer`, with the `FederationSync` protocol unchanged.

The contract that keeps the codebase clean: **the backend seam never learns federation exists** (no
`RemoteTaskBackend`, no remote-safe store, no per-backend DB split — all deleted by Model D), and
**federation never touches the DAG fold** (it holds handles, not tasks). Two honest seams, each simple,
meeting only at the router.

## 12. Decisions folded in from review (2026-07-01)

- **Task-level cache, not summary-only.** Mirror the peer's tasks so the parent renders native task
  views (§4.2). *(Comments: federated-tasks table, blob size, usability.)* **Refined in v4 (below):
  the rows live in the `tasks` table, not a separate projection.**
- **Global finelog, not a query-time proxy.** One shared store fed by peer-controller relays; uniform
  and cross-cluster-queryable, at the cost of an auth front + cross-region egress (§6). *(Comment: global finelog.)*
- **Bulk `FederationSync` delta protocol,** transport-agnostic so reverse-dial can defer (§5.2). *(Comments: bulk update, delta-in-either-case.)*
- **Retention mirrors the peer** via sync tombstones — no separate parent GC (§5.4). *(Comment: pruning handles it.)*
- **Always-on, combined execution-targets UI** — no `multiCluster` gate; backends + peers in one tab,
  the split kept in code only (§8, §8.1). *(Comment: gating complexity / users don't care.)*
- **Prefer-local routing** (§2, §9.1). *(Comment: prefer-local.)*
- **Dynamic `available:X` capability advertisement** on the sync channel (§9.1). *(Comment: dynamic availability.)*
- **`iris.cluster.federation` module** owns the abstractions (§9.1). *(Comment: peering/federation module.)*

### 12.1 Hardening from the codex review cycle (2026-07-01)

The v2 direction held ("Model D is the right boundary"), but codex flagged that v2 hand-waved where a
zero-task federated job meets the current DAG-shaped code. Folded in:

- **Federation decided at *submit*, before task materialization** — a separate `submit_federated_handle`
  path, because `ops.job.submit` unconditionally inserts local tasks (§2, §4.3). *(BLOCKER)*
- **Idempotency via a deterministic globally-unique `remote_job_id`** + the peer's existing same-name
  policy — no new idempotency field; `remote_job_id` known at handle creation (§5.1). *(BLOCKER)*
- **Per-(peer,requester) cursor in `federation_sync_state` + a durable peer changelog with retained
  tombstones + full-resync-as-set-replacement** so a missed tombstone can't leak (§4.2, §5.2). *(BLOCKER)*
- **`jobs.state`/`tasks.state` are the single source of truth** (sync mirrors peer state into `jobs`
  and `tasks`); no duplicate state on the sidecars; counts come from the `tasks` `GROUP BY` (§4.2). *(MAJOR)*
- **Exclude `child_cluster != ""` from the local pruner** — tombstones are the only deletion path (§5.4). *(BLOCKER)*
- **Federated tasks carry the fields the task views render**; attempt drill-down stays a deep-link —
  a deliberate reduced path (§4.2). *(MAJOR)* **(Superseded by v4's unified-table model below.)*
- **Cluster-namespaced global-finelog keys** (the "no remapping" claim was wrong) + a **new durable
  relay spool** (the finelog `Table` drops on overflow; `LogStack` is not a relay) + **mandatory auth** (§6). *(MAJOR)*
- **Peer routing is a separate submit-time layer** over live snapshots with rejection/retry — not folded
  into the static, startup-built `route_jobs_to_backends` index (§9.1). *(MAJOR)*
- **Federated budget admission is its own transaction** (local + cached federated spend + optional
  reservation); report-and-throttle marked explicitly weaker than local enforcement (§5.2). *(MAJOR)*
- **Cancel/exec race semantics defined** (peer `NOT_FOUND` after tombstone, cancel while
  `PENDING_HANDOFF`, exec against a stale projection) (§5.3, §7). *(MAJOR)*
- **Subtree lock scoped to the default in-task client**, with server-side handling of out-of-band child
  submits at the parent (§2 step 4, §9.2). *(MINOR)*

### 12.2 Unified tables (2026-07-01, maintainer)

*"Why a separate `federated_tasks` at all — could federated tasks live in `tasks` with a `cluster_id`
column so the dashboard queries just work, and jobs the same with `federated_jobs` as a join?"* Adopted
(§4): `child_cluster` goes on **both `jobs` and `tasks`**; the federated rows live in the main tables so
every read is unchanged; `federated_jobs`/`federated_tasks` are thin **join sidecars** for the handle /
per-task extras only. The v3 separate-projection is dropped. The single guarantee this moves — keeping
federated rows out of the *fold* — is enforced structurally by a **`local_tasks` view** the scheduler
reads, so the cost lands on ~4 write-side task-scans instead of every read. Tasks and jobs are now
modeled identically.

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
