---
plan: tighten-iris-control-boundaries-multi-backend-readiness
status: draft
---

# Tighten Iris control boundaries (multi-backend readiness)

> Revised three times. The decisive simplification (this round): **worker health is a
> backend concern.** The backend tracks health from its own I/O (RPC backend: reconcile-RPC
> + build failures; K8s: pod status) and returns per-worker verdicts in `BackendResult`; the
> controller just **stores** them. That deletes `on_workers_failed` / `on_worker_failed` /
> `ping_workers`, the ping loop, and the controller-side `WorkerHealthTracker` logic — and
> removes the last place backend-specific data flowed through the controller. Phase order is
> **schedule → reconcile → autoscale** (each as-needed). `cluster=` is a **hard** constraint;
> the single-tick model is the target; the `BackendResult` name stays (the `*Result` swarm
> collapses, so the earlier naming concern is moot).

## Problem & goal

The recent `TaskBackend` work pulled most backend specifics behind a contract, but the
control plane is still **loose**:

1. **Seven independent `ManagedThread` loops**, each on its own snapshot/cadence, coordinated
   only by a write `RLock` + wake events → scheduler, autoscaler and reconciler act on skewed
   snapshots.
2. **Demand is computed twice** — a dry-run `find_assignments` (`policy.py:283`) on a second
   snapshot against a second `Scheduler` instance (`controller.py:350` vs `backend.py:141`).
3. **Backend-specific data flows through the controller.** The controller derives worker
   liveness (ping loop), then *couriers* failed worker IDs into the backend
   (`on_workers_failed`, `controller.py:1202`) and loops over backend-returned siblings — even
   though the sibling/slice logic already lives in the autoscaler (`operations.py:36-102`). It
   also branches on `placement`/`manages_capacity` and reads type-specific result fields.
4. The reconcile **result shape is backend-specific** (`updates` vs `worker_results`).
5. **Calls the controller dispatches into a backend are not uniformly bounded** —
   `RpcTaskBackend.reconcile` runs `asyncio.run(...)` on the caller thread (`backend.py:153`);
   `exec_in_container` permits a ~1-hour RPC deadline (`backend.py:232-237`).

**Long-term goal:** a *multi-backend* Iris (GCP TPU + Slurm GPU, …), jobs routed by a **hard
`cluster=<name>` constraint**. The controller must build a snapshot, call **one uniform
interface** on every backend, and persist the DB-shaped `BackendResult` — never branching on
backend type, never holding backend-specific data.

**What "done" looks like:** a verified current-flow map, a target flow, and five improvements
with concrete code shapes + an honest feasibility analysis. Implementation = the tasks below.

### What is already good — and the precise caveats (verified)

- `Scheduler.find_assignments` is **pure** (`scheduling/scheduler.py`) but returns only
  `assignments` and marks a job `exhausted` on the first capacity miss without recording *why*
  — so residual demand is a deliberate limits-free fit, **not** a by-product (I2). And there
  are **two** `Scheduler` instances (`controller.py:350` feeds the building cap `:814` and the
  dry-run `:1242`; `backend.py:141` is the one the backend schedules with) — I2/I3 reconcile
  them to one.
- **Sibling-teardown already lives in the autoscaler** (`autoscaler/operations.py:36-102`,
  `runtime.py:774-802`): given dead workers, it finds the slice, derives siblings, detaches.
  The controller's `on_workers_failed` is a courier (`controller.py:1199-1217`) — deletable.
- **Worker health is purely in-memory in the controller** (`worker_health.py`; no DB
  persistence; reseeded from the workers table at startup, `controller.py:447-458`). Mutated
  by the ping loop (`consecutive_failures`, `:1157-1163`) **and** the reconcile path
  (heartbeat on success `batches.py:182`; `build_failed` on BUILDING→FAILED `task.py:441`;
  `mark_unhealthy` on the fail cascade `batches.py:226`). Read by scheduling-context filtering
  (`policy.py:1063` → `reads.py:1097`) and the read-only RPCs (`ListWorkers` `service.py:1578`,
  `GetWorkerStatus` `:1985`, `GetAutoscalerStatus` `:1766`). **Health is not in the scheduling
  snapshot today** — it's a separate post-filter.
- **K8s has no daemon-ping health model** — workers are pods, health is pod status
  (`backends/k8s/tasks.py`); it stubs `on_workers_failed`/`ping_workers`. This asymmetry is the
  reason health belongs *in* the backend, not in one controller-side tracker.
- The reconcile *kernel* (`reconcile/overlay|task|job|worker|effects`) is a **pure state
  machine**; `loader → snapshot → kernel → commit` is clean; `writes.validate()` enforces
  projection-table ownership.

## Architecture — current control flow

```mermaid
flowchart TB
    classDef problem fill:#ffe1e1,stroke:#c0392b,stroke-width:2px,color:#000;
    classDef pure fill:#e1f0ff,stroke:#2980b9,color:#000;
    classDef store fill:#f5f0d8,stroke:#7f8c8d,color:#000;

    subgraph CTRL["Controller — owns SQLite + write RLock + in-memory WorkerHealthTracker; 7 ManagedThreads"]
        direction TB
        L1["scheduling loop (~10s + RPC wake)"]
        L2["polling / reconcile loop (5s + wake)"]
        L3["dispatch loop (5s, TASK_BACKEND only)"]
        L4["ping loop (5s) — derives liveness, then couriers IDs to backend"]:::problem
        L5["autoscaler loop (10s)"]
        L6["prune (600s / 3600s)"]
        L7["checkpoint (optional)"]
    end

    DB[("Controller DB")]:::store
    HT["WorkerHealthTracker (in-memory, lost on restart)"]:::problem

    subgraph PURE["DB-free decision components"]
        SCH["Scheduler.find_assignments (PURE) — TWO instances"]:::pure
        DEM["compute_demand_entries — DRY-RUN on a 2nd snapshot"]:::problem
        ASP["Autoscaler — owns slices + sibling logic"]
    end

    subgraph BK["TaskBackend — 13-method contract; controller branches on type"]
        RPCB["RpcTaskBackend (IRIS)"]
        K8S["K8sTaskProvider (TASK_BACKEND) — stubs on_workers_failed / ping"]:::problem
    end

    L1 --> SCH
    L1 -->|"snapshot/commit"| DB
    L5 -->|"2nd snapshot"| DB
    L5 --> DEM -.->|"dry-run"| SCH
    L5 --> ASP --> DB
    L2 -->|"snapshot/commit"| DB
    L2 -->|"reconcile"| RPCB
    L3 --> K8S
    L4 -->|"ping"| RPCB
    L4 --> HT
    L4 -->|"on_workers_failed(ids)"| RPCB
    RPCB -.->|"siblings + AutoscalerState"| L4
    HT -.->|"health filter"| SCH
    L6 --> DB
    L7 --> DB

    L1 -.->|"⚠ demand skew"| L5
    L4 -.->|"⚠ backend-specific data through controller"| RPCB
```

## Architecture — target control flow

```mermaid
flowchart TB
    classDef good fill:#e1ffe1,stroke:#27ae60,color:#000;
    classDef store fill:#f5f0d8,stroke:#7f8c8d,color:#000;

    subgraph TICK["Control tick — ONE thread, ONE snapshot/tick"]
        direction TB
        S0["build ControlSnapshot (reads.py, one read txn) — incl. stored worker health"]
        S1["schedule (due ~10s / on wake): DECISION → assignments + residual_demand"]
        S2["reconcile (every tick): OPERATION → task observations + WORKER-HEALTH VERDICTS"]
        S3["autoscale (due): OPERATION → provision + tear down dead slices/siblings → AutoscalerState + verdicts"]
        S4["commit BackendResult (one write txn): pure kernel applies observations+health → store"]
        S0 --> S1 --> S2 --> S3 --> S4
    end

    subgraph HK["Housekeeping threads"]
        P["prune"]
        C["checkpoint"]
    end

    DB[("Controller DB — stores task state, worker health, slices")]:::store

    subgraph BD["per-backend bounded dispatch (threadpool + generous timeout) — I/O calls only"]
        D["submit(io_call, deadline)"]
    end

    subgraph BK["Backends — UNIFORM interface; OWN placement, capacity, AND health"]
        RPCB["RpcTaskBackend — 1 Scheduler + Autoscaler + health from reconcile-RPC/build"]:::good
        K8S["K8sTaskProvider — Kueue places; health from pod status"]:::good
    end

    S0 --> DB
    S1 -->|"in-process decision"| RPCB
    S2 --> BD
    S3 --> BD
    BD --> RPCB
    BD --> K8S
    RPCB -->|"BackendResult"| S4
    K8S -->|"BackendResult"| S4
    S4 --> DB
    P --> DB
    C --> DB
```

Net threads: **7 → 3**. No ping loop, no `on_workers_failed`, no controller-side health
tracker. The controller builds a snapshot (incl. stored health), calls one interface, and
persists `BackendResult` through one pure-kernel apply path — dispatching on result shape,
never backend type. `schedule` is an in-process decision; only the actual **I/O** (reconcile,
autoscale ops, one-offs) goes through the bounded dispatch.

## The five targeted improvements

### I1 — One phased control tick (schedule → reconcile → autoscale); drop the ping, dispatch & autoscaler loops

**Problem.** Five fast loops re-snapshot/commit independently; the dispatch loop is "reconcile
for TASK_BACKEND"; the autoscaler loop re-snapshots; the ping loop derives liveness and
couriers failures into the backend.

**Change.** One driver thread, one snapshot per tick, fixed order:

```
control_tick(now):
    snap = build_control_snapshot(db)              # ONE read txn via reads.py (I4); incl. stored health
    r = BackendResult()
    if schedule_phase.due(now) or woken:
        r += backend.schedule(snap)                # pure decision: assignments + residual_demand
    if reconcile_phase.due(now):
        r += backend.reconcile(snap)               # bounded I/O (I5): observations + health verdicts
    if autoscale_phase.due(now):
        r += backend.autoscale(snap, r.residual_demand, r.health)  # bounded I/O: scale + dead-slice teardown
    with db.transaction() as tx:                   # ONE write txn, one apply path
        commit(tx, r)                              # pure kernel: observations+health → task state; store health/slices
```

- **Drop the ping loop.** Reconcile already reaches every active worker (idle ones get an
  empty-rows heartbeat); liveness now comes from the backend tracking its own reconcile-RPC
  outcomes (I3). Keep the reconcile cadence ≈ the old ping interval so detection isn't slower.
- **Drop the dispatch loop** (it is reconcile for TASK_BACKEND placement; uniform after I3).
- **Drop the autoscaler loop** (now the autoscale phase, off `residual_demand` + reconcile's
  health verdicts).
- **Scheduling latency:** an RPC wake triggers a **schedule-only mini-tick** (reconcile/
  autoscale skipped unless due), so submit→assign = schedule time, not gated on in-flight
  reconcile.
- **Keep** prune + checkpoint as separate housekeeping threads.

**Serves:** poll over loose threading (#3); one snapshot fans out to DB-free abstractions (#1);
strong scheduler↔autoscaler sync (#6).
**Risk:** highest blast radius; land last; fallback flag; `marin-dev` bake. Never bounce the
live controller without explicit OK (AGENTS.md).

### I2 — One scheduler, one snapshot: co-locate residual demand; autoscaler off it

**Problem.** Demand is a second `find_assignments` dry-run on a second snapshot, limits
disabled (`policy.py:268-285`), against a second `Scheduler`.

**Change.** Keep the limits-free capacity fit (it is real work), but compute it **once, on the
schedule snapshot, with one `Scheduler` instance**, returning it on `BackendResult`:

```python
@dataclass(frozen=True)
class BackendResult:
    assignments: list[Assignment] = field(default_factory=list)
    residual_demand: list[DemandEntry] = field(default_factory=list)   # limits-free capacity-fit residual
    ...                                                                # (other fields below in I3)
```

The autoscale phase consumes `residual_demand` in the same tick. Delete the dry-run, the
second snapshot, the second `Scheduler`, and `compute_demand_entries`. The **golden-fixture
parity test is load-bearing** (same demand entries in/out, incl. reservation/holder cases).

**Serves:** one demand artifact across a typed boundary (#6); one way to compute demand (#5);
removes a snapshot + the unlocked dual-scheduler access (#1).
**Risk:** medium-high — reproduce reservation-taint / holder-task demand semantics exactly.

### I3 — One uniform backend interface; the backend owns placement, capacity, AND health; controller stores `BackendResult`

**Problem.** The controller orchestrates backend specifics: branches on `placement`/
`manages_capacity`; derives health itself; couriers failures via `on_workers_failed`; reads
type-specific result fields; K8s stubs methods.

**Change.** One uniform interface every backend implements; the Iris scheduler, autoscaler,
**and worker-health tracking** live inside `RpcTaskBackend`; the controller persists the
returned `BackendResult` through one pure-kernel apply path, dispatching on result shape only:

```python
class TaskBackend(Protocol):
    name: str
    def schedule(self, snap: ControlSnapshot) -> BackendResult: ...        # DECISION: assignments + residual_demand
    def reconcile(self, snap: ControlSnapshot) -> BackendResult: ...       # OPERATION: observations + worker-health verdicts
    def autoscale(self, snap, residual_demand, health) -> BackendResult: ...# OPERATION: scale + dead-slice/sibling teardown
    def set_log_sink(self, ...) -> None: ...
    def close(self) -> None: ...

@dataclass(frozen=True)
class BackendResult:
    """The DB-shaped write-back the controller commits this tick — applied through one
    writes.py / reconcile-kernel path. Empty fields = 'this backend has nothing here'."""
    assignments: list[Assignment] = field(default_factory=list)
    residual_demand: list[DemandEntry] = field(default_factory=list)
    observations: list[ReconcileObservation] = field(default_factory=list)   # task-state observed
    worker_health: list[WorkerHealthVerdict] = field(default_factory=list)   # healthy/active + counts, per worker
    autoscaler_state: AutoscalerState | None = None
    ...
```

**Worker health moves into the backend (the heart of this revision):**
- `RpcTaskBackend` owns the health tracker; `reconcile` counts its own reconcile-RPC outcomes
  (success → heartbeat; error → `consecutive_failures`++) plus build failures, applies the
  threshold, and returns per-worker `WorkerHealthVerdict`s. When it deems a worker dead it
  evicts its own cached stub (today's `on_worker_failed`) internally — no controller call.
- `autoscale`, given the dead-worker verdicts, tears down their slices **and** their healthy
  siblings (logic already in `autoscaler/operations.py`) and returns the siblings as further
  dead verdicts — so the controller never couriers IDs back in.
- `K8sTaskProvider.reconcile` derives verdicts from pod status — its native health source.
- The controller **stores** verdicts on the worker rows (persist + restore on restart, like
  autoscaler state — fixing the in-memory-only gap) and feeds them back via the next
  `ControlSnapshot` (so `schedule` can filter on health, and read-only RPCs serve stored
  health). The pure reconcile kernel still runs in `commit`, consuming
  `observations + dead-worker set` to compute task transitions (WORKER_FAILED cascades).

**Deleted:** `on_workers_failed`, `on_worker_failed`, `ping_workers`, the controller
`WorkerHealthTracker` mutation logic, and `placement`/`manages_capacity` + every branch.

Also in scope: reconcile the two `Scheduler` instances → one (backend-owned; building cap
sources from it); keep autoscaler startup-restore controller-driven so the backend stays
DB-less per-tick; collapse the two reconcile-result fields via a shared pure resolver.

**Naming (resolved):** keep `BackendResult`. The `*Result` swarm collapses — `PingResult`
deleted (ping gone), `BackendReconcileResult`/`CapacityResult`/`WorkersFailedResult` subsumed
by `BackendResult`, the per-worker `ReconcileResult` (`reconcile/worker.py`) becomes
backend-internal — so there is no remaining collision.

**Serves:** controller = state/storage, backend = operation, **no backend-specific data in the
controller** (#2); one interface + one result shape + one apply path (#5); the multi-backend
keystone (`BackendRegistry` over N backends, routed by the hard `cluster=` constraint).
**Risk:** largest churn; the genuine sub-projects are health-into-backend, autoscaler
DB-decoupling, and the dual-scheduler merge.

### I4 — `reads.py` is the single DB fan-out point feeding the typed `ControlSnapshot`

**Problem.** `_snapshot_reconcile_inputs`' raw join (`controller.py:1038-1059`) bypasses
`reads.py`; same in `policy.py`/`budget.py`/`checkpoint.py`.

**Change.** Build one typed `ControlSnapshot` per tick via `reads.py` (reuse
`reconcile/loader.load_closed_snapshot` where it fits), **including stored worker health** so
the backend can filter/track from it; move the raw selects behind `reads.py`. `reads.py`/
`writes.py`/projections become the only modules issuing schema queries.

**Serves:** central DB queries fan out to DB-free abstractions from one chokepoint (#1).
**Risk:** medium; independent; the `ControlSnapshot` is the backends' uniform input.

### I5 — Policy: every controller→backend RPC is threadpool-dispatched with a bounded (generous) timeout

**Problem.** `RpcTaskBackend.reconcile` runs `asyncio.run(...)` on the caller thread
(`backend.py:153`); `exec_in_container` allows a ~1-hour deadline (`backend.py:232-237`).

**Change — keep the policy, drop the ceremony.** Every call the controller dispatches into a
backend runs on a (per-backend) threadpool with a hard, generous timeout. It is fine for
reconcile to take ~5 s; no constant-time target, no global pool.

1. **One-offs are already pooled** (`profile`/`exec`/`get_process_status` run on the uvicorn
   `rpc-handler` pool, off the tick) — just replace the ~1-hour `exec` deadline with an
   explicit generous cap (e.g. 10–15 min). Do not re-route them.
2. **Wrap the inline calls** — `reconcile` and `autoscale` ops run under
   `future.result(timeout=cap)`. The inner per-worker RPCs already have a 10 s timeout +
   128-way semaphore, so the outer cap is a **fleet-size-aware watchdog**
   (`per_worker_timeout × ceil(workers/128) + slack`), not a fixed constant.
3. **Do not wrap in-process decisions** (`schedule`) — a watchdog over an uncancellable Python
   thread is ceremony; bound only real I/O.

**Serves:** any backend call completes in bounded time, threadpool-dispatched (#4); one
dispatch policy (#5).
**Risk:** low. Care item: choosing the reconcile/autoscale watchdog caps and the exec/profile caps.

## Constraint → improvement coverage

| Constraint | Addressed by |
|---|---|
| Central DB queries fan out to DB-free abstractions | I4 (one `reads.py` snapshot), I3 (snapshot in → `BackendResult` out), I2 (single demand artifact) |
| Controller handles state, backends handle operation | I3 (backend owns placement/capacity/**health**; controller only stores; **no backend-specific data in the controller**) |
| Prefer poll workflows vs loose threading | I1 (one control tick; drop ping/dispatch/autoscaler loops) |
| Any controller→backend call is threadpool-dispatched + bounded | I5 (per-backend pool + generous deadline; fix the 1h exec) |
| Only one way to do something | I3 (one interface + one result shape; delete `on_workers_failed`/`ping_workers`), I2 (one demand path) |
| Strong sync boundary scheduler ↔ autoscaler | I2 (autoscaler consumes the scheduler's `residual_demand`; one scheduler, one snapshot) + I1 |

## Tasks

`exec: session` tasks become weaver issues on `weaver plan sync … --apply`. Ordered by
recommended landing sequence.

### T1 — I5: bound controller→backend RPCs  `exec: session`  `value: high`  `deps: —`
Give the backend a small pool; run `reconcile`/`autoscale` ops under a fleet-size-aware
watchdog deadline; replace the ~1-hour `exec` deadline with an explicit generous cap; leave
the already-pooled one-offs bounded-in-place. Acceptance: a hung worker is surfaced as a
reconcile error within the cap without blocking the caller; no backend call path lacks a timeout.

### T2 — I2: single scheduler + co-located residual demand  `exec: session`  `value: high`  `deps: —`
Compute residual demand once on the schedule snapshot with one `Scheduler`; feed the autoscaler
off it; delete the dry-run + the separate autoscaler thread. Acceptance: golden-fixture parity
(same demand in/out incl. reservation/holder), one `Scheduler`, one scheduling snapshot/cycle.

### T3 — I4: one reads.py-built ControlSnapshot  `exec: session`  `value: medium`  `deps: —`
Build a typed per-tick snapshot via `reads.py`, carrying stored worker health; move the
reconcile-input join + policy/budget/checkpoint raw selects behind `reads.py`. Acceptance: no
schema query outside `reads.py`/`writes.py`/projections; snapshot includes health.

### T4 — I3: uniform backend interface + backend-owned health  `exec: session`  `value: high`  `deps: T2, T3`
Define `BackendResult` + the uniform `schedule`/`reconcile`/`autoscale(snapshot)` interface;
**move worker-health tracking into the backend** (RPC: reconcile-RPC + build failures + threshold;
K8s: pod status), returning per-worker verdicts; **persist health to worker rows** (+ restore on
restart) and feed it back via the snapshot; **delete `on_workers_failed`/`on_worker_failed`/
`ping_workers` + the controller health tracker**; backend tears down dead slices + siblings in
`autoscale`, returning sibling verdicts; reconcile the two `Scheduler` instances → one; keep
autoscaler startup-restore controller-driven; collapse the result fields via a shared resolver;
delete `placement`/`manages_capacity` + branches. Acceptance: `grep` finds no
`placement ==`/`manages_capacity` branch and no `on_workers_failed`/`ping_workers`; backend
imports no `db`/`reads`/`schema` on the per-tick path; a worker whose reconcile RPC fails the
threshold is failed and its slice torn down with no ping loop.

### T5 — I1: collapse the fast loops into one phased control tick  `exec: session`  `value: high`  `deps: T1, T2, T4`
One driver: snapshot → schedule → reconcile(+health) → autoscale → commit; per-phase `due()`;
wake → schedule-only mini-tick; drop ping/dispatch/autoscaler loops; keep prune/checkpoint.
Acceptance: one control thread; one read snapshot + one write txn per tick (counter/test);
submit→assign latency = schedule time; reconcile cadence ≈ old ping interval (no liveness
regression); chaos/integration suite green; behind a fallback flag.

## Feasibility analysis

**Overall: feasible, incremental, and now genuinely simplifying.** Moving health into the
backend *removes* code (ping loop, `on_workers_failed`/`on_worker_failed`/`ping_workers`,
controller `WorkerHealthTracker` mutation logic, the courier-and-loop sibling dance) rather than
adding it, and the sibling/slice logic it relies on already lives in the autoscaler. No
data-model change except adding stored health to the worker rows (a small, beneficial migration
that also fixes restart durability). End state: 3 threads, one interface, one apply path.

**Recommended landing sequence:**

1. **T1 (I5)** — independent; removes the 1-hour `exec` foot-gun and the inline-blocking
   reconcile. Prereq for a safe single tick.
2. **T2 (I2)** — independent; deletes the duplicate scheduling pass, the second snapshot, the
   second `Scheduler`. Care item: demand parity.
3. **T3 (I4)** — independent; produces the `ControlSnapshot` (with health) T4 consumes.
4. **T4 (I3)** — after T2 + T3; largest churn (uniform interface, health-into-backend, autoscaler
   restore decoupling, dual-scheduler merge, flag deletion). Land health-into-backend with the
   ping loop still present as belt-and-suspenders, then remove ping in the same task once the
   backend verdicts are proven.
5. **T5 (I1)** — last; after T1, T2, T4. The tick is then an ordering shell over functions that
   already exist. Fallback flag; `marin-dev` bake.

**Risk register:**

- *T4 (health-into-backend)* — health is mutated from several sites today (ping + reconcile +
  build + cascade); consolidating onto the backend's reconcile must preserve the threshold
  semantics (`PING_FAILURE_THRESHOLD`/`BUILD_FAILURE_THRESHOLD`). Mitigation: keep ping running
  until backend verdicts match it on `marin-dev`, then delete.
- *T5 (single tick)* — phase starvation/latency. Mitigation: per-phase `due()`; wake →
  schedule-only mini-tick; bounded reconcile (T1); fallback flag. Never bounce the live
  controller without explicit OK.
- *T2 (demand parity)* — golden-fixture test gates the dry-run deletion.
- *T4 (autoscaler DB-decoupling)* — `restore_from_db`/`recovery`/`persistence` touch the DB;
  keep restore controller-driven, off the per-tick path.
- *K8s health* — verdicts must come from pod status, not a daemon ping; verify the pod-diff
  reconcile already exposes enough to mark a pod-worker dead.
- *Cross-region / cost* — none; pure control-plane code.
- *Multi-backend itself is out of scope* — these five make it tractable; `BackendRegistry` +
  `cluster=` routing is a follow-on once T4 + T5 land.

**Independently shippable now:** T1, T2, T3. **Sequenced:** T4 (after T2+T3) → T5 (after T1, T2,
T4). Estimated ~6–7 focused sessions.

## Open questions

- **Health persistence shape:** add `healthy`/`active`/`consecutive_failures`/`build_failures`/
  `last_heartbeat_ms` columns to the worker rows, or a small dedicated health table? Either way
  it feeds the snapshot and survives restart.
- **Commit boundary in the tick (I1):** single end-of-tick write (schedule+reconcile+autoscale
  commit together; relies on reconcile self-heal after a crash) vs commit-after-schedule.
  Recommend single commit + self-heal; confirm.
- **Observation resolution (I3):** backend returns raw observations and the controller resolves
  (recommended), vs backend pre-resolves against the in-memory snapshot.
- **I5 caps:** the reconcile/autoscale watchdog caps and the exec/profile caps (values?).
- **`cluster=` routing (follow-on):** hard constraint resolved by a pre-scheduler dispatcher
  that partitions tasks per backend, or inside each backend's eligibility filter?
