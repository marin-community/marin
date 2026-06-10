# T4 — Uniform backend interface + backend-observed worker health

Weaver issue #83. Branch `weaver/i3-uniform-backend-interface-bac`. Builds on #6294/#6295/#6291.

## Goal

Collapse the `TaskBackend` protocol to three uniform per-tick methods
(`schedule` / `reconcile` / `autoscale`) returning ONE `BackendResult`, and make
worker health **observed by the backend, owned (and mutated in one place) by the
controller**. Delete the ping loop, the dispatch loop's separateness, the
`placement`/`manages_capacity` capability branches, and the `*Result` swarm.

## Target protocol

```python
class TaskBackend(Protocol):
    name: str
    capabilities: ClassVar[frozenset[BackendCapability]]   # replaces placement + manages_capacity
    autoscaler: Autoscaler | None
    def schedule(self, snap: ControlSnapshot) -> BackendResult: ...     # pure decision
    def reconcile(self, snap: ControlSnapshot) -> BackendResult: ...    # bounded I/O + health events
    def autoscale(self, snap, residual_demand, dead_workers) -> BackendResult: ...  # provision OR teardown
    # unchanged service-RPC one-offs: exec_in_container / profile_task / get_process_status
    # unchanged lifecycle: set_log_sink / close / attach_autoscaler
```

`BackendCapability` (StrEnum): `WORKER_DAEMON` ("workers"), `IRIS_AUTOSCALER`
("autoscaler"), `CLUSTER_VIEW` ("cluster"). RPC backend =
{WORKER_DAEMON, IRIS_AUTOSCALER}; K8s = {CLUSTER_VIEW}. Dashboard descriptor and
the on-demand service RPC routing (ListWorkers / GetAutoscalerStatus /
GetKubernetesClusterStatus / exec+profile route-by-worker-vs-task) key on
capability membership — NOT on a placement enum. The three per-tick control
loops never branch: they call the uniform method; the backend no-ops where
inapplicable.

## BackendResult (one type; replaces ScheduleResult / BackendReconcileResult /
CapacityResult / WorkersFailedResult / PingResult)

```python
@dataclass(frozen=True)
class BackendResult:
    # schedule outputs
    assignments: list[Assignment] = ()
    preemptions: list[TerminalDecision] = ()
    unschedulable: list[PendingTask] = ()
    residual_demand: list[DemandEntry] = ()
    diagnostics: dict[str, str] = {}
    post_taint_context: SchedulingContext | None = None
    # reconcile outputs (dispatch on FIELD CONTENT, not backend type):
    worker_results: list[tuple[WorkerReconcilePlan, ReconcileResult]] = ()  # IRIS: needs in-kernel overlay synthesis
    updates: list[TaskUpdate] = ()                                          # BACKEND (K8s): neutral updates
    health_events: list[WorkerHealthEvent] = ()                            # transport-observed REACHED/UNREACHABLE
    # autoscale outputs
    removed_workers: list[WorkerId] = ()        # dead + healthy siblings torn down
    autoscaler_state: AutoscalerState | None = None
```

`worker_results` carries `(plan, result)` pairs because the IRIS apply path runs
`ReconcileState.reconcile` whose worker-loss synthesis is overlay-aware
(cross-worker same-batch cascades) — it cannot be pre-converted to neutral
`TaskUpdate`s without the DB/overlay. `ReconcileResult` / `WorkerReconcilePlan`
stay as reconcile-kernel data types (not part of the deleted protocol swarm).
Controller apply dispatches on which field is populated.

## Worker health — observed, owned, single mutation site

New `WorkerHealthEvent{worker_id, kind}` with kind `REACHED | UNREACHABLE |
BUILD_FAILED`. `WorkerHealthTracker.apply(events, *, now_ms) -> list[WorkerId]`
folds events (REACHED → bump heartbeat + reset failures + healthy/active;
UNREACHABLE → consecutive_failures += 1; BUILD_FAILED → build_failures += 1) and
returns workers that *newly* crossed PING/BUILD threshold. `apply` is the SOLE
liveness-accounting mutation site. Lifecycle writes that remain: startup
`_seed_liveness_from_workers` (heartbeat seed), `register` on worker join (join
seed), `forget`/`forget_many` on removal.

Event sources:
- `RpcTaskBackend.reconcile`: per-worker RPC outcome → REACHED (responded) /
  UNREACHABLE (error/timeout). Evicts its own stub on failure (folds in the old
  `on_worker_failed`). These ride back as `BackendResult.health_events`.
- Kernel-derived BUILD_FAILED: `commit_effects` STOPS mutating health; the
  surviving `WorkerHealthEffect.build_failed` flows out in the returned
  `ControllerEffects`; the controller translates it to BUILD_FAILED events.
  (`emit_worker_heartbeat` and `emit_worker_make_unhealthy` are deleted — the
  transport REACHED replaces heartbeat; make_unhealthy was always immediately
  followed by `forget` on removal, so it is redundant.)
- `K8sTaskProvider.reconcile`: pod/node status → health events through the SAME
  `apply` (K8s has no Iris worker rows today, so this is effectively empty, but
  the path is uniform).

Controller per reconcile tick:
```
res = backend.reconcile(snap)                 # transport health_events + worker_results/updates
with db.transaction() as cur:
    effects = apply(...)                       # commits; no health mutation inside
build_failed = effects.health.build_failed
newly_dead = self._health.apply(res.health_events + [BUILD_FAILED(w) for w in build_failed], now_ms=...)
if newly_dead: self._fail_and_teardown(newly_dead, snap)
```

`_fail_and_teardown` (replaces `_terminate_workers` + ping choreography):
```
fr = ops.worker.fail(db, newly_dead, ...)                  # serialize failure
auto = backend.autoscale(snap, [], dead_workers=[w for w,_ in fr.removed_workers])
persist_autoscaler_state(db, auto.autoscaler_state)
siblings = [w for w in auto.removed_workers if w not in dead]
if siblings: ops.worker.fail(db, siblings, ...)            # serialize sibling removal
self._health.forget_many(auto.removed_workers)
```

`autoscale(snap, residual_demand, dead_workers)`: when `dead_workers` is set →
`terminate_slices_for_workers` (dead + siblings), return `removed_workers` +
state, no provisioning. Else → normal `refresh/probe_health/update(residual)` →
state. K8s returns empty either way.

## Loops: 7 → 5

Delete the **ping loop** and the **dispatch loop**. Remaining: scheduling
(`backend.schedule`), polling/reconcile (`backend.reconcile` + health apply +
fail/teardown), autoscaler (`backend.autoscale` provisioning), prune,
checkpoint. `heartbeat_interval` config becomes dead (only the ping + dispatch
loops read it) → removed.

**Cadence:** reconcile runs at `poll_interval`=1s ≤ old ping `heartbeat_interval`
=5s, same 10s per-RPC deadline + threshold-10 semantics → failure detection is
no slower. K8s pod sync moves 5s→1s (more responsive dispatch; acceptable
kubectl load at low-thousand pod scale). Idle workers still get an empty-rows
reconcile plan each tick → heartbeats continue without the ping loop.

## Dispatch fold (K8s)

Move `_dispatch_query` + `PENDING_DISPATCH_COLS` behind `reads.py`. The
controller runs `drain_for_dispatch` (a WRITE — promotes PENDING→ASSIGNED, stays
controller-side) as a capability-gated section of the reconcile snapshot when
the backend lacks an Iris scheduler (CLUSTER_VIEW). `K8sTaskProvider.reconcile`
consumes `tasks_to_run`/`running_tasks` from the snapshot and returns `updates`.
The separate dispatch loop is deleted. RPC backends build worker plans from
`reconcile_rows` internally (`build_reconcile_plans` moves into the RPC backend).
If this balloons, fall back to a thin `backend.reconcile`-calling dispatch loop
(flag in PR).

## Out of scope (later tasks): collapse loops into one phased tick (T5); replace
detached cloud-op threads with polled ops (T6).
