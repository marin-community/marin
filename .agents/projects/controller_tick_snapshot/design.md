# Controller Tick Snapshot

_Why are we doing this? What's the benefit?_

The Iris controller runs four concurrent loops (scheduler, polling, autoscaler, ping). Each opens its own SQLite read snapshots and queries near-identical worker / task / job state. A recent production profile (3.5 s wall) shows the four loops collectively burn **~2 s of CPU on redundant reads (~57% of one core)**, with `_schedulable_tasks` decoded twice per tick (scheduler + autoscaler, ~1.3 s combined) and `healthy_active_workers_with_attributes` read three times (scheduler + autoscaler + ping, ~0.5 s combined). Beyond CPU, the loops can disagree within a single tick — see [#5470](https://github.com/marin-community/marin/issues/5470), where the scheduler and polling threads observed different worker capacity and double-dispatched a gang.

This design replaces the three coupled loops (scheduler / polling / autoscaler) with a single **tick driver** thread that builds one consistent `ControlTick` snapshot per iteration and runs scheduling, polling, and autoscaler work directly off that snapshot. Ping stays on its own thread because worker-failure detection must remain sub-second. Once everyone shares the snapshot, intents collected during the tick are applied in one bulk write transaction — collapsing per-tick write transactions (scheduler section: up to 4 today; polling: 1 heartbeat tx; autoscaler: 0) into one. Net effect: ~half a core back, one canonical view of the world per tick, two long-running threads instead of four, and a single write-lock acquisition for the active section of each tick.

## Background

The pieces are already there. `_SchedulingStateRead` ([controller.py:210](https://github.com/marin-community/marin/blob/0d440a1b20d23f038fce7c5e0d6ab6d9833fb268/lib/iris/src/iris/cluster/controller/controller.py#L210)) is a frozen-dataclass snapshot bundling `pending_tasks` and `workers` for the scheduler — it just isn't shared. The four loops are independently threaded ([controller.py:1428–1472](https://github.com/marin-community/marin/blob/0d440a1b20d23f038fce7c5e0d6ab6d9833fb268/lib/iris/src/iris/cluster/controller/controller.py#L1428-L1472)) with different cadences (scheduler 1–10 s with `ExponentialBackoff`, polling 1 s, ping 5 s, autoscaler 10–30 s) and wired by per-loop wake events. The controller spawns two further threads (`_direct_provider_thread`, `_prune_thread`) which are out of scope for this design. All writes go through `transitions.*` under one shared `RLock` ([db.py:221](https://github.com/marin-community/marin/blob/0d440a1b20d23f038fce7c5e0d6ab6d9833fb268/lib/iris/src/iris/cluster/controller/db.py#L221)). Prior designs touch the perimeter: `iris-autoscaler-refactor-plan.md` introduces a frozen `ScalePlan` the autoscaler runtime would consume; `20260223_iris_resultectomy.md` deliberately removed an older cross-loop write-queue. Its rationale (don't entangle independent loops with shared mutable queues) is preserved here by making the tick driver the single *owner* of all batched mutations, not a shared queue between loops. See [research.md](./research.md) for the full profile breakdown and call-site inventory.

## Challenges

1. **Cadence asymmetry.** Today: scheduler 1–10 s (backoff), polling 1 s, autoscaler 10–30 s. We collapse the first three under one tick that runs at the most-frequent cadence (~1 s) and conditionally executes the slower work each iteration ("run autoscaler if ≥10 s since its last run"). Backoff/idle behavior — currently scheduler-only via `ExponentialBackoff` — moves into the tick driver as a per-section gate that preserves the existing hysteresis (reset on work-found, grow on idle).

2. **Heartbeat correctness.** Polling's `apply_heartbeats_batch` (~1.7 s inclusive in the profile) depends on the *latest* `task_attempts` rows to decide each transition. Bulk-applying intents built from a snapshot up to a tick old could miss transitions that happened mid-tick. We mitigate by including `task_attempts` in the snapshot *and* checking each intent's `attempt_id` against the live row inside the bulk-apply transaction (optimistic CAS); stale intents are dropped, recorded in the result, and a wake event re-triggers the next tick.

3. **Inline-HTTP in the autoscaler — promoted to risk, not Open Question.** The autoscaler currently calls `handle.describe()` → `_describe_cloud` → `tpu_describe` synchronously ([autoscaler/runtime.py:425](https://github.com/marin-community/marin/blob/0d440a1b20d23f038fce7c5e0d6ab6d9833fb268/lib/iris/src/iris/cluster/controller/autoscaler/runtime.py#L425)) — there's no TPU-monitor cache. Profile shows ~650 ms per autoscaler tick. We do not try to fix it here. Phase 1's tick driver runs `autoscaler.refresh()` inline and caps autoscaler frequency until a sibling design adds a background `TpuMonitor` cache.

4. **Ping stays separate** with its own RLock acquisition path. Ping writes are tiny (`update_worker_pings` is one UPDATE; `fail_workers_batch` is rare). The Phase 3 lock-hold-time risk (see Costs / Risks) is the main constraint on `apply_tick_intents` batch size.

5. **`_scheduling_diagnostics` is RPC-visible.** `get_job_scheduling_diagnostics` ([controller.py:2218](https://github.com/marin-community/marin/blob/0d440a1b20d23f038fce7c5e0d6ab6d9833fb268/lib/iris/src/iris/cluster/controller/controller.py#L2218)) reads `_scheduling_diagnostics` populated by `_cache_scheduling_diagnostics` ([controller.py:2180](https://github.com/marin-community/marin/blob/0d440a1b20d23f038fce7c5e0d6ab6d9833fb268/lib/iris/src/iris/cluster/controller/controller.py#L2180)). This must migrate to the `TickDriver` in Phase 1 — it is hidden Phase-1 scope.

## Costs / Risks

- **Bigger first PR.** Phase 1 replaces two production threads with a new driver. The smaller-PR alternative (pub/sub) was rejected because it preserves the thread topology we're trying to eliminate. Honest trade.
- **Memory grows with the snapshot.** Phase 1: single-digit MB (`pending_tasks` + `healthy_workers` at current scale). Phase 2 adds `task_attempts` (~200k rows in production); at ~200 B per `ActiveTaskRow` that is ~40 MB live for the tick body. Bounded by tick duration; GC'd at end of tick.
- **Ping and gRPC-handler write contention under Phase 3 bulk apply.** Phase 3 increases per-tick write-lock-hold time (one big transaction vs several small ones). Ping's `update_worker_pings` and the gRPC handler write paths (`RegisterEndpoint`, `LaunchJob`, `Register`, ...) all acquire the same `ControllerDB._lock`. We bound this by capping `apply_tick_intents` heartbeat sub-batch size at 200 updates (configurable). The bench measurement from the SA Core PR puts a 200-update `apply_heartbeats_batch` at ~10 ms p95 today; the spec budgets ~50 ms worst-case for headroom.
- **Phases are sequentially dependent, not independent.** Phase 2 requires Phase 1's `ControlTick`. Phase 3 requires Phase 2 because intent emission only makes sense when all three sections live in one thread. Each phase is independently *revertible* without rolling back the next-older phase, but each must land *after* the previous one.
- **Hidden Phase-1 scope** beyond the snapshot itself: `_scheduling_diagnostics` migration; deletion of `_run_scheduling_loop` and `_run_autoscaler_loop`; producers of `_scheduling_wake` rewired to set `TickDriver.wake` (full migration is in [spec.md](./spec.md#wake-event-migration)).
- **Rollback story** in [spec.md](./spec.md#rollback). Each phase ships behind a single revert-able commit; the previous loop code is *deleted*, so revert means re-introducing the loops from the prior commit. There is no feature flag — the cost is a deliberate forcing function to land each phase clean.
- **Snapshot-time consistency caveat.** `read_snapshot()` uses SQLite's deferred-transaction default; SQLite WAL gives readers a stable snapshot from the BEGIN onward. The spec documents the exact semantics so reviewers can confirm `ControlTick` truly does see one consistent DB state.

## Design

Three phases. Each is **revertible** by undoing one commit; **shipping order** is strictly 1 → 2 → 3.

### Phase 1 — Tick driver replaces scheduler + autoscaler threads (biggest single win)

Introduce `controller/tick.py` with `ControlTick` (frozen dataclass) and `TickDriver` (thread owner). The driver replaces `_run_scheduling_loop` and `_run_autoscaler_loop`. Polling and ping stay on their own threads. Each iteration of the driver:

```python
def _tick(self) -> None:
    snapshot = self._build_control_tick()        # one read_snapshot()
    if self._scheduler_due(snapshot):
        self._run_scheduler(snapshot)            # writes inline (Phase 1)
    if self._autoscaler_due(snapshot):
        self._run_autoscaler(snapshot)           # writes inline (Phase 1)
    self._update_diagnostics(snapshot)           # incl. _scheduling_diagnostics migration
```

`ControlTick` is a frozen dataclass bundling `pending_tasks`, `healthy_workers`, `resource_usage`, `tasks_index`, `reserved_jobs`, `state_read_ms`. Built inside one `read_snapshot()` so all fields are mutually consistent (see [spec.md](./spec.md#controltick-phase-1) for the snapshot-build order). The scheduler and autoscaler are called as plain functions taking `ControlTick`; their per-thread state (`ExponentialBackoff`, autoscaler `RateLimiter`, `_scheduling_diagnostics`) migrates into `TickDriver`.

**Reclaimed**: ~1.6 s per 3.5 s window when the autoscaler runs. Amortized across the full autoscaler cadence (10–30 s vs the scheduler's 1 s) the steady-state reclamation is ~50–150 ms per second of wall time, not 1.6/3.5. The dominant duplication (`_schedulable_tasks` decoded twice) goes away on every autoscaler tick.

### Phase 2 — Polling merges into the tick driver

Polling logic moves into the tick body. `ControlTick` gains `task_attempts` (unfinished worker-bound rows) and `run_request_templates`. The polling RPC fan-out runs in a small `ThreadPoolExecutor` owned by the `TickDriver`; the tick awaits results (with per-RPC timeout) before invoking `apply_heartbeats_batch` inline. After Phase 2, two long-running threads remain (tick + ping). `_scheduling_wake` and `_polling_wake` collapse into a single `TickDriver.wake` event; all producers migrate (full list in spec).

**Reclaimed**: dedupes polling reads with scheduler reads (`list_active_healthy_workers` + the task_attempts join); kills `_get_active_worker_addresses` (~0.8 s in the profile).

### Phase 3 — Bulk-apply writes

Each section returns typed `Intent` lists instead of calling `transitions.*` directly. The tick driver builds one `IntentBatch` and applies it via `transitions.apply_tick_intents(cur, batch) -> ApplyTickResult` under a single write-lock acquisition. Each intent CAS-validates against the live row inside the apply transaction; stale intents are dropped and the next tick is woken so the regenerated state can retry. Ping stays out of the batch — it acquires the write lock independently with its tiny writes.

Application order inside `apply_tick_intents` is fixed and **rationale-justified per pair** (see [spec.md](./spec.md#application-order-rationale)). Empty `IntentBatch` is a no-op (no lock acquisition). Heartbeat sub-batches are capped at 200 updates per transaction to bound ping-starvation risk (see Costs).

**Reclaimed**: per-tick write-transaction count drops from up to ~5 (scheduler-section worst case) to 1 in the steady state. The 3 → 1 acquisition reduction for the scheduler section is direct; the impact on `apply_heartbeats_batch` p95 is harder to predict (it depends on how often it currently contends with scheduler-section writes in production) and the design does not commit to a specific multiplier — the implementation PR is expected to measure it on a checkpoint replay before declaring victory.

## Testing

Each phase ships with integration tests loading the production checkpoint at `gs://marin-us-central2/iris/marin/state` (cached locally for CI; see spec for fixture details). Phase 1: parity test asserting `ControlTick` content equals the union of the prior per-loop reads on the same DB state. Phase 2: extends parity to polling RPC fan-out + heartbeat-apply results. Phase 3: property test — for any sequence of intents on a frozen snapshot, `apply_tick_intents` produces results equivalent to applying each intent in its old per-loop transaction (modulo CAS-failed drops). The existing controller integration suite (`lib/iris/tests/cluster/controller/`) runs against each phase to catch externally-visible regressions.

**Production rollout**: ship Phase 1, watch the tick-duration histogram and lock-wait p99 for a week against published thresholds (spec.md). If either trips, revert; otherwise proceed to Phase 2.

## Open Questions

1. **Backoff hysteresis preservation.** Today's `ExponentialBackoff` grows on idle (1 s → 10 s) and resets on work-found. The spec preserves this by resetting `scheduler_idle_max_s` to its minimum on any tick where the scheduler section produced work. Reviewers: is a single `tick_interval_s` cadence (always 1 s, scheduler skipped when idle) preferable to preserving the existing hysteresis? Simpler vs less-faithful.

2. **Lock-hold contention with ping.** Phase 3's bulk apply increases lock-hold per tick. The spec caps the heartbeat sub-batch at 200 updates and documents an expected worst-case lock-hold of ~50 ms. Open: do we accept that, or do we need a hard SLO with auto-split on overrun?

3. **Stale-intent convergence under tight regeneration loops.** When the same intent is regenerated and CAS-failed N ticks in a row (e.g. an assignment for a task whose `attempt_id` is being rapidly bumped by an external path), what bounds tight retry? The spec proposes "wake once, regenerate from fresh snapshot, no automatic re-enqueue" — sufficient, or do we need per-intent backoff to prevent thrashing?

4. **`replace_reservation_claims` placement in Phase 3.** Today three call sites ([controller.py:1763, 1811, 1908](https://github.com/marin-community/marin/blob/0d440a1b20d23f038fce7c5e0d6ab6d9833fb268/lib/iris/src/iris/cluster/controller/controller.py#L1763)) each rewrite the table. In Phase 3 they become one `ReservationClaimsIntent` per tick — but the three points in the scheduling cycle exist for reasons (recompute after preemption, after assignment, after timeout cancellation). Do we need three separate intents and re-validate after each, or can a single end-of-tick replacement preserve the invariant?

5. **`ControlTick` memory bound after Phase 2.** ~40 MB per tick at current scale (~200 k attempts). At 1 s ticks this is fine; at 10× scale we'd want streaming/lazy materialization of `task_attempts`. Open: do we put a size guard now, or revisit when growth materializes?
