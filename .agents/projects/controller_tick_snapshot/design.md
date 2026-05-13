# Controller Tick Snapshot

_Why are we doing this? What's the benefit?_

The Iris controller runs four concurrent loops (scheduler, polling, autoscaler, ping). Each opens its own SQLite read snapshots and queries near-identical worker / task / job state. Two of them (polling, autoscaler) also do **synchronous network I/O inside the loop body** — polling fans out `PollTasks` RPCs to every worker; autoscaler calls `tpu_describe` to GCE. A recent production profile (3.5 s wall) shows the four loops collectively burn **~2 s of CPU on redundant reads (~57% of one core)**; on autoscaler ticks ~650 ms is sunk in inline GCE HTTP; on polling ticks the tick wall time is dominated by waiting for slow worker RPCs to come back. Beyond CPU, the loops can disagree within a single tick — see [#5470](https://github.com/marin-community/marin/issues/5470), where the scheduler and polling threads observed different worker capacity and double-dispatched a gang.

This design moves to a clean separation: **decision work happens on one tick driver thread; network I/O happens on dedicated data threads that populate in-memory caches**. The tick driver builds a `ControlTick` snapshot (DB reads from one `read_snapshot()` + pinned references to the current cache values) and runs scheduler / polling / autoscaler sections synchronously against that snapshot. Intents collected during the tick are bulk-applied in one write transaction (Phase 3). Worker liveness ping stays on its own thread because failure detection must remain sub-second. Net effect: tick body is fully DB-bound (no network I/O), one canonical view of the world per tick, and one write-lock acquisition per tick.

## Background

The pieces are already there. `_SchedulingStateRead` ([controller.py:210](https://github.com/marin-community/marin/blob/0d440a1b20d23f038fce7c5e0d6ab6d9833fb268/lib/iris/src/iris/cluster/controller/controller.py#L210)) is a frozen-dataclass snapshot bundling `pending_tasks` and `workers` for the scheduler — it just isn't shared. The four control loops are independently threaded ([controller.py:1428–1472](https://github.com/marin-community/marin/blob/0d440a1b20d23f038fce7c5e0d6ab6d9833fb268/lib/iris/src/iris/cluster/controller/controller.py#L1428-L1472)) with different cadences (scheduler 1–10 s with `ExponentialBackoff`, polling 1 s, ping 5 s, autoscaler 10–30 s) and wired by per-loop wake events. The controller spawns two further threads (`_direct_provider_thread`, `_prune_thread`) which are out of scope for this design. All writes go through `transitions.*` under one shared `RLock` ([db.py:221](https://github.com/marin-community/marin/blob/0d440a1b20d23f038fce7c5e0d6ab6d9833fb268/lib/iris/src/iris/cluster/controller/db.py#L221)). Prior designs touch the perimeter: `iris-autoscaler-refactor-plan.md` introduces a frozen `ScalePlan` the autoscaler runtime would consume; `20260223_iris_resultectomy.md` deliberately removed an older cross-loop write-queue. Its rationale (don't entangle independent loops with shared mutable queues) is preserved here by making the tick driver the single *owner* of all batched mutations, not a shared queue between loops. See [research.md](./research.md) for the full profile breakdown and call-site inventory.

## Challenges

1. **Cadence asymmetry + I/O latency in the tick body.** Today's polling and autoscaler loops do synchronous network I/O. If we naively pull both into one tick body, the tick's wall time becomes the sum of "DB reads + worker RPCs + GCE HTTP" — many seconds in the worst case, blocking the scheduler. The design avoids this by introducing two background **data threads** (`TpuMonitor`, `WorkerPollFanout`) that own the I/O and publish to in-memory caches. The tick body reads the caches and never blocks on network. Scheduler backoff (currently `ExponentialBackoff`) is preserved as a per-section gate inside the tick driver.

2. **Heartbeat correctness with cached poll state.** Polling's `apply_heartbeats_batch` (~1.7 s inclusive in the profile) depends on the *latest* `task_attempts` rows to decide each transition. With `WorkerPollFanout` populating a cache, the cache value can be up to one poll-cadence stale (≤ 1 s). The tick body produces heartbeat intents from cache + snapshot, and each intent CAS-checks against the live `task_attempts.attempt_id` inside the bulk-apply transaction; stale intents drop and the next tick fires immediately.

3. **Ping stays separate.** Worker-failure detection must remain sub-second. Ping writes are tiny (`update_worker_pings` is a single UPDATE; `fail_workers_batch` is rare) and they must not be gated on tick latency. Ping continues to write directly under the shared `RLock`. The Phase 3 lock-hold-time concern (see Costs / Risks) is the main constraint on `apply_tick_intents` batch size.

4. **`_scheduling_diagnostics` is RPC-visible.** `get_job_scheduling_diagnostics` ([controller.py:2218](https://github.com/marin-community/marin/blob/0d440a1b20d23f038fce7c5e0d6ab6d9833fb268/lib/iris/src/iris/cluster/controller/controller.py#L2218)) reads `_scheduling_diagnostics` populated by `_cache_scheduling_diagnostics` ([controller.py:2180](https://github.com/marin-community/marin/blob/0d440a1b20d23f038fce7c5e0d6ab6d9833fb268/lib/iris/src/iris/cluster/controller/controller.py#L2180)). This must migrate to the `TickDriver` in Phase 1 — it is hidden Phase-1 scope.

## Costs / Risks

- **Bigger first PR.** Phase 1 replaces two production threads with the TickDriver and *adds* `TpuMonitor`, plus refactors `AutoscalerRuntime` to consume cached cloud state instead of calling GCE inline. The win is that the tick body is purely DB-bound from Phase 1 onward.
- **Memory.** `ControlTick` carries `task_attempts` from Phase 2 (~200 k rows at current scale ≈ 40 MB per tick). Held only during the tick body and GC'd at return. If this becomes unworkable at >10× scale, the answer is a larger controller VM — the data has to live somewhere and we already keep it in SQLite cache; one extra in-process copy is acceptable.
- **Ping and gRPC-handler write contention under Phase 3 bulk apply.** Phase 3 increases per-tick write-lock-hold time (one big transaction vs several small ones). Ping's `update_worker_pings` and the gRPC handler write paths (`RegisterEndpoint`, `LaunchJob`, `Register`, ...) all acquire the same `ControllerDB._lock`. We bound this by capping `apply_tick_intents` heartbeat sub-batch size at 200 updates (configurable); the bench measurement puts a 200-update `apply_heartbeats_batch` at ~10 ms p95 today.
- **Phases are sequentially dependent, not independent.** Phase 2 requires Phase 1's `ControlTick`. Phase 3 requires Phase 2 because intent emission only makes sense when all three sections live in one thread. Each phase is independently *revertible* without rolling back the next-older phase, but each must land *after* the previous one.
- **Hidden Phase-1 scope** beyond the snapshot itself: `_scheduling_diagnostics` migration; deletion of `_run_scheduling_loop` and `_run_autoscaler_loop`; producers of `_scheduling_wake` rewired to set `TickDriver.wake`; the `AutoscalerRuntime` cloud-probe refactor (see [spec.md](./spec.md)).
- **Rollback story** in [spec.md](./spec.md#rollback). Each phase ships behind a single revert-able commit; the previous loop code is *deleted*, so revert means re-introducing the loops from the prior commit. There is no feature flag — the cost is a deliberate forcing function to land each phase clean.
- **Cache staleness vs. correctness.** The cache pattern means the tick body sees data up to one cache-refresh-interval old. For polling (~1 s) this is the same as today's polling cadence. For cloud state (~10–30 s) this matches today's autoscaler cadence. The CAS check inside `apply_tick_intents` covers the DB-side staleness window; cloud-side staleness is bounded by the existing autoscaler tolerance.

## Design

Three phases. Each is **revertible** by undoing one commit; **shipping order** is strictly 1 → 2 → 3.

### Phase 1 — Tick driver + TpuMonitor (biggest single win)

Introduce `controller/tick.py` with `ControlTick` (frozen dataclass) and `TickDriver`. The driver replaces `_run_scheduling_loop` and `_run_autoscaler_loop`. Polling and ping stay on their own threads for Phase 1.

Also introduce `controller/cloud_monitor.py` with `TpuMonitor`: a background thread that periodically (~10 s, configurable) calls `slice_handle.describe()` for each managed slice and publishes the results to an in-memory `CloudStateCache`. The `AutoscalerRuntime.refresh()` path is refactored to read from this cache instead of calling `slice_handle.describe()` inline. The autoscaler section inside the tick body becomes pure decision work (~ms instead of ~650 ms).

```python
def _tick(self) -> None:
    snapshot = self._build_control_tick()        # DB reads + pinned cache refs
    if self._scheduler_due(snapshot):
        self._run_scheduler(snapshot)            # writes inline (Phase 1)
    if self._autoscaler_due(snapshot):
        self._run_autoscaler(snapshot)           # reads from cloud cache; writes inline (Phase 1)
    self._update_diagnostics(snapshot)
```

`ControlTick` is a frozen dataclass bundling `pending_tasks`, `healthy_workers`, `resource_usage`, `tasks_index`, `reserved_jobs`, `cloud_state` (pinned reference to `CloudStateCache` snapshot), and `state_read_ms`. Built inside one `read_snapshot()` so all DB fields are mutually consistent; the cache reference is captured at the same moment.

**Reclaimed**:
- Removes the autoscaler's inline GCE HTTP from the critical path (~650 ms / autoscaler tick).
- Dedupes `_schedulable_tasks` and `healthy_active_workers_with_attributes` between scheduler and autoscaler when both ticks fire in the same iteration (~1.6 s saved per 3.5 s window when they overlap; amortized ~50–150 ms/sec wall steady-state).
- One fewer long-running control loop (scheduler+autoscaler → tick), trading for one new I/O-only thread (`TpuMonitor`). Same headcount, very different shape.

**Implementation note**: Phase 1 may land as **two PRs** if the diff is too large for one review: 1a) `TpuMonitor` + `AutoscalerRuntime` cache refactor (standalone benefit — moves cloud probes out of the hot path regardless of whether the tick driver lands); 1b) `TickDriver` + `ControlTick` replacing the scheduler & autoscaler threads. The decision is the implementer's; the design is single-phase logically.

### Phase 2 — WorkerPollFanout; polling merges into the tick

Introduce `controller/worker_poll_fanout.py` with `WorkerPollFanout`: a thread (+ small `ThreadPoolExecutor` for parallel RPCs) that continuously fans out `PollTasks` RPCs at the polling cadence (~1 s) and publishes results to an in-memory `WorkerPollCache`. `_run_polling_loop` is deleted.

Polling logic moves into the tick body. `ControlTick` gains `task_attempts` (unfinished worker-bound rows) and `worker_poll_state` (pinned `WorkerPollCache` reference). The polling section consumes the cache plus `task_attempts` to decide heartbeat transitions; in Phase 2 it calls `transitions.apply_heartbeats_batch` inline.

After Phase 2: three threads in the control plane (tick + ping + `WorkerPollFanout`) plus one data-only thread (`TpuMonitor`). `_scheduling_wake` and `_polling_wake` collapse into a single `TickDriver.wake`.

**Dispatch path (RunTask RPCs)**: Phase 2 covers only the *read* side of worker communication. Writes — `RunTaskRequest` to newly-ASSIGNED tasks and `KillTaskRequest` to terminated ones — continue to be issued from the polling section (now living in the tick body), fire-and-forget through gRPC's async client. The worker's eventual state reaches us via the next `WorkerPollFanout` round. No separate dispatch thread is introduced.

**Reclaimed**: tick body no longer waits on worker RPCs; cache refresh is decoupled from decision cadence. Dedupes the polling-side `list_active_healthy_workers` against the scheduler's read.

### Phase 3 — Bulk-apply writes

Each section returns typed `Intent` lists (`QueueAssignmentIntent`, `HeartbeatApplyIntent`, etc.) instead of calling `transitions.*` directly. The tick driver builds one `IntentBatch` and applies via `transitions.apply_tick_intents(cur, batch) -> ApplyTickResult` under a single write-lock acquisition. Each intent CAS-validates against the live row inside the apply transaction; stale intents drop with a reason and the next tick is woken so the regenerated state can retry. Ping stays out of the batch — it acquires the write lock independently with its tiny writes.

Application order inside `apply_tick_intents` is fixed and **rationale-justified per pair** (see [spec.md](./spec.md#application-order-rationale)). Empty `IntentBatch` is a no-op (no lock acquisition). Heartbeat sub-batches are capped at 200 updates per transaction to bound contention with ping and gRPC writes.

**Reclaimed**: per-tick write-transaction count drops from up to ~5 (scheduler-section worst case) plus 1 (polling-section heartbeat apply) to 1 in the steady state. The impact on `apply_heartbeats_batch` p95 is harder to predict (it depends on how often it currently contends with scheduler-section writes in production); the implementation PR is expected to measure it on a checkpoint replay before declaring victory.

## Testing

Each phase ships with integration tests loading the production checkpoint at `gs://marin-us-central2/iris/marin/state` (cached locally for CI; see spec for fixture details).
- Phase 1: parity test asserting `ControlTick` content equals the union of the prior per-loop reads on the same DB state, plus a fake `TpuMonitor` cache exercising the autoscaler section against pre-canned cloud states.
- Phase 2: extends parity to the polling section consuming a fake `WorkerPollCache`; asserts heartbeat-apply results match the pre-refactor inline path.
- Phase 3: property test — for any sequence of intents on a frozen snapshot, `apply_tick_intents` produces results equivalent to applying each intent in its old per-loop transaction (modulo CAS-failed drops).

The existing controller integration suite (`lib/iris/tests/cluster/controller/`) runs against each phase to catch externally-visible regressions. **Production rollout**: ship Phase 1, watch tick-duration histogram and lock-wait p99 for a week against published thresholds; revert if either trips; proceed to Phase 2.

## Open Questions

1. **Backoff hysteresis preservation.** Today's `ExponentialBackoff` grows on idle (1 s → 10 s) and resets on work-found. The spec preserves this by resetting on any tick where the scheduler section produced work. Reviewers: is a single `tick_interval_s` cadence (always 1 s, scheduler skipped when idle) preferable to preserving the existing hysteresis? Simpler vs less-faithful.

2. **Lock-hold contention with ping.** Phase 3's bulk apply increases lock-hold per tick. The spec caps the heartbeat sub-batch at 200 updates (expected ~10 ms p95 lock-hold, budgeted at 50 ms). Open: do we accept that, or do we need a hard SLO with auto-split on overrun?

3. **Stale-intent convergence under tight regeneration loops.** When the same intent is regenerated and CAS-failed N ticks in a row (e.g. an assignment for a task whose `attempt_id` is being rapidly bumped by an external path), the spec proposes "wake once, regenerate from fresh snapshot, no automatic re-enqueue." Reviewers: sufficient, or do we need per-intent backoff?

4. **`replace_reservation_claims` placement in Phase 3.** Today three call sites ([controller.py:1763, 1811, 1908](https://github.com/marin-community/marin/blob/0d440a1b20d23f038fce7c5e0d6ab6d9833fb268/lib/iris/src/iris/cluster/controller/controller.py#L1763)) each rewrite the table. In Phase 3 they become one `ReservationClaimsIntent` per tick. Do we need three separate intents and re-validate after each, or can a single end-of-tick replacement preserve the invariant?

5. **`WorkerPollFanout` cadence and saturation.** Phase 2 introduces a continuously-fanning-out background thread. At 340 workers and a 1 s cadence we issue ~340 RPCs/sec; the existing polling loop already does this at the same rate. Open: should `WorkerPollFanout` be self-rate-limiting (slow down if the previous round didn't finish before the next was due), or should it pipeline (which can saturate a small TPE under tail latency)?
