# Iris Worker Health Score

## Problem

Today a worker is removed only when:

1. 10 consecutive heartbeat RPCs fail — `HEARTBEAT_FAILURE_THRESHOLD` at `lib/iris/src/iris/cluster/controller/transitions.py:89`, counter logic in `_record_heartbeat_failure` at `transitions.py:2166`.
2. Last heartbeat age exceeds 15 min — `_reap_stale_workers` at `lib/iris/src/iris/cluster/controller/controller.py:2377`.

Task-level failure signals that actually indicate a broken worker never feed the termination decision. A TPU worker that pings fine but returns `TASK_STATE_PREEMPTED` or `TASK_STATE_WORKER_FAILED` (see [issue #4817](https://github.com/marin-community/marin/issues/4817)) sits in the pool and keeps eating dispatches.

The current counter also resets to zero on any successful ping, so a worker that alternates (ping ok, task fails, ping ok, task fails) never reaches 10.

## Goals

- Terminate workers whose recent behavior indicates the TPU is bad, not just whose RPCs time out.
- Accumulate evidence across different signal types (RPC failures, preempted tasks, worker-failed tasks) with per-signal weights.
- Forget old evidence — one rough stretch three hours ago shouldn't kill an otherwise healthy worker.
- Keep termination decisions simple and testable in isolation from the provider sync and task-update code paths.
- One termination path, not two. The current inline `consecutive_failures >= 10` rule goes away.

Non-goals:

- Fold heartbeat age into the score equation. Age is a clean standalone predicate; it stays a separate check — but it moves onto the same reaper thread so there's one periodic termination loop, not two.
- Score-based autoscaling or worker ranking for scheduling. This is strictly about terminating obviously-broken workers.
- Cross-restart persistence. Controller restarts are rare and the signal rebuilds in minutes; see Tradeoffs.
- Error-string classification. The worker-reported `TaskState` is authoritative. If a new class of worker-fatal failure emerges that isn't currently reflected in the state enum, the fix is to surface it worker-side (a new state or error code), not to string-match on the controller.

## Proposed Solution

Introduce a `WorkerHealthTracker` — a pure in-memory object owned by the controller, mutated by the existing provider and task-update threads, and polled by a new reaper thread.

```python
# lib/iris/src/iris/cluster/controller/worker_health.py

class HealthSignal(StrEnum):
    RPC_FAILURE = "rpc_failure"
    TASK_WORKER_FAILED = "task_worker_failed"

SIGNAL_WEIGHT: dict[HealthSignal, float] = {
    HealthSignal.RPC_FAILURE: 1.0,
    HealthSignal.TASK_WORKER_FAILED: 1.0,
}
HEALTH_SCORE_THRESHOLD = 10.0
HEALTH_SCORE_HALF_LIFE_S = 300.0

@dataclass(slots=True)
class _ScoreEntry:
    score: float
    updated_ms: int

class WorkerHealthTracker:
    """Tracks per-worker failure scores with exponential decay.

    Thread-safe: bumped from the provider and task-update threads,
    read from the reaper thread.
    """

    def __init__(
        self,
        *,
        half_life_s: float = HEALTH_SCORE_HALF_LIFE_S,
        threshold: float = HEALTH_SCORE_THRESHOLD,
        weights: Mapping[HealthSignal, float] = SIGNAL_WEIGHT,
        clock: Callable[[], int] = lambda: Timestamp.now().epoch_ms(),
    ) -> None:
        self._lam = math.log(2) / half_life_s
        self._threshold = threshold
        self._weights = dict(weights)
        self._clock = clock
        self._lock = threading.Lock()
        self._entries: dict[WorkerId, _ScoreEntry] = {}

    def bump(self, worker_id: WorkerId, signal: HealthSignal) -> float:
        now = self._clock()
        weight = self._weights[signal]
        with self._lock:
            entry = self._entries.get(worker_id)
            decayed = self._decay(entry.score, now - entry.updated_ms) if entry else 0.0
            new_score = decayed + weight
            self._entries[worker_id] = _ScoreEntry(new_score, now)
            return new_score

    def current_score(self, worker_id: WorkerId) -> float:
        now = self._clock()
        with self._lock:
            entry = self._entries.get(worker_id)
            return self._decay(entry.score, now - entry.updated_ms) if entry else 0.0

    def workers_over_threshold(self) -> list[tuple[WorkerId, float]]:
        now = self._clock()
        with self._lock:
            return [
                (wid, self._decay(e.score, now - e.updated_ms))
                for wid, e in self._entries.items()
                if self._decay(e.score, now - e.updated_ms) >= self._threshold
            ]

    def forget(self, worker_id: WorkerId) -> None:
        with self._lock:
            self._entries.pop(worker_id, None)

    def _decay(self, score: float, dt_ms: int) -> float:
        if dt_ms <= 0:
            return score
        return score * math.exp(-self._lam * dt_ms / 1000.0)
```

### Wire-up

**Construction.** The controller owns a single `WorkerHealthTracker`, constructed in `Controller.__init__` next to the other per-controller collaborators.

**Bump sites.** Two call sites feed evidence into the tracker. The source already knows which signal it is — no string matching, no secondary classification.

1. Heartbeat RPC failures — `controller.py` around the `fail_heartbeats_batch` call site. Add `self._health.bump(worker_id, HealthSignal.RPC_FAILURE)` next to each recorded failure.

2. Task outcomes — in `_apply_task_transitions` at `transitions.py` (the function driven by `apply_task_updates` at `transitions.py:1959`), when a task transitions into `TASK_STATE_WORKER_FAILED`, emit a `bump`. Since `transitions.py` is DB-only today, the cleanest split is:

   - `_apply_task_transitions` returns a list of `(worker_id, HealthSignal)` observations in its `TxResult`. The mapping is direct: `TASK_STATE_WORKER_FAILED → TASK_WORKER_FAILED`. No other states produce a signal — preemption is an infrastructure event, not a worker-health signal.
   - The controller consumer that calls `apply_task_updates` forwards those observations to `self._health.bump(...)` after the transaction commits. This keeps `transitions.py` free of mutable non-DB state.

User-level `TASK_STATE_FAILED` does not emit a signal — that's user code failing, not a bad worker.

**Reaper thread.** New background thread mirroring the prune loop at `controller.py:1371`. Owns both termination predicates:

```python
def _run_reaper_loop(self, stop_event: threading.Event) -> None:
    interval = _REAPER_INTERVAL_S  # 30s
    while not stop_event.is_set():
        stop_event.wait(timeout=interval)
        if stop_event.is_set():
            break
        try:
            self._reap_once()
        except Exception:
            logger.exception("Reaper round failed")

def _reap_once(self) -> None:
    doomed: dict[WorkerId, str] = {}  # worker_id -> reason

    for wid, score in self._health.workers_over_threshold():
        doomed[wid] = f"health_score={score:.1f}"

    threshold_ms = HEARTBEAT_STALENESS_THRESHOLD.to_ms()
    for w in healthy_active_workers_with_attributes(self._db):
        if w.last_heartbeat.age_ms() > threshold_ms and w.worker_id not in doomed:
            doomed[w.worker_id] = "heartbeat_stale"

    if not doomed:
        return
    logger.warning("Reaping workers: %s", doomed)
    self._transitions.fail_workers_batch(
        list(doomed), reason_map=doomed
    )
    for wid in doomed:
        self._health.forget(wid)
```

Both `fail_workers_batch` (`transitions.py:2771`) and `healthy_active_workers_with_attributes` already exist — we reuse them.

The existing `_reap_stale_workers` at `controller.py:2377` and its call from `_sync_all_execution_units` are deleted. Provider sync stops making termination decisions entirely; it only records RPC failures as bumps and lets the reaper decide.

**Cleanup.** When a worker is removed (by the reaper, by the age-based `_reap_stale_workers`, or by normal scale-down), call `self._health.forget(worker_id)`. The tracker holds no references itself, so a missed `forget` would just hold a dict entry for a dead worker indefinitely — harmless until restart but worth wiring up cleanly. Simplest: call `forget` from the controller-side worker-removal code path that runs after any DB-level removal.

**Removal of the inline 10-strike rule.** `_record_heartbeat_failure`, `HEARTBEAT_FAILURE_THRESHOLD`, and the `workers.consecutive_failures` column all go away. The only termination paths are:

1. The reaper (health score or heartbeat-age — same thread, same predicate structure, different inputs).
2. Normal scale-down.

Schema migration: drop `consecutive_failures` from the `workers` table in `schema.py:851` and from `WorkerRow` in `schema.py:1539`. No replacement column — the tracker is in-memory.

### Diagram

```
┌─────────────── controller threads ───────────────────────────┐
│                                                               │
│  Provider loop        Task-update loop       Reaper (NEW)     │
│   ping / poll           apply_task_updates     every 30s      │
│     │                     │ returns obs list     │            │
│     ▼                     ▼                      ▼            │
│  bump(RPC_FAILURE)    bump(TASK_WORKER_      ┌──────────────┐ │
│                             FAILED)          │ score > T    │ │
│                                              │ heartbeat    │ │
│                                              │   age > 15m  │ │
│                                              └──────┬───────┘ │
│                                                     ▼         │
│                                          fail_workers_batch() │
│                                          tracker.forget()     │
│                                                               │
│                WorkerHealthTracker (in-memory)                │
│                 dict[worker_id] → (score, updated_ms)         │
│                 score_now = s * exp(-λ·Δt)                    │
└───────────────────────────────────────────────────────────────┘
```

## Why In-Memory

- **A dying worker recurs fast.** If a TPU is genuinely bad, the next RPC or next dispatch will hit it within seconds-to-minutes. Losing the accumulated score on controller restart doesn't save a worker that's actually broken — it just delays the kill by one signal window.
- **Simpler schema, not more.** We drop `workers.consecutive_failures` entirely; we don't add anything to replace it.
- **No SQL-side decay.** Decay-on-read is trivial in Python and ugly to express as a SQL predicate across the reaper's candidate scan.
- **Testability.** The tracker is a pure Python class with an injectable clock — unit tests set the clock, call `bump`, assert `workers_over_threshold`. No DB fixture needed.
- **No hot-path DB writes.** Every heartbeat failure and every task update today already touches the DB; adding a score write would put the reaper in the write path of those transactions. The tracker keeps that out.

The one thing we lose — score persistence across restart — doesn't matter: a controller restart invalidates worker liveness regardless, and `_reap_stale_workers` seeds the decision from heartbeat age. If a worker is truly bad it'll hit the tracker again within one signal cycle.

## Tradeoffs and Defaults

- **All signals weight 1.0, threshold 10.** Preserves today's "10 heartbeat failures = kill" behavior exactly when the signal stream is pure RPC failures, and extends it: 10 preempted tasks, 10 worker-failed tasks, or any mix summing to 10 will trip the reaper. Simple, uniform, easy to reason about.
- **Half-life 5 min.** Today's continuous failure run kills a worker in ~50s. With weight 1.0 and half-life 5 min, 10 failures within ~50s still cross the threshold; failures spaced by more than a half-life decay faster than they accumulate.
- **Don't bump on `TASK_STATE_FAILED`.** That's user code.
- **Reaper cadence 30s.** Low enough that worst-case latency after threshold crossing is bounded; high enough to stay cheap. Per tick: `O(workers)` in memory plus one DB query for heartbeat ages, one DB write (`fail_workers_batch`) only if there's work.

## Test Plan

- Unit tests for `WorkerHealthTracker` with an injected clock:
  - Single bump, query score = weight.
  - Bump, wait one half-life, query score = weight / 2.
  - Two bumps over time, check weighted accumulation.
  - Threshold detection over a synthetic dict of workers.
  - `forget()` drops an entry.
- Controller-level test: simulate a worker that passes pings but returns `TASK_STATE_PREEMPTED` for every task; assert reaper removes it within N ticks.
- Controller-level test: simulate 4 consecutive RPC failures on a worker; assert reaper removes it.
- Controller-level test: one `TASK_STATE_WORKER_FAILED` on a worker; assert removal on next reaper tick.
- Existing tests for the old 10-strike heartbeat path need to be updated or deleted — that code is gone.

## Migration

Schema: drop `workers.consecutive_failures` (column def at `schema.py:851`, `WorkerRow` field at `schema.py:1539`). Add a schema version bump; existing rows just lose the column.

Code removal:

- `HEARTBEAT_FAILURE_THRESHOLD` constant at `transitions.py:89`
- `_record_heartbeat_failure` at `transitions.py:2166`
- The `force_remove` branch flow in `fail_heartbeats_batch` — replaced by plain failure recording that bumps the tracker.
- Any callers that read `consecutive_failures` (scheduler, status projection, tests).

Config: no new config surface. Three module-level constants (`HEALTH_SCORE_THRESHOLD`, `HEALTH_SCORE_HALF_LIFE_S`, `SIGNAL_WEIGHT`) live in `worker_health.py`. If tuning becomes necessary, promote to `ControllerConfig` then.

## Open Questions

- Do we want a debug RPC / status field exposing current health scores? Useful for operator diagnosis but adds surface. Probably yes, read-only, in the existing worker status proto.
- Should the reaper emit a structured event (`WorkerReapedReason`) so dashboards can distinguish "died from health score" vs "died from heartbeat-age timeout"? Low cost, I'd add it.
