# Design: Periodic Coordinator-Side Profiling

**Issue:** [#3576](https://github.com/marin-community/marin/issues/3576)
**Status:** Draft (v2 — addresses Round 0 review)

## Problem

The cluster coordinator has no systematic way to periodically capture runtime profiles from workers. Debugging slow tasks or memory leaks requires manual intervention. We need periodic, automatic profiling that follows the same loop pattern as heartbeats.

## Approach

Add a fourth background loop to the controller — `_run_profiling_loop()` — that mirrors the heartbeat loop's three-phase pattern (snapshot → parallel RPCs → store results). Every 10 minutes it collects CPU thread dumps from all healthy active workers, batched 32 at a time. Profiles are stored in a new SQLite table with trigger-based ring buffer retention (last 10 per target).

### Why thread dumps (not flamegraphs or memray)?

- **Thread dumps are cheap**: `py-spy dump` completes in <1s vs 10s+ for `py-spy record`. At 200 workers, a full round takes ~7 batches × ~1s ≈ 7s. A `record`-based round would take ~70s, overlapping the next profiling interval.
- **Thread dumps are diagnostic enough**: they show what every thread is doing *right now*, which is exactly what periodic sampling needs. Flamegraphs are better for targeted deep-dives, which the existing on-demand `ProfileTask` RPC already supports.
- **Memory profiling is expensive and disruptive**: `memray` attaches via `ptrace`, pausing the target. Not suitable for automatic periodic capture on production workers.

The on-demand `ProfileTask` RPC remains available for flamegraphs, memory profiles, and other heavyweight profiling.

## Detailed Design

### 1. New Profiling Loop

Add `_run_profiling_loop` to `Controller`, spawned as a `ManagedThread` alongside the existing three loops:

```python
# In Controller.start():
if self._config.profiling_enabled:
    self._profiling_thread = self._threads.spawn(
        self._run_profiling_loop, name="profiling-loop"
    )
```

```python
def _run_profiling_loop(self, stop_event: threading.Event) -> None:
    limiter = RateLimiter(interval_seconds=self._config.profiling_interval.to_seconds())
    while not stop_event.is_set():
        stop_event.wait(timeout=limiter.time_until_next())
        limiter.mark_run()
        if stop_event.is_set():
            break
        if self._checkpoint_in_progress:
            continue
        try:
            self._profile_all_workers()
        except Exception:
            logger.exception("Profiling round failed, will retry next interval")
```

### 2. Three-Phase Worker Profiling

`_profile_all_workers()` follows the heartbeat's proven pattern, using `healthy_active_workers_with_attributes()` (the function that actually exists in `db.py`):

```python
def _profile_all_workers(self) -> None:
    # Phase 1: snapshot healthy workers
    workers = healthy_active_workers_with_attributes(self._db)
    if not workers:
        return

    # Phase 2: parallel RPCs, batched through dispatch executor
    work_queue: queue.Queue[Worker] = queue.Queue()
    result_queue: queue.Queue[tuple[WorkerId, bytes | None, str | None]] = queue.Queue()
    for w in workers:
        work_queue.put(w)

    batch_size = min(self._config.max_profiling_parallelism, len(workers))

    request = cluster_pb2.ProfileTaskRequest(
        target="/system/process",
        profile_type=cluster_pb2.ProfileType(
            threads=cluster_pb2.ThreadDumpSpec()
        ),
    )

    def _profile_one() -> None:
        while True:
            try:
                worker = work_queue.get_nowait()
            except queue.Empty:
                return
            try:
                stub = self.stub_factory.get_stub(worker.address)
                resp = stub.profile_task(request, timeout_ms=30_000)
                if resp.error:
                    result_queue.put((worker.worker_id, None, resp.error))
                else:
                    result_queue.put((worker.worker_id, resp.profile_data, None))
            except Exception as e:
                result_queue.put((worker.worker_id, None, str(e)))

    futures = [self._dispatch_executor.submit(_profile_one) for _ in range(batch_size)]

    # Phase 3: store results
    now_ms = Timestamp.now().epoch_ms()
    for _ in workers:
        worker_id, data, error = result_queue.get()
        if data is not None:
            store_profile(self._db, str(worker_id), "threads", data, now_ms)
        elif error:
            logger.debug("Profile failed for %s: %s", worker_id, error)

    for f in futures:
        f.result()
```

**Key difference from v1**: Uses `self.stub_factory` (the existing one) with an explicit `timeout_ms=30_000` on the RPC call, rather than creating a separate `StubFactory`. The `profile_task()` RPC already accepts `timeout_ms` as a keyword argument. This eliminates the need for a second stub factory and its lifecycle management.

### 3. Database Schema

New migration: `0004_profiles.sql`

```sql
CREATE TABLE IF NOT EXISTS profiles (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    target_id       TEXT    NOT NULL,
    profile_type    TEXT    NOT NULL,
    data            BLOB    NOT NULL,
    captured_at_ms  INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_profiles_target_type_time
    ON profiles(target_id, profile_type, captured_at_ms DESC);

-- Ring buffer: keep last 10 per (target_id, profile_type)
CREATE TRIGGER IF NOT EXISTS trg_profiles_retention
AFTER INSERT ON profiles
BEGIN
    DELETE FROM profiles
    WHERE target_id = NEW.target_id
      AND profile_type = NEW.profile_type
      AND id NOT IN (
        SELECT id FROM profiles
        WHERE target_id = NEW.target_id
          AND profile_type = NEW.profile_type
        ORDER BY id DESC
        LIMIT 10
      );
END;
```

**Schema decisions:**

- **`target_id`** is the worker ID (or task ID for future task-level profiling). Not a foreign key — profiles survive worker deregistration for post-mortem analysis.
- **`profile_type`** is a text enum (`"threads"`, `"cpu"`, `"memory"`) for readability in raw SQL queries.
- **Ring buffer** via `AFTER INSERT` trigger, matching the `txn_log` pattern. Scoped per `(target_id, profile_type)` so each worker independently keeps its last 10.
- **`data`** is BLOB. Thread dumps are text but stored as bytes to match `ProfileTaskResponse.profile_data` wire type.
- **No `duration_ms` column** (removed from v1): thread dumps are instantaneous; duration is only meaningful for flamegraph captures, which this table doesn't store by default.

### 4. Database Access Layer

```python
# In db.py

@db_row_model
class Profile:
    id: int = db_field("id", _decode_int)
    target_id: str = db_field("target_id", _decode_str)
    profile_type: str = db_field("profile_type", _decode_str)
    data: bytes = db_field("data", _decode_bytes)
    captured_at: Timestamp = db_field("captured_at_ms", _decode_timestamp_ms)

PROFILES = Table("profiles", Profile)


def store_profile(
    db: ControllerDB,
    target_id: str,
    profile_type: str,
    data: bytes,
    captured_at_ms: int,
) -> None:
    with db.transaction() as cur:
        cur.execute(
            "INSERT INTO profiles (target_id, profile_type, data, captured_at_ms) "
            "VALUES (?, ?, ?, ?)",
            (target_id, profile_type, data, captured_at_ms),
        )


def recent_profiles(
    db: ControllerDB,
    target_id: str,
    profile_type: str,
    limit: int = 10,
) -> list[Profile]:
    with db.snapshot() as q:
        return q.select(
            PROFILES,
            where=(PROFILES.c.target_id == target_id)
                  & (PROFILES.c.profile_type == profile_type),
            order=Order(PROFILES.c.captured_at, desc=True),
            limit=limit,
        )
```

### 5. Configuration

Add to `ControllerConfig`:

```python
profiling_interval: Duration = field(default_factory=lambda: Duration.from_seconds(600.0))
"""How often to run periodic profiling (default: 10 minutes)."""

max_profiling_parallelism: int = 32
"""Maximum concurrent profile RPCs per round."""

profiling_enabled: bool = True
"""Whether to enable the periodic profiling loop."""
```

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Profile type | Thread dumps (`py-spy dump`) | Fast (<1s), non-disruptive, sufficient for periodic monitoring |
| Worker query | `healthy_active_workers_with_attributes()` | The actual function in db.py; returns healthy+active workers with attributes populated |
| Parallelism | Reuse `_dispatch_executor` (ThreadPoolExecutor) | Same pattern as heartbeat; avoids creating a second executor |
| RPC timeout | Per-call `timeout_ms=30_000` | Avoids a separate stub factory; `profile_task()` already accepts `timeout_ms` |
| Storage | SQLite table with trigger-based ring buffer | Matches `txn_log` pattern; no external deps |
| Retention | 10 per (target, type) via trigger | Bounded storage; configurable via future migration |
| Target scope | Workers only (not individual tasks) | Workers are stable targets; task-level profiling available on-demand |
| No heartbeat lock | Profiling is read-only — no state transitions | Avoids blocking scheduling/heartbeat |
| Error handling | Log and skip unreachable workers | Best-effort; profiling failures never affect worker health |
| Checkpoint guard | Skip round if `_checkpoint_in_progress` | Same pattern as heartbeat |
| FK-free schema | `target_id` has no FK to `workers` | Profiles survive worker deregistration |

## Trade-offs

1. **Thread dumps vs flamegraphs**: Thread dumps are less detailed but capture the full cluster in seconds rather than minutes. The on-demand `ProfileTask` RPC covers the deep-dive case.

2. **SQLite storage vs cloud storage**: Profiles live in the controller's SQLite DB, not cloud storage. This keeps them queryable and co-located with other controller state, but they're lost if the DB is lost. The worker-side `ProfileCapture` already uploads to cloud storage for long-term retention; this feature is for operational visibility.

3. **Fixed retention (10) vs configurable**: The trigger hardcodes `LIMIT 10`. Changing retention requires a migration. Acceptable because 10 profiles × N workers is bounded, and the `txn_log` trigger has proven stable.

4. **Shared executor vs dedicated**: Reusing `_dispatch_executor` means profiling competes with heartbeat dispatch for threads. In practice, the two loops run at very different intervals (5s vs 600s) and are unlikely to overlap. Even if they do, the executor has enough capacity.

5. **`profiling_enabled` flag**: Adds a boolean config, which normally violates the "no boolean flags" guideline. Justified here because profiling has real cost (py-spy subprocess per worker) and operators need a kill switch. The alternative — setting `profiling_interval` to infinity — is less clear.

## Affected Files

| File | Change |
|------|--------|
| `lib/iris/src/iris/cluster/controller/migrations/0004_profiles.sql` | **New**: table, index, trigger |
| `lib/iris/src/iris/cluster/controller/db.py` | Add `Profile` model, `PROFILES` table, `store_profile()`, `recent_profiles()` |
| `lib/iris/src/iris/cluster/controller/controller.py` | Add `_run_profiling_loop()`, `_profile_all_workers()`, spawn thread in `start()` |
| `lib/iris/src/iris/cluster/controller/controller.py` (`ControllerConfig`) | Add `profiling_interval`, `max_profiling_parallelism`, `profiling_enabled` |

**No changes to:**
- `cluster.proto` — existing `ProfileTask` RPC suffices
- `profile_capture.py` — worker-side code, independent concern
- `runtime/profile.py` — utility code, no changes needed
- `service.py` — querying stored profiles can be a follow-up

## Test Strategy

### Unit Tests (`tests/iris/cluster/controller/test_profiles_db.py`)

1. **Ring buffer trigger**: Insert 15 profiles for the same `(target_id, profile_type)`, verify only 10 remain. Insert for a different target, verify independent retention.

2. **`store_profile()` / `recent_profiles()`**: Round-trip — store profiles, query back, verify ordering and content.

3. **Migration idempotency**: Apply `0004_profiles.sql` on a fresh DB and on a DB with existing migrations. Verify table and trigger exist.

### Integration Tests (`tests/iris/cluster/controller/test_controller.py`)

4. **Profiling loop end-to-end**: Stand up a `Controller` with mock workers (using `local.py` test harness). Run one profiling cycle. Verify profiles appear in the DB for each healthy worker.

5. **Unreachable worker**: Register a worker, shut it down, trigger profiling. Verify the loop completes without error and no profile is stored for that worker. Verify other workers in the same batch still get profiled.

6. **Checkpoint interaction**: Set `_checkpoint_in_progress`, trigger profiling loop, verify it skips the round.

### Test Fixture Strategy

Existing tests that construct `Controller` should be unaffected: set `profiling_enabled=False` in the shared test fixture/config. Only profiling-specific tests enable it. This avoids requiring mock stubs in every existing test.

### What NOT to test

- `py-spy` invocation — covered by existing `profile_capture.py` and `runtime/profile.py` tests
- `ProfileTask` RPC routing — already has coverage; this design reuses it unchanged
