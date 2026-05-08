# Spec — iris CPU profiles → finelog

Concrete contracts for the design in `design.md`. Reviewers should be able to read this and answer "would I build this exact API?" without inferring anything from prose.

## 1. Finelog namespace: `iris.cpu_profile`

### 1.1 Row dataclass

Location: `lib/iris/src/iris/cluster/worker/stats.py` (extended — the file already houses `IrisWorkerStat` and `IrisTaskStat`).

```python
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import ClassVar


CPU_PROFILE_NAMESPACE = "iris.cpu_profile"


class CpuProfileFormat(StrEnum):
    RAW = "raw"
    FLAMEGRAPH = "flamegraph"
    SPEEDSCOPE = "speedscope"


class CpuProfileTrigger(StrEnum):
    PERIODIC = "periodic"
    ON_DEMAND = "on_demand"


@dataclass
class IrisCpuProfile:
    """One row per CPU profile capture (periodic 10m loop or on-demand RPC).

    Written by the worker process. Read by the dashboard via finelog
    StatsService SQL. Retention is finelog segment-based, 7 days; see OPS.md.
    """

    key_column: ClassVar[str] = "captured_at"

    # identity
    task_id: str            # JobName.to_wire(); e.g. "/job/foo/task/3"
    attempt_id: int
    worker_id: str
    # capture metadata
    captured_at: datetime   # tz-naive UTC, segment key
    duration_seconds: int   # py-spy --duration; default 10
    rate_hz: int            # py-spy --rate; default 20
    native: bool            # py-spy --native flag at capture time
    format: str             # CpuProfileFormat value; periodic always RAW
    trigger: str            # CpuProfileTrigger value
    # payload
    profile_data: bytes     # raw py-spy output

    def __post_init__(self) -> None:
        # cheap validators — schema bugs surface here, not at query time
        CpuProfileFormat(self.format)
        CpuProfileTrigger(self.trigger)
```

**Contracts:**

- `key_column = "captured_at"` aligns with finelog's segment ordering for time-range pruning.
- `format` and `trigger` are stored as `str` (the StrEnum value), not enum types, so column types match the string-typed columns in `IrisWorkerStat.status`.
- No `profile_kind` column. This namespace is CPU-only; memory/threads do not write here.

### 1.2 Retention

Finelog segment-based retention, **7 days**. Configured via the standard finelog operator surface (per-namespace TTL in finelog catalog) — no application-side row-count cap. Documented in `lib/iris/OPS.md` alongside the existing `iris.worker` / `iris.task` retention notes.

## 2. Worker periodic loop

### 2.1 Module + entry point

Location: `lib/iris/src/iris/cluster/worker/profile_loop.py` (new).

```python
import logging
import threading
from collections.abc import Callable

from iris.cluster.worker.task_attempt import TaskAttempt
from iris.utils.duration import Duration
from iris.utils.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)


def run_profile_loop(
    *,
    stop_event: threading.Event,
    interval: Duration,
    list_running_attempts: Callable[[], list[TaskAttempt]],
    capture_one: Callable[[TaskAttempt, str], None],  # (attempt, trigger)
) -> None:
    """Periodically capture CPU profiles for every locally-running attempt.

    Pure function — takes injected callables for testability. No direct
    reference to `Worker` internals.

    Per-attempt errors are logged at exception level and do not propagate.
    Stops promptly between captures on `stop_event.set()`. An in-flight
    `capture_one` call may block stop for up to ~`profile_duration_seconds + 30`
    seconds (see §2.3 stop semantics).
    """
```

**Contracts:**

- Uses `RateLimiter(interval_seconds=interval.to_seconds())`. On stop, exits within one `stop_event.wait` slot.
- `list_running_attempts` returns a fresh snapshot each tick — the loop never holds a reference between ticks.
- `capture_one` raises on capture failure; the loop catches at `exception` level and continues.
- The loop does *not* write directly to finelog. `capture_one` owns the write — keeps the loop testable without a LogClient.

### 2.2 Worker integration

Edits in `lib/iris/src/iris/cluster/worker/worker.py`:

- New constructor params on `Worker.__init__`: `profile_interval: Duration = Duration.from_seconds(600)`, `profile_duration_seconds: int = 10`, `profile_concurrency: int = 1`. Defaults match today's controller config; tests can override. (The `profile_concurrency` knob is not used in v1 — the loop is sequential — but is reserved so a future bounded-pool change does not require a config-shape rev.)
- New `Worker._cpu_profile_table: Table[IrisCpuProfile] | None` field, populated in `start()` alongside `_worker_stats_table` and `_task_stats_table` *only when `_log_client is not None`* (matches the existing pattern at [`worker.py:281-282`](https://github.com/marin-community/marin/blob/24ebc3b1/lib/iris/src/iris/cluster/worker/worker.py#L281)). Cleared in `_detach_log_handler` ([`worker.py:543`](https://github.com/marin-community/marin/blob/24ebc3b1/lib/iris/src/iris/cluster/worker/worker.py#L543)) alongside the other stats tables.
- New thread spawn in `start()` (after `_log_client` build): `self._profile_thread = self._threads.spawn(self._run_profile_loop, name="profile-loop")`. Skipped when `_log_client is None` (test mode, no controller_address).
- New private method `Worker._capture_and_log_cpu_profile(attempt: TaskAttempt, *, trigger: str) -> bytes`. Single writer used by both the periodic loop and the `ProfileTask` RPC handler. Reads `attempt.task_id.to_wire()`, `attempt.attempt_id`, `attempt.pid` once at entry. Raises `RuntimeError("attempt no longer running")` if `attempt.pid is None or attempt.state != RUNNING` — the loop catches and logs at debug. Writes one `IrisCpuProfile` row when `_cpu_profile_table is not None`; if `None`, returns the bytes without writing (no crash).
- `stop()` sets `stop_event` and joins `_profile_thread` after the existing lifecycle thread join.

### 2.3 Stop semantics

`run_profile_loop` blocks on a synchronous `subprocess.run(timeout=duration_seconds + 30)` inside `_capture_and_log_cpu_profile`. `stop_event.set()` does not preempt subprocess. `Worker.stop()` may therefore block up to `profile_duration_seconds + 30` (default ≈40s) waiting for an in-flight capture to finish. Documented; not worth the complexity of a `Popen` + `poll(stop_event)` loop in v1.

## 3. Worker `ProfileTask` RPC behaviour change

`worker.proto:111` — proto signature unchanged.

Edits in `lib/iris/src/iris/cluster/worker/service.py` (the RPC handler):

```python
def profile_task(
    self,
    request: job_pb2.ProfileTaskRequest,
    ctx: RequestContext,
) -> job_pb2.ProfileTaskResponse:
    """Handle on-demand profile requests.

    Behaviour:
    - target == "/system/process": profile the worker process itself
      (uses profile_local_process). Result returned inline. Not persisted.
    - target == "/job/.../task/N[:attempt_id]": profile the task's container.
      For CPU: capture + write IrisCpuProfile row (trigger="on_demand")
      + return bytes inline.
      For memory/threads: capture + return bytes inline. Not persisted.
    - All other targets: INVALID_ARGUMENT.

    Errors:
    - INVALID_ARGUMENT if profile_type is missing.
    - NOT_FOUND if the task target does not match a known attempt.
    - FAILED_PRECONDITION if the matched attempt has no pid (Pending->Running race).
    - Runtime py-spy/memray failures returned as `error` field, not gRPC errors.
    """
```

**Contracts:**

- The CPU-on-task path goes through `Worker._capture_and_log_cpu_profile` (same writer as the loop). `trigger="on_demand"`.
- Memory and threads on task targets call `profile_local_process` against the task's pid and return inline. No finelog write.
- `/system/process` (worker self) does not write to finelog regardless of profile kind. The intent is to keep `iris.cpu_profile` semantically "task profiles" — worker-process diagnostics are ephemeral.

## 4. Controller deletions and slim k8s-only RPC

### 4.1 Code removed (worker-based providers)

| Symbol | File | Lines (at SHA `24ebc3b1`) |
|---|---|---|
| `_run_profile_loop` | `lib/iris/src/iris/cluster/controller/controller.py` | 1607-1626 |
| `_profile_all_running_tasks` | same | 1627-1651 |
| `_dispatch_profiles` | same | 1653-1670 |
| `_capture_one_profile` | same | 1672-1704 |
| `_profile_thread` spawn | same | 1351 |
| `profile_interval` config field | same | 1028 |
| `profile_duration` config field | same | 1031 |
| `profile_concurrency` config field | same | 1046 |
| `insert_task_profile` | `lib/iris/src/iris/cluster/controller/db.py` | 987 |
| `get_task_profiles` | same | 1000-1022 |
| `task_profiles_table` property | same | 628-629 |
| `PROFILES_DB_FILENAME`, `_profiles_db_path`, `ATTACH … profiles` | same | 297, 306, 314, 394 |
| `TASK_PROFILES = Table(...)` | `lib/iris/src/iris/cluster/controller/schema.py` | 1031-1071 |
| `Provider.profile_task` interface (non-k8s providers) | `lib/iris/src/iris/cluster/providers/*` | grep `profile_task` |

`WorkerService.ProfileTask` ([`worker.proto:111`](https://github.com/marin-community/marin/blob/24ebc3b1/lib/iris/src/iris/rpc/worker.proto#L111)) and the `CpuProfile`/`MemoryProfile`/`ThreadsProfile`/`ProfileType`/`ProfileTaskRequest`/`ProfileTaskResponse` messages in `job.proto` are kept — workers still use them.

### 4.2 Slim controller RPC retained for k8s mode

Edit `lib/iris/src/iris/cluster/controller/service.py:1893-1968`. Replace the existing handler with:

```python
def profile_task(
    self,
    request: job_pb2.ProfileTaskRequest,
    ctx: RequestContext,
) -> job_pb2.ProfileTaskResponse:
    """K8s-only on-demand profile dispatch.

    On clusters using K8sTaskProvider there is no worker process to host
    the profile RPC; the controller dispatches directly to the provider.
    On worker-based providers, this RPC raises UNIMPLEMENTED — the dashboard
    routes those requests at the worker proxy.

    Does not persist results — k8s mode has no periodic profile loop and
    no finelog writes for profiles.
    """
    if not self._controller.has_direct_provider:
        raise ConnectError(Code.UNIMPLEMENTED, "Use the worker proxy for profile requests")
    target = parse_profile_target(request.target)
    target.task_id.require_task()
    task = _read_task_with_attempts(self._db, target.task_id)
    if not task:
        raise ConnectError(Code.NOT_FOUND, f"Task {request.target} not found")
    attempt_id = target.attempt_id if target.attempt_id is not None else task.current_attempt_id
    return self._controller.provider.profile_task(task.task_id.to_wire(), attempt_id, request)
```

The `/system/process` and `/system/worker/<id>` target branches in the existing handler are **removed**. The controller no longer self-profiles or proxies worker self-profiles; dashboard worker-self-profile goes through the worker proxy directly.

### 4.3 SQLite migration

- Add new migration `lib/iris/src/iris/cluster/controller/migrations/0024_drop_profiles_db.py`:
  ```python
  def upgrade(conn: sqlite3.Connection, ctx: MigrationContext) -> None:
      # 1) Detach the schema if attached. SQLite raises if any cursor still
      # references it — the upgrade must run after all task_profiles_table
      # readers have been deleted in the same release.
      try:
          conn.execute("DETACH DATABASE profiles")
      except sqlite3.OperationalError:
          pass
      # 2) Remove the file. Idempotent — tolerates re-runs and missing files.
      try:
          ctx.profiles_db_path.unlink()
      except FileNotFoundError:
          pass
  ```
- Migrations 0005, 0014, 0020, 0023 stay on disk as no-op chain links once 0024 has run on a given DB. Their `upgrade()` bodies still execute on first-run upgrades from old snapshots; 0024 immediately reverses them. The path-resolver helper for `profiles_db_path` stays in `db.py` as a one-line method consumed only by 0024.
- **Commit ordering inside the migration step:** delete *all* call sites of `task_profiles_table` / `insert_task_profile` / `get_task_profiles` in the same commit as adding 0024. Otherwise a stale read after `DETACH` would 500.

### 4.4 Dashboard

- `lib/iris/dashboard/src/components/controller/StatusTab.vue` — remove the "profile this controller" button (`useProfileAction(controllerRpcCall, '/system/process')`).
- `lib/iris/dashboard/src/composables/useProfileAction.ts` — keep only the worker-RPC variant. Update call sites in `TaskDetail.vue`, `WorkerStatusPage.vue`, `WorkerTaskDetail.vue` to route through the worker proxy: `useRpc('proxy/worker/<worker_id>/iris.cluster.WorkerService', 'ProfileTask', body)`.
- `lib/iris/dashboard/src/components/controller/TaskDetail.vue` — add a "Profile history" panel that uses `useStatsRpc('Query', { sql: ... })` against `iris.cpu_profile`.

## 5. Dashboard SQL surface

### 5.1 List recent profiles for a task

```sql
SELECT
  captured_at,
  attempt_id,
  worker_id,
  duration_seconds,
  format,
  trigger,
  length(profile_data) AS size_bytes
FROM "iris.cpu_profile"
WHERE task_id = ?
ORDER BY captured_at DESC
LIMIT 50
```

### 5.2 Fetch one profile's bytes

```sql
SELECT profile_data, format
FROM "iris.cpu_profile"
WHERE task_id = ? AND captured_at = ?
LIMIT 1
```

Both queries go through the existing `useStatsRpc` composable, which posts to `proxy/system.log-server/finelog.stats.StatsService/Query` and decodes the Arrow IPC response.

## 6. Errors

No new error types. Existing behaviour:

- Worker `_capture_and_log_cpu_profile` raises `RuntimeError` from `profile_local_process` (py-spy missing, py-spy non-zero exit) and from the pid/state precondition. The periodic loop catches; the RPC handler returns `error=str(e)` in `ProfileTaskResponse`.
- Finelog write failures (`Table.write` errors) are logged by the LogClient bg-flush thread and surface in metrics — no change from today's `IrisTaskStat` write semantics.
- Controller k8s-only `profile_task` raises `UNIMPLEMENTED` on worker-based providers (clear signal to clients to use the worker proxy).

## 7. Out of scope

The following are **not** committed by this design and stay for follow-up PRs:

- Persisted memory/threads profiles (parallel `iris.memory_profile` / `iris.threads_profile` namespaces).
- A `purge profiles for task` dashboard action.
- Per-row gzip compression of `profile_data`. Decision deferred to dev-cluster measurement (see design Open Questions).
- Per-cluster `profile_interval` knob pushed by the controller via `Ping`. Today the value is per-worker config.
- Periodic profiles on the k8s direct-provider path. K8s stays on-demand-only (matches today).
- Per-task circuit breaker for repeated ptrace failures.
- Latency-sensitive opt-out attribute.
- Removal of the legacy migrations 0005/0014/0020/0023.

## 8. File summary

| Change | Path |
|---|---|
| New | `lib/iris/src/iris/cluster/worker/profile_loop.py` |
| Edit (extend) | `lib/iris/src/iris/cluster/worker/stats.py` (+ `IrisCpuProfile`, `CpuProfileFormat`, `CpuProfileTrigger`, `CPU_PROFILE_NAMESPACE`) |
| Edit | `lib/iris/src/iris/cluster/worker/worker.py` (spawn loop, register table, capture+log helper) |
| Edit | `lib/iris/src/iris/cluster/worker/service.py` (RPC writes finelog on CPU-task target) |
| Delete (large) | `lib/iris/src/iris/cluster/controller/controller.py` (profile loop + helpers + config) |
| Edit (slim) | `lib/iris/src/iris/cluster/controller/service.py` (`profile_task` becomes k8s-only dispatcher) |
| Delete | `lib/iris/src/iris/cluster/controller/schema.py` (`TASK_PROFILES`) |
| Delete | `lib/iris/src/iris/cluster/controller/db.py` (profile helpers + ATTACH; keep `profiles_db_path` method for 0024) |
| Delete | non-k8s provider `profile_task` interface implementations |
| New | `lib/iris/src/iris/cluster/controller/migrations/0024_drop_profiles_db.py` |
| Edit | `lib/iris/dashboard/src/components/controller/TaskDetail.vue` (history panel + worker-proxy route) |
| Edit | `lib/iris/dashboard/src/composables/useProfileAction.ts` (worker route only) |
| Edit | `lib/iris/dashboard/src/components/controller/StatusTab.vue` (remove profile-controller button) |
| Edit | `lib/iris/AGENTS.md`, `lib/iris/OPS.md` (document `iris.cpu_profile` retention + k8s exception) |
