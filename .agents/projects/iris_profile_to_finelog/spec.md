# Spec — iris CPU profiles → finelog

Concrete contracts for the design in `design.md`. Reviewers should be able to read this and answer "would I build this exact API?" without inferring anything from prose.

## 1. Finelog namespace: `iris.profile`

### 1.1 Row dataclass

Location: `lib/iris/src/iris/cluster/worker/stats.py` (extended — the file already houses `IrisWorkerStat` and `IrisTaskStat`).

```python
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import ClassVar


PROFILE_NAMESPACE = "iris.profile"


class ProfileType(StrEnum):
    CPU = "cpu"
    MEMORY = "memory"
    THREAD = "thread"


class ProfileFormat(StrEnum):
    # CPU
    RAW = "raw"
    FLAMEGRAPH = "flamegraph"
    SPEEDSCOPE = "speedscope"
    # Memory
    HTML = "html"
    TABLE = "table"
    STATS = "stats"


class ProfileTrigger(StrEnum):
    PERIODIC = "periodic"
    ON_DEMAND = "on_demand"


@dataclass
class IrisProfile:
    """One row per profile capture, regardless of type or trigger.

    Written by the worker process for task captures, by K8sTaskProvider
    (in the controller process) for k8s task captures, and by the
    controller for /system/controller self-captures. Read by the
    dashboard via finelog StatsService SQL. Retention is finelog
    segment-based, 7 days; see OPS.md.
    """

    key_column: ClassVar[str] = "captured_at"

    # identity — `source` is the proto request's target string, verbatim
    source: str             # "/job/.../task/N", "/system/worker/<id>",
                            # "/system/controller"
    attempt_id: int | None  # set for task sources; None for /system/*
    vm_id: str              # writer attribution: worker id or k8s pod/node name
    # capture metadata
    captured_at: datetime   # tz-naive UTC, segment key
    duration_seconds: int
    type: str               # ProfileType value
    format: str             # ProfileFormat value
    trigger: str            # ProfileTrigger value
    # type-specific (nullable; only the field for `type` is set)
    rate_hz: int | None = None        # CPU only — py-spy --rate
    native: bool | None = None        # CPU only — py-spy --native
    leaks: bool | None = None         # memory only — memray --leaks
    locals_dump: bool | None = None   # thread only — py-spy --locals
    # payload
    profile_data: bytes = b""

    def __post_init__(self) -> None:
        # cheap validators — schema bugs surface here, not at query time
        ProfileType(self.type)
        ProfileFormat(self.format)
        ProfileTrigger(self.trigger)
```

**Contracts:**

- `key_column = "captured_at"` aligns with finelog's segment ordering for time-range pruning.
- `source` is the verbatim string from `ProfileTaskRequest.target` after resolution. Three value families: `/job/.../task/N`, `/system/worker/<id>`, `/system/controller`. The dashboard's "Profile history" panel filters on this column.
- `attempt_id` is `None` for `/system/*` sources and an integer for task sources.
- `vm_id` is the **writer attribution** — the VM that captured the profile. Convention:
  - Task captured by a worker: `vm_id = <worker.id>`.
  - Task captured by k8s: `vm_id = f"k8s/{pod_node_name or pod_name}"`.
  - `/system/worker/<id>` captured by the named worker: `vm_id = <id>` (matches the source).
  - `/system/controller` captured by the controller: `vm_id = "controller-self"`.
- Enum-typed columns are stored as `str` (the StrEnum value), matching `IrisWorkerStat.status`.
- The four type-specific metadata fields are nullable. Exactly one is non-null per row, matching `type`. Periodic captures always have `type="cpu"`, `format="raw"`, `trigger="periodic"`.

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

- New constructor params on `Worker.__init__`: `profile_interval: Duration = Duration.from_seconds(600)`, `profile_duration_seconds: int = 10`. Defaults are constants (matching today's controller config). The loop runs captures sequentially — no concurrency knob.
- New `Worker._profile_table: Table[IrisProfile] | None` field, populated in `start()` alongside `_worker_stats_table` and `_task_stats_table` *only when `_log_client is not None`* (matches the existing pattern at [`worker.py:281-282`](https://github.com/marin-community/marin/blob/24ebc3b1/lib/iris/src/iris/cluster/worker/worker.py#L281)). Cleared in `_detach_log_handler` ([`worker.py:543`](https://github.com/marin-community/marin/blob/24ebc3b1/lib/iris/src/iris/cluster/worker/worker.py#L543)) alongside the other stats tables.
- New thread spawn in `start()` (after `_log_client` build): `self._profile_thread = self._threads.spawn(self._run_profile_loop, name="profile-loop")`. Skipped when `_log_client is None` (test mode, no controller_address).
- New private method `Worker._capture_and_log_profile(*, source: str, attempt_id: int | None, pid: int, request: job_pb2.ProfileTaskRequest, trigger: str) -> bytes`. Single writer used by the periodic loop and the `ProfileTask` RPC handler for **all** types. Calls `profile_local_process(duration, profile_type)`, builds an `IrisProfile` row from the request (deriving `type`, `format`, and the type-specific metadata from `request.profile_type`), and writes when `_profile_table is not None`; if `None`, returns the bytes without writing (no crash). The periodic loop calls it with `trigger="periodic"` and a fixed CPU `ProfileType`; the RPC handler passes through the request's `profile_type` and `trigger="on_demand"`.
- For task targets, the helper resolves `attempt: TaskAttempt` from `_tasks` first; raises `RuntimeError("attempt no longer running")` if `attempt.pid is None or attempt.state != RUNNING`. The loop catches and logs at debug.
- `stop()` sets `stop_event` and joins `_profile_thread` after the existing lifecycle thread join.

### 2.3 Stop semantics

`run_profile_loop` blocks on a synchronous `subprocess.run(timeout=duration_seconds + 30)` inside `_capture_and_log_profile`. `stop_event.set()` does not preempt subprocess. `Worker.stop()` may therefore block up to `profile_duration_seconds + 30` (default ≈40s) waiting for an in-flight capture to finish. Documented; not worth the complexity of a `Popen` + `poll(stop_event)` loop in v1.

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
    - target == "/system/process": profile the worker process itself.
      Captures via profile_local_process and writes one IrisProfile row
      with source set to "/system/worker/<this_worker_id>" so the
      dashboard can find it under the worker's URL. Bytes returned inline.
    - target == "/job/.../task/N[:attempt_id]": profile the task's container.
      Captures via _capture_and_log_profile, writes IrisProfile, returns bytes inline.
      All types (cpu/memory/thread) persist; trigger="on_demand".
    - All other targets: INVALID_ARGUMENT.

    Errors:
    - INVALID_ARGUMENT if profile_type is missing.
    - NOT_FOUND if the task target does not match a known attempt.
    - FAILED_PRECONDITION if the matched attempt has no pid (Pending->Running race).
    - Runtime py-spy/memray failures returned as `error` field, not gRPC errors.
    """
```

**Contracts:**

- All on-demand task captures (CPU, memory, thread) go through `Worker._capture_and_log_profile` with `trigger="on_demand"` and persist to `iris.profile`.
- `/system/process` (worker self) also persists, with `source = /system/worker/<id>` so the dashboard's per-worker history view finds it. `vm_id = <this worker's id>` (matches the source).
- The handler does not need to know about `type` — it forwards `request.profile_type` to the helper, which extracts the type from the proto oneof.

## 4. Controller changes

### 4.1 Code removed

| Symbol | File | Lines (at SHA `24ebc3b1`) | Reason |
|---|---|---|---|
| `_run_profile_loop` | `lib/iris/src/iris/cluster/controller/controller.py` | 1607-1626 | periodic loop |
| `_profile_all_running_tasks` | same | 1627-1651 | periodic loop |
| `_dispatch_profiles` | same | 1653-1670 | periodic loop |
| `_capture_one_profile` | same | 1672-1704 | periodic loop |
| `_profile_thread` spawn | same | 1351 | periodic loop |
| `profile_interval` config field | same | 1028 | controller no longer drives the loop |
| `profile_duration` config field | same | 1031 | controller no longer drives the loop |
| `profile_concurrency` config field | same | 1046 | controller no longer drives the loop |
| `profile_retention` config field | same | 1043 | prune sweep gone |
| `prune_old_data(profile_retention=…)` arg + call site | same | 1544 | prune sweep gone |
| `prune_old_data` `profile_retention` parameter | `lib/iris/src/iris/cluster/controller/transitions.py` | 2121, 2134 | prune sweep gone |
| `prune_stale_profiles` / `prune_orphan_profiles` invocations | same | 2174-2184 | prune sweep gone |
| `PruneResult.profiles_deleted` field | same | 2186-2197 | prune sweep gone |
| `prune_stale_profiles` (store helper) | `lib/iris/src/iris/cluster/controller/stores.py` | 1394-1409 | hardcodes `profiles.task_profiles` SQL |
| `prune_orphan_profiles` (store helper) | same | 1410-1425 | hardcodes `profiles.task_profiles` SQL |
| Checkpoint snapshot of `profiles.sqlite3` | `lib/iris/src/iris/cluster/controller/checkpoint.py` | 184-186, 233 | snapshot path gone |
| `profiles.sqlite3` restore + ATTACH branch | `lib/iris/src/iris/cluster/controller/db.py` | 740-758, 763-764 | restore path gone |
| `insert_task_profile` | same | 987 | DB persistence gone |
| `get_task_profiles` | same | 1000-1022 | DB persistence gone |
| `task_profiles_table` property | same | 628-629 | DB persistence gone |
| `PROFILES_DB_FILENAME` constant | same | 297 | survives until 0024; deleted in same commit |
| `_profiles_db_path` field assignment | same | 306 | survives until 0024; deleted in same commit |
| `ATTACH DATABASE … profiles` (startup) | same | 314 | survives until 0024; deleted in same commit |
| `ATTACH DATABASE … profiles` (read pool) | same | 394 | survives until 0024; deleted in same commit |
| `profiles_db_path` accessor property | same | 411-412 | kept as one-line method consumed only by 0024 |
| `TASK_PROFILES = Table(...)` | `lib/iris/src/iris/cluster/controller/schema.py` | 1031-1071 | DB persistence gone |

The controller's `profile_task` RPC handler **stays** — it is the dashboard-facing entry. `WorkerService.ProfileTask` ([`worker.proto:111`](https://github.com/marin-community/marin/blob/24ebc3b1/lib/iris/src/iris/rpc/worker.proto#L111)) and the `CpuProfile` / `MemoryProfile` / `ThreadsProfile` / `ProfileType` / `ProfileTaskRequest` / `ProfileTaskResponse` proto messages also stay. The provider abstraction's `profile_task` method on both worker-based providers and `K8sTaskProvider` stays.

### 4.2 Controller `profile_task` RPC handler (rewrite)

Edit `lib/iris/src/iris/cluster/controller/service.py:1893-1968`. The new handler delegates for task targets and self-captures for `/system/controller`:

```python
def profile_task(
    self,
    request: job_pb2.ProfileTaskRequest,
    ctx: RequestContext,
) -> job_pb2.ProfileTaskResponse:
    """Dashboard-facing on-demand profile dispatch.

    Behaviour by target:
      /job/.../task/N[:attempt_id]
        - Resolve task and worker; delegate to provider.profile_task.
          Worker-based: forwards to worker; worker writes IrisProfile
          (all types), returns bytes. K8s: K8sTaskProvider captures via
          kubectl exec, writes IrisProfile (all types), returns bytes.
      /system/worker/<id>
        - Forward as /system/process to the named worker via
          WorkerService.ProfileTask. Worker writes the row with
          source='/system/worker/<id>'.
      /system/controller
        - Capture this controller process via profile_local_process,
          write one IrisProfile row (source='/system/controller',
          vm_id='controller-self', attempt_id=None,
          trigger='on_demand'), return bytes inline.
      Anything else
        - INVALID_ARGUMENT.
    """
```

**Contracts:**

- For task targets and `/system/worker/<id>`, the controller does *not* hold a finelog Table — the writer is the worker (or k8s provider).
- For `/system/controller`, the controller has its own `Controller._profile_table: Table[IrisProfile] | None`, registered next to where `iris.task` / `iris.worker` are registered ([`controller.py:1186-1188`](https://github.com/marin-community/marin/blob/24ebc3b1/lib/iris/src/iris/cluster/controller/controller.py#L1186)) via `self._log_client.get_table(PROFILE_NAMESPACE, IrisProfile)`. Test mode (`_log_client is None`) → no-op write, bytes still returned.
- Today's controller-side `/system/process` target is **renamed** to `/system/controller`. The dashboard updates the literal in one place (`StatusTab.vue`); the worker-self path continues to use `/system/process` (semantically "the local process" relative to whichever endpoint receives it). The "profile this controller" button is preserved.
- All existing target-resolution and worker-liveness checks ([`service.py:1939-1968`](https://github.com/marin-community/marin/blob/24ebc3b1/lib/iris/src/iris/cluster/controller/service.py#L1939)) are preserved.

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
- Existing migrations 0005, 0014, 0020, 0023 are **not** modified — never edit landed migrations. They stay on disk as-is; 0024 runs after them on first upgrade from old snapshots and reverses their effect by `DETACH`-ing and `unlink`-ing the file. The path-resolver helper for `profiles_db_path` stays in `db.py` as a one-line method consumed only by 0024.
- This change ships as **one PR**. The reader deletions, prune-sweep deletion, checkpoint-branch deletion, and `0024_drop_profiles_db.py` all land together — splitting them risks intermediate states where the prune loop or checkpoint path references a table or file that no longer exists.
- The startup ATTACH (`db.py:314`), read-pool ATTACH (`db.py:394`), and `_profiles_db_path` field are deleted in the same PR, *after* 0024's migration body runs (which still needs them to `DETACH`).

### 4.4 K8s provider write path

`K8sTaskProvider` already has `log_client: LogWriterProtocol | None` injected ([`tasks.py:1083`](https://github.com/marin-community/marin/blob/24ebc3b1/lib/iris/src/iris/cluster/providers/k8s/tasks.py#L1083)), but `LogWriterProtocol` only exposes `write_batch`, not `get_table` ([`finelog/types.py:34-49`](https://github.com/marin-community/marin/blob/24ebc3b1/lib/finelog/src/finelog/types.py#L34)). Calling `self.log_client.get_table(PROFILE_NAMESPACE, IrisProfile)` would not type-check and would fail against the test fakes the protocol exists to support.

Mirror the existing `task_stats_table` pattern. The controller constructs the Table from its own `LogClient` and injects a typed field on the provider:

- **K8sTaskProvider field.** Add `profile_table: Table[IrisProfile] | None = None` next to `task_stats_table` at [`providers/k8s/tasks.py:1087`](https://github.com/marin-community/marin/blob/24ebc3b1/lib/iris/src/iris/cluster/providers/k8s/tasks.py#L1087).
- **Controller wiring.** In the controller's k8s-mode branch at [`controller.py:1186-1188`](https://github.com/marin-community/marin/blob/24ebc3b1/lib/iris/src/iris/cluster/controller/controller.py#L1186), next to the existing `self._provider.task_stats_table = k8s_log_client.get_table(TASK_STATS_NAMESPACE, IrisTaskStat)` line, add:
  ```python
  self._provider.profile_table = k8s_log_client.get_table(PROFILE_NAMESPACE, IrisProfile)
  ```
- **`K8sTaskProvider.profile_task` (edit at [`tasks.py:1155-1180`](https://github.com/marin-community/marin/blob/24ebc3b1/lib/iris/src/iris/cluster/providers/k8s/tasks.py#L1155)).** On every successful capture (CPU, memory, **and** thread), write one `IrisProfile` row via `self.profile_table.write(row)` with `type` derived from the request's `profile_type` oneof, `trigger="on_demand"`, and the type-specific metadata fields populated. No-op when `profile_table is None` (test mode).
- **`vm_id` value for k8s rows.** Use `vm_id = f"k8s/{pod_node_name or pod_name}"` — the pod's `spec.nodeName` (k8s scheduling-resolved host) when available, falling back to `pod_name`. Stable across captures of the same pod; distinguishes node moves; never collides with worker-based provider `vm_id`s (which never start with `k8s/`). The dashboard SQL surface in §5.1 selects `vm_id`, so the format is part of the contract.
- **Errors:** drop on failure — `ProfileTaskResponse(error=str(e))`, no finelog row written, no retry.

The k8s provider does not host a periodic loop in v1. Adding one is out of scope (see design Open Questions).

### 4.5 Dashboard

- `lib/iris/dashboard/src/composables/useProfileAction.ts` — **unchanged** other than the rename of the controller-self target string (`/system/process` → `/system/controller`). Continues to call the controller `profile_task` RPC.
- `lib/iris/dashboard/src/components/controller/StatusTab.vue` — keep the "profile this controller" button; only the target string changes. The same component grows a "Profile history" panel filtering on `source = '/system/controller'`.
- `lib/iris/dashboard/src/components/controller/TaskDetail.vue` — add a "Profile history" panel filtering on `source = '/job/.../task/N'` (the task's wire ID) using `useStatsRpc('Query', { sql: ... })`.
- `lib/iris/dashboard/src/components/controller/WorkerDetail.vue` (or equivalent) — add a "Profile history" panel filtering on `source = '/system/worker/<id>'`.

## 5. Dashboard SQL surface

### 5.1 List recent profiles for a source

Used by all three "Profile history" panels (task, worker-self, controller-self) — only the bound `source` parameter differs.

```sql
SELECT
  captured_at,
  type,
  attempt_id,
  vm_id,
  duration_seconds,
  format,
  trigger,
  length(profile_data) AS size_bytes
FROM "iris.profile"
WHERE source = ?
ORDER BY captured_at DESC
LIMIT 50
```

### 5.2 Fetch one profile's bytes

```sql
SELECT profile_data, type, format
FROM "iris.profile"
WHERE source = ? AND captured_at = ?
LIMIT 1
```

### 5.3 Optional: filter by type

```sql
SELECT captured_at, format, length(profile_data)
FROM "iris.profile"
WHERE source = ? AND type = ?
ORDER BY captured_at DESC
LIMIT 50
```

All queries go through the existing `useStatsRpc` composable, which posts to `proxy/system.log-server/finelog.stats.StatsService/Query` and decodes the Arrow IPC response.

## 6. Errors

No new error types. Drop on failure throughout:

- Worker `_capture_and_log_profile` raises `RuntimeError` from `profile_local_process` (py-spy/memray missing, non-zero exit) and from the pid/state precondition. The periodic loop catches and drops; the RPC handler returns `error=str(e)` in `ProfileTaskResponse`.
- K8s `_profile_cpu` / `_profile_memory` / `_profile_thread` raise on `kubectl exec` failure; `K8sTaskProvider.profile_task` catches and returns `error=str(e)`.
- Controller's `/system/controller` path raises `RuntimeError` on `profile_local_process` failure; the handler returns `error=str(e)`.
- Finelog write failures: trust finelog. The LogClient bg-flush thread handles its own retry/logging; a failed write drops that profile and the RPC still returns the bytes inline.
- Controller `profile_task` returns `INVALID_ARGUMENT` for unknown targets, including the legacy `/system/process` (renamed to `/system/controller`).

## 7. Out of scope

The following are **not** committed by this design and stay for follow-up PRs:

- A `purge profiles for source` dashboard action.
- Per-row compression of `profile_data` — trust finelog's segment-level compression.
- Per-cluster `profile_interval` push from controller. The interval is a constant for now.
- Periodic profiles on the k8s direct-provider path. K8s stays on-demand-only (matches today).
- Per-task circuit breaker for repeated ptrace failures.
- Latency-sensitive opt-out attribute.
- Any modification to existing migrations (0005/0014/0020/0023) — never edit landed migrations.

## 8. File summary

| Change | Path |
|---|---|
| New | `lib/iris/src/iris/cluster/worker/profile_loop.py` |
| Edit (extend) | `lib/iris/src/iris/cluster/worker/stats.py` (+ `IrisProfile`, `ProfileKind`, `ProfileFormat`, `ProfileTrigger`, `PROFILE_NAMESPACE`) |
| Edit | `lib/iris/src/iris/cluster/worker/worker.py` (spawn loop, register table, `_capture_and_log_profile` helper) |
| Edit | `lib/iris/src/iris/cluster/worker/service.py` (`ProfileTask` writes finelog for all kinds on task and `/system/process` targets) |
| Edit | `lib/iris/src/iris/cluster/providers/k8s/tasks.py` (`profile_task` writes finelog for all kinds; new `profile_table` field) |
| Delete (large) | `lib/iris/src/iris/cluster/controller/controller.py` (periodic profile loop + helpers + config) |
| Edit (rewrite) | `lib/iris/src/iris/cluster/controller/service.py` (`profile_task` dispatches for tasks, captures + writes finelog for `/system/controller`) |
| Edit | `lib/iris/src/iris/cluster/controller/controller.py` (register `_profile_table` next to existing stats tables) |
| Delete | `lib/iris/src/iris/cluster/controller/schema.py` (`TASK_PROFILES`) |
| Delete | `lib/iris/src/iris/cluster/controller/db.py` (profile helpers + ATTACH; keep `profiles_db_path` method for 0024) |
| New | `lib/iris/src/iris/cluster/controller/migrations/0024_drop_profiles_db.py` |
| Edit | `lib/iris/dashboard/src/components/controller/TaskDetail.vue` (Profile history panel filtering on `target`) |
| Edit | `lib/iris/dashboard/src/components/controller/StatusTab.vue` (rename target string to `/system/controller`; add Profile history panel) |
| Edit | `lib/iris/dashboard/src/components/controller/WorkerDetail.vue` (Profile history panel filtering on worker target) |
| Edit | `lib/iris/AGENTS.md`, `lib/iris/OPS.md` (document `iris.profile` retention + writer attribution conventions) |
