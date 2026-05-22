# Spec — task status text → finelog

Concrete contracts for the design in `design.md`. Reviewers should be able to read this and answer "would I build this exact API?" without inferring anything from prose.

## 1. Finelog namespace: `iris.zephyr_task_status`

### 1.1 Row dataclass

Location: `lib/iris/src/iris/cluster/worker/stats.py` (extended — already houses `IrisWorkerStat`, `IrisTaskStat`, `TASK_STATS_NAMESPACE`).

```python
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import ClassVar


ZEPHYR_TASK_STATUS_NAMESPACE = "iris.zephyr_task_status"


@dataclass
class ZephyrTaskStatusRow:
    """One row per status-text update pushed by a running task.

    Written by the worker process whenever task code calls
    ``RemoteClient.report_task_status_text``. Read by the dashboard via
    finelog ``StatsService.Query``. The dashboard surfaces the
    latest-by-``(ts, attempt_id)`` row per ``task_id``; finelog retains
    all prior rows and compacts them via the standard level-compactor.
    """

    key_column: ClassVar[str] = "ts"

    ts: datetime                    # tz-naive UTC, segment ordering key
    task_id: str                    # full task wire id, e.g. "/job/.../task/N"
    attempt_id: int                 # writer's current attempt; cross-attempt tiebreaker
    status_text_detail_md: str      # full markdown for the task detail page
    status_text_summary_md: str     # short markdown (~3 lines) for task lists
```

**Contracts:**

- `key_column = "ts"` aligns with finelog's segment ordering and matches the convention used by `iris.worker`, `iris.task`, `iris.profile`.
- `task_id` is the verbatim wire form of `JobName` (the same string used in `Controller.GetTaskStatus` and on the dashboard's task-detail URL). The row is keyed for read by `task_id`, not by attempt — status text is task-scoped, not attempt-scoped, but `attempt_id` is recorded for ordering when two attempts run briefly in parallel during preemption.
- Both markdown fields are required and stored as plain strings. The summary is informally capped at ~3 lines by the writer; no server-side validation.
- Writers pass `ts = datetime.now(UTC).replace(tzinfo=None)` (tz-naive UTC) — requires Python ≥3.11 for the `UTC` constant; this codebase is already ≥3.11. Concurrency: when two writes for the same `task_id` collide within a single millisecond, the dashboard's `ORDER BY ts DESC, attempt_id DESC LIMIT 1` deterministically picks the newer attempt; same-attempt sub-ms collisions fall back to finelog's row-write-order in the segment.

### 1.2 Retention

No application-side TTL. Finelog level-compactor manages segment count. If steady-state size becomes a problem after rollout, retention is added in a follow-up (see Open Questions in `design.md`); this design does not commit to a value.

## 2. Worker write path

### 2.1 `RemoteClient` table handle

Location: [`remote_client.py`](https://github.com/marin-community/marin/blob/b06b8cf5/lib/iris/src/iris/cluster/client/remote_client.py). `RemoteClient.__init__` grows two optional constructor parameters and an attribute:

```python
from finelog.client.log_client import LogClient, Table

class RemoteClient:
    def __init__(
        self,
        ...,
        log_client: LogClient | None = None,
        attempt_id: int | None = None,
    ):
        ...
        self._attempt_id: int | None = attempt_id
        self._zephyr_task_status_table: Table[ZephyrTaskStatusRow] | None = (
            log_client.get_table(
                ZEPHYR_TASK_STATUS_NAMESPACE, ZephyrTaskStatusRow
            )
            if log_client is not None
            else None
        )
```

Why on `RemoteClient` rather than `Worker`: `report_task_status_text` is called from `IrisClient.report_task_status_text` ([`client.py:782`](https://github.com/marin-community/marin/blob/b06b8cf5/lib/iris/src/iris/cluster/client/client.py#L782)), which is the public entry point used by task code (e.g. `lib/zephyr/src/zephyr/execution.py:240`). `IrisClient` is built from a `RemoteClient`; the table handle and attempt id must travel with it. The worker process passes both at `IrisClient` construction; the standalone iris CLI does not supply them (and does not call `report_task_status_text`).

### 2.2 `RemoteClient.report_task_status_text`

Location: [`remote_client.py:510`](https://github.com/marin-community/marin/blob/b06b8cf5/lib/iris/src/iris/cluster/client/remote_client.py#L510). Public signature unchanged; body rewritten.

```python
def report_task_status_text(
    self, task_id: JobName, detail_md: str, summary_md: str
) -> None:
    """Push markdown status text for ``task_id`` to finelog.

    Behaviour:

    - Last-write-wins per task from the dashboard's perspective (the
      dashboard reads ``ORDER BY ts DESC, attempt_id DESC LIMIT 1``).
      All prior versions are retained in ``iris.zephyr_task_status``
      until finelog compaction reclaims them.
    - **Never blocks.** ``Table.write`` buffers the row in a bounded
      in-memory queue and returns immediately; serialisation and
      network I/O run on finelog's background flush thread.
    - **Never raises on flush failure.** If the bg flush cannot reach
      the finelog server, the error is logged inside ``LogClient`` and
      the row is dropped silently. The contract is best-effort
      observability — status updates must not abort task execution.
      (This is a behaviour change from the prior controller-RPC
      implementation, which raised on failure; see design.md "Costs /
      Risks".)
    - When ``RemoteClient`` was constructed without a ``LogClient``
      (e.g. by the standalone iris CLI), the call is a no-op. Test
      and CLI codepaths therefore continue to work.
    """
    if self._zephyr_task_status_table is None:
        return
    self._zephyr_task_status_table.write([
        ZephyrTaskStatusRow(
            ts=datetime.now(UTC).replace(tzinfo=None),
            task_id=task_id.to_wire(),
            attempt_id=self._attempt_id or 0,
            status_text_detail_md=detail_md,
            status_text_summary_md=summary_md,
        )
    ])
```

## 3. Dashboard read path

### 3.1 Endpoint

`POST /finelog.stats.StatsService/Query` against the controller origin (same host:port as today's dashboard requests). The route is already mounted at [`dashboard.py:537`](https://github.com/marin-community/marin/blob/b06b8cf5/lib/iris/src/iris/cluster/controller/dashboard.py#L537) via `FinelogStatsServiceASGIApplication`, backed by `StatsServiceProxy(self._log_service_address)` ([`controller.py:1409`](https://github.com/marin-community/marin/blob/b06b8cf5/lib/iris/src/iris/cluster/controller/controller.py#L1409)). No new mount, no new auth path.

### 3.2 SQL

**Single-task (TaskDetail view)** — latest detail + summary for one `task_id`:

```sql
SELECT task_id, status_text_detail_md, status_text_summary_md
FROM "iris.zephyr_task_status"
WHERE task_id = $task_id
QUALIFY ROW_NUMBER() OVER (PARTITION BY task_id ORDER BY ts DESC, attempt_id DESC) = 1
```

**Batched (JobDetail view)** — latest summary for every non-terminal task in the job (potentially hundreds at once):

```sql
SELECT task_id, status_text_summary_md
FROM "iris.zephyr_task_status"
WHERE task_id IN ($task_ids)
QUALIFY ROW_NUMBER() OVER (PARTITION BY task_id ORDER BY ts DESC, attempt_id DESC) = 1
```

Both queries return at most one row per `task_id`. The dashboard treats missing rows as "no status text yet" (the current empty-string fallback in the Vue components).

### 3.3 Dashboard call sites

Three consumer files in `lib/iris/dashboard/src/components/controller/`, all of which today read `statusTextDetailMd` / `statusTextSummaryMd` from the `TaskStatus` proto:

| File:line | Consumer | Field | Query shape |
|---|---|---|---|
| `TaskDetail.vue:356,358` | one task's detail page | `statusTextDetailMd` | single-task |
| `JobDetail.vue:1129,1245` | job's task list (per non-terminal task) | `statusTextSummaryMd` | batched |

The proto field consumers (`lib/iris/dashboard/src/types/rpc.ts:101,102`) are deleted with the proto fields.

Each call site issues its `Query` when the view loads and **caches the result for the view's poll interval** (the existing cadence at which `GetTaskStatus` / `ListTasks` already refresh). Without this cache, an open `JobDetail` tab with a 2 s poll triggers one batched `Query` every 2 s through the controller per open tab — acceptable per tab, but the cache keeps it from being one Query per task per refresh. Reuse the existing finelog TS client (`finelog.stats.StatsService` is already generated for `ProfileHistory.vue:90` — a `useRpc` `Query` call exists in tree).

## 4. Removed surface

All deleted in the same PR. No back-compat shim.

### 4.1 Proto

- `ControllerService.SetTaskStatusText` RPC, [`controller.proto:639`](https://github.com/marin-community/marin/blob/b06b8cf5/lib/iris/src/iris/rpc/controller.proto#L639).
- `iris.job.SetTaskStatusTextRequest`, [`job.proto:635`](https://github.com/marin-community/marin/blob/b06b8cf5/lib/iris/src/iris/rpc/job.proto#L635).
- `iris.job.SetTaskStatusTextResponse`, [`job.proto:641`](https://github.com/marin-community/marin/blob/b06b8cf5/lib/iris/src/iris/rpc/job.proto#L641).
- Fields `status_text_detail_md` and `status_text_summary_md` on `iris.job.TaskStatus`, [`job.proto:21`](https://github.com/marin-community/marin/blob/b06b8cf5/lib/iris/src/iris/rpc/job.proto#L21). The field numbers (21 and 22) are reserved in-place via:

  ```protobuf
  message TaskStatus {
    // existing fields ...
    reserved 21, 22;
    reserved "status_text_detail_md", "status_text_summary_md";
  }
  ```

  Reserving both number and name blocks accidental reuse and prevents the previous wire numbers from being repurposed by a future field with different semantics.

### 4.2 Controller Python surface

- `ControllerServiceImpl.set_task_status_text`, [`service.py:2605`](https://github.com/marin-community/marin/blob/b06b8cf5/lib/iris/src/iris/cluster/controller/service.py#L2605).
- Population of `TaskStatus.status_text_detail_md` / `status_text_summary_md` in the `GetTaskStatus` handler, [`service.py:1511`](https://github.com/marin-community/marin/blob/b06b8cf5/lib/iris/src/iris/cluster/controller/service.py#L1511).
- `ControllerTransitions.record_task_status_text` and the two backing dicts (`_status_text_detail`, `_status_text_summary`).
- `ControllerTransitions.remove_status_text_by_job_ids` (only caller is the pruning loop that is now empty for status text).
- `RemoteClient._client.set_task_status_text` is no longer called from anywhere; the generated client method itself goes away with the proto.

### 4.3 Worker / client Python surface

No public method signatures removed. `RemoteClient.report_task_status_text` and `IrisClient.report_task_status_text` keep their signatures; only `RemoteClient.__init__` grows two optional parameters (`log_client`, `attempt_id`) and the body of `report_task_status_text` is rewritten.

### 4.4 Tests + benchmarks

- `lib/iris/tests/cluster/controller/test_service.py::test_set_task_status_text_persists_via_store` (and any sibling tests) — deleted with the handler.
- `lib/iris/scripts/benchmark_controller.py` — `benchmark_set_task_status_text` and any other `SetTaskStatusText`-referencing cases deleted, not ported. We do not benchmark the finelog write path under iris's benchmark harness; finelog has its own.
- Add a new worker-side test that asserts `RemoteClient.report_task_status_text` (a) buffers a row into a fake `StatsService`, (b) is a no-op when `_zephyr_task_status_table is None`, (c) does not raise on flush failure (mirrors the profile precedent's worker test in `iris_profile_to_finelog`).

## 5. Errors

- **Finelog unreachable from worker.** `Table.write` does **not** raise on transport failure — it enqueues into a bounded in-memory buffer and returns; the bg flush thread retries on its own cadence, logs persistent failures, and drops the row when the buffer fills. `report_task_status_text` does not see ConnectError. This is a **behaviour change** from today: today the controller-RPC implementation surfaces a Connect error synchronously into `IrisClient.report_task_status_text` and its caller (`lib/zephyr/src/zephyr/execution.py:240`), which does not wrap the call — so a controller outage during a status push currently crashes the task. After this change, the task continues; the operator loses one status update. This is the intended new contract.
- **`RemoteClient` constructed without `LogClient`.** `report_task_status_text` is a no-op. Standalone iris CLI usage (which never had a meaningful reason to push status text) is therefore not regressed. Tests that construct `RemoteClient` with the old signature continue to compile after the new params default to `None`.
- **Dashboard `Query` failure.** Identical handling to other dashboard stats panels that already issue `Query` calls (e.g. `ProfileHistory.vue:90`). The component renders the empty state.
- **Schema mismatch on worker start.** `LogClient.get_table` with a dataclass that doesn't match the registered schema raises eagerly at startup — same behaviour as `iris.worker` / `iris.task` today. Surfaces in tests and on first boot after a schema change.

No new error types.

## 6. File paths summary

| Piece | Location |
|---|---|
| Row dataclass + namespace constant | `lib/iris/src/iris/cluster/worker/stats.py` (extended) |
| `RemoteClient.__init__` two new params + table handle | `lib/iris/src/iris/cluster/client/remote_client.py` |
| `RemoteClient.report_task_status_text` rewrite | `lib/iris/src/iris/cluster/client/remote_client.py:510` |
| `IrisClient` wiring (passes `log_client`, `attempt_id` through) | `lib/iris/src/iris/cluster/client/client.py:782` (call-through), `__init__` (constructor wiring) |
| Worker / k8s provider call sites that construct `IrisClient` | wherever the worker today constructs its `IrisClient` (already has a `LogClient`); supply both new params |
| Proto deletions + reservations | `lib/iris/src/iris/rpc/controller.proto:639`, `lib/iris/src/iris/rpc/job.proto:21,635,641` |
| Controller handler deletion | `lib/iris/src/iris/cluster/controller/service.py:1511,2605` |
| Transitions deletion | `lib/iris/src/iris/cluster/controller/transitions.py` (status-text dicts + helpers) |
| Test deletions | `lib/iris/tests/cluster/controller/test_service.py::test_set_task_status_text_persists_via_store`; status-text cases in `lib/iris/scripts/benchmark_controller.py` |
| Dashboard queries | `lib/iris/dashboard/src/components/controller/TaskDetail.vue:356,358` (single-task); `lib/iris/dashboard/src/components/controller/JobDetail.vue:1129,1245` (batched) |
| Dashboard TS types deletion | `lib/iris/dashboard/src/types/rpc.ts:101,102` (`statusTextDetailMd`, `statusTextSummaryMd` on `TaskStatus`) |
| Dashboard route (already exists) | `lib/iris/src/iris/cluster/controller/dashboard.py:537` |

## 7. Out of scope

- No migration of the existing in-memory status text. Workers re-emit on next update; rollout drops the current text.
- No dashboard history UI. Even though finelog retains every version, the dashboard surface still shows the latest only.
- No coupling to #5574 (UpdateTaskStatus push removal). Independent rollout.
- No retention policy. Relies on finelog compaction. TTL is a follow-up.
- No rename of `report_task_status_text` despite the implementation change. Keeps call sites stable.
- No change to authentication on the dashboard ↔ finelog stats route — uses the existing controller-mount auth interceptor.
