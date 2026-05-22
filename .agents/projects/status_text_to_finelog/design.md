# Task status text → finelog

_Why are we doing this? What's the benefit?_

`ControllerService.SetTaskStatusText` is **73 % of all inbound RPCs** on the iris controller — 7566 of 10295 calls in a 60 s tcpdump (~5.8 MB of ~8.5 MB), ≈ 125 calls/sec. The handler is `@on_loop` and cheap (a dict write), but every call still spends the loop's GIL time on asgi + middleware + auth + timing-interceptor + stats `_record`. Skipping just the timing interceptor for this one RPC would shave some of that cost but leaves the loop dispatching every call. We move the data instead, because status text is pure display data — two markdown strings per task — that does not feed scheduling, heartbeats, or any control decision.

This is the next step in the same lift that `iris_stats_migration.md` and `iris_profile_to_finelog/design.md` already executed: registry + decisions stay in the controller DB; measurements and display state live in finelog namespaces. See [`research.md`](./research.md) for the full file:line inventory and Q&A.

## Challenges

_What's hard?_

The mechanical move is small — the current handler is an in-memory dict write ([`service.py:2605`](https://github.com/marin-community/marin/blob/b06b8cf5/lib/iris/src/iris/cluster/controller/service.py#L2605)), the worker side is a single client call ([`remote_client.py:510`](https://github.com/marin-community/marin/blob/b06b8cf5/lib/iris/src/iris/cluster/client/remote_client.py#L510)), and finelog already exposes generic `WriteRows` and `Query`. The interesting decisions are framing:

- Status text is **last-write-wins per task**, but finelog is append-only. The "current value" becomes a `SELECT … ORDER BY ts DESC, attempt_id DESC LIMIT 1` against a namespace that retains every prior version. Same `task_id` can be written by two attempts during preemption — spec pins an `attempt_id` tiebreaker column. We rely on finelog's level-compactor; no application-level dedup.
- Dashboard reads status text in two places: `TaskDetail.vue:356` (one task, detail) and `JobDetail.vue:1129,1245` (every non-terminal task in the job, summary). The latter forces a **batched** `WHERE task_id IN (...)` query — one Query per view, not one per task — cached for the view's poll interval. Spec pins the SQL.
- **Error semantics change.** `Table.write` is buffered and fire-and-forget; flush failures are logged in the bg thread and the row is dropped. Today's RPC implementation raises on failure, and the call site (`lib/zephyr/src/zephyr/execution.py:240`) doesn't wrap it — so a controller outage during a status push currently crashes the task. After the move, the task continues; the operator loses one status update. We treat this as a fix, but it's a real behaviour change worth flagging.

## Costs / Risks

- **Storage growth** in `iris.zephyr_task_status` (worst case in Open Questions). Relies on finelog level-compactor; no application-side TTL.
- **One extra round-trip per task-detail view.** Trivial latency cost; batched query keeps it bounded for the JobDetail list view.
- **In-flight status text is lost on rollout.** Today's dict is transient anyway — workers re-emit on the next update.
- **Naming.** `iris.zephyr_task_status` ties an iris-side namespace to a Zephyr concept (see Open Questions).

## Design

_How are we doing this?_

Same shape as `iris_profile_to_finelog`. Three flows: worker write, dashboard read, deletion.

**Worker write.** Drop `SetTaskStatusText` from `ControllerService`. `RemoteClient.report_task_status_text` (kept as the public worker API — call sites in `IrisClient` and `lib/zephyr/src/zephyr/execution.py:240` are untouched) is rewritten to call finelog's `Table.write` against a typed table handle injected at `RemoteClient` construction:

```python
from datetime import UTC, datetime

# at RemoteClient construction (when a LogClient is available)
self._zephyr_task_status_table = log_client.get_table(
    ZEPHYR_TASK_STATUS_NAMESPACE, ZephyrTaskStatusRow
)

# in report_task_status_text
if self._zephyr_task_status_table is None:
    return  # no LogClient — best-effort no-op, mirrors profile precedent
self._zephyr_task_status_table.write([
    ZephyrTaskStatusRow(
        ts=datetime.now(UTC).replace(tzinfo=None),
        task_id=task_id.to_wire(),
        attempt_id=self._attempt_id,
        status_text_detail_md=detail_md,
        status_text_summary_md=summary_md,
    )
])
```

Namespace registration on first use, matching the `iris.worker` / `iris.task` / `iris.profile` pattern. Schema in `spec.md`.

**Dashboard read.** The dashboard already issues authenticated requests to the controller origin. The controller already mounts `FinelogStatsServiceASGIApplication` at `/finelog.stats.StatsService/*` ([`dashboard.py:537`](https://github.com/marin-community/marin/blob/b06b8cf5/lib/iris/src/iris/cluster/controller/dashboard.py#L537)) — a thin ASGI sub-app whose `service` is a `StatsServiceProxy` forwarding to the finelog process. The task-detail view replaces its `task.status_text_detail_md` / `task.status_text_summary_md` access with a fresh `Query` RPC:

```sql
SELECT status_text_detail_md, status_text_summary_md
FROM "iris.zephyr_task_status"
WHERE task_id = ?
ORDER BY ts DESC
LIMIT 1
```

The mounted sub-app's only iris-side cost is the auth interceptor — no timing interceptor, no `_record`, no `to_thread`, no SQLite. The two `status_text_*` fields on `TaskStatus` are removed; nothing else on `GetTaskStatus` changes.

**Deletion of the controller path.** Removed in one PR with no compatibility shim (per AGENTS.md "NO BACKWARD COMPATIBILITY"):

- `ControllerService.SetTaskStatusText` RPC + `SetTaskStatusTextRequest` / `SetTaskStatusTextResponse` messages.
- `set_task_status_text` handler in `service.py`.
- `record_task_status_text`, the two dicts, and `remove_status_text_by_job_ids` in `ControllerTransitions`.
- `status_text_detail_md` / `status_text_summary_md` field names on `TaskStatus` — field numbers (21, 22) are **reserved** via `reserved 21, 22;` in `job.proto`, not deleted, so they cannot be silently reused.
- The handler test (`tests/cluster/controller/test_service.py::test_set_task_status_text_persists_via_store`) and any `benchmark_controller.py` cases for the deleted RPC.

Rollout = single PR. Workers, controller, and dashboard bundle redeploy together. Transient blank status text on long-open dashboard tabs is acceptable; a refresh clears it once the new bundle loads. Concrete public surface in [`spec.md`](./spec.md).

## Testing

_Agents make mistakes — how do we catch them?_

- **Unit-style:** worker calls `report_task_status_text` against a fake finelog `StatsService` and the test asserts a row appears in the namespace with the right (`task_id`, `ts`, fields). Mirrors the `iris.profile` worker test in `iris_profile_to_finelog`.
- **Integration:** spin up an iris dev cluster with the new controller + worker. Launch a job whose task calls `report_task_status_text` a few times with distinct strings. Verify (a) dashboard task-detail page renders the latest string; (b) `controller.sqlite3` has no status-text columns/dicts; (c) `Query("SELECT count(*) FROM \"iris.zephyr_task_status\"")` returns the expected row count.
- **Rollout check:** before/after tcpdump on the controller — confirm `SetTaskStatusText` count drops to zero and the inbound RPC total drops by ~70 %. This is the primary success signal for the design.
- **Negative test:** confirm an old worker calling the deleted RPC fails fast (404 / UNIMPLEMENTED) rather than silently dropping. Workers and controller redeploy together so this should never happen in practice, but we don't want the wrong behavior if it does.

## Open Questions

- **Namespace name.** `iris.zephyr_task_status` reflects the dominant producer but ties an iris-level namespace to a zephyr-level concept. The caller is iris-generic (`IrisClient.report_task_status_text` can be called from non-zephyr code too). Reviewers: keep the descriptive name, or stay generic (`iris.task_status_text`)?
- **No retention policy.** Relying on finelog compaction. Worst case: a task calling `report_task_status_text` once/sec for 7 days = ~600 k rows × ~1 KB ≈ 600 MB per task. With ~1000 long-running tasks that's ≈ 600 GB — non-trivial, but bounded. Defer to a follow-up that adds a per-namespace TTL once we measure real call rates, or add a TTL up front?
- **Silent-drop on flush failure.** `Table.write` swallows ConnectError in the bg flush thread (`log_client.py:399`). Today's RPC error path crashes the task (no caller wraps it). The design treats the new behaviour as a fix — best-effort observability should not abort tasks — but flagging in case a reviewer disagrees.
