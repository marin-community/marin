# Research: status text → finelog

Background for the 1-pager in `design.md`. Captures the file-level facts the design rests on, the precedents we are following, and the Q&A that fixed the load-bearing choices.

## 1. What the current path looks like

### Write path (worker → controller, in-memory)

- Proto: [`job.proto:635`](https://github.com/marin-community/marin/blob/b06b8cf5/lib/iris/src/iris/rpc/job.proto#L635) — `SetTaskStatusTextRequest { task_id, status_text_detail_md, status_text_summary_md }`. Response is empty.
- RPC: [`controller.proto:639`](https://github.com/marin-community/marin/blob/b06b8cf5/lib/iris/src/iris/rpc/controller.proto#L639) — `ControllerService.SetTaskStatusText`.
- Worker caller: [`remote_client.py:510`](https://github.com/marin-community/marin/blob/b06b8cf5/lib/iris/src/iris/cluster/client/remote_client.py#L510) — `RemoteClient.push_task_status_text(task_id, detail_md, summary_md)`. Called on-demand by task code (not periodic).
- Server handler: [`service.py:2605`](https://github.com/marin-community/marin/blob/b06b8cf5/lib/iris/src/iris/cluster/controller/service.py#L2605) — `set_task_status_text` is `@on_loop` (runs inline on the asyncio loop), delegates to `ControllerTransitions.record_task_status_text(...)`, which mutates two in-memory dicts (`_status_text_detail`, `_status_text_summary`).
- No SQLite, no persistence. Eviction happens via `remove_status_text_by_job_ids` during pruning.

### Read path (dashboard → controller)

- Dashboard reads via `GetTaskStatus` RPC on the controller. The handler reads the in-memory dicts and attaches both fields to the `TaskStatus` proto: [`service.py:1511`](https://github.com/marin-community/marin/blob/b06b8cf5/lib/iris/src/iris/cluster/controller/service.py#L1511).
- Proto fields on `TaskStatus`: [`job.proto:21`](https://github.com/marin-community/marin/blob/b06b8cf5/lib/iris/src/iris/rpc/job.proto#L21) (detail + summary).
- Dashboard render: `lib/iris/dashboard/src/components/controller/TaskDetail.vue:356-358` — `MarkdownRenderer` on `task.statusTextDetailMd`.

### Traffic share

60 s tcpdump on the live controller, 10 295 RPCs total (~8.5 MB):

| method | count | pct | KB |
|---|---|---|---|
| `SetTaskStatusText` | 7566 | **73.5%** | 5830 |
| `GetJobState` | 1154 | 11.2% | 957 |
| `UpdateTaskStatus` | 769 | 7.5% | 785 |
| `ListEndpoints` | 681 | 6.6% | 519 |

Per call is small (~789 B). Volume × per-RPC overhead (starlette middleware → auth interceptor → timing interceptor with `_record` + `_bump_bucket` + slow-RPC log → `@on_loop` handler) is the cost.

## 2. What finelog already gives us

- Process: `finelog.server` runs `LogService` and `StatsService` in the **same bundled server** (typically a local subprocess started by the controller, or a remote address via `--log-service-address`). See `controller.py:1402-1410`.
- Stats RPCs: [`finelog_stats.proto:139`](https://github.com/marin-community/marin/blob/b06b8cf5/lib/finelog/src/finelog/proto/finelog_stats.proto#L139) — `RegisterTable`, `WriteRows` (Arrow IPC batch), `Query` (DuckDB SQL → Arrow), `DropTable`, `ListNamespaces`, `GetTableSchema`.
- Persistence: DuckDB + parquet segments per namespace, level-compacted (see `2026-05-05_finelog_compactor.md`).
- Server impl: [`stats_service.py:48`](https://github.com/marin-community/marin/blob/b06b8cf5/lib/finelog/src/finelog/server/stats_service.py#L48). All writes are off-loop in finelog's process (no impact on iris controller loop).
- Client surface: workers already use `LogClient.get_table(namespace, Dataclass)` to get a typed write handle — same pattern as `iris.worker`, `iris.task`, `iris.profile`.
- Controller already proxies the stats RPC to finelog: `dashboard.py:537` mounts `FinelogStatsServiceASGIApplication(service=self._finelog_stats_service, interceptors=[auth_interceptor], ...)` where `self._finelog_stats_service = StatsServiceProxy(self._log_service_address, ...)`. **Dashboard `POST /finelog.stats.StatsService/Query` already works today** — it just isn't used for status text yet.

## 3. Precedents in `.agents/projects/`

The two relevant lift designs:

- `iris_stats_migration.md` — established the cut-line "registry + decisions live in controller DB; measurements live in finelog namespaces" and moved `worker_resource_history` / `task_resource_history` to `iris.worker` / `iris.task`. Same shape: define namespace, define dataclass, workers register on start, controller stops storing.
- `iris_profile_to_finelog/design.md` — moved CPU/memory/thread profiles to a single `iris.profile` namespace discriminated by `type` and `source`. Same shape; cleanest precedent for "single namespace covers a category of writes." Dashboard reads via `StatsService.Query` SQL.

Status text fits the same pattern: it is a per-task display string, not a control-plane datum. The cut-line says it belongs in finelog.

Other docs (no overlap, listed for completeness): `20260315_iris_controller_query_design.md`, `20260331_iris_sql_redesign.md`, `2026-05-05_finelog_compactor.md`.

## 4. Q&A that fixed load-bearing choices

1. **Write API: focused RPC or generic `WriteRows` + schema?** → Generic `WriteRows`. Namespace `iris.zephyr_task_status`. Matches `iris.worker` / `iris.task` / `iris.profile`. No new finelog RPC. (The namespace is `iris.zephyr_*` rather than `iris.task_*` because the heavy producers of status text in the wild are Zephyr tasks; status text is Zephyr-shaped.)
2. **Retention: indefinite, TTL, or terminal-purge?** → **Last-write-wins per task, indefinite, compacted by finelog.** Dashboard does `ORDER BY ts DESC LIMIT 1`. Storage grows with task count; finelog's level-compactor handles segment count.
3. **Dashboard read path: dashboard-direct, controller-proxy, or join in `GetTaskStatus`?** → **Dashboard calls `finelog.stats.StatsService.Query` via the controller-mounted ASGI sub-app** (`dashboard.py:537`). On the wire the request flows through the controller, but the controller's cost is auth-interceptor + network forward — no timing interceptor, no `_record`, no `to_thread`, no `@on_loop` handler. Logical separation, zero infra surgery (no CORS, no second auth, no new ingress). The alternative "browser → finelog host:port directly" buys nothing for the bottleneck and adds operational surface.
4. **Out of scope:**
   - No migration of the existing in-memory dict. Workers re-emit on next update; current text vanishes on rollout (it's transient by design today anyway).
   - No history UI — dashboard still surfaces "latest" only, even though finelog now retains history.
   - Independent from #5574 (UpdateTaskStatus push removal). Adjacent reduction in controller load; different RPC, different rollout.

## 5. Open issues / PRs

- #5909 — the issue this design implements.
- #5574 — adjacent: removes the push-based `UpdateTaskStatus` in favor of `PollTasks` reconcile. Independent rollout.

No prior PRs touch `SetTaskStatusText` directly.

## 6. Things that surprised me / worth flagging for review

- The handler is **already `@on_loop`**, so it does not pay `to_thread` cost. The win on the controller is **everything outside the handler** — middleware + interceptors + per-RPC log/stats accounting — multiplied by 7500+ calls per minute.
- Finelog stats is already proxied through the controller's ASGI app today (`dashboard.py:537`). The dashboard read path needs only client-side changes: replace one field read with a `StatsService.Query` call against the existing endpoint.
- Naming: `iris.zephyr_task_status` ties an iris-side namespace to a Zephyr concept. If non-Zephyr task code ever wants to push status text, the name reads oddly. Worth flagging as a minor open question for reviewers, but not blocking.
