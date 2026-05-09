# Research — iris profile collection → finelog

Companion to `design.md`. Captures what was found in-repo and the Q&A that shaped the design. Live links use SHA `24ebc3b1` (HEAD of `iris-sdist-scope` at the time of writing, equal to `main`).

## Current architecture (what we are removing / changing)

### Controller-driven profile loop

The controller spawns a dedicated `profile-loop` thread at startup and ticks every 10 minutes:

- Thread spawn: [`controller.py:1351`](https://github.com/marin-community/marin/blob/24ebc3b1/lib/iris/src/iris/cluster/controller/controller.py#L1351) — `self._profile_thread = self._threads.spawn(self._run_profile_loop, name="profile-loop")`.
- Loop body: [`controller.py:1607-1626`](https://github.com/marin-community/marin/blob/24ebc3b1/lib/iris/src/iris/cluster/controller/controller.py#L1607) (`_run_profile_loop`) — `RateLimiter(profile_interval)` then `_profile_all_running_tasks`.
- Fan-out: [`controller.py:1627-1670`](https://github.com/marin-community/marin/blob/24ebc3b1/lib/iris/src/iris/cluster/controller/controller.py#L1627) — `_profile_all_running_tasks` reads `healthy_active_workers_with_attributes` + `running_tasks_by_worker`, builds (task_id, worker) targets, dispatches via bounded `ThreadPoolExecutor` (`profile_concurrency=8`).
- One-shot capture: [`controller.py:1672-1704`](https://github.com/marin-community/marin/blob/24ebc3b1/lib/iris/src/iris/cluster/controller/controller.py#L1672) (`_capture_one_profile`) — issues `ProfileTaskRequest` to the worker via `provider.profile_task`, calls `insert_task_profile(db, …)` on success.
- Config knobs: [`controller.py:1028,1031,1046`](https://github.com/marin-community/marin/blob/24ebc3b1/lib/iris/src/iris/cluster/controller/controller.py#L1028) — `profile_interval=600s`, `profile_duration=10s`, `profile_concurrency=8`.

Memory profiling via memray was disabled in the loop (segfault notes in the docstring) — only CPU is captured today.

### Profile DB table + GC trigger

The table lives in an attached SQLite file `profiles.sqlite3`, separate from the main controller DB:

- Schema row class: [`schema.py:1031-1071`](https://github.com/marin-community/marin/blob/24ebc3b1/lib/iris/src/iris/cluster/controller/schema.py#L1031) — `TASK_PROFILES = Table("profiles.task_profiles", …)` with columns `id, task_id, profile_data BLOB, captured_at_ms, profile_kind`.
- Per-key cap trigger: [`schema.py:1056-1069`](https://github.com/marin-community/marin/blob/24ebc3b1/lib/iris/src/iris/cluster/controller/schema.py#L1056) — `trg_task_profiles_cap` keeps the most-recent 10 rows per `(task_id, profile_kind)`.
- DB attach: [`db.py:297,306,314,394,411-412,628-629`](https://github.com/marin-community/marin/blob/24ebc3b1/lib/iris/src/iris/cluster/controller/db.py#L297) — `PROFILES_DB_FILENAME = "profiles.sqlite3"`, attached as the `profiles` schema.
- Reader / writer: [`db.py:987,1000-1022`](https://github.com/marin-community/marin/blob/24ebc3b1/lib/iris/src/iris/cluster/controller/db.py#L987) — `insert_task_profile`, `get_task_profiles`.
- Migrations on the chopping block: `0005_task_profiles.py`, `0014_profile_kind.py`, `0020_perf_indices_and_profiles_fk.py`, `0023_separate_profiles_db.py`.

There is no separate pruning loop — the SQLite trigger is the only retention mechanism.

### On-demand profiling RPC

The `profile_task` controller RPC services dashboard "profile now" buttons:

- Handler: [`service.py:1893-1968`](https://github.com/marin-community/marin/blob/24ebc3b1/lib/iris/src/iris/cluster/controller/service.py#L1893) — three target types:
  - `/system/process` → profiles the controller process via `profile_local_process` ([`runtime/profile.py:158`](https://github.com/marin-community/marin/blob/24ebc3b1/lib/iris/src/iris/cluster/runtime/profile.py#L158)).
  - `/system/worker/<id>` → forwards `/system/process` to that worker.
  - `/job/.../task/N` → forwards to the task's worker.
- Returns synchronously with `profile_data` bytes; does **not** persist on-demand captures to the DB. (Only the periodic loop persists.)

### Worker side: existing infra

Workers already host `WorkerService.ProfileTask` ([`worker.proto:111`](https://github.com/marin-community/marin/blob/24ebc3b1/lib/iris/src/iris/rpc/worker.proto#L111)) and use `profile_local_process` for capture. They track running tasks in `_tasks: dict[(task_id, attempt_id), TaskAttempt]` ([`worker.py:197`](https://github.com/marin-community/marin/blob/24ebc3b1/lib/iris/src/iris/cluster/worker/worker.py#L197)).

Workers already write to finelog: `_log_client = LogClient.connect(...)` then `get_table(WORKER_STATS_NAMESPACE, IrisWorkerStat)` and `get_table(TASK_STATS_NAMESPACE, IrisTaskStat)` ([`worker.py:266,281-282`](https://github.com/marin-community/marin/blob/24ebc3b1/lib/iris/src/iris/cluster/worker/worker.py#L266)). Adding a third table is a paved-path change.

The worker has **no** existing periodic timer-driven loop — it is fundamentally RPC-driven. The new 10m profile loop is the first standalone periodic task on the worker side.

### Dashboard read path

The dashboard already queries the finelog StatsService for stats namespaces (and would need an additional query for profiles):

- StatsService composable: [`useRpc.ts:103-119`](https://github.com/marin-community/marin/blob/24ebc3b1/lib/iris/dashboard/src/composables/useRpc.ts#L103) — routes via `proxy/system.log-server/finelog.stats.StatsService/<Method>`.
- Existing usage: [`TaskDetail.vue:88`](https://github.com/marin-community/marin/blob/24ebc3b1/lib/iris/dashboard/src/components/controller/TaskDetail.vue#L88), [`WorkerDetail.vue:88`](https://github.com/marin-community/marin/blob/24ebc3b1/lib/iris/dashboard/src/components/controller/WorkerDetail.vue#L88), [`JobDetail.vue:121`](https://github.com/marin-community/marin/blob/24ebc3b1/lib/iris/dashboard/src/components/controller/JobDetail.vue#L121) — `FROM "iris.task"` / `FROM "iris.worker"` SQL via `useStatsRpc`.
- Profile UI: [`useProfileAction.ts`](https://github.com/marin-community/marin/blob/24ebc3b1/lib/iris/dashboard/src/composables/useProfileAction.ts), [`ProfileButtons.vue`](https://github.com/marin-community/marin/blob/24ebc3b1/lib/iris/dashboard/src/components/shared/ProfileButtons.vue) — both invoke the controller `profile_task` RPC and render the inline bytes; today there is no "list previous profiles" view.

## Prior art in-repo

The cleanest analogue is the **`iris_stats_migration.md`** design — same shape: lift a per-row controller SQLite table into a finelog stats namespace, drop the prune loop, repoint the dashboard. That migration created `iris.worker` and `iris.task`. The naming and dataclass conventions there are the template ([`worker/stats.py:47-92`](https://github.com/marin-community/marin/blob/24ebc3b1/lib/iris/src/iris/cluster/worker/stats.py#L47): `IrisWorkerStat`, `IrisTaskStat` — `key_column = "ts"`, datetime, top-level dataclass). We follow that pattern verbatim for `IrisProfile` (one namespace `iris.profile`, discriminated by `type` and `source` columns — see spec §1.1).

`AGENTS.md` boundary holds: finelog is iris-agnostic; iris-specific helpers live under `iris/cluster/log_store_helpers.py`. We will not introduce iris-knowledge into finelog ([`lib/finelog/AGENTS.md:27-35`](https://github.com/marin-community/marin/blob/24ebc3b1/lib/finelog/AGENTS.md#L27)).

## Q&A summary (load-bearing decisions)

1. **On-demand profiling:** Keep the "profile now" button. The controller `profile_task` RPC is **kept** (renamed self-target to `/system/controller`); it dispatches to the actual capturer, which writes the row. All types (cpu/memory/thread) persist to `iris.profile`.
2. **Profile types:** The periodic loop captures CPU only. Memory and thread are on-demand only. All on-demand captures land in `iris.profile`.
3. **Retention:** Time-based via finelog's standard segment retention. **7 days** for `iris.profile`, documented in `OPS.md`. No application-side row-count cap (the trigger is gone).
4. **Dashboard read:** A new "Profile history" panel in `TaskDetail.vue` / `WorkerDetail.vue` / `StatusTab.vue` runs SQL through the existing `useStatsRpc` composable: `SELECT captured_at, profile_data FROM "iris.profile" WHERE source = ? ORDER BY captured_at DESC`. Same pattern as today's `iris.task` queries.
5. **Schema:** single namespace `iris.profile` with `type ∈ {cpu, memory, thread}`, `source ∈ {/job/.../task/N, /system/worker/<id>, /system/controller}`, `vm_id` for writer attribution. See spec §1.1.

## Skipped passes

- **Web prior-art search:** skipped. This is a relocation within an established in-repo pattern (the `iris_stats_migration.md` precedent), not a novel category-of-system.
- **GitHub issue search:** none queried (skill says the design-doc skill itself does not run `gh` searches; reviewer can link related issues on the PR).

## Open questions resolved during review

- "Profile this controller" button: **kept**. Renamed target from `/system/process` to `/system/controller`; controller writes its own row to `iris.profile`.
- On-demand memory/thread persistence: **persist all types** in the single `iris.profile` namespace. No separate `iris.memory_profile` / `iris.thread_profile` namespaces.
