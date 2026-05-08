# Iris CPU profiles → finelog

_Why are we doing this? What's the benefit?_

Move CPU profile collection out of the Iris controller and into the workers themselves, with profiles persisted in finelog (`iris.cpu_profile`) instead of a controller-attached SQLite database. The controller stops bookkeeping time-series data it does not own — after this lands, the controller DB stores only registry and decisions, while every measurement (utilization, resource samples, profiles) lives in a finelog namespace. This is the same shape as the `iris_stats_migration.md` lift that moved `iris.worker` and `iris.task`; we are finishing the job.

The current controller-side fan-out is also a real bottleneck. On clusters with many workers, the loop dispatches `ProfileTask` RPCs through a bounded `ThreadPoolExecutor(profile_concurrency=8)` ([`controller.py:1663`](https://github.com/marin-community/marin/blob/24ebc3b1/lib/iris/src/iris/cluster/controller/controller.py#L1663)) — at 100+ workers the tail can run minutes behind. With each worker driving its own loop, captures fan out automatically and the central coordinator goes away.

## Background

The controller spawns a `profile-loop` thread that ticks every 10 minutes, fans `ProfileTask` RPCs across all workers, and persists results to `profiles.task_profiles` in an attached SQLite file ([`controller.py:1607`](https://github.com/marin-community/marin/blob/24ebc3b1/lib/iris/src/iris/cluster/controller/controller.py#L1607); [`schema.py:1031`](https://github.com/marin-community/marin/blob/24ebc3b1/lib/iris/src/iris/cluster/controller/schema.py#L1031)). The loop runs only on direct-provider clusters — k8s mode never had periodic profiles ([`controller.py:1344`](https://github.com/marin-community/marin/blob/24ebc3b1/lib/iris/src/iris/cluster/controller/controller.py#L1344)). Workers already write to finelog for stats and already host `WorkerService.ProfileTask` ([`worker.proto:111`](https://github.com/marin-community/marin/blob/24ebc3b1/lib/iris/src/iris/rpc/worker.proto#L111)) — they can run the loop themselves. See `research.md` for the full file:line inventory and the Q&A that fixed the four load-bearing choices (kept on-demand path via finelog, CPU-only auto loop, 7-day retention, dashboard reads via StatsService SQL).

## Challenges

The worker has never had an autonomous periodic loop — it is fundamentally RPC-driven, with the heartbeat deadline reset by inbound `Ping` / `PollTasks` calls. The 10-minute profile loop is the first wall-clock-driven cron in the worker process. We need a thread that respects the worker's lifecycle (`start` / `stop` / re-register / adopted-without-controller), does not pile up work if the previous round is still running, and tolerates `_log_client = None` modes (test, no-controller-address) without crashing.

The k8s direct-provider path is a real edge case. There is no worker process there to host a loop; the controller talks to pods through `K8sTaskProvider.profile_task` ([`providers/k8s/tasks.py:1155`](https://github.com/marin-community/marin/blob/24ebc3b1/lib/iris/src/iris/cluster/providers/k8s/tasks.py#L1155)), and today's controller already does not run periodic profiles in this mode. We preserve that status quo — k8s mode loses its on-demand profile-button if we naively delete the controller RPC, so the controller keeps a thin `profile_task` dispatcher *only* in k8s mode, with no DB writes.

## Costs / Risks

- **`/system/process` profiling on the controller goes away.** The dashboard "profile this controller" button on `StatusTab.vue` loses its target. SREs who want to py-spy the controller will SSH to the box. Worker self-profile (`/system/process` on the worker) stays via the worker RPC.
- **ptrace pause cost.** `py-spy record` uses `PTRACE_ATTACH` + stack-walk, which stops the target during sampling. At 10s every 600s that is ~1.7% steady wall-clock overhead, but on a TPU/GPU host with tightly-coupled NCCL collectives the pause can trip stalled-collective warnings or short timeouts. Mitigation: an opt-out attribute on tasks marked latency-sensitive (deferred — see Open Questions).
- **Capture sizing is unmeasured.** Today's `IrisTaskStat` rows are tens of bytes; profile rows are bytes blobs. We expect single-digit GB/day fleet-wide, but raw py-spy output on a JAX worker with hundreds of threads is hundreds of KB per capture, not the tens of KB the SQLite payload sized. We will measure on the dev cluster before enabling fleet-wide and decide on per-row compression then (see Open Questions).
- **No more per-task cap.** Today's SQLite trigger keeps 10 rows per `(task_id, profile_kind)`. Time-based retention will keep *all* captures inside the window — long-running tasks will accumulate ~1000 profiles over 7 days. The dashboard table needs `LIMIT` and a date filter; covered in spec §5.
- **Migration churn for a fourth time.** Fourth move for `task_profiles` (created → fk added → kind added → split DB → now deleted). Justified because the destination is the same finelog backend that already stores `iris.task` and `iris.worker` — the row class joins `IrisWorkerStat`/`IrisTaskStat` in `worker/stats.py`.

## Design

**Worker periodic loop.** A new `_run_profile_loop` thread spawned alongside the existing lifecycle thread, ticking every 10 minutes via the same `RateLimiter` pattern the controller uses today. Each tick iterates `self._tasks`, calls `profile_local_process(duration=10s, profile_type=cpu)` against each running attempt, and writes one row to the `iris.cpu_profile` finelog table. CPU profiles run sequentially within a worker by default (one py-spy invocation at a time on the host), with a `profile_concurrency: int = 1` worker config knob to tune for multi-task hosts. Across workers they run in parallel automatically. Per-task exceptions are logged at `exception` level; one flaky task does not skip the rest.

**Single writer.** The worker's `ProfileTask` RPC handler now also writes to `iris.cpu_profile` when the request is CPU and the target is a task — so the periodic loop and dashboard "profile now" go through one code path. The handler still returns bytes inline so the dashboard does not poll. Memory and threads captures keep their inline-only behaviour. Worker self-profile (`/system/process`) returns inline only.

```python
# lib/iris/src/iris/cluster/worker/profile_loop.py (new)
def run_profile_loop(*, stop_event, interval, list_running_attempts, capture_one):
    limiter = RateLimiter(interval_seconds=interval.to_seconds())
    while not stop_event.is_set():
        if (delay := limiter.time_until_next()) > 0:
            stop_event.wait(timeout=delay)
            if stop_event.is_set(): break
        limiter.mark_run()
        for attempt in list_running_attempts():
            try:
                capture_one(attempt, trigger="periodic")
            except Exception:
                logger.exception("profile capture failed for %s", attempt.task_id)
```

**Finelog namespace.** `iris.cpu_profile` with row class `IrisCpuProfile(key_column="captured_at")` — fields: `task_id, attempt_id, worker_id, captured_at, duration_seconds, rate_hz, native, format, trigger, profile_data`. `format` (`raw|flamegraph|speedscope`) and `trigger` (`periodic|on_demand`) follow the `StrEnum` convention used by `WorkerStatus` in `stats.py:32`. Registered eagerly at worker start via `LogClient.get_table` so schema mismatches surface on first ping. Retention: 7 days, configured via finelog's standard per-namespace TTL and documented in `OPS.md`.

**Controller deletions.** `_run_profile_loop`, `_profile_all_running_tasks`, `_dispatch_profiles`, `_capture_one_profile`, `_profile_thread`, the three `profile_*` config knobs, the `task_profiles` table, the attached `profiles` SQLite database, `insert_task_profile` / `get_task_profiles`, the `Provider.profile_task` interface (worker-based providers), and the controller-side `profile_task` RPC handler — gone. Migrations 0005/0014/0020/0023 stay on disk as no-op chain links (collapsing them is a separate cleanup); a new `0024_drop_profiles_db.py` `DETACH`-es and `unlink`-s the file.

**K8s mode.** The controller keeps a slim `profile_task` RPC *only* when `provider isinstance K8sTaskProvider`, dispatching to `K8sTaskProvider.profile_task` and returning bytes inline. No DB writes, no periodic loop on the k8s path — preserves status quo. Adding a periodic loop on the k8s direct-provider path is out of scope; flagged in Open Questions.

**Dashboard.** Drop the controller-side `useProfileAction` calls and route the "profile now" button at the worker via the existing `proxy/worker/<worker_id>/iris.cluster.WorkerService/ProfileTask` proxy. Add a "Profile history" panel on `TaskDetail.vue` running `SELECT captured_at, attempt_id, format, trigger, length(profile_data) FROM "iris.cpu_profile" WHERE task_id = ? ORDER BY captured_at DESC LIMIT 50` through `useStatsRpc`. Clicking a row downloads `profile_data` via a second targeted SQL.

**Commit ordering.** Each commit independently revertable, tests green at every step:
1. Introduce `IrisCpuProfile` + worker `_run_profile_loop` + table registration. Worker writes both old (controller-driven captures still hit `task_profiles`) and new namespace.
2. Repoint dashboard "profile now" buttons at worker proxy; add "Profile history" panel reading `iris.cpu_profile`.
3. Delete controller `_run_profile_loop` / helpers / config; delete worker-providers' `profile_task` plumbing; keep slim k8s-only controller RPC.
4. Drop `task_profiles` table and `profiles.sqlite3`: schema change + migration `0024_drop_profiles_db.py`.
5. Document in `lib/iris/AGENTS.md` and `OPS.md`.

## Testing

- **Unit:** `run_profile_loop` (advances on a clock; skips non-running attempts; swallows per-attempt failures; stops promptly between captures); worker `ProfileTask` handler's finelog write (uses `MemoryLogNamespace`); `_log_client = None` mode skips cleanly.
- **Integration:** end-to-end on the iris dev cluster — submit a long-running task, wait one tick, query `iris.cpu_profile` via StatsService, assert ≥1 row with non-empty `profile_data`. Run on both old-controller-with-new-workers and full-cluster-restart configurations. Add a k8s-provider integration test verifying on-demand still works through the slim controller RPC, and that no periodic rows appear in finelog (since the k8s loop is intentionally not added).
- **Migration:** `0024_drop_profiles_db.py` deletes `profiles.sqlite3` exactly once; tolerates re-runs and missing files. Profile data is diagnostic, not load-bearing — no backup.
- **No regression:** `iris.task` / `iris.worker` stats tests must keep passing.

## Open Questions

- **Capture size + compression.** Production py-spy raw output on JAX workers can be hundreds of KB per capture. Should we per-row gzip in the worker before writing (5-line change), rely on parquet block compression (low gain — bytes are mostly distinct), or measure on dev first and decide? Default plan: measure first.
- **Namespace name lock-in.** `iris.cpu_profile` is CPU-only by name. If we later persist memory/threads, we either add parallel namespaces (`iris.memory_profile`, `iris.threads_profile`) or rename today to `iris.task_profile` with a `kind` discriminator. The user's framing chose `iris.cpu_profile`; flagging in case reviewers prefer the discriminator-from-day-one shape.
- **Profile interval source.** Per-worker config knob (today's plan) vs. controller-pushed via `Ping.profile_interval` (one fleet-wide value). The latter is one extra proto field; the former is simpler but drifts across worker restarts.
- **Per-task circuit breaker.** Today the loop retries forever even if py-spy fails on the same attempt every tick. Add a per-`(task_id, attempt_id)` failure counter that backs off after N consecutive failures, or accept the noise?
- **Latency-sensitive opt-out.** Should tasks declare a `no-profile` attribute (e.g., on the JobConfig)? Tightly-coupled NCCL workloads may want to.
- **K8s periodic profiling.** Should the K8sTaskProvider grow its own loop writing to finelog from the controller process, or stay on-demand-only as today?
