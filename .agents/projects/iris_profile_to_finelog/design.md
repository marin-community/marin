# Iris CPU profiles → finelog

_Why are we doing this? What's the benefit?_

Move CPU profile collection out of the Iris controller and into the workers themselves, with profiles persisted in finelog (`iris.cpu_profile`) instead of a controller-attached SQLite database. The controller stops bookkeeping time-series data it does not own — after this lands, the controller DB stores only registry and decisions, while every measurement (utilization, resource samples, profiles) lives in a finelog namespace. This is the same shape as the `iris_stats_migration.md` lift that moved `iris.worker` and `iris.task`; we are finishing the job.

The current controller-side fan-out is also a real bottleneck. On clusters with many workers, the loop dispatches `ProfileTask` RPCs through a bounded `ThreadPoolExecutor(profile_concurrency=8)` ([`controller.py:1663`](https://github.com/marin-community/marin/blob/24ebc3b1/lib/iris/src/iris/cluster/controller/controller.py#L1663)) — at 100+ workers the tail can run minutes behind. With each worker driving its own loop, captures fan out automatically and the central coordinator goes away.

## Background

The controller spawns a `profile-loop` thread that ticks every 10 minutes, fans `ProfileTask` RPCs across all workers, and persists results to `profiles.task_profiles` in an attached SQLite file ([`controller.py:1607`](https://github.com/marin-community/marin/blob/24ebc3b1/lib/iris/src/iris/cluster/controller/controller.py#L1607); [`schema.py:1031`](https://github.com/marin-community/marin/blob/24ebc3b1/lib/iris/src/iris/cluster/controller/schema.py#L1031)). The loop runs only on direct-provider clusters — k8s mode never had periodic profiles ([`controller.py:1344`](https://github.com/marin-community/marin/blob/24ebc3b1/lib/iris/src/iris/cluster/controller/controller.py#L1344)). Workers already write to finelog for stats and already host `WorkerService.ProfileTask` ([`worker.proto:111`](https://github.com/marin-community/marin/blob/24ebc3b1/lib/iris/src/iris/rpc/worker.proto#L111)). The provider abstraction already exposes `provider.profile_task(...)` ([`controller.py:1688`](https://github.com/marin-community/marin/blob/24ebc3b1/lib/iris/src/iris/cluster/controller/controller.py#L1688), [`providers/k8s/tasks.py:1155`](https://github.com/marin-community/marin/blob/24ebc3b1/lib/iris/src/iris/cluster/providers/k8s/tasks.py#L1155)) — worker-based providers forward to the worker; the k8s provider captures via `kubectl exec`. We use that seam: keep the controller's `profile_task` RPC as the dashboard-facing entry, move persistence onto whichever party actually does the capture (worker or k8s provider), and delete only the controller's *periodic loop* and *DB storage*. See `research.md` for the full file:line inventory and the Q&A that fixed the four load-bearing choices (kept on-demand path via finelog, CPU-only auto loop, 7-day retention, dashboard reads via StatsService SQL).

## Architecture

### Before — controller-driven fan-out, SQLite storage

```mermaid
sequenceDiagram
    autonumber
    actor User as Dashboard
    participant CTL as Controller
    participant LOOP as profile-loop<br/>(controller thread)
    participant PROV as Provider
    participant W as Worker
    participant DB as profiles.sqlite3<br/>(attached to CTL)

    rect rgba(180, 200, 240, 0.18)
        Note over LOOP,DB: Periodic capture — every 600s, fan-out from controller
        LOOP->>CTL: read healthy workers + running tasks
        loop for each (task, worker) — bounded ThreadPool, concurrency=8
            LOOP->>+PROV: profile_task(worker_addr, request)
            PROV->>+W: WorkerService.ProfileTask
            W->>W: py-spy record (10s)
            W-->>-PROV: ProfileTaskResponse(bytes)
            PROV-->>-LOOP: bytes
            LOOP->>DB: insert_task_profile(task_id, bytes)
            Note right of DB: trigger caps to<br/>10 rows / (task, kind)
        end
    end

    rect rgba(245, 220, 195, 0.25)
        Note over User,DB: On-demand "profile now" — synchronous, NOT persisted
        User->>+CTL: profile_task RPC
        CTL->>+PROV: provider.profile_task(...)
        PROV->>+W: WorkerService.ProfileTask
        W-->>-PROV: bytes
        PROV-->>-CTL: bytes
        CTL-->>-User: bytes
    end

    rect rgba(200, 230, 200, 0.25)
        Note over User,DB: History view — controller reads SQLite
        User->>+CTL: GetTaskProfiles RPC
        CTL->>DB: SELECT profile_data FROM task_profiles<br/>WHERE task_id = ? ORDER BY id DESC
        DB-->>CTL: rows (≤ 10)
        CTL-->>-User: profiles
    end
```

### After — worker-driven loop, finelog storage, controller is a pure dispatcher

```mermaid
sequenceDiagram
    autonumber
    actor User as Dashboard
    participant CTL as Controller<br/>(pure dispatcher)
    participant PROV as Provider
    participant W as Worker
    participant LOOP as profile-loop<br/>(worker thread)
    participant K8S as K8sTaskProvider<br/>(controller process)
    participant FL as finelog<br/>iris.cpu_profile

    rect rgba(180, 200, 240, 0.18)
        Note over W,FL: Periodic capture — runs independently in each worker
        loop every 600s, sequential within worker, parallel across workers
            LOOP->>W: list local _tasks
            LOOP->>W: py-spy record (10s)
            LOOP->>FL: Table.write(IrisCpuProfile,<br/>trigger="periodic")
        end
    end

    rect rgba(245, 220, 195, 0.25)
        Note over User,FL: On-demand "profile now" — same dashboard path, provider abstraction routes
        User->>+CTL: profile_task RPC
        CTL->>+PROV: provider.profile_task(...)
        alt worker-based provider
            PROV->>+W: WorkerService.ProfileTask
            W->>W: py-spy record
            W->>FL: Table.write(trigger="on_demand")
            W-->>-PROV: bytes
        else K8sTaskProvider
            PROV->>+K8S: kubectl exec py-spy
            K8S->>FL: Table.write(trigger="on_demand")
            K8S-->>-PROV: bytes
        end
        PROV-->>-CTL: bytes
        CTL-->>-User: bytes (CTL never touches storage)
    end

    rect rgba(200, 230, 200, 0.25)
        Note over User,FL: History view — dashboard reads finelog directly
        User->>+FL: StatsService.Query<br/>SELECT … FROM "iris.cpu_profile" …
        FL-->>-User: rows (Arrow IPC)
    end
```

The two diagrams share the on-demand RPC shape — only the storage and the periodic loop's ownership change. Workers gain one new thread (`profile-loop`); the controller loses the loop, the table, and any `Table.write` for profile data.

## Challenges

The worker has never had an autonomous periodic loop — it is fundamentally RPC-driven, with the heartbeat deadline reset by inbound `Ping` / `PollTasks` calls. The 10-minute profile loop is the first wall-clock-driven cron in the worker process. We need a thread that respects the worker's lifecycle (`start` / `stop` / re-register / adopted-without-controller), does not pile up work if the previous round is still running, and tolerates `_log_client = None` modes (test, no-controller-address) without crashing.

The k8s direct-provider path needs care. There is no worker process there to host a loop, and `K8sTaskProvider` runs in the controller process. We do not add a periodic loop on k8s in v1 (matches today). For on-demand on k8s, the K8sTaskProvider already does the capture — it grows a finelog write before returning. That means the controller process is technically the writer for k8s captures, even though the controller no longer hosts a *loop* or *table*. The framing's "remove all vestiges of profiling from the iris controller" applies to collection orchestration and storage; the provider abstraction is the legitimate place for k8s to live.

## Costs / Risks

- **`/system/process` profiling on the controller goes away.** The dashboard "profile this controller" button on `StatusTab.vue` loses its target. SREs who want to py-spy the controller will SSH to the box. Worker self-profile (`/system/process` on the worker) stays via the worker RPC.
- **ptrace pause cost.** `py-spy record` uses `PTRACE_ATTACH` + stack-walk, which stops the target during sampling. At 10s every 600s that is ~1.7% steady wall-clock overhead, but on a TPU/GPU host with tightly-coupled NCCL collectives the pause can trip stalled-collective warnings or short timeouts. Mitigation: an opt-out attribute on tasks marked latency-sensitive (deferred — see Open Questions).
- **Capture sizing is unmeasured.** Today's `IrisTaskStat` rows are tens of bytes; profile rows are bytes blobs. We expect single-digit GB/day fleet-wide, but raw py-spy output on a JAX worker with hundreds of threads is hundreds of KB per capture, not the tens of KB the SQLite payload sized. We will measure on the dev cluster before enabling fleet-wide and decide on per-row compression then (see Open Questions).
- **No more per-task cap.** Today's SQLite trigger keeps 10 rows per `(task_id, profile_kind)`. Time-based retention will keep *all* captures inside the window — long-running tasks will accumulate ~1000 profiles over 7 days. The dashboard table needs `LIMIT` and a date filter; covered in spec §5.
- **Migration churn for a fourth time.** Fourth move for `task_profiles` (created → fk added → kind added → split DB → now deleted). Justified because the destination is the same finelog backend that already stores `iris.task` and `iris.worker` — the row class joins `IrisWorkerStat`/`IrisTaskStat` in `worker/stats.py`.

## Design

**Worker periodic loop.** A new `_run_profile_loop` thread spawned alongside the existing lifecycle thread, ticking every 10 minutes via the same `RateLimiter` pattern the controller uses today. Each tick iterates `self._tasks`, calls `profile_local_process(duration=10s, profile_type=cpu)` against each running attempt, and writes one row to the `iris.cpu_profile` finelog table. CPU profiles run sequentially within a worker by default (one py-spy invocation at a time on the host), with a `profile_concurrency: int = 1` worker config knob to tune for multi-task hosts. Across workers they run in parallel automatically. Per-task exceptions are logged at `exception` level; one flaky task does not skip the rest.

**Dashboard path unchanged.** The dashboard "profile now" button still calls the controller's `profile_task` RPC. The handler resolves the target and delegates to `provider.profile_task(...)` — same dispatch the existing periodic loop uses today. For worker-based providers, the provider forwards to the worker via `WorkerService.ProfileTask`; the worker captures, writes to `iris.cpu_profile`, returns bytes inline. For k8s, `K8sTaskProvider.profile_task` captures via `kubectl exec`, writes to `iris.cpu_profile`, returns bytes inline. The controller does not write to finelog itself — the writer is whoever holds the ptrace handle.

**Single writer per side.** The worker's `_capture_and_log_cpu_profile` helper is the only finelog writer on workers, called by both the periodic loop and the worker's `ProfileTask` RPC handler. K8s has its own writer inside `K8sTaskProvider.profile_task`. Memory and threads captures keep their inline-only behaviour on both sides. Worker self-profile (`/system/process`) returns inline only.

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

**Controller deletions.** What goes: `_run_profile_loop`, `_profile_all_running_tasks`, `_dispatch_profiles`, `_capture_one_profile`, `_profile_thread`, the three `profile_*` config knobs, the `task_profiles` table, the attached `profiles` SQLite database, and `insert_task_profile` / `get_task_profiles`. Migrations 0005/0014/0020/0023 stay on disk as no-op chain links (collapsing them is a separate cleanup); a new `0024_drop_profiles_db.py` `DETACH`-es and `unlink`-s the file. What stays: the controller's `profile_task` RPC handler — but stripped to "resolve target, dispatch via provider, return bytes." No DB writes from the controller in any mode.

**K8s mode.** `K8sTaskProvider.profile_task` adds a finelog write to `iris.cpu_profile` on CPU success before returning bytes. No periodic loop on k8s in v1 (matches today; flagged in Open Questions for follow-up).

**Dashboard.** "Profile now" button is unchanged — keeps calling controller `profile_task`, which now silently delegates through the provider abstraction. Add a "Profile history" panel on `TaskDetail.vue` running `SELECT captured_at, attempt_id, format, trigger, length(profile_data) FROM "iris.cpu_profile" WHERE task_id = ? ORDER BY captured_at DESC LIMIT 50` through `useStatsRpc`. Clicking a row downloads `profile_data` via a second targeted SQL. Drop the "profile this controller" button on `StatusTab.vue` (no controller self-profile path remains).

**Commit ordering.** Each commit independently revertable, tests green at every step:
1. Introduce `IrisCpuProfile` schema + namespace registration in `Worker.start()`. No writers yet — table exists but is empty.
2. Worker `ProfileTask` RPC handler writes to `iris.cpu_profile` on CPU-task success; `K8sTaskProvider.profile_task` does the same. On-demand captures now persist; periodic captures still go through the controller loop into `task_profiles`. Dual-write window.
3. Add worker `_run_profile_loop`. Add "Profile history" panel on `TaskDetail.vue`. Periodic captures land in finelog.
4. Delete controller `_run_profile_loop` / helpers / config / DB helpers. Strip the controller `profile_task` RPC handler down to "dispatch via provider; return bytes." Drop the `task_profiles` table and `profiles.sqlite3` via migration `0024_drop_profiles_db.py`.
5. Document in `lib/iris/AGENTS.md` and `OPS.md`.

## Testing

- **Unit:** `run_profile_loop` (advances on a clock; skips non-running attempts; swallows per-attempt failures; stops promptly between captures); worker `ProfileTask` handler's finelog write (uses `MemoryLogNamespace`); `_log_client = None` mode skips cleanly.
- **Integration:** end-to-end on the iris dev cluster — submit a long-running task, wait one tick, query `iris.cpu_profile` via StatsService, assert ≥1 row with non-empty `profile_data`. Run on both old-controller-with-new-workers and full-cluster-restart configurations. Add a k8s-provider integration test verifying on-demand still works through `controller.profile_task → K8sTaskProvider.profile_task` and produces a finelog row, and that no periodic rows appear (since the k8s loop is intentionally not added).
- **Migration:** `0024_drop_profiles_db.py` deletes `profiles.sqlite3` exactly once; tolerates re-runs and missing files. Profile data is diagnostic, not load-bearing — no backup.
- **No regression:** `iris.task` / `iris.worker` stats tests must keep passing.

## Open Questions

- **Capture size + compression.** Production py-spy raw output on JAX workers can be hundreds of KB per capture. Should we per-row gzip in the worker before writing (5-line change), rely on parquet block compression (low gain — bytes are mostly distinct), or measure on dev first and decide? Default plan: measure first.
- **Namespace name lock-in.** `iris.cpu_profile` is CPU-only by name. If we later persist memory/threads, we either add parallel namespaces (`iris.memory_profile`, `iris.threads_profile`) or rename today to `iris.task_profile` with a `kind` discriminator. The user's framing chose `iris.cpu_profile`; flagging in case reviewers prefer the discriminator-from-day-one shape.
- **Profile interval source.** Per-worker config knob (today's plan) vs. controller-pushed via `Ping.profile_interval` (one fleet-wide value). The latter is one extra proto field; the former is simpler but drifts across worker restarts.
- **Per-task circuit breaker.** Today the loop retries forever even if py-spy fails on the same attempt every tick. Add a per-`(task_id, attempt_id)` failure counter that backs off after N consecutive failures, or accept the noise?
- **Latency-sensitive opt-out.** Should tasks declare a `no-profile` attribute (e.g., on the JobConfig)? Tightly-coupled NCCL workloads may want to.
- **K8s periodic profiling.** Should the K8sTaskProvider grow its own loop writing to finelog from the controller process, or stay on-demand-only as today?
