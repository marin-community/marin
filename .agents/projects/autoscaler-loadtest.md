# Autoscaler / Controller Load-test: Reproduce the 2026-04-18 Spike

## Motivation

On 2026-04-18 the marin controller spiked CPU and workers/users saw slow
dashboard, failed "Task not found on worker" retries, and a sustained cascade
of `Failed to create slice … timed out` warnings (~960 in 2.5h) concentrated on
v6e-preemptible groups in europe-west4-a and us-east1-d. Autoscaler scale-ups
run as blocking HTTP calls on an **unbounded** managed-thread container inside
the controller process (`Autoscaler._do_scale_up` → `group.scale_up` →
`GcpWorkerProvider._gcp.tpu_create` — googleapiclient, not subprocess).

**Stage-0 corrections to code names used below:**
- Autoscaler class is `Autoscaler` (runtime.py:69), not `AutoscalerRuntime`.
- Fake service is `InMemoryGcpService` in `gcp/fake.py`, not `FakeGcpService`.
- Thread container (`Autoscaler._threads`) is **unbounded**; this is the
  primary Stage-4 fix candidate.
- Failure-path does **not** write to `txn_log`; it does two single-row
  `scaling_groups` UPDATEs per failed scale-up. Dashboard slowness under load
  is expected to come from sqlite writer-lock contention (`db.py` RLock +
  `BEGIN IMMEDIATE`), not row volume — instrument lock-hold time, not
  `txn_log` growth.

We want a deterministic reproducer so we can:

1. Confirm the root cause is autoscaler thread-pool churn + DB/log write
   amplification driven by slow/failed TPU REST calls.
2. Exercise the system under preemption + large submission bursts against a
   realistic fleet.
3. Measure tail latency on dashboard-ish queries and worker heartbeat paths
   while the spike is active.
4. Propose and validate a fix (backoff, thread-pool cap, or offload path).

## Inputs we already have

- `/tmp/iris-marin.sqlite3` — real controller state checkpoint from
  `gs://.../controller-state/1776527658577/`. 1.5 GB, 1,157 workers, ~143k
  tasks, 202 scale groups with real `consecutive_failures` / `quota_reason`.
- `/tmp/iris-logs/*.parquet` — 4 log parquets covering 08:00–10:30 UTC on
  2026-04-18 (process + worker logs).
- `lib/iris/src/iris/cluster/providers/gcp/fake.py` — existing in-memory fake
  that already implements `tpu_create` with failure-injection. This is the
  seam we mock at.

## Plan

### Stage 0 — Scaffolding and confirmation (senior-engineer)

- Confirm the seam: `AutoscalerRuntime` takes a `WorkerProvider` / `ScalingGroup`
  that ultimately calls `GcpClient.tpu_create`. Assert that swapping in
  `fake.FakeGcpService` (with configurable latency + error injection) bypasses
  all network and mutations.
- Audit `_do_scale_up` (runtime.py:334) and `ScalingGroup.scale_up`
  (scaling_group.py:513) for DB-write side-effects under failure:
  `group.record_failure`, `group.cancel_scale_up`, action log rows, `logger.exception`.
- Decide harness location: `lib/iris/tests/loadtest/` (gated off from the normal
  test run with a pytest marker `loadtest`, so CI doesn't run it).

### Stage 1 — Harness boot: run the controller against the captured DB (senior-engineer)

- Write `lib/iris/tests/loadtest/harness.py` that:
  - Takes a path to a controller sqlite snapshot (defaults to
    `/tmp/iris-marin.sqlite3`).
  - Opens it read-write into a **copy** at `/tmp/iris-loadtest.sqlite3` so the
    original stays intact.
  - Boots the `ControllerService` / `AutoscalerRuntime` in-process against that
    DB, with:
    - A fake `WorkerProvider` (extend `fake.FakeGcpService`) whose
      `tpu_create` / `tpu_delete` can be configured per-call to return:
      success, timeout (after N seconds of blocking sleep), RESOURCE_EXHAUSTED,
      or "internal error".
    - The real scheduler and autoscaler threads started normally.
  - Exposes measurement hooks: (a) live counters for scale-up threads active,
    (b) sampled `time.monotonic()` on a proxy "dashboard" query (e.g.
    `ExecuteRawQuery("SELECT state, count(*) FROM tasks GROUP BY state")`), and
    (c) txn_log write rate.
- Smoke test: boot harness → scheduler/autoscaler run one tick → shut down
  cleanly. Validate no network, no GCS access, DB copy untouched.

### Stage 2 — Stimulus generators (senior-engineer)

Three orthogonal generators. Each can be toggled independently in the harness.

- `submit_burst(user, job_count, tpu_kind, size)` — synthesize job
  submissions matching michaelryan's extract-v2 sweep shape (many single-task
  v6e-preemptible-8 jobs). Use the real job submission RPC / service call so we
  exercise the same code path.
- `preempt_workers(group_pattern, fraction)` — pick N% of currently-registered
  workers in matching groups and simulate preemption by marking their TPUs
  terminated in the fake GCP layer, which the autoscaler must observe and
  replace.
- `bad_tpu_api(group_pattern, failure_mode, duration)` — tell the fake
  `tpu_create` to return `timed out` / `internal error` for N minutes on matching
  groups, with configurable latency (default 120s blocking, as seen in prod).

### Stage 3 — Reproduce (senior-engineer + ml-engineer)

- Scenario A (overnight burst only): run `submit_burst` × 240 jobs in 60s
  across v6e-preemptible-4/8 in europe-west4-a. No API failures. Baseline.
- Scenario B (API timeouts only): no new submissions; just activate
  `bad_tpu_api` for 60 minutes across v6e-preemptible in europe-west4-a and
  us-east1-d. Watch autoscaler behavior.
- Scenario C (combined): burst + API timeouts + 10% preemption cycle every 5
  min. This should reproduce the prod pattern.
- Metrics collected per scenario:
  - CPU time spent in the python process (RUSAGE_SELF).
  - Dashboard-query P50/P95/P99 latency over time.
  - Scale-up thread count (peak, steady), thread-pool queue depth.
  - Txn_log writes/sec, DB size growth.
  - Scheduler tick latency.

### Stage 4 — Fix candidates and validation (senior-engineer)

Given measurements, propose and test fixes. Candidates:

1. **Per-group adaptive backoff on timeouts.** Extend `record_failure` to
   tag `TIMEOUT` vs `QUOTA` vs `GENERIC` and exponentially back off
   timeouts separately. Today everything just bumps `consecutive_failures`.
2. **Cap the autoscaler thread pool** (`self._threads`). If the pool is
   unbounded, a slow TPU API can accumulate hundreds of blocking threads.
3. **Deadline on `tpu_create`.** Replace `urlopen(…, timeout=120)` equivalent
   with a shorter deadline (e.g., 20 s) + retry, so a single stalled scale-up
   doesn't pin a thread for 2 minutes.
4. **Consolidate per-failure DB writes.** `record_failure` + `log_action` +
   `logger.exception` every failure is heavy when failing 1/sec.
5. **Queue-and-drain pattern for bursts.** Separate "intent to scale" from
   actual scale-up so the scheduler never blocks on a provisioning thread pool.

For each fix: re-run Scenario C, compare metrics, and keep ones that
demonstrably flatten the CPU curve and preserve dashboard latency P95 under
load.

### Stage 5 — Final report (senior-engineer)

Deliverables:

- `logs/autoscaler-loadtest/summary.md` — coordinator execution log.
- `logs/autoscaler-loadtest/report.md` — final report: hypothesis, stimuli
  used, measurements, which fix(es) worked, what the diff would look like,
  and any residual open questions (e.g., OPS.md Known Bug #2 heartbeat/gcloud
  delete path — only relevant in prod, not the loadtest).
- Load-test code committed under `lib/iris/tests/loadtest/` behind the
  `loadtest` pytest marker so it can be run on demand but does not slow CI.

## Constraints

- **No network**, **no GCS**, **no mutations to the real controller**.
- Use `uv run` per AGENTS.md; iris-specific code follows `lib/iris/AGENTS.md`
  (Connect/RPC, `rigging.timing`, `ThreadPoolExecutor` with hard timeouts, no
  `asyncio`, no `TYPE_CHECKING`).
- All experiments operate on the **copy** of the sqlite snapshot, not the
  original.
- Mark the harness `-m loadtest` in pytest so it's opt-in.

## Non-goals

- Reproducing OPS.md Known Bug #2 (heartbeat thread stuck on gcloud subprocess
  `tpu-vm delete`). That's a separate deletion-path issue; we call it out in
  the report but don't try to repro here.
- Reproducing the bundle-fetch-timeout storm from 2026-04-16. That's a bundle
  server issue, also separate.

## Progress log

`logs/autoscaler-loadtest/summary.md` — updated after each stage completes.
Per-stage detail lives in `logs/autoscaler-loadtest/stage-N-*.md`.

### Stage 6 — Close the magnitude gap (prod-faithful load)

Stage 3 harness peaked at ~24 concurrent scale-up threads; prod hit ~96 immediately. Goals:

1. **Synthetic workers in the fake.** On each successful `tpu_create`, `LoadtestGcpService` spawns a `SyntheticWorker` thread that registers via the real `ControllerService` RPC, emits heartbeats at prod cadence, polls for tasks, and reports completion. Lifecycle bound to matching `tpu_delete` or injected preemption. Goal: exercise heartbeat ingest + task-attempt writes + scheduler match load (the unmeasured CPU channel).
2. **Zone-agnostic job submission.** `submit_burst` drops zone/region constraints so jobs route across all eligible groups.
3. **Synthetic dashboard/RPC probe load.** 5–20 concurrent Connect clients at prod QPS issuing `list_jobs`, `get_scheduler_state`, task-table reads. Measure dashboard P95 under realistic read pressure.
4. **Diagnose the rate-limit gap.** Prod hits 96 threads immediately; harness needs minutes to reach 24. Re-read `ScalingGroup.scale_up` and the per-tick attempt budget — identify whether (a) parallel groups multiply the effective cap, (b) budget accumulates across ticks during 120s blocking calls, or (c) a prod path bypasses the budget. Fix the harness so it triggers immediately when all zones fail.
5. Keep real logger (don't silence `logger.exception`). Drop "log cost" as a load theory.

Scenario D (new): bad_tpu_api across **all** v6e-preemptible groups in **all** zones simultaneously + zone-agnostic 500-job burst + 10% preemption cycle + probe clients. Should trigger ≥80 concurrent scale-up threads within 60 s. Metrics collected as Stage 3.

Log: `logs/autoscaler-loadtest/stage-6-magnitude.md`. Summary update: `logs/autoscaler-loadtest/summary.md`.

#### 6.1a Synthetic-worker refinement (user-directed)

The `SyntheticWorker` must be a **real RPC server on a real localhost port** implementing the actual `WorkerService` interface the controller calls into. No in-process RPC shortcut.

- Each synthetic worker binds a free localhost port and runs the Connect/RPC `WorkerService` handler.
- It registers with the controller with its real `host:port` so the controller's worker-polling loop hits it over real sockets.
- Preempted workers: server stops responding (socket closed / handler returns errors / process-stopped equivalent) — controller's health check must observe the failure through the real RPC path.
- Task state progression on the synthetic worker: `ASSIGNED → BUILDING → RUNNING → COMPLETED` with realistic delays (~60 s between transitions; parameterizable for test speed). Reports state transitions back via the normal RPC surface.
- Many workers × real ports: use ephemeral ports (`bind(..., 0)`) and register the assigned port.

This exercises the **real** controller→worker RPC path, including connection pooling, retries, and timeouts — the suspected prod CPU channel.
