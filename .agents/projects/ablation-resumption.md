# Controller-toggle ablation — resumption notes

## What this is

We're quantifying a set of env-gated controller tweaks by running the
`prod-scale` loadtest scenario with toggles applied cumulatively. The
entry point is `iris-loadtest ablation`
(`lib/iris/src/iris/loadtest/ablation.py`), which spawns one
`iris-loadtest scenario prod-scale` subprocess per step, cleans up stray
child processes + `/tmp/loadtest-*` scratch dirs between steps, and writes
a combined `REPORT.md` with writer-lock / dashboard / per-RPC latency
tables and `Slow *` log counts.

Invocation on a low-variance box:

```
uv run iris-loadtest ablation \
  --preload-workers 100 --duration 300 \
  --burst-jobs 100 --cpu-jobs 10 --cpu-tasks-per-job 100 \
  --step-timeout-seconds 540 \
  --output-dir logs/ablation-$(date +%s)
```

Defaults match that invocation; split-heartbeat is held on throughout so
`--heartbeat-inmemory` (RAM HBM) is an isolable toggle. The step list
lives in `ablation.DEFAULT_STEPS`; edit there to re-order, swap, or add
toggles without touching the CLI.

## Toggles under test

Applied cumulatively. Steps 0-3 run against the legacy heartbeat path so
each tuning toggle's effect is visible on the path that's currently in
production; step 4 is the split-heartbeat flip; step 5 layers RAM HBM on
top (RAM HBM requires split-heartbeat — it's only read in the ping loop
and `_reap_stale_workers`, so reversing the order silently no-ops it).

0. baseline — legacy heartbeat, no tuning toggles
1. +`--sqlite-tuning` → `IRIS_DB_MMAP_BYTES=268435456 IRIS_DB_CACHE_KB=1048576`
2. +`--controller-yield` → `IRIS_CONTROLLER_YIELD=1`
3. +`--job-status-cache` → `IRIS_JOB_STATUS_CACHE_TTL_MS=1000`
4. flip to `--use-split-heartbeat` (same tuning toggles stay on)
5. +`--heartbeat-inmemory` → `IRIS_HEARTBEAT_INMEMORY=1`

## Metrics captured

`ScenarioMetrics` (`lib/iris/src/iris/loadtest/metrics.py`) records:

* `writer_lock_hold_ms` — samples from an `_InstrumentedLock` wrapper
  swapped onto `ControllerDB._lock`.
* `dashboard_query_ms` — 2 Hz read-only probe (`SELECT state, count(*) FROM tasks GROUP BY state`).
* `rpc_ms` — per-method batch wall-clock reservoirs for
  `WorkerProvider.{sync, ping_workers, start_tasks, stop_tasks, poll_workers}`.
  Installed by monkey-patching `harness._task_provider` in
  `ScenarioMetrics.start()`; restored in `stop()`. Batch-level granularity:
  one sample per call (fan-out across workers is aggregated).
* 1 Hz `_Sample` time series (`active_scale_up_threads`, `create_attempts`,
  `create_failures`, `rss_bytes`).

The ablation writer additionally greps each step's log for
`Slow heartbeat / provider / scheduling / buffer_assignments /
building_counts / dispatch_assignments_direct` events, so pathological
RPCs that escape the bounded reservoir still show up in the report.

## Prior run (on local laptop, 8 cores) — flagged for re-measurement

`logs/ablation-1776627554/REPORT.md` has the numbers and per-toggle reads.
Summary of what we already believe vs what needs re-verification on a
low-variance machine:

* **Clear** (holds up under the Slow-log view): split-heartbeat eliminates
  the monolithic heartbeat RPC path. `Slow heartbeat` count drops from
  ~1 500 (step 0, legacy path) to 0 (step 5). Writer-lock hold and dashboard
  P50 also improve with split-HB.
* **Needs re-verification on low-variance box**:
  * `sqlite-tuning` — 3× RSS cost is robust; the P50/P95 regression at
    100 workers may be a laptop-OS page-cache artifact.
  * `job-status-cache` — regression on default probe mix; re-run with a
    repeat-heavy probe (single `get_job_status` job_id at ≥ 10 Hz) to
    see whether the cache ever wins.
  * `heartbeat-inmemory` — in the prior run it was toggled only at the
    last step while the heartbeat path was *also* changing, so its
    isolated effect is unknown. With the updated ablation (split-HB on
    throughout) step 4 is the first clean HBM measurement.
* **Benchmark limitation (now fixed)**: prior metrics JSON only captured
  writer-lock hold + dashboard probe; heartbeat/ping/StartTasks/StopTasks
  batch latencies were invisible. That's why the prior report misread
  step 5 as "mixed evidence" — the real win was a 1 500 → 12 drop in
  `Slow heartbeat` events, which didn't show up in the recorded metrics.
  `rpc_ms` in `ScenarioMetrics` now exposes this directly.

## Known sharp edges

* Synthetic workers are OS subprocesses
  (`iris.loadtest.synthetic_worker_main`), not threads; the controller is
  still in-process. The loadtest box wants enough cores for
  `preload_workers + 1` concurrent Pythons.
* `/tmp/iris-marin.sqlite3` is the captured controller snapshot. Missing
  on a fresh box — copy from the originating host.
* `pkill -f` is how the runner cleans up between steps. If anything else
  on the box shares `iris-loadtest` / `synthetic_worker` /
  `loadtest.harness` in its command line, the runner will kill it too.
  Ideally run on a dedicated machine.
* Fake GCP service (`lib/iris/src/iris/cluster/providers/gcp/fake.py`)
  was previously racy under concurrent autoscaler threads; now guarded by
  a single RLock. Regression test: look for
  `RuntimeError: dictionary changed size during iteration` in step logs.

## Open questions to drive the next run

1. Does `sqlite-tuning` show a net win once page-cache pressure is not
   shared with 100 other local processes?
2. Does RAM HBM show a meaningful writer-lock-hold reduction when
   isolated? Hypothesis: yes, because each successful ping no longer
   takes the writer lock for `update_worker_pings`.
3. Is `controller-yield` measurably helpful under lower-variance
   scheduling, or is it lost in noise?
4. Job-status cache with a repeat-heavy probe mix — does it ever win?

Re-run the ablation on the low-variance box, read `REPORT.md`, and answer
those four.
