---
date: 2026-06-16
system: iris
severity: degraded
resolution: fixed
pr: pending
issue: weaver #202
---

# Iris control loop stalls ~60s/pass: autoscaler refresh() serializes blocking GCP describes

## TL;DR

- **Symptom:** a full control-loop pass (schedule → reconcile → autoscale) on the
  `marin` controller takes up to ~60s, so worker heartbeat/reconcile state goes
  stale and the autoscaler logs confusing demand against data that is tens of
  seconds old.
- **Root cause:** the control loop is a *single thread* (`single_control_tick=True`)
  that runs schedule → reconcile → autoscale **inline**. The autoscale phase calls
  `Autoscaler.refresh()`, which walks **every non-READY slice serially** and calls
  `handle.describe()` on each — a blocking GCP round-trip. For a reserved
  (queued-resource) slice `describe()` is **two** serial calls (`tpu_describe`
  miss → `queued_resource_describe`). On this cluster there were **64 reserved
  queued resources stuck `WAITING_FOR_RESOURCES`**, so one `refresh()` serializes
  ~130–140 blocking GCP calls. At GCP API latency that is tens of seconds, and it
  blocks reconcile the entire time.
- **Evidence:** live py-spy thread dumps of the `control-loop` thread caught it
  in `refresh → describe → tpu_describe`/`queued_resource_describe` in 2 of 5
  samples; `cluster status` showed `tpu_v4-reserved_8-us-central2-b` with **66
  booting**; `gcloud` confirmed **64 `WAITING_FOR_RESOURCES`** queued resources;
  controller logs showed concurrent `reconcile_rpc_failed ... Request timed out`
  (the stale-state symptom).
- **Fix (this PR):** parallelize the `refresh()` describes over a bounded thread
  pool (`_REFRESH_DESCRIBE_MAX_WORKERS = 64`) using the existing `_run_io_batch`
  find-then-fold pattern (`probe_health()` and `execute()` already do this). State
  folding stays single-threaded, so scale-group mutation is unchanged and
  race-free. This turns ~140 serial round-trips into ~3 bounded waves
  (≈seconds).
- **Secondary fix (this PR):** bump the reconcile fan-out semaphore
  `RECONCILE_FANOUT_PARALLELISM` 128 → 512 so the ~350-worker fleet reconciles in
  one wave; a single slow/timing-out worker then costs one 10s timeout window for
  the round instead of one-per-wave.
- **Not in this PR (follow-up):** the ~63 long-lived per-slice `bootstrap-*`
  threads. They are a *secondary* factor (redundant cloud polling + GCP API load
  + thread churn — ~10k threads created over the controller's life), not the
  direct cause. Removing them is an architectural change tracked separately.

## Original problem report

> It's taking up to ~60 seconds for [the marin controller] to complete a full
> reconcile pass. This leads to stale information on the worker & confusing
> autoscaler messages. Why is the loop taking so long?
>
> Open questions:
> * I thought we got rid of long lived threads in favor of polling TPU state, but
>   I see lots of threads like `bootstrap-marin-tpu-v4-preemptible-8-...`
>   (`_run_tpu_bootstrap`) — is that a factor?
> * what's the timeout/parallelism of the reconcile loop, the performance, how
>   can we make it faster (control-loop thread seen in `_fan_out` →
>   `reconcile`)?
> * what's a reasonable value for the parallelism semaphore, does it make sense
>   to go up to 1024?

## Architecture recap

`single_control_tick=True` (the production default,
`controller.py:183`) runs ONE driver thread, `_run_control_loop` →
`_control_tick` (`controller.py:814`, `:842`), executing the phases inline:

```
schedule (pure, in-memory)         every scheduler_min_interval / on wake
  -> reconcile (fan-out RPC, I/O)  every poll_interval (1.0s)
  -> autoscale (cloud I/O)         every autoscaler_evaluation_interval (10s)
  -> one end-of-tick write txn
```

The legacy separate `_run_autoscaler_loop` thread (`controller.py:785`) only
spawns when `single_control_tick=False`; in production it is dead. So **any time
the autoscale phase blocks, reconcile cannot run** — the two share the thread.

Relevant cadences/limits (marin config + code defaults):

| Knob | Value | Source |
|------|-------|--------|
| `poll_interval` (reconcile cadence) | 1.0s | `controller.py:193` |
| `autoscaler.evaluation_interval` | 10s | `config/marin.yaml:17` |
| `RECONCILE_FANOUT_PARALLELISM` | 128 → **512** | `rpc/backend.py:58` |
| `DEFAULT_WORKER_RPC_TIMEOUT` | 10s | `rpc/backend.py:55` |
| `_HEALTH_PROBE_MAX_WORKERS` | 64 | `autoscaler/runtime.py:83` |
| `_RPC_HANDLER_THREADS` | 1024 | `controller.py:108` |

## Investigation path

1. **`cluster status`** — controller healthy, git `b80b973b72`, **351/351 workers
   healthy**. Autoscaler table showed `tpu_v4-reserved_8-us-central2-b`:
   **66 booting / 102 ready / demand 256**. So the fleet was fine but a reserved
   group had a large non-ready backlog.

2. **`iris process profile threads`** (py-spy dump, repeated). The `control-loop`
   thread (Thread 45) across 5 samples:
   - **2/5** blocked in autoscale `refresh()`:
     ```
     read (ssl.py) ... get (httpx) -> _tpu_get (service.py:635)
     tpu_describe (service.py:641) -> _describe_cloud (handles.py:314)
     describe (handles.py:298) -> refresh (autoscaler/runtime.py:519)
     autoscale (rpc/backend.py:226) -> _control_tick (controller.py:909)
     ```
     and (queued-resource variant):
     ```
     _blocking (grpc/_channel.py) ... get_queued_resource (tpu client)
     queued_resource_describe (service.py:730) -> _describe_queued_resource (handles.py:368)
     _describe_cloud (handles.py:317) -> describe -> refresh (runtime.py:519)
     ```
   - **1/5** in autoscale `update()` → `_run_io_batch` (already parallel).
   - **2/5** in reconcile `_fan_out` (`rpc/backend.py:85`).

   The `refresh → describe` frames are the serial blocking I/O. (A CPU profile is
   uninformative here — the thread is GIL-released in a socket read, so the thread
   *dump* is the right tool.)

3. **Thread census** (full dump): **1100 live threads** = **1024 `rpc-handler`**
   (the RPC server executor pool) + **63 `bootstrap-*`** + control/housekeeping.
   Highest thread id ≈ **10,869** → ~10k threads created over the controller's
   life (heavy bootstrap-thread churn). The 63 live bootstrap threads are mostly
   `v4-reserved-8` (52 of 63).

4. **gcloud + logs** — `gcloud compute tpus queued-resources list
   --zone=us-central2-b`: **106 ACTIVE, 64 WAITING_FOR_RESOURCES**. Controller
   logs were a steady stream of `Queued resource ... is WAITING_FOR_RESOURCES,
   waiting...` (emitted by the bootstrap threads' `_wait_for_queued_resource_activation`)
   interleaved with `reconcile_rpc_failed ... Request timed out`.

## Root cause

`Autoscaler.refresh()` (`autoscaler/runtime.py`) before the fix:

```python
for group in self._groups.values():
    for slice_id, handle in group.non_ready_slice_handles():
        status = handle.describe()   # <-- blocking GCP round-trip, SERIAL
        ... fold status into group state ...
```

`non_ready_slice_handles()` returns every BOOTING/INITIALIZING slice. With 64
reserved queued resources stuck in `WAITING_FOR_RESOURCES` (+ a handful of other
booting slices), the loop issues ~140 serial blocking GCP calls per pass
(reserved slices cost two: `tpu_describe` returns None → `queued_resource_describe`).
Because autoscale is inline on the single control thread, reconcile is starved
for the whole duration — exactly the ~60s stalls and stale worker state reported.

This was a latent cost that only became visible once the reserved backlog grew:
at 2–3 non-ready slices the serial loop is sub-second; at 64+ it is a minute.

## The fix

### 1. Parallelize `refresh()` describes (primary)

Restructure `refresh()` into the same **find → fan-out I/O → fold serially**
shape `probe_health()` already uses:

- **Phase 1:** snapshot all `(group, slice_id, handle)` non-READY targets.
- **Phase 2:** `_run_io_batch(targets, _safe_describe, max_workers=64,
  thread_name_prefix="slice-describe")` — pure I/O on a bounded, joined pool that
  touches no autoscaler state, so it is race-free regardless of width.
  `_safe_describe` returns `SliceStatus | None` (None on exception, logged,
  retried next tick — same as the old per-iteration `try/except: continue`).
- **Phase 3:** fold the results into group state **serially** (unchanged logic),
  so all scale-group mutation stays single-threaded.

~140 serial round-trips → ~⌈140/64⌉ ≈ 3 bounded waves (seconds). 64 matches the
existing `_HEALTH_PROBE_MAX_WORKERS` precedent and is well within GCP TPU read
quota.

### 2. Reconcile fan-out width 128 → 512 (secondary)

A reconcile round costs `~RPC_TIMEOUT * ceil(num_workers / parallelism)`. At 128
with 351 workers that is 3 waves; a straggler/timeout (we observed several
`Request timed out`) in each wave adds another 10s. Setting the width ≥ fleet
size makes the fleet reconcile in one wave, so a straggler costs one timeout
window for the whole round, not one per wave. The RPCs are async coroutines over
per-worker pyqwest pools, so real concurrency is still bounded by the worker
count — a wider semaphore is cheap.

## Answers to the open questions

**Q: Are the long-lived `bootstrap-*` threads a factor?**
Secondary, not primary. Each non-ready slice still spawns a daemon thread
(`_spawn_bootstrap_thread` → `_run_tpu_bootstrap`) that polls cloud state to
drive create→READY. With 64 reserved slices queued they pile up (63 live, ~10k
created over the controller's life). They are mostly sleeping (GIL released on
I/O) so they do not directly burn the control loop, but they (a) **redundantly
poll the same cloud state** that `refresh()` also polls, doubling GCP load, and
(b) add thread/memory churn. The intended direction — *let the autoscaler's
polling be the lifecycle driver and drop the per-slice threads* — is the right
cleanup but is a larger, riskier change. Filed as a follow-up; not bundled here.
Note: the `refresh()` parallelization narrows, but does not remove, the
double-poll, so the follow-up still matters.

**Q: What is the reconcile loop's timeout/parallelism and how to make it faster?**
Per-worker RPC deadline 10s (`DEFAULT_WORKER_RPC_TIMEOUT`), fan-out width 128
(now 512). The fan-out itself was *not* the dominant cost (it appeared in 2/5
samples and completes in seconds when workers are responsive); the autoscale
`refresh()` was. Widening the semaphore removes the per-wave straggler tax; the
big win is the `refresh()` parallelization.

**Q: Is 1024 a reasonable semaphore value?**
It is *safe* (asyncio handles thousands of concurrent coroutines; concurrency is
capped by worker count anyway), but the meaningful threshold is "≥ fleet size."
512 already makes the current ~350-worker fleet a single wave with headroom.
Going to 1024 buys nothing today and only matters once the fleet grows past 512;
bump it then. Distinct from `_RPC_HANDLER_THREADS = 1024` (the inbound RPC server
pool), which is a separate knob.

## Verification

- `uv run --group dev pytest lib/iris/tests/cluster/controller/test_autoscaler.py
  test_autoscaler_integration.py -m "not requires_cluster"` → **90 passed**.
- `ruff@0.14.3 check` / `format --check` clean on the changed file.
- Behaviour preserved: describe failures still fold as "skip, retry next tick";
  READY/FAILED/UNKNOWN transitions and `scale_down_if_idle` logic unchanged;
  ordering preserved via `zip(targets, statuses, strict=True)`.

## Follow-ups

- Retire the per-slice `bootstrap-*` threads in favor of autoscaler-driven
  polling (the user's stated intent), eliminating the redundant cloud poll and
  the thread churn.
- Consider revisiting `_RPC_HANDLER_THREADS = 1024` (1024 idle threads × stack ≈
  real memory on `e2-highmem-4`) — separate from this issue.
- Consider a per-phase tick-duration log/metric so a slow phase is visible
  without a live thread dump.
