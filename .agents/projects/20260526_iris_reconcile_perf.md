# Reconcile RPC performance @ 5000-task zephyr scale

**Date:** 2026-05-26
**Owner:** Russell Power
**Branch:** `iris-reconcile-perf`
**Bench:** `lib/iris/scripts/benchmark_reconcile.py`

## Goal

Evaluate the Reconcile RPC (see `lib/iris/docs/reconcile_rpc.md`) as the primary
worker-control plane against a **5000-task zephyr job**, with particular focus
on the controller-side scheduling + dispatch + apply loop. The Reconcile wire
collapsed StartTasks + PollTasks + StopTasks into one unary RPC per worker per
tick; this report measures what that costs as task count scales and as the
per-job `RunTaskRequest` payload grows.

## Benchmark setup

`lib/iris/scripts/benchmark_reconcile.py` synthesizes a fresh controller DB
representing one zephyr-shaped job with N tasks distributed across M workers,
then spins up a single in-process fake worker (Connect ASGI on localhost) whose
`Reconcile` handler echoes every `DesiredAttempt.run` back as a
`TASK_STATE_RUNNING` observation. Every registered worker in the DB resolves
to that one address, so the run exercises the real `WorkerProvider`
`asyncio.gather`-over-Connect path with its `parallelism=128` semaphore — the
only thing it skips is N-way socket setup.

Each scenario reports two distinct ticks:

1. **Dispatch tick.** Fresh DB; every task is `ASSIGNED`, so every
   `DesiredAttempt` carries the full inline `RunTaskRequest` (this is the
   tick that puts spec bytes on the wire).
2. **Steady-state tick.** Repeated after the dispatch tick, every task is now
   `RUNNING`, so `DesiredAttempt.run.request` is unset (the worker is expected
   to have the spec cached). This is the polling tick that runs every 250 ms
   in production.

Hardware: single workstation (this Linux box), bench process and fake worker
share an asyncio loop and CPU pool, so `RPC fanout` is a **lower bound** on
production latency.

Sweep run: `uv run python lib/iris/scripts/benchmark_reconcile.py --scale-sweep`.

## Headline numbers

| scenario                              | workers | wire/tick | compute | dispatch tick | steady tick |
|---|---:|---:|---:|---:|---:|
| 500 tasks ×  1 KB payload             |   8 |  579 KB   |  3.7 ms |  **403 ms**   |  **24 ms**  |
| 500 tasks × 100 KB payload            |   8 |  47.8 MB  | 34.1 ms |  **440 ms**   |  **27 ms**  |
| 2000 tasks ×  1 KB payload            |  32 |   2.3 MB  | 14.5 ms | **1.28 s**    | **102 ms**  |
| 2000 tasks × 100 KB payload           |  32 | 191.1 MB  | 150 ms  | **1.66 s**    |  **93 ms**  |
| 5000 tasks ×  1 KB payload (64 t/w)   |  79 |   5.7 MB  | 35.7 ms | **3.75 s**    | **298 ms**  |
| 5000 tasks × 10 KB payload (64 t/w)   |  79 |  48.6 MB  |  77 ms  | **3.85 s**    | **249 ms**  |
| **5000 tasks × 100 KB payload (64 t/w)** | **79** | **477.8 MB** | **394 ms** | **4.33 s**  | **257 ms**  |
| 5000 tasks × 100 KB payload (8 t/w)   | 625 | 477.8 MB  | 359 ms  | **5.08 s**    | **742 ms**  |
| 5000 tasks × 100 KB payload (1 t/w)   |5000 | 477.8 MB  | 389 ms  | **12.7 s**    | **4.33 s**  |

(Means over n=4 steady ticks; dispatch is n=1 by construction.)

The user asked specifically: "5000 Reconcile RPCs with 100KB each → how much
does that hurt us?" Answer at the realistic **79-worker** density (the
zephyr-default 64 tasks/worker shape):

- **Total wire bytes per dispatch tick: 478 MB.** Each per-worker
  `ReconcileRequest` carries on average 6 MB of inline spec (≈ 64 ASSIGNED
  attempts × 100 KB).
- **Dispatch tick wall time: 4.3 s.** Breaks down as:
  - apply: 3.0 s (70 %)
  - RPC fanout: 0.85 s (20 %)
  - pure-compute proto build: 0.41 s (10 %)
  - DB snapshot: 0.04 s
- **Steady-state tick: 257 ms** — right at the 250 ms polling-loop budget. So
  on a 5000-task zephyr job we're already at saturation for the steady-state
  loop with realistic payloads, without any per-tick churn.

At 5000 workers × 1 task each (extreme zephyr fan-out), even the steady tick
takes 4.3 s — 16 × over the 250 ms budget. We do not run that shape today, but
it's a useful ceiling.

## Stage-by-stage analysis

### compute — `reconcile.reconcile_workers(inputs)`

Pure-Python proto build. Cost scales as
**(# ASSIGNED rows) × (sizeof RunTaskRequest)** because
`_reconcile_worker` does `req.CopyFrom(spec)` per ASSIGNED row
(`lib/iris/src/iris/cluster/controller/reconcile.py:86`). For 5000 ASSIGNED
attempts × 100 KB template, that's ~500 MB of deep copy → 400 ms of CPU on a
single Python thread. Once tasks move to RUNNING the spec is omitted, so
steady-state compute drops to ~10 ms.

### snapshot — `_snapshot_reconcile_inputs`

DB read over `task_attempts ⨝ tasks` filtered to the worker set, plus one
`run_request_template` build per unique job (LRU-cached). At 5000 rows it's
40–70 ms, dominated by the join and Python-side worker filter. Acceptable.

### RPC fanout — `WorkerProvider.reconcile_workers`

`asyncio.gather` across all workers under a `Semaphore(parallelism=128)`.
With 79 workers, all RPCs go in flight together — fanout in the 80–150 ms
range mostly reflects pyqwest's TLS-less HTTP roundtrip × the spec payload
size (845 ms for 478 MB across 79 connections ≈ 565 MB/s, close to localhost
loopback bandwidth).

The **parallelism cap** dominates at 5000 workers: 5000 / 128 ≈ 40 batches of
~40 ms each = ~1.6 s **per tick** (steady-state row in the table). The
controller stops scheduling, polling, and pinging while these batches drain.

### apply — `apply_reconcile_result` loop

The biggest single cost, and the surprising one: in the **steady-state** tick,
every per-worker apply still pays for `bulk_get_task_detail`,
`bulk_get_attempts`, and `resolve_attempt_uids` even when **no transitions
happen** (all observations are RUNNING → RUNNING no-ops). Costs:

- 5000 tasks / 79 workers steady-state: **150 ms** apply.
- 5000 tasks / 79 workers dispatch (real ASSIGNED → RUNNING transitions):
  **3.0 s**.

The dispatch apply is doing real work — writing 5000 row updates — but per-row
cost (3000 ms / 5000 = 600 µs) is high for SQLite. Per-worker
`apply_reconcile_result` is invoked **inside** a shared transaction
(`controller.py:2399`), so there's no commit fan-out, but each invocation does
its own `bulk_get_task_detail` and `resolve_attempt_uids` lookup — N
round-trips into SQLite where one batched fetch over all workers would do.

## Suggested fixes

Ranked by expected leverage on the dispatch-tick and steady-state numbers
above. The first three target the apply path — that's where the time
actually lives.

### 1. Batch the apply path across workers (biggest win)

**Where:** `iris/cluster/controller/transitions.py:1990` (`apply_reconcile_result`)
called from `iris/cluster/controller/controller.py:2399` in
`_reconcile_worker_batch`.

**What:** Today `_reconcile_worker_batch` invokes
`transitions.apply_reconcile_result(cur, plan, result, now)` once per worker.
Each invocation does its own `bulk_get_task_detail` / `bulk_get_attempts` /
`resolve_attempt_uids` lookup
(`transitions.py:2048-2055`). Replace with a single batched apply that walks
all per-worker observations once, collects every referenced (task_id,
attempt_id, attempt_uid), runs three bulk queries, then dispatches to
`_apply_task_transitions` per worker reusing the shared maps.

**Expected impact:** the 3.0 s dispatch apply at 5000 tasks should drop into
the low-hundreds-of-ms range (the SQL itself is fast — what we're paying is
N×3 query roundtrips through SQLAlchemy). Steady-state 150 ms drops further
because the bulk reads are skipped entirely once we detect "no observations
in this worker carry a state change."

**Suggested shape:**

```python
def apply_reconcile_batch(
    self,
    cur: Tx,
    plans_and_results: list[tuple[WorkerReconcilePlan, ReconcileResult]],
    now: Timestamp,
) -> list[TxResult]:
    # 1. Gate by worker existence (one filter_existing_workers call).
    # 2. Collect ALL observations across all workers; resolve uids once.
    # 3. bulk_get_task_detail / bulk_get_attempts on the union.
    # 4. Per worker, build HeartbeatApplyRequest using the shared maps and
    #    feed each through _apply_task_transitions.
```

### 2. Skip apply when no state changes

**Where:** `iris/cluster/controller/transitions.py:2100` (`_observations_to_updates`),
or earlier in the new batched entry point above.

**What:** In `_observations_to_updates`, short-circuit when every
observation's `state` matches the current attempt's state. The steady-state
tick is the common case — RUNNING → RUNNING — and we currently do three bulk
reads per worker for nothing. After item 1 lands, this is the next ~50 ms
per tick.

**Detection without DB:** since the controller already has the
`WorkerReconcilePlan` it sent (which encodes the desired states), and the
worker echoed back observations of the same attempts, we can compare
`obs.state` against the *task state implied by the desired intent* without
touching the DB. If they match across the whole response, return early.

### 3. Cache `attempt_uid → (task_id, attempt_id)` in process

**Where:** `iris/cluster/controller/reads.py:resolve_attempt_uids` callers in
`transitions.apply_reconcile_result` and the new batched apply.

**What:** `reads.resolve_attempt_uids` is a DB index lookup on every tick. The
mapping is immutable per attempt; an in-process `LRUCache` keyed by
`attempt_uid` (sized to a few × max active attempts) eliminates the lookup
entirely once the dispatch tick has populated it. Pair with eviction on
`task_attempts` insert/delete (the writer code paths are centralized in
`writes.insert_attempt` and the cancellation pathways).

### 4. Stop deep-copying the spec on every ASSIGNED row

**Where:** `iris/cluster/controller/reconcile.py:86` in `_reconcile_worker`.

**What:** Today each ASSIGNED row does:

```python
req = job_pb2.RunTaskRequest()
req.CopyFrom(spec)                # deep-copy of 100 KB+ proto
req.task_id = wire_task_id
req.attempt_id = row.attempt_id
req.attempt_uid = row.attempt_uid
```

The deep-copy of `workdir_files` (immutable bytes) is the expensive part.
Two options, in order of preference:

- **Serialize the template once, parse per attempt.** `template_bytes =
  spec.SerializeToString()` (called once when the template is built /
  cached, not per attempt). For each attempt:

  ```python
  req = job_pb2.RunTaskRequest()
  req.MergeFromString(template_bytes)   # C++ fast path
  req.task_id = wire_task_id
  req.attempt_id = row.attempt_id
  req.attempt_uid = row.attempt_uid
  ```

  The C++ parser is ~5–10 × faster than the pure-Python `CopyFrom` walk on
  large nested messages.

- **Share `workdir_files` by reference.** Extract them out of
  `AttemptSpec.request` into a sibling field (or a `bundle_id` content-hash
  reference) that the worker resolves to its local cache. Tradeoff: worker
  has to fetch out-of-band on cache miss — but it already does this for
  Docker images, so the machinery is there.

**Expected impact:** compute drops from 400 ms to ~50 ms on the 5000 × 100 KB
dispatch.

### 5. Raise (or remove) the RPC fanout parallelism cap

**Where:** `iris/cluster/controller/worker_provider.py:120`
(`WorkerProvider.parallelism: int = 128`) and the `asyncio.Semaphore` in
`_reconcile_all_via_rpc`.

**What:** Semaphore(128) costs ~1.6 s per tick at 5000 workers (≈ 40
sequential batches). The semaphore exists to bound concurrent open sockets,
but Connect/HTTP is multiplexed and pyqwest keeps a stub-per-address pool —
there's no actual N×N socket explosion. Either raise to ~1024 statically or
scale dynamically with worker count
(`max(self.parallelism, len(plans))`). At 79 workers this is a no-op; at
5000 it removes ~1.5 s from the steady-state tick.

### 6. Compressed payloads on the wire

**Where:** `iris/cluster/controller/worker_provider.py:67` —
`RpcWorkerStubFactory.get_stub` constructs the client with
`send_compression=None`.

**What:** `IRIS_RPC_COMPRESSIONS = (Zstd(-1), Gzip)` is negotiated on the
*receive* path, but the controller's stub doesn't compress what it *sends*.
For 100 KB workdir_files of typical Python pickle / tarball content, zstd
should hit 5–10 × compression. Flip `send_compression` to the zstd codec.
Mostly relevant on cross-region links; localhost bench won't show it, but in
production CoreWeave ↔ GCP this is meaningful.

## Concrete next steps

1. Land item 1 (apply-batch refactor) as a single PR with this bench as the
   regression gate. The dispatch-tick number should drop from 4.3 s to
   roughly 1.5 s at 5000 tasks × 100 KB.
2. Add the no-op-observation fast-path (item 2) on top — small change, big
   steady-state win.
3. Spike the template-via-serialize trick (item 4a) — tiny diff, possibly
   skip if (1) already gets us under tick budget.
4. Re-run the sweep and produce a v2 of this table in the same file.

## Files

- `lib/iris/scripts/benchmark_reconcile.py` — the new harness (run with
  `--scale-sweep` to reproduce).
- `lib/iris/docs/reconcile_rpc.md` — design doc this evaluates.
- `lib/iris/src/iris/cluster/controller/reconcile.py:66` — pure compute.
- `lib/iris/src/iris/cluster/controller/worker_provider.py:271` — RPC fanout.
- `lib/iris/src/iris/cluster/controller/transitions.py:1990` — apply path
  (target of items 1–3 above).
- `lib/iris/src/iris/cluster/controller/controller.py:2390` — caller that
  invokes the per-worker apply loop.
