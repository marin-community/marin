# Zephyr Agent Notes

Lazy dataset processing library. Start with the shared instructions in `/AGENTS.md`; only Zephyr-specific conventions are below.

## Key Docs

- `README.md` — overview, API reference, quick start
- `OPS.md` — debugging pipelines: dashboard, observability, profiling, diagnostic patterns (also used by skills: `debug`, `babysit-zephyr`)
- Echo — durable incident and debugging records; use `write-ops-log` after an
  infrastructure investigation and link the canonical Echo URL
- Archived: `.agents/projects/20260130_fray_lite_design.md` — Fray backend design (implemented; read `lib/fray/src/fray/` instead)

## Source Layout

- `src/zephyr/dataset.py` — `Dataset` class, `ShardInfo`, all transformation operations including `group_by`, `deduplicate`, `sorted_merge_join`
- `src/zephyr/execution.py` — `ZephyrContext`, `ZephyrCoordinator`, `ZephyrWorker`, `CounterSnapshot`, execution control flow
- `src/zephyr/plan.py` — `compute_plan`, `PhysicalPlan`, operation fusion
- `src/zephyr/readers.py` — `load_jsonl`, `load_parquet`, `load_vortex`, `InputFileSpec`
- `src/zephyr/writers.py` — `write_jsonl_file`, `write_parquet_file`, `write_vortex_file`, Levanter cache writer
- `src/zephyr/shuffle.py` — scatter pipeline internals (`ScatterFileIterator`, `ScatterReader`, hash-routing, combiner, zstd-chunk file format with byte-range sidecar)
- `src/zephyr/expr.py` — `Expr`, `col`, `lit` for filter expressions
- `src/zephyr/external_sort.py` — `external_sort_merge` k-way merge of sorted runs
- `src/zephyr/counters.py` — `ScopedCounters`, `pipeline`, `stage()`, `current_stage()` scoped counter API (`CounterSnapshot` lives in `worker_context.py`)

## Execution Model

Actor-based, pull-based task distribution. Workers are persistent across stages.

```
ZephyrContext → ZephyrCoordinator (fray actor) → ZephyrWorker actors (fray actor_group)
```

### One pool, two lifetimes

Every pipeline runs on a `ZephyrPool` — a fray job (`_run_pool_job`) hosting a
coordinator actor plus a worker actor group. There is no second code path:

- **Standing pool**: `with ZephyrPool(...) as endpoint:` (or `pool.start()`)
  publishes the coordinator's actor endpoint and serves many pipelines until
  `__exit__` / `pool.shutdown()`. Drivers connect with
  `ZephyrContext(coordinator_endpoint=...)`, or by setting
  `ZEPHYR_COORDINATOR_ENDPOINT` in the driver job's env (read as a fallback in
  `ZephyrContext.__post_init__`).
- **One-shot pool**: `ZephyrContext.execute()` with no endpoint builds a pool
  sized to the plan, runs one pipeline on it, and shuts it down in a `finally`.
  Each `max_execution_retries` attempt gets a fresh pool.

`run_pipeline` is safe to call concurrently. All per-pipeline coordinator state
lives in `_PipelineExecution` (registry `_executions[execution_id]`); the worker
pool (`_worker_states`, `_worker_counters`, …) is shared across them.
`pull_task` round-robins active executions and dispatches by
`ZephyrTaskResources.can_fit`. A per-run `fatal_error` fails only that pipeline;
`abort()`/`_pool_error` fails the whole pool. Finished executions are popped
from the registry — their counters are folded into one retained snapshot first —
so late worker reports for them are logged and ignored.

Results travel through storage, not the actor return value: `run_pipeline`
persists its payload (or the `_ensure_picklable_exception`-normalized error) to
`<chunk_prefix>/<execution_id>/results.pkl`, and the driver reads that. This
keeps an exception's original type even when the transport cannot carry it,
which `_NON_RETRYABLE_ERRORS` detection depends on.

### Releasing workers (`drain_idle_workers`)

A one-shot pool sets `drain_idle_workers`; a standing pool does not. When set,
`pull_task` hands `SHUTDOWN` to a worker that finds no dispatchable task, but
only once `_worker_is_releasable_locked` agrees: the registry is non-empty,
every active execution is on its last task-producing stage with an empty queue,
and this worker holds nothing in flight that could be requeued onto it. That
shrinks the pool through the last stage's tail while stragglers finish.

The non-empty check is load-bearing — workers boot before the driver submits, so
`all()` over an empty registry is vacuously true and would retire the whole pool
before its first task is queued. For the same reason `_check_worker_group` keys
off outstanding shards: with workers releasing themselves, a terminal worker job
is only a crash while shards remain.

### Data flow between stages

Stages pass data via **filesystem-backed chunk references** (`PickleDiskChunk`), not in-memory. Each stage reads chunks from storage, processes them, writes results back. Workers stream one chunk at a time to minimize memory.

### Critical `.result()` calls

These worker→coordinator RPCs **must** block (`.result()`). Removing them causes race conditions:

1. `coordinator.report_result.remote(worker_id, execution_id, ...).result()` — must complete before next `pull_task`, otherwise per-execution `in_flight` tracking breaks (assertion in `_assert_in_flight_consistent`). The `execution_id` routes the report to the right `_PipelineExecution`.
2. `coordinator.report_error.remote(worker_id, execution_id, ...).result()` — same ordering constraint as `report_result`
3. `coordinator.heartbeat.remote().result()` — prevents congesting the coordinator RPC pipe with fire-and-forget heartbeats
4. `coordinator.register_worker.remote().result()` — worker must be registered before polling starts

Shared data is uploaded to filesystem by `ZephyrContext._upload_shared_data()` before pipeline execution; workers read it lazily via `get_shared(name)`. Chunk config is passed inline with each `pull_task` response (not via a separate RPC).

### Error classification

- **Transient** (connection errors, preemption) → task re-queued
- **Permanent** (user code bugs, invalid data) → `fatal_error` set, exception raised

## Notes

### Backends

`ZephyrContext` uses `fray.current_client()`: Iris is auto-detected inside an Iris job, and `LocalClient` is the fallback for local tests and scripts.
