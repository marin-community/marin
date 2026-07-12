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

### Dedicated vs shared

- **Dedicated** (default): each `ctx.execute()` submits a fresh coordinator job
  (`_run_coordinator_job`) that runs one pipeline and tears everything down.
- **Shared**: `with ctx as endpoint:` (or `ctx.start()` directly) submits a
  long-lived coordinator job (`_run_shared_coordinator_job`) with a fixed
  worker pool and publishes the coordinator's actor endpoint; `__exit__` /
  `ctx.shutdown()` tears it down. `__enter__` starting the pool is the explicit
  opt-in to shared mode — plain `ZephyrContext(...).execute()` without a `with`
  or endpoint stays dedicated. Drivers connect with
  `ZephyrContext(coordinator_endpoint=...)` — or by setting
  `ZEPHYR_COORDINATOR_ENDPOINT` in the driver job's env (read as a fallback in
  `__post_init__`); their `execute()` calls submit pipelines via `run_pipeline`
  RPC, which the coordinator runs **concurrently**.
  All per-pipeline coordinator state lives in `_PipelineExecution` (registry
  `_executions[execution_id]`); the worker pool (`_worker_states`,
  `_worker_counters`, …) is shared. `pull_task` round-robins active executions
  and dispatches by `ZephyrTaskResources.can_fit`. A per-run `fatal_error`
  fails only that pipeline; `abort()`/`_pool_error` fails the whole pool.
  Finished executions are popped from the registry (shared mode) so late
  worker reports for them are logged and ignored.

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
