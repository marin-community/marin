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

Every pipeline runs on a pool — a fray job (`_run_pool_job`) hosting a
coordinator actor plus a worker actor group. There is no second code path:

- **Standing pool**: `ZephyrContext(mode=PoolMode.HOST, pool_name="ingest", ...)`
  as a `with` block; `__enter__` starts the pool, `__exit__` / `shutdown()`
  tears it down. `_ZephyrPool` is the implementation type and is not public —
  `ZephyrContext` is the whole surface.
- **One-shot pool**: any context not running on someone else's pool builds one
  sized to the plan per `execute()`, and shuts it down in a `finally`. Each
  `max_execution_retries` attempt gets a fresh pool.

`PoolMode` decides how a context treats the pool the environment offers: `AUTO`
joins it if present else runs one-shot, `INHERIT` requires it, `ISOLATED`
refuses it outright (and rejects `coordinator_endpoint` as a contradiction),
`HOST` creates it. A joining context takes the coordinator's address from
`coordinator_endpoint` or the env; `pool_name` only names a pool you host.

`start()` advertises the pool to child jobs itself, via
`_offer_pool_to_child_jobs`: it writes `ZEPHYR_COORDINATOR_ENDPOINT` into this job's declared
env (`JobInfo.env` and its serialized copy `IRIS_JOB_ENV`), which Iris then
copies into every job submitted afterwards. `shutdown()` restores the previous
value, or later jobs inherit a dead pool. The advertised value is the absolute
coordinator address, not the pool name: a name resolves relative to the caller
and is only correct one level below the host, so a grandchild would look beside
its own parent and find nothing. Only jobs submitted after `start()`
inherit it, and outside an Iris job it is a no-op.

This works because `get_job_info()` re-reads the environment unless something
populates its `ContextVar` — nothing does today. The live `JobInfo` is mutated
too, so it keeps working if that changes.

`run_pipeline` is safe to call concurrently; per-pipeline state lives in
`_PipelineExecution`, keyed by `execution_id`. A per-run `fatal_error` fails
only that pipeline, `abort()`/`_pool_error` fails the whole pool. Finished
executions are popped from the registry, so late reports for them are ignored —
their counters fold into a cumulative snapshot first, or the totals would lose
every pipeline that already ended.

Results go to `<chunk_prefix>/<execution_id>/results.pkl`, not the actor return
value, so an exception keeps its original type through a transport that cannot
carry it — which `_NON_RETRYABLE_ERRORS` detection depends on.

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
