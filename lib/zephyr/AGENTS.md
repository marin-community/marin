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
- `src/zephyr/execution.py` — `ZephyrContext`, `PoolMode`, the public entry point
- `src/zephyr/pool.py` — the fray job that hosts a coordinator and its workers
- `src/zephyr/coordinator.py` — `ZephyrCoordinator`, per-pipeline state, the pull protocol
- `src/zephyr/worker.py` — `ZephyrWorker` poll loop and task execution
- `src/zephyr/plan.py` — `compute_plan`, `PhysicalPlan`, operation fusion
- `src/zephyr/readers.py` — `load_jsonl`, `load_parquet`, `load_vortex`, `InputFileSpec`
- `src/zephyr/writers.py` — `write_jsonl_file`, `write_parquet_file`, `write_vortex_file`, Levanter cache writer
- `src/zephyr/shuffle.py` — scatter pipeline internals (`ScatterFileIterator`, `ScatterReader`, hash-routing, combiner, zstd-chunk file format with byte-range sidecar)
- `src/zephyr/expr.py` — `Expr`, `col`, `lit` for filter expressions
- `src/zephyr/external_sort.py` — `external_sort_merge` k-way merge of sorted runs
- `src/zephyr/counters.py` — `ScopedCounters`, `pipeline`, `stage()`, `current_stage()` scoped counter API (`CounterSnapshot` lives in `worker_context.py`)

## Execution Model

Actor-based, pull-based task distribution. Workers are persistent across stages,
and they stream one chunk at a time. Stages pass data as filesystem-backed chunk
references, not in memory.

```
ZephyrContext → ZephyrCoordinator (fray actor) → ZephyrWorker actors (fray actor_group)
```

Every pipeline runs on a pool: a fray job with one coordinator actor and a group
of worker actors. There is one code path. A pool is either standing (a host owns
it and many pipelines share it) or one-shot (built for a single `execute()` and
torn down after). `PoolMode` decides which one a context gets. `README.md`
documents the modes and how a host advertises its pool.

Read `execution.py` for the rest — the notes below are the parts that are easy
to break.

## Gotchas

- The worker→coordinator RPCs `report_result`, `report_error`, `heartbeat`, and
  `register_worker` must block on `.result()`. They are ordering constraints,
  not slow calls. Fire-and-forget versions of them cause race conditions.
- Classify errors correctly: transient errors (connection loss, preemption)
  re-queue the task, and permanent errors (bugs in user code, invalid data) fail
  the pipeline. A misclassified permanent error retries until the budget is
  gone.
- A pipeline result travels through storage, not through the actor return value,
  because an exception must keep its original type for retry classification.
- `ZephyrContext` uses `fray.current_client()`: Iris is auto-detected inside an
  Iris job, and `LocalClient` is the fallback for local tests and scripts.
