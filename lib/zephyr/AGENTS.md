# Zephyr Agent Notes

Lazy dataset processing library. Start with the shared instructions in `/AGENTS.md`; only Zephyr-specific conventions are below.

## Key Docs

- `README.md` — overview, API reference, quick start
- `OPS.md` — debugging pipelines: dashboard, observability, profiling, diagnostic patterns (also used by skills: `debug-infra`, `babysit-zephyr`)
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
- `src/zephyr/counters.py` — `increment` / `get_counters` per-worker counter API (`CounterSnapshot` lives in `execution.py`)

## Execution Model

Actor-based, pull-based task distribution. Workers are persistent across stages.

```
ZephyrContext → ZephyrCoordinator (fray actor) → ZephyrWorker actors (fray actor_group)
```

### Data flow between stages

Stages pass data via **filesystem-backed chunk references** (`PickleDiskChunk`), not in-memory. Each stage reads chunks from storage, processes them, writes results back. Workers stream one chunk at a time to minimize memory.

### Critical `.result()` calls

These worker→coordinator RPCs **must** block (`.result()`). Removing them causes race conditions:

1. `coordinator.report_result.remote().result()` — must complete before next `pull_task`, otherwise `_in_flight` tracking breaks (assertion at line ~584)
2. `coordinator.report_error.remote().result()` — same ordering constraint as `report_result`
3. `coordinator.heartbeat.remote().result()` — prevents congesting the coordinator RPC pipe with fire-and-forget heartbeats
4. `coordinator.register_worker.remote().result()` — worker must be registered before polling starts

Shared data is uploaded to filesystem by `ZephyrContext._upload_shared_data()` before pipeline execution; workers read it lazily via `get_shared(name)`. Chunk config is passed inline with each `pull_task` response (not via a separate RPC).

### Error classification

- **Transient** (connection errors, preemption) → task re-queued
- **Permanent** (user code bugs, invalid data) → `fatal_error` set, exception raised

## Notes

### MacOS

Ray 2.53 enables a `uv run` runtime_env hook by default. When tests run via `uv run pytest`, this can start workers with a different Python version or fail with psutil errors in sandboxed environments. Disable it for tests. See https://github.com/ray-project/ray/issues/59639.
