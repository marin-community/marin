# `marin.inference.distributed` — design

This document explains what the library is, what it's built out of, and the
key decisions that shaped it. For usage, see [README.md](README.md).

## What this is

A multi-region distributed inference library targeting vLLM-on-TPU. It
wraps Iris (cluster scheduler), Zephyr (per-region task orchestration), and
vLLM (the inference engine) behind a single stateless entry point so callers
don't have to reason about per-region job submission, worker scheduling,
XLA caching, or output aggregation.

It's intended for offline / batch inference patterns — large-scale
synthetic-data generation, eval-set scoring, dataset filtering. Not online
serving; not training. See "Non-goals" at the end.

## Architecture

Five layers, each with a single responsibility:

```
caller process
  └── api.inference()               public entry point; normalizes input
        └── meta_coordinator         submits one Fray job per region, waits
              └── regional_job       one CPU job per region; pre-flight + ctx
                    └── pipeline     Zephyr Dataset graph; per-region rotation
                          └── vllm_worker   actual vLLM compile + inference on TPU
```

| Module | Responsibility |
|---|---|
| `api.py` | Input normalization, run-id minting, results URI construction. `inference(model, dataset, config) -> InferenceResult`. |
| `meta_coordinator.py` | Submits one Fray job per region (each running `regional_job.main`), waits for all to terminate, computes `missing_shards` from the canonical output prefix. Regional failure is non-fatal here. |
| `regional_job.py` | Runs as a CPU Fray job in one region. Validates GCS paths are same-region, constructs a `ZephyrContext` carrying per-context heartbeat / failure caps and a worker `EnvironmentConfig` (uv extras + JAX/vLLM XLA-cache env vars), builds the pipeline, calls `ctx.execute`. |
| `pipeline.py` | Builds the Zephyr `Dataset` graph. Assigns content-stable shard IDs by sorted input-file position, applies per-region rotation, processes one input file per Zephyr shard. Output written directly to the canonical `shard-NNNNNNNN.jsonl.gz` path. |
| `vllm_worker.py` | Inside the worker actor: caches the vLLM engine at module scope (so it survives across shards via `InlineRunner`), dispatches on payload kind (`engine.generate` vs `engine.chat`), extracts response + extras. |
| `config.py`, `input.py`, `output.py`, `compile_cache.py` | Data classes and pure helpers — no I/O surprises. |

## Data flow (one `inference()` call)

1. Caller invokes `inference(model, dataset, config)` from wherever — a
   notebook, an ExecutorStep, an ad-hoc script.
2. `api.inference()` mints a `run_id`, builds the run-prefix
   (`gs://marin-{results_region}/{job_name}/{run_id}`), and resolves the
   input:
   - Inline records: written as JSONL.gz files under `{run_prefix}/inputs/`,
     `config.shard_size` records per file. Each file is one content shard.
   - Path/glob input: file list expanded as-is.
3. `meta_coordinator.run_meta_coordinator` submits one Fray job per region.
   Each carries the full input file list (content shard assignment is
   deterministic across regions).
4. Inside each regional job (`regional_job.main`):
   a. `_validate_region` runs the GCS pre-flight; cross-region paths crash
      here before vLLM loads.
   b. `_build_context` constructs the Zephyr context with the worker
      `EnvironmentConfig` (uv extras + JAX/vLLM XLA-cache env vars).
   c. `pipeline.build_dataset` builds the per-region work plan: shard IDs
      sorted by input-file position, rotated by region for cross-region
      load balance.
   d. `ctx.execute(dataset)` runs the pipeline on the Zephyr worker pool.
5. Each Zephyr worker actor lazily constructs the vLLM engine (cached at
   module scope so it survives across shards), processes its assigned
   shards, and writes outputs directly to
   `{run_prefix}/outputs/shard-NNNNNNNN.jsonl.gz`. The output path encodes
   the **content shard ID** (not the per-region Zephyr shard index), so
   different regions writing the same content shard land at the same path.
6. Once every regional job terminates, the meta-coordinator lists the
   output prefix and returns an `InferenceResult` with `missing_shards`
   set to whatever wasn't produced.

## Key decisions and why

### One input file = one content shard

The content shard ID is the file's position in the sorted input list. The
same file always maps to the same shard regardless of region. This makes
cross-region work assignment deterministic without coordination, lets us
write directly to a content-stable output path, and makes the
`skip_existing` race trivial: both regions racing the same shard land at
the same path; whoever writes last wins, and since the work is deterministic
the content is identical.

A consequence: `config.shard_size` controls inline-input chunking (one
chunk → one file → one shard). Path/glob input has its sharding fixed at
write time and `shard_size` is unused.

### Per-region rotation, not assignment

A central allocator would be one more piece of state to fail over. Instead,
each region rotates the full work list by `hash(region) % len(items)`.
Workers in different regions process disjoint shards at the start. If
work-shape skew means they collide later, `_output_exists` short-circuits
the duplicate work and the loser-region's worker moves on. The arbitration
is local; no inter-region communication.

### Regional failure is non-fatal at the run level

If one region's coordinator dies — preemption storm, infra outage, CPU
shortage — the other regions keep going. `missing_shards` surfaces what
nobody completed. This is intentional asymmetry: regional failures are
common in practice (preemptible TPUs, tier-blocked pools); they shouldn't
fail an otherwise-successful 8-region run. The caller decides what to do
with a partial result.

The pre-flight `check_gcs_paths_same_region` deliberately exempts
`results_region` — the canonical output sink is intentionally cross-region
when the writer region differs from the consumer region. This is the one
explicit cross-region path the library allows.

### Engine env vars must reach the worker process

The XLA compile cache needs `JAX_COMPILATION_CACHE_DIR` /
`VLLM_XLA_CACHE_PATH` / `JAX_ENABLE_COMPILATION_CACHE` to be set **on the
TPU worker process** — the regional CPU coordinator that submits the worker
job doesn't compile anything. Same for the uv extras (`marin:vllm`,
`marin:tpu`) that the worker needs to import vLLM in the first place.

The library delivers both via Fray's `EnvironmentConfig`, which iris
propagates to the worker job's environment spec:

```
regional_job._build_context
  → create_environment(extras=..., env_vars=...)
  → ZephyrContext.worker_environment
  → fray.create_actor_group(environment=...)
  → iris_backend.convert_environment(...)
  → iris.submit(environment=...)
  → worker process env at startup
```

Setting these on the coordinator's own `os.environ` is a no-op. Two
real-cluster bugs were caught by exactly this path: missing uv extras
(workers couldn't `import vllm`) and missing XLA cache env vars
(cache prefix stayed empty across runs, full cold compile every time).
Both fixes were the same shape: thread the state through
`EnvironmentConfig`, not through the coordinator's env.

### Compile cache hash includes `engine_kwargs`

XLA compiles a different program for different `tensor_parallel_size`,
`max_num_batched_tokens`, etc. The cache key is
`blake2b(repr(resolved_model_path, sorted(engine_kwargs.items())))[:8]`, so
changing any compile-affecting kwarg automatically gets its own cache
prefix without collision risk.

### vLLM engine cached at module scope, `InlineRunner`-driven

The vLLM `LLM` object is expensive to construct (cold XLA compile is ~5
min on v6e-4 even with a warm cache miss). Caching it at module scope means
a worker that handles 10 shards pays the construction cost once. Zephyr's
`InlineRunner` keeps the actor's Python process alive across shards, so
the module-scope cache holds. If a worker actor dies, the next replacement
re-pays the cost — that's acceptable; what we avoid is paying per-shard.

### `tpu_shapes` accepts a list of topology-matched alternatives

Specifying `("v6e-4", "v5litepod-4")` lets the scheduler land on whichever
has capacity. The constraint (enforced by `ResourceConfig.with_tpu`) is
that all listed variants share the same `vm_count` AND `chips_per_vm`.
`(v6e-4, v6e-8)` is rejected — same chip family but different chips-per-VM,
which would mean a different tensor-parallel layout.

### Output is verbatim — no markers stripped

The worker captures `RequestOutput.outputs[0].text` exactly as vLLM
produces it. `<think>`, `<reasoning>`, special tokens — none of it is
stripped. Downstream callers know their own domain; the library doesn't.
This is enforced as a unit test
(`test_extract_response_preserves_think_blocks_verbatim`) so a refactor
that adds incidental stripping will break CI.

## Failure model summary

| Failure | Where caught | Result |
|---|---|---|
| Cross-region GCS model URI | `regional_job._validate_region` pre-flight | Regional job crashes before vLLM loads. Surfaced as a missing shard. |
| Mixed payload kinds in one shard | `vllm_worker.infer_records` | Shard fails; logged as a `ValueError`. Surfaced as a missing shard. |
| Region preemption / tier-block | Iris / Zephyr | Worker pool shrinks. Surviving workers cover the rotated work; if every worker dies, surfaced as missing shards. |
| Whole region down | meta_coordinator wait | Regional `JobHandle.wait` raises; the meta-coordinator logs and continues. Other regions cover via rotation. |
| Engine cold-compile timeout | per-context `heartbeat_timeout` | Tunable on `InferenceConfig` (default 120s; raise for cold v5p compile). |
| Preemption thrash | per-context `max_shard_infra_failures` | Default 20; raise for preemption-heavy environments. |

The contract: `inference()` always returns. `is_complete` and
`missing_shards` carry the truth; callers act on those, not on exceptions.

## Why Zephyr + Fray (not Ray, not raw multiprocessing)

- We need **per-region scheduling** with regional capacity constraints
  (preemptible vs not, TPU shape, region affinity). Iris already handles
  this; Fray + Zephyr layer on top.
- We need **per-actor lifecycle** that survives across shards (the vLLM
  engine cache). Zephyr's `InlineRunner` is exactly this primitive.
- We need **failure isolation at the regional level** with no
  cross-region coordination on the happy path. Submitting one Fray job per
  region with `JobHandle.wait(raise_on_failure=False)` gives that for free.
- We need **stable resource shapes** (TPU vm_count + chips_per_vm) and
  alternatives in the same job. Fray's `ResourceConfig.with_tpu` enforces
  the topology invariant.

Ray was considered and rejected because Iris is already the cluster
authority for the marin org, and re-routing through a parallel scheduler
would have been a net cost.

## Non-goals

- **Online serving.** This is an offline / batch library. The engine
  cache is the wrong primitive for long-lived request/response loops; use
  `marin.inference.vllm_server` for that.
- **Training.** Use `levanter`.
- **Multi-host TPU.** Single-host shapes (v5p-8, v6e-4, v5litepod-*) only
  in v1.
- **`SamplingParams.n > 1`.** Rejected at config time. Multi-completion
  fans out elsewhere; landing in a follow-up.
- **Cross-region data movement we didn't ask for.** Hard-pinned
  `gs://marin-{X}/...` paths are validated against the worker's region;
  the only intentionally cross-region path is the canonical output sink.
