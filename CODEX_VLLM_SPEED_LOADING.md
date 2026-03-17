# Design: In-Process vLLM Startup with Fast fsspec Weight Loading

## Implementation Update (March 14, 2026)

Status: implemented in code with guarded fallback; 8B runtime validation succeeded on TPU. Full 70B/perf characterization is still pending.

### Logbook

- **2026-03-14**: Investigated in-process startup failure from live run. Root cause was confirmed as vLLM rejecting
  `load_format="dummy"` when `model` is an object-store URI (`gs://...` / `s3://...`).
- **2026-03-14**: Implemented bootstrap-source split in `vllm_inprocess.py`: keep remote path for fast weight loading,
  but initialize `LLM(dummy)` from a non-object-store bootstrap source (override, non-object-store `model.name`, or
  staged local metadata files from `model.path`).
- **2026-03-14**: Added bootstrap viability checks to in-process eligibility and cleanup for temporary staged metadata
  directories on both startup failure and normal shutdown.
- **2026-03-14**: Expanded unit tests in `tests/vllm/test_vllm_inprocess_backend.py` to cover bootstrap override rules
  and staged-local bootstrap resolution; local suite result is `9 passed`.
- **2026-03-14**: Ran repo-required checks on touched files using `./infra/pre-commit.py --fix ...`; lint, format, and
  pyrefly checks passed.
- **2026-03-14**: Added Iris-visible exception emission for in-process startup failures before fallback in
  `VllmEnvironment.__enter__` (`_emit_exception_to_iris`), including exception type/message and traceback lines.
  This removes the previous "silent fallback" failure mode in dashboard logs.
- **2026-03-14**: Updated fallback unit test to assert `_iris_emit` receives in-process failure details
  (`RuntimeError: in-process failed`) when fallback activates; local suite remains `9 passed`.
- **2026-03-14**: Added explicit vLLM API compatibility guards in `vllm_inprocess.py`:
  - fail fast when `build_app` has `build_app(args: Namespace)` signature (not `build_app(llm)`)
  - fail fast when `driver_worker.sync_weights` is unavailable
  These now raise `InProcessVllmUnsupportedError` with actionable messages instead of opaque runtime failures.
- **2026-03-14**: Added unit tests for both compatibility guards; local suite result is now `11 passed`.
- **2026-03-14**: Live in-process run succeeded end-to-end:
  - `load_format="dummy"` model skeleton initialized
  - no fallback to subprocess / `runai_streamer`
  - XLA compilation started for the expected Llama 8B architecture (32 layers, 8 KV heads, head dim 128)
  This confirms the in-process checkpoint load + weight injection path is working in the target runtime.
- **2026-03-14**: Coarse observed window from job logs was ~`02:38:18 -> 02:44:56` (~6.5 min) up to compile start,
  covering engine init + remote load + inject + compile bring-up. Fine-grained per-stage timings are not yet emitted.

### Bootstrap fix for `load_format="dummy"` on object-store paths

A runtime issue was discovered after the first implementation: when `LLM(..., load_format="dummy")` was called with
`model="gs://..."`, vLLM rejected startup and required streamer load formats for object-store model URIs. This caused
the expected fallback to subprocess + `runai_streamer`.

The in-process path has now been adjusted so `dummy` initialization never uses a `gs://`/`s3://` bootstrap model URI:
- New bootstrap-source resolution in `vllm_inprocess.py`:
  - `engine_kwargs["inprocess_bootstrap_model"]` override (must be non-object-store)
  - otherwise `model.name` if it is non-object-store
  - otherwise stage local metadata files (`config.json`, tokenizer files, etc.) from `model.path` and use that local dir
- In-process eligibility now checks bootstrap-source viability up front and returns ineligible if bootstrap cannot be
  resolved safely.
- Temporary staged metadata directories are cleaned up on normal shutdown and startup failure.

Completed in this change:
- Added `lib/marin/src/marin/inference/vllm_inprocess.py` with:
  - in-process eligibility checks (`evaluate_inprocess_eligibility`)
  - fast safetensor loading via `read_safetensors_fsspec` + `url_to_fs`
  - `LLM(load_format=\"dummy\")` startup + `sync_weights()` injection
  - in-process OpenAI server thread startup/shutdown wrappers
- Updated `lib/marin/src/marin/inference/vllm_server.py` to:
  - choose in-process backend only for eligible native/object-store models
  - defer `runai_streamer` auto-injection until subprocess path
  - add automatic fallback to subprocess native backend when in-process startup fails
  - preserve existing Docker behavior unchanged
- Added unit coverage in `tests/vllm/test_vllm_inprocess_backend.py` for:
  - eligibility decisions
  - backend selection behavior
  - fallback behavior from in-process -> subprocess

Validated locally in this workspace:
- `./infra/pre-commit.py --fix ...` on changed files: passed
- `uv run pytest tests/vllm/test_vllm_inprocess_backend.py -q`: passed

Still pending (requires TPU runtime with `vllm-tpu` installed):
- add stage-level timing emission for exact breakdown (bootstrap source resolution, safetensor load, NNX conversion, weight injection, server readiness)
- run 8B/70B startup timing and correctness checks against real checkpoints
- verify Harbor path behavior when `--served-model-name` is provided (currently treated as unsupported for in-process and expected to fall back to subprocess)

## Problem

`VllmEnvironment` currently starts `vllm serve` as a subprocess even in native mode (`lib/marin/src/marin/inference/vllm_server.py:895`).
For object-store checkpoints, startup defaults to `runai_streamer` (`lib/marin/src/marin/inference/vllm_server.py:351`), which is currently the dominant bottleneck for large models.

The repo already has a proven in-process weight-injection path in RL:
- Build engine with `load_format="dummy"` (`lib/marin/src/marin/rl/rl_experiment_utils.py:204`)
- Convert checkpoint tensors into NNX state (`lib/marin/src/marin/rl/weight_utils.py:75`)
- Inject via `driver_worker.sync_weights(...)` (`lib/marin/src/marin/rl/environments/inference_ctx/vllm.py:305`)

The repo also already has fast fsspec safetensor loading:
- `read_safetensors_fsspec` does concurrent byte-range reads (`lib/levanter/src/levanter/compat/fsspec_safetensor.py:203`)
- `url_to_fs` preserves Marin GCS guardrails (`lib/iris/src/iris/marin_fs.py:563`)

Backward compatibility requirement:
- Existing Docker and subprocess-native behavior must remain intact for unsupported cases.

## Goals

1. Keep the external contract unchanged: callers still interact with OpenAI-compatible HTTP (`/v1/*`).
2. For eligible native + object-store models, replace subprocess startup with in-process:
   - `LLM(..., load_format="dummy")`
   - fast fsspec shard loading
   - `sync_weights()` injection
3. Preserve current fallback paths (Docker, subprocess native, runai streamer defaults) with no behavior regressions.
4. Fail fast and automatically fall back when in-process prerequisites are not met.

Non-goals:
- No change to evaluator interfaces.
- No broad architecture expansion beyond mappings we can prove correct.
- No rework of Docker backend or evalchemy internals.

## Proposed Solution

### 1) Add a dedicated in-process runtime module

Create `lib/marin/src/marin/inference/vllm_inprocess.py` with:
- `InProcessVllmRuntime` dataclass (llm engine, uvicorn server, thread, model_id, diagnostics)
- `can_use_inprocess(...)` eligibility function
- startup/shutdown helpers
- checkpoint load + inject helpers

The module should be import-safe when `vllm` is not installed (defer hard imports until startup path is selected).

### 2) Select backend after eligibility, not before

Current `resolve_model_name_or_path()` mutates object-store models to `runai_streamer` early (`vllm_server.py:69` + `:351`).
For in-process loading, this mutation is wrong because we need `load_format="dummy"`.

Refine backend selection in `vllm_server.py`:
- Keep Docker path unchanged.
- In native mode, attempt in-process only when eligible.
- Only apply `_maybe_enable_streaming()` when taking subprocess path.

```python
# core routing idea (vllm_server.py)
if mode == "docker":
    backend = DockerVllmServerBackend(...)
elif mode == "native" and can_use_inprocess(model, extra_args):
    backend = InProcessVllmServerBackend(...)
else:
    model = _maybe_enable_streaming(model)  # existing behavior
    backend = NativeVllmServerBackend()
```

### 3) Load checkpoint shards with Levanter fsspec path

Use the same discovery pattern as HF checkpoint loader (`lib/levanter/src/levanter/compat/hf_checkpoints.py:671` + `:699`):
- detect sharded safetensor index (`model.safetensors.index.json`) or single-file safetensors
- load shard files deterministically
- call `read_safetensors_fsspec` through `fsspec.asyn.sync`

Important details:
- Use `url_to_fs(model_path)` so cross-region protections stay active (`marin_fs.py:563`).
- Keep load on CPU device context during conversion/injection to avoid accidental TPU placement spikes.
- Start with full-state load for parity with existing RL `sync_weights` flow; add a bounded-memory shard-streaming follow-up only after verifying partial-sync behavior.

```python
# core loading/injection idea (vllm_inprocess.py)
with jax.default_device(jax.devices("cpu")[0]):
    state_dict = {}
    for shard in shard_files:
        shard_state = fsspec_sync(loop, read_safetensors_fsspec, shard, fs=fs, sharding_fn=None)
        state_dict.update(shard_state)

nnx_state = levanter_state_dict_to_nnx_state_on_cpu(state_dict)
llm.llm_engine.model_executor.driver_worker.sync_weights(
    nnx_state,
    mappings=model_mapping,
    transpose_keys=transpose_keys,
    reshard_fn=None,
)
```

### 4) Reuse RL mapping/conversion machinery with stricter model-family checks

Reuse:
- `levanter_state_dict_to_nnx_state_on_cpu` (`weight_utils.py:75`)
- `MODEL_MAPPINGS` and `MODEL_TRANSPOSE_KEYS` (`vllm_utils.py:145`)

But the current mapping coverage is narrower than the original plan assumed:
- exact-name map entries + `Qwen2.5` fallback (`vllm_utils.py:110`)
- no generic Llama fallback today

Refinement:
- Add a local model-family resolver in `vllm_inprocess.py`:
  - first try `model.name` through `MODEL_MAPPINGS`
  - if absent, read `config.json` from model path and infer family from `architectures` / `model_type`
  - if still unresolved, fall back to subprocess

This avoids false negatives for names like path-derived run IDs (`evaluation/run.py:123`) while keeping unsupported models safe.

### 5) Run OpenAI HTTP server in background thread with explicit teardown

Mirror existing threaded serving pattern (`lib/marin/src/marin/rl/environments/inference_ctx/levanter.py:82`) and use explicit stop signaling on shutdown.

Shutdown order for in-process backend:
1. signal uvicorn server exit
2. join serve thread with timeout
3. call engine shutdown (`llm.llm_engine.shutdown()` when available)
4. clear runtime references

If vLLM API server symbols differ for `vllm-tpu==0.13.2.post6` (`lib/marin/pyproject.toml:171`), catch import/signature errors and fall back to subprocess with clear logs.

### 6) Handle `extra_args` conservatively

`VllmEnvironment` forwards CLI-style args (`vllm_server.py:421`), and Harbor currently depends on `--served-model-name` (`harbor_evaluator.py:247`).

In-process path should:
- support `--served-model-name` if API-server wiring allows it
- otherwise mark args as unsupported and fall back to subprocess

Do not silently ignore unknown CLI args in in-process mode.

## Implementation Outline

1. Add `vllm_inprocess.py` (eligibility, model-family detection, load/inject, threaded serve, shutdown).
2. Refactor `vllm_server.py` backend resolution so streaming defaults are applied only on subprocess path.
3. Add `InProcessVllmServerBackend` and wire fallback-on-error behavior with diagnostics.
4. Add unit tests for backend selection, eligibility, mapping resolution, and fallback behavior (no vLLM runtime required).
5. Run smoke tests on vLLM-enabled TPU env: 8B load-time + correctness, then 70B startup timing.
6. Validate evaluator compatibility (`lm-eval`, Harbor path with/without `--served-model-name`).

## Notes

- **Key-format risk**: HF safetensor keys are expected to work with current conversion logic because `levanter_state_dict_to_nnx_state_on_cpu` strips terminal `weight` and folds bias names (`weight_utils.py:85` and `:133`), matching mapping keys like `...q_proj` / `...q_proj_bias` (`vllm_utils.py:40`, `:58`). Validate with a real shard before enabling by default.
- **Memory risk**: Full-state loading can temporarily require significantly more than raw checkpoint size due to intermediate dicts and JAX wrapping. If this is tight on 70B hosts, implement shard-stream injection only after confirming `sync_weights` accepts partial states.
- **API stability risk**: `sync_weights` is a vLLM extension path used by RL here, not guaranteed stable across vLLM upgrades.
- **Concurrency knobs**: Loader throughput comes from `LEVANTER_FSSPEC_CHUNK_BYTES` and `LEVANTER_FSSPEC_MAX_CONCURRENT_CHUNKS` (`fsspec_safetensor.py:43`). Keep these tunable via env for bring-up.
- **Cross-region safety**: `url_to_fs` keeps cross-region GCS blocking (`marin_fs.py:433`), so this design does not bypass repo cost guardrails.

## Future Work

1. Stream shard-by-shard directly into `sync_weights` to bound host RAM during 70B+ loads.
2. Expand mapping inference to generic Llama/Qwen families in shared `vllm_utils.py` rather than local resolver glue.
3. Add dedicated performance regression tests (load MB/s, time-to-first-token) to CI/nightly TPU jobs.
4. Add optional startup telemetry for per-shard load and inject timings in `VllmEnvironment` diagnostics.
