# vLLM Fast Weight Loading on TPU

## Status: Async-Native Startup Hang Fix Applied — Awaiting TPU Validation

**Issue:** https://github.com/marin-community/marin/issues/3768
**Branch:** `vllm_load_fast` (off main, commit `dbeddbd45`)
**Related docs:** `FAST_VLLM_LOAD_TPU.md`, `CODEX_VLLM_SPEED_LOADING.md`
**Iris jobs:** `/ahmed/vllm-70b-smoke-tp4` (succeeded), `/ahmed/vllm-70b-smoke` (failed, OOM), `/ahmed/vllm-70b-smoke-2` (failed, RESOURCE_EXHAUSTED TP=1)
**TPU used for validation:** v5p-8, us-central1-a
**Model used for 70B test:** `gs://marin-us-central1/models/meta-llama--Llama-3-3-70B-Instruct--6f6073b`

---

## Problem

vLLM on TPU uses `runai_streamer` to load model weights from GCS. It downloads
safetensors via single-threaded HTTP at 53 MiB/s — regardless of concurrency
settings (tested up to 16 threads, no improvement). For a 70B model (131 GiB),
that's 41 minutes just for weight download. Combined with XLA compilation
(20-30 min), total cold start exceeds 1 hour.

---

## Timeline

### 2026-03-12 — Initial investigation (prior agent, Codex)
- Identified RunAI streamer as the bottleneck
- Found existing fast-path machinery in the RL codebase:
  - `load_format="dummy"` for empty model skeleton
  - `read_safetensors_fsspec` for parallel GCS downloads (Levanter)
  - `sync_weights()` for in-place weight injection
- Wrote design doc (`CODEX_VLLM_SPEED_LOADING.md`)

### 2026-03-13 — Implementation (prior agent, Codex)
- Created `lib/marin/src/marin/inference/vllm_inprocess.py`:
  - `InProcessVllmRuntime` dataclass
  - `evaluate_inprocess_eligibility()` — checks GCS path, model mappings, CLI args
  - `start_inprocess_vllm_server()` — full pipeline orchestration
  - `load_safetensors_from_remote()` — fsspec shard loading
  - `_create_inprocess_openai_app()` — minimal FastAPI wrapper around `LLM.generate()`
    (can't use vLLM's `build_app()` — it spawns child processes that fight over TPU lock)
- Updated `lib/marin/src/marin/inference/vllm_server.py`:
  - Backend selection: in-process for eligible GCS models, subprocess otherwise
  - `runai_streamer` injection deferred to subprocess path only
  - Automatic fallback on in-process failure
- Created `tests/vllm/test_vllm_inprocess_backend.py` — 11 unit tests
- Created experiment scripts in `experiments/inference/`:
  - `exp_vllm_smoke_test.py` — thin wrapper
  - `exp_vllm_inprocess_direct.py` — direct LLM.generate() test (no HTTP server)
  - `exp_vllm_eval.py` — lm-eval benchmarks via vLLM
  - `exp_vllm_stress_test.py` — 5000 concurrent prompts

### 2026-03-14 — Bootstrap fix and 8B validation (prior agent, Codex)
- **Bug found:** vLLM rejects `load_format="dummy"` when model path is `gs://`
- **Fix:** stage only metadata files (config.json, tokenizer) from GCS to local tmpdir,
  pass tmpdir to `LLM()`. Actual weights come from Levanter fsspec separately.
- Added bootstrap viability checks to eligibility
- Added Iris-visible exception emission for in-process failures
- Added vLLM API compatibility guards (`sync_weights` availability, `build_app` signature)
- Unit tests: 11 passing
- Lint/format/pyrefly: all passing

### 2026-03-15 — 8B end-to-end validated on TPU (prior agent, Claude)
- Llama 3.1 8B Instruct on v5p-8, us-central1
- Resolved issues:
  - TPU lock conflict → `enforce_eager=True` + direct `LLM.generate()`
  - asyncio event loop conflict → `asyncio.new_event_loop()`
  - HF 2D attention vs Levanter 3D → reshape q/k/v/o_proj
  - RESOURCE_EXHAUSTED on v6e-4 → switched to v5p-8
- **Result:** weight pipeline 139s vs RunAI 300s = **~3x speedup**
  - LLM skeleton: 108s, Load 291 tensors: 95.1s, Convert: 3.4s, Inject: 40.5s
  - Generated 128 tokens at 4.1 tok/s
- Per-stage timing already implemented via `_iris_emit` calls

### 2026-03-16 — 70B validation and streaming analysis (this session, Claude Opus)
- **70B smoke test script:** `experiments/inference/exp_vllm_70b_smoke_test.py`
  - Defaults to GCS path for Llama 3.3 70B in us-central1
  - 5 diverse test prompts (math, translation, factual, creative)
  - Reports download throughput in MiB/s and full timing breakdown
- **Failed attempts:**
  - `/ahmed/vllm-70b-smoke` — OOM killed, default `--memory 1GB` too low
  - `/ahmed/vllm-70b-smoke-2` — RESOURCE_EXHAUSTED, `tensor_parallel_size=1` can't fit 131 GiB on single 95.7 GiB chip
- **Successful run:** `/ahmed/vllm-70b-smoke-tp4`
  - `tensor_parallel_size=4`, `--memory 400GB`, `enforce_eager=True`
  - HBM: 32.86 GiB/chip for model, 86.16 GiB/chip after KV cache
  - Timing breakdown:
    - Bootstrap: 2.4s
    - LLM skeleton: 189.0s
    - **Weight download: 876.1s (14.6 min) at 154 MiB/s**
    - Reshape: 40.4s (320 attention projections)
    - NNX convert: 53.0s
    - Weight inject: 484.0s
    - **Weight pipeline total: 1453.4s (24.2 min)**
    - Generation: 94.7s (5 prompts, 568 tokens, 6.0 tok/s)
    - **Total: 1739.6s (29 min)**
  - All 5 prompts correct (17*23=391, proper translations, coherent haiku)
  - **vs RunAI: 41 min download alone, >60 min total (timed out)**

- **Key discovery: `sync_weights` supports partial state dicts**
  - Source: `tpu-inference/tpu_inference/models/jax/utils/weight_utils.py`
    (local copy at `/Users/ahmed/code/vllm_tpu_multi/tpu-inference/`)
  - `transfer_state_with_mappings` iterates over SOURCE keys, skips missing with a log
  - No "all keys must be present" assertion
  - This means shard-by-shard streaming is possible without any changes to vLLM

---

## Current Architecture

```
Pipeline (in-process path):
  1. Stage config.json + tokenizer from GCS → local tmpdir (~2 MB, ~2s)
  2. LLM(model=tmpdir, load_format="dummy", tensor_parallel_size=4)
     → empty model skeleton on HBM (~189s for 70B)
  3. load_safetensors_from_remote(gs://model)
     → Levanter fsspec: parse safetensors headers, download 2GB chunks
       in parallel via byte-range HTTP requests (~154 MiB/s)
     → ALL shards accumulated in single state_dict on host RAM (~131 GiB)
  4. Reshape attention: HF 2D → Levanter 3D (q/k/v/o_proj)
  5. levanter_state_dict_to_nnx_state_on_cpu → wrap in nnx.Param, pad to 128
  6. sync_weights(nnx_state) → overwrite dummy weights in-place on HBM
  7. Serve via minimal FastAPI app wrapping LLM.generate()

Fallback: any failure → subprocess + runai_streamer (original path)
```

---

## Key Files

| File | Purpose |
|------|---------|
| `lib/marin/src/marin/inference/vllm_inprocess.py` | In-process backend: eligibility, loading, injection, serving |
| `lib/marin/src/marin/inference/vllm_server.py` | Backend selection, fallback, VllmEnvironment |
| `lib/levanter/src/levanter/compat/fsspec_safetensor.py` | Parallel GCS weight loader (byte-range requests) |
| `lib/marin/src/marin/rl/weight_utils.py` | `levanter_state_dict_to_nnx_state_on_cpu` (NNX conversion + padding) |
| `lib/marin/src/marin/rl/environments/inference_ctx/vllm_utils.py` | `MODEL_MAPPINGS`, `MODEL_TRANSPOSE_KEYS` |
| `tests/vllm/test_vllm_inprocess_backend.py` | 11 unit tests |
| `experiments/inference/exp_vllm_70b_smoke_test.py` | 70B smoke test with timing breakdown |
| `experiments/inference/exp_vllm_inprocess_direct.py` | Direct LLM.generate() test (no HTTP) |
| `experiments/inference/exp_vllm_eval.py` | lm-eval benchmark script |
| `experiments/inference/exp_vllm_stress_test.py` | 5000-prompt stress test |
| `FAST_VLLM_LOAD_TPU.md` | Technical reference with results and streaming plan |
| `CODEX_VLLM_SPEED_LOADING.md` | Original design doc |

### External reference
| File | Purpose |
|------|---------|
| `/Users/ahmed/code/vllm_tpu_multi/tpu-inference/` | Local copy of vllm-tpu (github.com/vllm-project/tpu-inference) |
| `tpu_inference/models/jax/utils/weight_utils.py` | `transfer_state_with_mappings` — the sync_weights implementation |
| `tpu_inference/worker/tpu_worker.py:440` | `sync_weights` method on driver_worker |
| `tpu_inference/runner/tpu_runner.py:1770` | `_sync_weights` delegation to transfer_state_with_mappings |

---

## Validated Results

| Model | TPU | Weight Download | Weight Pipeline | Total | vs RunAI |
|-------|-----|-----------------|-----------------|-------|----------|
| 8B (Llama 3.1) | v5p-8, TP=1 | 95.1s | 139s | ~26 min* | ~3x faster |
| 70B (Llama 3.3) | v5p-8, TP=4 | 876.1s (14.6 min, 154 MiB/s) | 1453.4s (24.2 min) | 29 min | ~2x faster |

*8B total includes XLA compilation (~25 min), 70B used enforce_eager so no compilation.

---

## Shard-Streaming: Validated (March 16, 2026)

### Result
- Job `/ahmed/vllm-70b-stream-24g` — **succeeded at 24GB memory**
- 16GB OOM'd (skeleton needs ~15 GiB host RAM)
- Prior all-at-once at 64GB OOM'd — streaming fixes this completely
- Weight pipeline: 1379.8s (slightly faster than all-at-once 1453.4s)
- 568 tokens generated, correct output, 4.6 tok/s
- Memory reduction: **400GB → 24GB** (16.7x)

### What was done
Replaced all-at-once weight loading with per-shard streaming: load one shard (~4.6 GiB),
reshape + convert + inject, free, repeat. Peak host RAM = skeleton (~15 GiB) + one shard (~5 GiB).

### Why it works
1. **`sync_weights`** iterates over source keys, skips missing ones (`continue`, no error).
   Confirmed from `transfer_state_with_mappings` in tpu-inference source, then validated on TPU.
2. **`levanter_state_dict_to_nnx_state_on_cpu`** processes each key independently.
3. **Attention reshape** uses constants from config.json, independent of shard contents.

### Implementation

```python
def load_and_inject_streaming(model_path, llm, mapping_model_name, model_config):
    fs, remote_path = url_to_fs(model_path)
    shard_files = _discover_safetensor_shards(fs, remote_path)
    sync_weights_fn = _resolve_sync_weights_callable(llm)

    loop = asyncio.new_event_loop()
    cpu = jax.devices("cpu")[0]

    for i, shard_file in enumerate(shard_files):
        with jax.default_device(cpu):
            shard_dict = loop.run_until_complete(
                read_safetensors_fsspec(os.path.join(remote_path, shard_file), fs=fs, sharding_fn=None)
            )
        _reshape_attention_tensors(shard_dict, num_heads, num_kv_heads, head_dim)
        nnx_state = levanter_state_dict_to_nnx_state_on_cpu(shard_dict)
        sync_weights_fn(nnx_state, mappings=..., transpose_keys=..., reshard_fn=None)
        del shard_dict, nnx_state
        _iris_log(f"Shard {i+1}/{len(shard_files)} injected")
    loop.close()
```

### Where to change
- `lib/marin/src/marin/inference/vllm_inprocess.py`:
  - Add `load_and_inject_streaming()` function
  - Update `start_inprocess_vllm_server()` to use streaming instead of `load_safetensors_from_remote`
  - Extract attention reshape logic into `_reshape_attention_tensors()` helper
- `experiments/inference/exp_vllm_70b_smoke_test.py`:
  - Update to use the streaming path (or add a `--streaming` flag)

### Validated memory impact

| | All-at-once | Streaming |
|---|---|---|
| Iris `--memory` | 400GB | **24GB** |
| Peak host RAM | ~131 GiB + overhead | ~15 GiB (skeleton) + ~5 GiB (shard) |
| Weight pipeline | 1453.4s | **1379.8s** |
| `sync_weights` calls | 1 | 30 |

---

### 2026-03-17 — Stress test debugging and Iris logging fix (Claude Opus session)

#### Bugs found and fixed

1. **`_unsupported_extra_cli_args` too strict** — the other agent's commit `4dbaefaa3`
   rejected ALL raw CLI flags including `--max-model-len`, which comes from `engine_kwargs`
   and is already handled by `_llm_kwargs()`. This caused the stress test to fall back to
   subprocess (slow path). **Fixed:** removed the redundant guard from
   `start_inprocess_vllm_server` — eligibility checking already handles this correctly
   using only true `extra_args` (not engine_kwargs-derived flags).

2. **Iris dashboard showing 0 log lines** — ALL in-process runs (including `vllm-70b-stream-final`,
   `vllm-70b-stream-24g`) had zero visible logs in the Iris dashboard.
   - **Root cause:** `_iris_emit` wrote to `_REAL_STDOUT = sys.stdout`, but `sys.stdout` was
     captured AFTER `import jax` which redirects stdout on TPU (libtpu load).
   - **Attempted fixes:** moving `_REAL_STDOUT` before imports (worked for direct script
     `exp_vllm_inprocess_direct.py` but fragile — import order dependent).
   - **Final fix:** `os.dup(1)` to duplicate the raw stdout file descriptor at import time,
     then `os.write(_IRIS_LOG_FD, line.encode())` for all log output. This bypasses Python's
     `sys.stdout` entirely — immune to any JAX/vLLM stdout redirection. Applied to
     `vllm_inprocess.py`, `exp_vllm_stress_test.py`, and `exp_vllm_inprocess_direct.py`.
   - **Validated:** `vllm-stress-8b-v4` shows shard injection logs and timing in dashboard.

3. **Qwen3 235B model downloaded** — `Qwen/Qwen3-235B-A22B-Thinking-2507` (revision `6cbffae`)
   downloaded to both `gs://marin-us-east1/` and `gs://marin-eu-west4/`. 118 safetensors
   shards, 470 GB. Added to `experiments/models.py`.

#### Fixed: HTTP serving via queue-based generation server

- **Previous blocker:** `vllm-stress-8b-v4` — all 50 HTTP requests returned 500 errors.
  `llm.generate()` called from uvicorn's threadpool threads was not thread-safe.
- **Fix (commit `a260b1e69`):** Queue-based generation server. All HTTP handlers enqueue
  `GenerationRequest` objects (prompt + SamplingParams + Future) onto a `queue.Queue`.
  A single dedicated worker thread drains the queue, groups by SamplingParams for batching,
  calls `llm.generate()`, and distributes results back via futures.
- **Validated:** Job `/ahmed/vllm-stress-8b-v6` — **50/50 successful, 0 errors**.
  - Weight pipeline: 119.2s, 15.0 GiB at 128 MiB/s
  - Total tokens: 6,400 at 34.1 tok/s aggregate
  - p50 latency: 129.4s, p95: 185.8s
  - Results: `gs://marin-us-central1/inference/meta-llama--Llama-3-1-8B-Instruct--0e9e39f/stress_test/results_20260318-040743.json`

#### All paths now working

| Path | Weight Loading | Inference | HTTP Serving |
|------|---------------|-----------|--------------|
| `exp_vllm_inprocess_direct.py` (main thread) | ✅ | ✅ | N/A (no HTTP) |
| `VllmEnvironment` → in-process → FastAPI + queue | ✅ | ✅ | ✅ |
| `VllmEnvironment` → subprocess fallback | ✅ (slow, runai_streamer) | ✅ | ✅ |

---

### 2026-03-18 — Async-native backend refactor (Claude Opus session)

#### Motivation
The queue-based in-process backend (`InProcessVllmServerBackend`) from March 17 worked
but had two fundamental problems:
- It used a custom FastAPI app around blocking `LLM.generate()` — not the standard vLLM
  async serving stack
- Requests were serialized through a single worker thread, destroying concurrent throughput

The RL codebase already proved that vLLM's `AsyncLLM` + worker extensions work on TPU
(`lib/marin/src/marin/rl/environments/inference_ctx/inflight/worker.py`).

#### What was done

1. **Created `lib/marin/src/marin/inference/vllm_async.py`** — new async-native backend:
   - `AsyncVllmRuntime` dataclass (serve thread, server URL, events, error tracking)
   - `start_async_vllm_server()` — orchestrates the full pipeline in a daemon thread
   - `_run_async_server()` — the async core:
     - Stages bootstrap metadata via existing `_resolve_bootstrap_model_source_for_start`
     - Imports vLLM async symbols (`AsyncEngineArgs`, `build_app`, `build_async_engine_client_from_engine_args`, etc.)
     - Builds CLI args with `--load-format dummy`, `--worker-extension-cls` pointing to RL's `WorkerExtension`,
       `--served-model-name`, `--disable-frontend-multiprocessing`
     - Creates `AsyncLLM` engine via `build_async_engine_client_from_engine_args`
     - Streams shards via `_load_and_inject_streaming_async()`:
       - Downloads one shard at a time using Levanter `read_safetensors_fsspec`
       - Reshapes attention tensors (HF 2D → 3D)
       - Serializes via `serialize_state_dict_for_rpc` (from RL)
       - Injects via `engine_client.collective_rpc("update_weight", ...)`
       - Frees shard memory after each injection
     - Resets prefix cache after weight injection
     - Builds vLLM's standard OpenAI FastAPI app (`build_app` + `init_app_state`)
     - Serves via uvicorn with a watchdog task monitoring engine health
   - `stop_async_vllm_server()` — graceful shutdown + bootstrap cleanup
   - `_configure_async_vllm_environment()` — TPU env defaults + `VLLM_ALLOW_INSECURE_SERIALIZATION`

2. **Added `ManagedAsyncVllmServerBackend` to `vllm_server.py`**:
   - New `VllmServerBackend` subclass that wraps `start_async_vllm_server` / `stop_async_vllm_server`
   - `VllmEnvironment` now selects `ManagedAsyncVllmServerBackend` when native+eligible
     (replaces the old `InProcessVllmServerBackend`)
   - `VllmServerHandle` gained `async_runtime: AsyncVllmRuntime | None` field
   - Fallback to `NativeVllmServerBackend` (subprocess `vllm serve`) on async failure preserved

3. **Key architectural improvements over queue-based path:**
   - Uses vLLM's standard `AsyncLLM` engine — proper continuous batching, not serialized `LLM.generate()`
   - Uses vLLM's standard OpenAI app (`build_app`) — full API compatibility including `/v1/chat/completions`
   - Weight injection via `collective_rpc("update_weight")` through RL's `WorkerExtension` —
     no more direct `sync_weights` call from the frontend process
   - `--served-model-name` now fully supported (was a known constraint before)
   - No more custom queue/Future plumbing for thread safety

4. **Updated tests in `tests/vllm/test_vllm_inprocess_backend.py`:**
   - `test_vllm_environment_selects_inprocess_when_eligible` → asserts `ManagedAsyncVllmServerBackend`
   - `test_vllm_environment_falls_back_to_native_subprocess` → tests async failure → subprocess fallback
   - `test_build_openai_server_cli_args_sets_async_native_defaults` — validates CLI arg construction
   - `test_configure_async_vllm_environment_sets_defaults` — validates env var setup
   - Total: 13 tests

#### Weight injection path comparison

| | Queue-based (March 17) | Async-native (March 18) |
|---|---|---|
| Engine | `vllm.LLM` (blocking) | `AsyncLLM` (async) |
| Weight injection | `sync_weights()` direct call | `collective_rpc("update_weight")` via WorkerExtension |
| Serialization | `levanter_state_dict_to_nnx_state_on_cpu` | `serialize_state_dict_for_rpc` (RL helper) |
| HTTP serving | Custom FastAPI + queue + worker thread | vLLM's standard `build_app` + uvicorn |
| `--served-model-name` | Unsupported (fell back to subprocess) | Supported |
| Concurrent requests | Serialized through single thread | vLLM continuous batching |

---

## Current Architecture

```
Pipeline (async-native path, MARIN_VLLM_MODE=native):
  1. Stage config.json + tokenizer from GCS → local tmpdir (~2 MB, ~2s)
  2. AsyncLLM(model=tmpdir, load_format="dummy", worker_extension_cls=WorkerExtension,
     tensor_parallel_size=4, disable_frontend_multiprocessing=True)
     → empty model skeleton on HBM
  3. For each safetensor shard:
     a. Download shard via Levanter read_safetensors_fsspec (~4.6 GiB)
     b. Reshape attention tensors (HF 2D → 3D)
     c. Serialize via serialize_state_dict_for_rpc
     d. engine_client.collective_rpc("update_weight", shard, mapping_model_name)
     e. Free shard memory
  4. engine_client.reset_prefix_cache()
  5. build_app(args, supported_tasks) → vLLM's standard OpenAI FastAPI app
  6. init_app_state(engine_client, app.state, args, supported_tasks)
  7. Serve via uvicorn + watchdog loop

Fallback: any failure → subprocess `vllm serve` + runai_streamer (original path)
```

---

## Key Files

| File | Purpose |
|------|---------|
| `lib/marin/src/marin/inference/vllm_async.py` | Async-native backend: engine creation, shard streaming, OpenAI app serving |
| `lib/marin/src/marin/inference/vllm_inprocess.py` | Eligibility checking, bootstrap staging, shared helpers (reshape, emit, etc.) |
| `lib/marin/src/marin/inference/vllm_server.py` | Backend selection (`ManagedAsyncVllmServerBackend`), fallback, VllmEnvironment |
| `lib/levanter/src/levanter/compat/fsspec_safetensor.py` | Parallel GCS weight loader (byte-range requests) |
| `lib/marin/src/marin/rl/environments/inference_ctx/async_vllm.py` | `serialize_state_dict_for_rpc` (RPC serialization) |
| `lib/marin/src/marin/rl/environments/inference_ctx/inflight/worker.py` | `WorkerExtension` — handles `update_weight` RPC on worker side |
| `lib/marin/src/marin/rl/environments/inference_ctx/vllm_utils.py` | `MODEL_MAPPINGS`, `MODEL_TRANSPOSE_KEYS` |
| `tests/vllm/test_vllm_inprocess_backend.py` | 13 unit tests |
| `experiments/inference/exp_vllm_70b_smoke_test.py` | 70B smoke test with timing breakdown |
| `experiments/inference/exp_vllm_inprocess_direct.py` | Direct LLM.generate() test (no HTTP) |
| `experiments/inference/exp_vllm_eval.py` | lm-eval benchmark script |
| `experiments/inference/exp_vllm_stress_test.py` | 5000-prompt stress test |
| `.agents/projects/vllm_async_refactor.md` | Design doc for async refactor |

### External reference
| File | Purpose |
|------|---------|
| `/Users/ahmed/code/vllm_tpu_multi/tpu-inference/` | Local copy of vllm-tpu (github.com/vllm-project/tpu-inference) |
| `tpu_inference/models/jax/utils/weight_utils.py` | `transfer_state_with_mappings` — the sync_weights implementation |
| `tpu_inference/worker/tpu_worker.py:440` | `sync_weights` method on driver_worker |
| `tpu_inference/runner/tpu_runner.py:1770` | `_sync_weights` delegation to transfer_state_with_mappings |

---

## Known Constraints

- **Supported architectures:** Llama and Qwen only (MODEL_MAPPINGS coverage)
- **vLLM version:** tested on `vllm-tpu==0.13.2.post6`
- **`collective_rpc` stability:** depends on vLLM worker extension API, not guaranteed stable across upgrades
- **70B requires TP=4:** single v5p chip (95.7 GiB) cannot hold 131 GiB model skeleton
- **`enforce_eager=True`:** used to avoid XLA compilation overhead during testing;
  production runs may want to remove this for better inference throughput
- **Async-native path had startup hang on TPU** — engine subprocess hung at `vllm_get_model()`.
  Fix applied 2026-03-19: `VLLM_ENABLE_V1_MULTIPROCESSING=0` added to `_configure_async_vllm_environment()`
  to run engine in-process (matching RL path). Awaiting TPU validation via Iris job `vllm-async-no-mp-v1`.

---

### 2026-03-21 — Fork-native fsspec streaming weight loader (Claude Opus session)

#### Motivation

The previous approaches (in-process, async-native) loaded weights *outside* of
tpu-inference by using Marin-side machinery (Levanter fsspec, vllm_inprocess.py,
`sync_weights`, WorkerExtension RPCs). This worked but required complex Marin-side
orchestration and couldn't benefit `vllm serve` directly.

This session took a different approach: implement fsspec streaming *inside* the
tpu-inference fork, plugging into the existing weight-loading seam. This means
both `LLM.generate()` and `vllm serve` get fast loading through one unified path.

#### What was done

1. **New file: `tpu_inference/models/jax/streaming_weights.py`**
   - Ported from Levanter's `fsspec_safetensor.py` (header parsing, chunk building,
     dtype map, shard discovery)
   - `fsspec_weights_iterator(model_path)` yields `(str, jax.Array)` pairs
   - Sequential chunk-by-chunk streaming, peak host RAM ≈ chunk_size (~2 GiB)
   - All arrays materialized on CPU via `jax.default_device(cpu_device)`
   - Per-shard timing + peak RSS logging
   - Uses `gcsfs` for `gs://` paths, plain fsspec for local

2. **Extended `TpuBootstrapConfig`** with `weight_loader` field:
   - `"default"` — existing RunAI/file-based path
   - `"fsspec_streamer"` — new fsspec path
   - CLI: `--additional-config '{"tpu_bootstrap": {"model_bootstrap": "abstract_load", "weight_loader": "fsspec_streamer"}}'`

3. **New branch in `_build_abstract_model_and_load_weights()`**:
   - `fsspec_streamer` branch creates iterator, sets `model_weights_iterator`,
     calls `model.load_weights(rng)`, cleans up — same pattern as RunAI branch

4. **Widened `load_hf_weights()` iterator path**:
   - Lazy `torchax.default_env()` init — only when a `torch.Tensor` is encountered
   - `jax.Array` passes through directly (no torch→jax copy)
   - `np.ndarray` raises `TypeError`

5. **Dependencies**: added `fsspec>=2024.1.0`, `gcsfs>=2024.1.0` to requirements.txt

6. **Tests**:
   - `tests/models/jax/test_streaming_weights.py`: header parsing, chunk building,
     shard discovery, end-to-end iterator, CPU pinning, BF16 roundtrip, stable ordering
   - `tests/models/common/test_model_loader.py`: config parsing, invalid values,
     fsspec_streamer dispatch mock, ndarray rejection

7. **Marin-side fix: `MODEL_IMPL_TYPE` default changed from `"vllm"` to `"auto"`**
   in `vllm_server.py` (both `_vllm_jax_env()` and `_vllm_env()`).
   The `"vllm"` default was set by David Hall in PR #2320 (Jan 2026) as a defensive
   measure when flax_nnx lacked reliable mesh initialization. tpu-inference's
   `tpu_runner.py` now sets up the mesh correctly, and the `auto` default lets
   tpu-inference route supported architectures (Llama) to flax_nnx while falling
   back to vllm for unsupported ones.

#### Wheel and pin

- Fork branch: `marin-community/tpu-inference` `marin` branch
- Wheel release: `marin-ecc13081`
- Marin pin: `pyproject.toml` + `uv.lock` updated on `vllm_load_fast` branch

#### Smoke test: `/ahmed/vllm-smoke-fsspec`

- **Failed**: `MODEL_IMPL_TYPE=vllm` was hardcoded in `vllm_server.py`, forcing
  the vLLM PyTorch path. The fsspec code (flax_nnx path only) was never reached.
  RunAI loaded weights instead, OOM'd at 16GB.
- **Root cause identified and fixed**: changed default to `"auto"`.
- **Pending resubmission** after this commit.

#### Key files (fork)

| File | Repo | Action |
|------|------|--------|
| `tpu_inference/models/jax/streaming_weights.py` | fork | **New**: fsspec iterator |
| `tpu_inference/models/common/model_loader.py` | fork | Extended config + fsspec branch |
| `tpu_inference/models/jax/utils/weight_utils.py` | fork | jax.Array dispatch + lazy torchax |
| `requirements.txt` | fork | Added fsspec, gcsfs |
| `tests/models/jax/test_streaming_weights.py` | fork | **New**: iterator unit tests |
| `tests/models/common/test_model_loader.py` | fork | Config + dispatch tests |

#### Key files (marin)

| File | Action |
|------|--------|
| `lib/marin/src/marin/inference/vllm_server.py` | Changed MODEL_IMPL_TYPE default to "auto" |
| `pyproject.toml` | Updated wheel pin to marin-ecc13081 |
| `uv.lock` | Updated for new wheel |

---

## Future Work

- **Resubmit fsspec smoke test** with `MODEL_IMPL_TYPE=auto` fix
- Concurrent chunk downloads (v2 of streaming_weights.py)
- Expand model family mapping inference to generic Llama/Qwen in `vllm_utils.py`
- Performance regression tests (load MB/s, time-to-first-token) in CI
- Benchmark with `enforce_eager=False` to measure XLA compilation + inference throughput
- Tune `LEVANTER_FSSPEC_MAX_CONCURRENT_CHUNKS` for higher download throughput
- Compare async-native throughput vs subprocess `vllm serve` under concurrent load
- Extract RL async worker-extension utilities into a shared inference module
