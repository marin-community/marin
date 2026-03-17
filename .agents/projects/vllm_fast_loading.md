# vLLM Fast Weight Loading on TPU

## Status: Shard-Streaming Validated on 70B (24GB memory)

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

## Known Constraints

- **Supported architectures:** Llama and Qwen only (MODEL_MAPPINGS coverage)
- **`--served-model-name`:** unsupported in in-process path, falls back to subprocess
- **vLLM version:** tested on `vllm-tpu==0.13.2.post6`
- **`sync_weights` stability:** vLLM extension, not guaranteed stable across upgrades
- **70B requires TP=4:** single v5p chip (95.7 GiB) cannot hold 131 GiB model skeleton
- **Iris memory:** 400GB cgroups limit needed until shard-streaming is implemented
- **`enforce_eager=True`:** used to avoid XLA compilation overhead during testing;
  production runs may want to remove this for better inference throughput

## Future Work (beyond streaming)

- Expand model family mapping inference to generic Llama/Qwen in `vllm_utils.py`
- `--served-model-name` support for Harbor evaluator compatibility
- Performance regression tests (load MB/s, time-to-first-token) in CI
- Benchmark with `enforce_eager=False` to measure XLA compilation + inference throughput
- Tune `LEVANTER_FSSPEC_MAX_CONCURRENT_CHUNKS` for higher download throughput
