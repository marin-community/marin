# vLLM Async-Native Startup Hang — Investigation Log

## Summary

The new async-native vLLM serving backend (`vllm_async.py`) hangs for 15–60+ minutes
during engine startup on TPU v5p-8. The hang has been narrowed through 13 Iris debug
runs to a specific function call inside the engine subprocess's model loading path.

**Branch:** `vllm_load_fast`
**Affected path:** `VllmEnvironment` → `ManagedAsyncVllmServerBackend` → `start_async_vllm_server` → `build_async_engine_client_from_engine_args` → engine subprocess hangs
**Working path for comparison:** RL's `AsyncLLM.from_engine_args(start_engine_loop=False)` in `inflight/worker.py`

---

## Root Cause Narrowing (13 Iris runs)

### What was eliminated

| Hypothesis | Tested in | Result |
|------------|-----------|--------|
| `SKIP_JAX_PRECOMPILE` not set | v2 | Set to `1` — no effect |
| `enforce_eager` not taking effect | v2–v4 | Confirmed active in logs — no effect |
| Executor orchestration hang | v7 | `init_worker` (2.4s) and `init_device` (24.8s) complete normally |
| Hang before `load_model` | v7 | `START WorkerWrapperBase.load_model` reached |
| Hang before `tpu_runner.get_model` | v8 | `START tpu_runner.get_model` reached |
| Hang before `model_loader.get_vllm_model` | v9 | `START model_loader.get_vllm_model` reached |
| Hang before `VllmModelWrapper.load_weights` | v9 | `START VllmModelWrapper.load_weights` reached |
| Hang in upstream `vllm_model_loader.get_model` | v12 | That function was never entered (module alias issue) |

### Where the hang is (v13 — final narrowing)

```
(EngineCore_DP0 pid=276) [marin-vllm-startup] START VllmModelWrapper.load_weights
(EngineCore_DP0 pid=276) [marin-vllm-startup] START vllm_model_wrapper.vllm_get_model
(EngineCore_DP0 pid=276) "Initializing vLLM model with random weights, weight loading skipped."
                          ← HANG: vllm_get_model never returns
                          ← shard_model_to_tpu never starts
                          ← 900s startup timeout fires
```

The hang is **inside** `vllm_get_model()` — after the "random weights" log but before the
function returns. `shard_model_to_tpu()` is never reached.

---

## Code Path Analysis

### What `VllmModelWrapper.load_weights()` does

File: `tpu-inference/tpu_inference/models/vllm/vllm_model_wrapper.py:134`

```python
def load_weights(self):
    # 1. Deep-copy vllm_config, force device=cpu
    vllm_config_for_load.device_config.device = "cpu"

    # 2. Log "random weights" (this IS visible in Iris logs)
    if load_format == "dummy":
        logger.info("Initializing vLLM model with random weights, weight loading skipped.")

    # 3. Call vllm_get_model inside jax.default_device(cpu) context
    with load_context, jax_context:
        vllm_model = vllm_get_model(vllm_config=vllm_config_for_load)    # ← HANGS HERE

    # 4. Wrap and shard (never reached)
    self.model = _VllmRunner(vllm_model)
    params_and_buffers = shard_model_to_tpu(self.model, self.mesh)        # ← NEVER REACHED
```

### What `vllm_get_model()` does (upstream vLLM `get_model`)

For `load_format=dummy`:
1. `DummyModelLoader.load_model()` → `initialize_model()` → creates `LlamaForCausalLM` architecture
2. Initializes random weights on CPU via `torch.randn_like()`
3. Calls `process_weights_after_loading()` — may trigger weight reshaping/conversion
4. Returns model

The "random weights" log fires **before** `vllm_get_model` is called (line 168 vs 187),
so the log appearing in Iris does NOT mean `vllm_get_model` completed the dummy init.

---

## Key Architectural Difference: Inference vs RL

### RL path (works)

```python
# inflight/worker.py:492
engine = AsyncLLM.from_engine_args(engine_args=engine_args, start_engine_loop=False)
```

- Uses `AsyncLLM.from_engine_args()` directly
- `start_engine_loop=False` — engine doesn't begin processing
- `VLLM_ENABLE_V1_MULTIPROCESSING=0` is set (via `vllm_inprocess.py` import)
- Everything runs **in-process** — no engine subprocess
- Weight injection via `collective_rpc` happens after engine creation

### Inference path (hangs)

```python
# vllm_async.py:187
async with build_async_engine_client_from_engine_args(
    engine_args, disable_frontend_multiprocessing=True,
) as engine_client:
```

- Uses `build_async_engine_client_from_engine_args()` (upstream helper)
- This spawns `EngineCore` as a **subprocess** (`EngineCore_DP0 pid=276`)
- `--disable-frontend-multiprocessing` only disables frontend MP, NOT engine core MP
- Does NOT set `VLLM_ENABLE_V1_MULTIPROCESSING=0`
- The engine subprocess does the full model loading pipeline

### The critical missing piece

The inference path does NOT set `VLLM_ENABLE_V1_MULTIPROCESSING=0`. This means:

1. `AsyncLLM.__init__` creates `EngineCoreClient.make_async_mp_client()`
2. This spawns an engine core subprocess via `fork()`
3. The forked subprocess inherits the parent's JAX/TPU state
4. Model creation inside the subprocess (via `torchax` + `jax.default_device(cpu)`)
   may conflict with the parent's JAX initialization

The RL path avoids this entirely because `VLLM_ENABLE_V1_MULTIPROCESSING=0` keeps
everything in a single process.

---

## What Codex Built (instrumentation infrastructure)

Across 13 iterations, Codex built a comprehensive monkeypatch-based timing system:

- **Pre-fork hooks** in `vllm_async.py` (`_install_early_async_startup_instrumentation`):
  wraps 30+ vLLM/TPU methods before the engine subprocess is forked, so the subprocess
  inherits the timing wrappers

- **Worker-extension hooks** in `inflight/worker.py` (`_apply_startup_timing_instrumentation`):
  wraps the same methods from within the worker extension (backup path)

- **Logging**: all timing writes to `sys.stderr` with `flush=True` (bypasses Python logging
  configuration issues in subprocesses)

- **Format**: `[marin-vllm-startup] START/END/FAIL {method} in {seconds}s pid={pid}`

- **Env vars added**: `MARIN_VLLM_STARTUP_TIMING=1`, `SKIP_JAX_PRECOMPILE=1`,
  `VLLM_WORKER_MULTIPROC_METHOD=fork`

- **Stress test improvements**: `--startup-timeout`, `--native-startup-failure-mode raise`

---

## Plan: Resolving the Hang

### Track 1: Quick validation — avoid the subprocess (HIGH PRIORITY)

**Hypothesis**: The hang is caused by the engine subprocess, not the model loading itself.
The RL path works because it runs in-process.

**Action**: Set `VLLM_ENABLE_V1_MULTIPROCESSING=0` in `_configure_async_vllm_environment()`
and test whether the engine runs in-process like RL does.

**Files to modify**:
- `lib/marin/src/marin/inference/vllm_async.py`: add
  `os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")` in
  `_configure_async_vllm_environment()`

**Risk**: `build_async_engine_client_from_engine_args()` may still create a subprocess
regardless. If so, we need Track 1b.

### Track 1b: Use `AsyncLLM.from_engine_args()` directly (if Track 1 doesn't work)

**Action**: Replace `build_async_engine_client_from_engine_args()` with
`AsyncLLM.from_engine_args(start_engine_loop=False)`, matching the RL pattern exactly.

This requires:
- Creating the engine in-process
- Manually starting the engine loop
- Wiring up `build_app` / `init_app_state` with the `AsyncLLM` instance
- Verifying `AsyncLLM` satisfies the engine client interface that `build_app` expects

**Files to modify**:
- `lib/marin/src/marin/inference/vllm_async.py`: rewrite `_run_async_server()` to use
  `AsyncLLM.from_engine_args(start_engine_loop=False)` + manual loop management

**Reference implementation**: `inflight/worker.py:490-494` + `async_bridge.py`

### Track 2: Instrument deeper inside `vllm_get_model` (PARALLEL)

This is what Codex is doing next. Even if Track 1 fixes the hang, understanding what
blocks in the subprocess is valuable for future debugging.

**Action**: Add timing around:
- `DummyModelLoader.load_model()`
- `initialize_model()` (model architecture creation)
- `initialize_dummy_weights()` or `process_weights_after_loading()`
- Individual `jax.device_put` or `torchax` tensor creation calls

**Goal**: Determine whether the hang is in model architecture creation, dummy weight init,
or post-processing.

### Track 3: Fix the fallback TPU lock collision (FOLLOW-UP)

When async-native times out, the subprocess still holds the TPU lock, so the fallback
subprocess (`vllm serve`) crashes with "TPU already in use by pid 276".

**Action**: Ensure the hung engine subprocess is killed before attempting fallback.

**Files to modify**:
- `lib/marin/src/marin/inference/vllm_async.py`: in `stop_async_vllm_server()`, force-kill
  the engine subprocess
- `lib/marin/src/marin/inference/vllm_server.py`: in `_activate_fallback_backend()`, ensure
  async runtime is fully stopped before starting fallback

---

## Verification Plan

### For Track 1 (`VLLM_ENABLE_V1_MULTIPROCESSING=0`):

1. Add the env var to `_configure_async_vllm_environment()`
2. Run locally: `pytest -q tests/vllm/test_vllm_inprocess_backend.py`
3. Run on Iris:
   ```
   uv run iris --config=lib/iris/examples/marin.yaml job run \
     --tpu v5p-8 --memory 24GB --region us-central1 \
     --extra tpu --extra vllm \
     --job-name vllm-async-no-mp-v1 --no-wait \
     -- python experiments/inference/exp_vllm_stress_test.py \
     --model gs://marin-us-central1/models/meta-llama--Llama-3-1-8B-Instruct--0e9e39f \
     --num-prompts 50 --max-concurrent 4 --max-tokens 128 \
     --max-model-len 4096 --mode native --enforce-eager \
     --startup-timeout 900 --native-startup-failure-mode raise
   ```
4. Check logs for:
   - No `EngineCore_DP0 pid=` lines (engine should run in parent process)
   - `AsyncLLM engine created in Xs` within 2-3 minutes
   - `Shard N/M injected` lines
   - `50/50 successful` stress test results

### For Track 1b (direct `AsyncLLM.from_engine_args`):

Same Iris command as Track 1, but check logs for:
- Engine running in pid=1 (parent process)
- No subprocess spawned
- Weight injection via `collective_rpc` succeeds

### Success criteria

- Engine startup completes in < 3 minutes (matching the old in-process path's ~2 min)
- Weight streaming + injection completes in < 3 minutes for 8B
- HTTP serving responds correctly for all 50 stress test prompts
- No TPU lock collision on failure

---

## Iris Job History

| Job | Run | Key Finding |
|-----|-----|-------------|
| `vllm-async-stress-8b-eager` | v1 | First attempt, hung at Pallas V1 backend |
| `debug-async-engine-v1` | debug | Confirmed 300s timeout at `build_async_engine_client` |
| `vllm-async-stress-8b-eager-v2` | v2 | `SKIP_JAX_PRECOMPILE=1` — no effect |
| `vllm-async-stress-8b-eager-v3` | v3 | `MARIN_VLLM_STARTUP_TIMING=1` added — no timing visible |
| `vllm-async-stress-8b-eager-v4` | v4 | stderr logging fix — `init_worker` visible, then silence for 1hr, TPU lock collision on fallback |
| `vllm-async-stress-8b-eager-v5` | v5 | Same as v4, confirmed pattern |
| `vllm-async-stress-8b-eager-v6` | v6 | Pre-fork instrumentation — `_init_executor` crash (monkeypatch bug) |
| `vllm-async-stress-8b-eager-v7` | v7 | Fixed instrumentation — `init_device` 24.8s, `load_model` entered |
| `vllm-async-stress-8b-eager-v8` | v8 | `tpu_runner.get_model` entered, hangs |
| `vllm-async-stress-8b-eager-v9` | v9 | `VllmModelWrapper.load_weights` entered, hangs |
| `vllm-async-stress-8b-eager-v10` | v10 | Import packaging mismatch — `cleanup_sharding` not found |
| `vllm-async-stress-8b-eager-v11` | v11 | Fixed imports — same `load_weights` boundary |
| `vllm-async-stress-8b-eager-v12` | v12 | Module alias instrumentation — `vllm_get_model` entered |
| `vllm-async-stress-8b-eager-v13` | v13 | **`vllm_get_model` hangs** — "random weights" logged, never returns |

---

## 2026-03-19 — Track 1 insufficient, pivoting to Track 1b

### Track 1: `VLLM_ENABLE_V1_MULTIPROCESSING=0` — already set, still hangs

Added `os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")` to
`_configure_async_vllm_environment()` in `vllm_async.py` + startup log message.

**However, import chain analysis revealed this env var was ALREADY being set:**
- `vllm_inprocess.py:38` sets it at module level
- `vllm_async.py` imports from `vllm_inprocess` at module level (lines 25-35)
- No `import vllm` happens at module level anywhere in the chain
- So the env var IS set before any vLLM import in all 13 debug runs

**Conclusion:** `VLLM_ENABLE_V1_MULTIPROCESSING=0` alone does not prevent
`build_async_engine_client_from_engine_args()` from spawning a subprocess.
The function may bypass this env var, or vLLM reads it through a different
mechanism. Either way, the env var approach is necessary but insufficient.

### Track 1b: Replace `build_async_engine_client_from_engine_args` with direct `AsyncLLM`

**Action:** Rewrite `_run_async_server()` to use `AsyncLLM.from_engine_args(
start_engine_loop=False)` directly, matching the RL pattern that works.

This requires:
1. Create `AsyncLLM` in-process (no subprocess possible)
2. Stream shards via `engine.engine_core.collective_rpc("update_weight")`
   (not `engine_client.collective_rpc` — different interface)
3. Wire up `build_app` + `init_app_state` with the `AsyncLLM` instance
4. Start the engine loop manually

**Reference:** `inflight/worker.py:490-494` uses `AsyncLLM.from_engine_args(
start_engine_loop=False)` + `engine.engine_core.collective_rpc_async()`

### Track 1b implementation

Rewrote `_run_async_server()` and related code:

1. **`_import_async_vllm_symbols()`**: replaced `build_async_engine_client_from_engine_args`
   import with `from vllm.v1.engine.async_llm import AsyncLLM`

2. **`_run_async_server()`**:
   - Replaced `async with build_async_engine_client_from_engine_args(engine_args, ...)` context
     manager with `AsyncLLM.from_engine_args(engine_args=engine_args, start_engine_loop=False)`
   - Added `engine.shutdown()` in `finally` block (replaces context manager cleanup)
   - `build_app` / `init_app_state` / `reset_prefix_cache` / `get_supported_tasks` all called
     on `engine` directly — `AsyncLLM` satisfies the same interface as the old engine client

3. **`_load_and_inject_streaming_async()`**:
   - Parameter renamed `engine_client` → `engine`
   - `engine_client.collective_rpc("update_weight", ...)` →
     `engine.engine_core.collective_rpc_async("update_weight", ...)`
   - This matches the RL pattern at `inflight/worker.py:546`

### Files modified
- `lib/marin/src/marin/inference/vllm_async.py`: full rewrite of engine creation path

### Validation
- `pytest -q tests/vllm/test_vllm_inprocess_backend.py` — 18/18 passed
- `./infra/pre-commit.py` — all checks passed

### Iris result: `/ahmed/vllm-async-inprocess-v2` — FAILED (same hang)

**Key log findings:**
- `VLLM_ENABLE_V1_MULTIPROCESSING='0'` visible in startup log — env var IS set
- vLLM WARNING: "VLLM_ENABLE_V1_MULTIPROCESSING is set to False" — vLLM sees it
- `(EngineCore_DP0 pid=276)` — subprocess STILL spawned despite env var + `from_engine_args`
- Hang at same location: `vllm_get_model` entered (20:45:43), never returns
- Last log before hang: "Using Pallas V1 backend" (20:45:49)
- Timeout after 900s: "Engine core initialization failed. Failed core proc(s): {}"

**Conclusion:** Neither `VLLM_ENABLE_V1_MULTIPROCESSING=0` nor
`AsyncLLM.from_engine_args(start_engine_loop=False)` prevents the engine core
subprocess in vLLM 0.13.2.post6. The env var controls frontend multiprocessing,
not the engine core process. The hang is in the subprocess's TPU model loading.

**RL path clarification:** The RL `SyncVLLMWrapper` at `inflight/worker.py:492`
also spawns this subprocess — but the RL path works because it runs as a
standalone training process with a clean JAX/TPU context, while the inference
path runs inside an Iris container where JAX/TPU may be initialized differently.
Need to investigate: what makes the subprocess JAX context different between
RL and inference.

---

## 2026-03-19 — Cross-logbook analysis: the subprocess IS the problem

### Combined evidence from 18 Iris runs (13 Codex + 2 Claude + v14-v16 Codex)

Codex's 16-run instrumentation campaign (documented in `codex_vllm_refactor.md`)
progressively narrowed the hang from "engine startup" down to specific model loader
functions. Combined with Claude's Track 1/1b results, the full picture is:

| Run(s) | Change | Result |
|--------|--------|--------|
| v1-v5 | baseline | hang after `init_worker` |
| v6 | pre-fork instrumentation | monkeypatch bug |
| v7 | fixed instrumentation | narrowed to `load_model` |
| v8 | deeper hooks | narrowed to `tpu_runner.get_model` |
| v9 | deeper hooks | narrowed to `VllmModelWrapper.load_weights` |
| v13 | module-alias hooks | narrowed to `vllm_get_model` (hangs after "random weights") |
| v14 | dummy-loader hooks | narrowed to `initialize_dummy_weights` |
| **v15** | **1800s timeout** | **`DummyModelLoader.load_weights` COMPLETES in 879s**, then stalls in `process_weights_after_loading` |
| v16 | `MODEL_IMPL_TYPE=auto` | switches to `flax_nnx` path, but `get_flax_model` also hangs (>1200s) |
| inprocess-v2 (Claude) | `AsyncLLM.from_engine_args` | subprocess still spawns, same hang |

### The key insight: it's NOT a deadlock, it's a slow subprocess

v15 proved the "hang" is actually forward progress — just absurdly slow:
- `initialize_model` (create architecture): 9.9s
- `DummyModelLoader.load_weights` (random weights): **879s** (~14.5 min)
- `process_weights_after_loading`: started but didn't complete in 900s more

**ALL of this work is wasted** because Marin immediately overwrites these dummy
weights via `collective_rpc("update_weight")` with real streamed shards.

### Why the subprocess is slow: JAX/TPU context inheritance

The engine subprocess is created via `fork()`. It inherits the parent's
half-initialized JAX/TPU context. Every JAX operation inside the subprocess
is massively slower because of this inherited state conflict.

**Evidence**: The exact same `load_format=dummy` path completes in ~2-3 minutes
with the synchronous `LLM()` in-process path (March 17 queue-based approach
for 8B and 70B). The subprocess makes the same work 5-10x slower.

### Why neither env var nor API change prevents the subprocess

- `VLLM_ENABLE_V1_MULTIPROCESSING=0`: vLLM acknowledges it ("set to False") but
  the TPU platform forces a subprocess regardless via `EngineCoreProc`
- `AsyncLLM.from_engine_args(start_engine_loop=False)`: `start_engine_loop`
  controls the request processing loop, NOT the engine core subprocess
- `build_async_engine_client_from_engine_args(disable_frontend_multiprocessing=True)`:
  disables frontend MP only, not engine core MP
- vLLM 0.13.2.post6 on TPU always spawns an engine core subprocess — this is
  baked into the TPU platform's executor selection

### Why the RL path "works"

The RL `SyncVLLMWrapper` at `inflight/worker.py:492` also spawns this subprocess.
But the RL path uses `load_format="auto"` (real weight loading, not dummy), and
it just tolerates the long startup (~15-30 min) because RL training sessions run
for hours. The RL path doesn't claim to be fast — it just works eventually.

### Two viable paths forward

**Option A: Return to `LLM()` in-process + improve serving quality**
- The March 17 queue-based approach (`InProcessVllmServerBackend`) completed
  startup in ~2-3 min because `LLM()` runs everything in the parent process
- Problems: custom FastAPI app, serialized requests, no continuous batching
- But: it works, it's fast, and serving quality can be improved incrementally
- Could add request batching, connection pooling, or async generation wrapper

**Option B: Patch vLLM-TPU fork to respect in-process engine**
- Modify the TPU platform's executor selection to honor `VLLM_ENABLE_V1_MULTIPROCESSING=0`
- This is the "right" fix but requires changes to the `vllm-tpu` package
- Would need to be upstreamed or maintained as a fork patch

More instrumentation inside the subprocess will not help — the problem is
structural (forked JAX/TPU context), not a specific function being slow.

---

## Key Files

| File | Role |
|------|------|
| `lib/marin/src/marin/inference/vllm_async.py` | Async-native backend + pre-fork instrumentation |
| `lib/marin/src/marin/inference/vllm_server.py` | Backend selection, `VllmEnvironment`, fallback |
| `lib/marin/src/marin/inference/vllm_inprocess.py` | Eligibility, bootstrap staging, shared helpers |
| `lib/marin/src/marin/rl/environments/inference_ctx/inflight/worker.py` | WorkerExtension, worker timing hooks, MRO fix |
| `lib/marin/src/marin/rl/environments/inference_ctx/async_vllm.py` | RL async engine context (working reference) |
| `experiments/inference/exp_vllm_stress_test.py` | Stress test script with `--startup-timeout` and `--native-startup-failure-mode` |
| `tpu-inference/.../vllm_model_wrapper.py` | `load_weights`, `vllm_get_model` call site |
| `tpu-inference/.../cleanup_sharding.py` | `shard_model_to_tpu` (never reached) |
