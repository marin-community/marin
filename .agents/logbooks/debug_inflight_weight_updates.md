# Debug: Inflight Weight Updates Crashing in Manual Mode

## Scope

- **Goal**: Get `inflight_weight_updates=True` working in manual (no-Ray) mode so the sampler can fetch weights in the background while generating rollouts.
- **Status**: **DONE** — fix landed, end-to-end validated on TPU. Pending soak test on next full 500-step run.
- **TPU**: `debug-weight-sync` (v5p-8, us-central1-a, deleted after investigation), `exp2039-nb-*` (validation)
- **Baseline**: Inflight weight updates work correctly in Ray cluster mode. They were disabled in manual mode due to a suspected crash.
- **Outcome**: The "crash" was a fork deadlock caused by a flawed test. Inflight updates work in manual mode when JAX isn't pre-initialized before the AsyncLLM fork. Fix: removed the override, added constraint docs, fixed first-weight timeout.

## Background

### How inflight weight updates work (Ray mode — working)

1. `RolloutWorker.__init__` spawns a background thread running `_sync_weights_loop()`
2. The loop calls `_sync_weights()` → `_apply_weight_update()` → `AsyncvLLMInferenceContext.reload_model()`
3. `reload_model()` serializes the state dict for RPC, then calls `SyncVLLMWrapper.update_weights()`
4. `SyncVLLMWrapper` routes through `AsyncBridge` → `engine.engine_core.collective_rpc_async("update_weight", ...)` → `WorkerExtension.update_weight()`
5. `WorkerExtension.update_weight()` deserializes the state dict, converts to NNX format, calls `model_runner._sync_weights()`
6. Meanwhile the main thread generates rollouts via the same `AsyncLLM` engine
7. The async event loop interleaves generation and weight updates cooperatively

### What's different in manual mode

| Aspect | Ray mode | Manual mode |
|--------|----------|-------------|
| Process supervision | Ray manages workers | Bare Docker container |
| JAX initialization | `levanter.initialize()` runs first | Deliberately skipped (deadlock avoidance) |
| Docker image | Iris-managed, version-pinned | `levanter-ahmed` via `launch.py` |
| Weight coordinator | Ray actor | Filesystem (GCS JSON) |
| vLLM engine | Same `AsyncLLM` + `SyncVLLMWrapper` | Same (should be identical) |

### The crash

The comment in `exp2039_rl_math500.py` says:
> "the AsyncvLLMInferenceContext RPC path crashes the EngineCore subprocess after update_weights"

**We do not have the actual crash traceback.** The first step is to reproduce and capture it.

### Key code paths

| File | Role |
|------|------|
| `rollout_worker.py:253-260` | Selects `AsyncvLLMInferenceContext` when `inflight_weight_updates=True` |
| `async_vllm.py:42-47` | `reload_model()` — serializes state dict, calls `llm.update_weights()` |
| `inflight/worker.py:173-276` | `SyncVLLMWrapper` — wraps `AsyncLLM`, routes through `AsyncBridge` |
| `inflight/worker.py:156-170` | `WorkerExtension.update_weight()` — deserializes, calls `_sync_weights()` |
| `inflight/worker.py:23-130` | MRO fix monkeypatch for `WorkerWrapperBase.init_worker` |
| `vllm.py:51` | `os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"` |
| `vllm.py:158-178` | `_get_llm_engine()` — selects `LLM` vs `SyncVLLMWrapper` |

---

## Experiment Log

### Experiment 0: Reproduce the crash

**Date**: (pending)
**Hypothesis**: Re-enabling `inflight_weight_updates=True` in manual mode will crash the sampler during the first weight update from the trainer. The crash traceback will tell us the root cause.

**Plan**:

#### Step 1: Prepare a minimal reproduction

Don't use a full training run. Instead, test inflight weight updates in isolation on the sampler TPU:

1. SSH into `exp2039-sampler` (or a fresh TPU)
2. Start a Docker container with the current image
3. Run a minimal script that:
   - Creates an `AsyncvLLMInferenceContext` with the same config as exp2039
   - Loads the model (Llama 3.1 8B Instruct with `load_format=dummy`)
   - Generates a few completions (verify baseline works)
   - Calls `reload_model()` with a dummy state dict
   - Checks if the engine is still alive
   - Generates more completions (verify post-update works)

This isolates the crash from Arrow Flight, coordinators, curriculum, etc.

**Minimal repro script** (to run inside the container):
```python
import os
import time
import logging

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from marin.rl.environments.inference_ctx.vllm import vLLMInferenceContextConfig, InferenceMode
from marin.rl.environments.inference_ctx.async_vllm import AsyncvLLMInferenceContext

MODEL = "meta-llama/Llama-3.1-8B-Instruct"

config = vLLMInferenceContextConfig(
    model_name=MODEL,
    max_model_len=2048,
    tensor_parallel_size=4,
    gpu_memory_utilization=0.90,
    load_format="dummy",
    enforce_eager=True,
)

logger.info("Creating AsyncvLLMInferenceContext...")
ctx = AsyncvLLMInferenceContext(inference_config=config)
ctx.start_server(None)

logger.info("Testing generation before weight update...")
# Use ctx.llm to generate a simple completion
from vllm import SamplingParams
params = SamplingParams(temperature=0.0, max_tokens=32)
result = ctx.llm.generate(["Hello, how are you?"], params)
logger.info("Pre-update generation OK: %s", result[0].outputs[0].text[:50] if result else "EMPTY")

logger.info("Creating dummy state dict for weight update...")
import numpy as np
# Get param names from the model to create a realistic state dict
# For dummy test, just try with an empty dict first to see if the RPC path itself crashes
try:
    logger.info("Calling reload_model with empty state dict...")
    ctx.reload_model(None, {})
    logger.info("reload_model returned successfully!")
except Exception:
    logger.exception("reload_model CRASHED")

logger.info("Testing generation after weight update...")
try:
    result = ctx.llm.generate(["What is 2+2?"], params)
    logger.info("Post-update generation OK: %s", result[0].outputs[0].text[:50] if result else "EMPTY")
except Exception:
    logger.exception("Post-update generation CRASHED")

logger.info("Done. Shutting down.")
ctx.shutdown()
```

**Command**:
```bash
gcloud alpha compute tpus tpu-vm ssh exp2039-sampler --zone=us-central1-a \
  --command='docker exec levanter python /dev/stdin' < repro_script.py
```

Or if the container isn't running:
```bash
gcloud alpha compute tpus tpu-vm scp repro_script.py exp2039-sampler:~ --zone=us-central1-a
gcloud alpha compute tpus tpu-vm ssh exp2039-sampler --zone=us-central1-a \
  --command='docker run --rm --privileged --net=host \
    -e VLLM_ENABLE_V1_MULTIPROCESSING=0 \
    -e TPU_BACKEND_TYPE=jax \
    -e PJRT_DEVICE=TPU \
    us-central1-docker.pkg.dev/hai-gcp-models/levanter/levanter-ahmed:latest \
    python -c "$(cat ~/repro_script.py)"'
```

**Expected outcome**: Either the script succeeds (meaning the crash is env-specific or timing-dependent), or it crashes with a traceback we can analyze.

**Record**: exact traceback, stderr output, any EngineCore subprocess logs.

**ACTUAL RESULT** (2026-03-20):

- **Test 1 (sync context): PASSED** — `vLLMInferenceContext` works perfectly. `reload_model()` with empty state dict succeeds, generation works before and after.
- **Test 2 (async context): DEADLOCKED** — `AsyncvLLMInferenceContext` deadlocks during engine initialization. Not a crash — a fork deadlock.

**Key observations from the logs:**

1. `VLLM_ENABLE_V1_MULTIPROCESSING=0` does NOT prevent `AsyncLLM` from forking. The sync `LLM` class respects this env var (Test 1 ran entirely in-process). But `AsyncLLM` (used by `SyncVLLMWrapper`) always spawns an `EngineCore_DP0` subprocess via `os.fork()`.

2. Python warned explicitly: `os.fork() was called. os.fork() is incompatible with multithreaded code, and JAX is multithreaded, so this will likely lead to a deadlock.`

3. The forked `EngineCore_DP0` (pid=2116) initialized up to `Prepared token paddings` then froze. It never reached model loading or KV cache init. The deadlock occurs at JAX mesh/device initialization in the forked child.

4. In our test, Test 1 initialized JAX in the parent process first (via the sync context). This "poisoned" subsequent forks because JAX's internal threads and locks don't survive fork. Even without Test 1, the manual-mode sampler initializes JAX indirectly through Arrow Flight client setup or other imports.

**Why it works in Ray mode:**
- In Ray mode, the `RolloutWorker` only uses `AsyncvLLMInferenceContext` — it never runs the sync context first.
- Ray doesn't use `os.fork()` for worker management — it uses `spawn` (new processes from scratch).
- The sampler in Ray mode skips `levanter.initialize()`, so JAX may not be fully initialized before vLLM starts.
- The `AsyncLLM` subprocess inherits a clean JAX state (or Ray's managed subprocess avoids the fork entirely).

**RETRACTION:** This conclusion was premature. The deadlock was caused by Test 1 (sync context) initializing JAX before Test 2's fork. In a real manual-mode sampler, the sync context is never created first. We need to test each hypothesis in isolation to find the real root cause.

---

## Hypotheses

### H1: Test artifact — JAX pre-initialization from sync context caused the deadlock

**Theory:** The Experiment 0 deadlock was an artifact of running the sync `vLLMInferenceContext` (Test 1) before the async context (Test 2). Test 1 fully initialized JAX on the TPU, poisoning the fork for Test 2. In a real manual-mode sampler, the sync context is never created, so JAX might not be initialized before the `AsyncLLM` fork.

**If true:** The async engine initializes fine in a fresh process. The real bug is something else (H3 or H4).
**If false:** Something else initializes JAX before the fork (see H2).

### H2: Indirect JAX initialization from import chain triggers the fork deadlock

**Theory:** Even without a sync context, the manual-mode sampler's import chain or startup code initializes JAX before the `AsyncLLM` fork. Candidates:
- `import jax` at module level in `rollout_worker.py` (but `import jax` alone is lazy — doesn't initialize the runtime)
- `import haliax` which imports JAX transitively
- `RolloutTracker` (wandb init) touching JAX indirectly
- Arrow Flight client setup
- `levanter.utils.jax_utils.barrier_sync` import

**If true:** The fork deadlock IS the production bug, just triggered by imports rather than explicit `jax.devices()`. Need Option B.
**If false:** The import chain is safe. The bug is at the RPC layer (H3/H4).

### H3: Engine initializes fine, but `collective_rpc_async("update_weight")` crashes the subprocess

**Theory:** The original comment says "crashes the EngineCore subprocess after update_weights". The engine might initialize and generate successfully, but the actual weight update RPC crashes the subprocess. Possible causes:
- Serialization failure in the RPC path (numpy arrays corrupted during pickling)
- The `WorkerExtension.update_weight()` method crashing during `levanter_state_dict_to_nnx_state_on_cpu()` or `_sync_weights()`
- The subprocess dying after weight update completes (cleanup/state corruption)

**If true:** We need to fix the RPC serialization or weight application path. Or fall back to Option B.
**If false:** Weight updates work fine through the RPC path (good news).

### H4: MRO monkeypatch doesn't propagate correctly to the forked subprocess

**Theory:** The `WorkerExtension` class is injected into the worker via a dynamic MRO fix at import time. In the forked subprocess, the monkeypatch might not be applied correctly, causing `update_weight` to not be found or to misbehave.

**Evidence against:** Experiment 0 logs show `"Created extended worker class TPUWorkerWithExtension with WorkerExtension for collective_rpc calls ['update_weight']"` in the subprocess — the patch WAS applied. But we never got far enough to test if the method actually works.

**If true:** The `update_weight` RPC call fails with method-not-found or MRO error.
**If false:** The monkeypatch works correctly.

---

## Experiment Plan

**Critical design rule:** Each experiment runs in a **completely fresh subprocess** to avoid JAX state contamination between tests. The test runner spawns `python -c "..."` for each experiment.

### Experiment 1: Async context in fresh process, NO prior JAX

**Tests:** H1 vs H2
**What:** Create `AsyncvLLMInferenceContext` and generate a completion. No sync context, no explicit `jax.devices()`, nothing that touches JAX first.
**Timeout:** 180s (model init + XLA compilation takes ~60s normally; deadlock means no progress after ~30s)

| Outcome | Interpretation |
|---------|---------------|
| PASS (engine initializes, generates) | **H1 confirmed** — my Experiment 0 was a test artifact. The fork works when JAX isn't pre-initialized. Proceed to Exp 3. |
| TIMEOUT (deadlock) | **H2 confirmed** — even without explicit JAX init, the import chain or vLLM setup triggers JAX before fork. AsyncLLM is unusable in this environment. Proceed to Option B. |

### Experiment 2: Async context AFTER explicit `jax.devices()` (control)

**Tests:** Confirms H1 mechanism
**What:** Call `jax.devices()` to fully initialize JAX, then create `AsyncvLLMInferenceContext`.
**Timeout:** 120s

| Outcome | Interpretation |
|---------|---------------|
| TIMEOUT (deadlock) | **Expected** — confirms the fork-after-JAX deadlock mechanism from Experiment 0. |
| PASS | Surprising — would mean the deadlock is non-deterministic or our Experiment 0 finding was wrong. |

### Experiment 3: Async context + `update_weights` with empty dict

**Tests:** H3, H4
**Prerequisite:** Experiment 1 passed
**What:** Initialize async engine, generate, call `reload_model({})`, generate again.
**Timeout:** 180s

| Outcome | Interpretation |
|---------|---------------|
| PASS | Weight update RPC works with empty dict. H3 partially rejected (empty dict is trivial). Proceed to Exp 4. |
| FAIL (crash/error) | **H3 or H4 confirmed** — the RPC path itself is broken. Capture traceback. |

### Experiment 4: Async context + `update_weights` with real weights

**Tests:** H3, H4 with realistic data
**Prerequisite:** Experiment 3 passed
**What:** Initialize async engine, generate, call `reload_model()` with a real numpy state dict (embed_tokens layer), generate again.
**Timeout:** 300s (real weight transfer takes time)

| Outcome | Interpretation |
|---------|---------------|
| PASS | **All hypotheses rejected for the RPC path!** Inflight updates work. The production bug must be environmental (image version, timing, etc.). |
| FAIL (crash/error) | **H3 confirmed** — real weight data breaks the RPC. Capture traceback to see if it's serialization, NNX conversion, or sync_weights. |

### Experiment 5: Full RolloutWorker import chain + async context

**Tests:** H2 more precisely
**What:** Import everything that `rollout_worker.py` imports at module level (`jax`, `haliax`, `levanter`, `wandb`), create a `RolloutTracker`, THEN create `AsyncvLLMInferenceContext`.
**Timeout:** 180s

| Outcome | Interpretation |
|---------|---------------|
| PASS | **H2 rejected** — the RolloutWorker import chain does NOT initialize JAX. Inflight is viable in production if Exp 3/4 also passed. |
| TIMEOUT (deadlock) | **H2 confirmed** — one of the imports triggers JAX init. Need to bisect which import. |

---

## Decision matrix

| Exp 1 | Exp 2 | Exp 3 | Exp 4 | Exp 5 | Diagnosis | Action |
|-------|-------|-------|-------|-------|-----------|--------|
| PASS | TIMEOUT | PASS | PASS | PASS | Everything works. Original bug was env-specific. | Re-enable `inflight_weight_updates=True` and test in production. |
| PASS | TIMEOUT | PASS | PASS | TIMEOUT | Import chain poisons JAX. Engine + RPC work in isolation. | Fix import ordering (defer JAX imports), OR Option B. |
| PASS | TIMEOUT | PASS | FAIL | — | RPC works for empty dict but crashes with real weights. | Fix serialization/weight application, OR Option B. |
| PASS | TIMEOUT | FAIL | — | — | RPC path fundamentally broken. | Option B (background fetch, sync apply). |
| TIMEOUT | TIMEOUT | — | — | — | AsyncLLM fork always deadlocks in this env. | Option B (background fetch, sync apply). |

---

## Results Summary Table

| Experiment | Date | Result | Interpretation |
|-----------|------|--------|---------------|
| 0: Naive repro (sync then async) | 2026-03-20 | DEADLOCK | **Flawed** — sync context initialized JAX before async fork. Not representative of production. |
| **Run 1** (subprocesses, no cleanup) | | | |
| 1: Async only, no prior JAX | 2026-03-20 | **PASS** | Engine initializes and generates fine. **H1 confirmed.** |
| 2: Async after jax.devices() (control) | 2026-03-20 | **TIMEOUT** | Deadlock confirmed when JAX pre-initialized. |
| 3: update_weights empty dict | 2026-03-20 | FAILED | **INVALID** — stale EngineCore from EXP1 held TPU devices (`/dev/vfio/2: Device or resource busy`). |
| 4: update_weights real weights | 2026-03-20 | SKIPPED | — |
| 5: Full import chain + async | 2026-03-20 | FAILED | **INVALID** — same stale EngineCore device lock. |
| **Run 2** (separate Docker containers per experiment) | | | |
| 1: Async only, no prior JAX | 2026-03-20 | **PASS** | Reproduces Run 1 result. |
| 2: Async after jax.devices() (control) | 2026-03-20 | **TIMEOUT** | Reproduces. |
| 3: update_weights empty dict | 2026-03-20 | **PASS** | RPC path works with empty state dict. |
| 4: update_weights real weights | 2026-03-20 | FAILED | **Test bug** — used `np.bfloat16` which doesn't exist in numpy. Not a real failure. |
| 5: Full import chain + async | 2026-03-20 | **PASS** | RolloutWorker imports do NOT trigger JAX init. **H2 rejected.** |
| **Run 3** (EXP4 fixed with ml_dtypes.bfloat16) | | | |
| 4-fixed: update_weights real weights | 2026-03-20 | **PASS** | 1 GB bfloat16 weight transferred via RPC in 3.8s. Post-update generation works. **H3 rejected.** |

### Hypothesis verdicts

| Hypothesis | Verdict | Evidence |
|-----------|---------|----------|
| H1: Test artifact (JAX pre-init from sync context) | **CONFIRMED** | EXP1 passes when async runs alone. EXP2 deadlocks when JAX is pre-init'd. |
| H2: Import chain triggers JAX init | **REJECTED** | EXP5 passes — all rollout_worker imports + RolloutTracker do NOT trigger JAX init. |
| H3: `collective_rpc_async("update_weight")` crashes | **REJECTED** | EXP3 (empty) and EXP4-fixed (1 GB bfloat16) both pass. RPC works. |
| H4: MRO monkeypatch fails | **REJECTED** | WorkerExtension.update_weight is found and executes correctly. |

### Conclusion

**The inflight weight update system works in manual mode.** The original failure was likely caused by something initializing JAX before the `AsyncLLM` fork — possibly a test or debug session that ran the sync context first, or an earlier version of the code that initialized JAX earlier.

**The only constraint is: JAX must not be fully initialized before `AsyncvLLMInferenceContext` is created**, because `AsyncLLM` forks an EngineCore subprocess and fork+JAX deadlocks. In the current manual-mode code path (`RolloutWorker.__init__` → `create_inference_context`), this constraint is satisfied.

**Next step:** Simply re-enable `inflight_weight_updates=True` in manual mode by removing the override in `exp2039_rl_math500.py`. No code changes needed to the inflight machinery itself.

---

## Infrastructure notes

- Debug TPU `debug-weight-sync` (v5p-8, us-central1-a, on-demand) was provisioned for this investigation and **deleted** after completion.
- Test scripts: `on-demand-rl-scripts/debug_inflight.py` (subprocess-based), `debug_inflight_runner.sh` (container-based), `debug_exp4_fixed.py`
- Docker image used: `us-central1-docker.pkg.dev/hai-gcp-models/levanter/levanter-ahmed:1773961071` (commit `4fed32b32`)
- vLLM version in image: `0.13.2.post6`
- JAX version in image: `0.8.0`

## Lessons learned

1. **Isolate experiments properly.** Our first two test runs gave invalid results because experiments contaminated each other — stale EngineCore subprocesses held TPU devices (`/dev/vfio/*: Device or resource busy`). Each experiment needs its own Docker container, not just its own Python subprocess.

2. **Test bugs can masquerade as real bugs.** EXP4 "failed" because the test used `np.bfloat16` (doesn't exist in numpy). This could have been misdiagnosed as "the RPC path can't handle bfloat16 weights." Always read the actual traceback before drawing conclusions.

3. **Don't trust initial conclusions.** The Experiment 0 deadlock looked definitive — "AsyncLLM always forks, fork+JAX deadlocks, fundamental incompatibility." But the deadlock was caused by the test running the sync context first (initializing JAX), which doesn't happen in production. The retraction and hypothesis-driven re-investigation found the opposite conclusion.

4. **The fork+JAX constraint is real but narrow.** `AsyncLLM` does always fork an EngineCore subprocess. If JAX is fully initialized (runtime started, devices opened) before that fork, it WILL deadlock. But `import jax` alone doesn't trigger initialization — it's lazy. The current `RolloutWorker` code path creates the inference context before anything touches JAX devices, so the constraint is satisfied.

## Post-investigation review (Codex critique, 2026-03-20)

An independent review flagged three gaps:

1. **The fix was never landed.** The logbook concluded "just remove the override" but the code still has `inflight_weight_updates=False` and the stale "RPC path crashes" comment in `exp2039_rl_math500.py`. The investigation found the answer but didn't apply it.

2. **The JAX-before-fork invariant isn't encoded in code.** The conclusion identifies a brittle constraint (JAX must not be initialized before `AsyncvLLMInferenceContext` is created) but there's no guard, assertion, or even a comment at `create_inference_context()` in `rollout_worker.py`. A future import-order change could silently reintroduce the deadlock.

3. **No end-to-end validation.** The experiments proved the individual components work (engine init, generation, weight update RPC, import chain). They did not test the full `RolloutWorker` with the background sync thread, filesystem coordinator, and long-running lifecycle. The logbook leaves deployment verification pending.

**Assessment:** The diagnosis is correct. The minimal fix is sufficient — no architectural redesign needed. But the fix must actually be applied, the constraint must be documented in code, the debug scripts should be cleaned up, and it needs a real end-to-end run to verify.

## Action items

- [x] Remove `inflight_weight_updates=False` override and stale comment in `exp2039_rl_math500.py`
- [x] Add a comment documenting the JAX-before-fork constraint near `create_inference_context()` in `rollout_worker.py`
- [x] Delete one-off debug scripts (`debug_inflight.py`, `debug_inflight_runner.sh`, `debug_exp4_fixed.py`)
- [x] Fix first-weight timeout (default 20min when `max_weight_transfer_wait_time=0` and inflight enabled)
- [x] End-to-end validation: `exp2039-inflight-test2-20260320-192417` on nb TPUs, 3 training steps
- [ ] Option B plan in `.agents/projects/inflight_weight_updates.md` is no longer needed but kept for reference

### End-to-end validation (2026-03-20)

**Run**: `exp2039-inflight-test2-20260320-192417` on `exp2039-nb-trainer` / `exp2039-nb-sampler` (v5p-8, us-central1-a)

**Config**: 3 training steps, `inflight_weight_updates=True`, seed=42

**Results**:
- Sampler created `AsyncvLLMInferenceContext` with background weight sync thread — no deadlock
- Background thread received initial weights (step -1) while engine was compiling XLA graphs
- Trainer completed 3 steps, published weights at steps 0, 1, 2
- Sampler received weights from steps 0 and 1 **during generation** (inflight!) — interleaved logs confirm:
  - 19:32:00 GENERATE step=2 → 19:32:44 **Received weights step 0** → 19:34:01 WRITE_ROLLOUT step=2
  - 19:34:02 GENERATE step=3 → 19:34:40 **Received weights step 1** → during generation
- Trainer called `mark_completed()` → coordinator status `completed`
- Sampler saw completion signal, exited gracefully, wandb finalized
- Both wandb runs show as "finished"

**Issue found and fixed**: First attempt failed because `max_weight_transfer_wait_time=0` meant the inflight first-weight wait expired immediately. Fixed by defaulting to 1200s (20 min) for the first-weight wait when inflight is enabled and `max_weight_transfer_wait_time=0`.
