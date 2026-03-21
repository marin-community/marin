# Fix Inflight Weight Updates in Manual Mode

**Status**: planned
**Branch**: `on-demand-rl`
**Prerequisite for**: Phase 1 item 3 (eliminate trainer starvation)

## Problem

Inflight weight updates are disabled in manual (no-Ray) mode because the `AsyncvLLMInferenceContext` RPC path crashes the vLLM EngineCore subprocess after `update_weights`. This forces the sampler to block on synchronous weight sync every rollout cycle (~20s of dead time per cycle).

## How it works in Ray mode (where it's working)

The inflight system was built by Kevin and Chris for the Ray cluster. In Ray mode:

1. `RolloutWorker.__init__` spawns a background thread running `_sync_weights_loop()`
2. The loop calls `_sync_weights()` → `_apply_weight_update()` → `AsyncvLLMInferenceContext.reload_model()`
3. `reload_model()` calls `self.llm.update_weights()` on the `SyncVLLMWrapper`
4. `SyncVLLMWrapper` routes this through `AsyncBridge` → `collective_rpc_async("update_weight", ...)` → `WorkerExtension.update_weight()` on the vLLM worker
5. Meanwhile the main thread generates rollouts via the same `AsyncLLM` engine
6. The async event loop interleaves generation and weight updates — they yield to each other, never truly concurrent

This works because:
- vLLM's `AsyncLLM` engine handles request scheduling cooperatively
- `collective_rpc_async` is designed for this async interleaving
- Ray manages the process lifecycle cleanly

## What breaks in manual mode

The exact failure mode needs investigation. The comment says "RPC path crashes the EngineCore subprocess after update_weights." Despite `VLLM_ENABLE_V1_MULTIPROCESSING=0` being set, vLLM still spawns an `EngineCore_DP0` subprocess (visible in logs as `(EngineCore_DP0 pid=367)`).

Possible causes (need to verify with actual crash logs):

1. **Docker container differences**: The manual-mode Docker container may have different vLLM version/config than the Ray cluster image. The Ray cluster uses Iris-managed images; manual mode uses `levanter-ahmed` images built via `launch.py`.

2. **Process lifecycle**: In Ray mode, Ray manages worker processes and handles cleanup. In manual mode, the sampler runs as a bare Python process inside Docker. If the EngineCore subprocess crashes, there's no Ray supervisor to handle it.

3. **JAX initialization ordering**: In Ray mode, JAX is initialized by Levanter before vLLM starts. In manual mode for vLLM samplers, JAX initialization is deliberately avoided (to prevent deadlocks). The `WorkerExtension.update_weight()` calls `levanter_state_dict_to_nnx_state_on_cpu()` which may depend on JAX being initialized in a specific way.

4. **Timing/threading**: The background weight sync thread may interact differently with the `AsyncBridge` event loop in manual mode, e.g., thread contention or event loop lifecycle issues.

## Proposed approach: reproduce first, then decide

Before writing code, we need the actual crash traceback. Two options:

### Option A: Debug the AsyncvLLMInferenceContext crash

1. Re-enable `inflight_weight_updates=True` in manual mode on a test run
2. Capture the exact crash traceback from the sampler container
3. Fix the root cause (may be a one-line fix, may be deep in vLLM)

This preserves the original design where weight updates are truly concurrent with generation (best throughput).

### Option B: Background fetch, main-thread apply (bypass the crash)

If the crash is deep in vLLM internals and hard to fix, sidestep it:

Split weight sync into two phases:
1. **Fetch phase** (background thread): Poll coordinator + Arrow Flight receive → store `WeightUpdate` in a thread-safe slot
2. **Apply phase** (main thread): Before each generate call, check the slot. If a new update is available, call `reload_model()` synchronously using the working `vLLMInferenceContext`.

This avoids the `AsyncvLLMInferenceContext` entirely. Savings are modest (~10s/cycle from overlapping the fetch with generation) because the 10s `sync_weights` apply still blocks the main thread.

### Timeline comparison

**Current (synchronous, no overlap):**
```
[fetch 10s] → [apply 10s] → [generate 120s] → [fetch 10s] → [apply 10s] → ...
                                        Total per cycle: 140s
```

**Option A (true inflight, if we fix the crash):**
```
Main:       [generate ~~~~~120s~~~~~] → [generate ~~~~~120s~~~~~]
Background: [fetch+apply ~~20s~~]    → [fetch+apply ~~20s~~]
                                        Total per cycle: 120s (best case)
```

**Option B (background fetch only):**
```
Main:       [apply 10s] → [generate 120s] → [apply 10s] → [generate 120s]
Background: [fetch ~~~~10s~~~~]          → [fetch ~~~~10s~~~~]
                                        Total per cycle: 130s
```

## Recommended next step

**Try Option A first.** The inflight system was working in Ray mode — the crash may be something simple (env var, JAX init ordering, vLLM version mismatch). If we can get the crash traceback, we may fix it in an hour. If it's deep in vLLM, fall back to Option B.

To reproduce:
```bash
# On the sampler TPU, temporarily re-enable inflight:
# 1. Edit exp2039_rl_math500.py to remove the inflight_weight_updates=False override
# 2. Rebuild and redeploy the sampler container
# 3. Watch for the crash: docker logs -f levanter
```

## Code changes for Option B (if needed)

### 1. Add a pending-update slot to `RolloutWorker`

**File**: `lib/marin/src/marin/rl/rollout_worker.py`

Add to `__init__`:
```python
self._pending_weight_update: WeightUpdate | None = None
self._pending_update_lock = threading.Lock()
```

### 2. New background fetch method

**File**: `lib/marin/src/marin/rl/rollout_worker.py`

```python
def _fetch_weights_loop(self):
    """Background thread: fetch weights from Arrow Flight, store for main thread to apply."""
    logger.info("Starting background weight fetch loop")
    try:
        while self._running:
            try:
                update = self._transfer_client.receive_weights(self._policy_model)
            except Exception:
                logger.exception("Weight fetch failed")
                update = None

            if update and (update.is_done or update.is_failed):
                with self._pending_update_lock:
                    self._pending_weight_update = update
                break

            if update:
                with self._pending_update_lock:
                    self._pending_weight_update = update

                # If first weights, signal immediately
                if not self._first_weights_received.is_set():
                    self._first_weights_received.set()

            time.sleep(1.0)
    except Exception:
        logger.exception("Background weight fetch loop crashed")
    finally:
        logger.info("Background weight fetch loop exiting")
```

### 3. New main-thread apply method

**File**: `lib/marin/src/marin/rl/rollout_worker.py`

```python
def _apply_pending_weights(self) -> bool:
    """Check for and apply any pending weight update. Returns True if should stop."""
    with self._pending_update_lock:
        update = self._pending_weight_update
        self._pending_weight_update = None

    if update is None:
        return False

    if update.is_done or update.is_failed:
        reason = "Training complete" if update.is_done else "Trainer failed"
        logger.info("%s, stopping rollout worker", reason)
        self._running = False
        return True

    self._apply_weight_update(update)
    return False
```

### 4. Wire up in `__init__` and `run()`

**File**: `lib/marin/src/marin/rl/rollout_worker.py`

In `__init__`, change background thread target:
```python
if self.config.inflight_weight_updates:
    self.weight_transfer_thread = threading.Thread(
        target=self._fetch_weights_loop,  # was _sync_weights_loop
        ...
    )
```

In `run()`, add apply check in the inflight branch:
```python
if not self.config.inflight_weight_updates:
    self._sync_weights()
    if not self._running:
        break
else:
    if self._apply_pending_weights():
        break
```

### 5. Decouple from AsyncvLLMInferenceContext

**File**: `lib/marin/src/marin/rl/rollout_worker.py`

```python
# Always use sync context for vLLM
elif inference_type == "vllm":
    return vLLMInferenceContext(inference_config=inference_config)
```

### 6. Re-enable in manual mode

**File**: `experiments/exp2039_rl_math500.py`

Remove the `inflight_weight_updates=False` override.

## Files modified (Option B)

| File | What |
|------|------|
| `lib/marin/src/marin/rl/rollout_worker.py` | Background fetch loop, pending update slot, apply method, decouple from AsyncvLLMInferenceContext |
| `experiments/exp2039_rl_math500.py` | Remove `inflight_weight_updates=False` override |

## Thread safety analysis (Option B)

- `_pending_weight_update` protected by `_pending_update_lock`
- `_transfer_client.receive_weights()` called only from background thread
- `_policy_ctx.reload_model()` called only from main thread
- `_policy_ctx.generate()` called only from main thread
- No concurrent access to vLLM engine internals
