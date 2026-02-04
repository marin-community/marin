# Multi-Host Inference for SimPO Training

## Goal

Enable inference evaluation during multi-host training (e.g., v5p-64 with 8 hosts) so we can see model generations during training.

## Key Discovery

**The model is globally sharded across all hosts.** With Levanter's default data-parallel setup on v5p-64:
- 8 hosts, each with 4 TPU chips (total 32 chips, but JAX sees 64 devices due to megacore)
- The model weights are sharded across ALL hosts
- Each host only has 1/8th of the model weights

This means **we cannot run inference on a single host** - all hosts must participate in every forward pass.

## Failed Approaches

### Approach 1: Leader-only inference with barriers
```python
# NON-WORKING: Leader runs inference, others wait
if not is_leader:
    barrier_sync()  # Wait for leader
    return
# Leader does inference alone...
barrier_sync()  # Release others
```
**Why it failed:** The model's `initial_cache` and `decode` methods use `hax.named_jit` which requires all hosts to participate in collective operations. Leader hangs waiting for other hosts.

### Approach 2: Local devices only
```python
# NON-WORKING: Use only local devices
inference_devices = jax.local_devices()
engine_config = InferenceEngineConfig(..., devices=inference_devices)
```
**Why it failed:** Same issue - the model arrays are globally sharded, so any operation on them requires global coordination.

## Working Solution

**All hosts participate in inference together.** Every host:
1. Creates the inference engine
2. Runs generation with the same prompts
3. Gets the same results

Only process 0 logs results to wandb (to avoid duplicate logging).

### Code Changes

#### 1. `lib/levanter/src/levanter/inference/engine.py`

Added `devices` parameter to `InferenceEngineConfig` (for future single-host use cases):
```python
@dataclass(frozen=True)
class InferenceEngineConfig:
    # ... existing fields ...
    devices: Optional[Sequence] = None
    """Devices to use for inference. If None, uses jax.devices()."""
```

Updated `_available_hbm_budget_bytes()` to accept optional devices parameter.

#### 2. `lib/levanter/src/levanter/main/train_simpo.py`

The inference callback now:
- All processes run inference (no barriers for isolation)
- Uses `jax.local_devices()[0]` for memory stats (global devices not addressable)
- Only process 0 logs to wandb
- Does NOT call `jax.clear_caches()` (this corrupts TPU state!)
- Calls `barrier_sync()` at end to ensure all processes finish before resuming training

```python
def inference_eval_callback(step: StepInfo):
    # All processes participate - model is globally sharded

    # Memory stats use local device (global devices not addressable for memory_stats)
    memory_device = jax.local_devices()[0]

    # Create engine with global mesh (no devices= override)
    engine_config = InferenceEngineConfig(
        max_seq_len=max_seq_len,
        max_seqs=len(prompts),
        # ... other settings
        # devices=None means use all devices (required for sharded model)
    )

    # All processes run generation
    engine = InferenceEngine.from_model_with_config(model, tokenizer, engine_config)
    result = engine.generate(requests)

    # Only leader logs results
    if jax.process_index() == 0:
        levanter.tracker.log({"inference_eval/samples": samples_table}, step=step.step)

    # Cleanup
    del engine
    gc.collect()
    # NOTE: Do NOT call jax.clear_caches() - it corrupts TPU state!

    # Sync before returning to training
    if jax.process_count() > 1:
        barrier_sync(timeout=120.0)
```

#### 3. `lib/levanter/config/simpo_ultrafeedback_llama3_8b_v5p_64_inference.yaml`

New config for v5p-64 with inference enabled:
```yaml
trainer:
  train_batch_size: 256
  per_device_parallelism: -1
  per_device_eval_parallelism: -1

inference_eval:
  enabled: true
  eval_every: 2        # Every 2 steps for debugging (use 10-20 for production)
  max_new_tokens: 64
  temperature: 0.7
  max_seq_len: 512     # Keep short to minimize KV cache
  hbm_utilization: 0.2 # Low to leave room for training state
  max_pages: 64        # Cap pages to avoid VMEM exhaustion
  allow_multihost: true
```

## Current Status

**Inference works:** All 8 hosts successfully generate 320 tokens in ~38 seconds.

## Debug Attempts

### ATTEMPT #1: Remove jax.clear_caches()
**Hypothesis:** `jax.clear_caches()` was corrupting TPU state and invalidating training JIT functions.

**Change:** Removed both `jax.clear_caches()` calls from the inference callback.

**Result:** ❌ FAILED
```
jax.errors.JaxRuntimeError: FAILED_PRECONDITION: The program continuator has halted unexpectedly.
```
Error occurs at `state.step % h.every` when training tries to resume after callback exits.

---

### ATTEMPT #2: Add TPU-level sync with sync_global_devices
**Hypothesis:** `barrier_sync()` only does host-level synchronization. The TPU execution context might be out of sync across hosts, causing the training program to fail when it resumes.

**Change:** Added `jax.experimental.multihost_utils.sync_global_devices("inference_complete")` before the host-level barrier sync.

**Result:** ❌ FAILED - Error happens INSIDE sync_global_devices itself
```
jax.errors.JaxRuntimeError: FAILED_PRECONDITION: The program continuator has halted unexpectedly.
```
TPU was already corrupted before we even tried to sync.

---

### ATTEMPT #3: Block until ready on inference results + remove sync_global_devices
**Hypothesis:** The inference engine's async operations may not complete cleanly. The TPU program continuator fails because there are pending/incomplete operations. Using `jax.block_until_ready()` on inference results should force all pending TPU operations to complete before we return to training.

**Change:**
1. Remove sync_global_devices (fails when TPU already corrupted)
2. Add explicit `jax.block_until_ready()` on inference result tokens before cleanup
3. Keep barrier_sync for host coordination

**Result:** ❌ FAILED - block_until_ready completed successfully (logs show "All inference results materialized") but same error when training resumes:
```
jax.errors.JaxRuntimeError: FAILED_PRECONDITION: The program continuator has halted unexpectedly.
```

---

### ATTEMPT #4: Re-warm training context by touching training state arrays
**Hypothesis:** The inference JIT creates new XLA executables that somehow corrupt/invalidate the training execution context. By explicitly accessing training state arrays with a collective operation AFTER inference but BEFORE returning, we can re-establish the proper TPU execution context for training.

**Change:** After inference cleanup, explicitly call `jax.block_until_ready()` on the training model weights to ensure the training context is re-established before returning to the training loop.

**Result:** ❌ FAILED - **Different error!**
```
jax.errors.JaxRuntimeError: INTERNAL: Core halted unexpectedly: INTERNAL: Accelerator device halted prematurely
An unexpected peer shows up in the launch group with a different launch id than the current group leader.
If using multi-controller backends, non-determinism of 1) model code or 2) XLA compiler may also cause this
```
This indicates hosts compiled different XLA programs - XLA non-determinism across hosts.

---

### ATTEMPT #5: Ensure all hosts execute identical code paths
**Hypothesis:** The error "unexpected peer with different launch id" indicates XLA non-determinism. This could be caused by:
1. Different hosts taking different code paths (e.g., only leader does wandb logging)
2. Inference JIT creating collectives that get assigned different launch IDs

**Change:**
1. Remove the "re-warm training context" code (might trigger unexpected compilations)
2. Make ALL hosts execute `tokenizer.decode()` on results (moved outside `if is_leader:`)
3. Only leader does actual wandb logging after decoding

**Result:** ❌ FAILED - Same error as #4
```
Location: trainer.py:474 in train_step
  hooks_this_time = any(state.step % h.every == 0 for h in self.hooks.jit_hooks)
  -> jax._src.array.py:298 __bool__ -> _value -> _single_device_array_to_np_array_did_copy()

jax.errors.JaxRuntimeError: INTERNAL: Core halted unexpectedly: INTERNAL: Accelerator device halted prematurely
Node 0 halted unexpectedly at tag:pc TensorCoreSequencer:1:0x9e (from TensorCoreSequencer:1:0x1fe): scheckne:
***************
An unexpected peer shows up in the launch group with a different launch id than the current group leader.
If using multi-controller backends, non-determinism of 1) model code or 2) XLA compiler may also cause this,
enable HLO dump for all workers and check:
  1) if before_optimizations.txt are all the same
  2) if after_optimizations.txt are all the same
no HLO mapping
=== Source Location Trace: ===
learning/45eac/tpu/runtime/hal/internal/tpu_program_termination_validation.cc:180
```

**Key insight:** The error is NOT about Python code paths - it's about XLA compilation non-determinism. The inference JIT is compiling different programs on different hosts.

---

### ATTEMPT #6: Add barrier BEFORE inference starts
**Hypothesis:** The XLA non-determinism might be caused by hosts entering the inference callback at different times/states. If hosts are in different states when JIT compilation occurs, they might compile different programs. By adding a barrier at the START of the callback (not just the end), we ensure all hosts enter together and compile in sync.

**Change:** Add `barrier_sync()` immediately after checking `step.step == 0`, BEFORE any inference code runs. This ensures all 8 hosts are synchronized before any JIT compilation happens.

**Result:** PENDING

---

## Error Summary

| Error | Meaning |
|-------|---------|
| `FAILED_PRECONDITION: The program continuator has halted unexpectedly` | TPU execution context invalid - training's compiled program can't continue |
| `unexpected peer shows up in launch group with different launch id` | Hosts compiled different XLA programs - non-determinism across hosts |

## How to Test

```bash
cd /Users/ahmed/code/marin3/lib/levanter

python infra/launch.py --foreground --zone us-central1-a \
  --tpu_name simpo_worker --tpu_type v5p-64 --capacity_type on-demand -- \
  python src/levanter/main/train_simpo.py \
  --config_path config/simpo_ultrafeedback_llama3_8b_v5p_64_inference.yaml \
  --trainer.ray.auto_start_cluster false
```

**Success criteria:**
1. Step 2 inference completes (look for "Generation complete" from all 8 processes)
2. Training continues to step 3, 4, etc.
3. Step 4 inference completes
4. No JAX errors about "unexpected peer" or "different launch id"

## Debug Output

The callback includes debug prints like:
```
[INFERENCE DEBUG][Process 0][Step 2] >>> CALLBACK ENTERED
[INFERENCE DEBUG][Process 0][Step 2] <<< Generation complete in 37.83s, 320 tokens
[INFERENCE DEBUG][Process 0][Step 2] Syncing all processes before returning to training...
[INFERENCE DEBUG][Process 0][Step 2] Barrier passed, resuming training
[INFERENCE DEBUG][Process 0][Step 2] <<< CALLBACK EXITING
```

## Files Modified

| File | Changes |
|------|---------|
| `lib/levanter/src/levanter/inference/engine.py` | Added `devices` param to config |
| `lib/levanter/src/levanter/main/train_simpo.py` | Multi-host inference callback |
| `lib/levanter/config/simpo_ultrafeedback_llama3_8b_v5p_64_inference.yaml` | v5p-64 config |

## Future Improvements

1. **Remove debug prints** once stable
2. **Increase `eval_every`** to 10-20 for production runs
3. **Consider checkpointing** - save/restore model state around inference if issues persist

---

## Multi-Host Sampling (Status Update)

**What we implemented**

- Added a new multi-host sampler entrypoint (`src/levanter/main/sample_lm_multihost.py`) that:
  - Runs inference on all hosts with the globally sharded model.
  - Broadcasts tokenized prompts from leader for determinism.
  - Validates engine sizing constraints and prevents partial admission.
  - Logs throughput, samples, and HBM memory to W&B (leader only).
- Added configs:
  - `config/sampler/sample_llama8b_multihost.yaml` (W&B enabled, small run)
  - `config/sampler/sample_llama8b_multihost_real.yaml` (100 prompts, 4096 max_new_tokens)

**What worked**

- Multi-host sampling ran end-to-end with W&B logging for throughput and samples.

**Current focus: OOM debugging**

- We are **now debugging OOM issues** in the “realistic” config:
  - The TPU **scoped vmem** failure arises inside ragged paged attention prefill.
  - This is driven by **prefill shapes** (e.g., `max_prefill_size`, batch size).
- Mitigation in progress:
  - Added **batching** (`max_prompts_per_batch`) so 100 prompts run in smaller chunks.
  - Reduced engine sizes (max_seqs, max_prefill_size, max_pages) to fit vmem.
  - Latest `sample_llama8b_multihost_real.yaml` uses:
    - `max_prompts_per_batch: 8`
    - `max_prefill_size: 1024`
    - `max_seqs: 8`, `max_seqs_in_prefill: 8`
    - `max_pages: 288`

**Next steps**

- If OOM persists: drop `max_prompts_per_batch` to 4 and `max_prefill_size` to 768.
- Optionally add auto-batching based on prompt lengths to cap prefill size.


COMMAND TO RUN!!
python infra/launch.py --foreground --zone us-central1-a \
    --tpu_name simpo_worker --tpu_type v5p-64 --capacity_type on-demand -- \
    uv run src/levanter/main/sample_lm_multihost.py \
    --config_path config/sampler/sample_llama8b_multihost_real.yaml
