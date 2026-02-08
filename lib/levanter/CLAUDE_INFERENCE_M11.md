# M11 Analysis: Train -> Inference -> Train TPU Runtime Crash

Status: RESOLVED via subprocess isolation (2026-02-08)

## Resolution

The subprocess isolation approach (Strategy 1) was implemented and **successfully validated on v5p-32 (4 hosts)**. All 3 phases completed without TPU runtime errors:

- **Phase 1**: Train 2 steps (train_simpo.py), save HF checkpoint to GCS
- **Phase 2**: Host-data-parallel inference (sample_lm_multihost.py) from HF checkpoint, 128 prompts x 2048 tokens, ~286 tok/s per host
- **Phase 3**: Resume training (train_simpo.py) from HF checkpoint, 5 more steps, loss=1.14

### Files created

- `scripts/m11_interleaved_train_infer.py` — Orchestration script (runs 3 phases as subprocesses)
- `config/simpo_ultrafeedback_llama3_8b_v5p_32_m11_subprocess_phase1.yaml` — Phase 1 training config
- `config/sampler/sample_llama8b_v5p_32_m11_subprocess_phase2.yaml` — Phase 2 inference config
- `config/simpo_ultrafeedback_llama3_8b_v5p_32_m11_subprocess_phase3.yaml` — Phase 3 resume training config

### How to run

```bash
python infra/launch.py --foreground --zone us-central1-a \
    --tpu_name simpo_worker --tpu_type v5p-32 --capacity_type on-demand -- \
    python scripts/m11_interleaved_train_infer.py
```

---

## Problem Statement

M11 aims to implement an **in-process interleaved workflow** in `src/levanter/main/train_simpo.py`:

1. Train for 2 steps on a multi-host TPU (v5p-32, 4 hosts).
2. Pause training, run multi-host inference on 128 prompts with `max_new_tokens=2048`.
3. Resume training for 5 more steps (total 7).

**The code is fully implemented** (configs, callback, tests all pass locally). The workflow is **blocked by a TPU runtime crash** when transitioning from completed inference back into training. This crash has reproduced across 13+ runs on v5p-32, across both `host_data_parallel` and `global_mesh` inference modes.

### Crash Signature

Two related error families appear, always at the point where training tries to resume after inference:

```
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

```
jax.errors.JaxRuntimeError: FAILED_PRECONDITION: The program continuator has halted unexpectedly.
```

**Location in code where crash manifests:** after the inference callback returns and training attempts its next operation, typically at `trainer.py:474` (`state.step % h.every`) or inside `sync_global_devices`.

### Key Facts

- **M10 standalone host-DP inference works perfectly.** The crash is specific to running inference *within* a training process.
- **Both `host_data_parallel` and `global_mesh` inference modes crash identically** when embedded in training. This rules out local mesh creation as the root cause.
- **Inference itself completes successfully.** Decode runs to completion, tokens are generated, results are materialized. The crash happens *after* inference finishes and training tries to resume.
- **v5p-64 (8 hosts) shows the same crash family** (documented in `MULTI_HOST.md`, 6 failed attempts).
- **v5p-32 (4 hosts) shows the same crash family** (documented in `CODEX_INFERENCE_M11.md`, 13+ runs).

---

## Prior Work Synthesis

### `MULTI_HOST.md` (v5p-64, 6 debug attempts)

| # | Approach | Result | Insight |
|---|----------|--------|---------|
| 1 | Remove `jax.clear_caches()` | FAILED | `FAILED_PRECONDITION: program continuator halted` at resume. Crash occurs at `state.step % h.every`. |
| 2 | Add `sync_global_devices` after inference | FAILED | Error happens *inside* `sync_global_devices` itself. TPU was already corrupted before sync was attempted. |
| 3 | `jax.block_until_ready()` on inference results | FAILED | block_until_ready succeeds (logs confirm "All inference results materialized"), same crash at resume. |
| 4 | Re-warm training context by touching training weights | FAILED | **Different error**: launch-group mismatch ("unexpected peer with different launch id"). Indicates XLA programs diverged across hosts. |
| 5 | Ensure identical code paths across hosts | FAILED | Same launch-group error. Moved `tokenizer.decode()` to all hosts. Problem is *not* Python code paths. |
| 6 | Add barrier *before* inference starts | PENDING | Never validated. |

### `CODEX_INFERENCE_M11.md` (v5p-32, 13+ runs)

- Runs 20-32 across host-DP, global mesh, ragged/non-ragged, cleanup variants.
- Run 31: Inference completes (`DecodeStats[after_decode]: active=32 pages_in_use=1056 free=3168`), immediate crash at resume.
- Run 32: Added `sync_global_devices` barriers pre/post inference. Same crash.
- Run 33: Inconclusive (interrupted before completion marker).
- Added device-level sync barriers in `train_simpo.py` (lines 458, 739). Did not resolve.

### `CODEX_REFACTOR_KV.md` (M11/M12 sections)

- M11 implementation confirmed code-complete (2026-02-08).
- All unit tests pass locally.
- Runtime validation on multi-host TPU is the remaining blocker.
- M12 (periodic inference every 50 steps) is blocked on M11.

---

## Root Cause Analysis

### What the error tells us

The "unexpected peer shows up in the launch group with a different launch id" error comes from TPU runtime program validation (`tpu_program_termination_validation.cc:180`). It means:

1. When JAX JIT-compiles a function for multi-host execution, XLA assigns a **launch group** with a specific **launch ID** to coordinate the collective operations across hosts.
2. All hosts must enter the same compiled program (same launch ID) at the same time.
3. If one host presents a program with a different launch ID than what the group leader expects, the TPU runtime kills all hosts.

### Why this happens during train -> infer -> train

During training, the JAX runtime has an established set of compiled XLA executables with consistent launch IDs across hosts. These programs are cached and re-executed each step.

When inference runs, it JIT-compiles **new XLA executables** (for `InferenceEngine.generate`, `model.decode`, ragged paged attention, etc.). These new executables create **new launch groups** with new launch IDs.

The TPU runtime's "program continuator" mechanism appears to maintain state about which launch group is "current." When training tries to resume after inference, one of two things happens:

1. **Launch group contamination**: The inference-created launch groups have left the TPU runtime in a state where it expects different launch IDs than the training programs provide.
2. **XLA compilation non-determinism**: The inference code path may compile slightly differently on different hosts (e.g., due to different cache states, different JIT tracing order, or shape-dependent compilation), producing different launch IDs across hosts. When training resumes with its original (consistent) launch IDs, some hosts may still have the "wrong" launch group context.

### Why `jax.clear_caches()` makes it worse

`MULTI_HOST.md` documents that calling `jax.clear_caches()` **corrupts TPU state**. This is because clearing caches forces re-compilation of training programs. Re-compilation can produce new launch IDs that don't match the established training launch group, causing the exact same error. The training programs were previously compiled and consistent; clearing forces them to be re-compiled potentially non-deterministically.

### Why barriers don't help

Barriers (`sync_global_devices`, `barrier_sync`) ensure hosts reach the same *point* in execution, but they don't address the underlying issue: the TPU runtime's launch group state has been corrupted by the inference JIT. Even if all hosts synchronize perfectly, the next training program execution encounters the stale/conflicting launch group state.

### Why both `host_data_parallel` and `global_mesh` fail

- **`host_data_parallel`**: Creates a local mesh, replicates model, runs inference on local devices. Still JIT-compiles new programs that touch TPU runtime state.
- **`global_mesh`**: Runs inference on the same global mesh as training. Still JIT-compiles new programs with new launch groups.

Both paths introduce new XLA executables into the TPU runtime's launch group tracking, which contaminates the state that training's executables depend on.

---

## What Was Already Tried (and Why Each Failed)

| Approach | Why it failed |
|----------|---------------|
| Remove `jax.clear_caches()` | Prevents cache corruption but doesn't address launch-group contamination from inference JIT |
| `sync_global_devices` after inference | TPU state already corrupted before sync runs; sync itself triggers the error |
| `jax.block_until_ready()` on results | Ensures computation completion, doesn't reset launch-group state |
| Re-warm training arrays | Triggers more JIT compilation, potentially worsening launch-group divergence |
| Identical Python code paths | Problem is XLA-level, not Python-level |
| Pre/post device barriers | Synchronizes host entry/exit timing, doesn't address TPU runtime state |
| Host-DP mode (local mesh) | Same launch-group contamination, different mesh |
| Global mesh mode | Same launch-group contamination, same mesh |

---

## Proposed Solution Strategies

### Strategy 1: Subprocess Isolation (Highest Feasibility)

**Idea**: Run inference in a completely separate process, avoiding any contamination of the training process's TPU runtime state.

**Implementation sketch**:
```python
def inference_eval_callback(step: StepInfo):
    # 1. Save model checkpoint to a temp location
    temp_ckpt = f"/tmp/m11_inference_ckpt_step_{step_number}"
    save_checkpoint(step.eval_model, temp_ckpt)

    # 2. All hosts synchronize
    barrier_sync(...)

    # 3. Launch inference as a subprocess on each host
    #    (using sample_lm_multihost.py which works standalone)
    subprocess.run([
        sys.executable, "src/levanter/main/sample_lm_multihost.py",
        "--config_path", inference_config_path,
        "--initialize_from", temp_ckpt,
    ])

    # 4. All hosts synchronize after subprocess completes
    barrier_sync(...)

    # 5. Resume training - TPU runtime state is untouched
```

**Pros**:
- Completely isolates inference JIT from training JIT. No launch-group contamination possible.
- Reuses the proven `sample_lm_multihost.py` path that works standalone.
- No changes needed to JAX internals or TPU runtime.

**Cons**:
- Subprocess launch overhead (model load from checkpoint per inference eval).
- Checkpoint I/O cost at each inference step.
- More complex orchestration (subprocess error handling, timeout management).
- May need to serialize/deserialize prompts and results between processes.

**Feasibility**: HIGH. This is the safest path. The standalone sampler already works.

### Strategy 2: XLA HLO Dump + Determinism Fix

**Idea**: Follow the TPU error message's own suggestion: dump HLO for all workers, compare `before_optimizations.txt` and `after_optimizations.txt`, identify and fix the non-determinism.

**Implementation sketch**:
```bash
# Set XLA flags to dump HLO
export XLA_FLAGS="--xla_dump_to=/tmp/xla_dump --xla_dump_hlo_as_text"

# Run the M11 config
python src/levanter/main/train_simpo.py \
    --config_path config/simpo_ultrafeedback_llama3_8b_v5p_32_m11_train2_infer128_train5.yaml
```

Then compare HLO across hosts:
```bash
# On each host, compare the dumped HLO
diff host0/before_optimizations.txt host1/before_optimizations.txt
diff host0/after_optimizations.txt host1/after_optimizations.txt
```

**Pros**:
- Addresses the root cause directly.
- If the non-determinism is identifiable and fixable, enables the in-process workflow.

**Cons**:
- May be a JAX/XLA bug rather than user code, making it unfixable from our side.
- HLO dumps on multi-host TPU are large and hard to diff.
- Even if identified, the fix may require JAX version changes or XLA patches.

**Feasibility**: MEDIUM. Worth trying for diagnosis even if the fix is upstream.

### Strategy 3: Pre-compile Inference Programs During Training Setup

**Idea**: Force-compile all inference XLA programs during training initialization (before the first train step), so that they have consistent launch IDs established alongside training programs.

**Implementation sketch**:
```python
# During trainer setup, before training loop:
def _precompile_inference_programs(model, tokenizer, inference_config):
    """Compile inference XLA programs so launch groups are established early."""
    # Build a tiny dummy engine with the same shapes
    dummy_engine = InferenceEngine.from_model_with_config(
        model=model, tokenizer=tokenizer,
        config=_build_engine_config(inference_config, ...)
    )
    # Run one dummy inference to force JIT compilation
    dummy_requests = [Request(prompt_tokens=[1, 2, 3], ...)]
    dummy_engine.generate(dummy_requests)
    del dummy_engine
```

**Pros**:
- If the issue is that inference programs are compiled *after* training has established its launch groups, pre-compiling might establish them in the correct initial context.
- No subprocess overhead.

**Cons**:
- May not work if TPU launch group IDs are assigned per-invocation rather than per-compilation.
- The pre-compiled programs may get different launch IDs from the later inference invocations if any shape or parameter differs.
- Memory overhead of creating/destroying dummy engine during setup.

**Feasibility**: LOW-MEDIUM. The launch group mechanism may not work this way.

### Strategy 4: JAX Runtime Reset Between Modes

**Idea**: Find a JAX API that can cleanly reset the TPU runtime's launch group state without destroying compiled programs.

**Potential APIs to investigate**:
- `jax.live_arrays()` - inspect live arrays
- `jax._src.dispatch.xla_callable.cache_clear()` - targeted cache clearing
- `jax.experimental.multihost_utils` - multi-host coordination
- Custom XLA client operations for launch group management

**Pros**:
- Clean in-process solution if such an API exists.

**Cons**:
- No known public API for this exists.
- JAX internals are not stable.
- Would require deep JAX/XLA expertise.

**Feasibility**: LOW. This likely requires JAX team involvement.

### Strategy 5: Single-Process Sequential Mesh Switching

**Idea**: Instead of running inference alongside training state, completely tear down the training mesh, run inference on a fresh mesh, then re-establish the training mesh.

**Implementation sketch**:
```python
def inference_eval_callback(step: StepInfo):
    # 1. Checkpoint training state to host memory / disk
    # 2. Tear down training mesh context
    # 3. Create fresh mesh for inference
    # 4. Run inference
    # 5. Tear down inference mesh
    # 6. Re-establish training mesh
    # 7. Restore training state
```

**Pros**:
- Separates the two mesh contexts completely.
- Avoids overlapping launch groups.

**Cons**:
- JAX may not support clean mesh teardown/recreation in a single process.
- Training state restoration adds complexity and overhead.
- May trigger the same launch-group issues during mesh transition.

**Feasibility**: LOW. JAX mesh lifecycle is not designed for this pattern.

---

## Recommended Implementation Plan

### Phase 1: Diagnostic Run (Immediate)

Run an M11 config with XLA HLO dumps enabled to capture the exact point of launch-group divergence:

```bash
export XLA_FLAGS="--xla_dump_to=/tmp/xla_dump_m11 --xla_dump_hlo_as_text --xla_dump_hlo_pass_re=.*"

python infra/launch.py --foreground --zone us-central1-a \
    --tpu_name simpo_worker --tpu_type v5p-32 --capacity_type on-demand -- \
    python src/levanter/main/train_simpo.py \
    --config_path config/simpo_ultrafeedback_llama3_8b_v5p_32_m11_phase_a.yaml
```

Compare HLO across hosts to determine if the issue is:
- (a) Inference programs compiled differently across hosts (fixable by ensuring deterministic tracing).
- (b) Launch group state corruption regardless of compilation consistency (requires subprocess isolation).

### Phase 2: Implement Subprocess Isolation (Strategy 1)

If the diagnostic run confirms that launch-group contamination is inherent (not just non-determinism), implement subprocess-based inference:

1. Add a `checkpoint_for_inference()` helper that saves the current model to a temp directory.
2. Modify `inference_eval_callback` to:
   - Save a lightweight checkpoint at the inference step.
   - Launch `sample_lm_multihost.py` as a subprocess on each host.
   - Wait for subprocess completion.
   - Read back results from per-host JSONL files.
3. Add a new config option `inference_isolation: in_process | subprocess` (default `subprocess` for multi-host).

### Phase 3: Optimize Subprocess Path

If subprocess isolation works but is too slow:
- Use memory-mapped checkpoints to reduce I/O.
- Pre-fork the subprocess and keep it warm.
- Investigate `jax.distributed.initialize()` with separate device sets.

---

## Code References

| File | Relevant Lines | Purpose |
|------|---------------|---------|
| `src/levanter/main/train_simpo.py` | 424-743 | Inference callback implementation |
| `src/levanter/main/train_simpo.py` | 454-460 | Pre-inference barriers (added during debugging) |
| `src/levanter/main/train_simpo.py` | 731-741 | Post-inference cleanup and barriers |
| `src/levanter/main/sample_lm_multihost.py` | entire | Standalone host-DP inference (works) |
| `src/levanter/utils/jax_utils.py` | 539-542 | `sync_global_devices` (uses `assert_equal` for cross-host barrier) |
| `src/levanter/utils/jax_utils.py` | 545-598 | `replicate_model_to_local_mesh` (all-gather + local replication) |
| `src/levanter/utils/mesh.py` | 190-210 | `create_local_mesh` (host-local device mesh) |
| `src/levanter/inference/engine.py` | entire | `InferenceEngine` lifecycle |

## Config Files

| Config | Purpose | Status |
|--------|---------|--------|
| `config/simpo_ultrafeedback_llama3_8b_v5p_32_m11_train2_infer128_train5.yaml` | Main M11 config: 2 train steps, 128-prompt inference, 5 more train steps | Crash at resume |
| `config/simpo_ultrafeedback_llama3_8b_v5p_32_m11_phase_a.yaml` | Phase-A debug config (host-DP) | Crash at resume |
| `config/simpo_ultrafeedback_llama3_8b_v5p_32_m11_phase_a_global.yaml` | Phase-A debug config (global mesh) | Crash at resume |
| `config/simpo_ultrafeedback_llama3_8b_v5p_32_m11_phase_a_train2.yaml` | Phase-A train-only termination test | Inconclusive |

## Run Log Summary

| Run | Config Variant | Inference Mode | Result | Crash Location |
|-----|---------------|----------------|--------|----------------|
| 20-22 | Various | host_data_parallel | CRASH | Post-inference resume |
| 23 | Phase A | host_data_parallel | CRASH | Post-inference resume |
| 24 | Phase A + cleanup_mode=end | host_data_parallel | CRASH | Post-inference resume |
| 25 | Phase A | global_mesh | CRASH | Post-inference resume |
| 26 | Phase A + ragged | host_data_parallel | CRASH | Post-inference resume |
| 31 | Phase A + tpr256/r64/mqt512 | host_data_parallel | CRASH | Post-inference resume |
| 32 | Phase A + sync barriers | host_data_parallel | CRASH | Post-inference resume |
| 33 | Train-only termination | host_data_parallel | Inconclusive | Interrupted |

---

## Constraints on Solutions

1. **Do NOT use `jax.clear_caches()`**: documented to corrupt TPU state (`MULTI_HOST.md`).
2. **Do NOT add leader-only code paths inside JIT**: causes XLA non-determinism across hosts (`MULTI_HOST.md` Attempt 5, `CODEX_REFACTOR_KV.md` M5.2).
3. **Do NOT use `defer_tracker_logs_until_end=false`**: causes asymmetric tracker emission crashes (`CODEX_REFACTOR_KV.md` M5.2).
4. **All hosts must execute identical JIT-traced code paths**: required by multi-host TPU determinism invariants.
5. **Barriers alone are insufficient**: they synchronize timing but don't address launch-group state.

---

## Open Questions

1. Does JAX provide any API to "fence" or "reset" the TPU runtime's launch group state between different execution phases?
2. Is there a way to share compiled XLA executables between processes to reduce subprocess checkpoint/reload overhead?
3. Has the JAX team documented any supported pattern for running multiple distinct JIT workloads (training + inference) in a single multi-host process?
4. Would upgrading JAX to a newer version (with potential launch-group management improvements) resolve this?
