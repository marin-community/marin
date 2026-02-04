# Codex Multi-Host Inference: Best-Attempt Fix Plan

## Summary

The failures in `MULTI_HOST.md` look like multi-host XLA divergence after inference:
- `FAILED_PRECONDITION: The program continuator has halted unexpectedly` suggests device programs are left in an invalid state.
- `unexpected peer ... different launch id` points to different executables being compiled/used across hosts.

Based on the current code, the most likely sources of divergence are:
1. **Per-host inference configuration differences** (auto-sizing KV cache from free HBM, prompt tokenization, RNG seeds).
2. **Uncoordinated multi-host compilation** because inference uses `jax.jit` instead of `pjit`/`named_jit`.
3. **Device work not fully synchronized** before training resumes.

Below is a concrete, deterministic plan that should eliminate those failure modes while keeping inference in-process.

---

## Observations From the Codebase

- Inference kernels `_run_prefill` and `_run_generation_loop` are `jax.jit`, not `hax.named_jit`.
- `InferenceEngineConfig.max_pages` can be **auto-inferred from free HBM** (`_infer_max_pages_from_hbm`) which can differ per host.
- The inference callback tokenizes prompts **independently on each host**, and seeds PRNG using `step.step` (device array) directly.
- Training runs inside a `Trainer` context that sets the **parameter axis mapping**, while inference likely wants **compute axis mapping** for activations/KV cache.

---

## Attempt #1: Deterministic Multi-Host Inference (Plan)

### 1) Make inference config deterministic across hosts

**Goal:** all hosts build identical shapes and static args.

**Required changes:**
- **Always set `max_pages` explicitly** in config for multi-host inference.
- If `max_pages` is not provided, compute it on **process 0 and broadcast**.
- Ensure `max_prefill_size` is explicit (or derived from a broadcasted value), not from tokenizer internals.
- Broadcast tokenized prompts from leader so all hosts use identical token lists.

Suggested helper in the callback (pseudo-code):

```python
from levanter.utils.jax_utils import multihost_broadcast_sync

if is_multihost:
    # tokenize on leader once
    if is_leader:
        prompt_tokens = [tokenizer.encode(p) for p in prompts]
    else:
        prompt_tokens = None
    prompt_tokens = multihost_broadcast_sync(prompt_tokens, source=0)

    # ensure deterministic KV sizing
    if max_pages is None:
        max_pages = _infer_max_pages_from_hbm(model, engine_config)
    max_pages = multihost_broadcast_sync(max_pages, source=0)

    # explicit prefill size (avoid tokenizer.model_max_length divergence)
    if max_prefill_size is None:
        max_prefill_size = max_seq_len
    max_prefill_size = multihost_broadcast_sync(max_prefill_size, source=0)
```

Then build `Request` objects from `prompt_tokens` instead of re-encoding locally.


### 2) Force coordinated multi-host compilation (avoid per-host XLA divergence)

**Goal:** inference uses the same XLA executable on all hosts.

Best approach: **switch inference kernels to `hax.named_jit` (pjit)** with explicit axis resources. The commented-out decorator in `engine.py` is the right hint.

Changes in `src/levanter/inference/engine.py`:
- Replace `@jax.jit` on `_run_prefill` and `_run_generation_loop` with `@hax.named_jit`.
- Add `axis_resources` and `out_axis_resources` parameters (use compute axis mapping).
- Thread axis mapping through `InferenceEngineConfig` or pass via context.

Sketch (exact API may need adjustment):

```python
@hax.named_jit(
    axis_resources=compute_axis_mapping,
    out_axis_resources=compute_axis_mapping,
    donate_args=("gen_state",),
)
def _run_generation_loop(...):
    ...
```

Then, in the callback, ensure inference runs under the compute mapping:

```python
with hax.axis_mapping(trainer.compute_axis_mapping):
    engine = InferenceEngine.from_model_with_config(...)
    result = engine.generate(requests)
```

**Why this matters:** `pjit` compilation is coordinated across processes, while `jax.jit` can compile independently and produce divergent launch ids.


### 3) Precompile once before training

**Goal:** inference compilation happens once in a controlled state.

Add a warmup step (step 0 or before training loop):
- Build inference engine with deterministic config.
- Run a tiny generation on a single short prompt.
- Block and sync.
- Discard engine.

This ensures inference JITs are already compiled and cached before real training steps.


### 4) Make device-level synchronization explicit

**Goal:** no lingering inference work when training resumes.

After inference generation:
- Block on **engine state**, not just output tokens.
- Then perform a device-level barrier to align TPU execution across hosts.

Example:

```python
# ensure all inference device work is complete
jax.tree_util.tree_map(
    jax.block_until_ready,
    eqx.filter(engine.gen_state, lambda x: is_jax_array_like(x)),
)

# device-level barrier (after inference completes cleanly)
if is_multihost:
    jax.experimental.multihost_utils.sync_global_devices("inference_done")
```

Keep the host-level `barrier_sync` afterward.


### 5) Make RNG seeds identical and host-derived

**Goal:** ensure inference randomness is identical across hosts.

Use a host scalar (not a device array) and broadcast:

```python
if is_leader:
    base_seed = int(jax.device_get(step.state.step))
else:
    base_seed = None
base_seed = multihost_broadcast_sync(base_seed, source=0)

key = jax.random.PRNGKey(base_seed)
key = jax.random.fold_in(key, i)  # per-prompt
```


---

## Why This Should Fix the Errors

- **Deterministic shapes and configs** remove the most common source of multi-host divergence (different HLO per host).
- **Named jit/pjit** forces coordinated compilation on the multi-host runtime.
- **Blocking on all device outputs** + device barrier prevents the training program from starting while inference is still executing.
- **Broadcasted prompts and seeds** guarantee identical host-side inputs.

---

## Minimal Patch Sketch (High-Level)

1. `src/levanter/inference/engine.py`
   - Convert `_run_prefill` and `_run_generation_loop` to `hax.named_jit`.
   - Thread an axis mapping into inference kernels (config or context).

2. `src/levanter/main/train_simpo.py`
   - Broadcast prompts (tokenized), `max_pages`, `max_prefill_size`, and `base_seed` from leader.
   - Run inference under compute axis mapping.
   - Block on `engine.gen_state` and then `sync_global_devices`.

3. `config/...v5p_64_inference.yaml`
   - Require explicit `max_pages`, `max_seq_len`, `max_prefill_size` for multi-host inference.

---

## Fallback if Multi-Host Inference Still Fails

If the runtime still diverges, the safest operational alternative is:
- **Run inference as a separate eval job** on a checkpoint saved every N steps.
- This avoids interleaving inference JITs with training JITs entirely.

But given the evidence, the deterministic-config + named_jit + explicit device sync path above is the best attempt to keep inference in-process and stable.

## Attempt #1 Status

Implemented in:
- `src/levanter/inference/engine.py` (switch inference kernels to `hax.named_jit`; update jaxpr/HLO dump path)
- `src/levanter/main/train_simpo.py` (broadcast prompts/seed/max_pages; run inference under compute axis mapping; block+sync after inference)

## Attempt #1 Bug Report

**Failure:** `TypeError: multihost_broadcast_sync() got an unexpected keyword argument 'source'`

**Where:** `src/levanter/main/train_simpo.py` inside `inference_eval_callback` when calling `multihost_broadcast_sync(...)`.

**Root cause:** `multihost_broadcast_sync` accepts `is_source` (bool), not `source` (int). The call signature mismatch aborts inference before any JAX work.

**Observed log excerpt:**
```
[INFERENCE DEBUG][Process 0][Step 2] Barrier BEFORE inference (ensuring all hosts enter together)...
[INFERENCE DEBUG][Process 0][Step 2] Pre-inference barrier passed
TypeError: multihost_broadcast_sync() got an unexpected keyword argument 'source'
```

## Attempt #2: Fix Broadcast API + Single-Payload Sync (Plan)

1. **Fix the broadcast call sites** to use `is_source=(jax.process_index() == 0)`.\n
2. **Broadcast a single payload** (dict) instead of multiple independent calls to reduce KV-store churn and eliminate ordering issues. Example payload:\n
   - `prompt_tokens`\n
   - `base_seed`\n
   - `resolved_max_pages`\n
   - `max_prefill_size`\n
3. **Add a guard** that asserts the payload is received on all hosts and has expected lengths before proceeding.\n
4. Re-run the multi-host job and verify inference completes and training resumes.

## Attempt #2 Status

Implemented in:
- `src/levanter/main/train_simpo.py` (single-payload multihost broadcast using `is_source`, includes tokens/seed/max_pages/max_prefill_size)

## Attempt #2 Bug Report

**Failure:** `ValueError: Jitted function has invalid argnames {'max_seqs_in_prefill'} in static_argnames. Function does not take these args.`

**Where:** `src/levanter/inference/engine.py` in `_run_prefill` (triggered during `engine.generate()`).

**Root cause:** `hax.named_jit` wraps the function and ultimately calls `jax.jit` on an internal wrapper with signature `(dynamic_donated, dynamic_reserved, static)`. Passing `static_argnames=("max_seqs_in_prefill", ...)` is invalid for this wrapper, so JAX errors before compilation. This is why inference fails but training continues (exception is caught by the hook).

**Observed log excerpt:**
```
ValueError: Jitted function has invalid argnames {'max_seqs_in_prefill'} in static_argnames. Function does not take these args.
```

## Attempt #3: Remove static_argnames + Ensure Python ints (Plan)

1. **Remove `static_argnames` from `hax.named_jit` decorators** on `_run_prefill` and `_run_generation_loop`.\n
2. **Ensure static params are Python ints** at call sites (e.g., `int(self.config.max_seqs_in_prefill)`, `int(self.config.max_rounds)`) so `named_jit` treats them as static via its hashable partitioning.\n
3. Optionally **use `functools.partial`** to bind these ints, making the static nature explicit and avoiding accidental JAX array passing.\n
4. Re-run multi-host inference to verify the error disappears and generation proceeds.

## Attempt #3 Status

Implemented in:
- `src/levanter/inference/engine.py` (remove `static_argnames` from `hax.named_jit`; force Python ints at call sites)

## Attempt #3 Bug Report

**Failure:** `JaxRuntimeError: INTERNAL: Accelerator device halted prematurely ... unexpected peer ... different launch id`

**Where:** `trainer.py` when resuming training immediately after inference callback exits.

**Root cause (likely):** Multi-host launch group desync still occurring after inference completes. A constant tag is used for `sync_global_devices`, which may allow stale/overlapping rendezvous across steps or partial host participation. This can leave the TPU runtime in a mismatched state even though host barriers pass.

**Observed log excerpt:**
```
... <<< CALLBACK EXITING
jax.errors.JaxRuntimeError: INTERNAL: Accelerator device halted prematurely ...
An unexpected peer shows up in the launch group with a different launch id ...
```

## Attempt #4: Step-Tagged Global Device Sync (Plan)

1. **Use step-unique tags** for `multihost_utils.sync_global_devices`, e.g. `inference_complete_{step.step}`.\n
2. **Add a pre-inference device sync** with a step-unique tag (e.g., `inference_start_{step.step}`) to align device runtime state before compilation/execution.\n
3. Keep deterministic host barriers (`barrier_sync_with_tag`) before and after inference.

## Attempt #4 Status

Implemented in:
- `src/levanter/main/train_simpo.py` (step-tagged `sync_global_devices` before and after inference)

## Attempt #4 Bug Report

**Failure:** `JaxRuntimeError: INTERNAL: Accelerator device halted prematurely ... unexpected peer ... different launch id` immediately after the inference callback returns.

**Where:** `trainer.py` during the next training step (while evaluating `hooks_this_time`).

**Root cause (likely):** Training step work may still be in-flight on TPU when inference begins, so inference launches overlap with training launch groups. This can cause launch id mismatches even if inference itself is deterministic.

## Attempt #5: Block Training Completion Before Inference (Plan)

1. **Block on the completed training state** at the start of the inference callback (`block_until_ready_tree(step.state)`), ensuring all device work from the train step is finished before inference begins.\n
2. Keep pre/post host barriers + step-tagged `sync_global_devices` to maintain global alignment.\n
3. Re-run multi-host inference and confirm training resumes without launch id mismatches.

## Attempt #5 Status

Implemented in:
- `src/levanter/main/train_simpo.py` (block on `step.state` before inference to avoid overlap with training device work)
