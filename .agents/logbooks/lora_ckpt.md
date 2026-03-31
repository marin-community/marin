# LoRA Checkpoint Resume Bug Investigation

## Problem Statement

The lr=7.5e-6 LoRA DPO run (`endlboq3`) resumes from step-799 checkpoint but produces eval loss ~0.686 (untrained baseline = ln(2)), suggesting the LoRA weights are not being applied during inference even though they appear to load successfully.

## Key Identifiers

- **Iris job (broken)**: `/ahmed/lora-bsv2-lr7p5e6-debug-ckpt-v4` (latest debug run)
- **W&B run**: `endlboq3` at https://wandb.ai/marin-community/dpo/runs/endlboq3
- **Checkpoint**: `gs://marin-us-east5/checkpoints/dpo/lora_bloom_speceval_v2_marin_instruct_beta0.1_lr7.5e-6_seed2_v5p8_b64/endlboq3/step-799`
- **Config**: `lib/levanter/config/dpo/lora_dpo_bloom_speceval_v2_marin_instruct_beta0.1_lr7.5e-6_seed2_v5p8_east5_b64.yaml`
- **HF merged checkpoints (good, pre-preemption)**: steps 200, 400, 600 at `gs://marin-us-east5/checkpoints/dpo/lora_bloom_speceval_v2_marin_instruct_beta0.1_lr7.5e-6_seed2_v5p8_b64/merged_hf/endlboq3/`

## Working runs for comparison

| Run | W&B ID | Checkpoint | Resume works? | Preemptions |
|-----|--------|-----------|---------------|-------------|
| lr=7.5e-6 | `endlboq3` | step-799 | **NO** — eval loss 0.686 | 3 |
| lr=5e-6 | `uxviy7fz` | step-153 → step-332+ | **YES** — eval loss 0.068 | 1 |
| lr=1e-5 | `ki3eb0hi` | step-271 → step-399+ | **YES** (until code broke) | 1 |

## Investigation Timeline

### Phase 1: Initial discovery

- Train step 801 completes after resume but shows `postfix:-` (no loss reported in tqdm)
- Step-800 eval shows loss=0.686 ≈ ln(2) = untrained (policy == reference)
- For comparison, step-0 eval showed loss=0.692 (untrained), step-200 eval showed loss=0.012 (trained)
- This was present from the VERY FIRST resume after preemption (23:21 UTC eval), ruling out multi-preemption corruption

### Phase 2: Red herring — broken code deployment

Discovered that the git merge attempt (`git merge origin/main` at ~16:30 UTC) left `train_dpo.py` in an inconsistent state:
- Main's merge brought in `from levanter.dpo import ...` and removed inline `_logp_sum`
- The merge was "aborted" with `git merge --abort` but `git stash pop` left modified files
- Any Iris job submitted AFTER the merge used broken code with `NameError: _logp_sum`
- **Fix**: `git checkout lib/levanter/src/levanter/main/train_dpo.py` restored to HEAD

**However**, even with clean code (restored train_dpo.py), the eval loss is still 0.686. The code deployment issue caused ADDITIONAL crashes but is NOT the root cause of the checkpoint loading failure.

### Phase 3: Checkpoint data verification

Used `iris task exec` into the running container to directly read checkpoint data:

```
Total OCDBT keys: 159
LoRA keys: 147
LoRA B weight shape=(32, 4096, 64), dtype=float32, norm=1.738288
All zeros: False
```

**The checkpoint data is 100% valid.** Trained LoRA weights are present and non-zero.

Compared with the working lr=5e-6 checkpoint (step-332):
- Both have 159 total keys, 147 LoRA keys
- Key paths are IDENTICAL (`model/transformer/layers/stacked/mlp/down_proj/lora/lora_A/weight/...`)
- Data sizes are identical (~2.01 GB)
- dtypes are identical (float32)

### Phase 4: Debug logging in checkpoint.py

Added debug logging to `load_checkpoint_or_initialize` in `lib/levanter/src/levanter/checkpoint.py` (around line 520):

```python
# After load_checkpoint() call:
loaded_leaves = _jax.tree.leaves(loaded_state)
shape_leaves = [x for x in loaded_leaves if isinstance(x, _jax.ShapeDtypeStruct)]
array_leaves = [x for x in loaded_leaves if hasattr(x, 'shape') and not isinstance(x, _jax.ShapeDtypeStruct)]
logger.info(f"[DEBUG] Checkpoint loaded: {len(array_leaves)} arrays, {len(shape_leaves)} ShapeDtypeStructs...")

# Plus key path analysis showing which LoRA params loaded vs missing
```

#### Debug run v4 results (2026-03-31 01:32 UTC, job `/ahmed/lora-bsv2-lr7p5e6-debug-ckpt-v4`):

```
[DEBUG] Checkpoint loaded: 48 arrays, 0 ShapeDtypeStructs, 0 Nones, 0 other. Total: 48
[DEBUG] LoRA arrays in loaded_state: 42
[DEBUG] First LoRA array: key=model.transformer.layers.stacked.self_attn.q_proj.lora.lora_A.weight, shape=(32, 64, 4096), norm=44.753231
[DEBUG] LoRA keys in filtered_state_shape: 105
[DEBUG] LoRA loaded: 42, expected: 105, missing: 63
[DEBUG]   MISSING: model.transformer.layers.stacked.mlp.down_proj.lora.dropout.inference
[DEBUG]   MISSING: model.transformer.layers.stacked.mlp.down_proj.lora.lora_A.bias
[DEBUG]   MISSING: model.transformer.layers.stacked.mlp.down_proj.lora.lora_B.bias
[DEBUG]   MISSING: model.transformer.layers.stacked.mlp.gate_proj.lora.dropout.inference
[DEBUG]   MISSING: model.transformer.layers.stacked.mlp.gate_proj.lora.lora_A.bias
[DEBUG]   MISSING: model.transformer.layers.stacked.mlp.gate_proj.lora.lora_B.bias
...
```

#### Key finding

The 63 "missing" keys are ALL non-array leaves:
- `lora.dropout.inference` — boolean flag (not a tensor)
- `lora.lora_A.bias` — None (LoRA A has no bias in this config)
- `lora.lora_B.bias` — None (LoRA B has no bias in this config)

These are part of the `trainable_filter` (105 entries) but are NOT JAX arrays, so they're never serialized to the checkpoint. The serializer only saves array leaves. On restore, non-array leaves come from `init_state` via `equinox.combine`.

**The 42 actual LoRA weight arrays (lora_A.weight + lora_B.weight for 21 modules) ALL loaded successfully with non-zero trained values (norm=44.75).**

## Current Status: THE BUG IS STILL OPEN

The checkpoint loads 42 LoRA weight arrays correctly. The 63 "missing" are non-array leaves that are expected to come from `init_state`. **Yet the eval still shows loss=0.686 (untrained).**

This means either:

### Hypothesis A: `equinox.combine` is overwriting loaded weights with init_state zeros

The merge logic in `init_and_merge` (checkpoint.py line 493-497):
```python
@haliax.named_jit(...)
def init_and_merge(state, *args, **kwargs):
    init_state = init_fn(*args, **kwargs)
    state = equinox.filter(state, lambda x: not isinstance(x, jax.ShapeDtypeStruct))
    return equinox.combine(state, init_state)
```

`equinox.combine(state, init_state)` should take leaves from `state` where they exist and fall back to `init_state` where `state` has None. But maybe the filter on line 496 is removing the loaded LoRA arrays (if they're somehow wrapped in ShapeDtypeStruct)?

The debug output says "0 ShapeDtypeStructs" in loaded_state, so this seems unlikely. But the filter runs INSIDE `named_jit` so we can't debug it with logging.

### Hypothesis B: The model uses the wrong weights at eval time

The checkpoint loads correctly into `state.model`, but the eval function might be using a different model view. In DPO with `adapter_base` reference:
- Policy = base model + LoRA adapters (should use trained weights)
- Reference = base model without LoRA (adapter_base derives this)

If the eval function is accidentally using the reference view (no LoRA) for both policy AND reference, the loss would be ln(2).

### Hypothesis C: The step counter causes a skip

`state.step` is 799 after checkpoint load. The code at line 541-543 checks `if int(state.step) == 0` for fresh init vs resume. After resume, it sets `policy_model = state.model`. Then at line 557, `state = dataclasses.replace(state, model=policy_model)`. This SHOULD be correct.

But maybe `state.step` being 799 causes the training loop to skip step 800 and go directly to eval, and the eval is using the pre-checkpoint model somehow?

### Hypothesis D: `init_and_merge` runs inside JIT and the loaded arrays are being re-materialized

The `init_and_merge` function runs inside `@haliax.named_jit`. Inside JIT, `init_fn` re-creates the model from scratch (random init + zero LoRA). The `equinox.combine` should merge the loaded checkpoint values. But if JIT's tracing treats the loaded checkpoint arrays as "constant" and the init values as "traced", the combine might behave differently than expected.

## How to reproduce

1. Use config `lib/levanter/config/dpo/lora_dpo_bloom_speceval_v2_marin_instruct_beta0.1_lr7.5e-6_seed2_v5p8_east5_b64.yaml`
2. Set env `RUN_ID=endlboq3`
3. Run `python -m levanter.main.train_dpo --config_path <config>`
4. It will find checkpoint at step-799 and load it
5. Train step 801 will complete (fast, ~2.5 min including JIT)
6. Step-800 eval will show loss ≈ 0.686 (broken)

The working runs (lr=5e-6 `uxviy7fz`, lr=1e-5 `ki3eb0hi`) use the SAME code path and work correctly. The difference is that they were saved and loaded by the same code version, while lr=7.5e-6 was saved by codex's code and loaded by ours.

## Remaining debug code

Debug logging is currently in `lib/levanter/src/levanter/checkpoint.py` around line 520. It adds ~20 lines of logging after `load_checkpoint()` returns. This should be removed before merging.

## Files modified during investigation

- `lib/levanter/src/levanter/checkpoint.py` — debug logging added (TEMPORARY, remove before merge)
- `experiments/models.py` — added `marin_8b_instruct` model step
- Various YAML configs in `lib/levanter/config/dpo/` — LoRA DPO configs for different LRs/regions
- `.agents/logbooks/levanter_mesh_explained.md` — mesh analysis doc
- `.agents/logbooks/dpo-lora-claude.md` — main experiment logbook

## Next Steps for the Investigating Agent

1. **Check if the lr=5e-6 checkpoint was saved by different code than lr=7.5e-6**: The lr=7.5e-6 checkpoint was saved by codex's code bundle. The lr=5e-6 was saved by our code. If the pytree structure differs (e.g., different leaf ordering in equinox modules), `equinox.combine` might not merge correctly even though the OCDBT keys match.

2. **Add logging INSIDE the JIT** in `init_and_merge`: Use `jax.debug.print` (which works inside JIT) to check LoRA weight norms after combine:
   ```python
   def init_and_merge(state, *args, **kwargs):
       init_state = init_fn(*args, **kwargs)
       state = equinox.filter(state, lambda x: not isinstance(x, jax.ShapeDtypeStruct))
       merged = equinox.combine(state, init_state)
       # Debug: check a LoRA weight after merge
       jax.debug.print("LoRA B norm after combine: {}", jnp.linalg.norm(merged.model.transformer.layers.stacked.mlp.down_proj.lora.lora_B.weight.array))
       return merged
   ```

3. **Check the eval model vs train model**: The Trainer passes `step.eval_model` to eval hooks. Verify this is the same as `step.model` (with LoRA weights).

4. **Try loading the checkpoint on the lr=5e-6 run's infrastructure**: Same code, same checkpoint path, but using a checkpoint we KNOW works. If it works, the issue is checkpoint-specific. If it fails, the issue is code-specific.

5. **Compare equinox pytree structure**: Use `jax.tree_util.tree_structure` to compare the pytree structure of the loaded state vs the init state. If they differ, `equinox.combine` might not align leaves correctly.

## 2026-03-30 Follow-up: Root Cause Found

The checkpoint save/load path itself is not dropping LoRA tensors. The actual bug is in how unified `train_dpo.py` resumes adapter runs.

### What is preserved correctly

- The LoRA-only trainer checkpoint really does contain the trained adapter arrays.
- The debug output showing `42` loaded LoRA arrays is consistent with `21` adapted modules times `lora_A.weight` and `lora_B.weight`.
- The "missing" `63` LoRA-related leaves are expected non-array leaves:
  - `dropout.inference`
  - `bias=None`

### The actual failure mode

In unified `train_dpo.py`, we do:

1. build a fresh policy model
2. call `trainer.initial_state(...)`
3. let `load_checkpoint_or_initialize(...)` merge the checkpointed trainable leaves into that fresh policy model
4. only consult `initialize_from_hf` / `initialize_from_checkpoint_path` on fresh starts (`state.step == 0`)

That is correct for full-model checkpoints, but wrong for LoRA checkpoints because LoRA checkpoints only store adapter weights, not the frozen base model weights.

So on resume, the code was taking:

- **base model** = fresh/random init from `config.model.build(...)`
- **adapter weights** = trained LoRA tensors from checkpoint

and treating that as the resumed policy.

For `reference.type=adapter_base`, the reference is then `unwrap_lora_modules(policy_model)`, so both policy and reference share the same wrong fresh base model. That explains the resumed eval loss near `ln(2)`.

### Local reproduction

I reproduced this locally with a tiny model using the real `TrainerState.saveable_state` + `load_checkpoint_or_initialize(...)` path:

- save a LoRA-only trainer checkpoint
- reload it into a different base-model initialization
- observe:
  - the loaded `lora_B` weights match exactly
  - the resumed model output matches the **wrong base + loaded LoRA**
  - not the original pre-save model output

This is the key disambiguation:

- **LoRA checkpoint serialization is fine**
- **resume is reattaching those LoRA tensors to the wrong base model**

### Why old LoRA code worked

The pre-refactor `lib/levanter/src/levanter/main/lora_dpo.py` loaded the pretrained base model before calling `trainer.initial_state(...)`, just like `lora_lm.py`.

The unified `train_dpo.py` refactor changed that ordering. It only loaded the source model on fresh starts, which is why the regression showed up specifically in the unified LoRA-DPO path.

### Fix

`lib/levanter/src/levanter/main/train_dpo.py` now does the right thing on adapter resumes:

- if resuming and `adapter.type != none`
- rebuild the policy model from the configured source (`initialize_from_hf`, `initialize_from_checkpoint_path`, or scratch)
- then overlay the checkpointed trainable adapter weights onto that correctly sourced policy model

I added `_restore_policy_model_from_partial_checkpoint(...)` plus a regression test that simulates the broken state and verifies we recover the original model output when the correct base is reconstructed first.

### Verification

- `uv run --project lib/levanter --group test python -m pytest lib/levanter/tests/test_dpo.py lib/levanter/tests/test_lora_dpo.py lib/levanter/tests/test_checkpoint.py`
- Result: `48 passed, 1 skipped`
