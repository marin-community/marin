# DPO Implementation Analysis: Haliax Changes Review

This document analyzes the changes made to Haliax and Levanter for DPO support, evaluates whether each change is necessary, and proposes simplifications to get Haliax closer to main.

---

## Executive Summary

**Verdict**: All 4 Haliax changes are **necessary**. The Levanter trainer_state.py changes were **overly complex** and have been simplified.

| File | Change | Verdict | Action |
|------|--------|---------|--------|
| `haliax/nn/scan.py` | Add `auto_sharded` after vmap | **NECESSARY** | Keep |
| `haliax/partitioning.py` | Skip sharding for batch_dim | **NECESSARY** | Keep |
| `haliax/partitioning.py` | Handle array=None in pspec_for | **NECESSARY** | Keep |
| `haliax/quantization.py` | NamedArray as leaf + update=None | **NECESSARY** | Keep |
| `levanter/trainer_state.py` | Rewrite partition/cast functions | **OVERCOMPLICATED** | Simplify |

---

## Update: Fixes Merged to Main

The fixes described in this document have been implemented and merged to `main` via two PRs:

### PR #2463: "fix potential sharding inside hax.vmap"

This PR implemented fixes #1 and #2 from this analysis:

| Fix | Implementation |
|-----|----------------|
| `nn/scan.py`: Add `auto_sharded` after vmap | Added `stacked = haliax.auto_sharded(stacked)` after vmap in `Stacked.init` |
| `partitioning.py`: Skip sharding for batched tracers | Changed from `is_in_jit()` to `_is_jit_tracer(named.array)` for more precise tracer detection |

The PR also added 86 lines of regression tests in `test_scan.py` to verify vmap + sharding interactions.

### PR #2502: "fix getting pspec for None-backed named arrays"

This PR implemented fixes #3 and #4, plus the simplified `trainer_state.py` changes:

| Fix | Implementation |
|-----|----------------|
| `partitioning.py`: Handle `array=None` in `pspec_for` | Uses `node.axes` safely without accessing potentially-None array |
| `trainer_state.py`: Simplified partition/combine | Added `is_leaf=lambda x: isinstance(x, hax.NamedArray)` to `eqx.partition`/`eqx.combine` calls |
| `lora.py`: Consistent NamedArray handling | Switched to `haliax.tree_util.tree_map` |

The PR also added a regression test `test_pspec_for_namedarray_with_missing_array` to verify the fix.

### Why These Were Separate PRs

The fixes were split to:
1. Keep changes focused and reviewable
2. Allow independent testing of vmap/sharding vs None-array handling
3. Land in `main` without requiring the full DPO implementation

With these PRs merged, the DPO branch (`dpo_claude_opus`) can focus purely on the DPO-specific logic (preference data, loss function, `DpoModel`, training script) without carrying Haliax patches.

---

## Why DPO Surfaces These Issues (DPO vs LM Training)

DPO (Direct Preference Optimization) training differs fundamentally from standard LM pretraining in ways that expose latent bugs in JAX/Equinox/Haliax interactions:

### Key Architectural Differences

| Aspect | LM Pretraining | DPO Training |
|--------|---------------|--------------|
| **Models** | Single model | Two models: policy + reference |
| **Trainability** | All parameters trained | Policy trained, reference **frozen** |
| **Memory** | 1x model size | 2x model size (both loaded) |
| **Gradients** | All params get gradients | Only policy params get gradients |
| **Model structure** | Flat `LmHeadModel` | Nested `DpoModel(policy, reference)` |

### The DpoModel Structure

```python
@dataclass
class DpoModel:
    policy: LmHeadModel    # Trainable - gets gradients
    reference: LmHeadModel # Frozen - NO gradients, used for KL penalty
```

The DPO loss requires computing log-probabilities from **both** models:
```python
# Policy model (trainable)
logp_pi_chosen = _logp_sum(model.policy, example.chosen)
logp_pi_rejected = _logp_sum(model.policy, example.rejected)

# Reference model (frozen) - used for KL divergence penalty
logp_ref_chosen = _logp_sum(reference_model, example.chosen)
logp_ref_rejected = _logp_sum(reference_model, example.rejected)

# DPO loss combines both
loss = -log_sigmoid(beta * (delta_pi - delta_ref))
```

### Why Each Issue Manifests in DPO

1. **`eqx.partition` creating `NamedArray(array=None)`** (quantization.py, trainer_state.py)
   - In LM training: All params are trainable, partition rarely creates None placeholders
   - In DPO: Reference model is frozen → `eqx.partition(model, is_trainable)` creates `NamedArray(array=None)` for ALL reference params
   - This corrupts the model if NamedArray isn't treated as a leaf

2. **`pspec_for` failing on None arrays** (partitioning.py)
   - In LM training: Model state always has real arrays
   - In DPO: After partitioning for gradients, reference model becomes `NamedArray(array=None)` placeholders
   - When combining/sharding, `pspec_for` tries to read shape from None

3. **Memory pressure requiring proper sharding** (scan.py)
   - In LM training: Single 8B model fits in memory even with suboptimal sharding
   - In DPO: Two 8B models (16B total params) → OOM if layers aren't sharded immediately after vmap
   - The `auto_sharded` call after vmap becomes critical

4. **vmap + sharding dimension mismatch** (partitioning.py batch_dim check)
   - In LM training: Less common to call `auto_sharded` inside vmap
   - In DPO: Model initialization with `Stacked.init` uses vmap, and we want to shard immediately
   - The batch_dim check prevents crashes when sharding is attempted inside vmap

### The Frozen Reference Model Problem

The core issue is that DPO's **frozen reference model** creates a code path that standard LM training never exercises:

```python
# Standard LM training filter
is_trainable = True  # Everything trains

# DPO training filter
def is_trainable(model):
    return _bool_tree_like(model.policy, True), _bool_tree_like(model.reference, False)
```

When you partition with this filter:
```python
trainable, non_trainable = eqx.partition(dpo_model, is_trainable)
# trainable.policy = LmHeadModel(...)  ← Real weights
# trainable.reference = LmHeadModel(NamedArray(None), NamedArray(None), ...)  ← CORRUPTED!
```

Without treating NamedArray as a leaf, the reference model becomes a tree of `NamedArray(array=None)` objects that silently poison downstream operations.

---

## Detailed Analysis

### 1. `haliax/nn/scan.py` - auto_sharded after vmap

**Change**: Added `stacked = haliax.auto_sharded(stacked)` after `haliax.vmap` in `Stacked.init`

**Original code**:
```python
def fn(*args, **kwargs):
    stacked = haliax.vmap(module.init, Block)(*args, **kwargs)
    return Stacked(stacked, Block, gradient_checkpointing)
```

**New code**:
```python
def fn(*args, **kwargs):
    stacked = haliax.vmap(module.init, Block)(*args, **kwargs)
    stacked = haliax.auto_sharded(stacked)  # <-- Added
    return Stacked(stacked, Block, gradient_checkpointing)
```

**Why it's NECESSARY**: This change ensures stacked transformer layers are sharded across devices immediately after creation. Without it:

1. After vmap, the Block axis IS properly added to NamedArray axes
2. But the underlying arrays are not yet sharded - they're replicated on every device
3. For large models (e.g., Llama 8B with 32 layers), this causes OOM errors
4. The `auto_sharded` call distributes the stacked parameters according to the axis mapping

**Key distinction**: The `batch_dim` check in `partitioning.py` makes `auto_sharded` safe to call *inside* vmap (where it becomes a no-op). But *after* vmap completes, `auto_sharded` is essential for memory efficiency.

**Verified by testing**: Removing this line causes OOM on v5p-32 (43.84G needed, 37.13G available) because all transformer layers are replicated instead of sharded.

**DPO-specific impact**: DPO loads **two** Llama 8B models (policy + reference), doubling memory requirements. Without immediate sharding after vmap, OOM is guaranteed on typical TPU configurations. Standard LM training with a single model might survive with replicated layers on larger TPUs, masking this bug.

**Recommendation**: **KEEP** this change.

---

### 2. `haliax/partitioning.py` - Skip sharding for batched tracers

**Change**: Added early return when array has `batch_dim` attribute

```python
if getattr(named.array, "batch_dim", None) is not None:
    # Batched tracers from vmap can't be safely device_put with axis-mapped sharding
    # because the leading batch axis isn't represented in the NamedArray axes.
    # We'll shard after vmap adds the axis.
    return named
```

**Why it's NECESSARY**: This is the correct fix for the vmap + sharding issue. During `haliax.vmap`:

1. JAX's vmap adds a batch dimension to the underlying array (position 0)
2. NamedArray still has its original axes (without the batch axis)
3. `pspec_for` computes a PartitionSpec from NamedArray.axes
4. The pspec has N elements but the array has N+1 dimensions
5. `jax.device_put` fails with shape mismatch

The `batch_dim` attribute is set by JAX on batched tracers, making it a reliable indicator that we're inside a vmap. Deferring sharding is correct - the array will be sharded after vmap completes and axes are properly reconciled.

**DPO-specific impact**: DPO model initialization creates both policy and reference via `Stacked.init`, which uses vmap internally. The interaction between vmap (for layer stacking) and auto_sharded (for memory efficiency) is exercised more heavily in DPO due to initializing two complete models.

**Recommendation**: **KEEP** this change.

---

### 3. `haliax/partitioning.py` - Handle array=None in pspec_for

**Change**: Added check in `pspec_for` to handle NamedArray with None array

```python
def partition_spec(node: typing.Any):
    if isinstance(node, NamedArray):
        if not is_jax_array_like(node.array):  # <-- Added
            return None
        return pspec_for_axis(node.axes, resource_mapping)
```

**Why it's NECESSARY**: `NamedArray(array=None)` can legitimately exist in several scenarios:

1. After `eqx.filter(model, lambda _: False)` - creates placeholder with None
2. During shape-only evaluation (`eqx.filter_eval_shape`)
3. When combining partial model states

Without this check, `pspec_for` would try to access `node.axes`, which calls `jnp.shape(self.array)` and fails on None.

**DPO-specific impact**: DPO's frozen reference model creates `NamedArray(array=None)` placeholders throughout the non-trainable partition. Any operation that iterates over model parameters and calls `pspec_for` (e.g., checkpointing, sharding, model averaging) will crash without this fix. Standard LM training rarely creates None placeholders because all params are trainable.

**Recommendation**: **KEEP** this change.

---

### 4. `haliax/quantization.py` - NamedArray as leaf + handle None updates

**Changes**:
1. Added NamedArray to `is_leaf` in `partition_for_grad_overwrite`:
```python
def is_leaf(v):
    return isinstance(v, (OverwriteWithGradient, NamedArray))  # NamedArray added
```

2. Added NamedArray to `is_leaf` and None handling in `apply_updates`:
```python
def _apply_update(tree, update, overwrite):
    if overwrite is not None:
        return overwrite
    if update is None:  # <-- Added
        return tree
    return eqx.apply_updates(tree, update)

def is_leaf(x):
    return x is None or isinstance(x, OverwriteWithGradient) or isinstance(x, NamedArray)  # NamedArray added
```

**Why it's NECESSARY**: Without treating NamedArray as a leaf, `eqx.partition` recurses into its PyTree structure:

```python
# NamedArray PyTree: children=(array,), aux=axis_names
na = NamedArray(jnp.array([1.0, 2.0]), ('features',))
eqx.partition(na, lambda _: False)
# Returns: (NamedArray(array=None, axes=('features',)), None)  <- CORRUPTED!
```

This `NamedArray(array=None)` passes `is not None` checks and silently replaces real model weights. By treating NamedArray as a leaf:

```python
eqx.partition(na, lambda _: False, is_leaf=lambda x: isinstance(x, NamedArray))
# Returns: (None, NamedArray(...))  <- Correct!
```

The `update is None` check is also necessary because when a parameter has no gradient (frozen), `updates` will be None for that leaf, and we should keep the original value.

**DPO-specific impact**: This is the **most critical** fix for DPO. The entire reference model (billions of parameters) has `updates=None` because it's frozen. Without treating NamedArray as a leaf:
- `eqx.partition` decomposes each NamedArray into `(array, axes)`
- The filter returns `False` for frozen params
- Result: `NamedArray(array=None, axes=original_axes)` - a corrupted placeholder that looks valid but contains no data
- These corrupted NamedArrays propagate through the model, causing silent data loss or crashes

In standard LM training, all params are trainable, so this code path is never exercised.

**Recommendation**: **KEEP** these changes.

---

### 5. `levanter/trainer_state.py` - Overcomplicated rewrites

**Changes**: Completely rewrote `_partition_trainable_params` and `cast_params_by_trainability` using manual `tree_map` instead of `eqx.partition`.

**Original**:
```python
def _partition_trainable_params(model, filter):
    filter = make_floating_point_trainable_filter(filter)
    return eqx.partition(model, filter)

def cast_params_by_trainability(model, mp, is_trainable):
    trainable, non_trainable = _partition_trainable_params(model, is_trainable)
    trainable = mp.cast_to_param(trainable)
    non_trainable = mp.cast_to_compute(non_trainable)
    model = eqx.combine(trainable, non_trainable)
    return model
```

**New** (60+ lines of manual tree_map logic)

**Why it's OVERCOMPLICATED**: The fix in `quantization.py` already demonstrates the correct pattern - just add `is_leaf`. The same pattern works here:

```python
def _partition_trainable_params(model, filter):
    filter = make_floating_point_trainable_filter(filter)
    return eqx.partition(model, filter, is_leaf=lambda x: isinstance(x, hax.NamedArray))
```

The new code does work correctly, but it's ~50 lines of manual tree walking that duplicates what `eqx.partition` already does, just to avoid passing `is_leaf`.

**The helper functions** (`_fill_missing_namedarrays`, `_has_missing_namedarrays`) were added as safety nets for model averaging. With the quantization.py fixes, these may not be strictly necessary, but they don't hurt as defensive checks.

**DPO-specific impact**: These functions are called during:
- `_partition_trainable_params`: Separates policy (trainable) from reference (frozen)
- `cast_params_by_trainability`: Casts policy to f32 for gradients, reference to bf16 for compute
- Checkpointing: Saves/loads the DpoModel structure

In standard LM training, the trainable filter is simply `True` for all params, so partition/combine operations are trivial. DPO's mixed trainability (policy=True, reference=False) exercises these code paths with non-trivial filter specs.

**Recommendation**: **SIMPLIFY** by using `is_leaf` with `eqx.partition` instead of manual tree_map. Keep the helper functions as defensive checks.

---

## Proposed Changes

### Simplify (1 change to Levanter)

**`lib/levanter/src/levanter/trainer_state.py`** - Replace manual tree_map with is_leaf:

```python
import haliax as hax

def _partition_trainable_params(model, filter):
    """
    Partitions the model into trainable and non-trainable parameters.
    """
    filter = make_floating_point_trainable_filter(filter)
    return eqx.partition(model, filter, is_leaf=lambda x: isinstance(x, hax.NamedArray))


def cast_params_by_trainability(model, mp, is_trainable):
    """
    Casts the parameters of a model to the appropriate precision based on trainability.
    """
    trainable, non_trainable = _partition_trainable_params(model, is_trainable)
    trainable = mp.cast_to_param(trainable)
    non_trainable = mp.cast_to_compute(non_trainable)
    model = eqx.combine(trainable, non_trainable, is_leaf=lambda x: isinstance(x, hax.NamedArray))
    return model
```

Keep the `_fill_missing_namedarrays` and `_has_missing_namedarrays` helper functions as defensive checks in `eval_model`.

---

## Summary of Final Haliax State vs Main

After proposed changes:

| File | Lines Changed from Main | Reason |
|------|------------------------|--------|
| `nn/scan.py` | **+1** | auto_sharded after vmap for memory efficiency |
| `partitioning.py` | **+7** | batch_dim check + array=None check |
| `quantization.py` | **+6** | NamedArray as leaf + update=None handling |

**Total: 14 lines added to Haliax**

These changes are minimal, well-targeted fixes that align with Haliax's design philosophy: **NamedArray should be treated as an atomic leaf in tree operations**. This is not a workaround - it's the intended usage pattern, as evidenced by Haliax's own `tree_util` module which does this by default.

---

## Verification Plan

1. **Run DPO tests**: `pytest lib/levanter/tests/test_dpo.py -v`
2. **Run full Levanter test suite**: `pytest lib/levanter/tests/ -v`
3. **Test the specific regression cases**:
   - `test_vmapped_init_with_sharding_handles_layer_axis` - vmap + sharding
   - `test_partition_for_grad_overwrite_preserves_namedarrays` - quantization fix
   - `test_pspec_for_handles_filtered_namedarray` - pspec_for fix
4. **Run a small DPO training job** with the tiny config to verify end-to-end

---

## Root Cause Summary

### Is Treating NamedArray as a Leaf an Anti-Pattern?

**No - it's the INTENDED design pattern in Haliax.**

NamedArray is registered as a PyTree (`@jax.tree_util.register_pytree_node_class`) so it works with JAX transformations (jit, vmap, grad), but **Haliax's own tree utilities treat NamedArray as a leaf by default**:

```python
# From haliax/tree_util.py
def tree_map(fn, tree, *rest, is_leaf=None):
    """Version of jax.tree_util.tree_map that automatically treats NamedArrays as leaves."""
    if is_leaf is None:
        is_leaf = lambda x: isinstance(x, NamedArray)  # <-- DEFAULT BEHAVIOR
    # ...
```

Haliax provides `haliax.tree_util.tree_map`, `tree_flatten`, `tree_leaves`, and `tree_structure` - all of which treat NamedArray as a leaf by default. The docstrings explicitly state this:

> "Version of [jax.tree_util.tree_map][] that **automatically treats NamedArrays as leaves**."

**The design philosophy:**
1. NamedArray is a PyTree so JAX transformations work (jit, vmap, grad traverse into it)
2. For tree operations (mapping, partitioning, combining), treat it as an **atomic unit**
3. Haliax provides wrappers that do this automatically
4. When using raw JAX/Equinox (`jax.tree_util.*`, `eqx.partition`, `eqx.combine`), you must pass `is_leaf` yourself

**The issue in Levanter:** Code was using raw `eqx.partition`/`eqx.combine` without `is_leaf`, causing NamedArray to be decomposed into `(array, axis_names)` children. The fix aligns with Haliax's intended usage.

---

The core issue is that **NamedArray is a PyTree**, but many Equinox/JAX utilities assume they can safely recurse into PyTrees. When they do:

1. `eqx.partition` creates `NamedArray(array=None)` placeholders
2. `pspec_for` tries to read shape from None
3. `auto_sharded` during vmap sees mismatched array/axes dimensions

The fixes treat NamedArray as an **atomic leaf** - which is exactly what Haliax's own utilities do by default. This is the correct approach because:
1. **Haliax design**: `haliax.tree_util.*` functions all default to `is_leaf=lambda x: isinstance(x, NamedArray)`
2. **Semantic integrity**: A NamedArray's `array` and `axis_names` are inseparable - one without the other is meaningless
3. **Consistency**: When you partition/filter a model, you want whole NamedArrays, not their internal components

### Why Standard LM Training Doesn't Expose These Bugs

| Operation | LM Training | DPO Training |
|-----------|-------------|--------------|
| `eqx.partition(model, trainable)` | `trainable=True` for all → no None placeholders | `trainable=False` for reference → NamedArray(None) everywhere |
| Memory for model init | Single model fits even if not optimally sharded | Two models require immediate sharding or OOM |
| Gradient computation | All params get gradients | Reference params frozen → None updates |
| Model structure | Flat LmHeadModel | Nested DpoModel with mixed trainability |

DPO is essentially a **stress test** for JAX/Equinox/Haliax interactions because it:
1. Loads 2x the parameters (policy + reference)
2. Has mixed trainability (some params frozen)
3. Requires careful memory management (sharding both models)
4. Uses nested model structures (DpoModel wrapping two LmHeadModels)
