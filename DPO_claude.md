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

These changes are minimal, well-targeted fixes for real JAX/Equinox interaction issues with NamedArray, not workarounds or hacks.

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

The core issue is that **NamedArray is a PyTree**, but many Equinox/JAX utilities assume they can safely recurse into PyTrees. When they do:

1. `eqx.partition` creates `NamedArray(array=None)` placeholders
2. `pspec_for` tries to read shape from None
3. `auto_sharded` during vmap sees mismatched array/axes dimensions

The fixes treat NamedArray as an **atomic leaf** in contexts where recursion is problematic. This is the correct approach - NamedArray should be treated as a unit, not decomposed into its array + axis_names components during model transformations.
