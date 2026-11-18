# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Strict 10/05 checkpoint restore into the Mochi long-loop state layout."""

import dataclasses
import logging
import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from haliax.jax_utils import is_jax_array_like
from jax.experimental.multihost_utils import process_allgather
from levanter.checkpoint import load_checkpoint
from levanter.checkpoint_manifest import read_manifest
from levanter.utils.jax_utils import leaf_key_paths

logger = logging.getLogger(__name__)
_EXPECTED_MOMENT_LEAVES = 36
_OPTIMIZER_NON_MOMENT_PATHS = frozenset(
    {
        "opt_state/count",
        "opt_state/hyperparams/adam_lr",
        "opt_state/hyperparams/learning_rate",
        "opt_state/hyperparams_states/adam_lr/count",
        "opt_state/hyperparams_states/learning_rate/count",
        "opt_state/inner_state/inner_states/adam/inner_state/0/count",
        "opt_state/inner_state/inner_states/adamh/inner_state/0/count",
    }
)


def _shared(state, *, legacy: bool):
    """Change only the single shared-expert container on params and Muon momentum."""
    params_shared = state.params.stacked_blocks.stacked.shared
    momentum = state.opt_state.inner_state.inner_states["muonh"].inner_state[0].momentum_buffer
    momentum_shared = momentum.stacked_blocks.stacked.shared
    if legacy:
        if not isinstance(params_shared, tuple) or len(params_shared) != 1:
            raise ValueError("expected exactly one current-layout shared parameter expert")
        if not isinstance(momentum_shared, tuple) or len(momentum_shared) != 1:
            raise ValueError("expected exactly one current-layout shared momentum expert")
        params_value, momentum_value = params_shared[0], momentum_shared[0]
    else:
        if isinstance(params_shared, tuple) or isinstance(momentum_shared, tuple):
            raise ValueError("expected unwrapped legacy shared experts")
        params_value, momentum_value = (params_shared,), (momentum_shared,)
    params = eqx.tree_at(lambda x: x.stacked_blocks.stacked.shared, state.params, params_value)
    opt_state = eqx.tree_at(
        lambda x: x.inner_state.inner_states["muonh"].inner_state[0].momentum_buffer.stacked_blocks.stacked.shared,
        state.opt_state,
        momentum_value,
    )
    return dataclasses.replace(state, params=params, opt_state=opt_state)


def _array_specs(tree):
    serializable, _ = eqx.partition(tree, is_jax_array_like)
    paths = jax.tree.leaves(leaf_key_paths(serializable))
    values = jax.tree.leaves(serializable)
    if len(paths) != len(values):
        raise ValueError("checkpoint template path/leaf mismatch")
    specs = {}
    for path, value in zip(paths, values, strict=True):
        key = path.replace(".", "/")
        if key in specs:
            raise ValueError(f"duplicate checkpoint template path: {key}")
        specs[key] = (tuple(value.shape), str(value.dtype))
    return specs


def _fp32_template(tree):
    """Describe saved FP32 leaves without allocating or changing any optimizer value."""

    def source_leaf(leaf):
        if not is_jax_array_like(leaf):
            return leaf
        dtype = jnp.float32 if leaf.dtype == jnp.bfloat16 else leaf.dtype
        return jax.ShapeDtypeStruct(leaf.shape, dtype, sharding=getattr(leaf, "sharding", None))

    return jax.tree.map(source_leaf, tree)


def _is_moment_path(path):
    return path.startswith("opt_state/") and any(f"/{name}/" in path for name in ("mu", "nu", "momentum_buffer"))


def _audit_moment_cast_reference(path, source, converted, *, chunk_elements=1 << 20):
    """Report local shard cast statistics; reject any nonzero FP32 value becoming BF16 zero.

    Counts are shard-local and may include replicas; indices let a later receipt deduplicate them.
    This reference is for small CPU fixtures only; production uses device reductions.
    """
    if chunk_elements <= 0 or chunk_elements > (1 << 24):
        raise ValueError("invalid cast-audit chunk size")
    bad = False
    before_shards = source.addressable_shards
    after_shards = converted.addressable_shards
    if len(before_shards) != len(after_shards):
        raise ValueError("cast changed addressable shard count")
    for before, after in zip(before_shards, after_shards, strict=True):
        if before.index != after.index or before.device != after.device:
            raise ValueError("cast changed shard identity")
        # CPU-only oracle: materialize the small test shard before chunking.
        a = np.asarray(jax.device_get(before.data)).reshape(-1)
        b = np.asarray(jax.device_get(after.data)).reshape(-1)
        count = nonzero = lost = subnormal = nonfinite = 0
        min_abs_bits = None
        max_relative = 0.0
        for start in range(0, a.size, chunk_elements):
            x = np.asarray(a[start : start + chunk_elements], dtype=np.float32)
            y = np.asarray(b[start : start + chunk_elements])
            bits = x.view(np.uint32) & np.uint32(0x7FFFFFFF)
            ybits = y.view(np.uint16) & np.uint16(0x7FFF)
            finite = bits < np.uint32(0x7F800000)
            nonzero_mask = (bits != 0) & finite
            count += int(x.size)
            nonzero += int(np.count_nonzero(nonzero_mask))
            lost += int(np.count_nonzero(nonzero_mask & (ybits == 0)))
            subnormal += int(np.count_nonzero((ybits != 0) & (ybits < np.uint16(0x80))))
            nonfinite += int(np.count_nonzero(~finite))
            if np.any(nonzero_mask):
                smallest = int(np.min(bits[nonzero_mask]))
                min_abs_bits = smallest if min_abs_bits is None else min(min_abs_bits, smallest)
                xd = x[nonzero_mask].astype(np.float64)
                yd = y[nonzero_mask].astype(np.float64)
                max_relative = max(max_relative, float(np.max(np.abs(yd - xd) / np.abs(xd))))
        min_nonzero = None if min_abs_bits is None else float(np.asarray(min_abs_bits, dtype=np.uint32).view(np.float32))
        logger.info(
            "carried_moment_cast path=%s process=%d shard=%s device=%s elements=%d nonzero=%d "
            "nonzero_to_zero=%d fraction=%s bf16_subnormal=%d nonfinite=%d min_abs_nonzero=%s max_relative=%s "
            "scope=addressable_shard_may_include_replicas",
            path,
            jax.process_index(),
            before.index,
            before.device,
            count,
            nonzero,
            lost,
            lost / nonzero if nonzero else 0.0,
            subnormal,
            nonfinite,
            min_nonzero,
            max_relative if math.isfinite(max_relative) else str(max_relative),
        )
        bad |= lost > 0
    # Every process must agree before raising: the next per-leaf load may use collectives.
    any_bad = np.asarray(process_allgather(np.asarray([bad], dtype=np.int32), tiled=True)).any()
    if any_bad:
        raise ValueError(f"nonzero FP32 moment became BF16 zero in {path}")


def _device_moment_stats(source, converted):
    """Global reductions, with per-leading-slice uint32 counts for large leaves."""
    source_bits = jax.lax.bitcast_convert_type(source, jnp.uint32) & jnp.uint32(0x7FFFFFFF)
    converted_bits = jax.lax.bitcast_convert_type(converted, jnp.uint16) & jnp.uint16(0x7FFF)
    finite = source_bits < jnp.uint32(0x7F800000)
    nonzero = (source_bits != 0) & finite
    lost = nonzero & (converted_bits == 0)
    target_subnormal = (converted_bits != 0) & (converted_bits < jnp.uint16(0x80))
    source_subnormal = nonzero & (source_bits < jnp.uint32(0x800000))
    # The largest current leaf has >2^32 elements, but each leading slice is <2^31.
    large = math.prod(source.shape) >= (1 << 31)
    axes = tuple(range(1, source.ndim)) if large else tuple(range(source.ndim))
    count_nonzero = jnp.sum(nonzero, axis=axes, dtype=jnp.uint32)
    count_lost = jnp.sum(lost, axis=axes, dtype=jnp.uint32)
    count_subnormal = jnp.sum(target_subnormal, axis=axes, dtype=jnp.uint32)
    min_bits = jnp.min(jnp.where(nonzero, source_bits, jnp.uint32(0x7F800000)))
    # Device arithmetic may flush subnormals: flag that condition alongside this statistic.
    x = jnp.where(nonzero & ~source_subnormal, source, jnp.float32(1.0))
    relative = jnp.where(nonzero & ~source_subnormal, jnp.abs(converted.astype(jnp.float32) - x) / jnp.abs(x), 0.0)
    return (
        jnp.any(lost),
        jnp.any(~finite),
        jnp.any(source_subnormal),
        count_nonzero,
        count_lost,
        count_subnormal,
        min_bits,
        jnp.max(relative),
    )


def _audit_moment_cast(path, source, converted):
    """Gate on globally reduced lost values, retaining exact per-slice count sums on host."""
    if source.ndim and math.prod(source.shape[1:]) >= (1 << 31):
        raise ValueError(f"moment slice too large for exact uint32 counts: {path}")
    sharding = source.sharding
    if not isinstance(sharding, jax.sharding.NamedSharding):
        raise ValueError(f"moment source requires NamedSharding: {path}")
    replicated = jax.sharding.NamedSharding(sharding.mesh, jax.sharding.PartitionSpec())
    lost_flag, nonfinite_flag, source_subnormal, nonzero, lost, subnormal, min_bits, max_relative = jax.device_get(
        jax.jit(_device_moment_stats, out_shardings=replicated)(source, converted)
    )
    total_nonzero = sum(int(x) for x in np.asarray(nonzero).reshape(-1))
    total_lost = sum(int(x) for x in np.asarray(lost).reshape(-1))
    total_subnormal = sum(int(x) for x in np.asarray(subnormal).reshape(-1))
    min_nonzero = None if total_nonzero == 0 else float(np.asarray(min_bits, dtype=np.uint32).view(np.float32))
    if jax.process_index() == 0:
        logger.info(
            "carried_moment_cast path=%s elements=%d nonzero=%d nonzero_to_zero=%d fraction=%s "
            "bf16_subnormal=%d fp32_subnormal_present=%s min_abs_nonzero=%s max_relative_normal_only=%s "
            "scope=global_sharded_tensor",
            path,
            math.prod(source.shape),
            total_nonzero,
            total_lost,
            total_lost / total_nonzero if total_nonzero else 0.0,
            total_subnormal,
            bool(source_subnormal),
            min_nonzero,
            float(max_relative),
        )
    if bool(nonfinite_flag):
        raise ValueError(f"nonfinite FP32 moment in {path}")
    if bool(lost_flag):
        raise ValueError(f"nonzero FP32 moment became BF16 zero in {path}")


def _restore_one_leaf_at_a_time(target, checkpoint, *, mesh):
    """Load and convert one manifest leaf at a time; no complete FP32 tree is built."""
    serializable, nonserializable = eqx.partition(target, is_jax_array_like)
    leaves, treedef = jax.tree.flatten(serializable)
    paths = jax.tree.leaves(leaf_key_paths(serializable))
    if len(paths) != len(leaves):
        raise ValueError("checkpoint template path/leaf mismatch")
    restored = []
    for path, target_leaf in zip(paths, leaves, strict=True):
        key = path.replace(".", "/")
        parent, _, name = key.rpartition("/")
        dtype = jnp.float32 if target_leaf.dtype == jnp.bfloat16 else target_leaf.dtype
        exemplar = jax.ShapeDtypeStruct(target_leaf.shape, dtype, sharding=getattr(target_leaf, "sharding", None))
        # A one-key exemplar avoids a spurious trailing slash in OCDBT paths.
        loaded = load_checkpoint({name: exemplar}, checkpoint, subpath=parent or None, mesh=mesh, allow_partial=False)
        value = loaded.pop(name)
        del loaded
        if value.dtype != target_leaf.dtype:
            if value.dtype != jnp.float32 or target_leaf.dtype != jnp.bfloat16:
                raise ValueError(f"unsupported restored dtype conversion: {value.dtype} to {target_leaf.dtype}")
            converted = value.astype(jnp.bfloat16)
            converted.block_until_ready()
            if _is_moment_path(key):
                _audit_moment_cast(key, value, converted)
            value.delete()
            del value
            restored.append(converted)
        else:
            if _is_moment_path(key):
                if not isinstance(value.sharding, jax.sharding.NamedSharding):
                    raise ValueError(f"moment source requires NamedSharding: {key}")
                replicated = jax.sharding.NamedSharding(value.sharding.mesh, jax.sharding.PartitionSpec())
                nonfinite = jax.jit(lambda x: jnp.any(~jnp.isfinite(x)), out_shardings=replicated)(value)
                if bool(jax.device_get(nonfinite)):
                    raise ValueError(f"nonfinite FP32 moment in {key}")
                logger.info(
                    "carried_moment_cast path=%s process=%d dtype=%s cast=none; retained exactly",
                    key,
                    jax.process_index(),
                    value.dtype,
                )
            restored.append(value)
    return eqx.combine(jax.tree.unflatten(treedef, restored), nonserializable)


def restore_1005_carried_state(state, checkpoint: str, *, mesh, expected_step: int):
    """Strictly restore every saved array and explicitly round FP32 parameters/moments to BF16.

    First re-nest a source-layout exemplar, compare every source manifest path, shape and dtype,
    then load without partial restore. The only permitted conversion is FP32-to-BF16; optimizer
    counters remain int32. Injected hyperparameters are rounded if the target optimizer uses BF16.
    The trainer step is unchanged.
    """
    if expected_step <= 0 or state.master_params is not None or state.ema_params is not None:
        raise ValueError("unsupported carried-state template")
    param_dtypes = {str(leaf.dtype) for leaf in jax.tree.leaves(state.params) if is_jax_array_like(leaf)}
    if param_dtypes != {str(jnp.dtype("bfloat16"))}:
        raise ValueError("carried-state restore requires BF16 parameters")
    target_legacy = _shared(state, legacy=True)
    template = _fp32_template(target_legacy)
    expected = _array_specs(template)
    optimizer_paths = {key for key in expected if key.startswith("opt_state/")}
    moment_paths = {key for key in optimizer_paths if _is_moment_path(key)}
    if len(moment_paths) != _EXPECTED_MOMENT_LEAVES or optimizer_paths - moment_paths != _OPTIMIZER_NON_MOMENT_PATHS:
        raise ValueError("checkpoint optimizer moment path coverage mismatch")
    arrays = read_manifest(checkpoint).arrays
    actual = {a.path: (tuple(a.shape), str(a.dtype)) for a in arrays}
    if len(actual) != len(arrays):
        raise ValueError("duplicate checkpoint array paths")
    if actual != expected:
        missing = sorted(expected.keys() - actual.keys())
        extra = sorted(actual.keys() - expected.keys())
        mismatched = sorted(k for k in expected.keys() & actual.keys() if expected[k] != actual[k])
        raise ValueError(f"checkpoint layout mismatch: missing={missing}, extra={extra}, changed={mismatched}")
    restored = _restore_one_leaf_at_a_time(target_legacy, checkpoint, mesh=mesh)
    if int(restored.step) != expected_step:
        raise ValueError("restored step differs from the explicit carried checkpoint step")
    return _shared(restored, legacy=False)
