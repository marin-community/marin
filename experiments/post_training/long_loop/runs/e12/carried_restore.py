# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Strict 10/05 checkpoint restore into the Mochi long-loop state layout."""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
from haliax.jax_utils import is_jax_array_like
from levanter.checkpoint import load_checkpoint
from levanter.checkpoint_manifest import read_manifest
from levanter.utils.jax_utils import leaf_key_paths


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


def _cast_to_target(source, target):
    """Cast only saved FP32 arrays whose target template is BF16, leaving all other leaves exact."""

    def cast(value, exemplar):
        if not is_jax_array_like(value):
            return value
        if value.shape != exemplar.shape:
            raise ValueError("restored leaf shape differs from target")
        if value.dtype == exemplar.dtype:
            return value
        if value.dtype == jnp.float32 and exemplar.dtype == jnp.bfloat16:
            return value.astype(jnp.bfloat16)
        raise ValueError(f"unsupported restored dtype conversion: {value.dtype} to {exemplar.dtype}")

    return jax.tree.map(cast, source, target)


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
    arrays = read_manifest(checkpoint).arrays
    actual = {a.path: (tuple(a.shape), str(a.dtype)) for a in arrays}
    if len(actual) != len(arrays):
        raise ValueError("duplicate checkpoint array paths")
    if actual != expected:
        missing = sorted(expected.keys() - actual.keys())
        extra = sorted(actual.keys() - expected.keys())
        mismatched = sorted(k for k in expected.keys() & actual.keys() if expected[k] != actual[k])
        raise ValueError(f"checkpoint layout mismatch: missing={missing}, extra={extra}, changed={mismatched}")
    loaded = load_checkpoint(template, checkpoint, mesh=mesh, allow_partial=False)
    if int(loaded.step) != expected_step:
        raise ValueError("restored step differs from the explicit carried checkpoint step")
    cast = _cast_to_target(loaded, target_legacy)
    return _shared(cast, legacy=False)
