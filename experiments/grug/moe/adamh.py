# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
#
# Local copy of AdamH for iteration without modifying Levanter.
# Adapted from levanter.optim.adamh.

from collections import defaultdict
from typing import Any, NamedTuple

import chex
import jax
import jax.numpy as jnp
import optax
from optax import tree_utils as otu

from levanter.utils.jax_utils import leaf_key_paths


class ScaleByAdamHState(NamedTuple):
    count: chex.Array
    mu: optax.Updates
    nu: optax.Updates


def _module_key(path: str | None) -> str | None:
    if path is None:
        return None
    parts = path.split(".")
    if len(parts) <= 1:
        return path
    return ".".join(parts[:-1])


def normalize_module_gradients_to_unit_rms(updates: optax.Updates, eps: float = 1e-16) -> optax.Updates:
    """Normalize each module's gradient leaves to combined RMS 1."""
    leaves, treedef = jax.tree_util.tree_flatten(updates, is_leaf=lambda x: x is None)
    paths = treedef.flatten_up_to(leaf_key_paths(updates))
    groups: dict[str, list[int]] = defaultdict(list)

    for index, (leaf, path) in enumerate(zip(leaves, paths, strict=True)):
        if leaf is None or not hasattr(leaf, "shape"):
            continue
        module_key = _module_key(path)
        if module_key is None:
            continue
        groups[module_key].append(index)

    normalized_leaves = list(leaves)
    for group_indices in groups.values():
        square_sum = sum(
            (jnp.sum(jnp.square(leaves[index].astype(jnp.float32))) for index in group_indices),
            jnp.array(0.0, dtype=jnp.float32),
        )
        num_elements = sum(int(leaves[index].size) for index in group_indices)
        inv_rms = jax.lax.rsqrt(square_sum / num_elements + eps)
        for index in group_indices:
            leaf = leaves[index]
            normalized_leaves[index] = leaf * inv_rms.astype(leaf.dtype)

    return jax.tree_util.tree_unflatten(treedef, normalized_leaves)


def scale_by_adamh(
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    learning_rate: float = 0.02,
    mu_dtype: Any | None = None,
) -> optax.GradientTransformation:
    mu_dtype = jax.dtypes.canonicalize_dtype(mu_dtype)

    def init_fn(params):
        mu = otu.tree_zeros_like(params, dtype=mu_dtype)
        nu = otu.tree_zeros_like(params)
        return ScaleByAdamHState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu)

    def update_fn(updates, state, params):
        mu = otu.tree_update_moment(updates, state.mu, b1, 1)
        nu = otu.tree_update_moment_per_elem_norm(updates, state.nu, b2, 2)
        count_inc = optax.safe_increment(state.count)
        mu_hat = otu.tree_bias_correction(mu, b1, count_inc)
        nu_hat = otu.tree_bias_correction(nu, b2, count_inc)

        adam_updates = jax.tree.map(
            lambda m, v: None if m is None else m / (jnp.sqrt(v) + eps),
            mu_hat,
            nu_hat,
            is_leaf=lambda x: x is None,
        )
        mu = otu.tree_cast(mu, mu_dtype)

        def _scale_invariant_2d(p, u):
            """Core update for a 2-D (matrix) parameter."""
            p_norm = jnp.linalg.norm(p)
            u_norm = jnp.linalg.norm(u)
            new_p = p - learning_rate * u * p_norm / jnp.maximum(u_norm, 1e-10)
            return new_p / jnp.linalg.norm(new_p) * p_norm - p

        def scale_invariant_update(p, u):
            if p is None:
                return None
            if p.ndim <= 2:
                return _scale_invariant_2d(p, u)
            # For higher-rank tensors, vmap the 2-D logic over the leading axis.
            return jax.vmap(_scale_invariant_2d)(p, u)

        adamh_updates = jax.tree_util.tree_map(
            scale_invariant_update,
            params,
            adam_updates,
            is_leaf=lambda x: x is None,
        )

        return adamh_updates, ScaleByAdamHState(count=count_inc, mu=mu, nu=nu)

    return optax.GradientTransformation(init_fn, update_fn)


def scale_by_adamh_with_module_gradient_normalization(
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    learning_rate: float = 0.02,
    mu_dtype: Any | None = None,
    gradient_norm_eps: float = 1e-16,
) -> optax.GradientTransformation:
    """AdamH with module-wise gradient RMS normalization before moment updates."""
    adamh = scale_by_adamh(b1=b1, b2=b2, eps=eps, learning_rate=learning_rate, mu_dtype=mu_dtype)

    def update_fn(updates, state, params):
        normalized_updates = normalize_module_gradients_to_unit_rms(updates, eps=gradient_norm_eps)
        return adamh.update(normalized_updates, state, params)

    return optax.GradientTransformation(adamh.init, update_fn)


__all__ = [
    "ScaleByAdamHState",
    "normalize_module_gradients_to_unit_rms",
    "scale_by_adamh",
    "scale_by_adamh_with_module_gradient_normalization",
]
