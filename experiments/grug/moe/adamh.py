# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
#
# Local copy of AdamH for iteration without modifying Levanter.
# Adapted from levanter.optim.adamh.

from typing import Any, NamedTuple

import chex
import jax
import jax.numpy as jnp
import optax
from optax import tree_utils as otu

from experiments.grug.moe.optimizer_sharding import assert_update_sharding_matches_params


class ScaleByAdamHState(NamedTuple):
    count: chex.Array
    mu: optax.Updates
    nu: optax.Updates


def _scale_invariant_hyperball_update(param: jax.Array, update: jax.Array, learning_rate: float) -> jax.Array:
    """Return AdamH's norm-preserving update without materializing the new parameter."""
    if param.ndim <= 2:
        param_norm = jnp.linalg.norm(param)
        update_norm = jnp.linalg.norm(update)
        step_scale = learning_rate * param_norm / jnp.maximum(update_norm, 1e-10)
        dot = jnp.sum(param * update)
        new_param_norm_sq = param_norm**2 - 2 * step_scale * dot + step_scale**2 * update_norm**2
        new_param_norm = jnp.sqrt(jnp.maximum(new_param_norm_sq, 1e-30))
        rescale = param_norm / jnp.maximum(new_param_norm, 1e-10)
        return (rescale - 1) * param - rescale * step_scale * update

    axes = tuple(range(1, param.ndim))
    param_norm = jnp.sqrt(jnp.sum(jnp.square(param), axis=axes, keepdims=True))
    update_norm = jnp.sqrt(jnp.sum(jnp.square(update), axis=axes, keepdims=True))
    step_scale = learning_rate * param_norm / jnp.maximum(update_norm, 1e-10)
    dot = jnp.sum(param * update, axis=axes, keepdims=True)
    new_param_norm_sq = param_norm**2 - 2 * step_scale * dot + step_scale**2 * update_norm**2
    new_param_norm = jnp.sqrt(jnp.maximum(new_param_norm_sq, 1e-30))
    rescale = param_norm / jnp.maximum(new_param_norm, 1e-10)
    return (rescale - 1) * param - rescale * step_scale * update


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
        assert_update_sharding_matches_params(updates, params, "AdamH input updates")
        mu = otu.tree_update_moment(updates, state.mu, b1, 1)
        assert_update_sharding_matches_params(mu, params, "AdamH mu")
        nu = otu.tree_update_moment_per_elem_norm(updates, state.nu, b2, 2)
        assert_update_sharding_matches_params(nu, params, "AdamH nu")
        count_inc = optax.safe_increment(state.count)
        mu_hat = otu.tree_bias_correction(mu, b1, count_inc)
        assert_update_sharding_matches_params(mu_hat, params, "AdamH mu_hat")
        nu_hat = otu.tree_bias_correction(nu, b2, count_inc)
        assert_update_sharding_matches_params(nu_hat, params, "AdamH nu_hat")

        adam_updates = jax.tree.map(
            lambda m, v: None if m is None else m / (jnp.sqrt(v) + eps),
            mu_hat,
            nu_hat,
            is_leaf=lambda x: x is None,
        )
        assert_update_sharding_matches_params(adam_updates, params, "AdamH adam_updates")
        mu = otu.tree_cast(mu, mu_dtype)

        def scale_invariant_update(p, u):
            if p is None:
                return None
            return _scale_invariant_hyperball_update(p, u, learning_rate)

        adamh_updates = jax.tree_util.tree_map(
            scale_invariant_update,
            params,
            adam_updates,
            is_leaf=lambda x: x is None,
        )
        assert_update_sharding_matches_params(adamh_updates, params, "AdamH adamh_updates")

        return adamh_updates, ScaleByAdamHState(count=count_inc, mu=mu, nu=nu)

    return optax.GradientTransformation(init_fn, update_fn)


__all__ = ["ScaleByAdamHState", "scale_by_adamh"]
