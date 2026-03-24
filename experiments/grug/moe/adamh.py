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


class ScaleByAdamHState(NamedTuple):
    count: chex.Array
    mu: optax.Updates
    nu: optax.Updates


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

        def scale_invariant_update(p, u):
            if p is None:
                return None
            if p.ndim == 2:
                new_p = p - learning_rate * u * jnp.linalg.norm(p) / jnp.maximum(jnp.linalg.norm(u), 1e-10)
                return new_p / jnp.linalg.norm(new_p) * jnp.linalg.norm(p) - p
            else:
                axes = tuple(range(1, p.ndim))
                p_norm = jnp.sqrt(jnp.sum(jnp.square(p), axis=axes, keepdims=True))
                u_norm = jnp.sqrt(jnp.sum(jnp.square(u), axis=axes, keepdims=True))
                new_p = p - learning_rate * u * p_norm / jnp.maximum(u_norm, 1e-10)
                new_p_norm = jnp.sqrt(jnp.sum(jnp.square(new_p), axis=axes, keepdims=True))
                return new_p / jnp.maximum(new_p_norm, 1e-10) * p_norm - p

        adamh_updates = jax.tree_util.tree_map(
            scale_invariant_update,
            params,
            adam_updates,
            is_leaf=lambda x: x is None,
        )

        return adamh_updates, ScaleByAdamHState(count=count_inc, mu=mu, nu=nu)

    return optax.GradientTransformation(init_fn, update_fn)


__all__ = ["ScaleByAdamHState", "scale_by_adamh"]
