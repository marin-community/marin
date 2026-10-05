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


def _pin_sharding(x, ref):
    """Pin ``x`` to ``ref``'s named sharding so a following norm reduces correctly.

    The hyperball projection divides by ``norm(new_p)``; when ``new_p`` (a computed intermediate) is left
    with an SPMD-inferred sharding, the sharded reduction can over-count and collapse the whole tensor
    (issue #8073). Resharding to the parameter's own sharding is a same-layout no-op at runtime.
    """
    sharding = getattr(ref, "sharding", None)
    if sharding is None:
        sharding = getattr(jax.typeof(ref), "sharding", None)
    if isinstance(sharding, jax.sharding.NamedSharding):
        return jax.sharding.reshard(x, sharding)
    return x


class _StepWithStats(NamedTuple):
    """One leaf's hyperball update and its per-sphere stats (a distinct type: optax's MaskedNode is a tuple too)."""

    update: jax.Array
    stats: jax.Array


class ScaleByAdamHState(NamedTuple):
    count: chex.Array
    mu: optax.Updates
    nu: optax.Updates
    hyperball: optax.Updates | None = None
    """Last step's per-matrix ``[decay, cos, rescale]`` (``log_hyperball``; as ``optimizer.HYPERBALL_STATS``)."""


def scale_by_adamh(
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    learning_rate: float = 0.02,
    mu_dtype: Any | None = None,
    log_hyperball: bool = False,
) -> optax.GradientTransformation:
    """AdamH: the Adam direction taken as a hyperball step (fixed Frobenius norm per matrix). ``log_hyperball`` keeps
    each matrix's step ``[decay, cos, rescale]`` in the state: ``1 - decay = |W| / |W + u|`` is the re-projection's
    shrink, ``cos`` the step's radial cosine and ``rescale = lr |W| / |d|`` the factor on the Adam direction ``d``."""
    mu_dtype = jax.dtypes.canonicalize_dtype(mu_dtype)

    def stats_init(p):
        if p is None:
            return None
        return jnp.zeros((3, *p.shape[:-2], 1, 1), jnp.float32)

    def init_fn(params):
        mu = otu.tree_zeros_like(params, dtype=mu_dtype)
        nu = otu.tree_zeros_like(params)
        hyperball = jax.tree.map(stats_init, params, is_leaf=lambda x: x is None) if log_hyperball else None
        return ScaleByAdamHState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu, hyperball=hyperball)

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

        def _norm(x):
            # jnp.linalg.norm over a sharded matrix mis-lowers under SPMD and over-counts (issue #8073);
            # sum-of-squares in float32 reduces correctly.
            return jnp.sqrt(jnp.sum(jnp.square(x.astype(jnp.float32))))

        def _scale_invariant_2d(p, u):
            """Core update for a 2-D (matrix) parameter, and its ``[decay, cos, rescale]``."""
            p_norm = _norm(p)
            u_norm = _norm(u)
            rescale = learning_rate * p_norm / jnp.maximum(u_norm, 1e-10)
            new_p = p - u * rescale
            new_p = _pin_sharding(new_p, p)
            new_p_norm = _norm(new_p)
            step = new_p.astype(jnp.float32) - p.astype(jnp.float32)
            decay = 1.0 - p_norm / jnp.maximum(new_p_norm, 1e-10)
            cos = jnp.sum(p.astype(jnp.float32) * step) / jnp.maximum(p_norm * _norm(step), 1e-20)
            stats = jax.lax.stop_gradient(jnp.stack([decay, cos, rescale]))
            return new_p / jnp.maximum(new_p_norm, 1e-10) * p_norm - p, stats

        def scale_invariant_update(p, u):
            if p is None:
                return None
            if p.ndim <= 2:
                update, stats = _scale_invariant_2d(p, u)
                return _StepWithStats(update, stats.reshape(3, 1, 1))
            # For higher-rank tensors, vmap the 2-D logic over the leading axis.
            update, stats = jax.vmap(_scale_invariant_2d)(p, u)
            return _StepWithStats(update, jnp.moveaxis(stats, -1, 0)[..., None, None])

        out = jax.tree_util.tree_map(scale_invariant_update, params, adam_updates, is_leaf=lambda x: x is None)
        is_pair = lambda x: x is None or isinstance(x, _StepWithStats)  # noqa: E731
        adamh_updates = jax.tree.map(lambda x: x.update if isinstance(x, _StepWithStats) else x, out, is_leaf=is_pair)
        hyperball = (
            jax.tree.map(lambda x: x.stats if isinstance(x, _StepWithStats) else x, out, is_leaf=is_pair)
            if log_hyperball
            else None
        )

        return adamh_updates, ScaleByAdamHState(count=count_inc, mu=mu, nu=nu, hyperball=hyperball)

    return optax.GradientTransformation(init_fn, update_fn)


__all__ = ["ScaleByAdamHState", "scale_by_adamh"]
