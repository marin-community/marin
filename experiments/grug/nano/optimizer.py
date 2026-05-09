# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""AdamH optimizer for the nano (modded-nanogpt) experiments.

Matches `experiments/grug/nanogpt_adamh_ref.py` exactly:

    p_norm  = ||p||_F
    u_norm  = ||u||_F
    new_p   = p - lr * u * p_norm / max(u_norm, eps)
    new_norm = max(||new_p||_F, eps)
    p_next   = new_p / new_norm * p_norm

The Frobenius norm of every hidden 2D matrix is preserved across training.

Differences vs. `levanter.optim.adamh.scale_by_adamh` and the local copy in
`experiments/grug/moe/adamh.py`: those leave `new_p / ||new_p||` unprotected on
the 2-D code path (line 159 in `levanter/optim/adamh.py`), which can divide by
zero if AdamH is ever applied to a parameter whose `||p||` is zero or whose
update happens to land at the origin. The torch reference uses
`torch.clamp(new_param.norm(), min=eps)` — we mirror that with `jnp.maximum`.
"""

from dataclasses import dataclass
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


def scale_by_adamh_safe(
    b1: float = 0.9,
    b2: float = 0.95,
    eps: float = 1e-8,
    learning_rate: float = 0.018,
    norm_eps: float = 1e-10,
    mu_dtype: Any | None = None,
) -> optax.GradientTransformation:
    """AdamH with hardened divide-by-zero protection on both norms.

    Args:
        b1, b2, eps: standard Adam moments and denominator term.
        learning_rate: scalar AdamH learning rate (or a JAX scalar from
            `optax.inject_hyperparams`).
        norm_eps: clamp floor for both `||u||_F` and `||new_p||_F`. Mirrors the
            torch reference's `torch.clamp(..., min=1e-10)`.
        mu_dtype: optional dtype for first-moment storage.
    """
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

        def scale_invariant_2d(p, u):
            p_norm = jnp.linalg.norm(p)
            u_norm = jnp.linalg.norm(u)
            new_p = p - learning_rate * u * p_norm / jnp.maximum(u_norm, norm_eps)
            new_norm = jnp.maximum(jnp.linalg.norm(new_p), norm_eps)
            # Return the *delta* (new_p_renorm - p) so optax.apply_updates(p, delta) ≡ new_p_renorm.
            return new_p / new_norm * p_norm - p

        def scale_invariant_nd(p, u):
            # Per-row hyperball projection. Treats axis 0 as the batch and
            # applies the 2-D rule to each (..., -1) slice.
            axes = tuple(range(1, p.ndim))
            p_norm = jnp.sqrt(jnp.sum(jnp.square(p), axis=axes, keepdims=True))
            u_norm = jnp.sqrt(jnp.sum(jnp.square(u), axis=axes, keepdims=True))
            new_p = p - learning_rate * u * p_norm / jnp.maximum(u_norm, norm_eps)
            new_p_norm = jnp.sqrt(jnp.sum(jnp.square(new_p), axis=axes, keepdims=True))
            return new_p / jnp.maximum(new_p_norm, norm_eps) * p_norm - p

        def scale_invariant_update(p, u):
            if p is None:
                return None
            if p.ndim == 2:
                return scale_invariant_2d(p, u)
            if p.ndim > 2:
                return scale_invariant_nd(p, u)
            # 0-D or 1-D parameters should be routed away from AdamH by the
            # caller's mask; if one slips through, fall back to a plain step
            # rather than dividing by a degenerate norm.
            return -learning_rate * u

        adamh_updates = jax.tree_util.tree_map(
            scale_invariant_update,
            params,
            adam_updates,
            is_leaf=lambda x: x is None,
        )

        return adamh_updates, ScaleByAdamHState(count=count_inc, mu=mu, nu=nu)

    return optax.GradientTransformation(init_fn, update_fn)


@dataclass(frozen=True)
class AdamHHparams:
    """Convenience bundle for AdamH hyperparameters used by the nano launches."""

    lr: float = 0.018
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    norm_eps: float = 1e-10


__all__ = ["AdamHHparams", "ScaleByAdamHState", "scale_by_adamh_safe"]
