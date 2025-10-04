# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Any, NamedTuple, Optional

import chex
import jax
import jax.numpy as jnp
import optax
from optax import tree_utils as otu

from levanter.optim.config import OptimizerConfig


@OptimizerConfig.register_subclass("adopt")
@dataclass(frozen=True)
class AdoptConfig(OptimizerConfig):
    """
    Adopt optimizer configuration
    cf:
    Original Paper: https://arxiv.org/abs/2411.02853
    """

    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-6
    max_grad_norm: Optional[float] = 1.0
    haps: Optional[list[int]] = None
    schedule_list: Optional[list[str]] = None

    def build(self, num_train_steps):
        """Creates the optimizer"""

        # indirection makes it work with optax.inject_hyperparams so we can log the learning rate
        def _optimizer(learning_rate):
            components = []

            if self.max_grad_norm:
                components.append(optax.clip_by_global_norm(self.max_grad_norm))
            components.append(scale_by_adopt(self.beta1, self.beta2, self.epsilon))

            if self.weight_decay > 0:
                components.append(optax.add_decayed_weights(self.weight_decay, self.build_weight_decay_mask()))

            # - learning rate for descent
            components.append(optax.scale(-learning_rate))

            optimizer = optax.chain(*components)

            return optimizer

        return optax.inject_hyperparams(_optimizer)(learning_rate=self.lr_scheduler(num_train_steps))


class ScaleByAdoptState(NamedTuple):
    """State for the Adopt algorithm."""

    count: chex.Array  # shape=(), dtype=jnp.int32.
    mu: optax.Updates
    nu: optax.Updates


def scale_by_adopt(
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-6,
    eps_root: float = 0.0,
    mu_dtype: Optional[Any] = None,
    *,
    nesterov: bool = False,
) -> optax.GradientTransformation:
    r"""Rescale updates according to the Adopt algorithm.

    Adopt is a variant of Adam that uses a different update rule for the second moment.

    Args:
      b1: Decay rate for the exponentially weighted average of grads.
      b2: Decay rate for the exponentially weighted average of squared grads.
      eps: Term added to the denominator to improve numerical stability.
      eps_root: Term added to the denominator inside the square-root to improve
        numerical stability when backpropagating gradients through the rescaling.
      mu_dtype: Optional dtype to be used for the first order accumulator; if
        None then the dtype is inferred from params and updates.

    Returns:
      A :class:optax.GradientTransformation object.
    """

    mu_dtype = jax.dtypes.canonicalize_dtype(mu_dtype)

    def init_fn(params):
        mu = otu.tree_zeros_like(params, dtype=mu_dtype)  # First moment
        nu = otu.tree_zeros_like(params)  # Second moment
        return ScaleByAdoptState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu)

    def update_fn(updates, state, params=None):
        nu_hat = otu.tree_bias_correction(state.nu, b2, state.count)

        def zero_update(updates, nu_hat):
            # For the first step, return zero updates
            return jax.tree_util.tree_map(
                lambda u: None if u is None else jnp.zeros_like(u),
                updates,
                is_leaf=lambda x: x is None,
            )

        def regular_update(updates, nu_hat):
            # For subsequent steps, compute updates normally
            return jax.tree_util.tree_map(
                lambda m, v: None if m is None else m / (jnp.sqrt(v + eps_root) + eps),
                updates,
                nu_hat,
                is_leaf=lambda x: x is None,
            )

        adam_updates = jax.lax.cond(
            jnp.equal(state.count, 0),
            lambda args: zero_update(*args),
            lambda args: regular_update(*args),
            (updates, nu_hat),  # Arguments passed to the above functions
        )
        mu = otu.tree_update_moment(adam_updates, state.mu, b1, 1)
        nu = otu.tree_update_moment_per_elem_norm(updates, state.nu, b2, 2)
        count_inc = optax.safe_increment(state.count)
        mu_hat = otu.tree_bias_correction(mu, b1, count_inc)
        mu = otu.tree_cast(mu, mu_dtype)
        return mu_hat, ScaleByAdoptState(count=count_inc, mu=mu, nu=nu)

    return optax.GradientTransformation(init_fn, update_fn)
