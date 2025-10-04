# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Any, NamedTuple, Optional

import chex
import jax
import optax
from jax import numpy as jnp
from optax import tree_utils as otu

from levanter.optim.config import OptimizerConfig


@OptimizerConfig.register_subclass("mars")
@dataclass(frozen=True)
class MarsConfig(OptimizerConfig):
    """
    Mars optimizer configuration
    cf:
    Original Paper: https://arxiv.org/abs/2411.10438
    """

    weight_decay: float = 0.1
    beta1: float = 0.95
    beta2: float = 0.99
    gamma: float = 0.025
    epsilon: float = 1e-8
    max_grad_norm: Optional[float] = 1.0

    def build(self, num_train_steps):
        """Creates the optimizer"""

        # indirection makes it work with optax.inject_hyperparams so we can log the learning rate
        def _optimizer(learning_rate):
            components = []

            components.append(
                scale_by_mars(self.beta1, self.beta2, self.gamma, self.epsilon, max_grad_norm=self.max_grad_norm)
            )

            if self.weight_decay > 0:
                components.append(optax.add_decayed_weights(self.weight_decay, self.build_weight_decay_mask()))

            # - learning rate for descent
            components.append(optax.scale(-learning_rate))

            optimizer = optax.chain(*components)

            return optimizer

        return optax.inject_hyperparams(_optimizer)(learning_rate=self.lr_scheduler(num_train_steps))


class ScaleByMarsState(NamedTuple):
    """State for the Mars algorithm."""

    count: chex.Array  # shape=(), dtype=jnp.int32.
    mu: optax.Updates
    nu: optax.Updates
    mog: optax.Updates


def scale_by_mars(
    b1: float = 0.9,
    b2: float = 0.999,
    gamma: float = 0.05,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    max_grad_norm: float = 0.0,
    mu_dtype: Optional[Any] = None,
) -> optax.GradientTransformation:
    r"""Rescale updates according to the Mars algorithm.

    Args:
      b1: Decay rate for the exponentially weighted average of grads.
      b2: Decay rate for the exponentially weighted average of squared grads.
      gamma: Decay rate for the exponentially weighted average of the gradient from the previous step.
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
        mog = otu.tree_zeros_like(params, dtype=mu_dtype)  # gradient from
        return ScaleByMarsState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu, mog=mog)

    def update_fn(updates, state, params=None):
        c = jax.tree.map(
            lambda og, g: None if g is None else g + (gamma * b1 / (1 - b1)) * (g - og),
            state.mog,
            updates,
            is_leaf=lambda x: x is None,
        )
        if max_grad_norm:
            g_norm = optax.global_norm(c)
            scale = jnp.minimum(1.0, max_grad_norm / (g_norm + 1e-6))
            c = jax.tree_map(lambda g: None if g is None else g * scale, c, is_leaf=lambda x: x is None)
        mu = otu.tree_update_moment(c, state.mu, b1, 1)
        nu = otu.tree_update_moment_per_elem_norm(c, state.nu, b2, 2)
        count_inc = optax.safe_increment(state.count)
        mu_hat = otu.tree_bias_correction(mu, b1, count_inc)
        # Dozat 2016 https://openreview.net/pdf?id=OM0jvwB8jIp57ZJjtNEZ
        # Algorithm 2 further multiplies Adam's standard nu_hat by b2. It is
        # unclear why. Other Nadam implementations also omit the extra b2 factor.
        nu_hat = otu.tree_bias_correction(nu, b2, count_inc)
        adam_updates = jax.tree.map(
            lambda m, v: None if m is None else m / (jnp.sqrt(v + eps_root) + eps),
            mu_hat,
            nu_hat,
            is_leaf=lambda x: x is None,
        )
        mu = otu.tree_cast(mu, mu_dtype)
        return adam_updates, ScaleByMarsState(count=count_inc, mu=mu, nu=nu, mog=updates)

    return optax.GradientTransformation(init_fn, update_fn)
