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


@OptimizerConfig.register_subclass("cautious")
@dataclass(frozen=True)
class CautiousConfig(OptimizerConfig):
    """
    Cautious optimizer configuration
    cf:
    Original Paper: https://arxiv.org/abs/2411.16085
    """

    beta1: float = 0.95
    beta2: float = 0.95
    gamma: float = 0.025
    epsilon: float = 1e-8
    max_grad_norm: Optional[float] = 1.0
    adamc_weight_decay: bool = False
    """
    If set, use the AdamC corrected weight decay, which keeps
    ``weight_decay / lr`` constant across training.

    This follows Defazio, *On the Correct Treatment of Weight Decay in Adam*
    (2025, https://arxiv.org/abs/2506.02285v2).
    """

    def build(self, num_train_steps):
        """Creates the optimizer"""

        # indirection makes it work with optax.inject_hyperparams so we can log the learning rate
        def _optimizer(learning_rate):
            components = []

            if self.max_grad_norm:
                components.append(optax.clip_by_global_norm(self.max_grad_norm))

            components.append(scale_by_cautious(self.beta1, self.beta2, self.epsilon))

            if self.weight_decay > 0:
                if self.adamc_weight_decay:
                    max_lr = self.learning_rate
                    weight_decay = self.weight_decay * (learning_rate / max_lr)
                else:
                    weight_decay = self.weight_decay
                components.append(optax.add_decayed_weights(weight_decay, self.build_weight_decay_mask()))

            # - learning rate for descent
            components.append(optax.scale(-learning_rate))

            optimizer = optax.chain(*components)

            return optimizer

        return optax.inject_hyperparams(_optimizer)(learning_rate=self.lr_scheduler(num_train_steps))


class ScaleByCautiousState(NamedTuple):
    """State for the Cautious algorithm."""

    count: chex.Array  # shape=(), dtype=jnp.int32.
    mu: optax.Updates
    nu: optax.Updates


def scale_by_cautious(
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    mu_dtype: Optional[Any] = None,
) -> optax.GradientTransformation:
    r"""Rescale updates according to the Cautious algorithm.

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
        return ScaleByCautiousState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu)

    def update_fn(updates, state, params=None):
        mu = otu.tree_update_moment(updates, state.mu, b1, 1)
        nu = otu.tree_update_moment_per_elem_norm(updates, state.nu, b2, 2)
        count_inc = optax.safe_increment(state.count)
        mu_hat = otu.tree_bias_correction(mu, b1, count_inc)
        nu_hat = otu.tree_bias_correction(nu, b2, count_inc)
        adam_updates = jax.tree.map(
            lambda m, v: None if m is None else m / (jnp.sqrt(v + eps_root) + eps),
            mu_hat,
            nu_hat,
            is_leaf=lambda x: x is None,
        )
        cautious_m = jax.tree.map(
            lambda u, g: None if u is None else (g * u) > 0, adam_updates, updates, is_leaf=lambda x: x is None
        )
        cautious_update = jax.tree.map(
            lambda u, m: None if u is None else u * m / (m.mean() + eps),
            adam_updates,
            cautious_m,
            is_leaf=lambda x: x is None,
        )
        mu = otu.tree_cast(mu, mu_dtype)
        return cautious_update, ScaleByCautiousState(count=count_inc, mu=mu, nu=nu)

    return optax.GradientTransformation(init_fn, update_fn)
