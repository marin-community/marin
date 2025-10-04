# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

# Standard Library Imports
from dataclasses import dataclass
from typing import NamedTuple, Optional

# Third-Party Imports
import chex
import jax
import jax.numpy as jnp
import optax
from optax import tree_utils as otu

from levanter.optim.config import OptimizerConfig


@OptimizerConfig.register_subclass("rmsprop")
@dataclass(frozen=True)
class RMSPropMomentumConfig(OptimizerConfig):
    """Configuration for RMSProp with Momentum."""

    beta1: float = 0.9
    beta2: float = 0.99
    eps: float = 1e-30
    max_grad_norm: Optional[float] = 1.0
    weight_decay: float = 0.1

    def build(self, num_train_steps):
        """Creates and returns an Optax optimizer for RMSProp with Momentum."""

        # This indirection is used with `optax.inject_hyperparams` so that
        # the learning rate can be logged or updated dynamically.
        def _optimizer(learning_rate):
            transforms = []

            # 1) (Optional) Gradient clipping
            if self.max_grad_norm is not None:
                transforms.append(optax.clip_by_global_norm(self.max_grad_norm))

            # 2) RMSProp with momentum
            transforms.append(scale_by_rmsprop_momentum(beta2=self.beta2, beta1=self.beta1, eps=self.eps))

            # 3) (Optional) Weight beta2
            if self.weight_decay > 0.0:
                transforms.append(optax.add_decayed_weights(self.weight_decay, self.build_weight_decay_mask()))

            # 4) Finally, scale by negative learning rate for gradient descent
            transforms.append(optax.scale(-learning_rate))

            return optax.chain(*transforms)

        # Inject the (possibly schedule-driven) learning rate
        return optax.inject_hyperparams(_optimizer)(learning_rate=self.lr_scheduler(num_train_steps))


class ScaleByRMSPropMomState(NamedTuple):
    """State for RMSProp with momentum."""

    count: chex.Array  # shape=(), dtype=jnp.int32
    mean_square: optax.Updates
    momentum: optax.Updates


def scale_by_rmsprop_momentum(
    beta1: float = 0.9, beta2: float = 0.99, eps: float = 1e-30
) -> optax.GradientTransformation:
    r"""
    Scale updates according to the RMSProp algorithm, then apply momentum.

    RMSProp maintains an exponential moving average of the squared gradients.
    The effective step size is:
        update = g / sqrt( mean_square(g) + eps )

    With momentum, we store a separate buffer for momentum:
        momentum_buffer_{t+1} = momentum * momentum_buffer_t + update
        param_{t+1} = param_t - lr * momentum_buffer_{t+1}

    Args:
        beta2: Exponential beta2 rate for the squared gradient average.
        momentum: Momentum coefficient.
        eps: Term added to the denominator for numerical stability.

    Returns:
        An `optax.GradientTransformation` to be used in an Optax chain.
    """

    def init_fn(params):
        # Initialize mean_square and momentum buffers to zero trees
        mean_square = otu.tree_zeros_like(params)
        momentum_buf = otu.tree_zeros_like(params)
        return ScaleByRMSPropMomState(
            count=jnp.zeros([], jnp.int32),
            mean_square=mean_square,
            momentum=momentum_buf,
        )

    def update_fn(updates, state, params=None):

        count_inc = optax.safe_increment(state.count)
        # 1) Update the mean of squared gradients
        new_mean_square = jax.tree_map(
            lambda ms, g: beta2 * ms + (1.0 - beta2) * (g**2),
            state.mean_square,
            updates,
        )
        new_mean_square_hat = otu.tree_bias_correction(new_mean_square, beta2, count_inc)

        # 2) Compute the RMS-scaled updates
        scaled_updates = jax.tree_map(
            lambda g, ms: g / (jnp.sqrt(ms) + eps),
            updates,
            new_mean_square_hat,
        )

        # 3) Update momentum buffer:
        #    momentum_buffer = momentum * momentum_buffer + scaled_updates
        new_momentum = jax.tree_map(
            lambda mom, su: beta1 * mom + (1 - beta1) * su,
            state.momentum,
            scaled_updates,
        )

        # 4) Increment step count

        update = otu.tree_bias_correction(new_momentum, beta1, count_inc)

        # 5) Return final updates and new state
        #    (the final "raw" update is the momentum buffer itself)
        return update, ScaleByRMSPropMomState(
            count=count_inc,
            mean_square=new_mean_square,
            momentum=new_momentum,
        )

    return optax.GradientTransformation(init_fn, update_fn)
