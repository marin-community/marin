# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
from dataclasses import dataclass
from typing import NamedTuple

import chex
import jax
import jax.numpy as jnp
import optax
from optax import tree_utils as otu

from haliax.nn import Linear

from levanter.optim.config import OptimizerConfig
from levanter.utils.jax_utils import leaf_key_paths


@OptimizerConfig.register_subclass("mini")
@dataclass(frozen=True)
class MiniConfig(OptimizerConfig):
    """
    AdamW Mini optimizer configuration
    Original Paper: https://arxiv.org/abs/2406.16793
    Mini group the parameters into groups and only keep one scalar as second moment for each group.
    Note: The implementation here does not reduce the actual size of the second moment buffer but it has the same update rule as the original paper.
    """

    lr: float = 0.02
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    nesterov: bool = False

    def build(self, num_train_steps):
        """
        Creates the optimizer.
        """
        learning_rate_schedule = self.lr_scheduler(num_train_steps)

        def optimizer(learning_rate):
            def mini_transform(mean_axis):
                components = []
                components.append(scale_with_mini(self.beta1, self.beta2, self.epsilon, mean_axis=mean_axis))
                if self.weight_decay > 0:
                    components.append(optax.add_decayed_weights(self.weight_decay, self.build_weight_decay_mask()))
                components.append(optax.scale(-learning_rate))
                optimizer = optax.chain(*components)
                return optimizer

            def adamw_transform():
                components = []
                if self.max_grad_norm:
                    components.append(optax.clip_by_global_norm(self.max_grad_norm))
                components.append(optax.scale_by_adam(self.beta1, self.beta2, self.epsilon))
                if self.weight_decay > 0:
                    components.append(optax.add_decayed_weights(self.weight_decay, self.build_weight_decay_mask()))
                components.append(optax.scale(-learning_rate))
                optimizer = optax.chain(*components)
                return optimizer

            transformations = {
                "embedding": mini_transform(mean_axis=(1,)),
                "lm_head": mini_transform(mean_axis=(1,)),
                "query": mini_transform(mean_axis=(2, 3, 4)),
                "key": mini_transform(mean_axis=(2, 3)),
                "value": mini_transform(mean_axis=(3,)),
                "output": mini_transform(mean_axis=(2, 3)),
                "linear": mini_transform(mean_axis=(2,)),
                "adamw": adamw_transform(),
            }

            return optax.multi_transform(transformations, self.create_mask)

        return optax.inject_hyperparams(optimizer)(learning_rate=learning_rate_schedule)

    def create_mask(self, params):
        """
        Creates a mask that labels parameters by their functionality group.
        """
        paths = leaf_key_paths(params)

        def mask_fn(param, path):
            path_str = ".".join(path) if isinstance(path, (list, tuple)) else str(path)
            if "embedding" in path_str:
                return "embedding"
            elif "q_proj" in path_str:
                return "query"
            elif "k_proj" in path_str:
                return "key"
            elif "v_proj" in path_str:
                return "value"
            elif "o_proj" in path_str:
                return "output"
            elif "lm_head" in path_str:
                return "lm_head"
            elif isinstance(param, Linear):
                return dataclasses.replace(param, weight="linear", bias="adamw" if param.bias is not None else None)
            else:
                return "adamw"

        return jax.tree_util.tree_map(mask_fn, params, paths, is_leaf=lambda x: isinstance(x, Linear))


class ScaleByMiniState(NamedTuple):
    """State for the Mars algorithm."""

    count: chex.Array  # shape=(), dtype=jnp.int32.
    momentum_buffer: optax.Updates
    second_moment_buffer: optax.Updates


def scale_with_mini(beta1, beta2, epsilon, mean_axis=(1,)):
    """
    Implementation of the Mini algorithm: Momentum optimizer with second moment scaling.

    Args:
        beta1: momentum decay rate
        beta2: second moment decay rate
        epsilon: small constant for numerical stability
        mean_axis: axes over which to compute the mean for the second moment

    Returns:
        An optax GradientTransformation
    """

    def init_fn(params):
        momentum_buffer = otu.tree_zeros_like(params)  # First moment
        second_moment_buffer = otu.tree_zeros_like(params)  # Second moment
        count = jnp.zeros([], jnp.int32)

        return ScaleByMiniState(
            count=count, momentum_buffer=momentum_buffer, second_moment_buffer=second_moment_buffer
        )

    def update_fn(updates, state, params=None):
        # Update momentum buffer (step 6 in algorithm)
        momentum_buffer = otu.tree_update_moment(updates, state.momentum_buffer, beta1, 1)

        # Calculate second moment along specified axes (step 8 in algorithm)
        def calc_and_update_second_moment(update, prev_moment):
            # Calculate squared and take mean over specified axes
            squared = update * update
            for axis in sorted(mean_axis, reverse=True):
                if axis < len(update.shape):
                    squared = jnp.mean(squared, axis=axis, keepdims=True)
            # Update second moment buffer
            for axis in sorted(mean_axis, reverse=True):
                if axis < len(update.shape):
                    squared = squared.repeat(update.shape[axis], axis=axis)
            return prev_moment * beta2 + squared * (1 - beta2)

        second_moment_buffer = jax.tree_util.tree_map(
            calc_and_update_second_moment, updates, state.second_moment_buffer
        )

        # Bias correction (steps 7 and 9 in algorithm)
        count_inc = optax.safe_increment(state.count)
        momentum_hat = otu.tree_bias_correction(momentum_buffer, beta1, count_inc)
        second_moment_hat = otu.tree_bias_correction(second_moment_buffer, beta2, count_inc)

        # Calculate updates (step 10 in algorithm)
        def apply_update(m_hat, v_hat):
            return m_hat / (jnp.sqrt(v_hat) + epsilon)

        # Apply the update using the momentum and second moment
        updates = jax.tree_util.tree_map(lambda m, v: apply_update(m, v), momentum_hat, second_moment_hat)

        return updates, ScaleByMiniState(
            count=count_inc, momentum_buffer=momentum_buffer, second_moment_buffer=second_moment_buffer
        )

    return optax.GradientTransformation(init_fn, update_fn)
