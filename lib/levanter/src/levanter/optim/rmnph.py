# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import math
from dataclasses import dataclass
from typing import NamedTuple

import haliax
import jax
import jax.numpy as jnp
import optax
from optax import tree_utils as otu

from levanter.optim.adamh import scale_by_adamh
from levanter.optim.config import OptimizerConfig
from levanter.optim.util import matrix_optimizer_mask, map_flattened_matrix_layers, norm_preserving_update


@OptimizerConfig.register_subclass("rmnpH")
@dataclass(frozen=True)
class RMNPHConfig(OptimizerConfig):
    """Row-normalized momentum with norm-preserving hidden matrices and AdamH head."""

    embedding_lr: float = 0.012
    vector_lr: float = 0.060
    matrix_lr: float = 0.015
    output_lr: float = 0.015
    momentum: float = 0.95
    nesterov: bool = True
    beta1: float = 0.8
    beta2: float = 0.98
    epsilon: float = 1e-15
    rmnp_epsilon: float = 1e-10
    max_grad_norm: float = 1.0

    def build(self, num_train_steps):
        rates = dict(
            embedding_lr=self.embedding_lr,
            vector_lr=self.vector_lr,
            matrix_lr=self.matrix_lr,
            output_lr=self.output_lr,
        )
        if any(not math.isfinite(rate) or rate < 0 for rate in rates.values()):
            raise ValueError("Learning rates must be finite and non-negative")

        def optimizer(embedding_lr, vector_lr, matrix_lr, output_lr):
            def clipped(transform):
                if self.max_grad_norm:
                    return optax.chain(optax.clip_by_global_norm(self.max_grad_norm), transform)
                return transform

            embedding = [scale_with_rmnp(self.momentum, self.nesterov, self.rmnp_epsilon)]
            if self.weight_decay > 0:
                embedding.append(optax.add_decayed_weights(self.weight_decay, self.build_weight_decay_mask()))
            embedding.append(optax.scale(-embedding_lr))
            return optax.multi_transform(
                {
                    "rmnph": clipped(scale_with_rmnph(self.momentum, self.nesterov, self.rmnp_epsilon, matrix_lr)),
                    "adamh": clipped(scale_by_adamh(self.beta1, self.beta2, self.epsilon, output_lr)),
                    "rmnp": optax.chain(*embedding),
                    "adam": clipped(
                        optax.chain(optax.scale_by_adam(self.beta1, self.beta2, self.epsilon), optax.scale(-vector_lr))
                    ),
                },
                self.create_mask,
            )

        schedules = {name: self.lr_scheduler(num_train_steps, override_lr=rate) for name, rate in rates.items()}
        return optax.inject_hyperparams(optimizer)(**schedules)

    def create_mask(self, params):
        return matrix_optimizer_mask(
            params, matrix_label="rmnph", head_label="adamh", embedding_label="rmnp", fallback_label="adam"
        )


def scale_with_rmnph(momentum=0.95, nesterov=True, rmnp_eps=1e-10, learning_rate=0.015):
    """Project row-normalized momentum updates back onto each parameter sphere."""
    rmnp = scale_with_rmnp(momentum, nesterov, rmnp_eps)

    def update_fn(updates, state, params):
        directions, next_state = rmnp.update(updates, state, params)
        deltas = jax.tree.map(
            lambda p, u: norm_preserving_update(p, u, learning_rate), params, directions, is_leaf=lambda x: x is None
        )
        return deltas, next_state

    return optax.GradientTransformation(rmnp.init, update_fn)


class ScaleByRMNPState(NamedTuple):
    """State for the RMNP algorithm."""

    momentum_buffer: optax.Updates


def scale_with_rmnp(momentum=0.95, nesterov=True, rmnp_eps=1e-8, use_kimi_scaling=False):
    def init_fn(params):
        momentum_buffer = otu.tree_zeros_like(params)
        return ScaleByRMNPState(momentum_buffer=momentum_buffer)

    def update_fn(updates, state, params=None):
        buf = state.momentum_buffer
        buf = jax.tree.map(
            lambda m, g: None if g is None else momentum * m + g,
            buf,
            updates,
            is_leaf=lambda x: x is None,
        )
        if nesterov:
            updates = jax.tree.map(
                lambda m, g: None if g is None else momentum * m + g,
                buf,
                updates,
                is_leaf=lambda x: x is None,
            )
        else:
            updates = buf

        def transform_linear_layer(layer: haliax.nn.Linear):
            assert layer.weight.ndim == 2
            array = layer.weight.array
            # Normalize each output's update over its logical input dimension.
            input_axis = -1 if layer._out_first else -2
            row_norm = jnp.linalg.norm(array, axis=input_axis, keepdims=True)
            updated_weight_array = array / jnp.maximum(row_norm, rmnp_eps)

            if not use_kimi_scaling:
                # Preserve sqrt(Out/In) in either physical storage layout.
                out_dim, in_dim = array.shape if layer._out_first else array.shape[::-1]
                scale = jnp.sqrt(jnp.maximum(1, out_dim / in_dim))
            else:
                scale = 0.2 * jnp.sqrt(jnp.maximum(updated_weight_array.shape[0], updated_weight_array.shape[1]))
            updated_weight_array *= scale

            updated_weight = dataclasses.replace(layer.weight, array=updated_weight_array)

            return dataclasses.replace(layer, weight=updated_weight)  # type: ignore

        updates = map_flattened_matrix_layers(transform_linear_layer, updates)

        return updates, ScaleByRMNPState(momentum_buffer=buf)

    return optax.GradientTransformation(init_fn, update_fn)
