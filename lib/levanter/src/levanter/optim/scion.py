# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax
from optax import tree_utils as otu

import haliax

from levanter.optim.config import OptimizerConfig
from levanter.optim.util import (
    CoefficientType,
    is_linear_like_module,
    label_linear_like_module,
    linear_like_weight_array,
    map_flattened_linear_layers,
    replace_linear_like_weight_array,
    zeropower_via_newtonschulz5,
)
from levanter.utils.jax_utils import leaf_key_paths


@OptimizerConfig.register_subclass("scion")
@dataclass(frozen=True)
class ScionConfig(OptimizerConfig):
    """
    Scion optimizer configuration
    cf:
    Original Paper: https://arxiv.org/abs/2502.07529
    """

    lr: float = 0.02
    scion_to_signum_lr: float = 0.25  # Scaling factor between AdamW and Scion learning rates
    momentum: float = 0.95
    backend_steps: int = 10  # Number of steps for Newton-Schulz orthogonalization
    weight_decay: float = 0.0
    beta1: float = 0.9
    scion_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    coefficient_type: CoefficientType = "quintic"  # Type of Newton-Schulz coefficients to use

    def build(self, num_train_steps):
        """
        Creates the optimizer.
        """
        learning_rate_schedule = self.lr_scheduler(num_train_steps)

        def optimizer(learning_rate):
            signum_lr = learning_rate * self.scion_to_signum_lr

            def scion_transform():
                components = []
                components.append(
                    scale_with_scion(self.momentum, self.backend_steps, self.scion_epsilon, self.coefficient_type)
                )
                if self.weight_decay > 0:
                    components.append(optax.add_decayed_weights(self.weight_decay, self.build_weight_decay_mask()))
                components.append(optax.scale(-learning_rate))
                optimizer = optax.chain(*components)
                return optimizer

            def signum_transform():
                components = []
                if self.max_grad_norm:
                    components.append(optax.clip_by_global_norm(self.max_grad_norm))
                components.append(scale_by_signum(self.beta1))
                if self.weight_decay > 0:
                    components.append(optax.add_decayed_weights(self.weight_decay, self.build_weight_decay_mask()))
                components.append(optax.scale(-signum_lr))
                optimizer = optax.chain(*components)
                return optimizer

            transformations = {
                "scion": scion_transform(),
                "signum": signum_transform(),
            }

            return optax.multi_transform(transformations, self.create_mask)

        return optax.inject_hyperparams(optimizer)(learning_rate=learning_rate_schedule)

    def create_mask(self, params):
        """
        Creates a mask that labels parameters as 'scion' or 'signum' based on their
        dimensionality and module path, using AdamW for Embedding and lm_head parameters.
        """
        paths = leaf_key_paths(params)

        def mask_fn(param, path):
            path_str = ".".join(path) if isinstance(path, (list, tuple)) else str(path)
            if "Embedding" in path_str or "lm_head" in path_str:
                return "signum"
            elif is_linear_like_module(param):
                # scion for linear layers
                return label_linear_like_module(param, weight_label="scion", bias_label="signum")
            else:
                return "signum"

        return haliax.tree_util.tree_map(mask_fn, params, paths, is_leaf=is_linear_like_module)


class ScaleByScionState(NamedTuple):
    """State for the Scion algorithm."""

    momentum_buffer: optax.Updates


def scale_by_signum(momentum=0.95):
    def init_fn(params):
        momentum_buffer = otu.tree_zeros_like(params)  # First moment
        return ScaleByScionState(momentum_buffer=momentum_buffer)

    def update_fn(updates, state, params=None):
        buf = state.momentum_buffer
        buf = jax.tree.map(
            lambda m, g: None if g is None else momentum * m + (1 - momentum) * g,
            buf,
            updates,
            is_leaf=lambda x: x is None,
        )

        updates = jax.tree.map(lambda u: None if u is None else jnp.sign(u), buf, is_leaf=lambda x: x is None)

        return updates, ScaleByScionState(momentum_buffer=buf)

    return optax.GradientTransformation(init_fn, update_fn)


def scale_with_scion(momentum=0.95, steps=5, scion_eps=1e-8, coefficient_type="quintic"):
    def init_fn(params):
        momentum_buffer = otu.tree_zeros_like(params)  # First moment
        return ScaleByScionState(momentum_buffer=momentum_buffer)

    def update_fn(updates, state, params=None):
        buf = state.momentum_buffer
        buf = jax.tree.map(
            lambda m, g: None if g is None else momentum * m + (1 - momentum) * g,
            buf,
            updates,
            is_leaf=lambda x: x is None,
        )
        updates = buf

        def transform_linear_layer(layer):
            weight = linear_like_weight_array(layer)
            if weight.ndim < 2:
                raise ValueError(f"Expected linear-like weight with rank >= 2, got {weight.ndim}")

            out_dim, in_dim = weight.shape[-2], weight.shape[-1]
            if weight.ndim == 2:
                updated_weight_array = zeropower_via_newtonschulz5(
                    weight, steps=steps, eps=scion_eps, coefficient_type=coefficient_type
                )
            else:
                flat = weight.reshape((-1, out_dim, in_dim))
                flat_updated = jax.vmap(
                    lambda mat: zeropower_via_newtonschulz5(
                        mat, steps=steps, eps=scion_eps, coefficient_type=coefficient_type
                    )
                )(flat)
                updated_weight_array = flat_updated.reshape(weight.shape)

            scale = jnp.sqrt(jnp.maximum(1, out_dim / in_dim))
            updated_weight_array *= scale

            return replace_linear_like_weight_array(layer, updated_weight_array)

        updates = map_flattened_linear_layers(transform_linear_layer, updates)

        return updates, ScaleByScionState(momentum_buffer=buf)

    return optax.GradientTransformation(init_fn, update_fn)
