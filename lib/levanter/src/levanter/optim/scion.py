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

import haliax
from haliax.nn import Linear

from levanter.optim.config import OptimizerConfig
from levanter.optim.util import map_flattened_linear_layers
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

    def build(self, num_train_steps):
        """
        Creates the optimizer.
        """
        learning_rate_schedule = self.lr_scheduler(num_train_steps)

        def optimizer(learning_rate):
            signum_lr = learning_rate * self.scion_to_signum_lr

            def scion_transform():
                components = []
                components.append(scale_with_scion(self.momentum, self.backend_steps, self.scion_epsilon))
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
            elif isinstance(param, Linear):
                # scion for linear layers
                return dataclasses.replace(param, weight="scion", bias="signum" if param.bias is not None else None)
            else:
                return "signum"

        return jax.tree_util.tree_map(mask_fn, params, paths, is_leaf=lambda x: isinstance(x, Linear))


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

        updates = jax.tree_map(lambda u: None if u is None else jnp.sign(u), buf, is_leaf=lambda x: x is None)

        return updates, ScaleByScionState(momentum_buffer=buf)

    return optax.GradientTransformation(init_fn, update_fn)


def scale_with_scion(momentum=0.95, steps=5, scion_eps=1e-8):
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

        def transform_linear_layer(layer: haliax.nn.Linear):
            assert layer.weight.ndim == 2

            updated_weight_array = zeropower_via_newtonschulz5(layer.weight.array, steps=steps, eps=scion_eps)

            scale = jnp.sqrt(jnp.maximum(1, updated_weight_array.shape[0] / updated_weight_array.shape[1]))
            updated_weight_array *= scale

            updated_weight = dataclasses.replace(layer.weight, array=updated_weight_array)

            return dataclasses.replace(layer, weight=updated_weight)  # type: ignore

        updates = map_flattened_linear_layers(transform_linear_layer, updates)

        return updates, ScaleByScionState(momentum_buffer=buf)

    return optax.GradientTransformation(init_fn, update_fn)


def zeropower_via_newtonschulz5(X, steps=10, eps=1e-7):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.
    """
    chex.assert_rank(X, 2)
    a, b, c = (3.4445, -4.7750, 2.0315)
    X /= jnp.linalg.norm(X) + eps  # Ensure top singular value <= 1
    transpose = False
    if X.shape[0] > X.shape[1]:
        X = X.T
        transpose = True
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transpose:
        X = X.T
    # https://x.com/leloykun/status/1874358290093924849

    return X
