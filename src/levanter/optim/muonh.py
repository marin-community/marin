# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
from dataclasses import dataclass
from typing import NamedTuple

import chex
import jax
import jax.numpy as jnp
import optax
from jax.sharding import PartitionSpec
from optax import tree_utils as otu

import haliax
from haliax.nn import Linear

from levanter.optim.config import OptimizerConfig
from levanter.optim.util import map_flattened_linear_layers
from levanter.utils.jax_utils import leaf_key_paths
from levanter.optim.adamh import scale_by_adamh
from levanter.optim.muon import zeropower_via_newtonschulz5

@OptimizerConfig.register_subclass("muonH")
@dataclass(frozen=True)
class MuonHConfig(OptimizerConfig):
    """
    This is a variant of the Muon optimizer configuration: Momentum Orthogonalized by Newton-Schulz (https://github.com/KellerJordan/modded-nanogpt).

    We ensure that the linear weights stay exactly constant norm as initialization by applying the following update rule:

    p_new_intermediate = p - learning_rate * u * norm(p) / norm(u)
    p_new = p_new_intermediate / norm(p_new_intermediate) * norm(p)

    where p is the parameter, u is the update and norm is the Frobenius norm of a matrix.

    The default learning rate for the MuonH configuration should be sqrt(learning_rate * weight_decay) for Muon configuration with weight decay.

    """

    adam_lr: float = 6e-4  # Adam LR
    momentum: float = 0.95
    nesterov: bool = True
    backend_steps: int = 5  # Number of steps for Newton-Schulz orthogonalization
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    muon_epsilon: float = 1e-8
    max_grad_norm: float = 1.0

    def build(self, num_train_steps):
        """
        Creates the optimizer.
        """
        learning_rate_schedule = self.lr_scheduler(num_train_steps)
        adam_lr_schedule = self.lr_scheduler(num_train_steps, override_lr=self.adam_lr)

        def optimizer(learning_rate, adam_lr):
            def muonh_transform():
                components = []
                if self.max_grad_norm:
                    components.append(optax.clip_by_global_norm(self.max_grad_norm))
                components.append(
                    scale_with_muonh(
                        self.momentum, self.nesterov, self.backend_steps, self.muon_epsilon, learning_rate
                    )
                )
                optimizer = optax.chain(*components)
                return optimizer

            def adamh_transform():
                components = []
                if self.max_grad_norm:
                    components.append(optax.clip_by_global_norm(self.max_grad_norm))
                components.append(scale_by_adamh(self.beta1, self.beta2, self.epsilon, learning_rate))
                optimizer = optax.chain(*components)
                return optimizer

            def adam_transform():
                components = []
                if self.max_grad_norm:
                    components.append(optax.clip_by_global_norm(self.max_grad_norm))
                components.append(optax.scale_by_adam(self.beta1, self.beta2, self.epsilon))
                components.append(optax.scale(-adam_lr))
                optimizer = optax.chain(*components)
                return optimizer

            transformations = {
                "muonh": muonh_transform(),
                "adamh": adamh_transform(),
                "adam": adam_transform(),
            }

            return optax.multi_transform(transformations, self.create_mask)

        return optax.inject_hyperparams(optimizer)(learning_rate=learning_rate_schedule, adam_lr=adam_lr_schedule)

    def create_mask(self, params):
        """
        Creates a mask that labels parameters as 'muon' 'adamh' 'adam' based on their
        dimensionality and module path, using Adam for Embedding and vector parameters.
        using AdamH for lm_head parameters.
        using MuonH for other parameters.
        """
        paths = leaf_key_paths(params)

        def mask_fn(param, path):
            path_str = ".".join(path) if isinstance(path, (list, tuple)) else str(path)
            if "Embedding" in path_str:
                return "adam"
            elif "lm_head" in path_str:
                return "adamh"
            elif isinstance(param, Linear):
                # muonh for linear layers
                return dataclasses.replace(param, weight="muonh", bias="adam" if param.bias is not None else None)
            else:
                return "adam"

        return jax.tree_util.tree_map(mask_fn, params, paths, is_leaf=lambda x: isinstance(x, Linear))


class ScaleByMuonHState(NamedTuple):
    """State for the Muon algorithm."""

    momentum_buffer: optax.Updates


def scale_with_muonh(momentum=0.95, nesterov=True, steps=5, muon_eps=1e-8, learning_rate=0.02):
    # Convert steps to concrete int at function definition time
    steps = int(steps)

    def init_fn(params):
        momentum_buffer = otu.tree_zeros_like(params)
        return ScaleByMuonHState(momentum_buffer=momentum_buffer)

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
            # steps is now a concrete int
            array = layer.weight.array
            updated_weight_array = zeropower_via_newtonschulz5(array, steps=steps, eps=muon_eps)

            updated_weight = dataclasses.replace(layer.weight, array=updated_weight_array)

            return dataclasses.replace(layer, weight=updated_weight)  # type: ignore

        muon_updates = map_flattened_linear_layers(transform_linear_layer, updates)

        # projected training for linear weight
        def scale_invariant_update(p, u):
            if p is None:
                return None
            if p.ndim == 2:
                # this is the case for no layer stacking
                new_p = p - learning_rate * u * jnp.linalg.norm(p) / max(jnp.linalg.norm(u), 1e-10)
                return new_p / jnp.linalg.norm(new_p) * jnp.linalg.norm(p) - p
            else:
                axes = tuple(range(1, p.ndim))
                p_norm = jnp.sqrt(jnp.sum(jnp.square(p), axis=axes, keepdims=True))
                u_norm = jnp.sqrt(jnp.sum(jnp.square(u), axis=axes, keepdims=True))
                new_p = p - learning_rate * u * p_norm / max(u_norm, 1e-10)
                new_p_norm = jnp.sqrt(jnp.sum(jnp.square(new_p), axis=axes, keepdims=True))
                return new_p / max(new_p_norm, 1e-10) * p_norm - p

        muonh_updates = jax.tree_util.tree_map(
            scale_invariant_update,
            params,
            muon_updates,
            is_leaf=lambda x: x is None,
        )

        return muonh_updates, ScaleByMuonHState(momentum_buffer=buf)

    return optax.GradientTransformation(init_fn, update_fn)
