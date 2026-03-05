# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from functools import partial
from typing import NamedTuple, Optional

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


@OptimizerConfig.register_subclass("muon")
@dataclass(frozen=True)
class MuonConfig(OptimizerConfig):
    """
    Muon optimizer configuration: Momentum Orthogonalized by Newton-Schulz.
    cf:
    Original Implementation: https://github.com/KellerJordan/modded-nanogpt
    """

    lr: float = 0.02
    adam_lr: float = 6e-4  # Adam LR
    momentum: float = 0.95
    nesterov: bool = True
    backend_steps: int = 5  # Number of steps for Newton-Schulz orthogonalization
    weight_decay: float = 0.0
    adam_weight_decay: Optional[float] = None
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    muon_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    # Kimi scales the learning rate for every d_1 * d_2 module by 0.2 * jnp.sqrt{\max{d_1, d_2}}, instead of the jnp.sqrt{\max{1, d_1/d_2}} as in the original nanogpt speedrun.
    # When this scaling is enabled, it is recommended to use learning rate and weight decay similar to adam
    use_kimi_scaling: bool = False
    coefficient_type: CoefficientType = "quintic"  # Type of Newton-Schulz coefficients to use

    def build(self, num_train_steps):
        """
        Creates the optimizer.
        """
        learning_rate_schedule = self.lr_scheduler(num_train_steps)
        adam_lr_schedule = self.lr_scheduler(num_train_steps, override_lr=self.adam_lr)

        def optimizer(learning_rate, adam_lr):
            def muon_transform():
                components = []
                components.append(
                    scale_with_muon(
                        self.momentum,
                        self.nesterov,
                        self.backend_steps,
                        self.muon_epsilon,
                        self.use_kimi_scaling,
                        self.coefficient_type,
                    )
                )
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
                adam_weight_decay = self.adam_weight_decay if self.adam_weight_decay is not None else self.weight_decay
                if adam_weight_decay > 0:
                    components.append(optax.add_decayed_weights(adam_weight_decay, self.build_weight_decay_mask()))
                components.append(optax.scale(-adam_lr))
                optimizer = optax.chain(*components)
                return optimizer

            transformations = {
                "muon": muon_transform(),
                "adamw": adamw_transform(),
            }

            return optax.multi_transform(
                transformations, partial(self.create_mask, use_kimi_scaling=self.use_kimi_scaling)
            )

        return optax.inject_hyperparams(optimizer)(learning_rate=learning_rate_schedule, adam_lr=adam_lr_schedule)

    def create_mask(self, params, use_kimi_scaling=True):
        """
        Creates a mask that labels parameters as 'muon' or 'adamw' based on their
        dimensionality and module path, using AdamW for Embedding and lm_head parameters.
        """
        paths = leaf_key_paths(params)

        def mask_fn(param, path):
            path_str = ".".join(path) if isinstance(path, (list, tuple)) else str(path)
            if "Embedding" in path_str or "lm_head" in path_str:
                return "adamw"
            elif is_linear_like_module(param):
                # muon for linear layers
                if isinstance(param, haliax.nn.Linear):
                    assert (
                        param._out_first or use_kimi_scaling
                    )  # if we don't use kimi's version of scaling, then we need to assume out_first to ensure we are scaling like Out/In
                return label_linear_like_module(param, weight_label="muon", bias_label="adamw")
            else:
                return "adamw"

        return haliax.tree_util.tree_map(mask_fn, params, paths, is_leaf=is_linear_like_module)


class ScaleByMuonState(NamedTuple):
    """State for the Muon algorithm."""

    momentum_buffer: optax.Updates


def scale_with_muon(
    momentum=0.95, nesterov=True, steps=5, muon_eps=1e-8, use_kimi_scaling=False, coefficient_type="quintic"
):
    # Convert steps to concrete int at function definition time
    steps = int(steps)

    def init_fn(params):
        momentum_buffer = otu.tree_zeros_like(params)
        return ScaleByMuonState(momentum_buffer=momentum_buffer)

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

        def transform_linear_layer(layer):
            array = linear_like_weight_array(layer)
            if array.ndim < 2:
                raise ValueError(f"Expected linear-like weight with rank >= 2, got {array.ndim}")

            out_dim, in_dim = array.shape[-2], array.shape[-1]
            if array.ndim == 2:
                updated_weight_array = zeropower_via_newtonschulz5(
                    array, steps=steps, eps=muon_eps, coefficient_type=coefficient_type
                )
            else:
                flat = array.reshape((-1, out_dim, in_dim))
                flat_updated = jax.vmap(
                    lambda mat: zeropower_via_newtonschulz5(
                        mat, steps=steps, eps=muon_eps, coefficient_type=coefficient_type
                    )
                )(flat)
                updated_weight_array = flat_updated.reshape(array.shape)

            if not use_kimi_scaling:
                scale = jnp.sqrt(jnp.maximum(1, out_dim / in_dim))  # sqrt(Out/In)
            else:
                scale = 0.2 * jnp.sqrt(jnp.maximum(out_dim, in_dim))
            updated_weight_array *= scale

            return replace_linear_like_weight_array(layer, updated_weight_array)

        updates = map_flattened_linear_layers(transform_linear_layer, updates)

        return updates, ScaleByMuonState(momentum_buffer=buf)

    return optax.GradientTransformation(init_fn, update_fn)
