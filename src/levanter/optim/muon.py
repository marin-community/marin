# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
from dataclasses import dataclass
from functools import partial
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
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    muon_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    # Kimi scales the learning rate for every d_1 * d_2 module by 0.2 * jnp.sqrt{\max{d_1, d_2}}, instead of the jnp.sqrt{\max{1, d_1/d_2}} as in the original nanogpt speedrun.
    # When this scaling is enabled, it is recommended to use learning rate and weight decay similar to adam
    use_kimi_scaling: bool = False

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
                        self.momentum, self.nesterov, self.backend_steps, self.muon_epsilon, self.use_kimi_scaling
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
                if self.weight_decay > 0:
                    components.append(optax.add_decayed_weights(self.weight_decay, self.build_weight_decay_mask()))
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
            elif isinstance(param, Linear):
                # muon for linear layers
                assert (
                    param._out_first or use_kimi_scaling
                )  # if we don't use kimi's version of scaling, then we need to assume out_first to ensure we are scaling like Out/In
                return dataclasses.replace(param, weight="muon", bias="adamw" if param.bias is not None else None)
            else:
                return "adamw"

        return jax.tree_util.tree_map(mask_fn, params, paths, is_leaf=lambda x: isinstance(x, Linear))


class ScaleByMuonState(NamedTuple):
    """State for the Muon algorithm."""

    momentum_buffer: optax.Updates


def scale_with_muon(momentum=0.95, nesterov=True, steps=5, muon_eps=1e-8, use_kimi_scaling=False):
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

        def transform_linear_layer(layer: haliax.nn.Linear):
            assert layer.weight.ndim == 2
            # steps is now a concrete int
            array = layer.weight.array
            updated_weight_array = zeropower_via_newtonschulz5(array, steps=steps, eps=muon_eps)

            if not use_kimi_scaling:
                scale = jnp.sqrt(
                    jnp.maximum(1, updated_weight_array.shape[0] / updated_weight_array.shape[1])
                )  # sqrt(Out/In)
            else:
                scale = 0.2 * jnp.sqrt(jnp.maximum(updated_weight_array.shape[0], updated_weight_array.shape[1]))
            updated_weight_array *= scale

            updated_weight = dataclasses.replace(layer.weight, array=updated_weight_array)

            return dataclasses.replace(layer, weight=updated_weight)  # type: ignore

        updates = map_flattened_linear_layers(transform_linear_layer, updates)

        return updates, ScaleByMuonState(momentum_buffer=buf)

    return optax.GradientTransformation(init_fn, update_fn)


def zeropower_via_newtonschulz5(X, steps=5, eps=1e-7):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.
    """
    chex.assert_rank(X, 2)
    a, b, c = (3.4445, -4.7750, 2.0315)
    X /= jnp.linalg.norm(X) + eps  # Ensure top singular value <= 1
    transpose = False
    # assert X.shape[0] <= X.shape[1], "X should have more columns than rows for this implementation."
    # TODO: we should be smarter and also transpose if they're ~the same and X is already sharded along first axis
    if X.shape[0] > X.shape[1]:
        X = X.T
        transpose = True

    # TODO: because most things are in fact scan layers [L, m, n] (we vmap L)
    # it would be smarter to shard the layers so that basically each device gets its own layer
    # This doesn't quite optimally use the compute because there are usually more devices than layers, so we should
    # really do something even fancier.
    # It would be even smarter to stack similar layers together, but that would require more even more work
    # Let's call this good enough until we think it's not good enough
    if len(haliax.partitioning._get_mesh().devices):
        X = jax.lax.with_sharding_constraint(X, PartitionSpec(None, ("data", "model")))

    for i in range(steps):
        A = X @ X.T
        # doesn't seem to be necessary, so leaivng it out. When I used inspect_sharding it was a problem, but I dunno
        # A = jax.lax.with_sharding_constraint(A, PartitionSpec(None, None))  # ensure it's desharded
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transpose:
        X = X.T
    return X
