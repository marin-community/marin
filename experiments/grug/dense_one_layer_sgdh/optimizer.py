# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Raw-gradient Hyperball optimizer for the dense one-layer comparison."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import optax
from levanter.optim.config import OptimizerConfig
from levanter.utils.jax_utils import leaf_key_paths

from experiments.grug.dense_one_layer_sgdh.adamh import scale_by_adamh


def _target_named_sharding(array) -> jax.sharding.NamedSharding | None:
    if array is None or not hasattr(array, "shape"):
        return None
    sharding = getattr(array, "sharding", None)
    if sharding is None:
        sharding = getattr(jax.typeof(array), "sharding", None)
    if isinstance(sharding, jax.sharding.NamedSharding):
        return sharding
    return None


def _match_named_sharding_to_params(updates, params):
    def match_sharding(update, param):
        if update is None:
            return None
        target_sharding = _target_named_sharding(param)
        if target_sharding is None:
            return update
        return jax.sharding.reshard(update, target_sharding)

    return jax.tree.map(match_sharding, updates, params, is_leaf=lambda x: x is None)


def _match_named_update_sharding() -> optax.GradientTransformation:
    def init_fn(params):
        del params
        return optax.EmptyState()

    def update_fn(updates, state, params=None):
        if params is None:
            return updates, state
        return _match_named_sharding_to_params(updates, params), state

    return optax.GradientTransformation(init_fn, update_fn)


def _scale_invariant_hyperball_updates(params, direction_updates, learning_rate: float):
    direction_updates = _match_named_sharding_to_params(direction_updates, params)

    def scale_invariant_update(param, update):
        if update is None:
            return None
        if not hasattr(param, "ndim"):
            return update
        if param.ndim == 2:
            param_norm = jnp.linalg.norm(param)
            update_norm = jnp.linalg.norm(update)
            new_param = param - learning_rate * update * param_norm / jnp.maximum(update_norm, 1e-10)
            new_param_norm = jnp.linalg.norm(new_param)
            return new_param / jnp.maximum(new_param_norm, 1e-10) * param_norm - param

        axes = tuple(range(1, param.ndim))
        param_norm = jnp.sqrt(jnp.sum(jnp.square(param), axis=axes, keepdims=True))
        update_norm = jnp.sqrt(jnp.sum(jnp.square(update), axis=axes, keepdims=True))
        new_param = param - learning_rate * update * param_norm / jnp.maximum(update_norm, 1e-10)
        new_param_norm = jnp.sqrt(jnp.sum(jnp.square(new_param), axis=axes, keepdims=True))
        return new_param / jnp.maximum(new_param_norm, 1e-10) * param_norm - param

    return jax.tree.map(scale_invariant_update, params, direction_updates, is_leaf=lambda x: x is None)


def scale_with_grug_sgdh(learning_rate: float = 0.02) -> optax.GradientTransformation:
    """Apply the raw gradient followed by a norm-preserving Hyperball projection."""

    def init_fn(params):
        del params
        return optax.EmptyState()

    def update_fn(updates, state, params=None):
        if params is None:
            raise ValueError("scale_with_grug_sgdh requires params for norm-preserving updates")
        return _scale_invariant_hyperball_updates(params, updates, learning_rate), state

    return optax.GradientTransformation(init_fn, update_fn)


def _dense_hyperball_mask(params):
    paths = leaf_key_paths(params)

    def mask_fn(param, path):
        path_str = ".".join(path) if isinstance(path, (list, tuple)) else str(path)
        path_lower = path_str.lower()
        if "token_embed" in path_lower or path_lower.endswith(".weight"):
            return "adam"
        if "output_proj" in path_lower or "lm_head" in path_lower:
            return "adamh"
        if hasattr(param, "ndim") and param.ndim >= 2:
            return "sgdh"
        return "adam"

    return jax.tree.map(mask_fn, params, paths)


@OptimizerConfig.register_subclass("grug_dense_one_layer_sgdh_v1")
@dataclass(frozen=True)
class GrugDenseSGDHConfig(OptimizerConfig):
    """SGD-H for dense matrices with AdamH/Adam fallback parameter groups."""

    adam_lr: float = 6e-4
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    max_grad_norm: float | None = None

    def build(self, num_train_steps):
        learning_rate_schedule = self.lr_scheduler(num_train_steps)
        adam_lr_schedule = self.lr_scheduler(num_train_steps, override_lr=self.adam_lr)

        def optimizer(learning_rate, adam_lr):
            sgdh_components = []
            if self.max_grad_norm:
                sgdh_components.append(optax.clip_by_global_norm(self.max_grad_norm))
            sgdh_components.extend((scale_with_grug_sgdh(learning_rate), _match_named_update_sharding()))

            adamh_components = []
            if self.max_grad_norm:
                adamh_components.append(optax.clip_by_global_norm(self.max_grad_norm))
            adamh_components.append(scale_by_adamh(self.beta1, self.beta2, self.epsilon, learning_rate))

            adam_components = []
            if self.max_grad_norm:
                adam_components.append(optax.clip_by_global_norm(self.max_grad_norm))
            adam_components.extend((optax.scale_by_adam(self.beta1, self.beta2, self.epsilon), optax.scale(-adam_lr)))

            return optax.multi_transform(
                {
                    "sgdh": optax.chain(*sgdh_components),
                    "adamh": optax.chain(*adamh_components),
                    "adam": optax.chain(*adam_components),
                },
                self.create_mask,
            )

        return optax.inject_hyperparams(optimizer)(
            learning_rate=learning_rate_schedule,
            adam_lr=adam_lr_schedule,
        )

    def create_mask(self, params):
        return _dense_hyperball_mask(params)
