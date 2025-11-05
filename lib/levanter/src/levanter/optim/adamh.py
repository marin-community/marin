# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
from dataclasses import dataclass
from typing import NamedTuple, Optional, Any

import chex
import jax
import jax.numpy as jnp
import optax
from optax import tree_utils as otu

from haliax.nn import Linear

from levanter.optim.config import OptimizerConfig
from levanter.utils.jax_utils import leaf_key_paths


@OptimizerConfig.register_subclass("adamH")
@dataclass(frozen=True)
class AdamHConfig(OptimizerConfig):
    """
    This is a variant of the Adam optimizer configuration.

    We ensure that the linear weights stay exactly constant norm as initialization by applying the following update rule:

    p_new_intermediate = p - learning_rate * u * norm(p) / norm(u)
    p_new = p_new_intermediate / norm(p_new_intermediate) * norm(p)

    where p is the parameter, u is the update and norm is the Frobenius norm of a matrix.

    The default learning rate for the AdamH configuration should be sqrt(learning_rate * weight_decay) for Adam configuration with weight decay.
    """

    beta1: float = 0.9
    # cf https://docs.mosaicml.com/projects/composer/en/latest/api_reference/generated/composer.optim.DecoupledAdamW.html
    # https://x.com/giffmana/status/1692641748445438301
    beta2: float = 0.95
    epsilon: float = 1e-8
    max_grad_norm: Optional[float] = 1.0
    nesterov: bool = False
    adam_lr: float = 6e-4  # learning rate used for weight without weight decay

    def build(self, num_train_steps):
        """Creates the optimizer"""
        learning_rate_schedule = self.lr_scheduler(num_train_steps)
        adam_lr_schedule = self.lr_scheduler(num_train_steps, override_lr=self.adam_lr)

        # indirection makes it work with optax.inject_hyperparams so we can log the learning rate
        def optimizer(learning_rate, adam_lr):

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
                "adamh": adamh_transform(),
                "adam": adam_transform(),
            }

            return optax.multi_transform(transformations, self.create_mask)

        return optax.inject_hyperparams(optimizer)(learning_rate=learning_rate_schedule, adam_lr=adam_lr_schedule)

    def create_mask(self, params):
        """
        Creates a mask that labels parameters as 'adamh' or 'adam' based on their
        dimensionality and module path, using Adam for LayerNorm Gamma and Embedding, and AdamH for all Linear parameters.
        """
        paths = leaf_key_paths(params)

        def mask_fn(param, path):
            path_str = ".".join(path) if isinstance(path, (list, tuple)) else str(path)
            if "Embedding" in path_str:
                return "adam"
            elif isinstance(param, Linear):
                return dataclasses.replace(param, weight="adamh", bias="adam" if param.bias is not None else None)
            else:
                return "adam"

        return jax.tree_util.tree_map(mask_fn, params, paths, is_leaf=lambda x: isinstance(x, Linear))


class ScaleByAdamHState(NamedTuple):
    """State for the AdamH algorithm."""

    count: chex.Array  # shape=(), dtype=jnp.int32.
    mu: optax.Updates
    nu: optax.Updates


def scale_by_adamh(
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    learning_rate: float = 0.02,
    mu_dtype: Optional[Any] = None,
) -> optax.GradientTransformation:
    r"""Rescale updates according to the AdamH algorithm.

    Concretely,

    Args:
      b1: Decay rate for the exponentially weighted average of grads.
      b2: Decay rate for the exponentially weighted average of squared grads.
      eps: Term added to the denominator to improve numerical stability.
      learning_rate: Learning rate for the AdamH algorithm.
      mu_dtype: Optional dtype to be used for the first order accumulator; if
        None then the dtype is inferred from params and updates.


    Returns:
      A :class:optax.GradientTransformation object.
    """

    mu_dtype = jax.dtypes.canonicalize_dtype(mu_dtype)

    def init_fn(params):
        mu = otu.tree_zeros_like(params, dtype=mu_dtype)  # First moment
        nu = otu.tree_zeros_like(params)  # Second moment
        return ScaleByAdamHState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu)

    def update_fn(updates, state, params):
        mu = otu.tree_update_moment(updates, state.mu, b1, 1)
        nu = otu.tree_update_moment_per_elem_norm(updates, state.nu, b2, 2)
        count_inc = optax.safe_increment(state.count)
        mu_hat = otu.tree_bias_correction(mu, b1, count_inc)
        nu_hat = otu.tree_bias_correction(nu, b2, count_inc)

        adam_updates = jax.tree.map(
            lambda m, v: None if m is None else m / (jnp.sqrt(v) + eps),
            mu_hat,
            nu_hat,
            is_leaf=lambda x: x is None,
        )
        mu = otu.tree_cast(mu, mu_dtype)

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

        adamh_updates = jax.tree_util.tree_map(
            scale_invariant_update,
            params,
            adam_updates,
            is_leaf=lambda x: x is None,
        )

        return adamh_updates, ScaleByAdamHState(count=count_inc, mu=mu, nu=nu)

    return optax.GradientTransformation(init_fn, update_fn)
