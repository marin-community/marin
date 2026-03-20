# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import NamedTuple, Optional, Any

import chex
import jax
import jax.numpy as jnp
import optax
from optax import tree_utils as otu

import haliax

from levanter.optim.config import OptimizerConfig
from levanter.optim.util import is_linear_like_module, label_linear_like_module
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
            elif is_linear_like_module(param):
                return label_linear_like_module(param, weight_label="adamh", bias_label="adam")
            else:
                return "adam"

        return haliax.tree_util.tree_map(mask_fn, params, paths, is_leaf=is_linear_like_module)


class ScaleByAdamHState(NamedTuple):
    """State for the AdamH algorithm."""

    count: chex.Array  # shape=(), dtype=jnp.int32.
    mu: optax.Updates
    nu: optax.Updates
    target_norm: optax.Updates


def _adamh_norm_axes(param: jax.Array) -> tuple[int, ...] | None:
    if param.ndim <= 2:
        return None
    return tuple(range(1, param.ndim))


def _adamh_target_norm(param: jax.Array | None) -> jax.Array | None:
    if param is None:
        return None
    axes = _adamh_norm_axes(param)
    if axes is None:
        return jnp.linalg.norm(param)
    return jnp.sqrt(jnp.sum(jnp.square(param), axis=axes, keepdims=True))


def _adamh_projected_update(
    param: jax.Array | None,
    first_moment: jax.Array | None,
    second_moment: jax.Array | None,
    target_norm: jax.Array | None,
    *,
    beta1_correction: chex.Array,
    beta2_correction: chex.Array,
    eps: float,
    learning_rate: float,
) -> jax.Array | None:
    if param is None:
        return None
    if first_moment is None or second_moment is None or target_norm is None:
        return None

    corrected_first_moment = first_moment / beta1_correction
    corrected_second_moment = second_moment / beta2_correction
    adam_update = corrected_first_moment / (jnp.sqrt(corrected_second_moment) + eps)

    axes = _adamh_norm_axes(param)
    if axes is None:
        param_sq_norm = jnp.sum(jnp.square(param))
        update_dot = jnp.sum(param * adam_update)
        update_norm = jnp.sqrt(jnp.sum(jnp.square(adam_update)))
    else:
        param_sq_norm = jnp.sum(jnp.square(param), axis=axes, keepdims=True)
        update_dot = jnp.sum(param * adam_update, axis=axes, keepdims=True)
        update_norm = jnp.sqrt(jnp.sum(jnp.square(adam_update), axis=axes, keepdims=True))

    safe_update_norm = jnp.maximum(update_norm, 1e-10)
    projected_sq_norm = (
        param_sq_norm
        - 2.0 * learning_rate * target_norm * update_dot / safe_update_norm
        + (learning_rate**2) * jnp.square(target_norm)
    )
    projected_norm = jnp.sqrt(jnp.maximum(projected_sq_norm, 1e-10))
    scale_ratio = target_norm / projected_norm
    update_scale = learning_rate * target_norm * scale_ratio / safe_update_norm
    return param * (scale_ratio - 1.0) - adam_update * update_scale


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
        target_norm = jax.tree.map(_adamh_target_norm, params, is_leaf=lambda x: x is None)
        return ScaleByAdamHState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu, target_norm=target_norm)

    def update_fn(updates, state, params):
        mu = otu.tree_update_moment(updates, state.mu, b1, 1)
        nu = otu.tree_update_moment_per_elem_norm(updates, state.nu, b2, 2)
        count_inc = optax.safe_increment(state.count)
        beta1_correction = 1.0 - jnp.asarray(b1, dtype=jnp.float32) ** count_inc
        beta2_correction = 1.0 - jnp.asarray(b2, dtype=jnp.float32) ** count_inc
        mu = otu.tree_cast(mu, mu_dtype)

        adamh_updates = jax.tree.map(
            lambda p, m, v, target: _adamh_projected_update(
                p,
                m,
                v,
                target,
                beta1_correction=beta1_correction,
                beta2_correction=beta2_correction,
                eps=eps,
                learning_rate=learning_rate,
            ),
            params,
            mu,
            nu,
            state.target_norm,
            is_leaf=lambda x: x is None,
        )

        return adamh_updates, ScaleByAdamHState(count=count_inc, mu=mu, nu=nu, target_norm=state.target_norm)

    return optax.GradientTransformation(init_fn, update_fn)
