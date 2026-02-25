# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""NAMO and NAMO-D optimizer configs for Levanter.

This module implements:
- NAMO: Norm-Based Adaptive Moment Estimation with Orthogonalized Momentum.
- NAMO-D: A diagonal (column-wise) extension of NAMO with clamped adaptivity.

References:
- Paper: https://arxiv.org/abs/2602.17080
- Official implementation: https://github.com/minxin-zhg/namo/blob/main/src/namo.py

By default, NAMO/NAMO-D use the reference Newton-Schulz triplet via
``coefficient_type="simple"``. Other coefficient schedules remain available
for experimentation.
"""

import dataclasses
from dataclasses import dataclass
from typing import Any, NamedTuple, Optional

import jax
import jax.numpy as jnp
import optax
from optax import tree_utils as otu
from jaxtyping import Array

import haliax
from haliax.nn import Linear

from levanter.optim.config import OptimizerConfig
from levanter.optim.util import (
    CoefficientType,
    flatten_linear_layers,
    unflatten_linear_layers,
    zeropower_via_newtonschulz5,
)
from levanter.utils.jax_utils import leaf_key_paths


def _is_linear_or_none(x: Any) -> bool:
    return isinstance(x, Linear) or x is None


def _create_namo_mask(params: Any) -> Any:
    paths = leaf_key_paths(params)

    def mask_fn(param, path):
        path_str = ".".join(path) if isinstance(path, (list, tuple)) else str(path)
        if "Embedding" in path_str or "lm_head" in path_str:
            return "adamw"
        if isinstance(param, Linear):
            return dataclasses.replace(param, weight="namo", bias="adamw" if param.bias is not None else None)
        return "adamw"

    return haliax.tree_util.tree_map(mask_fn, params, paths, is_leaf=lambda x: isinstance(x, Linear))


def _adamw_transform(
    *,
    max_grad_norm: float,
    beta1: float,
    beta2: float,
    epsilon: float,
    adam_lr: float,
    weight_decay: float,
    build_weight_decay_mask,
) -> optax.GradientTransformation:
    components = []
    if max_grad_norm:
        components.append(optax.clip_by_global_norm(max_grad_norm))
    components.append(optax.scale_by_adam(beta1, beta2, epsilon))
    if weight_decay > 0:
        components.append(optax.add_decayed_weights(weight_decay, build_weight_decay_mask()))
    components.append(optax.scale(-adam_lr))
    return optax.chain(*components)


@OptimizerConfig.register_subclass("namo")
@dataclass(frozen=True)
class NamoConfig(OptimizerConfig):
    """NAMO optimizer with AdamW fallback for non-linear parameters."""

    learning_rate: float = 1e-2
    adam_lr: float = 6e-4
    momentum: float = 0.95
    mu2: float = 0.99
    adamnorm_eps: float = 1e-8
    nesterov: bool = True
    backend_steps: int = 5
    muon_epsilon: float = 1e-8
    scale_coeff: float = 0.2
    adam_weight_decay: Optional[float] = None
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    coefficient_type: CoefficientType = "simple"

    def build(self, num_train_steps: int) -> optax.GradientTransformation:
        learning_rate_schedule = self.lr_scheduler(num_train_steps)
        adam_lr_schedule = self.lr_scheduler(num_train_steps, override_lr=self.adam_lr)

        def optimizer(learning_rate, adam_lr):
            def namo_transform():
                return scale_with_namo(
                    momentum=self.momentum,
                    mu2=self.mu2,
                    nesterov=self.nesterov,
                    steps=self.backend_steps,
                    muon_eps=self.muon_epsilon,
                    learning_rate=learning_rate,
                    weight_decay=self.weight_decay,
                    adamnorm_eps=self.adamnorm_eps,
                    scale_coeff=self.scale_coeff,
                    coefficient_type=self.coefficient_type,
                )

            def adamw_transform():
                adam_weight_decay = self.adam_weight_decay if self.adam_weight_decay is not None else self.weight_decay
                return _adamw_transform(
                    max_grad_norm=self.max_grad_norm,
                    beta1=self.beta1,
                    beta2=self.beta2,
                    epsilon=self.epsilon,
                    adam_lr=adam_lr,
                    weight_decay=adam_weight_decay,
                    build_weight_decay_mask=self.build_weight_decay_mask,
                )

            return optax.multi_transform({"namo": namo_transform(), "adamw": adamw_transform()}, _create_namo_mask)

        return optax.inject_hyperparams(optimizer)(learning_rate=learning_rate_schedule, adam_lr=adam_lr_schedule)


@OptimizerConfig.register_subclass("namoD")
@dataclass(frozen=True)
class NamoDConfig(OptimizerConfig):
    """NAMO-D optimizer with column-wise adaptivity and AdamW fallback."""

    learning_rate: float = 1e-2
    adam_lr: float = 6e-4
    momentum: float = 0.95
    mu2: float = 0.99
    adamnorm_eps: float = 1e-8
    nesterov: bool = True
    backend_steps: int = 5
    muon_epsilon: float = 1e-8
    scale_coeff: float = 0.2
    col_state_clamp_c: float = 0.75
    adam_weight_decay: Optional[float] = None
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    coefficient_type: CoefficientType = "simple"

    def build(self, num_train_steps: int) -> optax.GradientTransformation:
        learning_rate_schedule = self.lr_scheduler(num_train_steps)
        adam_lr_schedule = self.lr_scheduler(num_train_steps, override_lr=self.adam_lr)

        def optimizer(learning_rate, adam_lr):
            def namod_transform():
                return scale_with_namod(
                    momentum=self.momentum,
                    mu2=self.mu2,
                    nesterov=self.nesterov,
                    steps=self.backend_steps,
                    muon_eps=self.muon_epsilon,
                    learning_rate=learning_rate,
                    weight_decay=self.weight_decay,
                    adamnorm_eps=self.adamnorm_eps,
                    scale_coeff=self.scale_coeff,
                    clamp_c=self.col_state_clamp_c,
                    coefficient_type=self.coefficient_type,
                )

            def adamw_transform():
                adam_weight_decay = self.adam_weight_decay if self.adam_weight_decay is not None else self.weight_decay
                return _adamw_transform(
                    max_grad_norm=self.max_grad_norm,
                    beta1=self.beta1,
                    beta2=self.beta2,
                    epsilon=self.epsilon,
                    adam_lr=adam_lr,
                    weight_decay=adam_weight_decay,
                    build_weight_decay_mask=self.build_weight_decay_mask,
                )

            return optax.multi_transform({"namo": namod_transform(), "adamw": adamw_transform()}, _create_namo_mask)

        return optax.inject_hyperparams(optimizer)(learning_rate=learning_rate_schedule, adam_lr=adam_lr_schedule)


class ScaleByNamoState(NamedTuple):
    """Optax-compatible NAMO state.

    We keep this as a ``NamedTuple`` to match existing optimizer-state
    conventions across ``levanter.optim`` and Optax examples.
    """

    step_count: jnp.ndarray
    momentum_buffer: optax.Updates
    v_squared: optax.Updates


class ScaleByNamoDState(NamedTuple):
    """Optax-compatible NAMO-D state."""

    step_count: jnp.ndarray
    momentum_buffer: optax.Updates
    v: optax.Updates


def _clamp_to_mean(col_scale: Array, clamp_c: float) -> Array:
    if not (0.0 < float(clamp_c) <= 1.0):
        return col_scale

    # Clamp per matrix (last axis = columns) so stacked/scanned linears are independent.
    mean = jnp.nanmean(col_scale, axis=-1, keepdims=True)
    floor = jnp.where(jnp.isfinite(mean) & (mean > 0), mean * clamp_c, 0.0)
    ceil = jnp.where(jnp.isfinite(mean) & (mean > 0), mean / clamp_c, jnp.inf)
    return jnp.clip(col_scale, floor, ceil)


def _orthogonalize_batched(
    matrix: Array,
    *,
    steps: int,
    muon_eps: float,
    coefficient_type: CoefficientType,
) -> Array:
    """Apply Newton-Schulz orthogonalization to [..., m, n] tensors."""
    if matrix.ndim == 2:
        return zeropower_via_newtonschulz5(matrix, steps=steps, eps=muon_eps, coefficient_type=coefficient_type)

    flat = matrix.reshape((-1, matrix.shape[-2], matrix.shape[-1]))
    flat_orth = jax.vmap(
        lambda m: zeropower_via_newtonschulz5(m, steps=steps, eps=muon_eps, coefficient_type=coefficient_type)
    )(flat)
    return flat_orth.reshape(matrix.shape)


def scale_with_namo(
    *,
    momentum: float = 0.95,
    mu2: float = 0.99,
    nesterov: bool = True,
    steps: int = 5,
    muon_eps: float = 1e-8,
    learning_rate: float = 1e-2,
    weight_decay: float = 0.1,
    adamnorm_eps: float = 1e-8,
    scale_coeff: float = 0.2,
    coefficient_type: CoefficientType = "simple",
) -> optax.GradientTransformation:
    """Build the NAMO gradient transformation.

    NAMO combines orthogonalized momentum directions with Adam-style norm-based
    adaptive scaling and applies adaptive decoupled weight decay.
    """

    def init_fn(params):
        flat_params = flatten_linear_layers(params)
        momentum_buffer = otu.tree_zeros_like(flat_params)

        def init_v(node):
            if isinstance(node, Linear):
                return jnp.zeros(node.weight.array.shape[:-2], dtype=jnp.float32)
            return None

        v_squared = jax.tree_util.tree_map(init_v, flat_params, is_leaf=_is_linear_or_none)
        return ScaleByNamoState(
            step_count=jnp.zeros([], jnp.int32),
            momentum_buffer=momentum_buffer,
            v_squared=v_squared,
        )

    def update_fn(updates, state, params=None):
        if params is None:
            raise ValueError("NAMO requires params to apply adaptive weight decay")

        flat_updates = flatten_linear_layers(updates)
        flat_params = flatten_linear_layers(params)

        buf = jax.tree.map(
            lambda m, g: None if g is None else momentum * m + (1.0 - momentum) * g,
            state.momentum_buffer,
            flat_updates,
            is_leaf=lambda x: x is None,
        )
        if nesterov:
            m_for_update = jax.tree.map(
                lambda m, g: None if g is None else momentum * m + (1.0 - momentum) * g,
                buf,
                flat_updates,
                is_leaf=lambda x: x is None,
            )
        else:
            m_for_update = buf

        count_inc = optax.safe_increment(state.step_count)
        bc1 = 1.0 - (momentum**count_inc)
        bc2 = 1.0 - (mu2**count_inc)

        def transform(param_node, grad_node, m_node, v_prev):
            if m_node is None:
                return grad_node, v_prev
            if not isinstance(m_node, Linear):
                return grad_node, v_prev

            if v_prev is None:
                v_prev = jnp.zeros(m_node.weight.array.shape[:-2], dtype=jnp.float32)

            grad_arr = grad_node.weight.array
            m_arr = m_node.weight.array
            param_arr = param_node.weight.array

            grad_norm = jnp.linalg.norm(grad_arr, axis=(-2, -1))
            momentum_norm = jnp.linalg.norm(m_arr, axis=(-2, -1))

            v_new = mu2 * v_prev + (1.0 - mu2) * grad_norm * grad_norm
            adaptive_lr = (learning_rate * jnp.sqrt(bc2) / (bc1 + 1e-12)) * (
                momentum_norm / (jnp.sqrt(v_new + adamnorm_eps))
            )
            adaptive_lr = jnp.minimum(adaptive_lr, 1.0)

            shape_scale = scale_coeff * jnp.sqrt(jnp.maximum(m_arr.shape[-2], m_arr.shape[-1]))
            shaped_lr = jnp.minimum(adaptive_lr * shape_scale, 1.0)

            orth_dir = _orthogonalize_batched(
                m_arr,
                steps=steps,
                muon_eps=muon_eps,
                coefficient_type=coefficient_type,
            )
            delta = -(weight_decay * adaptive_lr[..., None, None]) * param_arr - shaped_lr[..., None, None] * orth_dir

            return dataclasses.replace(grad_node, weight=dataclasses.replace(grad_node.weight, array=delta)), v_new

        new_flat_updates = jax.tree_util.tree_map(
            lambda p, g, m, v: transform(p, g, m, v)[0],
            flat_params,
            flat_updates,
            m_for_update,
            state.v_squared,
            is_leaf=_is_linear_or_none,
        )
        new_v = jax.tree_util.tree_map(
            lambda p, g, m, v: transform(p, g, m, v)[1],
            flat_params,
            flat_updates,
            m_for_update,
            state.v_squared,
            is_leaf=_is_linear_or_none,
        )

        final_updates = unflatten_linear_layers(updates, new_flat_updates)

        return final_updates, ScaleByNamoState(step_count=count_inc, momentum_buffer=buf, v_squared=new_v)

    return optax.GradientTransformation(init_fn, update_fn)


def scale_with_namod(
    *,
    momentum: float = 0.95,
    mu2: float = 0.99,
    nesterov: bool = True,
    steps: int = 5,
    muon_eps: float = 1e-8,
    learning_rate: float = 1e-2,
    weight_decay: float = 0.1,
    adamnorm_eps: float = 1e-8,
    scale_coeff: float = 0.2,
    clamp_c: float = 0.75,
    coefficient_type: CoefficientType = "simple",
) -> optax.GradientTransformation:
    """Build the NAMO-D gradient transformation.

    NAMO-D extends NAMO with column-wise adaptivity and clamped scaling to
    balance conditioning and fine-grained noise adaptation.
    """

    def init_fn(params):
        flat_params = flatten_linear_layers(params)
        momentum_buffer = otu.tree_zeros_like(flat_params)

        def init_v(node):
            if isinstance(node, Linear):
                shape = node.weight.array.shape
                return jnp.zeros(shape[:-2] + (shape[-1],), dtype=jnp.float32)
            return None

        v = jax.tree_util.tree_map(init_v, flat_params, is_leaf=_is_linear_or_none)
        return ScaleByNamoDState(
            step_count=jnp.zeros([], jnp.int32),
            momentum_buffer=momentum_buffer,
            v=v,
        )

    def update_fn(updates, state, params=None):
        if params is None:
            raise ValueError("NAMO-D requires params to apply adaptive weight decay")

        flat_updates = flatten_linear_layers(updates)
        flat_params = flatten_linear_layers(params)

        buf = jax.tree.map(
            lambda m, g: None if g is None else momentum * m + (1.0 - momentum) * g,
            state.momentum_buffer,
            flat_updates,
            is_leaf=lambda x: x is None,
        )
        if nesterov:
            m_for_update = jax.tree.map(
                lambda m, g: None if g is None else momentum * m + (1.0 - momentum) * g,
                buf,
                flat_updates,
                is_leaf=lambda x: x is None,
            )
        else:
            m_for_update = buf

        count_inc = optax.safe_increment(state.step_count)
        bc1 = 1.0 - (momentum**count_inc)
        bc2 = 1.0 - (mu2**count_inc)

        def transform(param_node, grad_node, m_node, v_prev):
            if m_node is None:
                return grad_node, v_prev
            if not isinstance(m_node, Linear):
                return grad_node, v_prev

            grad_arr = grad_node.weight.array
            m_arr = m_node.weight.array
            param_arr = param_node.weight.array

            if v_prev is None:
                v_prev = jnp.zeros(m_arr.shape[:-2] + (m_arr.shape[-1],), dtype=jnp.float32)

            col_norm_grad = jnp.linalg.norm(grad_arr, axis=-2)
            v_new = mu2 * v_prev + (1.0 - mu2) * jnp.square(col_norm_grad)

            base_lr = learning_rate * jnp.sqrt(bc2) / (bc1 + 1e-12)

            col_norm_m = jnp.linalg.norm(m_arr, axis=-2)
            col_scale = col_norm_m * jax.lax.rsqrt(v_new + adamnorm_eps)
            col_scale = _clamp_to_mean(col_scale, clamp_c)

            col_lr_wd = jnp.minimum(col_scale * base_lr, 1.0)
            decay = jnp.maximum(1.0 - weight_decay * col_lr_wd, 0.0)

            shape_scale = scale_coeff * jnp.sqrt(jnp.maximum(m_arr.shape[-2], m_arr.shape[-1]))
            col_lr_up = jnp.minimum(col_scale * (base_lr * shape_scale), 1.0)

            orth_dir = _orthogonalize_batched(
                m_arr,
                steps=steps,
                muon_eps=muon_eps,
                coefficient_type=coefficient_type,
            )

            delta = param_arr * (jnp.expand_dims(decay, axis=-2) - 1.0) - orth_dir * jnp.expand_dims(
                col_lr_up, axis=-2
            )
            return dataclasses.replace(grad_node, weight=dataclasses.replace(grad_node.weight, array=delta)), v_new

        new_flat_updates = jax.tree_util.tree_map(
            lambda p, g, m, v: transform(p, g, m, v)[0],
            flat_params,
            flat_updates,
            m_for_update,
            state.v,
            is_leaf=_is_linear_or_none,
        )
        new_v = jax.tree_util.tree_map(
            lambda p, g, m, v: transform(p, g, m, v)[1],
            flat_params,
            flat_updates,
            m_for_update,
            state.v,
            is_leaf=_is_linear_or_none,
        )

        final_updates = unflatten_linear_layers(updates, new_flat_updates)

        return final_updates, ScaleByNamoDState(step_count=count_inc, momentum_buffer=buf, v=new_v)

    return optax.GradientTransformation(init_fn, update_fn)
