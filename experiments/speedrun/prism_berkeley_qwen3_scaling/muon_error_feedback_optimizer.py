# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Error-aware Muon policies for the archived Qwen3 speedrun setting."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from functools import partial
from typing import Literal, NamedTuple

import chex
import haliax
import jax
import jax.numpy as jnp
import optax
from jax.sharding import PartitionSpec
from levanter.optim.config import OptimizerConfig
from levanter.utils.jax_utils import leaf_key_paths
from optax import tree_utils as otu

from experiments.speedrun.prism_berkeley_qwen3_scaling.optimizer_helpers import (
    flatten_linear_layers,
    label_linear_like_module,
    map_flattened_linear_layers,
    zeropower_via_newtonschulz5,
)
from experiments.speedrun.prism_berkeley_qwen3_scaling.prism_berkeley_optimizer import (
    _weight_decay_hyperparam,
)

ErrorAwareMuonPolicy = Literal["muon", "blend", "hesscorr"]

DEFAULT_CUBIC_POWER_ITERATIONS = 5
DEFAULT_SYLVESTER_STEPS = 400
DEFAULT_INVERSE_NEWTON_STEPS = 60
SPECTRAL_NORM_SAFETY_FACTOR = 1.1


def _compute_dtype(array: jax.Array) -> jnp.dtype:
    return jnp.promote_types(array.dtype, jnp.float32)


def quintic_newton_schulz(matrix: jax.Array, *, steps: int = 5, eps: float = 1e-12) -> jax.Array:
    """Apply Muon's constant-coefficient quintic Newton-Schulz iteration."""
    working = matrix.astype(_compute_dtype(matrix))
    return zeropower_via_newtonschulz5(working, steps=steps, eps=eps, coefficient_type="simple")


def _spectral_norm_power_iteration(matrix: jax.Array, *, steps: int, eps: float) -> jax.Array:
    """Estimate the spectral norm with deterministic power iteration."""
    right_dimension = matrix.shape[1]
    vector = jnp.linspace(1.0, 2.0, right_dimension, dtype=matrix.dtype)
    vector = vector / jnp.maximum(jnp.linalg.vector_norm(vector), jnp.asarray(eps, matrix.dtype))

    def iterate(_, value):
        value = matrix.T @ (matrix @ value)
        return value / jnp.maximum(jnp.linalg.vector_norm(value), jnp.asarray(eps, matrix.dtype))

    vector = jax.lax.fori_loop(0, steps, iterate, vector)
    return jnp.linalg.vector_norm(matrix @ vector)


def cubic_newton_schulz(
    matrix: jax.Array,
    *,
    steps: int = 15,
    power_iterations: int = DEFAULT_CUBIC_POWER_ITERATIONS,
    eps: float = 1e-12,
) -> jax.Array:
    """Apply cubic Newton-Schulz after power-iteration spectral normalization."""
    chex.assert_rank(matrix, 2)
    working = matrix.astype(_compute_dtype(matrix))
    spectral_norm = _spectral_norm_power_iteration(working, steps=power_iterations, eps=eps)
    denominator = jnp.maximum(
        SPECTRAL_NORM_SAFETY_FACTOR * spectral_norm,
        jnp.asarray(eps, working.dtype),
    )
    working = working / denominator

    transposed = working.shape[0] > working.shape[1]
    if transposed:
        working = working.T

    if not jax.sharding.get_abstract_mesh().empty:
        working = jax.lax.with_sharding_constraint(working, PartitionSpec(None, ("data", "model")))

    def iterate(_, value):
        return 1.5 * value - 0.5 * value @ value.T @ value

    working = jax.lax.fori_loop(0, int(steps), iterate, working)

    if transposed:
        working = working.T
    return working


def _inverse_spd_newton(matrix: jax.Array, *, steps: int, eps: float) -> jax.Array:
    """Approximate an SPD inverse with Newton--Hotelling matrix products."""
    dimension = matrix.shape[0]
    identity = jnp.eye(dimension, dtype=matrix.dtype)
    scale = jnp.maximum(jnp.linalg.norm(matrix, ord="fro"), jnp.asarray(eps, matrix.dtype))
    inverse = identity / scale

    def iterate(_, value):
        return value @ (2.0 * identity - matrix @ value)

    return jax.lax.fori_loop(0, steps, iterate, inverse)


def _solve_sylvester_fixed_point(
    hessian: jax.Array,
    skew_term: jax.Array,
    *,
    steps: int,
    eps: float,
) -> jax.Array:
    """Solve ``H S + S H = C`` by the damped product-only fixed point."""
    scale = jnp.maximum(jnp.linalg.norm(hessian, ord="fro"), jnp.asarray(eps, hessian.dtype))
    solution = jnp.zeros_like(skew_term)

    def iterate(_, value):
        return value + (skew_term - hessian @ value - value @ hessian) / (2.0 * scale)

    return jax.lax.fori_loop(0, steps, iterate, solution)


def _nuclear_hessian_sylvester(
    matrix: jax.Array,
    tangent: jax.Array,
    *,
    cubic_steps: int,
    sylvester_steps: int,
    inverse_steps: int,
    eps: float,
) -> jax.Array:
    """Apply the nuclear-norm Hessian through the SVD-free Sylvester identity."""
    if matrix.shape[0] < matrix.shape[1]:
        return _nuclear_hessian_sylvester(
            matrix.T,
            tangent.T,
            cubic_steps=cubic_steps,
            sylvester_steps=sylvester_steps,
            inverse_steps=inverse_steps,
            eps=eps,
        ).T

    polar = cubic_newton_schulz(matrix, steps=cubic_steps, eps=eps)
    polar_hessian = 0.5 * (polar.T @ matrix + matrix.T @ polar)
    skew_term = polar.T @ tangent - tangent.T @ polar
    skew_solution = _solve_sylvester_fixed_point(
        polar_hessian,
        skew_term,
        steps=sylvester_steps,
        eps=eps,
    )
    inverse_hessian = _inverse_spd_newton(polar_hessian, steps=inverse_steps, eps=eps)
    normal_complement = tangent - polar @ (polar.T @ tangent)
    return polar @ skew_solution + normal_complement @ inverse_hessian


def clipped_nuclear_hessian(
    matrix: jax.Array,
    tangent: jax.Array,
    *,
    steps: int = 15,
    sylvester_steps: int = DEFAULT_SYLVESTER_STEPS,
    inverse_steps: int = DEFAULT_INVERSE_NEWTON_STEPS,
    eps: float = 1e-12,
) -> jax.Array:
    """Apply the SVD-free nuclear-norm Hessian and cap its Frobenius norm."""
    if matrix.shape != tangent.shape:
        raise ValueError(f"Expected matching matrix and tangent shapes, got {matrix.shape} and {tangent.shape}.")

    compute_dtype = _compute_dtype(matrix)
    working_matrix = matrix.astype(compute_dtype)
    working_tangent = tangent.astype(compute_dtype)
    correction = _nuclear_hessian_sylvester(
        working_matrix,
        working_tangent,
        cubic_steps=steps,
        sylvester_steps=sylvester_steps,
        inverse_steps=inverse_steps,
        eps=eps,
    )

    cap = jnp.sqrt(jnp.asarray(min(matrix.shape), dtype=correction.dtype))
    correction_norm = jnp.linalg.norm(correction, ord="fro")
    clip_scale = jnp.minimum(1.0, cap / jnp.maximum(correction_norm, jnp.asarray(eps, correction.dtype)))
    return correction * clip_scale


def error_aware_muon_step(
    momentum_matrix: jax.Array,
    gradient: jax.Array,
    *,
    policy: ErrorAwareMuonPolicy,
    blend_gain: float = 0.0,
    correction_gain: float = 0.0,
    quintic_steps: int = 5,
    cubic_steps: int = 15,
    sylvester_steps: int = DEFAULT_SYLVESTER_STEPS,
    inverse_steps: int = DEFAULT_INVERSE_NEWTON_STEPS,
    eps: float = 1e-12,
) -> jax.Array:
    """Return a unit-learning-rate Muon, blend, or Hessian-corrected step."""
    if momentum_matrix.shape != gradient.shape:
        raise ValueError(
            f"Expected matching momentum and gradient shapes, got {momentum_matrix.shape} and {gradient.shape}."
        )

    compute_dtype = _compute_dtype(momentum_matrix)
    momentum_matrix = momentum_matrix.astype(compute_dtype)
    gradient = gradient.astype(compute_dtype)

    if policy == "muon":
        return quintic_newton_schulz(momentum_matrix, steps=quintic_steps, eps=eps)
    if policy == "blend":
        blended = momentum_matrix + blend_gain * (gradient - momentum_matrix)
        return quintic_newton_schulz(blended, steps=quintic_steps, eps=eps)
    if policy == "hesscorr":
        base_step = quintic_newton_schulz(momentum_matrix, steps=quintic_steps, eps=eps)
        correction = clipped_nuclear_hessian(
            momentum_matrix,
            gradient - momentum_matrix,
            steps=cubic_steps,
            sylvester_steps=sylvester_steps,
            inverse_steps=inverse_steps,
            eps=eps,
        )
        return base_step + correction_gain * correction
    raise ValueError(f"Unsupported error-aware Muon policy: {policy!r}.")


class ScaleByErrorAwareMuonState(NamedTuple):
    """State for normalized-EMA error-aware Muon."""

    momentum_buffer: optax.Updates


def _tree_zeros_like_float32(params):
    zeros = otu.tree_zeros_like(params)

    def to_float32(value):
        if hasattr(value, "dtype") and jnp.issubdtype(value.dtype, jnp.inexact):
            return value.astype(jnp.float32)
        return value

    return jax.tree.map(to_float32, zeros)


def scale_with_error_aware_muon(
    *,
    momentum: float = 0.95,
    nesterov: bool = False,
    policy: ErrorAwareMuonPolicy = "hesscorr",
    blend_gain: float = 0.0,
    correction_gain: float = 1.0,
    quintic_steps: int = 5,
    cubic_steps: int = 15,
    sylvester_steps: int = DEFAULT_SYLVESTER_STEPS,
    inverse_steps: int = DEFAULT_INVERSE_NEWTON_STEPS,
    muon_eps: float = 1e-12,
    use_kimi_scaling: bool = False,
):
    """Build the Optax transform for error-aware Muon matrix updates."""
    quintic_steps = int(quintic_steps)
    cubic_steps = int(cubic_steps)
    sylvester_steps = int(sylvester_steps)
    inverse_steps = int(inverse_steps)

    def init_fn(params):
        return ScaleByErrorAwareMuonState(momentum_buffer=_tree_zeros_like_float32(params))

    def update_fn(updates, state, params=None):
        del params
        raw_gradients = updates
        gradient_weight = 1.0 - momentum
        momentum_buffer = jax.tree.map(
            lambda old, gradient: (
                None if gradient is None else momentum * old + gradient_weight * gradient.astype(jnp.float32)
            ),
            state.momentum_buffer,
            raw_gradients,
            is_leaf=lambda value: value is None,
        )

        if nesterov:
            policy_momentum = jax.tree.map(
                lambda current, gradient: (
                    None if gradient is None else momentum * current + gradient_weight * gradient.astype(jnp.float32)
                ),
                momentum_buffer,
                raw_gradients,
                is_leaf=lambda value: value is None,
            )
        else:
            policy_momentum = momentum_buffer

        def transform_linear_layer(
            momentum_layer: haliax.nn.Linear,
            gradient_layer: haliax.nn.Linear,
        ) -> haliax.nn.Linear:
            if not isinstance(momentum_layer.weight, haliax.NamedArray) or not isinstance(
                gradient_layer.weight, haliax.NamedArray
            ):
                return momentum_layer

            momentum_array = momentum_layer.weight.array
            gradient_array = gradient_layer.weight.array
            if momentum_array.ndim != 2 or gradient_array.ndim != 2:
                raise ValueError(
                    "Error-aware Muon expects scan-aware flattened rank-2 linear weights, "
                    f"got {momentum_array.shape} and {gradient_array.shape}."
                )

            transformed = error_aware_muon_step(
                momentum_array,
                gradient_array,
                policy=policy,
                blend_gain=blend_gain,
                correction_gain=correction_gain,
                quintic_steps=quintic_steps,
                cubic_steps=cubic_steps,
                sylvester_steps=sylvester_steps,
                inverse_steps=inverse_steps,
                eps=muon_eps,
            )
            if use_kimi_scaling:
                scale = 0.2 * jnp.sqrt(jnp.maximum(transformed.shape[0], transformed.shape[1]))
            else:
                scale = jnp.sqrt(jnp.maximum(1, transformed.shape[0] / transformed.shape[1]))
            transformed = (transformed * scale).astype(gradient_array.dtype)

            return dataclasses.replace(
                momentum_layer,
                weight=dataclasses.replace(momentum_layer.weight, array=transformed),
            )

        transformed_updates = map_flattened_linear_layers(
            transform_linear_layer,
            policy_momentum,
            raw_gradients,
        )
        return transformed_updates, ScaleByErrorAwareMuonState(momentum_buffer=momentum_buffer)

    return optax.GradientTransformation(init_fn, update_fn)


@OptimizerConfig.register_subclass("error_aware_muon")
@dataclass(frozen=True)
class ErrorAwareMuonConfig(OptimizerConfig):
    """Optimizer config for Muon with filtering-error feedback policies."""

    adam_lr: float = 6e-4
    momentum: float = 0.95
    nesterov: bool = False
    policy: ErrorAwareMuonPolicy = "hesscorr"
    blend_gain: float = 0.0
    correction_gain: float = 1.0
    quintic_steps: int = 5
    cubic_steps: int = 15
    sylvester_steps: int = DEFAULT_SYLVESTER_STEPS
    inverse_steps: int = DEFAULT_INVERSE_NEWTON_STEPS
    adam_weight_decay: float | None = None
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    muon_epsilon: float = 1e-12
    adamc_weight_decay: bool = False
    max_grad_norm: float = 1.0
    use_kimi_scaling: bool = False
    min_matrix_dim: int = 8

    def _validate(self) -> None:
        if not 0.0 <= self.momentum < 1.0:
            raise ValueError(f"momentum must be in [0, 1), got {self.momentum}.")
        if self.policy not in ("muon", "blend", "hesscorr"):
            raise ValueError(f"Unsupported error-aware Muon policy: {self.policy!r}.")
        if not 0.0 <= self.blend_gain <= 1.0:
            raise ValueError(f"blend_gain must be in [0, 1], got {self.blend_gain}.")
        if self.correction_gain < 0.0:
            raise ValueError(f"correction_gain must be non-negative, got {self.correction_gain}.")
        if min(self.quintic_steps, self.cubic_steps, self.sylvester_steps, self.inverse_steps) <= 0:
            raise ValueError("All Newton-Schulz, Sylvester, and inverse iteration counts must be positive.")
        if self.muon_epsilon <= 0.0:
            raise ValueError(f"muon_epsilon must be positive, got {self.muon_epsilon}.")
        if self.min_matrix_dim <= 0:
            raise ValueError(f"min_matrix_dim must be positive, got {self.min_matrix_dim}.")

    def build(self, num_train_steps: int):
        self._validate()
        learning_rate_schedule = self.lr_scheduler(num_train_steps)
        adam_lr_schedule = self.lr_scheduler(num_train_steps, override_lr=self.adam_lr)
        weight_decay_hyperparam = _weight_decay_hyperparam(
            self.weight_decay,
            learning_rate_schedule=learning_rate_schedule,
            peak_learning_rate=self.learning_rate,
            adamc_weight_decay=self.adamc_weight_decay,
        )
        adam_base_weight_decay = self.adam_weight_decay if self.adam_weight_decay is not None else self.weight_decay
        adam_weight_decay_hyperparam = _weight_decay_hyperparam(
            adam_base_weight_decay,
            learning_rate_schedule=adam_lr_schedule,
            peak_learning_rate=self.adam_lr,
            adamc_weight_decay=self.adamc_weight_decay,
        )

        def optimizer(learning_rate, adam_lr, weight_decay, adam_weight_decay):
            muon_transform = optax.chain(
                scale_with_error_aware_muon(
                    momentum=self.momentum,
                    nesterov=self.nesterov,
                    policy=self.policy,
                    blend_gain=self.blend_gain,
                    correction_gain=self.correction_gain,
                    quintic_steps=self.quintic_steps,
                    cubic_steps=self.cubic_steps,
                    sylvester_steps=self.sylvester_steps,
                    inverse_steps=self.inverse_steps,
                    muon_eps=self.muon_epsilon,
                    use_kimi_scaling=self.use_kimi_scaling,
                ),
                optax.add_decayed_weights(weight_decay, self.build_weight_decay_mask()),
                optax.scale(-learning_rate),
            )

            adam_components = []
            if self.max_grad_norm:
                adam_components.append(optax.clip_by_global_norm(self.max_grad_norm))
            adam_components.extend(
                [
                    optax.scale_by_adam(self.beta1, self.beta2, self.epsilon),
                    optax.add_decayed_weights(adam_weight_decay, self.build_weight_decay_mask()),
                    optax.scale(-adam_lr),
                ]
            )
            return optax.multi_transform(
                {
                    "error_aware_muon": muon_transform,
                    "adamw": optax.chain(*adam_components),
                },
                partial(self.create_mask, use_kimi_scaling=self.use_kimi_scaling),
            )

        return optax.inject_hyperparams(optimizer)(
            learning_rate=learning_rate_schedule,
            adam_lr=adam_lr_schedule,
            weight_decay=weight_decay_hyperparam,
            adam_weight_decay=adam_weight_decay_hyperparam,
        )

    def create_mask(self, params, use_kimi_scaling: bool = True):
        paths = leaf_key_paths(params)

        def mask_fn(param, path):
            path_str = ".".join(path) if isinstance(path, (list, tuple)) else str(path)
            path_lower = path_str.lower()
            if "embedding" in path_lower or "lm_head" in path_lower:
                return "adamw"
            if isinstance(param, haliax.nn.Linear):
                flattened = flatten_linear_layers(param)
                weight = flattened.weight
                is_large_matrix = (
                    isinstance(weight, haliax.NamedArray)
                    and weight.array.ndim >= 2
                    and min(weight.array.shape[-2:]) >= self.min_matrix_dim
                )
                if is_large_matrix:
                    if not param._out_first and not use_kimi_scaling:
                        raise ValueError("Original Muon scaling requires output-first linear weights.")
                    return label_linear_like_module(
                        param,
                        weight_label="error_aware_muon",
                        bias_label="adamw",
                    )
                return label_linear_like_module(param, weight_label="adamw", bias_label="adamw")
            return "adamw"

        return haliax.tree_util.tree_map(
            mask_fn,
            params,
            paths,
            is_leaf=lambda value: isinstance(value, haliax.nn.Linear),
        )


OptimizerConfig.register_subclass("error-aware-muon", ErrorAwareMuonConfig)
