# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
from dataclasses import dataclass
from functools import partial
from typing import NamedTuple

import chex
import jax
import jax.numpy as jnp
import optax
from optax import tree_utils as otu

import haliax as hax

from levanter.optim.config import OptimizerConfig
from experiments.speedrun.muonremez_qwen3_scaling.optimizer_helpers import (
    label_linear_like_module,
    map_flattened_linear_layers,
)
from levanter.utils.jax_utils import leaf_key_paths


def _weight_decay_hyperparam(
    base_weight_decay: float,
    *,
    learning_rate_schedule,
    peak_learning_rate: float,
    adamc_weight_decay: bool,
):
    if not adamc_weight_decay:
        return base_weight_decay
    if peak_learning_rate <= 0:
        raise ValueError(f"peak_learning_rate must be positive, got {peak_learning_rate}.")

    def schedule(count):
        return base_weight_decay * (learning_rate_schedule(count) / peak_learning_rate)

    return schedule


@OptimizerConfig.register_subclass("muonremez")
@dataclass(frozen=True)
class MuonRemezConfig(OptimizerConfig):
    """MuonRemez uses a coupled Newton-Schulz matrix-square-root backend on linear weights."""

    adam_lr: float = 6e-4
    momentum: float = 0.95
    nesterov: bool = True
    backend_steps: int = 7
    adam_weight_decay: float | None = None
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    muon_epsilon: float = 1e-8
    adamc_weight_decay: bool = False
    max_grad_norm: float = 1.0
    use_kimi_scaling: bool = False

    def build(self, num_train_steps):
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
            def muonremez_transform():
                components = [
                    scale_with_muonremez(
                        self.momentum,
                        self.nesterov,
                        self.backend_steps,
                        self.muon_epsilon,
                        self.use_kimi_scaling,
                    )
                ]
                components.append(optax.add_decayed_weights(weight_decay, self.build_weight_decay_mask()))
                components.append(optax.scale(-learning_rate))
                return optax.chain(*components)

            def adamw_transform():
                components = []
                if self.max_grad_norm:
                    components.append(optax.clip_by_global_norm(self.max_grad_norm))
                components.append(optax.scale_by_adam(self.beta1, self.beta2, self.epsilon))
                components.append(optax.add_decayed_weights(adam_weight_decay, self.build_weight_decay_mask()))
                components.append(optax.scale(-adam_lr))
                return optax.chain(*components)

            transformations = {
                "muonremez": muonremez_transform(),
                "adamw": adamw_transform(),
            }

            return optax.multi_transform(
                transformations,
                partial(self.create_mask, use_kimi_scaling=self.use_kimi_scaling),
            )

        return optax.inject_hyperparams(optimizer)(
            learning_rate=learning_rate_schedule,
            adam_lr=adam_lr_schedule,
            weight_decay=weight_decay_hyperparam,
            adam_weight_decay=adam_weight_decay_hyperparam,
        )

    def create_mask(self, params, use_kimi_scaling=True):
        paths = leaf_key_paths(params)

        def mask_fn(param, path):
            path_str = ".".join(path) if isinstance(path, (list, tuple)) else str(path)
            if "Embedding" in path_str or "lm_head" in path_str:
                return "adamw"
            if isinstance(param, hax.nn.Linear):
                assert param._out_first or use_kimi_scaling
                return label_linear_like_module(param, weight_label="muonremez", bias_label="adamw")
            return "adamw"

        return hax.tree_util.tree_map(mask_fn, params, paths, is_leaf=lambda x: isinstance(x, hax.nn.Linear))


class ScaleByMuonRemezState(NamedTuple):
    momentum_buffer: optax.Updates


def scale_with_muonremez(momentum=0.95, nesterov=True, steps=7, muon_eps=1e-8, use_kimi_scaling=False):
    steps = int(steps)

    def init_fn(params):
        return ScaleByMuonRemezState(momentum_buffer=otu.tree_zeros_like(params))

    def update_fn(updates, state, params=None):
        del params
        momentum_buffer = jax.tree.map(
            lambda m, g: None if g is None else momentum * m + g,
            state.momentum_buffer,
            updates,
            is_leaf=lambda x: x is None,
        )
        if nesterov:
            updates = jax.tree.map(
                lambda m, g: None if g is None else momentum * m + g,
                momentum_buffer,
                updates,
                is_leaf=lambda x: x is None,
            )
        else:
            updates = momentum_buffer

        def transform_linear_layer(layer: hax.nn.Linear):
            assert layer.weight.ndim == 2
            transformed = compute_u_sigma_half_vt(layer.weight.array, steps=steps, eps=muon_eps)
            if use_kimi_scaling:
                scale = 0.2 * jnp.sqrt(jnp.maximum(transformed.shape[0], transformed.shape[1]))
            else:
                scale = jnp.sqrt(jnp.maximum(1.0, transformed.shape[0] / transformed.shape[1]))
            return dataclasses.replace(layer, weight=dataclasses.replace(layer.weight, array=transformed * scale))

        updates = map_flattened_linear_layers(transform_linear_layer, updates)
        return updates, ScaleByMuonRemezState(momentum_buffer=momentum_buffer)

    return optax.GradientTransformation(init_fn, update_fn)


def _get_quintic_coeffs(variant="higham"):
    if variant == "higham":
        return 1.875, -1.25, 0.375
    if variant == "muon_aggressive":
        return 3.4445, -4.7750, 2.0315
    raise ValueError(f"Unknown coupled Newton-Schulz variant: {variant!r}")


def coupled_newton_schulz_quintic(A, steps=5, variant="muon_aggressive", eps=1e-8):
    """Compute ``A^(1/2)`` and ``A^(-1/2)`` with a coupled quintic Newton-Schulz iteration."""

    chex.assert_rank(A, 2)
    alpha, beta, gamma = _get_quintic_coeffs(variant)

    norm_A = jnp.linalg.norm(A) + eps
    A_scaled = A / norm_A

    Y = A_scaled
    Z = jnp.eye(A.shape[0], dtype=A.dtype)
    I = jnp.eye(A.shape[0], dtype=A.dtype)

    for _ in range(steps):
        P = Z @ Y
        P2 = P @ P
        S = alpha * I + beta * P + gamma * P2
        Y = Y @ S
        Z = S @ Z

    sqrt_norm = jnp.sqrt(norm_A)
    return Y * sqrt_norm, Z / sqrt_norm


def compute_u_sigma_half_vt(M, steps=7, eps=1e-8, variant="higham"):
    """Approximate ``U Σ^(1/2) V^T`` for a matrix ``M = U Σ V^T``."""

    chex.assert_rank(M, 2)

    norm_M = jnp.linalg.norm(M) + eps
    M_scaled = M / norm_M

    if M_scaled.shape[0] >= M_scaled.shape[1]:
        A = M_scaled.T @ M_scaled
        _, Z1 = coupled_newton_schulz_quintic(A, steps, variant, eps)
        Y2, _ = coupled_newton_schulz_quintic(Z1, steps, variant, eps)
        result_scaled = M_scaled @ Y2
    else:
        A = M_scaled @ M_scaled.T
        _, Z1 = coupled_newton_schulz_quintic(A, steps, variant, eps)
        Y2, _ = coupled_newton_schulz_quintic(Z1, steps, variant, eps)
        result_scaled = Y2 @ M_scaled

    return result_scaled * jnp.sqrt(norm_M)
