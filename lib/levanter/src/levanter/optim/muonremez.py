# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
from dataclasses import dataclass
from functools import partial
from typing import NamedTuple, Optional

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


@OptimizerConfig.register_subclass("muonremez")
@dataclass(frozen=True)
class MuonRemezConfig(OptimizerConfig):
    """
    MuonRemez optimizer configuration: Similar to Muon but uses coupled Newton-Schulz iteration
    to compute U·Σ^(1/2)·V^T instead of Newton-Schulz orthogonalization (which computes Σ^0).

    Update rule:
        v_t = β v_{t-1} + g_t                    (momentum)
        u_t = compute_u_sigma_half_vt(v_t)       (matrix square root via coupled NS)
        W ← (1 - η*λ) W - η u_t                  (update with weight decay)
    """

    lr: float = 0.02
    adam_lr: float = 6e-4  # Adam LR
    momentum: float = 0.95
    nesterov: bool = True
    backend_steps: int = 7  # Number of steps for coupled Newton-Schulz iteration
    weight_decay: float = 0.0
    adam_weight_decay: Optional[float] = None
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    muon_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    use_kimi_scaling: bool = False

    def build(self, num_train_steps):
        """
        Creates the optimizer.
        """
        learning_rate_schedule = self.lr_scheduler(num_train_steps)
        adam_lr_schedule = self.lr_scheduler(num_train_steps, override_lr=self.adam_lr)

        def optimizer(learning_rate, adam_lr):
            def muonremez_transform():
                components = []
                components.append(
                    scale_with_muonremez(
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
                adam_weight_decay = self.adam_weight_decay if self.adam_weight_decay is not None else self.weight_decay
                if adam_weight_decay > 0:
                    components.append(optax.add_decayed_weights(adam_weight_decay, self.build_weight_decay_mask()))
                components.append(optax.scale(-adam_lr))
                optimizer = optax.chain(*components)
                return optimizer

            transformations = {
                "muonremez": muonremez_transform(),
                "adamw": adamw_transform(),
            }

            return optax.multi_transform(
                transformations, partial(self.create_mask, use_kimi_scaling=self.use_kimi_scaling)
            )

        return optax.inject_hyperparams(optimizer)(learning_rate=learning_rate_schedule, adam_lr=adam_lr_schedule)

    def create_mask(self, params, use_kimi_scaling=True):
        """
        Creates a mask that labels parameters as 'muonremez' or 'adamw' based on their
        dimensionality and module path, using AdamW for Embedding and lm_head parameters.
        """
        paths = leaf_key_paths(params)

        def mask_fn(param, path):
            path_str = ".".join(path) if isinstance(path, (list, tuple)) else str(path)
            if "Embedding" in path_str or "lm_head" in path_str:
                return "adamw"
            elif isinstance(param, Linear):
                # muonremez for linear layers
                assert (
                    param._out_first or use_kimi_scaling
                )  # if we don't use kimi's version of scaling, then we need to assume out_first to ensure we are scaling like Out/In
                return dataclasses.replace(param, weight="muonremez", bias="adamw" if param.bias is not None else None)
            else:
                return "adamw"

        return haliax.tree_util.tree_map(mask_fn, params, paths, is_leaf=lambda x: isinstance(x, Linear))


class ScaleByMuonRemezState(NamedTuple):
    """State for the MuonRemez algorithm."""

    momentum_buffer: optax.Updates


def scale_with_muonremez(momentum=0.95, nesterov=True, steps=7, muon_eps=1e-8, use_kimi_scaling=False):
    # Convert steps to concrete int at function definition time
    steps = int(steps)

    def init_fn(params):
        momentum_buffer = otu.tree_zeros_like(params)
        return ScaleByMuonRemezState(momentum_buffer=momentum_buffer)

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
            updated_weight_array = compute_u_sigma_half_vt(array, steps=steps, eps=muon_eps)

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

        return updates, ScaleByMuonRemezState(momentum_buffer=buf)

    return optax.GradientTransformation(init_fn, update_fn)


def get_quintic_coeffs(variant="higham"):
    """
    Returns polynomial coefficients (alpha, beta, gamma).
    Higham (Standard): Order 5 convergence, very stable.
    Muon Aggressive: Optimized for speed (steep slope at 0), theoretically derived for 5 steps.
    """
    if variant == "higham":
        return 1.875, -1.25, 0.375
    elif variant == "muon_aggressive":
        # These are approximate values often used to explain the "Greedy" approach
        # Real Muon implementations tune these based on specific constraints.
        return 3.4445, -4.7750, 2.0315
    else:
        return 1.875, -1.25, 0.375


def coupled_newton_schulz_quintic(A, steps=5, variant="muon_aggressive", eps=1e-8):
    """
    Coupled Iteration for Square Root.
    Target: Z -> A^(-1/2), Y -> A^(1/2)

    Args:
        A: Input matrix (2D array, should be square and positive semidefinite)
        steps: Number of Newton-Schulz iterations
        variant: Variant of the algorithm ('higham' or 'muon_aggressive')
        eps: Small constant for numerical stability

    Returns:
        Y: A^(1/2) (matrix square root)
        Z: A^(-1/2) (inverse matrix square root)
    """
    chex.assert_rank(A, 2)
    # A should be square for this algorithm (guaranteed by construction in compute_u_sigma_half_vt)

    alpha, beta, gamma = get_quintic_coeffs(variant)

    # Normalization (Crucial)
    norm_A = jnp.linalg.norm(A) + eps
    A_scaled = A / norm_A

    Y = A_scaled
    Z = jnp.eye(A.shape[0], dtype=A.dtype)
    I = jnp.eye(A.shape[0], dtype=A.dtype)

    for k in range(steps):
        P = Z @ Y
        P2 = P @ P
        S = alpha * I + beta * P + gamma * P2
        Y = Y @ S
        Z = S @ Z

    sqrt_norm = jnp.sqrt(norm_A)
    return Y * sqrt_norm, Z * (1.0 / sqrt_norm)


def compute_u_sigma_half_vt(M, steps=7, eps=1e-8, variant="higham"):
    """
    Chained Coupled Iteration to compute U·Σ^(1/2)·V^T (matrix square root).

    This computes the square root of a matrix via chained coupled Newton-Schulz iteration,
    which is different from the zeroth power (orthogonalization) used in standard Muon.

    For a matrix G, this computes an approximation to U·Σ^(1/2)·V^T where G = U·Σ·V^T is the SVD.

    Args:
        M: Input matrix (2D array)
        steps: Number of Newton-Schulz iterations
        eps: Small constant for numerical stability
        variant: Variant of the algorithm ('higham' or 'muon_aggressive')

    Returns:
        Matrix square root U·Σ^(1/2)·V^T
    """
    chex.assert_rank(M, 2)

    # Pre-normalize M
    # Check raw norm first (JIT-safe) to handle zero matrix case
    norm_M = jnp.linalg.norm(M) + eps
    M_scaled = M / norm_M

    if M_scaled.shape[0] >= M_scaled.shape[1]:
        # Tall matrix: compute M^T M first
        A = M_scaled.T @ M_scaled
        _, Z1 = coupled_newton_schulz_quintic(A, steps, variant, eps)
        Y2, _ = coupled_newton_schulz_quintic(Z1, steps, variant, eps)
        Result_scaled = M_scaled @ Y2
    else:
        # Wide matrix: compute M M^T first
        A = M_scaled @ M_scaled.T
        _, Z1 = coupled_newton_schulz_quintic(A, steps, variant, eps)
        Y2, _ = coupled_newton_schulz_quintic(Z1, steps, variant, eps)
        Result_scaled = Y2 @ M_scaled

    result = Result_scaled * jnp.sqrt(norm_M)
    return result
