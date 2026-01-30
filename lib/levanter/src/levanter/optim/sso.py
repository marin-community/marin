# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""
SSO (Spectral Sphere Optimizer) and MuonSphere optimizers.

SSO: Full spectral sphere optimization with lambda solver (Algorithm 1 from paper)
MuonSphere: Simplified version with lambda=0

Both optimizers:
- Retract 2D weight matrices to spectral sphere with radius R = radius_scaler * sqrt(d_out/d_in)
- Apply msign update (matrix sign function via Newton-Schulz iteration)
- SSO solves for lambda to enforce tangent constraint, MuonSphere uses lambda=0
"""

import dataclasses
from dataclasses import dataclass
from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
import optax
from optax import tree_utils as otu

import haliax
from haliax.nn import Linear

from levanter.optim.config import OptimizerConfig
from levanter.optim.util import flatten_linear_layers, map_flattened_linear_layers, unflatten_linear_layers
from levanter.utils.jax_utils import leaf_key_paths
from haliax.tree_util import scan_aware_tree_map


# -------------------------
# Small math helpers
# -------------------------
def _safe_l2_norm(x: jnp.ndarray, eps: float = 1e-12) -> jnp.ndarray:
    return jnp.sqrt(jnp.sum(x * x) + eps)


def _safe_fro_norm(A: jnp.ndarray, eps: float = 1e-12) -> jnp.ndarray:
    return jnp.sqrt(jnp.sum(A * A) + eps)


def spectral_mup_radius(d_out: int, d_in: int, radius_scaler: float = 1.0) -> jnp.ndarray:
    """R = radius_scaler * sqrt(d_out / d_in)."""
    return radius_scaler * jnp.sqrt(jnp.array(d_out, jnp.float32) / jnp.array(d_in, jnp.float32))


# -------------------------
# msign via Newton–Schulz with Polar Express coefficients
# -------------------------

# Coefficient sets from NVIDIA NeMo Emerging-Optimizers
# Each tuple is (a, b, c) for: X <- a*X + (b*A + c*A@A)@X, with A = X@X^T
_COEFFICIENT_SETS = {
    "simple": [
        (3.4445, -4.7750, 2.0315),
    ],
    "quintic": [
        (4.0848, -6.8946, 2.9270),
        (3.9505, -6.3029, 2.6377),
        (3.7418, -5.5913, 2.3037),
        (2.8769, -3.1427, 1.2046),
        (2.8366, -3.0525, 1.2012),
    ],
    "polar_express": [
        (8.2051, -22.9019, 16.4607),
        (4.0664, -2.8612, 0.5184),
        (3.9096, -2.8234, 0.5250),
        (3.2856, -2.4153, 0.4853),
        (2.2779, -1.6198, 0.3985),
        (1.8726, -1.2307, 0.3585),
        (1.8564, -1.2132, 0.3568),
        (1.8750, -1.2500, 0.3750),
    ],
    "aol": [
        (4.0098, -7.0585, 2.4635),
        (3.4585, -5.5479, 2.5959),
        (2.7573, -3.2939, 1.4254),
        (2.7215, -3.0494, 1.3169),
    ],
}


def msign_newton_schulz(
    A: jnp.ndarray,
    steps: int = 8,
    coefficient_type: str = "polar_express",
    eps: float = 1e-12,
) -> jnp.ndarray:
    """
    Approximate msign(A) = U V^T, where A = U Σ V^T.
    Uses Newton–Schulz iteration with Polar Express coefficients.

    This implementation matches the NVIDIA Emerging-Optimizers approach:
    - Frobenius normalization
    - Polar Express coefficient schedule (default)
    - 8 iterations (default for SSO)
    """
    X = A.astype(jnp.float32)
    m, n = X.shape

    coeffs = _COEFFICIENT_SETS[coefficient_type]
    L = len(coeffs)
    if steps % L != 0:
        raise ValueError(f"steps ({steps}) must be a multiple of len(coeffs) ({L}) for {coefficient_type!r}.")

    # Auto-transpose to operate on smaller dimension (cheaper A = X X^T)
    transpose = m > n
    if transpose:
        X = X.T

    # Ensure spectral norm <= 1 by Frobenius normalization
    X = X / _safe_fro_norm(X, eps=eps)

    # Use regular for loop which will be unrolled at trace time
    for i in range(steps):
        a, b, c = coeffs[i % L]
        # A = X @ X^T
        A = X @ X.T
        # B = b*A + c*A^2
        A2 = A @ A
        B = b * A + c * A2
        # X <- a*X + B@X
        X = a * X + (B @ X)

    if transpose:
        X = X.T
    return X


# -------------------------
# Power Iteration for top singular triplet
# -------------------------
def power_iteration_top_singular(
    W: jnp.ndarray,
    steps: int = 20,
    eps: float = 1e-12,
):
    """
    Returns (sigma, u, v) approx top singular value/vectors of W using power iteration.
    """
    W32 = W.astype(jnp.float32)
    m, n = W32.shape

    # Deterministic initialization
    v = jnp.ones((n,), dtype=jnp.float32)
    v = v / _safe_l2_norm(v, eps=eps)

    # Use regular for loop
    for _ in range(steps):
        u = W32 @ v
        u = u / _safe_l2_norm(u, eps=eps)
        v = W32.T @ u
        v = v / _safe_l2_norm(v, eps=eps)

    u = W32 @ v
    u = u / _safe_l2_norm(u, eps=eps)

    # Rayleigh quotient style estimate of sigma
    sigma = u @ (W32 @ v)
    return sigma, u, v


# -------------------------
# Lambda solver (bracket + bisection) for SSO
# -------------------------
def solve_lambda_bisection(
    G: jnp.ndarray,
    Theta: jnp.ndarray,
    tol: float = 1e-8,
    max_iter: int = 20,
    msign_steps: int = 8,
    eps: float = 1e-12,
) -> jnp.ndarray:
    """
    Solve for lambda* such that: h(lambda) = <Theta, msign(G + lambda Theta)> = 0.
    """
    G32 = G.astype(jnp.float32)
    Theta32 = Theta.astype(jnp.float32)

    def h(lmbd: jnp.ndarray) -> jnp.ndarray:
        Phi = msign_newton_schulz(G32 + lmbd * Theta32, steps=msign_steps, coefficient_type="polar_express", eps=eps)
        return jnp.vdot(Theta32, Phi)

    h0 = h(jnp.array(0.0, jnp.float32))

    # If already basically zero, return 0
    def return_zero(_):
        return jnp.array(0.0, jnp.float32)

    def do_solve(_):
        m, n = G32.shape
        min_dim = jnp.minimum(m, n).astype(jnp.float32)

        limit = 4.0 * jnp.sqrt(min_dim) + 1.0
        step0 = jnp.array(1.0, jnp.float32)

        # Case A: h(0) < 0 => root is to the right
        def bracket_right():
            lo = jnp.array(0.0, jnp.float32)
            hi = step0
            hlo = h0
            hhi = h(hi)

            def cond(state):
                hi, hhi = state
                return (hhi < 0.0) & (hi < limit)

            def body(state):
                hi, _hhi = state
                hi = hi * 2.0
                return (hi, h(hi))

            hi, hhi = jax.lax.while_loop(cond, body, (hi, hhi))
            return lo, hi, hlo, hhi

        # Case B: h(0) > 0 => root is to the left
        def bracket_left():
            hi = jnp.array(0.0, jnp.float32)
            lo = -step0
            hhi = h0
            hlo = h(lo)

            def cond(state):
                lo, hlo = state
                return (hlo > 0.0) & ((-lo) < limit)

            def body(state):
                lo, _hlo = state
                lo = lo * 2.0
                return (lo, h(lo))

            lo, hlo = jax.lax.while_loop(cond, body, (lo, hlo))
            return lo, hi, hlo, hhi

        lo, hi, hlo, hhi = jax.lax.cond(h0 < 0.0, lambda _: bracket_right(), lambda _: bracket_left(), operand=None)

        # If bracketing failed, fall back to lambda=0
        bracket_ok = (hlo <= 0.0) & (hhi >= 0.0)

        def bisect(_):
            lo2, hi2 = lo, hi
            for _ in range(max_iter):
                mid = 0.5 * (lo2 + hi2)
                hmid = h(mid)
                lo2 = jnp.where(hmid < 0.0, mid, lo2)
                hi2 = jnp.where(hmid < 0.0, hi2, mid)
            return 0.5 * (lo2 + hi2)

        return jax.lax.cond(bracket_ok, bisect, return_zero, operand=None)

    return jax.lax.cond(jnp.abs(h0) <= tol, return_zero, do_solve, operand=None)


# -------------------------
# Optimizer state
# -------------------------
class ScaleBySSOState(NamedTuple):
    """State for SSO/MuonSphere algorithms."""

    momentum_buffer: optax.Updates


# -------------------------
# SSO Config
# -------------------------
@OptimizerConfig.register_subclass("sso")
@dataclass(frozen=True)
class SSOConfig(OptimizerConfig):
    """
    SSO (Spectral Sphere Optimizer) configuration.

    Retracts 2D weight matrices to spectral sphere with radius R = radius_scaler * sqrt(d_out/d_in).
    Solves for lambda to enforce tangent constraint: <Theta, msign(G + lambda*Theta)> = 0.
    """

    adam_lr: float = 6e-4  # Adam LR for non-2D parameters
    momentum: float = 0.9
    nesterov: bool = True
    msign_steps: int = 8
    solver_tol: float = 1e-8
    solver_max_iter: int = 20
    power_iter_steps: int = 20
    radius_scaler: float = 1.0
    eps: float = 1e-12
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    max_grad_norm: float = 1.0

    def build(self, num_train_steps):
        """Creates the optimizer."""
        learning_rate_schedule = self.lr_scheduler(num_train_steps)
        adam_lr_schedule = self.lr_scheduler(num_train_steps, override_lr=self.adam_lr)

        def optimizer(learning_rate, adam_lr):
            def sso_transform():
                components = []
                if self.max_grad_norm:
                    components.append(optax.clip_by_global_norm(self.max_grad_norm))
                components.append(
                    scale_with_sso(
                        learning_rate=learning_rate,
                        momentum=self.momentum,
                        nesterov=self.nesterov,
                        msign_steps=self.msign_steps,
                        solver_tol=self.solver_tol,
                        solver_max_iter=self.solver_max_iter,
                        power_iter_steps=self.power_iter_steps,
                        radius_scaler=self.radius_scaler,
                        eps=self.eps,
                        use_lambda_solver=True,
                    )
                )
                return optax.chain(*components)

            def adam_transform():
                components = []
                if self.max_grad_norm:
                    components.append(optax.clip_by_global_norm(self.max_grad_norm))
                components.append(optax.scale_by_adam(self.beta1, self.beta2, self.epsilon))
                components.append(optax.scale(-adam_lr))
                return optax.chain(*components)

            transformations = {
                "sso": sso_transform(),
                "adam": adam_transform(),
            }

            return optax.multi_transform(transformations, self.create_mask)

        return optax.inject_hyperparams(optimizer)(learning_rate=learning_rate_schedule, adam_lr=adam_lr_schedule)

    def create_mask(self, params):
        """Creates a mask that labels parameters as 'sso' or 'adam'."""
        paths = leaf_key_paths(params)

        def mask_fn(param, path):
            path_str = ".".join(path) if isinstance(path, (list, tuple)) else str(path)
            if "Embedding" in path_str or "lm_head" in path_str:
                return "adam"
            elif isinstance(param, Linear):
                # sso for linear layer weights, adam for biases
                return dataclasses.replace(param, weight="sso", bias="adam" if param.bias is not None else None)
            else:
                return "adam"

        return haliax.tree_util.tree_map(mask_fn, params, paths, is_leaf=lambda x: isinstance(x, Linear))


# -------------------------
# MuonSphere Config
# -------------------------
@OptimizerConfig.register_subclass("muon_sphere")
@dataclass(frozen=True)
class MuonSphereConfig(OptimizerConfig):
    """
    MuonSphere optimizer configuration.

    Same as SSO but with lambda=0 (no tangent constraint solver).
    Simpler and faster than full SSO.
    """

    adam_lr: float = 6e-4
    momentum: float = 0.9
    nesterov: bool = True
    msign_steps: int = 8
    power_iter_steps: int = 20
    radius_scaler: float = 1.0
    eps: float = 1e-12
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    max_grad_norm: float = 1.0

    def build(self, num_train_steps):
        """Creates the optimizer."""
        learning_rate_schedule = self.lr_scheduler(num_train_steps)
        adam_lr_schedule = self.lr_scheduler(num_train_steps, override_lr=self.adam_lr)

        def optimizer(learning_rate, adam_lr):
            def muon_sphere_transform():
                components = []
                if self.max_grad_norm:
                    components.append(optax.clip_by_global_norm(self.max_grad_norm))
                components.append(
                    scale_with_sso(
                        learning_rate=learning_rate,
                        momentum=self.momentum,
                        nesterov=self.nesterov,
                        msign_steps=self.msign_steps,
                        solver_tol=1e-8,  # not used when use_lambda_solver=False
                        solver_max_iter=20,  # not used
                        power_iter_steps=self.power_iter_steps,
                        radius_scaler=self.radius_scaler,
                        eps=self.eps,
                        use_lambda_solver=False,  # MuonSphere uses lambda=0
                    )
                )
                return optax.chain(*components)

            def adam_transform():
                components = []
                if self.max_grad_norm:
                    components.append(optax.clip_by_global_norm(self.max_grad_norm))
                components.append(optax.scale_by_adam(self.beta1, self.beta2, self.epsilon))
                components.append(optax.scale(-adam_lr))
                return optax.chain(*components)

            transformations = {
                "muon_sphere": muon_sphere_transform(),
                "adam": adam_transform(),
            }

            return optax.multi_transform(transformations, self.create_mask)

        return optax.inject_hyperparams(optimizer)(learning_rate=learning_rate_schedule, adam_lr=adam_lr_schedule)

    def create_mask(self, params):
        """Creates a mask that labels parameters as 'muon_sphere' or 'adam'."""
        paths = leaf_key_paths(params)

        def mask_fn(param, path):
            path_str = ".".join(path) if isinstance(path, (list, tuple)) else str(path)
            if "Embedding" in path_str or "lm_head" in path_str:
                return "adam"
            elif isinstance(param, Linear):
                return dataclasses.replace(
                    param, weight="muon_sphere", bias="adam" if param.bias is not None else None
                )
            else:
                return "adam"

        return haliax.tree_util.tree_map(mask_fn, params, paths, is_leaf=lambda x: isinstance(x, Linear))


# -------------------------
# Core SSO/MuonSphere update function
# -------------------------
def scale_with_sso(
    learning_rate: float,
    momentum: float = 0.9,
    nesterov: bool = True,
    msign_steps: int = 8,
    solver_tol: float = 1e-8,
    solver_max_iter: int = 20,
    power_iter_steps: int = 20,
    radius_scaler: float = 1.0,
    eps: float = 1e-12,
    use_lambda_solver: bool = True,
):
    """
    Optax transformation for SSO/MuonSphere.

    If use_lambda_solver=True: full SSO with lambda solver
    If use_lambda_solver=False: MuonSphere (lambda=0)
    """
    steps = int(msign_steps)
    power_steps = int(power_iter_steps)
    max_iter = int(solver_max_iter)

    def init_fn(params):
        momentum_buffer = otu.tree_zeros_like(params)
        return ScaleBySSOState(momentum_buffer=momentum_buffer)

    def update_fn(updates, state, params=None):
        if params is None:
            raise ValueError("Parameters are required for SSO/MuonSphere.")

        # Momentum accumulation
        buf = state.momentum_buffer
        buf = jax.tree.map(
            lambda m, g: None if g is None else momentum * m + g,
            buf,
            updates,
            is_leaf=lambda x: x is None,
        )

        # Optional Nesterov
        if nesterov:
            eff_updates = jax.tree.map(
                lambda m, g: None if g is None else momentum * m + g,
                buf,
                updates,
                is_leaf=lambda x: x is None,
            )
        else:
            eff_updates = buf

        # Flatten params and updates for pairing
        flat_params = flatten_linear_layers(params)
        flat_eff_updates = flatten_linear_layers(eff_updates)

        # Transform linear layers
        def transform_linear_layer(update_layer: haliax.nn.Linear, param_layer: haliax.nn.Linear):
            if update_layer.weight is None or param_layer.weight is None:
                return update_layer

            W = param_layer.weight.array
            G = update_layer.weight.array

            # scan_aware_tree_map will vmap over the layer dimension if present,
            # so W and G should be 2D here
            assert W.ndim == 2, f"Expected 2D weight array, got {W.ndim}D"
            assert G.ndim == 2, f"Expected 2D gradient array, got {G.ndim}D"

            d_out, d_in = W.shape
            R = spectral_mup_radius(d_out, d_in, radius_scaler)

            # Normalize gradient by Frobenius norm
            G32 = G.astype(jnp.float32)
            G_norm = G32 / _safe_fro_norm(G32, eps=eps)

            # Power iteration to get top singular vectors
            sigma, u, v = power_iteration_top_singular(W, steps=power_steps, eps=eps)

            # Theta = u v^T
            Theta = jnp.outer(u, v)

            # Retract to spectral sphere
            sigma = jnp.maximum(sigma, eps)
            W32 = W.astype(jnp.float32)
            W_retr = W32 * (R / sigma)

            # Solve for lambda (or use 0 for MuonSphere)
            if use_lambda_solver:
                lmbd = solve_lambda_bisection(
                    G=G_norm,
                    Theta=Theta,
                    tol=solver_tol,
                    max_iter=max_iter,
                    msign_steps=steps,
                    eps=eps,
                )
            else:
                lmbd = jnp.array(0.0, jnp.float32)

            # Compute Phi = msign(G_norm + lambda * Theta)
            Phi = msign_newton_schulz(G_norm + lmbd * Theta, steps=steps, coefficient_type="polar_express", eps=eps)

            # Update: W <- W_retr - lr * R * Phi
            lr = jnp.array(learning_rate, jnp.float32)
            W_next = W_retr - lr * R * Phi

            # Compute the actual update (delta)
            delta = W_next - W

            updated_weight = dataclasses.replace(update_layer.weight, array=delta.astype(update_layer.weight.array.dtype))
            return dataclasses.replace(update_layer, weight=updated_weight)

        # Apply transformation using scan_aware_tree_map with both updates and params
        transformed_flat = scan_aware_tree_map(
            lambda u, p: transform_linear_layer(u, p) if isinstance(u, Linear) and isinstance(p, Linear) else u,
            flat_eff_updates,
            flat_params,
            is_leaf=lambda x: isinstance(x, Linear),
        )

        # Unflatten back to original structure
        new_updates = unflatten_linear_layers(eff_updates, transformed_flat)

        return new_updates, ScaleBySSOState(momentum_buffer=buf)

    return optax.GradientTransformation(init_fn, update_fn)
