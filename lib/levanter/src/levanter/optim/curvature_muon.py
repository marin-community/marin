# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Curvature-corrected Muon: Nesterov Muon with a one-sided Shampoo curvature penalty, on the
hyperball (MuonH) base.

Per matrix ``W`` (oriented so rows ≥ cols), the orthogonal update direction approximately solves

    max_{XᵀX=I}  ⟨N_t, X⟩  −  (λ/2)·tr(Xᵀ P_t X),     P_t = EMA(G_t G_tᵀ)   (left/output curvature)

via the inner Newton–Schulz fixed point

    X⁽⁰⁾   = msign(N_t)
    X⁽ᵏ⁺¹⁾ = msign( N_t + λ·(√e_max·I − P_t/√e_max)·X⁽ᵏ⁾ ),   e_max = λ_max(P_t)  (power iteration)

``msign`` is the usual Muon Newton–Schulz orthogonalization. The operator ``√e_max·I − P_t/√e_max`` is
**PSD** (eigenvalues ``(e_max − p_i)/√e_max ≥ 0``: ``√e_max`` in flat directions, ``0`` in the top-curvature
one), so the fixed point is a contraction — **stable at any λ, no α shift needed**. And ``P/√e_max`` has
gradient units (``P ~ G²`` ⟹ ``P/√e_max ~ G``), matching ``N``, so **λ is dimensionless** (no ``‖N‖``
factor) and its LR coupling is unambiguous. Curvature enters only through matmuls + a power iteration —
no eigendecomposition. ``X_t`` is then mapped through the MuonH hyperball (scale-invariant,
constant-Frobenius-norm) reparam instead of ``√(Out/In)`` scaling.

λ = 0 ⟹ exactly MuonH (Nesterov Muon + hyperball). K = 1 is a single curvature-corrected polishing step.
"""

import dataclasses
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax
from optax import tree_utils as otu

import haliax

from levanter.optim.adamh import scale_by_adamh
from levanter.optim.config import OptimizerConfig
from levanter.optim.util import (
    CoefficientType,
    flatten_linear_layers,
    label_linear_like_module,
    unflatten_linear_layers,
    zeropower_via_newtonschulz5,
)
from levanter.utils.jax_utils import leaf_key_paths


@OptimizerConfig.register_subclass("curvature_muon")
@dataclass(frozen=True)
class CurvatureMuonConfig(OptimizerConfig):
    """Curvature-corrected Muon on the hyperball base (cf. MuonH). λ=0 recovers MuonH exactly."""

    adam_lr: float = 6e-4
    momentum: float = 0.95  # μ
    nesterov: bool = True
    backend_steps: int = 5  # Newton-Schulz steps for msign
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    muon_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    coefficient_type: CoefficientType = "quintic"
    # --- curvature knobs ---
    # Inner fixed point: X = msign( N + λ·(√e_max·I − P/√e_max)·X ), P = EMA(G Gᵀ), e_max = λ_max(P).
    # The operator √e_max·I − P/√e_max is PSD (eigenvalues (e_max−p_i)/√e_max ≥ 0; 0 in the top-curvature
    # direction, √e_max in flat ones), so the iteration is a contraction — stable at any λ (no α needed).
    # P/√e_max has gradient units (P~G², so P/√e_max~G), matching N, so λ is dimensionless and its LR
    # coupling is clean (no ‖N‖ factor). λ=0 ⟹ MuonH.
    curvature_beta: float = 0.95  # ρ, EMA decay of P = EMA(G Gᵀ)
    curvature_lambda: float = 0.0  # λ, curvature strength (0 ⟹ MuonH)
    inner_steps: int = 1  # K, inner fixed-point iterations
    power_iters: int = 8  # power-iteration steps for the e_max(P) estimate (warm-started from stored q)
    # If set, the curvature strength tracks the LR schedule: lambda_t = curvature_lambda * lr_t / peak_lr.
    # So curvature is strongest at peak LR and fades during warmup / cosine decay (curvature_lambda = peak).
    lambda_tracks_lr: bool = False

    def build(self, num_train_steps):
        learning_rate_schedule = self.lr_scheduler(num_train_steps)
        adam_lr_schedule = self.lr_scheduler(num_train_steps, override_lr=self.adam_lr)

        def optimizer(learning_rate, adam_lr):
            def curvature_muon_transform():
                components = []
                if self.max_grad_norm:
                    components.append(optax.clip_by_global_norm(self.max_grad_norm))
                components.append(
                    scale_with_curvature_muon(
                        self.momentum,
                        self.nesterov,
                        self.backend_steps,
                        self.muon_epsilon,
                        learning_rate,
                        self.coefficient_type,
                        self.curvature_beta,
                        self.curvature_lambda,
                        self.inner_steps,
                        self.power_iters,
                        self.learning_rate,
                        self.lambda_tracks_lr,
                    )
                )
                return optax.chain(*components)

            def adamh_transform():
                components = []
                if self.max_grad_norm:
                    components.append(optax.clip_by_global_norm(self.max_grad_norm))
                components.append(scale_by_adamh(self.beta1, self.beta2, self.epsilon, learning_rate))
                return optax.chain(*components)

            def adam_transform():
                components = []
                if self.max_grad_norm:
                    components.append(optax.clip_by_global_norm(self.max_grad_norm))
                components.append(optax.scale_by_adam(self.beta1, self.beta2, self.epsilon))
                components.append(optax.scale(-adam_lr))
                return optax.chain(*components)

            transformations = {
                "curvature_muon": curvature_muon_transform(),
                "adamh": adamh_transform(),
                "adam": adam_transform(),
            }
            return optax.multi_transform(transformations, self.create_mask)

        return optax.inject_hyperparams(optimizer)(learning_rate=learning_rate_schedule, adam_lr=adam_lr_schedule)

    def create_mask(self, params):
        """Embeddings → adam, lm_head → adamh, Linear weights → curvature_muon, else → adam (matches MuonH)."""
        paths = leaf_key_paths(params)

        def mask_fn(param, path):
            path_str = ".".join(path) if isinstance(path, (list, tuple)) else str(path)
            if "Embedding" in path_str:
                return "adam"
            elif "lm_head" in path_str:
                return "adamh"
            elif isinstance(param, haliax.nn.Linear):
                return label_linear_like_module(param, weight_label="curvature_muon", bias_label="adam")
            else:
                return "adam"

        return haliax.tree_util.tree_map(mask_fn, params, paths, is_leaf=lambda x: isinstance(x, haliax.nn.Linear))


class ScaleByCurvatureMuonState(NamedTuple):
    momentum_buffer: optax.Updates  # B, full param tree
    curvature: optax.Updates  # P, flattened-linear tree of [..., M, M] (M = max(out, in))
    power_vec: optax.Updates  # q, flattened-linear tree of [..., M]


_EMAX_MARGIN = 1.05  # inflate the (lower-bound) Rayleigh e_max estimate so the operator stays strictly PSD


def _curv_direction_2d(g, n, p, q, *, rho, lam_static, lam_coef, steps, eps, ctype, inner_steps, power_iters):
    """One matrix. g, n: [out, in] (gradient, Nesterov signal). p: [M, M], q: [M], M = max(out, in).

    Inner fixed point X = msign( N + λ·(√e_max·I − P/√e_max)·X ). lam_static (python float) gates on/off;
    lam_coef is the actual coefficient (a traced scalar when it tracks the LR schedule).
    e_max(P) via a warm-started multi-step power iteration. Returns (new_p, new_q, x) in [out, in] orient.
    """
    out, inn = g.shape
    transpose = out < inn
    g_t = g.T if transpose else g  # [M, N], M ≥ N
    n_t = n.T if transpose else n

    new_p = rho * p + (1.0 - rho) * (g_t @ g_t.T)  # [M, M]
    new_q = q
    for _ in range(int(power_iters)):
        pq = new_p @ new_q
        new_q = pq / (jnp.linalg.norm(pq) + eps)
    # Rayleigh quotient lower-bounds λ_max; inflate by the margin so √e_max·I − P/√e_max is strictly PSD.
    emax = jnp.dot(new_q, new_p @ new_q) * _EMAX_MARGIN
    se = jnp.sqrt(emax) + eps

    x = zeropower_via_newtonschulz5(n_t, steps=steps, eps=eps, coefficient_type=ctype)
    if lam_static > 0.0:
        eye = jnp.eye(new_p.shape[0], dtype=new_p.dtype)
        operator = lam_coef * (se * eye - new_p / se)  # PSD: λ(√e_max I − P/√e_max)
        for _ in range(int(inner_steps)):
            x = zeropower_via_newtonschulz5(n_t + operator @ x, steps=steps, eps=eps, coefficient_type=ctype)

    x_out = x.T if transpose else x
    return new_p, new_q, x_out


def scale_with_curvature_muon(
    momentum=0.95,
    nesterov=True,
    steps=5,
    muon_eps=1e-8,
    learning_rate=0.02,
    coefficient_type="quintic",
    curvature_beta=0.95,
    curvature_lambda=0.0,
    inner_steps=1,
    power_iters=8,
    peak_lr=0.02,
    lambda_tracks_lr=False,
):
    steps = int(steps)
    mu = float(momentum)
    rho = float(curvature_beta)
    lam = float(curvature_lambda)  # static peak strength (also the on/off switch)
    peak_lr = float(peak_lr)
    tracks_lr = bool(lambda_tracks_lr)

    def _is_linear_weight(layer):
        return isinstance(layer, haliax.nn.Linear) and isinstance(layer.weight, haliax.NamedArray)

    def init_fn(params):
        momentum_buffer = otu.tree_zeros_like(params)
        flat = flatten_linear_layers(params)

        def to_p(layer):
            if not _is_linear_weight(layer):
                return layer
            a = layer.weight.array
            m = max(a.shape[-2], a.shape[-1])
            return jnp.broadcast_to(muon_eps * jnp.eye(m, dtype=a.dtype), a.shape[:-2] + (m, m))

        def to_q(layer):
            if not _is_linear_weight(layer):
                return layer
            a = layer.weight.array
            m = max(a.shape[-2], a.shape[-1])
            return jnp.broadcast_to(jnp.ones(m, dtype=a.dtype) / jnp.sqrt(m), a.shape[:-2] + (m,))

        curvature = haliax.tree_util.tree_map(to_p, flat, is_leaf=_is_linear_weight)
        power_vec = haliax.tree_util.tree_map(to_q, flat, is_leaf=_is_linear_weight)
        return ScaleByCurvatureMuonState(momentum_buffer=momentum_buffer, curvature=curvature, power_vec=power_vec)

    def update_fn(updates, state, params=None):
        # Momentum buffer + scale-preserving Nesterov signal N = (1-μ)G + μB.
        buf = jax.tree.map(
            lambda m, g: None if g is None else mu * m + (1.0 - mu) * g,
            state.momentum_buffer,
            updates,
            is_leaf=lambda x: x is None,
        )
        if nesterov:
            signal = jax.tree.map(
                lambda b, g: None if g is None else (1.0 - mu) * g + mu * b,
                buf,
                updates,
                is_leaf=lambda x: x is None,
            )
        else:
            signal = buf

        flat_grad = flatten_linear_layers(updates)
        flat_signal = flatten_linear_layers(signal)

        # Effective coefficient: tracks the LR schedule when requested (traced scalar), else the static peak.
        lam_coef = lam * (learning_rate / peak_lr) if tracks_lr else lam

        def per_layer(g_layer, n_layer, p, q):
            if not _is_linear_weight(g_layer):
                return n_layer  # passthrough (these are routed to adam/adamh anyway)
            g = g_layer.weight.array
            n = n_layer.weight.array
            fn = lambda gg, nn, pp, qq: _curv_direction_2d(
                gg,
                nn,
                pp,
                qq,
                rho=rho,
                lam_static=lam,
                lam_coef=lam_coef,
                steps=steps,
                eps=muon_eps,
                ctype=coefficient_type,
                inner_steps=inner_steps,
                power_iters=power_iters,
            )
            new_p, new_q, x = jax.vmap(fn)(g, n, p, q) if g.ndim == 3 else fn(g, n, p, q)
            new_w = dataclasses.replace(n_layer.weight, array=x)
            return (dataclasses.replace(n_layer, weight=new_w), new_p, new_q)  # type: ignore

        combined = haliax.tree_util.tree_map(
            per_layer, flat_grad, flat_signal, state.curvature, state.power_vec, is_leaf=_is_linear_weight
        )

        is_triple = lambda c: isinstance(c, tuple) and len(c) == 3
        flat_dir = jax.tree.map(lambda c: c[0] if is_triple(c) else c, combined, is_leaf=is_triple)
        new_curvature = jax.tree.map(lambda c: c[1] if is_triple(c) else c, combined, is_leaf=is_triple)
        new_power = jax.tree.map(lambda c: c[2] if is_triple(c) else c, combined, is_leaf=is_triple)
        direction = unflatten_linear_layers(signal, flat_dir)

        # Hyperball: constant-Frobenius-norm scale-invariant update (identical to MuonH).
        def scale_invariant_update(p, u):
            if p is None:
                return None
            if p.ndim == 2:
                new_p = p - learning_rate * u * jnp.linalg.norm(p) / jnp.maximum(jnp.linalg.norm(u), 1e-10)
                return new_p / jnp.linalg.norm(new_p) * jnp.linalg.norm(p) - p
            else:
                axes = tuple(range(1, p.ndim))
                p_norm = jnp.sqrt(jnp.sum(jnp.square(p), axis=axes, keepdims=True))
                u_norm = jnp.sqrt(jnp.sum(jnp.square(u), axis=axes, keepdims=True))
                new_p = p - learning_rate * u * p_norm / jnp.maximum(u_norm, 1e-10)
                new_p_norm = jnp.sqrt(jnp.sum(jnp.square(new_p), axis=axes, keepdims=True))
                return new_p / jnp.maximum(new_p_norm, 1e-10) * p_norm - p

        hyperball_updates = jax.tree_util.tree_map(
            scale_invariant_update, params, direction, is_leaf=lambda x: x is None
        )
        return hyperball_updates, ScaleByCurvatureMuonState(
            momentum_buffer=buf, curvature=new_curvature, power_vec=new_power
        )

    return optax.GradientTransformation(init_fn, update_fn)
