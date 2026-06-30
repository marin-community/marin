# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Left-preconditioned Muon (Ali Jadbabaie's "idea 4"), Mudam-style coupled-Newton solver.

    U = ρ · H^{-1/2} · polar( H^{-1/2} · M ) ,   H = EMA(G Gᵀ)   (output/left side)

``H^{-1/2}`` is the **saturating coarse inverse-sqrt** ``q_k`` (``solver="qk"``): eigh, then the k-step
Muon-coeff NS scalar map applied to the (λ_max-normalized) eigenvalues. This reproduces Mudam's
coupled-NS operator exactly (proven by spectral mapping) and beats the exact ``w^{-1/2}``
(``solver="eigh"``) because it *caps* the amplification of the smallest/noisiest eigen-directions
rather than blowing them up — the beneficial regularization that makes Mudam (1.165) beat exact
whitening (1.185).

``outer_precond=False`` drops the outer factor, giving ``polar(H^{-1/2} M)`` — provably identical to
Mudam-with-another_muon. ``outer_precond=True`` is the full idea-4 update; ``outer_msign`` re-
orthogonalizes after it; ``normalize_fro`` rescales to plain-Muon's Frobenius norm. ``H = I`` recovers
plain Muon. Gradient-only; no activation capture.
"""

import dataclasses
from dataclasses import dataclass
from functools import partial
from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
import optax
from optax import tree_utils as otu

import haliax

from levanter.optim.config import OptimizerConfig
from levanter.optim.util import (
    flatten_linear_layers,
    label_linear_like_module,
    unflatten_linear_layers,
    zeropower_via_newtonschulz5,
)
from levanter.utils.jax_utils import leaf_key_paths


@OptimizerConfig.register_subclass("left_precond_muon")
@dataclass(frozen=True)
class LeftPrecondMuonConfig(OptimizerConfig):
    """Left-preconditioned Muon (idea 4): U = ρ H^{-1/2} polar(H^{-1/2} M), H = EMA(G Gᵀ)."""

    # --- the idea-4 knobs ---
    h_beta: float = 0.95  # EMA decay for the gradient second moment H = EMA(G Gᵀ)
    solver: str = "qk"  # "qk": saturating coarse NS-polynomial inverse-sqrt (= Mudam); "eigh": exact w^{-1/2}
    qk_steps: int = 5  # k in the q_k saturating polynomial (Muon-coeff scalar map on eigenvalues)
    outer_precond: bool = True  # True: U = H^{-1/2} polar(H^{-1/2} M) (full); False: U = polar(H^{-1/2} M)
    outer_msign: bool = False  # re-orthogonalize after the outer factor: U = polar(H^{-1/2} polar(H^{-1/2} M))
    normalize_fro: bool = False  # rescale U to plain-Muon's Frobenius norm (polar(M))
    ns_steps: int = 5  # Newton-Schulz steps for the polar factor

    # --- Muon-shared knobs ---
    lr: float = 0.02
    adam_lr: float = 6e-4
    momentum: float = 0.95
    nesterov: bool = True
    weight_decay: float = 0.0
    adam_weight_decay: Optional[float] = None
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    matrix_epsilon: float = 1e-7  # NS / normalization floor
    max_grad_norm: float = 1.0
    use_kimi_scaling: bool = False

    def build(self, num_train_steps):
        learning_rate_schedule = self.lr_scheduler(num_train_steps)
        adam_lr_schedule = self.lr_scheduler(num_train_steps, override_lr=self.adam_lr)

        def optimizer(learning_rate, adam_lr):
            def matrix_transform():
                components = [
                    scale_with_left_precond_muon(
                        self.momentum,
                        self.nesterov,
                        self.h_beta,
                        self.solver,
                        self.qk_steps,
                        self.ns_steps,
                        self.matrix_epsilon,
                        self.use_kimi_scaling,
                        self.outer_precond,
                        self.outer_msign,
                        self.normalize_fro,
                    )
                ]
                if self.weight_decay > 0:
                    components.append(optax.add_decayed_weights(self.weight_decay, self.build_weight_decay_mask()))
                components.append(optax.scale(-learning_rate))
                return optax.chain(*components)

            def adamw_transform():
                components = []
                if self.max_grad_norm:
                    components.append(optax.clip_by_global_norm(self.max_grad_norm))
                components.append(optax.scale_by_adam(self.beta1, self.beta2, self.epsilon))
                adam_weight_decay = self.adam_weight_decay if self.adam_weight_decay is not None else self.weight_decay
                if adam_weight_decay > 0:
                    components.append(optax.add_decayed_weights(adam_weight_decay, self.build_weight_decay_mask()))
                components.append(optax.scale(-adam_lr))
                return optax.chain(*components)

            transformations = {"left_precond_muon": matrix_transform(), "adamw": adamw_transform()}
            return optax.multi_transform(
                transformations, partial(self.create_mask, use_kimi_scaling=self.use_kimi_scaling)
            )

        return optax.inject_hyperparams(optimizer)(learning_rate=learning_rate_schedule, adam_lr=adam_lr_schedule)

    def create_mask(self, params, use_kimi_scaling=True):
        """Label matrix (Linear) weights 'left_precond_muon', everything else 'adamw' (matches Muon)."""
        paths = leaf_key_paths(params)

        def mask_fn(param, path):
            path_str = ".".join(path) if isinstance(path, (list, tuple)) else str(path)
            if "Embedding" in path_str or "lm_head" in path_str:
                return "adamw"
            elif isinstance(param, haliax.nn.Linear):
                assert param._out_first or use_kimi_scaling
                return label_linear_like_module(param, weight_label="left_precond_muon", bias_label="adamw")
            else:
                return "adamw"

        return haliax.tree_util.tree_map(mask_fn, params, paths, is_leaf=lambda x: isinstance(x, haliax.nn.Linear))


class ScaleByLeftPrecondMuonState(NamedTuple):
    momentum_buffer: optax.Updates
    h_ema: optax.Updates  # per-matrix gradient second moment EMA, [..., d_out, d_out] (raw arrays)


_MUON_NS_COEFFS = (3.4445, -4.7750, 2.0315)


def _qk_inv_sqrt(h, steps, eps):
    """Saturating coarse inverse-sqrt: eigh + the k-step Muon-coeff NS scalar map q_k(w).

    Reproduces Mudam's coupled-NS operator exactly (proven by spectral mapping). Normalize the
    eigenvalues by λ_max so the spectrum is in (0,1] (the NS convergence basin). As k→∞, q_k(w) →
    w^{-1/2}; at finite k it SATURATES the amplification of the smallest/noisiest eigen-directions
    instead of blowing them up — that cap is the beneficial regularization over the exact inverse.
    Reduces to plain Muon when H ∝ I (all eigenvalues equal → uniform scaling, washed by the polar).
    """
    h = h.astype(jnp.float32)
    h = 0.5 * (h + h.T)
    w, u_vec = jnp.linalg.eigh(h)
    w = jnp.maximum(w, 0.0)
    lam = w / (jnp.max(w) + eps)  # spectrum in (0, 1]
    a, b, c = _MUON_NS_COEFFS
    al = lam
    mult = jnp.ones_like(lam)
    for _ in range(steps):
        f = a + b * al + c * al * al
        mult = mult * f
        al = f * f * al
    return (u_vec * mult) @ u_vec.T


def _exact_inv_sqrt(h, eps):
    """Exact mean-normalized H^{-1/2} via eigh (the un-regularized 'inner_muon' whitening)."""
    h = h.astype(jnp.float32)
    h = 0.5 * (h + h.T)
    w, u_vec = jnp.linalg.eigh(h)
    w = jnp.maximum(w, 0.0)
    lam = w / (jnp.mean(w) + eps)
    inv = jnp.where(lam > 1e-12, lam**-0.5, 0.0)
    return (u_vec * inv) @ u_vec.T


def _left_precond_matrix(
    m, h, *, solver, qk_steps, ns_steps, eps, use_kimi_scaling, outer_precond, outer_msign, normalize_fro
):
    """U = H^{-1/2} polar(H^{-1/2} M) for one weight M (d_out × d_in), H (d_out × d_out).

    outer_precond=False drops the outer H^{-1/2} (gives polar(H^{-1/2} M)).
    outer_msign re-orthogonalizes after the outer factor: U = polar(H^{-1/2} polar(H^{-1/2} M)).
    normalize_fro rescales U to plain-Muon's Frobenius norm (polar(M)).
    """
    orig_dtype = m.dtype
    x = m.astype(jnp.float32)
    hih = _qk_inv_sqrt(h, qk_steps, eps) if solver == "qk" else _exact_inv_sqrt(h, eps)  # (out, out)
    whitened = hih @ x  # H^{-1/2} M
    ortho = zeropower_via_newtonschulz5(whitened, steps=ns_steps, eps=eps, coefficient_type="quintic")
    u = (hih @ ortho) if outer_precond else ortho
    if outer_msign:
        u = zeropower_via_newtonschulz5(u, steps=ns_steps, eps=eps, coefficient_type="quintic")

    if normalize_fro:
        muon = zeropower_via_newtonschulz5(x, steps=ns_steps, eps=eps, coefficient_type="quintic")
        u = u * (jnp.linalg.norm(muon) / (jnp.linalg.norm(u) + eps))

    if not use_kimi_scaling:
        scale = jnp.sqrt(jnp.maximum(1.0, u.shape[0] / u.shape[1]))  # sqrt(Out/In)
    else:
        scale = 0.2 * jnp.sqrt(jnp.maximum(u.shape[0], u.shape[1]))
    return (u * scale).astype(orig_dtype)


def scale_with_left_precond_muon(
    momentum=0.95,
    nesterov=True,
    h_beta=0.95,
    solver="qk",
    qk_steps=5,
    ns_steps=5,
    eps=1e-7,
    use_kimi_scaling=False,
    outer_precond=True,
    outer_msign=False,
    normalize_fro=False,
):
    momentum = float(momentum)
    h_beta = float(h_beta)

    def _gram(weight_array):
        # weight_array: [..., out, in] → left Gram [..., out, out] = G Gᵀ (sum over in)
        x = weight_array.astype(jnp.float32)
        return jnp.einsum("...oi,...pi->...op", x, x)

    def init_fn(params):
        flat = flatten_linear_layers(params)

        def to_h(layer):
            if isinstance(layer, haliax.nn.Linear) and isinstance(layer.weight, haliax.NamedArray):
                a = layer.weight.array
                return jnp.zeros(a.shape[:-1] + (a.shape[-2],), jnp.float32)  # [..., out, out]
            return layer  # passthrough non-Linear leaves so h_ema matches the flattened structure

        h_ema = haliax.tree_util.tree_map(to_h, flat, is_leaf=lambda x: isinstance(x, haliax.nn.Linear))
        return ScaleByLeftPrecondMuonState(momentum_buffer=otu.tree_zeros_like(params), h_ema=h_ema)

    def update_fn(updates, state, params=None):
        buf = jax.tree.map(
            lambda m, g: None if g is None else momentum * m + g,
            state.momentum_buffer,
            updates,
            is_leaf=lambda x: x is None,
        )
        direction = (
            jax.tree.map(
                lambda m, g: None if g is None else momentum * m + g, buf, updates, is_leaf=lambda x: x is None
            )
            if nesterov
            else buf
        )

        # H = EMA(G Gᵀ) from the RAW gradient; newest term = current gradient.
        flat_grad = flatten_linear_layers(updates)

        def new_h(layer, h):
            if not (isinstance(layer, haliax.nn.Linear) and isinstance(layer.weight, haliax.NamedArray)):
                return h
            return h_beta * h + (1.0 - h_beta) * _gram(layer.weight.array)

        h_ema = haliax.tree_util.tree_map(
            new_h, flat_grad, state.h_ema, is_leaf=lambda x: isinstance(x, haliax.nn.Linear)
        )

        flat_dir = flatten_linear_layers(direction)

        def precond(layer, h):
            if not (isinstance(layer, haliax.nn.Linear) and isinstance(layer.weight, haliax.NamedArray)):
                return layer
            w = layer.weight
            arr = w.array
            fn = partial(
                _left_precond_matrix,
                solver=solver,
                qk_steps=qk_steps,
                ns_steps=ns_steps,
                eps=eps,
                use_kimi_scaling=use_kimi_scaling,
                outer_precond=outer_precond,
                outer_msign=outer_msign,
                normalize_fro=normalize_fro,
            )
            new_arr = jax.vmap(fn)(arr, h) if arr.ndim == 3 else fn(arr, h)
            return dataclasses.replace(layer, weight=dataclasses.replace(w, array=new_arr))  # type: ignore

        flat_out = haliax.tree_util.tree_map(
            precond, flat_dir, h_ema, is_leaf=lambda x: isinstance(x, haliax.nn.Linear)
        )
        new_updates = unflatten_linear_layers(direction, flat_out)
        return new_updates, ScaleByLeftPrecondMuonState(momentum_buffer=buf, h_ema=h_ema)

    return optax.GradientTransformation(init_fn, update_fn)
