# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Left-preconditioned Muon (Ali Jadbabaie's "idea 4").

Muon orthogonalizes the (momentum) gradient via its matrix sign / polar factor.
**Left-preconditioned** Muon additionally whitens on the *output* side by the gradient's
own left second moment ``H = EMA(G Gᵀ)`` (the Shampoo-style left factor). For a weight
``W`` (d_out × d_in) with momentum ``M`` and gradient second moment ``H`` (d_out × d_out):

    U = ρ · H^{-1/2} · polar( H^{-1/2} · M )           (polar = matrix sign = NS5)
    W ← W − η · U

i.e. whiten on the left by ``H^{-1/2}``, orthogonalize, whiten again. ``H = I`` recovers plain
Muon. Unlike idea 3 (which whitened by the *activation* second moment AAᵀ and needed forward
capture), ``H`` here is built from the gradient alone — so this is a pure optimizer with no
trainer/model changes. Using the EMA (rather than the current ``G Gᵀ``) keeps it non-degenerate:
``H^{-1/2}M`` is not the polar factor of ``M`` because ``H`` reflects the gradient's *history*.

``H^{-1/2}`` uses a **truncated (rank-)pseudo-inverse** (facebookresearch/optimizers#265): tiny
eigenvalues are zeroed rather than blown up by the inverse root, which is more stable than a
plain inverse-root or additive damping. The eigenvalues are normalized by the mean kept
eigenvalue first, so ``H^{-1/2}`` is scale-free and the update magnitude (and the learning rate)
stays comparable to Muon.
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

    # --- the new knobs (vs Muon) ---
    h_beta: float = 0.95  # EMA decay for the gradient second moment H = EMA(G Gᵀ)
    clamp_rel: float = 1e-15  # truncated pseudo-inverse: zero eigenvalues below clamp_rel · λ_max
    # 1e-15 (= Shampoo / fb#265) truncates only numerically-zero directions; larger values
    # aggressively drop real low-curvature directions (which hurts) — keep this tiny.
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
    matrix_epsilon: float = 1e-7  # NS5 normalization floor
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
                        self.clamp_rel,
                        self.ns_steps,
                        self.matrix_epsilon,
                        self.use_kimi_scaling,
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

            transformations = {
                "left_precond_muon": matrix_transform(),
                "adamw": adamw_transform(),
            }
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


def _trunc_inv_sqrt(h, clamp_rel, eps):
    """H^{-1/2} via truncated pseudo-inverse: zero eigenvalues below clamp_rel·λ_max (fb#265).

    Eigenvalues are normalized by the mean kept eigenvalue first, so the result is scale-free
    (H ∝ I ⟹ H^{-1/2} = I) and the update stays Muon-scaled.
    """
    h = h.astype(jnp.float32)
    h = 0.5 * (h + h.T)
    w, u = jnp.linalg.eigh(h)
    w = jnp.maximum(w, 0.0)
    wmax = jnp.max(w)
    keep = w > clamp_rel * wmax + eps
    mean_kept = jnp.sum(jnp.where(keep, w, 0.0)) / jnp.maximum(jnp.sum(keep), 1.0)
    wn = w / (mean_kept + eps)
    inv = jnp.where(keep, 1.0 / jnp.sqrt(wn + eps), 0.0)
    return (u * inv[None, :]) @ u.T


def _left_precond_matrix(m, h, *, clamp_rel, ns_steps, eps, use_kimi_scaling):
    """U = H^{-1/2} polar(H^{-1/2} M) for one weight M (d_out × d_in), H (d_out × d_out)."""
    orig_dtype = m.dtype
    x = m.astype(jnp.float32)
    hih = _trunc_inv_sqrt(h, clamp_rel, eps)  # (out, out)
    whitened = hih @ x  # H^{-1/2} M
    ortho = zeropower_via_newtonschulz5(whitened, steps=ns_steps, eps=eps, coefficient_type="quintic")
    u = hih @ ortho  # H^{-1/2} polar(·)

    if not use_kimi_scaling:
        scale = jnp.sqrt(jnp.maximum(1.0, u.shape[0] / u.shape[1]))  # sqrt(Out/In)
    else:
        scale = 0.2 * jnp.sqrt(jnp.maximum(u.shape[0], u.shape[1]))
    return (u * scale).astype(orig_dtype)


def scale_with_left_precond_muon(
    momentum=0.95, nesterov=True, h_beta=0.95, clamp_rel=1e-6, ns_steps=5, eps=1e-7, use_kimi_scaling=False
):
    momentum = float(momentum)
    h_beta = float(h_beta)
    clamp_rel = float(clamp_rel)

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
        # momentum on the search direction (like Muon)
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

        # update H = EMA(G Gᵀ) from the RAW gradient (flattened to 2D-per-layer)
        flat_grad = flatten_linear_layers(updates)

        def new_h(layer, h):
            if not (isinstance(layer, haliax.nn.Linear) and isinstance(layer.weight, haliax.NamedArray)):
                return h
            return h_beta * h + (1.0 - h_beta) * _gram(layer.weight.array)

        h_ema = haliax.tree_util.tree_map(
            new_h, flat_grad, state.h_ema, is_leaf=lambda x: isinstance(x, haliax.nn.Linear)
        )

        # apply U = H^{-1/2} polar(H^{-1/2} M) per matrix (vmapped over the stacked Layer axis)
        flat_dir = flatten_linear_layers(direction)
        paths = leaf_key_paths(flat_dir, is_leaf=lambda x: isinstance(x, haliax.nn.Linear))

        def precond(layer, h, path):
            if not (isinstance(layer, haliax.nn.Linear) and isinstance(layer.weight, haliax.NamedArray)):
                return layer
            w = layer.weight
            arr = w.array
            fn = partial(
                _left_precond_matrix,
                clamp_rel=clamp_rel,
                ns_steps=ns_steps,
                eps=eps,
                use_kimi_scaling=use_kimi_scaling,
            )
            new_arr = jax.vmap(fn)(arr, h) if arr.ndim == 3 else fn(arr, h)
            return dataclasses.replace(layer, weight=dataclasses.replace(w, array=new_arr))  # type: ignore

        flat_out = haliax.tree_util.tree_map(
            precond, flat_dir, h_ema, paths, is_leaf=lambda x: isinstance(x, haliax.nn.Linear)
        )
        new_updates = unflatten_linear_layers(direction, flat_out)
        return new_updates, ScaleByLeftPrecondMuonState(momentum_buffer=buf, h_ema=h_ema)

    return optax.GradientTransformation(init_fn, update_fn)
