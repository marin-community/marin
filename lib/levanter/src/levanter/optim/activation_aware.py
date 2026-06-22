# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Activation-aware Muon (Ali Jadbabaie's "idea 3").

Plain Muon builds the update only from the gradient/momentum matrix ``M`` (its
matrix-sign / polar factor). **Activation-aware** Muon additionally whitens by the
layer's *input activation* second moment ``Σ = A Aᵀ`` (``A`` = layer inputs,
d_in × n_tokens). For a weight ``W`` (d_out × d_in) with momentum ``M``:

    D = sign( M · Σ^{-1/2} ) · Σ^{-1/2}          (sign = matrix sign = polar = NS5)
    W ← W − η · D

i.e. whiten the gradient on the *input* side by Σ^{-1/2}, orthogonalize (Muon's
matrix-sign), then apply Σ^{-1/2} again. Σ=I recovers plain Muon (verified cos 1.0).
This is the natural-gradient / KFAC "A"-factor preconditioner: the activation
covariance is *independent* information not recoverable from the gradient (the
gradient G=δAᵀ conflates the output-grad δ with A), so it must be captured from the
forward pass — see ``levanter.optim.activation_capture``.

``Σ^{-1/2}`` is computed via ``eigh(Σ)`` with damping ``λ`` (``Σ + λ·mean(eig)·I``)
for numerical stability, and the resulting ``D`` is Frobenius-normalized to
``√(min(d_out,d_in))`` (Muon's update norm) so the activation factors change only the
*direction* of the update, keeping the learning rate transferable across layers and
the Muon baseline. Kaiyue's earlier torch attempt diverged; the damping +
normalization here is what makes it stable (verified on a standalone anisotropic task).

Scope (v1): the activation Gram is captured cheaply at the transformer-block boundary
for the five linears whose input is a normed residual-stream activation — attention
q/k/v (input = ``input_layernorm`` output) and MLP gate/up (input =
``post_attention_layernorm`` output). ``o_proj`` and ``down_proj`` (whose inputs live
deep inside attention/MLP) get plain Muon; embeddings/lm_head/norms/biases get AdamW.
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

# Which captured activation Gram feeds each linear (by path substring). o_proj/down_proj
# are absent → plain Muon (their inputs are not captured at the block boundary).
ATTN_IN_PROJ = ("q_proj", "k_proj", "v_proj")  # input = input_layernorm(residual)
MLP_IN_PROJ = ("gate_proj", "up_proj")  # input = post_attention_layernorm(residual)


@OptimizerConfig.register_subclass("activation_aware")
@dataclass(frozen=True)
class ActivationAwareConfig(OptimizerConfig):
    """Activation-aware Muon (idea 3): D = sign(M Σ^{-1/2}) Σ^{-1/2}, Σ = activation 2nd moment."""

    # --- the new knobs (vs Muon) ---
    damping: float = 1e-3  # λ in (Σ + λ·mean(eig)·I)^{-1/2}; larger = closer to plain Muon
    sigma_beta: float = 0.95  # EMA decay for the per-layer activation Gram Σ
    normalize_fro: bool = True  # rescale D to ‖D‖_F = √(min(d_out,d_in)) (Muon's update norm)
    ns_steps: int = 5  # Newton-Schulz steps for the matrix-sign

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
    matrix_epsilon: float = 1e-7  # normalization floor in the matrix path
    max_grad_norm: float = 1.0
    use_kimi_scaling: bool = False

    def build(self, num_train_steps):
        learning_rate_schedule = self.lr_scheduler(num_train_steps)
        adam_lr_schedule = self.lr_scheduler(num_train_steps, override_lr=self.adam_lr)

        def optimizer(learning_rate, adam_lr):
            def matrix_transform():
                components = [
                    scale_with_activation_aware(
                        self.momentum,
                        self.nesterov,
                        self.damping,
                        self.sigma_beta,
                        self.normalize_fro,
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
                "activation_aware": matrix_transform(),
                "adamw": adamw_transform(),
            }
            return optax.multi_transform(
                transformations, partial(self.create_mask, use_kimi_scaling=self.use_kimi_scaling)
            )

        return optax.inject_hyperparams(optimizer)(learning_rate=learning_rate_schedule, adam_lr=adam_lr_schedule)

    def create_mask(self, params, use_kimi_scaling=True):
        """Label matrix (Linear) weights 'activation_aware', everything else 'adamw' (matches Muon)."""
        paths = leaf_key_paths(params)

        def mask_fn(param, path):
            path_str = ".".join(path) if isinstance(path, (list, tuple)) else str(path)
            if "Embedding" in path_str or "lm_head" in path_str:
                return "adamw"
            elif isinstance(param, haliax.nn.Linear):
                assert param._out_first or use_kimi_scaling
                return label_linear_like_module(param, weight_label="activation_aware", bias_label="adamw")
            else:
                return "adamw"

        return haliax.tree_util.tree_map(mask_fn, params, paths, is_leaf=lambda x: isinstance(x, haliax.nn.Linear))


class ScaleByActivationAwareState(NamedTuple):
    momentum_buffer: optax.Updates
    sigma_ema: optax.Updates  # per-(attn_in/mlp_in)-per-layer EMA of the activation Gram


def _inv_sqrt_gram(gram, damping, eps):
    """Σ^{-1/2} for one (d_in × d_in) PSD activation Gram, damped by λ·mean(eig)."""
    gram = gram.astype(jnp.float32)
    gram = 0.5 * (gram + gram.T)  # symmetrize against fp error
    w, u = jnp.linalg.eigh(gram)
    w = jnp.maximum(w, 0.0)
    inv = 1.0 / jnp.sqrt(w + damping * jnp.mean(w) + eps)
    return (u * inv[None, :]) @ u.T


def _act_aware_matrix(m, sigma_inv_sqrt, *, normalize_fro, ns_steps, eps, use_kimi_scaling):
    """D = sign(M Σ^{-1/2}) Σ^{-1/2} for one weight M (d_out × d_in), Σ^{-1/2} given (d_in × d_in)."""
    orig_dtype = m.dtype
    x = m.astype(jnp.float32)
    whitened = x @ sigma_inv_sqrt  # M Σ^{-1/2}
    ortho = zeropower_via_newtonschulz5(whitened, steps=ns_steps, eps=eps, coefficient_type="quintic")  # sign(·)
    d = ortho @ sigma_inv_sqrt  # · Σ^{-1/2}

    if normalize_fro:
        k = min(d.shape[0], d.shape[1])
        d = d * (jnp.sqrt(jnp.float32(k)) / (jnp.linalg.norm(d) + eps))

    if not use_kimi_scaling:
        scale = jnp.sqrt(jnp.maximum(1.0, d.shape[0] / d.shape[1]))  # sqrt(Out/In)
    else:
        scale = 0.2 * jnp.sqrt(jnp.maximum(d.shape[0], d.shape[1]))
    return (d * scale).astype(orig_dtype)


def _muon_matrix(m, *, ns_steps, eps, use_kimi_scaling):
    """Plain Muon update for one weight (used for o_proj/down_proj)."""
    orig_dtype = m.dtype
    x = m.astype(jnp.float32)
    u = zeropower_via_newtonschulz5(x, steps=ns_steps, eps=eps, coefficient_type="quintic")
    if not use_kimi_scaling:
        scale = jnp.sqrt(jnp.maximum(1.0, u.shape[0] / u.shape[1]))
    else:
        scale = 0.2 * jnp.sqrt(jnp.maximum(u.shape[0], u.shape[1]))
    return (u * scale).astype(orig_dtype)


def _gram_key_for_path(path_str):
    """Return 'attn_in' / 'mlp_in' for activation-aware linears, None for plain-Muon ones."""
    if any(p in path_str for p in ATTN_IN_PROJ):
        return "attn_in"
    if any(p in path_str for p in MLP_IN_PROJ):
        return "mlp_in"
    return None


def scale_with_activation_aware(
    momentum=0.95,
    nesterov=True,
    damping=1e-3,
    sigma_beta=0.95,
    normalize_fro=True,
    ns_steps=5,
    eps=1e-7,
    use_kimi_scaling=False,
):
    """Optax transform consuming per-layer activation Grams via ``update(..., grams=...)``.

    ``grams`` is a dict ``{"attn_in": NamedArray[Layers, In, In2], "mlp_in": ...}`` produced by
    :mod:`levanter.optim.activation_capture`; the leading ``Layers`` axis aligns with the
    Stacked transformer layers (so each linear's stacked weight ``[Layers, Out, In]`` pairs
    with its Gram ``[Layers, In, In2]`` by ``vmap`` over ``Layers``). An EMA of the Grams is
    kept in the optimizer state for stability.
    """
    momentum = float(momentum)
    damping = float(damping)
    sigma_beta = float(sigma_beta)

    def init_fn(params):
        return ScaleByActivationAwareState(momentum_buffer=otu.tree_zeros_like(params), sigma_ema={})

    def update_fn(updates, state, params=None, *, grams=None, **extra_args):
        if grams is None:
            raise ValueError("activation_aware optimizer requires `grams` (per-layer activation Grams).")

        buf = jax.tree.map(
            lambda m, g: None if g is None else momentum * m + g,
            state.momentum_buffer,
            updates,
            is_leaf=lambda x: x is None,
        )
        if nesterov:
            momentum_updates = jax.tree.map(
                lambda m, g: None if g is None else momentum * m + g,
                buf,
                updates,
                is_leaf=lambda x: x is None,
            )
        else:
            momentum_updates = buf

        # EMA the captured Grams (raw float32 arrays, leading Layers axis).
        raw_grams = {k: v.array.astype(jnp.float32) for k, v in grams.items()}
        if state.sigma_ema:
            sigma_ema = {k: sigma_beta * state.sigma_ema[k] + (1.0 - sigma_beta) * raw_grams[k] for k in raw_grams}
        else:
            sigma_ema = raw_grams
        # Σ^{-1/2} per layer per gram-group, vmapped over the leading Layers axis.
        inv_sqrt = {
            k: jax.vmap(lambda g: _inv_sqrt_gram(g, damping, eps))(v) for k, v in sigma_ema.items()
        }  # [Layers, In, In2]

        # Flatten articulated linears to 2D-per-layer weights [Layers, Out, In] (In = embed for the
        # activation-aware projections), apply the per-layer matrix map (vmapped over Layers), unflatten.
        flat = flatten_linear_layers(momentum_updates)
        paths = leaf_key_paths(flat, is_leaf=lambda x: isinstance(x, haliax.nn.Linear))

        def transform(layer, path):
            if not isinstance(layer, haliax.nn.Linear) or not isinstance(layer.weight, haliax.NamedArray):
                return layer
            path_str = ".".join(path) if isinstance(path, (list, tuple)) else str(path)
            gram_key = _gram_key_for_path(path_str)
            w = layer.weight
            arr = w.array  # [Layers, Out, In] when stacked, else [Out, In]
            stacked = arr.ndim == 3
            if gram_key is not None:
                aa = partial(
                    _act_aware_matrix,
                    normalize_fro=normalize_fro,
                    ns_steps=ns_steps,
                    eps=eps,
                    use_kimi_scaling=use_kimi_scaling,
                )
                new_arr = jax.vmap(aa)(arr, inv_sqrt[gram_key]) if stacked else aa(arr, inv_sqrt[gram_key][0])
            else:
                mu = partial(_muon_matrix, ns_steps=ns_steps, eps=eps, use_kimi_scaling=use_kimi_scaling)
                new_arr = jax.vmap(mu)(arr) if stacked else mu(arr)
            return dataclasses.replace(layer, weight=dataclasses.replace(w, array=new_arr))  # type: ignore

        flat = haliax.tree_util.tree_map(transform, flat, paths, is_leaf=lambda x: isinstance(x, haliax.nn.Linear))
        new_updates = unflatten_linear_layers(momentum_updates, flat)
        return new_updates, ScaleByActivationAwareState(momentum_buffer=buf, sigma_ema=sigma_ema)

    return optax.GradientTransformationExtraArgs(init_fn, update_fn)
