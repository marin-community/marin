# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Gain-gated Muon (Ali Jadbabaie's "idea 2").

A one-parameter family of matrix updates that interpolates between Muon and the
square-root / HJB policy. From the SVD of the (momentum) gradient ``G = P Σ Qᵀ``,
the update is

    U = P · diag(aᵢ) · Qᵀ ,   aᵢ = min( ρ , √(2 g σᵢ) )

with a saturation threshold ``σ_c = ρ² / (2g)``:

    σᵢ <  σ_c   ⟹   aᵢ = √(2 g σᵢ)   (square-root / HJB regime — small singular values)
    σᵢ ≥  σ_c   ⟹   aᵢ = ρ            (saturated / Muon regime — large singular values)

Endpoints (with the singular values of the Frobenius-normalized gradient, ρ=1):

    g → ∞ : every mode saturates → aᵢ = ρ  → U = P Qᵀ  (pure Muon)
    g → 0 : no mode saturates    → aᵢ = √(2 g σᵢ) ∝ P Σ^{1/2} Qᵀ  (pure square-root / HJB)

Ali's intuition: Muon amplifies the rare (small-σ) directions at a constant rate,
which is good; but the large-σ directions "should not need" that — the ``min``
clamps them at ρ while the small ones follow the smooth √ law. ``g`` dials where
the clamp kicks in.

Unlike Muon (Newton-Schulz, no singular values), this needs the per-singular-value
gate, so we use an exact SVD (per Ali's §5 debug recipe). ρ is folded into the
learning rate (ρ=1), so the swept knobs are ``gain`` and the learning rate.

``normalize_fro`` selects the two reported variants:
  * False (original): apply U as derived; its Frobenius norm varies with g.
  * True (normalized): rescale U to ‖U‖_F = √(min(d₁,d₂)) — Muon's update norm —
    so g changes only the *direction* of the update, not its magnitude (decoupling
    g from the learning rate).
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
from levanter.optim.util import label_linear_like_module, map_flattened_linear_layers, zeropower_via_newtonschulz5
from levanter.utils.jax_utils import leaf_key_paths


@OptimizerConfig.register_subclass("gain_gated")
@dataclass(frozen=True)
class GainGatedConfig(OptimizerConfig):
    """Gain-gated Muon (interpolates Muon ↔ square-root/HJB via the gain ``g``)."""

    # --- the new knobs (vs Muon) ---
    gain: float = 1.0  # control-effort weight g; large g → Muon, small g → square-root
    rho: float = 1.0  # spectral cap ρ; folded into the learning rate, keep at 1.0
    normalize_fro: bool = False  # True → rescale the update to ‖U‖_F = √(min(d1,d2)) (Muon's norm)
    psi_eps: float = 0.0  # ε in ψ_ε(σ)=√(σ²+ε²)−ε (finite-gain HJB); 0 = original gate, >0 suppresses small σ

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
    svd_epsilon: float = 1e-7  # normalization floor for the SVD path
    max_grad_norm: float = 1.0
    use_kimi_scaling: bool = False

    def build(self, num_train_steps):
        learning_rate_schedule = self.lr_scheduler(num_train_steps)
        adam_lr_schedule = self.lr_scheduler(num_train_steps, override_lr=self.adam_lr)

        def optimizer(learning_rate, adam_lr):
            def gain_gated_transform():
                components = [
                    scale_with_gain_gated(
                        self.momentum,
                        self.nesterov,
                        self.gain,
                        self.rho,
                        self.normalize_fro,
                        self.psi_eps,
                        self.svd_epsilon,
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
                "gain_gated": gain_gated_transform(),
                "adamw": adamw_transform(),
            }
            return optax.multi_transform(
                transformations, partial(self.create_mask, use_kimi_scaling=self.use_kimi_scaling)
            )

        return optax.inject_hyperparams(optimizer)(learning_rate=learning_rate_schedule, adam_lr=adam_lr_schedule)

    def create_mask(self, params, use_kimi_scaling=True):
        """Label matrix (Linear) weights 'gain_gated', everything else 'adamw' (matches Muon)."""
        paths = leaf_key_paths(params)

        def mask_fn(param, path):
            path_str = ".".join(path) if isinstance(path, (list, tuple)) else str(path)
            if "Embedding" in path_str or "lm_head" in path_str:
                return "adamw"
            elif isinstance(param, haliax.nn.Linear):
                assert param._out_first or use_kimi_scaling
                return label_linear_like_module(param, weight_label="gain_gated", bias_label="adamw")
            else:
                return "adamw"

        return haliax.tree_util.tree_map(mask_fn, params, paths, is_leaf=lambda x: isinstance(x, haliax.nn.Linear))


class ScaleByGainGatedState(NamedTuple):
    momentum_buffer: optax.Updates


def _gain_gated_matrix(array, *, gain, rho, normalize_fro, psi_eps, eps, use_kimi_scaling):
    """Finite-gain HJB Muon: U = P diag(min(ρ, √(2 g ψ_ε(σᵢ)))) Qᵀ from the SVD.

    ψ_ε(s) = √(s² + ε²) − ε is a smoothed (Huber-like) version of the singular value
    that enters the nuclear-norm cost. ``psi_eps`` (ε) generalizes the gate:
      * ε = 0  ⟹  ψ_0(s) = s  ⟹  the original gain-gated update (a = min(ρ, √(2gσ))).
      * ε > 0  suppresses small singular values quadratically — for σ ≪ ε,
        ψ_ε(σ) ≈ σ²/(2ε), so √(2g ψ_ε(σ)) ≈ σ√(g/ε) scales ∝σ (linear) instead of ∝√σ,
        down-weighting the rare directions; for σ ≫ ε, ψ_ε(σ) ≈ σ (unchanged).

    Frobenius-normalizing first (so Σσᵢ²=1, matching Muon's NS normalization) makes
    ``gain`` and ``psi_eps`` transferable across layers/steps. Computed in float32 for
    SVD stability, cast back to the input dtype.
    """
    orig_dtype = array.dtype
    x = array.astype(jnp.float32)
    x = x / (jnp.linalg.norm(x) + eps)

    # Reconstruct U = P diag(a) Qᵀ WITHOUT jnp.linalg.svd. The orthogonal polar factor
    # P Qᵀ comes from Newton-Schulz (robust on TPU, same primitive as Muon); the singular
    # values + right vectors come from eigh of the Gram (XᵀX = Q diag(σ²) Qᵀ, stable).
    # Then U = (P Qᵀ)·(Q diag(a) Qᵀ) = P diag(a) Qᵀ.  This avoids the iterative svd path,
    # whose TPU implementation deterministically hangs on some inputs (the ε≥0.03 step
    # buffers). Gating is unchanged: a = min(ρ, √(2 g ψ_ε(σ))), ψ_ε(σ)=√(σ²+ε²)−ε.
    ortho = zeropower_via_newtonschulz5(x, steps=5, eps=eps, coefficient_type="quintic")
    gram = x.T @ x  # (n, n) symmetric PSD
    w, q = jnp.linalg.eigh(gram)  # w = σ² (ascending), q = right singular vectors
    sigma = jnp.sqrt(jnp.maximum(w, 0.0))
    s_eff = jnp.maximum(0.0, jnp.sqrt(sigma * sigma + psi_eps * psi_eps) - psi_eps)
    a = jnp.minimum(rho, jnp.sqrt(2.0 * gain * s_eff))
    u = ortho @ ((q * a[None, :]) @ q.T)

    if normalize_fro:
        k = min(x.shape[0], x.shape[1])
        u = u * (jnp.sqrt(jnp.float32(k)) / (jnp.linalg.norm(u) + eps))

    # Aspect-ratio scaling, identical to Muon (assumes out-first layout unless kimi).
    if not use_kimi_scaling:
        scale = jnp.sqrt(jnp.maximum(1.0, u.shape[0] / u.shape[1]))  # sqrt(Out/In)
    else:
        scale = 0.2 * jnp.sqrt(jnp.maximum(u.shape[0], u.shape[1]))
    u = u * scale

    return u.astype(orig_dtype)


def scale_with_gain_gated(
    momentum=0.95, nesterov=True, gain=1.0, rho=1.0, normalize_fro=False, psi_eps=0.0, eps=1e-7, use_kimi_scaling=False
):
    gain = float(gain)
    rho = float(rho)
    psi_eps = float(psi_eps)

    def init_fn(params):
        return ScaleByGainGatedState(momentum_buffer=otu.tree_zeros_like(params))

    def update_fn(updates, state, params=None):
        buf = jax.tree.map(
            lambda m, g: None if g is None else momentum * m + g,
            state.momentum_buffer,
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
            new_array = _gain_gated_matrix(
                layer.weight.array,
                gain=gain,
                rho=rho,
                normalize_fro=normalize_fro,
                psi_eps=psi_eps,
                eps=eps,
                use_kimi_scaling=use_kimi_scaling,
            )
            return dataclasses.replace(layer, weight=dataclasses.replace(layer.weight, array=new_array))  # type: ignore

        updates = map_flattened_linear_layers(transform_linear_layer, updates)
        return updates, ScaleByGainGatedState(momentum_buffer=buf)

    return optax.GradientTransformation(init_fn, update_fn)
