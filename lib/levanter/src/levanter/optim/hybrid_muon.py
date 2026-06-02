# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""HybridMuon: Muon orthogonalization blended with a nuclear-norm / effective-rank rescaling.

This is a JAX/optax port of the PyTorch ``HybridMuon`` optimizer (the ``opt_a`` / ``opt_f``
variants). For every 2D linear-layer update it computes:

    g       = (Nesterov) momentum of the gradient                    (raw update matrix)
    orth    = NS5(g)                                                  (Muon orthogonalization, sing. vals -> ~1)
    nuc     = <orth, g>          = ||g||_*   (nuclear norm)           (polar-factor identity)
    frob2   = <g, g>             = ||g||_F^2
    k       = <orth, orth>       = ||orth||_F^2  (~ effective rank)
    r       = nuc^2 / frob2      (effective rank ratio, scale-invariant)

and then blends ``orth`` with the raw update ``g`` using variant-specific coefficients:

    variant "a":
        c     = 1 / sqrt(1 + k / (8 r))
        alpha = c * sqrt(nuc / frob2)
        beta  = 0.5 c * sqrt(frob2 / nuc)
        u     = beta * g + alpha * orth

    variant "f":   (K = 4/7)
        C2    = 1.1 * K * r^2
        alpha = (r - C2 / (4 r)) * sqrt(nuc)
        beta  = (C2 - K r^2) / sqrt(nuc)
        u     = alpha * orth + beta * g

The update is finally scaled by the Muon aspect-ratio factor ``sqrt(max(1, out/in))``.

``nuc`` is evaluated via the polar-factor identity ``<orth, g> = trace(orth^T g) = sum_i sigma_i(g)``
(exact when ``orth`` is the polar unitary of ``g``), which is the cheap JAX equivalent of the
``trace(h)`` from the QDWH polar decomposition used in the PyTorch reference.

As in :mod:`levanter.optim.muon`, only linear-layer weights are routed through this transform;
embeddings, ``lm_head``, biases and other non-matrix parameters use AdamW.
"""

import dataclasses
from dataclasses import dataclass
from functools import partial
from typing import Literal, NamedTuple, Optional

import jax
import jax.numpy as jnp
import optax
from optax import tree_utils as otu

import haliax

from levanter.optim.config import OptimizerConfig
from levanter.optim.util import (
    CoefficientType,
    label_linear_like_module,
    map_flattened_linear_layers,
    zeropower_via_newtonschulz5,
)
from levanter.utils.jax_utils import leaf_key_paths

HybridMuonVariant = Literal["a", "f"]


def _hybrid_blend(raw: jax.Array, orth: jax.Array, *, variant: HybridMuonVariant, eps: float) -> jax.Array:
    """Blend the raw update ``raw`` with its orthogonalization ``orth`` (variant "a" or "f")."""
    nuc = jnp.sum(orth * raw)  # <orth, raw> = ||raw||_* (nuclear norm via polar-factor identity)
    frob2 = jnp.sum(raw * raw)  # ||raw||_F^2
    nuc = jnp.maximum(nuc, eps)
    frob2 = jnp.maximum(frob2, eps)
    r = nuc**2 / frob2  # effective-rank ratio (scale-invariant)

    if variant == "a":
        k_proxy = jnp.sum(orth * orth)  # ||orth||_F^2 ~ effective rank
        c = 1.0 / jnp.sqrt(1.0 + k_proxy / (8.0 * r))
        alpha = c * jnp.sqrt(nuc / frob2)
        beta = 0.5 * c * jnp.sqrt(frob2 / nuc)
        return beta * raw + alpha * orth
    elif variant == "f":
        k = 4.0 / 7.0
        c2 = 1.1 * k * r**2
        alpha = (r - c2 / (4.0 * r)) * jnp.sqrt(nuc)
        beta = (c2 - k * r**2) / jnp.sqrt(nuc)
        return alpha * orth + beta * raw
    else:
        raise ValueError(f"Unsupported HybridMuon variant: {variant!r} (expected 'a' or 'f').")


class ScaleByHybridMuonState(NamedTuple):
    """State for the HybridMuon algorithm."""

    momentum_buffer: optax.Updates


def scale_with_hybrid_muon(
    momentum=0.95,
    nesterov=True,
    steps=5,
    muon_eps=1e-8,
    use_kimi_scaling=False,
    coefficient_type: CoefficientType = "quintic",
    variant: HybridMuonVariant = "a",
):
    steps = int(steps)
    grad_coeff = 1.0 - momentum  # PyTorch HybridMuon uses the lerp momentum convention

    def init_fn(params):
        return ScaleByHybridMuonState(momentum_buffer=otu.tree_zeros_like(params))

    def update_fn(updates, state, params=None):
        del params
        buf = jax.tree.map(
            lambda m, g: None if g is None else momentum * m + grad_coeff * g,
            state.momentum_buffer,
            updates,
            is_leaf=lambda x: x is None,
        )
        if nesterov:
            updates = jax.tree.map(
                lambda m, g: None if g is None else momentum * m + grad_coeff * g,
                buf,
                updates,
                is_leaf=lambda x: x is None,
            )
        else:
            updates = buf

        def transform_linear_layer(layer: haliax.nn.Linear):
            assert layer.weight.ndim == 2
            raw = layer.weight.array
            orth = zeropower_via_newtonschulz5(raw, steps=steps, eps=muon_eps, coefficient_type=coefficient_type)
            blended = _hybrid_blend(raw, orth, variant=variant, eps=muon_eps)

            if not use_kimi_scaling:
                scale = jnp.sqrt(jnp.maximum(1, blended.shape[0] / blended.shape[1]))  # sqrt(Out/In)
            else:
                scale = 0.2 * jnp.sqrt(jnp.maximum(blended.shape[0], blended.shape[1]))
            blended = blended * scale

            updated_weight = dataclasses.replace(layer.weight, array=blended.astype(raw.dtype))
            return dataclasses.replace(layer, weight=updated_weight)  # type: ignore

        updates = map_flattened_linear_layers(transform_linear_layer, updates)
        return updates, ScaleByHybridMuonState(momentum_buffer=buf)

    return optax.GradientTransformation(init_fn, update_fn)


@OptimizerConfig.register_subclass("hybrid_muon")
@dataclass(frozen=True)
class HybridMuonConfig(OptimizerConfig):
    """HybridMuon optimizer: Muon orthogonalization blended with nuclear-norm rescaling.

    JAX port of the PyTorch ``HybridMuon`` (``opt_a`` / ``opt_f``). Set ``variant`` to ``"a"`` or
    ``"f"`` to select the blend formula. Linear-layer weights use the hybrid transform; everything
    else uses AdamW (matching :class:`~levanter.optim.muon.MuonConfig`).
    """

    variant: HybridMuonVariant = "a"
    lr: float = 0.02
    adam_lr: float = 6e-4  # Adam LR for the AdamW-routed parameters
    momentum: float = 0.95
    nesterov: bool = True
    backend_steps: int = 5  # Newton-Schulz iterations
    weight_decay: float = 0.0
    adam_weight_decay: Optional[float] = None
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    muon_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    use_kimi_scaling: bool = False
    coefficient_type: CoefficientType = "quintic"

    def build(self, num_train_steps):
        if self.variant not in ("a", "f"):
            raise ValueError(f"HybridMuonConfig.variant must be 'a' or 'f', got {self.variant!r}.")

        learning_rate_schedule = self.lr_scheduler(num_train_steps)
        adam_lr_schedule = self.lr_scheduler(num_train_steps, override_lr=self.adam_lr)

        def optimizer(learning_rate, adam_lr):
            def hybrid_muon_transform():
                components = [
                    scale_with_hybrid_muon(
                        self.momentum,
                        self.nesterov,
                        self.backend_steps,
                        self.muon_epsilon,
                        self.use_kimi_scaling,
                        self.coefficient_type,
                        self.variant,
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

            return optax.multi_transform(
                {"hybrid_muon": hybrid_muon_transform(), "adamw": adamw_transform()},
                partial(self.create_mask, use_kimi_scaling=self.use_kimi_scaling),
            )

        return optax.inject_hyperparams(optimizer)(learning_rate=learning_rate_schedule, adam_lr=adam_lr_schedule)

    def create_mask(self, params, use_kimi_scaling=True):
        """Label linear-layer weights 'hybrid_muon' and everything else 'adamw'."""
        paths = leaf_key_paths(params)

        def mask_fn(param, path):
            path_str = ".".join(path) if isinstance(path, (list, tuple)) else str(path)
            if "Embedding" in path_str or "lm_head" in path_str:
                return "adamw"
            elif isinstance(param, haliax.nn.Linear):
                assert param._out_first or use_kimi_scaling
                return label_linear_like_module(param, weight_label="hybrid_muon", bias_label="adamw")
            else:
                return "adamw"

        return haliax.tree_util.tree_map(mask_fn, params, paths, is_leaf=lambda x: isinstance(x, haliax.nn.Linear))
