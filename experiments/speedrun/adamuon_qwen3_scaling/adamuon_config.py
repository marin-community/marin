# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
from dataclasses import dataclass
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax
from optax import tree_utils as otu

import haliax

from levanter.optim.config import OptimizerConfig
from levanter.optim.muon import MuonConfig, zeropower_via_newtonschulz5
from levanter.optim.util import map_flattened_linear_layers


@OptimizerConfig.register_subclass("adamuon")
@dataclass(frozen=True)
class AdaMuonConfig(MuonConfig):
    """
    AdaMuon optimizer: Momentum Orthogonalized by Newton-Schulz, with adaptive
    element-wise second momentum and sign-stabilized orthogonal updates.

    Key differences from Muon (per https://arxiv.org/pdf/2507.11005):
      1. sign(Mt) is passed to Newton-Schulz instead of Mt directly.
      2. Element-wise second momentum Vt is accumulated on Ot (the orthogonalized output).
      3. Dynamic RMS-aligned rescaling: gamma_t = 0.2 * sqrt(m*n) / ||O_hat_t||_F.

    Per the paper, both first and second momentum use the SAME beta (the Muon
    momentum parameter), so no extra hyperparameters are introduced beyond Muon's.

    No bias correction on Vt — the RMS alignment step cancels any constant
    multiplicative bias exactly (paper Appendix B).

    cf: https://arxiv.org/pdf/2507.11005
    """

    # beta2 specifically for the AdamW side (embeddings, biases, lm_head).
    adam_beta2: float = 0.95

    def build(self, num_train_steps):
        learning_rate_schedule = self.lr_scheduler(num_train_steps)
        adam_lr_schedule = self.lr_scheduler(num_train_steps, override_lr=self.adam_lr)

        def optimizer(learning_rate, adam_lr):
            def muon_transform():
                components = []
                components.append(
                    scale_with_adamuon(
                        momentum=self.momentum,
                        steps=self.backend_steps,
                        eps=self.muon_epsilon,
                    )
                )
                if self.weight_decay > 0:
                    components.append(
                        optax.add_decayed_weights(
                            self.weight_decay, self.build_weight_decay_mask()
                        )
                    )
                components.append(optax.scale(-learning_rate))
                return optax.chain(*components)

            def adamw_transform():
                components = []
                if self.max_grad_norm:
                    components.append(optax.clip_by_global_norm(self.max_grad_norm))
                components.append(
                    optax.scale_by_adam(self.beta1, self.adam_beta2, self.epsilon)
                )
                adam_weight_decay = (
                    self.adam_weight_decay
                    if self.adam_weight_decay is not None
                    else self.weight_decay
                )
                if adam_weight_decay > 0:
                    components.append(
                        optax.add_decayed_weights(
                            adam_weight_decay, self.build_weight_decay_mask()
                        )
                    )
                components.append(optax.scale(-adam_lr))
                return optax.chain(*components)

            transformations = {
                "muon": muon_transform(),
                "adamw": adamw_transform(),
            }

            # use_kimi_scaling=True disables the out_first assert in create_mask,
            # since AdaMuon uses its own dynamic RMS scaling.
            return optax.multi_transform(
                transformations, partial(self.create_mask, use_kimi_scaling=True)
            )

        return optax.inject_hyperparams(optimizer)(
            learning_rate=learning_rate_schedule, adam_lr=adam_lr_schedule
        )


class ScaleByAdaMuonState(NamedTuple):
    momentum_buffer: optax.Updates
    second_momentum: optax.Updates  # EMA of O⊙O


def scale_with_adamuon(
    momentum: float = 0.95,
    steps: int = 5,
    eps: float = 1e-8,
):
    """
    AdaMuon gradient transformation (Algorithm 1 from the paper).

    Uses the SAME beta for both first and second momentum per the paper:
    "The coefficient beta is inherited directly from Muon's momentum parameter,
     ensuring that AdaMuon does not introduce any additional hyper-parameters."
    """
    steps = int(steps)
    beta = momentum  # paper uses same beta for V as for M

    def init_fn(params):
        return ScaleByAdaMuonState(
            momentum_buffer=otu.tree_zeros_like(params),
            second_momentum=otu.tree_zeros_like(params),
        )

    def update_fn(updates, state, params=None):
        del params

        # ── Algorithm 1, line 4: Mt = beta * M_{t-1} + Gt ──
        # Plain momentum, no Nesterov (sign() kills any benefit per the paper).
        buf = jax.tree.map(
            lambda m, g: None if g is None else beta * m + g,
            state.momentum_buffer,
            updates,
            is_leaf=lambda x: x is None,
        )

        # ── Line 5: Ot = Newton-Schulz(sign(Mt), T) ──
        # Uses map_flattened_linear_layers to access complete weight matrices,
        # matching the pattern used by Muon's scale_with_muon.
        def orthogonalize_layer(layer: haliax.nn.Linear):
            array = layer.weight.array
            o_array = zeropower_via_newtonschulz5(
                jnp.sign(array), steps=steps, eps=eps
            )
            return dataclasses.replace(
                layer, weight=dataclasses.replace(layer.weight, array=o_array)
            )

        o_updates = map_flattened_linear_layers(orthogonalize_layer, buf)

        # ── Line 6: Vt = beta * V_{t-1} + (1 - beta) * Ot ⊙ Ot ──
        # Element-wise on raw array leaves. jax.tree.map descends through
        # Linear/NamedArray containers to reach the raw jnp arrays.
        # Both o_updates and state.second_momentum have matching pytree structure
        # (both originate from multi_transform's muon-masked params).
        new_v = jax.tree.map(
            lambda oi, vi: None if oi is None else beta * vi + (1.0 - beta) * oi * oi,
            o_updates,
            state.second_momentum,
            is_leaf=lambda x: x is None,
        )

        # ── Line 7: O_hat_t = Ot / (sqrt(Vt) + eps) ──
        o_hat = jax.tree.map(
            lambda oi, vi: None if oi is None else oi / (jnp.sqrt(vi) + eps),
            o_updates,
            new_v,
            is_leaf=lambda x: x is None,
        )

        # ── Line 8: gamma_t = 0.2 * sqrt(m*n) / ||O_hat_t||_F ──
        # RMS-aligned rescaling needs the full weight matrix dimensions and its
        # Frobenius norm, so we use map_flattened_linear_layers again to ensure
        # we're operating on complete matrices (not sub-arrays).
        def rms_rescale_layer(layer: haliax.nn.Linear):
            array = layer.weight.array

            if array.ndim == 2:
                m, n = array.shape
                fro = jnp.linalg.norm(array)
                gamma = 0.2 * jnp.sqrt(
                    jnp.array(m * n, dtype=array.dtype)
                ) / jnp.maximum(fro, eps)
                scaled = gamma * array
            elif array.ndim == 3:
                # Stacked scan layers [L, m, n]: scale each slice independently
                m, n = array.shape[-2], array.shape[-1]
                fro = jnp.linalg.norm(
                    array.reshape(array.shape[0], -1), axis=-1
                )[:, None, None]
                gamma = 0.2 * jnp.sqrt(
                    jnp.array(m * n, dtype=array.dtype)
                ) / jnp.maximum(fro, eps)
                scaled = gamma * array
            else:
                # Non-matrix params should not reach here (routed to adamw),
                # but pass through safely.
                scaled = array

            return dataclasses.replace(
                layer, weight=dataclasses.replace(layer.weight, array=scaled)
            )

        final_updates = map_flattened_linear_layers(rms_rescale_layer, o_hat)

        return final_updates, ScaleByAdaMuonState(
            momentum_buffer=buf, second_momentum=new_v
        )

    return optax.GradientTransformation(init_fn, update_fn)