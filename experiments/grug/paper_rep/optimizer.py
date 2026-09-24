# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Paper-recipe optimizer for the arXiv 2609.19107 replication.

Mirrors the paper's Stage-2/3 training setup: Muon (modded-nanogpt scaling)
for the matrix parameters at the global learning rate (GLR), and AdamW for
the token embedding (lr = GLR * ELRM) and the LM head (lr = GLR * HLRM).

Interpretation choices (documented deviations from the paper text):

- Weight decay (WD, Table 5) is applied as decoupled AdamW weight decay on the
  embedding and head groups only; the Muon group trains without weight decay.
  The paper says "Muon for matrices and AdamW for embeddings/head" with a
  single WD knob; we read WD as an AdamW knob. Muon momentum/nesterov/steps
  are not specified in the paper and use the modded-nanogpt defaults
  (0.95 / True / 5).
- The LR schedule (warmup WU steps, then linear warmdown over the last WDR
  fraction, to zero) is expressed through the standard ``OptimizerConfig``
  knobs: ``lr_schedule="linear"``, ``warmup=WU``, ``decay=WDR``,
  ``min_lr_ratio=0.0``. All three groups share the schedule shape and differ
  only by their LR multipliers, matching the single GLR knob in the paper.
"""

from dataclasses import dataclass

import jax
import optax
from levanter.optim.config import OptimizerConfig
from levanter.optim.grugmuon import _grug_scale_with_muon
from levanter.utils.jax_utils import leaf_key_paths


@OptimizerConfig.register_subclass("paper_muon")
@dataclass(frozen=True)
class PaperMuonConfig(OptimizerConfig):
    """Three-group paper optimizer: muon / adamw_embed / adamw_head.

    - ``muon``: attention and MLP matrices (2-D weights except embed/head).
    - ``adamw_embed``: ``token_embed`` at ``learning_rate * embed_lr_multiplier``.
    - ``adamw_head``: ``output_proj`` at ``learning_rate * head_lr_multiplier``.

    The schedule comes from the inherited fields; recipes set
    ``lr_schedule="linear"``, ``warmup`` (WU steps), ``decay`` (WDR fraction)
    and ``min_lr_ratio=0.0``.
    """

    embed_lr_multiplier: float = 1.0
    head_lr_multiplier: float = 1.0
    momentum: float = 0.95
    nesterov: bool = True
    backend_steps: int = 5
    muon_epsilon: float = 1e-8
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-10

    def build(self, num_train_steps):
        learning_rate_schedule = self.lr_scheduler(num_train_steps)
        embed_lr_schedule = self.lr_scheduler(num_train_steps, override_lr=self.learning_rate * self.embed_lr_multiplier)
        head_lr_schedule = self.lr_scheduler(num_train_steps, override_lr=self.learning_rate * self.head_lr_multiplier)

        def optimizer(learning_rate, embed_lr, head_lr):
            def muon_transform():
                components = [
                    _grug_scale_with_muon(
                        momentum=self.momentum,
                        nesterov=self.nesterov,
                        steps=self.backend_steps,
                        muon_eps=self.muon_epsilon,
                        use_kimi_scaling=False,
                        coefficient_type="quintic",
                    )
                ]
                components.append(optax.scale(-learning_rate))
                return optax.chain(*components)

            def _adamw_transform(lr):
                components = [optax.scale_by_adam(self.beta1, self.beta2, self.epsilon)]
                if self.weight_decay > 0:
                    # Decoupled weight decay on the AdamW groups (documented reading
                    # of the paper's single WD knob; see module docstring).
                    components.append(optax.add_decayed_weights(self.weight_decay))
                components.append(optax.scale(-lr))
                return optax.chain(*components)

            return optax.multi_transform(
                {
                    "muon": muon_transform(),
                    "adamw_embed": _adamw_transform(embed_lr),
                    "adamw_head": _adamw_transform(head_lr),
                },
                self.create_mask,
            )

        return optax.inject_hyperparams(optimizer)(
            learning_rate=learning_rate_schedule,
            embed_lr=embed_lr_schedule,
            head_lr=head_lr_schedule,
        )

    def create_mask(self, params):
        paths = leaf_key_paths(params)

        def mask_fn(param, path):
            path_str = ".".join(path) if isinstance(path, (list, tuple)) else str(path)
            path_lower = path_str.lower()
            if "token_embed" in path_lower:
                return "adamw_embed"
            if "output_proj" in path_lower:
                return "adamw_head"
            if isinstance(param, jax.Array) and param.ndim >= 2:
                return "muon"
            return "adamw_embed"

        return jax.tree.map(mask_fn, params, paths)


__all__ = ["PaperMuonConfig"]
