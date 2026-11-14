# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""E12 full-rank attention-projection/head optimizer; other parameters are frozen."""

from dataclasses import dataclass

import jax
import optax
from levanter.optim.config import OptimizerConfig
from levanter.utils.jax_utils import leaf_key_paths

from experiments.grug.moe.optimizer import GrugMoeMuonHConfig

ATTENTION_PROJECTIONS = ("attn.w_q", "attn.w_k", "attn.w_v", "attn.w_o")


def trainable_path(path: str) -> bool:
    """Match only canonical attention matrices and the untied output projection."""
    return path == "output_proj" or any(path.endswith("." + suffix) for suffix in ATTENTION_PROJECTIONS)


def attention_head_mask(params):
    paths = leaf_key_paths(params)
    return jax.tree.map(lambda path: trainable_path(str(path)), paths)


@OptimizerConfig.register_subclass("e12_attention_head_muonh_v1")
@dataclass(frozen=True)
class E12AttentionHeadConfig(GrugMoeMuonHConfig):
    """Keep the qualified optimizer on selected leaves; zero every frozen update.

    optax.masked alone passes unselected gradients through unchanged. The second
    transform is required: it zeros those updates, including any decay/momentum.
    The first transform omits frozen leaves from optimizer slots and group norms.
    This freezes updates; it does not eliminate frozen-weight backward compute.
    """

    def build(self, num_train_steps):
        base = super().build(num_train_steps)

        def frozen_mask(params):
            return jax.tree.map(lambda active: not active, attention_head_mask(params))

        return optax.chain(
            optax.masked(base, attention_head_mask),
            optax.masked(optax.set_to_zero(), frozen_mask),
        )
