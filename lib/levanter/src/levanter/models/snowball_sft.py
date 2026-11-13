# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Explicit initialization and update controls for Snowball supervised tuning."""

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from levanter.models.snowball import SnowballLMHeadModel
from levanter.sharding import partition_spec_of
from levanter.utils.jax_utils import leaf_key_paths


@dataclass(frozen=True)
class SnowballSftConfig:
    special_token_ids: tuple[int, ...] = ()
    anchor_token_ids: tuple[int, ...] = ()
    special_token_lr_multiplier: float = 1.0
    freeze_router_bias: bool = True

    def __post_init__(self):
        if len(self.special_token_ids) != len(self.anchor_token_ids):
            raise ValueError("Every reseeded token requires one anchor")
        if self.special_token_lr_multiplier <= 0:
            raise ValueError("Special-token update multiplier must be positive")


def reseed_snowball_head(model: SnowballLMHeadModel, config: SnowballSftConfig) -> SnowballLMHeadModel:
    """Copy anchor output columns once at weights-only initialization, preserving input embeddings."""
    if not isinstance(model, SnowballLMHeadModel):
        raise ValueError("Snowball SFT controls require SnowballLMHeadModel")
    if not config.special_token_ids:
        return model
    vocab = model.transformer.output_proj.shape[-1]
    if any(i < 0 or i >= vocab for i in (*config.special_token_ids, *config.anchor_token_ids)):
        raise ValueError("Special-token or anchor ID is outside the model vocabulary")
    head = model.transformer.output_proj
    source = head
    for token, anchor in zip(config.special_token_ids, config.anchor_token_ids, strict=True):
        head = head.at[:, token].set(source[:, anchor], out_sharding=partition_spec_of(head))
    return eqx.tree_at(lambda m: m.transformer.output_proj, model, head)


def snowball_update_transform(config: SnowballSftConfig) -> optax.GradientTransformation:
    """Scale selected embedding/head updates and freeze router biases after the optimizer update."""

    def init(params):
        del params
        return optax.EmptyState()

    def update(updates, state, params=None):
        del params
        if config.special_token_ids:
            embed = updates.transformer.token_embed
            head = updates.transformer.output_proj
            for token in config.special_token_ids:
                embed = embed.at[token, :].multiply(
                    config.special_token_lr_multiplier, out_sharding=partition_spec_of(embed)
                )
                head = head.at[:, token].multiply(
                    config.special_token_lr_multiplier, out_sharding=partition_spec_of(head)
                )
            updates = eqx.tree_at(
                lambda m: (m.transformer.token_embed, m.transformer.output_proj), updates, (embed, head)
            )
        if config.freeze_router_bias:
            paths = leaf_key_paths(updates)
            updates = jax.tree.map(
                lambda value, path: jnp.zeros_like(value) if str(path).endswith("router_bias") else value,
                updates,
                paths,
            )
        return updates, state

    return optax.GradientTransformation(init, update)
