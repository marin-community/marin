# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Explicit initialization and update controls for Snowball supervised tuning."""

from dataclasses import dataclass

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import optax
from jax.sharding import PartitionSpec as P

from levanter.models.snowball import SnowballLMHeadModel
from levanter.models.lm_model import LmExample
from levanter.models.loss import maybe_fused_next_token_loss
from levanter.metrics import Metric, ReductionType
from levanter.sharding import partition_spec_of
from levanter.utils.jax_utils import leaf_key_paths


@dataclass(frozen=True)
class SnowballSftConfig:
    special_token_ids: tuple[int, ...] = ()
    anchor_token_ids: tuple[int, ...] = ()
    special_token_lr_multiplier: float = 1.0
    freeze_router_bias: bool = True
    record_dispatch_counts: bool = False

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
    source_spec = partition_spec_of(source)
    column_spec = P(source_spec[0]) if source_spec else P()
    for token, anchor in zip(config.special_token_ids, config.anchor_token_ids, strict=True):
        column = source.at[:, anchor].get(out_sharding=column_spec)
        head = head.at[:, token].set(column, out_sharding=partition_spec_of(head))
    return eqx.tree_at(lambda m: m.transformer.output_proj, model, head)


def snowball_loss_with_dispatch_counts(
    model: SnowballLMHeadModel,
    example: LmExample,
    *,
    z_loss_weight: float,
    reduction: hax.ReductionFunction,
) -> tuple[hax.NamedArray, dict[str, Metric]]:
    """Measure dispatch on the same unmasked-token forward used for training."""
    # All positions consume dispatch capacity in the regular training forward,
    # including padding. A supervision mask here would change that forward.
    valid = hax.ones(example.tokens.axes, dtype=jnp.bool_)
    hidden, stats = model.activations_with_stats(example.tokens, valid, attn_mask=example.attn_mask)
    loss = maybe_fused_next_token_loss(
        model.Pos,
        model.Embed,
        model.Vocab,
        hidden,
        model.get_lm_head(),
        example.tokens,
        loss_weight=example.loss_weight,
        reduction=reduction,
        logsumexp_weight=z_loss_weight,
    )
    return loss, {
        "moe/sender_dropped": Metric.from_value(jnp.sum(stats.sender_dropped), ReductionType.SUM),
        "moe/receiver_dropped": Metric.from_value(jnp.sum(stats.receiver_dropped), ReductionType.SUM),
    }


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
