# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray
from levanter.grug.attention import AttentionMask
from levanter.grug.loss import fused_linear_softmax_cross_entropy_loss

from experiments.grug.base.model import GrugModelConfig, Transformer


class CrossDomainTransformer(eqx.Module):
    """Shared Grug transformer with an optional scientific-position input adapter."""

    backbone: Transformer
    scientific_position_embed: jax.Array

    @staticmethod
    def init(
        config: GrugModelConfig,
        *,
        scientific_position_count: int,
        key: PRNGKeyArray,
    ) -> CrossDomainTransformer:
        if scientific_position_count <= 0:
            raise ValueError(f"scientific_position_count must be positive, got {scientific_position_count}")
        backbone_key, position_key = jax.random.split(key)
        backbone = Transformer.init(config, key=backbone_key)
        scientific_position_embed = config.initializer_std * jax.random.truncated_normal(
            position_key,
            -3,
            3,
            (scientific_position_count, config.hidden_dim),
        )
        return CrossDomainTransformer(backbone, scientific_position_embed)

    def input_embeddings(
        self,
        token_ids: Int[Array, "B S"],
        scientific_position_ids: Int[Array, "B S"],
    ) -> Float[Array, "B S D"]:
        token_embeddings = self.backbone.embed_tokens(token_ids)
        safe_position_ids = jnp.maximum(scientific_position_ids, 0)
        position_embeddings = self.scientific_position_embed[safe_position_ids]
        position_embeddings = jnp.where(
            scientific_position_ids[..., None] >= 0,
            position_embeddings,
            jnp.zeros_like(position_embeddings),
        )
        return token_embeddings + position_embeddings

    def logits(
        self,
        token_ids: Int[Array, "B S"],
        scientific_position_ids: Int[Array, "B S"],
        *,
        mask: AttentionMask | jax.Array,
        rotary_position_ids: Int[Array, "B S"],
    ) -> Float[Array, "B S V"]:
        inputs = self.input_embeddings(token_ids, scientific_position_ids)
        hidden = self.backbone.from_embeddings(inputs, mask=mask, position_ids=rotary_position_ids)
        return jnp.einsum("bsh,hv->bsv", hidden, self.backbone.output_proj)

    def aligned_token_loss(
        self,
        token_ids: Int[Array, "B S"],
        scientific_position_ids: Int[Array, "B S"],
        target_ids: Int[Array, "B S"],
        loss_weights: Float[Array, "B S"],
        *,
        mask: AttentionMask | jax.Array,
        rotary_position_ids: Int[Array, "B S"],
        reduction: str = "mean",
    ) -> jax.Array:
        inputs = self.input_embeddings(token_ids, scientific_position_ids)
        hidden = self.backbone.from_embeddings(inputs, mask=mask, position_ids=rotary_position_ids)
        labels = jnp.where(target_ids >= 0, target_ids, 0).astype(jnp.int32)
        return fused_linear_softmax_cross_entropy_loss(
            hidden,
            self.backbone.output_proj,
            labels,
            weight=loss_weights.astype(jnp.float32),
            reduction=reduction,
            dtype=jnp.float32,
        )
