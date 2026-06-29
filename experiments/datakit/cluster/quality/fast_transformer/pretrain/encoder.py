# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A token-level causal transformer encoder for pretrain-then-finetune.

The pooled :class:`~.model.FastTransformer` is cheap but data-limited: it
memorizes the 5.6k-doc oracle set (train loss -> 0, holdout Spearman plateaus at
~0.70 across architecture/capacity/context sweeps). The lever is to learn the
representation from *free* unlabeled text first.

This module is the standard recipe for that: a small GPT-style token-level
encoder we pretrain with next-token prediction (NTP) on a large free corpus,
then fine-tune for quality with a mean-pooled head. Unlike the pooled model,
attention runs over real tokens (causal), so NTP is a natural objective and the
embedding table -- the bulk of the parameters that otherwise overfit -- is
trained on billions of tokens instead of thousands of documents.

The classifier exposes the same ``(ids, key, inference) -> logit`` call as
:class:`~.model.FastTransformer`, so it reuses ``train_regressor`` / ``_predict``
/ ``_metrics`` unchanged.
"""

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from experiments.datakit.cluster.quality.fast_transformer.data import PAD_ID
from experiments.datakit.cluster.quality.fast_transformer.model import TransformerLayer, _layer_norm, _matmul


@dataclass(frozen=True)
class EncoderConfig:
    vocab_size: int
    max_tokens: int = 1024
    dim: int = 256
    num_layers: int = 4
    num_heads: int = 8
    mlp_ratio: int = 4
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.dim % self.num_heads != 0:
            raise ValueError(f"dim={self.dim} not divisible by num_heads={self.num_heads}")

    def flops_per_token(self) -> float:
        """Forward FLOPs per token (multiply-add = 2). Token-level attention is
        O(T) per token, unlike the pooled model's amortized cost."""
        d, t, d_ff = self.dim, self.max_tokens, self.dim * self.mlp_ratio
        attn_proj = 2 * (4 * d * d)
        attn_scores = 2 * (2 * t * d)  # QK^T over T keys + AV
        mlp = 2 * (2 * d * d_ff)
        head = 2 * d
        return self.num_layers * (attn_proj + attn_scores + mlp) + head


class TokenEncoder(eqx.Module):
    """Embedding + learned positions + causal pre-norm transformer stack."""

    config: EncoderConfig = eqx.field(static=True)
    embed: Array  # [vocab, dim]
    pos_embed: Array  # [max_tokens, dim]
    layers: list[TransformerLayer]
    final_g: Array
    final_b: Array

    def __init__(self, config: EncoderConfig, *, key: PRNGKeyArray):
        ke, kpos, klayers = jax.random.split(key, 3)
        self.config = config
        self.embed = jax.random.normal(ke, (config.vocab_size, config.dim)) * 0.02
        self.pos_embed = jax.random.normal(kpos, (config.max_tokens, config.dim)) * 0.02
        layer_keys = jax.random.split(klayers, config.num_layers)
        self.layers = [
            TransformerLayer(config.dim, config.num_heads, config.mlp_ratio, config.dropout, causal=True, key=lk)
            for lk in layer_keys
        ]
        self.final_g = jnp.ones(config.dim)
        self.final_b = jnp.zeros(config.dim)

    def encode(self, ids: Array, *, key: PRNGKeyArray | None, inference: bool, dropout: float | None = None) -> Array:
        """Token ids [B, T] -> contextual hidden states [B, T, dim]."""
        t = ids.shape[1]
        valid = (ids != PAD_ID).astype(jnp.float32)  # [B, T]
        h = jnp.take(self.embed, ids, axis=0) + self.pos_embed[:t]
        n = len(self.layers)
        layer_keys = [None] * n if key is None else list(jax.random.split(key, n))
        for layer, lk in zip(self.layers, layer_keys, strict=True):
            h = layer(h, valid, key=lk, inference=inference, dropout=dropout)
        return _layer_norm(h, self.final_g, self.final_b)

    def lm_logits(self, hidden: Array) -> Array:
        """Next-token logits [B, T, vocab] via weight-tied embeddings."""
        return _matmul(hidden, self.embed.T)


class EncoderClassifier(eqx.Module):
    """A :class:`TokenEncoder` with a mean-pooled quality-regression head.

    ``dropout`` overrides the encoder's own rate at fine-tune time, so one
    (expensive) pretrained encoder can be fine-tuned across a dropout sweep.
    """

    encoder: TokenEncoder
    head_g: Array
    head_b: Array
    head_w: Array  # [dim, 1]
    dropout: float = eqx.field(static=True)

    def __init__(
        self,
        config: EncoderConfig,
        *,
        key: PRNGKeyArray,
        encoder: TokenEncoder | None = None,
        dropout: float | None = None,
    ):
        kenc, khead = jax.random.split(key)
        self.encoder = encoder if encoder is not None else TokenEncoder(config, key=kenc)
        self.head_g = jnp.ones(config.dim)
        self.head_b = jnp.zeros(config.dim)
        self.head_w = jax.random.normal(khead, (config.dim, 1)) * (config.dim**-0.5)
        self.dropout = config.dropout if dropout is None else dropout

    def __call__(self, ids: Array, *, key: PRNGKeyArray | None = None, inference: bool = True) -> Array:
        hidden = self.encoder.encode(ids, key=key, inference=inference, dropout=self.dropout)
        mask = (ids != PAD_ID).astype(jnp.float32)[..., None]  # [B, T, 1]
        pooled = (hidden * mask).sum(axis=1) / jnp.maximum(mask.sum(axis=1), 1.0)
        normed = _layer_norm(pooled, self.head_g, self.head_b)
        return _matmul(normed, self.head_w)[:, 0]
