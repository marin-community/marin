# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A small "fast-transformer" document-quality regressor.

Architecture (the pooling step is the point):

    token ids ──embed──▶ [T, E]
              ──pool over windows of ``pool_window``──▶ [S, E_pool]   (S = T / w)
              ──input proj + learned position──▶ [S, D]
              ──N pre-norm transformer layers──▶ [S, D]
              ──final pool over S──▶ [D]
              ──head──▶ scalar quality (sigmoid)

Pooling at ``w``-token boundaries amortizes the transformer's per-token cost by
``w`` (~64x), which is what keeps the model under ~1M FLOPs/token while still
running real self-attention. ``pool_kind`` selects how a window of token
embeddings collapses to one super-token: plain ``mean`` / ``max``, the
multi-statistic ``meanmaxmin`` concat (captures spread, not just centroid, which
a bag-of-words mean cannot), or a learned ``attn`` pool.

The model is written per-example (no batch axis) and ``jax.vmap``-ed over the
batch by the trainer. ``PAD_ID`` (0) positions are masked everywhere: pooling
ignores them, empty windows become inactive super-tokens, and attention never
attends to inactive super-tokens.
"""

import math
from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from experiments.datakit.cluster.quality.fast_transformer.data import PAD_ID

POOL_KINDS = ("mean", "max", "meanmaxmin", "attn")
FINAL_POOLS = ("mean", "attn")
NEG_INF = -1e30


@dataclass(frozen=True)
class FastTransformerConfig:
    vocab_size: int
    max_tokens: int = 1024
    pool_window: int = 64
    pool_kind: str = "meanmaxmin"
    embed_dim: int = 256
    hidden_dim: int = 512
    num_layers: int = 4
    num_heads: int = 8
    mlp_ratio: int = 4
    dropout: float = 0.1
    final_pool: str = "mean"

    def __post_init__(self) -> None:
        if self.max_tokens % self.pool_window != 0:
            raise ValueError(f"max_tokens={self.max_tokens} must be divisible by pool_window={self.pool_window}")
        if self.pool_kind not in POOL_KINDS:
            raise ValueError(f"pool_kind={self.pool_kind} not in {POOL_KINDS}")
        if self.final_pool not in FINAL_POOLS:
            raise ValueError(f"final_pool={self.final_pool} not in {FINAL_POOLS}")
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(f"hidden_dim={self.hidden_dim} not divisible by num_heads={self.num_heads}")

    @property
    def num_super_tokens(self) -> int:
        return self.max_tokens // self.pool_window

    @property
    def pool_out_dim(self) -> int:
        return self.embed_dim * 3 if self.pool_kind == "meanmaxmin" else self.embed_dim

    def flops_per_token(self) -> float:
        """Forward FLOPs per *input* token (multiply-add counted as 2).

        Embedding lookup is a gather (~0 FLOPs). The dominant terms are the
        per-super-token linear layers (amortized by ``pool_window``) plus the
        (negligible) S^2 attention. This is the inference cost that matters when
        scoring a whole corpus, and the budget we hold under 1M.
        """
        d = self.hidden_dim
        s = self.num_super_tokens
        t = self.max_tokens
        d_ff = d * self.mlp_ratio
        # input projection of pooled vectors -> hidden
        proj = 2 * self.pool_out_dim * d * s
        attn_proj = 2 * (4 * d * d) * s  # qkv (3) + output (1) projections
        attn_scores = 2 * (2 * s * s * d)  # QK^T and AV
        mlp = 2 * (2 * d * d_ff) * s
        per_layer = attn_proj + attn_scores + mlp
        head = 2 * d  # final linear to scalar
        total = proj + self.num_layers * per_layer + head
        return total / t


def _vmap_linear(layer: eqx.nn.Linear, x: Array) -> Array:
    return jax.vmap(layer)(x)


class MultiHeadAttention(eqx.Module):
    """Masked multi-head self-attention over super-tokens."""

    qkv: eqx.nn.Linear
    out: eqx.nn.Linear
    num_heads: int = eqx.field(static=True)

    def __init__(self, dim: int, num_heads: int, *, key: PRNGKeyArray):
        k1, k2 = jax.random.split(key)
        self.qkv = eqx.nn.Linear(dim, 3 * dim, use_bias=True, key=k1)
        self.out = eqx.nn.Linear(dim, dim, use_bias=True, key=k2)
        self.num_heads = num_heads

    def __call__(self, x: Array, valid: Array) -> Array:
        s, d = x.shape
        h = self.num_heads
        hd = d // h
        qkv = _vmap_linear(self.qkv, x)  # [s, 3d]
        q, k, v = jnp.split(qkv, 3, axis=-1)
        q = q.reshape(s, h, hd).transpose(1, 0, 2)  # [h, s, hd]
        k = k.reshape(s, h, hd).transpose(1, 0, 2)
        v = v.reshape(s, h, hd).transpose(1, 0, 2)
        scores = jnp.einsum("hqd,hkd->hqk", q, k) / math.sqrt(hd)
        key_mask = valid.astype(bool)[None, None, :]  # [1, 1, s]
        scores = jnp.where(key_mask, scores, NEG_INF)
        attn = jax.nn.softmax(scores, axis=-1)
        ctx = jnp.einsum("hqk,hkd->hqd", attn, v)  # [h, s, hd]
        ctx = ctx.transpose(1, 0, 2).reshape(s, d)
        return _vmap_linear(self.out, ctx)


class MLP(eqx.Module):
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear

    def __init__(self, dim: int, hidden: int, *, key: PRNGKeyArray):
        k1, k2 = jax.random.split(key)
        self.fc1 = eqx.nn.Linear(dim, hidden, key=k1)
        self.fc2 = eqx.nn.Linear(hidden, dim, key=k2)

    def __call__(self, x: Array) -> Array:
        return _vmap_linear(self.fc2, jax.nn.gelu(_vmap_linear(self.fc1, x)))


class TransformerLayer(eqx.Module):
    norm1: eqx.nn.LayerNorm
    attn: MultiHeadAttention
    norm2: eqx.nn.LayerNorm
    mlp: MLP
    dropout: eqx.nn.Dropout

    def __init__(self, dim: int, num_heads: int, mlp_ratio: int, dropout: float, *, key: PRNGKeyArray):
        ka, km = jax.random.split(key)
        self.norm1 = eqx.nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads, key=ka)
        self.norm2 = eqx.nn.LayerNorm(dim)
        self.mlp = MLP(dim, dim * mlp_ratio, key=km)
        self.dropout = eqx.nn.Dropout(dropout)

    def __call__(self, x, valid, *, key, inference):
        ka, km = (None, None) if key is None else jax.random.split(key)
        normed = jax.vmap(self.norm1)(x)
        x = x + self.dropout(self.attn(normed, valid), key=ka, inference=inference)
        normed = jax.vmap(self.norm2)(x)
        x = x + self.dropout(self.mlp(normed), key=km, inference=inference)
        return x


class FastTransformer(eqx.Module):
    config: FastTransformerConfig = eqx.field(static=True)
    embed: eqx.nn.Embedding
    pool_query: Array  # [embed_dim], learned query for attn window pooling
    input_proj: eqx.nn.Linear
    pos_embed: Array  # [S, hidden]
    layers: list[TransformerLayer]
    final_query: Array  # [hidden], learned query for attn final pooling
    head_norm: eqx.nn.LayerNorm
    head: eqx.nn.Linear
    embed_dropout: eqx.nn.Dropout

    def __init__(self, config: FastTransformerConfig, *, key: PRNGKeyArray):
        keys = jax.random.split(key, 6)
        self.config = config
        self.embed = eqx.nn.Embedding(config.vocab_size, config.embed_dim, key=keys[0])
        self.pool_query = jax.random.normal(keys[1], (config.embed_dim,)) * 0.02
        self.input_proj = eqx.nn.Linear(config.pool_out_dim, config.hidden_dim, key=keys[2])
        self.pos_embed = jax.random.normal(keys[3], (config.num_super_tokens, config.hidden_dim)) * 0.02
        layer_keys = jax.random.split(keys[4], config.num_layers)
        self.layers = [
            TransformerLayer(config.hidden_dim, config.num_heads, config.mlp_ratio, config.dropout, key=lk)
            for lk in layer_keys
        ]
        self.final_query = jax.random.normal(keys[5], (config.hidden_dim,)) * 0.02
        self.head_norm = eqx.nn.LayerNorm(config.hidden_dim)
        self.head = eqx.nn.Linear(config.hidden_dim, 1, key=keys[0])
        self.embed_dropout = eqx.nn.Dropout(config.dropout)

    def _pool_windows(self, emb: Array, mask: Array):
        """Collapse windows of ``pool_window`` tokens to super-tokens.

        Returns (pooled [S, pool_out_dim], valid [S]).
        """
        cfg = self.config
        s, w, e = cfg.num_super_tokens, cfg.pool_window, cfg.embed_dim
        wemb = emb.reshape(s, w, e)
        wmask = mask.reshape(s, w)
        counts = wmask.sum(axis=1, keepdims=True)  # [s, 1]
        valid = (counts[:, 0] > 0).astype(jnp.float32)  # [s]
        denom = jnp.maximum(counts, 1.0)

        if cfg.pool_kind == "mean":
            pooled = (wemb * wmask[..., None]).sum(axis=1) / denom
        elif cfg.pool_kind == "max":
            pooled = jnp.where(wmask[..., None] > 0, wemb, NEG_INF).max(axis=1)
            pooled = jnp.where(valid[:, None] > 0, pooled, 0.0)
        elif cfg.pool_kind == "meanmaxmin":
            mean = (wemb * wmask[..., None]).sum(axis=1) / denom
            mx = jnp.where(wmask[..., None] > 0, wemb, NEG_INF).max(axis=1)
            mn = jnp.where(wmask[..., None] > 0, wemb, -NEG_INF).min(axis=1)
            mx = jnp.where(valid[:, None] > 0, mx, 0.0)
            mn = jnp.where(valid[:, None] > 0, mn, 0.0)
            pooled = jnp.concatenate([mean, mx, mn], axis=-1)
        else:  # attn: learned query, softmax over the window
            scores = (wemb @ self.pool_query) / math.sqrt(e)  # [s, w]
            scores = jnp.where(wmask > 0, scores, NEG_INF)
            attn = jax.nn.softmax(scores, axis=1)  # [s, w]
            pooled = jnp.einsum("sw,swe->se", attn, wemb)
            pooled = jnp.where(valid[:, None] > 0, pooled, 0.0)
        return pooled, valid

    def __call__(self, ids: Array, *, key=None, inference: bool = True) -> Array:
        cfg = self.config
        mask = (ids != PAD_ID).astype(jnp.float32)
        emb = jax.vmap(self.embed)(ids)  # [t, e]
        ek, lk_all = (None, None) if key is None else jax.random.split(key)
        emb = self.embed_dropout(emb, key=ek, inference=inference)

        pooled, valid = self._pool_windows(emb, mask)  # [s, pool_out], [s]
        h = _vmap_linear(self.input_proj, pooled) + self.pos_embed  # [s, d]

        layer_keys = [None] * cfg.num_layers if lk_all is None else list(jax.random.split(lk_all, cfg.num_layers))
        for layer, lk in zip(self.layers, layer_keys, strict=True):
            h = layer(h, valid, key=lk, inference=inference)

        if cfg.final_pool == "mean":
            pooled_doc = (h * valid[:, None]).sum(axis=0) / jnp.maximum(valid.sum(), 1.0)
        else:  # attn pool over super-tokens
            scores = (h @ self.final_query) / math.sqrt(cfg.hidden_dim)
            scores = jnp.where(valid > 0, scores, NEG_INF)
            attn = jax.nn.softmax(scores, axis=0)
            pooled_doc = jnp.einsum("s,sd->d", attn, h)

        logit = self.head(self.head_norm(pooled_doc))[0]
        return logit


def count_params(model: FastTransformer) -> int:
    return sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_inexact_array)))
