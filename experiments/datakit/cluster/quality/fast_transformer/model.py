# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A small "fast-transformer" document-quality regressor.

Architecture (the pooling step is the point):

    token ids ──embed──▶ [B, T, E]
              ──pool over windows of ``pool_window``──▶ [B, S, E_pool]   (S = T / w)
              ──input proj + learned position──▶ [B, S, D]
              ──N pre-norm transformer layers──▶ [B, S, D]
              ──final pool over S──▶ [B, D]
              ──head──▶ scalar quality logit (sigmoid at the loss/eval)

Pooling at ``w``-token boundaries amortizes the transformer's per-token cost by
``w`` (~64x), which is what keeps the model under ~1M FLOPs/token while still
running real self-attention. ``pool_kind`` selects how a window of token
embeddings collapses to one super-token: plain ``mean`` / ``max``, the
multi-statistic ``meanmaxmin`` concat (captures spread, not just centroid, which
a bag-of-words mean cannot), or a learned ``attn`` pool.

The model is written batched (leading ``B`` axis) with explicit einsums and a
bf16 matmul cast so XLA emits dense MXU matmuls on TPU. ``PAD_ID`` (0) positions
are masked everywhere: pooling ignores them, empty windows become inactive
super-tokens, and attention never attends to inactive super-tokens.
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
COMPUTE_DTYPE = jnp.bfloat16


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
        proj = 2 * self.pool_out_dim * d * s  # input projection of pooled vectors
        attn_proj = 2 * (4 * d * d) * s  # qkv (3) + output (1) projections
        attn_scores = 2 * (2 * s * s * d)  # QK^T and AV
        mlp = 2 * (2 * d * d_ff) * s
        per_layer = attn_proj + attn_scores + mlp
        head = 2 * d  # final linear to scalar
        total = proj + self.num_layers * per_layer + head
        return total / t


def _glorot(key: PRNGKeyArray, shape: tuple[int, ...]) -> Array:
    fan_in, fan_out = shape[0], shape[-1]
    return jax.random.normal(key, shape) * math.sqrt(2.0 / (fan_in + fan_out))


def _matmul(x: Array, w: Array) -> Array:
    """``x @ w`` in bf16 (TPU MXU) with f32 accumulation/output."""
    out = jnp.matmul(x.astype(COMPUTE_DTYPE), w.astype(COMPUTE_DTYPE), preferred_element_type=jnp.float32)
    return out.astype(jnp.float32)


def _layer_norm(x: Array, gamma: Array, beta: Array) -> Array:
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mu) * jax.lax.rsqrt(var + 1e-5) * gamma + beta


def _dropout(x: Array, p: float, key: PRNGKeyArray | None, inference: bool) -> Array:
    if inference or p == 0.0 or key is None:
        return x
    keep = jax.random.bernoulli(key, 1.0 - p, x.shape)
    return jnp.where(keep, x / (1.0 - p), 0.0)


class TransformerLayer(eqx.Module):
    """Batched masked pre-norm transformer block over super-tokens."""

    ln1_g: Array
    ln1_b: Array
    ln2_g: Array
    ln2_b: Array
    wqkv: Array  # [D, 3D]
    wo: Array  # [D, D]
    w1: Array  # [D, D_ff]
    w2: Array  # [D_ff, D]
    num_heads: int = eqx.field(static=True)
    dropout: float = eqx.field(static=True)

    def __init__(self, dim: int, num_heads: int, mlp_ratio: int, dropout: float, *, key: PRNGKeyArray):
        kqkv, ko, k1, k2 = jax.random.split(key, 4)
        self.ln1_g = jnp.ones(dim)
        self.ln1_b = jnp.zeros(dim)
        self.ln2_g = jnp.ones(dim)
        self.ln2_b = jnp.zeros(dim)
        self.wqkv = _glorot(kqkv, (dim, 3 * dim))
        self.wo = _glorot(ko, (dim, dim))
        self.w1 = _glorot(k1, (dim, dim * mlp_ratio))
        self.w2 = _glorot(k2, (dim * mlp_ratio, dim))
        self.num_heads = num_heads
        self.dropout = dropout

    def __call__(
        self, x: Array, valid: Array, *, key: PRNGKeyArray | None, inference: bool, dropout: float | None = None
    ) -> Array:
        b, s, d = x.shape
        h, hd = self.num_heads, d // self.num_heads
        p = self.dropout if dropout is None else dropout
        ka, km = (None, None) if key is None else jax.random.split(key)

        normed = _layer_norm(x, self.ln1_g, self.ln1_b)
        qkv = _matmul(normed, self.wqkv).reshape(b, s, 3, h, hd)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]  # [b, s, h, hd]
        scores = jnp.einsum("bqhd,bkhd->bhqk", q, k) / math.sqrt(hd)
        scores = jnp.where(valid[:, None, None, :].astype(bool), scores, NEG_INF)
        attn = jax.nn.softmax(scores, axis=-1)
        ctx = jnp.einsum("bhqk,bkhd->bqhd", attn, v).reshape(b, s, d)
        x = x + _dropout(_matmul(ctx, self.wo), p, ka, inference)

        normed = _layer_norm(x, self.ln2_g, self.ln2_b)
        mlp = _matmul(jax.nn.gelu(_matmul(normed, self.w1)), self.w2)
        x = x + _dropout(mlp, p, km, inference)
        return x


class FastTransformer(eqx.Module):
    config: FastTransformerConfig = eqx.field(static=True)
    embed: Array  # [vocab, E]
    pool_query: Array  # [E]
    proj_w: Array  # [pool_out_dim, D]
    proj_b: Array  # [D]
    pos_embed: Array  # [S, D]
    layers: list[TransformerLayer]
    final_query: Array  # [D]
    head_g: Array
    head_b: Array
    head_w: Array  # [D, 1]

    def __init__(self, config: FastTransformerConfig, *, key: PRNGKeyArray):
        ke, kpq, kpr, kpos, klayers, kfq, khead = jax.random.split(key, 7)
        self.config = config
        self.embed = jax.random.normal(ke, (config.vocab_size, config.embed_dim)) * 0.02
        self.pool_query = jax.random.normal(kpq, (config.embed_dim,)) * 0.02
        self.proj_w = _glorot(kpr, (config.pool_out_dim, config.hidden_dim))
        self.proj_b = jnp.zeros(config.hidden_dim)
        self.pos_embed = jax.random.normal(kpos, (config.num_super_tokens, config.hidden_dim)) * 0.02
        layer_keys = jax.random.split(klayers, max(1, config.num_layers))
        self.layers = [
            TransformerLayer(config.hidden_dim, config.num_heads, config.mlp_ratio, config.dropout, key=lk)
            for lk in layer_keys[: config.num_layers]
        ]
        self.final_query = jax.random.normal(kfq, (config.hidden_dim,)) * 0.02
        self.head_g = jnp.ones(config.hidden_dim)
        self.head_b = jnp.zeros(config.hidden_dim)
        self.head_w = _glorot(khead, (config.hidden_dim, 1))

    def _pool_windows(self, emb: Array, mask: Array) -> tuple[Array, Array]:
        """Collapse windows of ``pool_window`` tokens. Returns (pooled, valid)."""
        cfg = self.config
        b = emb.shape[0]
        s, w, e = cfg.num_super_tokens, cfg.pool_window, cfg.embed_dim
        wemb = emb.reshape(b, s, w, e)
        wmask = mask.reshape(b, s, w)
        counts = wmask.sum(axis=2, keepdims=True)  # [b, s, 1]
        valid = (counts[..., 0] > 0).astype(jnp.float32)  # [b, s]
        denom = jnp.maximum(counts, 1.0)
        m3 = wmask[..., None]

        if cfg.pool_kind == "mean":
            pooled = (wemb * m3).sum(axis=2) / denom
        elif cfg.pool_kind == "max":
            pooled = jnp.where(m3 > 0, wemb, NEG_INF).max(axis=2)
            pooled = jnp.where(valid[..., None] > 0, pooled, 0.0)
        elif cfg.pool_kind == "meanmaxmin":
            mean = (wemb * m3).sum(axis=2) / denom
            mx = jnp.where(valid[..., None] > 0, jnp.where(m3 > 0, wemb, NEG_INF).max(axis=2), 0.0)
            mn = jnp.where(valid[..., None] > 0, jnp.where(m3 > 0, wemb, -NEG_INF).min(axis=2), 0.0)
            pooled = jnp.concatenate([mean, mx, mn], axis=-1)
        else:  # attn: learned query, softmax over the window
            scores = (wemb @ self.pool_query) / math.sqrt(e)  # [b, s, w]
            scores = jnp.where(wmask > 0, scores, NEG_INF)
            attn = jax.nn.softmax(scores, axis=2)
            pooled = jnp.einsum("bsw,bswe->bse", attn, wemb)
            pooled = jnp.where(valid[..., None] > 0, pooled, 0.0)
        return pooled, valid

    def __call__(self, ids: Array, *, key: PRNGKeyArray | None = None, inference: bool = True) -> Array:
        cfg = self.config
        mask = (ids != PAD_ID).astype(jnp.float32)  # [b, t]
        emb = jnp.take(self.embed, ids, axis=0)  # [b, t, e]

        pooled, valid = self._pool_windows(emb, mask)  # [b, s, pool_out], [b, s]
        h = _matmul(pooled, self.proj_w) + self.proj_b + self.pos_embed  # [b, s, d]

        n = cfg.num_layers
        layer_keys = [None] * n if key is None else list(jax.random.split(key, n)) if n else []
        for layer, lk in zip(self.layers, layer_keys, strict=True):
            h = layer(h, valid, key=lk, inference=inference)

        if cfg.final_pool == "mean":
            pooled_doc = (h * valid[..., None]).sum(axis=1) / jnp.maximum(valid.sum(axis=1, keepdims=True), 1.0)
        else:  # attn pool over super-tokens
            scores = (h @ self.final_query) / math.sqrt(cfg.hidden_dim)  # [b, s]
            scores = jnp.where(valid > 0, scores, NEG_INF)
            attn = jax.nn.softmax(scores, axis=1)
            pooled_doc = jnp.einsum("bs,bsd->bd", attn, h)

        normed = _layer_norm(pooled_doc, self.head_g, self.head_b)
        return _matmul(normed, self.head_w)[:, 0]  # [b]


def count_params(model: FastTransformer) -> int:
    return sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_inexact_array)))
