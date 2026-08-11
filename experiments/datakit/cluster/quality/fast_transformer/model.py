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

# Knuth-style multiplicative mixing constants for the hashed-bigram side table.
# They are part of the serialized model contract: a checkpoint hashed with these
# constants must be scored with them, so do not change them without retraining.
BIGRAM_MIX_A = 2654435761
BIGRAM_MIX_B = 40503


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
    # Hashed-bigram side table: adjacent token-id pairs hash into this many buckets
    # of embed_dim vectors, summed into the token embedding before pooling. 0
    # disables it. The seed and bucket count are serialized with the config so a
    # checkpoint is always scored with the hash it was trained with.
    bigram_buckets: int = 0
    bigram_seed: int = 0
    # Per-document embedding side input (e.g. a 1024-d harrier doc vector). 0
    # disables it; a non-zero dim makes ``doc_embed`` a required forward input, so
    # a checkpoint that was trained with doc embeddings fails loudly when scored
    # without them. ``doc_embed_super_token`` additionally appends the projected
    # vector as an extra always-valid super-token so attention can condition on
    # it; the head-side skip connection is present in both cases.
    doc_embed_dim: int = 0
    doc_embed_super_token: bool = False
    # Frozen donor embedding: when > 0 the token embedding is a frozen
    # [vocab, frozen_donor_dim] donor table read through a learned projection to
    # embed_dim, replacing the trainable [vocab, embed_dim] table. The donor
    # rows are filled in by the caller, and must be excluded from optimization
    # via ``train_regressor``'s ``params_filter`` (the forward also
    # stop-gradients them, but weight decay is only stopped by the filter).
    frozen_donor_dim: int = 0

    def __post_init__(self) -> None:
        if self.max_tokens % self.pool_window != 0:
            raise ValueError(f"max_tokens={self.max_tokens} must be divisible by pool_window={self.pool_window}")
        if self.pool_kind not in POOL_KINDS:
            raise ValueError(f"pool_kind={self.pool_kind} not in {POOL_KINDS}")
        if self.final_pool not in FINAL_POOLS:
            raise ValueError(f"final_pool={self.final_pool} not in {FINAL_POOLS}")
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(f"hidden_dim={self.hidden_dim} not divisible by num_heads={self.num_heads}")
        if self.doc_embed_super_token and not self.doc_embed_dim:
            raise ValueError("doc_embed_super_token requires doc_embed_dim > 0")

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
        if self.frozen_donor_dim:
            # Training-time cost only: at deployment the frozen donor table and
            # the learned projection fold into one [vocab, embed_dim] table,
            # recovering the base model's inference cost.
            total += 2 * self.frozen_donor_dim * self.embed_dim * t
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
    embed: Array | None  # [vocab, E]; None when a frozen donor table replaces it
    donor_embed: Array | None  # [vocab, frozen_donor_dim] frozen donor rows
    donor_proj: Array | None  # [frozen_donor_dim, E], learned
    bigram_embed: Array | None  # [bigram_buckets, E], or None when disabled
    pool_query: Array  # [E]
    proj_w: Array  # [pool_out_dim, D]
    proj_b: Array  # [D]
    pos_embed: Array  # [S, D]
    layers: list[TransformerLayer]
    final_query: Array  # [D]
    head_g: Array
    head_b: Array
    head_w: Array  # [D, 1]
    # Per-document embedding side input (None unless config.doc_embed_dim > 0).
    doc_proj_w: Array | None  # [doc_embed_dim, D]
    doc_proj_b: Array | None  # [D]
    doc_ln_g: Array | None  # [D]
    doc_ln_b: Array | None  # [D]
    doc_head_w: Array | None  # [D, 1], zero-init so training starts as the base model
    doc_type_embed: Array | None  # [D], marks the appended super-token
    doc_gate: Array | None  # scalar, zero-init gate on the appended super-token

    def __init__(self, config: FastTransformerConfig, *, key: PRNGKeyArray):
        ke, kpq, kpr, kpos, klayers, kfq, khead = jax.random.split(key, 7)
        # Folded rather than added to the split so the base weight stream (and thus
        # any retrain of an existing arm) is unchanged when doc embeddings are off.
        kdoc_proj, kdoc_type = jax.random.split(jax.random.fold_in(key, 7))
        self.config = config
        if config.frozen_donor_dim:
            self.embed = None
            # Zeros, not random: the caller fills the donor rows, and tokens the
            # donor lacks (reserved ids, tail specials) then embed to zero.
            self.donor_embed = jnp.zeros((config.vocab_size, config.frozen_donor_dim))
            self.donor_proj = _glorot(jax.random.fold_in(key, 8), (config.frozen_donor_dim, config.embed_dim))
        else:
            self.embed = jax.random.normal(ke, (config.vocab_size, config.embed_dim)) * 0.02
            self.donor_embed = None
            self.donor_proj = None
        # Zero init keeps the side table a no-op at the start of training; each
        # bucket then learns a residual on top of the unigram embedding.
        self.bigram_embed = jnp.zeros((config.bigram_buckets, config.embed_dim)) if config.bigram_buckets else None
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
        if config.doc_embed_dim:
            self.doc_proj_w = _glorot(kdoc_proj, (config.doc_embed_dim, config.hidden_dim))
            self.doc_proj_b = jnp.zeros(config.hidden_dim)
            self.doc_ln_g = jnp.ones(config.hidden_dim)
            self.doc_ln_b = jnp.zeros(config.hidden_dim)
            # Zero-init head skip and super-token gate: the forward is exactly the
            # base model at step 0, and the doc-embedding path fades in by gradient.
            self.doc_head_w = jnp.zeros((config.hidden_dim, 1))
            self.doc_type_embed = (
                jax.random.normal(kdoc_type, (config.hidden_dim,)) * 0.02 if config.doc_embed_super_token else None
            )
            self.doc_gate = jnp.zeros(()) if config.doc_embed_super_token else None
        else:
            self.doc_proj_w = None
            self.doc_proj_b = None
            self.doc_ln_g = None
            self.doc_ln_b = None
            self.doc_head_w = None
            self.doc_type_embed = None
            self.doc_gate = None

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

    def _bigram_side(self, ids: Array) -> Array:
        """Hashed adjacent-token-pair embeddings, added at each pair's left token.

        The bucket of pair ``(a, b)`` is a fixed seeded multiplicative hash, so the
        mapping is reproducible from the config alone. Pairs touching PAD get no
        side vector, and the final position (which has no right neighbour) is zero.
        """
        cfg = self.config
        a = ids[:, :-1].astype(jnp.uint32)
        b = ids[:, 1:].astype(jnp.uint32)
        h = a * jnp.uint32(BIGRAM_MIX_A) + b * jnp.uint32(BIGRAM_MIX_B) + jnp.uint32(cfg.bigram_seed)
        bucket = (h % jnp.uint32(cfg.bigram_buckets)).astype(jnp.int32)
        pair_valid = ((ids[:, :-1] != PAD_ID) & (ids[:, 1:] != PAD_ID)).astype(jnp.float32)
        side = jnp.take(self.bigram_embed, bucket, axis=0) * pair_valid[..., None]
        return jnp.pad(side, ((0, 0), (0, 1), (0, 0)))

    def __call__(
        self,
        ids: Array,
        *,
        doc_embed: Array | None = None,
        key: PRNGKeyArray | None = None,
        inference: bool = True,
    ) -> Array:
        cfg = self.config
        if bool(cfg.doc_embed_dim) != (doc_embed is not None):
            raise ValueError(
                f"model has doc_embed_dim={cfg.doc_embed_dim} but doc_embed "
                f"{'is missing' if doc_embed is None else 'was passed'}; the two must agree"
            )
        mask = (ids != PAD_ID).astype(jnp.float32)  # [b, t]
        if cfg.frozen_donor_dim:
            # stop_gradient guards the gather; the optimizer-side params_filter is
            # still required so weight decay cannot erode the frozen table.
            donor = jnp.take(jax.lax.stop_gradient(self.donor_embed), ids, axis=0)
            emb = _matmul(donor, self.donor_proj)  # [b, t, e]
        else:
            emb = jnp.take(self.embed, ids, axis=0)  # [b, t, e]
        if self.bigram_embed is not None:
            emb = emb + self._bigram_side(ids)

        pooled, valid = self._pool_windows(emb, mask)  # [b, s, pool_out], [b, s]
        h = _matmul(pooled, self.proj_w) + self.proj_b + self.pos_embed  # [b, s, d]

        doc_vec = None
        if doc_embed is not None:
            doc_vec = _layer_norm(_matmul(doc_embed, self.doc_proj_w) + self.doc_proj_b, self.doc_ln_g, self.doc_ln_b)
            if cfg.doc_embed_super_token:
                token = self.doc_gate * (doc_vec + self.doc_type_embed)  # [b, d]
                h = jnp.concatenate([h, token[:, None, :]], axis=1)  # [b, s+1, d]
                valid = jnp.concatenate([valid, jnp.ones((valid.shape[0], 1))], axis=1)

        n = cfg.num_layers
        layer_keys = [None] * n if key is None else list(jax.random.split(key, n)) if n else []
        for layer, lk in zip(self.layers, layer_keys, strict=True):
            h = layer(h, valid, key=lk, inference=inference)

        if doc_vec is not None and cfg.doc_embed_super_token:
            # The doc token is a conditioning input the real tokens attend to, not
            # document content: keep it out of the final pool so the head-side skip
            # stays the only direct readout of the embedding.
            h, valid = h[:, : cfg.num_super_tokens], valid[:, : cfg.num_super_tokens]

        if cfg.final_pool == "mean":
            pooled_doc = (h * valid[..., None]).sum(axis=1) / jnp.maximum(valid.sum(axis=1, keepdims=True), 1.0)
        else:  # attn pool over super-tokens
            scores = (h @ self.final_query) / math.sqrt(cfg.hidden_dim)  # [b, s]
            scores = jnp.where(valid > 0, scores, NEG_INF)
            attn = jax.nn.softmax(scores, axis=1)
            pooled_doc = jnp.einsum("bs,bsd->bd", attn, h)

        normed = _layer_norm(pooled_doc, self.head_g, self.head_b)
        logit = _matmul(normed, self.head_w)[:, 0]  # [b]
        if doc_vec is not None:
            # Head-side skip: the concat([pooled_doc, doc_vec]) head, decomposed into
            # a sum of two linears so the base head (and its zero-init identity to
            # the no-embedding model) is untouched.
            logit = logit + _matmul(doc_vec, self.doc_head_w)[:, 0]
        return logit


def count_params(model: FastTransformer) -> int:
    return sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_inexact_array)))
