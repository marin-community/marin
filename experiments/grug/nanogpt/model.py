# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""NanoGPT-style dense transformer, matching the modded-nanogpt reference exactly.

Architecture: 12-layer, 768-dim, 6 heads (head_dim=128), squared ReLU MLP (4x),
QK-norm, half-truncated RoPE (base=1/1024), attention scale=0.12, logit soft-cap=15,
parametric RMSNorm, bias on all linear layers, zero-init projections.

Training: batch=512 seqs x 1024 tokens = 524288 tok/step, 3600 steps, ~1.89B tokens.
Schedule: no warmup, constant 30%, linear decay 70%.
Optimizer: AdamW for embed/head/1D, Muon for 2D block params.
"""

import math
from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
from einops import rearrange
from haliax.jax_utils import named_call
from jax import random
from jax.sharding import PartitionSpec as P
from jax.sharding import reshard
from jaxtyping import Array, Float, Int, PRNGKeyArray
from levanter.grug.attention import AttentionMask, attention
from levanter.grug.sharding import Pbatch, Pembed_vocab, Plm_head, unshard

# ---- Constants matching nanogpt_ref.py ----
VOCAB_SIZE = 50304
NUM_LAYERS = 12
MODEL_DIM = 768
HEAD_DIM = 128
NUM_HEADS = MODEL_DIM // HEAD_DIM  # 6
INTERMEDIATE_DIM = 4 * MODEL_DIM  # 3072
SEQ_LEN = 1024
BATCH_SIZE = 512  # sequences per step (= 524288 tokens / 1024)
TRAIN_STEPS = 3600
ATTN_SCALE = 0.12
LOGIT_CAP = 15.0
ROPE_BASE = 1.0 / 1024.0
LAYER_NORM_EPS = 1e-5


@dataclass(frozen=True)
class NanoGPTConfig:
    vocab_size: int = VOCAB_SIZE
    hidden_dim: int = MODEL_DIM
    intermediate_dim: int = INTERMEDIATE_DIM
    num_layers: int = NUM_LAYERS
    num_heads: int = NUM_HEADS
    head_dim: int = HEAD_DIM
    max_seq_len: int = SEQ_LEN
    layer_norm_eps: float = LAYER_NORM_EPS
    attn_scale: float = ATTN_SCALE
    logit_cap: float = LOGIT_CAP
    rope_base: float = ROPE_BASE


def _init_weight(key: PRNGKeyArray, shape: tuple[int, ...], std: float) -> Float[Array, "..."]:
    return std * random.truncated_normal(key, -3, 3, shape)


def _init_zeros(shape: tuple[int, ...]) -> jax.Array:
    return jnp.zeros(shape, dtype=jnp.float32)


# ---- Half-truncated RoPE ----
# First half of head_dim uses frequencies (1/1024)^linspace(0,1,dim//4),
# second half is zeros (no rotation).


def _build_rope_freqs(head_dim: int, base: float) -> jax.Array:
    """Build half-truncated RoPE frequencies. Returns shape (head_dim // 2,)."""
    quarter = head_dim // 4
    angular_freq = base ** jnp.linspace(0, 1, quarter)
    # Second quarter is zeros (no rotation for upper half of head)
    return jnp.concatenate([angular_freq, jnp.zeros(quarter)])


def _apply_half_truncated_rope(
    q: Float[Array, "B S N D"], k: Float[Array, "B S M D"], head_dim: int, base: float
) -> tuple[jax.Array, jax.Array]:
    seq_len = q.shape[1]
    freqs = _build_rope_freqs(head_dim, base)  # (head_dim // 2,)
    pos = jnp.arange(seq_len, dtype=jnp.float32)
    theta = jnp.outer(pos, freqs)  # (S, head_dim // 2)
    cos = jnp.cos(theta)[None, :, None, :]  # (1, S, 1, D/2)
    sin = jnp.sin(theta)[None, :, None, :]

    def rotate(x):
        x = x.astype(jnp.float32)
        x1, x2 = jnp.split(x, 2, axis=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return jnp.concatenate([y1, y2], axis=-1).astype(x.dtype)

    return rotate(q), rotate(k)


# ---- Parametric RMSNorm (with learnable gains) ----


class RMSNorm(eqx.Module):
    weight: jax.Array
    eps: float = eqx.field(static=True)

    @staticmethod
    def init(dim: int, eps: float = LAYER_NORM_EPS) -> "RMSNorm":
        return RMSNorm(weight=jnp.ones((dim,), dtype=jnp.float32), eps=eps)

    @named_call
    def __call__(self, x: Float[Array, "... D"]) -> Float[Array, "... D"]:
        weight = unshard(self.weight)
        dtype = x.dtype
        x = x.astype(jnp.float32)
        variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        normed = x * jax.lax.rsqrt(variance + self.eps)
        return (normed * weight).astype(dtype)


def rms_norm(x: jax.Array) -> jax.Array:
    """Non-parametric RMS norm for QK norm."""
    variance = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
    return (x * jax.lax.rsqrt(variance + 1e-6)).astype(x.dtype)


# ---- Linear with bias ----


class LinearWithBias(eqx.Module):
    weight: jax.Array
    bias: jax.Array

    @staticmethod
    def init(
        in_dim: int,
        out_dim: int,
        std: float,
        *,
        key: PRNGKeyArray,
        weight_pspec: P = P(None, None),
        zero_init: bool = False,
    ) -> "LinearWithBias":
        if zero_init:
            weight = jnp.zeros((in_dim, out_dim), dtype=jnp.float32)
        else:
            weight = _init_weight(key, (in_dim, out_dim), std)
        return LinearWithBias(
            weight=reshard(weight, weight_pspec),
            bias=jnp.zeros((out_dim,), dtype=jnp.float32),
        )


# ---- Attention ----


class CausalSelfAttention(eqx.Module):
    q: LinearWithBias
    k: LinearWithBias
    v: LinearWithBias
    proj: LinearWithBias
    cfg: NanoGPTConfig = eqx.field(static=True)

    @staticmethod
    def init(cfg: NanoGPTConfig, *, key: PRNGKeyArray) -> "CausalSelfAttention":
        k_q, k_k, k_v, k_proj = random.split(key, 4)
        hdim = cfg.num_heads * cfg.head_dim
        std = 0.02
        return CausalSelfAttention(
            q=LinearWithBias.init(cfg.hidden_dim, hdim, std, key=k_q, weight_pspec=P("data", "model")),
            k=LinearWithBias.init(cfg.hidden_dim, hdim, std, key=k_k, weight_pspec=P("data", "model")),
            v=LinearWithBias.init(cfg.hidden_dim, hdim, std, key=k_v, weight_pspec=P("data", "model")),
            proj=LinearWithBias.init(
                hdim, cfg.hidden_dim, std, key=k_proj, weight_pspec=P("model", "data"), zero_init=True
            ),
            cfg=cfg,
        )

    @named_call
    def __call__(self, x: Float[Array, "B S D"], mask: AttentionMask | jax.Array) -> Float[Array, "B S D"]:
        cfg = self.cfg
        h = cfg.head_dim

        def linear(mod, x):
            return jnp.einsum("bsd,df->bsf", x, mod.weight) + mod.bias

        q = rearrange(linear(self.q, x), "b s (n d) -> b s n d", d=h)
        k = rearrange(linear(self.k, x), "b s (n d) -> b s n d", d=h)
        v = rearrange(linear(self.v, x), "b s (n d) -> b s n d", d=h)
        # QK norm (non-parametric)
        q, k = rms_norm(q), rms_norm(k)
        # Half-truncated RoPE
        q, k = _apply_half_truncated_rope(q, k, h, cfg.rope_base)
        # Attention with fixed scale=0.12 (ref uses this instead of 1/sqrt(d)).
        # Levanter's attention uses 1/sqrt(d) internally, so we pre-scale q to compensate.
        levanter_scale = 1.0 / math.sqrt(h)
        q = q * (cfg.attn_scale / levanter_scale)
        attn_out = attention(q, k, v, mask)
        attn_out = rearrange(attn_out, "b s n d -> b s (n d)")
        return jnp.einsum("bsd,df->bsf", attn_out, self.proj.weight, out_sharding=Pbatch) + self.proj.bias


# ---- MLP with squared ReLU ----


class MLP(eqx.Module):
    fc: LinearWithBias
    proj: LinearWithBias

    @staticmethod
    def init(cfg: NanoGPTConfig, *, key: PRNGKeyArray) -> "MLP":
        k_fc, k_proj = random.split(key, 2)
        std = 0.02
        return MLP(
            fc=LinearWithBias.init(cfg.hidden_dim, cfg.intermediate_dim, std, key=k_fc, weight_pspec=P("data", "model")),
            proj=LinearWithBias.init(
                cfg.intermediate_dim, cfg.hidden_dim, std, key=k_proj, weight_pspec=P("model", "data"), zero_init=True
            ),
        )

    @named_call
    def __call__(self, x: Float[Array, "B S D"]) -> Float[Array, "B S D"]:
        h = jnp.einsum("bsd,df->bsf", x, self.fc.weight) + self.fc.bias
        h = jax.nn.relu(h) ** 2  # squared ReLU
        return jnp.einsum("bsf,fd->bsd", h, self.proj.weight, out_sharding=Pbatch) + self.proj.bias


# ---- Block ----


class Block(eqx.Module):
    attn: CausalSelfAttention
    mlp: MLP
    norm1: RMSNorm
    norm2: RMSNorm

    @staticmethod
    def init(cfg: NanoGPTConfig, *, key: PRNGKeyArray) -> "Block":
        k_attn, k_mlp = random.split(key, 2)
        return Block(
            attn=CausalSelfAttention.init(cfg, key=k_attn),
            mlp=MLP.init(cfg, key=k_mlp),
            norm1=RMSNorm.init(cfg.hidden_dim),
            norm2=RMSNorm.init(cfg.hidden_dim),
        )

    @named_call
    def __call__(self, x: Float[Array, "B S D"], mask: AttentionMask | jax.Array) -> Float[Array, "B S D"]:
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.mlp(self.norm2(x))
        return x


# ---- Transformer ----


class Transformer(eqx.Module):
    embed: jax.Array  # (vocab, dim), bf16 in ref
    blocks: tuple[Block, ...]
    proj: LinearWithBias  # (dim, vocab), zero-init
    norm1: RMSNorm  # pre-embed norm
    norm2: RMSNorm  # final norm
    config: NanoGPTConfig = eqx.field(static=True)

    @staticmethod
    def init(cfg: NanoGPTConfig, *, key: PRNGKeyArray) -> "Transformer":
        k_embed, k_proj, *block_keys = random.split(key, cfg.num_layers + 2)
        embed = reshard(_init_weight(k_embed, (cfg.vocab_size, cfg.hidden_dim), 0.02), Pembed_vocab)
        proj = LinearWithBias.init(
            cfg.hidden_dim, cfg.vocab_size, 0.02, key=k_proj, weight_pspec=Plm_head, zero_init=True
        )
        blocks = tuple(Block.init(cfg, key=bk) for bk in block_keys)
        return Transformer(
            embed=embed,
            blocks=blocks,
            proj=proj,
            norm1=RMSNorm.init(cfg.hidden_dim),
            norm2=RMSNorm.init(cfg.hidden_dim),
            config=cfg,
        )

    @named_call
    def __call__(
        self,
        token_ids: Int[Array, "B S"],
        mask: AttentionMask | jax.Array | None = None,
    ) -> Float[Array, "B S D"]:
        if mask is None:
            mask = AttentionMask.causal()
        hidden = self.norm1(self.embed.at[token_ids].get(out_sharding=Pbatch))
        for block in self.blocks:
            hidden = eqx.filter_checkpoint(block)(hidden, mask)
        return self.norm2(hidden)

    @named_call
    def logits(
        self,
        token_ids: Int[Array, "B S"],
        mask: AttentionMask | jax.Array | None = None,
    ) -> Float[Array, "B S V"]:
        hidden = self(token_ids, mask=mask)
        raw = jnp.einsum("bsd,dv->bsv", hidden, self.proj.weight) + self.proj.bias
        raw = raw.astype(jnp.float32)
        # Logit soft-capping: 15 * x * rsqrt(x^2 + 15^2)
        cap = self.config.logit_cap
        return cap * raw * jax.lax.rsqrt(raw**2 + cap**2)

    def next_token_loss(
        self,
        token_ids: Int[Array, "B S"],
        loss_weight: Float[Array, "B S"],
        *,
        mask: AttentionMask | jax.Array | None = None,
        reduction: str = "mean",
        logsumexp_weight: float | None = None,
        loss_dtype: jnp.dtype = jnp.float32,
    ) -> jax.Array:
        hidden = self(token_ids, mask=mask)
        labels = jnp.concatenate([token_ids[:, 1:], jnp.zeros_like(token_ids[:, :1])], axis=1).astype(jnp.int32)
        loss_weight = loss_weight.astype(loss_dtype)

        # Compute logits with soft-cap
        raw = jnp.einsum("bsd,dv->bsv", hidden, self.proj.weight, out_sharding=Pbatch) + self.proj.bias
        raw = raw.astype(jnp.float32)
        cap = self.config.logit_cap
        logits = cap * raw * jax.lax.rsqrt(raw**2 + cap**2)

        # Cross-entropy (logits fully replicated on vocab axis)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        token_losses = -jnp.take_along_axis(log_probs, labels[..., None], axis=-1).squeeze(-1)
        weighted = token_losses * loss_weight
        if reduction == "mean":
            return jnp.sum(weighted) / jnp.maximum(jnp.sum(loss_weight), 1.0)
        return weighted


def debug_mesh_and_token_pspec(num_devices: int) -> tuple[jax.sharding.AbstractMesh, P]:
    mesh = jax.sharding.AbstractMesh(
        axis_sizes=(num_devices, 1),
        axis_names=("data", "model"),
        axis_types=(jax.sharding.AxisType.Explicit, jax.sharding.AxisType.Explicit),
    )
    return mesh, P(("data",), None)
