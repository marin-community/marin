# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""NanoGPT architecture enhanced with grug MoE features.

Same base as model.py (12L, 768d, head_dim=128, half-truncated RoPE, logit
soft-cap, bias on all linears) but adds:
- XSA (Exclusive Self-Attention): subtract v-parallel component from attn output
- Attention gate: learned headwise sigmoid gate (zero-init)
- QK gain of 1.3 (multiplied after QK norm + RoPE)
- GatedNorm: rank-128 learnable gating on RMS-normalized input
- SwiGLU MLP: gate + up + down at 3x hidden_dim (replaces squared ReLU 4x)
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
from levanter.grug.attention import AttentionMask, align_kv_heads, attention
from levanter.grug.loss import fused_linear_softmax_cross_entropy_loss
from levanter.grug.sharding import Pbatch, Pembed_vocab, Plm_head

from experiments.grug.nanogpt.model import (
    LAYER_NORM_EPS,
    LOGIT_CAP,
    MODEL_DIM,
    NUM_HEADS,
    NUM_LAYERS,
    ROPE_BASE,
    SEQ_LEN,
    VOCAB_SIZE,
    LinearWithBias,
    RMSNorm,
    _apply_half_truncated_rope,
    _init_weight,
    rms_norm,
)

HEAD_DIM = 128
GRUG_INTERMEDIATE_DIM = 3 * MODEL_DIM  # 2304 (SwiGLU 3x, replaces 4x squared ReLU)
GATED_NORM_RANK = 128
QK_GAIN = 1.3


@dataclass(frozen=True)
class GrugNanoGPTConfig:
    vocab_size: int = VOCAB_SIZE
    hidden_dim: int = MODEL_DIM
    intermediate_dim: int = GRUG_INTERMEDIATE_DIM
    num_layers: int = NUM_LAYERS
    num_heads: int = NUM_HEADS
    head_dim: int = HEAD_DIM
    max_seq_len: int = SEQ_LEN
    layer_norm_eps: float = LAYER_NORM_EPS
    attn_scale: float = 0.12
    logit_cap: float = LOGIT_CAP
    rope_base: float = ROPE_BASE
    qk_gain: float = QK_GAIN
    gated_norm_rank: int = GATED_NORM_RANK
    zero_init_proj: bool = True


# ---- GatedNorm ----


class GatedNorm(eqx.Module):
    w_down: jax.Array
    w_up: jax.Array

    @staticmethod
    def init(hidden_dim: int, std: float, *, key: PRNGKeyArray) -> "GatedNorm":
        k_down, k_up = random.split(key)
        return GatedNorm(
            w_down=reshard(_init_weight(k_down, (hidden_dim, GATED_NORM_RANK), std), P(None, None)),
            w_up=reshard(_init_weight(k_up, (GATED_NORM_RANK, hidden_dim), std), P(None, None)),
        )

    @named_call
    def __call__(self, x: Float[Array, "... D"]) -> Float[Array, "... D"]:
        gate_hidden = jnp.einsum("...d,dr->...r", x, self.w_down)
        gate_hidden = jax.nn.silu(gate_hidden)
        gate = jax.nn.sigmoid(jnp.einsum("...r,rd->...d", gate_hidden, self.w_up))
        return x * gate.astype(x.dtype)


# ---- Attention with XSA + gate ----


class CausalSelfAttention(eqx.Module):
    q: LinearWithBias
    k: LinearWithBias
    v: LinearWithBias
    proj: LinearWithBias
    attn_gate: jax.Array  # (hidden_dim, num_heads), zero-init
    cfg: GrugNanoGPTConfig = eqx.field(static=True)

    @staticmethod
    def init(cfg: GrugNanoGPTConfig, *, key: PRNGKeyArray) -> "CausalSelfAttention":
        k_q, k_k, k_v, k_proj = random.split(key, 4)
        hdim = cfg.num_heads * cfg.head_dim
        std = 0.02
        return CausalSelfAttention(
            q=LinearWithBias.init(cfg.hidden_dim, hdim, std, key=k_q, weight_pspec=P("data", "model")),
            k=LinearWithBias.init(cfg.hidden_dim, hdim, std, key=k_k, weight_pspec=P("data", "model")),
            v=LinearWithBias.init(cfg.hidden_dim, hdim, std, key=k_v, weight_pspec=P("data", "model")),
            proj=LinearWithBias.init(
                hdim, cfg.hidden_dim, std, key=k_proj, weight_pspec=P("model", "data"), zero_init=cfg.zero_init_proj
            ),
            attn_gate=reshard(jnp.zeros((cfg.hidden_dim, cfg.num_heads)), P(None, None)),
            cfg=cfg,
        )

    @named_call
    def __call__(self, x: Float[Array, "B S D"], mask: AttentionMask | jax.Array) -> Float[Array, "B S D"]:
        cfg = self.cfg
        h = cfg.head_dim

        def linear(mod, inp):
            return jnp.einsum("bsd,df->bsf", inp, mod.weight) + mod.bias

        q = rearrange(linear(self.q, x), "b s (n d) -> b s n d", d=h)
        k = rearrange(linear(self.k, x), "b s (n d) -> b s n d", d=h)
        v = rearrange(linear(self.v, x), "b s (n d) -> b s n d", d=h)
        # QK norm
        q, k = rms_norm(q), rms_norm(k)
        # Half-truncated RoPE
        q, k = _apply_half_truncated_rope(q, k, h, cfg.rope_base)
        # QK gain (applied after norm + RoPE, before attention scale)
        q = q * cfg.qk_gain
        # Attention scale
        levanter_scale = 1.0 / math.sqrt(h)
        q = q * (cfg.attn_scale / levanter_scale)
        attn_out = attention(q, k, v, mask)

        # XSA: subtract the component of attn_out parallel to v
        aligned_v = align_kv_heads(v, num_q_heads=attn_out.shape[2])
        dot = jnp.sum(attn_out * aligned_v, axis=-1, keepdims=True)
        v_norm_sq = jnp.sum(aligned_v * aligned_v, axis=-1, keepdims=True)
        attn_out = attn_out - (dot / (v_norm_sq + 1e-6)) * aligned_v

        # Headwise gate: sigmoid(x @ attn_gate) -> one scalar per head
        gate = 2 * jax.nn.sigmoid(jnp.einsum("bsd,dn->bsn", x, self.attn_gate))[..., None]
        attn_out = gate * attn_out

        attn_out = rearrange(attn_out, "b s n d -> b s (n d)")
        return jnp.einsum("bsd,df->bsf", attn_out, self.proj.weight, out_sharding=Pbatch) + self.proj.bias


# ---- SwiGLU MLP ----


class SwiGLUMLP(eqx.Module):
    w_gate: LinearWithBias
    w_up: LinearWithBias
    w_down: LinearWithBias

    @staticmethod
    def init(cfg: GrugNanoGPTConfig, *, key: PRNGKeyArray) -> "SwiGLUMLP":
        k_gate, k_up, k_down = random.split(key, 3)
        std = 0.02
        return SwiGLUMLP(
            w_gate=LinearWithBias.init(
                cfg.hidden_dim, cfg.intermediate_dim, std, key=k_gate, weight_pspec=P("data", "model")
            ),
            w_up=LinearWithBias.init(
                cfg.hidden_dim, cfg.intermediate_dim, std, key=k_up, weight_pspec=P("data", "model")
            ),
            w_down=LinearWithBias.init(
                cfg.intermediate_dim,
                cfg.hidden_dim,
                std,
                key=k_down,
                weight_pspec=P("model", "data"),
                zero_init=cfg.zero_init_proj,
            ),
        )

    @named_call
    def __call__(self, x: Float[Array, "B S D"]) -> Float[Array, "B S D"]:
        gate = jnp.einsum("bsd,df->bsf", x, self.w_gate.weight) + self.w_gate.bias
        up = jnp.einsum("bsd,df->bsf", x, self.w_up.weight) + self.w_up.bias
        h = jax.nn.silu(gate) * up
        return jnp.einsum("bsf,fd->bsd", h, self.w_down.weight, out_sharding=Pbatch) + self.w_down.bias


# ---- Block ----


class Block(eqx.Module):
    attn: CausalSelfAttention
    mlp: SwiGLUMLP
    norm1: RMSNorm
    norm2: RMSNorm
    attn_gated_norm: GatedNorm
    mlp_gated_norm: GatedNorm

    @staticmethod
    def init(cfg: GrugNanoGPTConfig, *, key: PRNGKeyArray) -> "Block":
        k_attn, k_mlp, k_gn_attn, k_gn_mlp = random.split(key, 4)
        return Block(
            attn=CausalSelfAttention.init(cfg, key=k_attn),
            mlp=SwiGLUMLP.init(cfg, key=k_mlp),
            norm1=RMSNorm.init(cfg.hidden_dim),
            norm2=RMSNorm.init(cfg.hidden_dim),
            attn_gated_norm=GatedNorm.init(cfg.hidden_dim, 0.02, key=k_gn_attn),
            mlp_gated_norm=GatedNorm.init(cfg.hidden_dim, 0.02, key=k_gn_mlp),
        )

    @named_call
    def __call__(self, x: Float[Array, "B S D"], mask: AttentionMask | jax.Array) -> Float[Array, "B S D"]:
        x = x + self.attn(self.attn_gated_norm(self.norm1(x)), mask)
        x = x + self.mlp(self.mlp_gated_norm(self.norm2(x)))
        return x


# ---- Transformer ----


class Transformer(eqx.Module):
    embed: jax.Array
    blocks: tuple[Block, ...]
    proj: LinearWithBias
    norm1: RMSNorm  # pre-embed norm
    embed_gated_norm: GatedNorm
    norm2: RMSNorm  # final norm
    final_gated_norm: GatedNorm
    config: GrugNanoGPTConfig = eqx.field(static=True)

    @staticmethod
    def init(cfg: GrugNanoGPTConfig, *, key: PRNGKeyArray) -> "Transformer":
        k_embed, k_proj, k_gn_embed, k_gn_final, *block_keys = random.split(key, cfg.num_layers + 4)
        embed = reshard(_init_weight(k_embed, (cfg.vocab_size, cfg.hidden_dim), 0.02), Pembed_vocab)
        proj = LinearWithBias.init(
            cfg.hidden_dim, cfg.vocab_size, 0.02, key=k_proj, weight_pspec=Plm_head, zero_init=cfg.zero_init_proj
        )
        blocks = tuple(Block.init(cfg, key=bk) for bk in block_keys)
        return Transformer(
            embed=embed,
            blocks=blocks,
            proj=proj,
            norm1=RMSNorm.init(cfg.hidden_dim),
            embed_gated_norm=GatedNorm.init(cfg.hidden_dim, 0.02, key=k_gn_embed),
            norm2=RMSNorm.init(cfg.hidden_dim),
            final_gated_norm=GatedNorm.init(cfg.hidden_dim, 0.02, key=k_gn_final),
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
        hidden = self.embed_gated_norm(self.norm1(self.embed.at[token_ids].get(out_sharding=Pbatch)))
        for block in self.blocks:
            hidden = eqx.filter_checkpoint(block)(hidden, mask)
        return self.final_gated_norm(self.norm2(hidden))

    @named_call
    def logits(
        self,
        token_ids: Int[Array, "B S"],
        mask: AttentionMask | jax.Array | None = None,
    ) -> Float[Array, "B S V"]:
        hidden = self(token_ids, mask=mask)
        raw = jnp.einsum("bsd,dv->bsv", hidden, self.proj.weight) + self.proj.bias
        raw = raw.astype(jnp.float32)
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

        return fused_linear_softmax_cross_entropy_loss(
            hidden,
            self.proj.weight,
            labels,
            weight=loss_weight,
            reduction=reduction,
            logsumexp_weight=logsumexp_weight,
            logit_soft_cap=self.config.logit_cap,
            dtype=loss_dtype,
        )
