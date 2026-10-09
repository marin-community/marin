# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Dense Qwen3 decoder with optional Parallax attention (arXiv 2605.29157), copied from ``grug/base``.

Matches torchtitan's Qwen3 (the paper's code base): SwiGLU MLP, per-head RMSNorm on ``q`` and ``k`` before RoPE,
RMSNorm eps 1e-6, tied input/output embeddings, and every weight drawn from ``N(0, 0.02)``. With ``parallax`` the
attention also projects a per-query-head probe ``rho = W_R x`` (RMSNorm'd, optionally RoPE'd like ``q``) and runs
``levanter.kernels.pallas.parallax``; otherwise it runs FA4 (``gpu_fa4_cute``) on GPU.
"""

import dataclasses
from dataclasses import dataclass
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from einops import rearrange
from haliax.jax_utils import named_call
from jax import random, shard_map
from jax.sharding import PartitionSpec as P
from jax.sharding import get_abstract_mesh, reshard
from jaxtyping import Array, Float, Int, PRNGKeyArray
from levanter.grug.attention import AttentionMask, RotaryConfig, apply_rotary_embedding, attention
from levanter.grug.loss import fused_linear_softmax_cross_entropy_loss
from levanter.grug.sharding import Pbatch, Pembed_vocab, Plm_head, Plogits, unshard
from levanter.kernels.pallas.parallax import parallax_attention
from levanter.sharding import partition_spec_of

AttentionImplementation = Literal["gpu_fa4_cute", "reference"]


@dataclass(frozen=True)
class GrugModelConfig:
    """Hyperparameters for the Qwen3-style transformer."""

    vocab_size: int
    hidden_dim: int
    intermediate_dim: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    max_seq_len: int
    parallax: bool
    parallax_rope: bool = True  # RoPE on rho (the paper's best 0.6B arm)
    attention_implementation: AttentionImplementation = "gpu_fa4_cute"
    layer_norm_eps: float = 1e-6
    initializer_std: float = 0.02
    rope: RotaryConfig = dataclasses.field(default_factory=lambda: RotaryConfig(theta=1e6))

    def __post_init__(self) -> None:
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads for grouped-query attention")
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")


def flops_per_token(cfg: GrugModelConfig) -> float:
    """Forward matmul FLOPs per token, with causal attention over a full ``max_seq_len`` row (half the keys)."""
    d, h, hkv, hd, seq = cfg.hidden_dim, cfg.num_heads, cfg.num_kv_heads, cfg.head_dim, cfg.max_seq_len
    q_proj = h * hd * (2 if cfg.parallax else 1)  # W_R matches W_Q's shape
    proj = 2 * d * (q_proj + 2 * hkv * hd) + 2 * h * hd * d
    mlp = 2 * 3 * d * cfg.intermediate_dim
    # QK^T and PV over seq/2 keys on average; Parallax adds R K^T and P2 V.
    attn = 2 * 2 * h * hd * (seq / 2) * (2 if cfg.parallax else 1)
    return cfg.num_layers * (proj + mlp + attn) + 2 * d * cfg.vocab_size


class RMSNorm(eqx.Module):
    weight: jax.Array
    eps: float = eqx.field(static=True)

    @staticmethod
    def init(dim: int, eps: float) -> "RMSNorm":
        return RMSNorm(weight=jnp.ones((dim,), dtype=jnp.float32), eps=eps)

    @named_call
    def __call__(self, x: Float[Array, "... D"]) -> Float[Array, "... D"]:
        weight = unshard(self.weight)
        dtype = x.dtype
        x = x.astype(jnp.float32)
        variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        normed = x * jax.lax.rsqrt(variance + self.eps)
        return (normed * weight).astype(dtype)


def _parallax(q, r, k, v, mask: AttentionMask) -> jax.Array:
    """Parallax over ``[B, S, H, D]`` queries/probes and ``[B, S, Hkv, D]`` keys/values, one kernel call per shard."""
    segment_ids = mask.segment_ids[0] if mask.segment_ids is not None else None
    scale = q.shape[-1] ** -0.5

    def local(q, r, k, v, *seg):
        implementation = "pallas_gpu" if jax.default_backend() == "gpu" else "reference"
        return parallax_attention(q, r, k, v, seg[0] if seg else None, sm_scale=scale, implementation=implementation)

    seg_args = () if segment_ids is None else (segment_ids.astype(jnp.int32),)
    mesh = get_abstract_mesh()
    if mesh is None or mesh.empty:
        return local(q, r, k, v, *seg_args)
    q_spec = tuple(partition_spec_of(q))
    out_spec = P(*q_spec, *((None,) * (q.ndim - len(q_spec))))
    seg_args = tuple(reshard(s, P(out_spec[0], None)) for s in seg_args)
    return shard_map(local, mesh=mesh, out_specs=out_spec, check_vma=False)(q, r, k, v, *seg_args)


class CausalSelfAttention(eqx.Module):
    w_q: jax.Array
    w_k: jax.Array
    w_v: jax.Array
    w_o: jax.Array
    q_norm: RMSNorm
    k_norm: RMSNorm
    w_r: jax.Array | None
    r_norm: RMSNorm | None
    cfg: GrugModelConfig = eqx.field(static=True)

    @staticmethod
    def init(cfg: GrugModelConfig, *, key: PRNGKeyArray) -> "CausalSelfAttention":
        k_q, k_k, k_v, k_o, k_r = random.split(key, 5)
        d, h, hkv, hd, std = cfg.hidden_dim, cfg.num_heads, cfg.num_kv_heads, cfg.head_dim, cfg.initializer_std
        return CausalSelfAttention(
            w_q=reshard(_init_weight(k_q, (d, h * hd), std), P("data", "model")),
            w_k=reshard(_init_weight(k_k, (d, hkv * hd), std), P("data", "model")),
            w_v=reshard(_init_weight(k_v, (d, hkv * hd), std), P("data", "model")),
            w_o=reshard(_init_weight(k_o, (h * hd, d), std), P("model", "data")),
            q_norm=RMSNorm.init(hd, cfg.layer_norm_eps),
            k_norm=RMSNorm.init(hd, cfg.layer_norm_eps),
            w_r=reshard(_init_weight(k_r, (d, h * hd), std), P("data", "model")) if cfg.parallax else None,
            r_norm=RMSNorm.init(hd, cfg.layer_norm_eps) if cfg.parallax else None,
            cfg=cfg,
        )

    @named_call
    def __call__(self, x: Float[Array, "B S D"], mask: AttentionMask) -> Float[Array, "B S D"]:
        hd = self.cfg.head_dim
        seq_len = x.shape[1]

        def heads(w):
            return rearrange(jnp.einsum("bsh,hd->bsd", x, w), "... (n d) -> ... n d", d=hd)

        q = self.q_norm(heads(self.w_q))
        k = self.k_norm(heads(self.w_k))
        v = heads(self.w_v)
        q, k = apply_rotary_embedding(q, k, seq_len=seq_len, head_dim=hd, rope=self.cfg.rope)
        if self.w_r is None:
            attn_out = attention(q, k, v, mask, implementation=self.cfg.attention_implementation)
        else:
            assert self.r_norm is not None
            r = self.r_norm(heads(self.w_r))
            if self.cfg.parallax_rope:
                r, _ = apply_rotary_embedding(r, r, seq_len=seq_len, head_dim=hd, rope=self.cfg.rope)
            attn_out = _parallax(q, r, k, v, mask)
        attn_out = rearrange(attn_out, "... n d -> ... (n d)")
        return jnp.einsum("bsh,hd->bsd", attn_out, self.w_o, out_sharding=Pbatch)


class MLP(eqx.Module):
    w_gate: jax.Array
    w_up: jax.Array
    w_down: jax.Array

    @staticmethod
    def init(cfg: GrugModelConfig, *, key: PRNGKeyArray) -> "MLP":
        k_gate, k_up, k_down = random.split(key, 3)
        d, d_ff, std = cfg.hidden_dim, cfg.intermediate_dim, cfg.initializer_std
        return MLP(
            w_gate=reshard(_init_weight(k_gate, (d, d_ff), std), P("data", "model")),
            w_up=reshard(_init_weight(k_up, (d, d_ff), std), P("data", "model")),
            w_down=reshard(_init_weight(k_down, (d_ff, d), std), P("model", "data")),
        )

    @named_call
    def __call__(self, x: Float[Array, "B S D"]) -> Float[Array, "B S D"]:
        gate = jnp.einsum("bsh,hm->bsm", x, self.w_gate)
        up = jnp.einsum("bsh,hm->bsm", x, self.w_up)
        return jnp.einsum("bsm,mh->bsh", jax.nn.silu(gate) * up, self.w_down, out_sharding=Pbatch)


class Block(eqx.Module):
    rms_attn: RMSNorm
    attn: CausalSelfAttention
    rms_mlp: RMSNorm
    mlp: MLP

    @staticmethod
    def init(cfg: GrugModelConfig, *, key: PRNGKeyArray) -> "Block":
        attn_key, mlp_key = random.split(key, 2)
        return Block(
            rms_attn=RMSNorm.init(cfg.hidden_dim, cfg.layer_norm_eps),
            attn=CausalSelfAttention.init(cfg, key=attn_key),
            rms_mlp=RMSNorm.init(cfg.hidden_dim, cfg.layer_norm_eps),
            mlp=MLP.init(cfg, key=mlp_key),
        )

    @named_call
    def __call__(self, x: Float[Array, "B S D"], mask: AttentionMask) -> Float[Array, "B S D"]:
        x = x + self.attn(self.rms_attn(x), mask)
        return x + self.mlp(self.rms_mlp(x))


class Transformer(eqx.Module):
    token_embed: jax.Array  # [V, D]; also the (tied) output projection
    blocks: tuple[Block, ...]
    final_norm: RMSNorm
    config: GrugModelConfig = eqx.field(static=True)

    @staticmethod
    def init(cfg: GrugModelConfig, *, key: PRNGKeyArray) -> "Transformer":
        embed_key, *block_keys = random.split(key, cfg.num_layers + 1)
        token_embed = reshard(
            _init_weight(embed_key, (cfg.vocab_size, cfg.hidden_dim), cfg.initializer_std), Pembed_vocab
        )
        return Transformer(
            token_embed=token_embed,
            blocks=tuple(Block.init(cfg, key=layer_key) for layer_key in block_keys),
            final_norm=RMSNorm.init(cfg.hidden_dim, cfg.layer_norm_eps),
            config=cfg,
        )

    @named_call
    def __call__(self, token_ids: Int[Array, "B S"], mask: AttentionMask | None = None) -> Float[Array, "B S D"]:
        if mask is None:
            mask = AttentionMask.causal()
        hidden = self.token_embed.at[token_ids].get(out_sharding=Pbatch)
        for i, block in enumerate(self.blocks):
            with jax.named_scope(f"block_{i}"):
                hidden = eqx.filter_checkpoint(block)(hidden, mask)
        return self.final_norm(hidden)

    def _output_proj(self) -> jax.Array:
        return reshard(self.token_embed.T, Plm_head)

    @named_call
    def logits(self, token_ids: Int[Array, "B S"], mask: AttentionMask | None = None) -> Float[Array, "B S V"]:
        hidden = self(token_ids, mask=mask)
        return jnp.einsum("bsh,hd->bsd", hidden, self._output_proj(), out_sharding=Plogits)

    def next_token_loss(
        self,
        token_ids: Int[Array, "B S"],
        loss_weight: Float[Array, "B S"],
        *,
        mask: AttentionMask | None = None,
        reduction: str = "mean",
        logsumexp_weight: float | None = None,
        loss_dtype: jnp.dtype = jnp.float32,
        loss_implementation: str | tuple[str, ...] | None = None,
    ) -> jax.Array:
        """Compute next-token cross-entropy loss for a batch."""
        hidden = self(token_ids, mask=mask)
        labels = jnp.pad(token_ids[:, 1:], ((0, 0), (0, 1))).astype(jnp.int32)
        return fused_linear_softmax_cross_entropy_loss(
            hidden,
            self._output_proj(),
            labels,
            weight=loss_weight.astype(loss_dtype),
            reduction=reduction,
            logsumexp_weight=logsumexp_weight,
            dtype=loss_dtype,
            implementation=loss_implementation,
        )


def _init_weight(key: PRNGKeyArray, shape: tuple[int, ...], std: float) -> Float[Array, "..."]:
    return std * random.normal(key, shape)
