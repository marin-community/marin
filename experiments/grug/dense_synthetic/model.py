# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Dense Grug transformer for the dense synthetic harness: the June 67B block machinery with a dense MLP.

Llama-3 / Qwen3 layout: pre-norm blocks of grouped-query attention with full RoPE, optionally with a per-head RMSNorm
on q and k (QK-norm: Qwen3 and Marin 32B have it, Llama-3 does not), then a SwiGLU MLP; untied embeddings and a final
RMSNorm. From the June 67B
model (``experiments/june_tpu_67b_a2b/moe/model.py``) it takes the layers stacked into one ``jax.lax.scan``, the
per-block remat policy, the Grug attention dispatch, the FSDP (``data``) and tensor-parallel (``model``) weight
layout, and the shared expert's ``DenseMLP`` as the MLP. It drops the routed experts and June's other architecture
choices: GatedNorm, XSA, the attention gate, half RoPE and the sliding-window layers.
"""

import dataclasses
from dataclasses import dataclass
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from haliax.jax_utils import named_call
from jax import random
from jax.sharding import PartitionSpec as P
from jax.sharding import reshard
from jaxtyping import Array, Float, Int, PRNGKeyArray
from levanter.grug.attention import (
    AttentionMask,
    GrugAttentionImplementation,
    RotaryConfig,
    apply_rotary_embedding,
    attention,
)
from levanter.grug.loss import fused_linear_softmax_cross_entropy_loss
from levanter.grug.sharding import Pembed_vocab, Plm_head, unshard
from levanter.layers.stacked import ArrayStacked

from experiments.june_tpu_67b_a2b.moe.model import DenseMLP

_BATCH_AXES: tuple[str, ...] = ("replica_dcn", "data", "expert")
# Heads on the tensor-parallel axis, head_dim replicated.
_HEADS_SPEC = P(_BATCH_AXES, None, "model", None)

RematMode = Literal["recompute_all", "save_dots", "none"]
"""Per-block gradient checkpointing. "recompute_all" (June's default) reruns the whole block in backward and keeps
only its input; "save_dots" keeps the outputs of the projection and MLP matmuls and reruns the norms, RoPE, the
activation and attention; "none" keeps every activation."""

_REMAT_POLICIES = {
    "recompute_all": None,
    "save_dots": jax.checkpoint_policies.dots_with_no_batch_dims_saveable,
}


@dataclass(frozen=True)
class GrugModelConfig:
    """Shape knobs for the dense transformer; ``head_dim`` is explicit, as in Qwen3."""

    vocab_size: int
    hidden_dim: int
    intermediate_dim: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    max_seq_len: int
    qk_norm: bool = True
    layer_norm_eps: float = 1e-6
    initializer_std: float = 0.02
    # The rotary frequencies change no shape or cost, so every preset keeps the default.
    rope: RotaryConfig = dataclasses.field(default_factory=RotaryConfig)
    attention_implementation: GrugAttentionImplementation | None = None
    remat_mode: RematMode = "recompute_all"

    def __post_init__(self) -> None:
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads for grouped-query attention")


class RMSNorm(eqx.Module):
    """Grug base's RMSNorm. June's also pins the variance with an optimization barrier, for bitwise forward
    parity under AD, which a throughput benchmark does not need."""

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


class QKNorm(eqx.Module):
    """Per-head RMSNorm on queries and keys, as in Qwen3."""

    q: RMSNorm
    k: RMSNorm

    @staticmethod
    def init(head_dim: int, eps: float) -> "QKNorm":
        return QKNorm(q=RMSNorm.init(head_dim, eps), k=RMSNorm.init(head_dim, eps))


class CausalSelfAttention(eqx.Module):
    w_q: Float[Array, "D NH"]
    w_k: Float[Array, "D MH"]
    w_v: Float[Array, "D MH"]
    w_o: Float[Array, "NH D"]
    qk_norm: QKNorm | None
    cfg: GrugModelConfig = eqx.field(static=True)

    @staticmethod
    def init(cfg: GrugModelConfig, *, key: PRNGKeyArray) -> "CausalSelfAttention":
        k_q, k_k, k_v, k_o = random.split(key, 4)
        d, n, m, h = cfg.hidden_dim, cfg.num_heads, cfg.num_kv_heads, cfg.head_dim
        return CausalSelfAttention(
            w_q=reshard(_init_weight(k_q, (d, n * h), cfg.initializer_std), P("data", "model")),
            w_k=reshard(_init_weight(k_k, (d, m * h), cfg.initializer_std), P("data", "model")),
            w_v=reshard(_init_weight(k_v, (d, m * h), cfg.initializer_std), P("data", "model")),
            w_o=reshard(_init_weight(k_o, (n * h, d), cfg.initializer_std), P("model", "data")),
            qk_norm=QKNorm.init(h, cfg.layer_norm_eps) if cfg.qk_norm else None,
            cfg=cfg,
        )

    @named_call
    def __call__(self, x: Float[Array, "B S D"], mask: AttentionMask) -> Float[Array, "B S D"]:
        cfg = self.cfg
        batch, seq_len, _ = x.shape
        # As in June: the projection's flat head dim is sharded over ``model``, and an explicit-mesh reshape cannot
        # infer which split axis carries it, so the head split names it.
        q = jnp.einsum("bsh,hd->bsd", x, self.w_q).reshape((batch, seq_len, -1, cfg.head_dim), out_sharding=_HEADS_SPEC)
        k = jnp.einsum("bsh,hd->bsd", x, self.w_k).reshape((batch, seq_len, -1, cfg.head_dim), out_sharding=_HEADS_SPEC)
        v = jnp.einsum("bsh,hd->bsd", x, self.w_v).reshape((batch, seq_len, -1, cfg.head_dim), out_sharding=_HEADS_SPEC)
        if self.qk_norm is not None:
            q, k = self.qk_norm.q(q), self.qk_norm.k(k)
        q, k = apply_rotary_embedding(q, k, seq_len=seq_len, head_dim=cfg.head_dim, rope=cfg.rope)
        attn_out = attention(q, k, v, mask, implementation=cfg.attention_implementation)
        attn_out = reshard(attn_out, _HEADS_SPEC)
        attn_out = attn_out.reshape((batch, seq_len, -1), out_sharding=P(_BATCH_AXES, None, "model"))
        return jnp.einsum("bsh,hd->bsd", attn_out, self.w_o, out_sharding=P(_BATCH_AXES))


class Block(eqx.Module):
    rms_attn: RMSNorm
    attn: CausalSelfAttention
    rms_mlp: RMSNorm
    mlp: DenseMLP

    @staticmethod
    def init(cfg: GrugModelConfig, *, key: PRNGKeyArray) -> "Block":
        attn_key, mlp_key = random.split(key)
        return Block(
            rms_attn=RMSNorm.init(cfg.hidden_dim, cfg.layer_norm_eps),
            attn=CausalSelfAttention.init(cfg, key=attn_key),
            rms_mlp=RMSNorm.init(cfg.hidden_dim, cfg.layer_norm_eps),
            mlp=DenseMLP.init(cfg.hidden_dim, cfg.intermediate_dim, cfg.initializer_std, key=mlp_key),
        )

    @named_call
    def __call__(self, x: Float[Array, "B S D"], mask: AttentionMask) -> Float[Array, "B S D"]:
        x = x + self.attn(self.rms_attn(x), mask)
        return x + self.mlp(self.rms_mlp(x))


class Transformer(eqx.Module):
    token_embed: jax.Array
    output_proj: jax.Array
    blocks: ArrayStacked[Block]
    final_norm: RMSNorm
    config: GrugModelConfig = eqx.field(static=True)

    @staticmethod
    def init(cfg: GrugModelConfig, *, key: PRNGKeyArray) -> "Transformer":
        keys = random.split(key, cfg.num_layers + 2)
        embed_key, out_key = keys[:2]
        token_embed = reshard(
            _init_weight(embed_key, (cfg.vocab_size, cfg.hidden_dim), cfg.initializer_std), Pembed_vocab
        )
        output_proj = reshard(_init_weight(out_key, (cfg.hidden_dim, cfg.vocab_size), cfg.initializer_std), Plm_head)
        return Transformer(
            token_embed=token_embed,
            output_proj=output_proj,
            blocks=ArrayStacked.init(cfg.num_layers, Block)(cfg=cfg, key=keys[2:]),
            final_norm=RMSNorm.init(cfg.hidden_dim, cfg.layer_norm_eps),
            config=cfg,
        )

    @named_call
    def __call__(self, token_ids: Int[Array, "B S"], mask: AttentionMask | None = None) -> Float[Array, "B S D"]:
        if mask is None:
            mask = AttentionMask.causal()
        hidden = self.token_embed.at[token_ids].get(out_sharding=P(_BATCH_AXES))
        remat_mode = self.config.remat_mode

        def scan_layer(carry: Float[Array, "B S D"], layer: Block) -> tuple[Float[Array, "B S D"], None]:
            if remat_mode == "none":
                return layer(carry, mask), None
            return eqx.filter_checkpoint(layer, policy=_REMAT_POLICIES[remat_mode])(carry, mask), None

        hidden, _ = jax.lax.scan(scan_layer, hidden, xs=self.blocks.stacked)
        return self.final_norm(hidden)

    def next_token_loss(
        self,
        token_ids: Int[Array, "B S"],
        loss_weight: Float[Array, "B S"],
        *,
        mask: AttentionMask | None = None,
        reduction: str = "mean",
        loss_dtype: jnp.dtype = jnp.float32,
    ) -> jax.Array:
        hidden = self(token_ids, mask=mask)
        labels = jnp.concatenate([token_ids[:, 1:], token_ids[:, :1] * 0], axis=1).astype(jnp.int32)
        return fused_linear_softmax_cross_entropy_loss(
            hidden,
            self.output_proj,
            labels,
            weight=loss_weight.astype(loss_dtype),
            reduction=reduction,
            logsumexp_weight=None,
            dtype=loss_dtype,
        )


def _init_weight(key: PRNGKeyArray, shape: tuple[int, ...], std: float) -> Float[Array, "..."]:
    return std * random.truncated_normal(key, -3, 3, shape)
