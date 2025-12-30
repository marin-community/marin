# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from einops import rearrange
from jax import random
from jax.sharding import PartitionSpec as P, reshard
from jax.tree_util import register_dataclass

from .attention import AttentionMask, apply_rotary_embedding, attention
from .config import AttentionRuntimeConfig, GrugModelConfig


@register_dataclass
@dataclass(frozen=True)
class GrugAttentionParams:
    w_q: jax.Array
    w_k: jax.Array
    w_v: jax.Array
    w_o: jax.Array


@register_dataclass
@dataclass(frozen=True)
class GrugBlockParams:
    attn: GrugAttentionParams
    rms_attn: jax.Array
    rms_mlp: jax.Array
    mlp_gate: jax.Array
    mlp_up: jax.Array
    mlp_down: jax.Array


@register_dataclass
@dataclass(frozen=True)
class GrugModelParameters:
    token_embed: jax.Array
    output_proj: jax.Array
    blocks: tuple[GrugBlockParams, ...]
    final_norm: jax.Array


def _init_weight(key: jax.Array, shape: tuple[int, ...], std: float) -> jax.Array:
    return std * random.truncated_normal(key, -3, 3, shape)


@partial(jax.jit, static_argnames=("cfg",))
def init_parameters(cfg: GrugModelConfig, *, key: jax.Array) -> GrugModelParameters:
    head_dim = cfg.inferred_head_dim
    keys = random.split(key, 3 + 7 * cfg.num_layers)
    embed_key, out_key, final_norm_key, *rest = keys
    layer_keys = [rest[i * 7 : (i + 1) * 7] for i in range(cfg.num_layers)]

    token_embed = reshard(
        _init_weight(embed_key, (cfg.vocab_size, cfg.hidden_dim), cfg.initializer_std),
        P("data", None),
    )
    if cfg.tie_embeddings:
        output_proj = token_embed.T
    else:
        output_proj = reshard(
            _init_weight(out_key, (cfg.hidden_dim, cfg.vocab_size), cfg.initializer_std),
            P("data", None),
        )
    if cfg.tie_embeddings:
        output_proj = reshard(output_proj, P("data", None))
    final_norm = reshard(
        jnp.ones((cfg.hidden_dim,), dtype=jnp.float32),
        P(
            "data",
        ),
    )

    blocks: list[GrugBlockParams] = []
    for i in range(cfg.num_layers):
        k_q, k_k, k_v, k_o, k_gate, k_up, k_down = layer_keys[i]
        # extract shape sizes for brevity and consistency
        H, N, M, D, I = cfg.hidden_dim, cfg.num_heads, cfg.num_kv_heads, head_dim, cfg.intermediate_dim

        attn = GrugAttentionParams(
            w_q=reshard(_init_weight(k_q, (H, N * D), cfg.initializer_std), P("data", "tensor")),
            w_k=reshard(_init_weight(k_k, (H, M * D), cfg.initializer_std), P("data", "tensor")),
            w_v=reshard(_init_weight(k_v, (H, M * D), cfg.initializer_std), P("data", "tensor")),
            w_o=reshard(_init_weight(k_o, (N * D, H), cfg.initializer_std), P("tensor", "data")),
        )
        mlp_gate = reshard(_init_weight(k_gate, (H, I), cfg.initializer_std), P("data", "tensor"))
        mlp_up = reshard(_init_weight(k_up, (H, I), cfg.initializer_std), P("data", "tensor"))
        mlp_down = reshard(_init_weight(k_down, (I, H), cfg.initializer_std), P("tensor", "data"))
        # keep rms replicated
        rms_attn = jnp.ones((H,), dtype=jnp.float32)
        rms_mlp = jnp.ones((H,), dtype=jnp.float32)

        blocks.append(
            GrugBlockParams(
                attn=attn,
                rms_attn=rms_attn,
                rms_mlp=rms_mlp,
                mlp_gate=mlp_gate,
                mlp_up=mlp_up,
                mlp_down=mlp_down,
            )
        )

    return GrugModelParameters(
        token_embed=token_embed,
        output_proj=output_proj,
        blocks=tuple(blocks),
        final_norm=jnp.ones_like(final_norm),
    )


def _unshard(x: jax.Array) -> jax.Array:
    return reshard(x, P((None,) * x.ndim))


_Pbatch = P(("replica", "data"), None)


def _rms_norm(x: jax.Array, weight: jax.Array, eps: float) -> jax.Array:
    weight = _unshard(weight)
    variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
    normed = x * jax.lax.rsqrt(variance + eps)
    return normed * weight


def _mlp(block: GrugBlockParams, x: jax.Array) -> jax.Array:
    gate = jnp.einsum("bsh,hm->bsm", x, block.mlp_gate)
    up = jnp.einsum("bsh,hm->bsm", x, block.mlp_up)
    activated = jax.nn.silu(gate) * up
    return jnp.einsum("bsm,mh->bsh", activated, block.mlp_down, out_sharding=_Pbatch)


def forward(
    params: GrugModelParameters,
    token_ids: jax.Array,
    cfg: GrugModelConfig,
    runtime: AttentionRuntimeConfig,
    *,
    mask: AttentionMask | jax.Array | None = None,
    causal: bool = True,
) -> jax.Array:
    head_dim = cfg.inferred_head_dim
    seq_len = token_ids.shape[1]

    hidden = params.token_embed.at[token_ids].get(out_sharding=_Pbatch)

    for block in params.blocks:
        attn_in = _rms_norm(hidden, block.rms_attn, cfg.layer_norm_eps)
        q = rearrange(jnp.einsum("bsh,hd->bsd", attn_in, block.attn.w_q), "... (n d) -> ... n d", d=head_dim)
        k = rearrange(jnp.einsum("bsh,hd->bsd", attn_in, block.attn.w_k), "... (m d) -> ... m d", d=head_dim)
        v = rearrange(jnp.einsum("bsh,hd->bsd", attn_in, block.attn.w_v), "... (m d) -> ... m d", d=head_dim)
        q, k = apply_rotary_embedding(q, k, seq_len=seq_len, head_dim=head_dim, rope=cfg.rope)
        attn_out = attention(q, k, v, mask, causal=causal, runtime=runtime)
        attn_out = rearrange(attn_out, "... n d -> ... (n d)")
        attn_out = jnp.einsum("bsh,hd->bsd", attn_out, block.attn.w_o, out_sharding=_Pbatch)

        hidden = hidden + attn_out
        mlp_in = _rms_norm(hidden, block.rms_mlp, cfg.layer_norm_eps)
        mlp_out = _mlp(block, mlp_in)
        hidden = hidden + mlp_out

    hidden = _rms_norm(hidden, params.final_norm, cfg.layer_norm_eps)
    logits = jnp.einsum("bsh,hd->bsd", hidden, params.output_proj, out_sharding=_Pbatch)
    return logits


__all__ = [
    "GrugAttentionParams",
    "GrugBlockParams",
    "GrugModelParameters",
    "init_parameters",
    "forward",
]
