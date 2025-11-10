from __future__ import annotations

import math
from dataclasses import dataclass
from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
from einops import rearrange
from jax import random
from jax.sharding import PartitionSpec as P, reshard
from jax.tree_util import register_dataclass

from .config import AttentionRuntimeConfig, GruGPTModelConfig, RotaryConfig


AttentionBackend = Callable[[jax.Array, jax.Array, jax.Array, jax.Array | None, bool], jax.Array]


@register_dataclass
@dataclass(frozen=True)
class GruGPTAttentionParams:
    w_q: jax.Array
    w_k: jax.Array
    w_v: jax.Array
    w_o: jax.Array


@register_dataclass
@dataclass(frozen=True)
class GruGPTBlockParams:
    attn: GruGPTAttentionParams
    rms_attn: jax.Array
    rms_mlp: jax.Array
    mlp_gate: jax.Array
    mlp_up: jax.Array
    mlp_down: jax.Array


@register_dataclass
@dataclass(frozen=True)
class GruGPTModelParameters:
    token_embed: jax.Array
    output_proj: jax.Array
    blocks: tuple[GruGPTBlockParams, ...]
    final_norm: jax.Array


def _init_weight(key: jax.Array, shape: tuple[int, ...], std: float) -> jax.Array:
    return std * random.truncated_normal(key, -3, 3, shape)


@partial(jax.jit, static_argnames=("cfg",))
def init_parameters(cfg: GruGPTModelConfig, *, key: jax.Array) -> GruGPTModelParameters:
    head_dim = cfg.inferred_head_dim
    keys = random.split(key, 5 + cfg.num_layers)
    embed_key, out_key, final_norm_key, *layer_keys = keys

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
    final_norm = reshard(jnp.ones((cfg.hidden_dim,), dtype=jnp.float32), P("data",))

    blocks: list[GruGPTBlockParams] = []
    for i in range(cfg.num_layers):
        k_q, k_k, k_v, k_o, k_gate, k_up, k_down = jax.random.split(layer_keys[i], 7)
        # extract shape sizes for brevity and consistency
        H, N, M, D, I = cfg.hidden_dim, cfg.num_heads, cfg.num_kv_heads, head_dim, cfg.intermediate_dim

        attn = GruGPTAttentionParams(
            w_q=reshard( _init_weight(k_q, (H, N * D), cfg.initializer_std), P("data", "tensor")),
            w_k=reshard( _init_weight(k_k, (H, M * D), cfg.initializer_std), P("data", "tensor")),
            w_v=reshard( _init_weight(k_v, (H, M * D), cfg.initializer_std), P("data", "tensor")),
            w_o=reshard( _init_weight(k_o, (N * D, H), cfg.initializer_std), P("tensor", "data")),
        )
        rms_attn = reshard(jnp.ones((H,), dtype=jnp.float32), P("data",))
        rms_mlp = reshard(jnp.ones((H,), dtype=jnp.float32), P("data",))
        mlp_gate = reshard( _init_weight(k_gate, (H, I), cfg.initializer_std), P("data", "tensor"),)
        mlp_up = reshard( _init_weight(k_up, (H, I), cfg.initializer_std), P("data", "tensor"),)
        mlp_down = reshard( _init_weight(k_down, (I, H), cfg.initializer_std), P("tensor", "data"),)

        blocks.append(
            GruGPTBlockParams(
                attn=attn,
                rms_attn=rms_attn,
                rms_mlp=rms_mlp,
                mlp_gate=mlp_gate,
                mlp_up=mlp_up,
                mlp_down=mlp_down,
            )
        )

    return GruGPTModelParameters(
        token_embed=token_embed,
        output_proj=output_proj,
        blocks=tuple(blocks),
        final_norm=jnp.ones_like(final_norm),
    )


def _unshard(x: jax.Array) -> jax.Array:
    return reshard(x, P((None,) * x.ndim))

_Pbatch = P(("replica", "data"), None)


def default_attention_mask(seq_len: int, key_len: int | None = None, *, dtype=jnp.float32) -> jax.Array:
    if key_len is None:
        key_len = seq_len
    tri = jnp.tril(jnp.ones((seq_len, key_len), dtype=dtype))
    large_neg = jnp.array(-1e9, dtype=dtype)
    return jnp.where(tri == 1, 0.0, large_neg)


def _rotary_cache(seq_len: int, head_dim: int, rope: RotaryConfig) -> tuple[jax.Array, jax.Array]:
    half_dim = head_dim // 2
    inv_freq = 1.0 / (rope.theta ** (jnp.arange(0, half_dim, dtype=jnp.float32) / half_dim))
    positions = jnp.arange(seq_len, dtype=jnp.float32)
    angles = positions[:, None] * inv_freq[None, :]
    cos = jnp.cos(angles)
    sin = jnp.sin(angles)
    return cos, sin


def apply_rotary_embedding(q: jax.Array, k: jax.Array, seq_len: int, head_dim: int, rope: RotaryConfig) -> tuple[jax.Array, jax.Array]:
    cos, sin = _rotary_cache(seq_len, head_dim, rope)
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]

    def _apply(x: jax.Array) -> jax.Array:
        x1, x2 = jnp.split(x, 2, axis=-1)
        return jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)

    return _apply(q), _apply(k)


def _rms_norm(x: jax.Array, weight: jax.Array, eps: float) -> jax.Array:
    weight = _unshard(weight)
    variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
    normed = x * jax.lax.rsqrt(variance + eps)
    return normed * weight


def reference_attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    mask: jax.Array | None,
    *,
    causal: bool,
    logits_dtype: jnp.dtype | None,
) -> jax.Array:
    head_dim = q.shape[-1]
    num_q_heads = q.shape[2]
    num_kv_heads = k.shape[2]

    if num_q_heads != num_kv_heads:
        if num_q_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({num_q_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
            )
        repeat = num_q_heads // num_kv_heads
        k = jnp.repeat(k, repeat, axis=2)
        v = jnp.repeat(v, repeat, axis=2)

    scale = 1.0 / math.sqrt(head_dim)
    scores = jnp.einsum("bqhd,bkhd->bhqk", q * scale, k)
    if mask is None and causal:
        mask = default_attention_mask(scores.shape[-2], dtype=scores.dtype)
    if mask is not None:
        scores = scores + mask[None, None, :, :]
    if logits_dtype is not None:
        scores = scores.astype(logits_dtype)
    weights = jax.nn.softmax(scores, axis=-1).astype(v.dtype)
    ctx = jnp.einsum("bhqk,bkhd->bqhd", weights, v)
    ctx = ctx.astype(v.dtype)
    return ctx


def _maybe_splash_attention() -> AttentionBackend | None:
    return None


def resolve_attention_backend(runtime: AttentionRuntimeConfig) -> AttentionBackend:
    def _reference_backend(q, k, v, mask, causal):
        return reference_attention(q, k, v, mask, causal=causal, logits_dtype=runtime.logits_dtype)

    if runtime.backend == "reference":
        return _reference_backend
    if runtime.backend == "splash":
        backend = _maybe_splash_attention()
        if backend is None:
            raise RuntimeError("Splash attention requested but unavailable")
        return backend
    backend = _maybe_splash_attention() if runtime.backend == "auto" else None
    if backend is not None:
        return backend
    return _reference_backend


def _mlp(block: GruGPTBlockParams, x: jax.Array) -> jax.Array:
    gate = jnp.einsum("bsh,hm->bsm", x, block.mlp_gate)
    up = jnp.einsum("bsh,hm->bsm", x, block.mlp_up)
    activated = jax.nn.silu(gate) * up
    return jnp.einsum("bsm,mh->bsh", activated, block.mlp_down, out_sharding=_Pbatch)


def forward(
    params: GruGPTModelParameters,
    token_ids: jax.Array,
    cfg: GruGPTModelConfig,
    runtime: AttentionRuntimeConfig,
    *,
    mask: jax.Array | None = None,
    causal: bool = True,
) -> jax.Array:
    backend = resolve_attention_backend(runtime)
    head_dim = cfg.inferred_head_dim
    seq_len = token_ids.shape[1]

    hidden = params.token_embed.at[token_ids].get(out_sharding=_Pbatch)

    for block in params.blocks:
        attn_in = _rms_norm(hidden, block.rms_attn, cfg.layer_norm_eps)
        q = rearrange(jnp.einsum("bsh,hd->bsd", attn_in, block.attn.w_q), "... (n d) -> ... n d", d=head_dim)
        k = rearrange(jnp.einsum("bsh,hd->bsd", attn_in, block.attn.w_k), "... (m d) -> ... m d", d=head_dim)
        v = rearrange(jnp.einsum("bsh,hd->bsd", attn_in, block.attn.w_v), "... (m d) -> ... m d", d=head_dim)
        q, k = apply_rotary_embedding(q, k, seq_len=seq_len, head_dim=head_dim, rope=cfg.rope)
        attn_out = backend(q, k, v, mask, causal)
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
    "GruGPTAttentionParams",
    "GruGPTBlockParams",
    "GruGPTModelParameters",
    "init_parameters",
    "default_attention_mask",
    "apply_rotary_embedding",
    "forward",
    "resolve_attention_backend",
]
