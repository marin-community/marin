# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

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
from levanter.grug.loss import fused_linear_softmax_cross_entropy_loss
from levanter.grug.sharding import Pbatch, Pembed_vocab, Plm_head, Plogits, unshard
from levanter.kernels.pallas.mamba3 import (
    HybridModeConfig,
    intra_chunk_log_alpha_cumsum,
    local_log_alpha,
    mamba3_hybrid_chunked_forward,
    mamba3_mimo_attentionish_forward_from_transformed,
)

LayerType = Literal["full_attention", "sliding_attention", "linear_attention"]
LinearMode = Literal["siso", "mimo"]
DT_INIT_MIN = 1e-3
DT_INIT_MAX = 0.1
DT_INIT_FLOOR = 1e-4
A_INIT_MIN = 1.0
A_INIT_MAX = 16.0
A_FLOOR = 1e-4


@dataclass(frozen=True)
class HybridModelConfig:
    """Hyperparameters for the hybrid Grug Mamba-3/Splash transformer."""

    vocab_size: int
    hidden_dim: int = 512
    intermediate_dim: int = 1792
    num_layers: int = 12
    num_heads: int = 8
    max_seq_len: int = 4096
    sliding_window: int = 1024
    layer_types: tuple[LayerType, ...] = ()
    linear_mode: LinearMode = "siso"
    linear_state_dim: int = 64
    linear_rank: int = 4
    layer_norm_eps: float = 1e-6
    initializer_std: float = 0.02
    rope_theta: float = 10_000
    width_label: str = "d512"
    pattern_label: str = "full"

    def __post_init__(self) -> None:
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(f"hidden_dim={self.hidden_dim} must be divisible by num_heads={self.num_heads}")
        if len(self.layer_types) != self.num_layers:
            raise ValueError("layer_types must have one entry per layer")
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        if self.max_seq_len % self.linear_chunk_size != 0:
            raise ValueError(
                f"max_seq_len={self.max_seq_len} must be divisible by linear_chunk_size={self.linear_chunk_size}"
            )

    @property
    def head_dim(self) -> int:
        return self.hidden_dim // self.num_heads

    @property
    def num_kv_heads(self) -> int:
        return self.num_heads

    @property
    def linear_chunk_size(self) -> int:
        mode_cfg = HybridModeConfig(
            mode=self.linear_mode,
            mimo_rank=self.linear_rank if self.linear_mode == "mimo" else None,
        )
        return mode_cfg.resolved_chunk_size()

    @property
    def total_trainable_params(self) -> int:
        token_embedding = self.vocab_size * self.hidden_dim
        layer_norms = 2 * self.hidden_dim
        mlp = 2 * self.hidden_dim * self.intermediate_dim
        attention_proj = 4 * self.hidden_dim * self.hidden_dim

        if self.linear_mode == "siso":
            linear_proj = (
                2 * self.hidden_dim * self.num_heads * self.linear_state_dim
                + 3 * self.hidden_dim * self.hidden_dim
                + 3 * self.hidden_dim * self.num_heads
                + 2 * self.linear_state_dim
                + 2 * self.num_heads * self.linear_state_dim
                + self.hidden_dim
                + 4 * self.num_heads
            )
        else:
            linear_proj = (
                2 * self.hidden_dim * self.num_heads * self.linear_rank * self.linear_state_dim
                + 3 * self.hidden_dim * self.hidden_dim
                + 3 * self.hidden_dim * self.num_heads
                + 3 * self.hidden_dim * self.linear_rank
                + 2 * self.linear_state_dim
                + 2 * self.num_heads * self.linear_rank * self.linear_state_dim
                + self.hidden_dim
                + 4 * self.num_heads
            )

        layers = 0
        for layer_type in self.layer_types:
            mixer = attention_proj if layer_type != "linear_attention" else linear_proj
            layers += mixer + mlp + layer_norms

        final_norm = self.hidden_dim
        output_proj = self.hidden_dim * self.vocab_size
        return int(token_embedding + layers + final_norm + output_proj)

    @property
    def flops_per_token(self) -> float:
        mlp = 4 * self.hidden_dim * self.intermediate_dim
        lm_head = 2 * self.hidden_dim * self.vocab_size
        attention_proj = 8 * self.hidden_dim * self.hidden_dim

        linear_siso_proj = (
            4 * self.hidden_dim * self.num_heads * self.linear_state_dim
            + 6 * self.hidden_dim * self.hidden_dim
            + 6 * self.hidden_dim * self.num_heads
        )
        linear_mimo_proj = (
            4 * self.hidden_dim * self.num_heads * self.linear_rank * self.linear_state_dim
            + 6 * self.hidden_dim * self.hidden_dim
            + 6 * self.hidden_dim * self.num_heads
        )

        total = lm_head
        for layer_type in self.layer_types:
            if layer_type == "full_attention":
                mixer = attention_proj + _full_attention_token_flops(
                    self.max_seq_len,
                    self.num_heads,
                    self.head_dim,
                )
            elif layer_type == "sliding_attention":
                mixer = attention_proj + _sliding_attention_token_flops(
                    self.max_seq_len,
                    self.num_heads,
                    self.head_dim,
                    self.sliding_window,
                )
            elif self.linear_mode == "siso":
                mixer = linear_siso_proj + _linear_siso_kernel_token_flops(
                    num_heads=self.num_heads,
                    chunk_size=self.linear_chunk_size,
                    state_dim=self.linear_state_dim,
                    value_dim=self.head_dim,
                )
            else:
                mixer = linear_mimo_proj + _linear_mimo_kernel_token_flops(
                    num_heads=self.num_heads,
                    chunk_size=self.linear_chunk_size,
                    state_dim=self.linear_state_dim,
                    value_dim=self.head_dim,
                    rank=self.linear_rank,
                )
            total += mixer + mlp
        return float(total)


def _softplus_inverse(x: jax.Array) -> jax.Array:
    return x + jnp.log(-jnp.expm1(-x))


def _init_dt_bias(key: PRNGKeyArray, *, num_heads: int) -> jax.Array:
    dt = jnp.exp(
        random.uniform(
            key,
            (num_heads,),
            minval=math.log(DT_INIT_MIN),
            maxval=math.log(DT_INIT_MAX),
            dtype=jnp.float32,
        )
    ).clip(min=DT_INIT_FLOOR)
    return _softplus_inverse(dt)


def _init_a_bias(key: PRNGKeyArray, *, num_heads: int) -> jax.Array:
    a = random.uniform(key, (num_heads,), minval=A_INIT_MIN, maxval=A_INIT_MAX, dtype=jnp.float32)
    return _softplus_inverse(a)


class RMSNorm(eqx.Module):
    weight: jax.Array
    eps: float = eqx.field(static=True)

    @staticmethod
    def init(dim: int, eps: float) -> RMSNorm:
        return RMSNorm(weight=jnp.ones((dim,), dtype=jnp.float32), eps=eps)

    @named_call
    def __call__(self, x: Float[Array, "... D"]) -> Float[Array, "... D"]:
        weight = unshard(self.weight)
        dtype = x.dtype
        x = x.astype(jnp.float32)
        variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        normed = x * jax.lax.rsqrt(variance + self.eps)
        return (normed * weight).astype(dtype)


class HeadwiseGatedRMSNorm(eqx.Module):
    weight: jax.Array
    eps: float = eqx.field(static=True)

    @staticmethod
    def init(*, num_heads: int, head_dim: int, eps: float) -> HeadwiseGatedRMSNorm:
        return HeadwiseGatedRMSNorm(
            weight=reshard(jnp.ones((num_heads, head_dim), dtype=jnp.float32), P(None, "model")),
            eps=eps,
        )

    @named_call
    def __call__(self, x: Float[Array, "... H P"], gate: Float[Array, "... H P"]) -> Float[Array, "... H P"]:
        weight = unshard(self.weight)
        dtype = x.dtype
        x32 = x.astype(jnp.float32)
        variance = jnp.mean(jnp.square(x32), axis=-1, keepdims=True)
        normed = x32 * jax.lax.rsqrt(variance + self.eps)
        broadcast_weight = weight.reshape((1,) * (x.ndim - 2) + weight.shape)
        gated = normed * broadcast_weight * jax.nn.silu(gate.astype(jnp.float32))
        return gated.astype(dtype)


class MLP(eqx.Module):
    mlp_up: jax.Array
    mlp_down: jax.Array

    @staticmethod
    def init(cfg: HybridModelConfig, *, key: PRNGKeyArray) -> MLP:
        k_up, k_down = random.split(key, 2)
        return MLP(
            mlp_up=reshard(
                _init_weight(k_up, (cfg.hidden_dim, cfg.intermediate_dim), cfg.initializer_std),
                P("data", "model"),
            ),
            mlp_down=reshard(
                _init_weight(k_down, (cfg.intermediate_dim, cfg.hidden_dim), cfg.initializer_std),
                P("model", "data"),
            ),
        )

    @named_call
    def __call__(self, x: Float[Array, "B S D"]) -> Float[Array, "B S D"]:
        up = jnp.einsum("bsh,hm->bsm", x, self.mlp_up)
        activated = jax.nn.silu(up)
        return jnp.einsum("bsm,mh->bsh", activated, self.mlp_down, out_sharding=Pbatch)


class CausalSelfAttention(eqx.Module):
    w_q: jax.Array
    w_k: jax.Array
    w_v: jax.Array
    w_o: jax.Array
    cfg: HybridModelConfig = eqx.field(static=True)

    @staticmethod
    def init(cfg: HybridModelConfig, *, key: PRNGKeyArray) -> CausalSelfAttention:
        k_q, k_k, k_v, k_o = random.split(key, 4)
        head_dim = cfg.head_dim
        return CausalSelfAttention(
            w_q=reshard(
                _init_weight(k_q, (cfg.hidden_dim, cfg.num_heads * head_dim), cfg.initializer_std),
                P("data", "model"),
            ),
            w_k=reshard(
                _init_weight(k_k, (cfg.hidden_dim, cfg.num_heads * head_dim), cfg.initializer_std),
                P("data", "model"),
            ),
            w_v=reshard(
                _init_weight(k_v, (cfg.hidden_dim, cfg.num_heads * head_dim), cfg.initializer_std),
                P("data", "model"),
            ),
            w_o=reshard(
                _init_weight(k_o, (cfg.num_heads * head_dim, cfg.hidden_dim), cfg.initializer_std),
                P("model", "data"),
            ),
            cfg=cfg,
        )

    @named_call
    def __call__(self, x: Float[Array, "B S D"], mask: AttentionMask | jax.Array) -> Float[Array, "B S D"]:
        head_dim = self.cfg.head_dim
        q = rearrange(
            jnp.einsum("bsh,hd->bsd", x, self.w_q),
            "... (h d) -> ... h d",
            h=self.cfg.num_heads,
            d=head_dim,
        )
        k = rearrange(
            jnp.einsum("bsh,hd->bsd", x, self.w_k),
            "... (h d) -> ... h d",
            h=self.cfg.num_heads,
            d=head_dim,
        )
        v = rearrange(
            jnp.einsum("bsh,hd->bsd", x, self.w_v),
            "... (h d) -> ... h d",
            h=self.cfg.num_heads,
            d=head_dim,
        )
        q, k = _apply_rotary_embedding(q, k, seq_len=x.shape[1], head_dim=head_dim, rope_theta=self.cfg.rope_theta)
        attn_out = attention(q, k, v, mask)
        attn_out = rearrange(attn_out, "... h d -> ... (h d)")
        return jnp.einsum("bsh,hd->bsd", attn_out, self.w_o, out_sharding=Pbatch)


class Mamba3SisoMixer(eqx.Module):
    w_in: jax.Array
    w_o: jax.Array
    b_norm: RMSNorm
    c_norm: RMSNorm
    out_norm: HeadwiseGatedRMSNorm
    b_bias: jax.Array
    c_bias: jax.Array
    a_bias: jax.Array
    dt_bias: jax.Array
    trap_bias: jax.Array
    d_skip: jax.Array
    cfg: HybridModelConfig = eqx.field(static=True)

    @staticmethod
    def init(cfg: HybridModelConfig, *, key: PRNGKeyArray) -> Mamba3SisoMixer:
        keys = random.split(key, 4)
        flat_state = cfg.num_heads * cfg.linear_state_dim
        flat_value = cfg.hidden_dim
        in_proj_dim = 2 * flat_value + 2 * flat_state + 3 * cfg.num_heads
        return Mamba3SisoMixer(
            w_in=reshard(_init_weight(keys[0], (cfg.hidden_dim, in_proj_dim), cfg.initializer_std), P("data", "model")),
            w_o=reshard(_init_weight(keys[1], (flat_value, cfg.hidden_dim), cfg.initializer_std), P("model", "data")),
            b_norm=RMSNorm.init(cfg.linear_state_dim, cfg.layer_norm_eps),
            c_norm=RMSNorm.init(cfg.linear_state_dim, cfg.layer_norm_eps),
            out_norm=HeadwiseGatedRMSNorm.init(
                num_heads=cfg.num_heads,
                head_dim=cfg.head_dim,
                eps=cfg.layer_norm_eps,
            ),
            b_bias=reshard(jnp.ones((cfg.num_heads, cfg.linear_state_dim), dtype=jnp.float32), P(None, None)),
            c_bias=reshard(jnp.ones((cfg.num_heads, cfg.linear_state_dim), dtype=jnp.float32), P(None, None)),
            a_bias=reshard(_init_a_bias(keys[2], num_heads=cfg.num_heads), P(None)),
            dt_bias=reshard(_init_dt_bias(keys[3], num_heads=cfg.num_heads), P(None)),
            trap_bias=reshard(jnp.zeros((cfg.num_heads,), dtype=jnp.float32), P(None)),
            d_skip=reshard(jnp.ones((cfg.num_heads,), dtype=jnp.float32), P(None)),
            cfg=cfg,
        )

    @named_call
    def __call__(self, x: Float[Array, "B S D"]) -> Float[Array, "B S D"]:
        projected = jnp.einsum("bsh,hd->bsd", x, self.w_in)
        heads = self.cfg.num_heads
        state_dim = self.cfg.linear_state_dim
        value_dim = self.cfg.head_dim
        num_chunks = x.shape[1] // self.cfg.linear_chunk_size
        split_points = (
            value_dim * heads,
            2 * value_dim * heads,
            2 * value_dim * heads + state_dim * heads,
            2 * value_dim * heads + 2 * state_dim * heads,
            2 * value_dim * heads + 2 * state_dim * heads + heads,
            2 * value_dim * heads + 2 * state_dim * heads + 2 * heads,
        )
        z_proj, x_proj, b_proj, c_proj, dt_proj, a_proj, trap_proj = jnp.split(projected, split_points, axis=-1)
        x_base = jax.lax.reshape(
            x_proj,
            (x.shape[0], x.shape[1], heads, value_dim),
            out_sharding=P("data", None, "model", None),
        )
        z_base = jax.lax.reshape(
            z_proj,
            (x.shape[0], x.shape[1], heads, value_dim),
            out_sharding=P("data", None, "model", None),
        )
        b = (
            self.b_norm(
                jax.lax.reshape(
                    b_proj,
                    (x.shape[0], x.shape[1], heads, state_dim),
                    out_sharding=P("data", None, "model", None),
                )
            )
            + self.b_bias[None, None, :, :]
        )
        c = (
            self.c_norm(
                jax.lax.reshape(
                    c_proj,
                    (x.shape[0], x.shape[1], heads, state_dim),
                    out_sharding=P("data", None, "model", None),
                )
            )
            + self.c_bias[None, None, :, :]
        )
        dt = reshard(
            jnp.transpose(jax.nn.softplus(dt_proj + self.dt_bias[None, None, :]) + DT_INIT_FLOOR, (0, 2, 1)),
            P("data", "model", None),
        )
        trap = reshard(
            jnp.transpose(jax.nn.sigmoid(trap_proj + self.trap_bias[None, None, :]), (0, 2, 1)),
            P("data", "model", None),
        )
        a = reshard(
            jnp.transpose(
                jnp.clip(-jax.nn.softplus(a_proj + self.a_bias[None, None, :]), max=-A_FLOOR),
                (0, 2, 1),
            ),
            P("data", "model", None),
        )
        b_chunked = b.reshape(x.shape[0], num_chunks, self.cfg.linear_chunk_size, heads, state_dim).transpose(
            0, 3, 1, 2, 4
        )
        c_chunked = c.reshape(x.shape[0], num_chunks, self.cfg.linear_chunk_size, heads, state_dim).transpose(
            0, 3, 1, 2, 4
        )
        x_chunked = x_base.reshape(x.shape[0], num_chunks, self.cfg.linear_chunk_size, heads, value_dim).transpose(
            0, 3, 1, 2, 4
        )
        y_chunked, _ = mamba3_hybrid_chunked_forward(
            dt.reshape(x.shape[0], heads, num_chunks, self.cfg.linear_chunk_size),
            trap.reshape(x.shape[0], heads, num_chunks, self.cfg.linear_chunk_size),
            a.reshape(x.shape[0], heads, num_chunks, self.cfg.linear_chunk_size),
            b_chunked,
            c_chunked,
            x_chunked,
            mode="siso",
        )
        y_chunked = y_chunked + self.d_skip[None, :, None, None, None] * x_chunked
        y = y_chunked.transpose(0, 2, 3, 1, 4).reshape(x.shape[0], x.shape[1], heads, value_dim)
        y = self.out_norm(y, z_base)
        y = jax.lax.reshape(y, (y.shape[0], y.shape[1], heads * value_dim), out_sharding=P("data", None, "model"))
        return jnp.einsum("bsh,hd->bsd", y, self.w_o, out_sharding=Pbatch)


class Mamba3MimoMixer(eqx.Module):
    w_in: jax.Array
    w_o: jax.Array
    w_rank_x: jax.Array
    w_rank_z: jax.Array
    w_rank_o: jax.Array
    b_norm: RMSNorm
    c_norm: RMSNorm
    out_norm: HeadwiseGatedRMSNorm
    b_bias: jax.Array
    c_bias: jax.Array
    a_bias: jax.Array
    dt_bias: jax.Array
    trap_bias: jax.Array
    d_skip: jax.Array
    cfg: HybridModelConfig = eqx.field(static=True)

    @staticmethod
    def init(cfg: HybridModelConfig, *, key: PRNGKeyArray) -> Mamba3MimoMixer:
        keys = random.split(key, 4)
        flat_state = cfg.num_heads * cfg.linear_rank * cfg.linear_state_dim
        flat_value = cfg.hidden_dim
        in_proj_dim = 2 * flat_value + 2 * flat_state + 3 * cfg.num_heads
        rank_weight_shape = (cfg.num_heads, cfg.linear_rank, cfg.head_dim)
        return Mamba3MimoMixer(
            w_in=reshard(_init_weight(keys[0], (cfg.hidden_dim, in_proj_dim), cfg.initializer_std), P("data", "model")),
            w_o=reshard(_init_weight(keys[1], (flat_value, cfg.hidden_dim), cfg.initializer_std), P("model", "data")),
            w_rank_x=reshard(jnp.ones(rank_weight_shape, dtype=jnp.float32) / cfg.linear_rank, P(None, None, None)),
            w_rank_z=reshard(jnp.ones(rank_weight_shape, dtype=jnp.float32), P(None, None, None)),
            w_rank_o=reshard(jnp.ones(rank_weight_shape, dtype=jnp.float32) / cfg.linear_rank, P(None, None, None)),
            b_norm=RMSNorm.init(cfg.linear_state_dim, cfg.layer_norm_eps),
            c_norm=RMSNorm.init(cfg.linear_state_dim, cfg.layer_norm_eps),
            out_norm=HeadwiseGatedRMSNorm.init(
                num_heads=cfg.num_heads,
                head_dim=cfg.head_dim,
                eps=cfg.layer_norm_eps,
            ),
            b_bias=reshard(
                jnp.ones((cfg.num_heads, cfg.linear_rank, cfg.linear_state_dim), dtype=jnp.float32),
                P(None, None, None),
            ),
            c_bias=reshard(
                jnp.ones((cfg.num_heads, cfg.linear_rank, cfg.linear_state_dim), dtype=jnp.float32),
                P(None, None, None),
            ),
            a_bias=reshard(_init_a_bias(keys[2], num_heads=cfg.num_heads), P(None)),
            dt_bias=reshard(_init_dt_bias(keys[3], num_heads=cfg.num_heads), P(None)),
            trap_bias=reshard(jnp.zeros((cfg.num_heads,), dtype=jnp.float32), P(None)),
            d_skip=reshard(jnp.ones((cfg.num_heads,), dtype=jnp.float32), P(None)),
            cfg=cfg,
        )

    @named_call
    def __call__(self, x: Float[Array, "B S D"]) -> Float[Array, "B S D"]:
        projected = jnp.einsum("bsh,hd->bsd", x, self.w_in)
        heads = self.cfg.num_heads
        rank = self.cfg.linear_rank
        state_dim = self.cfg.linear_state_dim
        value_dim = self.cfg.head_dim
        split_points = (
            value_dim * heads,
            2 * value_dim * heads,
            2 * value_dim * heads + heads * rank * state_dim,
            2 * value_dim * heads + 2 * heads * rank * state_dim,
            2 * value_dim * heads + 2 * heads * rank * state_dim + heads,
            2 * value_dim * heads + 2 * heads * rank * state_dim + 2 * heads,
        )
        z_proj, x_proj, b_proj, c_proj, dt_proj, a_proj, trap_proj = jnp.split(projected, split_points, axis=-1)
        x_base = jax.lax.reshape(
            x_proj,
            (x.shape[0], x.shape[1], heads, value_dim),
            out_sharding=P("data", None, "model", None),
        )
        z_base = jax.lax.reshape(
            z_proj,
            (x.shape[0], x.shape[1], heads, value_dim),
            out_sharding=P("data", None, "model", None),
        )
        b = (
            self.b_norm(
                jax.lax.reshape(
                    b_proj,
                    (x.shape[0], x.shape[1], heads, rank, state_dim),
                    out_sharding=P("data", None, "model", None, None),
                ).transpose(0, 1, 3, 2, 4)
            )
            + self.b_bias.transpose(1, 0, 2)[None, None, :, :, :]
        )
        c = (
            self.c_norm(
                jax.lax.reshape(
                    c_proj,
                    (x.shape[0], x.shape[1], heads, rank, state_dim),
                    out_sharding=P("data", None, "model", None, None),
                ).transpose(0, 1, 3, 2, 4)
            )
            + self.c_bias.transpose(1, 0, 2)[None, None, :, :, :]
        )
        dt = reshard(
            jnp.transpose(jax.nn.softplus(dt_proj + self.dt_bias[None, None, :]) + DT_INIT_FLOOR, (0, 2, 1)),
            P("data", "model", None),
        )
        trap = reshard(
            jnp.transpose(jax.nn.sigmoid(trap_proj + self.trap_bias[None, None, :]), (0, 2, 1)),
            P("data", "model", None),
        )
        a = reshard(
            jnp.transpose(
                jnp.clip(-jax.nn.softplus(a_proj + self.a_bias[None, None, :]), max=-A_FLOOR),
                (0, 2, 1),
            ),
            P("data", "model", None),
        )
        log_alpha = reshard(
            intra_chunk_log_alpha_cumsum(
                local_log_alpha(
                    dt.reshape(x.shape[0], heads, x.shape[1] // self.cfg.linear_chunk_size, self.cfg.linear_chunk_size),
                    a.reshape(x.shape[0], heads, x.shape[1] // self.cfg.linear_chunk_size, self.cfg.linear_chunk_size),
                )
            ).reshape(x.shape[0], heads, x.shape[1]),
            P("data", "model", None),
        )
        y_ranked = mamba3_mimo_attentionish_forward_from_transformed(
            c,
            b,
            x_base,
            self.w_rank_x,
            self.w_rank_o,
            q_bias=self.c_bias,
            k_bias=self.b_bias,
            d=self.d_skip,
            da_cs=log_alpha,
            dt=dt,
            trap=trap,
            chunk_size=self.cfg.linear_chunk_size,
            reduce_o=False,
        )
        z_ranked = jnp.einsum("bshp,hrp->bsrhp", z_base.astype(jnp.float32), self.w_rank_z.astype(jnp.float32))
        y_ranked = self.out_norm(y_ranked, z_ranked.astype(y_ranked.dtype))
        y = jnp.einsum("bsrhp,hrp->bshp", y_ranked.astype(jnp.float32), self.w_rank_o.astype(jnp.float32)).astype(
            y_ranked.dtype
        )
        y = jax.lax.reshape(y, (y.shape[0], y.shape[1], heads * value_dim), out_sharding=P("data", None, "model"))
        return jnp.einsum("bsh,hd->bsd", y, self.w_o, out_sharding=Pbatch)


class HybridBlock(eqx.Module):
    rms_mixer: RMSNorm
    attn: CausalSelfAttention | None
    mixer: Mamba3SisoMixer | Mamba3MimoMixer | None
    rms_mlp: RMSNorm
    mlp: MLP
    layer_type: LayerType = eqx.field(static=True)
    sliding_window: int = eqx.field(static=True)

    @staticmethod
    def init(cfg: HybridModelConfig, *, layer_type: LayerType, key: PRNGKeyArray) -> HybridBlock:
        mixer_key, mlp_key = random.split(key, 2)
        attn: CausalSelfAttention | None = None
        mixer: Mamba3SisoMixer | Mamba3MimoMixer | None = None
        if layer_type == "linear_attention":
            if cfg.linear_mode == "siso":
                mixer = Mamba3SisoMixer.init(cfg, key=mixer_key)
            else:
                mixer = Mamba3MimoMixer.init(cfg, key=mixer_key)
        else:
            attn = CausalSelfAttention.init(cfg, key=mixer_key)
        return HybridBlock(
            rms_mixer=RMSNorm.init(cfg.hidden_dim, cfg.layer_norm_eps),
            attn=attn,
            mixer=mixer,
            rms_mlp=RMSNorm.init(cfg.hidden_dim, cfg.layer_norm_eps),
            mlp=MLP.init(cfg, key=mlp_key),
            layer_type=layer_type,
            sliding_window=cfg.sliding_window,
        )

    @named_call
    def __call__(self, x: Float[Array, "B S D"], mask: AttentionMask | jax.Array | None) -> Float[Array, "B S D"]:
        if self.layer_type == "linear_attention":
            if self.mixer is None:
                raise ValueError("linear HybridBlock is missing mixer")
            x = x + self.mixer(self.rms_mixer(x))
        else:
            if self.attn is None:
                raise ValueError("attention HybridBlock is missing attention module")
            x = x + self.attn(
                self.rms_mixer(x),
                _layer_mask(mask, layer_type=self.layer_type, sliding_window=self.sliding_window),
            )
        x = x + self.mlp(self.rms_mlp(x))
        return x


class Transformer(eqx.Module):
    token_embed: jax.Array
    output_proj: jax.Array
    blocks: tuple[HybridBlock, ...]
    final_norm: RMSNorm
    config: HybridModelConfig = eqx.field(static=True)

    @staticmethod
    def init(cfg: HybridModelConfig, *, key: PRNGKeyArray) -> Transformer:
        embed_key, out_key, *block_keys = random.split(key, cfg.num_layers + 2)
        token_embed = reshard(
            _init_weight(embed_key, (cfg.vocab_size, cfg.hidden_dim), cfg.initializer_std),
            Pembed_vocab,
        )
        output_proj = reshard(_init_weight(out_key, (cfg.hidden_dim, cfg.vocab_size), cfg.initializer_std), Plm_head)
        blocks = tuple(
            HybridBlock.init(cfg, layer_type=layer_type, key=block_key)
            for layer_type, block_key in zip(cfg.layer_types, block_keys, strict=True)
        )
        return Transformer(
            token_embed=token_embed,
            output_proj=output_proj,
            blocks=blocks,
            final_norm=RMSNorm.init(cfg.hidden_dim, cfg.layer_norm_eps),
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
        hidden = self.token_embed.at[token_ids].get(out_sharding=Pbatch)
        for block in self.blocks:
            hidden = eqx.filter_checkpoint(block)(hidden, mask)
        return self.final_norm(hidden)

    @named_call
    def logits(
        self,
        token_ids: Int[Array, "B S"],
        mask: AttentionMask | jax.Array | None = None,
    ) -> Float[Array, "B S V"]:
        hidden = self(token_ids, mask=mask)
        return jnp.einsum("bsh,hd->bsd", hidden, self.output_proj, out_sharding=Plogits)

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
        labels = jnp.concatenate([token_ids[:, 1:], token_ids[:, :1] * 0], axis=1).astype(jnp.int32)
        return fused_linear_softmax_cross_entropy_loss(
            hidden,
            self.output_proj,
            labels,
            weight=loss_weight.astype(loss_dtype),
            reduction=reduction,
            logsumexp_weight=logsumexp_weight,
            dtype=loss_dtype,
        )


def _init_weight(key: PRNGKeyArray, shape: tuple[int, ...], std: float) -> Float[Array, ...]:
    return std * random.truncated_normal(key, -3, 3, shape)


def _rotary_cache(seq_len: int, head_dim: int, rope_theta: float) -> tuple[Float[Array, "S D"], Float[Array, "S D"]]:
    half_dim = head_dim // 2
    inv_freq = 1.0 / (rope_theta ** (jnp.arange(0, half_dim, dtype=jnp.float32) / half_dim))
    positions = jnp.arange(seq_len, dtype=jnp.float32)
    angles = positions[:, None] * inv_freq[None, :]
    return jnp.cos(angles), jnp.sin(angles)


@named_call
def _apply_rotary_embedding(
    q: Float[Array, "B S H D"],
    k: Float[Array, "B S H D"],
    *,
    seq_len: int,
    head_dim: int,
    rope_theta: float,
) -> tuple[Float[Array, "B S H D"], Float[Array, "B S H D"]]:
    cos, sin = _rotary_cache(seq_len, head_dim, rope_theta)
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]

    def apply(x: Float[Array, "B S H D"]) -> Float[Array, "B S H D"]:
        x1, x2 = jnp.split(x, 2, axis=-1)
        return jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)

    return apply(q), apply(k)


def _layer_mask(
    mask: AttentionMask | jax.Array | None,
    *,
    layer_type: LayerType,
    sliding_window: int,
) -> AttentionMask | jax.Array:
    if isinstance(mask, AttentionMask):
        if layer_type == "sliding_attention":
            return mask.with_sliding_window(sliding_window)
        if layer_type == "full_attention":
            return mask.with_sliding_window(None)
        return mask
    if mask is None:
        if layer_type == "sliding_attention":
            return AttentionMask.causal(sliding_window=sliding_window)
        return AttentionMask.causal()
    return mask


def _full_attention_token_flops(seq_len: int, num_heads: int, head_dim: int) -> float:
    key_query_logits = 2 * seq_len * seq_len * num_heads * head_dim
    mask = 3 * seq_len * seq_len * num_heads
    mask_value = 2 * seq_len * seq_len * num_heads * head_dim
    return (key_query_logits + mask + mask_value) / seq_len


def _sliding_attention_token_flops(seq_len: int, num_heads: int, head_dim: int, sliding_window: int) -> float:
    window = min(seq_len, sliding_window)
    key_query_logits = 2 * seq_len * window * num_heads * head_dim
    mask = 3 * seq_len * window * num_heads
    mask_value = 2 * seq_len * window * num_heads * head_dim
    return (key_query_logits + mask + mask_value) / seq_len


def _linear_siso_kernel_token_flops(*, num_heads: int, chunk_size: int, state_dim: int, value_dim: int) -> float:
    return 2.0 * num_heads * (chunk_size * state_dim + chunk_size * value_dim + 2 * state_dim * value_dim)


def _linear_mimo_kernel_token_flops(
    *,
    num_heads: int,
    chunk_size: int,
    state_dim: int,
    value_dim: int,
    rank: int,
) -> float:
    return (
        2.0
        * num_heads
        * (
            chunk_size * state_dim * rank * rank
            + chunk_size * value_dim * rank * rank
            + 2 * state_dim * value_dim * rank
            + (state_dim + value_dim) * rank * rank
        )
    )


__all__ = [
    "MLP",
    "CausalSelfAttention",
    "HybridBlock",
    "HybridModelConfig",
    "LayerType",
    "LinearMode",
    "Mamba3MimoMixer",
    "Mamba3SisoMixer",
    "RMSNorm",
    "Transformer",
]
