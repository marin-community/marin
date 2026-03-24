# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Grug hybrid-attention sweep with Mamba-3 linear blocks.

This is a copy-paste Grug-style speedrun surface for testing hybrid layer
patterns that mix:
- full Splash attention,
- sliding-window Splash attention, and
- Mamba-3 linear blocks (SISO or MIMO).

The sweep targets a fixed training budget per run and compares:
- full-attention baseline,
- 3x sliding-window + 1x full baseline,
- SWA -> Linear -> SWA -> Full,
- SWA -> Linear -> Linear -> Full.
"""

# nodryrun
from __future__ import annotations

import json
import math
import os
import re
import shutil
import sqlite3
import tempfile
import sys
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, Literal

import equinox as eqx
import fsspec
import jax
import jax.numpy as jnp
import jax.random as random
from einops import rearrange
from fray.cluster import ResourceConfig
from haliax.jax_utils import named_call
from jax.sharding import PartitionSpec as P
from jax.sharding import reshard
from jaxtyping import Array, Float, Int, PRNGKeyArray

from experiments.defaults import default_validation_sets
from experiments.grug.hybrid_mamba3.launch import GrugHybridLaunchConfig, run_grug_hybrid_trial
from experiments.grug.hybrid_mamba3.model import HybridModelConfig as NativeHybridModelConfig
from experiments.grug.hybrid_mamba3.optimizer import HybridSplitAdamConfig
from experiments.grug.hybrid_mamba3.train import GrugEvalConfig, GrugTrainerConfig
from experiments.llama import llama3_tokenizer_vocab_size
from experiments.pretraining_datasets import nemotron_mix_block_shuffle
from experiments.simple_train_config import SimpleTrainConfig
from levanter.callbacks.profiler import ProfilerConfig
from levanter.grug.attention import AttentionMask, attention
from levanter.grug.loss import fused_linear_softmax_cross_entropy_loss
from levanter.grug.sharding import Pbatch, Pembed_vocab, Plm_head, unshard
from levanter.kernels.pallas.mamba3 import HybridModeConfig, mamba3_hybrid_chunked_forward
from levanter.optim import AdamConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path
from marin.execution.remote import remote
from marin.processing.tokenize import add_validation_sets_to_mixture

LayerType = Literal["full_attention", "sliding_attention", "linear_attention"]
LinearMode = Literal["siso", "mimo"]
SearchValue = float | int | str

TARGET_TRAIN_FLOPS = 3e18
DEFAULT_SEQ_LEN = 4096
SLIDING_WINDOWS = (1024, 2048)
FULL_ATTENTION_PATTERN = "full"
SWA_BASELINE_PATTERN = "swa3full1"
HYBRID_SWA_LINEAR_SWA_FULL = "swa-linear-swa-full"
HYBRID_SWA_LINEAR_LINEAR_FULL = "swa-linear-linear-full"
HYBRID_LINEAR_LINEAR_LINEAR_FULL = "linear3full1"
GRID_LAUNCH_MODE = "grid"
VIZIER_LAUNCH_MODE = "vizier"
TRACKER_METRICS_FILENAME = "tracker_metrics.jsonl"
SPEEDRUN_METRIC_KEY = "eval/paloma/c4_en/bpb"
SUGGESTIONS_FILENAME = "vizier_suggestions.json"
UPDATE_FILENAME = "vizier_update.json"
RESOURCE_FILENAME = "vizier_resource.json"
OPTIMAL_FILENAME = "vizier_optimal.json"
VIZIER_DB_FILENAME = "vizier.db"

HYBRID_DATA_WITH_EVAL = add_validation_sets_to_mixture(
    nemotron_mix_block_shuffle,
    default_validation_sets(tokenizer=nemotron_mix_block_shuffle.tokenizer),
)


def _init_weight(key: PRNGKeyArray, shape: tuple[int, ...], std: float) -> Float[Array, ...]:
    return std * random.truncated_normal(key, -3, 3, shape)


def _repeat_motif(motif: tuple[LayerType, ...], num_layers: int) -> tuple[LayerType, ...]:
    if num_layers % len(motif) != 0:
        raise ValueError(f"num_layers={num_layers} must be divisible by motif length {len(motif)}")
    return motif * (num_layers // len(motif))


def _pattern_to_layer_types(pattern: str, *, num_layers: int) -> tuple[LayerType, ...]:
    if pattern == FULL_ATTENTION_PATTERN:
        return ("full_attention",) * num_layers
    if pattern == SWA_BASELINE_PATTERN:
        return _repeat_motif(
            ("sliding_attention", "sliding_attention", "sliding_attention", "full_attention"), num_layers
        )
    if pattern == HYBRID_SWA_LINEAR_SWA_FULL:
        return _repeat_motif(
            ("sliding_attention", "linear_attention", "sliding_attention", "full_attention"), num_layers
        )
    if pattern == HYBRID_SWA_LINEAR_LINEAR_FULL:
        return _repeat_motif(("sliding_attention", "linear_attention", "linear_attention", "full_attention"), num_layers)
    if pattern == HYBRID_LINEAR_LINEAR_LINEAR_FULL:
        return _repeat_motif(("linear_attention", "linear_attention", "linear_attention", "full_attention"), num_layers)
    raise ValueError(f"Unknown pattern: {pattern}")


def _pattern_uses_sliding_window(pattern: str) -> bool:
    return pattern in {SWA_BASELINE_PATTERN, HYBRID_SWA_LINEAR_SWA_FULL, HYBRID_SWA_LINEAR_LINEAR_FULL}


def _linear_layer_fraction(layer_types: tuple[LayerType, ...]) -> float:
    linear_layers = sum(layer_type == "linear_attention" for layer_type in layer_types)
    return linear_layers / len(layer_types)


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


def _repoint_modules_for_ray(classes_in_main: list[type]) -> None:
    import_path = getattr(__spec__, "name", __name__)
    sys.modules[import_path] = sys.modules[__name__]
    for cls in classes_in_main:
        cls.__module__ = import_path


@dataclass(frozen=True)
class HybridModelConfig:
    vocab_size: int = llama3_tokenizer_vocab_size
    hidden_dim: int = 512
    intermediate_dim: int = 1792
    num_layers: int = 12
    num_heads: int = 8
    max_seq_len: int = DEFAULT_SEQ_LEN
    sliding_window: int = 1024
    layer_types: tuple[LayerType, ...] = ()
    linear_mode: LinearMode = "siso"
    linear_state_dim: int = 64
    linear_rank: int = 4
    layer_norm_eps: float = 1e-5
    initializer_std: float = 0.02
    rope_theta: float = 10_000
    width_label: str = "d512"
    pattern_label: str = FULL_ATTENTION_PATTERN

    def __post_init__(self) -> None:
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(f"hidden_dim={self.hidden_dim} must be divisible by num_heads={self.num_heads}")
        if len(self.layer_types) != self.num_layers:
            raise ValueError("layer_types must have one entry per layer")
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
            mode=self.linear_mode, mimo_rank=self.linear_rank if self.linear_mode == "mimo" else None
        )
        return mode_cfg.resolved_chunk_size()

    @property
    def total_trainable_params(self) -> int:
        token_embedding = self.vocab_size * self.hidden_dim
        layer_norms = 2 * self.hidden_dim
        mlp = 2 * self.hidden_dim * self.intermediate_dim
        attention = 4 * self.hidden_dim * self.hidden_dim

        linear_common = 2 * self.hidden_dim + 1
        if self.linear_mode == "siso":
            linear = (
                2 * self.hidden_dim * self.num_heads * self.linear_state_dim
                + self.hidden_dim * self.hidden_dim
                + self.hidden_dim * self.hidden_dim
                + 2 * self.hidden_dim * self.num_heads
                + self.num_heads * linear_common
            )
        else:
            linear = (
                2 * self.hidden_dim * self.num_heads * self.linear_rank * self.linear_state_dim
                + 3 * self.hidden_dim * self.hidden_dim
                + 2 * self.hidden_dim * self.num_heads
                + 3 * self.hidden_dim * self.linear_rank
                + self.num_heads * linear_common
            )

        layers = 0
        for layer_type in self.layer_types:
            mixer = attention if layer_type != "linear_attention" else linear
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
            + 4 * self.hidden_dim * self.hidden_dim
            + 4 * self.hidden_dim * self.num_heads
        )
        linear_mimo_proj = (
            4 * self.hidden_dim * self.num_heads * self.linear_rank * self.linear_state_dim
            + 6 * self.hidden_dim * self.hidden_dim
            + 4 * self.hidden_dim * self.num_heads
        )

        total = lm_head
        for layer_type in self.layer_types:
            if layer_type == "full_attention":
                mixer = attention_proj + _full_attention_token_flops(self.max_seq_len, self.num_heads, self.head_dim)
            elif layer_type == "sliding_attention":
                mixer = attention_proj + _sliding_attention_token_flops(
                    self.max_seq_len, self.num_heads, self.head_dim, self.sliding_window
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


class MLP(eqx.Module):
    mlp_up: jax.Array
    mlp_down: jax.Array

    @staticmethod
    def init(cfg: HybridModelConfig, *, key: PRNGKeyArray) -> MLP:
        k_up, k_down = random.split(key, 2)
        return MLP(
            mlp_up=reshard(
                _init_weight(k_up, (cfg.hidden_dim, cfg.intermediate_dim), cfg.initializer_std), P("data", "model")
            ),
            mlp_down=reshard(
                _init_weight(k_down, (cfg.intermediate_dim, cfg.hidden_dim), cfg.initializer_std), P("model", "data")
            ),
        )

    @named_call
    def __call__(self, x: Float[Array, "B S D"]) -> Float[Array, "B S D"]:
        up = jnp.einsum("bsh,hm->bsm", x, self.mlp_up)
        activated = jax.nn.relu(up)
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
                _init_weight(k_q, (cfg.hidden_dim, cfg.num_heads * head_dim), cfg.initializer_std), P("data", "model")
            ),
            w_k=reshard(
                _init_weight(k_k, (cfg.hidden_dim, cfg.num_heads * head_dim), cfg.initializer_std), P("data", "model")
            ),
            w_v=reshard(
                _init_weight(k_v, (cfg.hidden_dim, cfg.num_heads * head_dim), cfg.initializer_std), P("data", "model")
            ),
            w_o=reshard(
                _init_weight(k_o, (cfg.num_heads * head_dim, cfg.hidden_dim), cfg.initializer_std), P("model", "data")
            ),
            cfg=cfg,
        )

    @named_call
    def __call__(self, x: Float[Array, "B S D"], mask: AttentionMask | jax.Array) -> Float[Array, "B S D"]:
        head_dim = self.cfg.head_dim
        q = rearrange(jnp.einsum("bsh,hd->bsd", x, self.w_q), "... (h d) -> ... h d", h=self.cfg.num_heads, d=head_dim)
        k = rearrange(jnp.einsum("bsh,hd->bsd", x, self.w_k), "... (h d) -> ... h d", h=self.cfg.num_heads, d=head_dim)
        v = rearrange(jnp.einsum("bsh,hd->bsd", x, self.w_v), "... (h d) -> ... h d", h=self.cfg.num_heads, d=head_dim)
        q, k = _apply_rotary_embedding(q, k, seq_len=x.shape[1], head_dim=head_dim, rope_theta=self.cfg.rope_theta)
        attn_out = attention(q, k, v, mask)
        attn_out = rearrange(attn_out, "... h d -> ... (h d)")
        return jnp.einsum("bsh,hd->bsd", attn_out, self.w_o, out_sharding=Pbatch)


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


class Mamba3SisoMixer(eqx.Module):
    w_b: jax.Array
    w_c: jax.Array
    w_x: jax.Array
    w_dt: jax.Array
    w_lam: jax.Array
    w_o: jax.Array
    a_log: jax.Array
    dt_bias: jax.Array
    lam_bias: jax.Array
    cfg: HybridModelConfig = eqx.field(static=True)

    @staticmethod
    def init(cfg: HybridModelConfig, *, key: PRNGKeyArray) -> Mamba3SisoMixer:
        keys = random.split(key, 9)
        flat_state = cfg.num_heads * cfg.linear_state_dim
        flat_value = cfg.hidden_dim
        return Mamba3SisoMixer(
            w_b=reshard(_init_weight(keys[0], (cfg.hidden_dim, flat_state), cfg.initializer_std), P("data", "model")),
            w_c=reshard(_init_weight(keys[1], (cfg.hidden_dim, flat_state), cfg.initializer_std), P("data", "model")),
            w_x=reshard(_init_weight(keys[2], (cfg.hidden_dim, flat_value), cfg.initializer_std), P("data", "model")),
            w_dt=reshard(
                _init_weight(keys[3], (cfg.hidden_dim, cfg.num_heads), cfg.initializer_std), P("data", "model")
            ),
            w_lam=reshard(
                _init_weight(keys[4], (cfg.hidden_dim, cfg.num_heads), cfg.initializer_std), P("data", "model")
            ),
            w_o=reshard(_init_weight(keys[5], (flat_value, cfg.hidden_dim), cfg.initializer_std), P("model", "data")),
            a_log=reshard(jnp.zeros((cfg.num_heads,), dtype=jnp.float32), P(None)),
            dt_bias=reshard(jnp.zeros((cfg.num_heads,), dtype=jnp.float32), P(None)),
            lam_bias=reshard(jnp.zeros((cfg.num_heads,), dtype=jnp.float32), P(None)),
            cfg=cfg,
        )

    @named_call
    def __call__(self, x: Float[Array, "B S D"]) -> Float[Array, "B S D"]:
        chunk = self.cfg.linear_chunk_size
        chunks = x.shape[1] // chunk
        heads = self.cfg.num_heads
        state_dim = self.cfg.linear_state_dim
        value_dim = self.cfg.head_dim

        b = rearrange(
            jnp.einsum("bsh,hd->bsd", x, self.w_b),
            "b (chunks chunk) (h n) -> b h chunks chunk n",
            chunks=chunks,
            chunk=chunk,
            h=heads,
            n=state_dim,
        )
        c = rearrange(
            jnp.einsum("bsh,hd->bsd", x, self.w_c),
            "b (chunks chunk) (h n) -> b h chunks chunk n",
            chunks=chunks,
            chunk=chunk,
            h=heads,
            n=state_dim,
        )
        x_base = rearrange(
            jnp.einsum("bsh,hd->bsd", x, self.w_x),
            "b (chunks chunk) (h p) -> b h chunks chunk p",
            chunks=chunks,
            chunk=chunk,
            h=heads,
            p=value_dim,
        )
        dt = rearrange(
            jax.nn.softplus(jnp.einsum("bsd,dh->bsh", x, self.w_dt) + self.dt_bias[None, None, :]) + 1e-4,
            "b (chunks chunk) h -> b h chunks chunk",
            chunks=chunks,
            chunk=chunk,
            h=heads,
        )
        lam = rearrange(
            jax.nn.sigmoid(jnp.einsum("bsd,dh->bsh", x, self.w_lam) + self.lam_bias[None, None, :]),
            "b (chunks chunk) h -> b h chunks chunk",
            chunks=chunks,
            chunk=chunk,
            h=heads,
        )
        a = jnp.broadcast_to(-jax.nn.softplus(self.a_log)[None, :, None], dt.shape[:-1])
        y_chunked, _ = mamba3_hybrid_chunked_forward(dt, lam, a, b, c, x_base, mode="siso")
        y = rearrange(y_chunked, "b h chunks chunk p -> b (chunks chunk) (h p)")
        return jnp.einsum("bsh,hd->bsd", y, self.w_o, out_sharding=Pbatch)


class Mamba3MimoMixer(eqx.Module):
    w_b: jax.Array
    w_c: jax.Array
    w_x_base: jax.Array
    w_z_base: jax.Array
    w_dt: jax.Array
    w_lam: jax.Array
    w_o: jax.Array
    w_rank_x: jax.Array
    w_rank_z: jax.Array
    w_rank_o: jax.Array
    a_log: jax.Array
    dt_bias: jax.Array
    lam_bias: jax.Array
    cfg: HybridModelConfig = eqx.field(static=True)

    @staticmethod
    def init(cfg: HybridModelConfig, *, key: PRNGKeyArray) -> Mamba3MimoMixer:
        keys = random.split(key, 13)
        flat_state = cfg.num_heads * cfg.linear_rank * cfg.linear_state_dim
        flat_value = cfg.hidden_dim
        rank_weight_shape = (cfg.num_heads, cfg.head_dim, cfg.linear_rank)
        return Mamba3MimoMixer(
            w_b=reshard(_init_weight(keys[0], (cfg.hidden_dim, flat_state), cfg.initializer_std), P("data", "model")),
            w_c=reshard(_init_weight(keys[1], (cfg.hidden_dim, flat_state), cfg.initializer_std), P("data", "model")),
            w_x_base=reshard(
                _init_weight(keys[2], (cfg.hidden_dim, flat_value), cfg.initializer_std), P("data", "model")
            ),
            w_z_base=reshard(
                _init_weight(keys[3], (cfg.hidden_dim, flat_value), cfg.initializer_std), P("data", "model")
            ),
            w_dt=reshard(
                _init_weight(keys[4], (cfg.hidden_dim, cfg.num_heads), cfg.initializer_std), P("data", "model")
            ),
            w_lam=reshard(
                _init_weight(keys[5], (cfg.hidden_dim, cfg.num_heads), cfg.initializer_std), P("data", "model")
            ),
            w_o=reshard(_init_weight(keys[6], (flat_value, cfg.hidden_dim), cfg.initializer_std), P("model", "data")),
            w_rank_x=reshard(_init_weight(keys[7], rank_weight_shape, cfg.initializer_std), P(None, None, None)),
            w_rank_z=reshard(_init_weight(keys[8], rank_weight_shape, cfg.initializer_std), P(None, None, None)),
            w_rank_o=reshard(_init_weight(keys[9], rank_weight_shape, cfg.initializer_std), P(None, None, None)),
            a_log=reshard(jnp.zeros((cfg.num_heads,), dtype=jnp.float32), P(None)),
            dt_bias=reshard(jnp.zeros((cfg.num_heads,), dtype=jnp.float32), P(None)),
            lam_bias=reshard(jnp.zeros((cfg.num_heads,), dtype=jnp.float32), P(None)),
            cfg=cfg,
        )

    @named_call
    def __call__(self, x: Float[Array, "B S D"]) -> Float[Array, "B S D"]:
        chunk = self.cfg.linear_chunk_size
        chunks = x.shape[1] // chunk
        heads = self.cfg.num_heads
        rank = self.cfg.linear_rank
        state_dim = self.cfg.linear_state_dim
        value_dim = self.cfg.head_dim

        b = rearrange(
            jnp.einsum("bsh,hd->bsd", x, self.w_b),
            "b (chunks chunk) (h r n) -> b h chunks chunk n r",
            chunks=chunks,
            chunk=chunk,
            h=heads,
            r=rank,
            n=state_dim,
        )
        c = rearrange(
            jnp.einsum("bsh,hd->bsd", x, self.w_c),
            "b (chunks chunk) (h r n) -> b h chunks chunk n r",
            chunks=chunks,
            chunk=chunk,
            h=heads,
            r=rank,
            n=state_dim,
        )
        x_base = rearrange(
            jnp.einsum("bsh,hd->bsd", x, self.w_x_base),
            "b (chunks chunk) (h p) -> b h chunks chunk p",
            chunks=chunks,
            chunk=chunk,
            h=heads,
            p=value_dim,
        )
        z_base = rearrange(
            jnp.einsum("bsh,hd->bsd", x, self.w_z_base),
            "b (chunks chunk) (h p) -> b h chunks chunk p",
            chunks=chunks,
            chunk=chunk,
            h=heads,
            p=value_dim,
        )
        dt = rearrange(
            jax.nn.softplus(jnp.einsum("bsd,dh->bsh", x, self.w_dt) + self.dt_bias[None, None, :]) + 1e-4,
            "b (chunks chunk) h -> b h chunks chunk",
            chunks=chunks,
            chunk=chunk,
            h=heads,
        )
        lam = rearrange(
            jax.nn.sigmoid(jnp.einsum("bsd,dh->bsh", x, self.w_lam) + self.lam_bias[None, None, :]),
            "b (chunks chunk) h -> b h chunks chunk",
            chunks=chunks,
            chunk=chunk,
            h=heads,
        )
        a = jnp.broadcast_to(-jax.nn.softplus(self.a_log)[None, :, None], dt.shape[:-1])
        y_chunked, _ = mamba3_hybrid_chunked_forward(
            dt,
            lam,
            a,
            b,
            c,
            x_base,
            mode="mimo",
            z=z_base,
            w_x=self.w_rank_x,
            w_z=self.w_rank_z,
            w_o=self.w_rank_o,
        )
        y = rearrange(y_chunked, "b h chunks chunk p -> b (chunks chunk) (h p)")
        return jnp.einsum("bsh,hd->bsd", y, self.w_o, out_sharding=Pbatch)


def _layer_mask(
    mask: AttentionMask | jax.Array | None, *, layer_type: LayerType, sliding_window: int
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
                self.rms_mixer(x), _layer_mask(mask, layer_type=self.layer_type, sliding_window=self.sliding_window)
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
            _init_weight(embed_key, (cfg.vocab_size, cfg.hidden_dim), cfg.initializer_std), Pembed_vocab
        )
        output_proj = reshard(_init_weight(out_key, (cfg.hidden_dim, cfg.vocab_size), cfg.initializer_std), Plm_head)
        blocks: list[HybridBlock] = []
        for layer_type, block_key in zip(cfg.layer_types, block_keys, strict=True):
            blocks.append(HybridBlock.init(cfg, layer_type=layer_type, key=block_key))
        return Transformer(
            token_embed=token_embed,
            output_proj=output_proj,
            blocks=tuple(blocks),
            final_norm=RMSNorm.init(cfg.hidden_dim, cfg.layer_norm_eps),
            config=cfg,
        )

    @named_call
    def __call__(self, token_ids: Int[Array, "B S"], mask: AttentionMask | jax.Array | None) -> Float[Array, "B S D"]:
        if mask is None:
            mask = AttentionMask.causal()
        hidden = self.token_embed.at[token_ids].get(out_sharding=Pbatch)
        for block in self.blocks:
            hidden = eqx.filter_checkpoint(block)(hidden, mask)
        return self.final_norm(hidden)


def loss_fn(
    transformer: Transformer,
    token_ids: Int[Array, "B S"],
    loss_weight: Float[Array, "B S"],
    cfg: HybridModelConfig,
    *,
    mask: AttentionMask | jax.Array | None = None,
    reduction: str = "mean",
    logsumexp_weight: float | None = None,
    loss_dtype: jnp.dtype = jnp.float32,
) -> jax.Array:
    hidden = transformer(token_ids, mask=mask)
    labels = jnp.concatenate([token_ids[:, 1:], token_ids[:, :1] * 0], axis=1).astype(jnp.int32)
    return fused_linear_softmax_cross_entropy_loss(
        hidden,
        transformer.output_proj,
        labels,
        weight=loss_weight.astype(loss_dtype),
        reduction=reduction,
        logsumexp_weight=logsumexp_weight,
        dtype=loss_dtype,
    )


def _size_presets() -> dict[str, dict[str, int]]:
    return {
        "d512": {"hidden_dim": 512, "intermediate_dim": 1792, "num_layers": 12, "num_heads": 8},
        "d640": {"hidden_dim": 640, "intermediate_dim": 2304, "num_layers": 12, "num_heads": 10},
        "d768": {"hidden_dim": 768, "intermediate_dim": 2688, "num_layers": 12, "num_heads": 12},
    }


def _adamw_decay_targets() -> list[str]:
    return ["w_", "mlp_", "token_embed", "output_proj"]


def _optimizer_presets() -> dict[str, AdamConfig]:
    return {
        "d512": AdamConfig(
            learning_rate=0.0025,
            weight_decay=0.1,
            min_lr_ratio=0.1,
            warmup=256,
            beta1=0.9,
            beta2=0.95,
            epsilon=1e-8,
            max_grad_norm=1,
            lr_schedule="cosine",
            decay=1.0,
            weight_decay_modules=_adamw_decay_targets(),
        ),
        "d640": AdamConfig(
            learning_rate=0.002,
            weight_decay=0.1,
            min_lr_ratio=0.1,
            warmup=256,
            beta1=0.9,
            beta2=0.95,
            epsilon=1e-8,
            max_grad_norm=1,
            lr_schedule="cosine",
            decay=1.0,
            weight_decay_modules=_adamw_decay_targets(),
        ),
        "d768": AdamConfig(
            learning_rate=0.0016,
            weight_decay=0.1,
            min_lr_ratio=0.1,
            warmup=256,
            beta1=0.9,
            beta2=0.95,
            epsilon=1e-8,
            max_grad_norm=1,
            lr_schedule="cosine",
            decay=1.0,
            weight_decay_modules=_adamw_decay_targets(),
        ),
    }


def _batch_size_for_width(width_label: str) -> int:
    return {"d512": 256, "d640": 192, "d768": 128}[width_label]


def _batch_size_for_model(model_cfg: NativeHybridModelConfig) -> int:
    batch_size = _batch_size_for_width(model_cfg.width_label)
    linear_fraction = _linear_layer_fraction(model_cfg.layer_types)
    if model_cfg.linear_mode != "mimo" or linear_fraction == 0:
        return batch_size
    if linear_fraction >= 0.75:
        return max(32, batch_size // 2)
    return max(32, (batch_size * 3) // 4)


def _optimizer_for_model(model_cfg: NativeHybridModelConfig) -> AdamConfig:
    optimizer = _optimizer_presets()[model_cfg.width_label]
    linear_fraction = _linear_layer_fraction(model_cfg.layer_types)
    if linear_fraction == 0:
        return optimizer

    lr_scale = 0.7
    warmup = 512
    max_grad_norm = optimizer.max_grad_norm

    if linear_fraction >= 0.75:
        lr_scale = 0.55
        warmup = 768

    if model_cfg.linear_mode == "mimo":
        lr_scale = 0.4
        warmup = 1024
        max_grad_norm = min(max_grad_norm, 0.5)
        if linear_fraction >= 0.75:
            lr_scale = 0.35
            warmup = 1536
            max_grad_norm = min(max_grad_norm, 0.4)

    return replace(
        optimizer,
        learning_rate=optimizer.learning_rate * lr_scale,
        warmup=max(optimizer.warmup, warmup),
        max_grad_norm=max_grad_norm,
    )


def _split_optimizer_for_model(
    model_cfg: NativeHybridModelConfig,
    *,
    mamba_learning_rate: float,
    transformer_learning_rate: float,
) -> HybridSplitAdamConfig:
    base_optimizer = _optimizer_for_model(model_cfg)
    return HybridSplitAdamConfig(
        learning_rate=transformer_learning_rate,
        mamba_learning_rate=mamba_learning_rate,
        weight_decay=base_optimizer.weight_decay,
        min_lr_ratio=base_optimizer.min_lr_ratio,
        warmup=base_optimizer.warmup,
        decay=base_optimizer.decay,
        rewarmup=base_optimizer.rewarmup,
        cooldown=base_optimizer.cooldown,
        cycle_length=base_optimizer.cycle_length,
        cycles=base_optimizer.cycles,
        lr_schedule=base_optimizer.lr_schedule,
        haps=base_optimizer.haps,
        weight_decay_modules=base_optimizer.weight_decay_modules,
        default_weight_decay_mask=base_optimizer.default_weight_decay_mask,
        beta1=base_optimizer.beta1,
        beta2=base_optimizer.beta2,
        epsilon=base_optimizer.epsilon,
        max_grad_norm=base_optimizer.max_grad_norm,
        nesterov=base_optimizer.nesterov,
        update_rms_clipping=base_optimizer.update_rms_clipping,
        clip_update_norm=base_optimizer.clip_update_norm,
        skip_bad_steps=base_optimizer.skip_bad_steps,
        adamc_weight_decay=base_optimizer.adamc_weight_decay,
    )


def _steps_for_budget(model_cfg: NativeHybridModelConfig, batch_size: int) -> int:
    tokens_per_step = batch_size * model_cfg.max_seq_len
    step_flops = model_cfg.flops_per_token * 3 * tokens_per_step
    return max(1, math.ceil(TARGET_TRAIN_FLOPS / step_flops))


def _build_model_config(
    *,
    width_label: str,
    pattern_label: str,
    sliding_window: int,
    linear_mode: LinearMode,
) -> NativeHybridModelConfig:
    preset = _size_presets()[width_label]
    layer_types = _pattern_to_layer_types(pattern_label, num_layers=preset["num_layers"])
    return NativeHybridModelConfig(
        vocab_size=llama3_tokenizer_vocab_size,
        hidden_dim=preset["hidden_dim"],
        intermediate_dim=preset["intermediate_dim"],
        num_layers=preset["num_layers"],
        num_heads=preset["num_heads"],
        max_seq_len=DEFAULT_SEQ_LEN,
        sliding_window=sliding_window,
        layer_types=layer_types,
        linear_mode=linear_mode,
        linear_state_dim=64,
        linear_rank=4,
        width_label=width_label,
        pattern_label=pattern_label,
    )


def _build_train_config(
    model_cfg: NativeHybridModelConfig,
    *,
    optimizer_override: AdamConfig | HybridSplitAdamConfig | None = None,
) -> SimpleTrainConfig:
    batch_size = _batch_size_for_model(model_cfg)
    optimizer = optimizer_override or _optimizer_for_model(model_cfg)
    learning_rate = getattr(optimizer, "learning_rate", _optimizer_for_model(model_cfg).learning_rate)
    return SimpleTrainConfig(
        resources=ResourceConfig.with_tpu("v5p-8"),
        train_batch_size=batch_size,
        learning_rate=learning_rate,
        explicit_mesh_axes=True,
        profiler=ProfilerConfig(enabled=False),
        train_seq_len=model_cfg.max_seq_len,
        num_train_steps=_steps_for_budget(model_cfg, batch_size),
        steps_per_hf_export=-1,
        optimizer_config=optimizer,
    )


def _run_name(model_cfg: NativeHybridModelConfig) -> str:
    mode = f"-{model_cfg.linear_mode}" if "linear_attention" in model_cfg.layer_types else ""
    window = f"-sw{model_cfg.sliding_window}" if _pattern_uses_sliding_window(model_cfg.pattern_label) else ""
    suffix = os.environ.get("GRUG_HYBRID_RUN_SUFFIX", "").strip()
    if suffix:
        suffix = suffix if suffix.startswith("-") else f"-{suffix}"
    return f"grug-hybrid-{model_cfg.width_label}-{model_cfg.pattern_label}{mode}{window}{suffix}"


def _description(model_cfg: NativeHybridModelConfig) -> str:
    mode = f"; linear={model_cfg.linear_mode}" if "linear_attention" in model_cfg.layer_types else ""
    return (
        f"Grug hybrid sweep ({model_cfg.width_label}); pattern={model_cfg.pattern_label}; "
        f"attention=Splash/full+sliding{mode}; target_flops={TARGET_TRAIN_FLOPS:.1e}"
    )


def _selected_widths() -> list[str]:
    requested = os.environ.get("GRUG_HYBRID_WIDTHS")
    if requested is None:
        return list(_size_presets().keys())
    return [width.strip() for width in requested.split(",") if width.strip()]


def _selected_patterns() -> list[str]:
    requested = os.environ.get("GRUG_HYBRID_PATTERNS")
    defaults = [
        FULL_ATTENTION_PATTERN,
        SWA_BASELINE_PATTERN,
        HYBRID_SWA_LINEAR_SWA_FULL,
        HYBRID_SWA_LINEAR_LINEAR_FULL,
    ]
    if requested is None:
        return defaults
    return [pattern.strip() for pattern in requested.split(",") if pattern.strip()]


def _selected_modes() -> list[LinearMode]:
    requested = os.environ.get("GRUG_HYBRID_LINEAR_MODES")
    if requested is None:
        return ["siso", "mimo"]
    return [mode.strip() for mode in requested.split(",") if mode.strip()]  # type: ignore[return-value]


def _selected_sliding_windows() -> list[int]:
    requested = os.environ.get("GRUG_HYBRID_SLIDING_WINDOWS")
    if requested is None:
        return list(SLIDING_WINDOWS)
    return [int(window.strip()) for window in requested.split(",") if window.strip()]


def _selected_split_lr_ladder() -> list[tuple[float, float]]:
    requested = os.environ.get("GRUG_HYBRID_SPLIT_LR_LADDER")
    if requested is None or not requested.strip():
        return []

    pairs: list[tuple[float, float]] = []
    for raw_pair in requested.split(","):
        raw_pair = raw_pair.strip()
        if not raw_pair:
            continue
        try:
            mamba_lr, transformer_lr = raw_pair.split(":", 1)
        except ValueError as exc:
            raise ValueError(
                "GRUG_HYBRID_SPLIT_LR_LADDER entries must be formatted as 'mamba_lr:transformer_lr'"
            ) from exc
        pairs.append((float(mamba_lr), float(transformer_lr)))
    return pairs


def _lr_label(lr: float) -> str:
    return f"{lr * 1e3:.1f}".replace(".", "p")


def _split_lr_suffix(*, mamba_lr: float, transformer_lr: float) -> str:
    return f"-m{_lr_label(mamba_lr)}-t{_lr_label(transformer_lr)}"


def _build_sweep() -> list[tuple[str, NativeHybridModelConfig]]:
    runs: list[tuple[str, NativeHybridModelConfig]] = []
    sliding_windows = _selected_sliding_windows()
    for width_label in _selected_widths():
        for pattern_label in _selected_patterns():
            if pattern_label == FULL_ATTENTION_PATTERN:
                runs.append(
                    (
                        _run_name(
                            _build_model_config(
                                width_label=width_label,
                                pattern_label=pattern_label,
                                sliding_window=sliding_windows[0],
                                linear_mode="siso",
                            )
                        ),
                        _build_model_config(
                            width_label=width_label,
                            pattern_label=pattern_label,
                            sliding_window=sliding_windows[0],
                            linear_mode="siso",
                        ),
                    )
                )
                continue
            if pattern_label == SWA_BASELINE_PATTERN:
                for sliding_window in sliding_windows:
                    model_cfg = _build_model_config(
                        width_label=width_label,
                        pattern_label=pattern_label,
                        sliding_window=sliding_window,
                        linear_mode="siso",
                    )
                    runs.append((_run_name(model_cfg), model_cfg))
                continue
            pattern_sliding_windows = (
                sliding_windows if _pattern_uses_sliding_window(pattern_label) else [sliding_windows[0]]
            )
            for sliding_window in pattern_sliding_windows:
                for linear_mode in _selected_modes():
                    model_cfg = _build_model_config(
                        width_label=width_label,
                        pattern_label=pattern_label,
                        sliding_window=sliding_window,
                        linear_mode=linear_mode,
                    )
                    runs.append((_run_name(model_cfg), model_cfg))
    return runs


def _candidate_catalog() -> dict[str, NativeHybridModelConfig]:
    return {run_name: model_cfg for run_name, model_cfg in _build_sweep()}


def _build_grid_runs() -> list[GridRunSpec]:
    run_specs: list[GridRunSpec] = []
    split_lr_ladder = _selected_split_lr_ladder()

    for base_run_name, model_cfg in _build_sweep():
        base_tags = (
            "grug",
            "hybrid",
            "mamba3",
            f"pattern={model_cfg.pattern_label}",
            f"width={model_cfg.width_label}",
            f"window={model_cfg.sliding_window}",
            f"mode={model_cfg.linear_mode}",
        )

        if split_lr_ladder and "linear_attention" in model_cfg.layer_types:
            for mamba_lr, transformer_lr in split_lr_ladder:
                optimizer = _split_optimizer_for_model(
                    model_cfg,
                    mamba_learning_rate=mamba_lr,
                    transformer_learning_rate=transformer_lr,
                )
                run_specs.append(
                    GridRunSpec(
                        run_name=f"{base_run_name}{_split_lr_suffix(mamba_lr=mamba_lr, transformer_lr=transformer_lr)}",
                        model_cfg=model_cfg,
                        train_cfg=_build_train_config(model_cfg, optimizer_override=optimizer),
                        tags=(
                            *base_tags,
                            "split-lr",
                            f"mamba_lr={mamba_lr}",
                            f"transformer_lr={transformer_lr}",
                        ),
                    )
                )
            continue

        run_specs.append(
            GridRunSpec(
                run_name=base_run_name,
                model_cfg=model_cfg,
                train_cfg=_build_train_config(model_cfg),
                tags=base_tags,
            )
        )

    return run_specs


def _build_trial_train_config(
    model_cfg: NativeHybridModelConfig,
    parameters: Mapping[str, SearchValue],
) -> SimpleTrainConfig:
    base_train_cfg = _build_train_config(model_cfg)
    base_optimizer = _optimizer_presets()[model_cfg.width_label]
    lr_multiplier = float(parameters.get("lr_multiplier", 1.0))
    beta2 = float(parameters.get("beta2", base_optimizer.beta2))
    max_grad_norm = float(parameters.get("max_grad_norm", base_optimizer.max_grad_norm))
    weight_decay_multiplier = float(parameters.get("weight_decay_multiplier", 1.0))
    z_loss_weight = float(parameters.get("z_loss_weight", 0.0))

    optimizer = replace(
        base_optimizer,
        learning_rate=base_optimizer.learning_rate * lr_multiplier,
        beta2=beta2,
        max_grad_norm=max_grad_norm,
        weight_decay=base_optimizer.weight_decay * weight_decay_multiplier,
    )
    return replace(
        base_train_cfg,
        learning_rate=optimizer.learning_rate,
        optimizer_config=optimizer,
        z_loss_weight=z_loss_weight,
    )


@dataclass(frozen=True)
class HybridLaunchConfig:
    data: Any
    model_cfg: NativeHybridModelConfig
    train_cfg: SimpleTrainConfig
    output_path: str
    run_name: str
    wandb_group: str
    tags: tuple[str, ...]
    load_checkpoint_path: str | None = None


@dataclass(frozen=True)
class GridRunSpec:
    run_name: str
    model_cfg: NativeHybridModelConfig
    train_cfg: SimpleTrainConfig
    tags: tuple[str, ...]


def _resume_existing_checkpoints_enabled() -> bool:
    return os.environ.get("GRUG_HYBRID_RESUME_EXISTING", "").strip().lower() in {"1", "true", "yes", "on"}


def _checkpoint_root_for_output_path(output_path: str) -> str:
    if output_path.startswith("gs://"):
        bucket = output_path.split("/", 3)[:3]
        return "/".join([*bucket, "checkpoints"])
    return os.path.join(os.path.dirname(output_path), "checkpoints")


def _latest_checkpoint_path(output_path: str, run_name: str) -> str | None:
    if not _resume_existing_checkpoints_enabled():
        return None
    checkpoint_root = _checkpoint_root_for_output_path(output_path)
    fs, _, _ = fsspec.get_fs_token_paths(checkpoint_root)
    matches = fs.glob(os.path.join(checkpoint_root, f"{run_name}-*", "checkpoints", "step-*"))
    latest_step = -1
    latest_path: str | None = None
    for match in matches:
        normalized = match.rstrip("/")
        step_name = os.path.basename(normalized)
        if not step_name.startswith("step-"):
            continue
        try:
            step = int(step_name.split("-", 1)[1])
        except ValueError:
            continue
        if step > latest_step:
            latest_step = step
            latest_path = normalized
    return latest_path


def _build_grug_launch_config(config: HybridLaunchConfig) -> GrugHybridLaunchConfig:
    if not isinstance(config.train_cfg.train_batch_size, int):
        raise ValueError("Hybrid grug sweep expects an integer train_batch_size")
    optimizer = config.train_cfg.optimizer_config
    if optimizer is None:
        raise ValueError("Hybrid grug sweep requires optimizer_config to be set")
    load_checkpoint_path = config.load_checkpoint_path or _latest_checkpoint_path(config.output_path, config.run_name)
    if load_checkpoint_path is not None:
        print(f"[grug-hybrid] Resuming {config.run_name} from {load_checkpoint_path}")
    return GrugHybridLaunchConfig(
        model=config.model_cfg,
        data=config.data,
        output_path=config.output_path,
        run_id=config.run_name,
        resources=config.train_cfg.resources,
        steps=config.train_cfg.num_train_steps,
        batch_size=config.train_cfg.train_batch_size,
        seed=0,
        mp="params=float32,compute=bfloat16,output=bfloat16",
        tracker=WandbConfig(
            project="marin",
            tags=list(config.tags),
            group=config.wandb_group,
            name=None,
            replicate_path=config.output_path,
        ),
        optimizer=optimizer,
        load_checkpoint_path=load_checkpoint_path,
        grug_trainer=GrugTrainerConfig(
            log_every=1,
            ema_beta=config.train_cfg.ema_beta,
            z_loss_weight=config.train_cfg.z_loss_weight or 0.0,
        ),
        eval=GrugEvalConfig(
            eval_batch_size=config.train_cfg.train_batch_size,
            steps_per_eval=config.train_cfg.steps_per_eval,
            max_eval_batches=config.train_cfg.max_eval_batches,
            eval_current=True,
            eval_ema=False,
        ),
    )


def run_hybrid_trial(config: HybridLaunchConfig) -> None:
    run_grug_hybrid_trial(_build_grug_launch_config(config))


@dataclass(frozen=True)
class SearchSpaceParam:
    name: str
    kind: Literal["float", "int", "discrete", "categorical"]
    min_value: float | int | None = None
    max_value: float | int | None = None
    values: tuple[SearchValue, ...] = ()


@dataclass(frozen=True)
class HybridVizierSettings:
    experiment_name: str
    study_owner: str
    num_loops: int
    suggestions_per_loop: int
    metric_file: str
    metric_key: str
    metric_mode: str
    vizier_algorithm: str

    @property
    def study_id(self) -> str:
        return self.experiment_name

    @property
    def study_resource_name(self) -> str:
        return f"owners/{self.study_owner}/studies/{self.study_id}"

    @property
    def client_id_prefix(self) -> str:
        return self.experiment_name


def _vizier_settings() -> HybridVizierSettings:
    return HybridVizierSettings(
        experiment_name=os.environ.get("GRUG_HYBRID_VIZIER_EXPERIMENT", "grug-hybrid-mamba3-vizier-3e18"),
        study_owner=os.environ.get("GRUG_HYBRID_VIZIER_OWNER", "marin"),
        num_loops=int(os.environ.get("GRUG_HYBRID_VIZIER_NUM_LOOPS", "9")),
        suggestions_per_loop=int(os.environ.get("GRUG_HYBRID_VIZIER_SUGGESTIONS_PER_LOOP", "4")),
        metric_file=TRACKER_METRICS_FILENAME,
        metric_key=SPEEDRUN_METRIC_KEY,
        metric_mode="min",
        vizier_algorithm=os.environ.get("GRUG_HYBRID_VIZIER_ALGORITHM", "DEFAULT"),
    )


def _vizier_search_space() -> tuple[SearchSpaceParam, ...]:
    return (
        SearchSpaceParam("architecture_name", "categorical", values=tuple(_candidate_catalog().keys())),
        SearchSpaceParam("lr_multiplier", "discrete", values=(0.8, 1.0, 1.25)),
        SearchSpaceParam("weight_decay_multiplier", "discrete", values=(0.5, 1.0, 1.5)),
        SearchSpaceParam("beta2", "discrete", values=(0.95, 0.97, 0.98)),
        SearchSpaceParam("max_grad_norm", "discrete", values=(0.75, 1.0, 1.25)),
        SearchSpaceParam("z_loss_weight", "discrete", values=(0.0, 5e-6, 2e-5)),
    )


@dataclass(frozen=True)
class VizierSuggestConfig:
    study_owner: str
    study_id: str
    input_db_path: str | None
    output_path: str
    num_suggestions: int
    client_id: str
    metric_key: str
    mode: str
    algorithm: str
    search_space: tuple[SearchSpaceParam, ...]
    loop_index: int


@dataclass(frozen=True)
class VizierTrainConfig:
    suggestions_path: str
    suggestion_index: int
    data: Any
    output_path: str
    loop_index: int


@dataclass(frozen=True)
class VizierUpdateConfig:
    study_id: str
    study_resource_name: str
    input_db_path: str | None
    suggestions_path: str
    run_paths: list[str]
    metric_file: str
    metric_key: str
    mode: str
    output_path: str
    loop_index: int


@dataclass(frozen=True)
class VizierOptimalConfig:
    study_id: str
    study_resource_name: str
    input_db_path: str
    output_path: str


def best_run(runs: list[dict[str, Any]], mode: str = "min") -> dict[str, Any]:
    return (
        min(runs, key=lambda record: record["metric"])
        if mode == "min"
        else max(runs, key=lambda record: record["metric"])
    )


def _local_vizier_db_path(study_id: str) -> str:
    safe_study = re.sub(r"[^A-Za-z0-9_.-]+", "_", study_id)
    return os.path.join(tempfile.gettempdir(), f"vizier-{safe_study}.db")


def _configure_vizier_local_db(local_path: str) -> None:
    from vizier.service import clients

    clients.environment_variables.servicer_kwargs["database_url"] = f"sqlite:///{local_path}"


def _sqlite_sidecar_paths(path: str) -> tuple[str, ...]:
    return (f"{path}-wal", f"{path}-shm", f"{path}-journal")


def _remove_sqlite_sidecars(path: str) -> None:
    for sidecar_path in _sqlite_sidecar_paths(path):
        if os.path.exists(sidecar_path):
            os.remove(sidecar_path)


def _checkpoint_sqlite_db(path: str) -> None:
    if not os.path.exists(path):
        return
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA wal_checkpoint(FULL);")


def _sync_vizier_db_from_gcs(path: str | None, local_path: str) -> bool:
    if not path:
        return False
    fs, _, _ = fsspec.get_fs_token_paths(path)
    if not fs.exists(path):
        return False
    _remove_sqlite_sidecars(local_path)
    with fs.open(path, "rb") as src, open(local_path, "wb") as dst:
        shutil.copyfileobj(src, dst)
    return True


def _sync_vizier_db_to_gcs(local_path: str, path: str) -> None:
    _checkpoint_sqlite_db(local_path)
    _remove_sqlite_sidecars(local_path)
    fs, _, _ = fsspec.get_fs_token_paths(path)
    fs.makedirs(os.path.dirname(path), exist_ok=True)
    with open(local_path, "rb") as src, fs.open(path, "wb") as dst:
        shutil.copyfileobj(src, dst)
    for sidecar_suffix in ("-wal", "-shm", "-journal"):
        sidecar_path = f"{path}{sidecar_suffix}"
        if fs.exists(sidecar_path):
            fs.rm(sidecar_path)


def _load_suggestions(path: str) -> dict[str, Any]:
    fs, _, _ = fsspec.get_fs_token_paths(path)
    with fs.open(path, "r") as f:
        data = json.load(f)
    if "suggestions" not in data:
        raise ValueError(f"Missing 'suggestions' in {path}")
    return data


def _serialize_parameters(parameters: Mapping[str, object]) -> dict[str, SearchValue]:
    serialized: dict[str, SearchValue] = {}
    for key, value in parameters.items():
        raw_value = value.value if hasattr(value, "value") else value
        if isinstance(raw_value, bool):
            serialized[key] = int(raw_value)
        elif isinstance(raw_value, (int, float, str)):
            serialized[key] = raw_value
        else:
            raise ValueError(f"Unsupported parameter value for '{key}': {raw_value!r}")
    return serialized


def _metric_goal(mode: str) -> Any:
    from vizier.service import pyvizier as vz

    if mode == "min":
        return vz.ObjectiveMetricGoal.MINIMIZE
    if mode == "max":
        return vz.ObjectiveMetricGoal.MAXIMIZE
    raise ValueError(f"Unsupported metric mode: {mode}")


def _add_search_space_param(root: Any, parameter: SearchSpaceParam) -> None:
    if parameter.kind == "float":
        root.add_float_param(parameter.name, float(parameter.min_value), float(parameter.max_value))
        return
    if parameter.kind == "int":
        root.add_int_param(parameter.name, int(parameter.min_value), int(parameter.max_value))
        return
    if parameter.kind == "discrete":
        root.add_discrete_param(parameter.name, list(parameter.values))
        return
    if parameter.kind == "categorical":
        root.add_categorical_param(parameter.name, list(parameter.values))
        return
    raise ValueError(f"Unsupported parameter kind: {parameter.kind}")


def _vizier_trial_tags(
    *,
    architecture_name: str,
    parameters: Mapping[str, SearchValue],
    loop_index: int,
    trial_id: int,
) -> tuple[str, ...]:
    return (
        "grug",
        "hybrid",
        "mamba3",
        "vizier",
        f"arch={architecture_name}",
        f"trial={trial_id}",
        f"loop={loop_index}",
        f"lrx={parameters['lr_multiplier']}",
        f"wdx={parameters['weight_decay_multiplier']}",
        f"beta2={parameters['beta2']}",
        f"mgn={parameters['max_grad_norm']}",
        f"zloss={parameters['z_loss_weight']}",
    )


def run_vizier_suggest(config: VizierSuggestConfig) -> None:
    from vizier.service import clients
    from vizier.service import pyvizier as vz

    local_db_path = _local_vizier_db_path(config.study_id)
    output_db_path = os.path.join(config.output_path, VIZIER_DB_FILENAME)
    if not _sync_vizier_db_from_gcs(output_db_path, local_db_path):
        _sync_vizier_db_from_gcs(config.input_db_path, local_db_path)
    _configure_vizier_local_db(local_db_path)

    study_config = vz.StudyConfig(algorithm=config.algorithm)
    root = study_config.search_space.root
    for parameter in config.search_space:
        _add_search_space_param(root, parameter)
    study_config.metric_information.append(vz.MetricInformation(config.metric_key, goal=_metric_goal(config.mode)))

    study = clients.Study.from_study_config(study_config, owner=config.study_owner, study_id=config.study_id)
    suggestions = study.suggest(count=config.num_suggestions, client_id=config.client_id)
    output = {
        "study_resource_name": study.resource_name,
        "client_id": config.client_id,
        "suggestions": [
            {"trial_id": trial.id, "parameters": _serialize_parameters(trial.parameters)} for trial in suggestions
        ],
    }

    fs, _, _ = fsspec.get_fs_token_paths(config.output_path)
    fs.makedirs(config.output_path, exist_ok=True)
    with fs.open(os.path.join(config.output_path, SUGGESTIONS_FILENAME), "w") as f:
        json.dump(output, f, indent=2)

    _sync_vizier_db_to_gcs(local_db_path, output_db_path)


def run_vizier_train(config: VizierTrainConfig) -> None:
    settings = _vizier_settings()
    suggestions = _load_suggestions(config.suggestions_path)["suggestions"]
    if config.suggestion_index >= len(suggestions):
        raise IndexError(f"Suggestion index {config.suggestion_index} out of range")

    suggestion = suggestions[config.suggestion_index]
    parameters = suggestion["parameters"]
    architecture_name = str(parameters["architecture_name"])
    model_cfg = _candidate_catalog()[architecture_name]
    train_cfg = _build_trial_train_config(model_cfg, parameters)
    trial_id = int(suggestion["trial_id"])

    run_hybrid_trial(
        HybridLaunchConfig(
            data=config.data,
            model_cfg=model_cfg,
            train_cfg=train_cfg,
            output_path=config.output_path,
            run_name=f"{settings.experiment_name}-loop{config.loop_index}-trial{trial_id}",
            wandb_group=settings.experiment_name,
            tags=_vizier_trial_tags(
                architecture_name=architecture_name,
                parameters=parameters,
                loop_index=config.loop_index,
                trial_id=trial_id,
            ),
        )
    )


def run_vizier_update(config: VizierUpdateConfig) -> None:
    from vizier.service import clients
    from vizier.service import pyvizier as vz

    local_db_path = _local_vizier_db_path(config.study_id)
    if not config.input_db_path:
        raise ValueError("input_db_path is required for run_vizier_update")
    if not _sync_vizier_db_from_gcs(config.input_db_path, local_db_path):
        raise FileNotFoundError(f"Could not load Vizier DB from input path: {config.input_db_path}")

    output_db_path = os.path.join(config.output_path, VIZIER_DB_FILENAME)
    _configure_vizier_local_db(local_db_path)

    study = clients.Study.from_resource_name(config.study_resource_name)
    suggestions = _load_suggestions(config.suggestions_path)["suggestions"]
    if len(suggestions) != len(config.run_paths):
        raise ValueError(
            f"Expected {len(suggestions)} run paths but got {len(config.run_paths)} for loop {config.loop_index}"
        )

    results: list[dict[str, Any]] = []
    for run_path, suggestion in zip(config.run_paths, suggestions, strict=True):
        metric_path = os.path.join(run_path, config.metric_file)
        fs, _, _ = fsspec.get_fs_token_paths(metric_path)
        with fs.open(metric_path, "r") as f:
            lines = [line.strip() for line in f if line.strip()]
        if not lines:
            raise RuntimeError(f"No metrics found at {metric_path}")

        data = json.loads(lines[-1])
        value = data["summary"][config.metric_key]
        trial_id = int(suggestion["trial_id"])
        trial = study.get_trial(trial_id)
        trial.complete(vz.Measurement({config.metric_key: float(value)}))

        results.append(
            {
                "trial_id": trial_id,
                "metric": float(value),
                "hparams": suggestion["parameters"],
                "run_path": run_path,
            }
        )

    best = best_run(results, config.mode)
    fs, _, _ = fsspec.get_fs_token_paths(config.output_path)
    fs.makedirs(config.output_path, exist_ok=True)
    with fs.open(os.path.join(config.output_path, UPDATE_FILENAME), "w") as f:
        json.dump(
            {
                "study_resource_name": config.study_resource_name,
                "best_hparams": best["hparams"],
                "best_metric": best["metric"],
                "best_run_path": best["run_path"],
                "all_results": sorted(results, key=lambda r: r["metric"], reverse=(config.mode == "max")),
            },
            f,
            indent=2,
        )
    with fs.open(os.path.join(config.output_path, RESOURCE_FILENAME), "w") as f:
        json.dump({"study_resource_name": config.study_resource_name}, f, indent=2)

    _sync_vizier_db_to_gcs(local_db_path, output_db_path)


def run_vizier_optimal(config: VizierOptimalConfig) -> None:
    from vizier.service import clients

    local_db_path = _local_vizier_db_path(config.study_id)
    if not _sync_vizier_db_from_gcs(config.input_db_path, local_db_path):
        raise FileNotFoundError(f"Could not load Vizier DB from: {config.input_db_path}")
    _configure_vizier_local_db(local_db_path)

    study = clients.Study.from_resource_name(config.study_resource_name)
    optimal_trials = []
    for optimal_trial in study.optimal_trials():
        optimal_trial = optimal_trial.materialize()
        optimal_trials.append(
            {
                "trial_id": optimal_trial.id,
                "parameters": _serialize_parameters(optimal_trial.parameters),
                "final_measurement": str(optimal_trial.final_measurement),
            }
        )

    fs, _, _ = fsspec.get_fs_token_paths(config.output_path)
    fs.makedirs(config.output_path, exist_ok=True)
    with fs.open(os.path.join(config.output_path, OPTIMAL_FILENAME), "w") as f:
        json.dump({"optimal_trials": optimal_trials}, f, indent=2)


def _build_vizier_suggest_step(*, loop_index: int, input_db_path: str | None) -> ExecutorStep:
    settings = _vizier_settings()
    return ExecutorStep(
        name=f"{settings.experiment_name}-suggest-loop{loop_index}",
        fn=remote(run_vizier_suggest, resources=ResourceConfig.with_cpu(), pip_dependency_groups=["vizier"]),
        config=VizierSuggestConfig(
            study_owner=settings.study_owner,
            study_id=settings.study_id,
            input_db_path=input_db_path,
            output_path=this_output_path(),
            num_suggestions=settings.suggestions_per_loop,
            client_id=f"{settings.client_id_prefix}-loop-{loop_index}",
            metric_key=settings.metric_key,
            mode=settings.metric_mode,
            algorithm=settings.vizier_algorithm,
            search_space=_vizier_search_space(),
            loop_index=loop_index,
        ),
    )


def _build_vizier_train_step(*, loop_index: int, suggestion_index: int, suggestions_path: str) -> ExecutorStep:
    settings = _vizier_settings()
    return ExecutorStep(
        name=os.path.join("checkpoints", f"{settings.client_id_prefix}-loop{loop_index}-trial{suggestion_index}"),
        fn=remote(run_vizier_train, resources=ResourceConfig.with_cpu()),
        config=VizierTrainConfig(
            suggestions_path=suggestions_path,
            suggestion_index=suggestion_index,
            data=HYBRID_DATA_WITH_EVAL,
            output_path=this_output_path(),
            loop_index=loop_index,
        ),
    )


def _build_vizier_update_step(
    *,
    loop_index: int,
    study_resource_name: str,
    input_db_path: str | None,
    suggestions_path: str,
    training_steps: list[ExecutorStep],
) -> ExecutorStep:
    settings = _vizier_settings()
    return ExecutorStep(
        name=f"{settings.experiment_name}-update-loop{loop_index}",
        fn=remote(run_vizier_update, resources=ResourceConfig.with_cpu(), pip_dependency_groups=["vizier"]),
        config=VizierUpdateConfig(
            study_id=settings.study_id,
            study_resource_name=study_resource_name,
            input_db_path=input_db_path,
            suggestions_path=suggestions_path,
            run_paths=[step.as_input_name() for step in training_steps],
            metric_file=settings.metric_file,
            metric_key=settings.metric_key,
            mode=settings.metric_mode,
            output_path=this_output_path(),
            loop_index=loop_index,
        ),
    )


def _build_vizier_optimal_step(*, input_db_path: str, study_resource_name: str) -> ExecutorStep:
    settings = _vizier_settings()
    return ExecutorStep(
        name=f"{settings.experiment_name}-optimal",
        fn=remote(run_vizier_optimal, resources=ResourceConfig.with_cpu(), pip_dependency_groups=["vizier"]),
        config=VizierOptimalConfig(
            study_id=settings.study_id,
            study_resource_name=study_resource_name,
            input_db_path=input_db_path,
            output_path=this_output_path(),
        ),
    )


def _grid_main() -> None:
    module_classes = [HybridLaunchConfig]
    _repoint_modules_for_ray(module_classes)

    steps: list[ExecutorStep] = []
    for run_spec in _build_grid_runs():
        steps.append(
            ExecutorStep(
                name=os.path.join("checkpoints", run_spec.run_name),
                fn=remote(run_hybrid_trial, resources=ResourceConfig.with_cpu()),
                config=HybridLaunchConfig(
                    data=HYBRID_DATA_WITH_EVAL,
                    model_cfg=run_spec.model_cfg,
                    train_cfg=run_spec.train_cfg,
                    output_path=this_output_path(),
                    run_name=run_spec.run_name,
                    wandb_group="grug-hybrid-mamba3-grid-3e18",
                    tags=run_spec.tags,
                ),
            )
        )

    executor_main(steps=steps, description="Grug hybrid Mamba-3 vs Splash sweep at 3e18 FLOPs per run")


def _vizier_main() -> None:
    settings = _vizier_settings()
    num_loops = settings.num_loops
    if os.getenv("CI") is not None:
        num_loops = 1

    previous_update_step: ExecutorStep | None = None
    for loop_index in range(num_loops):
        input_db_path = previous_update_step / VIZIER_DB_FILENAME if previous_update_step else None
        suggest_step = _build_vizier_suggest_step(loop_index=loop_index, input_db_path=input_db_path)
        suggestions_path = suggest_step / SUGGESTIONS_FILENAME
        training_steps = [
            _build_vizier_train_step(
                loop_index=loop_index,
                suggestion_index=suggestion_index,
                suggestions_path=suggestions_path,
            )
            for suggestion_index in range(settings.suggestions_per_loop)
        ]
        previous_update_step = _build_vizier_update_step(
            loop_index=loop_index,
            study_resource_name=settings.study_resource_name,
            input_db_path=suggest_step / VIZIER_DB_FILENAME,
            suggestions_path=suggestions_path,
            training_steps=training_steps,
        )

    if previous_update_step is None:
        raise ValueError("Vizier sweep configured with zero loops")

    optimal_step = _build_vizier_optimal_step(
        input_db_path=previous_update_step / VIZIER_DB_FILENAME,
        study_resource_name=settings.study_resource_name,
    )
    executor_main(steps=[optimal_step], description="Vizier search over Grug hybrid Mamba-3 architecture candidates")


def main() -> None:
    module_classes = [
        HybridLaunchConfig,
        SearchSpaceParam,
        VizierSuggestConfig,
        VizierTrainConfig,
        VizierUpdateConfig,
        VizierOptimalConfig,
    ]
    _repoint_modules_for_ray(module_classes)

    launch_mode = os.environ.get("GRUG_HYBRID_LAUNCH_MODE", GRID_LAUNCH_MODE)
    if launch_mode == GRID_LAUNCH_MODE:
        _grid_main()
        return
    if launch_mode == VIZIER_LAUNCH_MODE:
        _vizier_main()
        return
    raise ValueError(f"Unknown GRUG_HYBRID_LAUNCH_MODE={launch_mode!r}")


if __name__ == "__main__":
    main()
