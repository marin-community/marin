# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Parameterized JAX reference/export for the connected dense debug region."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class DenseDebugConfig:
    """Static shapes for a small Llama block through the following QKV/RoPE."""

    batch: int = 1
    sequence: int = 4
    hidden: int = 128
    intermediate: int = 256
    query_heads: int = 2
    key_value_heads: int = 1
    head_dimension: int = 64
    epsilon: float = 1e-6

    def __post_init__(self) -> None:
        if self.query_heads * self.head_dimension != self.hidden:
            raise ValueError("query heads times head dimension must equal hidden size")
        if self.query_heads % self.key_value_heads != 0:
            raise ValueError("query head count must be divisible by KV head count")

    @property
    def tokens(self) -> int:
        return self.batch * self.sequence

    @property
    def qkv_width(self) -> int:
        return (self.query_heads + 2 * self.key_value_heads) * self.head_dimension


DENSE_REGION_INPUT_NAMES = (
    "x",
    "qkv_weight",
    "output_weight",
    "mlp_gamma",
    "gate_up_weight",
    "down_weight",
    "next_gamma",
    "next_qkv_weight",
    "rope_sine",
    "rope_cosine",
)


def export_debug_dense_region(config: DenseDebugConfig = DenseDebugConfig()) -> bytes:
    """Export a parameterized dense debug region as portable StableHLO bytecode."""
    bf16 = jnp.bfloat16
    specifications = (
        jax.ShapeDtypeStruct((config.tokens, config.hidden), bf16),
        jax.ShapeDtypeStruct((config.hidden, config.qkv_width), bf16),
        jax.ShapeDtypeStruct((config.hidden, config.hidden), bf16),
        jax.ShapeDtypeStruct((config.hidden,), bf16),
        jax.ShapeDtypeStruct((config.hidden, 2 * config.intermediate), bf16),
        jax.ShapeDtypeStruct((config.intermediate, config.hidden), bf16),
        jax.ShapeDtypeStruct((config.hidden,), bf16),
        jax.ShapeDtypeStruct((config.hidden, config.qkv_width), bf16),
        jax.ShapeDtypeStruct((config.sequence, config.head_dimension // 2), bf16),
        jax.ShapeDtypeStruct((config.sequence, config.head_dimension // 2), bf16),
    )
    exported = jax.export.export(jax.jit(_dense_region(config)))(*specifications)
    return exported.mlir_module_serialized


def dense_region(config: DenseDebugConfig):
    """Return the ordinary JAX function used as the dense numerical baseline."""
    return _dense_region(config)


def _dense_region(config: DenseDebugConfig):
    def region(
        x,
        qkv_weight,
        output_weight,
        mlp_gamma,
        gate_up_weight,
        down_weight,
        next_gamma,
        next_qkv_weight,
        rope_sine,
        rope_cosine,
    ):
        query, key, value = _project_qkv(x, qkv_weight, config)
        query = _rope(query, rope_sine, rope_cosine, config)
        key = _rope(key, rope_sine, rope_cosine, config)
        attention = _causal_gqa(query, key, value, config).reshape(config.tokens, config.hidden)
        projected = _linear(attention, output_weight)
        x1 = projected + x
        mlp_input = _rms_norm(x1, mlp_gamma, config)
        gate_up = _linear(mlp_input, gate_up_weight).reshape(config.tokens, config.intermediate, 2)
        gate = gate_up[..., 0]
        up = gate_up[..., 1]
        activated = (gate / (1.0 + jnp.exp(-gate))) * up
        down = _linear(activated, down_weight)
        x2 = down + x1
        next_input = _rms_norm(x2, next_gamma, config)
        next_query, next_key, next_value = _project_qkv(next_input, next_qkv_weight, config)
        next_query = _rope(next_query, rope_sine, rope_cosine, config)
        next_key = _rope(next_key, rope_sine, rope_cosine, config)
        return x2, next_query, next_key, next_value

    return region


def _linear(value, weight):
    return jnp.matmul(value, weight, preferred_element_type=jnp.float32).astype(jnp.bfloat16)


def _project_qkv(value, weight, config: DenseDebugConfig):
    projected = _linear(value, weight)
    query_width = config.query_heads * config.head_dimension
    key_value_width = config.key_value_heads * config.head_dimension
    query = projected[:, :query_width].reshape(
        config.batch,
        config.sequence,
        config.query_heads,
        config.head_dimension,
    )
    key = projected[:, query_width : query_width + key_value_width].reshape(
        config.batch,
        config.sequence,
        config.key_value_heads,
        config.head_dimension,
    )
    projected_value = projected[:, query_width + key_value_width :].reshape(
        config.batch,
        config.sequence,
        config.key_value_heads,
        config.head_dimension,
    )
    return query, key, projected_value


def _rope(value, sine, cosine, config: DenseDebugConfig):
    pairs = value.reshape(*value.shape[:-1], config.head_dimension // 2, 2)
    even = pairs[..., 0]
    odd = pairs[..., 1]
    sine = sine[None, :, None, :]
    cosine = cosine[None, :, None, :]
    rotated_even = even * cosine - odd * sine
    rotated_odd = even * sine + odd * cosine
    return jnp.stack((rotated_even, rotated_odd), axis=-1).reshape(value.shape)


def _causal_gqa(query, key, value, config: DenseDebugConfig):
    ratio = config.query_heads // config.key_value_heads
    key = jnp.broadcast_to(
        key.reshape(config.batch, config.sequence, config.key_value_heads, 1, config.head_dimension),
        (config.batch, config.sequence, config.key_value_heads, ratio, config.head_dimension),
    ).reshape(config.batch, config.sequence, config.query_heads, config.head_dimension)
    value = jnp.broadcast_to(
        value.reshape(config.batch, config.sequence, config.key_value_heads, 1, config.head_dimension),
        (config.batch, config.sequence, config.key_value_heads, ratio, config.head_dimension),
    ).reshape(config.batch, config.sequence, config.query_heads, config.head_dimension)
    scores = jnp.einsum("bqhd,bkhd->bhqk", query, key, preferred_element_type=jnp.float32)
    scores = scores * (config.head_dimension**-0.5)
    query_position = jnp.arange(config.sequence)[None, None, :, None]
    key_position = jnp.arange(config.sequence)[None, None, None, :]
    mask = key_position <= query_position
    mask = jnp.broadcast_to(mask, scores.shape)
    scores = jax.lax.select(mask, scores, jnp.full(scores.shape, -jnp.inf, dtype=scores.dtype))
    probabilities = jax.nn.softmax(scores, axis=-1).astype(jnp.bfloat16)
    output = jax.lax.dot_general(
        value,
        probabilities,
        dimension_numbers=(((1,), (3,)), ((0, 2), (0, 1))),
        preferred_element_type=jnp.float32,
    )
    return output.transpose(0, 3, 1, 2).astype(jnp.bfloat16)


def _rms_norm(value, gamma, config: DenseDebugConfig):
    value_fp32 = value.astype(jnp.float32)
    inverse_rms = jax.lax.rsqrt(jnp.sum(value_fp32 * value_fp32, axis=-1) / config.hidden + config.epsilon)
    return (value_fp32 * gamma.astype(jnp.float32) * inverse_rms[:, None]).astype(jnp.bfloat16)
