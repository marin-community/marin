# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Ordinary JAX causal-GQA reverse program used by StableHLO recovery tests."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class StreamingAttentionBackwardDebugConfig:
    """Small static shape for a natural differentiated GQA computation."""

    batch: int = 1
    query_length: int = 4
    key_length: int = 4
    query_heads: int = 4
    key_value_heads: int = 2
    head_dimension: int = 4
    scale: float = 0.5

    def __post_init__(self) -> None:
        dimensions = (
            self.batch,
            self.query_length,
            self.key_length,
            self.query_heads,
            self.key_value_heads,
            self.head_dimension,
        )
        if any(dimension <= 0 for dimension in dimensions):
            raise ValueError("streaming-attention reverse dimensions must be positive")
        if self.query_heads <= self.key_value_heads or self.query_heads % self.key_value_heads:
            raise ValueError("the recovery fixture requires genuine grouped-query attention")


STREAMING_ATTENTION_BACKWARD_INPUT_NAMES = ("query", "key", "value", "output_cotangent")


def causal_gqa_attention(config: StreamingAttentionBackwardDebugConfig):
    """Return ordinary JAX tensor algebra for exact causal GQA."""

    def attention(query, key, value):
        group_size = config.query_heads // config.key_value_heads
        grouped_query = query.reshape(
            config.batch,
            config.query_length,
            config.key_value_heads,
            group_size,
            config.head_dimension,
        )
        scores = jnp.einsum(
            "bqhgd,bkhd->bhgqk",
            grouped_query.astype(jnp.float32),
            key.astype(jnp.float32),
            preferred_element_type=jnp.float32,
        )
        scores *= config.scale
        query_position = jnp.arange(config.query_length)[:, None]
        key_position = jnp.arange(config.key_length)[None, :]
        valid = jnp.broadcast_to(key_position <= query_position, scores.shape)
        masked_scores = jax.lax.select(
            valid,
            scores,
            jnp.full(scores.shape, -jnp.inf, dtype=jnp.float32),
        )
        row_max = jnp.max(masked_scores, axis=-1, keepdims=True)
        exponentials = jnp.exp(masked_scores - row_max)
        probabilities = exponentials / jnp.sum(exponentials, axis=-1, keepdims=True)
        output = jnp.einsum(
            "bhgqk,bkhv->bqhgv",
            probabilities,
            value.astype(jnp.float32),
            preferred_element_type=jnp.float32,
        )
        return output.reshape(
            config.batch,
            config.query_length,
            config.query_heads,
            config.head_dimension,
        ).astype(jnp.bfloat16)

    return attention


def causal_gqa_attention_vjp(config: StreamingAttentionBackwardDebugConfig):
    """Return a reverse function whose derivative is owned by JAX."""

    attention = causal_gqa_attention(config)

    def reverse(query, key, value, output_cotangent):
        _, pullback = jax.vjp(attention, query, key, value)
        return pullback(output_cotangent)

    return reverse


def causal_gqa_attention_training(config: StreamingAttentionBackwardDebugConfig):
    """Return the natural forward plus JAX-owned reverse training boundary."""
    attention = causal_gqa_attention(config)

    def training(query, key, value, output_cotangent):
        output, pullback = jax.vjp(attention, query, key, value)
        query_cotangent, key_cotangent, value_cotangent = pullback(output_cotangent)
        return output, query_cotangent, key_cotangent, value_cotangent

    return training


def export_debug_streaming_attention_backward(
    config: StreamingAttentionBackwardDebugConfig = StreamingAttentionBackwardDebugConfig(),
) -> bytes:
    """Export the JAX-owned reverse as portable StableHLO."""
    bf16 = jnp.bfloat16
    specifications = (
        jax.ShapeDtypeStruct(
            (config.batch, config.query_length, config.query_heads, config.head_dimension),
            bf16,
        ),
        jax.ShapeDtypeStruct(
            (config.batch, config.key_length, config.key_value_heads, config.head_dimension),
            bf16,
        ),
        jax.ShapeDtypeStruct(
            (config.batch, config.key_length, config.key_value_heads, config.head_dimension),
            bf16,
        ),
        jax.ShapeDtypeStruct(
            (config.batch, config.query_length, config.query_heads, config.head_dimension),
            bf16,
        ),
    )
    exported = jax.export.export(jax.jit(causal_gqa_attention_vjp(config)))(*specifications)
    return exported.mlir_module_serialized


def export_debug_streaming_attention_training(
    config: StreamingAttentionBackwardDebugConfig = StreamingAttentionBackwardDebugConfig(),
) -> bytes:
    """Export a natural forward-plus-reverse JAX training boundary."""
    bf16 = jnp.bfloat16
    specifications = (
        jax.ShapeDtypeStruct(
            (config.batch, config.query_length, config.query_heads, config.head_dimension),
            bf16,
        ),
        jax.ShapeDtypeStruct(
            (config.batch, config.key_length, config.key_value_heads, config.head_dimension),
            bf16,
        ),
        jax.ShapeDtypeStruct(
            (config.batch, config.key_length, config.key_value_heads, config.head_dimension),
            bf16,
        ),
        jax.ShapeDtypeStruct(
            (config.batch, config.query_length, config.query_heads, config.head_dimension),
            bf16,
        ),
    )
    exported = jax.export.export(jax.jit(causal_gqa_attention_training(config)))(*specifications)
    return exported.mlir_module_serialized
