# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Ordinary JAX routed block-attention frontend and StableHLO export."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

ROUTED_ATTENTION_INPUT_NAMES = (
    "query",
    "key",
    "value",
    "query_metadata",
    "key_value_metadata",
)


@dataclass(frozen=True)
class RoutedAttentionDebugConfig:
    """Static dimensions for the natural routed-attention fixture."""

    sequence: int = 16
    block_size: int = 4
    selected_blocks: int = 2
    query_heads: int = 4
    key_value_heads: int = 2
    head_dimension: int = 8
    router_dimension: int = 4

    def __post_init__(self) -> None:
        if (
            min(
                self.sequence,
                self.block_size,
                self.selected_blocks,
                self.query_heads,
                self.key_value_heads,
                self.head_dimension,
                self.router_dimension,
            )
            <= 0
        ):
            raise ValueError("routed-attention dimensions must be positive")
        if self.sequence % self.block_size:
            raise ValueError("sequence length must be divisible by block size")
        if self.query_heads % self.key_value_heads:
            raise ValueError("query heads must be divisible by key/value heads")
        if self.selected_blocks > self.block_count:
            raise ValueError("selected block count cannot exceed the number of blocks")

    @property
    def block_count(self) -> int:
        """Number of query and key/value blocks."""
        return self.sequence // self.block_size

    @property
    def scale(self) -> float:
        """Source-level score scale."""
        return self.head_dimension**-0.5


def export_debug_routed_attention(config: RoutedAttentionDebugConfig = RoutedAttentionDebugConfig()) -> bytes:
    """Export natural routing, selected attention, and merge as StableHLO."""
    bf16 = jnp.bfloat16
    specifications = (
        jax.ShapeDtypeStruct((1, config.sequence, config.query_heads, config.head_dimension), bf16),
        jax.ShapeDtypeStruct((1, config.sequence, config.key_value_heads, config.head_dimension), bf16),
        jax.ShapeDtypeStruct((1, config.sequence, config.key_value_heads, config.head_dimension), bf16),
        jax.ShapeDtypeStruct((config.block_count, config.router_dimension), jnp.float32),
        jax.ShapeDtypeStruct((config.block_count, config.router_dimension), jnp.float32),
    )
    exported = jax.export.export(jax.jit(routed_attention_region(config)))(*specifications)
    return exported.mlir_module_serialized


def routed_attention_region(config: RoutedAttentionDebugConfig):
    """Return ordinary JAX math for routed exact attention."""

    def region(query, key, value, query_metadata, key_value_metadata):
        routing_scores = jnp.einsum(
            "qr,kr->qk",
            query_metadata,
            key_value_metadata,
            preferred_element_type=jnp.float32,
        )
        query_block = jnp.arange(config.block_count)[:, None]
        key_block = jnp.arange(config.block_count)[None, :]
        routing_valid = key_block <= query_block
        routing_scores = jax.lax.select(
            routing_valid,
            routing_scores,
            jnp.full(routing_scores.shape, -jnp.inf, dtype=routing_scores.dtype),
        )
        _, selected = jax.lax.top_k(routing_scores, config.selected_blocks)

        blocked_query = query[0].reshape(
            config.block_count,
            config.block_size,
            config.query_heads,
            config.head_dimension,
        )
        blocked_key = key[0].reshape(
            config.block_count,
            config.block_size,
            config.key_value_heads,
            config.head_dimension,
        )
        blocked_value = value[0].reshape(
            config.block_count,
            config.block_size,
            config.key_value_heads,
            config.head_dimension,
        )
        selected_key = blocked_key[selected]
        selected_value = blocked_value[selected]
        head_group = config.query_heads // config.key_value_heads
        selected_key = jnp.broadcast_to(
            selected_key[..., None, :],
            (
                config.block_count,
                config.selected_blocks,
                config.block_size,
                config.key_value_heads,
                head_group,
                config.head_dimension,
            ),
        ).reshape(
            config.block_count,
            config.selected_blocks,
            config.block_size,
            config.query_heads,
            config.head_dimension,
        )
        selected_value = jnp.broadcast_to(
            selected_value[..., None, :],
            (
                config.block_count,
                config.selected_blocks,
                config.block_size,
                config.key_value_heads,
                head_group,
                config.head_dimension,
            ),
        ).reshape(
            config.block_count,
            config.selected_blocks,
            config.block_size,
            config.query_heads,
            config.head_dimension,
        )

        scores = jnp.einsum(
            "qthd,qskhd->qhtsk",
            blocked_query,
            selected_key,
            preferred_element_type=jnp.float32,
        )
        scores *= config.scale
        query_position = jnp.arange(config.sequence).reshape(config.block_count, config.block_size)
        key_position = selected[..., None] * config.block_size + jnp.arange(config.block_size)
        causal = key_position[:, None, :, :] <= query_position[:, :, None, None]
        causal = jnp.broadcast_to(causal[:, None, :, :, :], scores.shape)
        scores = jax.lax.select(causal, scores, jnp.full(scores.shape, -jnp.inf, dtype=scores.dtype))
        flattened_scores = scores.reshape(
            config.block_count,
            config.query_heads,
            config.block_size,
            config.selected_blocks * config.block_size,
        )
        probabilities = jax.nn.softmax(flattened_scores, axis=-1).astype(jnp.bfloat16)
        flattened_value = selected_value.reshape(
            config.block_count,
            config.selected_blocks * config.block_size,
            config.query_heads,
            config.head_dimension,
        )
        output = jnp.einsum(
            "qhtk,qkhd->qthd",
            probabilities,
            flattened_value,
            preferred_element_type=jnp.float32,
        )
        return output.reshape(1, config.sequence, config.query_heads, config.head_dimension).astype(jnp.bfloat16)

    return region
