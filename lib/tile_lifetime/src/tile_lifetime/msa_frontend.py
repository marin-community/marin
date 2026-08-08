# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Ordinary JAX projected-relation sparse attention and StableHLO export."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

MSA_INPUT_NAMES = (
    "query_hidden",
    "key_value_hidden",
    "query_weight",
    "key_weight",
    "value_weight",
    "left_index_weight",
    "right_index_weight",
)


@dataclass(frozen=True)
class MSADebugConfig:
    """Small static shape for the natural projected-relation fixture."""

    query_length: int = 8
    key_value_length: int = 16
    hidden_dimension: int = 12
    query_heads: int = 4
    key_value_heads: int = 2
    head_dimension: int = 8
    index_dimension: int = 4
    block_size: int = 4
    selected_blocks: int = 2

    def __post_init__(self) -> None:
        dimensions = (
            self.query_length,
            self.key_value_length,
            self.hidden_dimension,
            self.query_heads,
            self.key_value_heads,
            self.head_dimension,
            self.index_dimension,
            self.block_size,
            self.selected_blocks,
        )
        if min(dimensions) <= 0:
            raise ValueError("projected sparse-attention dimensions must be positive")
        if self.key_value_length % self.block_size:
            raise ValueError("key/value length must be divisible by block size")
        if self.query_length > self.key_value_length:
            raise ValueError("bottom-right causal prefill requires query length no larger than key/value length")
        if self.query_heads % self.key_value_heads:
            raise ValueError("query heads must be divisible by key/value heads")
        if self.selected_blocks > self.block_count:
            raise ValueError("selected block count cannot exceed block count")

    @property
    def block_count(self) -> int:
        """Number of key/value blocks."""
        return self.key_value_length // self.block_size

    @property
    def query_position_offset(self) -> int:
        """Absolute position of the first query in bottom-right causal alignment."""
        return self.key_value_length - self.query_length

    @property
    def attention_scale(self) -> float:
        """Source-level main-attention score scale."""
        return self.head_dimension**-0.5

    @property
    def index_scale(self) -> float:
        """Source-level index score scale."""
        return self.index_dimension**-0.5


def export_debug_msa(config: MSADebugConfig = MSADebugConfig()) -> bytes:
    """Export projections, relation selection, and selected exact attention."""
    bf16 = jnp.bfloat16
    specifications = (
        jax.ShapeDtypeStruct((config.query_length, config.hidden_dimension), bf16),
        jax.ShapeDtypeStruct((config.key_value_length, config.hidden_dimension), bf16),
        jax.ShapeDtypeStruct(
            (config.hidden_dimension, config.query_heads * config.head_dimension),
            bf16,
        ),
        jax.ShapeDtypeStruct(
            (config.hidden_dimension, config.key_value_heads * config.head_dimension),
            bf16,
        ),
        jax.ShapeDtypeStruct(
            (config.hidden_dimension, config.key_value_heads * config.head_dimension),
            bf16,
        ),
        jax.ShapeDtypeStruct(
            (config.hidden_dimension, config.key_value_heads * config.index_dimension),
            jnp.float32,
        ),
        jax.ShapeDtypeStruct((config.hidden_dimension, config.index_dimension), jnp.float32),
    )
    exported = jax.export.export(jax.jit(msa_region(config)))(*specifications)
    return exported.mlir_module_serialized


def msa_region(config: MSADebugConfig):
    """Return natural projected-routing and exact selected-attention math."""

    def region(
        query_hidden,
        key_value_hidden,
        query_weight,
        key_weight,
        value_weight,
        left_index_weight,
        right_index_weight,
    ):
        query = _project_main(query_hidden, query_weight).reshape(
            config.query_length,
            config.query_heads,
            config.head_dimension,
        )
        key = _project_main(key_value_hidden, key_weight).reshape(
            config.key_value_length,
            config.key_value_heads,
            config.head_dimension,
        )
        value = _project_main(key_value_hidden, value_weight).reshape(
            config.key_value_length,
            config.key_value_heads,
            config.head_dimension,
        )

        detached_query = jax.lax.stop_gradient(query_hidden).astype(jnp.float32)
        detached_key_value = jax.lax.stop_gradient(key_value_hidden).astype(jnp.float32)
        left_index = jnp.matmul(
            detached_query,
            left_index_weight,
            preferred_element_type=jnp.float32,
        ).astype(jnp.bfloat16)
        left_index = left_index.reshape(config.query_length, config.key_value_heads, config.index_dimension)
        right_index = jnp.matmul(
            detached_key_value,
            right_index_weight,
            preferred_element_type=jnp.float32,
        ).astype(jnp.bfloat16)
        token_scores = jnp.einsum(
            "qhd,kd->qhk",
            left_index,
            right_index,
            preferred_element_type=jnp.float32,
        )
        token_scores *= config.index_scale
        query_position = (jnp.arange(config.query_length) + config.query_position_offset)[:, None]
        key_position = jnp.arange(config.key_value_length)[None, :]
        token_scores = jax.lax.select(
            jnp.broadcast_to((key_position <= query_position)[:, None, :], token_scores.shape),
            token_scores,
            jnp.full(token_scores.shape, -jnp.inf, dtype=token_scores.dtype),
        )
        block_scores = jnp.max(
            token_scores.reshape(
                config.query_length,
                config.key_value_heads,
                config.block_count,
                config.block_size,
            ),
            axis=-1,
        )
        all_local_blocks = jnp.repeat(jnp.arange(config.block_count), config.block_size)
        local_block = jax.lax.slice_in_dim(
            all_local_blocks,
            config.query_position_offset,
            config.query_position_offset + config.query_length,
        )
        block_identity = jnp.arange(config.block_count)
        forced_local = block_identity[None, None, :] == local_block[:, None, None]
        ranked_scores = jax.lax.select(
            jnp.broadcast_to(forced_local, block_scores.shape),
            jnp.full(block_scores.shape, jnp.inf, dtype=block_scores.dtype),
            block_scores,
        )
        _, selected = jax.lax.top_k(ranked_scores, config.selected_blocks)
        selected_valid = selected <= local_block[:, None, None]
        safe_selected = jax.lax.select(selected_valid, selected, jnp.zeros_like(selected))

        blocked_key = key.reshape(
            config.block_count,
            config.block_size,
            config.key_value_heads,
            config.head_dimension,
        ).transpose(2, 0, 1, 3)
        blocked_value = value.reshape(
            config.block_count,
            config.block_size,
            config.key_value_heads,
            config.head_dimension,
        ).transpose(2, 0, 1, 3)
        group_indices = safe_selected.transpose(1, 0, 2)
        selected_key = jax.vmap(lambda blocks, indices: blocks[indices])(blocked_key, group_indices).transpose(
            1, 0, 2, 3, 4
        )
        selected_value = jax.vmap(lambda blocks, indices: blocks[indices])(
            blocked_value,
            group_indices,
        ).transpose(1, 0, 2, 3, 4)

        heads_per_group = config.query_heads // config.key_value_heads
        grouped_query = query.reshape(
            config.query_length,
            config.key_value_heads,
            heads_per_group,
            config.head_dimension,
        )
        scores = jnp.einsum(
            "qhgd,qhkbd->qhgkb",
            grouped_query,
            selected_key,
            preferred_element_type=jnp.float32,
        )
        scores *= config.attention_scale
        selected_key_position = (
            safe_selected[..., None] * config.block_size + jnp.arange(config.block_size)[None, None, None, :]
        )
        query_position = jnp.arange(config.query_length) + config.query_position_offset
        score_valid = selected_key_position <= query_position[:, None, None, None]
        scores = jax.lax.select(
            jnp.broadcast_to(score_valid[:, :, None, :, :], scores.shape),
            scores,
            jnp.full(scores.shape, -jnp.inf, dtype=scores.dtype),
        )
        scores = jax.lax.select(
            jnp.broadcast_to(selected_valid[:, :, None, :, None], scores.shape),
            scores,
            jnp.full(scores.shape, -jnp.inf, dtype=scores.dtype),
        )
        flattened_scores = scores.reshape(
            config.query_length,
            config.key_value_heads,
            heads_per_group,
            config.selected_blocks * config.block_size,
        )
        probability = jax.nn.softmax(flattened_scores, axis=-1).astype(jnp.bfloat16)
        flattened_value = selected_value.reshape(
            config.query_length,
            config.key_value_heads,
            config.selected_blocks * config.block_size,
            config.head_dimension,
        )
        output = jnp.einsum(
            "qhgk,qhkd->qhgd",
            probability,
            flattened_value,
            preferred_element_type=jnp.float32,
        )
        return output.reshape(config.query_length, config.query_heads, config.head_dimension).astype(jnp.bfloat16)

    return region


def _project_main(hidden, weight):
    return jnp.matmul(hidden, weight, preferred_element_type=jnp.float32).astype(jnp.bfloat16)
