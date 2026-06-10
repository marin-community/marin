# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import jax.numpy as jnp
import pytest
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask
from jax.sharding import PartitionSpec as P

from levanter.kernels.pallas.splash_attention import (
    SplashAttentionMaskSpec,
    lower_splash_attention_mask,
    lower_splash_segment_ids,
    splash_attention_block_sizes,
)


def test_lower_splash_attention_mask_builds_multi_head_mask():
    lowering = lower_splash_attention_mask(
        mask=SplashAttentionMaskSpec(is_causal=True, sliding_window=128),
        q_seq_len=256,
        kv_seq_len=256,
        num_heads=3,
        q_seq_shards=1,
    )

    assert lowering.base_mask.shape == (256, 256)
    assert isinstance(lowering.base_mask, splash_attention_mask.LogicalAnd)
    assert len(lowering.kernel_mask.masks) == 3
    assert all(mask.shape == (256, 256) for mask in lowering.kernel_mask.masks)
    assert all(isinstance(mask, splash_attention_mask.LogicalAnd) for mask in lowering.kernel_mask.masks)


def test_lower_splash_attention_mask_rejects_unsupported_structured_fields():
    with pytest.raises(NotImplementedError):
        lower_splash_attention_mask(
            mask=SplashAttentionMaskSpec(is_causal=True, causal_offset=1),
            q_seq_len=128,
            kv_seq_len=128,
            num_heads=1,
            q_seq_shards=1,
        )

    with pytest.raises(NotImplementedError):
        lower_splash_attention_mask(
            mask=SplashAttentionMaskSpec(has_explicit_mask=True),
            q_seq_len=128,
            kv_seq_len=128,
            num_heads=1,
            q_seq_shards=1,
        )


def test_lower_splash_segment_ids_packages_arrays_specs_and_batch_axes():
    q_segment_ids = jnp.array([[0, 0, 1, 1]], dtype=jnp.int32)
    kv_segment_ids = jnp.array([[0, 0, 1, 1]], dtype=jnp.int32)

    lowering = lower_splash_segment_ids(
        q_segment_ids=q_segment_ids,
        kv_segment_ids=kv_segment_ids,
        q_segment_ids_axes=P("data", None),
        kv_segment_ids_axes=P("data", None),
        q_segment_batch_axis=0,
        kv_segment_batch_axis=0,
    )

    assert lowering.segment_ids is not None
    assert lowering.segment_ids.q is q_segment_ids
    assert lowering.segment_ids.kv is kv_segment_ids
    assert lowering.segment_ids_axes is not None
    assert lowering.segment_ids_axes.q == P("data", None)
    assert lowering.segment_batch_axis is not None
    assert lowering.segment_batch_axis.q == 0


def test_splash_attention_block_sizes_match_existing_defaults():
    block_sizes = splash_attention_block_sizes(
        q_seq_len=1024,
        kv_seq_len=1024,
        q_seq_shards=2,
        kv_seq_shards=1,
    )

    assert block_sizes.block_q == 512
    assert block_sizes.block_kv == 512
    assert block_sizes.block_kv_compute == 512
