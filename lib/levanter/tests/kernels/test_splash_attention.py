# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask
from jax.sharding import PartitionSpec as P

from levanter.kernels.pallas.splash_attention import (
    PrefixMask,
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

    assert len(lowering.kernel_mask.masks) == 3
    expected = np.tril(np.ones((256, 256), dtype=bool))
    expected &= np.arange(256)[None, :] >= np.arange(256)[:, None] - 127
    np.testing.assert_array_equal(_materialize_splash_mask(lowering.kernel_mask.masks[0]), expected)
    np.testing.assert_array_equal(_materialize_splash_mask(lowering.kernel_mask.masks[1]), expected)


def test_lower_splash_attention_mask_builds_static_prefix_lm_mask():
    lowering = lower_splash_attention_mask(
        mask=SplashAttentionMaskSpec(is_causal=True, prefix_length=64),
        q_seq_len=256,
        kv_seq_len=256,
        num_heads=2,
        q_seq_shards=1,
    )

    expected = np.tril(np.ones((256, 256), dtype=bool))
    expected |= np.arange(256)[None, :] < 64
    np.testing.assert_array_equal(_materialize_splash_mask(lowering.kernel_mask.masks[0]), expected)
    assert isinstance(lowering.kernel_mask.masks[0].right, PrefixMask)


def test_lower_splash_attention_mask_combines_static_prefix_lm_and_sliding_window():
    lowering = lower_splash_attention_mask(
        mask=SplashAttentionMaskSpec(is_causal=True, sliding_window=32, prefix_length=64),
        q_seq_len=256,
        kv_seq_len=256,
        num_heads=1,
        q_seq_shards=1,
    )

    q = np.arange(256)[:, None]
    kv = np.arange(256)[None, :]
    expected = (kv < 64) | ((kv <= q) & (kv >= q - 31))
    np.testing.assert_array_equal(_materialize_splash_mask(lowering.kernel_mask.masks[0]), expected)


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

    with pytest.raises(NotImplementedError):
        lower_splash_attention_mask(
            mask=SplashAttentionMaskSpec(is_causal=True, has_prefix_mask=True),
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


def _materialize_splash_mask(mask):
    if isinstance(mask, splash_attention_mask.LogicalAnd):
        return _materialize_splash_mask(mask.left) & _materialize_splash_mask(mask.right)
    if isinstance(mask, splash_attention_mask.LogicalOr):
        return _materialize_splash_mask(mask.left) | _materialize_splash_mask(mask.right)

    q_indices = jnp.arange(mask.shape[0])[:, None]
    kv_indices = jnp.arange(mask.shape[1])[None, :]
    return np.asarray(mask.mask_function(q_indices, kv_indices))
