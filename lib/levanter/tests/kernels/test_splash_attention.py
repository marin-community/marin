# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from enum import StrEnum

import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel
from jax.sharding import PartitionSpec as P

from levanter.kernels.pallas.splash_attention import (
    BLOCK_MASK_EMPTY,
    BLOCK_MASK_FULL,
    BLOCK_MASK_PARTIAL,
    SplashAttentionMaskSpec,
    SplashDynamicPrefixMask,
    lower_splash_attention_mask,
    lower_splash_segment_ids,
    packed_prefix_lm_mask_infos,
    prefix_lm_dkv_mask_info,
    prefix_lm_forward_mask_info,
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


@pytest.mark.parametrize("prefix_length", [0, 64, 130, 256])
def test_prefix_lm_forward_mask_info_matches_dense_prefix_lm(prefix_length):
    mask_info = prefix_lm_forward_mask_info(
        prefix_length=jnp.asarray(prefix_length, dtype=jnp.int32),
        q_seq_len=256,
        kv_seq_len=256,
        num_heads=2,
        block_q=64,
        block_kv=64,
    )

    actual = _materialize_mask_info(mask_info, head=0, q_seq_len=256, kv_seq_len=256, block_q=64, block_kv=64)
    q = np.arange(256)[:, None]
    kv = np.arange(256)[None, :]
    expected = (kv < prefix_length) | (kv <= q)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("prefix_length", [0, 64, 130, 256])
def test_prefix_lm_dkv_mask_info_matches_dense_prefix_lm(prefix_length):
    mask_info = prefix_lm_dkv_mask_info(
        prefix_length=jnp.asarray(prefix_length, dtype=jnp.int32),
        q_seq_len=256,
        kv_seq_len=256,
        num_heads=2,
        block_q=64,
        block_kv=64,
    )

    actual = _materialize_mask_info(
        mask_info,
        head=0,
        q_seq_len=256,
        kv_seq_len=256,
        block_q=64,
        block_kv=64,
        partial_block_layout=_PartialBlockLayout.COMPACT_TRANSPOSED,
    )
    q = np.arange(256)[:, None]
    kv = np.arange(256)[None, :]
    expected = (kv < prefix_length) | (kv <= q)
    np.testing.assert_array_equal(actual, expected)

    data_next = np.asarray(mask_info.data_next)
    block_mask = np.asarray(mask_info.block_mask)
    q_block_ids = np.arange(4, dtype=data_next.dtype)[:, None]
    np.testing.assert_array_equal(data_next[0], np.where(block_mask[0] > 0, q_block_ids, 0))


def test_packed_prefix_lm_mask_info_matches_dense_packed_prefix_lm():
    seq_len = 128
    block_size = 16
    segment_ids = jnp.concatenate(
        [
            jnp.zeros((64,), dtype=jnp.int32),
            jnp.ones((64,), dtype=jnp.int32),
        ]
    )
    prefix_mask = jnp.zeros((seq_len,), dtype=jnp.bool_)
    prefix_mask = prefix_mask.at[:20].set(True)
    prefix_mask = prefix_mask.at[64:80].set(True)
    block_sizes = splash_attention_kernel.BlockSizes(
        block_q=block_size,
        block_kv_compute=block_size,
        block_kv=block_size,
        block_q_dkv=block_size,
        block_kv_dkv=block_size,
        block_kv_dkv_compute=block_size,
        block_q_dq=block_size,
        block_kv_dq=block_size,
    )

    metadata = packed_prefix_lm_mask_infos(
        prefix_mask=prefix_mask,
        q_segment_ids=segment_ids,
        kv_segment_ids=segment_ids,
        q_seq_len=seq_len,
        kv_seq_len=seq_len,
        block_sizes=block_sizes,
    )

    q = np.arange(seq_len)[:, None]
    kv = np.arange(seq_len)[None, :]
    segments = np.asarray(segment_ids)
    expected = (segments[:, None] == segments[None, :]) & ((kv <= q) | np.asarray(prefix_mask)[None, :])
    actual_fwd = _materialize_mask_info(
        metadata.fwd_mask_info,
        head=0,
        q_seq_len=seq_len,
        kv_seq_len=seq_len,
        block_q=block_size,
        block_kv=block_size,
        partial_block_layout=_PartialBlockLayout.DYNAMIC,
    )
    actual_dkv = _materialize_mask_info(
        metadata.dkv_mask_info,
        head=0,
        q_seq_len=seq_len,
        kv_seq_len=seq_len,
        block_q=block_size,
        block_kv=block_size,
        partial_block_layout=_PartialBlockLayout.DYNAMIC_TRANSPOSED,
    )

    np.testing.assert_array_equal(actual_fwd, expected)
    np.testing.assert_array_equal(actual_dkv, expected)

    block_mask = np.asarray(metadata.fwd_mask_info.block_mask)
    # The helper's contract includes block skipping, not only dense mask parity.
    assert block_mask[0, 0, 4] == BLOCK_MASK_EMPTY  # noqa: ml-slop-test
    assert block_mask[0, 5, 4] == BLOCK_MASK_FULL  # noqa: ml-slop-test
    assert block_mask[0, 0, 1] == BLOCK_MASK_PARTIAL  # noqa: ml-slop-test


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
            mask=SplashAttentionMaskSpec(is_causal=True, dynamic_prefix=SplashDynamicPrefixMask.PREFIX_MASK),
            q_seq_len=128,
            kv_seq_len=128,
            num_heads=1,
            q_seq_shards=1,
        )

    with pytest.raises(NotImplementedError):
        lower_splash_attention_mask(
            mask=SplashAttentionMaskSpec(is_causal=True, dynamic_prefix=SplashDynamicPrefixMask.PREFIX_LENGTHS),
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


class _PartialBlockLayout(StrEnum):
    COMPACT = "compact"
    COMPACT_TRANSPOSED = "compact_transposed"
    DYNAMIC = "dynamic"
    DYNAMIC_TRANSPOSED = "dynamic_transposed"


def _materialize_mask_info(
    mask_info,
    *,
    head: int,
    q_seq_len: int,
    kv_seq_len: int,
    block_q: int,
    block_kv: int,
    partial_block_layout: _PartialBlockLayout = _PartialBlockLayout.COMPACT,
):
    q_blocks = q_seq_len // block_q
    kv_blocks = kv_seq_len // block_kv
    out = np.zeros((q_seq_len, kv_seq_len), dtype=bool)
    block_mask = np.asarray(mask_info.block_mask)
    mask_next = np.asarray(mask_info.mask_next)
    partial_mask_blocks = np.asarray(mask_info.partial_mask_blocks)
    for q_block in range(q_blocks):
        q_slice = slice(q_block * block_q, (q_block + 1) * block_q)
        for kv_block in range(kv_blocks):
            kv_slice = slice(kv_block * block_kv, (kv_block + 1) * block_kv)
            block_kind = int(block_mask[head, q_block, kv_block])
            if block_kind == BLOCK_MASK_FULL:
                out[q_slice, kv_slice] = True
            elif block_kind == BLOCK_MASK_PARTIAL:
                if partial_block_layout in (
                    _PartialBlockLayout.DYNAMIC,
                    _PartialBlockLayout.DYNAMIC_TRANSPOSED,
                ):
                    partial_block = partial_mask_blocks[head, q_block, kv_block]
                else:
                    partial_block = partial_mask_blocks[int(mask_next[head, q_block, kv_block])]
                if partial_block_layout in (
                    _PartialBlockLayout.COMPACT_TRANSPOSED,
                    _PartialBlockLayout.DYNAMIC_TRANSPOSED,
                ):
                    partial_block = partial_block.T
                out[q_slice, kv_slice] = partial_block
    return out
