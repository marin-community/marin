# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared lowering helpers for JAX Splash Attention."""

from collections.abc import Sequence
from dataclasses import dataclass

import jax
from jax.experimental.pallas.ops.tpu.splash_attention import SegmentIds as SplashSegmentIds
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel, splash_attention_mask
from jax.sharding import Mesh, PartitionSpec


SPLASH_BLOCK_GRANULARITY = 128
DEFAULT_SPLASH_BLOCK_SIZE = 512
PartitionSpecEntry = str | Sequence[str | None] | None


@dataclass(frozen=True)
class SplashAttentionMaskSpec:
    """Static mask fields consumed by Splash Attention lowering."""

    is_causal: bool = False
    causal_offset: int | None = None
    sliding_window: int | None = None
    bidirectional_window: int | None = None
    has_explicit_mask: bool = False


@dataclass(frozen=True)
class SplashAttentionMaskLowering:
    """JAX Splash masks lowered from a structured mask spec."""

    base_mask: splash_attention_mask.Mask
    kernel_mask: splash_attention_mask.MultiHeadMask


@dataclass(frozen=True)
class SplashSegmentIdsLowering:
    """Segment ID arrays and sharding metadata for Splash Attention."""

    segment_ids: SplashSegmentIds | None
    segment_ids_axes: SplashSegmentIds | None
    segment_batch_axis: SplashSegmentIds | None


def lower_splash_attention_mask(
    *,
    mask: SplashAttentionMaskSpec | None,
    q_seq_len: int,
    kv_seq_len: int,
    num_heads: int,
    q_seq_shards: int,
) -> SplashAttentionMaskLowering:
    """Lower static structured mask fields to JAX Splash mask objects."""
    if mask is None:
        base_mask = splash_attention_mask.FullMask(_shape=(q_seq_len, kv_seq_len))
    else:
        if mask.is_causal:
            if mask.causal_offset is not None:
                raise NotImplementedError(
                    "Causal offsets are not supported for splash attention. Please use a standard causal mask."
                )
            base_mask = splash_attention_mask.CausalMask(
                (q_seq_len, kv_seq_len),
                offset=0,
                shard_count=q_seq_shards,
            )
        else:
            base_mask = splash_attention_mask.FullMask(_shape=(q_seq_len, kv_seq_len))

        if mask.sliding_window is not None:
            local_mask = splash_attention_mask.LocalMask(
                shape=(q_seq_len, kv_seq_len),
                window_size=(mask.sliding_window - 1, None),
                offset=0,
                shard_count=q_seq_shards,
            )
            base_mask = splash_attention_mask.LogicalAnd(base_mask, local_mask)

        if mask.bidirectional_window is not None:
            # Symmetric window: attend to keys within `bidirectional_window` positions on either side.
            window_mask = splash_attention_mask.LocalMask(
                shape=(q_seq_len, kv_seq_len),
                window_size=(mask.bidirectional_window, mask.bidirectional_window),
                offset=0,
                shard_count=q_seq_shards,
            )
            base_mask = splash_attention_mask.LogicalAnd(base_mask, window_mask)

        if mask.has_explicit_mask:
            raise NotImplementedError("Explicit masks are not yet supported for splash attention")

    kernel_mask = splash_attention_mask.MultiHeadMask(masks=[base_mask for _ in range(num_heads)])
    return SplashAttentionMaskLowering(base_mask=base_mask, kernel_mask=kernel_mask)


def lower_splash_segment_ids(
    *,
    q_segment_ids: jax.Array | None = None,
    kv_segment_ids: jax.Array | None = None,
    q_segment_ids_axes: PartitionSpec | None = None,
    kv_segment_ids_axes: PartitionSpec | None = None,
    q_segment_batch_axis: int | None = None,
    kv_segment_batch_axis: int | None = None,
) -> SplashSegmentIdsLowering:
    """Package segment ID arrays, input specs, and vmap axes for Splash."""
    if q_segment_ids is None and kv_segment_ids is None:
        return SplashSegmentIdsLowering(segment_ids=None, segment_ids_axes=None, segment_batch_axis=None)
    if q_segment_ids is None or kv_segment_ids is None:
        raise ValueError("Both q_segment_ids and kv_segment_ids must be provided.")

    return SplashSegmentIdsLowering(
        segment_ids=SplashSegmentIds(q_segment_ids, kv_segment_ids),
        segment_ids_axes=SplashSegmentIds(q_segment_ids_axes, kv_segment_ids_axes),
        segment_batch_axis=SplashSegmentIds(q_segment_batch_axis, kv_segment_batch_axis),
    )


def splash_attention_block_sizes(
    *,
    q_seq_len: int,
    kv_seq_len: int,
    q_seq_shards: int,
    kv_seq_shards: int,
    max_block_size: int = DEFAULT_SPLASH_BLOCK_SIZE,
) -> splash_attention_kernel.BlockSizes:
    """Choose Splash Attention block sizes compatible with per-shard sequence lengths."""
    shard_q_seq_len = max(1, q_seq_len // max(1, q_seq_shards))
    shard_kv_seq_len = max(1, kv_seq_len // max(1, kv_seq_shards))

    block_q = _compatible_splash_block(shard_q_seq_len, max_block_size)
    block_kv = _compatible_splash_block(shard_kv_seq_len, max_block_size)

    return splash_attention_kernel.BlockSizes(
        block_q=block_q,
        block_kv_compute=block_kv,
        block_kv=block_kv,
        block_q_dkv=block_q,
        block_kv_dkv=block_kv,
        block_kv_dkv_compute=block_q,
        block_q_dq=block_q,
        block_kv_dq=block_kv,
    )


def splash_partition_spec_shard_factor(entry: PartitionSpecEntry, mesh: Mesh | None) -> int:
    """Compute product of mesh axis sizes referenced by a PartitionSpec entry."""
    if mesh is None:
        return 1
    if entry is None or entry is PartitionSpec.UNCONSTRAINED:
        return 1
    if isinstance(entry, str):
        return int(mesh.shape.get(entry, 1))

    product = 1
    for axis_name in entry:
        if axis_name is None or axis_name is PartitionSpec.UNCONSTRAINED:
            continue
        product *= int(mesh.shape.get(axis_name, 1))
    return product


def _compatible_splash_block(shard_len: int, max_block: int) -> int:
    """Pick largest block <= max_block that divides shard_len; prefer Splash's granularity."""
    if shard_len <= 0:
        return max_block
    cap = min(max_block, shard_len)
    for step in (SPLASH_BLOCK_GRANULARITY, 1):
        candidate = cap - (cap % step)
        while candidate >= step:
            if shard_len % candidate == 0:
                return candidate
            candidate -= step
    return 1
