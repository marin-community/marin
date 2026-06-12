# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared lowering helpers for JAX Splash Attention."""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

import jax
import numpy as np
from jax.experimental.pallas.ops.tpu.splash_attention import SegmentIds as SplashSegmentIds
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel, splash_attention_mask
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask_info
from jax import numpy as jnp
from jax.sharding import Mesh, PartitionSpec


SPLASH_BLOCK_GRANULARITY = 128
DEFAULT_SPLASH_BLOCK_SIZE = 512
BLOCK_MASK_EMPTY = 0
BLOCK_MASK_PARTIAL = 1
BLOCK_MASK_FULL = 2
PARTIAL_MASK_SLOTS_PER_Q_BLOCK = 2
CAUSAL_PARTIAL_MASK_SLOT_OFFSET = 0
PREFIX_PARTIAL_MASK_SLOT_OFFSET = 1
PartitionSpecEntry = str | Sequence[str | None] | None


class SplashDynamicPrefixMask(StrEnum):
    NONE = "none"
    PREFIX_LENGTHS = "prefix_lengths"
    PREFIX_MASK = "prefix_mask"


@dataclass(frozen=True)
class SplashAttentionMaskSpec:
    """Static mask fields consumed by Splash Attention lowering."""

    is_causal: bool = False
    causal_offset: int | None = None
    sliding_window: int | None = None
    prefix_length: int | None = None
    dynamic_prefix: SplashDynamicPrefixMask = SplashDynamicPrefixMask.NONE
    has_explicit_mask: bool = False


@dataclass(frozen=True)
class SplashAttentionMaskLowering:
    """JAX Splash masks lowered from a structured mask spec."""

    base_mask: splash_attention_mask.Mask
    kernel_mask: splash_attention_mask.MultiHeadMask


def splash_attention_mask_spec_from_attention_mask(mask) -> SplashAttentionMaskSpec:
    """Convert an AttentionMask-like object to the static fields used by Splash lowering."""
    dynamic_prefix = SplashDynamicPrefixMask.NONE
    if mask.prefix_lengths is not None and mask.prefix_mask is not None:
        raise ValueError("Splash attention mask spec cannot combine prefix_lengths and prefix_mask.")
    if mask.prefix_lengths is not None:
        dynamic_prefix = SplashDynamicPrefixMask.PREFIX_LENGTHS
    elif mask.prefix_mask is not None:
        dynamic_prefix = SplashDynamicPrefixMask.PREFIX_MASK

    return SplashAttentionMaskSpec(
        is_causal=mask.is_causal,
        causal_offset=mask.causal_offset,
        sliding_window=mask.sliding_window,
        prefix_length=mask.prefix_length,
        dynamic_prefix=dynamic_prefix,
        has_explicit_mask=mask.explicit_mask is not None,
    )


@dataclass(frozen=True)
class SplashSegmentIdsLowering:
    """Segment ID arrays and sharding metadata for Splash Attention."""

    segment_ids: SplashSegmentIds | None
    segment_ids_axes: SplashSegmentIds | None
    segment_batch_axis: SplashSegmentIds | None


@dataclass(frozen=True)
class SplashPrefixLmMetadata:
    """Compact block metadata for a dynamic prefix-LM mask."""

    fwd_mask_info: splash_attention_mask_info.MaskInfo
    dq_mask_info: splash_attention_mask_info.MaskInfo
    dkv_mask_info: splash_attention_mask_info.MaskInfo


@dataclass(frozen=True)
class _PrefixLmMaskInfoComponents:
    data_next: jax.Array
    mask_next: jax.Array
    block_mask: jax.Array
    partial_mask_blocks: jax.Array


class _PrefixLmDataNextAxis(StrEnum):
    Q = "q"
    KV = "kv"


def prefix_lm_mask_infos(
    *,
    prefix_length: jax.Array,
    q_seq_len: int,
    kv_seq_len: int,
    num_heads: int,
    block_sizes: splash_attention_kernel.BlockSizes,
) -> SplashPrefixLmMetadata:
    """Build compact Splash mask metadata for dynamic prefix-LM attention."""
    if not block_sizes.has_backward_blocks:
        raise ValueError("prefix_lm_mask_infos requires backward block sizes.")

    fwd_mask_info = prefix_lm_forward_mask_info(
        prefix_length=prefix_length,
        q_seq_len=q_seq_len,
        kv_seq_len=kv_seq_len,
        num_heads=num_heads,
        block_q=block_sizes.block_q,
        block_kv=block_sizes.block_kv,
    )
    dq_mask_info = prefix_lm_forward_mask_info(
        prefix_length=prefix_length,
        q_seq_len=q_seq_len,
        kv_seq_len=kv_seq_len,
        num_heads=num_heads,
        block_q=block_sizes.block_q_dq,
        block_kv=block_sizes.block_kv_dq,
    )
    dkv_mask_info = prefix_lm_dkv_mask_info(
        prefix_length=prefix_length,
        q_seq_len=q_seq_len,
        kv_seq_len=kv_seq_len,
        num_heads=num_heads,
        block_q=block_sizes.block_q_dkv,
        block_kv=block_sizes.block_kv_dkv,
    )
    return SplashPrefixLmMetadata(
        fwd_mask_info=fwd_mask_info,
        dq_mask_info=dq_mask_info,
        dkv_mask_info=dkv_mask_info,
    )


def packed_prefix_lm_mask_infos(
    *,
    prefix_mask: jax.Array,
    q_segment_ids: jax.Array,
    kv_segment_ids: jax.Array,
    q_seq_len: int,
    kv_seq_len: int,
    block_sizes: splash_attention_kernel.BlockSizes,
    head_shards: int = 1,
    q_seq_shards: int = 1,
) -> SplashPrefixLmMetadata:
    """Build Splash metadata for packed prefix-LM masks.

    The represented mask is ``same_segment(q, kv) & (kv <= q | prefix_mask[kv])``.
    This supports packed docs by letting callers mark prefix tokens per key
    position while segment IDs keep those prefix tokens local to their document.
    """
    if not block_sizes.has_backward_blocks:
        raise ValueError("packed_prefix_lm_mask_infos requires backward block sizes.")

    # The packed prefix mask is identical for every attention head. Splash treats
    # a mask-info head dimension of size 1 as shared across all heads, avoiding
    # an H-fold blowup in dynamic partial mask blocks at long sequence lengths.
    dense_mask = packed_prefix_lm_dense_mask(
        prefix_mask=prefix_mask,
        q_segment_ids=q_segment_ids,
        kv_segment_ids=kv_segment_ids,
        q_seq_len=q_seq_len,
        kv_seq_len=kv_seq_len,
        num_heads=1,
    )
    fwd_mask_info = _dynamic_forward_dense_mask_info(
        dense_mask,
        block_q=block_sizes.block_q,
        block_kv=block_sizes.block_kv,
        head_shards=head_shards,
        q_seq_shards=q_seq_shards,
    )
    dq_mask_info = _dynamic_forward_dense_mask_info(
        dense_mask,
        block_q=block_sizes.block_q_dq,
        block_kv=block_sizes.block_kv_dq,
        head_shards=head_shards,
        q_seq_shards=q_seq_shards,
    )
    dkv_mask_info = _dynamic_dkv_dense_mask_info(
        dense_mask,
        block_q=block_sizes.block_q_dkv,
        block_kv=block_sizes.block_kv_dkv,
        head_shards=head_shards,
        q_seq_shards=q_seq_shards,
    )
    return SplashPrefixLmMetadata(
        fwd_mask_info=fwd_mask_info,
        dq_mask_info=dq_mask_info,
        dkv_mask_info=dkv_mask_info,
    )


def packed_prefix_lm_dense_mask(
    *,
    prefix_mask: jax.Array,
    q_segment_ids: jax.Array,
    kv_segment_ids: jax.Array,
    q_seq_len: int,
    kv_seq_len: int,
    num_heads: int,
) -> jax.Array:
    """Materialize the per-example dense mask used by packed prefix-LM lowering."""
    prefix_mask = jnp.asarray(prefix_mask, dtype=jnp.bool_)
    q_segment_ids = jnp.asarray(q_segment_ids)
    kv_segment_ids = jnp.asarray(kv_segment_ids)

    if prefix_mask.ndim != 1:
        raise ValueError(f"prefix_mask must be rank 1 after batching, got shape {prefix_mask.shape}.")
    if q_segment_ids.ndim != 1 or kv_segment_ids.ndim != 1:
        raise ValueError(
            "q_segment_ids and kv_segment_ids must be rank 1 after batching, "
            f"got {q_segment_ids.shape} and {kv_segment_ids.shape}."
        )
    if prefix_mask.shape[0] != kv_seq_len:
        raise ValueError(f"prefix_mask length {prefix_mask.shape[0]} must match kv_seq_len={kv_seq_len}.")
    if q_segment_ids.shape[0] != q_seq_len:
        raise ValueError(f"q_segment_ids length {q_segment_ids.shape[0]} must match q_seq_len={q_seq_len}.")
    if kv_segment_ids.shape[0] != kv_seq_len:
        raise ValueError(f"kv_segment_ids length {kv_segment_ids.shape[0]} must match kv_seq_len={kv_seq_len}.")

    q_positions = jnp.arange(q_seq_len, dtype=jnp.int32)[:, None]
    kv_positions = jnp.arange(kv_seq_len, dtype=jnp.int32)[None, :]
    same_segment = q_segment_ids[:, None] == kv_segment_ids[None, :]
    prefix_or_causal = (kv_positions <= q_positions) | prefix_mask[None, :]
    mask = same_segment & prefix_or_causal
    return jnp.broadcast_to(mask[None, :, :], (num_heads, q_seq_len, kv_seq_len))


def _dynamic_forward_dense_mask_info(
    dense_mask: jax.Array,
    *,
    block_q: int,
    block_kv: int,
    head_shards: int,
    q_seq_shards: int,
) -> splash_attention_mask_info.MaskInfo:
    mask_info, _ = splash_attention_mask_info._process_dynamic_mask(
        dense_mask,
        (block_q, block_kv),
        is_dkv=False,
        head_shards=head_shards,
        q_seq_shards=q_seq_shards,
    )
    return mask_info


def _dynamic_dkv_dense_mask_info(
    dense_mask: jax.Array,
    *,
    block_q: int,
    block_kv: int,
    head_shards: int,
    q_seq_shards: int,
) -> splash_attention_mask_info.MaskInfo:
    mask_info, _ = splash_attention_mask_info._process_dynamic_mask(
        dense_mask,
        (block_q, block_kv),
        is_dkv=True,
        head_shards=head_shards,
        q_seq_shards=q_seq_shards,
    )
    return mask_info


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
        if mask.dynamic_prefix != SplashDynamicPrefixMask.NONE:
            raise NotImplementedError("Dynamic prefix-LM masks are not yet supported for splash attention")
        if mask.prefix_length is not None and not mask.is_causal:
            raise NotImplementedError("Splash prefix-LM masks must also be causal.")

        if mask.is_causal:
            if mask.causal_offset is not None:
                raise NotImplementedError(
                    "Causal offsets are not supported for splash attention. Please use a standard causal mask."
                )
            causal_mask = splash_attention_mask.CausalMask(
                (q_seq_len, kv_seq_len),
                offset=0,
                shard_count=q_seq_shards,
            )
        else:
            causal_mask = splash_attention_mask.FullMask(_shape=(q_seq_len, kv_seq_len))

        if mask.sliding_window is not None:
            local_mask = splash_attention_mask.LocalMask(
                shape=(q_seq_len, kv_seq_len),
                window_size=(mask.sliding_window - 1, None),
                offset=0,
                shard_count=q_seq_shards,
            )
            causal_mask = splash_attention_mask.LogicalAnd(causal_mask, local_mask)

        if mask.prefix_length is not None:
            prefix_mask = PrefixMask(
                shape=(q_seq_len, kv_seq_len),
                prefix_length=mask.prefix_length,
                shard_count=q_seq_shards,
            )
            base_mask = splash_attention_mask.LogicalOr(causal_mask, prefix_mask)
        else:
            base_mask = causal_mask

        if mask.has_explicit_mask:
            raise NotImplementedError("Explicit masks are not yet supported for splash attention")

    kernel_mask = splash_attention_mask.MultiHeadMask(masks=[base_mask for _ in range(num_heads)])
    return SplashAttentionMaskLowering(base_mask=base_mask, kernel_mask=kernel_mask)


def prefix_lm_forward_mask_info(
    *,
    prefix_length: jax.Array,
    q_seq_len: int,
    kv_seq_len: int,
    num_heads: int,
    block_q: int,
    block_kv: int,
) -> splash_attention_mask_info.MaskInfo:
    """Build forward Splash metadata for ``kv < prefix_length OR kv <= q``."""
    components = _prefix_lm_mask_info_components(
        prefix_length=prefix_length,
        q_seq_len=q_seq_len,
        kv_seq_len=kv_seq_len,
        num_heads=num_heads,
        block_q=block_q,
        block_kv=block_kv,
        data_next_axis=_PrefixLmDataNextAxis.KV,
    )
    return _prefix_lm_mask_info_from_components(components)


def prefix_lm_dkv_mask_info(
    *,
    prefix_length: jax.Array,
    q_seq_len: int,
    kv_seq_len: int,
    num_heads: int,
    block_q: int,
    block_kv: int,
) -> splash_attention_mask_info.MaskInfo:
    """Build compact dKV Splash metadata for a prefix-LM mask."""
    components = _prefix_lm_mask_info_components(
        prefix_length=prefix_length,
        q_seq_len=q_seq_len,
        kv_seq_len=kv_seq_len,
        num_heads=num_heads,
        block_q=block_q,
        block_kv=block_kv,
        data_next_axis=_PrefixLmDataNextAxis.Q,
    )
    return _prefix_lm_mask_info_from_components(
        _PrefixLmMaskInfoComponents(
            data_next=components.data_next,
            mask_next=components.mask_next,
            block_mask=components.block_mask,
            partial_mask_blocks=jnp.swapaxes(components.partial_mask_blocks, -1, -2),
        )
    )


def _prefix_lm_mask_info_from_components(
    components: _PrefixLmMaskInfoComponents,
) -> splash_attention_mask_info.MaskInfo:
    return splash_attention_mask_info.MaskInfo(
        data_next=components.data_next,
        mask_next=components.mask_next,
        block_mask=components.block_mask.astype(jnp.int8),
        partial_mask_blocks=components.partial_mask_blocks,
        q_sequence=None,
        is_dynamic_mask=None,
    )


def _prefix_lm_mask_info_components(
    *,
    prefix_length: jax.Array,
    q_seq_len: int,
    kv_seq_len: int,
    num_heads: int,
    block_q: int,
    block_kv: int,
    data_next_axis: _PrefixLmDataNextAxis,
) -> _PrefixLmMaskInfoComponents:
    if block_q != block_kv:
        raise NotImplementedError("Compact prefix-LM metadata currently requires block_q == block_kv.")
    if q_seq_len % block_q != 0:
        raise ValueError(f"block_q={block_q} must divide q_seq_len={q_seq_len}.")
    if kv_seq_len % block_kv != 0:
        raise ValueError(f"block_kv={block_kv} must divide kv_seq_len={kv_seq_len}.")
    q_blocks = q_seq_len // block_q
    kv_blocks = kv_seq_len // block_kv
    prefix_length = jnp.asarray(prefix_length, dtype=jnp.int32)

    q_block_ids = jnp.arange(q_blocks, dtype=jnp.int32)[:, None]
    kv_block_ids = jnp.arange(kv_blocks, dtype=jnp.int32)[None, :]
    q_start = q_block_ids * block_q
    q_end = q_start + block_q - 1
    kv_start = kv_block_ids * block_kv
    kv_end = kv_start + block_kv - 1

    has_prefix = kv_start < prefix_length
    causal_nonzero = kv_start <= q_end
    nonzero = has_prefix | causal_nonzero

    full_prefix = kv_end < prefix_length
    full_causal = kv_end <= q_start
    full = full_prefix | full_causal
    partial = nonzero & ~full

    block_mask = jnp.where(full, BLOCK_MASK_FULL, jnp.where(partial, BLOCK_MASK_PARTIAL, BLOCK_MASK_EMPTY)).astype(
        jnp.int32
    )
    block_mask = jnp.broadcast_to(block_mask[None, :, :], (num_heads, q_blocks, kv_blocks))

    data_block_ids = kv_block_ids if data_next_axis == _PrefixLmDataNextAxis.KV else q_block_ids
    data_next = jnp.broadcast_to(data_block_ids, (q_blocks, kv_blocks))
    data_next = jnp.where(block_mask[0] != BLOCK_MASK_EMPTY, data_next, 0).astype(jnp.int32)
    data_next = jnp.broadcast_to(data_next[None, :, :], (num_heads, q_blocks, kv_blocks))

    causal_slot = q_block_ids * PARTIAL_MASK_SLOTS_PER_Q_BLOCK + CAUSAL_PARTIAL_MASK_SLOT_OFFSET
    prefix_boundary_block = prefix_length // block_kv
    is_prefix_boundary = kv_block_ids == prefix_boundary_block
    mask_next = jnp.where(
        is_prefix_boundary & (kv_block_ids != q_block_ids),
        causal_slot + PREFIX_PARTIAL_MASK_SLOT_OFFSET,
        causal_slot,
    )
    mask_next = jnp.where(partial, mask_next, 0).astype(jnp.int32)
    mask_next = jnp.broadcast_to(mask_next[None, :, :], (num_heads, q_blocks, kv_blocks))

    q_offsets = jnp.arange(block_q, dtype=jnp.int32)[:, None]
    kv_offsets = jnp.arange(block_kv, dtype=jnp.int32)[None, :]
    q_global = jnp.arange(q_blocks, dtype=jnp.int32)[:, None, None] * block_q + q_offsets[None, :, :]
    causal_kv_global = jnp.arange(q_blocks, dtype=jnp.int32)[:, None, None] * block_kv + kv_offsets[None, :, :]
    causal_blocks = (causal_kv_global <= q_global) | (causal_kv_global < prefix_length)

    prefix_kv_start = prefix_boundary_block * block_kv
    prefix_kv_global = prefix_kv_start + kv_offsets
    prefix_block = prefix_kv_global < prefix_length
    prefix_blocks = jnp.broadcast_to(prefix_block[None, :, :], (q_blocks, block_q, block_kv))
    prefix_blocks = prefix_blocks | (prefix_kv_global[None, :, :] <= q_global)

    partial_mask_blocks = jnp.zeros((PARTIAL_MASK_SLOTS_PER_Q_BLOCK * q_blocks, block_q, block_kv), dtype=jnp.bool_)
    partial_mask_blocks = partial_mask_blocks.at[CAUSAL_PARTIAL_MASK_SLOT_OFFSET::PARTIAL_MASK_SLOTS_PER_Q_BLOCK].set(
        causal_blocks
    )
    partial_mask_blocks = partial_mask_blocks.at[PREFIX_PARTIAL_MASK_SLOT_OFFSET::PARTIAL_MASK_SLOTS_PER_Q_BLOCK].set(
        prefix_blocks
    )

    return _PrefixLmMaskInfoComponents(
        data_next=data_next,
        mask_next=mask_next,
        block_mask=block_mask,
        partial_mask_blocks=partial_mask_blocks,
    )


class PrefixMask(splash_attention_mask.Mask):
    """Splash mask that allows every query to attend to prefix keys."""

    _shape: tuple[int, int]
    prefix_length: int
    q_sequence: np.ndarray

    def __init__(self, *, shape: tuple[int, int], prefix_length: int, shard_count: int = 1):
        if prefix_length < 0:
            raise ValueError(f"prefix_length must be non-negative, got {prefix_length}.")
        if prefix_length > shape[1]:
            raise ValueError(f"prefix_length must be <= kv sequence length {shape[1]}, got {prefix_length}.")
        q_seq_len = shape[0]
        if q_seq_len % (shard_count * shard_count) != 0:
            raise ValueError(
                f"Shard count squared ({shard_count * shard_count}) must divide Q seq_len ({q_seq_len}) evenly."
            )
        self._shape = shape
        self.prefix_length = prefix_length
        self.q_sequence = np.arange(q_seq_len, dtype=np.int32)

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    def __getitem__(self, idx) -> np.ndarray:
        if len(idx) != 2:
            raise NotImplementedError(f"Unsupported slice: {idx}")
        q_slice, kv_slice = idx
        if not isinstance(q_slice, slice) or not isinstance(kv_slice, slice):
            raise NotImplementedError(f"Unsupported slice: {idx}")

        q_start, q_stop, q_step = q_slice.indices(self.shape[0])
        kv_start, kv_stop, kv_step = kv_slice.indices(self.shape[1])
        if q_step != 1 or kv_step != 1:
            raise NotImplementedError(f"Unsupported strided slice: {idx}")

        q_size = q_stop - q_start
        kv_ids = np.arange(kv_start, kv_stop, dtype=np.int32)
        prefix = kv_ids < self.prefix_length
        return np.broadcast_to(prefix[None, :], (q_size, kv_stop - kv_start))

    def mask_function(self, _q_ids, kv_ids):
        return kv_ids < self.prefix_length

    def __eq__(self, other: object):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.shape == other.shape and self.prefix_length == other.prefix_length

    def __hash__(self):
        return hash((type(self), self.shape, self.prefix_length))


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
