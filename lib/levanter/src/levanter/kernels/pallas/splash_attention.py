# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared lowering helpers for JAX Splash Attention."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum

import equinox as eqx
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
PARTIAL_BLOCK_CAPACITY_ERROR = "Packed Splash metadata partial block capacity exceeded."
PartitionSpecEntry = str | Sequence[str | None] | None


class SplashDynamicPrefixMask(StrEnum):
    NONE = "none"
    PREFIX_LENGTHS = "prefix_lengths"
    PREFIX_MASK = "prefix_mask"


@dataclass(frozen=True)
class SplashPrefixLmMaskSpec:
    """Static or dynamic prefix-LM controls for Splash mask lowering."""

    prefix_length: int | None = None
    dynamic_prefix: SplashDynamicPrefixMask = SplashDynamicPrefixMask.NONE


@dataclass(frozen=True)
class SplashAttentionMaskSpec:
    """Static mask fields consumed by Splash Attention lowering."""

    is_causal: bool = False
    causal_offset: object | None = None
    sliding_window: int | None = None
    prefix_lm: SplashPrefixLmMaskSpec | None = None
    has_explicit_mask: bool = False


@dataclass(frozen=True)
class SplashAttentionMaskLowering:
    """JAX Splash masks lowered from a structured mask spec."""

    base_mask: splash_attention_mask.Mask
    kernel_mask: splash_attention_mask.MultiHeadMask


def splash_attention_mask_spec_from_fields(
    *,
    is_causal: bool,
    causal_offset: object | None = None,
    sliding_window: int | None = None,
    prefix_length: int | None = None,
    prefix_lengths: object | None = None,
    prefix_mask: object | None = None,
    explicit_mask: object | None = None,
) -> SplashAttentionMaskSpec:
    dynamic_prefix = SplashDynamicPrefixMask.NONE
    if prefix_lengths is not None and prefix_mask is not None:
        raise ValueError("Splash attention mask spec cannot combine prefix_lengths and prefix_mask.")
    if prefix_lengths is not None:
        dynamic_prefix = SplashDynamicPrefixMask.PREFIX_LENGTHS
    elif prefix_mask is not None:
        dynamic_prefix = SplashDynamicPrefixMask.PREFIX_MASK

    return SplashAttentionMaskSpec(
        is_causal=is_causal,
        causal_offset=causal_offset,
        sliding_window=sliding_window,
        prefix_lm=(
            None
            if prefix_length is None and dynamic_prefix == SplashDynamicPrefixMask.NONE
            else SplashPrefixLmMaskSpec(prefix_length=prefix_length, dynamic_prefix=dynamic_prefix)
        ),
        has_explicit_mask=explicit_mask is not None,
    )


@dataclass(frozen=True)
class SplashSegmentIdsLowering:
    """Segment ID arrays and sharding metadata for Splash Attention."""

    segment_ids: SplashSegmentIds | None
    segment_ids_axes: SplashSegmentIds | None
    segment_batch_axis: SplashSegmentIds | None


@dataclass(frozen=True)
class SplashDynamicMaskMetadata:
    """Compact block metadata for a dynamic Splash mask."""

    fwd_mask_info: splash_attention_mask_info.MaskInfo
    dq_mask_info: splash_attention_mask_info.MaskInfo
    dkv_mask_info: splash_attention_mask_info.MaskInfo


@dataclass(frozen=True)
class _PrefixLmMaskInfoComponents:
    data_next: jax.Array
    mask_next: jax.Array
    block_mask: jax.Array
    partial_mask_blocks: jax.Array


@dataclass(frozen=True)
class _PackedDynamicMaskContext:
    q_seq_len: int
    kv_seq_len: int
    block_sizes: splash_attention_kernel.BlockSizes
    head_shards: int
    q_seq_shards: int


@dataclass(frozen=True)
class _PartialMaskCapacities:
    fwd: int
    dq: int
    dkv: int


@dataclass(frozen=True)
class _SegmentRunBoundaries:
    active: jax.Array
    starts: jax.Array
    ends: jax.Array


class _PrefixLmDataNextAxis(StrEnum):
    Q = "q"
    KV = "kv"


class _DynamicMaskInfoRole(StrEnum):
    FWD_OR_DQ = "fwd_or_dq"
    DKV = "dkv"


def _packed_dynamic_mask_context(
    *,
    q_seq_len: int,
    kv_seq_len: int,
    block_sizes: splash_attention_kernel.BlockSizes,
    head_shards: int,
    q_seq_shards: int,
    caller: str,
) -> _PackedDynamicMaskContext:
    if not block_sizes.has_backward_blocks:
        raise ValueError(f"{caller} requires backward block sizes.")
    return _PackedDynamicMaskContext(
        q_seq_len=q_seq_len,
        kv_seq_len=kv_seq_len,
        block_sizes=block_sizes,
        head_shards=head_shards,
        q_seq_shards=q_seq_shards,
    )


def prefix_lm_mask_infos(
    *,
    prefix_length: jax.Array,
    q_seq_len: int,
    kv_seq_len: int,
    num_heads: int,
    block_sizes: splash_attention_kernel.BlockSizes,
) -> SplashDynamicMaskMetadata:
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
    return SplashDynamicMaskMetadata(
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
) -> SplashDynamicMaskMetadata:
    """Build Splash metadata for packed prefix-LM masks.

    The represented mask is ``same_segment(q, kv) & (kv <= q | prefix_mask[kv])``.
    This supports packed docs by letting callers mark prefix tokens per key
    position while segment IDs keep those prefix tokens local to their document.
    """
    context = _packed_dynamic_mask_context(
        q_seq_len=q_seq_len,
        kv_seq_len=kv_seq_len,
        block_sizes=block_sizes,
        head_shards=head_shards,
        q_seq_shards=q_seq_shards,
        caller="packed_prefix_lm_mask_infos",
    )

    def mask_block_builder(*, block_q: int, block_kv: int) -> jax.Array:
        return packed_prefix_lm_block_mask_blocks(
            prefix_mask=prefix_mask,
            q_segment_ids=q_segment_ids,
            kv_segment_ids=kv_segment_ids,
            q_seq_len=context.q_seq_len,
            kv_seq_len=context.kv_seq_len,
            block_q=block_q,
            block_kv=block_kv,
        )

    return _blocked_packed_dynamic_mask_infos(mask_block_builder=mask_block_builder, context=context)


def packed_prefix_lm_block_mask_blocks(
    *,
    prefix_mask: jax.Array,
    q_segment_ids: jax.Array,
    kv_segment_ids: jax.Array,
    q_seq_len: int,
    kv_seq_len: int,
    block_q: int,
    block_kv: int,
) -> jax.Array:
    """Build mask blocks for ``same_segment(q, kv) & (kv <= q | prefix_mask[kv])``."""
    prefix_mask = jnp.asarray(prefix_mask, dtype=jnp.bool_)
    if prefix_mask.ndim != 1:
        raise ValueError(f"prefix_mask must be rank 1 after batching, got shape {prefix_mask.shape}.")
    if prefix_mask.shape[0] != kv_seq_len:
        raise ValueError(f"prefix_mask length {prefix_mask.shape[0]} must match kv_seq_len={kv_seq_len}.")

    mask_inputs = _blocked_packed_segment_mask_inputs(
        q_segment_ids=q_segment_ids,
        kv_segment_ids=kv_segment_ids,
        q_seq_len=q_seq_len,
        kv_seq_len=kv_seq_len,
        block_q=block_q,
        block_kv=block_kv,
    )
    causal_or_prefix = mask_inputs.kv_positions <= mask_inputs.q_positions
    prefix = prefix_mask.reshape(mask_inputs.kv_blocks, block_kv)[None, :, None, :]
    return mask_inputs.same_segment & (causal_or_prefix | prefix)


def packed_causal_segment_block_mask_blocks(
    *,
    q_segment_ids: jax.Array,
    kv_segment_ids: jax.Array,
    q_seq_len: int,
    kv_seq_len: int,
    block_q: int,
    block_kv: int,
) -> jax.Array:
    """Build mask blocks for ``same_segment(q, kv) & (kv <= q)``."""
    mask_inputs = _blocked_packed_segment_mask_inputs(
        q_segment_ids=q_segment_ids,
        kv_segment_ids=kv_segment_ids,
        q_seq_len=q_seq_len,
        kv_seq_len=kv_seq_len,
        block_q=block_q,
        block_kv=block_kv,
    )
    return mask_inputs.same_segment & (mask_inputs.kv_positions <= mask_inputs.q_positions)


@dataclass(frozen=True)
class _BlockedPackedSegmentMaskInputs:
    kv_blocks: int
    q_positions: jax.Array
    kv_positions: jax.Array
    same_segment: jax.Array


def _blocked_packed_segment_mask_inputs(
    *,
    q_segment_ids: jax.Array,
    kv_segment_ids: jax.Array,
    q_seq_len: int,
    kv_seq_len: int,
    block_q: int,
    block_kv: int,
) -> _BlockedPackedSegmentMaskInputs:
    q_segment_ids = jnp.asarray(q_segment_ids)
    kv_segment_ids = jnp.asarray(kv_segment_ids)

    if q_segment_ids.ndim != 1 or kv_segment_ids.ndim != 1:
        raise ValueError(
            "q_segment_ids and kv_segment_ids must be rank 1 after batching, "
            f"got {q_segment_ids.shape} and {kv_segment_ids.shape}."
        )
    if q_segment_ids.shape[0] != q_seq_len:
        raise ValueError(f"q_segment_ids length {q_segment_ids.shape[0]} must match q_seq_len={q_seq_len}.")
    if kv_segment_ids.shape[0] != kv_seq_len:
        raise ValueError(f"kv_segment_ids length {kv_segment_ids.shape[0]} must match kv_seq_len={kv_seq_len}.")
    if q_seq_len % block_q != 0:
        raise ValueError(f"block_q={block_q} must divide q_seq_len={q_seq_len}.")
    if kv_seq_len % block_kv != 0:
        raise ValueError(f"block_kv={block_kv} must divide kv_seq_len={kv_seq_len}.")

    q_blocks = q_seq_len // block_q
    kv_blocks = kv_seq_len // block_kv
    q_segment_blocks = q_segment_ids.reshape(q_blocks, block_q)
    kv_segment_blocks = kv_segment_ids.reshape(kv_blocks, block_kv)
    same_segment = q_segment_blocks[:, None, :, None] == kv_segment_blocks[None, :, None, :]

    q_positions = jnp.arange(q_seq_len, dtype=jnp.int32).reshape(q_blocks, block_q)
    kv_positions = jnp.arange(kv_seq_len, dtype=jnp.int32).reshape(kv_blocks, block_kv)
    return _BlockedPackedSegmentMaskInputs(
        kv_blocks=kv_blocks,
        q_positions=q_positions[:, None, :, None],
        kv_positions=kv_positions[None, :, None, :],
        same_segment=same_segment,
    )


def _blocked_packed_dynamic_mask_infos(
    *,
    mask_block_builder: Callable[..., jax.Array],
    context: _PackedDynamicMaskContext,
    partial_capacities: _PartialMaskCapacities | None = None,
) -> SplashDynamicMaskMetadata:
    fwd_mask_info = _blocked_packed_dynamic_mask_info(
        mask_block_builder(
            block_q=context.block_sizes.block_q,
            block_kv=context.block_sizes.block_kv,
        ),
        role=_DynamicMaskInfoRole.FWD_OR_DQ,
        head_shards=context.head_shards,
        q_seq_shards=context.q_seq_shards,
        partial_capacity=None if partial_capacities is None else partial_capacities.fwd,
    )
    dq_mask_info = _blocked_packed_dynamic_mask_info(
        mask_block_builder(
            block_q=context.block_sizes.block_q_dq,
            block_kv=context.block_sizes.block_kv_dq,
        ),
        role=_DynamicMaskInfoRole.FWD_OR_DQ,
        head_shards=context.head_shards,
        q_seq_shards=context.q_seq_shards,
        partial_capacity=None if partial_capacities is None else partial_capacities.dq,
    )
    dkv_mask_info = _blocked_packed_dynamic_mask_info(
        mask_block_builder(
            block_q=context.block_sizes.block_q_dkv,
            block_kv=context.block_sizes.block_kv_dkv,
        ),
        role=_DynamicMaskInfoRole.DKV,
        head_shards=context.head_shards,
        q_seq_shards=context.q_seq_shards,
        partial_capacity=None if partial_capacities is None else partial_capacities.dkv,
    )
    return SplashDynamicMaskMetadata(
        fwd_mask_info=fwd_mask_info,
        dq_mask_info=dq_mask_info,
        dkv_mask_info=dkv_mask_info,
    )


def _blocked_packed_dynamic_mask_info(
    partial_mask_blocks: jax.Array,
    *,
    role: _DynamicMaskInfoRole,
    head_shards: int,
    q_seq_shards: int,
    partial_capacity: int | None,
) -> splash_attention_mask_info.MaskInfo:
    if partial_mask_blocks.ndim != 4:
        raise ValueError(f"Expected blocked mask with rank 4, got {partial_mask_blocks.shape}.")
    if partial_mask_blocks.dtype != jnp.bool_:
        raise ValueError(f"Expected bool blocked mask, got {partial_mask_blocks.dtype}.")
    if head_shards != 1:
        raise NotImplementedError("Packed dynamic Splash metadata with shared heads currently requires head_shards=1.")

    q_blocks, kv_blocks, _, _ = partial_mask_blocks.shape
    if q_blocks % q_seq_shards != 0:
        raise ValueError(f"q_seq_shards={q_seq_shards} must divide q block count {q_blocks}.")

    is_full_mask = jnp.all(partial_mask_blocks, axis=(-1, -2))
    is_empty_mask = jnp.logical_not(jnp.any(partial_mask_blocks, axis=(-1, -2)))

    block_mask = jnp.ones((1, q_blocks, kv_blocks), dtype=jnp.int32)
    block_mask = jnp.where(is_full_mask[None, :, :], BLOCK_MASK_FULL, block_mask)
    block_mask = jnp.where(is_empty_mask[None, :, :], BLOCK_MASK_EMPTY, block_mask)

    compact_mask_blocks, mask_next = _compact_partial_mask_blocks(
        partial_mask_blocks,
        block_mask=block_mask,
        partial_capacity=partial_capacity or (q_blocks * kv_blocks),
    )

    return _packed_dynamic_mask_info_from_components(
        block_mask=block_mask,
        compact_mask_blocks=compact_mask_blocks,
        mask_next=mask_next,
        q_blocks=q_blocks,
        kv_blocks=kv_blocks,
        role=role,
        partial_capacity=partial_capacity or (q_blocks * kv_blocks),
    )


def _compact_partial_mask_blocks(
    partial_mask_blocks: jax.Array,
    *,
    block_mask: jax.Array,
    partial_capacity: int,
) -> tuple[jax.Array, jax.Array]:
    q_blocks, kv_blocks, block_q, block_kv = partial_mask_blocks.shape
    flat_blocks = partial_mask_blocks.reshape(q_blocks * kv_blocks, block_q, block_kv)
    partial_indexing = _partial_block_indexing(block_mask, q_blocks=q_blocks, kv_blocks=kv_blocks)

    partial_flat_indices = jnp.nonzero(partial_indexing.is_partial, size=partial_capacity, fill_value=0)[0]
    compact_blocks = flat_blocks[partial_flat_indices]
    valid_slots = jnp.arange(partial_capacity, dtype=jnp.int32) < partial_indexing.num_partial
    compact_blocks = jnp.where(valid_slots[:, None, None], compact_blocks, False)
    compact_blocks = eqx.error_if(
        compact_blocks,
        partial_indexing.num_partial > partial_capacity,
        PARTIAL_BLOCK_CAPACITY_ERROR,
    )

    return compact_blocks, partial_indexing.mask_next


@dataclass(frozen=True)
class _PartialBlockIndexing:
    is_partial: jax.Array
    num_partial: jax.Array
    mask_next: jax.Array


def _partial_block_indexing(block_mask: jax.Array, *, q_blocks: int, kv_blocks: int) -> _PartialBlockIndexing:
    is_partial = block_mask.reshape(-1) == BLOCK_MASK_PARTIAL
    partial_indices = jnp.cumsum(is_partial.astype(jnp.int32), dtype=jnp.int32) - 1
    num_partial = jnp.sum(is_partial.astype(jnp.int32), dtype=jnp.int32)
    mask_next = jnp.where(is_partial, partial_indices, 0).reshape(1, q_blocks, kv_blocks)
    return _PartialBlockIndexing(is_partial=is_partial, num_partial=num_partial, mask_next=mask_next)


def _packed_dynamic_mask_info_from_components(
    *,
    block_mask: jax.Array,
    compact_mask_blocks: jax.Array,
    mask_next: jax.Array,
    q_blocks: int,
    kv_blocks: int,
    role: _DynamicMaskInfoRole,
    partial_capacity: int,
) -> splash_attention_mask_info.MaskInfo:
    if role == _DynamicMaskInfoRole.DKV:
        data_next = jnp.arange(q_blocks, dtype=jnp.int32)[None, :, None]
        data_next = jnp.broadcast_to(data_next, (1, q_blocks, kv_blocks))
        compact_mask_blocks = compact_mask_blocks.swapaxes(-1, -2)
    else:
        data_next = jnp.arange(kv_blocks, dtype=jnp.int32)[None, None, :]
        data_next = jnp.broadcast_to(data_next, (1, q_blocks, kv_blocks))
    data_next = jnp.where(block_mask == BLOCK_MASK_EMPTY, 0, data_next)

    return splash_attention_mask_info.MaskInfo(
        data_next=_downcast_dynamic_mask_index(
            data_next,
            q_blocks if role == _DynamicMaskInfoRole.DKV else kv_blocks,
        ),
        mask_next=_downcast_dynamic_mask_index(mask_next, partial_capacity),
        block_mask=block_mask.astype(jnp.int8),
        partial_mask_blocks=compact_mask_blocks,
        q_sequence=None,
        is_dynamic_mask=True,
    )


def _downcast_dynamic_mask_index(array: jax.Array, max_value: int) -> jax.Array:
    if array.size == 0:
        return array
    if max_value <= np.iinfo(np.int8).max:
        return array.astype(jnp.int8)
    if max_value <= np.iinfo(np.int16).max:
        return array.astype(jnp.int16)
    return array.astype(jnp.int32)


def packed_causal_segment_mask_infos(
    *,
    q_segment_ids: jax.Array,
    kv_segment_ids: jax.Array,
    q_seq_len: int,
    kv_seq_len: int,
    block_sizes: splash_attention_kernel.BlockSizes,
    head_shards: int = 1,
    q_seq_shards: int = 1,
) -> SplashDynamicMaskMetadata:
    """Build Splash metadata for packed causal masks.

    The represented mask is ``same_segment(q, kv) & (kv <= q)``.
    """
    context = _packed_dynamic_mask_context(
        q_seq_len=q_seq_len,
        kv_seq_len=kv_seq_len,
        block_sizes=block_sizes,
        head_shards=head_shards,
        q_seq_shards=q_seq_shards,
        caller="packed_causal_segment_mask_infos",
    )

    def mask_block_builder(*, block_q: int, block_kv: int) -> jax.Array:
        return packed_causal_segment_block_mask_blocks(
            q_segment_ids=q_segment_ids,
            kv_segment_ids=kv_segment_ids,
            q_seq_len=context.q_seq_len,
            kv_seq_len=context.kv_seq_len,
            block_q=block_q,
            block_kv=block_kv,
        )

    return _blocked_packed_dynamic_mask_infos(mask_block_builder=mask_block_builder, context=context)


def packed_causal_segment_run_mask_infos(
    *,
    segment_lengths: jax.Array,
    num_segments: jax.Array,
    q_seq_len: int,
    kv_seq_len: int,
    block_sizes: splash_attention_kernel.BlockSizes,
    head_shards: int = 1,
    q_seq_shards: int = 1,
) -> SplashDynamicMaskMetadata:
    """Build packed causal metadata from fixed-shape contiguous segment lengths."""
    if q_seq_len != kv_seq_len:
        raise NotImplementedError("Segment-run Splash metadata currently supports self-attention only.")
    context = _packed_dynamic_mask_context(
        q_seq_len=q_seq_len,
        kv_seq_len=kv_seq_len,
        block_sizes=block_sizes,
        head_shards=head_shards,
        q_seq_shards=q_seq_shards,
        caller="packed_causal_segment_run_mask_infos",
    )

    segment_lengths = jnp.asarray(segment_lengths, dtype=jnp.int32)
    if segment_lengths.ndim != 1:
        raise ValueError(f"segment_lengths must be rank 1 after batching, got shape {segment_lengths.shape}.")

    max_segments = segment_lengths.shape[0]
    partial_capacities = _segment_run_partial_capacities(
        q_seq_len=context.q_seq_len,
        kv_seq_len=context.kv_seq_len,
        block_sizes=context.block_sizes,
        max_segments=max_segments,
    )

    return SplashDynamicMaskMetadata(
        fwd_mask_info=_segment_run_dynamic_mask_info(
            segment_lengths=segment_lengths,
            num_segments=num_segments,
            seq_len=context.q_seq_len,
            block_q=context.block_sizes.block_q,
            block_kv=context.block_sizes.block_kv,
            role=_DynamicMaskInfoRole.FWD_OR_DQ,
            head_shards=context.head_shards,
            q_seq_shards=context.q_seq_shards,
            partial_capacity=partial_capacities.fwd,
        ),
        dq_mask_info=_segment_run_dynamic_mask_info(
            segment_lengths=segment_lengths,
            num_segments=num_segments,
            seq_len=context.q_seq_len,
            block_q=_required_block_size(context.block_sizes.block_q_dq, "block_q_dq"),
            block_kv=_required_block_size(context.block_sizes.block_kv_dq, "block_kv_dq"),
            role=_DynamicMaskInfoRole.FWD_OR_DQ,
            head_shards=context.head_shards,
            q_seq_shards=context.q_seq_shards,
            partial_capacity=partial_capacities.dq,
        ),
        dkv_mask_info=_segment_run_dynamic_mask_info(
            segment_lengths=segment_lengths,
            num_segments=num_segments,
            seq_len=context.q_seq_len,
            block_q=_required_block_size(context.block_sizes.block_q_dkv, "block_q_dkv"),
            block_kv=_required_block_size(context.block_sizes.block_kv_dkv, "block_kv_dkv"),
            role=_DynamicMaskInfoRole.DKV,
            head_shards=context.head_shards,
            q_seq_shards=context.q_seq_shards,
            partial_capacity=partial_capacities.dkv,
        ),
    )


def _segment_run_dynamic_mask_info(
    *,
    segment_lengths: jax.Array,
    num_segments: jax.Array,
    seq_len: int,
    block_q: int,
    block_kv: int,
    role: _DynamicMaskInfoRole,
    head_shards: int,
    q_seq_shards: int,
    partial_capacity: int,
) -> splash_attention_mask_info.MaskInfo:
    if head_shards != 1:
        raise NotImplementedError("Packed dynamic Splash metadata with shared heads currently requires head_shards=1.")
    if seq_len % block_q != 0:
        raise ValueError(f"block_q={block_q} must divide seq_len={seq_len}.")
    if seq_len % block_kv != 0:
        raise ValueError(f"block_kv={block_kv} must divide seq_len={seq_len}.")

    q_blocks = seq_len // block_q
    kv_blocks = seq_len // block_kv
    if q_blocks % q_seq_shards != 0:
        raise ValueError(f"q_seq_shards={q_seq_shards} must divide q block count {q_blocks}.")

    boundaries = _segment_run_boundaries(
        segment_lengths=segment_lengths,
        num_segments=num_segments,
        seq_len=seq_len,
    )
    block_mask = _segment_run_block_mask(
        active=boundaries.active,
        segment_starts=boundaries.starts,
        segment_ends=boundaries.ends,
        q_blocks=q_blocks,
        kv_blocks=kv_blocks,
        block_q=block_q,
        block_kv=block_kv,
    )
    partial_indexing = _partial_block_indexing(block_mask, q_blocks=q_blocks, kv_blocks=kv_blocks)

    compact_mask_blocks = _segment_run_partial_mask_blocks(
        segment_lengths=segment_lengths,
        num_segments=num_segments,
        seq_len=seq_len,
        kv_blocks=kv_blocks,
        block_q=block_q,
        block_kv=block_kv,
        is_partial=partial_indexing.is_partial,
        num_partial=partial_indexing.num_partial,
        partial_capacity=partial_capacity,
    )
    compact_mask_blocks = eqx.error_if(
        compact_mask_blocks,
        partial_indexing.num_partial > partial_capacity,
        PARTIAL_BLOCK_CAPACITY_ERROR,
    )

    return _packed_dynamic_mask_info_from_components(
        block_mask=block_mask,
        compact_mask_blocks=compact_mask_blocks,
        mask_next=partial_indexing.mask_next,
        q_blocks=q_blocks,
        kv_blocks=kv_blocks,
        role=role,
        partial_capacity=partial_capacity,
    )


def _segment_run_boundaries(
    *,
    segment_lengths: jax.Array,
    num_segments: jax.Array,
    seq_len: int,
) -> _SegmentRunBoundaries:
    num_segments = jnp.asarray(num_segments, dtype=jnp.int32)
    active = jnp.arange(segment_lengths.shape[0], dtype=jnp.int32) < num_segments
    lengths = jnp.where(active, segment_lengths, jnp.zeros_like(segment_lengths))
    lengths = eqx.error_if(
        lengths,
        jnp.sum(lengths, dtype=jnp.int32) != seq_len,
        "Packed Splash segment lengths must cover the full sequence.",
    )
    segment_ends = jnp.cumsum(lengths, dtype=jnp.int32)
    segment_starts = segment_ends - lengths
    return _SegmentRunBoundaries(active=active, starts=segment_starts, ends=segment_ends)


def _segment_run_block_mask(
    *,
    active: jax.Array,
    segment_starts: jax.Array,
    segment_ends: jax.Array,
    q_blocks: int,
    kv_blocks: int,
    block_q: int,
    block_kv: int,
) -> jax.Array:
    q_start = (jnp.arange(q_blocks, dtype=jnp.int32) * block_q)[:, None, None]
    q_end = q_start + block_q - 1
    kv_start = (jnp.arange(kv_blocks, dtype=jnp.int32) * block_kv)[None, :, None]
    kv_end = kv_start + block_kv - 1
    segment_start = segment_starts[None, None, :]
    segment_end = segment_ends[None, None, :] - 1
    active = active[None, None, :]

    q_overlap_start = jnp.maximum(q_start, segment_start)
    q_overlap_end = jnp.minimum(q_end, segment_end)
    kv_overlap_start = jnp.maximum(kv_start, segment_start)
    kv_overlap_end = jnp.minimum(kv_end, segment_end)

    q_intersects = q_overlap_start <= q_overlap_end
    kv_intersects = kv_overlap_start <= kv_overlap_end
    non_empty = active & q_intersects & kv_intersects & (kv_overlap_start <= q_overlap_end)
    full = active & (q_start >= segment_start) & (q_end <= segment_end) & (kv_start >= segment_start)
    full = full & (kv_end <= segment_end) & (kv_end <= q_start)

    any_non_empty = jnp.any(non_empty, axis=-1)
    any_full = jnp.any(full, axis=-1)
    block_mask = jnp.where(any_full, BLOCK_MASK_FULL, jnp.where(any_non_empty, BLOCK_MASK_PARTIAL, BLOCK_MASK_EMPTY))
    return block_mask[None, :, :].astype(jnp.int32)


def _segment_run_partial_mask_blocks(
    *,
    segment_lengths: jax.Array,
    num_segments: jax.Array,
    seq_len: int,
    kv_blocks: int,
    block_q: int,
    block_kv: int,
    is_partial: jax.Array,
    num_partial: jax.Array,
    partial_capacity: int,
) -> jax.Array:
    partial_flat_indices = jnp.nonzero(is_partial, size=partial_capacity, fill_value=0)[0]
    q_block_ids = partial_flat_indices // kv_blocks
    kv_block_ids = partial_flat_indices % kv_blocks

    q_positions = q_block_ids[:, None] * block_q + jnp.arange(block_q, dtype=jnp.int32)[None, :]
    kv_positions = kv_block_ids[:, None] * block_kv + jnp.arange(block_kv, dtype=jnp.int32)[None, :]
    segment_ids = _segment_ids_from_lengths(
        segment_lengths=segment_lengths,
        num_segments=num_segments,
        seq_len=seq_len,
    )
    q_segment_ids = segment_ids[q_positions]
    kv_segment_ids = segment_ids[kv_positions]
    compact_blocks = (q_segment_ids[:, :, None] == kv_segment_ids[:, None, :]) & (
        kv_positions[:, None, :] <= q_positions[:, :, None]
    )
    valid_slots = jnp.arange(partial_capacity, dtype=jnp.int32) < num_partial
    return jnp.where(valid_slots[:, None, None], compact_blocks, False)


def _segment_ids_from_lengths(
    *,
    segment_lengths: jax.Array,
    num_segments: jax.Array,
    seq_len: int,
) -> jax.Array:
    boundaries = _segment_run_boundaries(
        segment_lengths=segment_lengths,
        num_segments=num_segments,
        seq_len=seq_len,
    )
    positions = jnp.arange(seq_len, dtype=jnp.int32)
    return jnp.sum(positions[:, None] >= boundaries.ends[None, :], axis=1, dtype=jnp.int32)


def _segment_run_partial_capacities(
    *,
    q_seq_len: int,
    kv_seq_len: int,
    block_sizes: splash_attention_kernel.BlockSizes,
    max_segments: int,
) -> _PartialMaskCapacities:
    return _PartialMaskCapacities(
        fwd=_segment_run_partial_capacity(
            q_blocks=q_seq_len // block_sizes.block_q,
            kv_blocks=kv_seq_len // block_sizes.block_kv,
            max_segments=max_segments,
        ),
        dq=_segment_run_partial_capacity(
            q_blocks=q_seq_len // _required_block_size(block_sizes.block_q_dq, "block_q_dq"),
            kv_blocks=kv_seq_len // _required_block_size(block_sizes.block_kv_dq, "block_kv_dq"),
            max_segments=max_segments,
        ),
        dkv=_segment_run_partial_capacity(
            q_blocks=q_seq_len // _required_block_size(block_sizes.block_q_dkv, "block_q_dkv"),
            kv_blocks=kv_seq_len // _required_block_size(block_sizes.block_kv_dkv, "block_kv_dkv"),
            max_segments=max_segments,
        ),
    )


def _required_block_size(block_size: int | None, name: str) -> int:
    if block_size is None:
        raise ValueError(f"{name} must be set.")
    return block_size


def _segment_run_partial_capacity(*, q_blocks: int, kv_blocks: int, max_segments: int) -> int:
    dense_capacity = q_blocks * kv_blocks
    segment_capacity = max_segments * (q_blocks + kv_blocks + 1)
    return min(dense_capacity, segment_capacity)


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
        prefix_lm = mask.prefix_lm
        if prefix_lm is not None and prefix_lm.dynamic_prefix != SplashDynamicPrefixMask.NONE:
            raise NotImplementedError("Dynamic prefix-LM masks are not yet supported for splash attention")
        if prefix_lm is not None and prefix_lm.prefix_length is not None and not mask.is_causal:
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

        if prefix_lm is not None and prefix_lm.prefix_length is not None:
            prefix_mask = PrefixMask(
                shape=(q_seq_len, kv_seq_len),
                prefix_length=prefix_lm.prefix_length,
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
