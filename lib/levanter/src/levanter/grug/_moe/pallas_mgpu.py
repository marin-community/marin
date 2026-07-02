# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import functools
import itertools
import math
import os
import warnings
from collections.abc import Callable
from dataclasses import dataclass

import jax
import numpy as np
from jax import lax, random
from jax import numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.mosaic.gpu import profiler
from jax.experimental.pallas import mosaic_gpu as mgpu
from jaxtyping import Array, Float, Int

from haliax.nn.ragged_dot import ragged_dot
from levanter.grug._moe.ep_common import (
    _clip_receiver_group_sizes,
    _compact_by_keep_mask,
    _expand_from_keep_mask,
    _expert_prefix_keep_mask,
    _prefix_cap_counts,
)
from levanter.grug._moe.ep_ragged_all_to_all import _moe_mlp_ep_ragged_a2a_local


_WGMMA_M_TILING = 8
_DISPATCH_COPY_TILE = 128
_DISPATCH_COPY_ASSIGNMENT_MAJOR = "assignment_major"
_DISPATCH_COPY_EXPERT_GROUP_PEER = "expert_group_peer"
_DISPATCH_COPY_SCHEDULES = frozenset(
    {
        _DISPATCH_COPY_ASSIGNMENT_MAJOR,
        _DISPATCH_COPY_EXPERT_GROUP_PEER,
    }
)


@dataclass(frozen=True)
class MoeMgpuConfig:
    block_m: int = 64
    block_n: int = 128
    block_k: int = 64
    max_concurrent_steps: int = 4
    grid_block_n: int = 2
    capacity_factor: float = 1.25
    num_sms: int | None = None
    deterministic: bool = True
    dispatch_copy_schedule: str = _DISPATCH_COPY_EXPERT_GROUP_PEER
    dispatch_peer_phase_order: tuple[int, ...] | None = None
    dispatch_swizzle_expert_groups: bool = False
    dispatch_expert_group_size: int = 8
    dispatch_chunk_copy_tile: int = _DISPATCH_COPY_TILE
    dispatch_chunk_copy_rows: int = 1
    dispatch_chunk_vectorized_copy_rows: bool = False
    dispatch_user_kernel_permute_up: bool = False
    dispatch_static_workqueue_permute_up: bool = False
    dispatch_static_workqueue_max_rows_per_src_expert: int | None = None
    dispatch_static_workqueue_init_recv_x: bool = False
    dispatch_static_workqueue_init_hidden: bool = False
    dispatch_static_workqueue_direct_store_hidden: bool = False
    dispatch_static_workqueue_copy_pl_loop: bool = False
    dispatch_static_workqueue_dense_balanced_compute: bool = False
    dispatch_static_workqueue_destination_pull: bool = False
    dispatch_static_workqueue_destination_pull_n_group: int = 2
    dispatch_fuse_metadata: bool = True
    dispatch_chunked_permute_up: bool = False
    dispatch_split_wg_permute_up: bool = False
    dispatch_split_wg_overlap_permute_up: bool = False
    combine_bwd_block_n: int = 512
    dx_unpermute_block_n: int = 2560

    @classmethod
    def destination_pull_dense_hopper(
        cls,
        *,
        capacity_factor: float = 1.25,
        max_rows_per_src_expert: int | None = None,
    ) -> "MoeMgpuConfig":
        """Tuned Hopper destination-pull fast path for block-aligned balanced routing."""
        return cls(
            block_m=64,
            block_n=128,
            block_k=128,
            grid_block_n=4,
            capacity_factor=capacity_factor,
            dispatch_expert_group_size=32,
            dispatch_static_workqueue_permute_up=True,
            dispatch_static_workqueue_max_rows_per_src_expert=max_rows_per_src_expert,
            dispatch_static_workqueue_dense_balanced_compute=True,
            dispatch_static_workqueue_destination_pull=True,
            dispatch_static_workqueue_destination_pull_n_group=2,
        )

    def __post_init__(self) -> None:
        if not self.deterministic:
            raise ValueError("MoeMgpuConfig only supports deterministic=True")
        for field_name in ("block_m", "block_n", "block_k", "max_concurrent_steps", "grid_block_n"):
            value = getattr(self, field_name)
            if value <= 0:
                raise ValueError(f"{field_name} must be positive, got {value}")
        if self.capacity_factor <= 0:
            raise ValueError(f"capacity_factor must be positive, got {self.capacity_factor}")
        if self.num_sms is not None and self.num_sms <= 0:
            raise ValueError(f"num_sms must be positive when set, got {self.num_sms}")
        if self.dispatch_copy_schedule not in _DISPATCH_COPY_SCHEDULES:
            raise ValueError(
                f"unknown dispatch_copy_schedule={self.dispatch_copy_schedule!r}; "
                f"expected one of {sorted(_DISPATCH_COPY_SCHEDULES)}"
            )
        if self.dispatch_peer_phase_order is not None:
            if not self.dispatch_peer_phase_order:
                raise ValueError("dispatch_peer_phase_order must not be empty when set")
            if min(self.dispatch_peer_phase_order) < 0:
                raise ValueError(
                    f"dispatch_peer_phase_order entries must be non-negative, got {self.dispatch_peer_phase_order}"
                )
            if len(set(self.dispatch_peer_phase_order)) != len(self.dispatch_peer_phase_order):
                raise ValueError(
                    f"dispatch_peer_phase_order entries must be unique, got {self.dispatch_peer_phase_order}"
                )
        if self.dispatch_expert_group_size <= 0:
            raise ValueError(f"dispatch_expert_group_size must be positive, got {self.dispatch_expert_group_size}")
        if self.dispatch_chunk_copy_tile <= 0:
            raise ValueError(f"dispatch_chunk_copy_tile must be positive, got {self.dispatch_chunk_copy_tile}")
        if self.dispatch_chunk_copy_rows <= 0:
            raise ValueError(f"dispatch_chunk_copy_rows must be positive, got {self.dispatch_chunk_copy_rows}")
        if self.dispatch_static_workqueue_destination_pull_n_group not in (1, 2, 5):
            raise ValueError(
                "dispatch_static_workqueue_destination_pull_n_group currently supports only 1, 2, or 5, got "
                f"{self.dispatch_static_workqueue_destination_pull_n_group}"
            )
        if self.dispatch_user_kernel_permute_up and (
            self.dispatch_static_workqueue_permute_up or self.dispatch_chunked_permute_up
        ):
            raise ValueError(
                "dispatch_user_kernel_permute_up is a standalone permute_up experiment; do not combine it with "
                "dispatch_static_workqueue_permute_up or dispatch_chunked_permute_up"
            )
        if self.dispatch_static_workqueue_permute_up and self.dispatch_chunked_permute_up:
            raise ValueError(
                "dispatch_static_workqueue_permute_up and dispatch_chunked_permute_up are separate experiments; "
                "enable only one"
            )
        if (
            self.dispatch_static_workqueue_max_rows_per_src_expert is not None
            and self.dispatch_static_workqueue_max_rows_per_src_expert <= 0
        ):
            raise ValueError(
                "dispatch_static_workqueue_max_rows_per_src_expert must be positive when set, got "
                f"{self.dispatch_static_workqueue_max_rows_per_src_expert}"
            )
        if self.dispatch_static_workqueue_destination_pull:
            if not self.dispatch_static_workqueue_permute_up:
                raise ValueError(
                    "dispatch_static_workqueue_destination_pull requires " "dispatch_static_workqueue_permute_up=True"
                )
            if self.dispatch_split_wg_permute_up:
                if self.dispatch_split_wg_overlap_permute_up:
                    raise ValueError("dispatch_static_workqueue_destination_pull does not support split-wg overlap")
                if self.dispatch_static_workqueue_destination_pull_n_group != 1:
                    raise ValueError(
                        "dispatch_static_workqueue_destination_pull split-wg RHS buffering currently requires "
                        "dispatch_static_workqueue_destination_pull_n_group=1"
                    )
            if self.block_m % 64 != 0:
                raise ValueError(
                    "dispatch_static_workqueue_destination_pull requires block_m to be a multiple of 64 for WGMMA; "
                    f"got block_m={self.block_m}"
                )
        if self.dispatch_split_wg_permute_up and not (
            self.dispatch_chunked_permute_up or self.dispatch_static_workqueue_permute_up
        ):
            raise ValueError(
                "dispatch_split_wg_permute_up requires dispatch_chunked_permute_up=True or "
                "dispatch_static_workqueue_permute_up=True"
            )
        if self.dispatch_split_wg_overlap_permute_up and not self.dispatch_split_wg_permute_up:
            raise ValueError("dispatch_split_wg_overlap_permute_up requires dispatch_split_wg_permute_up=True")
        if self.combine_bwd_block_n <= 0:
            raise ValueError(f"combine_bwd_block_n must be positive, got {self.combine_bwd_block_n}")
        if self.dx_unpermute_block_n <= 0:
            raise ValueError(f"dx_unpermute_block_n must be positive, got {self.dx_unpermute_block_n}")


def _raise_unless_chunked_counts_balanced(is_balanced: jax.Array, rows_per_source_expert: jax.Array) -> None:
    if not bool(np.asarray(is_balanced)):
        expected = int(np.asarray(rows_per_source_expert))
        raise ValueError(
            "dispatch_chunked_permute_up is a synthetic balanced-routing diagnostic. "
            "Every send_counts[dst, expert] bucket must equal "
            f"assignments // (EP * E_local) == {expected}; use dispatch_static_workqueue_permute_up for real routing."
        )


def _require_chunked_balanced_counts(send_counts: jax.Array, rows_per_source_expert: int) -> None:
    """Reject the old chunked diagnostic unless its balanced-count assumption holds."""
    expected = jnp.asarray(rows_per_source_expert, dtype=send_counts.dtype)
    is_balanced = jnp.all(send_counts == expected)
    jax.debug.callback(
        _raise_unless_chunked_counts_balanced,
        is_balanced,
        jnp.asarray(rows_per_source_expert, dtype=jnp.int32),
    )


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class MoeMgpuRoutingMetadata:
    """Source-side deterministic assignment metadata for the MGPU MoE layout."""

    assignment_ids_sorted: Int[Array, "A"]
    token_ids_sorted: Int[Array, "A"]
    expert_ids_sorted: Int[Array, "A"]
    dst_ranks_sorted: Int[Array, "A"]
    local_experts_sorted: Int[Array, "A"]
    local_pos_sorted: Int[Array, "A"]
    send_counts: Int[Array, "EP Elocal"]
    global_counts: Int[Array, "EP EP Elocal"]


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class _MoeMgpuUpMetadata:
    global_expert_counts: Int[Array, "EP E"]  # Global expert counts across all ranks


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class MoeMgpuReceiveLayout:
    recv_x: Float[Array, "capacity D"]
    recv_src_rank: Int[Array, "capacity"]
    recv_src_assignment: Int[Array, "capacity"]
    rows_per_expert: Int[Array, "Elocal"]
    clipped_global_counts: Int[Array, "EP EP Elocal"]
    dropped: Int[Array, ""]


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class MoeMgpuUpLayout:
    hidden: Float[Array, "capacity I"]
    recv_src_rank: Int[Array, "capacity"]
    recv_src_assignment: Int[Array, "capacity"]
    rows_per_expert: Int[Array, "Elocal"]
    clipped_global_counts: Int[Array, "EP EP Elocal"]
    dropped: Int[Array, ""]


@dataclass(frozen=True)
class GroupInfo:
    """Information regarding the group being processed in a block."""

    group_id: jax.Array
    block: jax.Array
    block_start: jax.Array
    actual_start: jax.Array
    actual_end: jax.Array
    start_within_block: jax.Array
    actual_size: jax.Array

    @classmethod
    def create(cls, group_lengths, tile, tid):
        """Get the group info for the current block."""

        tile = jnp.int32(tile)
        group_boundaries = [group_lengths[i] for i in range(len(group_lengths))]

        # We usually only have very few groups, so we unroll the loop processing
        # them. Normally we'd break out of the loop early, once we'd have found our
        # boundary, but we can't do that when unrolling, so we rely on many selects
        # to mask out the epilogue of the loop.
        group_end = group_start = block = group = end = jnp.array(0, dtype=jnp.int32)

        for i, b in enumerate(group_boundaries):
            # Start/end are inclusive
            start = end
            end = start + b
            final = end - 1
            start_block = lax.div(start, tile)
            final_block = lax.div(final, tile)
            block_end = final_block + 1
            tid_begin = start_block + i
            tid_end = block_end + i
            # How many blocks after is our block?
            this_is_group = (b > 0) & (tid_begin <= tid) & (tid < tid_end)
            block = lax.select(this_is_group, tid - tid_begin + start_block, block)
            group = lax.select(this_is_group, jnp.int32(i), group)
            group_start = lax.select(this_is_group, start, group_start)
            group_end = lax.select(this_is_group, end, group_end)

        block_start = block * tile
        actual_start = jnp.maximum(group_start, block_start)
        actual_end = jnp.minimum(group_end, block_start + tile)
        start_within_block = actual_start - block_start
        actual_size = actual_end - actual_start
        return cls(
            group_id=group,
            block=block,
            block_start=block_start,
            actual_start=actual_start,
            actual_end=actual_end,
            start_within_block=start_within_block,
            actual_size=actual_size,
        )


def _receiver_capacity(tokens_per_shard: int, topk: int, local_experts: int, capacity_factor: float) -> int:
    assignments_per_shard = tokens_per_shard * topk
    requested_capacity = max(local_experts, int(math.ceil(capacity_factor * assignments_per_shard)))
    return _pad_receiver_capacity_for_wgmma(requested_capacity)


def _pad_receiver_capacity_for_wgmma(capacity: int) -> int:
    if capacity < _WGMMA_M_TILING:
        return capacity
    return int(math.ceil(capacity / _WGMMA_M_TILING) * _WGMMA_M_TILING)


def _warn_if_receiver_capacity_padded(
    tokens_per_shard: int,
    topk: int,
    local_experts: int,
    capacity_factor: float,
) -> None:
    assignments_per_shard = tokens_per_shard * topk
    requested_capacity = max(local_experts, int(math.ceil(capacity_factor * assignments_per_shard)))
    padded_capacity = _pad_receiver_capacity_for_wgmma(requested_capacity)
    if padded_capacity == requested_capacity:
        return
    warnings.warn(
        "implementation='pallas_mgpu' padded receiver capacity from "
        f"{requested_capacity} to {padded_capacity} rows to satisfy Mosaic WGMMA tiling",
        RuntimeWarning,
        stacklevel=2,
    )


def _warn_if_wgmma_m_padded(requested_rows: int, padded_rows: int) -> None:
    if padded_rows == requested_rows:
        return
    warnings.warn(
        "implementation='pallas_mgpu' padded WGMMA M dimension from "
        f"{requested_rows} to {padded_rows} rows to satisfy Mosaic WGMMA tiling",
        RuntimeWarning,
        stacklevel=2,
    )


def _effective_padded_capacity_factor(
    tokens_per_shard: int,
    topk: int,
    local_experts: int,
    capacity_factor: float,
) -> float:
    assignments_per_shard = tokens_per_shard * topk
    receiver_capacity = _receiver_capacity(tokens_per_shard, topk, local_experts, capacity_factor)
    return receiver_capacity / assignments_per_shard


def _stable_assignment_sort(expert_ids_flat: jax.Array) -> jax.Array:
    return jnp.argsort(expert_ids_flat, axis=0, stable=True).astype(jnp.int32)


def _expert_group_peer_copy_order(
    dst_ranks_sorted: jax.Array,
    local_experts_sorted: jax.Array,
    local_pos_sorted: jax.Array,
    *,
    rank: jax.Array,
    ep_size: int,
    local_experts: int,
    expert_group_size: int,
) -> jax.Array:
    """Return sorted-assignment offsets in expert-group outer, rotating-peer inner order."""
    if local_experts % expert_group_size != 0:
        raise ValueError(
            f"local_experts={local_experts} must be divisible by " f"dispatch_expert_group_size={expert_group_size}"
        )
    assignments = dst_ranks_sorted.shape[0]
    expert_group = local_experts_sorted // expert_group_size
    peer_phase = (dst_ranks_sorted - rank) % ep_size
    key = (
        expert_group * (ep_size * local_experts * assignments)
        + peer_phase * (local_experts * assignments)
        + local_experts_sorted * assignments
        + local_pos_sorted
    )
    return jnp.argsort(key, stable=True).astype(jnp.int32)


def prepare_mgpu_moe_metadata(
    selected_experts_local: Int[Array, "T K"],
    *,
    local_experts: int,
    ep_size: int | None = None,
    expert_axis: str | None = None,
) -> MoeMgpuRoutingMetadata:
    """Build deterministic source-side assignment metadata for the MGPU layout."""
    if selected_experts_local.ndim != 2:
        raise ValueError(f"selected_experts must be rank-2 [T, K], got shape={selected_experts_local.shape}")
    if local_experts <= 0:
        raise ValueError(f"local_experts must be positive, got {local_experts}")
    if ep_size is None:
        if expert_axis is None:
            raise ValueError("expert_axis is required when ep_size is not provided")
        ep_size = int(lax.axis_size(expert_axis))
    if ep_size <= 0:
        raise ValueError(f"ep_size must be positive, got {ep_size}")

    tokens, topk = selected_experts_local.shape
    assignments = tokens * topk
    num_experts = ep_size * local_experts
    assignment_ids = jnp.arange(assignments, dtype=jnp.int32)
    expert_ids_flat = selected_experts_local.reshape(assignments).astype(jnp.int32)

    sort_indices = _stable_assignment_sort(expert_ids_flat)
    assignment_ids_sorted = assignment_ids[sort_indices]
    token_ids_sorted = assignment_ids_sorted // topk
    expert_ids_sorted = expert_ids_flat[sort_indices]
    dst_ranks_sorted = expert_ids_sorted // local_experts
    local_experts_sorted = expert_ids_sorted % local_experts

    send_counts_flat = jnp.bincount(expert_ids_flat, length=num_experts).astype(jnp.int32)
    send_counts = send_counts_flat.reshape(ep_size, local_experts)
    expert_offsets = jnp.cumsum(send_counts_flat, dtype=jnp.int32) - send_counts_flat
    sorted_positions = jnp.arange(assignments, dtype=jnp.int32)
    local_pos_sorted = sorted_positions - expert_offsets[expert_ids_sorted]

    if expert_axis is None:
        global_counts = send_counts[jnp.newaxis, :, :]
    else:
        global_counts = lax.all_gather(send_counts, expert_axis)

    return MoeMgpuRoutingMetadata(
        assignment_ids_sorted=assignment_ids_sorted,
        token_ids_sorted=token_ids_sorted,
        expert_ids_sorted=expert_ids_sorted,
        dst_ranks_sorted=dst_ranks_sorted,
        local_experts_sorted=local_experts_sorted,
        local_pos_sorted=local_pos_sorted,
        send_counts=send_counts,
        global_counts=global_counts,
    )


def _group_sizes_with_padding(group_sizes: jax.Array, *, total_size: int) -> jax.Array:
    padding = total_size - jnp.sum(group_sizes, dtype=jnp.int32)
    return group_sizes.at[-1].add(padding)


def _comm_block_n(hidden_dim: int, preferred_block_n: int, config: MoeMgpuConfig) -> int:
    if hidden_dim % preferred_block_n == 0:
        return preferred_block_n
    return config.block_n


def _max_blocks_touched_by_unaligned_range(max_rows: int, block_m: int) -> int:
    """Maximum `block_m` GMEM blocks touched by a compact row range."""
    if max_rows <= 0:
        raise ValueError(f"max_rows must be positive, got {max_rows}")
    if block_m <= 0:
        raise ValueError(f"block_m must be positive, got {block_m}")
    return int(math.ceil((max_rows + block_m - 1) / block_m))


def _static_workqueue_phase_indices(
    global_phase: jax.Array,
    rank: jax.Array,
    *,
    ep_size: int,
    expert_groups: int,
    dispatch_copy_schedule: str,
    peer_phase_order: jax.Array | None,
    swizzle_expert_groups: bool,
) -> tuple[jax.Array, jax.Array]:
    """Map a static workqueue phase to `(expert_group, peer_phase)`."""
    if dispatch_copy_schedule == _DISPATCH_COPY_EXPERT_GROUP_PEER:
        raw_expert_group = global_phase // ep_size
        raw_peer_phase = global_phase - raw_expert_group * ep_size
    else:
        raw_expert_group = global_phase % expert_groups
        raw_peer_phase = global_phase // expert_groups

    if swizzle_expert_groups:
        expert_group = (raw_expert_group + rank) % expert_groups
    else:
        expert_group = raw_expert_group

    if peer_phase_order is None:
        peer_phase = raw_peer_phase
    else:
        peer_phase = peer_phase_order[raw_peer_phase]

    return expert_group, peer_phase


def _remote_rows_for_sorted_assignments(
    *,
    rank: jax.Array,
    dst_ranks_sorted: jax.Array,
    local_experts_sorted: jax.Array,
    local_pos_sorted: jax.Array,
    clipped_counts: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Compute deterministic remote rows for this source rank's sorted assignments."""
    rows_per_dst_expert = jnp.sum(clipped_counts, axis=0, dtype=jnp.int32)
    expert_base_by_dst = jnp.cumsum(rows_per_dst_expert, axis=1, dtype=jnp.int32) - rows_per_dst_expert
    src_base_by_dst_expert = jnp.cumsum(clipped_counts, axis=0, dtype=jnp.int32) - clipped_counts
    accepted_by_assignment = clipped_counts[rank, dst_ranks_sorted, local_experts_sorted]
    remote_rows = (
        expert_base_by_dst[dst_ranks_sorted, local_experts_sorted]
        + src_base_by_dst_expert[rank, dst_ranks_sorted, local_experts_sorted]
        + local_pos_sorted
    )
    keep = local_pos_sorted < accepted_by_assignment
    return remote_rows.astype(jnp.int32), keep


def _permute_up_tiled_values_with_schedule(
    x_local: jax.Array,
    metadata: MoeMgpuRoutingMetadata,
    remote_rows_sorted: jax.Array,
    keep_sorted: jax.Array,
    *,
    rank: jax.Array,
    capacity: int,
    ep_size: int,
    local_experts: int,
    expert_axis: str,
    config: MoeMgpuConfig,
) -> jax.Array:
    if config.dispatch_copy_schedule != _DISPATCH_COPY_EXPERT_GROUP_PEER:
        return _permute_up_tiled_values_mgpu_kernel(
            x_local,
            metadata.token_ids_sorted,
            metadata.dst_ranks_sorted,
            remote_rows_sorted,
            keep_sorted,
            capacity=capacity,
            ep_size=ep_size,
            expert_axis=expert_axis,
            config=config,
        )

    copy_order = _expert_group_peer_copy_order(
        metadata.dst_ranks_sorted,
        metadata.local_experts_sorted,
        metadata.local_pos_sorted,
        rank=rank,
        ep_size=ep_size,
        local_experts=local_experts,
        expert_group_size=config.dispatch_expert_group_size,
    )
    return _permute_up_tiled_values_scheduled_mgpu_kernel(
        x_local,
        metadata.token_ids_sorted,
        metadata.dst_ranks_sorted,
        remote_rows_sorted,
        keep_sorted,
        copy_order,
        capacity=capacity,
        ep_size=ep_size,
        expert_axis=expert_axis,
        config=config,
    )


def _copy_order_for_schedule(
    metadata: MoeMgpuRoutingMetadata,
    *,
    rank: jax.Array,
    ep_size: int,
    local_experts: int,
    config: MoeMgpuConfig,
) -> jax.Array:
    if config.dispatch_copy_schedule != _DISPATCH_COPY_EXPERT_GROUP_PEER:
        return jnp.arange(metadata.token_ids_sorted.shape[0], dtype=jnp.int32)
    return _expert_group_peer_copy_order(
        metadata.dst_ranks_sorted,
        metadata.local_experts_sorted,
        metadata.local_pos_sorted,
        rank=rank,
        ep_size=ep_size,
        local_experts=local_experts,
        expert_group_size=config.dispatch_expert_group_size,
    )


def _permute_up_tiled_metadata_values_with_schedule(
    x_local: jax.Array,
    metadata: MoeMgpuRoutingMetadata,
    remote_rows_sorted: jax.Array,
    keep_sorted: jax.Array,
    *,
    rank: jax.Array,
    capacity: int,
    ep_size: int,
    local_experts: int,
    expert_axis: str,
    config: MoeMgpuConfig,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    copy_order = _copy_order_for_schedule(
        metadata,
        rank=rank,
        ep_size=ep_size,
        local_experts=local_experts,
        config=config,
    )
    return _permute_up_tiled_metadata_values_mgpu_kernel(
        x_local,
        metadata.assignment_ids_sorted,
        metadata.token_ids_sorted,
        metadata.dst_ranks_sorted,
        remote_rows_sorted,
        keep_sorted,
        copy_order,
        capacity=capacity,
        ep_size=ep_size,
        expert_axis=expert_axis,
        config=config,
    )


def permute_mgpu_reference(
    x_local: Float[Array, "Tlocal D"],
    selected_experts_local: Int[Array, "Tlocal K"],
    *,
    local_experts: int,
    expert_axis: str = "expert",
    config: MoeMgpuConfig = MoeMgpuConfig(),
) -> MoeMgpuReceiveLayout:
    """Reference expert-major receive layout for the future permute MGPU kernel."""
    if x_local.ndim != 2:
        raise ValueError(f"x_local must be rank-2 [T, D], got shape={x_local.shape}")
    if selected_experts_local.ndim != 2:
        raise ValueError(f"selected_experts_local must be rank-2 [T, K], got shape={selected_experts_local.shape}")
    if selected_experts_local.shape[0] != x_local.shape[0]:
        raise ValueError("selected_experts_local token dimension must match x_local")

    ep_size = int(lax.axis_size(expert_axis))
    rank = lax.axis_index(expert_axis)
    tokens, topk = selected_experts_local.shape
    assignments = tokens * topk
    capacity = _receiver_capacity(tokens, topk, local_experts, config.capacity_factor)

    metadata = prepare_mgpu_moe_metadata(
        selected_experts_local,
        local_experts=local_experts,
        ep_size=ep_size,
        expert_axis=expert_axis,
    )
    global_counts_flat = metadata.global_counts.reshape(ep_size, ep_size * local_experts)
    clipped_counts = _clip_receiver_group_sizes(
        global_counts_flat,
        local_expert_size=local_experts,
        receiver_capacity=capacity,
    ).reshape(ep_size, ep_size, local_experts)

    rows_per_expert = jnp.sum(clipped_counts[:, rank, :], axis=0, dtype=jnp.int32)
    expert_base = jnp.cumsum(rows_per_expert, dtype=jnp.int32) - rows_per_expert
    src_base_by_expert = jnp.cumsum(clipped_counts[:, rank, :], axis=0, dtype=jnp.int32) - clipped_counts[:, rank, :]

    x_by_src = lax.all_gather(x_local, expert_axis)
    assignment_ids_by_src = lax.all_gather(metadata.assignment_ids_sorted, expert_axis)
    token_ids_by_src = lax.all_gather(metadata.token_ids_sorted, expert_axis)
    dst_ranks_by_src = lax.all_gather(metadata.dst_ranks_sorted, expert_axis)
    local_experts_by_src = lax.all_gather(metadata.local_experts_sorted, expert_axis)
    local_pos_by_src = lax.all_gather(metadata.local_pos_sorted, expert_axis)

    recv_x = jnp.zeros((capacity, x_local.shape[1]), dtype=x_local.dtype)
    recv_src_rank = jnp.full((capacity,), -1, dtype=jnp.int32)
    recv_src_assignment = jnp.full((capacity,), -1, dtype=jnp.int32)
    source_positions = jnp.arange(assignments, dtype=jnp.int32)

    for src in range(ep_size):
        source_x = x_by_src[src]
        token_ids = token_ids_by_src[src]
        local_expert = local_experts_by_src[src]
        local_pos = local_pos_by_src[src]
        accepted_for_expert = clipped_counts[src, rank, local_expert]
        keep = (source_positions < assignments) & (dst_ranks_by_src[src] == rank) & (local_pos < accepted_for_expert)
        remote_row = expert_base[local_expert] + src_base_by_expert[src, local_expert] + local_pos
        remote_row = jnp.where(keep, remote_row, capacity)
        recv_x = recv_x.at[remote_row].set(source_x[token_ids], mode="drop")
        recv_src_rank = recv_src_rank.at[remote_row].set(jnp.int32(src), mode="drop")
        recv_src_assignment = recv_src_assignment.at[remote_row].set(assignment_ids_by_src[src], mode="drop")

    dropped = jnp.sum(metadata.global_counts, dtype=jnp.int32) - jnp.sum(clipped_counts, dtype=jnp.int32)
    return MoeMgpuReceiveLayout(
        recv_x=recv_x,
        recv_src_rank=recv_src_rank,
        recv_src_assignment=recv_src_assignment,
        rows_per_expert=rows_per_expert,
        clipped_global_counts=clipped_counts,
        dropped=dropped,
    )


def permute_mgpu(
    x_local: Float[Array, "Tlocal D"],
    selected_experts_local: Int[Array, "Tlocal K"],
    *,
    local_experts: int,
    expert_axis: str = "expert",
    config: MoeMgpuConfig = MoeMgpuConfig(),
) -> MoeMgpuReceiveLayout:
    """Pallas MGPU remote-write permute kernel."""
    if x_local.ndim != 2:
        raise ValueError(f"x_local must be rank-2 [T, D], got shape={x_local.shape}")
    if selected_experts_local.ndim != 2:
        raise ValueError(f"selected_experts_local must be rank-2 [T, K], got shape={selected_experts_local.shape}")
    if selected_experts_local.shape[0] != x_local.shape[0]:
        raise ValueError("selected_experts_local token dimension must match x_local")

    ep_size = int(lax.axis_size(expert_axis))
    rank = lax.axis_index(expert_axis)
    tokens, topk = selected_experts_local.shape
    capacity = _receiver_capacity(tokens, topk, local_experts, config.capacity_factor)
    metadata = prepare_mgpu_moe_metadata(
        selected_experts_local,
        local_experts=local_experts,
        ep_size=ep_size,
        expert_axis=expert_axis,
    )
    global_counts_flat = metadata.global_counts.reshape(ep_size, ep_size * local_experts)
    clipped_counts = _clip_receiver_group_sizes(
        global_counts_flat,
        local_expert_size=local_experts,
        receiver_capacity=capacity,
    ).reshape(ep_size, ep_size, local_experts)
    rows_per_expert = jnp.sum(clipped_counts[:, rank, :], axis=0, dtype=jnp.int32)
    dropped = jnp.sum(metadata.global_counts, dtype=jnp.int32) - jnp.sum(clipped_counts, dtype=jnp.int32)
    remote_rows_sorted, keep_sorted = _remote_rows_for_sorted_assignments(
        rank=rank,
        dst_ranks_sorted=metadata.dst_ranks_sorted,
        local_experts_sorted=metadata.local_experts_sorted,
        local_pos_sorted=metadata.local_pos_sorted,
        clipped_counts=clipped_counts,
    )

    if config.dispatch_fuse_metadata:
        recv_x, recv_src_rank, recv_src_assignment = _permute_up_tiled_metadata_values_with_schedule(
            x_local,
            metadata,
            remote_rows_sorted,
            keep_sorted,
            rank=rank,
            capacity=capacity,
            ep_size=ep_size,
            local_experts=local_experts,
            expert_axis=expert_axis,
            config=config,
        )
    else:
        recv_src_rank, recv_src_assignment = _permute_up_tiled_metadata_mgpu_kernel(
            metadata.assignment_ids_sorted,
            metadata.dst_ranks_sorted,
            remote_rows_sorted,
            keep_sorted,
            capacity=capacity,
            ep_size=ep_size,
            expert_axis=expert_axis,
            config=config,
        )
        recv_x = _permute_up_tiled_values_with_schedule(
            x_local,
            metadata,
            remote_rows_sorted,
            keep_sorted,
            rank=rank,
            capacity=capacity,
            ep_size=ep_size,
            local_experts=local_experts,
            expert_axis=expert_axis,
            config=config,
        )
    return MoeMgpuReceiveLayout(
        recv_x=recv_x,
        recv_src_rank=recv_src_rank,
        recv_src_assignment=recv_src_assignment,
        rows_per_expert=rows_per_expert,
        clipped_global_counts=clipped_counts,
        dropped=dropped,
    )


def _permute_mgpu_kernel(
    x_local: jax.Array,
    assignment_ids_sorted: jax.Array,
    token_ids_sorted: jax.Array,
    dst_ranks_sorted: jax.Array,
    remote_rows_sorted: jax.Array,
    keep_sorted: jax.Array,
    *,
    capacity: int,
    ep_size: int,
    expert_axis: str,
    config: MoeMgpuConfig,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    tokens, hidden = x_local.shape
    assignments = assignment_ids_sorted.shape[0]
    num_sms = config.num_sms
    if num_sms is None:
        num_sms = jax.devices()[0].core_count
    init_steps = int(math.ceil(capacity / num_sms))
    assignment_steps = int(math.ceil(assignments / num_sms))

    def body(
        x_ref,
        assignment_ids_ref,
        token_ids_ref,
        dst_ranks_ref,
        remote_rows_ref,
        keep_ref,
        recv_x_ref,
        recv_src_rank_ref,
        recv_src_assignment_ref,
    ):
        arrivals_sem = pl.get_global(mgpu.SemaphoreType.REGULAR)
        rank = lax.axis_index(expert_axis)
        sm_id = lax.axis_index("sm")

        @pl.loop(0, init_steps)
        def _init_step(step):
            row = step * num_sms + sm_id

            @pl.when(row < capacity)
            def _init_row():
                recv_src_rank_ref[row] = jnp.int32(-1)
                recv_src_assignment_ref[row] = jnp.int32(-1)

                @pl.loop(0, hidden)
                def _init_col(col):
                    recv_x_ref[row, col] = jnp.zeros_like(x_ref[0, col])

        @pl.loop(0, ep_size)
        def _signal_init(peer):
            pl.semaphore_signal(arrivals_sem, device_id=peer, device_id_type=pl.DeviceIdType.LOGICAL)

        pl.semaphore_wait(arrivals_sem, value=ep_size * num_sms, decrement=False)

        @pl.loop(0, assignment_steps)
        def _send_step(step):
            offset = step * num_sms + sm_id

            @pl.when(offset < assignments)
            def _send_assignment():
                dst = dst_ranks_ref[offset]
                should_send = keep_ref[offset]
                remote_row = remote_rows_ref[offset]
                remote_recv_x_ref = mgpu.remote_ref(recv_x_ref, dst, device_id_type=pl.DeviceIdType.LOGICAL)
                remote_src_rank_ref = mgpu.remote_ref(recv_src_rank_ref, dst, device_id_type=pl.DeviceIdType.LOGICAL)
                remote_src_assignment_ref = mgpu.remote_ref(
                    recv_src_assignment_ref, dst, device_id_type=pl.DeviceIdType.LOGICAL
                )

                @pl.when(should_send)
                def _write_remote():
                    remote_src_rank_ref[remote_row] = rank
                    remote_src_assignment_ref[remote_row] = assignment_ids_ref[offset]
                    token_id = token_ids_ref[offset]

                    @pl.loop(0, hidden)
                    def _copy_col(col):
                        remote_recv_x_ref[remote_row, col] = x_ref[token_id, col]

        @pl.loop(0, ep_size)
        def _signal_done(peer):
            pl.semaphore_signal(arrivals_sem, device_id=peer, device_id_type=pl.DeviceIdType.LOGICAL)

        pl.semaphore_wait(arrivals_sem, value=2 * ep_size * num_sms, decrement=False)

    kernel = mgpu.kernel(
        body,
        out_shape=[
            jax.ShapeDtypeStruct((capacity, hidden), x_local.dtype),
            jax.ShapeDtypeStruct((capacity,), jnp.int32),
            jax.ShapeDtypeStruct((capacity,), jnp.int32),
        ],
        grid=(num_sms,),
        grid_names=("sm",),
        compiler_params=mgpu.CompilerParams(
            lowering_semantics=mgpu.LoweringSemantics.Warpgroup,
        ),
    )
    del tokens
    return kernel(
        x_local,
        assignment_ids_sorted,
        token_ids_sorted,
        dst_ranks_sorted,
        remote_rows_sorted,
        keep_sorted,
    )


def _fused_w13_reference(
    recv_x: Float[Array, "capacity D"],
    moe_w13: Float[Array, "Elocal D I2"],
    rows_per_expert: Int[Array, "Elocal"],
    *,
    activation_fn: Callable[[jax.Array], jax.Array],
) -> Float[Array, "capacity I"]:
    """Reference with the fused W13/SwiGLU semantics used by the Pallas up kernel."""
    intermediate = moe_w13.shape[-1] // 2
    compute_group_sizes = _group_sizes_with_padding(rows_per_expert, total_size=recv_x.shape[0])
    gate = ragged_dot(recv_x, moe_w13[..., :intermediate], compute_group_sizes)
    up = ragged_dot(recv_x, moe_w13[..., intermediate:], compute_group_sizes)
    return (activation_fn(gate) * up).astype(recv_x.dtype)


def permute_up_mgpu_reference(
    x_local: Float[Array, "Tlocal D"],
    selected_experts_local: Int[Array, "Tlocal K"],
    moe_w13: Float[Array, "Elocal D I2"],
    *,
    local_experts: int,
    activation_fn: Callable[[jax.Array], jax.Array],
    expert_axis: str = "expert",
    config: MoeMgpuConfig = MoeMgpuConfig(),
) -> MoeMgpuUpLayout:
    """Reference for the staged permute-up MGPU boundary."""
    layout = permute_mgpu_reference(
        x_local,
        selected_experts_local,
        local_experts=local_experts,
        expert_axis=expert_axis,
        config=config,
    )
    hidden = _fused_w13_reference(
        layout.recv_x,
        moe_w13,
        layout.rows_per_expert,
        activation_fn=activation_fn,
    )
    return MoeMgpuUpLayout(
        hidden=hidden,
        recv_src_rank=layout.recv_src_rank,
        recv_src_assignment=layout.recv_src_assignment,
        rows_per_expert=layout.rows_per_expert,
        clipped_global_counts=layout.clipped_global_counts,
        dropped=layout.dropped,
    )


def permute_up_mgpu(
    x_local: Float[Array, "Tlocal D"],
    selected_experts_local: Int[Array, "Tlocal K"],
    moe_w13: Float[Array, "Elocal D I2"],
    *,
    local_experts: int,
    activation_fn: Callable[[jax.Array], jax.Array],
    expert_axis: str = "expert",
    config: MoeMgpuConfig = MoeMgpuConfig(),
) -> MoeMgpuUpLayout:
    """Fused MGPU permute plus local W13/SwiGLU over the expert-major receive layout."""
    if x_local.ndim != 2:
        raise ValueError(f"x_local must be rank-2 [T, D], got shape={x_local.shape}")
    if selected_experts_local.ndim != 2:
        raise ValueError(f"selected_experts_local must be rank-2 [T, K], got shape={selected_experts_local.shape}")
    if selected_experts_local.shape[0] != x_local.shape[0]:
        raise ValueError("selected_experts_local token dimension must match x_local")

    ep_size = int(lax.axis_size(expert_axis))
    rank = lax.axis_index(expert_axis)
    tokens, topk = selected_experts_local.shape
    capacity = _receiver_capacity(tokens, topk, local_experts, config.capacity_factor)
    metadata = prepare_mgpu_moe_metadata(
        selected_experts_local,
        local_experts=local_experts,
        ep_size=ep_size,
        expert_axis=expert_axis,
    )
    global_counts_flat = metadata.global_counts.reshape(ep_size, ep_size * local_experts)
    clipped_counts = _clip_receiver_group_sizes(
        global_counts_flat,
        local_expert_size=local_experts,
        receiver_capacity=capacity,
    ).reshape(ep_size, ep_size, local_experts)
    rows_per_expert = jnp.sum(clipped_counts[:, rank, :], axis=0, dtype=jnp.int32)
    compute_group_sizes = _group_sizes_with_padding(rows_per_expert, total_size=capacity)
    dropped = jnp.sum(metadata.global_counts, dtype=jnp.int32) - jnp.sum(clipped_counts, dtype=jnp.int32)
    remote_rows_sorted, keep_sorted = _remote_rows_for_sorted_assignments(
        rank=rank,
        dst_ranks_sorted=metadata.dst_ranks_sorted,
        local_experts_sorted=metadata.local_experts_sorted,
        local_pos_sorted=metadata.local_pos_sorted,
        clipped_counts=clipped_counts,
    )

    if config.dispatch_user_kernel_permute_up:
        hidden, recv_src_rank, recv_src_assignment = _permute_up_mgpu_user_kernel(
            x_local,
            metadata,
            remote_rows_sorted,
            keep_sorted,
            clipped_counts,
            rows_per_expert,
            moe_w13,
            activation_fn,
            rank=rank,
            capacity=capacity,
            ep_size=ep_size,
            local_experts=local_experts,
            expert_axis=expert_axis,
            config=config,
        )
        return MoeMgpuUpLayout(
            hidden=hidden,
            recv_src_rank=recv_src_rank,
            recv_src_assignment=recv_src_assignment,
            rows_per_expert=rows_per_expert,
            clipped_global_counts=clipped_counts,
            dropped=dropped,
        )

    if config.dispatch_static_workqueue_permute_up:
        hidden, recv_src_rank, recv_src_assignment = _permute_up_mgpu_static_workqueue_kernel(
            x_local,
            metadata,
            clipped_counts,
            rows_per_expert,
            moe_w13,
            activation_fn,
            rank=rank,
            capacity=capacity,
            ep_size=ep_size,
            local_experts=local_experts,
            expert_axis=expert_axis,
            config=config,
        )
        hidden, recv_src_rank, recv_src_assignment, rows_per_expert = lax.optimization_barrier(
            (hidden, recv_src_rank, recv_src_assignment, rows_per_expert)
        )
        return MoeMgpuUpLayout(
            hidden=hidden,
            recv_src_rank=recv_src_rank,
            recv_src_assignment=recv_src_assignment,
            rows_per_expert=rows_per_expert,
            clipped_global_counts=clipped_counts,
            dropped=dropped,
        )

    if config.dispatch_chunked_permute_up:
        hidden, recv_src_rank, recv_src_assignment = _permute_up_mgpu_fused_chunked_kernel(
            x_local,
            metadata.assignment_ids_sorted,
            metadata.token_ids_sorted,
            metadata.send_counts,
            clipped_counts,
            moe_w13,
            activation_fn,
            capacity=capacity,
            ep_size=ep_size,
            local_experts=local_experts,
            expert_axis=expert_axis,
            config=config,
        )
        return MoeMgpuUpLayout(
            hidden=hidden,
            recv_src_rank=recv_src_rank,
            recv_src_assignment=recv_src_assignment,
            rows_per_expert=rows_per_expert,
            clipped_global_counts=clipped_counts,
            dropped=dropped,
        )

    if config.dispatch_fuse_metadata:
        recv_x, recv_src_rank, recv_src_assignment = _permute_up_tiled_metadata_values_with_schedule(
            x_local,
            metadata,
            remote_rows_sorted,
            keep_sorted,
            rank=rank,
            capacity=capacity,
            ep_size=ep_size,
            local_experts=local_experts,
            expert_axis=expert_axis,
            config=config,
        )
    else:
        recv_src_rank, recv_src_assignment = _permute_up_tiled_metadata_mgpu_kernel(
            metadata.assignment_ids_sorted,
            metadata.dst_ranks_sorted,
            remote_rows_sorted,
            keep_sorted,
            capacity=capacity,
            ep_size=ep_size,
            expert_axis=expert_axis,
            config=config,
        )
        recv_x = _permute_up_tiled_values_with_schedule(
            x_local,
            metadata,
            remote_rows_sorted,
            keep_sorted,
            rank=rank,
            capacity=capacity,
            ep_size=ep_size,
            local_experts=local_experts,
            expert_axis=expert_axis,
            config=config,
        )
    hidden = _moe_mgpu_dispatch_w13_activation(
        recv_x,
        moe_w13,
        activation_fn,
        _MoeMgpuUpMetadata(global_expert_counts=compute_group_sizes[jnp.newaxis, :]),
        config,
    )
    return MoeMgpuUpLayout(
        hidden=hidden,
        recv_src_rank=recv_src_rank,
        recv_src_assignment=recv_src_assignment,
        rows_per_expert=rows_per_expert,
        clipped_global_counts=clipped_counts,
        dropped=dropped,
    )


def _permute_up_tiled_metadata_mgpu_kernel(
    assignment_ids_sorted: jax.Array,
    dst_ranks_sorted: jax.Array,
    remote_rows_sorted: jax.Array,
    keep_sorted: jax.Array,
    *,
    capacity: int,
    ep_size: int,
    expert_axis: str,
    config: MoeMgpuConfig,
) -> tuple[jax.Array, jax.Array]:
    assignments = assignment_ids_sorted.shape[0]
    num_sms = config.num_sms
    if num_sms is None:
        num_sms = jax.devices()[0].core_count
    num_threads = 3
    workers = num_sms * num_threads
    init_steps = int(math.ceil(capacity / workers))
    metadata_steps = int(math.ceil(assignments / workers))

    def body(assignment_ids_ref, dst_ranks_ref, remote_rows_ref, keep_ref, recv_src_rank_ref, recv_assignment_ref):
        arrivals_sem = pl.get_global(mgpu.SemaphoreType.REGULAR)
        rank = lax.axis_index(expert_axis)
        sm_id = lax.axis_index("sm")
        wg_id = lax.axis_index("wg")
        worker_id = sm_id * num_threads + wg_id

        @pl.loop(0, init_steps)
        def _init_step(step):
            row = step * workers + worker_id

            @pl.when(row < capacity)
            def _init_row():
                recv_src_rank_ref[row] = jnp.int32(-1)
                recv_assignment_ref[row] = jnp.int32(-1)

        @pl.loop(0, ep_size)
        def _signal_init(peer):
            pl.semaphore_signal(arrivals_sem, device_id=peer, device_id_type=pl.DeviceIdType.LOGICAL)

        pl.semaphore_wait(arrivals_sem, value=ep_size * workers, decrement=False)

        @pl.loop(0, metadata_steps)
        def _metadata_step(step):
            offset = step * workers + worker_id

            @pl.when((offset < assignments) & keep_ref[offset])
            def _write_metadata():
                dst = dst_ranks_ref[offset]
                remote_row = remote_rows_ref[offset]
                remote_src_rank_ref = mgpu.remote_ref(recv_src_rank_ref, dst, device_id_type=pl.DeviceIdType.LOGICAL)
                remote_assignment_ref = mgpu.remote_ref(
                    recv_assignment_ref, dst, device_id_type=pl.DeviceIdType.LOGICAL
                )
                remote_src_rank_ref[remote_row] = rank
                remote_assignment_ref[remote_row] = assignment_ids_ref[offset]

        @pl.loop(0, ep_size)
        def _signal_done(peer):
            pl.semaphore_signal(arrivals_sem, device_id=peer, device_id_type=pl.DeviceIdType.LOGICAL)

        pl.semaphore_wait(arrivals_sem, value=2 * ep_size * workers, decrement=False)

    kernel = mgpu.kernel(
        body,
        out_shape=[
            jax.ShapeDtypeStruct((capacity,), jnp.int32),
            jax.ShapeDtypeStruct((capacity,), jnp.int32),
        ],
        grid=(num_sms,),
        grid_names=("sm",),
        num_threads=num_threads,
        thread_name="wg",
        compiler_params=mgpu.CompilerParams(
            lowering_semantics=mgpu.LoweringSemantics.Warpgroup,
        ),
    )
    return kernel(assignment_ids_sorted, dst_ranks_sorted, remote_rows_sorted, keep_sorted)


def _permute_up_tiled_values_mgpu_kernel(
    x_local: jax.Array,
    token_ids_sorted: jax.Array,
    dst_ranks_sorted: jax.Array,
    remote_rows_sorted: jax.Array,
    keep_sorted: jax.Array,
    *,
    capacity: int,
    ep_size: int,
    expert_axis: str,
    config: MoeMgpuConfig,
) -> jax.Array:
    _tokens, hidden_dim = x_local.shape
    assignments = token_ids_sorted.shape[0]
    copy_tile = config.dispatch_chunk_copy_tile
    if hidden_dim % copy_tile != 0:
        raise ValueError(f"D={hidden_dim} must be divisible by dispatch copy tile={copy_tile}")
    d_tiles = hidden_dim // copy_tile
    num_sms = config.num_sms
    if num_sms is None:
        num_sms = jax.devices()[0].core_count
    workers = num_sms
    init_tiles = capacity * d_tiles
    dispatch_tiles = assignments * d_tiles
    init_steps = int(math.ceil(init_tiles / workers))
    dispatch_steps = int(math.ceil(dispatch_tiles / workers))

    def body(x_ref, token_ids_ref, dst_ranks_ref, remote_rows_ref, keep_ref, recv_x_ref):
        arrivals_sem = pl.get_global(mgpu.SemaphoreType.REGULAR)
        sm_id = lax.axis_index("sm")
        worker_id = sm_id

        @pl.loop(0, init_steps)
        def _init_step(step):
            linear_tile = step * workers + worker_id

            @pl.when(linear_tile < init_tiles)
            def _zero_tile():
                row = linear_tile // d_tiles
                d_tile = linear_tile - row * d_tiles
                d_start = d_tile * copy_tile
                recv_x_ref[row, pl.ds(d_start, copy_tile)] = jnp.zeros((copy_tile,), dtype=x_ref.dtype)

        @pl.loop(0, ep_size)
        def _signal_init(peer):
            pl.semaphore_signal(arrivals_sem, device_id=peer, device_id_type=pl.DeviceIdType.LOGICAL)

        pl.semaphore_wait(arrivals_sem, value=ep_size * workers, decrement=False)

        @pl.loop(0, dispatch_steps)
        def _send_tile_step(step):
            linear_tile = step * workers + worker_id

            @pl.when(linear_tile < dispatch_tiles)
            def _send_tile():
                offset = linear_tile // d_tiles

                @pl.when(keep_ref[offset])
                def _copy_remote_tile():
                    dst = dst_ranks_ref[offset]
                    remote_row = remote_rows_ref[offset]
                    d_tile = linear_tile - offset * d_tiles
                    d_start = d_tile * copy_tile
                    token_id = token_ids_ref[offset]
                    remote_recv_x_ref = mgpu.remote_ref(recv_x_ref, dst, device_id_type=pl.DeviceIdType.LOGICAL)
                    remote_recv_x_ref[remote_row, pl.ds(d_start, copy_tile)] = x_ref[
                        token_id, pl.ds(d_start, copy_tile)
                    ]

        @pl.loop(0, ep_size)
        def _signal_done(peer):
            pl.semaphore_signal(arrivals_sem, device_id=peer, device_id_type=pl.DeviceIdType.LOGICAL)

        pl.semaphore_wait(arrivals_sem, value=2 * ep_size * workers, decrement=False)

    kernel = mgpu.kernel(
        body,
        out_shape=jax.ShapeDtypeStruct((capacity, hidden_dim), x_local.dtype),
        grid=(num_sms,),
        grid_names=("sm",),
        compiler_params=mgpu.CompilerParams(
            lowering_semantics=mgpu.LoweringSemantics.Lane,
        ),
    )
    return kernel(x_local, token_ids_sorted, dst_ranks_sorted, remote_rows_sorted, keep_sorted)


def _permute_up_tiled_values_scheduled_mgpu_kernel(
    x_local: jax.Array,
    token_ids_sorted: jax.Array,
    dst_ranks_sorted: jax.Array,
    remote_rows_sorted: jax.Array,
    keep_sorted: jax.Array,
    copy_order: jax.Array,
    *,
    capacity: int,
    ep_size: int,
    expert_axis: str,
    config: MoeMgpuConfig,
) -> jax.Array:
    _tokens, hidden_dim = x_local.shape
    assignments = token_ids_sorted.shape[0]
    copy_tile = config.dispatch_chunk_copy_tile
    if hidden_dim % copy_tile != 0:
        raise ValueError(f"D={hidden_dim} must be divisible by dispatch copy tile={copy_tile}")
    d_tiles = hidden_dim // copy_tile
    num_sms = config.num_sms
    if num_sms is None:
        num_sms = jax.devices()[0].core_count
    workers = num_sms
    init_tiles = capacity * d_tiles
    dispatch_tiles = assignments * d_tiles
    init_steps = int(math.ceil(init_tiles / workers))
    dispatch_steps = int(math.ceil(dispatch_tiles / workers))

    def body(x_ref, token_ids_ref, dst_ranks_ref, remote_rows_ref, keep_ref, copy_order_ref, recv_x_ref):
        arrivals_sem = pl.get_global(mgpu.SemaphoreType.REGULAR)
        sm_id = lax.axis_index("sm")
        worker_id = sm_id

        @pl.loop(0, init_steps)
        def _init_step(step):
            linear_tile = step * workers + worker_id

            @pl.when(linear_tile < init_tiles)
            def _zero_tile():
                row = linear_tile // d_tiles
                d_tile = linear_tile - row * d_tiles
                d_start = d_tile * copy_tile
                recv_x_ref[row, pl.ds(d_start, copy_tile)] = jnp.zeros((copy_tile,), dtype=x_ref.dtype)

        @pl.loop(0, ep_size)
        def _signal_init(peer):
            pl.semaphore_signal(arrivals_sem, device_id=peer, device_id_type=pl.DeviceIdType.LOGICAL)

        pl.semaphore_wait(arrivals_sem, value=ep_size * workers, decrement=False)

        @pl.loop(0, dispatch_steps)
        def _send_tile_step(step):
            linear_tile = step * workers + worker_id

            @pl.when(linear_tile < dispatch_tiles)
            def _send_tile():
                schedule_pos = linear_tile // d_tiles
                offset = copy_order_ref[schedule_pos]

                @pl.when(keep_ref[offset])
                def _copy_remote_tile():
                    dst = dst_ranks_ref[offset]
                    remote_row = remote_rows_ref[offset]
                    d_tile = linear_tile - schedule_pos * d_tiles
                    d_start = d_tile * copy_tile
                    token_id = token_ids_ref[offset]
                    remote_recv_x_ref = mgpu.remote_ref(recv_x_ref, dst, device_id_type=pl.DeviceIdType.LOGICAL)
                    remote_recv_x_ref[remote_row, pl.ds(d_start, copy_tile)] = x_ref[
                        token_id, pl.ds(d_start, copy_tile)
                    ]

        @pl.loop(0, ep_size)
        def _signal_done(peer):
            pl.semaphore_signal(arrivals_sem, device_id=peer, device_id_type=pl.DeviceIdType.LOGICAL)

        pl.semaphore_wait(arrivals_sem, value=2 * ep_size * workers, decrement=False)

    kernel = mgpu.kernel(
        body,
        out_shape=jax.ShapeDtypeStruct((capacity, hidden_dim), x_local.dtype),
        grid=(num_sms,),
        grid_names=("sm",),
        compiler_params=mgpu.CompilerParams(
            lowering_semantics=mgpu.LoweringSemantics.Lane,
        ),
    )
    return kernel(x_local, token_ids_sorted, dst_ranks_sorted, remote_rows_sorted, keep_sorted, copy_order)


def _permute_up_tiled_metadata_values_mgpu_kernel(
    x_local: jax.Array,
    assignment_ids_sorted: jax.Array,
    token_ids_sorted: jax.Array,
    dst_ranks_sorted: jax.Array,
    remote_rows_sorted: jax.Array,
    keep_sorted: jax.Array,
    copy_order: jax.Array,
    *,
    capacity: int,
    ep_size: int,
    expert_axis: str,
    config: MoeMgpuConfig,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    _tokens, hidden_dim = x_local.shape
    assignments = token_ids_sorted.shape[0]
    copy_tile = config.dispatch_chunk_copy_tile
    if hidden_dim % copy_tile != 0:
        raise ValueError(f"D={hidden_dim} must be divisible by dispatch copy tile={copy_tile}")
    d_tiles = hidden_dim // copy_tile
    num_sms = config.num_sms
    if num_sms is None:
        num_sms = jax.devices()[0].core_count
    workers = num_sms
    init_value_tiles = capacity * d_tiles
    dispatch_tiles = assignments * d_tiles
    init_value_steps = int(math.ceil(init_value_tiles / workers))
    init_metadata_steps = int(math.ceil(capacity / workers))
    dispatch_steps = int(math.ceil(dispatch_tiles / workers))

    def body(
        x_ref,
        assignment_ids_ref,
        token_ids_ref,
        dst_ranks_ref,
        remote_rows_ref,
        keep_ref,
        copy_order_ref,
        recv_x_ref,
        recv_src_rank_ref,
        recv_assignment_ref,
    ):
        arrivals_sem = pl.get_global(mgpu.SemaphoreType.REGULAR)
        rank = lax.axis_index(expert_axis)
        sm_id = lax.axis_index("sm")
        worker_id = sm_id

        @pl.loop(0, init_value_steps)
        def _init_value_step(step):
            linear_tile = step * workers + worker_id

            @pl.when(linear_tile < init_value_tiles)
            def _zero_tile():
                row = linear_tile // d_tiles
                d_tile = linear_tile - row * d_tiles
                d_start = d_tile * copy_tile
                recv_x_ref[row, pl.ds(d_start, copy_tile)] = jnp.zeros((copy_tile,), dtype=x_ref.dtype)

        @pl.loop(0, init_metadata_steps)
        def _init_metadata_step(step):
            row = step * workers + worker_id

            @pl.when(row < capacity)
            def _init_row():
                recv_src_rank_ref[row] = jnp.int32(-1)
                recv_assignment_ref[row] = jnp.int32(-1)

        @pl.loop(0, ep_size)
        def _signal_init(peer):
            pl.semaphore_signal(arrivals_sem, device_id=peer, device_id_type=pl.DeviceIdType.LOGICAL)

        pl.semaphore_wait(arrivals_sem, value=ep_size * workers, decrement=False)

        @pl.loop(0, dispatch_steps)
        def _send_tile_step(step):
            linear_tile = step * workers + worker_id

            @pl.when(linear_tile < dispatch_tiles)
            def _send_tile():
                schedule_pos = linear_tile // d_tiles
                offset = copy_order_ref[schedule_pos]

                @pl.when(keep_ref[offset])
                def _copy_remote_tile():
                    dst = dst_ranks_ref[offset]
                    remote_row = remote_rows_ref[offset]
                    d_tile = linear_tile - schedule_pos * d_tiles
                    d_start = d_tile * copy_tile
                    token_id = token_ids_ref[offset]
                    remote_recv_x_ref = mgpu.remote_ref(recv_x_ref, dst, device_id_type=pl.DeviceIdType.LOGICAL)
                    remote_recv_x_ref[remote_row, pl.ds(d_start, copy_tile)] = x_ref[
                        token_id, pl.ds(d_start, copy_tile)
                    ]

                    @pl.when(d_tile == 0)
                    def _write_metadata():
                        remote_src_rank_ref = mgpu.remote_ref(
                            recv_src_rank_ref,
                            dst,
                            device_id_type=pl.DeviceIdType.LOGICAL,
                        )
                        remote_assignment_ref = mgpu.remote_ref(
                            recv_assignment_ref,
                            dst,
                            device_id_type=pl.DeviceIdType.LOGICAL,
                        )
                        remote_src_rank_ref[remote_row] = rank
                        remote_assignment_ref[remote_row] = assignment_ids_ref[offset]

        @pl.loop(0, ep_size)
        def _signal_done(peer):
            pl.semaphore_signal(arrivals_sem, device_id=peer, device_id_type=pl.DeviceIdType.LOGICAL)

        pl.semaphore_wait(arrivals_sem, value=2 * ep_size * workers, decrement=False)

    kernel = mgpu.kernel(
        body,
        out_shape=[
            jax.ShapeDtypeStruct((capacity, hidden_dim), x_local.dtype),
            jax.ShapeDtypeStruct((capacity,), jnp.int32),
            jax.ShapeDtypeStruct((capacity,), jnp.int32),
        ],
        grid=(num_sms,),
        grid_names=("sm",),
        compiler_params=mgpu.CompilerParams(
            lowering_semantics=mgpu.LoweringSemantics.Lane,
        ),
    )
    return kernel(
        x_local,
        assignment_ids_sorted,
        token_ids_sorted,
        dst_ranks_sorted,
        remote_rows_sorted,
        keep_sorted,
        copy_order,
    )


def _permute_up_mgpu_user_kernel(
    x_local: jax.Array,
    metadata: MoeMgpuRoutingMetadata,
    remote_rows_sorted: jax.Array,
    keep_sorted: jax.Array,
    clipped_counts: jax.Array,
    rows_per_expert: jax.Array,
    moe_w13: jax.Array,
    activation_fn: Callable[[jax.Array], jax.Array],
    *,
    rank: jax.Array,
    capacity: int,
    ep_size: int,
    local_experts: int,
    expert_axis: str,
    config: MoeMgpuConfig,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Edit point for an experimental fused MGPU permute_up kernel.

    This hook is selected with `MoeMgpuConfig(dispatch_user_kernel_permute_up=True)`
    or benchmark flag `--dispatch-user-kernel-permute-up`.
    Inputs are already normalized to the deterministic expert-major routing metadata:
    `metadata.send_counts` is local per-destination/expert send metadata, and
    `clipped_counts[src, dst, local_expert]` is aggregate receive count metadata.
    Return `(hidden, recv_src_rank, recv_src_assignment)` with shapes
    `(capacity, I)`, `(capacity,)`, and `(capacity,)`.

    `remote_rows_sorted` and `keep_sorted` map each locally sorted assignment to
    the destination row selected after receiver-capacity clipping. They are
    provided for kernels that want to reuse the existing staged metadata while
    replacing the token exchange/W13 body.
    """
    raise NotImplementedError(
        "dispatch_user_kernel_permute_up=True routes here; implement "
        "_permute_up_mgpu_user_kernel in lib/levanter/src/levanter/grug/_moe/pallas_mgpu.py. "
        "Start with: --implementations none --include-pallas-stages "
        "--pallas-stages permute_up_compare_split --dispatch-user-kernel-permute-up"
    )


def _permute_up_mgpu_static_workqueue_kernel(
    x_local: jax.Array,
    metadata: MoeMgpuRoutingMetadata,
    clipped_counts: jax.Array,
    rows_per_expert: jax.Array,
    moe_w13: jax.Array,
    activation_fn: Callable[[jax.Array], jax.Array],
    *,
    rank: jax.Array,
    capacity: int,
    ep_size: int,
    local_experts: int,
    expert_axis: str,
    config: MoeMgpuConfig,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Edit point for the spec's static workqueue fused permute_up kernel.

    This path is selected with `MoeMgpuConfig(dispatch_static_workqueue_permute_up=True)`.
    With `dispatch_copy_schedule="expert_group_peer"`, the schedule is:

    ```text
    for expert_group in local expert groups:
        for peer_phase in 0 .. EP-1:
            dst = (rank + peer_phase) % EP
            src = (rank - peer_phase + EP) % EP
    ```

    With `dispatch_copy_schedule="assignment_major"`, the same phase set runs
    with the loop order exchanged: peer phase outer, expert group inner.

    The kernel should remote-write source rows into destination `recv_x`, signal
    readiness, then compute local W13/SwiGLU into destination expert-major
    `hidden`. The arrays below are precomputed outside Pallas to keep the kernel
    body focused on the static producer/consumer phase loop.
    """
    _tokens, hidden_dim = x_local.shape
    local_experts_w13, hidden_dim_w13, intermediate2 = moe_w13.shape
    if local_experts_w13 != local_experts:
        raise ValueError(f"moe_w13 local experts {local_experts_w13} must match {local_experts=}")
    if hidden_dim_w13 != hidden_dim:
        raise ValueError(f"moe_w13 hidden dim {hidden_dim_w13} must match x hidden dim {hidden_dim}")
    intermediate = intermediate2 // 2
    if intermediate2 != 2 * intermediate:
        raise ValueError(f"moe_w13 last dimension must be even, got {intermediate2}")
    if hidden_dim % config.block_k != 0:
        raise ValueError(f"D={hidden_dim} must be divisible by block_k={config.block_k}")
    if intermediate % config.block_n != 0:
        raise ValueError(f"I={intermediate} must be divisible by block_n={config.block_n}")
    copy_tile = config.dispatch_chunk_copy_tile
    if hidden_dim % copy_tile != 0:
        raise ValueError(f"D={hidden_dim} must be divisible by dispatch chunk copy tile={copy_tile}")
    if local_experts % config.dispatch_expert_group_size != 0:
        raise ValueError(
            "dispatch_static_workqueue_permute_up requires E_local to be divisible by "
            f"dispatch_expert_group_size; got E_local={local_experts} and "
            f"dispatch_expert_group_size={config.dispatch_expert_group_size}"
        )
    if config.dispatch_peer_phase_order is not None and tuple(sorted(config.dispatch_peer_phase_order)) != tuple(
        range(ep_size)
    ):
        raise ValueError(
            f"dispatch_peer_phase_order must be a permutation of range(ep_size={ep_size}), "
            f"got {config.dispatch_peer_phase_order}"
        )

    source_expert_offsets = jnp.cumsum(metadata.send_counts.reshape(ep_size * local_experts), dtype=jnp.int32)
    source_expert_offsets = source_expert_offsets - metadata.send_counts.reshape(ep_size * local_experts)
    source_expert_offsets_by_src = lax.all_gather(source_expert_offsets, expert_axis)
    rows_per_dst_expert = jnp.sum(clipped_counts, axis=0, dtype=jnp.int32)
    expert_base_by_dst = jnp.cumsum(rows_per_dst_expert, axis=1, dtype=jnp.int32) - rows_per_dst_expert
    src_base_by_dst_expert = jnp.cumsum(clipped_counts, axis=0, dtype=jnp.int32) - clipped_counts

    assignment_ids_by_src = lax.all_gather(metadata.assignment_ids_sorted, expert_axis)
    dst_ranks_by_src = lax.all_gather(metadata.dst_ranks_sorted, expert_axis)
    local_experts_by_src = lax.all_gather(metadata.local_experts_sorted, expert_axis)
    local_pos_by_src = lax.all_gather(metadata.local_pos_sorted, expert_axis)
    assignments = metadata.assignment_ids_sorted.shape[0]
    source_positions = jnp.arange(assignments, dtype=jnp.int32)
    recv_src_rank = jnp.full((capacity,), -1, dtype=jnp.int32)
    recv_src_assignment = jnp.full((capacity,), -1, dtype=jnp.int32)
    for src in range(ep_size):
        local_expert = local_experts_by_src[src]
        local_pos = local_pos_by_src[src]
        accepted_for_expert = clipped_counts[src, rank, local_expert]
        keep = (source_positions < assignments) & (dst_ranks_by_src[src] == rank) & (local_pos < accepted_for_expert)
        remote_row = (
            expert_base_by_dst[rank, local_expert] + src_base_by_dst_expert[src, rank, local_expert] + local_pos
        )
        remote_row = jnp.where(keep, remote_row, capacity)
        recv_src_rank = recv_src_rank.at[remote_row].set(jnp.int32(src), mode="drop")
        recv_src_assignment = recv_src_assignment.at[remote_row].set(assignment_ids_by_src[src], mode="drop")

    num_sms = config.num_sms
    if num_sms is None:
        num_sms = jax.devices()[0].core_count
    workers = num_sms
    split_wg_threads = 3 if config.dispatch_split_wg_overlap_permute_up else 2
    split_wg_pipeline_stages = 2
    split_wg_tail_slots = min(split_wg_pipeline_stages, hidden_dim // config.block_k)
    expert_group_size = config.dispatch_expert_group_size
    expert_groups = local_experts // expert_group_size
    peer_phase_order = jnp.asarray(
        config.dispatch_peer_phase_order or tuple(range(ep_size)),
        dtype=jnp.int32,
    )
    total_phases = expert_groups * ep_size
    d_copy_tiles = hidden_dim // copy_tile
    n_tiles = intermediate // config.block_n
    destination_pull_n_group = (
        config.dispatch_static_workqueue_destination_pull_n_group
        if config.dispatch_static_workqueue_destination_pull
        and n_tiles % config.dispatch_static_workqueue_destination_pull_n_group == 0
        else 1
    )
    destination_pull_n_groups = n_tiles // destination_pull_n_group
    max_rows_per_src_expert = config.dispatch_static_workqueue_max_rows_per_src_expert or capacity
    max_global_blocks_per_src_expert = _max_blocks_touched_by_unaligned_range(
        max_rows_per_src_expert,
        config.block_m,
    )
    copy_tiles_per_phase = expert_group_size * max_rows_per_src_expert * d_copy_tiles
    copy_steps = int(math.ceil(copy_tiles_per_phase / workers))
    init_recv_x_steps = int(math.ceil((capacity * d_copy_tiles) / workers))
    init_hidden_steps = int(math.ceil((capacity * n_tiles) / workers))

    if config.dispatch_static_workqueue_destination_pull:
        if not os.environ.get("MARIN_MGPU_DEST_PULL_DEBUG_ALLOW"):
            raise NotImplementedError(
                "dispatch_static_workqueue_destination_pull is wired as an experimental permute_up edit point, "
                "but the direct destination-pull W13 prototype is blocked by current Mosaic GPU lowering. "
                "Observed H100 failures: vector remote gathers into SMEM raise "
                "`NotImplementedError: int_indexer_shape non-empty`; scalar peer remote loads into WGMMA lhs SMEM, "
                "including explicit `mgpu.load(remote_ref, idx, optimized=False)`, raise "
                "`ValueError: Failed to infer a possible set of layouts`; and "
                "`mgpu.copy_gmem_to_smem` from `mgpu.remote_ref(..., peer_id)` raises "
                "`NotImplementedError: GMEM refs with peer ids are not supported in warpgroup lowering.` "
                "Use the source-push static workqueue path, or rewrite this branch to stage peer data through a "
                "supported local GMEM/multimem path before WGMMA."
            )

        def body_destination_pull(
            sorted_x_ref,
            clipped_counts_ref,
            source_expert_offsets_by_src_ref,
            expert_base_by_dst_ref,
            src_base_by_dst_expert_ref,
            peer_phase_order_ref,
            rhs_ref,
            hidden_ref,
        ):
            rank = lax.axis_index(expert_axis)
            if config.dispatch_split_wg_permute_up:
                wg_id = lax.axis_index("wg")
                is_compute_wg = wg_id == jnp.int32(0)
                is_loader_wg = wg_id == jnp.int32(1)

            def _phase_indices(global_phase):
                return _static_workqueue_phase_indices(
                    global_phase,
                    rank,
                    ep_size=ep_size,
                    expert_groups=expert_groups,
                    dispatch_copy_schedule=config.dispatch_copy_schedule,
                    peer_phase_order=peer_phase_order_ref,
                    swizzle_expert_groups=config.dispatch_swizzle_expert_groups,
                )

            def _remote_row(dst, src_rank, expert, local_pos):
                return (
                    expert_base_by_dst_ref[dst, expert] + src_base_by_dst_expert_ref[src_rank, dst, expert] + local_pos
                )

            def _source_row(src_rank, dst, expert, local_pos):
                global_expert = dst * local_experts + expert
                return source_expert_offsets_by_src_ref[src_rank, global_expert] + local_pos

            def _compute_phase(global_phase):
                expert_group, peer_phase = _phase_indices(global_phase)
                expert_group_start = expert_group * expert_group_size
                src = (rank + ep_size - peer_phase) % ep_size

                remote_sorted_x_ref = mgpu.remote_ref(
                    sorted_x_ref,
                    src,
                    device_id_type=pl.DeviceIdType.LOGICAL,
                )

                @mgpu.nd_loop(
                    (expert_group_size * max_global_blocks_per_src_expert * destination_pull_n_groups,),
                    collective_axes="sm",
                )
                def _compute_loop(loop_info):
                    mi, n_group_i = mgpu.planar_snake(
                        loop_info.index[0],
                        (expert_group_size * max_global_blocks_per_src_expert, destination_pull_n_groups),
                        1,
                        config.grid_block_n,
                    )
                    expert_in_group = mi // max_global_blocks_per_src_expert
                    block_in_subrange = mi - expert_in_group * max_global_blocks_per_src_expert
                    expert = expert_group_start + expert_in_group
                    local_pos_start = block_in_subrange * config.block_m
                    row_count = clipped_counts_ref[src, rank, expert]
                    if config.dispatch_static_workqueue_dense_balanced_compute:
                        should_compute_block = local_pos_start + config.block_m <= row_count
                    else:
                        should_compute_block = local_pos_start < row_count

                    @pl.when(should_compute_block)
                    def _compute_block():
                        token_offset = _source_row(src, rank, expert, local_pos_start)
                        remote_row_start = _remote_row(rank, src, expert, local_pos_start)

                        if config.dispatch_static_workqueue_dense_balanced_compute:

                            def _store_hidden(hidden_tile, n_tile):
                                hidden_ref[
                                    pl.ds(remote_row_start, config.block_m),
                                    pl.ds(n_tile * config.block_n, config.block_n),
                                ] = hidden_tile.astype(hidden_ref.dtype)

                        else:
                            valid_rows = jnp.minimum(jnp.int32(config.block_m), row_count - local_pos_start)

                            def _store_hidden(hidden_tile, n_tile):
                                @pl.when(valid_rows == config.block_m)
                                def _store_full_block():
                                    hidden_ref[
                                        pl.ds(remote_row_start, config.block_m),
                                        pl.ds(n_tile * config.block_n, config.block_n),
                                    ] = hidden_tile.astype(hidden_ref.dtype)

                                @pl.when(valid_rows < config.block_m)
                                def _store_partial_block():
                                    def store_scope(hidden_smem):
                                        hidden_smem[...] = hidden_tile.astype(hidden_smem.dtype)
                                        mgpu.commit_smem()
                                        chunk_rows = min(8, config.block_m)
                                        row_offset = 0
                                        while row_offset < config.block_m:
                                            full_chunk_end = row_offset + chunk_rows

                                            @pl.when(valid_rows >= full_chunk_end)
                                            def _store_full_tail_chunk():
                                                smem_slice = hidden_smem.at[pl.ds(row_offset, chunk_rows)]
                                                gmem_slice = hidden_ref.at[
                                                    pl.ds(remote_row_start + row_offset, chunk_rows),
                                                    pl.ds(n_tile * config.block_n, config.block_n),
                                                ]
                                                mgpu.copy_smem_to_gmem(smem_slice, gmem_slice)

                                            @pl.when((valid_rows > row_offset) & (valid_rows < full_chunk_end))
                                            def _store_partial_tail_chunk():
                                                for row_in_chunk in range(chunk_rows):
                                                    row = row_offset + row_in_chunk

                                                    @pl.when(valid_rows > row)
                                                    def _store_one_tail_row():
                                                        smem_slice = hidden_smem.at[pl.ds(row, 1)]
                                                        gmem_slice = hidden_ref.at[
                                                            pl.ds(remote_row_start + row, 1),
                                                            pl.ds(n_tile * config.block_n, config.block_n),
                                                        ]
                                                        mgpu.copy_smem_to_gmem(smem_slice, gmem_slice)

                                            row_offset += chunk_rows

                                        mgpu.wait_smem_to_gmem(0, wait_read_only=True)

                                    pl.run_scoped(
                                        store_scope,
                                        hidden_smem=mgpu.SMEM(
                                            (config.block_m, config.block_n),
                                            dtype=hidden_ref.dtype,
                                        ),
                                    )

                        if destination_pull_n_group == 5:

                            def acc_scope(
                                gate0_acc_ref,
                                up0_acc_ref,
                                gate1_acc_ref,
                                up1_acc_ref,
                                gate2_acc_ref,
                                up2_acc_ref,
                                gate3_acc_ref,
                                up3_acc_ref,
                                gate4_acc_ref,
                                up4_acc_ref,
                            ):
                                def smem_scope(
                                    lhs_smem,
                                    gate0_smem,
                                    up0_smem,
                                    gate1_smem,
                                    up1_smem,
                                    gate2_smem,
                                    up2_smem,
                                    gate3_smem,
                                    up3_smem,
                                    gate4_smem,
                                    up4_smem,
                                    ready_barrier,
                                ):
                                    @pl.loop(0, hidden_dim // config.block_k)
                                    def _wgmma_k(kk):
                                        k_start = kk * config.block_k
                                        n0 = n_group_i * 5
                                        n1 = n0 + 1
                                        n2 = n0 + 2
                                        n3 = n0 + 3
                                        n4 = n0 + 4

                                        lhs_smem[:, :] = mgpu.load(
                                            remote_sorted_x_ref,
                                            (
                                                pl.ds(token_offset, config.block_m),
                                                pl.ds(k_start, config.block_k),
                                            ),
                                            layout=mgpu.Layout.WGMMA,
                                            optimized=False,
                                        )

                                        mgpu.copy_gmem_to_smem(
                                            rhs_ref.at[
                                                expert,
                                                pl.ds(k_start, config.block_k),
                                                pl.ds(n0 * config.block_n, config.block_n),
                                            ],
                                            gate0_smem,
                                            ready_barrier,
                                        )
                                        mgpu.copy_gmem_to_smem(
                                            rhs_ref.at[
                                                expert,
                                                pl.ds(k_start, config.block_k),
                                                pl.ds((n0 + n_tiles) * config.block_n, config.block_n),
                                            ],
                                            up0_smem,
                                            ready_barrier,
                                        )
                                        mgpu.copy_gmem_to_smem(
                                            rhs_ref.at[
                                                expert,
                                                pl.ds(k_start, config.block_k),
                                                pl.ds(n1 * config.block_n, config.block_n),
                                            ],
                                            gate1_smem,
                                            ready_barrier,
                                        )
                                        mgpu.copy_gmem_to_smem(
                                            rhs_ref.at[
                                                expert,
                                                pl.ds(k_start, config.block_k),
                                                pl.ds((n1 + n_tiles) * config.block_n, config.block_n),
                                            ],
                                            up1_smem,
                                            ready_barrier,
                                        )
                                        mgpu.copy_gmem_to_smem(
                                            rhs_ref.at[
                                                expert,
                                                pl.ds(k_start, config.block_k),
                                                pl.ds(n2 * config.block_n, config.block_n),
                                            ],
                                            gate2_smem,
                                            ready_barrier,
                                        )
                                        mgpu.copy_gmem_to_smem(
                                            rhs_ref.at[
                                                expert,
                                                pl.ds(k_start, config.block_k),
                                                pl.ds((n2 + n_tiles) * config.block_n, config.block_n),
                                            ],
                                            up2_smem,
                                            ready_barrier,
                                        )
                                        mgpu.copy_gmem_to_smem(
                                            rhs_ref.at[
                                                expert,
                                                pl.ds(k_start, config.block_k),
                                                pl.ds(n3 * config.block_n, config.block_n),
                                            ],
                                            gate3_smem,
                                            ready_barrier,
                                        )
                                        mgpu.copy_gmem_to_smem(
                                            rhs_ref.at[
                                                expert,
                                                pl.ds(k_start, config.block_k),
                                                pl.ds((n3 + n_tiles) * config.block_n, config.block_n),
                                            ],
                                            up3_smem,
                                            ready_barrier,
                                        )
                                        mgpu.copy_gmem_to_smem(
                                            rhs_ref.at[
                                                expert,
                                                pl.ds(k_start, config.block_k),
                                                pl.ds(n4 * config.block_n, config.block_n),
                                            ],
                                            gate4_smem,
                                            ready_barrier,
                                        )
                                        mgpu.copy_gmem_to_smem(
                                            rhs_ref.at[
                                                expert,
                                                pl.ds(k_start, config.block_k),
                                                pl.ds((n4 + n_tiles) * config.block_n, config.block_n),
                                            ],
                                            up4_smem,
                                            ready_barrier,
                                        )
                                        mgpu.barrier_wait(ready_barrier)
                                        mgpu.commit_smem()
                                        mgpu.wgmma(gate0_acc_ref, lhs_smem, gate0_smem)
                                        mgpu.wgmma(up0_acc_ref, lhs_smem, up0_smem)
                                        mgpu.wgmma(gate1_acc_ref, lhs_smem, gate1_smem)
                                        mgpu.wgmma(up1_acc_ref, lhs_smem, up1_smem)
                                        mgpu.wgmma(gate2_acc_ref, lhs_smem, gate2_smem)
                                        mgpu.wgmma(up2_acc_ref, lhs_smem, up2_smem)
                                        mgpu.wgmma(gate3_acc_ref, lhs_smem, gate3_smem)
                                        mgpu.wgmma(up3_acc_ref, lhs_smem, up3_smem)
                                        mgpu.wgmma(gate4_acc_ref, lhs_smem, gate4_smem)
                                        mgpu.wgmma(up4_acc_ref, lhs_smem, up4_smem)
                                        mgpu.wgmma_wait(0)

                                pl.run_scoped(
                                    smem_scope,
                                    lhs_smem=mgpu.SMEM((config.block_m, config.block_k), dtype=sorted_x_ref.dtype),
                                    gate0_smem=mgpu.SMEM((config.block_k, config.block_n), dtype=rhs_ref.dtype),
                                    up0_smem=mgpu.SMEM((config.block_k, config.block_n), dtype=rhs_ref.dtype),
                                    gate1_smem=mgpu.SMEM((config.block_k, config.block_n), dtype=rhs_ref.dtype),
                                    up1_smem=mgpu.SMEM((config.block_k, config.block_n), dtype=rhs_ref.dtype),
                                    gate2_smem=mgpu.SMEM((config.block_k, config.block_n), dtype=rhs_ref.dtype),
                                    up2_smem=mgpu.SMEM((config.block_k, config.block_n), dtype=rhs_ref.dtype),
                                    gate3_smem=mgpu.SMEM((config.block_k, config.block_n), dtype=rhs_ref.dtype),
                                    up3_smem=mgpu.SMEM((config.block_k, config.block_n), dtype=rhs_ref.dtype),
                                    gate4_smem=mgpu.SMEM((config.block_k, config.block_n), dtype=rhs_ref.dtype),
                                    up4_smem=mgpu.SMEM((config.block_k, config.block_n), dtype=rhs_ref.dtype),
                                    ready_barrier=mgpu.Barrier(num_arrivals=10),
                                )
                                return (
                                    activation_fn(gate0_acc_ref[...]) * up0_acc_ref[...],
                                    activation_fn(gate1_acc_ref[...]) * up1_acc_ref[...],
                                    activation_fn(gate2_acc_ref[...]) * up2_acc_ref[...],
                                    activation_fn(gate3_acc_ref[...]) * up3_acc_ref[...],
                                    activation_fn(gate4_acc_ref[...]) * up4_acc_ref[...],
                                )

                            hidden0, hidden1, hidden2, hidden3, hidden4 = pl.run_scoped(
                                acc_scope,
                                gate0_acc_ref=mgpu.ACC((config.block_m, config.block_n)),
                                up0_acc_ref=mgpu.ACC((config.block_m, config.block_n)),
                                gate1_acc_ref=mgpu.ACC((config.block_m, config.block_n)),
                                up1_acc_ref=mgpu.ACC((config.block_m, config.block_n)),
                                gate2_acc_ref=mgpu.ACC((config.block_m, config.block_n)),
                                up2_acc_ref=mgpu.ACC((config.block_m, config.block_n)),
                                gate3_acc_ref=mgpu.ACC((config.block_m, config.block_n)),
                                up3_acc_ref=mgpu.ACC((config.block_m, config.block_n)),
                                gate4_acc_ref=mgpu.ACC((config.block_m, config.block_n)),
                                up4_acc_ref=mgpu.ACC((config.block_m, config.block_n)),
                            )
                            _store_hidden(hidden0, n_group_i * 5)
                            _store_hidden(hidden1, n_group_i * 5 + 1)
                            _store_hidden(hidden2, n_group_i * 5 + 2)
                            _store_hidden(hidden3, n_group_i * 5 + 3)
                            _store_hidden(hidden4, n_group_i * 5 + 4)
                        elif destination_pull_n_group == 2:

                            def acc_scope(gate0_acc_ref, up0_acc_ref, gate1_acc_ref, up1_acc_ref):
                                def smem_scope(
                                    lhs_smem,
                                    gate0_smem,
                                    up0_smem,
                                    gate1_smem,
                                    up1_smem,
                                    ready_barrier,
                                ):
                                    @pl.loop(0, hidden_dim // config.block_k)
                                    def _wgmma_k(kk):
                                        k_start = kk * config.block_k
                                        n0 = n_group_i * 2
                                        n1 = n0 + 1

                                        lhs_smem[:, :] = mgpu.load(
                                            remote_sorted_x_ref,
                                            (
                                                pl.ds(token_offset, config.block_m),
                                                pl.ds(k_start, config.block_k),
                                            ),
                                            layout=mgpu.Layout.WGMMA,
                                            optimized=False,
                                        )

                                        mgpu.copy_gmem_to_smem(
                                            rhs_ref.at[
                                                expert,
                                                pl.ds(k_start, config.block_k),
                                                pl.ds(n0 * config.block_n, config.block_n),
                                            ],
                                            gate0_smem,
                                            ready_barrier,
                                        )
                                        mgpu.copy_gmem_to_smem(
                                            rhs_ref.at[
                                                expert,
                                                pl.ds(k_start, config.block_k),
                                                pl.ds((n0 + n_tiles) * config.block_n, config.block_n),
                                            ],
                                            up0_smem,
                                            ready_barrier,
                                        )
                                        mgpu.copy_gmem_to_smem(
                                            rhs_ref.at[
                                                expert,
                                                pl.ds(k_start, config.block_k),
                                                pl.ds(n1 * config.block_n, config.block_n),
                                            ],
                                            gate1_smem,
                                            ready_barrier,
                                        )
                                        mgpu.copy_gmem_to_smem(
                                            rhs_ref.at[
                                                expert,
                                                pl.ds(k_start, config.block_k),
                                                pl.ds((n1 + n_tiles) * config.block_n, config.block_n),
                                            ],
                                            up1_smem,
                                            ready_barrier,
                                        )
                                        mgpu.barrier_wait(ready_barrier)
                                        mgpu.commit_smem()
                                        mgpu.wgmma(gate0_acc_ref, lhs_smem, gate0_smem)
                                        mgpu.wgmma(up0_acc_ref, lhs_smem, up0_smem)
                                        mgpu.wgmma(gate1_acc_ref, lhs_smem, gate1_smem)
                                        mgpu.wgmma(up1_acc_ref, lhs_smem, up1_smem)
                                        mgpu.wgmma_wait(0)

                                pl.run_scoped(
                                    smem_scope,
                                    lhs_smem=mgpu.SMEM((config.block_m, config.block_k), dtype=sorted_x_ref.dtype),
                                    gate0_smem=mgpu.SMEM((config.block_k, config.block_n), dtype=rhs_ref.dtype),
                                    up0_smem=mgpu.SMEM((config.block_k, config.block_n), dtype=rhs_ref.dtype),
                                    gate1_smem=mgpu.SMEM((config.block_k, config.block_n), dtype=rhs_ref.dtype),
                                    up1_smem=mgpu.SMEM((config.block_k, config.block_n), dtype=rhs_ref.dtype),
                                    ready_barrier=mgpu.Barrier(num_arrivals=4),
                                )
                                return (
                                    activation_fn(gate0_acc_ref[...]) * up0_acc_ref[...],
                                    activation_fn(gate1_acc_ref[...]) * up1_acc_ref[...],
                                )

                            hidden0, hidden1 = pl.run_scoped(
                                acc_scope,
                                gate0_acc_ref=mgpu.ACC((config.block_m, config.block_n)),
                                up0_acc_ref=mgpu.ACC((config.block_m, config.block_n)),
                                gate1_acc_ref=mgpu.ACC((config.block_m, config.block_n)),
                                up1_acc_ref=mgpu.ACC((config.block_m, config.block_n)),
                            )
                            _store_hidden(hidden0, n_group_i * 2)
                            _store_hidden(hidden1, n_group_i * 2 + 1)
                        else:
                            if config.dispatch_split_wg_permute_up:

                                def acc_scope(gate_acc_ref, up_acc_ref):
                                    def smem_scope(
                                        lhs_smem,
                                        gate_smem,
                                        up_smem,
                                        ready_barrier,
                                        consumed_barrier,
                                    ):
                                        k_tiles = hidden_dim // config.block_k
                                        pipeline_stages = 2

                                        @pl.when(is_loader_wg)
                                        def _load_rhs_tiles():
                                            @pl.loop(0, k_tiles)
                                            def _load_k(kk):
                                                slot = kk % pipeline_stages

                                                @pl.when(kk >= pipeline_stages)
                                                def _wait_consumed_before_reuse():
                                                    mgpu.barrier_wait(consumed_barrier.at[slot])

                                                k_start = kk * config.block_k
                                                mgpu.copy_gmem_to_smem(
                                                    rhs_ref.at[
                                                        expert,
                                                        pl.ds(k_start, config.block_k),
                                                        pl.ds(n_group_i * config.block_n, config.block_n),
                                                    ],
                                                    gate_smem.at[slot],
                                                    ready_barrier.at[slot],
                                                )
                                                mgpu.copy_gmem_to_smem(
                                                    rhs_ref.at[
                                                        expert,
                                                        pl.ds(k_start, config.block_k),
                                                        pl.ds(
                                                            (n_group_i + n_tiles) * config.block_n,
                                                            config.block_n,
                                                        ),
                                                    ],
                                                    up_smem.at[slot],
                                                    ready_barrier.at[slot],
                                                )

                                            @pl.loop(0, pipeline_stages)
                                            def _wait_tail_consumed(tail):
                                                @pl.when(tail < jnp.minimum(pipeline_stages, k_tiles))
                                                def _wait_tail_slot():
                                                    kk = k_tiles - tail - 1
                                                    slot = kk % pipeline_stages
                                                    mgpu.barrier_wait(consumed_barrier.at[slot])

                                        @pl.when(is_compute_wg)
                                        def _compute_w13():
                                            @pl.loop(0, k_tiles)
                                            def _wgmma_k(kk):
                                                slot = kk % pipeline_stages
                                                k_start = kk * config.block_k
                                                lhs_smem[:, :] = mgpu.load(
                                                    remote_sorted_x_ref,
                                                    (
                                                        pl.ds(token_offset, config.block_m),
                                                        pl.ds(k_start, config.block_k),
                                                    ),
                                                    layout=mgpu.Layout.WGMMA,
                                                    optimized=False,
                                                )
                                                mgpu.barrier_wait(ready_barrier.at[slot])
                                                mgpu.commit_smem()
                                                mgpu.wgmma(gate_acc_ref, lhs_smem, gate_smem.at[slot])
                                                mgpu.wgmma(up_acc_ref, lhs_smem, up_smem.at[slot])
                                                mgpu.wgmma_wait(0)
                                                mgpu.barrier_arrive(consumed_barrier.at[slot])

                                    pl.run_scoped(
                                        smem_scope,
                                        lhs_smem=mgpu.SMEM(
                                            (config.block_m, config.block_k),
                                            dtype=sorted_x_ref.dtype,
                                        ),
                                        gate_smem=mgpu.SMEM(
                                            (2, config.block_k, config.block_n),
                                            dtype=rhs_ref.dtype,
                                        ),
                                        up_smem=mgpu.SMEM(
                                            (2, config.block_k, config.block_n),
                                            dtype=rhs_ref.dtype,
                                        ),
                                        ready_barrier=mgpu.Barrier(
                                            num_arrivals=2,
                                            num_barriers=2,
                                        ),
                                        consumed_barrier=mgpu.Barrier(
                                            num_arrivals=1,
                                            num_barriers=2,
                                        ),
                                        collective_axes="wg",
                                    )
                                    return activation_fn(gate_acc_ref[...]) * up_acc_ref[...]

                                hidden = pl.run_scoped(
                                    acc_scope,
                                    gate_acc_ref=mgpu.ACC((config.block_m, config.block_n)),
                                    up_acc_ref=mgpu.ACC((config.block_m, config.block_n)),
                                )
                                _store_hidden(hidden, n_group_i)
                            else:

                                def acc_scope(gate_acc_ref, up_acc_ref):
                                    def smem_scope(lhs_smem, gate_smem, up_smem, ready_barrier):
                                        @pl.loop(0, hidden_dim // config.block_k)
                                        def _wgmma_k(kk):
                                            k_start = kk * config.block_k

                                            lhs_smem[:, :] = mgpu.load(
                                                remote_sorted_x_ref,
                                                (
                                                    pl.ds(token_offset, config.block_m),
                                                    pl.ds(k_start, config.block_k),
                                                ),
                                                layout=mgpu.Layout.WGMMA,
                                                optimized=False,
                                            )

                                            mgpu.copy_gmem_to_smem(
                                                rhs_ref.at[
                                                    expert,
                                                    pl.ds(k_start, config.block_k),
                                                    pl.ds(n_group_i * config.block_n, config.block_n),
                                                ],
                                                gate_smem,
                                                ready_barrier,
                                            )
                                            mgpu.copy_gmem_to_smem(
                                                rhs_ref.at[
                                                    expert,
                                                    pl.ds(k_start, config.block_k),
                                                    pl.ds((n_group_i + n_tiles) * config.block_n, config.block_n),
                                                ],
                                                up_smem,
                                                ready_barrier,
                                            )
                                            mgpu.barrier_wait(ready_barrier)
                                            mgpu.commit_smem()
                                            mgpu.wgmma(gate_acc_ref, lhs_smem, gate_smem)
                                            mgpu.wgmma(up_acc_ref, lhs_smem, up_smem)
                                            mgpu.wgmma_wait(0)

                                    pl.run_scoped(
                                        smem_scope,
                                        lhs_smem=mgpu.SMEM(
                                            (config.block_m, config.block_k),
                                            dtype=sorted_x_ref.dtype,
                                        ),
                                        gate_smem=mgpu.SMEM((config.block_k, config.block_n), dtype=rhs_ref.dtype),
                                        up_smem=mgpu.SMEM((config.block_k, config.block_n), dtype=rhs_ref.dtype),
                                        ready_barrier=mgpu.Barrier(num_arrivals=2),
                                    )
                                    return activation_fn(gate_acc_ref[...]) * up_acc_ref[...]

                                hidden = pl.run_scoped(
                                    acc_scope,
                                    gate_acc_ref=mgpu.ACC((config.block_m, config.block_n)),
                                    up_acc_ref=mgpu.ACC((config.block_m, config.block_n)),
                                )
                                _store_hidden(hidden, n_group_i)

            @pl.loop(0, total_phases)
            def _phase_loop(global_phase):
                _compute_phase(global_phase)

        kernel_kwargs = {}
        if config.dispatch_split_wg_permute_up:
            kernel_kwargs = {"num_threads": 2, "thread_name": "wg"}
        kernel_destination_pull = mgpu.kernel(
            body_destination_pull,
            out_shape=jax.ShapeDtypeStruct((capacity, intermediate), x_local.dtype),
            grid=(num_sms,),
            grid_names=("sm",),
            compiler_params=mgpu.CompilerParams(
                lowering_semantics=mgpu.LoweringSemantics.Warpgroup,
            ),
            **kernel_kwargs,
        )
        # Source-side packed order matching `source_expert_offsets`: destination rank,
        # local expert, then local position within that source/destination/expert run.
        # This makes each routed-token block contiguous for destination-pull WGMMA loads.
        sorted_x = x_local[metadata.token_ids_sorted]
        sorted_x = jnp.concatenate(
            [sorted_x, jnp.zeros((config.block_m, hidden_dim), dtype=sorted_x.dtype)],
            axis=0,
        )
        hidden = kernel_destination_pull(
            sorted_x,
            clipped_counts,
            source_expert_offsets_by_src,
            expert_base_by_dst,
            src_base_by_dst_expert,
            peer_phase_order,
            moe_w13,
        )
        return hidden, recv_src_rank, recv_src_assignment

    def body(
        x_ref,
        token_ids_ref,
        source_expert_offsets_ref,
        clipped_counts_ref,
        expert_base_by_dst_ref,
        src_base_by_dst_expert_ref,
        peer_phase_order_ref,
        rhs_ref,
        hidden_ref,
        recv_x_ref,
    ):
        arrivals_sem = pl.get_global(mgpu.SemaphoreType.REGULAR)
        rank = lax.axis_index(expert_axis)
        sm_id = lax.axis_index("sm")
        wg_id = lax.axis_index("wg") if config.dispatch_split_wg_permute_up else jnp.int32(0)
        if config.dispatch_split_wg_overlap_permute_up:
            is_producer_wg = wg_id == jnp.int32(0)
            is_loader_wg = wg_id == jnp.int32(1)
            is_compute_wg = wg_id == jnp.int32(2)
        else:
            is_producer_wg = wg_id == jnp.int32(0)
            is_loader_wg = wg_id == jnp.int32(0)
            is_compute_wg = wg_id == jnp.int32(1)

        def _phase_indices(global_phase):
            return _static_workqueue_phase_indices(
                global_phase,
                rank,
                ep_size=ep_size,
                expert_groups=expert_groups,
                dispatch_copy_schedule=config.dispatch_copy_schedule,
                peer_phase_order=peer_phase_order_ref,
                swizzle_expert_groups=config.dispatch_swizzle_expert_groups,
            )

        def _remote_row(dst, src_rank, expert, local_pos):
            return expert_base_by_dst_ref[dst, expert] + src_base_by_dst_expert_ref[src_rank, dst, expert] + local_pos

        def _signal_phase_ready(global_phase):
            _expert_group, peer_phase = _phase_indices(global_phase)
            dst = (rank + peer_phase) % ep_size
            mgpu.semaphore_signal_parallel(
                mgpu.SemaphoreSignal(arrivals_sem, device_id=(jnp.int32(0), dst, jnp.int32(0))),
            )

        def _wait_phase_ready(global_phase):
            expected_arrivals = (global_phase + 1) * workers
            pl.semaphore_wait(arrivals_sem, value=expected_arrivals, decrement=False)

        def _init_outputs():
            if config.dispatch_static_workqueue_init_recv_x:

                @pl.loop(0, init_recv_x_steps)
                def _init_recv_x_step(step):
                    linear = step * workers + sm_id

                    @pl.when(linear < capacity * d_copy_tiles)
                    def _zero_recv_x():
                        row = linear // d_copy_tiles
                        d_tile = linear - row * d_copy_tiles
                        recv_x_ref[row, pl.ds(d_tile * copy_tile, copy_tile)] = jnp.zeros(
                            (copy_tile,), dtype=recv_x_ref.dtype
                        )

            if config.dispatch_static_workqueue_init_hidden:

                @pl.loop(0, init_hidden_steps)
                def _init_hidden_step(step):
                    linear = step * workers + sm_id

                    @pl.when(linear < capacity * n_tiles)
                    def _zero_hidden():
                        row = linear // n_tiles
                        n_tile = linear - row * n_tiles
                        hidden_ref[row, pl.ds(n_tile * config.block_n, config.block_n)] = jnp.zeros(
                            (config.block_n,), dtype=hidden_ref.dtype
                        )

        def _send_phase(global_phase):
            expert_group, peer_phase = _phase_indices(global_phase)
            expert_group_start = expert_group * expert_group_size
            dst = (rank + peer_phase) % ep_size

            def _copy_linear(linear):
                expert_in_group = linear // (max_rows_per_src_expert * d_copy_tiles)
                remainder = linear - expert_in_group * (max_rows_per_src_expert * d_copy_tiles)
                local_pos = remainder // d_copy_tiles
                d_tile = remainder - local_pos * d_copy_tiles
                expert = expert_group_start + expert_in_group
                count = clipped_counts_ref[rank, dst, expert]

                @pl.when(local_pos < count)
                def _copy_payload():
                    global_expert = dst * local_experts + expert
                    token_offset = source_expert_offsets_ref[global_expert] + local_pos
                    token_id = token_ids_ref[token_offset]
                    d_start = d_tile * copy_tile
                    row = _remote_row(dst, rank, expert, local_pos)
                    remote_recv_x_ref = mgpu.remote_ref(
                        recv_x_ref,
                        dst,
                        device_id_type=pl.DeviceIdType.LOGICAL,
                    )
                    remote_recv_x_ref[row, pl.ds(d_start, copy_tile)] = x_ref[token_id, pl.ds(d_start, copy_tile)]

            if config.dispatch_static_workqueue_copy_pl_loop:

                @pl.loop(0, copy_steps)
                def _copy_step(step):
                    linear = step * workers + sm_id

                    @pl.when(linear < copy_tiles_per_phase)
                    def _copy_tile():
                        _copy_linear(linear)

            else:

                @mgpu.nd_loop((copy_tiles_per_phase,), collective_axes="sm")
                def _copy_loop(loop_info):
                    _copy_linear(loop_info.index[0])

            _signal_phase_ready(global_phase)

        def _compute_phase(global_phase):
            expert_group, peer_phase = _phase_indices(global_phase)
            expert_group_start = expert_group * expert_group_size
            src = (rank + ep_size - peer_phase) % ep_size
            _wait_phase_ready(global_phase)

            @mgpu.nd_loop(
                (expert_group_size * max_global_blocks_per_src_expert * n_tiles,),
                collective_axes="sm",
            )
            def _compute_loop(loop_info):
                if config.dispatch_static_workqueue_dense_balanced_compute:
                    mi, ni = mgpu.planar_snake(
                        loop_info.index[0],
                        (expert_group_size * max_global_blocks_per_src_expert, n_tiles),
                        1,
                        config.grid_block_n,
                    )
                    expert_in_group = mi // max_global_blocks_per_src_expert
                    block_in_subrange = mi - expert_in_group * max_global_blocks_per_src_expert
                else:
                    linear = loop_info.index[0]
                    expert_in_group = linear // (max_global_blocks_per_src_expert * n_tiles)
                    remainder = linear - expert_in_group * (max_global_blocks_per_src_expert * n_tiles)
                    block_in_subrange = remainder // n_tiles
                    ni = remainder - block_in_subrange * n_tiles
                expert = expert_group_start + expert_in_group
                row_count = clipped_counts_ref[src, rank, expert]
                row_start = expert_base_by_dst_ref[rank, expert] + src_base_by_dst_expert_ref[src, rank, expert]
                if config.dispatch_static_workqueue_dense_balanced_compute:
                    del row_count
                    global_block = row_start // config.block_m + block_in_subrange
                    block_start = global_block * config.block_m
                    actual_size = jnp.int32(config.block_m)
                    start_within_block = jnp.int32(0)
                else:
                    row_end = row_start + row_count
                    first_global_block = row_start // config.block_m
                    global_block = first_global_block + block_in_subrange
                    block_start = global_block * config.block_m
                    actual_start = jnp.maximum(row_start, block_start)
                    actual_end = jnp.minimum(row_end, block_start + config.block_m)
                    actual_size = jnp.maximum(actual_end - actual_start, jnp.int32(0))
                    start_within_block = actual_start - block_start

                def _compute_tile():
                    def _store_hidden_from_smem(hidden_smem):
                        smem_start = start_within_block
                        remaining_rows = min(config.block_m, capacity)

                        while remaining_rows > 0:
                            const_rows_len = 1 << int(math.log2(remaining_rows))
                            remaining_rows //= 2

                            @pl.when(actual_size & const_rows_len != 0)
                            def _store_rows():
                                smem_slice = hidden_smem.at[pl.ds(smem_start, const_rows_len)]
                                gmem_slice = hidden_ref.at[
                                    pl.ds(block_start + smem_start, const_rows_len),
                                    pl.ds(ni * config.block_n, config.block_n),
                                ]
                                mgpu.copy_smem_to_gmem(smem_slice, gmem_slice)

                            smem_start += actual_size & const_rows_len

                        mgpu.wait_smem_to_gmem(0, wait_read_only=True)

                    if config.dispatch_split_wg_permute_up:
                        k_tiles = hidden_dim // config.block_k

                        def split_scope(
                            lhs_smem,
                            gate_smem,
                            up_smem,
                            hidden_smem,
                            ready_barrier,
                            consumed_barrier,
                        ):
                            @pl.when(is_loader_wg)
                            def _load_tiles():
                                @pl.loop(0, k_tiles)
                                def _load_k(kk):
                                    slot = kk % split_wg_pipeline_stages

                                    @pl.when(kk >= split_wg_pipeline_stages)
                                    def _wait_consumed_before_reuse():
                                        mgpu.barrier_wait(consumed_barrier.at[slot])

                                    k_start = kk * config.block_k
                                    mgpu.copy_gmem_to_smem(
                                        recv_x_ref.at[
                                            pl.ds(global_block * config.block_m, config.block_m),
                                            pl.ds(k_start, config.block_k),
                                        ],
                                        lhs_smem.at[slot],
                                        ready_barrier.at[slot],
                                    )
                                    mgpu.copy_gmem_to_smem(
                                        rhs_ref.at[
                                            expert,
                                            pl.ds(k_start, config.block_k),
                                            pl.ds(ni * config.block_n, config.block_n),
                                        ],
                                        gate_smem.at[slot],
                                        ready_barrier.at[slot],
                                    )
                                    mgpu.copy_gmem_to_smem(
                                        rhs_ref.at[
                                            expert,
                                            pl.ds(k_start, config.block_k),
                                            pl.ds((ni + n_tiles) * config.block_n, config.block_n),
                                        ],
                                        up_smem.at[slot],
                                        ready_barrier.at[slot],
                                    )

                                @pl.loop(0, split_wg_tail_slots)
                                def _wait_tail_consumed(tail):
                                    kk = k_tiles - tail - 1
                                    slot = kk % split_wg_pipeline_stages
                                    mgpu.barrier_wait(consumed_barrier.at[slot])

                            @pl.when(is_compute_wg)
                            def _compute_w13():
                                def acc_scope(gate_acc_ref, up_acc_ref):
                                    @pl.loop(0, k_tiles)
                                    def _wgmma_k(kk):
                                        slot = kk % split_wg_pipeline_stages
                                        mgpu.barrier_wait(ready_barrier.at[slot])
                                        mgpu.wgmma(gate_acc_ref, lhs_smem.at[slot], gate_smem.at[slot])
                                        mgpu.wgmma(up_acc_ref, lhs_smem.at[slot], up_smem.at[slot])
                                        mgpu.wgmma_wait(0)
                                        mgpu.barrier_arrive(consumed_barrier.at[slot])

                                    hidden = activation_fn(gate_acc_ref[...]) * up_acc_ref[...]
                                    hidden_smem[...] = hidden.astype(hidden_smem.dtype)
                                    mgpu.commit_smem()
                                    _store_hidden_from_smem(hidden_smem)

                                pl.run_scoped(
                                    acc_scope,
                                    gate_acc_ref=mgpu.ACC((config.block_m, config.block_n)),
                                    up_acc_ref=mgpu.ACC((config.block_m, config.block_n)),
                                )

                        pl.run_scoped(
                            split_scope,
                            lhs_smem=mgpu.SMEM(
                                (split_wg_pipeline_stages, config.block_m, config.block_k),
                                dtype=recv_x_ref.dtype,
                            ),
                            gate_smem=mgpu.SMEM(
                                (split_wg_pipeline_stages, config.block_k, config.block_n),
                                dtype=rhs_ref.dtype,
                            ),
                            up_smem=mgpu.SMEM(
                                (split_wg_pipeline_stages, config.block_k, config.block_n),
                                dtype=rhs_ref.dtype,
                            ),
                            hidden_smem=mgpu.SMEM((config.block_m, config.block_n), dtype=hidden_ref.dtype),
                            ready_barrier=mgpu.Barrier(
                                num_arrivals=3,
                                num_barriers=split_wg_pipeline_stages,
                            ),
                            consumed_barrier=mgpu.Barrier(
                                num_arrivals=1,
                                num_barriers=split_wg_pipeline_stages,
                            ),
                            collective_axes="wg",
                        )
                    else:

                        def acc_scope(gate_acc_ref, up_acc_ref):
                            def wgmma_step(_, lhs_smem, gate_smem, up_smem):
                                mgpu.wgmma(gate_acc_ref, lhs_smem, gate_smem)
                                mgpu.wgmma(up_acc_ref, lhs_smem, up_smem)

                            mgpu.emit_pipeline(
                                wgmma_step,
                                grid=(hidden_dim // config.block_k,),
                                in_specs=[
                                    mgpu.BlockSpec(
                                        (config.block_m, config.block_k),
                                        lambda kk: (global_block, kk),
                                        delay_release=1,
                                    ),
                                    mgpu.BlockSpec(
                                        (config.block_k, config.block_n),
                                        lambda kk: (kk, ni),
                                        delay_release=1,
                                    ),
                                    mgpu.BlockSpec(
                                        (config.block_k, config.block_n),
                                        lambda kk: (kk, ni + n_tiles),
                                        delay_release=1,
                                    ),
                                ],
                                max_concurrent_steps=config.max_concurrent_steps,
                            )(
                                recv_x_ref,
                                rhs_ref.at[expert],
                                rhs_ref.at[expert],
                            )
                            return activation_fn(gate_acc_ref[...]) * up_acc_ref[...]

                        hidden = pl.run_scoped(
                            acc_scope,
                            gate_acc_ref=mgpu.ACC((config.block_m, config.block_n)),
                            up_acc_ref=mgpu.ACC((config.block_m, config.block_n)),
                        )

                        if config.dispatch_static_workqueue_direct_store_hidden:
                            hidden_ref[
                                pl.ds(global_block * config.block_m, config.block_m),
                                pl.ds(ni * config.block_n, config.block_n),
                            ] = hidden.astype(hidden_ref.dtype)
                        else:

                            def store_scope(hidden_smem):
                                hidden_smem[...] = hidden.astype(hidden_smem.dtype)
                                mgpu.commit_smem()
                                _store_hidden_from_smem(hidden_smem)

                            pl.run_scoped(
                                store_scope,
                                hidden_smem=mgpu.SMEM((config.block_m, config.block_n), dtype=hidden_ref.dtype),
                            )

                if config.dispatch_static_workqueue_dense_balanced_compute:
                    _compute_tile()
                else:

                    @pl.when(actual_size > 0)
                    def _compute_nonempty_tile():
                        _compute_tile()

        if config.dispatch_split_wg_permute_up:

            @pl.when(is_producer_wg)
            def _run_producer_init():
                _init_outputs()

            if config.dispatch_split_wg_overlap_permute_up:

                @pl.when(is_producer_wg)
                def _run_producer_loop():
                    @pl.loop(0, total_phases)
                    def _producer_phase_loop(global_phase):
                        _send_phase(global_phase)

                @pl.when(is_loader_wg | is_compute_wg)
                def _run_compute_loop():
                    @pl.loop(0, total_phases)
                    def _compute_phase_loop(global_phase):
                        _compute_phase(global_phase)

            else:

                @pl.loop(0, total_phases)
                def _lockstep_phase_loop(global_phase):
                    @pl.when(is_producer_wg)
                    def _producer_phase():
                        _send_phase(global_phase)

                    @pl.when(is_loader_wg | is_compute_wg)
                    def _compute_phase_step():
                        _compute_phase(global_phase)

        else:
            _init_outputs()

            @pl.loop(0, total_phases)
            def _phase_loop(global_phase):
                _send_phase(global_phase)
                _compute_phase(global_phase)

    kernel = mgpu.kernel(
        body,
        out_shape=[
            jax.ShapeDtypeStruct((capacity, intermediate), x_local.dtype),
            jax.ShapeDtypeStruct((capacity, hidden_dim), x_local.dtype),
        ],
        grid=(num_sms,),
        grid_names=("sm",),
        num_threads=split_wg_threads if config.dispatch_split_wg_permute_up else None,
        thread_name="wg" if config.dispatch_split_wg_permute_up else None,
        compiler_params=mgpu.CompilerParams(
            lowering_semantics=mgpu.LoweringSemantics.Warpgroup,
        ),
    )
    hidden, _recv_x = kernel(
        x_local,
        metadata.token_ids_sorted,
        source_expert_offsets,
        clipped_counts,
        expert_base_by_dst,
        src_base_by_dst_expert,
        peer_phase_order,
        moe_w13,
    )
    return hidden, recv_src_rank, recv_src_assignment


def _permute_up_mgpu_fused_chunked_kernel(
    x_local: jax.Array,
    assignment_ids_sorted: jax.Array,
    token_ids_sorted: jax.Array,
    send_counts: jax.Array,
    clipped_counts: jax.Array,
    moe_w13: jax.Array,
    activation_fn: Callable[[jax.Array], jax.Array],
    *,
    capacity: int,
    ep_size: int,
    local_experts: int,
    expert_axis: str,
    config: MoeMgpuConfig,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    _tokens, hidden_dim = x_local.shape
    assignments = token_ids_sorted.shape[0]
    local_experts_w13, hidden_dim_w13, intermediate2 = moe_w13.shape
    if local_experts_w13 != local_experts:
        raise ValueError(f"moe_w13 local experts {local_experts_w13} must match {local_experts=}")
    if hidden_dim_w13 != hidden_dim:
        raise ValueError(f"moe_w13 hidden dim {hidden_dim_w13} must match x hidden dim {hidden_dim}")
    if assignments % (ep_size * local_experts) != 0:
        raise ValueError(
            "chunked permute_up currently requires balanced routing: "
            f"assignments={assignments} must be divisible by global experts={ep_size * local_experts}"
        )

    expert_group_size = config.dispatch_expert_group_size
    if local_experts % expert_group_size != 0:
        raise ValueError(f"{local_experts=} must be divisible by {expert_group_size=}")
    if config.dispatch_peer_phase_order is not None and tuple(sorted(config.dispatch_peer_phase_order)) != tuple(
        range(ep_size)
    ):
        raise ValueError(
            f"dispatch_peer_phase_order must be a permutation of range(ep_size={ep_size}), "
            f"got {config.dispatch_peer_phase_order}"
        )
    rows_per_source_expert = assignments // (ep_size * local_experts)
    _require_chunked_balanced_counts(send_counts, rows_per_source_expert)
    if rows_per_source_expert % config.block_m != 0:
        raise ValueError(
            "chunked permute_up currently requires rows per source/expert to be divisible by block_m; "
            f"got {rows_per_source_expert=} and block_m={config.block_m}"
        )
    copy_tile = config.dispatch_chunk_copy_tile
    if hidden_dim % copy_tile != 0:
        raise ValueError(f"D={hidden_dim} must be divisible by dispatch chunk copy tile={copy_tile}")
    copy_rows = config.dispatch_chunk_copy_rows
    if rows_per_source_expert % copy_rows != 0:
        raise ValueError(
            "chunked permute_up currently requires rows per source/expert to be divisible by "
            f"dispatch_chunk_copy_rows; got {rows_per_source_expert=} and {copy_rows=}"
        )

    intermediate = intermediate2 // 2
    if intermediate2 != 2 * intermediate:
        raise ValueError(f"moe_w13 last dimension must be even, got {intermediate2}")
    if intermediate % config.block_n != 0:
        raise ValueError(f"I={intermediate} must be divisible by block_n={config.block_n}")
    if hidden_dim % config.block_k != 0:
        raise ValueError(f"D={hidden_dim} must be divisible by block_k={config.block_k}")

    num_sms = config.num_sms
    if num_sms is None:
        num_sms = jax.devices()[0].core_count
    workers = num_sms
    d_copy_tiles = hidden_dim // copy_tile
    chunk_rows = rows_per_source_expert * expert_group_size
    copy_row_groups = chunk_rows // copy_rows
    copy_tiles_per_chunk = copy_row_groups * d_copy_tiles
    copy_steps = int(math.ceil(copy_tiles_per_chunk / workers))
    init_metadata_steps = int(math.ceil(capacity / workers))
    padding_rows = capacity - assignments
    init_hidden_tiles = padding_rows * (intermediate // config.block_n)
    init_hidden_steps = int(math.ceil(init_hidden_tiles / workers))
    blocks_per_source_expert = rows_per_source_expert // config.block_m
    chunk_grid_m = expert_group_size * blocks_per_source_expert
    chunk_grid_n = intermediate // config.block_n
    expert_groups = local_experts // expert_group_size
    peer_phase_order = jnp.asarray(
        config.dispatch_peer_phase_order or tuple(range(ep_size)),
        dtype=jnp.int32,
    )
    total_phases = expert_groups * ep_size
    scratch_slots = total_phases
    split_wg_threads = 3 if config.dispatch_split_wg_overlap_permute_up else 2
    split_wg_pipeline_stages = 2
    compute_tiles = chunk_grid_m * chunk_grid_n
    compute_tiles_per_sm = int(math.ceil(compute_tiles / workers))
    compute_k_steps_per_sm = compute_tiles_per_sm * (hidden_dim // config.block_k)
    overlap_copy_steps_per_k_step = int(math.ceil(copy_steps / compute_k_steps_per_sm))
    source_expert_offsets = jnp.cumsum(send_counts.reshape(ep_size * local_experts), dtype=jnp.int32)
    source_expert_offsets = source_expert_offsets - send_counts.reshape(ep_size * local_experts)
    rows_by_dst_expert = jnp.sum(clipped_counts, axis=0, dtype=jnp.int32)
    expert_base_by_dst = (jnp.cumsum(rows_by_dst_expert, axis=1, dtype=jnp.int32) - rows_by_dst_expert).reshape(
        ep_size * local_experts
    )
    source_base_by_src_dst = (jnp.cumsum(clipped_counts, axis=0, dtype=jnp.int32) - clipped_counts).reshape(
        ep_size * ep_size * local_experts
    )

    def body(
        x_ref,
        assignment_ids_ref,
        token_ids_ref,
        source_expert_offsets_ref,
        expert_base_by_dst_ref,
        source_base_by_src_dst_ref,
        peer_phase_order_ref,
        rhs_ref,
        hidden_ref,
        recv_src_rank_ref,
        recv_assignment_ref,
        scratch_ref,
    ):
        arrivals_sem = pl.get_global(mgpu.SemaphoreType.REGULAR)
        rank = lax.axis_index(expert_axis)
        sm_id = lax.axis_index("sm")
        wg_id = lax.axis_index("wg") if config.dispatch_split_wg_permute_up else jnp.int32(0)
        is_compute_wg = wg_id == jnp.int32(0)
        is_overlap_remote_wg = wg_id == jnp.int32(1)
        is_loader_wg = wg_id == jnp.int32(2 if config.dispatch_split_wg_overlap_permute_up else 1)
        is_remote_wg = wg_id == jnp.int32(1 if config.dispatch_split_wg_permute_up else 0)
        worker_id = sm_id

        def _remote_row(dst, src_rank, expert, local_pos):
            expert_base_idx = dst * local_experts + expert
            source_base_idx = (src_rank * ep_size + dst) * local_experts + expert
            return expert_base_by_dst_ref[expert_base_idx] + source_base_by_src_dst_ref[source_base_idx] + local_pos

        def _phase_indices(global_phase):
            return _static_workqueue_phase_indices(
                global_phase,
                rank,
                ep_size=ep_size,
                expert_groups=expert_groups,
                dispatch_copy_schedule=config.dispatch_copy_schedule,
                peer_phase_order=peer_phase_order_ref,
                swizzle_expert_groups=config.dispatch_swizzle_expert_groups,
            )

        def _signal_phase_copied(global_phase):
            expert_group, peer_phase = _phase_indices(global_phase)
            del expert_group
            dst = (rank + peer_phase) % ep_size
            mgpu.semaphore_signal_parallel(
                mgpu.SemaphoreSignal(arrivals_sem, device_id=(jnp.int32(0), dst, jnp.int32(0))),
            )

        def _copy_phase_step(global_phase, step):
            expert_group, peer_phase = _phase_indices(global_phase)
            expert_group_start = expert_group * expert_group_size
            dst = (rank + peer_phase) % ep_size
            scratch_slot = global_phase
            linear_tile = step * workers + worker_id

            @pl.when(linear_tile < copy_tiles_per_chunk)
            def _copy_tile():
                chunk_row_group = linear_tile // d_copy_tiles
                d_tile = linear_tile - chunk_row_group * d_copy_tiles
                chunk_row_start = chunk_row_group * copy_rows
                d_start = d_tile * copy_tile
                remote_scratch_ref = mgpu.remote_ref(
                    scratch_ref,
                    dst,
                    device_id_type=pl.DeviceIdType.LOGICAL,
                )

                def _copy_one_row(chunk_row):
                    local_expert_in_group = chunk_row // rows_per_source_expert
                    local_pos = chunk_row - local_expert_in_group * rows_per_source_expert
                    expert = expert_group_start + local_expert_in_group
                    global_expert = dst * local_experts + expert
                    token_offset = source_expert_offsets_ref[global_expert] + local_pos
                    token_id = token_ids_ref[token_offset]
                    remote_scratch_ref[scratch_slot, chunk_row, pl.ds(d_start, copy_tile)] = x_ref[
                        token_id, pl.ds(d_start, copy_tile)
                    ]

                    @pl.when(d_tile == 0)
                    def _write_metadata():
                        remote_row = _remote_row(dst, rank, expert, local_pos)
                        remote_src_rank_ref = mgpu.remote_ref(
                            recv_src_rank_ref,
                            dst,
                            device_id_type=pl.DeviceIdType.LOGICAL,
                        )
                        remote_assignment_ref = mgpu.remote_ref(
                            recv_assignment_ref,
                            dst,
                            device_id_type=pl.DeviceIdType.LOGICAL,
                        )
                        remote_src_rank_ref[remote_row] = rank
                        remote_assignment_ref[remote_row] = assignment_ids_ref[token_offset]

                if copy_rows == 1:
                    _copy_one_row(chunk_row_start)
                elif config.dispatch_chunk_vectorized_copy_rows:
                    local_expert_in_group = chunk_row_start // rows_per_source_expert
                    local_pos = chunk_row_start - local_expert_in_group * rows_per_source_expert
                    expert = expert_group_start + local_expert_in_group
                    global_expert = dst * local_experts + expert
                    token_offset = source_expert_offsets_ref[global_expert] + local_pos
                    token_ids = token_ids_ref[pl.ds(token_offset, copy_rows)]
                    remote_scratch_ref[
                        scratch_slot,
                        pl.ds(chunk_row_start, copy_rows),
                        pl.ds(d_start, copy_tile),
                    ] = x_ref[token_ids, pl.ds(d_start, copy_tile)]

                    @pl.when(d_tile == 0)
                    def _write_metadata_vector():
                        remote_row_start = _remote_row(dst, rank, expert, local_pos)
                        remote_src_rank_ref = mgpu.remote_ref(
                            recv_src_rank_ref,
                            dst,
                            device_id_type=pl.DeviceIdType.LOGICAL,
                        )
                        remote_assignment_ref = mgpu.remote_ref(
                            recv_assignment_ref,
                            dst,
                            device_id_type=pl.DeviceIdType.LOGICAL,
                        )
                        remote_src_rank_ref[pl.ds(remote_row_start, copy_rows)] = jnp.full(
                            (copy_rows,),
                            rank,
                            dtype=jnp.int32,
                        )
                        remote_assignment_ref[pl.ds(remote_row_start, copy_rows)] = assignment_ids_ref[
                            pl.ds(token_offset, copy_rows)
                        ]

                else:

                    @pl.loop(0, copy_rows)
                    def _copy_row(row_offset):
                        _copy_one_row(chunk_row_start + row_offset)

        def _copy_phase(global_phase):
            @pl.loop(0, copy_steps)
            def _copy_step(step):
                _copy_phase_step(global_phase, step)

            _signal_phase_copied(global_phase)

        def _compute_phase(global_phase):
            expert_group, peer_phase = _phase_indices(global_phase)
            expert_group_start = expert_group * expert_group_size
            scratch_slot = global_phase
            expected_arrivals = ep_size * workers + (global_phase + 1) * workers
            pl.semaphore_wait(arrivals_sem, value=expected_arrivals, decrement=False)

            src = (rank + ep_size - peer_phase) % ep_size

            @mgpu.nd_loop((chunk_grid_m * chunk_grid_n,), collective_axes="sm")
            def _compute_chunk(loop_info):
                mi, ni = mgpu.planar_snake(
                    loop_info.index[0],
                    (chunk_grid_m, chunk_grid_n),
                    1,
                    config.grid_block_n,
                )
                local_expert_in_group = mi // blocks_per_source_expert
                block_in_expert = mi - local_expert_in_group * blocks_per_source_expert
                expert = expert_group_start + local_expert_in_group
                local_pos_start = block_in_expert * config.block_m
                chunk_block_start = mi * config.block_m

                def acc_scope(gate_acc_ref, up_acc_ref):
                    def wgmma_step(_, lhs_smem, gate_smem, up_smem):
                        mgpu.wgmma(gate_acc_ref, lhs_smem, gate_smem)
                        mgpu.wgmma(up_acc_ref, lhs_smem, up_smem)

                    mgpu.emit_pipeline(
                        wgmma_step,
                        grid=(hidden_dim // config.block_k,),
                        in_specs=[
                            mgpu.BlockSpec(
                                (config.block_m, config.block_k),
                                lambda kk: (chunk_block_start // config.block_m, kk),
                                delay_release=1,
                            ),
                            mgpu.BlockSpec(
                                (config.block_k, config.block_n),
                                lambda kk: (kk, ni),
                                delay_release=1,
                            ),
                            mgpu.BlockSpec(
                                (config.block_k, config.block_n),
                                lambda kk: (kk, ni + intermediate // config.block_n),
                                delay_release=1,
                            ),
                        ],
                        max_concurrent_steps=config.max_concurrent_steps,
                    )(
                        scratch_ref.at[scratch_slot],
                        rhs_ref.at[expert],
                        rhs_ref.at[expert],
                    )
                    return activation_fn(gate_acc_ref[...]) * up_acc_ref[...]

                hidden = pl.run_scoped(
                    acc_scope,
                    gate_acc_ref=mgpu.ACC((config.block_m, config.block_n)),
                    up_acc_ref=mgpu.ACC((config.block_m, config.block_n)),
                )

                remote_row_start = _remote_row(rank, src, expert, local_pos_start)
                hidden_ref[
                    pl.ds(remote_row_start, config.block_m),
                    pl.ds(ni * config.block_n, config.block_n),
                ] = hidden.astype(hidden_ref.dtype)

        def _compute_phase_manual_split(global_phase):
            expert_group, peer_phase = _phase_indices(global_phase)
            expert_group_start = expert_group * expert_group_size
            scratch_slot = global_phase
            expected_arrivals = ep_size * workers + (global_phase + 1) * workers
            pl.semaphore_wait(arrivals_sem, value=expected_arrivals, decrement=False)

            src = (rank + ep_size - peer_phase) % ep_size

            @pl.when(is_overlap_remote_wg & config.dispatch_split_wg_overlap_permute_up)
            def _copy_next_phase():
                next_phase = global_phase + 1

                @pl.when(next_phase < total_phases)
                def _copy_next():
                    _copy_phase(next_phase)

            @mgpu.nd_loop((chunk_grid_m * chunk_grid_n,), collective_axes="sm")
            def _compute_chunk(loop_info):
                mi, ni = mgpu.planar_snake(
                    loop_info.index[0],
                    (chunk_grid_m, chunk_grid_n),
                    1,
                    config.grid_block_n,
                )
                local_expert_in_group = mi // blocks_per_source_expert
                block_in_expert = mi - local_expert_in_group * blocks_per_source_expert
                expert = expert_group_start + local_expert_in_group
                local_pos_start = block_in_expert * config.block_m
                chunk_block_start = mi * config.block_m
                k_tiles = hidden_dim // config.block_k

                def manual_pipeline_scope(lhs_smem, gate_smem, up_smem, ready_barrier, consumed_barrier):
                    @pl.when(is_loader_wg)
                    def _load_w13_operands():
                        @pl.loop(0, k_tiles)
                        def _load_k(kk):
                            slot = kk % split_wg_pipeline_stages

                            @pl.when(kk >= split_wg_pipeline_stages)
                            def _wait_consumed():
                                mgpu.barrier_wait(consumed_barrier.at[slot])

                            mgpu.copy_gmem_to_smem(
                                scratch_ref.at[
                                    scratch_slot,
                                    pl.ds(chunk_block_start, config.block_m),
                                    pl.ds(kk * config.block_k, config.block_k),
                                ],
                                lhs_smem.at[slot],
                                ready_barrier.at[slot],
                            )
                            mgpu.copy_gmem_to_smem(
                                rhs_ref.at[
                                    expert,
                                    pl.ds(kk * config.block_k, config.block_k),
                                    pl.ds(ni * config.block_n, config.block_n),
                                ],
                                gate_smem.at[slot],
                                ready_barrier.at[slot],
                            )
                            mgpu.copy_gmem_to_smem(
                                rhs_ref.at[
                                    expert,
                                    pl.ds(kk * config.block_k, config.block_k),
                                    pl.ds((ni + intermediate // config.block_n) * config.block_n, config.block_n),
                                ],
                                up_smem.at[slot],
                                ready_barrier.at[slot],
                            )

                        @pl.loop(0, split_wg_pipeline_stages)
                        def _wait_tail_consumed(tail):
                            @pl.when(tail < jnp.minimum(split_wg_pipeline_stages, k_tiles))
                            def _wait_tail_slot():
                                kk = k_tiles - tail - 1
                                slot = kk % split_wg_pipeline_stages
                                mgpu.barrier_wait(consumed_barrier.at[slot])

                    @pl.when(is_compute_wg)
                    def _compute_w13_tile():
                        def acc_scope(gate_acc_ref, up_acc_ref):
                            @pl.loop(0, k_tiles)
                            def _wgmma_k(kk):
                                slot = kk % split_wg_pipeline_stages
                                mgpu.barrier_wait(ready_barrier.at[slot])
                                mgpu.wgmma(gate_acc_ref, lhs_smem.at[slot], gate_smem.at[slot])
                                mgpu.wgmma(up_acc_ref, lhs_smem.at[slot], up_smem.at[slot])
                                mgpu.wgmma_wait(0)
                                mgpu.barrier_arrive(consumed_barrier.at[slot])

                            hidden = activation_fn(gate_acc_ref[...]) * up_acc_ref[...]
                            remote_row_start = _remote_row(rank, src, expert, local_pos_start)
                            hidden_ref[
                                pl.ds(remote_row_start, config.block_m),
                                pl.ds(ni * config.block_n, config.block_n),
                            ] = hidden.astype(hidden_ref.dtype)

                        pl.run_scoped(
                            acc_scope,
                            gate_acc_ref=mgpu.ACC((config.block_m, config.block_n)),
                            up_acc_ref=mgpu.ACC((config.block_m, config.block_n)),
                        )

                pl.run_scoped(
                    manual_pipeline_scope,
                    lhs_smem=mgpu.SMEM(
                        (split_wg_pipeline_stages, config.block_m, config.block_k),
                        dtype=scratch_ref.dtype,
                    ),
                    gate_smem=mgpu.SMEM(
                        (split_wg_pipeline_stages, config.block_k, config.block_n),
                        dtype=rhs_ref.dtype,
                    ),
                    up_smem=mgpu.SMEM(
                        (split_wg_pipeline_stages, config.block_k, config.block_n),
                        dtype=rhs_ref.dtype,
                    ),
                    ready_barrier=mgpu.Barrier(
                        num_arrivals=3,
                        num_barriers=split_wg_pipeline_stages,
                    ),
                    consumed_barrier=mgpu.Barrier(
                        num_arrivals=1,
                        num_barriers=split_wg_pipeline_stages,
                    ),
                    collective_axes="wg",
                )

        def _compute_phase_warp_specialized(global_phase):
            expert_group, peer_phase = _phase_indices(global_phase)
            expert_group_start = expert_group * expert_group_size
            scratch_slot = global_phase
            expected_arrivals = ep_size * workers + (global_phase + 1) * workers
            pl.semaphore_wait(arrivals_sem, value=expected_arrivals, decrement=False)

            src = (rank + ep_size - peer_phase) % ep_size

            @mgpu.nd_loop((chunk_grid_m * chunk_grid_n,), collective_axes="sm")
            def _compute_chunk(loop_info):
                mi, ni = mgpu.planar_snake(
                    loop_info.index[0],
                    (chunk_grid_m, chunk_grid_n),
                    1,
                    config.grid_block_n,
                )
                local_expert_in_group = mi // blocks_per_source_expert
                block_in_expert = mi - local_expert_in_group * blocks_per_source_expert
                expert = expert_group_start + local_expert_in_group
                local_pos_start = block_in_expert * config.block_m
                chunk_block_start = mi * config.block_m
                k_tiles = hidden_dim // config.block_k

                def _pipeline_body_manual_consumed(
                    indices,
                    lhs_smem,
                    gate_smem,
                    up_smem,
                    lhs_consumed_barrier,
                    gate_consumed_barrier,
                    up_consumed_barrier,
                    acc_refs,
                ):
                    (kk,) = indices
                    gate_acc_ref, up_acc_ref = acc_refs

                    @pl.when(is_compute_wg)
                    def _compute_wgmma():
                        mgpu.wgmma(gate_acc_ref, lhs_smem, gate_smem)
                        mgpu.wgmma(up_acc_ref, lhs_smem, up_smem)
                        mgpu.wgmma_wait(0)

                    @pl.when(is_overlap_remote_wg & config.dispatch_split_wg_overlap_permute_up)
                    def _copy_next_phase_step():
                        next_phase = global_phase + 1

                        @pl.when(next_phase < total_phases)
                        def _copy_next():
                            base_copy_step = (loop_info.local_index * k_tiles + kk) * overlap_copy_steps_per_k_step

                            @pl.loop(0, overlap_copy_steps_per_k_step)
                            def _copy_step(extra_step):
                                _copy_phase_step(next_phase, base_copy_step + extra_step)

                            num_local_steps = loop_info.num_local_steps
                            assert num_local_steps is not None

                            @pl.when((loop_info.local_index == num_local_steps - 1) & (kk == k_tiles - 1))
                            def _signal_next_phase():
                                _signal_phase_copied(next_phase)

                    mgpu.barrier_arrive(lhs_consumed_barrier)
                    mgpu.barrier_arrive(gate_consumed_barrier)
                    mgpu.barrier_arrive(up_consumed_barrier)
                    return gate_acc_ref, up_acc_ref

                def _pipeline_body(indices, lhs_smem, gate_smem, up_smem, acc_refs):
                    (kk,) = indices
                    gate_acc_ref, up_acc_ref = acc_refs
                    mgpu.wgmma(gate_acc_ref, lhs_smem, gate_smem)
                    mgpu.wgmma(up_acc_ref, lhs_smem, up_smem)
                    mgpu.wgmma_wait(0)

                    @pl.when(config.dispatch_split_wg_overlap_permute_up)
                    def _copy_next_phase_step():
                        next_phase = global_phase + 1

                        @pl.when(next_phase < total_phases)
                        def _copy_next():
                            copy_step = loop_info.local_index * k_tiles + kk
                            _copy_phase_step(next_phase, copy_step)
                            num_local_steps = loop_info.num_local_steps
                            assert num_local_steps is not None

                            @pl.when((loop_info.local_index == num_local_steps - 1) & (kk == k_tiles - 1))
                            def _signal_next_phase():
                                _signal_phase_copied(next_phase)

                    return gate_acc_ref, up_acc_ref

                def _compute_context(eval_pipeline):
                    def _acc_scope(gate_acc_ref, up_acc_ref):
                        gate_acc_ref, up_acc_ref = eval_pipeline((gate_acc_ref, up_acc_ref))

                        @pl.when(is_compute_wg)
                        def _store_hidden():
                            hidden = activation_fn(gate_acc_ref[...]) * up_acc_ref[...]
                            remote_row_start = _remote_row(rank, src, expert, local_pos_start)
                            hidden_ref[
                                pl.ds(remote_row_start, config.block_m),
                                pl.ds(ni * config.block_n, config.block_n),
                            ] = hidden.astype(hidden_ref.dtype)

                    pl.run_scoped(
                        _acc_scope,
                        gate_acc_ref=mgpu.ACC((config.block_m, config.block_n)),
                        up_acc_ref=mgpu.ACC((config.block_m, config.block_n)),
                    )

                delay_release = 1
                pipeline_body = (
                    _pipeline_body_manual_consumed if config.dispatch_split_wg_overlap_permute_up else _pipeline_body
                )
                pipeline = mgpu.emit_pipeline_warp_specialized(
                    pipeline_body,
                    grid=(hidden_dim // config.block_k,),
                    memory_registers=40,
                    in_specs=[
                        mgpu.BlockSpec(
                            (config.block_m, config.block_k),
                            lambda kk: (chunk_block_start // config.block_m, kk),
                            delay_release=delay_release,
                        ),
                        mgpu.BlockSpec(
                            (config.block_k, config.block_n),
                            lambda kk: (kk, ni),
                            delay_release=delay_release,
                        ),
                        mgpu.BlockSpec(
                            (config.block_k, config.block_n),
                            lambda kk: (kk, ni + intermediate // config.block_n),
                            delay_release=delay_release,
                        ),
                    ],
                    max_concurrent_steps=config.max_concurrent_steps,
                    wg_axis="wg",
                    num_compute_wgs=2 if config.dispatch_split_wg_overlap_permute_up else 1,
                    manual_consumed_barriers=config.dispatch_split_wg_overlap_permute_up,
                    compute_context=_compute_context,
                )
                pipeline(
                    scratch_ref.at[scratch_slot],
                    rhs_ref.at[expert],
                    rhs_ref.at[expert],
                )

        def _init_copy_outputs():
            @pl.loop(0, init_metadata_steps)
            def _init_metadata_step(step):
                row = step * workers + worker_id

                @pl.when(row < capacity)
                def _init_row():
                    recv_src_rank_ref[row] = jnp.int32(-1)
                    recv_assignment_ref[row] = jnp.int32(-1)

            @pl.loop(0, init_hidden_steps)
            def _init_hidden_step(step):
                linear_tile = step * workers + worker_id

                @pl.when(linear_tile < init_hidden_tiles)
                def _zero_hidden_tile():
                    row = assignments + linear_tile // chunk_grid_n
                    n_tile = linear_tile % chunk_grid_n
                    hidden_ref[row, pl.ds(n_tile * config.block_n, config.block_n)] = jnp.zeros(
                        (config.block_n,), dtype=hidden_ref.dtype
                    )

            @pl.loop(0, ep_size)
            def _signal_init(peer):
                pl.semaphore_signal(arrivals_sem, device_id=peer, device_id_type=pl.DeviceIdType.LOGICAL)

        def _run_split_wg():
            @pl.when(is_remote_wg)
            def _copy_first_phase():
                _init_copy_outputs()
                pl.semaphore_wait(arrivals_sem, value=ep_size * workers, decrement=False)
                _copy_phase(jnp.int32(0))

            @pl.loop(0, total_phases)
            def _phase_compute_loop(global_phase):
                @pl.when(is_remote_wg & ~config.dispatch_split_wg_overlap_permute_up & (global_phase > 0))
                def _copy_current_phase():
                    _copy_phase(global_phase)

                _compute_phase_manual_split(global_phase)

        def _run_serial_wg():
            _init_copy_outputs()
            pl.semaphore_wait(arrivals_sem, value=ep_size * workers, decrement=False)

            _copy_phase(jnp.int32(0))

            @pl.loop(0, total_phases)
            def _phase_loop(global_phase):
                next_phase = global_phase + 1

                @pl.when(next_phase < total_phases)
                def _prefetch_next():
                    _copy_phase(next_phase)

                _compute_phase(global_phase)

        if config.dispatch_split_wg_permute_up:
            _run_split_wg()

        else:
            _run_serial_wg()

    kernel = mgpu.kernel(
        body,
        out_shape=[
            jax.ShapeDtypeStruct((capacity, intermediate), x_local.dtype),
            jax.ShapeDtypeStruct((capacity,), jnp.int32),
            jax.ShapeDtypeStruct((capacity,), jnp.int32),
            jax.ShapeDtypeStruct((scratch_slots, chunk_rows, hidden_dim), x_local.dtype),
        ],
        grid=(num_sms,),
        grid_names=("sm",),
        num_threads=split_wg_threads if config.dispatch_split_wg_permute_up else None,
        thread_name="wg" if config.dispatch_split_wg_permute_up else None,
        compiler_params=mgpu.CompilerParams(
            lowering_semantics=mgpu.LoweringSemantics.Warpgroup,
        ),
    )
    hidden, recv_src_rank, recv_src_assignment, _scratch = kernel(
        x_local,
        assignment_ids_sorted,
        token_ids_sorted,
        source_expert_offsets,
        expert_base_by_dst,
        source_base_by_src_dst,
        peer_phase_order,
        moe_w13,
    )
    return hidden, recv_src_rank, recv_src_assignment


def local_producer_consumer_copy_mgpu(
    x: Float[Array, "T D"],
    *,
    config: MoeMgpuConfig = MoeMgpuConfig(),
) -> Float[Array, "T D"]:
    """Diagnostic split-WG local copy kernel for producer/consumer barrier experiments."""
    rows, hidden_dim = x.shape
    if rows % config.block_m != 0:
        raise ValueError(f"rows={rows} must be divisible by block_m={config.block_m}")
    if hidden_dim % config.block_k != 0:
        raise ValueError(f"D={hidden_dim} must be divisible by block_k={config.block_k}")

    num_sms = config.num_sms
    if num_sms is None:
        num_sms = jax.devices()[0].core_count

    row_tiles = rows // config.block_m
    col_tiles = hidden_dim // config.block_k

    def body(x_ref, y_ref):
        wg_id = lax.axis_index("wg")
        is_consumer_wg = wg_id == jnp.int32(0)
        is_producer_wg = wg_id == jnp.int32(1)

        @mgpu.nd_loop((row_tiles * col_tiles,), collective_axes="sm")
        def _copy_tile(loop_info):
            linear_tile = loop_info.index[0]
            row_tile = linear_tile // col_tiles
            col_tile = linear_tile - row_tile * col_tiles
            row_start = row_tile * config.block_m
            col_start = col_tile * config.block_k

            def _scope(tile_smem, ready_barrier, consumed_barrier):
                @pl.when(is_producer_wg)
                def _produce():
                    mgpu.copy_gmem_to_smem(
                        x_ref.at[pl.ds(row_start, config.block_m), pl.ds(col_start, config.block_k)],
                        tile_smem,
                        ready_barrier,
                    )
                    mgpu.barrier_wait(consumed_barrier)

                @pl.when(is_consumer_wg)
                def _consume():
                    mgpu.barrier_wait(ready_barrier)
                    mgpu.copy_smem_to_gmem(
                        tile_smem,
                        y_ref.at[pl.ds(row_start, config.block_m), pl.ds(col_start, config.block_k)],
                    )
                    mgpu.wait_smem_to_gmem(0, wait_read_only=False)
                    mgpu.barrier_arrive(consumed_barrier)

            pl.run_scoped(
                _scope,
                tile_smem=mgpu.SMEM((config.block_m, config.block_k), dtype=x_ref.dtype),
                ready_barrier=mgpu.Barrier(num_arrivals=1),
                consumed_barrier=mgpu.Barrier(num_arrivals=1),
                collective_axes="wg",
            )

    kernel = mgpu.kernel(
        body,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(num_sms,),
        grid_names=("sm",),
        num_threads=2,
        thread_name="wg",
        compiler_params=mgpu.CompilerParams(
            lowering_semantics=mgpu.LoweringSemantics.Warpgroup,
        ),
    )
    return kernel(x)


def local_split_wg_w13_mgpu(
    x: Float[Array, "T D"],
    w_up_gate: Float[Array, "D twoI"],
    activation_fn: Callable[[jax.Array], jax.Array],
    *,
    config: MoeMgpuConfig = MoeMgpuConfig(),
) -> Float[Array, "T I"]:
    """Diagnostic split-WG local W13 kernel for producer/consumer WGMMA experiments."""
    rows, hidden_dim = x.shape
    hidden_dim_w, intermediate2 = w_up_gate.shape
    if hidden_dim_w != hidden_dim:
        raise ValueError(f"w_up_gate hidden dim {hidden_dim_w} must match x hidden dim {hidden_dim}")
    intermediate = intermediate2 // 2
    if intermediate2 != 2 * intermediate:
        raise ValueError(f"w_up_gate output dim must be even, got {intermediate2}")
    if rows % config.block_m != 0:
        raise ValueError(f"rows={rows} must be divisible by block_m={config.block_m}")
    if hidden_dim % config.block_k != 0:
        raise ValueError(f"D={hidden_dim} must be divisible by block_k={config.block_k}")
    if intermediate % config.block_n != 0:
        raise ValueError(f"I={intermediate} must be divisible by block_n={config.block_n}")

    num_sms = config.num_sms
    if num_sms is None:
        num_sms = jax.devices()[0].core_count

    row_tiles = rows // config.block_m
    n_tiles = intermediate // config.block_n
    k_tiles = hidden_dim // config.block_k
    pipeline_stages = 2
    tail_slots = min(pipeline_stages, k_tiles)

    def body(x_ref, w_ref, y_ref):
        wg_id = lax.axis_index("wg")
        is_compute_wg = wg_id == jnp.int32(0)
        is_loader_wg = wg_id == jnp.int32(1)

        @mgpu.nd_loop((row_tiles * n_tiles,), collective_axes="sm")
        def _compute_tile(loop_info):
            mi, ni = mgpu.planar_snake(
                loop_info.index[0],
                (row_tiles, n_tiles),
                1,
                config.grid_block_n,
            )
            row_start = mi * config.block_m
            n_start = ni * config.block_n
            up_n_start = intermediate + n_start

            def _scope(lhs_smem, gate_smem, up_smem, ready_barrier, consumed_barrier):
                @pl.when(is_loader_wg)
                def _load_tiles():
                    @pl.loop(0, k_tiles)
                    def _load_k(kk):
                        slot = kk % pipeline_stages

                        @pl.when(kk >= pipeline_stages)
                        def _wait_consumed_before_reuse():
                            mgpu.barrier_wait(consumed_barrier.at[slot])

                        k_start = kk * config.block_k
                        mgpu.copy_gmem_to_smem(
                            x_ref.at[pl.ds(row_start, config.block_m), pl.ds(k_start, config.block_k)],
                            lhs_smem.at[slot],
                            ready_barrier.at[slot],
                        )
                        mgpu.copy_gmem_to_smem(
                            w_ref.at[pl.ds(k_start, config.block_k), pl.ds(n_start, config.block_n)],
                            gate_smem.at[slot],
                            ready_barrier.at[slot],
                        )
                        mgpu.copy_gmem_to_smem(
                            w_ref.at[pl.ds(k_start, config.block_k), pl.ds(up_n_start, config.block_n)],
                            up_smem.at[slot],
                            ready_barrier.at[slot],
                        )

                    @pl.loop(0, tail_slots)
                    def _wait_tail_consumed(tail):
                        kk = k_tiles - tail - 1
                        slot = kk % pipeline_stages
                        mgpu.barrier_wait(consumed_barrier.at[slot])

                @pl.when(is_compute_wg)
                def _compute_w13():
                    def _acc_scope(gate_acc_ref, up_acc_ref):
                        @pl.loop(0, k_tiles)
                        def _wgmma_k(kk):
                            slot = kk % pipeline_stages
                            mgpu.barrier_wait(ready_barrier.at[slot])
                            mgpu.wgmma(gate_acc_ref, lhs_smem.at[slot], gate_smem.at[slot])
                            mgpu.wgmma(up_acc_ref, lhs_smem.at[slot], up_smem.at[slot])
                            mgpu.wgmma_wait(0)
                            mgpu.barrier_arrive(consumed_barrier.at[slot])

                        hidden = activation_fn(gate_acc_ref[...]) * up_acc_ref[...]
                        y_ref[
                            pl.ds(row_start, config.block_m),
                            pl.ds(n_start, config.block_n),
                        ] = hidden.astype(y_ref.dtype)

                    pl.run_scoped(
                        _acc_scope,
                        gate_acc_ref=mgpu.ACC((config.block_m, config.block_n)),
                        up_acc_ref=mgpu.ACC((config.block_m, config.block_n)),
                    )

            pl.run_scoped(
                _scope,
                lhs_smem=mgpu.SMEM(
                    (pipeline_stages, config.block_m, config.block_k),
                    dtype=x_ref.dtype,
                ),
                gate_smem=mgpu.SMEM(
                    (pipeline_stages, config.block_k, config.block_n),
                    dtype=w_ref.dtype,
                ),
                up_smem=mgpu.SMEM(
                    (pipeline_stages, config.block_k, config.block_n),
                    dtype=w_ref.dtype,
                ),
                ready_barrier=mgpu.Barrier(
                    num_arrivals=3,
                    num_barriers=pipeline_stages,
                ),
                consumed_barrier=mgpu.Barrier(
                    num_arrivals=1,
                    num_barriers=pipeline_stages,
                ),
                collective_axes="wg",
            )

    kernel = mgpu.kernel(
        body,
        out_shape=jax.ShapeDtypeStruct((rows, intermediate), x.dtype),
        grid=(num_sms,),
        grid_names=("sm",),
        num_threads=2,
        thread_name="wg",
        compiler_params=mgpu.CompilerParams(
            lowering_semantics=mgpu.LoweringSemantics.Warpgroup,
        ),
    )
    return kernel(x, w_up_gate)


def ragged_w2_reference(
    hidden: Float[Array, "capacity I"],
    moe_w2: Float[Array, "Elocal I D"],
    rows_per_expert: Int[Array, "Elocal"],
) -> Float[Array, "capacity D"]:
    """Reference local expert W2 over the expert-major receive layout."""
    compute_group_sizes = _group_sizes_with_padding(rows_per_expert, total_size=hidden.shape[0])
    return ragged_dot(hidden, moe_w2, compute_group_sizes).astype(hidden.dtype)


def ragged_w2_mgpu(
    hidden: Float[Array, "capacity I"],
    moe_w2: Float[Array, "Elocal I D"],
    rows_per_expert: Int[Array, "Elocal"],
    *,
    config: MoeMgpuConfig = MoeMgpuConfig(),
) -> Float[Array, "capacity D"]:
    """Local Pallas MGPU grouped W2 kernel over expert-major hidden rows."""
    local_experts, intermediate, hidden_dim = moe_w2.shape
    capacity = hidden.shape[0]
    if hidden.shape[1] != intermediate:
        raise ValueError(f"hidden dimension {hidden.shape[1]} must match moe_w2 input dimension {intermediate}")
    if intermediate % config.block_k != 0:
        raise ValueError(f"I={intermediate} must be divisible by block_k={config.block_k}")
    if hidden_dim % config.block_n != 0:
        raise ValueError(f"D={hidden_dim} must be divisible by block_n={config.block_n}")

    block_m = config.block_m
    block_n = config.block_n
    block_k = config.block_k
    max_concurrent_steps = config.max_concurrent_steps
    grid_block_n = config.grid_block_n

    compute_group_sizes = _group_sizes_with_padding(rows_per_expert, total_size=capacity)

    def body(rows_per_expert_gmem, lhs_gmem, rhs_gmem, o_gmem):
        grid_m = pl.cdiv(capacity, block_m) + local_experts - 1
        grid_n = pl.cdiv(hidden_dim, block_n)
        grid = (grid_m * grid_n,)

        @mgpu.nd_loop(grid, collective_axes="sm")
        def mn_loop(loop_info):
            mi, ni = mgpu.planar_snake(
                loop_info.index[0],
                (grid_m, grid_n),
                1,
                grid_block_n,
            )

            group_info = GroupInfo.create(
                rows_per_expert_gmem,
                block_m,
                mi,
            )

            def acc_scope(acc_ref):
                def wgmma_step(_, lhs_smem, rhs_smem):
                    mgpu.wgmma(
                        acc_ref,
                        lhs_smem,
                        rhs_smem,
                    )

                mgpu.emit_pipeline(
                    wgmma_step,
                    grid=(intermediate // block_k,),
                    in_specs=[
                        mgpu.BlockSpec(
                            (block_m, block_k),
                            lambda kk: (group_info.block, kk),
                            delay_release=1,
                        ),
                        mgpu.BlockSpec(
                            (block_k, block_n),
                            lambda kk: (kk, ni),
                            delay_release=1,
                        ),
                    ],
                    max_concurrent_steps=max_concurrent_steps,
                )(
                    lhs_gmem,
                    rhs_gmem.at[group_info.group_id],
                )

                return acc_ref[...]

            out_tile = pl.run_scoped(
                acc_scope,
                acc_ref=mgpu.ACC((block_m, block_n)),
            )

            @functools.partial(pl.run_scoped, o_smem=mgpu.SMEM((block_m, block_n), dtype=o_gmem.dtype))
            def store_scope(o_smem):
                o_smem[...] = out_tile.astype(o_smem.dtype)
                mgpu.commit_smem()

                smem_start = group_info.start_within_block
                remaining_rows = min(block_m, capacity)

                while remaining_rows > 0:
                    const_rows_len = 1 << int(math.log2(remaining_rows))
                    remaining_rows //= 2

                    @pl.when(group_info.actual_size & const_rows_len != 0)
                    def _():
                        o_smem_slice = o_smem.at[pl.ds(smem_start, const_rows_len)]
                        o_gref_slice = o_gmem.at[
                            pl.ds(
                                group_info.block_start + smem_start,
                                const_rows_len,
                            ),
                            pl.ds(ni * block_n, block_n),
                        ]
                        mgpu.copy_smem_to_gmem(o_smem_slice, o_gref_slice)

                    smem_start += group_info.actual_size & const_rows_len

                mgpu.wait_smem_to_gmem(0, wait_read_only=True)

    num_sms = config.num_sms
    if num_sms is None:
        num_sms = jax.devices()[0].core_count
    kernel = mgpu.kernel(
        body,
        out_shape=jax.ShapeDtypeStruct((capacity, hidden_dim), hidden.dtype),
        grid=(num_sms,),
        grid_names=("sm",),
        compiler_params=mgpu.CompilerParams(
            lowering_semantics=mgpu.LoweringSemantics.Warpgroup,
        ),
    )
    return kernel(compute_group_sizes, hidden, moe_w2)


def unpermute_mgpu_reference(
    y_dispatch: Float[Array, "capacity D"],
    recv_src_rank: Int[Array, "capacity"],
    recv_src_assignment: Int[Array, "capacity"],
    combine_weights: Float[Array, "Tlocal K"],
    *,
    expert_axis: str = "expert",
) -> Float[Array, "Tlocal D"]:
    """Reference return/unpermute and deterministic fixed-slot combine."""
    rank = lax.axis_index(expert_axis)
    ep_size = int(lax.axis_size(expert_axis))
    tokens, topk = combine_weights.shape
    hidden_dim = y_dispatch.shape[1]
    assignments = tokens * topk

    y_by_rank = lax.all_gather(y_dispatch, expert_axis)
    src_rank_by_rank = lax.all_gather(recv_src_rank, expert_axis)
    assignment_by_rank = lax.all_gather(recv_src_assignment, expert_axis)

    return_slots = jnp.zeros((assignments, hidden_dim), dtype=y_dispatch.dtype)
    for expert_rank in range(ep_size):
        source_matches = src_rank_by_rank[expert_rank] == rank
        assignment = assignment_by_rank[expert_rank]
        valid_assignment = (assignment >= 0) & (assignment < assignments)
        keep = source_matches & valid_assignment
        safe_assignment = jnp.where(valid_assignment, assignment, 0)
        values = jnp.where(keep[:, None], y_by_rank[expert_rank], jnp.zeros_like(y_by_rank[expert_rank]))
        return_slots = return_slots.at[safe_assignment].add(values)

    return_slots = return_slots.reshape(tokens, topk, hidden_dim)
    out = jnp.zeros((tokens, hidden_dim), dtype=y_dispatch.dtype)
    weights = combine_weights.astype(y_dispatch.dtype)
    for route_slot in range(topk):
        out = out + return_slots[:, route_slot, :] * weights[:, route_slot, None]
    return out


def unpermute_mgpu(
    y_dispatch: Float[Array, "capacity D"],
    recv_src_rank: Int[Array, "capacity"],
    recv_src_assignment: Int[Array, "capacity"],
    combine_weights: Float[Array, "Tlocal K"],
    *,
    expert_axis: str = "expert",
    config: MoeMgpuConfig = MoeMgpuConfig(),
) -> Float[Array, "Tlocal D"]:
    """Remote return slots plus deterministic local fixed-route-slot combine."""
    capacity, hidden_dim = y_dispatch.shape
    tokens, topk = combine_weights.shape
    assignments = tokens * topk
    ep_size = int(lax.axis_size(expert_axis))
    num_sms = config.num_sms
    if num_sms is None:
        num_sms = jax.devices()[0].core_count

    def body(y_ref, src_rank_ref, assignment_ref, weights_ref, out_ref, return_slots_ref):
        arrivals_sem = pl.get_global(mgpu.SemaphoreType.REGULAR)
        sm_id = lax.axis_index("sm")

        @pl.when(sm_id == 0)
        def _single_sm_work():
            @pl.loop(0, tokens)
            def _init_out_row(token):
                @pl.loop(0, hidden_dim)
                def _init_out_col(col):
                    out_ref[token, col] = jnp.zeros_like(y_ref[0, col])

            @pl.loop(0, assignments)
            def _init_slot(slot):
                @pl.loop(0, hidden_dim)
                def _init_slot_col(col):
                    return_slots_ref[slot, col] = jnp.zeros_like(y_ref[0, col])

            @pl.loop(0, ep_size)
            def _signal_init(peer):
                pl.semaphore_signal(arrivals_sem, device_id=peer, device_id_type=pl.DeviceIdType.LOGICAL)

            pl.semaphore_wait(arrivals_sem, value=ep_size, decrement=False)

            @pl.loop(0, capacity)
            def _send_row(row):
                src = src_rank_ref[row]
                src_safe = jnp.maximum(src, jnp.int32(0))
                assignment = assignment_ref[row]
                should_send = (src >= 0) & (assignment >= 0) & (assignment < assignments)
                remote_return_slots_ref = mgpu.remote_ref(
                    return_slots_ref, src_safe, device_id_type=pl.DeviceIdType.LOGICAL
                )

                @pl.when(should_send)
                def _write_remote():
                    @pl.loop(0, hidden_dim)
                    def _copy_col(col):
                        remote_return_slots_ref[assignment, col] = y_ref[row, col]

            @pl.loop(0, ep_size)
            def _signal_done(peer):
                pl.semaphore_signal(arrivals_sem, device_id=peer, device_id_type=pl.DeviceIdType.LOGICAL)

            pl.semaphore_wait(arrivals_sem, value=2 * ep_size, decrement=False)

            @pl.loop(0, tokens)
            def _combine_token(token):
                @pl.loop(0, hidden_dim)
                def _combine_col(col):
                    acc = jnp.zeros_like(y_ref[0, col])
                    for route_slot in range(topk):
                        slot = token * topk + route_slot
                        weight = weights_ref[token, route_slot].astype(y_ref.dtype)
                        acc += return_slots_ref[slot, col] * weight
                    out_ref[token, col] = acc

    kernel = mgpu.kernel(
        body,
        out_shape=[
            jax.ShapeDtypeStruct((tokens, hidden_dim), y_dispatch.dtype),
            jax.ShapeDtypeStruct((assignments, hidden_dim), y_dispatch.dtype),
        ],
        grid=(num_sms,),
        grid_names=("sm",),
        compiler_params=mgpu.CompilerParams(
            lowering_semantics=mgpu.LoweringSemantics.Warpgroup,
        ),
    )
    out, _return_slots = kernel(y_dispatch, recv_src_rank, recv_src_assignment, combine_weights)
    return out


def return_combine_mgpu(
    y_dispatch: Float[Array, "capacity D"],
    recv_src_rank: Int[Array, "capacity"],
    recv_src_assignment: Int[Array, "capacity"],
    combine_weights: Float[Array, "Tlocal K"],
    *,
    expert_axis: str = "expert",
    config: MoeMgpuConfig = MoeMgpuConfig(),
) -> Float[Array, "Tlocal D"]:
    """Benchmark/debug helper for the production return-slot and combine path."""
    capacity, hidden_dim = y_dispatch.shape
    tokens, topk = combine_weights.shape
    assignments = tokens * topk
    ep_size = int(lax.axis_size(expert_axis))
    if recv_src_rank.shape != (capacity,):
        raise ValueError(f"recv_src_rank must have shape {(capacity,)}, got {recv_src_rank.shape}")
    if recv_src_assignment.shape != (capacity,):
        raise ValueError(f"recv_src_assignment must have shape {(capacity,)}, got {recv_src_assignment.shape}")

    num_sms = config.num_sms
    if num_sms is None:
        num_sms = jax.devices()[0].core_count
    return_slot_elements = assignments * hidden_dim
    init_return_slot_steps = int(math.ceil(return_slot_elements / num_sms))
    return_send_steps = int(math.ceil((capacity * hidden_dim) / num_sms))
    combine_steps = int(math.ceil((tokens * hidden_dim) / num_sms))

    def body(y_ref, src_rank_ref, assignment_ref, weights_ref, out_ref, return_slots_ref):
        arrivals_sem = pl.get_global(mgpu.SemaphoreType.REGULAR)
        sm_id = lax.axis_index("sm")

        @pl.loop(0, init_return_slot_steps)
        def _init_slot_step(step):
            element = step * num_sms + sm_id

            @pl.when(element < return_slot_elements)
            def _init_slot_element():
                slot = element // hidden_dim
                col = element % hidden_dim
                return_slots_ref[slot, col] = jnp.zeros_like(y_ref[0, 0])

        @pl.loop(0, ep_size)
        def _signal_init(peer):
            pl.semaphore_signal(arrivals_sem, device_id=peer, device_id_type=pl.DeviceIdType.LOGICAL)

        pl.semaphore_wait(arrivals_sem, value=ep_size * num_sms, decrement=False)

        @pl.loop(0, return_send_steps)
        def _send_return_step(step):
            element = step * num_sms + sm_id

            @pl.when(element < capacity * hidden_dim)
            def _send_return_element():
                row = element // hidden_dim
                col = element % hidden_dim
                src = src_rank_ref[row]
                src_safe = jnp.maximum(src, jnp.int32(0))
                assignment = assignment_ref[row]
                should_send = (src >= 0) & (assignment >= 0) & (assignment < assignments)
                remote_return_slots_ref = mgpu.remote_ref(
                    return_slots_ref,
                    src_safe,
                    device_id_type=pl.DeviceIdType.LOGICAL,
                )

                @pl.when(should_send)
                def _write_remote():
                    remote_return_slots_ref[assignment, col] = y_ref[row, col]

        @pl.loop(0, ep_size)
        def _signal_return_done(peer):
            pl.semaphore_signal(arrivals_sem, device_id=peer, device_id_type=pl.DeviceIdType.LOGICAL)

        pl.semaphore_wait(arrivals_sem, value=2 * ep_size * num_sms, decrement=False)

        @pl.loop(0, combine_steps)
        def _combine_step(step):
            element = step * num_sms + sm_id

            @pl.when(element < tokens * hidden_dim)
            def _combine_element():
                token = element // hidden_dim
                col = element % hidden_dim
                acc = jnp.zeros_like(y_ref[0, 0])
                for route_slot in range(topk):
                    slot = token * topk + route_slot
                    weight = weights_ref[token, route_slot].astype(y_ref.dtype)
                    acc += return_slots_ref[slot, col] * weight
                out_ref[token, col] = acc

    kernel = mgpu.kernel(
        body,
        out_shape=[
            jax.ShapeDtypeStruct((tokens, hidden_dim), y_dispatch.dtype),
            jax.ShapeDtypeStruct((assignments, hidden_dim), y_dispatch.dtype),
        ],
        grid=(num_sms,),
        grid_names=("sm",),
        compiler_params=mgpu.CompilerParams(
            lowering_semantics=mgpu.LoweringSemantics.Warpgroup,
        ),
    )
    out, _return_slots = kernel(y_dispatch, recv_src_rank, recv_src_assignment, combine_weights)
    return out


def return_slots_mgpu(
    y_dispatch: Float[Array, "capacity D"],
    recv_src_rank: Int[Array, "capacity"],
    recv_src_assignment: Int[Array, "capacity"],
    combine_weights: Float[Array, "Tlocal K"],
    *,
    expert_axis: str = "expert",
    config: MoeMgpuConfig = MoeMgpuConfig(),
) -> Float[Array, "A D"]:
    """Benchmark/debug helper for return-slot initialization and remote writes."""
    capacity, hidden_dim = y_dispatch.shape
    tokens, topk = combine_weights.shape
    assignments = tokens * topk
    ep_size = int(lax.axis_size(expert_axis))
    if recv_src_rank.shape != (capacity,):
        raise ValueError(f"recv_src_rank must have shape {(capacity,)}, got {recv_src_rank.shape}")
    if recv_src_assignment.shape != (capacity,):
        raise ValueError(f"recv_src_assignment must have shape {(capacity,)}, got {recv_src_assignment.shape}")

    num_sms = config.num_sms
    if num_sms is None:
        num_sms = jax.devices()[0].core_count
    return_slot_elements = assignments * hidden_dim
    init_return_slot_steps = int(math.ceil(return_slot_elements / num_sms))
    return_send_steps = int(math.ceil((capacity * hidden_dim) / num_sms))

    def body(y_ref, src_rank_ref, assignment_ref, return_slots_ref):
        arrivals_sem = pl.get_global(mgpu.SemaphoreType.REGULAR)
        sm_id = lax.axis_index("sm")

        @pl.loop(0, init_return_slot_steps)
        def _init_slot_step(step):
            element = step * num_sms + sm_id

            @pl.when(element < return_slot_elements)
            def _init_slot_element():
                slot = element // hidden_dim
                col = element % hidden_dim
                return_slots_ref[slot, col] = jnp.zeros_like(y_ref[0, 0])

        @pl.loop(0, ep_size)
        def _signal_init(peer):
            pl.semaphore_signal(arrivals_sem, device_id=peer, device_id_type=pl.DeviceIdType.LOGICAL)

        pl.semaphore_wait(arrivals_sem, value=ep_size * num_sms, decrement=False)

        @pl.loop(0, return_send_steps)
        def _send_return_step(step):
            element = step * num_sms + sm_id

            @pl.when(element < capacity * hidden_dim)
            def _send_return_element():
                row = element // hidden_dim
                col = element % hidden_dim
                src = src_rank_ref[row]
                src_safe = jnp.maximum(src, jnp.int32(0))
                assignment = assignment_ref[row]
                should_send = (src >= 0) & (assignment >= 0) & (assignment < assignments)
                remote_return_slots_ref = mgpu.remote_ref(
                    return_slots_ref,
                    src_safe,
                    device_id_type=pl.DeviceIdType.LOGICAL,
                )

                @pl.when(should_send)
                def _write_remote():
                    remote_return_slots_ref[assignment, col] = y_ref[row, col]

        @pl.loop(0, ep_size)
        def _signal_return_done(peer):
            pl.semaphore_signal(arrivals_sem, device_id=peer, device_id_type=pl.DeviceIdType.LOGICAL)

        pl.semaphore_wait(arrivals_sem, value=2 * ep_size * num_sms, decrement=False)

    kernel = mgpu.kernel(
        body,
        out_shape=jax.ShapeDtypeStruct((assignments, hidden_dim), y_dispatch.dtype),
        grid=(num_sms,),
        grid_names=("sm",),
        compiler_params=mgpu.CompilerParams(
            lowering_semantics=mgpu.LoweringSemantics.Warpgroup,
        ),
    )
    return kernel(y_dispatch, recv_src_rank, recv_src_assignment)


def combine_slots_mgpu(
    return_slots: Float[Array, "A D"],
    combine_weights: Float[Array, "Tlocal K"],
    *,
    config: MoeMgpuConfig = MoeMgpuConfig(),
) -> Float[Array, "Tlocal D"]:
    """Benchmark/debug helper for local fixed-route-slot combine only."""
    tokens, topk = combine_weights.shape
    assignments, hidden_dim = return_slots.shape
    if assignments != tokens * topk:
        raise ValueError(f"return_slots first dimension must be {tokens * topk}, got {assignments}")

    num_sms = config.num_sms
    if num_sms is None:
        num_sms = jax.devices()[0].core_count
    combine_steps = int(math.ceil((tokens * hidden_dim) / num_sms))

    def body(return_slots_ref, weights_ref, out_ref):
        sm_id = lax.axis_index("sm")

        @pl.loop(0, combine_steps)
        def _combine_step(step):
            element = step * num_sms + sm_id

            @pl.when(element < tokens * hidden_dim)
            def _combine_element():
                token = element // hidden_dim
                col = element % hidden_dim
                acc = jnp.zeros_like(return_slots_ref[0, 0])
                for route_slot in range(topk):
                    slot = token * topk + route_slot
                    weight = weights_ref[token, route_slot].astype(return_slots_ref.dtype)
                    acc += return_slots_ref[slot, col] * weight
                out_ref[token, col] = acc

    kernel = mgpu.kernel(
        body,
        out_shape=jax.ShapeDtypeStruct((tokens, hidden_dim), return_slots.dtype),
        grid=(num_sms,),
        grid_names=("sm",),
        compiler_params=mgpu.CompilerParams(
            lowering_semantics=mgpu.LoweringSemantics.Warpgroup,
        ),
    )
    return kernel(return_slots, combine_weights)


def pull_combine_mgpu(
    y_dispatch: Float[Array, "capacity D"],
    selected_experts_local: Int[Array, "Tlocal K"],
    combine_weights: Float[Array, "Tlocal K"],
    *,
    local_experts: int,
    expert_axis: str = "expert",
    config: MoeMgpuConfig = MoeMgpuConfig(),
) -> Float[Array, "Tlocal D"]:
    """Benchmark/debug source-side pull from expert-owner rows plus fixed-slot combine."""
    capacity, hidden_dim = y_dispatch.shape
    tokens, topk = combine_weights.shape
    assignments = tokens * topk
    if selected_experts_local.shape != (tokens, topk):
        raise ValueError(
            f"selected_experts_local must have shape {(tokens, topk)}, got {selected_experts_local.shape}"
        )
    ep_size = int(lax.axis_size(expert_axis))
    rank = lax.axis_index(expert_axis)
    metadata = prepare_mgpu_moe_metadata(
        selected_experts_local,
        local_experts=local_experts,
        ep_size=ep_size,
        expert_axis=expert_axis,
    )
    global_counts_flat = metadata.global_counts.reshape(ep_size, ep_size * local_experts)
    clipped_counts = _clip_receiver_group_sizes(
        global_counts_flat,
        local_expert_size=local_experts,
        receiver_capacity=capacity,
    ).reshape(ep_size, ep_size, local_experts)

    expert_ids_flat = selected_experts_local.reshape(assignments).astype(jnp.int32)
    dst_rank_by_assignment = expert_ids_flat // local_experts
    local_expert_by_assignment = expert_ids_flat % local_experts
    local_pos_by_assignment = (
        jnp.zeros((assignments,), dtype=jnp.int32).at[metadata.assignment_ids_sorted].set(metadata.local_pos_sorted)
    )

    rows_per_dst_expert = jnp.sum(clipped_counts, axis=0, dtype=jnp.int32)
    expert_base_by_dst = jnp.cumsum(rows_per_dst_expert, axis=1, dtype=jnp.int32) - rows_per_dst_expert
    src_base_by_dst_expert = jnp.cumsum(clipped_counts, axis=0, dtype=jnp.int32) - clipped_counts
    accepted_by_assignment = clipped_counts[rank, dst_rank_by_assignment, local_expert_by_assignment]
    remote_row_by_assignment = (
        expert_base_by_dst[dst_rank_by_assignment, local_expert_by_assignment]
        + src_base_by_dst_expert[rank, dst_rank_by_assignment, local_expert_by_assignment]
        + local_pos_by_assignment
    )
    keep_by_assignment = local_pos_by_assignment < accepted_by_assignment
    remote_row_by_assignment = jnp.where(keep_by_assignment, remote_row_by_assignment, jnp.int32(-1))

    num_sms = config.num_sms
    if num_sms is None:
        num_sms = jax.devices()[0].core_count
    combine_steps = int(math.ceil((tokens * hidden_dim) / num_sms))

    def body(y_ref, dst_rank_ref, remote_row_ref, weights_ref, out_ref):
        sm_id = lax.axis_index("sm")

        @pl.loop(0, combine_steps)
        def _combine_step(step):
            element = step * num_sms + sm_id

            @pl.when(element < tokens * hidden_dim)
            def _combine_element():
                token = element // hidden_dim
                col = element % hidden_dim
                acc = jnp.zeros_like(y_ref[0, 0])
                for route_slot in range(topk):
                    assignment = token * topk + route_slot
                    dst = dst_rank_ref[assignment]
                    dst_safe = jnp.maximum(dst, jnp.int32(0))
                    remote_row = remote_row_ref[assignment]
                    remote_row_safe = jnp.maximum(remote_row, jnp.int32(0))
                    should_read = remote_row >= 0
                    remote_y_ref = mgpu.remote_ref(y_ref, dst_safe, device_id_type=pl.DeviceIdType.LOGICAL)
                    value = remote_y_ref[remote_row_safe, col]
                    weight = weights_ref[token, route_slot].astype(y_ref.dtype)
                    acc += jnp.where(should_read, value * weight, jnp.zeros_like(value))
                out_ref[token, col] = acc

    kernel = mgpu.kernel(
        body,
        out_shape=jax.ShapeDtypeStruct((tokens, hidden_dim), y_dispatch.dtype),
        grid=(num_sms,),
        grid_names=("sm",),
        compiler_params=mgpu.CompilerParams(
            lowering_semantics=mgpu.LoweringSemantics.Warpgroup,
        ),
    )
    return kernel(y_dispatch, dst_rank_by_assignment, remote_row_by_assignment, combine_weights)


def pull_combine_vector_mgpu(
    y_dispatch: Float[Array, "capacity D"],
    selected_experts_local: Int[Array, "Tlocal K"],
    combine_weights: Float[Array, "Tlocal K"],
    *,
    local_experts: int,
    expert_axis: str = "expert",
    config: MoeMgpuConfig = MoeMgpuConfig(),
) -> Float[Array, "Tlocal D"]:
    """Benchmark/debug source-side pull/combine using block_n-wide remote reads."""
    capacity, hidden_dim = y_dispatch.shape
    tokens, topk = combine_weights.shape
    assignments = tokens * topk
    block_n = config.block_n
    if selected_experts_local.shape != (tokens, topk):
        raise ValueError(
            f"selected_experts_local must have shape {(tokens, topk)}, got {selected_experts_local.shape}"
        )
    if hidden_dim % block_n != 0:
        raise ValueError(f"D={hidden_dim} must be divisible by block_n={block_n}")
    ep_size = int(lax.axis_size(expert_axis))
    rank = lax.axis_index(expert_axis)
    metadata = prepare_mgpu_moe_metadata(
        selected_experts_local,
        local_experts=local_experts,
        ep_size=ep_size,
        expert_axis=expert_axis,
    )
    global_counts_flat = metadata.global_counts.reshape(ep_size, ep_size * local_experts)
    clipped_counts = _clip_receiver_group_sizes(
        global_counts_flat,
        local_expert_size=local_experts,
        receiver_capacity=capacity,
    ).reshape(ep_size, ep_size, local_experts)

    expert_ids_flat = selected_experts_local.reshape(assignments).astype(jnp.int32)
    dst_rank_by_assignment = expert_ids_flat // local_experts
    local_expert_by_assignment = expert_ids_flat % local_experts
    local_pos_by_assignment = (
        jnp.zeros((assignments,), dtype=jnp.int32).at[metadata.assignment_ids_sorted].set(metadata.local_pos_sorted)
    )

    rows_per_dst_expert = jnp.sum(clipped_counts, axis=0, dtype=jnp.int32)
    expert_base_by_dst = jnp.cumsum(rows_per_dst_expert, axis=1, dtype=jnp.int32) - rows_per_dst_expert
    src_base_by_dst_expert = jnp.cumsum(clipped_counts, axis=0, dtype=jnp.int32) - clipped_counts
    accepted_by_assignment = clipped_counts[rank, dst_rank_by_assignment, local_expert_by_assignment]
    remote_row_by_assignment = (
        expert_base_by_dst[dst_rank_by_assignment, local_expert_by_assignment]
        + src_base_by_dst_expert[rank, dst_rank_by_assignment, local_expert_by_assignment]
        + local_pos_by_assignment
    )
    keep_by_assignment = local_pos_by_assignment < accepted_by_assignment
    remote_row_by_assignment = jnp.where(keep_by_assignment, remote_row_by_assignment, jnp.int32(-1))

    num_sms = config.num_sms
    if num_sms is None:
        num_sms = jax.devices()[0].core_count
    col_blocks = hidden_dim // block_n
    combine_steps = int(math.ceil((tokens * col_blocks) / num_sms))

    def body(y_ref, dst_rank_ref, remote_row_ref, weights_ref, out_ref):
        sm_id = lax.axis_index("sm")

        @pl.loop(0, combine_steps)
        def _combine_step(step):
            element = step * num_sms + sm_id

            @pl.when(element < tokens * col_blocks)
            def _combine_element():
                token = element // col_blocks
                col_block = element % col_blocks
                col_start = col_block * block_n
                acc = jnp.zeros((block_n,), dtype=jnp.float32)
                for route_slot in range(topk):
                    assignment = token * topk + route_slot
                    dst = dst_rank_ref[assignment]
                    dst_safe = jnp.maximum(dst, jnp.int32(0))
                    remote_row = remote_row_ref[assignment]
                    remote_row_safe = jnp.maximum(remote_row, jnp.int32(0))
                    should_read = remote_row >= 0
                    remote_y_ref = mgpu.remote_ref(y_ref, dst_safe, device_id_type=pl.DeviceIdType.LOGICAL)
                    value = remote_y_ref[remote_row_safe, pl.ds(col_start, block_n)]
                    weight = weights_ref[token, route_slot].astype(jnp.float32)
                    weight_vec = jnp.full((block_n,), weight, dtype=jnp.float32)
                    should_read_vec = jnp.full((block_n,), should_read, dtype=jnp.bool_)
                    product = value.astype(jnp.float32) * weight_vec
                    acc += jnp.where(should_read_vec, product, jnp.zeros_like(product))
                out_ref[token, pl.ds(col_start, block_n)] = acc.astype(y_ref.dtype)

    kernel = mgpu.kernel(
        body,
        out_shape=jax.ShapeDtypeStruct((tokens, hidden_dim), y_dispatch.dtype),
        grid=(num_sms,),
        grid_names=("sm",),
        compiler_params=mgpu.CompilerParams(
            lowering_semantics=mgpu.LoweringSemantics.Warpgroup,
        ),
    )
    return kernel(y_dispatch, dst_rank_by_assignment, remote_row_by_assignment, combine_weights)


def combine_bwd_mgpu(
    y_dispatch: Float[Array, "capacity D"],
    out_bar: Float[Array, "Tlocal D"],
    selected_experts_local: Int[Array, "Tlocal K"],
    combine_weights: Float[Array, "Tlocal K"],
    *,
    local_experts: int,
    expert_axis: str = "expert",
    config: MoeMgpuConfig = MoeMgpuConfig(),
) -> tuple[Float[Array, "capacity D"], Float[Array, "Tlocal K"]]:
    """Backward of source-side pull/combine over the MGPU expert layout."""
    capacity, hidden_dim = y_dispatch.shape
    tokens, topk = combine_weights.shape
    assignments = tokens * topk
    block_n = _comm_block_n(hidden_dim, config.combine_bwd_block_n, config)
    if out_bar.shape != (tokens, hidden_dim):
        raise ValueError(f"out_bar must have shape {(tokens, hidden_dim)}, got {out_bar.shape}")
    if selected_experts_local.shape != (tokens, topk):
        raise ValueError(
            f"selected_experts_local must have shape {(tokens, topk)}, got {selected_experts_local.shape}"
        )
    if hidden_dim % block_n != 0:
        raise ValueError(f"D={hidden_dim} must be divisible by block_n={block_n}")

    ep_size = int(lax.axis_size(expert_axis))
    rank = lax.axis_index(expert_axis)
    metadata = prepare_mgpu_moe_metadata(
        selected_experts_local,
        local_experts=local_experts,
        ep_size=ep_size,
        expert_axis=expert_axis,
    )
    global_counts_flat = metadata.global_counts.reshape(ep_size, ep_size * local_experts)
    clipped_counts = _clip_receiver_group_sizes(
        global_counts_flat,
        local_expert_size=local_experts,
        receiver_capacity=capacity,
    ).reshape(ep_size, ep_size, local_experts)

    expert_ids_flat = selected_experts_local.reshape(assignments).astype(jnp.int32)
    dst_rank_by_assignment = expert_ids_flat // local_experts
    local_expert_by_assignment = expert_ids_flat % local_experts
    local_pos_by_assignment = (
        jnp.zeros((assignments,), dtype=jnp.int32).at[metadata.assignment_ids_sorted].set(metadata.local_pos_sorted)
    )

    rows_per_dst_expert = jnp.sum(clipped_counts, axis=0, dtype=jnp.int32)
    expert_base_by_dst = jnp.cumsum(rows_per_dst_expert, axis=1, dtype=jnp.int32) - rows_per_dst_expert
    src_base_by_dst_expert = jnp.cumsum(clipped_counts, axis=0, dtype=jnp.int32) - clipped_counts
    accepted_by_assignment = clipped_counts[rank, dst_rank_by_assignment, local_expert_by_assignment]
    remote_row_by_assignment = (
        expert_base_by_dst[dst_rank_by_assignment, local_expert_by_assignment]
        + src_base_by_dst_expert[rank, dst_rank_by_assignment, local_expert_by_assignment]
        + local_pos_by_assignment
    )
    keep_by_assignment = local_pos_by_assignment < accepted_by_assignment
    remote_row_by_assignment = jnp.where(keep_by_assignment, remote_row_by_assignment, jnp.int32(-1))

    num_sms = config.num_sms
    if num_sms is None:
        num_sms = jax.devices()[0].core_count
    col_blocks = hidden_dim // block_n
    dy_init_steps = int(math.ceil((capacity * col_blocks) / num_sms))
    assignment_col_steps = int(math.ceil((assignments * col_blocks) / num_sms))

    def body(y_ref, out_bar_ref, dst_rank_ref, remote_row_ref, weights_ref, dy_ref, dweight_partials_ref):
        arrivals_sem = pl.get_global(mgpu.SemaphoreType.REGULAR)
        sm_id = lax.axis_index("sm")

        @pl.loop(0, dy_init_steps)
        def _init_dy_step(step):
            element = step * num_sms + sm_id

            @pl.when(element < capacity * col_blocks)
            def _zero_dy_block():
                row = element // col_blocks
                col_block = element - row * col_blocks
                col_start = col_block * block_n
                dy_ref[row, pl.ds(col_start, block_n)] = jnp.zeros((block_n,), dtype=y_ref.dtype)

        @pl.loop(0, ep_size)
        def _signal_init(peer):
            pl.semaphore_signal(arrivals_sem, device_id=peer, device_id_type=pl.DeviceIdType.LOGICAL)

        pl.semaphore_wait(arrivals_sem, value=ep_size * num_sms, decrement=False)

        @pl.loop(0, assignment_col_steps)
        def _assignment_col_step(step):
            element = step * num_sms + sm_id

            @pl.when(element < assignments * col_blocks)
            def _process_assignment_col():
                assignment = element // col_blocks
                col_block = element - assignment * col_blocks
                col_start = col_block * block_n
                dst = dst_rank_ref[assignment]
                dst_safe = jnp.maximum(dst, jnp.int32(0))
                remote_row = remote_row_ref[assignment]
                remote_row_safe = jnp.maximum(remote_row, jnp.int32(0))
                should_process = remote_row >= 0
                token = assignment // topk
                route_slot = assignment - token * topk
                weight = weights_ref[token, route_slot].astype(y_ref.dtype)
                weight_vec = jnp.full((block_n,), weight, dtype=y_ref.dtype)
                remote_y_ref = mgpu.remote_ref(y_ref, dst_safe, device_id_type=pl.DeviceIdType.LOGICAL)
                remote_dy_ref = mgpu.remote_ref(dy_ref, dst_safe, device_id_type=pl.DeviceIdType.LOGICAL)
                out_vec = out_bar_ref[token, pl.ds(col_start, block_n)]
                y_vec = remote_y_ref[remote_row_safe, pl.ds(col_start, block_n)]
                dy_vec = out_vec * weight_vec

                @pl.when(should_process)
                def _write_remote_dy():
                    remote_dy_ref[remote_row_safe, pl.ds(col_start, block_n)] = dy_vec

                product = out_vec.astype(jnp.float32) * y_vec.astype(jnp.float32)
                dweight_partials_ref[assignment, col_block] = jnp.where(
                    should_process,
                    jnp.sum(product),
                    jnp.zeros((), dtype=jnp.float32),
                )

        @pl.loop(0, ep_size)
        def _signal_done(peer):
            pl.semaphore_signal(arrivals_sem, device_id=peer, device_id_type=pl.DeviceIdType.LOGICAL)

        pl.semaphore_wait(arrivals_sem, value=2 * ep_size * num_sms, decrement=False)

    kernel = mgpu.kernel(
        body,
        out_shape=[
            jax.ShapeDtypeStruct((capacity, hidden_dim), y_dispatch.dtype),
            jax.ShapeDtypeStruct((assignments, col_blocks), jnp.float32),
        ],
        grid=(num_sms,),
        grid_names=("sm",),
        compiler_params=mgpu.CompilerParams(
            lowering_semantics=mgpu.LoweringSemantics.Warpgroup,
        ),
    )
    dy_dispatch, dweight_partials = kernel(
        y_dispatch,
        out_bar,
        dst_rank_by_assignment,
        remote_row_by_assignment,
        combine_weights,
    )
    dcombine_weights = jnp.sum(dweight_partials.reshape(tokens, topk, col_blocks), axis=-1)
    return dy_dispatch, dcombine_weights.astype(combine_weights.dtype)


def dx_unpermute_vector_mgpu(
    drecv_x: Float[Array, "capacity D"],
    recv_src_rank: Int[Array, "capacity"],
    recv_src_assignment: Int[Array, "capacity"],
    combine_weights: Float[Array, "Tlocal K"],
    *,
    expert_axis: str = "expert",
    config: MoeMgpuConfig = MoeMgpuConfig(),
) -> Float[Array, "Tlocal D"]:
    """Vectorized metadata-driven return/combine used for dispatch backward."""
    capacity, hidden_dim = drecv_x.shape
    tokens, topk = combine_weights.shape
    assignments = tokens * topk
    block_n = _comm_block_n(hidden_dim, config.dx_unpermute_block_n, config)
    ep_size = int(lax.axis_size(expert_axis))
    if recv_src_rank.shape != (capacity,):
        raise ValueError(f"recv_src_rank must have shape {(capacity,)}, got {recv_src_rank.shape}")
    if recv_src_assignment.shape != (capacity,):
        raise ValueError(f"recv_src_assignment must have shape {(capacity,)}, got {recv_src_assignment.shape}")
    if hidden_dim % block_n != 0:
        raise ValueError(f"D={hidden_dim} must be divisible by block_n={block_n}")

    num_sms = config.num_sms
    if num_sms is None:
        num_sms = jax.devices()[0].core_count
    col_blocks = hidden_dim // block_n
    init_steps = int(math.ceil((assignments * col_blocks) / num_sms))
    return_steps = int(math.ceil((capacity * col_blocks) / num_sms))
    combine_steps = int(math.ceil((tokens * col_blocks) / num_sms))

    def body(drecv_ref, src_rank_ref, assignment_ref, weights_ref, dx_ref, return_slots_ref):
        arrivals_sem = pl.get_global(mgpu.SemaphoreType.REGULAR)
        sm_id = lax.axis_index("sm")

        @pl.loop(0, init_steps)
        def _init_step(step):
            element = step * num_sms + sm_id

            @pl.when(element < assignments * col_blocks)
            def _zero_slot_block():
                slot = element // col_blocks
                col_block = element - slot * col_blocks
                col_start = col_block * block_n
                return_slots_ref[slot, pl.ds(col_start, block_n)] = jnp.zeros((block_n,), dtype=drecv_ref.dtype)

        @pl.loop(0, ep_size)
        def _signal_init(peer):
            pl.semaphore_signal(arrivals_sem, device_id=peer, device_id_type=pl.DeviceIdType.LOGICAL)

        pl.semaphore_wait(arrivals_sem, value=ep_size * num_sms, decrement=False)

        @pl.loop(0, return_steps)
        def _return_step(step):
            element = step * num_sms + sm_id

            @pl.when(element < capacity * col_blocks)
            def _return_block():
                row = element // col_blocks
                col_block = element - row * col_blocks
                col_start = col_block * block_n
                src = src_rank_ref[row]
                src_safe = jnp.maximum(src, jnp.int32(0))
                assignment = assignment_ref[row]
                assignment_safe = jnp.maximum(assignment, jnp.int32(0))
                should_send = (src >= 0) & (assignment >= 0) & (assignment < assignments)
                remote_return_slots_ref = mgpu.remote_ref(
                    return_slots_ref,
                    src_safe,
                    device_id_type=pl.DeviceIdType.LOGICAL,
                )
                value = drecv_ref[row, pl.ds(col_start, block_n)]

                @pl.when(should_send)
                def _write_remote_return_slot():
                    remote_return_slots_ref[assignment_safe, pl.ds(col_start, block_n)] = value

        @pl.loop(0, ep_size)
        def _signal_return_done(peer):
            pl.semaphore_signal(arrivals_sem, device_id=peer, device_id_type=pl.DeviceIdType.LOGICAL)

        pl.semaphore_wait(arrivals_sem, value=2 * ep_size * num_sms, decrement=False)

        @pl.loop(0, combine_steps)
        def _combine_step(step):
            element = step * num_sms + sm_id

            @pl.when(element < tokens * col_blocks)
            def _combine_block():
                token = element // col_blocks
                col_block = element - token * col_blocks
                col_start = col_block * block_n
                acc = jnp.zeros((block_n,), dtype=jnp.float32)
                for route_slot in range(topk):
                    slot = token * topk + route_slot
                    weight = weights_ref[token, route_slot].astype(jnp.float32)
                    weight_vec = jnp.full((block_n,), weight, dtype=jnp.float32)
                    acc += return_slots_ref[slot, pl.ds(col_start, block_n)].astype(jnp.float32) * weight_vec
                dx_ref[token, pl.ds(col_start, block_n)] = acc.astype(drecv_ref.dtype)

    kernel = mgpu.kernel(
        body,
        out_shape=[
            jax.ShapeDtypeStruct((tokens, hidden_dim), drecv_x.dtype),
            jax.ShapeDtypeStruct((assignments, hidden_dim), drecv_x.dtype),
        ],
        grid=(num_sms,),
        grid_names=("sm",),
        compiler_params=mgpu.CompilerParams(
            lowering_semantics=mgpu.LoweringSemantics.Warpgroup,
        ),
    )
    dx, _return_slots = kernel(drecv_x, recv_src_rank, recv_src_assignment, combine_weights)
    return dx


def down_unpermute_mgpu(
    hidden: Float[Array, "capacity I"],
    recv_src_rank: Int[Array, "capacity"],
    recv_src_assignment: Int[Array, "capacity"],
    rows_per_expert: Int[Array, "Elocal"],
    moe_w2: Float[Array, "Elocal I D"],
    combine_weights: Float[Array, "Tlocal K"],
    selected_experts: Int[Array, "Tlocal K"],
    *,
    expert_axis: str = "expert",
    config: MoeMgpuConfig = MoeMgpuConfig(),
) -> Float[Array, "Tlocal D"]:
    """Fused local W2 plus deterministic source-side pull/combine."""
    out, _y_dispatch = down_unpermute_mgpu_with_dispatch(
        hidden,
        recv_src_rank,
        recv_src_assignment,
        rows_per_expert,
        moe_w2,
        combine_weights,
        selected_experts,
        expert_axis=expert_axis,
        config=config,
    )
    return out


def down_unpermute_mgpu_with_dispatch(
    hidden: Float[Array, "capacity I"],
    recv_src_rank: Int[Array, "capacity"],
    recv_src_assignment: Int[Array, "capacity"],
    rows_per_expert: Int[Array, "Elocal"],
    moe_w2: Float[Array, "Elocal I D"],
    combine_weights: Float[Array, "Tlocal K"],
    selected_experts: Int[Array, "Tlocal K"],
    *,
    expert_axis: str = "expert",
    config: MoeMgpuConfig = MoeMgpuConfig(),
) -> tuple[Float[Array, "Tlocal D"], Float[Array, "capacity D"]]:
    """Fused local W2 plus combine, returning the intermediate W2 dispatch rows."""
    return _down_unpermute_mgpu_kernel(
        hidden,
        moe_w2,
        recv_src_rank,
        recv_src_assignment,
        rows_per_expert,
        combine_weights,
        selected_experts,
        expert_axis=expert_axis,
        config=config,
    )


def _down_unpermute_mgpu_kernel(
    hidden: jax.Array,
    moe_w2: jax.Array,
    recv_src_rank: jax.Array,
    recv_src_assignment: jax.Array,
    rows_per_expert: jax.Array,
    combine_weights: jax.Array,
    selected_experts: jax.Array,
    *,
    expert_axis: str,
    config: MoeMgpuConfig,
) -> tuple[jax.Array, jax.Array]:
    local_experts, intermediate, hidden_dim = moe_w2.shape
    capacity = hidden.shape[0]
    tokens, topk = combine_weights.shape
    assignments = tokens * topk
    ep_size = int(lax.axis_size(expert_axis))
    rank = lax.axis_index(expert_axis)
    if hidden.shape[1] != intermediate:
        raise ValueError(f"hidden dimension {hidden.shape[1]} must match moe_w2 input dimension {intermediate}")
    if recv_src_rank.shape != (capacity,):
        raise ValueError(f"recv_src_rank must have shape {(capacity,)}, got {recv_src_rank.shape}")
    if recv_src_assignment.shape != (capacity,):
        raise ValueError(f"recv_src_assignment must have shape {(capacity,)}, got {recv_src_assignment.shape}")
    if selected_experts.shape != (tokens, topk):
        raise ValueError(f"selected_experts must have shape {(tokens, topk)}, got {selected_experts.shape}")
    if intermediate % config.block_k != 0:
        raise ValueError(f"I={intermediate} must be divisible by block_k={config.block_k}")
    if hidden_dim % config.block_n != 0:
        raise ValueError(f"D={hidden_dim} must be divisible by block_n={config.block_n}")

    block_m = config.block_m
    block_n = config.block_n
    block_k = config.block_k
    max_concurrent_steps = config.max_concurrent_steps
    grid_block_n = config.grid_block_n
    compute_group_sizes = _group_sizes_with_padding(rows_per_expert, total_size=capacity)
    num_sms = config.num_sms
    if num_sms is None:
        num_sms = jax.devices()[0].core_count
    col_blocks = hidden_dim // block_n
    combine_steps = int(math.ceil((tokens * col_blocks) / num_sms))
    metadata = prepare_mgpu_moe_metadata(
        selected_experts,
        local_experts=local_experts,
        ep_size=ep_size,
        expert_axis=expert_axis,
    )
    global_counts_flat = metadata.global_counts.reshape(ep_size, ep_size * local_experts)
    clipped_counts = _clip_receiver_group_sizes(
        global_counts_flat,
        local_expert_size=local_experts,
        receiver_capacity=capacity,
    ).reshape(ep_size, ep_size, local_experts)
    expert_ids_flat = selected_experts.reshape(assignments).astype(jnp.int32)
    dst_rank_by_assignment = expert_ids_flat // local_experts
    local_expert_by_assignment = expert_ids_flat % local_experts
    local_pos_by_assignment = (
        jnp.zeros((assignments,), dtype=jnp.int32).at[metadata.assignment_ids_sorted].set(metadata.local_pos_sorted)
    )
    rows_per_dst_expert = jnp.sum(clipped_counts, axis=0, dtype=jnp.int32)
    expert_base_by_dst = jnp.cumsum(rows_per_dst_expert, axis=1, dtype=jnp.int32) - rows_per_dst_expert
    src_base_by_dst_expert = jnp.cumsum(clipped_counts, axis=0, dtype=jnp.int32) - clipped_counts
    accepted_by_assignment = clipped_counts[rank, dst_rank_by_assignment, local_expert_by_assignment]
    remote_row_by_assignment = (
        expert_base_by_dst[dst_rank_by_assignment, local_expert_by_assignment]
        + src_base_by_dst_expert[rank, dst_rank_by_assignment, local_expert_by_assignment]
        + local_pos_by_assignment
    )
    keep_by_assignment = local_pos_by_assignment < accepted_by_assignment
    remote_row_by_assignment = jnp.where(keep_by_assignment, remote_row_by_assignment, jnp.int32(-1))

    def body(
        rows_per_expert_gmem,
        hidden_gmem,
        w2_gmem,
        dst_rank_gmem,
        remote_row_gmem,
        weights_gmem,
        out_gmem,
        y_dispatch_gmem,
    ):
        arrivals_sem = pl.get_global(mgpu.SemaphoreType.REGULAR)
        sm_id = lax.axis_index("sm")

        grid_m = pl.cdiv(capacity, block_m) + local_experts - 1
        grid_n = pl.cdiv(hidden_dim, block_n)
        grid = (grid_m * grid_n,)

        @mgpu.nd_loop(grid, collective_axes="sm")
        def mn_loop(loop_info):
            mi, ni = mgpu.planar_snake(
                loop_info.index[0],
                (grid_m, grid_n),
                1,
                grid_block_n,
            )

            group_info = GroupInfo.create(
                rows_per_expert_gmem,
                block_m,
                mi,
            )

            def acc_scope(acc_ref):
                def wgmma_step(_, lhs_smem, rhs_smem):
                    mgpu.wgmma(
                        acc_ref,
                        lhs_smem,
                        rhs_smem,
                    )

                mgpu.emit_pipeline(
                    wgmma_step,
                    grid=(intermediate // block_k,),
                    in_specs=[
                        mgpu.BlockSpec(
                            (block_m, block_k),
                            lambda kk: (group_info.block, kk),
                            delay_release=1,
                        ),
                        mgpu.BlockSpec(
                            (block_k, block_n),
                            lambda kk: (kk, ni),
                            delay_release=1,
                        ),
                    ],
                    max_concurrent_steps=max_concurrent_steps,
                )(
                    hidden_gmem,
                    w2_gmem.at[group_info.group_id],
                )

                return acc_ref[...]

            out_tile = pl.run_scoped(
                acc_scope,
                acc_ref=mgpu.ACC((block_m, block_n)),
            )

            @functools.partial(pl.run_scoped, o_smem=mgpu.SMEM((block_m, block_n), dtype=hidden_gmem.dtype))
            def store_scope(o_smem):
                o_smem[...] = out_tile.astype(o_smem.dtype)
                mgpu.commit_smem()

                smem_start = group_info.start_within_block
                remaining_rows = min(block_m, capacity)

                while remaining_rows > 0:
                    const_rows_len = 1 << int(math.log2(remaining_rows))
                    remaining_rows //= 2

                    @pl.when(group_info.actual_size & const_rows_len != 0)
                    def _():
                        o_smem_slice = o_smem.at[pl.ds(smem_start, const_rows_len)]
                        o_gref_slice = y_dispatch_gmem.at[
                            pl.ds(
                                group_info.block_start + smem_start,
                                const_rows_len,
                            ),
                            pl.ds(ni * block_n, block_n),
                        ]
                        mgpu.copy_smem_to_gmem(o_smem_slice, o_gref_slice)

                    smem_start += group_info.actual_size & const_rows_len

                mgpu.wait_smem_to_gmem(0, wait_read_only=True)

        @pl.loop(0, ep_size)
        def _signal_w2_done(peer):
            pl.semaphore_signal(arrivals_sem, device_id=peer, device_id_type=pl.DeviceIdType.LOGICAL)

        pl.semaphore_wait(arrivals_sem, value=ep_size * num_sms, decrement=False)

        @pl.loop(0, combine_steps)
        def _combine_step(step):
            element = step * num_sms + sm_id

            @pl.when(element < tokens * col_blocks)
            def _combine_element():
                token = element // col_blocks
                col_block = element % col_blocks
                col_start = col_block * block_n
                acc = jnp.zeros((block_n,), dtype=jnp.float32)
                for route_slot in range(topk):
                    assignment = token * topk + route_slot
                    dst = dst_rank_gmem[assignment]
                    dst_safe = jnp.maximum(dst, jnp.int32(0))
                    remote_row = remote_row_gmem[assignment]
                    remote_row_safe = jnp.maximum(remote_row, jnp.int32(0))
                    should_read = remote_row >= 0
                    remote_y_dispatch_ref = mgpu.remote_ref(
                        y_dispatch_gmem,
                        dst_safe,
                        device_id_type=pl.DeviceIdType.LOGICAL,
                    )
                    value = remote_y_dispatch_ref[remote_row_safe, pl.ds(col_start, block_n)]
                    weight = weights_gmem[token, route_slot].astype(jnp.float32)
                    weight_vec = jnp.full((block_n,), weight, dtype=jnp.float32)
                    should_read_vec = jnp.full((block_n,), should_read, dtype=jnp.bool_)
                    product = value.astype(jnp.float32) * weight_vec
                    acc += jnp.where(should_read_vec, product, jnp.zeros_like(product))
                out_gmem[token, pl.ds(col_start, block_n)] = acc.astype(hidden_gmem.dtype)

    kernel = mgpu.kernel(
        body,
        out_shape=[
            jax.ShapeDtypeStruct((tokens, hidden_dim), hidden.dtype),
            jax.ShapeDtypeStruct((capacity, hidden_dim), hidden.dtype),
        ],
        grid=(num_sms,),
        grid_names=("sm",),
        compiler_params=mgpu.CompilerParams(
            lowering_semantics=mgpu.LoweringSemantics.Warpgroup,
        ),
    )
    out, y_dispatch = kernel(
        compute_group_sizes,
        hidden,
        moe_w2,
        dst_rank_by_assignment,
        remote_row_by_assignment,
        combine_weights,
    )
    return out, y_dispatch


def moe_mlp_pallas_mgpu_staged(
    x: Float[Array, "Tlocal D"],
    selected_experts: Int[Array, "Tlocal K"],
    combine_weights: Float[Array, "Tlocal K"],
    moe_w13: Float[Array, "Elocal D I2"],
    moe_w2: Float[Array, "Elocal I D"],
    *,
    activation_fn: Callable[[jax.Array], jax.Array],
    expert_axis: str = "expert",
    config: MoeMgpuConfig = MoeMgpuConfig(),
) -> tuple[Float[Array, "Tlocal D"], Int[Array, ""]]:
    """End-to-end staged Pallas MGPU forward path used for kernel bring-up tests."""
    local_experts = moe_w13.shape[0]
    up_layout = permute_up_mgpu(
        x,
        selected_experts,
        moe_w13,
        local_experts=local_experts,
        activation_fn=activation_fn,
        expert_axis=expert_axis,
        config=config,
    )
    out = down_unpermute_mgpu(
        up_layout.hidden,
        up_layout.recv_src_rank,
        up_layout.recv_src_assignment,
        up_layout.rows_per_expert,
        moe_w2,
        combine_weights,
        selected_experts,
        expert_axis=expert_axis,
        config=config,
    )
    return out.astype(x.dtype), up_layout.dropped


def _validate_pallas_mgpu_reference_static_shapes(
    x: jax.Array,
    selected_experts: jax.Array,
    combine_weights: jax.Array,
    moe_w13: jax.Array,
    moe_w2: jax.Array,
) -> tuple[int, int, int, int, int]:
    if x.ndim != 2:
        raise ValueError(f"x must be rank-2 [T, D], got shape={x.shape}")
    if selected_experts.ndim != 2:
        raise ValueError(f"selected_experts must be rank-2 [T, K], got shape={selected_experts.shape}")
    if combine_weights.shape != selected_experts.shape:
        raise ValueError(
            "combine_weights must have the same [T, K] shape as selected_experts; "
            f"got {combine_weights.shape} vs {selected_experts.shape}"
        )
    if selected_experts.shape[0] != x.shape[0]:
        raise ValueError("selected_experts/combine_weights token dimension must match x")
    if x.shape[0] <= 0:
        raise ValueError(f"pallas_mgpu reference requires a positive token dimension, got T={x.shape[0]}")
    if selected_experts.shape[1] <= 0:
        raise ValueError(
            f"pallas_mgpu reference requires a positive top-k route dimension, got K={selected_experts.shape[1]}"
        )
    if moe_w13.ndim != 3:
        raise ValueError(f"moe_w13 must be rank-3 [E_local, D, 2I], got shape={moe_w13.shape}")
    if moe_w2.ndim != 3:
        raise ValueError(f"moe_w2 must be rank-3 [E_local, I, D], got shape={moe_w2.shape}")

    tokens, hidden_dim = x.shape
    local_experts, w13_hidden_dim, intermediate_twice = moe_w13.shape
    if local_experts <= 0:
        raise ValueError(
            f"pallas_mgpu reference requires a positive local expert dimension, got E_local={local_experts}"
        )
    if w13_hidden_dim != hidden_dim:
        raise ValueError(f"moe_w13 hidden dimension {w13_hidden_dim} must match x dimension {hidden_dim}")
    if hidden_dim <= 0:
        raise ValueError(f"pallas_mgpu reference requires a positive hidden dimension, got D={hidden_dim}")
    if intermediate_twice % 2 != 0:
        raise ValueError(f"moe_w13 output dimension must be even [2I], got {intermediate_twice}")
    intermediate = intermediate_twice // 2
    if intermediate <= 0:
        raise ValueError(f"pallas_mgpu reference requires a positive intermediate dimension, got I={intermediate}")
    if moe_w2.shape != (local_experts, intermediate, hidden_dim):
        raise ValueError(f"moe_w2 must have shape {(local_experts, intermediate, hidden_dim)}, got {moe_w2.shape}")
    topk = selected_experts.shape[1]
    return tokens, hidden_dim, local_experts, intermediate, topk


def moe_mlp_pallas_mgpu_reference(
    x: Float[Array, "Tlocal D"],
    selected_experts: Int[Array, "Tlocal K"],
    combine_weights: Float[Array, "Tlocal K"],
    moe_w13: Float[Array, "Elocal D I2"],
    moe_w2: Float[Array, "Elocal I D"],
    *,
    activation_fn: Callable[[jax.Array], jax.Array],
    config: MoeMgpuConfig = MoeMgpuConfig(),
    expert_axis: str | None = None,
    num_experts: int | None = None,
) -> tuple[Float[Array, "Tlocal D"], Int[Array, ""]]:
    """Readable staged reference for the MGPU MoE dataflow."""
    if expert_axis is not None:
        if num_experts is None:
            raise ValueError("num_experts is required for expert-parallel pallas_mgpu reference execution")
        tokens, _hidden_dim, local_experts, _intermediate, topk = _validate_pallas_mgpu_reference_static_shapes(
            x,
            selected_experts,
            combine_weights,
            moe_w13,
            moe_w2,
        )
        effective_capacity_factor = _effective_padded_capacity_factor(
            tokens,
            topk,
            local_experts,
            config.capacity_factor,
        )
        return _moe_mlp_ep_ragged_a2a_local(
            x,
            selected_experts,
            combine_weights,
            moe_w13,
            moe_w2,
            activation_fn=activation_fn,
            num_experts=num_experts,
            capacity_factor=effective_capacity_factor,
        )

    T, D, localE, I, K = _validate_pallas_mgpu_reference_static_shapes(
        x,
        selected_experts,
        combine_weights,
        moe_w13,
        moe_w2,
    )

    metadata = prepare_mgpu_moe_metadata(selected_experts, local_experts=localE, ep_size=1)
    x_dispatch = x[metadata.token_ids_sorted]

    group_sizes = metadata.send_counts.reshape(localE)
    accepted_group_sizes = _prefix_cap_counts(
        group_sizes,
        capacity=_receiver_capacity(T, K, localE, config.capacity_factor),
    )
    keep_mask = _expert_prefix_keep_mask(group_sizes, accepted_group_sizes, total_size=T * K)
    x_dispatch = _compact_by_keep_mask(x_dispatch, keep_mask)
    compute_group_sizes = _group_sizes_with_padding(accepted_group_sizes, total_size=T * K)

    w13_out = ragged_dot(x_dispatch, moe_w13, compute_group_sizes)
    gate, up = jnp.split(w13_out, [I], axis=-1)
    hidden = activation_fn(gate) * up
    out_dispatch = ragged_dot(hidden, moe_w2, compute_group_sizes)
    out_dispatch = _expand_from_keep_mask(out_dispatch, keep_mask)

    return_slots = jnp.zeros((T * K, D), dtype=out_dispatch.dtype).at[metadata.assignment_ids_sorted].set(out_dispatch)
    return_slots = return_slots.reshape(T, K, D)
    out = jnp.zeros((T, D), dtype=out_dispatch.dtype)
    weights = combine_weights.astype(out_dispatch.dtype)
    for route_slot in range(K):
        out = out + return_slots[:, route_slot, :] * weights[:, route_slot, None]

    dropped = jnp.sum(group_sizes, dtype=jnp.int32) - jnp.sum(accepted_group_sizes, dtype=jnp.int32)
    return out.astype(x.dtype), dropped


def _moe_mlp_pallas_mgpu_dropped(
    selected_experts: Int[Array, "Tlocal K"],
    *,
    local_experts: int,
    expert_axis: str,
    config: MoeMgpuConfig,
) -> Int[Array, ""]:
    ep_size = int(lax.axis_size(expert_axis))
    tokens, topk = selected_experts.shape
    metadata = prepare_mgpu_moe_metadata(
        selected_experts,
        local_experts=local_experts,
        ep_size=ep_size,
        expert_axis=expert_axis,
    )
    capacity = _receiver_capacity(tokens, topk, local_experts, config.capacity_factor)
    clipped_counts = _clip_receiver_group_sizes(
        metadata.global_counts.reshape(ep_size, ep_size * local_experts),
        local_expert_size=local_experts,
        receiver_capacity=capacity,
    ).reshape(ep_size, ep_size, local_experts)
    return jnp.sum(metadata.global_counts, dtype=jnp.int32) - jnp.sum(clipped_counts, dtype=jnp.int32)


def _validate_pallas_mgpu_requirements(
    x: jax.Array,
    selected_experts: jax.Array,
    combine_weights: jax.Array,
    moe_w13: jax.Array,
    moe_w2: jax.Array,
    *,
    expert_axis: str,
    config: MoeMgpuConfig,
) -> None:
    if x.ndim != 2:
        raise ValueError(f"x must be rank-2 [T, D], got shape={x.shape}")
    if selected_experts.ndim != 2:
        raise ValueError(f"selected_experts must be rank-2 [T, K], got shape={selected_experts.shape}")
    if combine_weights.shape != selected_experts.shape:
        raise ValueError(
            "combine_weights must have the same [T, K] shape as selected_experts; "
            f"got {combine_weights.shape} vs {selected_experts.shape}"
        )
    if selected_experts.shape[0] != x.shape[0]:
        raise ValueError("selected_experts/combine_weights token dimension must match x")
    if x.shape[0] <= 0:
        raise ValueError(f"implementation='pallas_mgpu' requires a positive token dimension, got T={x.shape[0]}")
    if selected_experts.shape[1] <= 0:
        raise ValueError(
            f"implementation='pallas_mgpu' requires a positive top-k route dimension, got K={selected_experts.shape[1]}"
        )
    if selected_experts.dtype != jnp.int32:
        raise ValueError(f"selected_experts must have dtype int32, got {selected_experts.dtype}")
    if combine_weights.dtype not in (jnp.bfloat16, jnp.float32):
        raise ValueError(f"combine_weights must have dtype bfloat16 or float32, got {combine_weights.dtype}")
    if moe_w13.ndim != 3:
        raise ValueError(f"moe_w13 must be rank-3 [E_local, D, 2I], got shape={moe_w13.shape}")
    if moe_w2.ndim != 3:
        raise ValueError(f"moe_w2 must be rank-3 [E_local, I, D], got shape={moe_w2.shape}")
    local_experts, w13_hidden_dim, intermediate_twice = moe_w13.shape
    if local_experts <= 0:
        raise ValueError(
            f"implementation='pallas_mgpu' requires a positive local expert dimension, got E_local={local_experts}"
        )
    if (
        config.dispatch_copy_schedule == _DISPATCH_COPY_EXPERT_GROUP_PEER
        and local_experts % config.dispatch_expert_group_size != 0
    ):
        raise ValueError(
            "implementation='pallas_mgpu' requires E_local to be divisible by dispatch_expert_group_size "
            "when dispatch_copy_schedule='expert_group_peer'; "
            f"got E_local={local_experts} and dispatch_expert_group_size={config.dispatch_expert_group_size}"
        )
    if w13_hidden_dim != x.shape[1]:
        raise ValueError(f"moe_w13 hidden dimension {w13_hidden_dim} must match x dimension {x.shape[1]}")
    if x.shape[1] <= 0:
        raise ValueError(f"implementation='pallas_mgpu' requires a positive hidden dimension, got D={x.shape[1]}")
    if intermediate_twice % 2 != 0:
        raise ValueError(f"moe_w13 output dimension must be even [2I], got {intermediate_twice}")
    intermediate = intermediate_twice // 2
    if intermediate <= 0:
        raise ValueError(
            f"implementation='pallas_mgpu' requires a positive intermediate dimension, got I={intermediate}"
        )
    if moe_w2.shape != (local_experts, intermediate, x.shape[1]):
        raise ValueError(f"moe_w2 must have shape {(local_experts, intermediate, x.shape[1])}, got {moe_w2.shape}")
    if x.dtype != jnp.bfloat16 or moe_w13.dtype != jnp.bfloat16 or moe_w2.dtype != jnp.bfloat16:
        raise ValueError(
            "implementation='pallas_mgpu' currently requires bfloat16 activations and weights; "
            f"got x={x.dtype}, moe_w13={moe_w13.dtype}, moe_w2={moe_w2.dtype}"
        )
    if x.shape[1] % config.dispatch_chunk_copy_tile != 0:
        raise ValueError(
            f"D={x.shape[1]} must be divisible by dispatch_chunk_copy_tile={config.dispatch_chunk_copy_tile}"
        )
    if x.shape[1] % config.block_k != 0:
        raise ValueError(f"D={x.shape[1]} must be divisible by block_k={config.block_k}")
    if intermediate % config.block_n != 0:
        raise ValueError(f"I={intermediate} must be divisible by block_n={config.block_n}")

    gpu_devices = [device for device in jax.local_devices() if device.platform == "gpu"]
    if not gpu_devices:
        raise ValueError("implementation='pallas_mgpu' requires a GPU backend; no local GPU devices are visible")
    device_kind = getattr(gpu_devices[0], "device_kind", "").lower()
    if "h100" not in device_kind and "hopper" not in device_kind:
        raise ValueError(
            "implementation='pallas_mgpu' currently targets Hopper/H100 GPUs only; "
            f"first local GPU device kind is {device_kind!r}"
        )
    ep_size = int(lax.axis_size(expert_axis))
    if ep_size > 8:
        raise ValueError(f"implementation='pallas_mgpu' supports EP <= 8, got EP={ep_size}")
    if ep_size > len(gpu_devices):
        raise ValueError(
            "implementation='pallas_mgpu' requires all expert-parallel ranks to be local GPU devices; "
            f"got EP={ep_size} with {len(gpu_devices)} local GPU devices"
        )
    _warn_if_receiver_capacity_padded(x.shape[0], selected_experts.shape[1], local_experts, config.capacity_factor)


def moe_mlp_pallas_mgpu(
    x: Float[Array, "Tlocal D"],
    selected_experts: Int[Array, "Tlocal K"],
    combine_weights: Float[Array, "Tlocal K"],
    moe_w13: Float[Array, "Elocal D I2"],
    moe_w2: Float[Array, "Elocal I D"],
    *,
    activation_fn: Callable[[jax.Array], jax.Array],
    expert_axis: str = "expert",
    config: MoeMgpuConfig = MoeMgpuConfig(),
) -> tuple[Float[Array, "Tlocal D"], Int[Array, ""]]:
    """Experimental Hopper MGPU MoE backend entrypoint.

    Supports a single local NVLink expert-parallel group on Hopper/H100 GPUs,
    `EP <= 8`, bfloat16 activations/weights, and deterministic fixed-slot
    combine. This path intentionally does not support NIC/InfiniBand, multi-host
    expert parallelism, FP8, or remote atomic combine.
    """
    _validate_pallas_mgpu_requirements(
        x,
        selected_experts,
        combine_weights,
        moe_w13,
        moe_w2,
        expert_axis=expert_axis,
        config=config,
    )
    out = _moe_mlp_pallas_mgpu_out_custom_vjp(
        x,
        selected_experts,
        combine_weights,
        moe_w13,
        moe_w2,
        activation_fn=activation_fn,
        expert_axis=expert_axis,
        config=config,
    )
    dropped = _moe_mlp_pallas_mgpu_dropped(
        selected_experts,
        local_experts=moe_w13.shape[0],
        expert_axis=expert_axis,
        config=config,
    )
    return out, dropped


@functools.partial(jax.custom_vjp, nondiff_argnums=(5, 6, 7))
def _moe_mlp_pallas_mgpu_out_custom_vjp(
    x: Float[Array, "Tlocal D"],
    selected_experts: Int[Array, "Tlocal K"],
    combine_weights: Float[Array, "Tlocal K"],
    moe_w13: Float[Array, "Elocal D I2"],
    moe_w2: Float[Array, "Elocal I D"],
    activation_fn: Callable[[jax.Array], jax.Array],
    expert_axis: str,
    config: MoeMgpuConfig,
) -> Float[Array, "Tlocal D"]:
    out, _dropped = moe_mlp_pallas_mgpu_staged(
        x,
        selected_experts,
        combine_weights,
        moe_w13,
        moe_w2,
        activation_fn=activation_fn,
        expert_axis=expert_axis,
        config=config,
    )
    return out


def _moe_mlp_pallas_mgpu_out_custom_vjp_fwd(
    x: jax.Array,
    selected_experts: jax.Array,
    combine_weights: jax.Array,
    moe_w13: jax.Array,
    moe_w2: jax.Array,
    activation_fn: Callable[[jax.Array], jax.Array],
    expert_axis: str,
    config: MoeMgpuConfig,
) -> tuple[jax.Array, tuple[jax.Array, ...]]:
    if config.dispatch_chunked_permute_up:
        out, _dropped = moe_mlp_pallas_mgpu_staged(
            x,
            selected_experts,
            combine_weights,
            moe_w13,
            moe_w2,
            activation_fn=activation_fn,
            expert_axis=expert_axis,
            config=config,
        )
        return out, (x, selected_experts, combine_weights, moe_w13, moe_w2)

    local_experts = moe_w13.shape[0]
    recv_layout = permute_mgpu(
        x,
        selected_experts,
        local_experts=local_experts,
        expert_axis=expert_axis,
        config=config,
    )
    compute_group_sizes = _group_sizes_with_padding(
        recv_layout.rows_per_expert, total_size=recv_layout.recv_x.shape[0]
    )
    hidden = _moe_mgpu_dispatch_w13_activation(
        recv_layout.recv_x,
        moe_w13,
        activation_fn,
        _MoeMgpuUpMetadata(global_expert_counts=compute_group_sizes[jnp.newaxis, :]),
        config,
    )
    out, y_dispatch = down_unpermute_mgpu_with_dispatch(
        hidden,
        recv_layout.recv_src_rank,
        recv_layout.recv_src_assignment,
        recv_layout.rows_per_expert,
        moe_w2,
        combine_weights,
        selected_experts,
        expert_axis=expert_axis,
        config=config,
    )
    return out.astype(x.dtype), (
        x,
        recv_layout.recv_x,
        recv_layout.recv_src_rank,
        recv_layout.recv_src_assignment,
        recv_layout.rows_per_expert,
        hidden,
        y_dispatch,
        selected_experts,
        combine_weights,
        moe_w13,
        moe_w2,
    )


def _moe_mlp_pallas_mgpu_out_custom_vjp_bwd(
    activation_fn: Callable[[jax.Array], jax.Array],
    expert_axis: str,
    config: MoeMgpuConfig,
    residuals: tuple[jax.Array, ...],
    out_bar: jax.Array | jax.custom_derivatives.SymbolicZero,
) -> tuple[jax.Array, None, jax.Array, jax.Array, jax.Array]:
    if config.dispatch_chunked_permute_up:
        x, selected_experts, combine_weights, moe_w13, moe_w2 = residuals
        if isinstance(out_bar, jax.custom_derivatives.SymbolicZero):
            return (
                jnp.zeros_like(x),
                None,
                jnp.zeros_like(combine_weights),
                jnp.zeros_like(moe_w13),
                jnp.zeros_like(moe_w2),
            )

        local_experts = moe_w13.shape[0]
        recv_layout = permute_mgpu(
            x,
            selected_experts,
            local_experts=local_experts,
            expert_axis=expert_axis,
            config=config,
        )
        compute_group_sizes = _group_sizes_with_padding(
            recv_layout.rows_per_expert, total_size=recv_layout.recv_x.shape[0]
        )

        def w13_activation(recv_x_arg: jax.Array, moe_w13_arg: jax.Array) -> jax.Array:
            w13_out = ragged_dot(recv_x_arg, moe_w13_arg, compute_group_sizes)
            intermediate = moe_w2.shape[1]
            gate, up = jnp.split(w13_out, [intermediate], axis=-1)
            return (activation_fn(gate) * up).astype(recv_x_arg.dtype)

        hidden, w13_pullback = jax.vjp(w13_activation, recv_layout.recv_x, moe_w13)

        def w2_forward(hidden_arg: jax.Array, moe_w2_arg: jax.Array) -> jax.Array:
            return ragged_dot(hidden_arg, moe_w2_arg, compute_group_sizes).astype(hidden_arg.dtype)

        y_dispatch, w2_pullback = jax.vjp(w2_forward, hidden, moe_w2)
        dy_dispatch, dcombine_weights = combine_bwd_mgpu(
            y_dispatch,
            out_bar.astype(y_dispatch.dtype),
            selected_experts,
            combine_weights,
            local_experts=local_experts,
            expert_axis=expert_axis,
            config=config,
        )
        dhidden, dmoe_w2 = w2_pullback(dy_dispatch)
        drecv_x, dmoe_w13 = w13_pullback(dhidden)
        dx = dx_unpermute_vector_mgpu(
            drecv_x,
            recv_layout.recv_src_rank,
            recv_layout.recv_src_assignment,
            jnp.ones_like(combine_weights, dtype=drecv_x.dtype),
            expert_axis=expert_axis,
            config=config,
        )
        return (
            dx.astype(x.dtype),
            None,
            dcombine_weights.astype(combine_weights.dtype),
            dmoe_w13.astype(moe_w13.dtype),
            dmoe_w2.astype(moe_w2.dtype),
        )

    (
        x,
        recv_x,
        recv_src_rank,
        recv_src_assignment,
        rows_per_expert,
        hidden,
        y_dispatch,
        selected_experts,
        combine_weights,
        moe_w13,
        moe_w2,
    ) = residuals
    if isinstance(out_bar, jax.custom_derivatives.SymbolicZero):
        return (
            jnp.zeros_like(x),
            None,
            jnp.zeros_like(combine_weights),
            jnp.zeros_like(moe_w13),
            jnp.zeros_like(moe_w2),
        )

    local_experts = moe_w13.shape[0]
    compute_group_sizes = _group_sizes_with_padding(rows_per_expert, total_size=recv_x.shape[0])

    def w13_activation(recv_x_arg: jax.Array, moe_w13_arg: jax.Array) -> jax.Array:
        w13_out = ragged_dot(recv_x_arg, moe_w13_arg, compute_group_sizes)
        intermediate = moe_w2.shape[1]
        gate, up = jnp.split(w13_out, [intermediate], axis=-1)
        return (activation_fn(gate) * up).astype(recv_x_arg.dtype)

    _hidden, w13_pullback = jax.vjp(w13_activation, recv_x, moe_w13)

    def w2_forward(hidden_arg: jax.Array, moe_w2_arg: jax.Array) -> jax.Array:
        return ragged_dot(hidden_arg, moe_w2_arg, compute_group_sizes).astype(hidden_arg.dtype)

    _y_dispatch, w2_pullback = jax.vjp(w2_forward, hidden, moe_w2)
    dy_dispatch, dcombine_weights = combine_bwd_mgpu(
        y_dispatch,
        out_bar.astype(y_dispatch.dtype),
        selected_experts,
        combine_weights,
        local_experts=local_experts,
        expert_axis=expert_axis,
        config=config,
    )
    dhidden, dmoe_w2 = w2_pullback(dy_dispatch)
    drecv_x, dmoe_w13 = w13_pullback(dhidden)
    dx = dx_unpermute_vector_mgpu(
        drecv_x,
        recv_src_rank,
        recv_src_assignment,
        jnp.ones_like(combine_weights, dtype=drecv_x.dtype),
        expert_axis=expert_axis,
        config=config,
    )
    return (
        dx.astype(x.dtype),
        None,
        dcombine_weights.astype(combine_weights.dtype),
        dmoe_w13.astype(moe_w13.dtype),
        dmoe_w2.astype(moe_w2.dtype),
    )


_moe_mlp_pallas_mgpu_out_custom_vjp.defvjp(
    _moe_mlp_pallas_mgpu_out_custom_vjp_fwd,
    _moe_mlp_pallas_mgpu_out_custom_vjp_bwd,
)


def _moe_mlp_ep_pallas_mgpu_local(
    x_local: Float[Array, "TL D"],
    selected_experts_local: Int[Array, "TL K"],
    combine_weights_local: Float[Array, "TL K"],
    moe_w13_local: Float[Array, "EL D I2"],
    moe_w2_local: Float[Array, "EL I D"],
    *,
    activation_fn: Callable[[jax.Array], jax.Array],
    num_experts: int,
    capacity_factor: float,
    pallas_mgpu_config: MoeMgpuConfig | None = None,
) -> tuple[Float[Array, "TL D"], Int[Array, ""]]:
    config = pallas_mgpu_config or MoeMgpuConfig(capacity_factor=capacity_factor)
    return moe_mlp_pallas_mgpu(
        x_local,
        selected_experts_local,
        combine_weights_local,
        moe_w13_local,
        moe_w2_local,
        activation_fn=activation_fn,
        expert_axis="expert",
        config=config,
    )


def _moe_mgpu_dispatch_w13_activation(
    tokens_sorted: Float[Array, "localTK D"],
    moe_w13: Float[Array, "localE D I2"],
    activation_fn: Callable[[jax.Array], jax.Array],
    metadata: _MoeMgpuUpMetadata,
    config: MoeMgpuConfig,
):
    """
    Dispatches tokens to experts, applies moe_w13 and activation_fn, and returns the intermediate representation.

    Broadly follows:
       - https://github.com/jax-ml/jax/blob/main/jax/experimental/pallas/ops/gpu/ragged_dot_mgpu.py and
       - https://github.com/jax-ml/jax/blob/main/jax/experimental/pallas/ops/gpu/collective_matmul_mgpu.py
    """
    # my_rank = lax.axis_index("expert")
    # ep = lax.axis_size("expert")

    # assert ep == 1, "Multi-rank support is not implemented yet."

    localE, D, I2 = moe_w13.shape
    I = I2 // 2

    localTK = tokens_sorted.shape[0]
    m = _pad_receiver_capacity_for_wgmma(localTK)
    _warn_if_wgmma_m_padded(localTK, m)
    if m != localTK:
        pad_rows = m - localTK
        tokens_sorted = jnp.pad(tokens_sorted, ((0, pad_rows), (0, 0)))
    i = I
    g = localE
    d = D

    assert D % config.block_k == 0
    assert I % config.block_n == 0
    block_m = config.block_m
    block_n = config.block_n
    block_k = config.block_k
    max_concurrent_steps = config.max_concurrent_steps
    grid_block_n = config.grid_block_n

    def body(rows_per_expert_gmem, lhs_gmem, rhs_gmem, o_gmem):
        grid_m = pl.cdiv(m, block_m) + g - 1
        grid_n = pl.cdiv(i, block_n)
        grid = (grid_m * grid_n,)

        @mgpu.nd_loop(grid, collective_axes="sm")
        def mn_loop(loop_info):
            mi, ni = mgpu.planar_snake(
                loop_info.index[0],
                (grid_m, grid_n),
                1,
                grid_block_n,
            )

            group_info = GroupInfo.create(
                rows_per_expert_gmem,
                block_m,
                mi,
            )

            def acc_scope(gate_acc_ref, up_acc_ref):
                def wgmma_step(_, lhs_smem, gate_smem, up_smem):
                    mgpu.wgmma(
                        gate_acc_ref,
                        lhs_smem,
                        gate_smem,
                    )
                    mgpu.wgmma(
                        up_acc_ref,
                        lhs_smem,
                        up_smem,
                    )

                mgpu.emit_pipeline(
                    wgmma_step,
                    grid=(d // block_k,),
                    in_specs=[
                        # LHS tile: [block_m, block_k]
                        mgpu.BlockSpec(
                            (block_m, block_k),
                            lambda kk: (group_info.block, kk),
                            delay_release=1,
                        ),
                        # RHS gate tile: rhs[group, kk, ni]
                        mgpu.BlockSpec(
                            (block_k, block_n),
                            lambda kk: (kk, ni),
                            delay_release=1,
                        ),
                        # RHS up tile: rhs[group, kk, ni + i / block_n]
                        mgpu.BlockSpec(
                            (block_k, block_n),
                            lambda kk: (kk, ni + i // block_n),
                            delay_release=1,
                        ),
                    ],
                    max_concurrent_steps=max_concurrent_steps,
                )(
                    lhs_gmem,
                    rhs_gmem.at[group_info.group_id],
                    rhs_gmem.at[group_info.group_id],
                )

                gate = gate_acc_ref[...]
                up = up_acc_ref[...]

                # hidden = (gate / (1.0 + exp(-gate))) * up
                hidden = activation_fn(gate) * up
                return hidden

            hidden = pl.run_scoped(
                acc_scope,
                gate_acc_ref=mgpu.ACC((block_m, block_n)),
                up_acc_ref=mgpu.ACC((block_m, block_n)),
            )

            # ridiculous log2 loop to get around lack of dynamic shapes
            @functools.partial(pl.run_scoped, o_smem=mgpu.SMEM((block_m, block_n), dtype=o_gmem.dtype))
            def store_scope(o_smem):
                o_smem[...] = hidden.astype(o_smem.dtype)
                mgpu.commit_smem()

                smem_start = group_info.start_within_block
                remaining_rows = min(block_m, m)

                while remaining_rows > 0:
                    const_rows_len = 1 << int(math.log2(remaining_rows))
                    remaining_rows //= 2

                    @pl.when(group_info.actual_size & const_rows_len != 0)
                    def _():
                        o_smem_slice = o_smem.at[pl.ds(smem_start, const_rows_len)]
                        o_gref_slice = o_gmem.at[
                            pl.ds(
                                group_info.block_start + smem_start,
                                const_rows_len,
                            ),
                            pl.ds(ni * block_n, block_n),
                        ]
                        mgpu.copy_smem_to_gmem(o_smem_slice, o_gref_slice)

                    smem_start += group_info.actual_size & const_rows_len

                mgpu.wait_smem_to_gmem(0, wait_read_only=True)

    num_sms = config.num_sms
    if num_sms is None:
        num_sms = jax.devices()[0].core_count
    kernel = mgpu.kernel(
        body,
        out_shape=jax.ShapeDtypeStruct((m, I), tokens_sorted.dtype),
        grid=(num_sms,),
        grid_names=("sm",),
        compiler_params=mgpu.CompilerParams(
            lowering_semantics=mgpu.LoweringSemantics.Warpgroup,
        ),
    )

    local_group_sizes = metadata.global_expert_counts[0]

    return kernel(local_group_sizes, tokens_sorted, moe_w13)[:localTK]


def main(unused_argv):
    m, k, n, num_groups = 4096 * 8, 2560, 1280, 16
    kx, ky, kz = random.split(random.key(1234), num=3)

    lhs = jax.random.normal(kx, (m, k), jnp.bfloat16) * 0.1
    rhs = jax.random.normal(ky, (num_groups, k, n * 2), jnp.bfloat16) * 0.1
    group_boundaries = jax.lax.sort(jax.random.randint(kz, (num_groups - 1,), 0, m, jnp.int32))
    group_starts = lax.concatenate([jnp.array([0], dtype=jnp.int32), group_boundaries], 0)
    group_ends = lax.concatenate([group_boundaries, jnp.array([m], dtype=jnp.int32)], 0)
    group_sizes = group_ends - group_starts
    assert group_sizes.shape == (num_groups,)

    block_m = block_n = (64, 128, 192)
    block_k = (64,)
    max_concurrent_steps = (2, 4, 5, 6)
    grid_block_n = (1, 2, 4, 8, 16)
    configs = itertools.product(block_m, block_n, block_k, max_concurrent_steps, grid_block_n)
    names = ("block_m", "block_n", "block_k", "max_concurrent_steps", "grid_block_n")
    best_runtime = float("inf")
    best_kwargs: dict[str, int] = {}
    metadata = _MoeMgpuUpMetadata(global_expert_counts=group_sizes.reshape(1, -1))
    for config in configs:
        kwargs = dict(zip(names, config))
        if n % (kwargs["grid_block_n"] * kwargs["block_n"]):
            continue

        try:
            f = functools.partial(
                _moe_mgpu_dispatch_w13_activation, config=MoeMgpuConfig(**kwargs), activation_fn=jax.nn.silu
            )
            _, runtime = profiler.measure(f)(lhs, rhs, metadata=metadata)
        except ValueError as e:
            if "Mosaic GPU kernel exceeds available shared memory" not in str(e):
                raise
            runtime = float("inf")
        # Enable this to get more detailed information.
        else:
            assert runtime is not None
            print(" ".join(f"{k}={v}" for k, v in kwargs.items()), int(runtime * 1000))
        if runtime < best_runtime:
            best_runtime = runtime
            best_kwargs = kwargs
    if not best_kwargs:
        raise ValueError("No valid configuration found")

    def ref_ragged_dot(lhs, rhs, group_sizes):
        up = jax.lax.ragged_dot(lhs, rhs[..., n:], group_sizes=group_sizes)
        gate = jax.lax.ragged_dot(lhs, rhs[..., :n], group_sizes=group_sizes)
        return jax.nn.silu(gate) * up

    ref, ref_runtime = profiler.measure(ref_ragged_dot)(lhs, rhs, group_sizes=metadata.global_expert_counts[0])
    assert ref_runtime is not None
    result = _moe_mgpu_dispatch_w13_activation(
        lhs, rhs, metadata=metadata, config=MoeMgpuConfig(**best_kwargs), activation_fn=jax.nn.silu
    )
    np.testing.assert_allclose(result, ref, atol=1e-2, rtol=5e-2)

    tflops = float(4 * k * m * n) / (best_runtime / 1e3) / 1e12
    ref_tflops = float(4 * k * m * n) / (ref_runtime / 1e3) / 1e12
    print("Best parameters: ", " ".join(f"{k}={v}" for k, v in best_kwargs.items()))
    print(f"Kernel:    {best_runtime * 1000:.1f} us = {tflops:.1f} TFLOPS")
    print(f"Reference: {ref_runtime * 1000:.1f} us = {ref_tflops:.1f} TFLOPS")


if __name__ == "__main__":
    from absl import app

    jax.config.config_with_absl()
    app.run(main)
