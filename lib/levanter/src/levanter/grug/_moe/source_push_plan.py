# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Host-side plan for an invertible source-push MGPU MoE forward path."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Bool, Float, Int

from levanter.grug._moe.ep_common import _clip_receiver_group_sizes


SOURCE_PUSH_META_SRC_RANK = 0
SOURCE_PUSH_META_LOCAL_EXPERT = 1
SOURCE_PUSH_META_LOCAL_ROW_START = 2
SOURCE_PUSH_META_VALID_ROWS = 3
SOURCE_PUSH_META_FIELDS = 4
INVALID_ASSIGNMENT_ID = -1
SOURCE_PUSH_MESH_AXIS = "expert"


def _source_push_out_sharding(*parts):
    if jax.sharding.get_abstract_mesh().empty:
        return None
    return P(*parts)


def _with_source_push_sharding(value, *parts):
    sharding = _source_push_out_sharding(*parts)
    if sharding is None:
        return value
    return jax.sharding.reshard(value, sharding)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SourcePushPlan:
    """Invertible source-push queue metadata for all EP ranks in a local group.

    Queue-owned fields use source-major transport order:
    `[src, dst_ordinal, entry, row_in_block]`.

    Destination offset fields use destination-major order:
    `rows_per_local_expert[dst, expert]`,
    `expert_base[dst, expert]`, and
    `src_base_by_expert[dst, src, expert]`.
    """

    assignment_ids: Int[Array, "S Dst Q M"]
    token_ids: Int[Array, "S Dst Q M"]
    route_slots: Int[Array, "S Dst Q M"]
    combine_weights: Float[Array, "S Dst Q M"]
    valid_mask: Bool[Array, "S Dst Q M"]
    local_experts: Int[Array, "S Dst Q"]
    local_row_starts: Int[Array, "S Dst Q"]
    send_meta: Int[Array, "S Dst Q F"]
    recv_meta: Int[Array, "Dst Src Q F"]
    counts_by_src_dst_expert: Int[Array, "S Dst E"]
    rows_per_local_expert: Int[Array, "Dst E"]
    expert_base: Int[Array, "Dst E"]
    src_base_by_expert: Int[Array, "Dst S E"]
    dropped_routes: Int[Array, ""]
    tokens_per_source: int = field(metadata={"static": True})
    topk: int = field(metadata={"static": True})


@dataclass(frozen=True)
class SourcePushPlanRowStats:
    """Row accounting for the source-push transport queue."""

    useful_rows: int
    rounded_rows: int
    live_entries: int
    dropped_routes: int
    row_efficiency: float
    masked_row_fraction: float


class SourcePushQueueEntryMetadata(NamedTuple):
    """Expanded queue-entry metadata derived from per-source destination counts."""

    local_experts: np.ndarray
    local_row_starts: np.ndarray
    send_meta: np.ndarray
    recv_meta: np.ndarray


class SourcePushRouteRowsHostData(NamedTuple):
    """Host route identity and compact expert-row placement derived from a plan."""

    src: np.ndarray
    dst: np.ndarray
    local_expert: np.ndarray
    expert_row: np.ndarray
    token_id: np.ndarray
    route_slot: np.ndarray
    assignment_id: np.ndarray
    valid: np.ndarray


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SourcePushSemanticReverseRoute:
    """Source-owned route metadata for semantic expert-major rows."""

    route_dst: Int[Array, "S T K"]
    route_expert: Int[Array, "S T K"]
    route_expert_row: Int[Array, "S T K"]
    route_valid: Bool[Array, "S T K"]
    assignment_id: Int[Array, "S T K"]


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SourcePushSemanticPlan:
    """JAX-native expert-major routing metadata, independent of inbox slots.

    Rows are grouped as `[source, destination, row]`, with each pair's rows
    stored in local-expert order. The row identity is the source assignment id
    `token * topk + route_slot`; token ids, route slots, combine weights, and
    source-owned reverse metadata are derived from that identity.
    """

    assignment_ids: Int[Array, "S Dst R"]
    token_ids: Int[Array, "S Dst R"]
    route_slots: Int[Array, "S Dst R"]
    route_weights: Float[Array, "S Dst R"]
    valid_mask: Bool[Array, "S Dst R"]
    xcounts: Int[Array, "S Dst E"]
    pair_expert_base: Int[Array, "S Dst E"]
    rows_per_local_expert: Int[Array, "Dst E"]
    expert_base: Int[Array, "Dst E"]
    src_base_by_expert: Int[Array, "Dst S E"]
    reverse_route: SourcePushSemanticReverseRoute
    routing_dropped_routes: Int[Array, ""]
    metadata_overflow_routes: Int[Array, ""]
    dropped_routes: Int[Array, ""]
    tokens_per_source: int = field(metadata={"static": True})
    topk: int = field(metadata={"static": True})


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SourcePushSemanticQueueMetadata:
    """Static return-queue entries and their source-owned inverse mapping."""

    local_expert: Int[Array, "S DstOrd Q"]
    local_row_start: Int[Array, "S DstOrd Q"]
    valid_rows: Int[Array, "S DstOrd Q"]
    required_entries_per_dst: Int[Array, "S DstOrd"]
    route_dst_ordinal: Int[Array, "S T K"]
    route_entry: Int[Array, "S T K"]
    route_queue_row: Int[Array, "S T K"]
    route_valid: Bool[Array, "S T K"]
    overflow_entries: Int[Array, ""]
    overflow_routes: Int[Array, ""]
    return_row_block: int = field(metadata={"static": True})
    entries_per_dst: int = field(metadata={"static": True})


def dst_ordinal(src: int, dst: int, ep_size: int) -> int:
    """Return the source-local destination ordinal used by the transport queue."""

    return (dst - src) % ep_size


def recv_src_ordinal(dst: int, src: int, ep_size: int) -> int:
    """Return the destination-local source ordinal used by receive metadata."""

    return (src - dst) % ep_size


def source_push_expert_offsets_from_counts(
    counts_by_src_dst_expert: Int[Array, "S Dst E"] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Derive exact expert-major row counts and prefix offsets from accepted counts."""

    counts = _counts_host(counts_by_src_dst_expert)
    rows_per_local_expert = np.asarray(np.sum(counts, axis=0, dtype=np.int32), dtype=np.int32)
    expert_base = _exclusive_cumsum(rows_per_local_expert, axis=1)
    src_base_by_expert = np.zeros((counts.shape[1], counts.shape[0], counts.shape[2]), dtype=np.int32)
    for dst in range(counts.shape[1]):
        src_base_by_expert[dst] = _exclusive_cumsum(counts[:, dst, :], axis=0)
    return rows_per_local_expert, expert_base, src_base_by_expert


def source_push_queue_entry_metadata_from_counts(
    counts_by_src_dst_expert: Int[Array, "S Dst E"] | np.ndarray,
    block_m: int,
    *,
    entries_per_dst: int | None = None,
) -> SourcePushQueueEntryMetadata:
    """Derive queue-entry expert, row-start, and send/receive metadata from counts.

    This is the analytical representation behind ``local_experts``,
    ``local_row_starts``, ``send_meta``, and ``recv_meta``. The arrays remain on
    ``SourcePushPlan`` as cached kernel inputs, but their ownership is the
    per-``(source, destination, local_expert)`` count tensor plus ``block_m``.
    """

    if block_m <= 0:
        raise ValueError(f"block_m must be positive, got {block_m}")
    counts = _counts_host(counts_by_src_dst_expert)
    ep_size, dst_count, experts_per_rank = counts.shape
    if ep_size != dst_count:
        raise ValueError(f"source-push counts must be square in source/destination dims, got {counts.shape}")

    required_entries_per_dst = _required_entries_per_dst(counts, block_m)
    if entries_per_dst is None:
        entries_per_dst = required_entries_per_dst
    if entries_per_dst < required_entries_per_dst:
        raise ValueError(
            "source-push queue capacity overflow: "
            f"entries_per_dst={entries_per_dst} but required {required_entries_per_dst}"
        )

    local_experts = np.full((ep_size, ep_size, entries_per_dst), INVALID_ASSIGNMENT_ID, dtype=np.int32)
    local_row_starts = np.zeros((ep_size, ep_size, entries_per_dst), dtype=np.int32)
    send_meta = np.zeros((ep_size, ep_size, entries_per_dst, SOURCE_PUSH_META_FIELDS), dtype=np.int32)

    for src in range(ep_size):
        for dst in range(ep_size):
            dst_entry = 0
            dst_ord = dst_ordinal(src, dst, ep_size)
            for local_expert in range(experts_per_rank):
                accepted_count = int(counts[src, dst, local_expert])
                for local_row_start in range(0, accepted_count, block_m):
                    valid_rows = min(block_m, accepted_count - local_row_start)
                    local_experts[src, dst_ord, dst_entry] = local_expert
                    local_row_starts[src, dst_ord, dst_entry] = local_row_start
                    send_meta[src, dst_ord, dst_entry, :] = (
                        src,
                        local_expert,
                        local_row_start,
                        valid_rows,
                    )
                    dst_entry += 1

    recv_meta = np.zeros_like(send_meta)
    for dst in range(ep_size):
        for src in range(ep_size):
            send_dst_ord = dst_ordinal(src, dst, ep_size)
            recv_src_ord = recv_src_ordinal(dst, src, ep_size)
            recv_meta[dst, recv_src_ord, :, :] = send_meta[src, send_dst_ord, :, :]

    return SourcePushQueueEntryMetadata(
        local_experts=local_experts,
        local_row_starts=local_row_starts,
        send_meta=send_meta,
        recv_meta=recv_meta,
    )


def build_source_push_semantic_plan_jax(
    selected_experts: Int[Array, "S T K"],
    combine_weights: Float[Array, "S T K"],
    *,
    ep_size: int,
    experts_per_rank: int,
    rows_per_src_dst_capacity: int,
    rows_per_expert_capacity: int | None = None,
    capacity_factor: float = 1.25,
) -> SourcePushSemanticPlan:
    """Build slot-free source-push routing metadata with JAX array operations.

    The output shape is padded by ``rows_per_src_dst_capacity``. Rows inside
    each source/destination pair are grouped by local expert. After router and
    pair clipping, ``rows_per_expert_capacity`` optionally caps each destination
    expert across sources, with earlier sources taking precedence. All metadata
    clipping is counted in ``metadata_overflow_routes``.
    """

    if ep_size <= 0:
        raise ValueError(f"ep_size must be positive, got {ep_size}")
    if experts_per_rank <= 0:
        raise ValueError(f"experts_per_rank must be positive, got {experts_per_rank}")
    if rows_per_src_dst_capacity <= 0:
        raise ValueError(f"rows_per_src_dst_capacity must be positive, got {rows_per_src_dst_capacity}")
    if rows_per_expert_capacity is not None and rows_per_expert_capacity <= 0:
        raise ValueError(f"rows_per_expert_capacity must be positive, got {rows_per_expert_capacity}")
    if capacity_factor <= 0:
        raise ValueError(f"capacity_factor must be positive, got {capacity_factor}")
    if selected_experts.ndim != 3:
        raise ValueError(f"selected_experts must have shape [ep_size, T, K], got {selected_experts.shape}")
    if combine_weights.shape != selected_experts.shape:
        raise ValueError(
            f"combine_weights shape {combine_weights.shape} must match selected_experts {selected_experts.shape}"
        )

    source_count, tokens_per_source, topk = selected_experts.shape
    if source_count != ep_size:
        raise ValueError(f"selected_experts leading dim must match ep_size={ep_size}, got {source_count}")

    assignments_per_source = tokens_per_source * topk
    global_experts = ep_size * experts_per_rank
    flat_expert = selected_experts.reshape(ep_size, assignments_per_source).astype(jnp.int32)
    flat_weights = combine_weights.reshape(ep_size, assignments_per_source)
    flat_assignment_id = jnp.arange(assignments_per_source, dtype=jnp.int32)

    group_sizes = jax.vmap(lambda experts: jnp.bincount(experts, length=global_experts).astype(jnp.int32))(flat_expert)
    receiver_capacity = max(experts_per_rank, int(math.ceil(capacity_factor * assignments_per_source)))
    clipped_group_sizes = _clip_receiver_group_sizes(
        group_sizes,
        local_expert_size=experts_per_rank,
        receiver_capacity=receiver_capacity,
    )
    clipped_counts = clipped_group_sizes.reshape(ep_size, ep_size, experts_per_rank)
    clipped_pair_counts = jnp.sum(clipped_counts, axis=2, dtype=jnp.int32)
    stored_pair_counts = jnp.minimum(clipped_pair_counts, rows_per_src_dst_capacity)
    pair_expert_base_unclipped = _exclusive_cumsum_jax(clipped_counts, axis=2)
    stored_counts = jnp.clip(rows_per_src_dst_capacity - pair_expert_base_unclipped, 0, clipped_counts)
    metadata_overflow_routes = jnp.sum(clipped_pair_counts - stored_pair_counts, dtype=jnp.int32)
    if rows_per_expert_capacity is not None:
        source_expert_base = _exclusive_cumsum_jax(stored_counts, axis=0)
        expert_clipped_counts = jnp.clip(
            rows_per_expert_capacity - source_expert_base,
            0,
            stored_counts,
        )
        metadata_overflow_routes += jnp.sum(stored_counts - expert_clipped_counts, dtype=jnp.int32)
        stored_counts = expert_clipped_counts
    routing_dropped_routes = selected_experts.size - jnp.sum(clipped_counts, dtype=jnp.int32)

    pair_expert_base = _exclusive_cumsum_jax(stored_counts, axis=2)
    rows_per_local_expert = jnp.sum(stored_counts, axis=0, dtype=jnp.int32)
    expert_base = _exclusive_cumsum_jax(rows_per_local_expert, axis=1)
    src_base_by_expert = jnp.transpose(_exclusive_cumsum_jax(stored_counts, axis=0), (1, 0, 2))

    sort_key = flat_expert * assignments_per_source + flat_assignment_id[None, :]
    sorted_positions = jnp.argsort(sort_key, axis=1, stable=True).astype(jnp.int32)
    sorted_expert = jnp.take_along_axis(flat_expert, sorted_positions, axis=1)
    sorted_weights = jnp.take_along_axis(flat_weights, sorted_positions, axis=1)
    sorted_assignment_ids = jnp.take_along_axis(
        jnp.broadcast_to(flat_assignment_id[None, :], sorted_positions.shape),
        sorted_positions,
        axis=1,
    )

    group_offsets = _exclusive_cumsum_jax(group_sizes, axis=1)
    sorted_group_start = jnp.take_along_axis(group_offsets, sorted_expert, axis=1)
    sorted_stored = jnp.take_along_axis(stored_counts.reshape(ep_size, global_experts), sorted_expert, axis=1)
    sorted_local_row = jnp.arange(assignments_per_source, dtype=jnp.int32)[None, :] - sorted_group_start
    keep = sorted_local_row < sorted_stored

    sorted_dst = sorted_expert // experts_per_rank
    sorted_pair_base = jnp.take_along_axis(pair_expert_base.reshape(ep_size, global_experts), sorted_expert, axis=1)
    sorted_pair_row = sorted_pair_base + sorted_local_row

    rows_per_pair_capacity = rows_per_src_dst_capacity
    stored_pair_row = jnp.where(keep, sorted_pair_row, rows_per_pair_capacity)
    source_index = jnp.arange(ep_size, dtype=jnp.int32)[:, None]
    assignment_by_pair = jnp.full(
        (ep_size, ep_size, rows_per_pair_capacity),
        INVALID_ASSIGNMENT_ID,
        dtype=jnp.int32,
    )
    assignment_by_pair = assignment_by_pair.at[
        source_index,
        sorted_dst,
        stored_pair_row,
    ].set(jnp.where(keep, sorted_assignment_ids, INVALID_ASSIGNMENT_ID), mode="drop")

    weight_by_pair = jnp.zeros(
        (ep_size, ep_size, rows_per_pair_capacity),
        dtype=combine_weights.dtype,
    )
    weight_by_pair = weight_by_pair.at[
        source_index,
        sorted_dst,
        stored_pair_row,
    ].set(jnp.where(keep, sorted_weights, jnp.zeros((), dtype=combine_weights.dtype)), mode="drop")

    assignment_ids = assignment_by_pair
    route_weights = weight_by_pair
    valid_mask = assignment_ids != INVALID_ASSIGNMENT_ID
    safe_assignment_ids = jnp.maximum(assignment_ids, 0)
    token_ids = jnp.where(valid_mask, safe_assignment_ids // topk, INVALID_ASSIGNMENT_ID)
    route_slots = jnp.where(valid_mask, safe_assignment_ids % topk, INVALID_ASSIGNMENT_ID)

    reverse_route = _source_push_semantic_reverse_route_from_metadata_jax(
        assignment_ids=assignment_ids,
        token_ids=token_ids,
        route_slots=route_slots,
        valid_mask=valid_mask,
        xcounts=stored_counts,
        pair_expert_base=pair_expert_base,
        src_base_by_expert=src_base_by_expert,
        tokens_per_source=tokens_per_source,
        topk=topk,
    )
    dropped_routes = routing_dropped_routes + metadata_overflow_routes
    return SourcePushSemanticPlan(
        assignment_ids=assignment_ids,
        token_ids=token_ids,
        route_slots=route_slots,
        route_weights=route_weights,
        valid_mask=valid_mask,
        xcounts=stored_counts,
        pair_expert_base=pair_expert_base,
        rows_per_local_expert=rows_per_local_expert,
        expert_base=expert_base,
        src_base_by_expert=src_base_by_expert,
        reverse_route=reverse_route,
        routing_dropped_routes=routing_dropped_routes,
        metadata_overflow_routes=metadata_overflow_routes,
        dropped_routes=dropped_routes,
        tokens_per_source=tokens_per_source,
        topk=topk,
    )


def source_push_semantic_queue_metadata_jax(
    plan: SourcePushSemanticPlan,
    *,
    return_row_block: int,
    entries_per_dst: int,
) -> SourcePushSemanticQueueMetadata:
    """Expand semantic counts into a static source-local return queue.

    Entries are ordered by local expert and then row block for each
    ``(source, destination)`` pair. Destination axes use the source-local
    ordinal ``(dst - src) % ep_size``. The inverse arrays map every stored
    source route to its queue entry and row; routes beyond ``entries_per_dst``
    are excluded from ``route_valid`` and reported in the overflow counters.
    """

    if return_row_block <= 0:
        raise ValueError(f"return_row_block must be positive, got {return_row_block}")
    if entries_per_dst <= 0:
        raise ValueError(f"entries_per_dst must be positive, got {entries_per_dst}")

    source_count, dst_count, experts_per_rank = plan.xcounts.shape
    if source_count != dst_count:
        raise ValueError(f"semantic source/destination dims must match, got {plan.xcounts.shape}")

    source_index = jnp.arange(source_count, dtype=jnp.int32)[:, None]
    dst_ordinal = jnp.arange(dst_count, dtype=jnp.int32)[None, :]
    actual_dst = (source_index + dst_ordinal) % dst_count
    counts_by_dst_ordinal = plan.xcounts.at[source_index, actual_dst].get()
    entry_counts_by_expert = (counts_by_dst_ordinal + return_row_block - 1) // return_row_block
    expert_entry_base = _exclusive_cumsum_jax(entry_counts_by_expert, axis=2)
    required_entries_per_dst = jnp.sum(entry_counts_by_expert, axis=2, dtype=jnp.int32)

    local_expert = _source_push_semantic_pair_expert_ids_from_counts_jax(
        entry_counts_by_expert,
        rows_per_pair_capacity=entries_per_dst,
    )
    queue_entry = jnp.arange(entries_per_dst, dtype=jnp.int32)[None, None, :]
    source_queue_index = jnp.arange(source_count, dtype=jnp.int32)[:, None, None]
    dst_queue_index = jnp.arange(dst_count, dtype=jnp.int32)[None, :, None]
    local_entry = queue_entry - expert_entry_base.at[source_queue_index, dst_queue_index, local_expert].get()
    entry_valid = queue_entry < required_entries_per_dst[..., None]
    local_row_start = jnp.maximum(local_entry, 0) * return_row_block
    expert_count = counts_by_dst_ordinal.at[source_queue_index, dst_queue_index, local_expert].get()
    valid_rows = jnp.clip(expert_count - local_row_start, 0, return_row_block)
    local_expert = jnp.where(entry_valid, local_expert, INVALID_ASSIGNMENT_ID)
    local_row_start = jnp.where(entry_valid, local_row_start, 0).astype(jnp.int32)
    valid_rows = jnp.where(entry_valid, valid_rows, 0).astype(jnp.int32)

    pair_expert, _expert_row = _source_push_semantic_expert_row_indices(plan)
    pair_row = jnp.arange(plan.assignment_ids.shape[-1], dtype=jnp.int32)[None, None, :]
    source_pair_index = jnp.arange(source_count, dtype=jnp.int32)[:, None, None]
    dst_pair_index = jnp.arange(dst_count, dtype=jnp.int32)[None, :, None]
    pair_base = plan.pair_expert_base.at[source_pair_index, dst_pair_index, pair_expert].get()
    local_pair_row = jnp.maximum(pair_row - pair_base, 0)
    actual_entry_counts = (plan.xcounts + return_row_block - 1) // return_row_block
    actual_expert_entry_base = _exclusive_cumsum_jax(actual_entry_counts, axis=2)
    route_entry_by_pair = (
        actual_expert_entry_base.at[source_pair_index, dst_pair_index, pair_expert].get()
        + local_pair_row // return_row_block
    )
    route_queue_row_by_pair = local_pair_row % return_row_block
    route_dst_ordinal_by_pair = (dst_pair_index - source_pair_index) % dst_count
    pair_route_valid = plan.valid_mask & (route_entry_by_pair < entries_per_dst)

    safe_token_ids = jnp.where(pair_route_valid, jnp.maximum(plan.token_ids, 0), plan.tokens_per_source)
    safe_route_slots = jnp.where(pair_route_valid, jnp.maximum(plan.route_slots, 0), plan.topk)
    route_shape = (source_count, plan.tokens_per_source, plan.topk)
    route_dst_ordinal = jnp.zeros(route_shape, dtype=jnp.int32)
    route_dst_ordinal = route_dst_ordinal.at[source_pair_index, safe_token_ids, safe_route_slots].set(
        jnp.where(pair_route_valid, route_dst_ordinal_by_pair, 0),
        mode="drop",
    )
    route_entry = jnp.zeros(route_shape, dtype=jnp.int32)
    route_entry = route_entry.at[source_pair_index, safe_token_ids, safe_route_slots].set(
        jnp.where(pair_route_valid, route_entry_by_pair, 0),
        mode="drop",
    )
    route_queue_row = jnp.zeros(route_shape, dtype=jnp.int32)
    route_queue_row = route_queue_row.at[source_pair_index, safe_token_ids, safe_route_slots].set(
        jnp.where(pair_route_valid, route_queue_row_by_pair, 0),
        mode="drop",
    )
    route_valid = jnp.zeros(route_shape, dtype=jnp.bool_)
    route_valid = route_valid.at[source_pair_index, safe_token_ids, safe_route_slots].set(
        pair_route_valid,
        mode="drop",
    )

    overflow_entries = jnp.sum(
        jnp.maximum(required_entries_per_dst - entries_per_dst, 0),
        dtype=jnp.int32,
    )
    overflow_routes = jnp.sum(plan.valid_mask & ~pair_route_valid, dtype=jnp.int32)
    return SourcePushSemanticQueueMetadata(
        local_expert=local_expert,
        local_row_start=local_row_start,
        valid_rows=valid_rows,
        required_entries_per_dst=required_entries_per_dst,
        route_dst_ordinal=route_dst_ordinal,
        route_entry=route_entry,
        route_queue_row=route_queue_row,
        route_valid=route_valid,
        overflow_entries=overflow_entries,
        overflow_routes=overflow_routes,
        return_row_block=return_row_block,
        entries_per_dst=entries_per_dst,
    )


def source_push_semantic_source_aligned_expert_offsets_jax(
    plan: SourcePushSemanticPlan,
    *,
    row_alignment: int,
) -> tuple[Int[Array, "Dst E"], Int[Array, "Dst S E"]]:
    """Place each source segment at an aligned expert-local row offset.

    The semantic pair order is unchanged. Only the gaps between adjacent source
    segments are padded, which keeps tiled GMEM transfers aligned without the
    much larger per-source compute padding used by the inbox layout.
    """

    if row_alignment <= 0:
        raise ValueError(f"row_alignment must be positive, got {row_alignment}")
    aligned_counts = (plan.xcounts + jnp.asarray(row_alignment - 1, dtype=jnp.int32)) // row_alignment * row_alignment
    source_bases = jnp.cumsum(aligned_counts, axis=0, dtype=jnp.int32) - aligned_counts
    rows_per_local_expert = jnp.sum(aligned_counts, axis=0, dtype=jnp.int32)
    return rows_per_local_expert, jnp.transpose(source_bases, (1, 0, 2))


def build_source_push_plan(
    selected_experts: Int[Array, "S T K"],
    combine_weights: Float[Array, "S T K"],
    *,
    ep_size: int,
    experts_per_rank: int,
    block_m: int,
    capacity_factor: float = 1.25,
    entries_per_dst: int | None = None,
) -> SourcePushPlan:
    """Build source-owned inverse metadata and destination expert-major offsets.

    Capacity clipping matches the existing EP ragged all-to-all helper:
    receiver capacity is applied per destination rank, experts are accepted in
    local expert order, and earlier source ranks win ties inside each expert.
    """

    if ep_size <= 0:
        raise ValueError(f"ep_size must be positive, got {ep_size}")
    if experts_per_rank <= 0:
        raise ValueError(f"experts_per_rank must be positive, got {experts_per_rank}")
    if block_m <= 0:
        raise ValueError(f"block_m must be positive, got {block_m}")
    if capacity_factor <= 0:
        raise ValueError(f"capacity_factor must be positive, got {capacity_factor}")

    selected_host = np.asarray(jax.device_get(selected_experts), dtype=np.int32)
    weights_host = np.asarray(jax.device_get(combine_weights))
    if selected_host.ndim != 3:
        raise ValueError(f"selected_experts must have shape [ep_size, T, K], got {selected_host.shape}")
    if weights_host.shape != selected_host.shape:
        raise ValueError(
            f"combine_weights shape {weights_host.shape} must match selected_experts {selected_host.shape}"
        )

    source_count, tokens, topk = selected_host.shape
    if source_count != ep_size:
        raise ValueError(f"selected_experts leading dim must match ep_size={ep_size}, got {source_count}")
    assignments_per_source = tokens * topk
    global_experts = ep_size * experts_per_rank
    if np.any(selected_host < 0) or np.any(selected_host >= global_experts):
        raise ValueError(f"selected_experts must be in [0, {global_experts})")

    group_sizes = _source_group_sizes(selected_host, global_experts)
    receiver_capacity = max(experts_per_rank, int(math.ceil(capacity_factor * assignments_per_source)))
    clipped_group_sizes = np.asarray(
        jax.device_get(
            _clip_receiver_group_sizes(
                jnp.asarray(group_sizes, dtype=jnp.int32),
                local_expert_size=experts_per_rank,
                receiver_capacity=receiver_capacity,
            )
        ),
        dtype=np.int32,
    )
    counts_by_src_dst_expert = clipped_group_sizes.reshape(ep_size, ep_size, experts_per_rank)

    required_entries_per_dst = _required_entries_per_dst(counts_by_src_dst_expert, block_m)
    if entries_per_dst is None:
        entries_per_dst = required_entries_per_dst
    if entries_per_dst < required_entries_per_dst:
        raise ValueError(
            "source-push queue capacity overflow: "
            f"entries_per_dst={entries_per_dst} but required {required_entries_per_dst}"
        )

    rows_per_local_expert, expert_base, src_base_by_expert = source_push_expert_offsets_from_counts(
        counts_by_src_dst_expert
    )
    entry_metadata = source_push_queue_entry_metadata_from_counts(
        counts_by_src_dst_expert,
        block_m,
        entries_per_dst=entries_per_dst,
    )

    queue_shape = (ep_size, ep_size, entries_per_dst, block_m)
    assignment_ids = np.full(queue_shape, INVALID_ASSIGNMENT_ID, dtype=np.int32)
    token_ids = np.full(queue_shape, INVALID_ASSIGNMENT_ID, dtype=np.int32)
    route_slots = np.full(queue_shape, INVALID_ASSIGNMENT_ID, dtype=np.int32)
    combine_weights_host = np.zeros(queue_shape, dtype=weights_host.dtype)
    valid_mask = np.zeros(queue_shape, dtype=np.bool_)
    local_experts = entry_metadata.local_experts
    local_row_starts = entry_metadata.local_row_starts
    send_meta = entry_metadata.send_meta
    recv_meta = entry_metadata.recv_meta

    flat_assignment_ids = np.arange(assignments_per_source, dtype=np.int32)
    for src in range(ep_size):
        source_experts = selected_host[src].reshape(assignments_per_source)
        source_weights = weights_host[src].reshape(assignments_per_source)
        sort_key = source_experts.astype(np.int64) * assignments_per_source + flat_assignment_ids
        sorted_assignment_ids = flat_assignment_ids[np.argsort(sort_key, kind="stable")]

        for dst in range(ep_size):
            dst_entry = 0
            dst_ord = dst_ordinal(src, dst, ep_size)
            for local_expert in range(experts_per_rank):
                global_expert = dst * experts_per_rank + local_expert
                accepted_count = int(counts_by_src_dst_expert[src, dst, local_expert])
                if accepted_count == 0:
                    continue
                expert_assignment_ids = sorted_assignment_ids[source_experts[sorted_assignment_ids] == global_expert]
                expert_assignment_ids = expert_assignment_ids[:accepted_count]
                for local_row_start in range(0, accepted_count, block_m):
                    valid_rows = min(block_m, accepted_count - local_row_start)
                    block_assignment_ids = expert_assignment_ids[local_row_start : local_row_start + valid_rows]
                    row_slice = slice(0, valid_rows)

                    assignment_ids[src, dst_ord, dst_entry, row_slice] = block_assignment_ids
                    token_ids[src, dst_ord, dst_entry, row_slice] = block_assignment_ids // topk
                    route_slots[src, dst_ord, dst_entry, row_slice] = block_assignment_ids % topk
                    combine_weights_host[src, dst_ord, dst_entry, row_slice] = source_weights[block_assignment_ids]
                    valid_mask[src, dst_ord, dst_entry, row_slice] = True
                    dst_entry += 1

    dropped_routes = selected_host.size - int(np.sum(counts_by_src_dst_expert, dtype=np.int64))
    return SourcePushPlan(
        assignment_ids=jnp.asarray(assignment_ids, dtype=jnp.int32),
        token_ids=jnp.asarray(token_ids, dtype=jnp.int32),
        route_slots=jnp.asarray(route_slots, dtype=jnp.int32),
        combine_weights=jnp.asarray(combine_weights_host),
        valid_mask=jnp.asarray(valid_mask, dtype=jnp.bool_),
        local_experts=jnp.asarray(local_experts, dtype=jnp.int32),
        local_row_starts=jnp.asarray(local_row_starts, dtype=jnp.int32),
        send_meta=jnp.asarray(send_meta, dtype=jnp.int32),
        recv_meta=jnp.asarray(recv_meta, dtype=jnp.int32),
        counts_by_src_dst_expert=jnp.asarray(counts_by_src_dst_expert, dtype=jnp.int32),
        rows_per_local_expert=jnp.asarray(rows_per_local_expert, dtype=jnp.int32),
        expert_base=jnp.asarray(expert_base, dtype=jnp.int32),
        src_base_by_expert=jnp.asarray(src_base_by_expert, dtype=jnp.int32),
        dropped_routes=jnp.asarray(dropped_routes, dtype=jnp.int32),
        tokens_per_source=tokens,
        topk=topk,
    )


def source_push_plan_row_stats(plan: SourcePushPlan) -> SourcePushPlanRowStats:
    """Return useful-vs-rounded row accounting for benchmark reporting."""

    valid_rows = np.asarray(plan.send_meta[..., SOURCE_PUSH_META_VALID_ROWS], dtype=np.int64)
    live_entries = int(np.sum(valid_rows > 0))
    useful_rows = int(np.sum(valid_rows))
    block_m = int(plan.assignment_ids.shape[-1])
    rounded_rows = live_entries * block_m
    row_efficiency = useful_rows / rounded_rows if rounded_rows else 1.0
    return SourcePushPlanRowStats(
        useful_rows=useful_rows,
        rounded_rows=rounded_rows,
        live_entries=live_entries,
        dropped_routes=int(jax.device_get(plan.dropped_routes)),
        row_efficiency=row_efficiency,
        masked_row_fraction=1.0 - row_efficiency,
    )


def source_push_source_padded_row_bases(
    plan: SourcePushPlan,
    block_m: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return source-padded expert-major row bases for full-tile WGMMA stores.

    The exact plan bases pack sources contiguously inside each local expert.
    Current Lane/WGMMA lowering cannot store partial GMEM tiles, so the W13 v0
    gives each source `block_m`-rounded room and leaves invalid rows as padding.
    """

    if block_m <= 0:
        raise ValueError(f"block_m must be positive, got {block_m}")

    counts = np.asarray(jax.device_get(plan.counts_by_src_dst_expert), dtype=np.int32)
    rounded_counts = _ceil_div(counts, block_m) * block_m
    rows_per_local_expert = np.sum(rounded_counts, axis=0, dtype=np.int32)
    expert_base = np.zeros_like(rows_per_local_expert)
    src_base_by_expert = np.zeros((counts.shape[1], counts.shape[0], counts.shape[2]), dtype=np.int32)
    for dst in range(counts.shape[1]):
        row = 0
        for expert in range(counts.shape[2]):
            expert_base[dst, expert] = row
            src_running = 0
            for src in range(counts.shape[0]):
                src_base_by_expert[dst, src, expert] = src_running
                src_running += int(rounded_counts[src, dst, expert])
            row += src_running
    return rounded_counts, expert_base, src_base_by_expert


def _exclusive_cumsum_jax(values: Int[Array, "..."], axis: int) -> Int[Array, "..."]:
    return jnp.cumsum(values, axis=axis, dtype=jnp.int32) - values


def source_push_route_rows_host_from_plan(
    plan: SourcePushPlan,
    *,
    src_base_by_expert: Int[Array, "Dst S E"] | np.ndarray | None = None,
) -> SourcePushRouteRowsHostData:
    """Derive queue route identity and compact expert rows from plan counts.

    ``local_experts`` and ``local_row_starts`` are intentionally not read from
    the plan here. They are cached kernel metadata; the analytical owner is the
    count tensor plus ``block_m`` and the selected compact source bases.
    """

    assignment_ids = np.asarray(jax.device_get(plan.assignment_ids), dtype=np.int32)
    token_ids = np.asarray(jax.device_get(plan.token_ids), dtype=np.int32)
    route_slots = np.asarray(jax.device_get(plan.route_slots), dtype=np.int32)
    valid = np.asarray(jax.device_get(plan.valid_mask), dtype=np.bool_)
    block_m = assignment_ids.shape[-1]
    metadata = source_push_queue_entry_metadata_from_counts(
        plan.counts_by_src_dst_expert,
        block_m,
        entries_per_dst=assignment_ids.shape[2],
    )
    if src_base_by_expert is None:
        src_base_host = np.asarray(jax.device_get(plan.src_base_by_expert), dtype=np.int32)
    else:
        src_base_host = np.asarray(jax.device_get(src_base_by_expert), dtype=np.int32)

    _validate_route_rows_shapes(assignment_ids, token_ids, route_slots, valid, metadata, src_base_host)

    ep_size, dst_ord_count, entries_per_dst, _ = valid.shape
    src_by_entry = np.arange(ep_size, dtype=np.int32)[:, None, None]
    dst_ord_by_entry = np.arange(dst_ord_count, dtype=np.int32)[None, :, None]
    src_by_entry = np.broadcast_to(src_by_entry, (ep_size, dst_ord_count, entries_per_dst))
    dst_by_entry = (src_by_entry + dst_ord_by_entry) % ep_size
    dst_by_entry = np.broadcast_to(dst_by_entry, src_by_entry.shape)

    safe_expert_by_entry = np.maximum(metadata.local_experts, 0)
    row_base = src_base_host[dst_by_entry, src_by_entry, safe_expert_by_entry]
    row_offsets = np.arange(block_m, dtype=np.int32)[None, None, None, :]
    expert_row = row_base[..., None] + metadata.local_row_starts[..., None] + row_offsets

    src = np.broadcast_to(src_by_entry[..., None], valid.shape)
    dst = np.broadcast_to(dst_by_entry[..., None], valid.shape)
    local_expert = np.broadcast_to(safe_expert_by_entry[..., None], valid.shape)
    zeros = np.zeros((), dtype=np.int32)
    return SourcePushRouteRowsHostData(
        src=np.where(valid, src, zeros),
        dst=np.where(valid, dst, zeros),
        local_expert=np.where(valid, local_expert, zeros),
        expert_row=np.where(valid, expert_row, zeros),
        token_id=np.where(valid, np.maximum(token_ids, 0), zeros),
        route_slot=np.where(valid, np.maximum(route_slots, 0), zeros),
        assignment_id=np.where(valid, np.maximum(assignment_ids, 0), zeros),
        valid=valid,
    )


def pack_source_push_tokens(
    x: Float[Array, "S T D"],
    plan: SourcePushPlan,
) -> Float[Array, "S Dst Q M D"]:
    """Pack source tokens into source-push queue order using the inverse plan."""

    x_host = np.asarray(jax.device_get(x))
    token_ids = np.asarray(jax.device_get(plan.token_ids), dtype=np.int32)
    valid_mask = np.asarray(jax.device_get(plan.valid_mask), dtype=np.bool_)
    if x_host.ndim != 3:
        raise ValueError(f"x must have shape [ep_size, T, D], got {x_host.shape}")
    if token_ids.shape[0] != x_host.shape[0]:
        raise ValueError(f"x leading dim {x_host.shape[0]} must match plan source dim {token_ids.shape[0]}")

    packed = np.zeros((*token_ids.shape, x_host.shape[-1]), dtype=x_host.dtype)
    for src in range(token_ids.shape[0]):
        source_valid = valid_mask[src]
        packed_src = packed[src]
        packed_src[source_valid] = x_host[src, token_ids[src][source_valid], :]
        packed[src] = packed_src
    return jnp.asarray(packed)


def pack_source_push_tokens_jax(
    x: Float[Array, "S T D"],
    plan: SourcePushPlan,
) -> Float[Array, "S Dst Q M D"]:
    """Pack source tokens in queue order using JAX gathers from a fixed plan."""

    source_indices = jnp.arange(plan.assignment_ids.shape[0], dtype=jnp.int32)[:, None, None, None]
    token_ids = jnp.maximum(plan.token_ids, 0)
    packed = x.at[source_indices, token_ids].get(
        out_sharding=_source_push_out_sharding(SOURCE_PUSH_MESH_AXIS, None, None, None, None)
    )
    packed = jnp.where(plan.valid_mask[..., None], packed, jnp.zeros((), dtype=x.dtype))
    return _with_source_push_sharding(packed, SOURCE_PUSH_MESH_AXIS, None, None, None, None)


def source_push_queue_route_weights_jax(
    route_weights: Float[Array, "S T K"],
    plan: SourcePushPlan,
) -> Float[Array, "S Dst Q M"]:
    """Gather route weights in source-owned queue order using JAX from a fixed plan."""

    source_indices = jnp.arange(plan.assignment_ids.shape[0], dtype=jnp.int32)[:, None, None, None]
    token_ids = jnp.maximum(plan.token_ids, 0)
    route_slots = jnp.maximum(plan.route_slots, 0)
    queue_weights = route_weights.at[source_indices, token_ids, route_slots].get(
        out_sharding=_source_push_out_sharding(SOURCE_PUSH_MESH_AXIS, None, None, None)
    )
    queue_weights = jnp.where(plan.valid_mask, queue_weights, jnp.zeros((), dtype=route_weights.dtype))
    return _with_source_push_sharding(queue_weights, SOURCE_PUSH_MESH_AXIS, None, None, None)


def source_push_h_row_route_weights_jax(
    route_weights: Float[Array, "S T K"],
    plan: SourcePushPlan,
    send_meta: Int[Array, "S Dst Q F"] | np.ndarray,
    expert_base: Int[Array, "Dst E"] | np.ndarray,
    src_base_by_expert: Int[Array, "Dst S E"] | np.ndarray,
    *,
    hidden_rows_per_rank: int,
    use_exact_expert_major: bool,
) -> Float[Array, "Dst rows"]:
    """Gather route weights into the same flat destination row layout as H."""

    queue_weights = source_push_queue_route_weights_jax(route_weights, plan)
    send_meta = jnp.asarray(send_meta, dtype=jnp.int32)
    expert_base = jnp.asarray(expert_base, dtype=jnp.int32)
    src_base_by_expert = jnp.asarray(src_base_by_expert, dtype=jnp.int32)
    valid_mask = jnp.asarray(plan.valid_mask, dtype=jnp.bool_)

    ep_size, _, entries_per_dst, block_m = valid_mask.shape
    src = jnp.arange(ep_size, dtype=jnp.int32)[:, None, None]
    dst_ordinal = jnp.arange(ep_size, dtype=jnp.int32)[None, :, None]
    src = jnp.broadcast_to(src, (ep_size, ep_size, entries_per_dst))
    dst = (src + dst_ordinal) % ep_size

    metadata_row_start = send_meta[..., SOURCE_PUSH_META_LOCAL_ROW_START]
    if use_exact_expert_major:
        expert = jnp.maximum(send_meta[..., SOURCE_PUSH_META_LOCAL_EXPERT], 0)
        base_row = expert_base.at[dst, expert].get()
        src_base = src_base_by_expert.at[dst, src, expert].get()
        row_start = base_row + src_base + metadata_row_start
    else:
        row_start = metadata_row_start

    row_offsets = jnp.arange(block_m, dtype=jnp.int32)[None, None, None, :]
    flat_row = jnp.where(valid_mask, row_start[..., None] + row_offsets, jnp.zeros((), dtype=jnp.int32))
    flat_dst = jnp.where(valid_mask, jnp.broadcast_to(dst[..., None], flat_row.shape), jnp.zeros((), dtype=jnp.int32))
    weighted_rows = jnp.where(valid_mask, queue_weights, jnp.zeros((), dtype=queue_weights.dtype))
    weighted_rows = _with_source_push_sharding(weighted_rows, None, None, None, None)
    out = jnp.zeros((ep_size, hidden_rows_per_rank), dtype=route_weights.dtype)
    h_row_weights = out.at[flat_dst, flat_row].add(
        weighted_rows,
        out_sharding=_source_push_out_sharding(SOURCE_PUSH_MESH_AXIS, None),
    )
    return _with_source_push_sharding(h_row_weights, SOURCE_PUSH_MESH_AXIS, None)


def source_push_w2_return(
    hidden_expert_major: Float[Array, "Dst rows I"],
    w_down: Float[Array, "Dst E I D"],
    plan: SourcePushPlan,
    *,
    expert_base: Int[Array, "Dst E"] | np.ndarray | None = None,
    src_base_by_expert: Int[Array, "Dst S E"] | np.ndarray | None = None,
) -> Float[Array, "S Dst Q M D"]:
    """Compute W2 from expert-major hidden rows and return rows to source queues.

    Optional bases allow the same source-owned plan to address either exact
    contiguous expert-major rows or the source-padded row layout used by the
    current W13 kernel.
    """

    hidden_host = np.asarray(jax.device_get(hidden_expert_major), dtype=np.float32)
    w_down_host = np.asarray(jax.device_get(w_down), dtype=np.float32)
    assignment_ids = np.asarray(jax.device_get(plan.assignment_ids), dtype=np.int32)
    valid_mask = np.asarray(jax.device_get(plan.valid_mask), dtype=np.bool_)
    local_experts = np.asarray(jax.device_get(plan.local_experts), dtype=np.int32)
    local_row_starts = np.asarray(jax.device_get(plan.local_row_starts), dtype=np.int32)

    if expert_base is None:
        expert_base_host = np.asarray(jax.device_get(plan.expert_base), dtype=np.int32)
    else:
        expert_base_host = np.asarray(jax.device_get(expert_base), dtype=np.int32)
    if src_base_by_expert is None:
        src_base_host = np.asarray(jax.device_get(plan.src_base_by_expert), dtype=np.int32)
    else:
        src_base_host = np.asarray(jax.device_get(src_base_by_expert), dtype=np.int32)

    _validate_w2_return_shapes(hidden_host, w_down_host, assignment_ids, expert_base_host, src_base_host)

    ep_size, dst_ord_count, entries_per_dst, block_m = assignment_ids.shape
    return_y = np.zeros(
        (ep_size, dst_ord_count, entries_per_dst, block_m, w_down_host.shape[-1]), dtype=hidden_host.dtype
    )
    for src in range(ep_size):
        for dst_ord in range(dst_ord_count):
            dst = (src + dst_ord) % ep_size
            for entry in range(entries_per_dst):
                rows = valid_mask[src, dst_ord, entry]
                valid_rows = int(np.sum(rows))
                if valid_rows == 0:
                    continue
                expert = int(local_experts[src, dst_ord, entry])
                row_start = (
                    int(expert_base_host[dst, expert])
                    + int(src_base_host[dst, src, expert])
                    + int(local_row_starts[src, dst_ord, entry])
                )
                hidden_rows = hidden_host[dst, row_start : row_start + valid_rows, :]
                return_y[src, dst_ord, entry, :valid_rows, :] = hidden_rows @ w_down_host[dst, expert]
    return jnp.asarray(return_y)


def source_push_route_buffer(
    return_y: Float[Array, "S Dst Q M D"],
    plan: SourcePushPlan,
) -> Float[Array, "S T K D"]:
    """Scatter returned queue rows into the deterministic source route buffer."""

    return_host = np.asarray(jax.device_get(return_y))
    assignment_ids = np.asarray(jax.device_get(plan.assignment_ids), dtype=np.int32)
    token_ids = np.asarray(jax.device_get(plan.token_ids), dtype=np.int32)
    route_slots = np.asarray(jax.device_get(plan.route_slots), dtype=np.int32)
    weights = np.asarray(jax.device_get(plan.combine_weights))
    valid_mask = np.asarray(jax.device_get(plan.valid_mask), dtype=np.bool_)
    if return_host.shape[:4] != assignment_ids.shape:
        raise ValueError(f"return_y queue shape {return_host.shape[:4]} must match plan {assignment_ids.shape}")

    route_buffer = np.zeros(
        (assignment_ids.shape[0], plan.tokens_per_source, plan.topk, return_host.shape[-1]),
        dtype=return_host.dtype,
    )
    for src in range(assignment_ids.shape[0]):
        for dst_ord in range(assignment_ids.shape[1]):
            for entry in range(assignment_ids.shape[2]):
                for row in range(assignment_ids.shape[3]):
                    if not valid_mask[src, dst_ord, entry, row]:
                        continue
                    token = token_ids[src, dst_ord, entry, row]
                    route_slot = route_slots[src, dst_ord, entry, row]
                    route_buffer[src, token, route_slot, :] = (
                        return_host[src, dst_ord, entry, row, :] * weights[src, dst_ord, entry, row]
                    )
    return jnp.asarray(route_buffer)


def source_push_combine(
    return_y: Float[Array, "S Dst Q M D"],
    plan: SourcePushPlan,
) -> Float[Array, "S T D"]:
    """Combine returned route rows into source-token outputs in fixed slot order."""

    return jnp.sum(source_push_route_buffer(return_y, plan), axis=2)


def source_push_semantic_pair_expert_ids_jax(plan: SourcePushSemanticPlan) -> Int[Array, "S Dst R"]:
    """Return the local expert owning each pair-flat semantic row."""

    return _source_push_semantic_pair_expert_ids_from_counts_jax(
        plan.xcounts,
        rows_per_pair_capacity=plan.assignment_ids.shape[-1],
    )


def source_push_semantic_gather_x_jax(
    x: Float[Array, "S T H"],
    plan: SourcePushSemanticPlan,
) -> Float[Array, "S Dst R H"]:
    """Gather source token rows into pair-flat semantic route order."""

    source_index = jnp.arange(plan.assignment_ids.shape[0], dtype=jnp.int32)[:, None, None]
    token_ids = jnp.maximum(plan.token_ids, 0)
    x_pair = x.at[source_index, token_ids].get()
    return jnp.where(plan.valid_mask[..., None], x_pair, jnp.zeros((), dtype=x.dtype))


def source_push_semantic_x_to_expert_major_jax(
    x: Float[Array, "S T H"],
    plan: SourcePushSemanticPlan,
    *,
    rows_per_expert_capacity: int | None = None,
) -> tuple[Float[Array, "Dst E C H"], Bool[Array, "Dst E C"]]:
    """Produce destination-local expert-major source activations.

    The output row order is the semantic expert-major order defined by
    ``src_base_by_expert`` and ``pair_expert_base``. For each live pair row
    ``(src, dst, pair_row)``, the local expert is inferred from ``xcounts`` and
    the destination expert row is:

    ``src_base_by_expert[dst, src, expert] + pair_row - pair_expert_base[src, dst, expert]``.
    """

    if x.ndim != 3:
        raise ValueError(f"x must have shape [source, token, hidden], got {x.shape}")
    if x.shape[0] != plan.assignment_ids.shape[0]:
        raise ValueError(f"x source dim {x.shape[0]} must match plan source dim {plan.assignment_ids.shape[0]}")
    if rows_per_expert_capacity is None:
        rows_per_expert_capacity = plan.assignment_ids.shape[0] * plan.assignment_ids.shape[-1]
    if rows_per_expert_capacity <= 0:
        raise ValueError(f"rows_per_expert_capacity must be positive, got {rows_per_expert_capacity}")

    expert_ids, expert_rows = _source_push_semantic_expert_row_indices(plan)
    valid = plan.valid_mask & (expert_rows < rows_per_expert_capacity)
    scatter_rows = jnp.where(valid, expert_rows, rows_per_expert_capacity)
    source_index = jnp.arange(plan.assignment_ids.shape[0], dtype=jnp.int32)[:, None, None]
    dst_index = jnp.arange(plan.assignment_ids.shape[1], dtype=jnp.int32)[None, :, None]
    token_ids = jnp.maximum(plan.token_ids, 0)
    source_rows = x.at[source_index, token_ids].get()
    zero = jnp.zeros((), dtype=x.dtype)
    out = jnp.zeros(
        (
            plan.assignment_ids.shape[1],
            plan.xcounts.shape[-1],
            rows_per_expert_capacity,
            x.shape[-1],
        ),
        dtype=x.dtype,
    )
    x_expert = out.at[dst_index, expert_ids, scatter_rows].set(
        jnp.where(valid[..., None], source_rows, zero),
        mode="drop",
    )
    valid_out = jnp.zeros(x_expert.shape[:3], dtype=jnp.bool_)
    valid_out = valid_out.at[dst_index, expert_ids, scatter_rows].set(valid, mode="drop")
    return x_expert, valid_out


def source_push_semantic_w13_reference_jax(
    x: Float[Array, "S T H"],
    w_gate_up: Float[Array, "Dst E H twoI"],
    plan: SourcePushSemanticPlan,
) -> tuple[Float[Array, "S Dst R twoI"], Float[Array, "S Dst R I"]]:
    """Reference W13/SwiGLU forward over pair-flat semantic rows."""

    x_pair = source_push_semantic_gather_x_jax(x, plan).astype(jnp.float32)
    expert_ids = source_push_semantic_pair_expert_ids_jax(plan)
    dst_index = jnp.arange(plan.assignment_ids.shape[1], dtype=jnp.int32)[None, :, None]
    w_pair = w_gate_up.at[dst_index, expert_ids].get().astype(jnp.float32)
    z_pair = jnp.einsum("sdrh,sdrho->sdro", x_pair, w_pair, preferred_element_type=jnp.float32)
    z_pair = jnp.where(plan.valid_mask[..., None], z_pair, jnp.zeros((), dtype=z_pair.dtype))
    gate, up = jnp.split(z_pair, 2, axis=-1)
    h_pair = jax.nn.silu(gate) * up
    return z_pair, h_pair


def source_push_semantic_w2_reference_jax(
    h_pair: Float[Array, "S Dst R I"],
    w_down: Float[Array, "Dst E I H"],
    plan: SourcePushSemanticPlan,
) -> Float[Array, "S Dst R H"]:
    """Reference W2 over pair-flat semantic rows, before route weighting."""

    expert_ids = source_push_semantic_pair_expert_ids_jax(plan)
    dst_index = jnp.arange(plan.assignment_ids.shape[1], dtype=jnp.int32)[None, :, None]
    w_pair = w_down.at[dst_index, expert_ids].get().astype(jnp.float32)
    route_y = jnp.einsum(
        "sdri,sdrih->sdrh",
        h_pair.astype(jnp.float32),
        w_pair,
        preferred_element_type=jnp.float32,
    )
    return jnp.where(plan.valid_mask[..., None], route_y, jnp.zeros((), dtype=route_y.dtype))


def source_push_semantic_combine_jax(
    route_y: Float[Array, "S Dst R H"],
    plan: SourcePushSemanticPlan,
    *,
    preweighted: bool = False,
) -> Float[Array, "S T H"]:
    """Combine pair-flat route rows back to source-token outputs."""

    route_values = route_y if preweighted else route_y * plan.route_weights[..., None].astype(route_y.dtype)
    route_values = jnp.where(plan.valid_mask[..., None], route_values, jnp.zeros((), dtype=route_values.dtype))
    source_index = jnp.arange(plan.assignment_ids.shape[0], dtype=jnp.int32)[:, None, None]
    token_ids = jnp.maximum(plan.token_ids, 0)
    out = jnp.zeros((plan.assignment_ids.shape[0], plan.tokens_per_source, route_y.shape[-1]), dtype=route_y.dtype)
    return out.at[source_index, token_ids].add(route_values)


def source_push_semantic_reverse_route_jax(
    plan: SourcePushSemanticPlan,
    *,
    source_row_base_by_expert: Int[Array, "Dst S E"] | None = None,
) -> SourcePushSemanticReverseRoute:
    """Return source-owned inverse routes for the requested expert-row layout."""

    if source_row_base_by_expert is None:
        return plan.reverse_route
    if source_row_base_by_expert.shape != plan.src_base_by_expert.shape:
        raise ValueError(
            f"source_row_base_by_expert shape {source_row_base_by_expert.shape} must match "
            f"plan source-base shape {plan.src_base_by_expert.shape}"
        )
    if source_row_base_by_expert.dtype != jnp.int32:
        raise ValueError(f"source_row_base_by_expert must have dtype int32, got {source_row_base_by_expert.dtype}")
    return _source_push_semantic_reverse_route_from_metadata_jax(
        assignment_ids=plan.assignment_ids,
        token_ids=plan.token_ids,
        route_slots=plan.route_slots,
        valid_mask=plan.valid_mask,
        xcounts=plan.xcounts,
        pair_expert_base=plan.pair_expert_base,
        src_base_by_expert=source_row_base_by_expert,
        tokens_per_source=plan.tokens_per_source,
        topk=plan.topk,
    )


def source_push_semantic_forward_reference_jax(
    x: Float[Array, "S T H"],
    w_gate_up: Float[Array, "Dst E H twoI"],
    w_down: Float[Array, "Dst E I H"],
    plan: SourcePushSemanticPlan,
) -> tuple[
    Float[Array, "S T H"],
    Float[Array, "S Dst R twoI"],
    Float[Array, "S Dst R I"],
    Float[Array, "S Dst R H"],
]:
    """Reference full MLP forward over the slot-free semantic plan."""

    z_pair, h_pair = source_push_semantic_w13_reference_jax(x, w_gate_up, plan)
    route_y = source_push_semantic_w2_reference_jax(h_pair, w_down, plan)
    y = source_push_semantic_combine_jax(route_y, plan)
    return y, z_pair, h_pair, route_y


def source_push_semantic_backward_source_expand_jax(
    dy: Float[Array, "S T H"],
    route_y: Float[Array, "S Dst R H"],
    plan: SourcePushSemanticPlan,
) -> tuple[Float[Array, "S Dst R H"], Float[Array, "S T K"]]:
    """Expand source-token dy to route dy and compute route-weight gradients."""

    source_index = jnp.arange(plan.assignment_ids.shape[0], dtype=jnp.int32)[:, None, None]
    token_ids = jnp.maximum(plan.token_ids, 0)
    route_slots = jnp.maximum(plan.route_slots, 0)
    dy_pair = dy.at[source_index, token_ids].get()
    dy_pair = jnp.where(plan.valid_mask[..., None], dy_pair, jnp.zeros((), dtype=dy.dtype))
    dy_route = dy_pair * plan.route_weights[..., None].astype(dy_pair.dtype)
    dcombine_values = jnp.sum(
        dy_pair.astype(jnp.float32) * route_y.astype(jnp.float32),
        axis=-1,
    )
    dcombine_values = jnp.where(plan.valid_mask, dcombine_values, jnp.zeros((), dtype=dcombine_values.dtype))
    dcombine = jnp.zeros(
        (plan.assignment_ids.shape[0], plan.tokens_per_source, plan.topk),
        dtype=dcombine_values.dtype,
    )
    dcombine = dcombine.at[source_index, token_ids, route_slots].add(dcombine_values)
    return dy_route, dcombine


def source_push_semantic_w2_backward_reference_jax(
    h_pair: Float[Array, "S Dst R I"],
    dy_route: Float[Array, "S Dst R H"],
    w_down: Float[Array, "Dst E I H"],
    plan: SourcePushSemanticPlan,
) -> tuple[Float[Array, "S Dst R I"], Float[Array, "Dst E I H"]]:
    """Reference W2 backward over pair-flat semantic rows."""

    expert_ids = source_push_semantic_pair_expert_ids_jax(plan)
    dh = jnp.zeros_like(h_pair, dtype=jnp.float32)
    dw2_parts = []
    for expert in range(w_down.shape[1]):
        mask = (expert_ids == expert) & plan.valid_mask
        h_expert = jnp.where(mask[..., None], h_pair.astype(jnp.float32), jnp.zeros((), dtype=jnp.float32))
        dy_expert = jnp.where(mask[..., None], dy_route.astype(jnp.float32), jnp.zeros((), dtype=jnp.float32))
        dw2_parts.append(jnp.einsum("sdri,sdrh->dih", h_expert, dy_expert, preferred_element_type=jnp.float32))
        dh_expert = jnp.einsum(
            "sdrh,dih->sdri",
            dy_expert,
            w_down[:, expert].astype(jnp.float32),
            preferred_element_type=jnp.float32,
        )
        dh = dh + jnp.where(mask[..., None], dh_expert, jnp.zeros((), dtype=dh.dtype))
    return dh, jnp.stack(dw2_parts, axis=1)


def source_push_semantic_swiglu_backward_reference_jax(
    dh_pair: Float[Array, "S Dst R I"],
    z_pair: Float[Array, "S Dst R twoI"],
    plan: SourcePushSemanticPlan,
) -> Float[Array, "S Dst R twoI"]:
    """Reference SwiGLU backward over pair-flat semantic rows."""

    gate, up = jnp.split(z_pair.astype(jnp.float32), 2, axis=-1)
    sigmoid_gate = jax.nn.sigmoid(gate)
    silu_gate = gate * sigmoid_gate
    d_silu_gate = sigmoid_gate * (1.0 + gate * (1.0 - sigmoid_gate))
    dz_gate = dh_pair.astype(jnp.float32) * up * d_silu_gate
    dz_up = dh_pair.astype(jnp.float32) * silu_gate
    dz_pair = jnp.concatenate([dz_gate, dz_up], axis=-1)
    return jnp.where(plan.valid_mask[..., None], dz_pair, jnp.zeros((), dtype=dz_pair.dtype))


def source_push_semantic_w13_backward_reference_jax(
    x: Float[Array, "S T H"],
    dz_pair: Float[Array, "S Dst R twoI"],
    w_gate_up: Float[Array, "Dst E H twoI"],
    plan: SourcePushSemanticPlan,
) -> tuple[Float[Array, "S Dst R H"], Float[Array, "Dst E H twoI"]]:
    """Reference W13 backward over pair-flat semantic rows."""

    x_pair = source_push_semantic_gather_x_jax(x, plan).astype(jnp.float32)
    expert_ids = source_push_semantic_pair_expert_ids_jax(plan)
    dx_pair = jnp.zeros_like(x_pair, dtype=jnp.float32)
    dw13_parts = []
    for expert in range(w_gate_up.shape[1]):
        mask = (expert_ids == expert) & plan.valid_mask
        x_expert = jnp.where(mask[..., None], x_pair, jnp.zeros((), dtype=jnp.float32))
        dz_expert = jnp.where(mask[..., None], dz_pair.astype(jnp.float32), jnp.zeros((), dtype=jnp.float32))
        dw13_parts.append(jnp.einsum("sdrh,sdro->dho", x_expert, dz_expert, preferred_element_type=jnp.float32))
        dx_expert = jnp.einsum(
            "sdro,dho->sdrh",
            dz_expert,
            w_gate_up[:, expert].astype(jnp.float32),
            preferred_element_type=jnp.float32,
        )
        dx_pair = dx_pair + jnp.where(mask[..., None], dx_expert, jnp.zeros((), dtype=dx_pair.dtype))
    return dx_pair, jnp.stack(dw13_parts, axis=1)


def source_push_semantic_dx_combine_jax(
    dx_pair: Float[Array, "S Dst R H"],
    plan: SourcePushSemanticPlan,
) -> Float[Array, "S T H"]:
    """Combine route-level dx rows back to source-token dx."""

    return source_push_semantic_combine_jax(dx_pair, plan, preweighted=True)


def source_push_semantic_pair_to_expert_major_jax(
    pair_values: Float[Array, "S Dst R F"],
    plan: SourcePushSemanticPlan,
    *,
    rows_per_expert_capacity: int,
) -> tuple[Float[Array, "Dst E C F"], Bool[Array, "Dst E C"]]:
    """Scatter pair-flat semantic rows to destination-local expert-major rows."""

    if rows_per_expert_capacity <= 0:
        raise ValueError(f"rows_per_expert_capacity must be positive, got {rows_per_expert_capacity}")
    if pair_values.shape[:3] != plan.assignment_ids.shape:
        raise ValueError(
            f"pair_values shape {pair_values.shape[:3]} must match semantic rows {plan.assignment_ids.shape}"
        )

    expert_ids, expert_rows = _source_push_semantic_expert_row_indices(plan)
    valid = plan.valid_mask & (expert_rows < rows_per_expert_capacity)
    scatter_rows = jnp.where(valid, expert_rows, rows_per_expert_capacity)
    dst_index = jnp.arange(plan.assignment_ids.shape[1], dtype=jnp.int32)[None, :, None]
    out = jnp.zeros(
        (
            plan.assignment_ids.shape[1],
            plan.xcounts.shape[-1],
            rows_per_expert_capacity,
            pair_values.shape[-1],
        ),
        dtype=pair_values.dtype,
    )
    expert_values = out.at[dst_index, expert_ids, scatter_rows].set(
        jnp.where(valid[..., None], pair_values, jnp.zeros((), dtype=pair_values.dtype)),
        mode="drop",
    )
    valid_out = jnp.zeros(expert_values.shape[:3], dtype=jnp.bool_)
    valid_out = valid_out.at[dst_index, expert_ids, scatter_rows].set(valid, mode="drop")
    return expert_values, valid_out


def source_push_semantic_expert_major_to_pair_jax(
    expert_values: Float[Array, "Dst E C F"],
    plan: SourcePushSemanticPlan,
) -> Float[Array, "S Dst R F"]:
    """Gather destination-local expert-major rows back to pair-flat semantic order."""

    if expert_values.ndim != 4:
        raise ValueError(f"expert_values must have shape [Dst, E, C, F], got {expert_values.shape}")
    if expert_values.shape[0] != plan.assignment_ids.shape[1] or expert_values.shape[1] != plan.xcounts.shape[-1]:
        raise ValueError(
            f"expert_values leading shape {expert_values.shape[:2]} must match "
            f"{(plan.assignment_ids.shape[1], plan.xcounts.shape[-1])}"
        )

    expert_ids, expert_rows = _source_push_semantic_expert_row_indices(plan)
    dst_index = jnp.arange(plan.assignment_ids.shape[1], dtype=jnp.int32)[None, :, None]
    safe_rows = jnp.minimum(expert_rows, expert_values.shape[2] - 1)
    pair_out_sharding = _source_push_out_sharding(None, None, None, None)
    if pair_out_sharding is None:
        pair_values = expert_values.at[dst_index, expert_ids, safe_rows].get()
    else:
        pair_values = expert_values.at[dst_index, expert_ids, safe_rows].get(out_sharding=pair_out_sharding)
    valid = plan.valid_mask & (expert_rows < expert_values.shape[2])
    return jnp.where(valid[..., None], pair_values, jnp.zeros((), dtype=expert_values.dtype))


def source_push_semantic_route_weights_expert_major_jax(
    plan: SourcePushSemanticPlan,
    *,
    rows_per_expert_capacity: int,
) -> tuple[Float[Array, "Dst E C"], Bool[Array, "Dst E C"]]:
    """Scatter semantic route weights to destination-local expert-major rows."""

    route_weights, valid = source_push_semantic_pair_to_expert_major_jax(
        plan.route_weights[..., None],
        plan,
        rows_per_expert_capacity=rows_per_expert_capacity,
    )
    return route_weights[..., 0], valid


def _source_push_semantic_pair_expert_ids_from_counts_jax(
    xcounts: Int[Array, "S Dst E"],
    *,
    rows_per_pair_capacity: int,
) -> Int[Array, "S Dst R"]:
    rows = jnp.arange(rows_per_pair_capacity, dtype=jnp.int32)
    pair_ends = jnp.cumsum(xcounts, axis=2, dtype=jnp.int32)

    def pair_expert_ids(ends):
        expert = jnp.searchsorted(ends, rows, side="right").astype(jnp.int32)
        return jnp.minimum(expert, xcounts.shape[-1] - 1)

    return jax.vmap(jax.vmap(pair_expert_ids, in_axes=0), in_axes=0)(pair_ends)


def _source_push_semantic_expert_row_indices_from_metadata_jax(
    *,
    xcounts: Int[Array, "S Dst E"],
    pair_expert_base: Int[Array, "S Dst E"],
    src_base_by_expert: Int[Array, "Dst S E"],
    rows_per_pair_capacity: int,
) -> tuple[Int[Array, "S Dst R"], Int[Array, "S Dst R"]]:
    expert_ids = _source_push_semantic_pair_expert_ids_from_counts_jax(
        xcounts,
        rows_per_pair_capacity=rows_per_pair_capacity,
    )
    rows = jnp.arange(rows_per_pair_capacity, dtype=jnp.int32)[None, None, :]
    src_index = jnp.arange(xcounts.shape[0], dtype=jnp.int32)[:, None, None]
    dst_index = jnp.arange(xcounts.shape[1], dtype=jnp.int32)[None, :, None]
    pair_base = pair_expert_base.at[src_index, dst_index, expert_ids].get()
    src_base = src_base_by_expert.at[dst_index, src_index, expert_ids].get()
    expert_rows = src_base + rows - pair_base
    return expert_ids, jnp.maximum(expert_rows, 0).astype(jnp.int32)


def _source_push_semantic_reverse_route_from_metadata_jax(
    *,
    assignment_ids: Int[Array, "S Dst R"],
    token_ids: Int[Array, "S Dst R"],
    route_slots: Int[Array, "S Dst R"],
    valid_mask: Bool[Array, "S Dst R"],
    xcounts: Int[Array, "S Dst E"],
    pair_expert_base: Int[Array, "S Dst E"],
    src_base_by_expert: Int[Array, "Dst S E"],
    tokens_per_source: int,
    topk: int,
) -> SourcePushSemanticReverseRoute:
    """Build the static-shape source-owned inverse table from canonical row metadata."""

    expert_ids, expert_rows = _source_push_semantic_expert_row_indices_from_metadata_jax(
        xcounts=xcounts,
        pair_expert_base=pair_expert_base,
        src_base_by_expert=src_base_by_expert,
        rows_per_pair_capacity=assignment_ids.shape[-1],
    )
    source_index = jnp.arange(assignment_ids.shape[0], dtype=jnp.int32)[:, None, None]
    dst_index = jnp.arange(assignment_ids.shape[1], dtype=jnp.int32)[None, :, None]
    safe_token_ids = jnp.where(valid_mask, jnp.maximum(token_ids, 0), tokens_per_source)
    safe_route_slots = jnp.where(valid_mask, jnp.maximum(route_slots, 0), topk)
    route_shape = (assignment_ids.shape[0], tokens_per_source, topk)

    route_dst = jnp.zeros(route_shape, dtype=jnp.int32)
    route_dst = route_dst.at[source_index, safe_token_ids, safe_route_slots].set(
        jnp.where(valid_mask, dst_index, jnp.zeros((), dtype=jnp.int32)),
        mode="drop",
    )
    route_expert = jnp.zeros(route_shape, dtype=jnp.int32)
    route_expert = route_expert.at[source_index, safe_token_ids, safe_route_slots].set(
        jnp.where(valid_mask, expert_ids, jnp.zeros((), dtype=jnp.int32)),
        mode="drop",
    )
    route_expert_row = jnp.zeros(route_shape, dtype=jnp.int32)
    route_expert_row = route_expert_row.at[source_index, safe_token_ids, safe_route_slots].set(
        jnp.where(valid_mask, expert_rows, jnp.zeros((), dtype=jnp.int32)),
        mode="drop",
    )
    route_valid = jnp.zeros(route_shape, dtype=jnp.bool_)
    route_valid = route_valid.at[source_index, safe_token_ids, safe_route_slots].set(valid_mask, mode="drop")
    inverse_assignment_id = jnp.full(route_shape, INVALID_ASSIGNMENT_ID, dtype=jnp.int32)
    inverse_assignment_id = inverse_assignment_id.at[source_index, safe_token_ids, safe_route_slots].set(
        jnp.where(valid_mask, jnp.maximum(assignment_ids, 0), INVALID_ASSIGNMENT_ID),
        mode="drop",
    )
    return SourcePushSemanticReverseRoute(
        route_dst=route_dst,
        route_expert=route_expert,
        route_expert_row=route_expert_row,
        route_valid=route_valid,
        assignment_id=inverse_assignment_id,
    )


def _source_push_semantic_expert_row_indices(
    plan: SourcePushSemanticPlan,
) -> tuple[Int[Array, "S Dst R"], Int[Array, "S Dst R"]]:
    return _source_push_semantic_expert_row_indices_from_metadata_jax(
        xcounts=plan.xcounts,
        pair_expert_base=plan.pair_expert_base,
        src_base_by_expert=plan.src_base_by_expert,
        rows_per_pair_capacity=plan.assignment_ids.shape[-1],
    )


def _source_group_sizes(selected_experts: np.ndarray, global_experts: int) -> np.ndarray:
    source_count = selected_experts.shape[0]
    group_sizes = np.zeros((source_count, global_experts), dtype=np.int32)
    for src in range(source_count):
        group_sizes[src] = np.bincount(selected_experts[src].reshape(-1), minlength=global_experts).astype(np.int32)
    return group_sizes


def _ceil_div(values: np.ndarray, divisor: int) -> np.ndarray:
    return (values + divisor - 1) // divisor


def _exclusive_cumsum(values: np.ndarray, axis: int) -> np.ndarray:
    cumsum = np.cumsum(values, axis=axis, dtype=np.int32)
    return cumsum - values


def _counts_host(counts_by_src_dst_expert: Int[Array, "S Dst E"] | np.ndarray) -> np.ndarray:
    counts = np.asarray(jax.device_get(counts_by_src_dst_expert), dtype=np.int32)
    if counts.ndim != 3:
        raise ValueError(f"counts_by_src_dst_expert must have shape [source, destination, expert], got {counts.shape}")
    if np.any(counts < 0):
        raise ValueError("counts_by_src_dst_expert must be non-negative")
    return counts


def _required_entries_per_dst(counts_by_src_dst_expert: np.ndarray, block_m: int) -> int:
    entries_required = np.sum(_ceil_div(counts_by_src_dst_expert, block_m), axis=2)
    return int(np.max(entries_required)) if entries_required.size else 0


def _validate_route_rows_shapes(
    assignment_ids: np.ndarray,
    token_ids: np.ndarray,
    route_slots: np.ndarray,
    valid: np.ndarray,
    metadata: SourcePushQueueEntryMetadata,
    src_base_by_expert: np.ndarray,
) -> None:
    if token_ids.shape != assignment_ids.shape:
        raise ValueError(f"token_ids shape {token_ids.shape} must match assignment_ids {assignment_ids.shape}")
    if route_slots.shape != assignment_ids.shape:
        raise ValueError(f"route_slots shape {route_slots.shape} must match assignment_ids {assignment_ids.shape}")
    if valid.shape != assignment_ids.shape:
        raise ValueError(f"valid shape {valid.shape} must match assignment_ids {assignment_ids.shape}")

    entry_shape = assignment_ids.shape[:3]
    if metadata.local_experts.shape != entry_shape:
        raise ValueError(f"derived local_experts shape {metadata.local_experts.shape} must match {entry_shape}")
    if metadata.local_row_starts.shape != entry_shape:
        raise ValueError(f"derived local_row_starts shape {metadata.local_row_starts.shape} must match {entry_shape}")

    ep_size = assignment_ids.shape[0]
    if src_base_by_expert.shape[0] != ep_size or src_base_by_expert.shape[1] != ep_size:
        raise ValueError(f"src_base_by_expert shape {src_base_by_expert.shape} must start with {(ep_size, ep_size)}")


def _validate_w2_return_shapes(
    hidden: np.ndarray,
    w_down: np.ndarray,
    assignment_ids: np.ndarray,
    expert_base: np.ndarray,
    src_base_by_expert: np.ndarray,
) -> None:
    if hidden.ndim != 3:
        raise ValueError(f"hidden_expert_major must have shape [dst, rows, I], got {hidden.shape}")
    if w_down.ndim != 4:
        raise ValueError(f"w_down must have shape [dst, expert, I, D], got {w_down.shape}")
    ep_size = assignment_ids.shape[0]
    experts_per_rank = w_down.shape[1]
    if hidden.shape[0] != ep_size:
        raise ValueError(f"hidden destination dim {hidden.shape[0]} must match plan ep_size {ep_size}")
    if w_down.shape[0] != ep_size:
        raise ValueError(f"w_down destination dim {w_down.shape[0]} must match plan ep_size {ep_size}")
    if hidden.shape[-1] != w_down.shape[-2]:
        raise ValueError(f"hidden I dim {hidden.shape[-1]} must match w_down I dim {w_down.shape[-2]}")
    if expert_base.shape != (ep_size, experts_per_rank):
        raise ValueError(f"expert_base shape {expert_base.shape} must be {(ep_size, experts_per_rank)}")
    if src_base_by_expert.shape != (ep_size, ep_size, experts_per_rank):
        raise ValueError(
            f"src_base_by_expert shape {src_base_by_expert.shape} must be {(ep_size, ep_size, experts_per_rank)}"
        )


def source_push_w13_h(
    x: Float[Array, "S Dst Q M D"],
    w_gate_up: Float[Array, "Dst E D twoI"],
    plan: SourcePushPlan,
    *,
    src_base_by_expert: Int[Array, "Dst S E"] | np.ndarray | None = None,
    expert_capacity: int | None = None,
) -> Float[Array, "Dst E C twoI"]:
    """Compute W13 preactivation rows in source-push expert-major layout."""

    x_host = np.asarray(jax.device_get(x), dtype=np.float32)
    w_host = np.asarray(jax.device_get(w_gate_up), dtype=np.float32)
    valid_mask = np.asarray(jax.device_get(plan.valid_mask), dtype=np.bool_)
    local_experts = np.asarray(jax.device_get(plan.local_experts), dtype=np.int32)
    local_row_starts = np.asarray(jax.device_get(plan.local_row_starts), dtype=np.int32)
    if src_base_by_expert is None:
        src_base_host = np.asarray(jax.device_get(plan.src_base_by_expert), dtype=np.int32)
    else:
        src_base_host = np.asarray(jax.device_get(src_base_by_expert), dtype=np.int32)

    _validate_w13_h_shapes(x_host, w_host, valid_mask, src_base_host)
    if expert_capacity is None:
        expert_capacity = _expert_capacity_for_source_bases(plan, src_base_host)

    h = np.zeros((valid_mask.shape[1], src_base_host.shape[-1], expert_capacity, w_host.shape[-1]), dtype=np.float32)
    for src in range(valid_mask.shape[0]):
        for dst_ord in range(valid_mask.shape[1]):
            dst = (src + dst_ord) % valid_mask.shape[1]
            for entry in range(valid_mask.shape[2]):
                rows = valid_mask[src, dst_ord, entry]
                valid_rows = int(np.sum(rows))
                if valid_rows == 0:
                    continue
                expert = int(local_experts[src, dst_ord, entry])
                row_start = int(src_base_host[dst, src, expert]) + int(local_row_starts[src, dst_ord, entry])
                x_rows = x_host[src, dst_ord, entry, :valid_rows, :]
                h[dst, expert, row_start : row_start + valid_rows, :] = x_rows @ w_host[dst, expert]
    return jnp.asarray(h)


def source_push_w2_from_h_return(
    h_expert_major: Float[Array, "Dst E C twoI"],
    route_weights: Float[Array, "S T K"],
    w_down: Float[Array, "Dst E I D"],
    plan: SourcePushPlan,
    *,
    src_base_by_expert: Int[Array, "Dst S E"] | np.ndarray | None = None,
) -> Float[Array, "S Dst Q M D"]:
    """Compute W2 returns from W13 preactivation H with route weights before W2."""

    h_host = np.asarray(jax.device_get(h_expert_major), dtype=np.float32)
    route_weights_host = np.asarray(jax.device_get(route_weights), dtype=np.float32)
    w_down_host = np.asarray(jax.device_get(w_down), dtype=np.float32)
    valid_mask = np.asarray(jax.device_get(plan.valid_mask), dtype=np.bool_)
    token_ids = np.asarray(jax.device_get(plan.token_ids), dtype=np.int32)
    route_slots = np.asarray(jax.device_get(plan.route_slots), dtype=np.int32)
    local_experts = np.asarray(jax.device_get(plan.local_experts), dtype=np.int32)
    local_row_starts = np.asarray(jax.device_get(plan.local_row_starts), dtype=np.int32)
    if src_base_by_expert is None:
        src_base_host = np.asarray(jax.device_get(plan.src_base_by_expert), dtype=np.int32)
    else:
        src_base_host = np.asarray(jax.device_get(src_base_by_expert), dtype=np.int32)

    _validate_w2_from_h_shapes(h_host, route_weights_host, w_down_host, valid_mask, src_base_host)
    return_y = np.zeros((*valid_mask.shape, w_down_host.shape[-1]), dtype=np.float32)
    for src in range(valid_mask.shape[0]):
        for dst_ord in range(valid_mask.shape[1]):
            dst = (src + dst_ord) % valid_mask.shape[1]
            for entry in range(valid_mask.shape[2]):
                rows = valid_mask[src, dst_ord, entry]
                valid_rows = int(np.sum(rows))
                if valid_rows == 0:
                    continue
                expert = int(local_experts[src, dst_ord, entry])
                row_start = int(src_base_host[dst, src, expert]) + int(local_row_starts[src, dst_ord, entry])
                h_rows = h_host[dst, expert, row_start : row_start + valid_rows, :]
                intermediate_dim = h_rows.shape[-1] // 2
                gate = h_rows[:, :intermediate_dim]
                up = h_rows[:, intermediate_dim:]
                activation = gate * (1.0 / (1.0 + np.exp(-gate))) * up
                tokens = token_ids[src, dst_ord, entry, :valid_rows]
                slots = route_slots[src, dst_ord, entry, :valid_rows]
                weights = route_weights_host[src, tokens, slots]
                weighted_activation = activation * weights[:, None]
                return_y[src, dst_ord, entry, :valid_rows, :] = weighted_activation @ w_down_host[dst, expert]
    return jnp.asarray(return_y)


def source_push_combine_preweighted(
    return_y: Float[Array, "S Dst Q M D"],
    plan: SourcePushPlan,
) -> Float[Array, "S T D"]:
    """Combine W2 rows that already include route weights."""

    return_host = np.asarray(jax.device_get(return_y))
    token_ids = np.asarray(jax.device_get(plan.token_ids), dtype=np.int32)
    route_slots = np.asarray(jax.device_get(plan.route_slots), dtype=np.int32)
    valid_mask = np.asarray(jax.device_get(plan.valid_mask), dtype=np.bool_)
    if return_host.shape[:4] != valid_mask.shape:
        raise ValueError(f"return_y queue shape {return_host.shape[:4]} must match plan {valid_mask.shape}")

    route_buffer = np.zeros(
        (valid_mask.shape[0], plan.tokens_per_source, plan.topk, return_host.shape[-1]),
        dtype=return_host.dtype,
    )
    for src in range(valid_mask.shape[0]):
        for dst_ord in range(valid_mask.shape[1]):
            for entry in range(valid_mask.shape[2]):
                for row in range(valid_mask.shape[3]):
                    if not valid_mask[src, dst_ord, entry, row]:
                        continue
                    token = token_ids[src, dst_ord, entry, row]
                    route_slot = route_slots[src, dst_ord, entry, row]
                    route_buffer[src, token, route_slot, :] = return_host[src, dst_ord, entry, row, :]
    return jnp.asarray(np.sum(route_buffer, axis=2))


def _expert_capacity_for_source_bases(plan: SourcePushPlan, src_base_by_expert: np.ndarray) -> int:
    valid_mask = np.asarray(jax.device_get(plan.valid_mask), dtype=np.bool_)
    local_experts = np.asarray(jax.device_get(plan.local_experts), dtype=np.int32)
    local_row_starts = np.asarray(jax.device_get(plan.local_row_starts), dtype=np.int32)
    max_row = 0
    for src in range(valid_mask.shape[0]):
        for dst_ord in range(valid_mask.shape[1]):
            dst = (src + dst_ord) % valid_mask.shape[1]
            for entry in range(valid_mask.shape[2]):
                valid_rows = int(np.sum(valid_mask[src, dst_ord, entry]))
                if valid_rows == 0:
                    continue
                expert = int(local_experts[src, dst_ord, entry])
                row_start = int(src_base_by_expert[dst, src, expert]) + int(local_row_starts[src, dst_ord, entry])
                max_row = max(max_row, row_start + valid_rows)
    return max_row


def _validate_w13_h_shapes(
    x: np.ndarray,
    w_gate_up: np.ndarray,
    valid_mask: np.ndarray,
    src_base_by_expert: np.ndarray,
) -> None:
    if x.ndim != 5:
        raise ValueError(f"x must have shape [src, dst, entry, row, D], got {x.shape}")
    if w_gate_up.ndim != 4:
        raise ValueError(f"w_gate_up must have shape [dst, expert, D, 2I], got {w_gate_up.shape}")
    ep_size = valid_mask.shape[0]
    experts_per_rank = w_gate_up.shape[1]
    if x.shape[:4] != valid_mask.shape:
        raise ValueError(f"x queue shape {x.shape[:4]} must match plan {valid_mask.shape}")
    if w_gate_up.shape[0] != ep_size or w_gate_up.shape[2] != x.shape[-1]:
        raise ValueError(f"w_gate_up shape {w_gate_up.shape} is incompatible with x shape {x.shape}")
    if src_base_by_expert.shape != (ep_size, ep_size, experts_per_rank):
        raise ValueError(
            f"src_base_by_expert shape {src_base_by_expert.shape} must be {(ep_size, ep_size, experts_per_rank)}"
        )


def _validate_w2_from_h_shapes(
    h: np.ndarray,
    route_weights: np.ndarray,
    w_down: np.ndarray,
    valid_mask: np.ndarray,
    src_base_by_expert: np.ndarray,
) -> None:
    if h.ndim != 4:
        raise ValueError(f"h_expert_major must have shape [dst, expert, capacity, 2I], got {h.shape}")
    if route_weights.ndim != 3:
        raise ValueError(f"route_weights must have shape [src, token, topk], got {route_weights.shape}")
    if w_down.ndim != 4:
        raise ValueError(f"w_down must have shape [dst, expert, I, D], got {w_down.shape}")
    ep_size = valid_mask.shape[0]
    experts_per_rank = w_down.shape[1]
    if route_weights.shape[0] != ep_size:
        raise ValueError(f"route_weights source dim {route_weights.shape[0]} must match plan ep_size {ep_size}")
    if h.shape[0] != ep_size or h.shape[1] != experts_per_rank:
        raise ValueError(f"h shape {h.shape} must start with {(ep_size, experts_per_rank)}")
    if w_down.shape[0] != ep_size:
        raise ValueError(f"w_down destination dim {w_down.shape[0]} must match plan ep_size {ep_size}")
    if h.shape[-1] != 2 * w_down.shape[-2]:
        raise ValueError(f"h trailing dim {h.shape[-1]} must equal 2 * w_down I dim {w_down.shape[-2]}")
    if src_base_by_expert.shape != (ep_size, ep_size, experts_per_rank):
        raise ValueError(
            f"src_base_by_expert shape {src_base_by_expert.shape} must be {(ep_size, ep_size, experts_per_rank)}"
        )
