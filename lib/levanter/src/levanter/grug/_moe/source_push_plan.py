# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Host-side plan for an invertible source-push MGPU MoE forward path."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float, Int

from levanter.grug._moe.ep_common import _clip_receiver_group_sizes


SOURCE_PUSH_META_SRC_RANK = 0
SOURCE_PUSH_META_LOCAL_EXPERT = 1
SOURCE_PUSH_META_LOCAL_ROW_START = 2
SOURCE_PUSH_META_VALID_ROWS = 3
SOURCE_PUSH_META_FIELDS = 4
INVALID_ASSIGNMENT_ID = -1


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


def dst_ordinal(src: int, dst: int, ep_size: int) -> int:
    """Return the source-local destination ordinal used by the transport queue."""

    return (dst - src) % ep_size


def recv_src_ordinal(dst: int, src: int, ep_size: int) -> int:
    """Return the destination-local source ordinal used by receive metadata."""

    return (src - dst) % ep_size


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

    entries_required = np.sum(_ceil_div(counts_by_src_dst_expert, block_m), axis=2)
    required_entries_per_dst = int(np.max(entries_required)) if entries_required.size else 0
    if entries_per_dst is None:
        entries_per_dst = required_entries_per_dst
    if entries_per_dst < required_entries_per_dst:
        raise ValueError(
            "source-push queue capacity overflow: "
            f"entries_per_dst={entries_per_dst} but required {required_entries_per_dst}"
        )

    rows_per_local_expert = np.sum(counts_by_src_dst_expert, axis=0, dtype=np.int32)
    expert_base = _exclusive_cumsum(rows_per_local_expert, axis=1)
    src_base_by_expert = np.zeros((ep_size, ep_size, experts_per_rank), dtype=np.int32)
    for dst in range(ep_size):
        src_base_by_expert[dst] = _exclusive_cumsum(counts_by_src_dst_expert[:, dst, :], axis=0)

    queue_shape = (ep_size, ep_size, entries_per_dst, block_m)
    assignment_ids = np.full(queue_shape, INVALID_ASSIGNMENT_ID, dtype=np.int32)
    token_ids = np.full(queue_shape, INVALID_ASSIGNMENT_ID, dtype=np.int32)
    route_slots = np.full(queue_shape, INVALID_ASSIGNMENT_ID, dtype=np.int32)
    combine_weights_host = np.zeros(queue_shape, dtype=weights_host.dtype)
    valid_mask = np.zeros(queue_shape, dtype=np.bool_)
    local_experts = np.full((ep_size, ep_size, entries_per_dst), INVALID_ASSIGNMENT_ID, dtype=np.int32)
    local_row_starts = np.zeros((ep_size, ep_size, entries_per_dst), dtype=np.int32)
    send_meta = np.zeros((ep_size, ep_size, entries_per_dst, SOURCE_PUSH_META_FIELDS), dtype=np.int32)

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
