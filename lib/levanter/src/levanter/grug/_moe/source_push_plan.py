# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Host-side plan for an invertible source-push MGPU MoE forward path."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

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


def source_push_recv_route_weights_jax(
    route_weights: Float[Array, "S T K"],
    plan: SourcePushPlan,
) -> Float[Array, "Dst Src Q M"]:
    """Gather route weights in destination receive order using JAX from a fixed plan."""

    queue_weights = source_push_queue_route_weights_jax(route_weights, plan)
    ep_size = plan.assignment_ids.shape[0]
    recv_by_dst = []
    for dst in range(ep_size):
        recv_sources = []
        for recv_ord in range(ep_size):
            src = (dst + recv_ord) % ep_size
            send_dst_ord = dst_ordinal(src, dst, ep_size)
            recv_sources.append(
                queue_weights.at[src, send_dst_ord].get(out_sharding=_source_push_out_sharding(None, None))
            )
        recv_by_dst.append(jnp.stack(recv_sources, axis=0))
    recv_weights = jnp.stack(recv_by_dst, axis=0)
    return _with_source_push_sharding(recv_weights, SOURCE_PUSH_MESH_AXIS, None, None, None)


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


def source_push_destination_local_x_jax(
    packed_x: Float[Array, "S Dst Q M D"],
    plan: SourcePushPlan,
    send_meta: Int[Array, "S Dst Q F"] | np.ndarray,
    expert_base: Int[Array, "Dst E"] | np.ndarray,
    src_base_by_expert: Int[Array, "Dst S E"] | np.ndarray,
    *,
    hidden_rows_per_rank: int,
    use_exact_expert_major: bool,
) -> Float[Array, "Dst rows D"]:
    """Scatter source-packed token rows into the destination-local H row layout."""

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
    x_rows = jnp.where(valid_mask[..., None], packed_x, jnp.zeros((), dtype=packed_x.dtype))
    x_rows = _with_source_push_sharding(x_rows, None, None, None, None, None)
    out = jnp.zeros((ep_size, hidden_rows_per_rank, packed_x.shape[-1]), dtype=packed_x.dtype)
    destination_x = out.at[flat_dst, flat_row].add(
        x_rows,
        out_sharding=_source_push_out_sharding(SOURCE_PUSH_MESH_AXIS, None, None),
    )
    return _with_source_push_sharding(destination_x, SOURCE_PUSH_MESH_AXIS, None, None)


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


def source_push_recv_route_weights(
    route_weights: Float[Array, "S T K"],
    plan: SourcePushPlan,
) -> Float[Array, "Dst Src Q M"]:
    """Gather live route weights into destination receive order."""

    route_weights_host = np.asarray(jax.device_get(route_weights))
    token_ids = np.asarray(jax.device_get(plan.token_ids), dtype=np.int32)
    route_slots = np.asarray(jax.device_get(plan.route_slots), dtype=np.int32)
    valid_mask = np.asarray(jax.device_get(plan.valid_mask), dtype=np.bool_)
    if route_weights_host.ndim != 3:
        raise ValueError(f"route_weights must have shape [src, token, topk], got {route_weights_host.shape}")
    if route_weights_host.shape[0] != valid_mask.shape[0]:
        raise ValueError(
            f"route_weights source dim {route_weights_host.shape[0]} must match plan ep_size {valid_mask.shape[0]}"
        )

    recv_weights = np.zeros(
        (valid_mask.shape[1], valid_mask.shape[0], *valid_mask.shape[2:]), dtype=route_weights_host.dtype
    )
    for src in range(valid_mask.shape[0]):
        for dst_ord in range(valid_mask.shape[1]):
            dst = (src + dst_ord) % valid_mask.shape[1]
            recv_ord = recv_src_ordinal(dst, src, valid_mask.shape[1])
            rows = valid_mask[src, dst_ord]
            tokens = token_ids[src, dst_ord]
            slots = route_slots[src, dst_ord]
            recv_weights[dst, recv_ord][rows] = route_weights_host[src, tokens[rows], slots[rows]]
    return jnp.asarray(recv_weights)


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
