# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from levanter.grug._moe.common import _prepare_moe_dispatch_indices_with_assignment_ids
from levanter.grug._moe.ep_common import _clip_receiver_group_sizes, _expert_prefix_keep_mask
from levanter.grug._moe.source_push_plan import (
    INVALID_ASSIGNMENT_ID,
    SOURCE_PUSH_META_LOCAL_EXPERT,
    SOURCE_PUSH_META_LOCAL_ROW_START,
    SOURCE_PUSH_META_SRC_RANK,
    SOURCE_PUSH_META_VALID_ROWS,
    build_source_push_plan,
    dst_ordinal,
    pack_source_push_tokens,
    source_push_combine,
    recv_src_ordinal,
    source_push_plan_row_stats,
    source_push_route_buffer,
    source_push_source_padded_row_bases,
    source_push_w2_return,
)


EP_SIZE = 2
EXPERTS_PER_RANK = 2
BLOCK_M = 2


def _small_routing_inputs() -> tuple[jax.Array, jax.Array]:
    selected_experts = jnp.array(
        [
            [[2, 0], [3, 2], [1, 3], [0, 2]],
            [[0, 2], [1, 3], [2, 0], [1, 3]],
        ],
        dtype=jnp.int32,
    )
    combine_weights = jnp.arange(selected_experts.size, dtype=jnp.float32).reshape(selected_experts.shape) + 1.0
    return selected_experts, combine_weights


def _expert_major_row_count(expert_base: np.ndarray, rows_per_local_expert: np.ndarray) -> int:
    return int(np.max(expert_base + rows_per_local_expert))


def test_source_push_plan_queues_source_assignments_in_destination_expert_order():
    selected_experts, combine_weights = _small_routing_inputs()
    plan = build_source_push_plan(
        selected_experts,
        combine_weights,
        ep_size=EP_SIZE,
        experts_per_rank=EXPERTS_PER_RANK,
        block_m=BLOCK_M,
        capacity_factor=2.0,
    )

    assignment_ids = np.asarray(plan.assignment_ids)
    valid_mask = np.asarray(plan.valid_mask)
    local_experts = np.asarray(plan.local_experts)
    local_row_starts = np.asarray(plan.local_row_starts)

    assert assignment_ids.shape == (EP_SIZE, EP_SIZE, 3, BLOCK_M)
    np.testing.assert_array_equal(assignment_ids[0, dst_ordinal(0, 0, EP_SIZE), 0], [1, 6])
    np.testing.assert_array_equal(assignment_ids[0, dst_ordinal(0, 0, EP_SIZE), 1], [4, INVALID_ASSIGNMENT_ID])
    np.testing.assert_array_equal(assignment_ids[0, dst_ordinal(0, 1, EP_SIZE), 0], [0, 3])
    np.testing.assert_array_equal(assignment_ids[0, dst_ordinal(0, 1, EP_SIZE), 1], [7, INVALID_ASSIGNMENT_ID])
    np.testing.assert_array_equal(assignment_ids[0, dst_ordinal(0, 1, EP_SIZE), 2], [2, 5])

    np.testing.assert_array_equal(local_experts[0, dst_ordinal(0, 1, EP_SIZE)], [0, 0, 1])
    np.testing.assert_array_equal(local_row_starts[0, dst_ordinal(0, 1, EP_SIZE)], [0, 2, 0])
    np.testing.assert_array_equal(valid_mask[0, dst_ordinal(0, 0, EP_SIZE), 1], [True, False])


def test_source_push_plan_metadata_preserves_assignment_identity_and_receive_order():
    selected_experts, combine_weights = _small_routing_inputs()
    plan = build_source_push_plan(
        selected_experts,
        combine_weights,
        ep_size=EP_SIZE,
        experts_per_rank=EXPERTS_PER_RANK,
        block_m=BLOCK_M,
        capacity_factor=2.0,
    )

    assignment_ids = np.asarray(plan.assignment_ids)
    valid_mask = np.asarray(plan.valid_mask)
    token_ids = np.asarray(plan.token_ids)
    route_slots = np.asarray(plan.route_slots)
    packed_weights = np.asarray(plan.combine_weights)
    flat_weights = np.asarray(combine_weights).reshape(EP_SIZE, -1)
    valid_assignment_ids = assignment_ids[valid_mask]

    np.testing.assert_array_equal(token_ids[valid_mask], valid_assignment_ids // combine_weights.shape[-1])
    np.testing.assert_array_equal(route_slots[valid_mask], valid_assignment_ids % combine_weights.shape[-1])
    for src in range(EP_SIZE):
        source_valid = valid_mask[src]
        np.testing.assert_array_equal(
            packed_weights[src][source_valid], flat_weights[src, assignment_ids[src][source_valid]]
        )

    send_meta = np.asarray(plan.send_meta)
    recv_meta = np.asarray(plan.recv_meta)
    send_dst_ord = dst_ordinal(0, 1, EP_SIZE)
    recv_ord = recv_src_ordinal(1, 0, EP_SIZE)
    np.testing.assert_array_equal(
        send_meta[0, send_dst_ord, 1],
        [0, 0, 2, 1],
    )
    np.testing.assert_array_equal(recv_meta[1, recv_ord, 1], send_meta[0, send_dst_ord, 1])
    assert send_meta[0, send_dst_ord, 1, SOURCE_PUSH_META_SRC_RANK] == 0
    assert send_meta[0, send_dst_ord, 1, SOURCE_PUSH_META_LOCAL_EXPERT] == 0
    assert send_meta[0, send_dst_ord, 1, SOURCE_PUSH_META_LOCAL_ROW_START] == 2
    assert send_meta[0, send_dst_ord, 1, SOURCE_PUSH_META_VALID_ROWS] == 1


def test_source_push_plan_derives_expert_major_offsets_from_accepted_counts():
    selected_experts, combine_weights = _small_routing_inputs()
    plan = build_source_push_plan(
        selected_experts,
        combine_weights,
        ep_size=EP_SIZE,
        experts_per_rank=EXPERTS_PER_RANK,
        block_m=BLOCK_M,
        capacity_factor=2.0,
    )

    expected_counts = np.array(
        [
            [[2, 1], [3, 2]],
            [[2, 2], [2, 2]],
        ],
        dtype=np.int32,
    )
    np.testing.assert_array_equal(np.asarray(plan.counts_by_src_dst_expert), expected_counts)
    np.testing.assert_array_equal(np.asarray(plan.rows_per_local_expert), [[4, 3], [5, 4]])
    np.testing.assert_array_equal(np.asarray(plan.expert_base), [[0, 4], [0, 5]])
    np.testing.assert_array_equal(np.asarray(plan.src_base_by_expert), [[[0, 0], [2, 1]], [[0, 0], [3, 2]]])

    send_meta = np.asarray(plan.send_meta)
    valid_mask = np.asarray(plan.valid_mask)
    expert_base = np.asarray(plan.expert_base)
    src_base_by_expert = np.asarray(plan.src_base_by_expert)
    occupied_offsets: dict[tuple[int, int], set[int]] = {}
    for src in range(EP_SIZE):
        for dst in range(EP_SIZE):
            dst_ord = dst_ordinal(src, dst, EP_SIZE)
            for entry in range(send_meta.shape[2]):
                valid_rows = send_meta[src, dst_ord, entry, SOURCE_PUSH_META_VALID_ROWS]
                if valid_rows == 0:
                    continue
                local_expert = send_meta[src, dst_ord, entry, SOURCE_PUSH_META_LOCAL_EXPERT]
                local_row_start = send_meta[src, dst_ord, entry, SOURCE_PUSH_META_LOCAL_ROW_START]
                key = (dst, local_expert)
                occupied_offsets.setdefault(key, set())
                for row in range(valid_rows):
                    assert valid_mask[src, dst_ord, entry, row]
                    offset = (
                        expert_base[dst, local_expert]
                        + src_base_by_expert[dst, src, local_expert]
                        + local_row_start
                        + row
                    )
                    occupied_offsets[key].add(int(offset))

    assert occupied_offsets[(0, 0)] == {0, 1, 2, 3}
    assert occupied_offsets[(0, 1)] == {4, 5, 6}
    assert occupied_offsets[(1, 0)] == {0, 1, 2, 3, 4}
    assert occupied_offsets[(1, 1)] == {5, 6, 7, 8}


def test_source_push_plan_packs_tokens_and_restores_source_route_buffer():
    selected_experts, combine_weights = _small_routing_inputs()
    plan = build_source_push_plan(
        selected_experts,
        combine_weights,
        ep_size=EP_SIZE,
        experts_per_rank=EXPERTS_PER_RANK,
        block_m=BLOCK_M,
        capacity_factor=2.0,
    )
    x = jnp.arange(EP_SIZE * selected_experts.shape[1] * 3, dtype=jnp.float32).reshape(EP_SIZE, -1, 3)

    packed = pack_source_push_tokens(x, plan)
    packed_host = np.asarray(packed)
    assignment_ids = np.asarray(plan.assignment_ids)
    token_ids = np.asarray(plan.token_ids)
    route_slots = np.asarray(plan.route_slots)
    valid_mask = np.asarray(plan.valid_mask)
    for src in range(EP_SIZE):
        for dst_ord in range(EP_SIZE):
            for entry in range(assignment_ids.shape[2]):
                for row in range(BLOCK_M):
                    if valid_mask[src, dst_ord, entry, row]:
                        np.testing.assert_array_equal(
                            packed_host[src, dst_ord, entry, row],
                            np.asarray(x)[src, token_ids[src, dst_ord, entry, row]],
                        )
                    else:
                        np.testing.assert_array_equal(packed_host[src, dst_ord, entry, row], np.zeros(3))

    return_y = np.zeros_like(packed_host)
    return_y[..., 0] = np.where(valid_mask, assignment_ids * 10 + 1, 0)
    return_y[..., 1] = np.where(valid_mask, assignment_ids * 10 + 2, 0)
    return_y[..., 2] = np.where(valid_mask, assignment_ids * 10 + 3, 0)
    route_buffer = np.asarray(source_push_route_buffer(jnp.asarray(return_y), plan))
    combined = np.asarray(source_push_combine(jnp.asarray(return_y), plan))

    assert route_buffer.shape == (EP_SIZE, selected_experts.shape[1], selected_experts.shape[2], 3)
    for src in range(EP_SIZE):
        for dst_ord in range(EP_SIZE):
            for entry in range(assignment_ids.shape[2]):
                for row in range(BLOCK_M):
                    if not valid_mask[src, dst_ord, entry, row]:
                        continue
                    token = token_ids[src, dst_ord, entry, row]
                    route_slot = route_slots[src, dst_ord, entry, row]
                    expected = (
                        return_y[src, dst_ord, entry, row] * np.asarray(plan.combine_weights)[src, dst_ord, entry, row]
                    )
                    np.testing.assert_array_equal(route_buffer[src, token, route_slot], expected)
    np.testing.assert_array_equal(combined, np.sum(route_buffer, axis=2))


def test_source_push_w2_return_preserves_queue_identity_from_exact_expert_major_hidden():
    selected_experts, combine_weights = _small_routing_inputs()
    plan = build_source_push_plan(
        selected_experts,
        combine_weights,
        ep_size=EP_SIZE,
        experts_per_rank=EXPERTS_PER_RANK,
        block_m=BLOCK_M,
        capacity_factor=2.0,
    )
    expert_base = np.asarray(plan.expert_base)
    src_base_by_expert = np.asarray(plan.src_base_by_expert)
    rows_per_local_expert = np.asarray(plan.rows_per_local_expert)
    assignment_ids = np.asarray(plan.assignment_ids)
    valid_mask = np.asarray(plan.valid_mask)
    local_experts = np.asarray(plan.local_experts)
    local_row_starts = np.asarray(plan.local_row_starts)

    hidden = np.zeros((EP_SIZE, _expert_major_row_count(expert_base, rows_per_local_expert), 2), dtype=np.float32)
    for src in range(EP_SIZE):
        for dst_ord in range(EP_SIZE):
            dst = (src + dst_ord) % EP_SIZE
            for entry in range(assignment_ids.shape[2]):
                expert = local_experts[src, dst_ord, entry]
                if expert == INVALID_ASSIGNMENT_ID:
                    continue
                row_start = (
                    expert_base[dst, expert]
                    + src_base_by_expert[dst, src, expert]
                    + local_row_starts[src, dst_ord, entry]
                )
                for row in range(BLOCK_M):
                    if not valid_mask[src, dst_ord, entry, row]:
                        continue
                    assignment = assignment_ids[src, dst_ord, entry, row]
                    hidden[dst, row_start + row] = [assignment + 1, 100 + 10 * src + dst_ord]

    w_down = np.arange(EP_SIZE * EXPERTS_PER_RANK * 2 * 3, dtype=np.float32).reshape(EP_SIZE, EXPERTS_PER_RANK, 2, 3)
    return_y = np.asarray(source_push_w2_return(jnp.asarray(hidden), jnp.asarray(w_down), plan))

    for src in range(EP_SIZE):
        for dst_ord in range(EP_SIZE):
            dst = (src + dst_ord) % EP_SIZE
            for entry in range(assignment_ids.shape[2]):
                expert = local_experts[src, dst_ord, entry]
                for row in range(BLOCK_M):
                    if not valid_mask[src, dst_ord, entry, row]:
                        np.testing.assert_array_equal(return_y[src, dst_ord, entry, row], np.zeros(3))
                        continue
                    row_start = (
                        expert_base[dst, expert]
                        + src_base_by_expert[dst, src, expert]
                        + local_row_starts[src, dst_ord, entry]
                    )
                    expected = hidden[dst, row_start + row] @ w_down[dst, expert]
                    np.testing.assert_allclose(return_y[src, dst_ord, entry, row], expected)


def test_source_push_w2_return_with_source_padded_bases_combines_like_direct_reference():
    selected_experts, combine_weights = _small_routing_inputs()
    plan = build_source_push_plan(
        selected_experts,
        combine_weights,
        ep_size=EP_SIZE,
        experts_per_rank=EXPERTS_PER_RANK,
        block_m=BLOCK_M,
        capacity_factor=2.0,
    )
    rounded_counts, expert_base, src_base_by_expert = source_push_source_padded_row_bases(plan, BLOCK_M)
    rows_per_local_expert = np.sum(rounded_counts, axis=0)
    assignment_ids = np.asarray(plan.assignment_ids)
    token_ids = np.asarray(plan.token_ids)
    valid_mask = np.asarray(plan.valid_mask)
    local_experts = np.asarray(plan.local_experts)
    local_row_starts = np.asarray(plan.local_row_starts)
    weights = np.asarray(plan.combine_weights)

    hidden = np.zeros((EP_SIZE, _expert_major_row_count(expert_base, rows_per_local_expert), 2), dtype=np.float32)
    for src in range(EP_SIZE):
        for dst_ord in range(EP_SIZE):
            dst = (src + dst_ord) % EP_SIZE
            for entry in range(assignment_ids.shape[2]):
                expert = local_experts[src, dst_ord, entry]
                if expert == INVALID_ASSIGNMENT_ID:
                    continue
                row_start = (
                    expert_base[dst, expert]
                    + src_base_by_expert[dst, src, expert]
                    + local_row_starts[src, dst_ord, entry]
                )
                for row in range(BLOCK_M):
                    if not valid_mask[src, dst_ord, entry, row]:
                        continue
                    assignment = assignment_ids[src, dst_ord, entry, row]
                    hidden[dst, row_start + row] = [assignment + 1, 1 + src + 2 * dst_ord]

    w_down = (
        np.arange(EP_SIZE * EXPERTS_PER_RANK * 2 * 4, dtype=np.float32).reshape(EP_SIZE, EXPERTS_PER_RANK, 2, 4) + 1.0
    ) / 10.0
    return_y = source_push_w2_return(
        jnp.asarray(hidden),
        jnp.asarray(w_down),
        plan,
        expert_base=expert_base,
        src_base_by_expert=src_base_by_expert,
    )
    combined = np.asarray(source_push_combine(return_y, plan))

    expected = np.zeros((EP_SIZE, selected_experts.shape[1], 4), dtype=np.float32)
    for src in range(EP_SIZE):
        for dst_ord in range(EP_SIZE):
            dst = (src + dst_ord) % EP_SIZE
            for entry in range(assignment_ids.shape[2]):
                expert = local_experts[src, dst_ord, entry]
                if expert == INVALID_ASSIGNMENT_ID:
                    continue
                row_start = (
                    expert_base[dst, expert]
                    + src_base_by_expert[dst, src, expert]
                    + local_row_starts[src, dst_ord, entry]
                )
                for row in range(BLOCK_M):
                    if not valid_mask[src, dst_ord, entry, row]:
                        continue
                    token = token_ids[src, dst_ord, entry, row]
                    expected[src, token] += (hidden[dst, row_start + row] @ w_down[dst, expert]) * weights[
                        src, dst_ord, entry, row
                    ]

    np.testing.assert_allclose(combined, expected, rtol=1e-6, atol=1e-6)


def test_source_push_plan_capacity_clipping_matches_existing_ep_reference():
    selected_experts, combine_weights = _small_routing_inputs()
    plan = build_source_push_plan(
        selected_experts,
        combine_weights,
        ep_size=EP_SIZE,
        experts_per_rank=EXPERTS_PER_RANK,
        block_m=BLOCK_M,
        capacity_factor=1.0,
    )

    assignments_per_source = selected_experts.shape[1] * selected_experts.shape[2]
    receiver_capacity = max(EXPERTS_PER_RANK, int(math.ceil(assignments_per_source)))
    group_sizes = []
    sorted_assignment_ids = []
    for src in range(EP_SIZE):
        _, _, source_group_sizes, source_sorted_assignment_ids = _prepare_moe_dispatch_indices_with_assignment_ids(
            selected_experts[src],
            num_experts=EP_SIZE * EXPERTS_PER_RANK,
        )
        group_sizes.append(source_group_sizes)
        sorted_assignment_ids.append(source_sorted_assignment_ids)
    group_sizes_array = jnp.stack(group_sizes, axis=0)
    clipped_group_sizes = _clip_receiver_group_sizes(
        group_sizes_array,
        local_expert_size=EXPERTS_PER_RANK,
        receiver_capacity=receiver_capacity,
    )

    plan_assignment_ids = np.asarray(plan.assignment_ids)
    valid_mask = np.asarray(plan.valid_mask)
    for src in range(EP_SIZE):
        keep_mask = _expert_prefix_keep_mask(
            group_sizes_array[src],
            clipped_group_sizes[src],
            total_size=assignments_per_source,
        )
        expected_accepted = set(np.asarray(sorted_assignment_ids[src][keep_mask], dtype=np.int32).tolist())
        actual_accepted = set(plan_assignment_ids[src][valid_mask[src]].tolist())
        assert actual_accepted == expected_accepted

    assert int(plan.dropped_routes) == 1
    stats = source_push_plan_row_stats(plan)
    assert stats.useful_rows == selected_experts.size - 1
    assert stats.dropped_routes == 1
    assert stats.rounded_rows == stats.live_entries * BLOCK_M
    assert stats.masked_row_fraction == pytest.approx(1.0 - stats.useful_rows / stats.rounded_rows)


def test_source_push_plan_rejects_queue_capacity_overflow():
    selected_experts, combine_weights = _small_routing_inputs()

    with pytest.raises(ValueError, match="source-push queue capacity overflow"):
        build_source_push_plan(
            selected_experts,
            combine_weights,
            ep_size=EP_SIZE,
            experts_per_rank=EXPERTS_PER_RANK,
            block_m=BLOCK_M,
            capacity_factor=2.0,
            entries_per_dst=1,
        )
