# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import math
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AbstractMesh, AxisType, NamedSharding, PartitionSpec as P, use_abstract_mesh

from levanter.grug._moe.common import _prepare_moe_dispatch_indices_with_assignment_ids
from levanter.grug._moe.ep_common import _clip_receiver_group_sizes, _expert_prefix_keep_mask
from levanter.grug._moe.source_push_plan import (
    INVALID_ASSIGNMENT_ID,
    SOURCE_PUSH_META_LOCAL_EXPERT,
    SOURCE_PUSH_META_LOCAL_ROW_START,
    SOURCE_PUSH_META_SRC_RANK,
    SOURCE_PUSH_META_VALID_ROWS,
    build_source_push_plan,
    build_source_push_semantic_plan_jax,
    dst_ordinal,
    pack_source_push_tokens,
    pack_source_push_tokens_jax,
    source_push_combine,
    source_push_combine_preweighted,
    source_push_expert_offsets_from_counts,
    source_push_h_row_route_weights_jax,
    source_push_queue_entry_metadata_from_counts,
    source_push_queue_route_weights_jax,
    source_push_route_rows_host_from_plan,
    source_push_w13_h,
    source_push_w2_from_h_return,
    recv_src_ordinal,
    source_push_plan_row_stats,
    source_push_route_buffer,
    source_push_source_padded_row_bases,
    source_push_semantic_backward_source_expand_jax,
    source_push_semantic_dx_combine_jax,
    source_push_semantic_expert_major_to_pair_jax,
    source_push_semantic_forward_reference_jax,
    source_push_semantic_pair_to_expert_major_jax,
    source_push_semantic_reverse_route_jax,
    source_push_semantic_route_weights_expert_major_jax,
    source_push_semantic_queue_metadata_jax,
    source_push_semantic_source_aligned_expert_offsets_jax,
    source_push_semantic_swiglu_backward_reference_jax,
    source_push_semantic_w13_backward_reference_jax,
    source_push_semantic_w2_backward_reference_jax,
    source_push_w2_return,
)
from levanter.grug._moe.source_push_token_pack import (
    SourcePushTokenPackPallasBlockSizes,
    source_push_pack_tokens_pallas_mgpu,
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


def _rough_balanced_routing_inputs() -> tuple[jax.Array, jax.Array]:
    ep_size = 4
    experts_per_rank = 2
    tokens_per_source = 6
    topk = 2
    global_experts = ep_size * experts_per_rank
    base_routes = np.arange(ep_size * tokens_per_source * topk, dtype=np.int32).reshape(
        ep_size, tokens_per_source, topk
    )
    selected_experts = (base_routes + np.arange(ep_size, dtype=np.int32)[:, None, None]) % global_experts
    combine_weights = np.arange(selected_experts.size, dtype=np.float32).reshape(selected_experts.shape) + 0.5
    return jnp.asarray(selected_experts), jnp.asarray(combine_weights)


def _skewed_destination_expert_inputs() -> tuple[jax.Array, jax.Array]:
    selected_experts = jnp.array(
        [
            [[2], [2], [2], [3], [4], [5]],
            [[2], [2], [2], [2], [3], [0]],
            [[2], [2], [2], [2], [2], [3]],
        ],
        dtype=jnp.int32,
    )
    combine_weights = jnp.arange(selected_experts.size, dtype=jnp.float32).reshape(selected_experts.shape) + 1.0
    return selected_experts, combine_weights


def test_source_push_semantic_source_aligned_offsets_are_jittable_and_preserve_source_order():
    selected_experts, combine_weights = _small_routing_inputs()
    plan = build_source_push_semantic_plan_jax(
        selected_experts,
        combine_weights,
        ep_size=EP_SIZE,
        experts_per_rank=EXPERTS_PER_RANK,
        rows_per_src_dst_capacity=8,
        capacity_factor=1.25,
    )

    rows_per_expert, source_bases = jax.jit(
        lambda semantic_plan: source_push_semantic_source_aligned_expert_offsets_jax(
            semantic_plan,
            row_alignment=8,
        )
    )(plan)
    aligned_counts = ((plan.xcounts + 7) // 8) * 8
    expected_bases = jnp.transpose(jnp.cumsum(aligned_counts, axis=0) - aligned_counts, (1, 0, 2))

    np.testing.assert_array_equal(np.asarray(source_bases), np.asarray(expected_bases))
    np.testing.assert_array_equal(np.asarray(rows_per_expert), np.asarray(jnp.sum(aligned_counts, axis=0)))
    assert np.all(np.asarray(source_bases) % 8 == 0)


def _plan_cases():
    return (
        pytest.param(_small_routing_inputs(), EP_SIZE, EXPERTS_PER_RANK, BLOCK_M, 2.0, id="uneven"),
        pytest.param(_rough_balanced_routing_inputs(), 4, 2, 2, 1.25, id="rough-balanced"),
    )


def _expert_major_row_count(expert_base: np.ndarray, rows_per_local_expert: np.ndarray) -> int:
    return int(np.max(expert_base + rows_per_local_expert))


def _source_padded_h_row_metadata(plan):
    rounded_counts, expert_base, src_base_by_expert = source_push_source_padded_row_bases(plan, BLOCK_M)
    send_meta = np.asarray(plan.send_meta).copy()
    local_experts = np.asarray(plan.local_experts)
    local_row_starts = np.asarray(plan.local_row_starts)
    valid_entries = send_meta[..., SOURCE_PUSH_META_VALID_ROWS] > 0
    for src, dst_ord, entry in np.argwhere(valid_entries):
        dst = (src + dst_ord) % EP_SIZE
        expert = local_experts[src, dst_ord, entry]
        send_meta[src, dst_ord, entry, SOURCE_PUSH_META_LOCAL_ROW_START] = (
            expert_base[dst, expert] + src_base_by_expert[dst, src, expert] + local_row_starts[src, dst_ord, entry]
        )
    rows_per_local_expert = np.sum(rounded_counts, axis=0, dtype=np.int32)
    hidden_rows = _expert_major_row_count(expert_base, rows_per_local_expert)
    return send_meta, expert_base, src_base_by_expert, hidden_rows


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


@pytest.mark.parametrize(("inputs", "ep_size", "experts_per_rank", "block_m", "capacity_factor"), _plan_cases())
def test_source_push_semantic_plan_jax_matches_host_plan_rows(
    inputs,
    ep_size,
    experts_per_rank,
    block_m,
    capacity_factor,
):
    selected_experts, combine_weights = inputs
    host_plan = build_source_push_plan(
        selected_experts,
        combine_weights,
        ep_size=ep_size,
        experts_per_rank=experts_per_rank,
        block_m=block_m,
        capacity_factor=capacity_factor,
    )
    rows_per_pair = int(np.max(np.sum(np.asarray(host_plan.counts_by_src_dst_expert), axis=2)))

    semantic_plan = build_source_push_semantic_plan_jax(
        selected_experts,
        combine_weights,
        ep_size=ep_size,
        experts_per_rank=experts_per_rank,
        rows_per_src_dst_capacity=rows_per_pair,
        capacity_factor=capacity_factor,
    )

    np.testing.assert_array_equal(semantic_plan.xcounts, host_plan.counts_by_src_dst_expert)
    np.testing.assert_array_equal(
        semantic_plan.pair_expert_base,
        np.cumsum(np.asarray(host_plan.counts_by_src_dst_expert), axis=2, dtype=np.int32)
        - np.asarray(host_plan.counts_by_src_dst_expert),
    )
    np.testing.assert_array_equal(semantic_plan.rows_per_local_expert, host_plan.rows_per_local_expert)
    np.testing.assert_array_equal(semantic_plan.expert_base, host_plan.expert_base)
    np.testing.assert_array_equal(semantic_plan.src_base_by_expert, host_plan.src_base_by_expert)
    assert int(semantic_plan.metadata_overflow_routes) == 0
    assert int(semantic_plan.dropped_routes) == int(host_plan.dropped_routes)

    host_assignment_ids = np.asarray(host_plan.assignment_ids)
    host_valid = np.asarray(host_plan.valid_mask)
    host_experts = np.asarray(host_plan.local_experts)
    semantic_assignment_ids = np.asarray(semantic_plan.assignment_ids)
    semantic_token_ids = np.asarray(semantic_plan.token_ids)
    semantic_route_slots = np.asarray(semantic_plan.route_slots)
    semantic_weights = np.asarray(semantic_plan.route_weights)
    semantic_pair_base = np.asarray(semantic_plan.pair_expert_base)
    semantic_counts = np.asarray(semantic_plan.xcounts)
    flat_weights = np.asarray(combine_weights).reshape(ep_size, -1)

    for src in range(ep_size):
        for dst in range(ep_size):
            dst_ord = dst_ordinal(src, dst, ep_size)
            for expert in range(experts_per_rank):
                expected = []
                for entry in range(host_assignment_ids.shape[2]):
                    if host_experts[src, dst_ord, entry] != expert:
                        continue
                    expected.extend(host_assignment_ids[src, dst_ord, entry][host_valid[src, dst_ord, entry]])
                row_start = semantic_pair_base[src, dst, expert]
                row_end = row_start + semantic_counts[src, dst, expert]
                observed = semantic_assignment_ids[src, dst, row_start:row_end]
                np.testing.assert_array_equal(observed, np.asarray(expected, dtype=np.int32))
                np.testing.assert_array_equal(
                    semantic_token_ids[src, dst, row_start:row_end],
                    observed // combine_weights.shape[-1],
                )
                np.testing.assert_array_equal(
                    semantic_route_slots[src, dst, row_start:row_end],
                    observed % combine_weights.shape[-1],
                )
                np.testing.assert_array_equal(
                    semantic_weights[src, dst, row_start:row_end],
                    flat_weights[src, observed],
                )


def test_source_push_semantic_plan_jax_is_jittable():
    selected_experts, combine_weights = _small_routing_inputs()

    @jax.jit
    def build(selected_experts_arg, combine_weights_arg):
        return build_source_push_semantic_plan_jax(
            selected_experts_arg,
            combine_weights_arg,
            ep_size=EP_SIZE,
            experts_per_rank=EXPERTS_PER_RANK,
            rows_per_src_dst_capacity=8,
            capacity_factor=2.0,
        )

    semantic_plan = build(selected_experts, combine_weights)

    assert semantic_plan.assignment_ids.shape == (EP_SIZE, EP_SIZE, 8)
    assert int(semantic_plan.metadata_overflow_routes) == 0


def test_source_push_semantic_plan_jax_destination_expert_capacity_clips_sources_in_prefix_order():
    selected_experts, combine_weights = _skewed_destination_expert_inputs()

    pair_clipped_plan = build_source_push_semantic_plan_jax(
        selected_experts,
        combine_weights,
        ep_size=3,
        experts_per_rank=2,
        rows_per_src_dst_capacity=5,
        capacity_factor=3.0,
    )
    plan = build_source_push_semantic_plan_jax(
        selected_experts,
        combine_weights,
        ep_size=3,
        experts_per_rank=2,
        rows_per_src_dst_capacity=5,
        rows_per_expert_capacity=5,
        capacity_factor=3.0,
    )

    expected_counts = np.array(
        [
            [[0, 0], [3, 1], [1, 1]],
            [[1, 0], [2, 1], [0, 0]],
            [[0, 0], [0, 0], [0, 0]],
        ],
        dtype=np.int32,
    )
    np.testing.assert_array_equal(np.asarray(pair_clipped_plan.xcounts[:, 1]), [[3, 1], [4, 1], [5, 0]])
    assert int(pair_clipped_plan.metadata_overflow_routes) == 1
    assert int(pair_clipped_plan.dropped_routes) == 1
    np.testing.assert_array_equal(np.asarray(plan.xcounts), expected_counts)
    np.testing.assert_array_equal(
        np.asarray(plan.pair_expert_base),
        np.cumsum(expected_counts, axis=2, dtype=np.int32) - expected_counts,
    )
    np.testing.assert_array_equal(np.asarray(plan.rows_per_local_expert), [[1, 0], [5, 2], [1, 1]])
    np.testing.assert_array_equal(np.asarray(plan.expert_base), [[0, 1], [0, 5], [0, 1]])
    np.testing.assert_array_equal(
        np.asarray(plan.src_base_by_expert),
        [
            [[0, 0], [0, 0], [1, 0]],
            [[0, 0], [3, 1], [5, 2]],
            [[0, 0], [1, 1], [1, 1]],
        ],
    )

    np.testing.assert_array_equal(np.asarray(plan.assignment_ids[0, 1]), [0, 1, 2, 3, -1])
    np.testing.assert_array_equal(np.asarray(plan.assignment_ids[1, 1]), [0, 1, 4, -1, -1])
    np.testing.assert_array_equal(np.asarray(plan.valid_mask[2]), np.zeros((3, 5), dtype=np.bool_))
    assert int(plan.routing_dropped_routes) == 0
    assert int(plan.metadata_overflow_routes) == 8
    assert int(plan.metadata_overflow_routes - pair_clipped_plan.metadata_overflow_routes) == 7
    assert int(plan.dropped_routes) == 8
    assert int(jnp.sum(plan.valid_mask)) == selected_experts.size - int(plan.dropped_routes)

    reverse = plan.reverse_route
    expected_reverse_valid = np.array(
        [
            [[True], [True], [True], [True], [True], [True]],
            [[True], [True], [False], [False], [True], [True]],
            [[False], [False], [False], [False], [False], [False]],
        ],
        dtype=np.bool_,
    )
    np.testing.assert_array_equal(np.asarray(reverse.route_valid), expected_reverse_valid)
    np.testing.assert_array_equal(np.asarray(reverse.route_expert_row[0, :4, 0]), [0, 1, 2, 0])
    np.testing.assert_array_equal(np.asarray(reverse.route_expert_row[1, [0, 1, 4], 0]), [3, 4, 1])
    np.testing.assert_array_equal(
        np.asarray(reverse.assignment_id[~reverse.route_valid]),
        np.full(8, INVALID_ASSIGNMENT_ID, dtype=np.int32),
    )


def test_source_push_semantic_plan_jax_destination_expert_capacity_is_static_under_jit():
    selected_experts, combine_weights = _skewed_destination_expert_inputs()
    build = jax.jit(
        build_source_push_semantic_plan_jax,
        static_argnames=(
            "ep_size",
            "experts_per_rank",
            "rows_per_src_dst_capacity",
            "rows_per_expert_capacity",
            "capacity_factor",
        ),
    )

    plan = build(
        selected_experts,
        combine_weights,
        ep_size=3,
        experts_per_rank=2,
        rows_per_src_dst_capacity=5,
        rows_per_expert_capacity=5,
        capacity_factor=3.0,
    )

    np.testing.assert_array_equal(np.asarray(plan.rows_per_local_expert), [[1, 0], [5, 2], [1, 1]])
    assert int(plan.metadata_overflow_routes) == 8


@pytest.mark.parametrize("rows_per_expert_capacity", [0, -1])
def test_source_push_semantic_plan_jax_rejects_non_positive_destination_expert_capacity(
    rows_per_expert_capacity,
):
    selected_experts, combine_weights = _small_routing_inputs()

    with pytest.raises(ValueError, match="rows_per_expert_capacity must be positive"):
        build_source_push_semantic_plan_jax(
            selected_experts,
            combine_weights,
            ep_size=EP_SIZE,
            experts_per_rank=EXPERTS_PER_RANK,
            rows_per_src_dst_capacity=8,
            rows_per_expert_capacity=rows_per_expert_capacity,
        )


def test_source_push_semantic_queue_metadata_jax_expands_entries_and_accounts_for_static_capacity():
    selected_experts = jnp.zeros((1, 4, 1), dtype=jnp.int32)
    route_weights = jnp.ones(selected_experts.shape, dtype=jnp.float32)

    @jax.jit
    def build_queue(selected_experts_arg, route_weights_arg):
        plan = build_source_push_semantic_plan_jax(
            selected_experts_arg,
            route_weights_arg,
            ep_size=1,
            experts_per_rank=1,
            rows_per_src_dst_capacity=4,
            capacity_factor=2.0,
        )
        return plan, source_push_semantic_queue_metadata_jax(
            plan,
            return_row_block=2,
            entries_per_dst=1,
        )

    plan, queue = build_queue(selected_experts, route_weights)

    np.testing.assert_array_equal(np.asarray(plan.assignment_ids), [[[0, 1, 2, 3]]])
    np.testing.assert_array_equal(np.asarray(plan.valid_mask), np.ones((1, 1, 4), dtype=np.bool_))
    np.testing.assert_array_equal(np.asarray(plan.xcounts), [[[4]]])
    np.testing.assert_array_equal(np.asarray(plan.pair_expert_base), [[[0]]])
    np.testing.assert_array_equal(np.asarray(plan.expert_base), [[0]])
    np.testing.assert_array_equal(np.asarray(plan.src_base_by_expert), [[[0]]])
    np.testing.assert_array_equal(np.asarray(queue.local_expert), [[[0]]])
    np.testing.assert_array_equal(np.asarray(queue.local_row_start), [[[0]]])
    np.testing.assert_array_equal(np.asarray(queue.valid_rows), [[[2]]])
    np.testing.assert_array_equal(np.asarray(queue.required_entries_per_dst), [[2]])
    np.testing.assert_array_equal(np.asarray(queue.route_dst_ordinal), [[[0], [0], [0], [0]]])
    np.testing.assert_array_equal(np.asarray(queue.route_entry), [[[0], [0], [0], [0]]])
    np.testing.assert_array_equal(np.asarray(queue.route_queue_row), [[[0], [1], [0], [0]]])
    np.testing.assert_array_equal(np.asarray(queue.route_valid), [[[True], [True], [False], [False]]])
    assert int(queue.overflow_entries) == 1
    assert int(queue.overflow_routes) == 2
    assert queue.return_row_block == 2
    assert queue.entries_per_dst == 1


def test_source_push_semantic_queue_metadata_jax_inverse_indexes_every_stored_route():
    selected_experts, route_weights = _small_routing_inputs()

    @jax.jit
    def build_queue(selected_experts_arg, route_weights_arg):
        plan = build_source_push_semantic_plan_jax(
            selected_experts_arg,
            route_weights_arg,
            ep_size=EP_SIZE,
            experts_per_rank=EXPERTS_PER_RANK,
            rows_per_src_dst_capacity=8,
            capacity_factor=2.0,
        )
        return plan, source_push_semantic_queue_metadata_jax(
            plan,
            return_row_block=2,
            entries_per_dst=4,
        )

    plan, queue = build_queue(selected_experts, route_weights)
    selected_host = np.asarray(selected_experts)
    queue_expert = np.asarray(queue.local_expert)
    queue_row_start = np.asarray(queue.local_row_start)
    queue_valid_rows = np.asarray(queue.valid_rows)
    route_dst_ordinal = np.asarray(queue.route_dst_ordinal)
    route_entry = np.asarray(queue.route_entry)
    route_queue_row = np.asarray(queue.route_queue_row)
    route_valid = np.asarray(queue.route_valid)

    np.testing.assert_array_equal(route_valid, np.ones(selected_experts.shape, dtype=np.bool_))
    assert int(queue.overflow_entries) == 0
    assert int(queue.overflow_routes) == 0
    for src, token, route_slot in np.ndindex(selected_experts.shape):
        global_expert = selected_host[src, token, route_slot]
        expected_dst = global_expert // EXPERTS_PER_RANK
        expected_local_expert = global_expert % EXPERTS_PER_RANK
        dst_ord = route_dst_ordinal[src, token, route_slot]
        entry = route_entry[src, token, route_slot]
        queue_row = route_queue_row[src, token, route_slot]

        assert (src + dst_ord) % EP_SIZE == expected_dst
        assert queue_expert[src, dst_ord, entry] == expected_local_expert
        assert queue_row < queue_valid_rows[src, dst_ord, entry]
        local_expert_row = queue_row_start[src, dst_ord, entry] + queue_row
        earlier_assignments = selected_host[src].reshape(-1)[: token * plan.topk + route_slot]
        assert local_expert_row == np.count_nonzero(earlier_assignments == global_expert)


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


@pytest.mark.parametrize("inputs, ep_size, experts_per_rank, block_m, capacity_factor", _plan_cases())
def test_source_push_plan_cached_entry_metadata_is_derived_from_counts(
    inputs,
    ep_size,
    experts_per_rank,
    block_m,
    capacity_factor,
):
    selected_experts, combine_weights = inputs
    plan = build_source_push_plan(
        selected_experts,
        combine_weights,
        ep_size=ep_size,
        experts_per_rank=experts_per_rank,
        block_m=block_m,
        capacity_factor=capacity_factor,
    )

    rows_per_expert, expert_base, src_base_by_expert = source_push_expert_offsets_from_counts(
        plan.counts_by_src_dst_expert
    )
    np.testing.assert_array_equal(rows_per_expert, np.asarray(plan.rows_per_local_expert))
    np.testing.assert_array_equal(expert_base, np.asarray(plan.expert_base))
    np.testing.assert_array_equal(src_base_by_expert, np.asarray(plan.src_base_by_expert))

    entry_metadata = source_push_queue_entry_metadata_from_counts(
        plan.counts_by_src_dst_expert,
        block_m,
        entries_per_dst=plan.assignment_ids.shape[2],
    )
    np.testing.assert_array_equal(entry_metadata.local_experts, np.asarray(plan.local_experts))
    np.testing.assert_array_equal(entry_metadata.local_row_starts, np.asarray(plan.local_row_starts))
    np.testing.assert_array_equal(entry_metadata.send_meta, np.asarray(plan.send_meta))
    np.testing.assert_array_equal(entry_metadata.recv_meta, np.asarray(plan.recv_meta))


@pytest.mark.parametrize("inputs, ep_size, experts_per_rank, block_m, capacity_factor", _plan_cases())
def test_source_push_plan_route_rows_derive_destination_expert_and_rows_from_global_assignments(
    inputs,
    ep_size,
    experts_per_rank,
    block_m,
    capacity_factor,
):
    selected_experts, combine_weights = inputs
    plan = build_source_push_plan(
        selected_experts,
        combine_weights,
        ep_size=ep_size,
        experts_per_rank=experts_per_rank,
        block_m=block_m,
        capacity_factor=capacity_factor,
    )

    route_rows = source_push_route_rows_host_from_plan(plan)
    selected_host = np.asarray(selected_experts)
    token_ids = np.asarray(plan.token_ids)
    route_slots = np.asarray(plan.route_slots)
    valid_mask = np.asarray(plan.valid_mask)
    local_experts = np.asarray(plan.local_experts)
    local_row_starts = np.asarray(plan.local_row_starts)
    src_base_by_expert = np.asarray(plan.src_base_by_expert)

    np.testing.assert_array_equal(route_rows.token_id[valid_mask], token_ids[valid_mask])
    np.testing.assert_array_equal(route_rows.route_slot[valid_mask], route_slots[valid_mask])
    for src, dst_ord, entry, row in np.argwhere(valid_mask):
        token = token_ids[src, dst_ord, entry, row]
        route_slot = route_slots[src, dst_ord, entry, row]
        global_expert = selected_host[src, token, route_slot]
        expected_dst = global_expert // experts_per_rank
        expected_expert = global_expert % experts_per_rank
        expected_expert_row = (
            src_base_by_expert[expected_dst, src, expected_expert] + local_row_starts[src, dst_ord, entry] + row
        )

        assert route_rows.dst[src, dst_ord, entry, row] == expected_dst
        assert route_rows.local_expert[src, dst_ord, entry, row] == expected_expert
        assert local_experts[src, dst_ord, entry] == expected_expert
        assert route_rows.expert_row[src, dst_ord, entry, row] == expected_expert_row


def test_source_push_plan_route_rows_ignore_cached_kernel_entry_metadata():
    selected_experts, combine_weights = _rough_balanced_routing_inputs()
    plan = build_source_push_plan(
        selected_experts,
        combine_weights,
        ep_size=4,
        experts_per_rank=2,
        block_m=2,
        capacity_factor=1.25,
    )

    expected = source_push_route_rows_host_from_plan(plan)
    corrupted_plan = replace(
        plan,
        local_experts=jnp.full_like(plan.local_experts, 1),
        local_row_starts=jnp.full_like(plan.local_row_starts, 123),
    )
    observed = source_push_route_rows_host_from_plan(corrupted_plan)

    np.testing.assert_array_equal(observed.src, expected.src)
    np.testing.assert_array_equal(observed.dst, expected.dst)
    np.testing.assert_array_equal(observed.local_expert, expected.local_expert)
    np.testing.assert_array_equal(observed.expert_row, expected.expert_row)
    np.testing.assert_array_equal(observed.token_id, expected.token_id)
    np.testing.assert_array_equal(observed.route_slot, expected.route_slot)
    np.testing.assert_array_equal(observed.assignment_id, expected.assignment_id)
    np.testing.assert_array_equal(observed.valid, expected.valid)


def test_source_push_plan_derived_route_rows_round_trip_topk_return_slots():
    selected_experts, combine_weights = _rough_balanced_routing_inputs()
    plan = build_source_push_plan(
        selected_experts,
        combine_weights,
        ep_size=4,
        experts_per_rank=2,
        block_m=2,
        capacity_factor=1.25,
    )
    route_rows = source_push_route_rows_host_from_plan(plan)

    return_y = np.zeros((*route_rows.valid.shape, 1), dtype=np.float32)
    return_y[..., 0] = np.where(
        route_rows.valid, 100 * route_rows.src + 10 * route_rows.token_id + route_rows.route_slot, 0
    )
    route_buffer = np.asarray(source_push_route_buffer(jnp.asarray(return_y), plan))
    combined = np.asarray(source_push_combine(jnp.asarray(return_y), plan))

    weights = np.asarray(plan.combine_weights)
    for src, dst_ord, entry, row in np.argwhere(route_rows.valid):
        token = route_rows.token_id[src, dst_ord, entry, row]
        route_slot = route_rows.route_slot[src, dst_ord, entry, row]
        expected = return_y[src, dst_ord, entry, row] * weights[src, dst_ord, entry, row]
        np.testing.assert_array_equal(route_buffer[src, token, route_slot], expected)
    np.testing.assert_array_equal(combined, np.sum(route_buffer, axis=2))


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


def test_source_push_plan_jax_pack_and_route_weight_gathers_match_host_plan():
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

    observed_packed = jax.jit(lambda x_arg: pack_source_push_tokens_jax(x_arg, plan))(x)
    observed_queue_weights = jax.jit(lambda weights_arg: source_push_queue_route_weights_jax(weights_arg, plan))(
        combine_weights
    )

    np.testing.assert_array_equal(np.asarray(observed_packed), np.asarray(pack_source_push_tokens(x, plan)))
    np.testing.assert_array_equal(np.asarray(observed_queue_weights), np.asarray(plan.combine_weights))


def test_source_push_pallas_token_pack_matches_jax_pack_in_interpret_mode():
    selected_experts, combine_weights = _small_routing_inputs()
    plan = build_source_push_plan(
        selected_experts,
        combine_weights,
        ep_size=EP_SIZE,
        experts_per_rank=EXPERTS_PER_RANK,
        block_m=BLOCK_M,
        capacity_factor=2.0,
    )
    x = jnp.arange(EP_SIZE * selected_experts.shape[1] * 8, dtype=jnp.float32).reshape(EP_SIZE, -1, 8)

    observed = source_push_pack_tokens_pallas_mgpu(
        x,
        plan,
        block_sizes=SourcePushTokenPackPallasBlockSizes(hidden_block=4),
        interpret=True,
    )
    expected = pack_source_push_tokens_jax(x, plan).astype(jnp.bfloat16)

    np.testing.assert_array_equal(np.asarray(observed), np.asarray(expected))


def test_source_push_h_row_route_weights_match_exact_expert_major_rows():
    selected_experts, combine_weights = _small_routing_inputs()
    plan = build_source_push_plan(
        selected_experts,
        combine_weights,
        ep_size=EP_SIZE,
        experts_per_rank=EXPERTS_PER_RANK,
        block_m=BLOCK_M,
        capacity_factor=2.0,
    )
    hidden_rows = _expert_major_row_count(np.asarray(plan.expert_base), np.asarray(plan.rows_per_local_expert))

    h_route_weights = np.asarray(
        source_push_h_row_route_weights_jax(
            combine_weights,
            plan,
            plan.send_meta,
            plan.expert_base,
            plan.src_base_by_expert,
            hidden_rows_per_rank=hidden_rows,
            use_exact_expert_major=True,
        )
    )

    assignment_ids = np.asarray(plan.assignment_ids)
    token_ids = np.asarray(plan.token_ids)
    route_slots = np.asarray(plan.route_slots)
    valid_mask = np.asarray(plan.valid_mask)
    local_experts = np.asarray(plan.local_experts)
    local_row_starts = np.asarray(plan.local_row_starts)
    expert_base = np.asarray(plan.expert_base)
    src_base_by_expert = np.asarray(plan.src_base_by_expert)
    route_weights = np.asarray(combine_weights)
    for src, dst_ord, entry, row in np.argwhere(valid_mask):
        dst = (src + dst_ord) % EP_SIZE
        expert = local_experts[src, dst_ord, entry]
        expert_row = src_base_by_expert[dst, src, expert] + local_row_starts[src, dst_ord, entry] + row
        flat_row = expert_base[dst, expert] + expert_row
        assignment = assignment_ids[src, dst_ord, entry, row]
        token = token_ids[src, dst_ord, entry, row]
        route_slot = route_slots[src, dst_ord, entry, row]
        assert token == assignment // combine_weights.shape[-1]
        assert route_slot == assignment % combine_weights.shape[-1]
        np.testing.assert_array_equal(h_route_weights[dst, flat_row], route_weights[src, token, route_slot])


def test_source_push_h_row_route_weights_match_source_padded_metadata_rows():
    selected_experts, combine_weights = _small_routing_inputs()
    plan = build_source_push_plan(
        selected_experts,
        combine_weights,
        ep_size=EP_SIZE,
        experts_per_rank=EXPERTS_PER_RANK,
        block_m=BLOCK_M,
        capacity_factor=2.0,
    )
    send_meta, expert_base, src_base_by_expert, hidden_rows = _source_padded_h_row_metadata(plan)

    h_route_weights = np.asarray(
        source_push_h_row_route_weights_jax(
            combine_weights,
            plan,
            send_meta,
            expert_base,
            src_base_by_expert,
            hidden_rows_per_rank=hidden_rows,
            use_exact_expert_major=False,
        )
    )

    token_ids = np.asarray(plan.token_ids)
    route_slots = np.asarray(plan.route_slots)
    route_weights = np.asarray(combine_weights)
    for src, dst_ord, entry, row in np.argwhere(np.asarray(plan.valid_mask)):
        dst = (src + dst_ord) % EP_SIZE
        flat_row = send_meta[src, dst_ord, entry, SOURCE_PUSH_META_LOCAL_ROW_START] + row
        token = token_ids[src, dst_ord, entry, row]
        route_slot = route_slots[src, dst_ord, entry, row]
        np.testing.assert_array_equal(h_route_weights[dst, flat_row], route_weights[src, token, route_slot])


def test_source_push_h_row_route_weights_lowers_from_source_sharded_route_weights():
    selected_experts, combine_weights = _small_routing_inputs()
    plan = build_source_push_plan(
        selected_experts,
        combine_weights,
        ep_size=EP_SIZE,
        experts_per_rank=EXPERTS_PER_RANK,
        block_m=BLOCK_M,
        capacity_factor=2.0,
    )
    send_meta, expert_base, src_base_by_expert, hidden_rows = _source_padded_h_row_metadata(plan)
    mesh = AbstractMesh(
        axis_sizes=(EP_SIZE,),
        axis_names=("expert",),
        axis_types=(AxisType.Explicit,),
    )

    def h_row_weights(weights_arg):
        return source_push_h_row_route_weights_jax(
            weights_arg,
            plan,
            send_meta,
            expert_base,
            src_base_by_expert,
            hidden_rows_per_rank=hidden_rows,
            use_exact_expert_major=False,
        )

    with use_abstract_mesh(mesh):
        source_sharded_weights = jax.ShapeDtypeStruct(
            combine_weights.shape,
            combine_weights.dtype,
            sharding=NamedSharding(mesh, P("expert", None, None)),
        )
        out_shape = jax.eval_shape(h_row_weights, source_sharded_weights)
        assert out_shape.shape == (EP_SIZE, hidden_rows)
        assert out_shape.sharding.spec == P("expert", None)
        jax.jit(h_row_weights).trace(source_sharded_weights).lower(lowering_platforms=("cpu",))


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


def test_source_push_h_forward_applies_route_weights_before_w2():
    selected_experts, route_weights = _small_routing_inputs()
    plan = build_source_push_plan(
        selected_experts,
        route_weights,
        ep_size=EP_SIZE,
        experts_per_rank=EXPERTS_PER_RANK,
        block_m=BLOCK_M,
        capacity_factor=2.0,
    )
    x = (jnp.arange(EP_SIZE * selected_experts.shape[1] * 3, dtype=jnp.float32).reshape(EP_SIZE, -1, 3) + 1.0) / 10.0
    packed_x = pack_source_push_tokens(x, plan)
    w_gate_up = (
        jnp.arange(EP_SIZE * EXPERTS_PER_RANK * 3 * 4, dtype=jnp.float32).reshape(EP_SIZE, EXPERTS_PER_RANK, 3, 4)
        + 1.0
    ) / 20.0
    w_down = (
        jnp.arange(EP_SIZE * EXPERTS_PER_RANK * 2 * 3, dtype=jnp.float32).reshape(EP_SIZE, EXPERTS_PER_RANK, 2, 3)
        + 1.0
    ) / 30.0

    h = source_push_w13_h(packed_x, w_gate_up, plan)
    return_y = source_push_w2_from_h_return(h, route_weights, w_down, plan)
    observed = np.asarray(source_push_combine_preweighted(return_y, plan), dtype=np.float32)

    expected = np.zeros((EP_SIZE, selected_experts.shape[1], x.shape[-1]), dtype=np.float32)
    selected_host = np.asarray(selected_experts)
    x_host = np.asarray(x, dtype=np.float32)
    route_weights_host = np.asarray(route_weights, dtype=np.float32)
    w_gate_up_host = np.asarray(w_gate_up, dtype=np.float32)
    w_down_host = np.asarray(w_down, dtype=np.float32)
    for src in range(EP_SIZE):
        for token in range(selected_experts.shape[1]):
            for route_slot in range(selected_experts.shape[2]):
                global_expert = int(selected_host[src, token, route_slot])
                dst = global_expert // EXPERTS_PER_RANK
                expert = global_expert % EXPERTS_PER_RANK
                preactivation = x_host[src, token] @ w_gate_up_host[dst, expert]
                gate, up = np.split(preactivation, 2)
                activation = gate * (1.0 / (1.0 + np.exp(-gate))) * up
                expected[src, token] += (route_weights_host[src, token, route_slot] * activation) @ w_down_host[
                    dst, expert
                ]

    assignment_ids = np.asarray(plan.assignment_ids)
    valid_mask = np.asarray(plan.valid_mask)
    local_experts = np.asarray(plan.local_experts)
    local_row_starts = np.asarray(plan.local_row_starts)
    src_base_by_expert = np.asarray(plan.src_base_by_expert)
    h_host = np.asarray(h, dtype=np.float32)
    for src, dst_ord, entry, row in np.argwhere(valid_mask):
        dst = (src + dst_ord) % EP_SIZE
        expert = local_experts[src, dst_ord, entry]
        expert_row = src_base_by_expert[dst, src, expert] + local_row_starts[src, dst_ord, entry] + row
        assignment = assignment_ids[src, dst_ord, entry, row]
        token = assignment // selected_experts.shape[2]
        route_slot = assignment % selected_experts.shape[2]
        assert np.asarray(plan.combine_weights)[src, dst_ord, entry, row] == route_weights_host[src, token, route_slot]
        np.testing.assert_allclose(
            h_host[dst, expert, expert_row],
            x_host[src, token] @ w_gate_up_host[dst, expert],
            rtol=1e-6,
            atol=1e-6,
        )
    np.testing.assert_allclose(observed, expected, rtol=1e-6, atol=1e-6)


def test_source_push_semantic_forward_reference_matches_host_plan_reference():
    selected_experts, route_weights = _small_routing_inputs()
    plan = build_source_push_plan(
        selected_experts,
        route_weights,
        ep_size=EP_SIZE,
        experts_per_rank=EXPERTS_PER_RANK,
        block_m=BLOCK_M,
        capacity_factor=2.0,
    )
    semantic_plan = build_source_push_semantic_plan_jax(
        selected_experts,
        route_weights,
        ep_size=EP_SIZE,
        experts_per_rank=EXPERTS_PER_RANK,
        rows_per_src_dst_capacity=selected_experts.shape[1] * selected_experts.shape[2],
        capacity_factor=2.0,
    )
    x = (jnp.arange(EP_SIZE * selected_experts.shape[1] * 3, dtype=jnp.float32).reshape(EP_SIZE, -1, 3) + 1.0) / 10.0
    w_gate_up = (
        jnp.arange(EP_SIZE * EXPERTS_PER_RANK * 3 * 4, dtype=jnp.float32).reshape(EP_SIZE, EXPERTS_PER_RANK, 3, 4)
        + 1.0
    ) / 20.0
    w_down = (
        jnp.arange(EP_SIZE * EXPERTS_PER_RANK * 2 * 3, dtype=jnp.float32).reshape(EP_SIZE, EXPERTS_PER_RANK, 2, 3)
        + 1.0
    ) / 30.0

    packed_x = pack_source_push_tokens(x, plan)
    h = source_push_w13_h(packed_x, w_gate_up, plan)
    return_y = source_push_w2_from_h_return(h, route_weights, w_down, plan)
    expected = source_push_combine_preweighted(return_y, plan)

    observed, _z_pair, _h_pair, _route_y = source_push_semantic_forward_reference_jax(
        x,
        w_gate_up,
        w_down,
        semantic_plan,
    )

    np.testing.assert_allclose(observed, expected, rtol=1e-5, atol=1e-5)


def test_source_push_semantic_forward_backward_references_match_jax_autodiff_inside_jit():
    selected_experts, route_weights = _small_routing_inputs()
    semantic_plan = build_source_push_semantic_plan_jax(
        selected_experts,
        route_weights,
        ep_size=EP_SIZE,
        experts_per_rank=EXPERTS_PER_RANK,
        rows_per_src_dst_capacity=selected_experts.shape[1] * selected_experts.shape[2],
        capacity_factor=2.0,
    )
    x = (jnp.arange(EP_SIZE * selected_experts.shape[1] * 3, dtype=jnp.float32).reshape(EP_SIZE, -1, 3) + 1.0) / 10.0
    w_gate_up = (
        jnp.arange(EP_SIZE * EXPERTS_PER_RANK * 3 * 4, dtype=jnp.float32).reshape(EP_SIZE, EXPERTS_PER_RANK, 3, 4)
        + 1.0
    ) / 20.0
    w_down = (
        jnp.arange(EP_SIZE * EXPERTS_PER_RANK * 2 * 3, dtype=jnp.float32).reshape(EP_SIZE, EXPERTS_PER_RANK, 2, 3)
        + 1.0
    ) / 30.0

    def loss(x_arg, w_gate_up_arg, w_down_arg):
        y, _z_pair, _h_pair, _route_y = source_push_semantic_forward_reference_jax(
            x_arg,
            w_gate_up_arg,
            w_down_arg,
            semantic_plan,
        )
        return jnp.sum(y)

    y_expected, _z_pair, _h_pair, _route_y = source_push_semantic_forward_reference_jax(
        x,
        w_gate_up,
        w_down,
        semantic_plan,
    )
    dx_expected, dw13_expected, dw2_expected = jax.grad(loss, argnums=(0, 1, 2))(x, w_gate_up, w_down)

    @jax.jit
    def references(selected_experts_arg, route_weights_arg, x_arg, w_gate_up_arg, w_down_arg):
        plan = build_source_push_semantic_plan_jax(
            selected_experts_arg,
            route_weights_arg,
            ep_size=EP_SIZE,
            experts_per_rank=EXPERTS_PER_RANK,
            rows_per_src_dst_capacity=selected_experts.shape[1] * selected_experts.shape[2],
            capacity_factor=2.0,
        )
        y, z_pair, h_pair, route_y = source_push_semantic_forward_reference_jax(
            x_arg,
            w_gate_up_arg,
            w_down_arg,
            plan,
        )
        dy_route, _dcombine = source_push_semantic_backward_source_expand_jax(jnp.ones_like(y), route_y, plan)
        dh_pair, dw2 = source_push_semantic_w2_backward_reference_jax(h_pair, dy_route, w_down_arg, plan)
        dz_pair = source_push_semantic_swiglu_backward_reference_jax(dh_pair, z_pair, plan)
        dx_pair, dw13 = source_push_semantic_w13_backward_reference_jax(x_arg, dz_pair, w_gate_up_arg, plan)
        dx = source_push_semantic_dx_combine_jax(dx_pair, plan)
        return y, dx, dw13, dw2

    y, dx, dw13, dw2 = references(selected_experts, route_weights, x, w_gate_up, w_down)

    np.testing.assert_allclose(y, y_expected, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(dx, dx_expected, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(dw13, dw13_expected, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(dw2, dw2_expected, rtol=1e-5, atol=1e-5)


def test_source_push_semantic_pair_expert_major_round_trip():
    selected_experts, route_weights = _small_routing_inputs()
    semantic_plan = build_source_push_semantic_plan_jax(
        selected_experts,
        route_weights,
        ep_size=EP_SIZE,
        experts_per_rank=EXPERTS_PER_RANK,
        rows_per_src_dst_capacity=selected_experts.shape[1] * selected_experts.shape[2],
        capacity_factor=2.0,
    )
    pair_values = (
        jnp.arange(math.prod(semantic_plan.assignment_ids.shape) * 3, dtype=jnp.float32).reshape(
            *semantic_plan.assignment_ids.shape,
            3,
        )
        + 1.0
    )
    pair_values = jnp.where(semantic_plan.valid_mask[..., None], pair_values, jnp.asarray(-999.0))
    rows_per_expert_capacity = int(np.max(np.asarray(semantic_plan.rows_per_local_expert)))

    expert_values, valid_by_expert = source_push_semantic_pair_to_expert_major_jax(
        pair_values,
        semantic_plan,
        rows_per_expert_capacity=rows_per_expert_capacity,
    )
    observed_pair = source_push_semantic_expert_major_to_pair_jax(expert_values, semantic_plan)
    expert_route_weights, route_valid = source_push_semantic_route_weights_expert_major_jax(
        semantic_plan,
        rows_per_expert_capacity=rows_per_expert_capacity,
    )

    np.testing.assert_allclose(
        np.asarray(observed_pair),
        np.asarray(
            jnp.where(semantic_plan.valid_mask[..., None], pair_values, jnp.zeros((), dtype=pair_values.dtype))
        ),
        atol=0,
        rtol=0,
    )
    np.testing.assert_array_equal(np.asarray(valid_by_expert), np.asarray(route_valid))
    assert expert_values.shape == (EP_SIZE, EXPERTS_PER_RANK, rows_per_expert_capacity, 3)
    assert expert_route_weights.shape == (EP_SIZE, EXPERTS_PER_RANK, rows_per_expert_capacity)


def test_source_push_semantic_reverse_route_indexes_expert_major_rows():
    selected_experts, route_weights = _small_routing_inputs()
    semantic_plan = build_source_push_semantic_plan_jax(
        selected_experts,
        route_weights,
        ep_size=EP_SIZE,
        experts_per_rank=EXPERTS_PER_RANK,
        rows_per_src_dst_capacity=2,
        capacity_factor=2.0,
    )
    rows_per_expert_capacity = int(np.max(np.asarray(semantic_plan.rows_per_local_expert)))
    reverse_route = source_push_semantic_reverse_route_jax(semantic_plan)
    pair_assignment_ids = semantic_plan.assignment_ids[..., None].astype(jnp.float32)
    expert_assignment_ids, valid_by_expert = source_push_semantic_pair_to_expert_major_jax(
        pair_assignment_ids,
        semantic_plan,
        rows_per_expert_capacity=rows_per_expert_capacity,
    )

    expected_valid = np.zeros(selected_experts.shape, dtype=np.bool_)
    selected_host = np.asarray(selected_experts)
    token_ids = np.asarray(semantic_plan.token_ids)
    route_slots = np.asarray(semantic_plan.route_slots)
    assignment_ids = np.asarray(semantic_plan.assignment_ids)
    valid_mask = np.asarray(semantic_plan.valid_mask)
    reverse_dst = np.asarray(reverse_route.route_dst)
    reverse_expert = np.asarray(reverse_route.route_expert)
    reverse_expert_row = np.asarray(reverse_route.route_expert_row)
    reverse_assignment = np.asarray(reverse_route.assignment_id)
    reverse_valid = np.asarray(reverse_route.route_valid)
    expert_assignment_ids_host = np.asarray(expert_assignment_ids[..., 0])
    valid_by_expert_host = np.asarray(valid_by_expert)

    for src, dst, pair_row in np.argwhere(valid_mask):
        token = token_ids[src, dst, pair_row]
        route_slot = route_slots[src, dst, pair_row]
        expert = selected_host[src, token, route_slot] % EXPERTS_PER_RANK
        expert_row = reverse_expert_row[src, token, route_slot]
        expected_valid[src, token, route_slot] = True

        assert reverse_valid[src, token, route_slot]
        assert reverse_dst[src, token, route_slot] == dst
        assert reverse_expert[src, token, route_slot] == expert
        assert reverse_assignment[src, token, route_slot] == assignment_ids[src, dst, pair_row]
        assert valid_by_expert_host[dst, expert, expert_row]
        assert expert_assignment_ids_host[dst, expert, expert_row] == assignment_ids[src, dst, pair_row]

    np.testing.assert_array_equal(reverse_valid, expected_valid)
    assert not np.all(reverse_valid)
    np.testing.assert_array_equal(reverse_dst[~reverse_valid], 0)
    np.testing.assert_array_equal(reverse_expert[~reverse_valid], 0)
    np.testing.assert_array_equal(reverse_expert_row[~reverse_valid], 0)
    np.testing.assert_array_equal(reverse_assignment[~reverse_valid], INVALID_ASSIGNMENT_ID)


def test_source_push_semantic_reverse_route_jax_is_jittable():
    selected_experts, route_weights = _small_routing_inputs()
    semantic_plan = build_source_push_semantic_plan_jax(
        selected_experts,
        route_weights,
        ep_size=EP_SIZE,
        experts_per_rank=EXPERTS_PER_RANK,
        rows_per_src_dst_capacity=selected_experts.shape[1] * selected_experts.shape[2],
        capacity_factor=2.0,
    )

    reverse_route = jax.jit(source_push_semantic_reverse_route_jax)(semantic_plan)

    assert reverse_route.route_dst.shape == selected_experts.shape
    np.testing.assert_array_equal(
        np.asarray(reverse_route.route_valid),
        np.ones(selected_experts.shape, dtype=np.bool_),
    )


def test_source_push_semantic_reverse_route_accepts_custom_source_row_bases():
    selected_experts, combine_weights = _small_routing_inputs()
    semantic_plan = build_source_push_semantic_plan_jax(
        selected_experts,
        combine_weights,
        ep_size=EP_SIZE,
        experts_per_rank=EXPERTS_PER_RANK,
        rows_per_src_dst_capacity=6,
        capacity_factor=2.0,
    )
    source_offset = jnp.arange(EP_SIZE, dtype=jnp.int32)[None, :, None] * 8
    custom_bases = semantic_plan.src_base_by_expert + source_offset
    default = source_push_semantic_reverse_route_jax(semantic_plan)
    custom = source_push_semantic_reverse_route_jax(
        semantic_plan,
        source_row_base_by_expert=custom_bases,
    )
    source = jnp.arange(EP_SIZE, dtype=jnp.int32)[:, None, None]
    expected_delta = source_offset[0, source, 0]
    expected_rows = jnp.where(default.route_valid, default.route_expert_row + expected_delta, 0)

    np.testing.assert_array_equal(np.asarray(custom.route_expert_row), np.asarray(expected_rows))
    np.testing.assert_array_equal(np.asarray(custom.route_dst), np.asarray(default.route_dst))
    np.testing.assert_array_equal(np.asarray(custom.route_expert), np.asarray(default.route_expert))


def test_source_push_semantic_w2_backward_matches_expert_major_matmul_boundary():
    selected_experts, route_weights = _small_routing_inputs()
    semantic_plan = build_source_push_semantic_plan_jax(
        selected_experts,
        route_weights,
        ep_size=EP_SIZE,
        experts_per_rank=EXPERTS_PER_RANK,
        rows_per_src_dst_capacity=selected_experts.shape[1] * selected_experts.shape[2],
        capacity_factor=2.0,
    )
    pair_shape = semantic_plan.assignment_ids.shape
    h_pair = (jnp.arange(math.prod(pair_shape) * 2, dtype=jnp.float32).reshape(*pair_shape, 2) + 1.0) / 11.0
    dy_route = (jnp.arange(math.prod(pair_shape) * 3, dtype=jnp.float32).reshape(*pair_shape, 3) - 7.0) / 13.0
    w_down = (
        jnp.arange(EP_SIZE * EXPERTS_PER_RANK * 2 * 3, dtype=jnp.float32).reshape(EP_SIZE, EXPERTS_PER_RANK, 2, 3)
        - 3.0
    ) / 17.0
    h_pair = jnp.where(semantic_plan.valid_mask[..., None], h_pair, jnp.asarray(0.0, dtype=h_pair.dtype))
    dy_route = jnp.where(semantic_plan.valid_mask[..., None], dy_route, jnp.asarray(0.0, dtype=dy_route.dtype))
    rows_per_expert_capacity = int(np.max(np.asarray(semantic_plan.rows_per_local_expert)))

    h_expert, valid_by_expert = source_push_semantic_pair_to_expert_major_jax(
        h_pair,
        semantic_plan,
        rows_per_expert_capacity=rows_per_expert_capacity,
    )
    dy_expert, dy_valid_by_expert = source_push_semantic_pair_to_expert_major_jax(
        dy_route,
        semantic_plan,
        rows_per_expert_capacity=rows_per_expert_capacity,
    )
    dh_expert = jnp.einsum("dech,deih->deci", dy_expert, w_down, preferred_element_type=jnp.float32)
    dw2 = jnp.einsum("deci,dech->deih", h_expert, dy_expert, preferred_element_type=jnp.float32)
    dh_pair = source_push_semantic_expert_major_to_pair_jax(dh_expert, semantic_plan)

    expected_dh, expected_dw2 = source_push_semantic_w2_backward_reference_jax(
        h_pair,
        dy_route,
        w_down,
        semantic_plan,
    )
    np.testing.assert_array_equal(np.asarray(valid_by_expert), np.asarray(dy_valid_by_expert))
    np.testing.assert_allclose(np.asarray(dh_pair), np.asarray(expected_dh), atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(dw2), np.asarray(expected_dw2), atol=1e-6, rtol=1e-6)


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


def test_source_push_plan_balanced_capacity_factor_one_has_no_drops():
    ep_size = 4
    experts_per_rank = 2
    tokens_per_source = 8
    topk = 2
    block_m = 2
    global_experts = ep_size * experts_per_rank
    assignments_per_source = tokens_per_source * topk
    source_experts = np.tile(np.arange(global_experts, dtype=np.int32), assignments_per_source // global_experts)
    selected_experts = np.broadcast_to(
        source_experts.reshape(tokens_per_source, topk), (ep_size, tokens_per_source, topk)
    )
    combine_weights = np.ones((ep_size, tokens_per_source, topk), dtype=np.float32)

    plan = build_source_push_plan(
        jnp.asarray(selected_experts),
        jnp.asarray(combine_weights),
        ep_size=ep_size,
        experts_per_rank=experts_per_rank,
        block_m=block_m,
        capacity_factor=1.0,
    )
    stats = source_push_plan_row_stats(plan)

    assert int(plan.dropped_routes) == 0
    assert stats.useful_rows == ep_size * assignments_per_source
    assert stats.rounded_rows == stats.useful_rows
    assert stats.masked_row_fraction == 0.0
    np.testing.assert_array_equal(
        np.asarray(plan.counts_by_src_dst_expert),
        np.full((ep_size, ep_size, experts_per_rank), 2, dtype=np.int32),
    )


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
