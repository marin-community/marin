# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

from levanter.grug._moe.source_push_plan import build_source_push_semantic_plan_jax
from levanter.grug._moe.source_push_semantic_metadata_pallas import (
    SourcePushSemanticMetadataPallasBlockSizes,
    build_source_push_semantic_plan_pallas_mgpu,
    build_source_push_semantic_tile_metadata_pallas_mgpu,
    source_push_semantic_row_scatter_pallas_mgpu,
    source_push_semantic_tile_histogram_pallas_mgpu,
    source_push_semantic_tile_metadata_apply_pair_capacity_jax,
)


EP_SIZE = 2
EXPERTS_PER_RANK = 2


def _routing_inputs() -> tuple[jax.Array, jax.Array]:
    selected_experts = jnp.array(
        [
            [[0, 1], [2, 3], [0, 2], [1, 2], [3, 3]],
            [[3, 2], [1, 0], [2, 2], [0, 1], [3, 0]],
        ],
        dtype=jnp.int32,
    )
    route_weights = jnp.arange(selected_experts.size, dtype=jnp.float32).reshape(selected_experts.shape) / 10.0
    return selected_experts, route_weights


def _clipped_routing_inputs() -> tuple[jax.Array, jax.Array]:
    selected_experts = jnp.array(
        [
            [[0, 0], [0, 0], [1, 1], [2, 3], [0, 1]],
            [[0, 0], [0, 0], [0, 0], [1, 1], [2, 3]],
        ],
        dtype=jnp.int32,
    )
    route_weights = jnp.arange(selected_experts.size, dtype=jnp.float32).reshape(selected_experts.shape) / 10.0
    return selected_experts, route_weights


def _expert_capacity_routing_inputs() -> tuple[jax.Array, jax.Array]:
    selected_experts = jnp.array(
        [
            [[0, 0], [0, 0], [1, 1], [1, 2]],
            [[0, 0], [0, 1], [1, 2], [3, 3]],
        ],
        dtype=jnp.int32,
    )
    route_weights = jnp.arange(selected_experts.size, dtype=jnp.float32).reshape(selected_experts.shape) / 10.0
    return selected_experts, route_weights


def _assert_semantic_plans_equal(observed, expected) -> None:
    np.testing.assert_array_equal(np.asarray(observed.assignment_ids), np.asarray(expected.assignment_ids))
    np.testing.assert_array_equal(np.asarray(observed.token_ids), np.asarray(expected.token_ids))
    np.testing.assert_array_equal(np.asarray(observed.route_slots), np.asarray(expected.route_slots))
    np.testing.assert_array_equal(np.asarray(observed.valid_mask), np.asarray(expected.valid_mask))
    np.testing.assert_allclose(np.asarray(observed.route_weights), np.asarray(expected.route_weights))
    np.testing.assert_array_equal(np.asarray(observed.xcounts), np.asarray(expected.xcounts))
    np.testing.assert_array_equal(np.asarray(observed.pair_expert_base), np.asarray(expected.pair_expert_base))
    np.testing.assert_array_equal(
        np.asarray(observed.rows_per_local_expert),
        np.asarray(expected.rows_per_local_expert),
    )
    np.testing.assert_array_equal(np.asarray(observed.expert_base), np.asarray(expected.expert_base))
    np.testing.assert_array_equal(
        np.asarray(observed.src_base_by_expert),
        np.asarray(expected.src_base_by_expert),
    )
    np.testing.assert_array_equal(
        np.asarray(observed.reverse_route.route_dst),
        np.asarray(expected.reverse_route.route_dst),
    )
    np.testing.assert_array_equal(
        np.asarray(observed.reverse_route.route_expert),
        np.asarray(expected.reverse_route.route_expert),
    )
    np.testing.assert_array_equal(
        np.asarray(observed.reverse_route.route_expert_row),
        np.asarray(expected.reverse_route.route_expert_row),
    )
    np.testing.assert_array_equal(
        np.asarray(observed.reverse_route.route_valid),
        np.asarray(expected.reverse_route.route_valid),
    )
    np.testing.assert_array_equal(
        np.asarray(observed.reverse_route.assignment_id),
        np.asarray(expected.reverse_route.assignment_id),
    )
    assert int(observed.routing_dropped_routes) == int(expected.routing_dropped_routes)
    assert int(observed.metadata_overflow_routes) == int(expected.metadata_overflow_routes)
    assert int(observed.dropped_routes) == int(expected.dropped_routes)


def _tile_counts_reference(
    selected_experts: jax.Array,
    *,
    ep_size: int,
    experts_per_rank: int,
    tile_assignments: int,
) -> np.ndarray:
    selected_np = np.asarray(selected_experts)
    source_count, tokens_per_source, topk = selected_np.shape
    assignments_per_source = tokens_per_source * topk
    tile_count = math.ceil(assignments_per_source / tile_assignments)
    counts = np.zeros((source_count, tile_count, ep_size, experts_per_rank), dtype=np.int32)
    flat = selected_np.reshape(source_count, assignments_per_source)
    for src in range(source_count):
        for assignment in range(assignments_per_source):
            tile = assignment // tile_assignments
            global_expert = int(flat[src, assignment])
            counts[src, tile, global_expert // experts_per_rank, global_expert % experts_per_rank] += 1
    return counts


def test_source_push_semantic_tile_histogram_pallas_interpret_matches_reference():
    selected_experts, _route_weights = _routing_inputs()
    block_sizes = SourcePushSemanticMetadataPallasBlockSizes(tile_assignments=3)

    observed = source_push_semantic_tile_histogram_pallas_mgpu(
        selected_experts,
        ep_size=EP_SIZE,
        experts_per_rank=EXPERTS_PER_RANK,
        block_sizes=block_sizes,
        interpret=True,
    )

    expected = _tile_counts_reference(
        selected_experts,
        ep_size=EP_SIZE,
        experts_per_rank=EXPERTS_PER_RANK,
        tile_assignments=block_sizes.tile_assignments,
    )
    np.testing.assert_array_equal(np.asarray(observed), expected)


def test_source_push_semantic_plan_jax_is_jittable_and_preserves_metadata_contract():
    selected_experts, route_weights = _clipped_routing_inputs()

    @jax.jit
    def build_plan(selected_experts_arg, route_weights_arg):
        return build_source_push_semantic_plan_jax(
            selected_experts_arg,
            route_weights_arg,
            ep_size=EP_SIZE,
            experts_per_rank=EXPERTS_PER_RANK,
            rows_per_src_dst_capacity=3,
            capacity_factor=0.5,
        )

    observed = build_plan(selected_experts, route_weights)
    expected = build_source_push_semantic_plan_jax(
        selected_experts,
        route_weights,
        ep_size=EP_SIZE,
        experts_per_rank=EXPERTS_PER_RANK,
        rows_per_src_dst_capacity=3,
        capacity_factor=0.5,
    )

    _assert_semantic_plans_equal(observed, expected)


def test_source_push_semantic_tile_metadata_pallas_matches_jax_plan_offsets():
    selected_experts, route_weights = _routing_inputs()
    block_sizes = SourcePushSemanticMetadataPallasBlockSizes(tile_assignments=4)
    rows_per_pair = selected_experts.shape[1] * selected_experts.shape[2]

    tile_metadata = build_source_push_semantic_tile_metadata_pallas_mgpu(
        selected_experts,
        ep_size=EP_SIZE,
        experts_per_rank=EXPERTS_PER_RANK,
        capacity_factor=2.0,
        block_sizes=block_sizes,
        interpret=True,
    )
    semantic_plan = build_source_push_semantic_plan_jax(
        selected_experts,
        route_weights,
        ep_size=EP_SIZE,
        experts_per_rank=EXPERTS_PER_RANK,
        rows_per_src_dst_capacity=rows_per_pair,
        capacity_factor=2.0,
    )

    np.testing.assert_array_equal(np.asarray(tile_metadata.xcounts), np.asarray(semantic_plan.xcounts))
    np.testing.assert_array_equal(
        np.asarray(tile_metadata.pair_expert_base), np.asarray(semantic_plan.pair_expert_base)
    )
    np.testing.assert_array_equal(
        np.asarray(tile_metadata.rows_per_local_expert),
        np.asarray(semantic_plan.rows_per_local_expert),
    )
    np.testing.assert_array_equal(np.asarray(tile_metadata.expert_base), np.asarray(semantic_plan.expert_base))
    np.testing.assert_array_equal(
        np.asarray(tile_metadata.src_base_by_expert),
        np.asarray(semantic_plan.src_base_by_expert),
    )
    assert int(tile_metadata.routing_dropped_routes) == int(semantic_plan.routing_dropped_routes)


def test_source_push_semantic_tile_metadata_pallas_exposes_tile_local_bases_after_pair_capacity():
    selected_experts, _route_weights = _clipped_routing_inputs()
    block_sizes = SourcePushSemanticMetadataPallasBlockSizes(tile_assignments=3)

    tile_metadata = build_source_push_semantic_tile_metadata_pallas_mgpu(
        selected_experts,
        ep_size=EP_SIZE,
        experts_per_rank=EXPERTS_PER_RANK,
        capacity_factor=0.5,
        block_sizes=block_sizes,
        interpret=True,
    )
    capped_metadata = source_push_semantic_tile_metadata_apply_pair_capacity_jax(
        tile_metadata,
        rows_per_src_dst_capacity=3,
    )

    tile_prefix = jnp.cumsum(capped_metadata.tile_counts, axis=1, dtype=jnp.int32) - capped_metadata.tile_counts
    expected_tile_pair_base = jnp.minimum(tile_prefix, capped_metadata.xcounts[:, None, :, :])

    np.testing.assert_array_equal(np.asarray(capped_metadata.tile_pair_base), np.asarray(expected_tile_pair_base))
    assert int(jnp.sum(capped_metadata.tile_counts)) == selected_experts.size
    assert int(capped_metadata.routing_dropped_routes) > 0
    assert np.asarray(capped_metadata.tile_counts).shape[:2] == (EP_SIZE, 4)


def test_source_push_semantic_tile_metadata_pallas_applies_receiver_capacity_clipping():
    selected_experts, route_weights = _routing_inputs()
    block_sizes = SourcePushSemanticMetadataPallasBlockSizes(tile_assignments=5)

    tile_metadata = build_source_push_semantic_tile_metadata_pallas_mgpu(
        selected_experts,
        ep_size=EP_SIZE,
        experts_per_rank=EXPERTS_PER_RANK,
        capacity_factor=0.5,
        block_sizes=block_sizes,
        interpret=True,
    )
    semantic_plan = build_source_push_semantic_plan_jax(
        selected_experts,
        route_weights,
        ep_size=EP_SIZE,
        experts_per_rank=EXPERTS_PER_RANK,
        rows_per_src_dst_capacity=selected_experts.shape[1] * selected_experts.shape[2],
        capacity_factor=0.5,
    )

    np.testing.assert_array_equal(np.asarray(tile_metadata.xcounts), np.asarray(semantic_plan.xcounts))
    assert int(tile_metadata.routing_dropped_routes) == int(semantic_plan.routing_dropped_routes)


def test_source_push_semantic_plan_pallas_facade_is_jittable_in_interpret_mode():
    selected_experts, route_weights = _routing_inputs()
    block_sizes = SourcePushSemanticMetadataPallasBlockSizes(tile_assignments=3)

    @jax.jit
    def build_plan(selected_experts_arg, route_weights_arg):
        return build_source_push_semantic_plan_pallas_mgpu(
            selected_experts_arg,
            route_weights_arg,
            ep_size=EP_SIZE,
            experts_per_rank=EXPERTS_PER_RANK,
            rows_per_src_dst_capacity=10,
            capacity_factor=2.0,
            block_sizes=block_sizes,
            interpret=True,
        )

    observed = build_plan(selected_experts, route_weights)
    expected = build_source_push_semantic_plan_jax(
        selected_experts,
        route_weights,
        ep_size=EP_SIZE,
        experts_per_rank=EXPERTS_PER_RANK,
        rows_per_src_dst_capacity=10,
        capacity_factor=2.0,
    )

    _assert_semantic_plans_equal(observed, expected)


def test_source_push_semantic_tile_histogram_pallas_interpret_is_jittable():
    selected_experts, _route_weights = _routing_inputs()
    block_sizes = SourcePushSemanticMetadataPallasBlockSizes(tile_assignments=3)

    @jax.jit
    def build_counts(selected_experts_arg):
        return source_push_semantic_tile_histogram_pallas_mgpu(
            selected_experts_arg,
            ep_size=EP_SIZE,
            experts_per_rank=EXPERTS_PER_RANK,
            block_sizes=block_sizes,
            interpret=True,
        )

    observed = build_counts(selected_experts)
    expected = _tile_counts_reference(
        selected_experts,
        ep_size=EP_SIZE,
        experts_per_rank=EXPERTS_PER_RANK,
        tile_assignments=block_sizes.tile_assignments,
    )
    np.testing.assert_array_equal(np.asarray(observed), expected)


def test_source_push_semantic_row_scatter_pallas_interpret_matches_jax_reference_rows():
    selected_experts, route_weights = _routing_inputs()
    block_sizes = SourcePushSemanticMetadataPallasBlockSizes(tile_assignments=3)
    rows_per_pair = selected_experts.shape[1] * selected_experts.shape[2]
    tile_metadata = build_source_push_semantic_tile_metadata_pallas_mgpu(
        selected_experts,
        ep_size=EP_SIZE,
        experts_per_rank=EXPERTS_PER_RANK,
        capacity_factor=2.0,
        block_sizes=block_sizes,
        interpret=True,
    )
    tile_metadata = source_push_semantic_tile_metadata_apply_pair_capacity_jax(
        tile_metadata,
        rows_per_src_dst_capacity=rows_per_pair,
    )

    observed_assignment_ids, observed_weights = source_push_semantic_row_scatter_pallas_mgpu(
        selected_experts,
        route_weights,
        tile_metadata,
        rows_per_src_dst_capacity=rows_per_pair,
        interpret=True,
    )
    expected = build_source_push_semantic_plan_jax(
        selected_experts,
        route_weights,
        ep_size=EP_SIZE,
        experts_per_rank=EXPERTS_PER_RANK,
        rows_per_src_dst_capacity=rows_per_pair,
        capacity_factor=2.0,
    )

    np.testing.assert_array_equal(np.asarray(observed_assignment_ids), np.asarray(expected.assignment_ids))
    np.testing.assert_allclose(np.asarray(observed_weights), np.asarray(expected.route_weights))


def test_source_push_semantic_plan_pallas_facade_matches_jax_reference_scatter():
    selected_experts, route_weights = _routing_inputs()
    block_sizes = SourcePushSemanticMetadataPallasBlockSizes(tile_assignments=3)

    observed = build_source_push_semantic_plan_pallas_mgpu(
        selected_experts,
        route_weights,
        ep_size=EP_SIZE,
        experts_per_rank=EXPERTS_PER_RANK,
        rows_per_src_dst_capacity=10,
        capacity_factor=2.0,
        block_sizes=block_sizes,
        interpret=True,
    )
    expected = build_source_push_semantic_plan_jax(
        selected_experts,
        route_weights,
        ep_size=EP_SIZE,
        experts_per_rank=EXPERTS_PER_RANK,
        rows_per_src_dst_capacity=10,
        capacity_factor=2.0,
    )

    _assert_semantic_plans_equal(observed, expected)


def test_source_push_semantic_plan_pallas_facade_applies_pair_capacity_like_jax_reference():
    selected_experts, route_weights = _routing_inputs()
    block_sizes = SourcePushSemanticMetadataPallasBlockSizes(tile_assignments=3)
    rows_per_pair = 3

    observed = build_source_push_semantic_plan_pallas_mgpu(
        selected_experts,
        route_weights,
        ep_size=EP_SIZE,
        experts_per_rank=EXPERTS_PER_RANK,
        rows_per_src_dst_capacity=rows_per_pair,
        capacity_factor=2.0,
        block_sizes=block_sizes,
        interpret=True,
    )
    expected = build_source_push_semantic_plan_jax(
        selected_experts,
        route_weights,
        ep_size=EP_SIZE,
        experts_per_rank=EXPERTS_PER_RANK,
        rows_per_src_dst_capacity=rows_per_pair,
        capacity_factor=2.0,
    )

    np.testing.assert_array_equal(np.asarray(observed.assignment_ids), np.asarray(expected.assignment_ids))
    np.testing.assert_allclose(np.asarray(observed.route_weights), np.asarray(expected.route_weights))
    np.testing.assert_array_equal(np.asarray(observed.xcounts), np.asarray(expected.xcounts))
    assert int(observed.metadata_overflow_routes) == int(expected.metadata_overflow_routes)
    assert int(observed.dropped_routes) == int(expected.dropped_routes)


def test_source_push_semantic_plan_pallas_facade_applies_receiver_and_pair_capacity_like_jax_reference():
    selected_experts, route_weights = _clipped_routing_inputs()
    block_sizes = SourcePushSemanticMetadataPallasBlockSizes(tile_assignments=3)

    observed = build_source_push_semantic_plan_pallas_mgpu(
        selected_experts,
        route_weights,
        ep_size=EP_SIZE,
        experts_per_rank=EXPERTS_PER_RANK,
        rows_per_src_dst_capacity=3,
        capacity_factor=0.5,
        block_sizes=block_sizes,
        interpret=True,
    )
    expected = build_source_push_semantic_plan_jax(
        selected_experts,
        route_weights,
        ep_size=EP_SIZE,
        experts_per_rank=EXPERTS_PER_RANK,
        rows_per_src_dst_capacity=3,
        capacity_factor=0.5,
    )

    _assert_semantic_plans_equal(observed, expected)
    assert int(observed.routing_dropped_routes) > 0
    assert int(observed.metadata_overflow_routes) > 0


def test_source_push_semantic_plan_pallas_facade_applies_expert_capacity_after_pair_clipping():
    selected_experts, route_weights = _expert_capacity_routing_inputs()
    block_sizes = SourcePushSemanticMetadataPallasBlockSizes(tile_assignments=3)

    observed = build_source_push_semantic_plan_pallas_mgpu(
        selected_experts,
        route_weights,
        ep_size=EP_SIZE,
        experts_per_rank=EXPERTS_PER_RANK,
        rows_per_src_dst_capacity=5,
        rows_per_expert_capacity=5,
        capacity_factor=2.0,
        block_sizes=block_sizes,
        interpret=True,
    )
    expected = build_source_push_semantic_plan_jax(
        selected_experts,
        route_weights,
        ep_size=EP_SIZE,
        experts_per_rank=EXPERTS_PER_RANK,
        rows_per_src_dst_capacity=5,
        rows_per_expert_capacity=5,
        capacity_factor=2.0,
    )

    _assert_semantic_plans_equal(observed, expected)
    np.testing.assert_array_equal(
        np.asarray(observed.xcounts),
        np.array(
            [
                [[4, 1], [1, 0]],
                [[1, 2], [1, 2]],
            ],
            dtype=np.int32,
        ),
    )
    assert int(observed.routing_dropped_routes) == 0
    assert int(observed.metadata_overflow_routes) == 4
    assert int(observed.dropped_routes) == 4


def test_source_push_semantic_plan_pallas_facade_expert_capacity_is_jittable_and_matches_jax_reverse_routes():
    selected_experts, route_weights = _expert_capacity_routing_inputs()
    block_sizes = SourcePushSemanticMetadataPallasBlockSizes(tile_assignments=3)

    @jax.jit
    def build_plan(selected_experts_arg, route_weights_arg):
        return build_source_push_semantic_plan_pallas_mgpu(
            selected_experts_arg,
            route_weights_arg,
            ep_size=EP_SIZE,
            experts_per_rank=EXPERTS_PER_RANK,
            rows_per_src_dst_capacity=5,
            rows_per_expert_capacity=5,
            capacity_factor=2.0,
            block_sizes=block_sizes,
            interpret=True,
        )

    observed = build_plan(selected_experts, route_weights)
    expected = build_source_push_semantic_plan_jax(
        selected_experts,
        route_weights,
        ep_size=EP_SIZE,
        experts_per_rank=EXPERTS_PER_RANK,
        rows_per_src_dst_capacity=5,
        rows_per_expert_capacity=5,
        capacity_factor=2.0,
    )

    _assert_semantic_plans_equal(observed, expected)
    np.testing.assert_array_equal(
        np.asarray(observed.reverse_route.route_valid),
        np.asarray(expected.reverse_route.route_valid),
    )
    np.testing.assert_array_equal(
        np.asarray(observed.reverse_route.route_expert_row),
        np.asarray(expected.reverse_route.route_expert_row),
    )
    assert int(observed.metadata_overflow_routes) == 4
    assert int(observed.dropped_routes) == 4
