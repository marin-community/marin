# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh, PartitionSpec as P

from levanter.grug._moe.source_push_backward_w2 import SOURCE_PUSH_MESH_AXIS
from levanter.grug._moe.source_push_plan import (
    build_source_push_semantic_plan_jax,
    source_push_semantic_backward_source_expand_jax,
    source_push_semantic_dx_combine_jax,
    source_push_semantic_pair_to_expert_major_jax,
    source_push_semantic_queue_metadata_jax,
    source_push_semantic_swiglu_backward_reference_jax,
)
from levanter.grug._moe.source_push_semantic_backward_pallas import (
    SourcePushSemanticBackwardPallasBlockSizes,
    SourcePushSemanticDxReturnPallasBlockSizes,
    source_push_semantic_backward_dcombine_from_return_queue_pallas_mgpu,
    source_push_semantic_backward_dcombine_source_gather_jax,
    source_push_semantic_backward_dcombine_source_gather_pallas_mgpu,
    source_push_semantic_backward_dy_route_expert_major_jax,
    source_push_semantic_backward_dy_route_source_push_pallas_mgpu,
    source_push_semantic_backward_source_expand_from_return_queue_jax,
    source_push_semantic_backward_source_expand_from_return_queue_pallas_mgpu,
    source_push_semantic_backward_source_expand_from_expert_major_jax,
    source_push_semantic_backward_source_expand_from_expert_major_owner_sharded_dcombine_pallas_mgpu,
    source_push_semantic_backward_source_expand_from_expert_major_pallas_mgpu,
    source_push_semantic_backward_source_expand_from_expert_major_source_gather_pallas_mgpu,
    source_push_semantic_backward_source_expand_from_expert_major_source_push_pallas_mgpu,
    source_push_semantic_backward_source_expand_expert_major_jax,
    source_push_semantic_backward_source_expand_pallas_mgpu,
    source_push_semantic_dx_combine_expert_major_jax,
    source_push_semantic_dx_combine_pallas_mgpu,
    source_push_semantic_dx_combine_source_queue_pallas_mgpu,
    source_push_semantic_dx_return_copy_only_pallas_mgpu,
    source_push_semantic_dx_return_direct_to_source_combine_pallas_mgpu,
    source_push_semantic_dx_return_direct_to_source_combine_reference_jax,
    source_push_semantic_dx_return_direct_to_source_pallas_mgpu,
    source_push_semantic_dx_return_direct_to_source_reference_jax,
    source_push_semantic_dx_return_expert_major_jax,
    source_push_semantic_dx_return_pallas_mgpu,
    source_push_semantic_dx_return_remote_source_gather_pallas_mgpu,
    source_push_semantic_dx_return_slot_reduce_pallas_mgpu,
    source_push_semantic_dx_return_source_gather_jax,
    source_push_semantic_dx_return_source_gather_pallas_mgpu,
    source_push_semantic_dx_return_sum_pallas_mgpu,
    source_push_semantic_swiglu_backward_expert_major_jax,
)
from levanter.grug._moe.source_push_semantic_inbox_pallas import source_push_semantic_inbox_layout_jax


def _semantic_plan():
    selected_experts = jnp.asarray(
        [
            [[0, 2], [1, 3], [0, 1]],
            [[2, 0], [3, 1], [2, 3]],
        ],
        dtype=jnp.int32,
    )
    combine_weights = jnp.asarray(
        [
            [[0.7, 0.3], [0.6, 0.4], [0.9, 0.1]],
            [[0.5, 0.5], [0.2, 0.8], [0.1, 0.9]],
        ],
        dtype=jnp.float32,
    )
    return build_source_push_semantic_plan_jax(
        selected_experts,
        combine_weights,
        ep_size=2,
        experts_per_rank=2,
        rows_per_src_dst_capacity=6,
        capacity_factor=2.0,
    )


def _rough_semantic_plan():
    selected_experts = jnp.asarray(
        [
            [[0, 0], [0, 1], [0, 2], [3, 0], [0, 0]],
            [[0, 2], [2, 2], [2, 3], [1, 2], [2, 2]],
        ],
        dtype=jnp.int32,
    )
    combine_weights = jnp.full(selected_experts.shape, 0.5, dtype=jnp.float32)
    return build_source_push_semantic_plan_jax(
        selected_experts,
        combine_weights,
        ep_size=2,
        experts_per_rank=2,
        rows_per_src_dst_capacity=10,
        capacity_factor=2.0,
    )


def _source_padded_b64_plan_and_queue():
    token = jnp.arange(64, dtype=jnp.int32)
    selected_experts = jnp.stack(
        (
            jnp.stack((token % 4, (token + 1) % 4), axis=1),
            jnp.stack(((token + 2) % 4, (token + 3) % 4), axis=1),
        ),
        axis=0,
    )
    combine_weights = jnp.broadcast_to(jnp.asarray([0.25, 0.75], dtype=jnp.float32), selected_experts.shape)
    plan = build_source_push_semantic_plan_jax(
        selected_experts,
        combine_weights,
        ep_size=2,
        experts_per_rank=2,
        rows_per_src_dst_capacity=64,
        capacity_factor=1.0,
    )
    queue = source_push_semantic_queue_metadata_jax(
        plan,
        return_row_block=64,
        entries_per_dst=4,
    )
    layout = source_push_semantic_inbox_layout_jax(
        plan,
        queue,
        rows_per_expert_capacity=128,
    )
    return plan, queue, layout


def _source_padded_row_bases(plan, *, row_block: int):
    counts = np.asarray(plan.xcounts, dtype=np.int32)
    rounded_counts = ((counts + row_block - 1) // row_block) * row_block
    padded_bases = np.cumsum(rounded_counts, axis=0, dtype=np.int32) - rounded_counts
    rows_per_expert_capacity = int(np.max(np.sum(rounded_counts, axis=0, dtype=np.int32)))
    return jnp.asarray(np.transpose(padded_bases, (1, 0, 2)), dtype=jnp.int32), rows_per_expert_capacity


def _independent_route_rows(plan, source_row_base_by_expert):
    counts = np.asarray(plan.xcounts, dtype=np.int32)
    pair_expert_base = np.asarray(plan.pair_expert_base, dtype=np.int32)
    valid = np.asarray(plan.valid_mask, dtype=np.bool_)
    source_bases = np.asarray(source_row_base_by_expert, dtype=np.int32)
    expert_ids = np.zeros(valid.shape, dtype=np.int32)
    expert_rows = np.zeros(valid.shape, dtype=np.int32)
    for source, destination, pair_row in np.ndindex(valid.shape):
        if not valid[source, destination, pair_row]:
            continue
        expert = int(np.searchsorted(np.cumsum(counts[source, destination]), pair_row, side="right"))
        expert_ids[source, destination, pair_row] = expert
        expert_rows[source, destination, pair_row] = (
            source_bases[destination, source, expert] + pair_row - pair_expert_base[source, destination, expert]
        )
    return expert_ids, expert_rows


def _independent_dy_route(dy, plan, source_row_base_by_expert, *, rows_per_expert_capacity: int):
    dy_host = np.asarray(dy, dtype=np.float32)
    weights = np.asarray(plan.route_weights, dtype=np.float32)
    token_ids = np.asarray(plan.token_ids, dtype=np.int32)
    valid = np.asarray(plan.valid_mask, dtype=np.bool_)
    expert_ids, expert_rows = _independent_route_rows(plan, source_row_base_by_expert)
    output = np.zeros(
        (plan.xcounts.shape[1], plan.xcounts.shape[2], rows_per_expert_capacity, dy.shape[-1]),
        dtype=np.float32,
    )
    for source, destination, pair_row in np.ndindex(valid.shape):
        if not valid[source, destination, pair_row]:
            continue
        expert = expert_ids[source, destination, pair_row]
        expert_row = expert_rows[source, destination, pair_row]
        token = token_ids[source, destination, pair_row]
        output[destination, expert, expert_row] = dy_host[source, token] * weights[source, destination, pair_row]
    return jnp.asarray(output, dtype=dy.dtype)


def _independent_expert_major_values(plan, source_row_base_by_expert, *, rows_per_expert_capacity: int, hidden: int):
    valid = np.asarray(plan.valid_mask, dtype=np.bool_)
    expert_ids, expert_rows = _independent_route_rows(plan, source_row_base_by_expert)
    output = np.zeros(
        (plan.xcounts.shape[1], plan.xcounts.shape[2], rows_per_expert_capacity, hidden),
        dtype=np.float32,
    )
    for source, destination, pair_row in np.ndindex(valid.shape):
        if not valid[source, destination, pair_row]:
            continue
        row_value = 100 * (source + 1) + 20 * destination + pair_row
        output[destination, expert_ids[source, destination, pair_row], expert_rows[source, destination, pair_row]] = (
            row_value + np.arange(hidden, dtype=np.float32) / hidden
        ) / 31
    return jnp.asarray(output)


def _independent_dx_combine(dx_route, plan, source_row_base_by_expert, *, output_dtype):
    route_values = np.asarray(dx_route.astype(jnp.bfloat16), dtype=np.float32)
    token_ids = np.asarray(plan.token_ids, dtype=np.int32)
    valid = np.asarray(plan.valid_mask, dtype=np.bool_)
    expert_ids, expert_rows = _independent_route_rows(plan, source_row_base_by_expert)
    output = np.zeros((plan.xcounts.shape[0], plan.tokens_per_source, dx_route.shape[-1]), dtype=np.float32)
    for source, destination, pair_row in np.ndindex(valid.shape):
        if not valid[source, destination, pair_row]:
            continue
        output[source, token_ids[source, destination, pair_row]] += route_values[
            destination,
            expert_ids[source, destination, pair_row],
            expert_rows[source, destination, pair_row],
        ]
    return jnp.asarray(output, dtype=output_dtype)


def _independent_dcombine(dy, route_y_expert, plan, source_row_base_by_expert):
    dy_host = np.asarray(dy, dtype=np.float32)
    route_values = np.asarray(route_y_expert.astype(jnp.bfloat16), dtype=np.float32)
    token_ids = np.asarray(plan.token_ids, dtype=np.int32)
    route_slots = np.asarray(plan.route_slots, dtype=np.int32)
    valid = np.asarray(plan.valid_mask, dtype=np.bool_)
    expert_ids, expert_rows = _independent_route_rows(plan, source_row_base_by_expert)
    output = np.zeros((plan.xcounts.shape[0], plan.tokens_per_source, plan.topk), dtype=np.float32)
    for source, destination, pair_row in np.ndindex(valid.shape):
        if not valid[source, destination, pair_row]:
            continue
        token = token_ids[source, destination, pair_row]
        output[source, token, route_slots[source, destination, pair_row]] = np.sum(
            dy_host[source, token]
            * route_values[
                destination,
                expert_ids[source, destination, pair_row],
                expert_rows[source, destination, pair_row],
            ],
            dtype=np.float32,
        )
    return jnp.asarray(output)


def _single_device_mesh() -> Mesh:
    return Mesh(np.asarray(jax.devices()[:1]), (SOURCE_PUSH_MESH_AXIS,))


def _single_device_explicit_mesh() -> Mesh:
    return Mesh(
        np.asarray(jax.devices()[:1]),
        (SOURCE_PUSH_MESH_AXIS,),
        axis_types=(AxisType.Explicit,),
    )


def _independent_dx_return_from_reverse_route(dx_route, plan, *, output_dtype):
    reverse_route = plan.reverse_route
    safe_dst = jnp.where(reverse_route.route_valid, reverse_route.route_dst, 0)
    safe_expert = jnp.where(reverse_route.route_valid, reverse_route.route_expert, 0)
    safe_row = jnp.where(reverse_route.route_valid, reverse_route.route_expert_row, 0)
    route_dx = dx_route.at[safe_dst, safe_expert, safe_row].get(mode="clip").astype(jnp.bfloat16)
    route_dx = jnp.where(
        reverse_route.route_valid[..., None],
        route_dx.astype(jnp.float32),
        jnp.zeros((), dtype=jnp.float32),
    )
    return jnp.sum(route_dx, axis=2, dtype=jnp.float32).astype(output_dtype)


def test_source_push_semantic_swiglu_backward_expert_major_matches_autodiff():
    z_expert = (jnp.arange(2 * 2 * 3 * 8, dtype=jnp.float32).reshape(2, 2, 3, 8) - 41.0) / 13.0
    dh_expert = (jnp.arange(2 * 2 * 3 * 4, dtype=jnp.float32).reshape(2, 2, 3, 4) - 17.0) / 11.0
    valid = jnp.asarray(
        [
            [[True, False, True], [False, True, True]],
            [[True, True, False], [True, False, True]],
        ]
    )

    def activation(preactivation):
        gate, up = jnp.split(preactivation, 2, axis=-1)
        return jnp.where(valid[..., None], jax.nn.silu(gate) * up, 0.0)

    _, pullback = jax.vjp(activation, z_expert)
    (expected,) = pullback(dh_expert)
    observed = source_push_semantic_swiglu_backward_expert_major_jax(dh_expert, z_expert, valid)

    np.testing.assert_allclose(np.asarray(observed), np.asarray(expected), atol=1e-6, rtol=1e-6)
    assert observed.dtype == jnp.float32
    np.testing.assert_array_equal(np.asarray(observed[~valid]), 0.0)


def test_source_push_semantic_swiglu_backward_expert_major_matches_pair_flat_reference_with_invalid_rows():
    plan = _semantic_plan()
    z_pair = (jnp.arange(2 * 2 * 6 * 8, dtype=jnp.float32).reshape(2, 2, 6, 8) - 83.0).astype(jnp.bfloat16)
    dh_pair = (jnp.arange(2 * 2 * 6 * 4, dtype=jnp.float32).reshape(2, 2, 6, 4) / 9.0).astype(jnp.bfloat16)
    z_expert, valid = source_push_semantic_pair_to_expert_major_jax(
        z_pair,
        plan,
        rows_per_expert_capacity=4,
    )
    dh_expert, _ = source_push_semantic_pair_to_expert_major_jax(
        dh_pair,
        plan,
        rows_per_expert_capacity=4,
    )
    z_expert = jnp.where(valid[..., None], z_expert, jnp.asarray(jnp.nan, dtype=z_expert.dtype))
    dh_expert = jnp.where(valid[..., None], dh_expert, jnp.asarray(jnp.inf, dtype=dh_expert.dtype))

    expected_pair = source_push_semantic_swiglu_backward_reference_jax(dh_pair, z_pair, plan)
    expected, expected_valid = source_push_semantic_pair_to_expert_major_jax(
        expected_pair,
        plan,
        rows_per_expert_capacity=4,
    )
    observed = source_push_semantic_swiglu_backward_expert_major_jax(dh_expert, z_expert, valid)

    np.testing.assert_array_equal(np.asarray(valid), np.asarray(expected_valid))
    np.testing.assert_allclose(np.asarray(observed), np.asarray(expected), atol=1e-5, rtol=1e-5)
    assert observed.dtype == jnp.float32
    np.testing.assert_array_equal(np.asarray(observed[~valid]), 0.0)


def test_source_push_semantic_backward_source_expand_pallas_interpret_matches_jax_reference():
    plan = _semantic_plan()
    dy = (jnp.arange(2 * 3 * 8, dtype=jnp.float32).reshape(2, 3, 8) / 16).astype(jnp.bfloat16)
    route_y = (jnp.arange(2 * 2 * 6 * 8, dtype=jnp.float32).reshape(2, 2, 6, 8) / 64).astype(jnp.float32)
    rows_per_expert_capacity = 4

    observed_dy_route, observed_dcombine = source_push_semantic_backward_source_expand_pallas_mgpu(
        dy,
        route_y,
        plan,
        rows_per_expert_capacity=rows_per_expert_capacity,
        block_sizes=SourcePushSemanticBackwardPallasBlockSizes(row_block=1, hidden_block=4),
        interpret=True,
    )
    expected_dy_route, expected_dcombine = source_push_semantic_backward_source_expand_expert_major_jax(
        dy,
        route_y,
        plan,
        rows_per_expert_capacity=rows_per_expert_capacity,
    )

    np.testing.assert_allclose(np.asarray(observed_dy_route), np.asarray(expected_dy_route), atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(np.asarray(observed_dcombine), np.asarray(expected_dcombine), atol=1e-4, rtol=1e-4)


def test_source_push_semantic_backward_source_expand_from_expert_major_pallas_matches_jax_reference():
    plan = _semantic_plan()
    dy = (jnp.arange(2 * 3 * 8, dtype=jnp.float32).reshape(2, 3, 8) / 16).astype(jnp.bfloat16)
    route_y_pair = (jnp.arange(2 * 2 * 6 * 8, dtype=jnp.float32).reshape(2, 2, 6, 8) / 64).astype(jnp.float32)
    route_y_expert, _valid = source_push_semantic_pair_to_expert_major_jax(
        route_y_pair,
        plan,
        rows_per_expert_capacity=4,
    )

    observed_dy_route, observed_dcombine = source_push_semantic_backward_source_expand_from_expert_major_pallas_mgpu(
        dy,
        route_y_expert,
        plan,
        block_sizes=SourcePushSemanticBackwardPallasBlockSizes(row_block=1, hidden_block=4),
        interpret=True,
    )
    expected_dy_route, expected_dcombine = source_push_semantic_backward_source_expand_from_expert_major_jax(
        dy,
        route_y_expert,
        plan,
    )

    np.testing.assert_allclose(np.asarray(observed_dy_route), np.asarray(expected_dy_route), atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(np.asarray(observed_dcombine), np.asarray(expected_dcombine), atol=1e-4, rtol=1e-4)


def test_source_push_semantic_backward_source_expand_owner_sharded_dcombine_interpret_matches_jax_reference():
    plan = _semantic_plan()
    mesh = _single_device_mesh()
    dy = (jnp.arange(2 * 3 * 8, dtype=jnp.float32).reshape(2, 3, 8) / 16).astype(jnp.bfloat16)
    route_y_pair = (jnp.arange(2 * 2 * 6 * 8, dtype=jnp.float32).reshape(2, 2, 6, 8) / 64).astype(jnp.float32)
    route_y_expert, _valid = source_push_semantic_pair_to_expert_major_jax(
        route_y_pair,
        plan,
        rows_per_expert_capacity=4,
    )

    observed_dy_route, observed_dcombine = (
        source_push_semantic_backward_source_expand_from_expert_major_owner_sharded_dcombine_pallas_mgpu(
            dy,
            route_y_expert,
            plan,
            block_sizes=SourcePushSemanticBackwardPallasBlockSizes(row_block=1, hidden_block=4),
            interpret=True,
            mesh=mesh,
        )
    )
    expected_dy_route, expected_dcombine = source_push_semantic_backward_source_expand_from_expert_major_jax(
        dy,
        route_y_expert,
        plan,
    )

    np.testing.assert_allclose(np.asarray(observed_dy_route), np.asarray(expected_dy_route), atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(np.asarray(observed_dcombine), np.asarray(expected_dcombine), atol=1e-4, rtol=1e-4)


def test_source_push_semantic_backward_source_expand_owner_sharded_dcombine_interpret_validates_shape():
    plan = _semantic_plan()
    mesh = _single_device_mesh()
    dy = (jnp.arange(2 * 3 * 8, dtype=jnp.float32).reshape(2, 3, 8) / 16).astype(jnp.bfloat16)
    route_y_pair = (jnp.arange(2 * 2 * 6 * 8, dtype=jnp.float32).reshape(2, 2, 6, 8) / 64).astype(jnp.float32)
    route_y_expert, _valid = source_push_semantic_pair_to_expert_major_jax(
        route_y_pair,
        plan,
        rows_per_expert_capacity=4,
    )

    with pytest.raises(ValueError, match="hidden dim 8 must be divisible by hidden_block=3"):
        source_push_semantic_backward_source_expand_from_expert_major_owner_sharded_dcombine_pallas_mgpu(
            dy,
            route_y_expert,
            plan,
            block_sizes=SourcePushSemanticBackwardPallasBlockSizes(row_block=1, hidden_block=3),
            interpret=True,
            mesh=mesh,
        )


def test_source_push_semantic_backward_dy_route_source_push_interpret_matches_jax_reference():
    plan = _semantic_plan()
    dy = (jnp.arange(2 * 3 * 8, dtype=jnp.float32).reshape(2, 3, 8) / 16).astype(jnp.bfloat16)

    observed = source_push_semantic_backward_dy_route_source_push_pallas_mgpu(
        dy,
        plan,
        rows_per_expert_capacity=4,
        block_sizes=SourcePushSemanticBackwardPallasBlockSizes(row_block=1, hidden_block=4),
        interpret=True,
    )
    expected = source_push_semantic_backward_dy_route_expert_major_jax(
        dy,
        plan,
        rows_per_expert_capacity=4,
    )

    np.testing.assert_allclose(np.asarray(observed), np.asarray(expected), atol=0, rtol=0)


def test_source_push_semantic_backward_source_expand_source_push_interpret_matches_jax_reference():
    plan = _semantic_plan()
    dy = (jnp.arange(2 * 3 * 8, dtype=jnp.float32).reshape(2, 3, 8) / 16).astype(jnp.bfloat16)
    route_y_pair = (jnp.arange(2 * 2 * 6 * 8, dtype=jnp.float32).reshape(2, 2, 6, 8) / 64).astype(jnp.float32)
    route_y_expert, _valid = source_push_semantic_pair_to_expert_major_jax(
        route_y_pair,
        plan,
        rows_per_expert_capacity=4,
    )

    observed_dy_route, observed_dcombine = (
        source_push_semantic_backward_source_expand_from_expert_major_source_push_pallas_mgpu(
            dy,
            route_y_expert,
            plan,
            block_sizes=SourcePushSemanticBackwardPallasBlockSizes(row_block=1, hidden_block=4),
            interpret=True,
        )
    )
    expected_dy_route, expected_dcombine = source_push_semantic_backward_source_expand_from_expert_major_jax(
        dy,
        route_y_expert,
        plan,
    )

    np.testing.assert_allclose(np.asarray(observed_dy_route), np.asarray(expected_dy_route), atol=0, rtol=0)
    np.testing.assert_allclose(np.asarray(observed_dcombine), np.asarray(expected_dcombine), atol=0, rtol=0)


def test_source_push_semantic_backward_dcombine_source_gather_interpret_matches_jax_reference():
    plan = _semantic_plan()
    dy = (jnp.arange(2 * 3 * 8, dtype=jnp.float32).reshape(2, 3, 8) / 16).astype(jnp.bfloat16)
    route_y_pair = (jnp.arange(2 * 2 * 6 * 8, dtype=jnp.float32).reshape(2, 2, 6, 8) / 64).astype(jnp.float32)
    route_y_expert, _valid = source_push_semantic_pair_to_expert_major_jax(
        route_y_pair,
        plan,
        rows_per_expert_capacity=4,
    )

    observed = source_push_semantic_backward_dcombine_source_gather_pallas_mgpu(
        dy,
        route_y_expert,
        plan,
        block_sizes=SourcePushSemanticBackwardPallasBlockSizes(row_block=1, hidden_block=4),
        interpret=True,
    )
    expected = source_push_semantic_backward_dcombine_source_gather_jax(dy, route_y_expert, plan)

    np.testing.assert_allclose(np.asarray(observed), np.asarray(expected), atol=1e-4, rtol=1e-4)


def test_source_push_semantic_backward_dcombine_source_gather_jax_accepts_named_sharding():
    plan = _semantic_plan()
    dy = (jnp.arange(2 * 3 * 8, dtype=jnp.float32).reshape(2, 3, 8) / 16).astype(jnp.bfloat16)
    route_y_pair = (jnp.arange(2 * 2 * 6 * 8, dtype=jnp.float32).reshape(2, 2, 6, 8) / 64).astype(jnp.float32)
    route_y_expert, _valid = source_push_semantic_pair_to_expert_major_jax(
        route_y_pair,
        plan,
        rows_per_expert_capacity=4,
    )
    mesh = _single_device_mesh()
    route_y_expert = jax.device_put(
        route_y_expert,
        jax.sharding.NamedSharding(mesh, P(SOURCE_PUSH_MESH_AXIS, None, None, None)),
    )

    observed = jax.jit(source_push_semantic_backward_dcombine_source_gather_jax)(dy, route_y_expert, plan)
    expected = source_push_semantic_backward_dcombine_source_gather_jax(dy, route_y_expert, plan)

    np.testing.assert_allclose(np.asarray(observed), np.asarray(expected), atol=1e-4, rtol=1e-4)


def test_source_push_semantic_backward_source_expand_source_gather_interpret_matches_jax_reference():
    plan = _semantic_plan()
    dy = (jnp.arange(2 * 3 * 8, dtype=jnp.float32).reshape(2, 3, 8) / 16).astype(jnp.bfloat16)
    route_y_pair = (jnp.arange(2 * 2 * 6 * 8, dtype=jnp.float32).reshape(2, 2, 6, 8) / 64).astype(jnp.float32)
    route_y_expert, _valid = source_push_semantic_pair_to_expert_major_jax(
        route_y_pair,
        plan,
        rows_per_expert_capacity=4,
    )

    observed_dy_route, observed_dcombine = (
        source_push_semantic_backward_source_expand_from_expert_major_source_gather_pallas_mgpu(
            dy,
            route_y_expert,
            plan,
            block_sizes=SourcePushSemanticBackwardPallasBlockSizes(row_block=1, hidden_block=4),
            interpret=True,
        )
    )
    expected_dy_route, expected_dcombine = source_push_semantic_backward_source_expand_from_expert_major_jax(
        dy,
        route_y_expert,
        plan,
    )

    np.testing.assert_allclose(np.asarray(observed_dy_route), np.asarray(expected_dy_route), atol=0, rtol=0)
    np.testing.assert_allclose(np.asarray(observed_dcombine), np.asarray(expected_dcombine), atol=1e-4, rtol=1e-4)


def test_source_push_semantic_dx_combine_pallas_interpret_matches_jax_reference():
    plan = _semantic_plan()
    dx_pair = (jnp.arange(2 * 2 * 6 * 8, dtype=jnp.float32).reshape(2, 2, 6, 8) / 32).astype(jnp.float32)
    rows_per_expert_capacity = 4
    dx_route, _valid = source_push_semantic_pair_to_expert_major_jax(
        dx_pair,
        plan,
        rows_per_expert_capacity=rows_per_expert_capacity,
    )

    observed = source_push_semantic_dx_combine_pallas_mgpu(
        dx_route,
        plan,
        block_sizes=SourcePushSemanticBackwardPallasBlockSizes(row_block=1, hidden_block=4),
        interpret=True,
    )
    expected = source_push_semantic_dx_combine_expert_major_jax(dx_route, plan)

    np.testing.assert_allclose(np.asarray(observed), np.asarray(expected), atol=1e-5, rtol=1e-5)


def test_source_push_semantic_dx_direct_return_queue_jit_matches_independent_reverse_route():
    plan = _semantic_plan()
    dx_pair = (jnp.arange(2 * 2 * 6 * 8, dtype=jnp.float32).reshape(2, 2, 6, 8) / 29).astype(jnp.float32)
    dx_route, _valid = source_push_semantic_pair_to_expert_major_jax(
        dx_pair,
        plan,
        rows_per_expert_capacity=4,
    )
    block_sizes = SourcePushSemanticDxReturnPallasBlockSizes(row_block=1, hidden_block=4)

    observed = jax.jit(
        lambda route: source_push_semantic_dx_return_direct_to_source_combine_pallas_mgpu(
            route,
            plan,
            block_sizes=block_sizes,
            output_dtype=jnp.float32,
            interpret=True,
        )
    )(dx_route)
    expected = _independent_dx_return_from_reverse_route(dx_route, plan, output_dtype=jnp.float32)

    return_dx = source_push_semantic_dx_return_direct_to_source_reference_jax(
        dx_route,
        plan,
        block_sizes=block_sizes,
    )
    metadata = source_push_semantic_queue_metadata_jax(
        plan,
        return_row_block=block_sizes.row_block,
        entries_per_dst=return_dx.shape[2],
    )
    observed_from_saved_queue = source_push_semantic_dx_combine_source_queue_pallas_mgpu(
        return_dx,
        plan,
        block_sizes=block_sizes,
        output_dtype=jnp.float32,
        interpret=True,
    )

    assert return_dx.dtype == jnp.bfloat16
    assert not bool(np.asarray(metadata.overflow_routes))
    np.testing.assert_allclose(np.asarray(observed), np.asarray(expected), atol=0, rtol=0)
    np.testing.assert_allclose(np.asarray(observed_from_saved_queue), np.asarray(expected), atol=0, rtol=0)


@pytest.mark.parametrize(
    ("row_block", "hidden_block"),
    ((257, 4), (1, 512)),
)
def test_source_push_semantic_dx_direct_return_rejects_async_copy_dimensions_over_256(row_block, hidden_block):
    plan = _semantic_plan()
    dx_route = jnp.zeros((2, 2, 4, 8), dtype=jnp.bfloat16)

    with pytest.raises(ValueError, match="exceeds the Mosaic async-copy limit 256"):
        source_push_semantic_dx_return_direct_to_source_pallas_mgpu(
            dx_route,
            plan,
            block_sizes=SourcePushSemanticDxReturnPallasBlockSizes(
                row_block=row_block,
                hidden_block=hidden_block,
            ),
            interpret=True,
        )


def test_source_push_semantic_dx_direct_return_queue_has_independent_linear_gradient():
    plan = _semantic_plan()
    dx_pair = (jnp.arange(2 * 2 * 6 * 8, dtype=jnp.float32).reshape(2, 2, 6, 8) / 31).astype(jnp.float32)
    dx_route, _valid = source_push_semantic_pair_to_expert_major_jax(
        dx_pair,
        plan,
        rows_per_expert_capacity=4,
    )
    cotangent = jnp.linspace(-0.75, 1.25, 2 * 3 * 8, dtype=jnp.float32).reshape(2, 3, 8)
    block_sizes = SourcePushSemanticDxReturnPallasBlockSizes(row_block=1, hidden_block=4)

    def queue_loss(route):
        dx = source_push_semantic_dx_return_direct_to_source_combine_pallas_mgpu(
            route,
            plan,
            block_sizes=block_sizes,
            output_dtype=jnp.float32,
            interpret=True,
        )
        return jnp.sum(dx * cotangent)

    def independent_loss(route):
        dx = _independent_dx_return_from_reverse_route(route, plan, output_dtype=jnp.float32)
        return jnp.sum(dx * cotangent)

    observed_grad = jax.grad(queue_loss)(dx_route)
    expected_grad = jax.grad(independent_loss)(dx_route)

    np.testing.assert_allclose(np.asarray(observed_grad), np.asarray(expected_grad), atol=0, rtol=0)


def test_source_padded_dy_route_jit_and_source_push_interpret_match_independent_reference():
    plan = _rough_semantic_plan()
    source_row_bases, rows_per_expert_capacity = _source_padded_row_bases(plan, row_block=4)
    dy = jnp.arange(2 * 5 * 8, dtype=jnp.float32).reshape(2, 5, 8) / 8
    expected = _independent_dy_route(
        dy,
        plan,
        source_row_bases,
        rows_per_expert_capacity=rows_per_expert_capacity,
    )

    observed_jit = jax.jit(
        lambda source_dy, row_bases: source_push_semantic_backward_dy_route_expert_major_jax(
            source_dy,
            plan,
            rows_per_expert_capacity=rows_per_expert_capacity,
            source_row_base_by_expert=row_bases,
        )
    )(dy, source_row_bases)
    observed_interpret = source_push_semantic_backward_dy_route_source_push_pallas_mgpu(
        dy,
        plan,
        rows_per_expert_capacity=rows_per_expert_capacity,
        source_row_base_by_expert=source_row_bases,
        block_sizes=SourcePushSemanticBackwardPallasBlockSizes(row_block=1, hidden_block=4),
        interpret=True,
    )

    assert not np.array_equal(np.asarray(source_row_bases), np.asarray(plan.src_base_by_expert))
    np.testing.assert_allclose(np.asarray(observed_jit), np.asarray(expected), atol=0, rtol=0)
    np.testing.assert_allclose(np.asarray(observed_interpret), np.asarray(expected), atol=0, rtol=0)


def test_source_padded_saved_return_source_expand_jit_and_interpret_match_independent_reference():
    plan = _rough_semantic_plan()
    source_row_bases, rows_per_expert_capacity = _source_padded_row_bases(plan, row_block=4)
    dy = jnp.arange(2 * 5 * 8, dtype=jnp.float32).reshape(2, 5, 8) / 13
    route_y_expert = _independent_expert_major_values(
        plan,
        source_row_bases,
        rows_per_expert_capacity=rows_per_expert_capacity,
        hidden=8,
    )
    queue_block_sizes = SourcePushSemanticDxReturnPallasBlockSizes(row_block=1, hidden_block=4)
    return_y = source_push_semantic_dx_return_direct_to_source_reference_jax(
        route_y_expert,
        plan,
        source_row_base_by_expert=source_row_bases,
        block_sizes=queue_block_sizes,
    )
    expected_dy_route = _independent_dy_route(
        dy,
        plan,
        source_row_bases,
        rows_per_expert_capacity=rows_per_expert_capacity,
    )
    expected_dcombine = _independent_dcombine(dy, route_y_expert, plan, source_row_bases)
    backward_block_sizes = SourcePushSemanticBackwardPallasBlockSizes(row_block=1, hidden_block=4)

    observed_jax = jax.jit(
        lambda source_dy, saved_return, row_bases: source_push_semantic_backward_source_expand_from_return_queue_jax(
            source_dy,
            saved_return,
            plan,
            rows_per_expert_capacity=rows_per_expert_capacity,
            source_row_base_by_expert=row_bases,
        )
    )(dy, return_y, source_row_bases)
    observed_pallas = source_push_semantic_backward_source_expand_from_return_queue_pallas_mgpu(
        dy,
        return_y,
        plan,
        rows_per_expert_capacity=rows_per_expert_capacity,
        source_row_base_by_expert=source_row_bases,
        block_sizes=backward_block_sizes,
        interpret=True,
    )

    for observed_dy_route, observed_dcombine in (observed_jax, observed_pallas):
        np.testing.assert_allclose(np.asarray(observed_dy_route), np.asarray(expected_dy_route), atol=0, rtol=0)
        np.testing.assert_allclose(np.asarray(observed_dcombine), np.asarray(expected_dcombine), atol=1e-4, rtol=1e-4)


def test_source_padded_direct_dx_return_and_combine_match_independent_reference():
    plan = _rough_semantic_plan()
    source_row_bases, rows_per_expert_capacity = _source_padded_row_bases(plan, row_block=4)
    dx_route = _independent_expert_major_values(
        plan,
        source_row_bases,
        rows_per_expert_capacity=rows_per_expert_capacity,
        hidden=8,
    )
    block_sizes = SourcePushSemanticDxReturnPallasBlockSizes(row_block=1, hidden_block=4)
    expected_dx = _independent_dx_combine(
        dx_route,
        plan,
        source_row_bases,
        output_dtype=jnp.float32,
    )

    reference_queue = jax.jit(
        lambda route, row_bases: source_push_semantic_dx_return_direct_to_source_reference_jax(
            route,
            plan,
            source_row_base_by_expert=row_bases,
            block_sizes=block_sizes,
        )
    )(dx_route, source_row_bases)
    observed_queue = source_push_semantic_dx_return_direct_to_source_pallas_mgpu(
        dx_route,
        plan,
        source_row_base_by_expert=source_row_bases,
        block_sizes=block_sizes,
        interpret=True,
    )
    observed_dx = source_push_semantic_dx_return_direct_to_source_combine_pallas_mgpu(
        dx_route,
        plan,
        source_row_base_by_expert=source_row_bases,
        block_sizes=block_sizes,
        output_dtype=jnp.float32,
        interpret=True,
    )

    np.testing.assert_allclose(np.asarray(observed_queue), np.asarray(reference_queue), atol=0, rtol=0)
    np.testing.assert_allclose(np.asarray(observed_dx), np.asarray(expected_dx), atol=0, rtol=0)


def test_source_padded_b64_saved_return_backward_consumes_exact_queue():
    plan, queue, layout = _source_padded_b64_plan_and_queue()
    route_y_expert = _independent_expert_major_values(
        plan,
        layout.src_base_by_expert,
        rows_per_expert_capacity=layout.rows_per_expert_capacity,
        hidden=8,
    )
    queue_block_sizes = SourcePushSemanticDxReturnPallasBlockSizes(row_block=64, hidden_block=4)
    return_y = source_push_semantic_dx_return_direct_to_source_reference_jax(
        route_y_expert,
        plan,
        source_row_base_by_expert=layout.src_base_by_expert,
        block_sizes=queue_block_sizes,
        queue=queue,
    )
    dy = (jnp.arange(2 * 64 * 8, dtype=jnp.float32).reshape(2, 64, 8) / 101).astype(jnp.bfloat16)
    backward_block_sizes = SourcePushSemanticBackwardPallasBlockSizes(row_block=64, hidden_block=4)
    expected_dy_route = _independent_dy_route(
        dy,
        plan,
        layout.src_base_by_expert,
        rows_per_expert_capacity=layout.rows_per_expert_capacity,
    )
    expected_dcombine = _independent_dcombine(dy, route_y_expert, plan, layout.src_base_by_expert)

    reference_dy_route, reference_dcombine = jax.jit(
        lambda source_dy, saved_return, row_bases, saved_queue: (
            source_push_semantic_backward_source_expand_from_return_queue_jax(
                source_dy,
                saved_return,
                plan,
                rows_per_expert_capacity=layout.rows_per_expert_capacity,
                source_row_base_by_expert=row_bases,
                queue=saved_queue,
            )
        )
    )(dy, return_y, layout.src_base_by_expert, queue)
    observed_dcombine = source_push_semantic_backward_dcombine_from_return_queue_pallas_mgpu(
        dy,
        return_y,
        plan,
        block_sizes=backward_block_sizes,
        queue=queue,
        interpret=True,
    )
    observed_dy_route, observed_expand_dcombine = (
        source_push_semantic_backward_source_expand_from_return_queue_pallas_mgpu(
            dy,
            return_y,
            plan,
            rows_per_expert_capacity=layout.rows_per_expert_capacity,
            source_row_base_by_expert=layout.src_base_by_expert,
            block_sizes=backward_block_sizes,
            queue=queue,
            interpret=True,
        )
    )

    assert queue.entries_per_dst == 4
    assert int(np.asarray(jnp.max(queue.required_entries_per_dst))) == 2
    assert not bool(np.asarray(queue.overflow_routes))
    assert not bool(np.asarray(layout.overflow_rows))
    assert not np.array_equal(np.asarray(layout.src_base_by_expert), np.asarray(plan.src_base_by_expert))
    np.testing.assert_allclose(np.asarray(reference_dy_route), np.asarray(expected_dy_route), atol=0, rtol=0)
    np.testing.assert_allclose(np.asarray(observed_dy_route), np.asarray(expected_dy_route), atol=0, rtol=0)
    np.testing.assert_allclose(np.asarray(reference_dcombine), np.asarray(expected_dcombine), atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(np.asarray(observed_dcombine), np.asarray(expected_dcombine), atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(
        np.asarray(observed_expand_dcombine),
        np.asarray(expected_dcombine),
        atol=1e-4,
        rtol=1e-4,
    )


def test_source_padded_b64_direct_dx_return_combine_consumes_exact_queue():
    plan, queue, layout = _source_padded_b64_plan_and_queue()
    dx_route = _independent_expert_major_values(
        plan,
        layout.src_base_by_expert,
        rows_per_expert_capacity=layout.rows_per_expert_capacity,
        hidden=8,
    )
    block_sizes = SourcePushSemanticDxReturnPallasBlockSizes(row_block=64, hidden_block=4)
    expected_dx = _independent_dx_combine(
        dx_route,
        plan,
        layout.src_base_by_expert,
        output_dtype=jnp.float32,
    )

    reference_dx = jax.jit(
        lambda route, row_bases, saved_queue: source_push_semantic_dx_return_direct_to_source_combine_reference_jax(
            route,
            plan,
            source_row_base_by_expert=row_bases,
            block_sizes=block_sizes,
            output_dtype=jnp.float32,
            queue=saved_queue,
        )
    )(dx_route, layout.src_base_by_expert, queue)
    return_dx = source_push_semantic_dx_return_direct_to_source_pallas_mgpu(
        dx_route,
        plan,
        source_row_base_by_expert=layout.src_base_by_expert,
        block_sizes=block_sizes,
        queue=queue,
        interpret=True,
    )
    combined_dx = source_push_semantic_dx_combine_source_queue_pallas_mgpu(
        return_dx,
        plan,
        block_sizes=block_sizes,
        output_dtype=jnp.float32,
        queue=queue,
        interpret=True,
    )
    composed_dx = source_push_semantic_dx_return_direct_to_source_combine_pallas_mgpu(
        dx_route,
        plan,
        source_row_base_by_expert=layout.src_base_by_expert,
        block_sizes=block_sizes,
        output_dtype=jnp.float32,
        queue=queue,
        interpret=True,
    )

    assert return_dx.shape == (2, 2, queue.entries_per_dst, 64, 8)
    np.testing.assert_array_equal(np.asarray(return_dx[:, :, 2:]), np.zeros((2, 2, 2, 64, 8)))
    np.testing.assert_allclose(np.asarray(reference_dx), np.asarray(expected_dx), atol=0, rtol=0)
    np.testing.assert_allclose(np.asarray(combined_dx), np.asarray(expected_dx), atol=0, rtol=0)
    np.testing.assert_allclose(np.asarray(composed_dx), np.asarray(expected_dx), atol=0, rtol=0)


def test_compact_direct_dx_return_reference_jit_accepts_named_sharding():
    plan = _semantic_plan()
    dx_pair = (jnp.arange(2 * 2 * 6 * 8, dtype=jnp.float32).reshape(2, 2, 6, 8) / 29).astype(jnp.float32)
    dx_route, _valid = source_push_semantic_pair_to_expert_major_jax(
        dx_pair,
        plan,
        rows_per_expert_capacity=4,
    )
    mesh = _single_device_mesh()
    dx_route_sharded = jax.device_put(
        dx_route,
        jax.sharding.NamedSharding(mesh, P(SOURCE_PUSH_MESH_AXIS, None, None, None)),
    )
    block_sizes = SourcePushSemanticDxReturnPallasBlockSizes(row_block=1, hidden_block=4)

    observed = jax.jit(
        lambda route: source_push_semantic_dx_return_direct_to_source_reference_jax(
            route,
            plan,
            block_sizes=block_sizes,
        )
    )(dx_route_sharded)
    expected = source_push_semantic_dx_return_direct_to_source_reference_jax(
        dx_route,
        plan,
        block_sizes=block_sizes,
    )

    np.testing.assert_allclose(np.asarray(observed), np.asarray(expected), atol=0, rtol=0)


def test_compact_direct_dx_return_combine_reference_jit_accepts_explicit_mesh():
    plan = _semantic_plan()
    dx_pair = (jnp.arange(2 * 2 * 6 * 8, dtype=jnp.float32).reshape(2, 2, 6, 8) / 29).astype(jnp.float32)
    dx_route, _valid = source_push_semantic_pair_to_expert_major_jax(
        dx_pair,
        plan,
        rows_per_expert_capacity=4,
    )
    block_sizes = SourcePushSemanticDxReturnPallasBlockSizes(row_block=1, hidden_block=4)
    mesh = _single_device_explicit_mesh()

    with jax.set_mesh(mesh):
        sharded_dx_route = jax.sharding.reshard(
            dx_route,
            P(SOURCE_PUSH_MESH_AXIS, None, None, None),
        )
        observed = jax.jit(
            lambda route: source_push_semantic_dx_return_direct_to_source_combine_reference_jax(
                route,
                plan,
                block_sizes=block_sizes,
                output_dtype=jnp.float32,
            )
        )(sharded_dx_route)

    expected = source_push_semantic_dx_return_direct_to_source_combine_reference_jax(
        dx_route,
        plan,
        block_sizes=block_sizes,
        output_dtype=jnp.float32,
    )
    np.testing.assert_allclose(np.asarray(observed), np.asarray(expected), atol=0, rtol=0)


def test_source_row_base_by_expert_validates_shape_and_dtype():
    plan = _rough_semantic_plan()
    dy = jnp.zeros((2, 5, 8), dtype=jnp.float32)

    with pytest.raises(ValueError, match="source_row_base_by_expert shape"):
        source_push_semantic_backward_dy_route_expert_major_jax(
            dy,
            plan,
            rows_per_expert_capacity=12,
            source_row_base_by_expert=jnp.zeros((2, 2, 1), dtype=jnp.int32),
        )
    with pytest.raises(ValueError, match="source_row_base_by_expert must have dtype int32"):
        source_push_semantic_backward_dy_route_expert_major_jax(
            dy,
            plan,
            rows_per_expert_capacity=12,
            source_row_base_by_expert=jnp.zeros((2, 2, 2), dtype=jnp.float32),
        )


def test_source_push_semantic_backward_source_expand_from_saved_return_queue_matches_expert_reference():
    plan = _semantic_plan()
    dy = (jnp.arange(2 * 3 * 8, dtype=jnp.float32).reshape(2, 3, 8) / 17).astype(jnp.bfloat16)
    route_y_pair = (jnp.arange(2 * 2 * 6 * 8, dtype=jnp.float32).reshape(2, 2, 6, 8) / 53).astype(jnp.float32)
    route_y_expert, _valid = source_push_semantic_pair_to_expert_major_jax(
        route_y_pair,
        plan,
        rows_per_expert_capacity=4,
    )
    queue_block_sizes = SourcePushSemanticDxReturnPallasBlockSizes(row_block=1, hidden_block=4)
    return_y = source_push_semantic_dx_return_direct_to_source_reference_jax(
        route_y_expert,
        plan,
        block_sizes=queue_block_sizes,
    )
    backward_block_sizes = SourcePushSemanticBackwardPallasBlockSizes(row_block=1, hidden_block=4)

    observed_dcombine = jax.jit(
        lambda source_dy, saved_return: source_push_semantic_backward_dcombine_from_return_queue_pallas_mgpu(
            source_dy,
            saved_return,
            plan,
            block_sizes=backward_block_sizes,
            interpret=True,
        )
    )(dy, return_y)
    observed_dy_route, observed_expand_dcombine = (
        source_push_semantic_backward_source_expand_from_return_queue_pallas_mgpu(
            dy,
            return_y,
            plan,
            rows_per_expert_capacity=4,
            block_sizes=backward_block_sizes,
            interpret=True,
        )
    )
    expected_dy_route, expected_dcombine = source_push_semantic_backward_source_expand_from_expert_major_jax(
        dy,
        route_y_expert.astype(jnp.bfloat16),
        plan,
    )
    reference_dy_route, reference_dcombine = source_push_semantic_backward_source_expand_from_return_queue_jax(
        dy,
        return_y,
        plan,
        rows_per_expert_capacity=4,
    )

    np.testing.assert_allclose(np.asarray(observed_dcombine), np.asarray(expected_dcombine), atol=0, rtol=0)
    np.testing.assert_allclose(np.asarray(observed_dy_route), np.asarray(expected_dy_route), atol=0, rtol=0)
    np.testing.assert_allclose(np.asarray(observed_expand_dcombine), np.asarray(expected_dcombine), atol=0, rtol=0)
    np.testing.assert_allclose(np.asarray(reference_dy_route), np.asarray(expected_dy_route), atol=0, rtol=0)
    np.testing.assert_allclose(np.asarray(reference_dcombine), np.asarray(expected_dcombine), atol=0, rtol=0)


def test_source_push_semantic_dx_return_source_gather_jax_matches_expert_major_reference():
    plan = _semantic_plan()
    dx_pair = (jnp.arange(2 * 2 * 6 * 8, dtype=jnp.float32).reshape(2, 2, 6, 8) / 32).astype(jnp.float32)
    dx_route, _valid = source_push_semantic_pair_to_expert_major_jax(
        dx_pair,
        plan,
        rows_per_expert_capacity=5,
    )

    observed = source_push_semantic_dx_return_source_gather_jax(dx_route, plan)
    expected, _expected_by_slot = source_push_semantic_dx_return_expert_major_jax(dx_route, plan)

    np.testing.assert_allclose(np.asarray(observed), np.asarray(expected), atol=1e-5, rtol=1e-5)


def test_source_push_semantic_dx_return_source_gather_jax_accepts_named_sharding():
    plan = _semantic_plan()
    dx_pair = (jnp.arange(2 * 2 * 6 * 8, dtype=jnp.float32).reshape(2, 2, 6, 8) / 32).astype(jnp.float32)
    dx_route, _valid = source_push_semantic_pair_to_expert_major_jax(
        dx_pair,
        plan,
        rows_per_expert_capacity=5,
    )
    mesh = _single_device_mesh()
    dx_route = jax.device_put(
        dx_route,
        jax.sharding.NamedSharding(mesh, P(SOURCE_PUSH_MESH_AXIS, None, None, None)),
    )

    observed = jax.jit(source_push_semantic_dx_return_source_gather_jax)(dx_route, plan)
    expected, _expected_by_slot = source_push_semantic_dx_return_expert_major_jax(dx_route, plan)

    np.testing.assert_allclose(np.asarray(observed), np.asarray(expected), atol=1e-5, rtol=1e-5)


def test_source_push_semantic_dx_return_source_gather_jax_accepts_explicit_mesh_sharding():
    plan = _semantic_plan()
    dx_pair = (jnp.arange(2 * 2 * 6 * 8, dtype=jnp.float32).reshape(2, 2, 6, 8) / 32).astype(jnp.float32)
    dx_route, _valid = source_push_semantic_pair_to_expert_major_jax(
        dx_pair,
        plan,
        rows_per_expert_capacity=5,
    )
    expected, _expected_by_slot = source_push_semantic_dx_return_expert_major_jax(dx_route, plan)
    mesh = _single_device_explicit_mesh()

    with jax.set_mesh(mesh):
        dx_route = jax.sharding.reshard(dx_route, P("expert", None, None, None))
        observed = jax.jit(source_push_semantic_dx_return_source_gather_jax)(dx_route, plan)

    np.testing.assert_allclose(np.asarray(observed), np.asarray(expected), atol=1e-5, rtol=1e-5)


def test_source_push_semantic_dx_return_source_gather_pallas_interpret_matches_jax_reference():
    plan = _semantic_plan()
    dx_pair = (jnp.arange(2 * 2 * 6 * 8, dtype=jnp.float32).reshape(2, 2, 6, 8) / 32).astype(jnp.float32)
    dx_route, _valid = source_push_semantic_pair_to_expert_major_jax(
        dx_pair,
        plan,
        rows_per_expert_capacity=5,
    )

    observed = source_push_semantic_dx_return_source_gather_pallas_mgpu(
        dx_route,
        plan,
        block_sizes=SourcePushSemanticBackwardPallasBlockSizes(row_block=2, hidden_block=4),
        interpret=True,
    )
    expected, _expected_by_slot = source_push_semantic_dx_return_expert_major_jax(dx_route, plan)

    np.testing.assert_allclose(np.asarray(observed), np.asarray(expected), atol=1e-5, rtol=1e-5)


def test_source_push_semantic_dx_return_remote_source_gather_pallas_interpret_matches_jax_reference():
    plan = _semantic_plan()
    dx_pair = (jnp.arange(2 * 2 * 6 * 8, dtype=jnp.float32).reshape(2, 2, 6, 8) / 32).astype(jnp.float32)
    dx_route, _valid = source_push_semantic_pair_to_expert_major_jax(
        dx_pair,
        plan,
        rows_per_expert_capacity=5,
    )

    observed = source_push_semantic_dx_return_remote_source_gather_pallas_mgpu(
        dx_route,
        plan,
        block_sizes=SourcePushSemanticBackwardPallasBlockSizes(row_block=1, hidden_block=4),
        interpret=True,
        mesh=_single_device_mesh(),
    )
    expected, _expected_by_slot = source_push_semantic_dx_return_expert_major_jax(dx_route, plan)

    np.testing.assert_allclose(np.asarray(observed), np.asarray(expected), atol=1e-5, rtol=1e-5)


def test_source_push_semantic_dx_return_pallas_interpret_matches_jax_reference():
    plan = _semantic_plan()
    dx_pair = (jnp.arange(2 * 2 * 6 * 8, dtype=jnp.float32).reshape(2, 2, 6, 8) / 32).astype(jnp.float32)
    dx_route, _valid = source_push_semantic_pair_to_expert_major_jax(
        dx_pair,
        plan,
        rows_per_expert_capacity=4,
    )

    observed_dx, observed_dx_by_slot = source_push_semantic_dx_return_pallas_mgpu(
        dx_route,
        plan,
        block_sizes=SourcePushSemanticBackwardPallasBlockSizes(row_block=1, hidden_block=4),
        interpret=True,
    )
    expected_dx, expected_dx_by_slot = source_push_semantic_dx_return_expert_major_jax(dx_route, plan)

    np.testing.assert_allclose(np.asarray(observed_dx), np.asarray(expected_dx), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(
        np.asarray(observed_dx_by_slot),
        np.asarray(expected_dx_by_slot),
        atol=1e-5,
        rtol=1e-5,
    )


def test_source_push_semantic_dx_return_sum_pallas_interpret_matches_jax_reference():
    plan = _semantic_plan()
    dx_pair = (jnp.arange(2 * 2 * 6 * 8, dtype=jnp.float32).reshape(2, 2, 6, 8) / 32).astype(jnp.float32)
    dx_route, _valid = source_push_semantic_pair_to_expert_major_jax(
        dx_pair,
        plan,
        rows_per_expert_capacity=4,
    )

    observed = source_push_semantic_dx_return_sum_pallas_mgpu(
        dx_route,
        plan,
        block_sizes=SourcePushSemanticBackwardPallasBlockSizes(row_block=1, hidden_block=4),
        interpret=True,
    )
    expected, _expected_by_slot = source_push_semantic_dx_return_expert_major_jax(dx_route, plan)

    np.testing.assert_allclose(np.asarray(observed), np.asarray(expected), atol=1e-5, rtol=1e-5)


def test_source_push_semantic_dx_return_slot_reduce_pallas_interpret_matches_jax_reference():
    plan = _semantic_plan()
    dx_pair = (jnp.arange(2 * 2 * 6 * 8, dtype=jnp.float32).reshape(2, 2, 6, 8) / 32).astype(jnp.float32)
    dx_route, _valid = source_push_semantic_pair_to_expert_major_jax(
        dx_pair,
        plan,
        rows_per_expert_capacity=4,
    )

    observed = source_push_semantic_dx_return_slot_reduce_pallas_mgpu(
        dx_route,
        plan,
        block_sizes=SourcePushSemanticBackwardPallasBlockSizes(row_block=1, hidden_block=4),
        interpret=True,
    )
    expected, _expected_by_slot = source_push_semantic_dx_return_expert_major_jax(dx_route, plan)

    np.testing.assert_allclose(np.asarray(observed), np.asarray(expected), atol=1e-5, rtol=1e-5)


def test_source_push_semantic_dx_return_copy_only_pallas_interpret_matches_jax_reference():
    plan = _semantic_plan()
    dx_pair = (jnp.arange(2 * 2 * 6 * 8, dtype=jnp.float32).reshape(2, 2, 6, 8) / 32).astype(jnp.float32)
    dx_route, _valid = source_push_semantic_pair_to_expert_major_jax(
        dx_pair,
        plan,
        rows_per_expert_capacity=4,
    )

    observed = source_push_semantic_dx_return_copy_only_pallas_mgpu(
        dx_route,
        plan,
        block_sizes=SourcePushSemanticBackwardPallasBlockSizes(row_block=1, hidden_block=4),
        interpret=True,
    )
    masked_pair = jnp.where(plan.valid_mask[..., None], dx_pair, jnp.zeros((), dtype=dx_pair.dtype))
    expected, _expected_valid = source_push_semantic_pair_to_expert_major_jax(
        masked_pair,
        plan,
        rows_per_expert_capacity=dx_route.shape[2],
    )

    np.testing.assert_allclose(np.asarray(observed), np.asarray(expected), atol=1e-5, rtol=1e-5)


def test_source_push_semantic_dx_return_sum_interpret_validates_pallas_shape_constraints():
    plan = _semantic_plan()
    dx_pair = (jnp.arange(2 * 2 * 6 * 8, dtype=jnp.float32).reshape(2, 2, 6, 8) / 32).astype(jnp.float32)
    dx_route, _valid = source_push_semantic_pair_to_expert_major_jax(
        dx_pair,
        plan,
        rows_per_expert_capacity=4,
    )

    with pytest.raises(ValueError, match="hidden dim 8 must be divisible by hidden_block=3"):
        source_push_semantic_dx_return_sum_pallas_mgpu(
            dx_route,
            plan,
            block_sizes=SourcePushSemanticBackwardPallasBlockSizes(row_block=1, hidden_block=3),
            interpret=True,
        )


def test_source_push_semantic_backward_pair_flat_references_remain_equivalent():
    plan = _semantic_plan()
    dy = (jnp.arange(2 * 3 * 8, dtype=jnp.float32).reshape(2, 3, 8) / 16).astype(jnp.float32)
    route_y = (jnp.arange(2 * 2 * 6 * 8, dtype=jnp.float32).reshape(2, 2, 6, 8) / 64).astype(jnp.float32)
    dx_pair = (jnp.arange(2 * 2 * 6 * 8, dtype=jnp.float32).reshape(2, 2, 6, 8) / 32).astype(jnp.float32)
    rows_per_expert_capacity = 4

    dy_pair, expected_dcombine = source_push_semantic_backward_source_expand_jax(dy, route_y, plan)
    dy_route, observed_dcombine = source_push_semantic_backward_source_expand_expert_major_jax(
        dy,
        route_y,
        plan,
        rows_per_expert_capacity=rows_per_expert_capacity,
    )
    expected_dy_route, _valid = source_push_semantic_pair_to_expert_major_jax(
        dy_pair,
        plan,
        rows_per_expert_capacity=rows_per_expert_capacity,
    )
    dx_route, _valid = source_push_semantic_pair_to_expert_major_jax(
        dx_pair,
        plan,
        rows_per_expert_capacity=rows_per_expert_capacity,
    )

    np.testing.assert_allclose(np.asarray(dy_route), np.asarray(expected_dy_route), atol=0, rtol=0)
    np.testing.assert_allclose(np.asarray(observed_dcombine), np.asarray(expected_dcombine), atol=0, rtol=0)
    np.testing.assert_allclose(
        np.asarray(source_push_semantic_dx_combine_expert_major_jax(dx_route, plan)),
        np.asarray(source_push_semantic_dx_combine_jax(dx_pair, plan)),
        atol=0,
        rtol=0,
    )
