# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh, PartitionSpec as P

from levanter.grug._moe.source_push_backward_w2 import SOURCE_PUSH_MESH_AXIS
from levanter.grug._moe.source_push_plan import (
    build_source_push_semantic_plan_jax,
    source_push_semantic_combine_jax,
    source_push_semantic_expert_major_to_pair_jax,
    source_push_semantic_gather_x_jax,
    source_push_semantic_pair_to_expert_major_jax,
    source_push_semantic_queue_metadata_jax,
    source_push_semantic_w2_reference_jax,
)
from levanter.grug._moe.source_push_semantic_inbox_pallas import source_push_semantic_inbox_kernel_inputs_jax
from levanter.grug._moe.source_push_semantic_w2_pallas import (
    SourcePushSemanticForwardReturnPallasBlockSizes,
    SourcePushSemanticW2ExpertMajorPallasBlockSizes,
    SourcePushSemanticW2PallasBlockSizes,
    _source_push_semantic_forward_return_expert_major_pallas_call,
    source_push_semantic_forward_combine_source_gather_pallas_mgpu,
    source_push_semantic_forward_combine_source_gather_reference_jax,
    source_push_semantic_forward_expert_major_direct_return_combine_pallas_mgpu,
    source_push_semantic_forward_return_direct_to_source_reference_jax,
    source_push_semantic_forward_return_copy_only_pallas_mgpu,
    source_push_semantic_forward_return_expert_major_pallas_mgpu,
    source_push_semantic_forward_return_expert_major_lookup_metadata_jax,
    source_push_semantic_forward_return_expert_major_reference_jax,
    source_push_semantic_forward_return_queue_metadata_jax,
    source_push_semantic_forward_return_remote_source_gather_pallas_mgpu,
    source_push_semantic_forward_return_source_gather_pallas_mgpu,
    source_push_semantic_forward_return_source_gather_reference_jax,
    source_push_semantic_forward_return_slot_reduce_owner_sharded_pallas_mgpu,
    source_push_semantic_forward_return_slot_reduce_pallas_mgpu,
    source_push_semantic_forward_return_sum_lookup_pallas_mgpu,
    source_push_semantic_forward_return_sum_pallas_mgpu,
    source_push_semantic_w2_expert_major_assume_zero_invalid_pallas_mgpu,
    source_push_semantic_w2_and_combine_pallas_scaffold_mgpu,
    source_push_semantic_w2_expert_major_pallas_mgpu,
    source_push_semantic_w2_expert_major_reference_jax,
    source_push_semantic_w2_pallas_mgpu,
)


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


def _w2_inputs():
    plan = _semantic_plan()
    h_pair = (jnp.arange(2 * 2 * 6 * 4, dtype=jnp.float32).reshape(2, 2, 6, 4) / 16).astype(jnp.bfloat16)
    w_down = (jnp.arange(2 * 2 * 4 * 8, dtype=jnp.float32).reshape(2, 2, 4, 8) / 64).astype(jnp.bfloat16)
    block_sizes = SourcePushSemanticW2PallasBlockSizes(row_block=2, intermediate_block=2, hidden_block=4)
    return plan, h_pair, w_down, block_sizes


def _single_device_mesh() -> Mesh:
    return Mesh(np.asarray(jax.devices()[:1]), (SOURCE_PUSH_MESH_AXIS,))


def _single_device_explicit_mesh() -> Mesh:
    return Mesh(
        np.asarray(jax.devices()[:1]),
        (SOURCE_PUSH_MESH_AXIS,),
        axis_types=(AxisType.Explicit,),
    )


def _rough_source_padded_direct_return_inputs():
    selected_experts = jnp.asarray(
        [
            [[0, 0], [0, 1], [0, 2], [1, 3], [0, 3], [2, 2]],
            [[0, 2], [2, 2], [2, 3], [2, 3], [1, 2], [0, 0]],
        ],
        dtype=jnp.int32,
    )
    route_weights = (jnp.arange(selected_experts.size, dtype=jnp.float32).reshape(selected_experts.shape) + 1) / 32
    plan = build_source_push_semantic_plan_jax(
        selected_experts,
        route_weights,
        ep_size=2,
        experts_per_rank=2,
        rows_per_src_dst_capacity=8,
        capacity_factor=4.0,
    )
    row_block = 2
    entries_per_dst = 4
    queue = source_push_semantic_queue_metadata_jax(
        plan,
        return_row_block=row_block,
        entries_per_dst=entries_per_dst,
    )
    rows_per_expert_capacity = 10
    x = ((jnp.arange(2 * 6 * 4, dtype=jnp.float32).reshape(2, 6, 4) + 1) / 16).astype(jnp.bfloat16)
    inbox_inputs = source_push_semantic_inbox_kernel_inputs_jax(
        x,
        plan,
        queue,
        rows_per_expert_capacity=rows_per_expert_capacity,
    )

    source_index = jnp.arange(2, dtype=jnp.int32)[:, None, None, None]
    dst_ordinal = jnp.arange(2, dtype=jnp.int32)[None, :, None, None]
    actual_dst = jnp.broadcast_to((source_index + dst_ordinal) % 2, inbox_inputs.packed_x.shape[:-1])
    queue_row = jnp.arange(row_block, dtype=jnp.int32)[None, None, None, :]
    flat_row = inbox_inputs.send_meta[..., 2, None] + queue_row
    row_valid = queue_row < inbox_inputs.send_meta[..., 3, None]
    scatter_row = jnp.where(row_valid, flat_row, 2 * rows_per_expert_capacity)
    h_flat = jnp.zeros((2, 2 * rows_per_expert_capacity, x.shape[-1]), dtype=x.dtype)
    h_flat = h_flat.at[actual_dst, scatter_row].set(
        jnp.where(row_valid[..., None], inbox_inputs.packed_x, jnp.zeros((), dtype=x.dtype)),
        mode="drop",
    )
    h_expert = h_flat.reshape(2, 2, rows_per_expert_capacity, x.shape[-1])
    w_down = ((jnp.arange(2 * 2 * 4 * 4, dtype=jnp.float32).reshape(2, 2, 4, 4) + 1) / 64).astype(jnp.bfloat16)
    return plan, x, h_expert, inbox_inputs, w_down, queue


def test_source_push_semantic_w2_pallas_interpret_matches_jax_reference():
    plan, h_pair, w_down, block_sizes = _w2_inputs()

    observed = source_push_semantic_w2_pallas_mgpu(
        h_pair,
        w_down,
        plan,
        block_sizes=block_sizes,
        interpret=True,
    )
    expected = source_push_semantic_w2_reference_jax(h_pair, w_down, plan)

    np.testing.assert_allclose(np.asarray(observed), np.asarray(expected), atol=1e-4, rtol=1e-4)


def test_source_push_semantic_w2_and_combine_scaffold_interpret_matches_jax_reference():
    plan, h_pair, w_down, block_sizes = _w2_inputs()

    observed_y, observed_route_y = source_push_semantic_w2_and_combine_pallas_scaffold_mgpu(
        h_pair,
        w_down,
        plan,
        block_sizes=block_sizes,
        interpret=True,
    )
    expected_route_y = source_push_semantic_w2_reference_jax(h_pair, w_down, plan)
    expected_y = source_push_semantic_combine_jax(expected_route_y, plan)

    np.testing.assert_allclose(np.asarray(observed_route_y), np.asarray(expected_route_y), atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(np.asarray(observed_y), np.asarray(expected_y), atol=1e-4, rtol=1e-4)


def test_source_push_semantic_w2_expert_major_reference_reconstructs_pair_rows():
    plan, h_pair, w_down, _block_sizes = _w2_inputs()
    h_expert, valid = source_push_semantic_pair_to_expert_major_jax(h_pair, plan, rows_per_expert_capacity=6)

    route_y_expert = source_push_semantic_w2_expert_major_reference_jax(h_expert, w_down, valid)
    observed_pair = source_push_semantic_expert_major_to_pair_jax(route_y_expert, plan)
    expected_pair = source_push_semantic_w2_reference_jax(h_pair, w_down, plan)

    np.testing.assert_allclose(np.asarray(observed_pair), np.asarray(expected_pair), atol=1e-4, rtol=1e-4)


def test_source_push_semantic_w2_expert_major_masks_invalid_rows_before_matmul():
    plan, h_pair, w_down, _block_sizes = _w2_inputs()
    h_expert, valid = source_push_semantic_pair_to_expert_major_jax(h_pair, plan, rows_per_expert_capacity=6)
    invalid_payload = jnp.full_like(h_expert, 128)
    h_with_invalid_payload = jnp.where(valid[..., None], h_expert, invalid_payload)

    observed = source_push_semantic_w2_expert_major_reference_jax(h_with_invalid_payload, w_down, valid)
    expected = source_push_semantic_w2_expert_major_reference_jax(h_expert, w_down, valid)

    np.testing.assert_allclose(np.asarray(observed), np.asarray(expected), atol=1e-4, rtol=1e-4)
    np.testing.assert_array_equal(np.asarray(observed)[np.asarray(~valid)], 0)


def test_source_push_semantic_w2_expert_major_interpret_matches_reference():
    plan, h_pair, w_down, _block_sizes = _w2_inputs()
    h_expert, valid = source_push_semantic_pair_to_expert_major_jax(h_pair, plan, rows_per_expert_capacity=6)
    block_sizes = SourcePushSemanticW2ExpertMajorPallasBlockSizes(
        row_block=64, intermediate_block=128, hidden_block=128
    )

    observed = source_push_semantic_w2_expert_major_pallas_mgpu(
        h_expert,
        w_down,
        valid,
        block_sizes=block_sizes,
        interpret=True,
    )
    expected = source_push_semantic_w2_expert_major_reference_jax(h_expert, w_down, valid)

    np.testing.assert_allclose(np.asarray(observed), np.asarray(expected), atol=1e-4, rtol=1e-4)


def test_source_push_semantic_w2_expert_major_assume_zero_invalid_matches_when_invalid_rows_are_zero():
    plan, h_pair, w_down, _block_sizes = _w2_inputs()
    h_expert, valid = source_push_semantic_pair_to_expert_major_jax(h_pair, plan, rows_per_expert_capacity=6)
    zero_invalid_h = jnp.where(valid[..., None], h_expert, jnp.zeros((), dtype=h_expert.dtype))
    block_sizes = SourcePushSemanticW2ExpertMajorPallasBlockSizes(
        row_block=64, intermediate_block=128, hidden_block=128
    )

    observed = source_push_semantic_w2_expert_major_assume_zero_invalid_pallas_mgpu(
        zero_invalid_h,
        w_down,
        block_sizes=block_sizes,
        interpret=True,
    )
    expected = source_push_semantic_w2_expert_major_reference_jax(zero_invalid_h, w_down, valid)

    np.testing.assert_allclose(np.asarray(observed), np.asarray(expected), atol=1e-4, rtol=1e-4)


def test_source_push_semantic_w2_expert_major_assume_zero_invalid_requires_zero_invalid_rows():
    plan, h_pair, w_down, _block_sizes = _w2_inputs()
    h_expert, valid = source_push_semantic_pair_to_expert_major_jax(h_pair, plan, rows_per_expert_capacity=6)
    invalid_payload = jnp.full_like(h_expert, 16)
    h_with_invalid_payload = jnp.where(valid[..., None], h_expert, invalid_payload)
    block_sizes = SourcePushSemanticW2ExpertMajorPallasBlockSizes(
        row_block=64, intermediate_block=128, hidden_block=128
    )

    observed = source_push_semantic_w2_expert_major_assume_zero_invalid_pallas_mgpu(
        h_with_invalid_payload,
        w_down,
        block_sizes=block_sizes,
        interpret=True,
    )
    expected = source_push_semantic_w2_expert_major_reference_jax(h_with_invalid_payload, w_down, valid)

    assert np.max(np.abs(np.asarray(observed - expected))) > 1.0


def test_source_push_semantic_w2_expert_major_default_blocks_match_target_tile():
    block_sizes = SourcePushSemanticW2ExpertMajorPallasBlockSizes.get_default()

    assert block_sizes == SourcePushSemanticW2ExpertMajorPallasBlockSizes(
        row_block=128,
        intermediate_block=64,
        hidden_block=128,
    )


def test_source_push_semantic_forward_return_expert_major_pallas_interpret_matches_reference():
    plan, h_pair, w_down, _block_sizes = _w2_inputs()
    route_y_pair = source_push_semantic_w2_reference_jax(h_pair, w_down, plan)
    route_y_expert, _valid = source_push_semantic_pair_to_expert_major_jax(
        route_y_pair,
        plan,
        rows_per_expert_capacity=6,
    )

    observed_y, observed_route_by_slot = source_push_semantic_forward_return_expert_major_pallas_mgpu(
        route_y_expert,
        plan,
        block_sizes=SourcePushSemanticForwardReturnPallasBlockSizes(row_block=1, hidden_block=4),
        interpret=True,
    )
    expected_y, expected_route_by_slot = source_push_semantic_forward_return_expert_major_reference_jax(
        route_y_expert,
        plan,
    )

    np.testing.assert_allclose(np.asarray(observed_y), np.asarray(expected_y), atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(
        np.asarray(observed_route_by_slot),
        np.asarray(expected_route_by_slot),
        atol=1e-4,
        rtol=1e-4,
    )


def test_source_push_semantic_forward_return_sum_pallas_interpret_matches_reference():
    plan, h_pair, w_down, _block_sizes = _w2_inputs()
    route_y_pair = source_push_semantic_w2_reference_jax(h_pair, w_down, plan)
    route_y_expert, _valid = source_push_semantic_pair_to_expert_major_jax(
        route_y_pair,
        plan,
        rows_per_expert_capacity=6,
    )

    observed_y = source_push_semantic_forward_return_sum_pallas_mgpu(
        route_y_expert,
        plan,
        block_sizes=SourcePushSemanticForwardReturnPallasBlockSizes(row_block=1, hidden_block=4),
        interpret=True,
    )
    expected_y, _expected_route_by_slot = source_push_semantic_forward_return_expert_major_reference_jax(
        route_y_expert,
        plan,
    )

    np.testing.assert_allclose(np.asarray(observed_y), np.asarray(expected_y), atol=1e-4, rtol=1e-4)


def test_source_push_semantic_forward_return_lookup_metadata_reconstructs_reference_sum():
    plan, h_pair, w_down, _block_sizes = _w2_inputs()
    route_y_pair = source_push_semantic_w2_reference_jax(h_pair, w_down, plan)
    route_y_expert, _valid = source_push_semantic_pair_to_expert_major_jax(
        route_y_pair,
        plan,
        rows_per_expert_capacity=6,
    )
    source_lookup, token_lookup, weight_lookup, valid_lookup = (
        source_push_semantic_forward_return_expert_major_lookup_metadata_jax(
            plan,
            rows_per_expert_capacity=route_y_expert.shape[2],
        )
    )
    source_index = jnp.where(valid_lookup, source_lookup, 0)
    token_index = jnp.where(valid_lookup, token_lookup, 0)
    weighted = route_y_expert.astype(jnp.float32) * weight_lookup.astype(jnp.float32)[..., None]
    weighted = jnp.where(valid_lookup[..., None], weighted, jnp.zeros((), dtype=weighted.dtype))

    observed_y = jnp.zeros(
        (plan.assignment_ids.shape[0], plan.tokens_per_source, route_y_expert.shape[-1]),
        dtype=jnp.float32,
    )
    observed_y = observed_y.at[source_index, token_index].add(weighted)
    expected_y, _expected_route_by_slot = source_push_semantic_forward_return_expert_major_reference_jax(
        route_y_expert,
        plan,
    )

    assert source_lookup.dtype == jnp.int32
    assert token_lookup.dtype == jnp.int32
    assert weight_lookup.dtype == plan.route_weights.dtype
    assert valid_lookup.dtype == jnp.bool_
    np.testing.assert_allclose(np.asarray(observed_y), np.asarray(expected_y), atol=1e-4, rtol=1e-4)


def test_source_push_semantic_forward_return_queue_metadata_reconstructs_reference_sum():
    plan, h_pair, w_down, _block_sizes = _w2_inputs()
    route_y_pair = source_push_semantic_w2_reference_jax(h_pair, w_down, plan)
    route_y_expert, _valid = source_push_semantic_pair_to_expert_major_jax(
        route_y_pair,
        plan,
        rows_per_expert_capacity=6,
    )
    return_block_sizes = SourcePushSemanticForwardReturnPallasBlockSizes(row_block=2, hidden_block=4)
    metadata = source_push_semantic_forward_return_queue_metadata_jax(
        plan,
        rows_per_expert_capacity=route_y_expert.shape[2],
        return_row_block=return_block_sizes.row_block,
    )
    return_y = source_push_semantic_forward_return_direct_to_source_reference_jax(
        route_y_expert,
        plan,
        block_sizes=return_block_sizes,
        route_buffer_dtype=jnp.bfloat16,
    )

    observed_y = source_push_semantic_forward_combine_source_gather_reference_jax(
        return_y,
        metadata,
        output_dtype=jnp.float32,
    )
    expected_y, _expected_route_by_slot = source_push_semantic_forward_return_expert_major_reference_jax(
        route_y_expert,
        plan,
    )

    assert metadata.queue_dst_ord.dtype == jnp.int32
    assert metadata.queue_entry.dtype == jnp.int32
    assert metadata.queue_row.dtype == jnp.int32
    assert metadata.route_weight.dtype == plan.route_weights.dtype
    assert metadata.route_valid.dtype == jnp.bool_
    np.testing.assert_allclose(np.asarray(observed_y), np.asarray(expected_y), atol=1e-1, rtol=1e-2)


def test_source_push_semantic_forward_combine_source_gather_reference_accepts_explicit_mesh():
    plan, h_pair, w_down, _block_sizes = _w2_inputs()
    route_y_pair = source_push_semantic_w2_reference_jax(h_pair, w_down, plan)
    route_y_expert, _valid = source_push_semantic_pair_to_expert_major_jax(
        route_y_pair,
        plan,
        rows_per_expert_capacity=6,
    )
    return_block_sizes = SourcePushSemanticForwardReturnPallasBlockSizes(row_block=2, hidden_block=4)
    metadata = source_push_semantic_forward_return_queue_metadata_jax(
        plan,
        rows_per_expert_capacity=route_y_expert.shape[2],
        return_row_block=return_block_sizes.row_block,
    )
    return_y = source_push_semantic_forward_return_direct_to_source_reference_jax(
        route_y_expert,
        plan,
        block_sizes=return_block_sizes,
        route_buffer_dtype=jnp.bfloat16,
    )
    mesh = _single_device_explicit_mesh()

    with jax.set_mesh(mesh):
        sharded_return_y = jax.sharding.reshard(
            return_y,
            P(SOURCE_PUSH_MESH_AXIS, None, None, None, None),
        )
        observed_y = jax.jit(source_push_semantic_forward_combine_source_gather_reference_jax)(
            sharded_return_y,
            metadata,
        )

    expected_y, _expected_route_by_slot = source_push_semantic_forward_return_expert_major_reference_jax(
        route_y_expert,
        plan,
    )
    np.testing.assert_allclose(np.asarray(observed_y), np.asarray(expected_y), atol=1e-1, rtol=1e-2)


def test_source_push_semantic_forward_direct_return_source_padded_reference_is_jittable():
    plan, x, h_expert, inbox_inputs, w_down, queue = _rough_source_padded_direct_return_inputs()
    layout = inbox_inputs.layout
    route_y_expert = source_push_semantic_w2_expert_major_reference_jax(h_expert, w_down, layout.valid)
    return_block_sizes = SourcePushSemanticForwardReturnPallasBlockSizes(row_block=2, hidden_block=4)

    metadata = jax.jit(
        lambda semantic_plan, source_row_bases, semantic_queue: source_push_semantic_forward_return_queue_metadata_jax(
            semantic_plan,
            rows_per_expert_capacity=h_expert.shape[2],
            return_row_block=return_block_sizes.row_block,
            source_row_base_by_expert=source_row_bases,
            queue=semantic_queue,
        )
    )(plan, layout.src_base_by_expert, queue)
    return_y = jax.jit(
        lambda route_y, semantic_plan, source_row_bases, semantic_queue: source_push_semantic_forward_return_direct_to_source_reference_jax(
            route_y,
            semantic_plan,
            block_sizes=return_block_sizes,
            route_buffer_dtype=jnp.float32,
            source_row_base_by_expert=source_row_bases,
            queue=semantic_queue,
        )
    )(route_y_expert, plan, layout.src_base_by_expert, queue)
    observed_y = source_push_semantic_forward_combine_source_gather_reference_jax(
        return_y,
        metadata,
        output_dtype=jnp.float32,
    )

    h_pair = source_push_semantic_gather_x_jax(x, plan)
    expected_route_y = source_push_semantic_w2_reference_jax(h_pair, w_down, plan)
    expected_y = source_push_semantic_combine_jax(expected_route_y, plan)
    recv_expert = jnp.maximum(metadata.recv_local_expert, 0)
    dst_index = jnp.arange(2, dtype=jnp.int32)[:, None, None]
    recv_expert_base = layout.expert_base.at[dst_index, recv_expert].get()
    expected_recv_row_start = inbox_inputs.recv_meta[..., 2] - recv_expert_base
    expected_recv_row_start = jnp.where(metadata.recv_valid_rows > 0, expected_recv_row_start, 0)
    conservative_metadata = source_push_semantic_forward_return_queue_metadata_jax(
        plan,
        rows_per_expert_capacity=h_expert.shape[2],
        return_row_block=return_block_sizes.row_block,
        source_row_base_by_expert=layout.src_base_by_expert,
    )

    assert not np.array_equal(np.asarray(layout.src_base_by_expert), np.asarray(plan.src_base_by_expert))
    assert int(queue.overflow_entries) == 0
    assert int(queue.overflow_routes) == 0
    assert metadata.queue_local_expert.shape[2] == queue.entries_per_dst == 4
    assert conservative_metadata.queue_local_expert.shape[2] == 5
    assert return_y.shape[2] == queue.entries_per_dst
    np.testing.assert_array_equal(np.asarray(metadata.recv_expert_row_start), np.asarray(expected_recv_row_start))
    np.testing.assert_allclose(np.asarray(observed_y), np.asarray(expected_y), atol=1e-4, rtol=1e-4)


def test_source_push_semantic_forward_direct_return_combine_source_padded_interpret_matches_semantic_reference():
    plan, x, h_expert, inbox_inputs, w_down, queue = _rough_source_padded_direct_return_inputs()
    layout = inbox_inputs.layout
    return_block_sizes = SourcePushSemanticForwardReturnPallasBlockSizes(row_block=2, hidden_block=4)

    observed_y = source_push_semantic_forward_expert_major_direct_return_combine_pallas_mgpu(
        h_expert,
        w_down,
        layout.valid,
        plan,
        w2_block_sizes=SourcePushSemanticW2ExpertMajorPallasBlockSizes(
            row_block=2,
            intermediate_block=2,
            hidden_block=4,
        ),
        return_block_sizes=return_block_sizes,
        route_buffer_dtype=jnp.float32,
        output_dtype=jnp.float32,
        source_row_base_by_expert=layout.src_base_by_expert,
        queue=queue,
        interpret=True,
    )

    h_pair = source_push_semantic_gather_x_jax(x, plan)
    expected_route_y = source_push_semantic_w2_reference_jax(h_pair, w_down, plan)
    expected_y = source_push_semantic_combine_jax(expected_route_y, plan)

    np.testing.assert_allclose(np.asarray(observed_y), np.asarray(expected_y), atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize(
    "source_row_bases",
    (
        jnp.zeros((2, 2, 3), dtype=jnp.int32),
        jnp.zeros((2, 2, 2), dtype=jnp.int16),
    ),
)
def test_source_push_semantic_forward_return_queue_metadata_rejects_invalid_source_row_bases(source_row_bases):
    plan = _semantic_plan()

    with pytest.raises(ValueError, match="source_row_base_by_expert"):
        source_push_semantic_forward_return_queue_metadata_jax(
            plan,
            rows_per_expert_capacity=6,
            return_row_block=2,
            source_row_base_by_expert=source_row_bases,
        )


def test_source_push_semantic_forward_return_queue_metadata_rejects_mismatched_queue():
    plan, _x, h_expert, _inbox_inputs, _w_down, queue = _rough_source_padded_direct_return_inputs()

    with pytest.raises(ValueError, match="queue local_expert shape"):
        source_push_semantic_forward_return_queue_metadata_jax(
            plan,
            rows_per_expert_capacity=h_expert.shape[2],
            return_row_block=2,
            queue=replace(queue, local_expert=queue.local_expert[:1]),
        )
    with pytest.raises(ValueError, match="queue return_row_block"):
        source_push_semantic_forward_return_queue_metadata_jax(
            plan,
            rows_per_expert_capacity=h_expert.shape[2],
            return_row_block=2,
            queue=replace(queue, return_row_block=1),
        )


def test_source_push_semantic_forward_combine_source_gather_pallas_interpret_matches_reference():
    plan, h_pair, w_down, _block_sizes = _w2_inputs()
    route_y_pair = source_push_semantic_w2_reference_jax(h_pair, w_down, plan)
    route_y_expert, _valid = source_push_semantic_pair_to_expert_major_jax(
        route_y_pair,
        plan,
        rows_per_expert_capacity=6,
    )
    return_block_sizes = SourcePushSemanticForwardReturnPallasBlockSizes(row_block=1, hidden_block=4)
    metadata = source_push_semantic_forward_return_queue_metadata_jax(
        plan,
        rows_per_expert_capacity=route_y_expert.shape[2],
        return_row_block=return_block_sizes.row_block,
    )
    return_y = source_push_semantic_forward_return_direct_to_source_reference_jax(
        route_y_expert,
        plan,
        block_sizes=return_block_sizes,
        route_buffer_dtype=jnp.bfloat16,
    )

    observed_y = source_push_semantic_forward_combine_source_gather_pallas_mgpu(
        return_y,
        plan,
        block_sizes=return_block_sizes,
        output_dtype=jnp.float32,
        interpret=True,
    )
    expected_y = source_push_semantic_forward_combine_source_gather_reference_jax(
        return_y,
        metadata,
        output_dtype=jnp.float32,
    )

    np.testing.assert_allclose(np.asarray(observed_y), np.asarray(expected_y), atol=1e-4, rtol=1e-4)


def test_source_push_semantic_forward_direct_return_combine_interpret_matches_reference():
    plan, h_pair, w_down, _block_sizes = _w2_inputs()
    h_expert, valid = source_push_semantic_pair_to_expert_major_jax(h_pair, plan, rows_per_expert_capacity=64)
    return_block_sizes = SourcePushSemanticForwardReturnPallasBlockSizes(row_block=1, hidden_block=4)
    w2_block_sizes = SourcePushSemanticW2ExpertMajorPallasBlockSizes(
        row_block=64,
        intermediate_block=128,
        hidden_block=128,
    )

    observed_y = source_push_semantic_forward_expert_major_direct_return_combine_pallas_mgpu(
        h_expert,
        w_down,
        valid,
        plan,
        w2_block_sizes=w2_block_sizes,
        return_block_sizes=return_block_sizes,
        output_dtype=jnp.float32,
        interpret=True,
    )
    route_y_expert = source_push_semantic_w2_expert_major_reference_jax(h_expert, w_down, valid)
    expected_y, _expected_route_by_slot = source_push_semantic_forward_return_expert_major_reference_jax(
        route_y_expert,
        plan,
    )

    np.testing.assert_allclose(np.asarray(observed_y), np.asarray(expected_y), atol=1e-1, rtol=1e-2)


def test_source_push_semantic_forward_return_sum_lookup_pallas_interpret_matches_reference():
    plan, h_pair, w_down, _block_sizes = _w2_inputs()
    route_y_pair = source_push_semantic_w2_reference_jax(h_pair, w_down, plan)
    route_y_expert, _valid = source_push_semantic_pair_to_expert_major_jax(
        route_y_pair,
        plan,
        rows_per_expert_capacity=6,
    )

    observed_y = source_push_semantic_forward_return_sum_lookup_pallas_mgpu(
        route_y_expert,
        plan,
        block_sizes=SourcePushSemanticForwardReturnPallasBlockSizes(row_block=1, hidden_block=4),
        interpret=True,
    )
    expected_y, _expected_route_by_slot = source_push_semantic_forward_return_expert_major_reference_jax(
        route_y_expert,
        plan,
    )

    np.testing.assert_allclose(np.asarray(observed_y), np.asarray(expected_y), atol=1e-4, rtol=1e-4)


def test_source_push_semantic_forward_return_source_gather_reference_matches_slot_reference():
    plan, h_pair, w_down, _block_sizes = _w2_inputs()
    route_y_pair = source_push_semantic_w2_reference_jax(h_pair, w_down, plan)
    route_y_expert, _valid = source_push_semantic_pair_to_expert_major_jax(
        route_y_pair,
        plan,
        rows_per_expert_capacity=6,
    )

    observed_y = source_push_semantic_forward_return_source_gather_reference_jax(route_y_expert, plan)
    expected_y, _expected_route_by_slot = source_push_semantic_forward_return_expert_major_reference_jax(
        route_y_expert,
        plan,
    )

    np.testing.assert_allclose(np.asarray(observed_y), np.asarray(expected_y), atol=1e-4, rtol=1e-4)


def test_source_push_semantic_forward_return_source_gather_reference_is_jittable():
    plan, h_pair, w_down, _block_sizes = _w2_inputs()
    route_y_pair = source_push_semantic_w2_reference_jax(h_pair, w_down, plan)
    route_y_expert, _valid = source_push_semantic_pair_to_expert_major_jax(
        route_y_pair,
        plan,
        rows_per_expert_capacity=6,
    )

    observed_y = jax.jit(source_push_semantic_forward_return_source_gather_reference_jax)(route_y_expert, plan)
    expected_y, _expected_route_by_slot = source_push_semantic_forward_return_expert_major_reference_jax(
        route_y_expert,
        plan,
    )

    np.testing.assert_allclose(np.asarray(observed_y), np.asarray(expected_y), atol=1e-4, rtol=1e-4)


def test_source_push_semantic_forward_return_source_gather_reference_accepts_named_sharding():
    plan, h_pair, w_down, _block_sizes = _w2_inputs()
    route_y_pair = source_push_semantic_w2_reference_jax(h_pair, w_down, plan)
    route_y_expert, _valid = source_push_semantic_pair_to_expert_major_jax(
        route_y_pair,
        plan,
        rows_per_expert_capacity=6,
    )
    mesh = _single_device_mesh()
    route_y_expert = jax.device_put(
        route_y_expert,
        jax.sharding.NamedSharding(mesh, P(SOURCE_PUSH_MESH_AXIS, None, None, None)),
    )

    observed_y = jax.jit(source_push_semantic_forward_return_source_gather_reference_jax)(route_y_expert, plan)
    expected_y, _expected_route_by_slot = source_push_semantic_forward_return_expert_major_reference_jax(
        route_y_expert,
        plan,
    )

    np.testing.assert_allclose(np.asarray(observed_y), np.asarray(expected_y), atol=1e-4, rtol=1e-4)


def test_source_push_semantic_forward_return_source_gather_pallas_interpret_matches_reference():
    plan, h_pair, w_down, _block_sizes = _w2_inputs()
    route_y_pair = source_push_semantic_w2_reference_jax(h_pair, w_down, plan)
    route_y_expert, _valid = source_push_semantic_pair_to_expert_major_jax(
        route_y_pair,
        plan,
        rows_per_expert_capacity=6,
    )

    observed_y = source_push_semantic_forward_return_source_gather_pallas_mgpu(
        route_y_expert,
        plan,
        block_sizes=SourcePushSemanticForwardReturnPallasBlockSizes(row_block=1, hidden_block=4),
        interpret=True,
    )
    expected_y = source_push_semantic_forward_return_source_gather_reference_jax(route_y_expert, plan)

    np.testing.assert_allclose(np.asarray(observed_y), np.asarray(expected_y), atol=1e-4, rtol=1e-4)


def test_source_push_semantic_forward_return_remote_source_gather_interpret_matches_reference():
    plan, h_pair, w_down, _block_sizes = _w2_inputs()
    route_y_pair = source_push_semantic_w2_reference_jax(h_pair, w_down, plan)
    route_y_expert, _valid = source_push_semantic_pair_to_expert_major_jax(
        route_y_pair,
        plan,
        rows_per_expert_capacity=6,
    )

    observed_y = source_push_semantic_forward_return_remote_source_gather_pallas_mgpu(
        route_y_expert,
        plan,
        block_sizes=SourcePushSemanticForwardReturnPallasBlockSizes(row_block=1, hidden_block=4),
        interpret=True,
        mesh=None,
    )
    expected_y = source_push_semantic_forward_return_source_gather_reference_jax(route_y_expert, plan)

    np.testing.assert_allclose(np.asarray(observed_y), np.asarray(expected_y), atol=1e-4, rtol=1e-4)


def test_source_push_semantic_forward_return_slot_reduce_pallas_interpret_matches_reference():
    plan, h_pair, w_down, _block_sizes = _w2_inputs()
    route_y_pair = source_push_semantic_w2_reference_jax(h_pair, w_down, plan)
    route_y_expert, _valid = source_push_semantic_pair_to_expert_major_jax(
        route_y_pair,
        plan,
        rows_per_expert_capacity=6,
    )

    observed_y = source_push_semantic_forward_return_slot_reduce_pallas_mgpu(
        route_y_expert,
        plan,
        block_sizes=SourcePushSemanticForwardReturnPallasBlockSizes(row_block=1, hidden_block=4),
        interpret=True,
    )
    expected_y, _expected_route_by_slot = source_push_semantic_forward_return_expert_major_reference_jax(
        route_y_expert,
        plan,
    )

    np.testing.assert_allclose(np.asarray(observed_y), np.asarray(expected_y), atol=1e-4, rtol=1e-4)


def test_source_push_semantic_forward_return_slot_reduce_owner_sharded_interpret_matches_reference():
    plan, h_pair, w_down, _block_sizes = _w2_inputs()
    route_y_pair = source_push_semantic_w2_reference_jax(h_pair, w_down, plan)
    route_y_expert, _valid = source_push_semantic_pair_to_expert_major_jax(
        route_y_pair,
        plan,
        rows_per_expert_capacity=6,
    )

    observed_y = source_push_semantic_forward_return_slot_reduce_owner_sharded_pallas_mgpu(
        route_y_expert,
        plan,
        block_sizes=SourcePushSemanticForwardReturnPallasBlockSizes(row_block=1, hidden_block=4),
        interpret=True,
    )
    expected_y, _expected_route_by_slot = source_push_semantic_forward_return_expert_major_reference_jax(
        route_y_expert,
        plan,
    )

    np.testing.assert_allclose(np.asarray(observed_y), np.asarray(expected_y), atol=1e-4, rtol=1e-4)


def test_source_push_semantic_forward_return_copy_only_pallas_interpret_matches_reference():
    plan, h_pair, w_down, _block_sizes = _w2_inputs()
    route_y_pair = source_push_semantic_w2_reference_jax(h_pair, w_down, plan)
    route_y_expert, _valid = source_push_semantic_pair_to_expert_major_jax(
        route_y_pair,
        plan,
        rows_per_expert_capacity=6,
    )

    observed = source_push_semantic_forward_return_copy_only_pallas_mgpu(
        route_y_expert,
        plan,
        block_sizes=SourcePushSemanticForwardReturnPallasBlockSizes(row_block=1, hidden_block=4),
        interpret=True,
    )
    weighted_pair = route_y_pair * plan.route_weights[..., None].astype(jnp.float32)
    weighted_pair = jnp.where(plan.valid_mask[..., None], weighted_pair, jnp.zeros((), dtype=weighted_pair.dtype))
    expected, _expected_valid = source_push_semantic_pair_to_expert_major_jax(
        weighted_pair,
        plan,
        rows_per_expert_capacity=route_y_expert.shape[2],
    )

    np.testing.assert_allclose(np.asarray(observed), np.asarray(expected), atol=1e-4, rtol=1e-4)


def test_source_push_semantic_forward_return_local_pallas_uses_local_destination_grid():
    plan, h_pair, w_down, _block_sizes = _w2_inputs()
    route_y_pair = source_push_semantic_w2_reference_jax(h_pair, w_down, plan)
    route_y_expert, _valid = source_push_semantic_pair_to_expert_major_jax(
        route_y_pair,
        plan,
        rows_per_expert_capacity=6,
    )
    destination = 1

    observed_route_by_slot = _source_push_semantic_forward_return_expert_major_pallas_call(
        route_y_expert[destination : destination + 1],
        plan.token_ids,
        plan.route_slots,
        plan.route_weights,
        plan.xcounts,
        plan.pair_expert_base,
        plan.src_base_by_expert,
        jnp.asarray(destination, dtype=jnp.int32),
        tokens_per_source=plan.tokens_per_source,
        topk=plan.topk,
        row_block=1,
        hidden_block=4,
        interpret=True,
    )
    destination_only_route_y = jnp.zeros_like(route_y_expert).at[destination].set(route_y_expert[destination])
    _expected_y, expected_route_by_slot = source_push_semantic_forward_return_expert_major_reference_jax(
        destination_only_route_y,
        plan,
    )

    np.testing.assert_allclose(
        np.asarray(observed_route_by_slot),
        np.asarray(expected_route_by_slot),
        atol=1e-4,
        rtol=1e-4,
    )


def test_source_push_semantic_forward_return_expert_major_sharded_interpret_matches_reference():
    selected_experts = jnp.asarray([[[0, 0], [0, 0], [0, 0]]], dtype=jnp.int32)
    combine_weights = jnp.asarray([[[0.7, 0.3], [0.6, 0.4], [0.9, 0.1]]], dtype=jnp.float32)
    plan = build_source_push_semantic_plan_jax(
        selected_experts,
        combine_weights,
        ep_size=1,
        experts_per_rank=1,
        rows_per_src_dst_capacity=8,
        capacity_factor=2.0,
    )
    h_pair = (jnp.arange(1 * 1 * 8 * 4, dtype=jnp.float32).reshape(1, 1, 8, 4) / 16).astype(jnp.bfloat16)
    w_down = (jnp.arange(1 * 1 * 4 * 8, dtype=jnp.float32).reshape(1, 1, 4, 8) / 64).astype(jnp.bfloat16)
    route_y_pair = source_push_semantic_w2_reference_jax(h_pair, w_down, plan)
    route_y_expert, _valid = source_push_semantic_pair_to_expert_major_jax(
        route_y_pair,
        plan,
        rows_per_expert_capacity=8,
    )
    mesh = Mesh(np.asarray(jax.devices()[:1]), (SOURCE_PUSH_MESH_AXIS,))

    observed_y, observed_route_by_slot = source_push_semantic_forward_return_expert_major_pallas_mgpu(
        route_y_expert,
        plan,
        block_sizes=SourcePushSemanticForwardReturnPallasBlockSizes(row_block=1, hidden_block=4),
        interpret=True,
        mesh=mesh,
    )
    expected_y, expected_route_by_slot = source_push_semantic_forward_return_expert_major_reference_jax(
        route_y_expert,
        plan,
    )

    np.testing.assert_allclose(np.asarray(observed_y), np.asarray(expected_y), atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(
        np.asarray(observed_route_by_slot),
        np.asarray(expected_route_by_slot),
        atol=1e-4,
        rtol=1e-4,
    )
