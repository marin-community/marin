# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

import levanter.grug._moe.source_push_mlp as source_push_mlp
from levanter.grug._moe.source_push_backward_w2 import (
    SOURCE_PUSH_W2_BACKWARD_IMPLEMENTATION_REFERENCE_MATMUL_PALLAS_MGPU_SWIGLU,
    SOURCE_PUSH_W2_MATMUL_BACKWARD_IMPLEMENTATION_PALLAS_MGPU,
    SOURCE_PUSH_W2_MATMUL_BACKWARD_IMPLEMENTATION_REFERENCE,
    SOURCE_PUSH_W2_SWIGLU_BACKWARD_IMPLEMENTATION_PALLAS_MGPU,
    SOURCE_PUSH_W2_SWIGLU_BACKWARD_IMPLEMENTATION_REFERENCE,
    SourcePushW2MatmulBackwardPallasBlockSizes,
    SourcePushW2SwiGLUBackwardPallasBlockSizes,
    _dst_indices,
    _expert_flat_rows,
    _source_push_w2_activation_and_weighted_activation_reference,
    _source_push_w2_backward_expert_blocks,
    _source_push_w2_backward_for_expert_block_reference,
    _source_push_w2_backward_expert_blocks_reference,
    _source_push_w2_backward_from_flat_h,
    _source_push_w2_backward_from_flat_h_reference,
    _source_push_w2_matmul_backward,
    _source_push_w2_matmul_backward_reference,
    _source_push_w2_swiglu_backward_pallas_mgpu,
    _source_push_w2_swiglu_backward_reference,
)
from levanter.grug._moe.source_push_plan import SOURCE_PUSH_MESH_AXIS


DST = 2
EXPERTS = 2
ROWS_PER_EXPERT = 3
INTERMEDIATE_DIM = 2
HIDDEN_DIM = 3


def _w2_backward_inputs(dtype=jnp.float32):
    h = jnp.linspace(
        -0.7,
        0.8,
        DST * EXPERTS * ROWS_PER_EXPERT * 2 * INTERMEDIATE_DIM,
        dtype=jnp.float32,
    ).reshape(DST, EXPERTS, ROWS_PER_EXPERT, 2 * INTERMEDIATE_DIM)
    route_weight = jnp.linspace(
        0.1,
        0.9,
        DST * EXPERTS * ROWS_PER_EXPERT,
        dtype=jnp.float32,
    ).reshape(DST, EXPERTS, ROWS_PER_EXPERT)
    dy = jnp.linspace(
        -0.5,
        0.6,
        DST * EXPERTS * ROWS_PER_EXPERT * HIDDEN_DIM,
        dtype=jnp.float32,
    ).reshape(DST, EXPERTS, ROWS_PER_EXPERT, HIDDEN_DIM)
    w2 = jnp.linspace(
        -0.4,
        0.5,
        DST * EXPERTS * INTERMEDIATE_DIM * HIDDEN_DIM,
        dtype=jnp.float32,
    ).reshape(DST, EXPERTS, INTERMEDIATE_DIM, HIDDEN_DIM)
    valid = jnp.array(
        [
            [[True, False, True], [True, True, False]],
            [[False, True, True], [True, False, True]],
        ],
        dtype=jnp.bool_,
    )
    h = jnp.where(valid[..., None], h, jnp.asarray(31.0, dtype=h.dtype))
    route_weight = jnp.where(valid, route_weight, jnp.asarray(-17.0, dtype=route_weight.dtype))
    dy = jnp.where(valid[..., None], dy, jnp.asarray(23.0, dtype=dy.dtype))
    return h.astype(dtype), route_weight.astype(dtype), dy.astype(dtype), w2.astype(dtype), valid


def _flat_w2_backward_inputs():
    h_blocks, route_blocks, dy_blocks, w2, valid = _w2_backward_inputs()
    expert_base = jnp.array([[0, 5], [1, 6]], dtype=jnp.int32)
    flat_rows = _expert_flat_rows(expert_base, ROWS_PER_EXPERT)
    dst_index = _dst_indices(DST, EXPERTS, ROWS_PER_EXPERT)
    h = jnp.full((DST, 9, 2 * INTERMEDIATE_DIM), 101.0, dtype=h_blocks.dtype)
    route_weight = jnp.full((DST, 9), -101.0, dtype=route_blocks.dtype)
    dy = jnp.full((DST, 9, HIDDEN_DIM), 101.0, dtype=dy_blocks.dtype)
    h = h.at[dst_index, flat_rows].set(h_blocks)
    route_weight = route_weight.at[dst_index, flat_rows].set(route_blocks)
    dy = dy.at[dst_index, flat_rows].set(dy_blocks)
    return h_blocks, route_blocks, dy_blocks, w2, valid, expert_base, flat_rows, dst_index, h, route_weight, dy


@pytest.mark.parametrize("dtype,atol", [(jnp.float32, 1e-6), (jnp.bfloat16, 2e-2)])
def test_source_push_w2_backward_expert_blocks_matches_existing_helper(dtype, atol):
    h, route_weight, dy, w2, valid = _w2_backward_inputs(dtype)

    observed = _source_push_w2_backward_expert_blocks_reference(h, route_weight, dy, w2, valid)

    for expert in range(EXPERTS):
        expected = source_push_mlp._source_push_mlp_w2_swiglu_backward_for_expert(
            h[:, expert],
            route_weight[:, expert],
            dy[:, expert].astype(jnp.float32),
            w2[:, expert].astype(jnp.float32),
            valid[:, expert].astype(jnp.float32),
        )
        np.testing.assert_allclose(
            np.asarray(observed.d_h[:, expert]),
            np.asarray(expected.d_h_block),
            atol=atol,
            rtol=atol,
        )
        np.testing.assert_allclose(
            np.asarray(observed.d_route_weight[:, expert]),
            np.asarray(expected.d_route_block),
            atol=atol,
            rtol=atol,
        )
        np.testing.assert_allclose(
            np.asarray(observed.dw2[:, expert]),
            np.asarray(expected.dw2_block),
            atol=atol,
            rtol=atol,
        )
        direct_d_h, direct_d_route, direct_dw2 = _source_push_w2_backward_for_expert_block_reference(
            h[:, expert],
            route_weight[:, expert],
            dy[:, expert],
            w2[:, expert],
            valid[:, expert],
        )
        np.testing.assert_allclose(
            np.asarray(direct_d_h),
            np.asarray(expected.d_h_block),
            atol=atol,
            rtol=atol,
        )
        np.testing.assert_allclose(
            np.asarray(direct_d_route),
            np.asarray(expected.d_route_block),
            atol=atol,
            rtol=atol,
        )
        np.testing.assert_allclose(
            np.asarray(direct_dw2),
            np.asarray(expected.dw2_block),
            atol=atol,
            rtol=atol,
        )

    invalid = np.logical_not(np.asarray(valid))
    np.testing.assert_array_equal(
        np.asarray(observed.d_h)[invalid],
        np.zeros((invalid.sum(), 2 * INTERMEDIATE_DIM)),
    )
    np.testing.assert_array_equal(np.asarray(observed.d_route_weight)[invalid], np.zeros(invalid.sum()))


def test_source_push_w2_backward_default_boundary_matches_reference():
    h, route_weight, dy, w2, valid = _w2_backward_inputs()

    expected = _source_push_w2_backward_expert_blocks_reference(h, route_weight, dy, w2, valid)
    observed = _source_push_w2_backward_expert_blocks(h, route_weight, dy, w2, valid)

    np.testing.assert_allclose(np.asarray(observed.d_h), np.asarray(expected.d_h), atol=0, rtol=0)
    np.testing.assert_allclose(
        np.asarray(observed.d_route_weight),
        np.asarray(expected.d_route_weight),
        atol=0,
        rtol=0,
    )
    np.testing.assert_allclose(np.asarray(observed.dw2), np.asarray(expected.dw2), atol=0, rtol=0)


def test_source_push_w2_swiglu_backward_pallas_interpret_matches_reference():
    h, route_weight, dy, w2, valid = _w2_backward_inputs()
    _activation, weighted_activation = _source_push_w2_activation_and_weighted_activation_reference(
        h,
        route_weight,
        valid,
    )
    matmul_output = _source_push_w2_matmul_backward_reference(weighted_activation, dy, w2, valid)

    expected = _source_push_w2_swiglu_backward_reference(
        h,
        route_weight,
        matmul_output.d_weighted_activation,
        valid,
    )
    observed = _source_push_w2_swiglu_backward_pallas_mgpu(
        h,
        route_weight,
        matmul_output.d_weighted_activation,
        valid,
        block_sizes=SourcePushW2SwiGLUBackwardPallasBlockSizes(row_block=1),
        interpret=True,
    )

    np.testing.assert_allclose(np.asarray(observed.d_h), np.asarray(expected.d_h), atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(
        np.asarray(observed.d_route_weight),
        np.asarray(expected.d_route_weight),
        atol=1e-6,
        rtol=1e-6,
    )


def test_source_push_w2_backward_from_flat_h_matches_expert_blocks():
    h_blocks, route_blocks, dy_blocks, w2, valid, expert_base, flat_rows, dst_index, h, route_weight, dy = (
        _flat_w2_backward_inputs()
    )

    expected_blocks = _source_push_w2_backward_expert_blocks_reference(
        h_blocks,
        route_blocks,
        dy_blocks,
        w2,
        valid,
    )
    expected_d_h = jnp.zeros(h.shape, dtype=expected_blocks.d_h.dtype)
    expected_d_route = jnp.zeros(route_weight.shape, dtype=expected_blocks.d_route_weight.dtype)
    expected_d_h = expected_d_h.at[dst_index, flat_rows].add(expected_blocks.d_h * valid[..., None])
    expected_d_route = expected_d_route.at[dst_index, flat_rows].add(expected_blocks.d_route_weight * valid)

    observed = _source_push_w2_backward_from_flat_h_reference(
        expert_base,
        h,
        route_weight,
        dy,
        w2,
        valid,
    )

    np.testing.assert_allclose(
        np.asarray(observed.d_h),
        np.asarray(expected_d_h),
        atol=1e-6,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(observed.d_route_weight),
        np.asarray(expected_d_route),
        atol=1e-6,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(observed.dw2),
        np.asarray(expected_blocks.dw2),
        atol=1e-6,
        rtol=1e-6,
    )


def test_source_push_w2_backward_from_flat_h_contiguous_gather_matches_indexed_path():
    _h_blocks, _route_blocks, _dy_blocks, w2, valid, expert_base, _flat_rows, _dst_index, h, route_weight, dy = (
        _flat_w2_backward_inputs()
    )

    def indexed_path(expert_base_value, h_value, route_weight_value, dy_value, w2_value, valid_value):
        return _source_push_w2_backward_from_flat_h(
            expert_base_value,
            h_value,
            route_weight_value,
            dy_value,
            w2_value,
            valid_value,
            contiguous_expert_gather=False,
        )

    def contiguous_path(expert_base_value, h_value, route_weight_value, dy_value, w2_value, valid_value):
        return _source_push_w2_backward_from_flat_h(
            expert_base_value,
            h_value,
            route_weight_value,
            dy_value,
            w2_value,
            valid_value,
            contiguous_expert_gather=True,
        )

    expected = jax.jit(indexed_path)(expert_base, h, route_weight, dy, w2, valid)
    observed = jax.jit(contiguous_path)(expert_base, h, route_weight, dy, w2, valid)

    np.testing.assert_allclose(np.asarray(observed.d_h), np.asarray(expected.d_h), atol=0, rtol=0)
    np.testing.assert_allclose(
        np.asarray(observed.d_route_weight),
        np.asarray(expected.d_route_weight),
        atol=0,
        rtol=0,
    )
    np.testing.assert_allclose(np.asarray(observed.dw2), np.asarray(expected.dw2), atol=0, rtol=0)


def test_source_push_w2_backward_contiguous_gather_accepts_dst_sharded_inputs():
    if len(jax.local_devices()) < DST:
        pytest.skip("requires enough local devices to shard the destination axis")

    _h_blocks, _route_blocks, _dy_blocks, w2, valid, expert_base, _flat_rows, _dst_index, h, route_weight, dy = (
        _flat_w2_backward_inputs()
    )
    mesh = Mesh(np.asarray(jax.local_devices()[:DST]), (SOURCE_PUSH_MESH_AXIS,))
    h = jax.device_put(h, NamedSharding(mesh, P(SOURCE_PUSH_MESH_AXIS, None, None)))
    route_weight = jax.device_put(route_weight, NamedSharding(mesh, P(SOURCE_PUSH_MESH_AXIS, None)))
    dy = jax.device_put(dy, NamedSharding(mesh, P(SOURCE_PUSH_MESH_AXIS, None, None)))
    w2 = jax.device_put(w2, NamedSharding(mesh, P(SOURCE_PUSH_MESH_AXIS, None, None, None)))
    valid = jax.device_put(valid, NamedSharding(mesh, P(SOURCE_PUSH_MESH_AXIS, None, None)))
    expert_base = jax.device_put(expert_base, NamedSharding(mesh, P(SOURCE_PUSH_MESH_AXIS, None)))

    expected = _source_push_w2_backward_from_flat_h(
        expert_base,
        h,
        route_weight,
        dy,
        w2,
        valid,
        contiguous_expert_gather=False,
    )
    observed = _source_push_w2_backward_from_flat_h(
        expert_base,
        h,
        route_weight,
        dy,
        w2,
        valid,
        contiguous_expert_gather=True,
    )

    np.testing.assert_allclose(np.asarray(observed.d_h), np.asarray(expected.d_h), atol=0, rtol=0)
    np.testing.assert_allclose(
        np.asarray(observed.d_route_weight),
        np.asarray(expected.d_route_weight),
        atol=0,
        rtol=0,
    )
    np.testing.assert_allclose(np.asarray(observed.dw2), np.asarray(expected.dw2), atol=0, rtol=0)


def test_source_push_w2_backward_contiguous_gather_jits_dst_sharded_inputs():
    if len(jax.local_devices()) < DST:
        pytest.skip("requires enough local devices to shard the destination axis")

    _h_blocks, _route_blocks, _dy_blocks, w2, valid, expert_base, _flat_rows, _dst_index, h, route_weight, dy = (
        _flat_w2_backward_inputs()
    )
    mesh = Mesh(np.asarray(jax.local_devices()[:DST]), (SOURCE_PUSH_MESH_AXIS,))
    h = jax.device_put(h, NamedSharding(mesh, P(SOURCE_PUSH_MESH_AXIS, None, None)))
    route_weight = jax.device_put(route_weight, NamedSharding(mesh, P(SOURCE_PUSH_MESH_AXIS, None)))
    dy = jax.device_put(dy, NamedSharding(mesh, P(SOURCE_PUSH_MESH_AXIS, None, None)))
    w2 = jax.device_put(w2, NamedSharding(mesh, P(SOURCE_PUSH_MESH_AXIS, None, None, None)))
    valid = jax.device_put(valid, NamedSharding(mesh, P(SOURCE_PUSH_MESH_AXIS, None, None)))
    expert_base = jax.device_put(expert_base, NamedSharding(mesh, P(SOURCE_PUSH_MESH_AXIS, None)))

    def indexed_path(expert_base_value, h_value, route_weight_value, dy_value, w2_value, valid_value):
        return _source_push_w2_backward_from_flat_h(
            expert_base_value,
            h_value,
            route_weight_value,
            dy_value,
            w2_value,
            valid_value,
            contiguous_expert_gather=False,
        )

    def contiguous_path(expert_base_value, h_value, route_weight_value, dy_value, w2_value, valid_value):
        return _source_push_w2_backward_from_flat_h(
            expert_base_value,
            h_value,
            route_weight_value,
            dy_value,
            w2_value,
            valid_value,
            contiguous_expert_gather=True,
        )

    expected = jax.jit(indexed_path)(expert_base, h, route_weight, dy, w2, valid)
    observed = jax.jit(contiguous_path)(expert_base, h, route_weight, dy, w2, valid)

    np.testing.assert_allclose(np.asarray(observed.d_h), np.asarray(expected.d_h), atol=0, rtol=0)
    np.testing.assert_allclose(
        np.asarray(observed.d_route_weight),
        np.asarray(expected.d_route_weight),
        atol=0,
        rtol=0,
    )
    np.testing.assert_allclose(np.asarray(observed.dw2), np.asarray(expected.dw2), atol=0, rtol=0)


def test_source_push_w2_backward_partial_pallas_boundary_matches_reference_from_flat_h():
    _h_blocks, _route_blocks, _dy_blocks, w2, valid, expert_base, _flat_rows, _dst_index, h, route_weight, dy = (
        _flat_w2_backward_inputs()
    )

    expected = _source_push_w2_backward_from_flat_h_reference(
        expert_base,
        h,
        route_weight,
        dy,
        w2,
        valid,
    )
    observed = _source_push_w2_backward_from_flat_h(
        expert_base,
        h,
        route_weight,
        dy,
        w2,
        valid,
        implementation=SOURCE_PUSH_W2_BACKWARD_IMPLEMENTATION_REFERENCE_MATMUL_PALLAS_MGPU_SWIGLU,
        block_sizes=SourcePushW2SwiGLUBackwardPallasBlockSizes(row_block=1),
        interpret=True,
    )

    np.testing.assert_allclose(np.asarray(observed.d_h), np.asarray(expected.d_h), atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(
        np.asarray(observed.d_route_weight),
        np.asarray(expected.d_route_weight),
        atol=1e-6,
        rtol=1e-6,
    )
    np.testing.assert_allclose(np.asarray(observed.dw2), np.asarray(expected.dw2), atol=1e-6, rtol=1e-6)


def test_source_push_w2_backward_from_flat_h_with_pallas_matmul_matches_reference():
    _h_blocks, _route_blocks, _dy_blocks, w2, valid, expert_base, _flat_rows, _dst_index, h, route_weight, dy = (
        _flat_w2_backward_inputs()
    )

    expected = _source_push_w2_backward_from_flat_h_reference(
        expert_base,
        h,
        route_weight,
        dy,
        w2,
        valid,
    )
    observed = _source_push_w2_backward_from_flat_h(
        expert_base,
        h,
        route_weight,
        dy,
        w2,
        valid,
        matmul_implementation=SOURCE_PUSH_W2_MATMUL_BACKWARD_IMPLEMENTATION_PALLAS_MGPU,
        swiglu_implementation=SOURCE_PUSH_W2_SWIGLU_BACKWARD_IMPLEMENTATION_REFERENCE,
        interpret=True,
    )

    np.testing.assert_allclose(np.asarray(observed.d_h), np.asarray(expected.d_h), atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(
        np.asarray(observed.d_route_weight),
        np.asarray(expected.d_route_weight),
        atol=1e-6,
        rtol=1e-6,
    )
    np.testing.assert_allclose(np.asarray(observed.dw2), np.asarray(expected.dw2), atol=1e-6, rtol=1e-6)


def test_source_push_w2_backward_stage_selectors_match_combined_implementations():
    h, route_weight, dy, w2, valid = _w2_backward_inputs()

    expected_reference = _source_push_w2_backward_expert_blocks_reference(h, route_weight, dy, w2, valid)
    observed_reference = _source_push_w2_backward_expert_blocks(
        h,
        route_weight,
        dy,
        w2,
        valid,
        matmul_implementation=SOURCE_PUSH_W2_MATMUL_BACKWARD_IMPLEMENTATION_REFERENCE,
        swiglu_implementation=SOURCE_PUSH_W2_SWIGLU_BACKWARD_IMPLEMENTATION_REFERENCE,
    )

    np.testing.assert_allclose(np.asarray(observed_reference.d_h), np.asarray(expected_reference.d_h), atol=0, rtol=0)
    np.testing.assert_allclose(
        np.asarray(observed_reference.d_route_weight),
        np.asarray(expected_reference.d_route_weight),
        atol=0,
        rtol=0,
    )
    np.testing.assert_allclose(np.asarray(observed_reference.dw2), np.asarray(expected_reference.dw2), atol=0, rtol=0)

    expected_partial = _source_push_w2_backward_expert_blocks(
        h,
        route_weight,
        dy,
        w2,
        valid,
        implementation=SOURCE_PUSH_W2_BACKWARD_IMPLEMENTATION_REFERENCE_MATMUL_PALLAS_MGPU_SWIGLU,
        block_sizes=SourcePushW2SwiGLUBackwardPallasBlockSizes(row_block=1),
        interpret=True,
    )
    observed_partial = _source_push_w2_backward_expert_blocks(
        h,
        route_weight,
        dy,
        w2,
        valid,
        matmul_implementation=SOURCE_PUSH_W2_MATMUL_BACKWARD_IMPLEMENTATION_REFERENCE,
        swiglu_implementation=SOURCE_PUSH_W2_SWIGLU_BACKWARD_IMPLEMENTATION_PALLAS_MGPU,
        block_sizes=SourcePushW2SwiGLUBackwardPallasBlockSizes(row_block=1),
        interpret=True,
    )

    np.testing.assert_allclose(np.asarray(observed_partial.d_h), np.asarray(expected_partial.d_h), atol=0, rtol=0)
    np.testing.assert_allclose(
        np.asarray(observed_partial.d_route_weight),
        np.asarray(expected_partial.d_route_weight),
        atol=0,
        rtol=0,
    )
    np.testing.assert_allclose(np.asarray(observed_partial.dw2), np.asarray(expected_partial.dw2), atol=0, rtol=0)


def test_source_push_w2_backward_stage_selectors_reject_ambiguous_combined_implementation():
    h, route_weight, dy, w2, valid = _w2_backward_inputs()

    with pytest.raises(ValueError, match="stage-specific W2 backward selectors"):
        _source_push_w2_backward_expert_blocks(
            h,
            route_weight,
            dy,
            w2,
            valid,
            implementation=SOURCE_PUSH_W2_BACKWARD_IMPLEMENTATION_REFERENCE_MATMUL_PALLAS_MGPU_SWIGLU,
            matmul_implementation=SOURCE_PUSH_W2_MATMUL_BACKWARD_IMPLEMENTATION_REFERENCE,
        )


def test_source_push_w2_matmul_pallas_mgpu_interpret_matches_reference():
    h, route_weight, dy, w2, valid = _w2_backward_inputs()
    _activation, weighted_activation = _source_push_w2_activation_and_weighted_activation_reference(
        h,
        route_weight,
        valid,
    )

    expected = _source_push_w2_matmul_backward_reference(weighted_activation, dy, w2, valid)
    observed = _source_push_w2_matmul_backward(
        weighted_activation,
        dy,
        w2,
        valid,
        implementation=SOURCE_PUSH_W2_MATMUL_BACKWARD_IMPLEMENTATION_PALLAS_MGPU,
        block_sizes=SourcePushW2MatmulBackwardPallasBlockSizes(
            row_block=1,
            intermediate_block=1,
            hidden_block=1,
        ),
        interpret=True,
    )

    np.testing.assert_allclose(
        np.asarray(observed.d_weighted_activation),
        np.asarray(expected.d_weighted_activation),
        atol=1e-6,
        rtol=1e-6,
    )
    np.testing.assert_allclose(np.asarray(observed.dw2), np.asarray(expected.dw2), atol=1e-6, rtol=1e-6)
