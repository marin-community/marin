# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh

from levanter.grug._moe.source_push_plan import (
    SOURCE_PUSH_MESH_AXIS,
    build_source_push_semantic_plan_jax,
    source_push_semantic_gather_x_jax,
    source_push_semantic_pair_to_expert_major_jax,
    source_push_semantic_w13_backward_reference_jax,
)
from levanter.grug._moe.source_push_semantic_backward_w13_pallas import (
    SourcePushSemanticW13BackwardPallasBlockSizes,
    source_push_semantic_w13_backward_dx_route_expert_major_pallas_mgpu,
    source_push_semantic_w13_backward_dw13_expert_major_pallas_mgpu,
    source_push_semantic_w13_backward_expert_major_pallas_mgpu,
    source_push_semantic_w13_backward_expert_major_reference_jax,
    source_push_semantic_w13_backward_dw13_pallas_mgpu,
    source_push_semantic_w13_backward_dx_pair_pallas_mgpu,
    source_push_semantic_w13_backward_pallas_scaffold_mgpu,
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


def _w13_backward_inputs():
    plan = _semantic_plan()
    x = (jnp.arange(2 * 3 * 8, dtype=jnp.float32).reshape(2, 3, 8) / 32 + 0.1).astype(jnp.bfloat16)
    dz_pair = (jnp.arange(2 * 2 * 6 * 4, dtype=jnp.float32).reshape(2, 2, 6, 4) / 64 - 0.5).astype(jnp.float32)
    w_gate_up = (jnp.arange(2 * 2 * 8 * 4, dtype=jnp.float32).reshape(2, 2, 8, 4) / 128 + 0.2).astype(jnp.bfloat16)
    block_sizes = SourcePushSemanticW13BackwardPallasBlockSizes(row_block=2, hidden_block=4, output_block=2)
    return plan, x, dz_pair, w_gate_up, block_sizes


def test_source_push_semantic_w13_backward_dx_pair_pallas_interpret_matches_jax_reference():
    plan, x, dz_pair, w_gate_up, block_sizes = _w13_backward_inputs()

    observed = source_push_semantic_w13_backward_dx_pair_pallas_mgpu(
        dz_pair,
        w_gate_up,
        plan,
        block_sizes=block_sizes,
        interpret=True,
    )
    expected, _expected_dw13 = source_push_semantic_w13_backward_reference_jax(x, dz_pair, w_gate_up, plan)

    np.testing.assert_allclose(np.asarray(observed), np.asarray(expected), atol=1e-4, rtol=1e-4)


def test_source_push_semantic_w13_backward_expert_major_reference_matches_pair_reference():
    plan, x, dz_pair, w_gate_up, _block_sizes = _w13_backward_inputs()
    rows_per_expert_capacity = 8
    x_pair = source_push_semantic_gather_x_jax(x, plan)
    x_expert, valid = source_push_semantic_pair_to_expert_major_jax(
        x_pair,
        plan,
        rows_per_expert_capacity=rows_per_expert_capacity,
    )
    dz13, _dz_valid = source_push_semantic_pair_to_expert_major_jax(
        dz_pair,
        plan,
        rows_per_expert_capacity=rows_per_expert_capacity,
    )

    observed_dx_route, observed_dw13 = source_push_semantic_w13_backward_expert_major_reference_jax(
        x_expert,
        dz13,
        w_gate_up,
        valid,
    )
    expected_dx_pair, expected_dw13 = source_push_semantic_w13_backward_reference_jax(x, dz_pair, w_gate_up, plan)
    expected_dx_route, _expected_valid = source_push_semantic_pair_to_expert_major_jax(
        expected_dx_pair,
        plan,
        rows_per_expert_capacity=rows_per_expert_capacity,
    )

    np.testing.assert_allclose(np.asarray(observed_dx_route), np.asarray(expected_dx_route), atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(np.asarray(observed_dw13), np.asarray(expected_dw13), atol=1e-4, rtol=1e-4)


def test_source_push_semantic_w13_backward_expert_major_interpret_masks_invalid_rows():
    _plan, _x, _dz_pair, w_gate_up, block_sizes = _w13_backward_inputs()
    valid = jnp.asarray(
        [
            [[True, False, True], [False, True, False]],
            [[True, True, False], [False, False, True]],
        ],
        dtype=jnp.bool_,
    )
    x_expert = jnp.arange(2 * 2 * 3 * 8, dtype=jnp.float32).reshape(2, 2, 3, 8).astype(jnp.bfloat16)
    dz13 = (jnp.arange(2 * 2 * 3 * 4, dtype=jnp.float32).reshape(2, 2, 3, 4) / 17.0).astype(jnp.bfloat16)
    dirty_x = jnp.where(valid[..., None], x_expert, jnp.full_like(x_expert, 1.0e4))
    dirty_dz13 = jnp.where(valid[..., None], dz13, jnp.full_like(dz13, -1.0e4))

    expected_dx, expected_dw13 = source_push_semantic_w13_backward_expert_major_reference_jax(
        x_expert,
        dz13,
        w_gate_up,
        valid,
    )
    observed_dx = source_push_semantic_w13_backward_dx_route_expert_major_pallas_mgpu(
        dirty_dz13,
        w_gate_up,
        valid,
        block_sizes=block_sizes,
        interpret=True,
    )
    observed_dw13 = source_push_semantic_w13_backward_dw13_expert_major_pallas_mgpu(
        dirty_x,
        dirty_dz13,
        valid,
        block_sizes=block_sizes,
        interpret=True,
    )

    np.testing.assert_allclose(np.asarray(observed_dx), np.asarray(expected_dx), atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(np.asarray(observed_dw13), np.asarray(expected_dw13), atol=1e-4, rtol=1e-4)


def test_source_push_semantic_w13_backward_expert_major_pallas_masks_invalid_rows_on_gpu():
    if jax.default_backend() != "gpu":
        pytest.skip("Pallas/MGPU production W13 backward requires a GPU backend")

    valid_rows = (jnp.arange(64, dtype=jnp.int32) % 5) != 1
    valid = valid_rows[None, None, :]
    x_expert = (jnp.arange(1 * 1 * 64 * 64, dtype=jnp.float32).reshape(1, 1, 64, 64) / 128.0).astype(jnp.bfloat16)
    dz13 = (jnp.arange(1 * 1 * 64 * 64, dtype=jnp.float32).reshape(1, 1, 64, 64) / 256.0 - 0.5).astype(jnp.bfloat16)
    w13 = (jnp.arange(1 * 1 * 64 * 64, dtype=jnp.float32).reshape(1, 1, 64, 64) / 512.0 + 0.1).astype(jnp.bfloat16)
    dirty_x = jnp.where(valid[..., None], x_expert, jnp.full_like(x_expert, 1.0e4))
    dirty_dz13 = jnp.where(valid[..., None], dz13, jnp.full_like(dz13, -1.0e4))
    block_sizes = SourcePushSemanticW13BackwardPallasBlockSizes(row_block=64, hidden_block=64, output_block=64)
    mesh = Mesh(np.asarray(jax.local_devices()[:1]), (SOURCE_PUSH_MESH_AXIS,))

    observed_dx, observed_dw13 = source_push_semantic_w13_backward_expert_major_pallas_mgpu(
        dirty_x,
        dirty_dz13,
        w13,
        valid,
        block_sizes=block_sizes,
        mesh=mesh,
    )
    expected_dx, expected_dw13 = source_push_semantic_w13_backward_expert_major_reference_jax(
        x_expert,
        dz13,
        w13,
        valid,
    )

    np.testing.assert_allclose(np.asarray(observed_dx), np.asarray(expected_dx), atol=2e-2, rtol=2e-2)
    np.testing.assert_allclose(np.asarray(observed_dw13), np.asarray(expected_dw13), atol=2e-2, rtol=2e-2)


def test_source_push_semantic_w13_backward_dw13_pallas_interpret_matches_jax_reference():
    plan, x, dz_pair, w_gate_up, block_sizes = _w13_backward_inputs()

    observed = source_push_semantic_w13_backward_dw13_pallas_mgpu(
        x,
        dz_pair,
        plan,
        block_sizes=block_sizes,
        interpret=True,
    )
    _expected_dx_pair, expected = source_push_semantic_w13_backward_reference_jax(x, dz_pair, w_gate_up, plan)

    np.testing.assert_allclose(np.asarray(observed), np.asarray(expected), atol=1e-4, rtol=1e-4)


def test_source_push_semantic_w13_backward_scaffold_interpret_matches_jax_reference():
    plan, x, dz_pair, w_gate_up, block_sizes = _w13_backward_inputs()

    observed_dx_pair, observed_dw13 = source_push_semantic_w13_backward_pallas_scaffold_mgpu(
        x,
        dz_pair,
        w_gate_up,
        plan,
        block_sizes=block_sizes,
        interpret=True,
    )
    expected_dx_pair, expected_dw13 = source_push_semantic_w13_backward_reference_jax(x, dz_pair, w_gate_up, plan)

    np.testing.assert_allclose(np.asarray(observed_dx_pair), np.asarray(expected_dx_pair), atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(np.asarray(observed_dw13), np.asarray(expected_dw13), atol=1e-4, rtol=1e-4)
