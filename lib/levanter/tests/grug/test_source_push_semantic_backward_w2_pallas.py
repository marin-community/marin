# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh

from levanter.grug._moe.source_push_plan import (
    SOURCE_PUSH_MESH_AXIS,
    build_source_push_semantic_plan_jax,
    source_push_semantic_expert_major_to_pair_jax,
    source_push_semantic_pair_to_expert_major_jax,
    source_push_semantic_w2_backward_reference_jax,
)
from levanter.grug._moe.source_push_semantic_backward_w2_pallas import (
    SourcePushSemanticW2BackwardExpertMajorPallasBlockSizes,
    SourcePushSemanticW2BackwardPallasBlockSizes,
    _make_source_push_semantic_w2_backward_dh_expert_major_mgpu_kernel,
    _make_source_push_semantic_w2_backward_dw2_expert_major_mgpu_kernel,
    source_push_semantic_w2_backward_dh_expert_major_pallas_mgpu,
    source_push_semantic_w2_backward_dh_pallas_mgpu,
    source_push_semantic_w2_backward_dw2_expert_major_pallas_mgpu,
    source_push_semantic_w2_backward_dw2_pallas_mgpu,
    source_push_semantic_w2_backward_expert_major_pallas_mgpu,
    source_push_semantic_w2_backward_expert_major_reference_jax,
    source_push_semantic_w2_backward_pallas_mgpu,
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


def _w2_backward_inputs():
    plan = _semantic_plan()
    h_pair = (jnp.arange(2 * 2 * 6 * 4, dtype=jnp.float32).reshape(2, 2, 6, 4) / 16).astype(jnp.bfloat16)
    dy_route = (jnp.arange(2 * 2 * 6 * 8, dtype=jnp.float32).reshape(2, 2, 6, 8) / 32).astype(jnp.bfloat16)
    w_down = (jnp.arange(2 * 2 * 4 * 8, dtype=jnp.float32).reshape(2, 2, 4, 8) / 64).astype(jnp.bfloat16)
    block_sizes = SourcePushSemanticW2BackwardPallasBlockSizes(
        row_block=2,
        intermediate_block=2,
        hidden_block=4,
    )
    return plan, h_pair, dy_route, w_down, block_sizes


def test_source_push_semantic_w2_backward_pallas_interpret_matches_jax_reference():
    plan, h_pair, dy_route, w_down, block_sizes = _w2_backward_inputs()

    observed_dh, observed_dw2 = source_push_semantic_w2_backward_pallas_mgpu(
        h_pair,
        dy_route,
        w_down,
        plan,
        block_sizes=block_sizes,
        interpret=True,
    )
    expected_dh, expected_dw2 = source_push_semantic_w2_backward_reference_jax(h_pair, dy_route, w_down, plan)

    np.testing.assert_allclose(np.asarray(observed_dh), np.asarray(expected_dh), atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(np.asarray(observed_dw2), np.asarray(expected_dw2), atol=1e-4, rtol=1e-4)


def test_source_push_semantic_w2_backward_individual_pallas_interpret_matches_jax_reference():
    plan, h_pair, dy_route, w_down, block_sizes = _w2_backward_inputs()

    observed_dh = source_push_semantic_w2_backward_dh_pallas_mgpu(
        dy_route,
        w_down,
        plan,
        block_sizes=block_sizes,
        interpret=True,
    )
    observed_dw2 = source_push_semantic_w2_backward_dw2_pallas_mgpu(
        h_pair,
        dy_route,
        plan,
        w_down_shape=w_down.shape,
        block_sizes=block_sizes,
        interpret=True,
    )
    expected_dh, expected_dw2 = source_push_semantic_w2_backward_reference_jax(h_pair, dy_route, w_down, plan)

    np.testing.assert_allclose(np.asarray(observed_dh), np.asarray(expected_dh), atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(np.asarray(observed_dw2), np.asarray(expected_dw2), atol=1e-4, rtol=1e-4)


def test_source_push_semantic_w2_backward_expert_major_reference_matches_pair_reference():
    plan, h_pair, dy_route, w_down, _block_sizes = _w2_backward_inputs()
    h_expert, valid = source_push_semantic_pair_to_expert_major_jax(
        h_pair,
        plan,
        rows_per_expert_capacity=8,
    )
    dy_expert, _dy_valid = source_push_semantic_pair_to_expert_major_jax(
        dy_route,
        plan,
        rows_per_expert_capacity=8,
    )

    observed_dh_expert, observed_dw2 = source_push_semantic_w2_backward_expert_major_reference_jax(
        h_expert,
        dy_expert,
        w_down,
        valid,
    )
    observed_dh_pair = source_push_semantic_expert_major_to_pair_jax(observed_dh_expert, plan)
    expected_dh_pair, expected_dw2 = source_push_semantic_w2_backward_reference_jax(h_pair, dy_route, w_down, plan)

    np.testing.assert_allclose(np.asarray(observed_dh_pair), np.asarray(expected_dh_pair), atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(np.asarray(observed_dw2), np.asarray(expected_dw2), atol=1e-4, rtol=1e-4)


def test_source_push_semantic_w2_backward_expert_major_interpret_masks_invalid_rows():
    valid = jnp.asarray(
        [
            [[True, False, True], [False, True, False]],
            [[True, True, False], [False, False, True]],
        ],
        dtype=jnp.bool_,
    )
    h_expert = (jnp.arange(2 * 2 * 3 * 4, dtype=jnp.float32).reshape(2, 2, 3, 4) / 11.0).astype(jnp.bfloat16)
    dy_expert = (jnp.arange(2 * 2 * 3 * 8, dtype=jnp.float32).reshape(2, 2, 3, 8) / 13.0).astype(jnp.bfloat16)
    w_down = (jnp.arange(2 * 2 * 4 * 8, dtype=jnp.float32).reshape(2, 2, 4, 8) / 17.0).astype(jnp.bfloat16)
    dirty_h = jnp.where(valid[..., None], h_expert, jnp.full_like(h_expert, 1.0e4))
    dirty_dy = jnp.where(valid[..., None], dy_expert, jnp.full_like(dy_expert, -1.0e4))

    expected_dh, expected_dw2 = source_push_semantic_w2_backward_expert_major_reference_jax(
        h_expert,
        dy_expert,
        w_down,
        valid,
    )
    block_sizes = SourcePushSemanticW2BackwardExpertMajorPallasBlockSizes(
        row_block=1, intermediate_block=1, hidden_block=1
    )
    observed_dh, observed_dw2 = source_push_semantic_w2_backward_expert_major_pallas_mgpu(
        dirty_h,
        dirty_dy,
        w_down,
        valid,
        block_sizes=block_sizes,
        interpret=True,
    )

    np.testing.assert_allclose(np.asarray(observed_dh), np.asarray(expected_dh), atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(np.asarray(observed_dw2), np.asarray(expected_dw2), atol=1e-4, rtol=1e-4)


def test_source_push_semantic_w2_backward_expert_major_kernel_factories_use_explicit_wgmma():
    dh_source = inspect.getsource(_make_source_push_semantic_w2_backward_dh_expert_major_mgpu_kernel)
    dw2_source = inspect.getsource(_make_source_push_semantic_w2_backward_dw2_expert_major_mgpu_kernel)

    assert "mgpu.wgmma" in dh_source
    assert "mgpu.wgmma" in dw2_source
    assert "pl.dot" not in dh_source
    assert "pl.dot" not in dw2_source


def test_source_push_semantic_w2_backward_expert_major_pallas_matches_reference_on_gpu():
    if jax.default_backend() != "gpu":
        pytest.skip("Pallas/MGPU production W2 backward requires a GPU backend")

    valid_rows = (jnp.arange(64, dtype=jnp.int32) % 7) != 2
    valid = valid_rows[None, None, :]
    h_expert = (jnp.arange(1 * 1 * 64 * 64, dtype=jnp.float32).reshape(1, 1, 64, 64) / 128.0).astype(jnp.bfloat16)
    dy_expert = (jnp.arange(1 * 1 * 64 * 64, dtype=jnp.float32).reshape(1, 1, 64, 64) / 256.0 - 0.5).astype(
        jnp.bfloat16
    )
    w_down = (jnp.arange(1 * 1 * 64 * 64, dtype=jnp.float32).reshape(1, 1, 64, 64) / 512.0 + 0.1).astype(jnp.bfloat16)
    block_sizes = SourcePushSemanticW2BackwardExpertMajorPallasBlockSizes(
        row_block=64,
        intermediate_block=64,
        hidden_block=64,
    )
    mesh = Mesh(np.asarray(jax.local_devices()[:1]), (SOURCE_PUSH_MESH_AXIS,))

    observed_dh, observed_dw2 = source_push_semantic_w2_backward_expert_major_pallas_mgpu(
        h_expert,
        dy_expert,
        w_down,
        valid,
        block_sizes=block_sizes,
        mesh=mesh,
    )
    observed_dh_only = source_push_semantic_w2_backward_dh_expert_major_pallas_mgpu(
        dy_expert,
        w_down,
        valid,
        block_sizes=block_sizes,
        mesh=mesh,
    )
    observed_dw2_only = source_push_semantic_w2_backward_dw2_expert_major_pallas_mgpu(
        h_expert,
        dy_expert,
        valid,
        block_sizes=block_sizes,
        mesh=mesh,
    )
    expected_dh, expected_dw2 = source_push_semantic_w2_backward_expert_major_reference_jax(
        h_expert,
        dy_expert,
        w_down,
        valid,
    )

    np.testing.assert_allclose(np.asarray(observed_dh), np.asarray(expected_dh), atol=2e-2, rtol=2e-2)
    np.testing.assert_allclose(np.asarray(observed_dw2), np.asarray(expected_dw2), atol=2e-2, rtol=2e-2)
    np.testing.assert_allclose(np.asarray(observed_dh_only), np.asarray(expected_dh), atol=2e-2, rtol=2e-2)
    np.testing.assert_allclose(np.asarray(observed_dw2_only), np.asarray(expected_dw2), atol=2e-2, rtol=2e-2)
