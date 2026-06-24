# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from levanter.kernels.pallas.moe_dispatch_up import api as dispatch_up_api
from levanter.kernels.pallas.moe_dispatch_up.reference import (
    MoeDispatchUpLayout,
    dispatch_prepacked_moe_dispatch_up_source_expert_reference,
    dispatch_prepacked_moe_dispatch_up_reference,
    compute_moe_up_from_layout_reference,
    compute_moe_up_from_layout_ragged_dot,
    prepack_moe_dispatch_up_source_expert_reference,
    prepack_moe_dispatch_up_reference,
)


def _hand_routing_inputs():
    x_by_rank = jnp.array(
        [
            [[10.0, 11.0], [12.0, 13.0], [14.0, 15.0]],
            [[20.0, 21.0], [22.0, 23.0], [24.0, 25.0]],
        ],
        dtype=jnp.float32,
    )
    expert_ids = jnp.array(
        [
            [[0, 2], [1, 3], [0, 3]],
            [[1, 2], [0, 3], [2, 1]],
        ],
        dtype=jnp.int32,
    )
    router_weights = jnp.array(
        [
            [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
            [[0.7, 0.8], [0.9, 1.0], [1.1, 1.2]],
        ],
        dtype=jnp.float32,
    )
    return x_by_rank, expert_ids, router_weights


def test_moe_dispatch_up_layout_reference_uses_expert_major_source_rank_order():
    x_by_rank, expert_ids, router_weights = _hand_routing_inputs()

    prepacked = prepack_moe_dispatch_up_reference(
        x_by_rank,
        expert_ids,
        router_weights,
        num_experts=4,
        recv_capacity=6,
    )
    layout = dispatch_prepacked_moe_dispatch_up_reference(prepacked, recv_capacity=6)

    np.testing.assert_array_equal(
        np.asarray(layout.rows_per_expert),
        np.array([[3, 3], [3, 3]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        np.asarray(layout.expert_base),
        np.array([[0, 3], [0, 3]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        np.asarray(prepacked.send_expert_count_by_dst),
        np.array(
            [
                [[2, 1], [1, 2]],
                [[1, 2], [2, 1]],
            ],
            dtype=np.int32,
        ),
    )
    np.testing.assert_array_equal(
        np.asarray(prepacked.send_expert_base_by_dst),
        np.array(
            [
                [[0, 2], [0, 1]],
                [[0, 1], [0, 2]],
            ],
            dtype=np.int32,
        ),
    )
    np.testing.assert_array_equal(
        np.asarray(prepacked.recv_source_expert_count),
        np.array(
            [
                [[2, 1], [1, 2]],
                [[1, 2], [2, 1]],
            ],
            dtype=np.int32,
        ),
    )
    np.testing.assert_array_equal(
        np.asarray(prepacked.recv_source_expert_base),
        np.array(
            [
                [[0, 3], [2, 4]],
                [[0, 3], [1, 5]],
            ],
            dtype=np.int32,
        ),
    )
    np.testing.assert_array_equal(
        np.asarray(layout.recv_valid),
        np.ones((2, 6), dtype=np.bool_),
    )
    np.testing.assert_allclose(
        np.asarray(layout.recv_x[0]),
        np.array(
            [
                [10.0, 11.0],
                [14.0, 15.0],
                [22.0, 23.0],
                [12.0, 13.0],
                [20.0, 21.0],
                [24.0, 25.0],
            ],
            dtype=np.float32,
        ),
        rtol=0,
        atol=0,
    )
    np.testing.assert_array_equal(np.asarray(layout.recv_src_rank[0]), np.array([0, 0, 1, 0, 1, 1], dtype=np.int32))
    np.testing.assert_array_equal(
        np.asarray(layout.recv_src_token_idx[0]),
        np.array([0, 2, 1, 1, 0, 2], dtype=np.int32),
    )
    np.testing.assert_array_equal(np.asarray(layout.recv_topk_slot[0]), np.array([0, 0, 0, 0, 0, 1], dtype=np.int32))
    np.testing.assert_allclose(
        np.asarray(layout.recv_router_weight[0]),
        np.array([0.1, 0.5, 0.9, 0.3, 0.7, 1.2], dtype=np.float32),
        rtol=1e-6,
        atol=1e-6,
    )


def test_moe_dispatch_up_prepack_matches_direct_layout_reference():
    x_by_rank, expert_ids, router_weights = _hand_routing_inputs()

    prepacked = dispatch_up_api.prepack_moe_dispatch_up(
        x_by_rank,
        expert_ids,
        router_weights,
        num_experts=4,
        recv_capacity=6,
    )
    from_prepack = dispatch_up_api.moe_dispatch_up_layout(prepacked, recv_capacity=6)
    direct = dispatch_up_api.moe_dispatch_up_layout_reference(
        x_by_rank,
        expert_ids,
        router_weights,
        num_experts=4,
        recv_capacity=6,
    )

    np.testing.assert_allclose(np.asarray(from_prepack.recv_x), np.asarray(direct.recv_x), rtol=0, atol=0)
    np.testing.assert_array_equal(np.asarray(from_prepack.recv_valid), np.asarray(direct.recv_valid))
    np.testing.assert_array_equal(np.asarray(from_prepack.recv_src_rank), np.asarray(direct.recv_src_rank))
    np.testing.assert_array_equal(np.asarray(from_prepack.recv_src_token_idx), np.asarray(direct.recv_src_token_idx))
    np.testing.assert_array_equal(np.asarray(from_prepack.recv_topk_slot), np.asarray(direct.recv_topk_slot))


def test_source_expert_prepack_matches_standard_prepack():
    x_by_rank, expert_ids, router_weights = _hand_routing_inputs()

    standard_prepacked = prepack_moe_dispatch_up_reference(
        x_by_rank,
        expert_ids,
        router_weights,
        num_experts=4,
        recv_capacity=6,
    )
    compact_prepacked = prepack_moe_dispatch_up_source_expert_reference(
        x_by_rank,
        expert_ids,
        router_weights,
        num_experts=4,
        recv_capacity=6,
    )
    standard = dispatch_prepacked_moe_dispatch_up_reference(standard_prepacked, recv_capacity=6)
    compact = dispatch_prepacked_moe_dispatch_up_source_expert_reference(compact_prepacked, recv_capacity=6)

    np.testing.assert_allclose(np.asarray(compact.recv_x), np.asarray(standard.recv_x), rtol=0, atol=0)
    np.testing.assert_array_equal(np.asarray(compact.recv_valid), np.asarray(standard.recv_valid))
    np.testing.assert_array_equal(np.asarray(compact.recv_local_expert), np.asarray(standard.recv_local_expert))
    np.testing.assert_array_equal(np.asarray(compact.recv_src_rank), np.asarray(standard.recv_src_rank))
    np.testing.assert_array_equal(np.asarray(compact.recv_src_token_idx), np.asarray(standard.recv_src_token_idx))
    np.testing.assert_array_equal(np.asarray(compact.recv_topk_slot), np.asarray(standard.recv_topk_slot))
    np.testing.assert_allclose(
        np.asarray(compact.recv_router_weight),
        np.asarray(standard.recv_router_weight),
        rtol=1e-6,
        atol=1e-6,
    )
    assert compact_prepacked.send_x_by_dst_expert.shape == (2, 2, 2, 2, 2)
    assert int(np.asarray(compact.overflow_count)) == 0


def test_source_expert_prepack_reports_source_capacity_overflow():
    x_by_rank, expert_ids, router_weights = _hand_routing_inputs()

    compact_prepacked = prepack_moe_dispatch_up_source_expert_reference(
        x_by_rank,
        expert_ids,
        router_weights,
        num_experts=4,
        recv_capacity=6,
        source_expert_capacity=1,
    )
    compact = dispatch_prepacked_moe_dispatch_up_source_expert_reference(compact_prepacked, recv_capacity=6)

    assert int(np.asarray(compact.overflow_count)) == 4
    assert int(np.asarray(jnp.sum(compact.recv_valid))) == 8


def _manual_w13_silu(layout: MoeDispatchUpLayout, w_gate_up_by_rank: jax.Array) -> jax.Array:
    ep_size, recv_capacity, _hidden = layout.recv_x.shape
    intermediate = w_gate_up_by_rank.shape[-1] // 2
    out = []
    for dst_rank in range(ep_size):
        rows = []
        for row in range(recv_capacity):
            if not bool(np.asarray(layout.recv_valid[dst_rank, row])):
                rows.append(np.zeros((intermediate,), dtype=np.float32))
                continue
            local_expert = int(np.asarray(layout.recv_local_expert[dst_rank, row]))
            gate_up = np.asarray(layout.recv_x[dst_rank, row]) @ np.asarray(w_gate_up_by_rank[dst_rank, local_expert])
            gate, up = np.split(gate_up, 2)
            rows.append(np.asarray(jax.nn.silu(jnp.asarray(gate)) * jnp.asarray(up)))
        out.append(np.stack(rows, axis=0))
    return jnp.asarray(np.stack(out, axis=0), dtype=layout.recv_x.dtype)


@pytest.mark.parametrize(
    ("ep_size", "experts_per_rank", "top_k"),
    [
        (2, 1, 1),
        (2, 2, 4),
        (8, 1, 1),
        (8, 4, 4),
    ],
)
def test_moe_dispatch_up_matches_manual_reference(ep_size: int, experts_per_rank: int, top_k: int):
    tokens_per_rank = 5
    hidden = 8
    intermediate = 6
    num_experts = ep_size * experts_per_rank
    key = jax.random.key(ep_size * 100 + experts_per_rank * 10 + top_k)
    k_x, k_w = jax.random.split(key)
    x_by_rank = jax.random.normal(k_x, (ep_size, tokens_per_rank, hidden), dtype=jnp.float32)
    token_ids = jnp.arange(tokens_per_rank, dtype=jnp.int32)[None, :, None]
    rank_ids = jnp.arange(ep_size, dtype=jnp.int32)[:, None, None]
    slot_ids = jnp.arange(top_k, dtype=jnp.int32)[None, None, :]
    expert_ids = (rank_ids * experts_per_rank + token_ids + slot_ids) % num_experts
    router_weights = jax.nn.softmax(
        jnp.arange(ep_size * tokens_per_rank * top_k, dtype=jnp.float32).reshape(ep_size, tokens_per_rank, top_k),
        axis=-1,
    )
    w_gate_up = jax.random.normal(k_w, (ep_size, experts_per_rank, hidden, 2 * intermediate), dtype=jnp.float32)
    recv_capacity = ep_size * tokens_per_rank * top_k

    dispatch_up, layout = dispatch_up_api.moe_dispatch_up(
        x_by_rank,
        expert_ids,
        router_weights,
        w_gate_up,
        num_experts=num_experts,
        recv_capacity=recv_capacity,
    )
    expected = _manual_w13_silu(layout, w_gate_up)

    np.testing.assert_allclose(np.asarray(dispatch_up), np.asarray(expected), rtol=1e-5, atol=1e-5)
    assert int(np.asarray(layout.overflow_count)) == 0


def test_moe_dispatch_up_reference_is_differentiable():
    x_by_rank, expert_ids, router_weights = _hand_routing_inputs()
    w_gate_up = jnp.arange(2 * 2 * 2 * 6, dtype=jnp.float32).reshape(2, 2, 2, 6) / 13.0
    prepacked = prepack_moe_dispatch_up_reference(
        x_by_rank,
        expert_ids,
        router_weights,
        num_experts=4,
        recv_capacity=6,
    )
    layout = dispatch_prepacked_moe_dispatch_up_reference(prepacked, recv_capacity=6)

    def loss(weights):
        return jnp.sum(compute_moe_up_from_layout_reference(layout, weights).astype(jnp.float32))

    grad = jax.grad(loss)(w_gate_up)

    assert grad.shape == w_gate_up.shape
    assert jnp.isfinite(grad).all()


def test_moe_dispatch_up_ragged_dot_matches_reference_with_capacity_overflow():
    x_by_rank, expert_ids, router_weights = _hand_routing_inputs()
    w_gate_up = jnp.arange(2 * 2 * 2 * 6, dtype=jnp.float32).reshape(2, 2, 2, 6) / 13.0
    prepacked = prepack_moe_dispatch_up_reference(
        x_by_rank,
        expert_ids,
        router_weights,
        num_experts=4,
        recv_capacity=4,
    )
    layout = dispatch_prepacked_moe_dispatch_up_reference(prepacked, recv_capacity=4)

    expected = compute_moe_up_from_layout_reference(layout, w_gate_up)
    actual = compute_moe_up_from_layout_ragged_dot(layout, w_gate_up, implementation="xla")

    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-5, atol=1e-5)
    assert int(np.asarray(layout.overflow_count)) == 4


def test_moe_dispatch_up_api_can_use_ragged_dot_w13():
    x_by_rank, expert_ids, router_weights = _hand_routing_inputs()
    w_gate_up = jnp.arange(2 * 2 * 2 * 6, dtype=jnp.float32).reshape(2, 2, 2, 6) / 13.0

    expected, expected_layout = dispatch_up_api.moe_dispatch_up(
        x_by_rank,
        expert_ids,
        router_weights,
        w_gate_up,
        num_experts=4,
        recv_capacity=6,
    )
    actual, actual_layout = dispatch_up_api.moe_dispatch_up(
        x_by_rank,
        expert_ids,
        router_weights,
        w_gate_up,
        num_experts=4,
        recv_capacity=6,
        w13_implementation="ragged_dot",
        ragged_dot_implementation="xla",
    )

    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(np.asarray(actual_layout.recv_x), np.asarray(expected_layout.recv_x))


def test_moe_dispatch_up_backward_reference_matches_autodiff():
    ep_size = 2
    tokens_per_rank = 4
    experts_per_rank = 2
    top_k = 2
    hidden = 3
    intermediate = 5
    num_experts = ep_size * experts_per_rank
    recv_capacity = ep_size * tokens_per_rank * top_k
    key = jax.random.key(6597)
    k_x, k_w, k_g = jax.random.split(key, 3)
    x_by_rank = jax.random.normal(k_x, (ep_size, tokens_per_rank, hidden), dtype=jnp.float32)
    w_gate_up = jax.random.normal(k_w, (ep_size, experts_per_rank, hidden, 2 * intermediate), dtype=jnp.float32)
    rank_ids = jnp.arange(ep_size, dtype=jnp.int32)[:, None, None]
    token_ids = jnp.arange(tokens_per_rank, dtype=jnp.int32)[None, :, None]
    slot_ids = jnp.arange(top_k, dtype=jnp.int32)[None, None, :]
    expert_ids = (rank_ids * experts_per_rank + token_ids + slot_ids) % num_experts
    router_weights = jax.nn.softmax(
        jnp.arange(ep_size * tokens_per_rank * top_k, dtype=jnp.float32).reshape(ep_size, tokens_per_rank, top_k),
        axis=-1,
    )
    grad_dispatch_up = jax.random.normal(k_g, (ep_size, recv_capacity, intermediate), dtype=jnp.float32)

    def loss(x_arg, w_arg):
        dispatch_up, _layout = dispatch_up_api.moe_dispatch_up(
            x_arg,
            expert_ids,
            router_weights,
            w_arg,
            num_experts=num_experts,
            recv_capacity=recv_capacity,
        )
        return jnp.sum(dispatch_up * grad_dispatch_up)

    autodiff_grad_x, autodiff_grad_w = jax.grad(loss, argnums=(0, 1))(x_by_rank, w_gate_up)
    explicit_grad_x, explicit_grad_w = dispatch_up_api.moe_dispatch_up_bwd_reference(
        x_by_rank,
        expert_ids,
        router_weights,
        w_gate_up,
        grad_dispatch_up,
        num_experts=num_experts,
        recv_capacity=recv_capacity,
    )

    np.testing.assert_allclose(np.asarray(explicit_grad_x), np.asarray(autodiff_grad_x), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(explicit_grad_w), np.asarray(autodiff_grad_w), rtol=1e-5, atol=1e-5)


def test_explicit_mosaic_gpu_dispatch_requires_local_gpu_mesh():
    if len([device for device in jax.local_devices() if device.platform == "gpu"]) >= 2:
        pytest.skip("local host has an MGPU runtime")
    x_by_rank, expert_ids, router_weights = _hand_routing_inputs()
    prepacked = prepack_moe_dispatch_up_reference(
        x_by_rank,
        expert_ids,
        router_weights,
        num_experts=4,
        recv_capacity=6,
    )

    with pytest.raises(dispatch_up_api.MosaicGpuUnsupportedError, match="at least two local GPU"):
        dispatch_up_api.moe_dispatch_up_layout(prepacked, recv_capacity=6, implementation="mosaic_gpu")
