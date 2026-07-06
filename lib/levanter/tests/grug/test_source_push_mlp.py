# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh

from levanter.grug._moe import source_push_forward
from levanter.grug._moe.source_push_inbox import AXIS, PushInboxConfig
import levanter.grug._moe.source_push_mlp as source_push_mlp
from levanter.grug._moe.source_push_mlp import (
    SOURCE_PUSH_MLP_IMPLEMENTATION_PALLAS_MGPU,
    SOURCE_PUSH_MLP_IMPLEMENTATION_REFERENCE,
    build_source_push_mlp_route_table,
    source_push_moe_mlp,
    source_push_moe_mlp_from_plan,
    source_push_moe_mlp_custom_vjp,
    source_push_moe_mlp_reference,
    source_push_moe_mlp_reference_with_h_flat,
    source_push_moe_mlp_reference_with_h,
    source_push_mlp_route_table_from_plan,
)


EP_SIZE = 2
EXPERTS_PER_RANK = 2
BLOCK_M = 2
TOKENS_PER_RANK = 4
TOPK = 2
HIDDEN_DIM = 3
INTERMEDIATE_DIM = 2


def _small_mlp_inputs():
    route_assignments = jnp.array(
        [
            [[0, 2], [1, 3], [0, 2], [1, 3]],
            [[2, 0], [3, 1], [2, 0], [3, 1]],
        ],
        dtype=jnp.int32,
    )
    route_weights = jnp.array(
        [
            [[0.50, 0.25], [0.75, 0.125], [0.375, 0.625], [0.25, 0.50]],
            [[0.20, 0.80], [0.60, 0.40], [0.30, 0.70], [0.90, 0.10]],
        ],
        dtype=jnp.float32,
    )
    x = jnp.linspace(
        -0.2,
        0.4,
        EP_SIZE * TOKENS_PER_RANK * HIDDEN_DIM,
        dtype=jnp.float32,
    ).reshape(EP_SIZE, TOKENS_PER_RANK, HIDDEN_DIM)
    w13 = jnp.linspace(
        -0.3,
        0.5,
        EP_SIZE * EXPERTS_PER_RANK * HIDDEN_DIM * 2 * INTERMEDIATE_DIM,
        dtype=jnp.float32,
    ).reshape(EP_SIZE, EXPERTS_PER_RANK, HIDDEN_DIM, 2 * INTERMEDIATE_DIM)
    w2 = jnp.linspace(
        -0.4,
        0.2,
        EP_SIZE * EXPERTS_PER_RANK * INTERMEDIATE_DIM * HIDDEN_DIM,
        dtype=jnp.float32,
    ).reshape(EP_SIZE, EXPERTS_PER_RANK, INTERMEDIATE_DIM, HIDDEN_DIM)
    return x, route_assignments, route_weights, w13, w2


def _route_table(route_assignments, route_weights):
    assert route_assignments.shape == route_weights.shape
    route_table, dropped_routes = build_source_push_mlp_route_table(
        route_assignments,
        ep_size=EP_SIZE,
        experts_per_rank=EXPERTS_PER_RANK,
        block_m=BLOCK_M,
        capacity_factor=2.0,
    )
    assert int(dropped_routes) == 0
    return route_table


def _small_forward_config():
    return PushInboxConfig(
        ep_size=EP_SIZE,
        entries_per_rank=2,
        inbox_slots=2,
        hidden_dim=HIDDEN_DIM,
        intermediate_dim=INTERMEDIATE_DIM,
        block_m=BLOCK_M,
        block_k=1,
        block_n=1,
        experts_per_rank=EXPERTS_PER_RANK,
        send_worker_programs_per_peer=1,
        worker_programs_per_peer=4,
        send_pipeline_depth=1,
        n_groups_per_job=1,
        routing="balanced",
        tokens_per_rank=TOKENS_PER_RANK,
        topk=TOPK,
        capacity_factor=2.0,
    )


def _small_forward_plan_inputs(*, use_exact_expert_major=False):
    x, route_assignments, route_weights, w13, w2 = _small_mlp_inputs()
    config = _small_forward_config()
    host_inputs = source_push_forward.make_source_push_forward_inputs(
        config,
        x,
        route_assignments,
        route_weights,
        w13,
        w2,
        input_mode="exact_source_push_plan" if use_exact_expert_major else "real_arrays",
        use_exact_expert_major=use_exact_expert_major,
    )
    route_table = source_push_mlp_route_table_from_plan(
        host_inputs.plan,
        src_base_by_expert=host_inputs.src_base_by_expert,
    )
    assert int(host_inputs.plan.dropped_routes) == 0
    return config, host_inputs, route_table, x, route_weights, w13, w2


def test_source_push_moe_mlp_reference_saves_w13_preactivation_h():
    x, route_assignments, route_weights, w13, w2 = _small_mlp_inputs()
    route_table = _route_table(route_assignments, route_weights)

    _, h = source_push_moe_mlp_reference_with_h(route_table, x, route_weights, w13, w2)

    assert h.shape == (EP_SIZE, EXPERTS_PER_RANK, route_table.expert_capacity, 2 * INTERMEDIATE_DIM)
    for route in range(route_table.source_rank.shape[0]):
        src = int(route_table.source_rank[route])
        token = int(route_table.token_id[route])
        dst = int(route_table.destination_rank[route])
        expert = int(route_table.local_expert[route])
        expert_row = int(route_table.expert_row[route])
        expected_h = x[src, token] @ w13[dst, expert]
        np.testing.assert_allclose(np.asarray(h[dst, expert, expert_row]), np.asarray(expected_h), atol=1e-6)


def test_source_push_w13_h_expert_major_from_plan_returns_trimmed_layout(monkeypatch):
    config, host_inputs, route_table, x, route_weights, w13, w2 = _small_forward_plan_inputs()
    _y, expected_h = source_push_moe_mlp_reference_with_h(route_table, x, route_weights, w13, w2)
    expected_h = jnp.asarray(expected_h, dtype=jnp.bfloat16)
    calls = []
    source_push_forward._STAGED_FORWARD_CALL_CACHE.clear()

    def fake_w13_h_compact_kernel(_mesh, config_arg, *, compact_expert_capacity, use_exact_expert_major):
        calls.append(
            {
                "compact_expert_capacity": compact_expert_capacity,
                "use_exact_expert_major": use_exact_expert_major,
            }
        )

        def kernel(x_arg, _send_meta, _recv_meta, _expert_base, _src_base_by_expert, _w13_arg):
            padded_h = jnp.pad(expected_h, ((0, 0), (0, 0), (0, 3), (0, 0)), constant_values=17)
            return jnp.zeros((1,), dtype=x_arg.dtype), padded_h

        return kernel

    monkeypatch.setattr(source_push_forward, "_sharded_w13_h_compact_kernel", fake_w13_h_compact_kernel)

    mesh = Mesh(np.asarray(jax.devices()[:1]), (AXIS,))
    observed_h, dropped_routes = source_push_forward.source_push_w13_h_expert_major_from_plan(
        config,
        host_inputs,
        x,
        w13,
        mesh=mesh,
    )

    assert int(dropped_routes) == 0
    assert calls
    assert calls[0]["compact_expert_capacity"] >= route_table.expert_capacity
    assert observed_h.shape == (EP_SIZE, EXPERTS_PER_RANK, route_table.expert_capacity, 2 * INTERMEDIATE_DIM)
    np.testing.assert_allclose(np.asarray(observed_h, dtype=np.float32), np.asarray(expected_h, dtype=np.float32))


def test_source_push_moe_mlp_custom_vjp_residual_names_h_checkpoint():
    x, route_assignments, route_weights, w13, w2 = _small_mlp_inputs()
    route_table = _route_table(route_assignments, route_weights)

    y, residual = source_push_mlp._source_push_moe_mlp_fwd(route_table, x, route_weights, w13, w2)
    expected_y, expected_h = source_push_moe_mlp_reference_with_h(route_table, x, route_weights, w13, w2)

    assert isinstance(residual, source_push_mlp.SourcePushMlpReferenceResidual)
    assert residual._fields == ("route_table", "x", "expert_route_weights", "w13", "w2", "h")
    assert source_push_mlp.SourcePushMlpPlanResidual._fields == ("x", "expert_route_weights", "w13", "w2", "h")
    assert not {"recv_x", "packed_x", "activation", "route_y", "scratch"}.intersection(residual._fields)
    np.testing.assert_allclose(np.asarray(y), np.asarray(expected_y), atol=0, rtol=0)
    np.testing.assert_allclose(np.asarray(residual.h), np.asarray(expected_h), atol=0, rtol=0)
    assert residual.expert_route_weights.shape == (EP_SIZE, EXPERTS_PER_RANK, route_table.expert_capacity)
    for route in range(route_table.source_rank.shape[0]):
        src = int(route_table.source_rank[route])
        token = int(route_table.token_id[route])
        slot = int(route_table.route_slot[route])
        dst = int(route_table.destination_rank[route])
        expert = int(route_table.local_expert[route])
        expert_row = int(route_table.expert_row[route])
        np.testing.assert_allclose(
            np.asarray(residual.expert_route_weights[dst, expert, expert_row]),
            np.asarray(route_weights[src, token, slot]),
            atol=0,
            rtol=0,
        )


def test_source_push_moe_mlp_reference_with_h_flat_matches_compact_h_reference():
    config, host_inputs, route_table, x, route_weights, w13, w2 = _small_forward_plan_inputs()

    flat_y, flat_h = source_push_moe_mlp_reference_with_h_flat(
        route_table,
        jnp.asarray(host_inputs.expert_base, dtype=jnp.int32),
        config.hidden_rows_per_rank,
        x,
        route_weights,
        w13,
        w2,
    )
    compact_y, compact_h = source_push_moe_mlp_reference_with_h(route_table, x, route_weights, w13, w2)

    np.testing.assert_allclose(np.asarray(flat_y), np.asarray(compact_y), atol=0, rtol=0)
    for route in range(route_table.source_rank.shape[0]):
        dst = int(route_table.destination_rank[route])
        expert = int(route_table.local_expert[route])
        expert_row = int(route_table.expert_row[route])
        flat_row = int(host_inputs.expert_base[dst, expert]) + expert_row
        np.testing.assert_allclose(
            np.asarray(flat_h[dst, flat_row]),
            np.asarray(compact_h[dst, expert, expert_row]),
            atol=0,
            rtol=0,
        )


def test_source_push_mlp_flat_h_to_compact_matches_source_padded_rows():
    config, host_inputs, route_table, x, route_weights, w13, w2 = _small_forward_plan_inputs()
    expert_base = jnp.asarray(host_inputs.expert_base, dtype=jnp.int32)
    _flat_y, flat_h = source_push_moe_mlp_reference_with_h_flat(
        route_table,
        expert_base,
        config.hidden_rows_per_rank,
        x,
        route_weights,
        w13,
        w2,
    )
    _compact_y, compact_h = source_push_moe_mlp_reference_with_h(route_table, x, route_weights, w13, w2)

    observed_compact = source_push_mlp._source_push_mlp_flat_h_to_compact(route_table, expert_base, flat_h)
    observed_flat = source_push_mlp._source_push_mlp_compact_h_to_flat(
        route_table,
        expert_base,
        config.hidden_rows_per_rank,
        observed_compact,
    )

    assert observed_compact.shape == (EP_SIZE, EXPERTS_PER_RANK, route_table.expert_capacity, 2 * INTERMEDIATE_DIM)
    np.testing.assert_allclose(np.asarray(observed_compact), np.asarray(compact_h), atol=0, rtol=0)
    for route in range(route_table.source_rank.shape[0]):
        dst = int(route_table.destination_rank[route])
        expert = int(route_table.local_expert[route])
        expert_row = int(route_table.expert_row[route])
        flat_row = int(host_inputs.expert_base[dst, expert]) + expert_row
        np.testing.assert_allclose(
            np.asarray(observed_flat[dst, flat_row]),
            np.asarray(flat_h[dst, flat_row]),
            atol=0,
            rtol=0,
        )


def test_source_push_moe_mlp_h_route_weights_align_with_flat_h_rows():
    config, host_inputs, route_table, _x, route_weights, _w13, _w2 = _small_forward_plan_inputs()

    expert_route_weights = source_push_mlp._source_push_mlp_route_weights_to_all_expert_major(
        route_table,
        route_weights,
    )
    h_route_weights = source_push_mlp.source_push_h_row_route_weights_jax(
        route_weights,
        host_inputs.plan,
        host_inputs.send_meta,
        host_inputs.expert_base,
        host_inputs.src_base_by_expert,
        hidden_rows_per_rank=config.hidden_rows_per_rank,
        use_exact_expert_major=host_inputs.use_exact_expert_major,
    )

    assert source_push_mlp.SourcePushMlpPlanResidual._fields == ("x", "expert_route_weights", "w13", "w2", "h")
    assert expert_route_weights.shape == (EP_SIZE, EXPERTS_PER_RANK, route_table.expert_capacity)
    assert h_route_weights.shape == (EP_SIZE, config.hidden_rows_per_rank)
    for route in range(route_table.source_rank.shape[0]):
        src = int(route_table.source_rank[route])
        token = int(route_table.token_id[route])
        slot = int(route_table.route_slot[route])
        dst = int(route_table.destination_rank[route])
        expert = int(route_table.local_expert[route])
        expert_row = int(route_table.expert_row[route])
        flat_row = int(host_inputs.expert_base[dst, expert]) + expert_row
        expected_weight = route_weights[src, token, slot]

        np.testing.assert_array_equal(
            np.asarray(expert_route_weights[dst, expert, expert_row]),
            np.asarray(expected_weight),
        )
        np.testing.assert_array_equal(
            np.asarray(host_inputs.h_route_weights[dst, flat_row]),
            np.asarray(expected_weight),
        )
        np.testing.assert_array_equal(
            np.asarray(h_route_weights[dst, flat_row]),
            np.asarray(expected_weight),
        )


def test_source_push_mlp_source_tensor_to_expert_major_masks_padding_rows():
    x, route_assignments, route_weights, _w13, _w2 = _small_mlp_inputs()
    route_table = _route_table(route_assignments, route_weights)
    expert = 0
    route_indices = source_push_mlp._source_push_mlp_expert_route_indices(
        route_table,
        expert,
    )

    observed = source_push_mlp._source_push_mlp_source_tensor_to_expert_major(
        x,
        route_indices.safe_src,
        route_indices.safe_token,
        route_indices.valid_f,
    )

    expected = np.zeros((EP_SIZE, route_table.expert_capacity, HIDDEN_DIM), dtype=np.float32)
    valid_host = np.asarray(route_indices.valid)
    safe_src_host = np.asarray(route_indices.safe_src)
    safe_token_host = np.asarray(route_indices.safe_token)
    x_host = np.asarray(x, dtype=np.float32)
    for dst in range(EP_SIZE):
        for row in range(route_table.expert_capacity):
            if valid_host[dst, row]:
                expected[dst, row] = x_host[safe_src_host[dst, row], safe_token_host[dst, row]]

    np.testing.assert_allclose(np.asarray(observed), expected, atol=0, rtol=0)


def test_source_push_mlp_expert_route_indices_named_contract():
    _x, route_assignments, route_weights, _w13, _w2 = _small_mlp_inputs()
    route_table = _route_table(route_assignments, route_weights)
    route_indices = source_push_mlp._source_push_mlp_expert_route_indices(route_table, expert=1)

    assert route_indices._fields == ("valid", "safe_src", "safe_token", "safe_slot", "valid_f")
    np.testing.assert_array_equal(np.asarray(route_indices.valid), np.asarray(route_table.valid_by_expert[:, 1]))
    np.testing.assert_array_equal(
        np.asarray(route_indices.safe_src),
        np.maximum(np.asarray(route_table.source_rank_by_expert[:, 1]), 0),
    )
    np.testing.assert_array_equal(
        np.asarray(route_indices.safe_token),
        np.maximum(np.asarray(route_table.token_id_by_expert[:, 1]), 0),
    )
    np.testing.assert_array_equal(
        np.asarray(route_indices.safe_slot),
        np.maximum(np.asarray(route_table.route_slot_by_expert[:, 1]), 0),
    )
    np.testing.assert_array_equal(
        np.asarray(route_indices.valid_f),
        np.asarray(route_indices.valid, dtype=np.float32),
    )


def test_source_push_mlp_backward_for_expert_carries_named_route_indices():
    x, route_assignments, route_weights, w13, w2 = _small_mlp_inputs()
    route_table = _route_table(route_assignments, route_weights)
    expert = 1
    dy = jnp.linspace(-0.5, 0.7, x.size, dtype=jnp.float32).reshape(x.shape)
    _, h = source_push_moe_mlp_reference_with_h(route_table, x, route_weights, w13, w2)
    expert_route_weights = source_push_mlp._source_push_mlp_route_weights_to_all_expert_major(
        route_table,
        route_weights,
    )

    backward = source_push_mlp._source_push_mlp_backward_for_expert(
        route_table,
        x,
        expert_route_weights,
        w13,
        w2,
        dy,
        h[:, expert],
        expert,
    )
    expected_route_indices = source_push_mlp._source_push_mlp_expert_route_indices(route_table, expert)

    assert backward._fields == ("route_indices", "dx_block", "d_route_block", "dw13_block", "dw2_block")
    for observed, expected in zip(backward.route_indices, expected_route_indices, strict=True):
        np.testing.assert_array_equal(np.asarray(observed), np.asarray(expected))


def test_source_push_mlp_expert_backward_inputs_route_dy_and_weights():
    x, route_assignments, route_weights, w13, w2 = _small_mlp_inputs()
    route_table = _route_table(route_assignments, route_weights)
    expert = 1
    dy = jnp.linspace(-0.5, 0.7, x.size, dtype=jnp.float32).reshape(x.shape)
    _, h = source_push_moe_mlp_reference_with_h(route_table, x, route_weights, w13, w2)
    expert_route_weights = source_push_mlp._source_push_mlp_route_weights_to_all_expert_major(
        route_table,
        route_weights,
    )

    observed = source_push_mlp._source_push_mlp_expert_backward_inputs(
        route_table,
        expert_route_weights,
        w13,
        w2,
        dy,
        h[:, expert],
        expert,
    )
    expected_route_indices = source_push_mlp._source_push_mlp_expert_route_indices(route_table, expert)

    assert observed._fields == ("route_indices", "h_block", "weights", "dy_block", "w2_block", "w13_block")
    for observed_index, expected_index in zip(observed.route_indices, expected_route_indices, strict=True):
        np.testing.assert_array_equal(np.asarray(observed_index), np.asarray(expected_index))
    np.testing.assert_array_equal(np.asarray(observed.h_block), np.asarray(h[:, expert]))
    np.testing.assert_allclose(
        np.asarray(observed.weights),
        np.asarray(expert_route_weights[:, expert] * expected_route_indices.valid_f),
        atol=0,
        rtol=0,
    )
    np.testing.assert_array_equal(np.asarray(observed.w2_block), np.asarray(w2[:, expert], dtype=np.float32))
    np.testing.assert_array_equal(np.asarray(observed.w13_block), np.asarray(w13[:, expert], dtype=np.float32))

    expected_dy = np.zeros((EP_SIZE, route_table.expert_capacity, HIDDEN_DIM), dtype=np.float32)
    valid_host = np.asarray(expected_route_indices.valid)
    safe_src_host = np.asarray(expected_route_indices.safe_src)
    safe_token_host = np.asarray(expected_route_indices.safe_token)
    dy_host = np.asarray(dy, dtype=np.float32)
    for dst in range(EP_SIZE):
        for row in range(route_table.expert_capacity):
            if valid_host[dst, row]:
                expected_dy[dst, row] = dy_host[safe_src_host[dst, row], safe_token_host[dst, row]]
    np.testing.assert_allclose(np.asarray(observed.dy_block), expected_dy, atol=0, rtol=0)


def test_source_push_mlp_w2_swiglu_backward_for_expert_matches_autograd():
    h_block = jnp.linspace(
        -0.6,
        0.7,
        EP_SIZE * BLOCK_M * 2 * INTERMEDIATE_DIM,
        dtype=jnp.float32,
    ).reshape(EP_SIZE, BLOCK_M, 2 * INTERMEDIATE_DIM)
    weights = jnp.array([[0.5, 0.25], [0.75, 0.125]], dtype=jnp.float32)
    dy_block = jnp.linspace(
        -0.4,
        0.3,
        EP_SIZE * BLOCK_M * HIDDEN_DIM,
        dtype=jnp.float32,
    ).reshape(EP_SIZE, BLOCK_M, HIDDEN_DIM)
    w2_block = jnp.linspace(
        -0.2,
        0.6,
        EP_SIZE * INTERMEDIATE_DIM * HIDDEN_DIM,
        dtype=jnp.float32,
    ).reshape(EP_SIZE, INTERMEDIATE_DIM, HIDDEN_DIM)
    valid_f = jnp.array([[1.0, 0.0], [1.0, 1.0]], dtype=jnp.float32)

    def forward_loss(h_arg, weights_arg, w2_arg):
        live_h = h_arg * valid_f[..., None]
        live_weights = weights_arg * valid_f
        gate, up = jnp.split(live_h, 2, axis=-1)
        activation = jax.nn.silu(gate) * up
        route_y = jnp.einsum("sci,sid->scd", activation * live_weights[..., None], w2_arg)
        return jnp.sum(route_y * dy_block)

    expected_d_h, expected_d_weights, expected_dw2 = jax.grad(forward_loss, argnums=(0, 1, 2))(
        h_block,
        weights,
        w2_block,
    )
    observed = source_push_mlp._source_push_mlp_w2_swiglu_backward_for_expert(
        h_block,
        weights,
        dy_block,
        w2_block,
        valid_f,
    )

    np.testing.assert_allclose(np.asarray(observed.d_h_block), np.asarray(expected_d_h), atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(
        np.asarray(observed.d_route_block), np.asarray(expected_d_weights), atol=1e-6, rtol=1e-6
    )
    np.testing.assert_allclose(np.asarray(observed.dw2_block), np.asarray(expected_dw2), atol=1e-6, rtol=1e-6)
    np.testing.assert_array_equal(np.asarray(observed.d_h_block[0, 1]), np.zeros(2 * INTERMEDIATE_DIM))
    np.testing.assert_array_equal(np.asarray(observed.d_route_block[0, 1]), np.asarray(0.0, dtype=np.float32))


def test_source_push_mlp_x_w13_backward_for_expert_remats_x_matches_reference():
    x, route_assignments, route_weights, w13, _w2 = _small_mlp_inputs()
    route_table = _route_table(route_assignments, route_weights)
    expert = 1
    route_indices = source_push_mlp._source_push_mlp_expert_route_indices(route_table, expert)
    d_h_block = jnp.linspace(
        -0.3,
        0.5,
        EP_SIZE * route_table.expert_capacity * 2 * INTERMEDIATE_DIM,
        dtype=jnp.float32,
    ).reshape(EP_SIZE, route_table.expert_capacity, 2 * INTERMEDIATE_DIM)
    d_h_block = d_h_block * route_indices.valid_f[..., None]
    w13_block = w13[:, expert].astype(jnp.float32)

    observed = source_push_mlp._source_push_mlp_x_w13_backward_for_expert(
        x,
        route_indices,
        d_h_block,
        w13_block,
    )

    expected_x_block = np.zeros((EP_SIZE, route_table.expert_capacity, HIDDEN_DIM), dtype=np.float32)
    valid_host = np.asarray(route_indices.valid)
    safe_src_host = np.asarray(route_indices.safe_src)
    safe_token_host = np.asarray(route_indices.safe_token)
    x_host = np.asarray(x, dtype=np.float32)
    for dst in range(EP_SIZE):
        for row in range(route_table.expert_capacity):
            if valid_host[dst, row]:
                expected_x_block[dst, row] = x_host[safe_src_host[dst, row], safe_token_host[dst, row]]

    expected_dx_block = np.einsum(
        "sco,sdo->scd",
        np.asarray(d_h_block, dtype=np.float32),
        np.asarray(w13_block, dtype=np.float32),
    )
    expected_dw13_block = np.einsum(
        "scd,sco->sdo",
        expected_x_block,
        np.asarray(d_h_block, dtype=np.float32),
    )

    assert observed._fields == ("dx_block", "dw13_block")
    np.testing.assert_allclose(np.asarray(observed.dx_block), expected_dx_block, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(observed.dw13_block), expected_dw13_block, atol=1e-6, rtol=1e-6)


def test_source_push_mlp_accumulate_expert_major_to_source_sums_duplicates():
    source_tokens = jnp.zeros((EP_SIZE, TOKENS_PER_RANK, HIDDEN_DIM), dtype=jnp.float32)
    source_routes = jnp.zeros((EP_SIZE, TOKENS_PER_RANK, TOPK), dtype=jnp.float32)
    safe_src = jnp.array([[0, 0], [1, 1]], dtype=jnp.int32)
    safe_token = jnp.array([[2, 2], [1, 3]], dtype=jnp.int32)
    safe_slot = jnp.array([[1, 1], [0, 1]], dtype=jnp.int32)
    route_indices = source_push_mlp._SourcePushMlpExpertRouteIndices(
        valid=jnp.ones_like(safe_src, dtype=jnp.bool_),
        safe_src=safe_src,
        safe_token=safe_token,
        safe_slot=safe_slot,
        valid_f=jnp.ones(safe_src.shape, dtype=jnp.float32),
    )
    token_block = jnp.array(
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
        ],
        dtype=jnp.float32,
    )
    route_block = jnp.array([[0.25, 0.75], [1.25, 1.75]], dtype=jnp.float32)

    observed_tokens = source_push_mlp._source_push_mlp_accumulate_expert_major_to_source_tokens(
        source_tokens,
        safe_src,
        safe_token,
        token_block,
    )
    observed_routes = source_push_mlp._source_push_mlp_accumulate_expert_major_to_source_routes(
        source_routes,
        safe_src,
        safe_token,
        safe_slot,
        route_block,
    )
    observed_source_grads = source_push_mlp._source_push_mlp_return_expert_backward_to_sources(
        source_tokens,
        source_routes,
        route_indices,
        token_block,
        route_block,
    )

    expected_tokens = np.zeros((EP_SIZE, TOKENS_PER_RANK, HIDDEN_DIM), dtype=np.float32)
    expected_tokens[0, 2] = np.array([5.0, 7.0, 9.0], dtype=np.float32)
    expected_tokens[1, 1] = np.array([7.0, 8.0, 9.0], dtype=np.float32)
    expected_tokens[1, 3] = np.array([10.0, 11.0, 12.0], dtype=np.float32)
    expected_routes = np.zeros((EP_SIZE, TOKENS_PER_RANK, TOPK), dtype=np.float32)
    expected_routes[0, 2, 1] = 1.0
    expected_routes[1, 1, 0] = 1.25
    expected_routes[1, 3, 1] = 1.75

    np.testing.assert_allclose(np.asarray(observed_tokens), expected_tokens, atol=0, rtol=0)
    np.testing.assert_allclose(np.asarray(observed_routes), expected_routes, atol=0, rtol=0)
    assert observed_source_grads._fields == ("dx", "d_route_weights")
    np.testing.assert_allclose(np.asarray(observed_source_grads.dx), expected_tokens, atol=0, rtol=0)
    np.testing.assert_allclose(np.asarray(observed_source_grads.d_route_weights), expected_routes, atol=0, rtol=0)


def test_source_push_mlp_accumulate_expert_weight_gradients_places_local_expert_blocks():
    dw13 = jnp.zeros((EP_SIZE, EXPERTS_PER_RANK, HIDDEN_DIM, 2 * INTERMEDIATE_DIM), dtype=jnp.float32)
    dw2 = jnp.zeros((EP_SIZE, EXPERTS_PER_RANK, INTERMEDIATE_DIM, HIDDEN_DIM), dtype=jnp.float32)
    expert = 1
    dw13_block = jnp.arange(EP_SIZE * HIDDEN_DIM * 2 * INTERMEDIATE_DIM, dtype=jnp.float32).reshape(
        EP_SIZE,
        HIDDEN_DIM,
        2 * INTERMEDIATE_DIM,
    )
    dw2_block = (jnp.arange(EP_SIZE * INTERMEDIATE_DIM * HIDDEN_DIM, dtype=jnp.float32) + 100).reshape(
        EP_SIZE,
        INTERMEDIATE_DIM,
        HIDDEN_DIM,
    )

    observed_dw13, observed_dw2 = source_push_mlp._source_push_mlp_accumulate_expert_weight_gradients(
        dw13,
        dw2,
        expert,
        dw13_block,
        dw2_block,
    )

    expected_dw13 = np.zeros(dw13.shape, dtype=np.float32)
    expected_dw2 = np.zeros(dw2.shape, dtype=np.float32)
    expected_dw13[:, expert] = np.asarray(dw13_block)
    expected_dw2[:, expert] = np.asarray(dw2_block)
    np.testing.assert_allclose(np.asarray(observed_dw13), expected_dw13, atol=0, rtol=0)
    np.testing.assert_allclose(np.asarray(observed_dw2), expected_dw2, atol=0, rtol=0)


def test_source_push_moe_mlp_matches_independent_loop_reference():
    x, route_assignments, route_weights, w13, w2 = _small_mlp_inputs()
    route_table = _route_table(route_assignments, route_weights)

    observed = source_push_moe_mlp_reference(route_table, x, route_weights, w13, w2)
    observed_from_api, dropped_routes = source_push_moe_mlp(
        x,
        route_assignments,
        route_weights,
        w13,
        w2,
        ep_size=EP_SIZE,
        experts_per_rank=EXPERTS_PER_RANK,
        block_m=BLOCK_M,
        capacity_factor=2.0,
    )
    expected = _naive_source_push_moe_mlp(x, route_assignments, route_weights, w13, w2)

    assert int(dropped_routes) == 0
    np.testing.assert_allclose(np.asarray(observed), np.asarray(expected), atol=1e-6)
    np.testing.assert_allclose(np.asarray(observed_from_api), np.asarray(expected), atol=1e-6)


def test_source_push_moe_mlp_rejects_route_weight_shape_mismatch_at_boundary():
    x, route_assignments, route_weights, w13, w2 = _small_mlp_inputs()

    with pytest.raises(ValueError, match="route_assignments shape"):
        source_push_moe_mlp(
            x,
            route_assignments,
            route_weights[..., :1],
            w13,
            w2,
            ep_size=EP_SIZE,
            experts_per_rank=EXPERTS_PER_RANK,
            block_m=BLOCK_M,
            capacity_factor=2.0,
        )


def test_source_push_moe_mlp_rejects_staged_implementation_without_config():
    x, route_assignments, route_weights, w13, w2 = _small_mlp_inputs()

    with pytest.raises(ValueError, match="requires an explicit PushInboxConfig"):
        source_push_moe_mlp(
            x,
            route_assignments,
            route_weights,
            w13,
            w2,
            ep_size=EP_SIZE,
            experts_per_rank=EXPERTS_PER_RANK,
            block_m=BLOCK_M,
            capacity_factor=2.0,
            implementation=SOURCE_PUSH_MLP_IMPLEMENTATION_PALLAS_MGPU,
        )


def test_source_push_moe_mlp_rejects_staged_execution_mode_without_config():
    x, route_assignments, route_weights, w13, w2 = _small_mlp_inputs()

    with pytest.raises(ValueError, match="execution_mode is only used with config"):
        source_push_moe_mlp(
            x,
            route_assignments,
            route_weights,
            w13,
            w2,
            ep_size=EP_SIZE,
            experts_per_rank=EXPERTS_PER_RANK,
            block_m=BLOCK_M,
            capacity_factor=2.0,
            execution_mode=source_push_forward.FORWARD_EXECUTION_SINGLE_JIT,
        )


def test_source_push_moe_mlp_with_config_delegates_to_from_plan_boundary(monkeypatch):
    x, route_assignments, route_weights, w13, w2 = _small_mlp_inputs()
    config = _small_forward_config()
    calls = {}

    def fake_source_push_moe_mlp_from_plan(
        config_arg,
        host_inputs_arg,
        route_table_arg,
        x_arg,
        route_weights_arg,
        w13_arg,
        w2_arg,
        *,
        implementation,
        execution_mode,
        mesh,
    ):
        calls["config"] = config_arg
        calls["host_inputs"] = host_inputs_arg
        calls["route_table"] = route_table_arg
        calls["x"] = x_arg
        calls["route_weights"] = route_weights_arg
        calls["w13"] = w13_arg
        calls["w2"] = w2_arg
        calls["implementation"] = implementation
        calls["execution_mode"] = execution_mode
        calls["mesh"] = mesh
        return x_arg + 3.0, host_inputs_arg.plan.dropped_routes

    monkeypatch.setattr(source_push_mlp, "source_push_moe_mlp_from_plan", fake_source_push_moe_mlp_from_plan)

    observed, dropped_routes = source_push_moe_mlp(
        x,
        route_assignments,
        route_weights,
        w13,
        w2,
        ep_size=EP_SIZE,
        experts_per_rank=EXPERTS_PER_RANK,
        block_m=BLOCK_M,
        capacity_factor=2.0,
        entries_per_dst=config.entries_per_rank,
        implementation=SOURCE_PUSH_MLP_IMPLEMENTATION_PALLAS_MGPU,
        config=config,
    )

    assert calls["config"] is config
    assert calls["host_inputs"].plan.assignment_ids.shape[:3] == (EP_SIZE, EP_SIZE, config.entries_per_rank)
    assert calls["route_table"].ep_size == EP_SIZE
    assert calls["route_table"].topk == TOPK
    assert calls["implementation"] == SOURCE_PUSH_MLP_IMPLEMENTATION_PALLAS_MGPU
    assert calls["execution_mode"] == source_push_mlp.FORWARD_EXECUTION_STAGED_HOST_SYNC
    assert calls["mesh"] is None
    np.testing.assert_array_equal(np.asarray(calls["x"]), np.asarray(x))
    np.testing.assert_array_equal(np.asarray(calls["route_weights"]), np.asarray(route_weights))
    np.testing.assert_array_equal(np.asarray(calls["w13"]), np.asarray(w13))
    np.testing.assert_array_equal(np.asarray(calls["w2"]), np.asarray(w2))
    np.testing.assert_array_equal(np.asarray(observed), np.asarray(x + 3.0))
    assert int(dropped_routes) == 0


def test_source_push_moe_mlp_rejects_traced_route_assignments_with_static_plan_message():
    x, route_assignments, route_weights, w13, w2 = _small_mlp_inputs()

    def loss(assignments):
        y, dropped_routes = source_push_moe_mlp(
            x,
            assignments,
            route_weights,
            w13,
            w2,
            ep_size=EP_SIZE,
            experts_per_rank=EXPERTS_PER_RANK,
            block_m=BLOCK_M,
            capacity_factor=2.0,
        )
        return jnp.sum(y.astype(jnp.float32)) + dropped_routes.astype(jnp.float32) * 0.0

    with pytest.raises(ValueError, match="route_assignments must be concrete/static"):
        jax.jit(loss)(route_assignments)


def test_source_push_moe_mlp_from_plan_rejects_route_weight_shape_mismatch_at_boundary():
    config, host_inputs, route_table, x, route_weights, w13, w2 = _small_forward_plan_inputs()

    with pytest.raises(ValueError, match="route_weights shape"):
        source_push_moe_mlp_from_plan(
            config,
            host_inputs,
            route_table,
            x,
            route_weights[..., :1],
            w13,
            w2,
            implementation=SOURCE_PUSH_MLP_IMPLEMENTATION_REFERENCE,
        )


def test_source_push_moe_mlp_from_plan_rejects_route_table_plan_mismatch():
    config, _host_inputs, route_table, x, route_weights, w13, w2 = _small_forward_plan_inputs()
    _x, route_assignments, _route_weights, _w13, _w2 = _small_mlp_inputs()
    mismatched_assignments = route_assignments.at[0, 0, 0].set(1).at[0, 1, 0].set(0)
    mismatched_host_inputs = source_push_forward.make_source_push_forward_inputs(
        config,
        x,
        mismatched_assignments,
        route_weights,
        w13,
        w2,
    )

    with pytest.raises(ValueError, match="route_table field"):
        source_push_moe_mlp_from_plan(
            config,
            mismatched_host_inputs,
            route_table,
            x,
            route_weights,
            w13,
            w2,
            implementation=SOURCE_PUSH_MLP_IMPLEMENTATION_REFERENCE,
        )


@pytest.mark.parametrize(
    "field_name",
    ("send_meta", "expert_base"),
)
def test_source_push_moe_mlp_from_plan_rejects_stale_host_layout_metadata(field_name):
    config, host_inputs, route_table, x, route_weights, w13, w2 = _small_forward_plan_inputs()
    bad_value = np.array(getattr(host_inputs, field_name), copy=True)
    bad_value.flat[0] += 1
    bad_host_inputs = replace(host_inputs, **{field_name: bad_value})

    with pytest.raises(ValueError, match=f"host_inputs.{field_name}"):
        source_push_moe_mlp_from_plan(
            config,
            bad_host_inputs,
            route_table,
            x,
            route_weights,
            w13,
            w2,
            implementation=SOURCE_PUSH_MLP_IMPLEMENTATION_REFERENCE,
        )


@pytest.mark.parametrize(
    "field_name",
    ("queue_dst_ord", "queue_entry", "queue_row", "route_valid_mask"),
)
def test_source_push_moe_mlp_from_plan_rejects_stale_host_inverse_metadata(field_name):
    config, host_inputs, route_table, x, route_weights, w13, w2 = _small_forward_plan_inputs()
    bad_value = np.array(getattr(host_inputs, field_name), copy=True)
    if bad_value.dtype == np.bool_:
        bad_value.flat[0] = not bool(bad_value.flat[0])
    else:
        bad_value.flat[0] += 1
    bad_host_inputs = replace(host_inputs, **{field_name: bad_value})

    with pytest.raises(ValueError, match=f"host_inputs.{field_name}"):
        source_push_moe_mlp_from_plan(
            config,
            bad_host_inputs,
            route_table,
            x,
            route_weights,
            w13,
            w2,
            implementation=SOURCE_PUSH_MLP_IMPLEMENTATION_REFERENCE,
        )


def test_source_push_moe_mlp_from_plan_rejects_stale_route_combine_weight_shape():
    config, host_inputs, route_table, x, route_weights, w13, w2 = _small_forward_plan_inputs()
    bad_host_inputs = replace(host_inputs, route_combine_weights=host_inputs.route_combine_weights[..., :1])

    with pytest.raises(ValueError, match="host_inputs.route_combine_weights shape"):
        source_push_moe_mlp_from_plan(
            config,
            bad_host_inputs,
            route_table,
            x,
            route_weights,
            w13,
            w2,
            implementation=SOURCE_PUSH_MLP_IMPLEMENTATION_REFERENCE,
        )


def test_source_push_moe_mlp_from_plan_uses_route_weight_argument_not_host_weight_values():
    config, host_inputs, route_table, x, route_weights, w13, w2 = _small_forward_plan_inputs()
    stale_host_inputs = replace(
        host_inputs,
        h_route_weights=np.full_like(host_inputs.h_route_weights, -5.0),
        route_combine_weights=np.full_like(host_inputs.route_combine_weights, 7.0),
    )
    cotangent = jnp.linspace(-0.5, 0.7, x.size, dtype=jnp.float32).reshape(x.shape)
    dynamic_route_weights = route_weights * 0.25 + 0.125

    def reference_loss(route_weights_arg):
        y = source_push_moe_mlp_reference(route_table, x, route_weights_arg, w13, w2)
        return jnp.sum(y * cotangent)

    def from_plan_loss(route_weights_arg):
        y, dropped_routes = source_push_moe_mlp_from_plan(
            config,
            stale_host_inputs,
            route_table,
            x,
            route_weights_arg,
            w13,
            w2,
            implementation=SOURCE_PUSH_MLP_IMPLEMENTATION_REFERENCE,
        )
        assert int(dropped_routes) == 0
        return jnp.sum(y * cotangent)

    reference_value, reference_grad = jax.value_and_grad(reference_loss)(dynamic_route_weights)
    from_plan_value, from_plan_grad = jax.value_and_grad(from_plan_loss)(dynamic_route_weights)

    np.testing.assert_allclose(np.asarray(from_plan_value), np.asarray(reference_value), atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(from_plan_grad), np.asarray(reference_grad), atol=1e-5, rtol=1e-5)
    assert float(jnp.max(jnp.abs(from_plan_grad))) > 0.0


def test_source_push_moe_mlp_from_plan_uses_dynamic_arrays_not_host_placeholders():
    config, host_inputs, route_table, x, route_weights, w13, w2 = _small_forward_plan_inputs()
    stale_host_inputs = replace(
        host_inputs,
        x=np.full_like(host_inputs.x, 9.0),
        w_gate_up=np.full_like(host_inputs.w_gate_up, -7.0),
        w_down=np.full_like(host_inputs.w_down, 11.0),
        h_route_weights=np.full_like(host_inputs.h_route_weights, -5.0),
        route_combine_weights=np.full_like(host_inputs.route_combine_weights, 7.0),
    )
    cotangent = jnp.linspace(-0.5, 0.7, x.size, dtype=jnp.float32).reshape(x.shape)
    dynamic_x = x + 0.13
    dynamic_route_weights = route_weights * 0.25 + 0.125
    dynamic_w13 = w13 - 0.2
    dynamic_w2 = w2 + 0.17

    def reference_loss(x_arg, route_weights_arg, w13_arg, w2_arg):
        y = source_push_moe_mlp_reference(route_table, x_arg, route_weights_arg, w13_arg, w2_arg)
        return jnp.sum(y * cotangent)

    def from_plan_loss(x_arg, route_weights_arg, w13_arg, w2_arg):
        y, dropped_routes = source_push_moe_mlp_from_plan(
            config,
            stale_host_inputs,
            route_table,
            x_arg,
            route_weights_arg,
            w13_arg,
            w2_arg,
            implementation=SOURCE_PUSH_MLP_IMPLEMENTATION_REFERENCE,
        )
        assert int(dropped_routes) == 0
        return jnp.sum(y * cotangent)

    reference_value, reference_grads = jax.value_and_grad(reference_loss, argnums=(0, 1, 2, 3))(
        dynamic_x,
        dynamic_route_weights,
        dynamic_w13,
        dynamic_w2,
    )
    from_plan_value, from_plan_grads = jax.value_and_grad(from_plan_loss, argnums=(0, 1, 2, 3))(
        dynamic_x,
        dynamic_route_weights,
        dynamic_w13,
        dynamic_w2,
    )

    np.testing.assert_allclose(np.asarray(from_plan_value), np.asarray(reference_value), atol=1e-6, rtol=1e-6)
    for observed, expected in zip(from_plan_grads, reference_grads, strict=True):
        np.testing.assert_allclose(np.asarray(observed), np.asarray(expected), atol=1e-5, rtol=1e-5)
    assert float(jnp.max(jnp.abs(from_plan_grads[1]))) > 0.0


def test_source_push_moe_mlp_from_plan_rejects_stale_config_queue_shape():
    config, host_inputs, route_table, x, route_weights, w13, w2 = _small_forward_plan_inputs()
    bad_config = replace(config, entries_per_rank=config.entries_per_rank + 1)

    with pytest.raises(ValueError, match="host_inputs.plan assignment_ids shape"):
        source_push_moe_mlp_from_plan(
            bad_config,
            host_inputs,
            route_table,
            x,
            route_weights,
            w13,
            w2,
            implementation=SOURCE_PUSH_MLP_IMPLEMENTATION_REFERENCE,
        )


def test_source_push_moe_mlp_from_plan_rejects_unknown_execution_mode_at_boundary():
    config, host_inputs, route_table, x, route_weights, w13, w2 = _small_forward_plan_inputs()

    with pytest.raises(ValueError, match="unknown execution_mode"):
        source_push_moe_mlp_from_plan(
            config,
            host_inputs,
            route_table,
            x,
            route_weights,
            w13,
            w2,
            implementation=SOURCE_PUSH_MLP_IMPLEMENTATION_REFERENCE,
            execution_mode="not_a_mode",
        )


def test_source_push_moe_mlp_from_plan_rejects_pallas_mesh_that_does_not_match_plan():
    config, host_inputs, route_table, x, route_weights, w13, w2 = _small_forward_plan_inputs()
    bad_mesh = Mesh(np.asarray(jax.devices()[:1], dtype=object), ("expert",))

    with pytest.raises(ValueError, match="mesh axis 'expert' size 1 must match config ep_size 2"):
        source_push_moe_mlp_from_plan(
            config,
            host_inputs,
            route_table,
            x,
            route_weights,
            w13,
            w2,
            implementation=SOURCE_PUSH_MLP_IMPLEMENTATION_PALLAS_MGPU,
            mesh=bad_mesh,
        )


def test_source_push_moe_mlp_from_plan_rejects_exact_layout_tail_blocks():
    config = PushInboxConfig(
        ep_size=2,
        entries_per_rank=1,
        inbox_slots=1,
        hidden_dim=2,
        intermediate_dim=1,
        block_m=2,
        block_k=1,
        block_n=1,
        experts_per_rank=1,
        send_worker_programs_per_peer=1,
        worker_programs_per_peer=4,
        send_pipeline_depth=1,
        n_groups_per_job=1,
        routing="balanced",
        tokens_per_rank=3,
        topk=1,
        capacity_factor=2.0,
    )
    x = jnp.arange(12, dtype=jnp.float32).reshape(2, 3, 2) * 0.01
    route_assignments = jnp.array([[[0], [1], [0]], [[1], [0], [1]]], dtype=jnp.int32)
    route_weights = jnp.ones((2, 3, 1), dtype=jnp.float32)
    w13 = jnp.arange(8, dtype=jnp.float32).reshape(2, 1, 2, 2) * 0.01
    w2 = jnp.arange(4, dtype=jnp.float32).reshape(2, 1, 1, 2) * 0.01
    host_inputs = source_push_forward.make_source_push_forward_inputs(
        config,
        x,
        route_assignments,
        route_weights,
        w13,
        w2,
    )
    route_table = source_push_mlp_route_table_from_plan(
        host_inputs.plan,
        src_base_by_expert=host_inputs.src_base_by_expert,
    )
    bad_host_inputs = replace(host_inputs, use_exact_expert_major=True)

    with pytest.raises(ValueError, match="exact source-push MLP layout requires block_m-aligned live blocks"):
        source_push_moe_mlp_from_plan(
            config,
            bad_host_inputs,
            route_table,
            x,
            route_weights,
            w13,
            w2,
            implementation=SOURCE_PUSH_MLP_IMPLEMENTATION_REFERENCE,
        )


def test_source_push_moe_mlp_duplicate_topk_routes_accumulate_deterministically():
    ep_size = 2
    experts_per_rank = 1
    block_m = 2
    x = jnp.array(
        [
            [[1.0, -0.5], [0.25, 0.75]],
            [[-0.25, 0.5], [0.5, -1.0]],
        ],
        dtype=jnp.float32,
    )
    route_assignments = jnp.array(
        [
            [[0, 0], [1, 0]],
            [[1, 1], [0, 1]],
        ],
        dtype=jnp.int32,
    )
    route_weights = jnp.array(
        [
            [[0.25, 0.75], [0.5, 0.125]],
            [[0.6, 0.4], [0.3, 0.2]],
        ],
        dtype=jnp.float32,
    )
    w13 = jnp.array(
        [
            [[[0.2, -0.4], [0.5, 0.3]]],
            [[[-0.1, 0.6], [0.7, -0.2]]],
        ],
        dtype=jnp.float32,
    )
    w2 = jnp.array(
        [
            [[[1.25, -0.5]]],
            [[[-0.75, 0.4]]],
        ],
        dtype=jnp.float32,
    )
    route_table, dropped_routes = build_source_push_mlp_route_table(
        route_assignments,
        ep_size=ep_size,
        experts_per_rank=experts_per_rank,
        block_m=block_m,
        capacity_factor=4.0,
    )

    observed = source_push_moe_mlp_custom_vjp(route_table, x, route_weights, w13, w2)
    observed_from_api, api_dropped_routes = source_push_moe_mlp(
        x,
        route_assignments,
        route_weights,
        w13,
        w2,
        ep_size=ep_size,
        experts_per_rank=experts_per_rank,
        block_m=block_m,
        capacity_factor=4.0,
    )
    expected = _naive_source_push_moe_mlp(x, route_assignments, route_weights, w13, w2)

    assert int(dropped_routes) == 0
    assert int(api_dropped_routes) == 0
    np.testing.assert_allclose(np.asarray(observed), np.asarray(expected), atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(observed_from_api), np.asarray(expected), atol=1e-6, rtol=1e-6)

    cotangent = jnp.array(
        [
            [[0.2, -0.3], [0.4, 0.1]],
            [[-0.25, 0.5], [0.3, -0.2]],
        ],
        dtype=jnp.float32,
    )

    def loss(route_weights_arg):
        y = source_push_moe_mlp_custom_vjp(route_table, x, route_weights_arg, w13, w2)
        return jnp.sum(y * cotangent)

    d_route_weights = jax.grad(loss)(route_weights)
    np.testing.assert_allclose(
        np.asarray(d_route_weights[0, 0, 0]),
        np.asarray(d_route_weights[0, 0, 1]),
        atol=1e-6,
        rtol=1e-6,
    )
    assert float(jnp.max(jnp.abs(d_route_weights))) > 0.0


def test_source_push_moe_mlp_dropped_routes_have_zero_route_weight_gradient():
    ep_size = 2
    experts_per_rank = 1
    block_m = 2
    capacity_factor = 0.25
    x = jnp.array(
        [
            [[1.0, 2.0], [-0.5, 0.25]],
            [[0.75, -1.5], [0.5, 0.5]],
        ],
        dtype=jnp.float32,
    )
    route_assignments = jnp.zeros((ep_size, 2, 2), dtype=jnp.int32)
    route_weights = jnp.array(
        [
            [[0.2, 0.3], [0.4, 0.5]],
            [[0.6, 0.7], [0.8, 0.9]],
        ],
        dtype=jnp.float32,
    )
    w13 = jnp.array(
        [
            [[[0.2, -0.4], [0.5, 0.3]]],
            [[[-0.1, 0.6], [0.7, -0.2]]],
        ],
        dtype=jnp.float32,
    )
    w2 = jnp.array(
        [
            [[[1.25, -0.5]]],
            [[[-0.75, 0.4]]],
        ],
        dtype=jnp.float32,
    )
    route_table, dropped_routes = build_source_push_mlp_route_table(
        route_assignments,
        ep_size=ep_size,
        experts_per_rank=experts_per_rank,
        block_m=block_m,
        capacity_factor=capacity_factor,
    )
    observed, api_dropped_routes = source_push_moe_mlp(
        x,
        route_assignments,
        route_weights,
        w13,
        w2,
        ep_size=ep_size,
        experts_per_rank=experts_per_rank,
        block_m=block_m,
        capacity_factor=capacity_factor,
    )
    expected = source_push_moe_mlp_reference(route_table, x, route_weights, w13, w2)

    accepted_mask = _accepted_route_mask(route_table, route_weights.shape)
    assert int(dropped_routes) == route_weights.size - int(np.sum(accepted_mask))
    assert int(api_dropped_routes) == int(dropped_routes)
    assert int(dropped_routes) > 0
    np.testing.assert_allclose(np.asarray(observed), np.asarray(expected), atol=1e-6, rtol=1e-6)

    cotangent = jnp.array(
        [
            [[0.2, -0.3], [0.4, 0.1]],
            [[-0.25, 0.5], [0.3, -0.2]],
        ],
        dtype=jnp.float32,
    )

    def loss(route_weights_arg):
        y, _ = source_push_moe_mlp(
            x,
            route_assignments,
            route_weights_arg,
            w13,
            w2,
            ep_size=ep_size,
            experts_per_rank=experts_per_rank,
            block_m=block_m,
            capacity_factor=capacity_factor,
        )
        return jnp.sum(y * cotangent)

    d_route_weights = np.asarray(jax.grad(loss)(route_weights))
    np.testing.assert_array_equal(d_route_weights[~accepted_mask], np.zeros_like(d_route_weights[~accepted_mask]))
    assert np.any(np.abs(d_route_weights[accepted_mask]) > 1e-8)


def test_source_push_moe_mlp_api_gradients_do_not_capture_route_weights_in_plan():
    x, route_assignments, route_weights, w13, w2 = _small_mlp_inputs()
    route_table = _route_table(route_assignments, route_weights)

    def reference_loss(x_arg, route_weights_arg, w13_arg, w2_arg):
        y = source_push_moe_mlp_reference(route_table, x_arg, route_weights_arg, w13_arg, w2_arg)
        return jnp.sum(y.astype(jnp.float32))

    def api_loss(x_arg, route_weights_arg, w13_arg, w2_arg):
        y, dropped_routes = source_push_moe_mlp(
            x_arg,
            route_assignments,
            route_weights_arg,
            w13_arg,
            w2_arg,
            ep_size=EP_SIZE,
            experts_per_rank=EXPERTS_PER_RANK,
            block_m=BLOCK_M,
            capacity_factor=2.0,
        )
        assert int(dropped_routes) == 0
        return jnp.sum(y.astype(jnp.float32))

    reference_value, reference_grads = jax.value_and_grad(reference_loss, argnums=(0, 1, 2, 3))(
        x,
        route_weights,
        w13,
        w2,
    )
    api_value, api_grads = jax.value_and_grad(api_loss, argnums=(0, 1, 2, 3))(
        x,
        route_weights,
        w13,
        w2,
    )

    np.testing.assert_allclose(np.asarray(api_value), np.asarray(reference_value), atol=1e-6, rtol=1e-6)
    for observed, expected in zip(api_grads, reference_grads, strict=True):
        np.testing.assert_allclose(np.asarray(observed), np.asarray(expected), atol=1e-5, rtol=1e-5)
    assert float(jnp.max(jnp.abs(api_grads[1]))) > 0.0


def test_source_push_moe_mlp_api_gradients_work_under_jit_for_static_routes():
    x, route_assignments, route_weights, w13, w2 = _small_mlp_inputs()
    route_table = _route_table(route_assignments, route_weights)

    def reference_loss(x_arg, route_weights_arg, w13_arg, w2_arg):
        y = source_push_moe_mlp_reference(route_table, x_arg, route_weights_arg, w13_arg, w2_arg)
        return jnp.sum(y.astype(jnp.float32))

    def api_loss(x_arg, route_weights_arg, w13_arg, w2_arg):
        y, dropped_routes = source_push_moe_mlp(
            x_arg,
            route_assignments,
            route_weights_arg,
            w13_arg,
            w2_arg,
            ep_size=EP_SIZE,
            experts_per_rank=EXPERTS_PER_RANK,
            block_m=BLOCK_M,
            capacity_factor=2.0,
        )
        return jnp.sum(y.astype(jnp.float32)) + dropped_routes.astype(jnp.float32) * 0.0

    reference_grads = jax.jit(jax.grad(reference_loss, argnums=(0, 1, 2, 3)))(x, route_weights, w13, w2)
    api_grads = jax.jit(jax.grad(api_loss, argnums=(0, 1, 2, 3)))(x, route_weights, w13, w2)

    for observed, expected in zip(api_grads, reference_grads, strict=True):
        np.testing.assert_allclose(np.asarray(observed), np.asarray(expected), atol=1e-5, rtol=1e-5)
    assert float(jnp.max(jnp.abs(api_grads[1]))) > 0.0


def test_source_push_moe_mlp_with_config_reference_matches_reference_gradients():
    config, _host_inputs, route_table, x, route_weights, w13, w2 = _small_forward_plan_inputs()
    _x, route_assignments, _route_weights, _w13, _w2 = _small_mlp_inputs()
    cotangent = jnp.linspace(-0.5, 0.7, x.size, dtype=jnp.float32).reshape(x.shape)

    def reference_loss(x_arg, route_weights_arg, w13_arg, w2_arg):
        y = source_push_moe_mlp_reference(route_table, x_arg, route_weights_arg, w13_arg, w2_arg)
        return jnp.sum(y * cotangent)

    def api_loss(x_arg, route_weights_arg, w13_arg, w2_arg):
        y, dropped_routes = source_push_moe_mlp(
            x_arg,
            route_assignments,
            route_weights_arg,
            w13_arg,
            w2_arg,
            ep_size=EP_SIZE,
            experts_per_rank=EXPERTS_PER_RANK,
            block_m=BLOCK_M,
            capacity_factor=config.capacity_factor,
            entries_per_dst=config.entries_per_rank,
            implementation=SOURCE_PUSH_MLP_IMPLEMENTATION_REFERENCE,
            config=config,
        )
        assert int(dropped_routes) == 0
        return jnp.sum(y * cotangent)

    reference_value, reference_grads = jax.value_and_grad(reference_loss, argnums=(0, 1, 2, 3))(
        x,
        route_weights,
        w13,
        w2,
    )
    api_value, api_grads = jax.value_and_grad(api_loss, argnums=(0, 1, 2, 3))(
        x,
        route_weights,
        w13,
        w2,
    )

    np.testing.assert_allclose(np.asarray(api_value), np.asarray(reference_value), atol=1e-6, rtol=1e-6)
    for observed, expected in zip(api_grads, reference_grads, strict=True):
        np.testing.assert_allclose(np.asarray(observed), np.asarray(expected), atol=1e-5, rtol=1e-5)
    assert float(jnp.max(jnp.abs(api_grads[1]))) > 0.0


def test_source_push_moe_mlp_with_config_pallas_path_uses_compact_h_boundary(monkeypatch):
    config, _host_inputs, route_table, x, route_weights, w13, w2 = _small_forward_plan_inputs()
    _x, route_assignments, _route_weights, _w13, _w2 = _small_mlp_inputs()
    cotangent = jnp.linspace(-0.5, 0.7, x.size, dtype=jnp.float32).reshape(x.shape)
    calls = []

    def fail_flat_h_conversion(*_args, **_kwargs):
        raise AssertionError("Pallas compact-H MLP path must not use flat-H conversion")

    monkeypatch.setattr(source_push_mlp, "_source_push_mlp_flat_h_to_compact", fail_flat_h_conversion)
    monkeypatch.setattr(source_push_mlp, "_source_push_mlp_compact_h_to_flat", fail_flat_h_conversion)

    def fake_source_push_forward_with_compact_h_from_plan(
        config_arg,
        host_inputs_arg,
        x_arg,
        h_route_weights_arg,
        w13_arg,
        w2_arg,
        *,
        compact_expert_capacity,
        mesh,
    ):
        staged_route_table = source_push_mlp_route_table_from_plan(
            host_inputs_arg.plan,
            src_base_by_expert=host_inputs_arg.src_base_by_expert,
        )
        y, h = source_push_moe_mlp_reference_with_h(
            staged_route_table,
            x_arg,
            route_weights,
            w13_arg,
            w2_arg,
        )
        if compact_expert_capacity > staged_route_table.expert_capacity:
            h = jnp.pad(
                h,
                ((0, 0), (0, 0), (0, compact_expert_capacity - staged_route_table.expert_capacity), (0, 0)),
            )
        calls.append(
            {
                "config": config_arg,
                "mesh": mesh,
                "route_table": staged_route_table,
                "compact_expert_capacity": compact_expert_capacity,
                "h_route_weights": h_route_weights_arg,
            }
        )
        return y, h, host_inputs_arg.plan.dropped_routes

    monkeypatch.setattr(
        source_push_mlp,
        "source_push_forward_with_compact_h_from_plan",
        fake_source_push_forward_with_compact_h_from_plan,
    )

    def reference_loss(x_arg, route_weights_arg, w13_arg, w2_arg):
        y = source_push_moe_mlp_reference(route_table, x_arg, route_weights_arg, w13_arg, w2_arg)
        return jnp.sum(y * cotangent)

    def api_loss(x_arg, route_weights_arg, w13_arg, w2_arg):
        y, dropped_routes = source_push_moe_mlp(
            x_arg,
            route_assignments,
            route_weights_arg,
            w13_arg,
            w2_arg,
            ep_size=EP_SIZE,
            experts_per_rank=EXPERTS_PER_RANK,
            block_m=BLOCK_M,
            capacity_factor=config.capacity_factor,
            entries_per_dst=config.entries_per_rank,
            implementation=SOURCE_PUSH_MLP_IMPLEMENTATION_PALLAS_MGPU,
            config=config,
        )
        assert int(dropped_routes) == 0
        return jnp.sum(y * cotangent)

    reference_value, reference_grads = jax.value_and_grad(reference_loss, argnums=(0, 1, 2, 3))(
        x,
        route_weights,
        w13,
        w2,
    )
    api_value, api_grads = jax.value_and_grad(api_loss, argnums=(0, 1, 2, 3))(
        x,
        route_weights,
        w13,
        w2,
    )

    assert calls
    assert all(call["config"] is config for call in calls)
    assert all(call["mesh"] is None for call in calls)
    assert all(call["route_table"].ep_size == EP_SIZE for call in calls)
    assert all(call["route_table"].topk == TOPK for call in calls)
    expected_route_weights = source_push_mlp._source_push_mlp_route_weights_to_all_expert_major(
        calls[0]["route_table"],
        route_weights,
    )
    for call in calls:
        assert call["compact_expert_capacity"] >= call["route_table"].expert_capacity
        np.testing.assert_allclose(
            np.asarray(call["h_route_weights"][:, :, : call["route_table"].expert_capacity]),
            np.asarray(expected_route_weights),
            atol=0,
            rtol=0,
        )
        assert call["h_route_weights"].shape[:2] == (EP_SIZE, EXPERTS_PER_RANK)
    np.testing.assert_allclose(np.asarray(api_value), np.asarray(reference_value), atol=1e-6, rtol=1e-6)
    for observed, expected in zip(api_grads, reference_grads, strict=True):
        np.testing.assert_allclose(np.asarray(observed), np.asarray(expected), atol=1e-5, rtol=1e-5)
    assert float(jnp.max(jnp.abs(api_grads[1]))) > 0.0


def test_source_push_moe_mlp_custom_vjp_matches_reference_gradients():
    x, route_assignments, route_weights, w13, w2 = _small_mlp_inputs()
    route_table = _route_table(route_assignments, route_weights)
    cotangent = jnp.linspace(-0.5, 0.7, x.size, dtype=jnp.float32).reshape(x.shape)

    def reference_loss(x_arg, route_weights_arg, w13_arg, w2_arg):
        y = source_push_moe_mlp_reference(route_table, x_arg, route_weights_arg, w13_arg, w2_arg)
        return jnp.sum(y * cotangent)

    def custom_vjp_loss(x_arg, route_weights_arg, w13_arg, w2_arg):
        y = source_push_moe_mlp_custom_vjp(route_table, x_arg, route_weights_arg, w13_arg, w2_arg)
        return jnp.sum(y * cotangent)

    reference_grads = jax.grad(reference_loss, argnums=(0, 1, 2, 3))(x, route_weights, w13, w2)
    custom_vjp_grads = jax.grad(custom_vjp_loss, argnums=(0, 1, 2, 3))(x, route_weights, w13, w2)

    for observed, expected in zip(custom_vjp_grads, reference_grads, strict=True):
        np.testing.assert_allclose(np.asarray(observed), np.asarray(expected), atol=1e-5, rtol=1e-5)
    assert float(jnp.max(jnp.abs(custom_vjp_grads[1]))) > 0.0


def test_source_push_moe_mlp_saved_h_backward_returns_named_mlp_gradients():
    x, route_assignments, route_weights, w13, w2 = _small_mlp_inputs()
    route_table = _route_table(route_assignments, route_weights)
    cotangent = jnp.linspace(-0.5, 0.7, x.size, dtype=jnp.float32).reshape(x.shape)
    _, h, expert_route_weights = source_push_mlp._source_push_moe_mlp_reference_with_h_and_expert_route_weights(
        route_table,
        x,
        route_weights,
        w13,
        w2,
    )

    def reference_loss(x_arg, route_weights_arg, w13_arg, w2_arg):
        y = source_push_moe_mlp_reference(route_table, x_arg, route_weights_arg, w13_arg, w2_arg)
        return jnp.sum(y * cotangent)

    expected_dx, expected_d_route_weights, expected_dw13, expected_dw2 = jax.grad(
        reference_loss,
        argnums=(0, 1, 2, 3),
    )(x, route_weights, w13, w2)
    observed = source_push_mlp._source_push_moe_mlp_backward_from_h(
        route_table,
        x,
        expert_route_weights,
        w13,
        w2,
        h,
        cotangent,
    )

    np.testing.assert_allclose(np.asarray(observed.dx), np.asarray(expected_dx), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(
        np.asarray(observed.d_route_weights),
        np.asarray(expected_d_route_weights),
        atol=1e-5,
        rtol=1e-5,
    )
    np.testing.assert_allclose(np.asarray(observed.dw13), np.asarray(expected_dw13), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(np.asarray(observed.dw2), np.asarray(expected_dw2), atol=1e-5, rtol=1e-5)
    assert float(jnp.max(jnp.abs(observed.d_route_weights))) > 0.0


def test_source_push_moe_mlp_compact_h_backward_uses_shared_compact_dy_route(monkeypatch):
    x, route_assignments, route_weights, w13, w2 = _small_mlp_inputs()
    route_table = _route_table(route_assignments, route_weights)
    cotangent = jnp.linspace(-0.5, 0.7, x.size, dtype=jnp.float32).reshape(x.shape)
    _, h, expert_route_weights = source_push_mlp._source_push_moe_mlp_reference_with_h_and_expert_route_weights(
        route_table,
        x,
        route_weights,
        w13,
        w2,
    )
    original_dy_route = source_push_mlp._source_push_backward_dy_to_expert_major
    calls = []

    def tracking_dy_route(dy, source_rank_by_expert, token_id_by_expert, valid_by_expert, **kwargs):
        calls.append(
            (
                tuple(source_rank_by_expert.shape),
                tuple(token_id_by_expert.shape),
                tuple(valid_by_expert.shape),
                kwargs["implementation"],
            )
        )
        return original_dy_route(dy, source_rank_by_expert, token_id_by_expert, valid_by_expert, **kwargs)

    monkeypatch.setattr(source_push_mlp, "_source_push_backward_dy_to_expert_major", tracking_dy_route)

    observed = source_push_mlp._source_push_moe_mlp_backward_from_h(
        route_table,
        x,
        expert_route_weights,
        w13,
        w2,
        h,
        cotangent,
    )

    assert (
        calls
        == [
            (
                (EP_SIZE, 1, route_table.expert_capacity),
                (EP_SIZE, 1, route_table.expert_capacity),
                (EP_SIZE, 1, route_table.expert_capacity),
                source_push_mlp.SOURCE_PUSH_DY_ROUTE_IMPLEMENTATION_REFERENCE,
            )
        ]
        * EXPERTS_PER_RANK
    )
    assert float(jnp.max(jnp.abs(observed.d_route_weights))) > 0.0


def test_source_push_moe_mlp_route_weight_gradient_matches_h_formula():
    x, route_assignments, route_weights, w13, w2 = _small_mlp_inputs()
    route_table = _route_table(route_assignments, route_weights)
    cotangent = jnp.linspace(-0.5, 0.7, x.size, dtype=jnp.float32).reshape(x.shape)

    def custom_vjp_loss(route_weights_arg):
        y = source_push_moe_mlp_custom_vjp(route_table, x, route_weights_arg, w13, w2)
        return jnp.sum(y.astype(jnp.float32) * cotangent)

    observed = jax.grad(custom_vjp_loss)(route_weights)
    _, h = source_push_moe_mlp_reference_with_h(route_table, x, route_weights, w13, w2)

    expected = np.zeros(route_weights.shape, dtype=np.float32)
    h_host = np.asarray(h, dtype=np.float32)
    w2_host = np.asarray(w2, dtype=np.float32)
    dy_host = np.asarray(cotangent, dtype=np.float32)
    for route in range(route_table.source_rank.shape[0]):
        src = int(route_table.source_rank[route])
        token = int(route_table.token_id[route])
        slot = int(route_table.route_slot[route])
        dst = int(route_table.destination_rank[route])
        expert = int(route_table.local_expert[route])
        expert_row = int(route_table.expert_row[route])
        gate, up = np.split(h_host[dst, expert, expert_row], 2)
        activation = gate * (1.0 / (1.0 + np.exp(-gate))) * up
        d_weighted_activation = dy_host[src, token] @ w2_host[dst, expert].T
        expected[src, token, slot] += np.sum(d_weighted_activation * activation)

    np.testing.assert_allclose(np.asarray(observed), expected, atol=1e-5, rtol=1e-5)


def test_source_push_moe_mlp_custom_vjp_bf16_uses_saved_h_with_reference_tolerance():
    x, route_assignments, route_weights, w13, w2 = _small_mlp_inputs()
    x = x.astype(jnp.bfloat16)
    route_weights = route_weights.astype(jnp.bfloat16)
    w13 = w13.astype(jnp.bfloat16)
    w2 = w2.astype(jnp.bfloat16)
    route_table = _route_table(route_assignments, route_weights)
    cotangent = jnp.linspace(-0.5, 0.7, x.size, dtype=jnp.float32).reshape(x.shape)

    _, residual = source_push_mlp._source_push_moe_mlp_fwd(route_table, x, route_weights, w13, w2)

    assert residual.h.dtype == jnp.bfloat16
    for route in range(route_table.source_rank.shape[0]):
        src = int(route_table.source_rank[route])
        token = int(route_table.token_id[route])
        dst = int(route_table.destination_rank[route])
        expert = int(route_table.local_expert[route])
        expert_row = int(route_table.expert_row[route])
        expected_h = x[src, token].astype(jnp.float32) @ w13[dst, expert].astype(jnp.float32)
        expected_h = expected_h.astype(jnp.bfloat16)
        np.testing.assert_array_equal(
            np.asarray(residual.h[dst, expert, expert_row]),
            np.asarray(expected_h),
        )

    def reference_loss(x_arg, route_weights_arg, w13_arg, w2_arg):
        y = source_push_moe_mlp_reference(route_table, x_arg, route_weights_arg, w13_arg, w2_arg)
        return jnp.sum(y.astype(jnp.float32) * cotangent)

    def custom_vjp_loss(x_arg, route_weights_arg, w13_arg, w2_arg):
        y = source_push_moe_mlp_custom_vjp(route_table, x_arg, route_weights_arg, w13_arg, w2_arg)
        return jnp.sum(y.astype(jnp.float32) * cotangent)

    reference_value, reference_grads = jax.value_and_grad(reference_loss, argnums=(0, 1, 2, 3))(
        x,
        route_weights,
        w13,
        w2,
    )
    custom_value, custom_vjp_grads = jax.value_and_grad(custom_vjp_loss, argnums=(0, 1, 2, 3))(
        x,
        route_weights,
        w13,
        w2,
    )

    np.testing.assert_allclose(np.asarray(custom_value), np.asarray(reference_value), atol=1e-2, rtol=1e-2)
    for observed, expected in zip(custom_vjp_grads, reference_grads, strict=True):
        np.testing.assert_allclose(
            np.asarray(observed, dtype=np.float32),
            np.asarray(expected, dtype=np.float32),
            atol=2e-2,
            rtol=2e-2,
        )
    assert float(jnp.max(jnp.abs(custom_vjp_grads[1].astype(jnp.float32)))) > 0.0


def test_source_push_moe_mlp_from_plan_reference_custom_vjp_matches_reference_gradients():
    config, host_inputs, route_table, x, route_weights, w13, w2 = _small_forward_plan_inputs()
    cotangent = jnp.linspace(-0.5, 0.7, x.size, dtype=jnp.float32).reshape(x.shape)

    def reference_loss(x_arg, route_weights_arg, w13_arg, w2_arg):
        y = source_push_moe_mlp_reference(route_table, x_arg, route_weights_arg, w13_arg, w2_arg)
        return jnp.sum(y * cotangent)

    def from_plan_loss(x_arg, route_weights_arg, w13_arg, w2_arg):
        y, dropped_routes = source_push_moe_mlp_from_plan(
            config,
            host_inputs,
            route_table,
            x_arg,
            route_weights_arg,
            w13_arg,
            w2_arg,
            implementation=SOURCE_PUSH_MLP_IMPLEMENTATION_REFERENCE,
        )
        assert int(dropped_routes) == 0
        return jnp.sum(y * cotangent)

    reference_grads = jax.grad(reference_loss, argnums=(0, 1, 2, 3))(x, route_weights, w13, w2)
    from_plan_grads = jax.grad(from_plan_loss, argnums=(0, 1, 2, 3))(x, route_weights, w13, w2)

    for observed, expected in zip(from_plan_grads, reference_grads, strict=True):
        np.testing.assert_allclose(np.asarray(observed), np.asarray(expected), atol=1e-5, rtol=1e-5)
    assert float(jnp.max(jnp.abs(from_plan_grads[1]))) > 0.0


def test_source_push_moe_mlp_from_plan_reference_custom_vjp_matches_reference_gradients_under_jit():
    config, host_inputs, route_table, x, route_weights, w13, w2 = _small_forward_plan_inputs()
    cotangent = jnp.linspace(-0.5, 0.7, x.size, dtype=jnp.float32).reshape(x.shape)

    def reference_loss(x_arg, route_weights_arg, w13_arg, w2_arg):
        y = source_push_moe_mlp_reference(route_table, x_arg, route_weights_arg, w13_arg, w2_arg)
        return jnp.sum(y * cotangent)

    def from_plan_loss(x_arg, route_weights_arg, w13_arg, w2_arg):
        y, _ = source_push_moe_mlp_from_plan(
            config,
            host_inputs,
            route_table,
            x_arg,
            route_weights_arg,
            w13_arg,
            w2_arg,
            implementation=SOURCE_PUSH_MLP_IMPLEMENTATION_REFERENCE,
        )
        return jnp.sum(y * cotangent)

    reference_grads = jax.jit(jax.grad(reference_loss, argnums=(0, 1, 2, 3)))(x, route_weights, w13, w2)
    from_plan_grads = jax.jit(jax.grad(from_plan_loss, argnums=(0, 1, 2, 3)))(x, route_weights, w13, w2)

    for observed, expected in zip(from_plan_grads, reference_grads, strict=True):
        np.testing.assert_allclose(np.asarray(observed), np.asarray(expected), atol=1e-5, rtol=1e-5)
    assert float(jnp.max(jnp.abs(from_plan_grads[1]))) > 0.0


def test_source_push_moe_mlp_from_plan_pallas_path_uses_compact_h_boundary(monkeypatch):
    config, host_inputs, route_table, x, route_weights, w13, w2 = _small_forward_plan_inputs()
    cotangent = jnp.linspace(-0.5, 0.7, x.size, dtype=jnp.float32).reshape(x.shape)
    calls = []

    def fail_flat_h_conversion(*_args, **_kwargs):
        raise AssertionError("Pallas compact-H MLP path must not use flat-H conversion")

    monkeypatch.setattr(source_push_mlp, "_source_push_mlp_flat_h_to_compact", fail_flat_h_conversion)
    monkeypatch.setattr(source_push_mlp, "_source_push_mlp_compact_h_to_flat", fail_flat_h_conversion)

    def fake_source_push_forward_with_compact_h_from_plan(
        config_arg,
        host_inputs_arg,
        x_arg,
        h_route_weights_arg,
        w13_arg,
        w2_arg,
        *,
        compact_expert_capacity,
        mesh,
    ):
        calls.append(
            {
                "config": config_arg,
                "host_inputs": host_inputs_arg,
                "mesh": mesh,
                "compact_expert_capacity": compact_expert_capacity,
                "h_route_weights": h_route_weights_arg,
            }
        )
        y, h = source_push_moe_mlp_reference_with_h(
            route_table,
            x_arg,
            route_weights,
            w13_arg,
            w2_arg,
        )
        if compact_expert_capacity > route_table.expert_capacity:
            h = jnp.pad(
                h,
                ((0, 0), (0, 0), (0, compact_expert_capacity - route_table.expert_capacity), (0, 0)),
            )
        return y, h, host_inputs_arg.plan.dropped_routes

    monkeypatch.setattr(
        source_push_mlp,
        "source_push_forward_with_compact_h_from_plan",
        fake_source_push_forward_with_compact_h_from_plan,
    )

    def reference_loss(x_arg, route_weights_arg, w13_arg, w2_arg):
        y = source_push_moe_mlp_reference(route_table, x_arg, route_weights_arg, w13_arg, w2_arg)
        return jnp.sum(y * cotangent)

    def pallas_loss(x_arg, route_weights_arg, w13_arg, w2_arg):
        y, dropped_routes = source_push_moe_mlp_from_plan(
            config,
            host_inputs,
            route_table,
            x_arg,
            route_weights_arg,
            w13_arg,
            w2_arg,
            implementation=SOURCE_PUSH_MLP_IMPLEMENTATION_PALLAS_MGPU,
        )
        assert int(dropped_routes) == 0
        return jnp.sum(y * cotangent)

    reference_value, reference_grads = jax.value_and_grad(reference_loss, argnums=(0, 1, 2, 3))(
        x,
        route_weights,
        w13,
        w2,
    )
    pallas_value, pallas_grads = jax.value_and_grad(pallas_loss, argnums=(0, 1, 2, 3))(
        x,
        route_weights,
        w13,
        w2,
    )

    assert calls
    assert all(call["config"] is config for call in calls)
    assert all(call["host_inputs"] is host_inputs for call in calls)
    assert all(call["mesh"] is None for call in calls)
    expected_route_weights = source_push_mlp._source_push_mlp_route_weights_to_all_expert_major(
        route_table,
        route_weights,
    )
    for call in calls:
        assert call["compact_expert_capacity"] >= route_table.expert_capacity
        np.testing.assert_allclose(
            np.asarray(call["h_route_weights"][:, :, : route_table.expert_capacity]),
            np.asarray(expected_route_weights),
            atol=0,
            rtol=0,
        )
        assert call["h_route_weights"].shape[:2] == (EP_SIZE, EXPERTS_PER_RANK)
    np.testing.assert_allclose(np.asarray(pallas_value), np.asarray(reference_value), atol=1e-6, rtol=1e-6)
    for observed, expected in zip(pallas_grads, reference_grads, strict=True):
        np.testing.assert_allclose(np.asarray(observed), np.asarray(expected), atol=1e-5, rtol=1e-5)
    assert float(jnp.max(jnp.abs(pallas_grads[1]))) > 0.0


def test_source_push_moe_mlp_from_plan_pallas_backward_uses_compact_h_residual(monkeypatch):
    config, host_inputs, route_table, x, route_weights, w13, w2 = _small_forward_plan_inputs()
    cotangent = jnp.linspace(-0.5, 0.7, x.size, dtype=jnp.float32).reshape(x.shape)
    _, reference_h = source_push_moe_mlp_reference_with_h(
        route_table,
        x,
        route_weights,
        w13,
        w2,
    )
    h_delta = jnp.linspace(0.05, 0.25, reference_h.size, dtype=reference_h.dtype).reshape(reference_h.shape)
    staged_h = reference_h + h_delta
    assert staged_h.shape == (
        EP_SIZE,
        EXPERTS_PER_RANK,
        route_table.expert_capacity,
        2 * INTERMEDIATE_DIM,
    )

    def fail_flat_h_conversion(*_args, **_kwargs):
        raise AssertionError("Pallas compact-H MLP backward residual must not use flat-H conversion")

    monkeypatch.setattr(source_push_mlp, "_source_push_mlp_flat_h_to_compact", fail_flat_h_conversion)
    monkeypatch.setattr(source_push_mlp, "_source_push_mlp_compact_h_to_flat", fail_flat_h_conversion)

    def fake_source_push_forward_with_compact_h_from_plan(
        config_arg,
        host_inputs_arg,
        x_arg,
        h_route_weights_arg,
        w13_arg,
        w2_arg,
        *,
        compact_expert_capacity,
        mesh,
    ):
        del h_route_weights_arg, mesh
        y, _ = source_push_moe_mlp_reference_with_h(
            route_table,
            x_arg,
            route_weights,
            w13_arg,
            w2_arg,
        )
        h = staged_h
        if compact_expert_capacity > route_table.expert_capacity:
            h = jnp.pad(
                h,
                ((0, 0), (0, 0), (0, compact_expert_capacity - route_table.expert_capacity), (0, 0)),
            )
        return y, h, host_inputs_arg.plan.dropped_routes

    monkeypatch.setattr(
        source_push_mlp,
        "source_push_forward_with_compact_h_from_plan",
        fake_source_push_forward_with_compact_h_from_plan,
    )

    def pallas_loss(x_arg, route_weights_arg, w13_arg, w2_arg):
        y, dropped_routes = source_push_moe_mlp_from_plan(
            config,
            host_inputs,
            route_table,
            x_arg,
            route_weights_arg,
            w13_arg,
            w2_arg,
            implementation=SOURCE_PUSH_MLP_IMPLEMENTATION_PALLAS_MGPU,
        )
        assert int(dropped_routes) == 0
        return jnp.sum(y * cotangent)

    pallas_grads = jax.grad(pallas_loss, argnums=(0, 1, 2, 3))(x, route_weights, w13, w2)
    expert_route_weights = source_push_mlp._source_push_mlp_route_weights_to_all_expert_major(
        route_table,
        route_weights,
    )
    expected = source_push_mlp._source_push_moe_mlp_backward_from_h(
        route_table,
        x,
        expert_route_weights,
        w13,
        w2,
        staged_h,
        cotangent,
    )

    for observed, expected_grad in zip(
        pallas_grads,
        (expected.dx, expected.d_route_weights, expected.dw13, expected.dw2),
        strict=True,
    ):
        np.testing.assert_allclose(np.asarray(observed), np.asarray(expected_grad), atol=1e-5, rtol=1e-5)
    assert float(jnp.max(jnp.abs(staged_h - reference_h))) > 0.0
    assert float(jnp.max(jnp.abs(pallas_grads[1]))) > 0.0


def test_source_push_moe_mlp_from_plan_backward_does_not_use_flat_return_bridge(monkeypatch):
    config, host_inputs, _route_table, x, route_weights, w13, w2 = _small_forward_plan_inputs()
    cotangent = jnp.linspace(-0.5, 0.7, x.size, dtype=jnp.float32).reshape(x.shape)

    def fail_flat_return(*_args, **_kwargs):
        raise AssertionError("from-plan MLP backward should consume compact H directly")

    monkeypatch.setattr(source_push_mlp, "source_push_backward_return_flat", fail_flat_return)

    def loss(x_arg, route_weights_arg, w13_arg, w2_arg):
        y, dropped_routes = source_push_moe_mlp_from_plan(
            config,
            host_inputs,
            source_push_mlp.source_push_mlp_route_table_from_plan(
                host_inputs.plan,
                src_base_by_expert=host_inputs.src_base_by_expert,
            ),
            x_arg,
            route_weights_arg,
            w13_arg,
            w2_arg,
            implementation=SOURCE_PUSH_MLP_IMPLEMENTATION_REFERENCE,
        )
        assert int(dropped_routes) == 0
        return jnp.sum(y * cotangent)

    grads = jax.grad(loss, argnums=(0, 1, 2, 3))(x, route_weights, w13, w2)

    assert float(jnp.max(jnp.abs(grads[1]))) > 0.0


def test_source_push_moe_mlp_pallas_backward_selectors_avoid_full_output_pallas_stages(monkeypatch):
    assert (
        source_push_mlp._source_push_mlp_backward_dy_route_implementation(SOURCE_PUSH_MLP_IMPLEMENTATION_PALLAS_MGPU)
        == source_push_mlp.SOURCE_PUSH_DY_ROUTE_IMPLEMENTATION_REFERENCE
    )
    assert (
        source_push_mlp._source_push_mlp_backward_w2_implementation(SOURCE_PUSH_MLP_IMPLEMENTATION_PALLAS_MGPU)
        == source_push_mlp.SOURCE_PUSH_W2_BACKWARD_IMPLEMENTATION_REFERENCE
    )
    assert (
        source_push_mlp._source_push_mlp_backward_return_implementation(SOURCE_PUSH_MLP_IMPLEMENTATION_PALLAS_MGPU)
        == source_push_mlp.SOURCE_PUSH_BACKWARD_RETURN_IMPLEMENTATION_JAX
    )

    monkeypatch.setattr(source_push_mlp.jax, "default_backend", lambda: "gpu")

    assert (
        source_push_mlp._source_push_mlp_backward_dy_route_implementation(SOURCE_PUSH_MLP_IMPLEMENTATION_PALLAS_MGPU)
        == source_push_mlp.SOURCE_PUSH_DY_ROUTE_IMPLEMENTATION_REFERENCE
    )
    assert (
        source_push_mlp._source_push_mlp_backward_w2_implementation(SOURCE_PUSH_MLP_IMPLEMENTATION_PALLAS_MGPU)
        == source_push_mlp.SOURCE_PUSH_W2_BACKWARD_IMPLEMENTATION_REFERENCE
    )
    assert (
        source_push_mlp._source_push_mlp_backward_return_implementation(SOURCE_PUSH_MLP_IMPLEMENTATION_PALLAS_MGPU)
        == source_push_mlp.SOURCE_PUSH_BACKWARD_RETURN_IMPLEMENTATION_PALLAS_MGPU
    )


def test_source_push_moe_mlp_from_plan_exact_layout_matches_reference_gradients():
    config, host_inputs, route_table, x, route_weights, w13, w2 = _small_forward_plan_inputs(
        use_exact_expert_major=True,
    )
    cotangent = jnp.linspace(-0.5, 0.7, x.size, dtype=jnp.float32).reshape(x.shape)

    assert host_inputs.use_exact_expert_major

    def reference_loss(x_arg, route_weights_arg, w13_arg, w2_arg):
        y = source_push_moe_mlp_reference(route_table, x_arg, route_weights_arg, w13_arg, w2_arg)
        return jnp.sum(y * cotangent)

    def from_plan_loss(x_arg, route_weights_arg, w13_arg, w2_arg):
        y, dropped_routes = source_push_moe_mlp_from_plan(
            config,
            host_inputs,
            route_table,
            x_arg,
            route_weights_arg,
            w13_arg,
            w2_arg,
            implementation=SOURCE_PUSH_MLP_IMPLEMENTATION_REFERENCE,
        )
        assert int(dropped_routes) == 0
        return jnp.sum(y * cotangent)

    reference_grads = jax.grad(reference_loss, argnums=(0, 1, 2, 3))(x, route_weights, w13, w2)
    from_plan_grads = jax.grad(from_plan_loss, argnums=(0, 1, 2, 3))(x, route_weights, w13, w2)

    for observed, expected in zip(from_plan_grads, reference_grads, strict=True):
        np.testing.assert_allclose(np.asarray(observed), np.asarray(expected), atol=1e-5, rtol=1e-5)
    assert float(jnp.max(jnp.abs(from_plan_grads[1]))) > 0.0


def test_source_push_moe_mlp_flat_h_backward_matches_compact_h_backward():
    x, route_assignments, route_weights, w13, w2 = _small_mlp_inputs()
    route_table = _route_table(route_assignments, route_weights)
    cotangent = jnp.linspace(-0.5, 0.7, x.size, dtype=jnp.float32).reshape(x.shape)
    _, compact_h = source_push_moe_mlp_reference_with_h(route_table, x, route_weights, w13, w2)
    expert_base = jnp.array(
        [
            [0, route_table.expert_capacity + 1],
            [2, route_table.expert_capacity + 4],
        ],
        dtype=jnp.int32,
    )
    flat_rows = int(np.max(np.asarray(expert_base))) + route_table.expert_capacity
    flat_h = jnp.zeros((EP_SIZE, flat_rows, 2 * INTERMEDIATE_DIM), dtype=compact_h.dtype)
    for dst in range(EP_SIZE):
        for expert in range(EXPERTS_PER_RANK):
            start = int(expert_base[dst, expert])
            flat_h = flat_h.at[dst, start : start + route_table.expert_capacity].set(compact_h[dst, expert])
    expert_route_weights = source_push_mlp._source_push_mlp_route_weights_to_all_expert_major(
        route_table,
        route_weights,
    )

    compact_grads = source_push_mlp._source_push_moe_mlp_backward_from_h(
        route_table,
        x,
        expert_route_weights,
        w13,
        w2,
        compact_h,
        cotangent,
    )
    flat_grads = source_push_mlp._source_push_moe_mlp_backward_from_h_flat(
        route_table,
        expert_base,
        x,
        expert_route_weights,
        w13,
        w2,
        flat_h,
        cotangent,
    )

    for observed, expected in zip(flat_grads, compact_grads, strict=True):
        np.testing.assert_allclose(np.asarray(observed), np.asarray(expected), atol=0, rtol=0)


def _naive_source_push_moe_mlp(x, route_assignments, route_weights, w13, w2):
    ep_size, tokens_per_rank, topk = route_assignments.shape
    hidden_dim = x.shape[-1]
    experts_per_rank = w13.shape[1]
    out = np.zeros((ep_size, tokens_per_rank, hidden_dim), dtype=np.float32)
    x_host = np.asarray(x, dtype=np.float32)
    route_assignments_host = np.asarray(route_assignments)
    route_weights_host = np.asarray(route_weights, dtype=np.float32)
    w13_host = np.asarray(w13, dtype=np.float32)
    w2_host = np.asarray(w2, dtype=np.float32)
    for src in range(ep_size):
        for token in range(tokens_per_rank):
            for route_slot in range(topk):
                global_expert = int(route_assignments_host[src, token, route_slot])
                dst = global_expert // experts_per_rank
                local_expert = global_expert % experts_per_rank
                h = x_host[src, token] @ w13_host[dst, local_expert]
                gate, up = np.split(h, 2)
                activation = gate * (1.0 / (1.0 + np.exp(-gate))) * up
                weighted_activation = route_weights_host[src, token, route_slot] * activation
                out[src, token] += weighted_activation @ w2_host[dst, local_expert]
    return out


def _accepted_route_mask(route_table, route_weights_shape):
    accepted = np.zeros(route_weights_shape, dtype=np.bool_)
    for route in range(route_table.source_rank.shape[0]):
        accepted[
            int(route_table.source_rank[route]),
            int(route_table.token_id[route]),
            int(route_table.route_slot[route]),
        ] = True
    return accepted
