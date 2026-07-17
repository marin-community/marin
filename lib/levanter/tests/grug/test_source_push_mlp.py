# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import numpy as np

from levanter.grug._moe import source_push_forward
from levanter.grug._moe.source_push_inbox import PushInboxConfig
import levanter.grug._moe.source_push_mlp as source_push_mlp
from levanter.grug._moe.source_push_mlp import (
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
    route_table, dropped_routes = build_source_push_mlp_route_table(
        route_assignments,
        route_weights,
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


def _small_forward_plan_inputs():
    x, route_assignments, route_weights, w13, w2 = _small_mlp_inputs()
    config = _small_forward_config()
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


def test_source_push_moe_mlp_custom_vjp_residual_names_h_checkpoint():
    x, route_assignments, route_weights, w13, w2 = _small_mlp_inputs()
    route_table = _route_table(route_assignments, route_weights)

    y, residual = source_push_mlp._source_push_moe_mlp_fwd(route_table, x, route_weights, w13, w2)
    expected_y, expected_h = source_push_moe_mlp_reference_with_h(route_table, x, route_weights, w13, w2)

    assert isinstance(residual, source_push_mlp.SourcePushMlpReferenceResidual)
    assert residual._fields == ("route_table", "x", "route_weights", "w13", "w2", "h")
    assert source_push_mlp.SourcePushMlpFlatResidual._fields == ("x", "route_weights", "w13", "w2", "h_flat")
    assert not {"recv_x", "packed_x", "activation", "route_y", "scratch"}.intersection(residual._fields)
    np.testing.assert_allclose(np.asarray(y), np.asarray(expected_y), atol=0, rtol=0)
    np.testing.assert_allclose(np.asarray(residual.h), np.asarray(expected_h), atol=0, rtol=0)


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

    compact_grads = source_push_mlp._source_push_moe_mlp_backward_from_h(
        route_table,
        x,
        route_weights,
        w13,
        w2,
        compact_h,
        cotangent,
    )
    flat_grads = source_push_mlp._source_push_moe_mlp_backward_from_h_flat(
        route_table,
        expert_base,
        x,
        route_weights,
        w13,
        w2,
        flat_h,
        cotangent,
    )

    for observed, expected in zip(flat_grads, compact_grads, strict=True):
        np.testing.assert_allclose(np.asarray(observed), np.asarray(expected), atol=0, rtol=0)


def _naive_source_push_moe_mlp(x, route_assignments, route_weights, w13, w2):
    out = np.zeros((EP_SIZE, TOKENS_PER_RANK, HIDDEN_DIM), dtype=np.float32)
    x_host = np.asarray(x, dtype=np.float32)
    route_assignments_host = np.asarray(route_assignments)
    route_weights_host = np.asarray(route_weights, dtype=np.float32)
    w13_host = np.asarray(w13, dtype=np.float32)
    w2_host = np.asarray(w2, dtype=np.float32)
    for src in range(EP_SIZE):
        for token in range(TOKENS_PER_RANK):
            for route_slot in range(TOPK):
                global_expert = int(route_assignments_host[src, token, route_slot])
                dst = global_expert // EXPERTS_PER_RANK
                local_expert = global_expert % EXPERTS_PER_RANK
                h = x_host[src, token] @ w13_host[dst, local_expert]
                gate, up = np.split(h, 2)
                activation = gate * (1.0 / (1.0 + np.exp(-gate))) * up
                weighted_activation = route_weights_host[src, token, route_slot] * activation
                out[src, token] += weighted_activation @ w2_host[dst, local_expert]
    return out
