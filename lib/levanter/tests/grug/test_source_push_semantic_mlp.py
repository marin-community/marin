# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh, PartitionSpec as P

from levanter.grug._moe.source_push_semantic_mlp import (
    SourcePushSemanticMlpCapacity,
    _build_source_push_semantic_mlp_plan,
    _constrain_destination_major,
    _constrain_replicated,
    _source_push_moe_mlp_semantic_fused_pallas_mgpu,
    _source_push_moe_mlp_semantic_pallas_mgpu,
    _source_push_semantic_fused_queue_geometry,
    source_push_moe_mlp_semantic_pallas_mgpu,
)


def _dense_moe_reference(
    selected_experts: jax.Array,
    x: jax.Array,
    route_weights: jax.Array,
    w13: jax.Array,
    w2: jax.Array,
    accepted_routes: jax.Array,
) -> jax.Array:
    hidden_dim = x.shape[-1]
    intermediate_dim = w2.shape[-2]
    selected_w13 = w13.reshape(-1, hidden_dim, 2 * intermediate_dim)[selected_experts]
    z = jnp.einsum("sth,stkho->stko", x, selected_w13)
    gate, up = jnp.split(z, 2, axis=-1)
    h = jax.nn.silu(gate) * up
    selected_w2 = w2.reshape(-1, intermediate_dim, hidden_dim)[selected_experts]
    route_y = jnp.einsum("stki,stkih->stkh", h, selected_w2)
    weights = route_weights * accepted_routes.astype(route_weights.dtype)
    return jnp.sum(route_y * weights[..., None], axis=2)


def _tiny_inputs():
    selected_experts = jnp.asarray(
        [
            [[0, 1], [0, 2], [3, 3]],
            [[0, 0], [1, 2], [2, 3]],
        ],
        dtype=jnp.int32,
    )
    keys = jax.random.split(jax.random.key(0), 5)
    x = jax.random.normal(keys[0], (2, 3, 4), dtype=jnp.float32)
    route_weights = jax.random.normal(keys[1], selected_experts.shape, dtype=jnp.float32)
    w13 = jax.random.normal(keys[2], (2, 2, 4, 6), dtype=jnp.float32)
    w2 = jax.random.normal(keys[3], (2, 2, 3, 4), dtype=jnp.float32)
    cotangent = jax.random.normal(keys[4], x.shape, dtype=jnp.float32)
    return selected_experts, x, route_weights, w13, w2, cotangent


def test_semantic_mlp_interpret_matches_dense_value_and_custom_vjp_gradients_under_jit():
    selected_experts, x, route_weights, w13, w2, cotangent = _tiny_inputs()
    capacity = SourcePushSemanticMlpCapacity(rows_per_src_dst=2, rows_per_expert=3)
    plan = _build_source_push_semantic_mlp_plan(
        selected_experts,
        route_weights,
        experts_per_rank=w13.shape[1],
        capacity=capacity,
        capacity_factor=10.0,
    )
    accepted_routes = plan.reverse_route.route_valid
    assert int(plan.metadata_overflow_routes) > 0

    def observed_loss(x_arg, route_weights_arg, w13_arg, w2_arg):
        y, dropped = _source_push_moe_mlp_semantic_pallas_mgpu(
            selected_experts,
            x_arg,
            route_weights_arg,
            w13_arg,
            w2_arg,
            capacity=capacity,
            capacity_factor=10.0,
            mesh=None,
            interpret=True,
        )
        return jnp.sum(y * cotangent), (y, dropped)

    def reference_loss(x_arg, route_weights_arg, w13_arg, w2_arg):
        y = _dense_moe_reference(
            selected_experts,
            x_arg,
            route_weights_arg,
            w13_arg,
            w2_arg,
            accepted_routes,
        )
        return jnp.sum(y * cotangent), y

    observed = jax.jit(jax.value_and_grad(observed_loss, argnums=(0, 1, 2, 3), has_aux=True))(
        x,
        route_weights,
        w13,
        w2,
    )
    expected = jax.jit(jax.value_and_grad(reference_loss, argnums=(0, 1, 2, 3), has_aux=True))(
        x,
        route_weights,
        w13,
        w2,
    )
    (observed_loss_value, (observed_y, observed_dropped)), observed_grads = observed
    (expected_loss_value, expected_y), expected_grads = expected

    np.testing.assert_allclose(observed_y, expected_y, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(observed_loss_value, expected_loss_value, rtol=2e-5, atol=2e-5)
    for observed_grad, expected_grad in zip(observed_grads, expected_grads, strict=True):
        np.testing.assert_allclose(observed_grad, expected_grad, rtol=2e-5, atol=2e-5)
    assert int(observed_dropped) == int(plan.dropped_routes)
    assert bool(jnp.all(observed_grads[1][~accepted_routes] == 0.0))


def test_semantic_mlp_interpret_bfloat16_matches_dense_value():
    selected_experts, x, route_weights, w13, w2, cotangent = _tiny_inputs()
    x = x.astype(jnp.bfloat16)
    route_weights = route_weights.astype(jnp.bfloat16)
    w13 = w13.astype(jnp.bfloat16)
    w2 = w2.astype(jnp.bfloat16)
    cotangent = cotangent.astype(jnp.bfloat16)
    capacity = SourcePushSemanticMlpCapacity(rows_per_src_dst=4, rows_per_expert=6)
    plan = _build_source_push_semantic_mlp_plan(
        selected_experts,
        route_weights,
        experts_per_rank=w13.shape[1],
        capacity=capacity,
        capacity_factor=10.0,
    )

    def observed_loss(x_arg, route_weights_arg, w13_arg, w2_arg):
        y, dropped = _source_push_moe_mlp_semantic_pallas_mgpu(
            selected_experts,
            x_arg,
            route_weights_arg,
            w13_arg,
            w2_arg,
            capacity=capacity,
            capacity_factor=10.0,
            mesh=None,
            interpret=True,
        )
        return jnp.sum(y * cotangent), (y, dropped)

    def reference_loss(x_arg, route_weights_arg, w13_arg, w2_arg):
        y = _dense_moe_reference(
            selected_experts,
            x_arg,
            route_weights_arg,
            w13_arg,
            w2_arg,
            plan.reverse_route.route_valid,
        )
        return jnp.sum(y * cotangent), y

    (observed_loss_value, (observed_y, observed_dropped)), observed_grads = jax.jit(
        jax.value_and_grad(observed_loss, argnums=(0, 1, 2, 3), has_aux=True)
    )(x, route_weights, w13, w2)
    (expected_loss_value, expected_y), expected_grads = jax.jit(
        jax.value_and_grad(reference_loss, argnums=(0, 1, 2, 3), has_aux=True)
    )(x, route_weights, w13, w2)

    np.testing.assert_allclose(
        np.asarray(observed_y, dtype=np.float32),
        np.asarray(expected_y, dtype=np.float32),
        rtol=0.03,
        atol=0.03,
    )
    np.testing.assert_allclose(
        np.asarray(observed_loss_value, dtype=np.float32),
        np.asarray(expected_loss_value, dtype=np.float32),
        rtol=0.03,
        atol=0.03,
    )
    for observed_grad, expected_grad in zip(observed_grads, expected_grads, strict=True):
        np.testing.assert_allclose(
            np.asarray(observed_grad, dtype=np.float32),
            np.asarray(expected_grad, dtype=np.float32),
            rtol=0.08,
            atol=0.08,
        )
    assert int(observed_dropped) == 0


def test_fused_semantic_mlp_interpret_matches_dense_value_and_gradients_with_duplicate_topk_under_jit():
    selected_experts = jnp.asarray(
        [
            [[0, 0], [1, 3]],
            [[2, 3], [3, 3]],
        ],
        dtype=jnp.int32,
    )
    keys = jax.random.split(jax.random.key(17), 5)
    x = jax.random.normal(keys[0], (2, 2, 512), dtype=jnp.bfloat16) / 8
    route_weights = jax.random.normal(keys[1], selected_experts.shape, dtype=jnp.bfloat16) / 8
    w13 = jax.random.normal(keys[2], (2, 2, 512, 256), dtype=jnp.bfloat16) / 16
    w2 = jax.random.normal(keys[3], (2, 2, 128, 512), dtype=jnp.bfloat16) / 16
    cotangent = jax.random.normal(keys[4], x.shape, dtype=jnp.bfloat16) / 8
    capacity = SourcePushSemanticMlpCapacity(rows_per_src_dst=4, rows_per_expert=128)
    plan = _build_source_push_semantic_mlp_plan(
        selected_experts,
        route_weights,
        experts_per_rank=w13.shape[1],
        capacity=capacity,
        capacity_factor=10.0,
    )

    def observed_loss(x_arg, route_weights_arg, w13_arg, w2_arg):
        y, dropped = _source_push_moe_mlp_semantic_fused_pallas_mgpu(
            selected_experts,
            x_arg,
            route_weights_arg,
            w13_arg,
            w2_arg,
            capacity=capacity,
            capacity_factor=10.0,
            mesh=None,
            interpret=True,
        )
        return jnp.sum(y.astype(jnp.float32) * cotangent.astype(jnp.float32)), (y, dropped)

    def reference_loss(x_arg, route_weights_arg, w13_arg, w2_arg):
        y = _dense_moe_reference(
            selected_experts,
            x_arg,
            route_weights_arg,
            w13_arg,
            w2_arg,
            plan.reverse_route.route_valid,
        )
        return jnp.sum(y.astype(jnp.float32) * cotangent.astype(jnp.float32)), y

    (observed_loss_value, (observed_y, observed_dropped)), observed_grads = jax.jit(
        jax.value_and_grad(observed_loss, argnums=(0, 1, 2, 3), has_aux=True)
    )(x, route_weights, w13, w2)
    (expected_loss_value, expected_y), expected_grads = jax.jit(
        jax.value_and_grad(reference_loss, argnums=(0, 1, 2, 3), has_aux=True)
    )(x, route_weights, w13, w2)

    np.testing.assert_allclose(
        np.asarray(observed_y, dtype=np.float32), np.asarray(expected_y, dtype=np.float32), rtol=0.04, atol=0.04
    )
    np.testing.assert_allclose(observed_loss_value, expected_loss_value, rtol=0.04, atol=0.04)
    for observed_grad, expected_grad in zip(observed_grads, expected_grads, strict=True):
        np.testing.assert_allclose(
            np.asarray(observed_grad, dtype=np.float32),
            np.asarray(expected_grad, dtype=np.float32),
            rtol=0.1,
            atol=0.1,
        )
    assert int(observed_dropped) == 0


def test_fused_semantic_mlp_queue_geometry_covers_expert_fragmentation():
    send_chunks_per_dst, entries_per_dst = _source_push_semantic_fused_queue_geometry(
        rows_per_src_dst=65,
        experts_per_rank=8,
    )

    assert send_chunks_per_dst == 3
    assert entries_per_dst == 4 * send_chunks_per_dst


def test_fused_semantic_mlp_reports_layout_overflow_in_dropped_accounting():
    selected_experts = jnp.zeros((2, 1, 1), dtype=jnp.int32)
    x = jnp.zeros((2, 1, 512), dtype=jnp.bfloat16)
    route_weights = jnp.ones(selected_experts.shape, dtype=jnp.bfloat16)
    w13 = jnp.zeros((2, 1, 512, 256), dtype=jnp.bfloat16)
    w2 = jnp.zeros((2, 1, 128, 512), dtype=jnp.bfloat16)
    capacity = SourcePushSemanticMlpCapacity(rows_per_src_dst=1, rows_per_expert=64)

    _y, dropped = jax.jit(
        lambda x_arg, route_weights_arg, w13_arg, w2_arg: _source_push_moe_mlp_semantic_fused_pallas_mgpu(
            selected_experts,
            x_arg,
            route_weights_arg,
            w13_arg,
            w2_arg,
            capacity=capacity,
            capacity_factor=2.0,
            mesh=None,
            interpret=True,
        )
    )(x, route_weights, w13, w2)

    assert int(dropped) == 64


def test_fused_semantic_mlp_non_interpreted_profile_requires_bfloat16():
    selected_experts, x, route_weights, w13, w2, _cotangent = _tiny_inputs()
    capacity = SourcePushSemanticMlpCapacity(rows_per_src_dst=4, rows_per_expert=128)
    mesh = Mesh(np.asarray(jax.devices("cpu")[:1]), ("expert",), axis_types=(AxisType.Explicit,))

    with pytest.raises(ValueError, match="requires bfloat16 x"):
        _source_push_moe_mlp_semantic_fused_pallas_mgpu(
            selected_experts,
            x,
            route_weights,
            w13,
            w2,
            capacity=capacity,
            capacity_factor=1.0,
            mesh=mesh,
            interpret=False,
        )


def test_semantic_mlp_reports_router_and_pair_overflow_separately():
    selected_experts = jnp.zeros((1, 4, 1), dtype=jnp.int32)
    route_weights = jnp.ones(selected_experts.shape, dtype=jnp.float32)
    capacity = SourcePushSemanticMlpCapacity(rows_per_src_dst=1, rows_per_expert=1)
    plan = jax.jit(
        lambda: _build_source_push_semantic_mlp_plan(
            selected_experts,
            route_weights,
            experts_per_rank=1,
            capacity=capacity,
            capacity_factor=0.5,
        )
    )()

    assert int(plan.routing_dropped_routes) == 2
    assert int(plan.metadata_overflow_routes) == 1
    assert int(plan.dropped_routes) == 3


def test_semantic_mlp_reports_cross_source_expert_capacity_overflow():
    selected_experts = jnp.zeros((2, 2, 1), dtype=jnp.int32)
    route_weights = jnp.ones(selected_experts.shape, dtype=jnp.float32)
    capacity = SourcePushSemanticMlpCapacity(rows_per_src_dst=2, rows_per_expert=2)
    plan = _build_source_push_semantic_mlp_plan(
        selected_experts,
        route_weights,
        experts_per_rank=1,
        capacity=capacity,
        capacity_factor=2.0,
    )
    x = jnp.ones((2, 2, 2), dtype=jnp.float32)
    w13 = jnp.ones((2, 1, 2, 4), dtype=jnp.float32)
    w2 = jnp.ones((2, 1, 2, 2), dtype=jnp.float32)
    _y, dropped = jax.jit(
        lambda x_arg, route_weights_arg, w13_arg, w2_arg: _source_push_moe_mlp_semantic_pallas_mgpu(
            selected_experts,
            x_arg,
            route_weights_arg,
            w13_arg,
            w2_arg,
            capacity=capacity,
            capacity_factor=2.0,
            mesh=None,
            interpret=True,
        )
    )(x, route_weights, w13, w2)

    assert int(plan.routing_dropped_routes) == 0
    assert int(plan.metadata_overflow_routes) == 2
    assert int(plan.dropped_routes) == 2
    assert int(dropped) == int(plan.dropped_routes)


def test_semantic_mlp_shape_and_capacity_validation_rejects_incompatible_inputs():
    selected_experts, x, route_weights, w13, w2, _cotangent = _tiny_inputs()
    capacity = SourcePushSemanticMlpCapacity(rows_per_src_dst=2, rows_per_expert=3)

    with pytest.raises(ValueError, match="route_weights shape"):
        _source_push_moe_mlp_semantic_pallas_mgpu(
            selected_experts,
            x,
            route_weights[:, :, :1],
            w13,
            w2,
            capacity=capacity,
            capacity_factor=1.0,
            mesh=None,
            interpret=True,
        )
    with pytest.raises(ValueError, match="w2 shape"):
        _source_push_moe_mlp_semantic_pallas_mgpu(
            selected_experts,
            x,
            route_weights,
            w13,
            w2[..., :2],
            capacity=capacity,
            capacity_factor=1.0,
            mesh=None,
            interpret=True,
        )
    with pytest.raises(ValueError, match="rows_per_expert must be positive"):
        _source_push_moe_mlp_semantic_pallas_mgpu(
            selected_experts,
            x,
            route_weights,
            w13,
            w2,
            capacity=SourcePushSemanticMlpCapacity(rows_per_src_dst=2, rows_per_expert=0),
            capacity_factor=1.0,
            mesh=None,
            interpret=True,
        )


def test_semantic_mlp_public_entrypoint_requires_explicit_gpu_mesh():
    devices = np.asarray(jax.devices("cpu")[:1])
    selected_experts = jnp.zeros((1, 1, 1), dtype=jnp.int32)
    x = jnp.ones((1, 1, 512), dtype=jnp.float32)
    route_weights = jnp.ones(selected_experts.shape, dtype=jnp.float32)
    w13 = jnp.ones((1, 1, 512, 256), dtype=jnp.float32)
    w2 = jnp.ones((1, 1, 128, 512), dtype=jnp.float32)
    capacity = SourcePushSemanticMlpCapacity(rows_per_src_dst=1, rows_per_expert=128)

    implicit_mesh = Mesh(devices, ("expert",), axis_types=(AxisType.Auto,))
    with pytest.raises(ValueError, match="AxisType.Explicit"):
        source_push_moe_mlp_semantic_pallas_mgpu(
            selected_experts,
            x,
            route_weights,
            w13,
            w2,
            capacity=capacity,
            capacity_factor=1.0,
            mesh=implicit_mesh,
        )

    cpu_mesh = Mesh(devices, ("expert",), axis_types=(AxisType.Explicit,))
    with pytest.raises(NotImplementedError, match="GPU mesh devices"):
        source_push_moe_mlp_semantic_pallas_mgpu(
            selected_experts,
            x,
            route_weights,
            w13,
            w2,
            capacity=capacity,
            capacity_factor=1.0,
            mesh=cpu_mesh,
        )


def test_semantic_mlp_constrains_expert_major_intermediates_to_the_destination_axis():
    mesh = Mesh(np.asarray(jax.devices("cpu")[:1]), ("expert",), axis_types=(AxisType.Explicit,))
    value = jnp.ones((1, 2, 3, 4), dtype=jnp.bfloat16)

    with jax.set_mesh(mesh):
        constrained = jax.jit(lambda value_arg: _constrain_destination_major(value_arg, mesh))(value)

    assert constrained.sharding.spec == P("expert", None, None, None)


def test_semantic_mlp_constrains_destination_pull_dy_to_replicated():
    mesh = Mesh(np.asarray(jax.devices("cpu")[:1]), ("expert",), axis_types=(AxisType.Explicit,))
    value = jnp.ones((1, 3, 4), dtype=jnp.float32)

    with jax.set_mesh(mesh):
        constrained = jax.jit(lambda value_arg: _constrain_replicated(value_arg, mesh))(value)

    assert constrained.sharding.spec == P(None, None, None)


def test_semantic_mlp_constrains_interpret_gradient_under_explicit_mesh():
    mesh = Mesh(np.asarray(jax.devices("cpu")[:1]), ("expert",), axis_types=(AxisType.Explicit,))
    value = jnp.ones((1, 3, 4), dtype=jnp.float32)

    with jax.set_mesh(mesh):
        constrained = jax.jit(lambda value_arg: _constrain_replicated(value_arg, None))(value)

    assert constrained.sharding.spec == P(None, None, None)
