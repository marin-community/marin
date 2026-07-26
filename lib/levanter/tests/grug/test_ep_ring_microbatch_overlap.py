# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from experiments.grug.moe.benchmark_ep_ring_microbatch_overlap import (
    _PARITY_RELATIVE_L2,
    _VAG_PROMOTION_SPEEDUP,
    _parity_metrics,
    _parser,
    _promotion,
    _training_loss,
    _time_arms,
    _validate_args,
    _value_and_grad_parity_metrics,
)
from levanter.grug._moe.ep_ring import (
    _BulkRingDispatchState,
    _BulkRingExpertState,
    _moe_mlp_ep_ring_combine_local,
    _moe_mlp_ep_ring_dispatch_local,
    _moe_mlp_ep_ring_expert_local,
    _moe_mlp_ep_ring_local,
)


def _single_device_mesh() -> Mesh:
    return Mesh(
        np.asarray(jax.devices()[:1]).reshape(1, 1, 1),
        axis_names=("data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )


@pytest.mark.parametrize("overflow", [False, True])
def test_ep_ring_phases_match_full_output_drop_and_vjp(overflow: bool) -> None:
    mesh = _single_device_mesh()
    tokens = 8
    hidden_dim = 4
    intermediate_dim = 6
    num_experts = 2
    top_k = 2
    capacity_factor = 0.5 if overflow else 1.0
    keys = jax.random.split(jax.random.key(0), 5)
    x = jax.random.normal(keys[0], (tokens, hidden_dim), dtype=jnp.bfloat16)
    if overflow:
        selected_experts = jnp.zeros((tokens, top_k), dtype=jnp.int32)
    else:
        selected_experts = jnp.arange(tokens * top_k, dtype=jnp.int32).reshape(tokens, top_k) % num_experts
    combine_weights = jax.nn.softmax(
        jax.random.normal(keys[1], (tokens, top_k), dtype=jnp.float32),
        axis=-1,
    )
    w13 = jax.random.normal(keys[2], (num_experts, hidden_dim, 2 * intermediate_dim), dtype=jnp.bfloat16)
    w2 = jax.random.normal(keys[3], (num_experts, intermediate_dim, hidden_dim), dtype=jnp.bfloat16)
    cotangent = jax.random.normal(keys[4], x.shape, dtype=jnp.bfloat16)

    batch_spec = P(("data", "expert"), None)
    expert_spec = P("expert", None, None)
    phase_vector_spec = P(("data", "expert"))
    dispatch_spec = _BulkRingDispatchState(
        x_dispatch=batch_spec,
        weight_dispatch=phase_vector_spec,
        token_global=phase_vector_spec,
        group_sizes=phase_vector_spec,
        dropped_local=phase_vector_spec,
    )
    expert_state_spec = _BulkRingExpertState(out_dispatch=batch_spec)
    batch_sharding = NamedSharding(mesh, batch_spec)
    expert_sharding = NamedSharding(mesh, expert_spec)
    x = jax.device_put(x, batch_sharding)
    selected_experts = jax.device_put(selected_experts, batch_sharding)
    combine_weights = jax.device_put(combine_weights, batch_sharding)
    cotangent = jax.device_put(cotangent, batch_sharding)
    w13 = jax.device_put(w13, expert_sharding)
    w2 = jax.device_put(w2, expert_sharding)
    full = jax.shard_map(
        partial(
            _moe_mlp_ep_ring_local,
            activation_fn=jax.nn.silu,
            num_experts=num_experts,
            capacity_factor=capacity_factor,
        ),
        mesh=mesh,
        in_specs=(batch_spec, batch_spec, batch_spec, expert_spec, expert_spec),
        out_specs=(batch_spec, P()),
        check_vma=False,
    )
    dispatch = jax.shard_map(
        partial(
            _moe_mlp_ep_ring_dispatch_local,
            local_experts=num_experts,
            num_experts=num_experts,
            capacity_factor=capacity_factor,
        ),
        mesh=mesh,
        in_specs=(batch_spec, batch_spec, batch_spec),
        out_specs=dispatch_spec,
        check_vma=False,
    )
    expert = jax.shard_map(
        partial(_moe_mlp_ep_ring_expert_local, activation_fn=jax.nn.silu),
        mesh=mesh,
        in_specs=(dispatch_spec, expert_spec, expert_spec),
        out_specs=expert_state_spec,
        check_vma=False,
    )
    combine = jax.shard_map(
        partial(
            _moe_mlp_ep_ring_combine_local,
            tokens_per_shard=tokens,
            expert_axis_size=1,
        ),
        mesh=mesh,
        in_specs=(dispatch_spec, expert_state_spec),
        out_specs=(batch_spec, P()),
        check_vma=False,
    )

    def phased(x: jax.Array, combine_weights: jax.Array, w13: jax.Array, w2: jax.Array):
        dispatched = dispatch(x, selected_experts, combine_weights)
        computed = expert(dispatched, w13, w2)
        return combine(dispatched, computed)

    def full_loss(x, combine_weights, w13, w2):
        out, _ = full(x, selected_experts, combine_weights, w13, w2)
        return jnp.sum(out.astype(jnp.float32) * cotangent.astype(jnp.float32))

    def phased_loss(x, combine_weights, w13, w2):
        out, _ = phased(x, combine_weights, w13, w2)
        return jnp.sum(out.astype(jnp.float32) * cotangent.astype(jnp.float32))

    with jax.set_mesh(mesh):
        full_result = full(x, selected_experts, combine_weights, w13, w2)
        phased_result = phased(x, combine_weights, w13, w2)
        full_gradients = jax.grad(full_loss, argnums=(0, 1, 2, 3))(x, combine_weights, w13, w2)
        phased_gradients = jax.grad(phased_loss, argnums=(0, 1, 2, 3))(x, combine_weights, w13, w2)

    np.testing.assert_array_equal(np.asarray(phased_result[0]), np.asarray(full_result[0]))
    assert int(phased_result[1]) == int(full_result[1])
    assert (int(full_result[1]) > 0) == overflow
    for phased_gradient, full_gradient in zip(phased_gradients, full_gradients, strict=True):
        np.testing.assert_array_equal(np.asarray(phased_gradient), np.asarray(full_gradient))


def test_ep_ring_microbatch_overlap_defaults_match_target_geometry() -> None:
    args = _parser().parse_args([])

    _validate_args(args)

    assert args.microbatch_size == 32
    assert args.sequence_length == 4096
    assert args.hidden_dim == 2560
    assert args.intermediate_dim == 1280
    assert args.num_experts == 64
    assert args.top_k == 4
    assert args.group_size == 2


def test_ep_ring_microbatch_overlap_requires_multiple_microbatches() -> None:
    args = _parser().parse_args(["--group-size", "1"])

    with pytest.raises(ValueError, match="group_size must be at least 2"):
        _validate_args(args)


def test_ep_ring_microbatch_overlap_forward_parity_uses_fixed_threshold() -> None:
    reference = (
        (jnp.asarray([1.0, 2.0], dtype=jnp.float32), jnp.asarray(0)),
        (jnp.asarray([3.0, 4.0], dtype=jnp.float32), jnp.asarray(1)),
    )
    exact = _parity_metrics(reference, reference)
    outside_tolerance = (
        (jnp.asarray([1.01, 2.0], dtype=jnp.float32), jnp.asarray(0)),
        reference[1],
    )
    rejected = _parity_metrics(outside_tolerance, reference)

    assert exact["passed"] is True
    assert exact["all_bitwise_equal"] is True
    assert rejected["max_relative_l2_error"] > _PARITY_RELATIVE_L2


def test_ep_ring_microbatch_overlap_training_loss_means_tokens_after_hidden_sum() -> None:
    output = jnp.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.bfloat16)

    assert float(_training_loss(output)) == pytest.approx((1.0 + 4.0 + 9.0 + 16.0) / 2.0)


def test_ep_ring_microbatch_overlap_vag_parity_uses_ordered_shared_gradient_sum() -> None:
    loss_0 = jnp.asarray(2.0, dtype=jnp.float32)
    loss_1 = jnp.asarray(3.0, dtype=jnp.float32)
    output_0 = jnp.asarray([[1.0, 2.0]], dtype=jnp.bfloat16)
    output_1 = jnp.asarray([[3.0, 4.0]], dtype=jnp.bfloat16)
    x_gradient_0 = jnp.asarray([[0.1, 0.2]], dtype=jnp.bfloat16)
    x_gradient_1 = jnp.asarray([[0.3, 0.4]], dtype=jnp.bfloat16)
    combine_gradient_0 = jnp.asarray([[0.5]], dtype=jnp.float32)
    combine_gradient_1 = jnp.asarray([[0.6]], dtype=jnp.float32)
    w13_gradient_0 = jnp.asarray([1.0, 2.0], dtype=jnp.bfloat16)
    w13_gradient_1 = jnp.asarray([3.0, 4.0], dtype=jnp.bfloat16)
    w2_gradient_0 = jnp.asarray([5.0, 6.0], dtype=jnp.bfloat16)
    w2_gradient_1 = jnp.asarray([7.0, 8.0], dtype=jnp.bfloat16)
    queued = (
        (
            (loss_0, (output_0, jnp.asarray(0))),
            (x_gradient_0, combine_gradient_0, w13_gradient_0, w2_gradient_0),
        ),
        (
            (loss_1, (output_1, jnp.asarray(1))),
            (x_gradient_1, combine_gradient_1, w13_gradient_1, w2_gradient_1),
        ),
    )
    fused = (
        (
            loss_0 + loss_1,
            (
                (loss_0, output_0, jnp.asarray(0)),
                (loss_1, output_1, jnp.asarray(1)),
            ),
        ),
        (
            (x_gradient_0, x_gradient_1),
            (combine_gradient_0, combine_gradient_1),
            w13_gradient_0 + w13_gradient_1,
            w2_gradient_0 + w2_gradient_1,
        ),
    )

    parity = _value_and_grad_parity_metrics(fused, queued)
    bad_fused = (
        fused[0],
        (
            fused[1][0],
            (combine_gradient_0 + 0.01, combine_gradient_1),
            fused[1][2],
            fused[1][3],
        ),
    )
    bad_parity = _value_and_grad_parity_metrics(bad_fused, queued)
    promotion = _promotion(
        parity,
        {
            "queued_full": {"p50_ms": 11.2},
            "fused_full": {"p50_ms": 10.0},
        },
    )

    assert parity["passed"] is True
    assert parity["all_finite"] is True
    assert parity["gradients"]["w13"]["bitwise_equal"] is True
    assert parity["gradients"]["w2"]["bitwise_equal"] is True
    assert parity["shared_gradient_reference"] == "ordered_sum_of_queued_microbatch_gradients"
    assert bad_parity["passed"] is False
    assert bad_parity["gradients"]["combine_weights"][0]["accepted"] is False
    assert promotion["passed"] is True
    assert promotion["promoted_arm"] == "fused_full"
    assert promotion["basis"] == "value_and_grad_p50"
    assert promotion["required_speedup"] == _VAG_PROMOTION_SPEEDUP
    assert promotion["speedup_vs_queued_full"] == pytest.approx(1.12)
    assert promotion["parity_passed"] is True
    assert promotion["phased_forward_can_promote"] is False


def test_ep_ring_microbatch_overlap_alternates_all_arm_orders() -> None:
    calls = {"queued_full": 0, "phased": 0, "fused_full": 0}

    def runner(arm: str) -> None:
        calls[arm] += 1

    timings, orders = _time_arms(
        {arm: partial(runner, arm) for arm in calls},
        arm_names=tuple(calls),
        warmup=0,
        iterations=6,
    )

    assert len({tuple(order) for order in orders}) == 6
    assert all(set(order) == set(calls) for order in orders)
    assert calls == {"queued_full": 6, "phased": 6, "fused_full": 6}
    assert all(timing["samples"] == 6 for timing in timings.values())

    vag_calls = {"queued_full": 0, "fused_full": 0}

    def vag_runner(arm: str) -> None:
        vag_calls[arm] += 1

    vag_timings, vag_orders = _time_arms(
        {arm: partial(vag_runner, arm) for arm in vag_calls},
        arm_names=tuple(vag_calls),
        warmup=0,
        iterations=2,
    )

    assert vag_orders == [["queued_full", "fused_full"], ["fused_full", "queued_full"]]
    assert vag_calls == {"queued_full": 2, "fused_full": 2}
    assert all(timing["samples"] == 2 for timing in vag_timings.values())
