# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from functools import partial

import equinox as eqx
import numpy as np
import pytest

import jax
import jax.numpy as jnp
from haliax.quantization import OverwriteWithGradient
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P

from levanter.grug._moe.common import MoeRaggedDotOps
from levanter.grug._moe.ep_common import _pmax_replicated_cotangent
from levanter.grug._moe.ep_ring import _moe_mlp_ep_ring_local


class _ScaledRaggedDotOp(eqx.Module):
    scale: jax.Array

    def __call__(self, lhs, rhs, group_sizes):
        return jax.lax.ragged_dot(lhs * self.scale, rhs, group_sizes)


class _OverwriteState(OverwriteWithGradient):
    value: jax.Array


def test_replicated_cotangent_maxes_overwrite_state_and_sums_plain_gradient() -> None:
    if len(jax.devices()) < 2:
        pytest.skip("requires at least two devices")
    devices = np.asarray(jax.devices()).reshape(1, len(jax.devices()), 1)
    mesh = Mesh(
        devices,
        axis_names=("data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )
    per_shard = jnp.arange(1.0, len(jax.devices()) + 1.0)
    weights = jax.device_put(per_shard, NamedSharding(mesh, P("expert")))

    def shard_fn(tree, local_weight):
        tree = _pmax_replicated_cotangent(tree)
        return (tree["overwrite"].value + tree["plain"]) * local_weight

    mapped = jax.shard_map(
        shard_fn,
        mesh=mesh,
        in_specs=(P(), P("expert")),
        out_specs=P("expert"),
        check_vma=False,
    )

    def loss(tree):
        return jnp.sum(mapped(tree, weights))

    tree = {"overwrite": _OverwriteState(jnp.ones(())), "plain": jnp.ones(())}
    with jax.set_mesh(mesh):
        gradient = jax.grad(loss)(tree)

    np.testing.assert_allclose(float(gradient["overwrite"].value), float(per_shard.max()), rtol=1e-6)
    np.testing.assert_allclose(float(gradient["plain"]), float(per_shard.sum()), rtol=1e-6)


def test_ring_stateful_ragged_dot_ops_match_default_and_receive_gradients() -> None:
    devices = np.asarray(jax.devices()).reshape(1, len(jax.devices()), 1)
    mesh = Mesh(
        devices,
        axis_names=("data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )
    tokens = len(jax.devices()) * 8
    num_experts = len(jax.devices()) * 2
    topk = 2
    key_x, key_weights, key_w13, key_w2 = jax.random.split(jax.random.key(0), 4)
    x = jax.random.normal(key_x, (tokens, 4), dtype=jnp.float32)
    selected_experts = jnp.arange(tokens * topk, dtype=jnp.int32).reshape(tokens, topk) % num_experts
    combine_weights = jax.nn.softmax(jax.random.normal(key_weights, (tokens, topk)), axis=-1)
    w13 = jax.random.normal(key_w13, (num_experts, 4, 12), dtype=jnp.float32) * 0.05
    w2 = jax.random.normal(key_w2, (num_experts, 6, 4), dtype=jnp.float32) * 0.05
    ops = MoeRaggedDotOps(
        w13=_ScaledRaggedDotOp(jnp.ones(())),
        w2=_ScaledRaggedDotOp(jnp.ones(())),
    )
    local_fn = partial(
        _moe_mlp_ep_ring_local,
        activation_fn=jax.nn.silu,
        num_experts=num_experts,
        capacity_factor=1.0,
    )
    batch_spec = P(("data", "expert"), None)
    expert_spec = P("expert", None, None)
    default = jax.shard_map(
        local_fn,
        mesh=mesh,
        in_specs=(batch_spec, batch_spec, batch_spec, expert_spec, expert_spec),
        out_specs=(batch_spec, P()),
        check_vma=False,
    )
    injected = jax.shard_map(
        local_fn,
        mesh=mesh,
        in_specs=(batch_spec, batch_spec, batch_spec, expert_spec, expert_spec, P()),
        out_specs=(batch_spec, P()),
        check_vma=False,
    )

    with jax.set_mesh(mesh):
        batch_sharding = NamedSharding(mesh, batch_spec)
        expert_sharding = NamedSharding(mesh, expert_spec)
        x = jax.sharding.reshard(x, batch_sharding)
        selected_experts = jax.sharding.reshard(selected_experts, batch_sharding)
        combine_weights = jax.sharding.reshard(combine_weights, batch_sharding)
        w13 = jax.sharding.reshard(w13, expert_sharding)
        w2 = jax.sharding.reshard(w2, expert_sharding)
        expected, expected_dropped = default(x, selected_experts, combine_weights, w13, w2)
        actual, actual_dropped = injected(x, selected_experts, combine_weights, w13, w2, ops)

        def loss(state):
            output, _ = injected(x, selected_experts, combine_weights, w13, w2, state)
            return jnp.sum(jnp.square(output))

        state_gradient = jax.grad(loss)(ops)

    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-5, atol=1e-6)
    assert int(actual_dropped) == int(expected_dropped)
    assert np.isfinite(np.asarray(state_gradient.w13.scale)).all()
    assert np.isfinite(np.asarray(state_gradient.w2.scale)).all()
    assert float(jnp.abs(state_gradient.w13.scale)) > 0.0
    assert float(jnp.abs(state_gradient.w2.scale)) > 0.0
