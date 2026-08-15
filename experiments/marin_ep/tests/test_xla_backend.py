# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""XLA stepping-stone backend vs oracle, on a real 8-device CPU mesh.

Runs in a subprocess so `--xla_force_host_platform_device_count` applies
before jax initializes (the pattern from tests/test_moe_hero_ep.py). The
backend executes inside shard_map over a grug-shaped 4-axis mesh; values,
drop counts, and gradients must match the keep-masked dense oracle.
"""

import os
import subprocess
import sys
import textwrap
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import shard_map
from jax.sharding import AbstractMesh, AxisType, Mesh, NamedSharding, use_abstract_mesh
from jax.sharding import PartitionSpec as P

from experiments.marin_ep.kernels.xla_backend import _static_capacity, marin_ep_moe_local
from experiments.marin_ep.oracle import moe_oracle, pooled_keep_mask

_SCRIPT = """
    import jax
    import jax.numpy as jnp
    import numpy as np
    from functools import partial
    from jax import shard_map
    from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P

    from experiments.marin_ep.kernels.xla_backend import marin_ep_moe_local, _static_capacity
    from experiments.marin_ep.oracle import moe_oracle, pooled_keep_mask

    devices, tokens, topk, num_experts, hidden, intermediate = 8, 16, 3, 16, 8, 12
    group_size, capacity_factor = 2, 1.1
    local_experts = num_experts // devices

    rng = np.random.default_rng(seed=11)
    probs = rng.dirichlet(np.full(num_experts, 0.4))
    experts = rng.choice(num_experts, size=(devices * tokens, topk), p=probs).astype(np.int32)
    x = rng.standard_normal((devices * tokens, hidden)).astype(np.float32)
    weights = (rng.random((devices * tokens, topk)) + 0.05).astype(np.float32)
    w13 = (0.3 * rng.standard_normal((num_experts, hidden, 2 * intermediate))).astype(np.float32)
    w2 = (0.3 * rng.standard_normal((num_experts, intermediate, hidden))).astype(np.float32)
    cot = rng.standard_normal((devices * tokens, hidden)).astype(np.float32)

    mesh = Mesh(
        np.asarray(jax.devices()).reshape(1, 1, devices, 1),
        ("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 4,
    )
    batch_spec = P(("replica_dcn", "data", "expert"))
    weight_spec = P("expert", None, None)

    local_fn = partial(
        marin_ep_moe_local,
        activation_fn=jax.nn.silu,
        num_experts=num_experts,
        capacity_factor=capacity_factor,
        pool_group_size=group_size,
        transport="gathered",  # ragged_all_to_all is unimplemented on XLA:CPU
    )
    shard_fn = shard_map(
        local_fn,
        mesh=mesh,
        in_specs=(batch_spec, batch_spec, batch_spec, weight_spec, weight_spec),
        out_specs=(batch_spec, P()),
        check_vma=False,
    )

    put = lambda a, spec: jax.device_put(jnp.asarray(a), NamedSharding(mesh, spec))
    experts_sharded = put(experts, batch_spec)

    def forward(x_, weights_, w13_, w2_):
        y, dropped = shard_fn(x_, experts_sharded, weights_, w13_, w2_)
        return y, dropped

    args = (put(x, batch_spec), put(weights, batch_spec), put(w13, weight_spec), put(w2, weight_spec))
    with jax.set_mesh(mesh):
        y, dropped = jax.jit(forward)(*args)

    capacity = _static_capacity(devices * tokens * topk, num_experts, capacity_factor)
    keep, dropped_oracle = pooled_keep_mask(
        experts.reshape(devices, tokens, topk), num_experts=num_experts,
        capacity=capacity, group_size=group_size,
    )
    y_oracle = moe_oracle(
        jnp.asarray(x), jnp.asarray(experts), jnp.asarray(weights),
        jnp.asarray(w13), jnp.asarray(w2), jnp.asarray(keep.reshape(devices * tokens, topk)),
        activation_fn=jax.nn.silu,
    )
    assert int(dropped) == dropped_oracle, (int(dropped), dropped_oracle)
    assert dropped_oracle > 0, "test must exercise the drop path"
    np.testing.assert_allclose(np.asarray(y), np.asarray(y_oracle), rtol=1e-5, atol=1e-5)

    def loss(x_, weights_, w13_, w2_):
        y_, _ = forward(x_, weights_, w13_, w2_)
        return jnp.sum(y_ * jnp.asarray(cot))

    with jax.set_mesh(mesh):
        grads = jax.jit(jax.grad(loss, argnums=(0, 1, 2, 3)))(*args)

    def oracle_loss(x_, weights_, w13_, w2_):
        y_ = moe_oracle(
            x_, jnp.asarray(experts), weights_, w13_, w2_,
            jnp.asarray(keep.reshape(devices * tokens, topk)), activation_fn=jax.nn.silu,
        )
        return jnp.sum(y_ * jnp.asarray(cot))

    grads_oracle = jax.grad(oracle_loss, argnums=(0, 1, 2, 3))(
        jnp.asarray(x), jnp.asarray(weights), jnp.asarray(w13), jnp.asarray(w2)
    )
    for got, want, name in zip(grads, grads_oracle, ("dx", "dweights", "dw13", "dw2")):
        np.testing.assert_allclose(
            np.asarray(got), np.asarray(want), rtol=2e-5, atol=2e-5, err_msg=name
        )
    print("OK")
"""


def test_ragged_transport_abstractly_evaluates_at_hero_ep64_topology():
    """The production transport path traces at the hero EP degree and expert
    count (shape-level guard; execution needs real devices)."""
    mesh = AbstractMesh(
        axis_sizes=(1, 64, 1),
        axis_names=("data", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 3,
    )
    batch_spec = P(("data", "expert"))
    weight_spec = P("expert", None, None)
    tokens, topk, num_experts, hidden, intermediate = 64 * 128, 4, 192, 64, 96
    local_fn = partial(
        marin_ep_moe_local,
        activation_fn=jax.nn.silu,
        num_experts=num_experts,
        capacity_factor=1.33,
        pool_group_size=3,
    )
    shard_fn = shard_map(
        local_fn,
        mesh=mesh,
        in_specs=(batch_spec, batch_spec, batch_spec, weight_spec, weight_spec),
        out_specs=(batch_spec, P()),
        check_vma=False,
    )

    def sds(shape, dtype, spec):
        return jax.ShapeDtypeStruct(shape, dtype, sharding=NamedSharding(mesh, spec))

    with use_abstract_mesh(mesh):
        y, dropped = jax.eval_shape(
            shard_fn,
            sds((tokens, hidden), jnp.bfloat16, batch_spec),
            sds((tokens, topk), jnp.int32, batch_spec),
            sds((tokens, topk), jnp.float32, batch_spec),
            sds((num_experts, hidden, 2 * intermediate), jnp.bfloat16, weight_spec),
            sds((num_experts, intermediate, hidden), jnp.bfloat16, weight_spec),
        )
    assert y.shape == (tokens, hidden) and y.dtype == jnp.bfloat16
    assert dropped.shape == () and dropped.dtype == jnp.int32


def test_xla_backend_matches_oracle_values_drops_and_grads_on_8_device_mesh():
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(_SCRIPT)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


@pytest.mark.parametrize("transport", ["ragged", "mgpu"])
@pytest.mark.parametrize("group_size", [1, 2, 4])
def test_real_gpu_transport_matches_oracle(group_size, transport):
    """Production transports on real GPUs (ragged passed on GB200, MEP-006).

    Dims must be Triton-grouped-GEMM safe: haliax ragged_dot's GPU path
    silently miscomputes below its tile minimums (H=32/I=48 gives O(10)
    errors), and it accumulates in tf32 regardless of the jax matmul
    precision setting — hence the 1e-2 tolerance. For mgpu, hidden must be
    LANE-divisible, and tokens are sized so some segments exceed the SMEM
    tile height (both kernel copy paths run).
    """
    if jax.default_backend() != "gpu":
        pytest.skip("needs real GPUs")
    devices = len(jax.devices())
    if devices < 2:
        pytest.skip("needs >= 2 GPUs")
    tokens, topk, hidden, intermediate = (512 if transport == "mgpu" else 64), 3, 256, 256
    num_experts = devices * 4
    capacity_factor = 1.1

    rng = np.random.default_rng(seed=11)
    probs = rng.dirichlet(np.full(num_experts, 0.4))
    experts = rng.choice(num_experts, size=(devices * tokens, topk), p=probs).astype(np.int32)
    x = rng.standard_normal((devices * tokens, hidden)).astype(np.float32)
    weights = (rng.random((devices * tokens, topk)) + 0.05).astype(np.float32)
    w13 = (0.3 * rng.standard_normal((num_experts, hidden, 2 * intermediate))).astype(np.float32)
    w2 = (0.3 * rng.standard_normal((num_experts, intermediate, hidden))).astype(np.float32)
    cot = rng.standard_normal((devices * tokens, hidden)).astype(np.float32)

    mesh = Mesh(
        np.asarray(jax.devices()).reshape(1, 1, devices, 1),
        ("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 4,
    )
    batch_spec = P(("replica_dcn", "data", "expert"))
    weight_spec = P("expert", None, None)
    shard_fn = shard_map(
        partial(
            marin_ep_moe_local,
            activation_fn=jax.nn.silu,
            num_experts=num_experts,
            capacity_factor=capacity_factor,
            pool_group_size=group_size,
            transport=transport,
        ),
        mesh=mesh,
        in_specs=(batch_spec, batch_spec, batch_spec, weight_spec, weight_spec),
        out_specs=(batch_spec, P()),
        check_vma=False,
    )

    def put(a, spec):
        return jax.device_put(jnp.asarray(a), NamedSharding(mesh, spec))

    experts_sharded = put(experts, batch_spec)

    def forward(x_, weights_, w13_, w2_):
        return shard_fn(x_, experts_sharded, weights_, w13_, w2_)

    args = (put(x, batch_spec), put(weights, batch_spec), put(w13, weight_spec), put(w2, weight_spec))
    with jax.set_mesh(mesh):
        y, dropped = jax.jit(forward)(*args)

    capacity = _static_capacity(devices * tokens * topk, num_experts, capacity_factor)
    keep, dropped_oracle = pooled_keep_mask(
        experts.reshape(devices, tokens, topk), num_experts=num_experts, capacity=capacity, group_size=group_size
    )
    keep_flat = jnp.asarray(keep.reshape(devices * tokens, topk))
    y_oracle = moe_oracle(
        jnp.asarray(x),
        jnp.asarray(experts),
        jnp.asarray(weights),
        jnp.asarray(w13),
        jnp.asarray(w2),
        keep_flat,
        activation_fn=jax.nn.silu,
    )
    assert int(dropped) == dropped_oracle
    assert dropped_oracle > 0  # the drop path must be exercised
    np.testing.assert_allclose(np.asarray(y), np.asarray(y_oracle), rtol=1e-2, atol=1e-2)

    def loss(x_, weights_, w13_, w2_):
        y_, _ = forward(x_, weights_, w13_, w2_)
        return jnp.sum(y_ * jnp.asarray(cot))

    with jax.set_mesh(mesh):
        grads = jax.jit(jax.grad(loss, argnums=(0, 1, 2, 3)))(*args)

    def oracle_loss(x_, weights_, w13_, w2_):
        y_ = moe_oracle(x_, jnp.asarray(experts), weights_, w13_, w2_, keep_flat, activation_fn=jax.nn.silu)
        return jnp.sum(y_ * jnp.asarray(cot))

    grads_oracle = jax.grad(oracle_loss, argnums=(0, 1, 2, 3))(
        jnp.asarray(x), jnp.asarray(weights), jnp.asarray(w13), jnp.asarray(w2)
    )
    for got, want, name in zip(grads, grads_oracle, ("dx", "dweights", "dw13", "dw2"), strict=True):
        np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-2, atol=1e-2, err_msg=name)
