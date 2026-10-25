# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Remote CPU operator oracles; these tests perform no optimizer updates."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh

from experiments.grug.belay_tpu.local_experts import clip_capacity_pools, gathered_expert_mlp


def accepted_oracle(selected, experts, pools, capacity):
    keep = np.zeros(selected.shape, dtype=bool)
    for pool in range(pools):
        remaining = capacity
        for expert in range(pool * experts // pools, (pool + 1) * experts // pools):
            positions = np.flatnonzero(selected.reshape(-1) == expert)[:remaining]
            keep.reshape(-1)[positions] = True
            remaining -= len(positions)
    return keep


@pytest.mark.parametrize("senders", [2, 4, 8])
def test_capacity_is_independent_of_sender_count(senders):
    rng = np.random.default_rng(43)
    selected = rng.integers(0, 8, (32, 2))
    counts = np.stack([np.bincount(part.reshape(-1), minlength=8) for part in np.split(selected, senders)])
    keep = accepted_oracle(selected, 8, 2, 27)
    expected = np.stack(
        [
            np.bincount(ids[mask], minlength=8)
            for ids, mask in zip(np.split(selected, senders), np.split(keep, senders), strict=True)
        ]
    )
    np.testing.assert_array_equal(clip_capacity_pools(jnp.array(counts), pools=2, capacity=27), expected)


@pytest.mark.parametrize("skewed", [False, True])
@pytest.mark.parametrize("capacity_factor", [0.3, 1.05, 3.0])
def test_sharded_forward_and_gradients(skewed, capacity_factor):
    if jax.device_count() < 4:
        pytest.skip("Requires --xla_force_host_platform_device_count=4 for remote CPU operator checks")
    rng = np.random.default_rng(92)
    tokens, experts, topk = 32, 8, 2
    selected = rng.integers(0, 2 if skewed else experts, (tokens, topk), dtype=np.int32)
    x = jnp.asarray(rng.normal(size=(tokens, 4)), jnp.float32)
    up = jnp.asarray(rng.normal(scale=0.2, size=(experts, 4, 6)), jnp.float32)
    down = jnp.asarray(rng.normal(scale=0.2, size=(experts, 6, 4)), jnp.float32)
    weights = jnp.asarray(rng.uniform(size=(tokens, topk)), jnp.float32).at[0, 0].set(0)
    capacity = max(experts // 2, math.ceil(capacity_factor * selected.size / 2))
    keep = jnp.asarray(accepted_oracle(selected, experts, 2, capacity))
    ids = jnp.asarray(selected)
    cotangent = jnp.asarray(rng.normal(size=(tokens, 4)), jnp.float32)
    mesh = Mesh(np.array(jax.devices()[:4]), ("expert",), axis_types=(AxisType.Explicit,))

    def dense(x, weights, up, down):
        hidden = jnp.square(jax.nn.relu(jnp.einsum("td,tkdh->tkh", x, up[ids])))
        slots = jnp.einsum("tkh,tkhd->tkd", hidden, down[ids])
        return (jnp.where(keep[..., None], slots, 0) * weights[..., None]).sum(axis=1)

    def candidate(x, weights, up, down):
        return gathered_expert_mlp(
            x, ids, weights, up, down, mesh=mesh, capacity_factor=capacity_factor, capacity_pools=2
        )[0]

    def objective(fn):
        def loss(*values):
            y = fn(*values)
            return (y * cotangent).sum(), y

        return jax.jit(jax.value_and_grad(loss, argnums=(0, 1, 2, 3), has_aux=True))

    with jax.set_mesh(mesh):
        expected = objective(dense)(x, weights, up, down)
        observed = objective(candidate)(x, weights, up, down)
        _, counts = jax.jit(
            lambda: gathered_expert_mlp(
                x, ids, weights, up, down, mesh=mesh, capacity_factor=capacity_factor, capacity_pools=2
            )
        )()
    assert int(counts.dropped) == selected.size - int(keep.sum())
    for actual, wanted in zip(jax.tree.leaves(observed), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(actual, wanted, atol=3e-6, rtol=3e-5)
