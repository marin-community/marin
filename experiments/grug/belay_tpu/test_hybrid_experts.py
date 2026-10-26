# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check hybrid expert ownership, global capacity and all differentiable inputs."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh

from experiments.grug.belay_tpu.hybrid_experts import hybrid_gathered_expert_mlp


def accepted_routes(ids, experts, pools, capacity_factor):
    capacity = max(experts // pools, math.ceil(capacity_factor * ids.size / pools))
    keep = np.zeros_like(ids, dtype=bool)
    for pool in range(pools):
        remaining = capacity
        for expert in range(pool * experts // pools, (pool + 1) * experts // pools):
            positions = np.flatnonzero(ids.ravel() == expert)[:remaining]
            keep.ravel()[positions] = True
            remaining -= positions.size
    return keep


def dense_reference(x, weights, up, down, ids, keep):
    routed = jnp.einsum("td,tkdh->tkh", x, up[ids]).astype(x.dtype)
    hidden = jnp.square(jax.nn.relu(routed))
    outputs = jnp.einsum("tkh,tkhd->tkd", hidden, down[ids]).astype(x.dtype)
    outputs = jnp.where(keep[..., None], outputs, 0)
    return (outputs.astype(jnp.float32) * weights[..., None].astype(jnp.float32)).sum(axis=1).astype(x.dtype)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("distribution", ["balanced", "skewed", "last_expert"])
@pytest.mark.parametrize("group_size", [2, 4])
@pytest.mark.parametrize("peers", [4, 16])
def test_hybrid_experts_preserve_global_capacity_and_gradients(dtype, distribution, group_size, peers):
    assert len(jax.devices()) >= peers, "Run this non-training oracle with sixteen CPU devices or TPU devices"
    mesh = Mesh(np.asarray(jax.devices()[:peers]), ("expert",), axis_types=(AxisType.Explicit,))
    rng = np.random.default_rng(438)
    tokens, topk, experts, width, hidden, pools = 48, 3, 32, 8, 16, 16
    ids = rng.integers(experts, size=(tokens, topk), dtype=np.int32)
    if distribution == "skewed":
        ids[rng.random(ids.shape) < 0.85] = 0
    elif distribution == "last_expert":
        ids[:] = experts - 1
    keep = jnp.asarray(accepted_routes(ids, experts, pools, 1.05))
    shapes = ((tokens, width), (tokens, topk), (experts, width, hidden), (experts, hidden, width))
    values = tuple(jnp.asarray(rng.normal(scale=0.2, size=shape), dtype) for shape in shapes)
    dy = jnp.asarray(rng.normal(size=(tokens, width)), dtype)
    ids = jnp.asarray(ids)

    def candidate(x, weights, up, down):
        return hybrid_gathered_expert_mlp(
            x,
            ids,
            weights,
            up,
            down,
            mesh=mesh,
            capacity_factor=1.05,
            capacity_pools=pools,
            expert_group_size=group_size,
        )[0]

    expected, reference_backward = jax.vjp(lambda *args: dense_reference(*args, ids, keep), *values)
    actual, backward = jax.vjp(candidate, *values)
    results = (actual, *backward(jax.device_put(dy, actual.sharding)))
    references = (expected, *reference_backward(dy))
    tolerance = 1e-4 if dtype == jnp.float32 else 2e-2
    for got, want in zip(results, references, strict=True):
        assert np.isfinite(got).all()
        np.testing.assert_allclose(
            np.asarray(got, np.float32), np.asarray(want, np.float32), atol=tolerance, rtol=tolerance
        )
    _, drops = hybrid_gathered_expert_mlp(
        values[0],
        ids,
        values[1],
        values[2],
        values[3],
        mesh=mesh,
        capacity_factor=1.05,
        capacity_pools=pools,
        expert_group_size=group_size,
    )
    assert int(drops.receiver_dropped) == ids.size - int(keep.sum())
