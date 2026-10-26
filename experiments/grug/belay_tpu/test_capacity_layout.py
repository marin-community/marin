# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare capacity layouts with an independent expert-prefix oracle."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.grug.belay_tpu import local_experts
from experiments.grug.belay_tpu.capacity_layout import factorized_route_counts, interval_route_layout


def oracle(flat, accepted):
    keep = np.zeros(flat.shape, bool)
    for expert, count in enumerate(accepted):
        keep[np.flatnonzero(flat == expert)[:count]] = True
    order = np.argsort(np.where(keep, flat, len(accepted)), kind="stable")
    return order, np.argsort(order), keep


@pytest.mark.parametrize("size,experts", [(57, 7), (1280, 384), (327680, 384)])
@pytest.mark.parametrize("distribution", ["balanced", "skewed", "last_expert"])
@pytest.mark.parametrize("acceptance", ["all", "none", "partial"])
def test_integer_capacity_layout(size, experts, distribution, acceptance):
    rng = np.random.default_rng(427)
    flat = rng.integers(experts, size=size, dtype=np.int32)
    if distribution == "skewed":
        flat[rng.random(size) < 0.95] = 0
    elif distribution == "last_expert":
        flat[:] = experts - 1
    counts = np.bincount(flat, minlength=experts).astype(np.int32)
    accepted = counts.copy()
    if acceptance == "none":
        accepted[:] = 0
    elif acceptance == "partial":
        accepted = rng.integers(counts + 1, dtype=np.int32)
    actual_counts = jax.jit(lambda ids: factorized_route_counts(ids, experts))(jnp.asarray(flat))
    np.testing.assert_array_equal(actual_counts, counts)
    expected = oracle(flat, accepted)
    for layout in (local_experts.route_layout, interval_route_layout):
        actual = jax.jit(layout)(jnp.asarray(flat), actual_counts, jnp.asarray(accepted))
        for got, want in zip(actual, expected, strict=True):
            np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("acceptance", ["all", "none", "partial"])
def test_local_expert_values_and_gradients(monkeypatch, dtype, acceptance):
    rng = np.random.default_rng(428)
    tokens, topk, experts, width, hidden = 17, 3, 7, 16, 32
    ids = jnp.asarray(rng.integers(experts, size=(tokens, topk)), jnp.int32)
    counts = np.bincount(np.asarray(ids).ravel(), minlength=experts)
    accepted = counts if acceptance == "all" else np.zeros_like(counts)
    if acceptance == "partial":
        accepted = rng.integers(counts + 1)
    accepted = jnp.asarray(accepted, jnp.int32)
    shapes = ((tokens, width), (tokens, topk), (experts, width, hidden), (experts, hidden, width))
    values = tuple(jnp.asarray(rng.normal(size=shape) * 0.1, dtype) for shape in shapes)
    dy = jnp.asarray(rng.normal(size=(tokens, width)), dtype)

    def evaluate(x, weights, up, down):
        return local_experts.local_expert_mlp(x, ids, weights, up, down, accepted_counts=accepted)[0]

    def values_and_gradients():
        output, backward = jax.vjp(evaluate, *values)
        return output, backward(dy)

    expected = values_and_gradients()
    monkeypatch.setattr(local_experts, "route_counts", factorized_route_counts)
    monkeypatch.setattr(local_experts, "route_layout", interval_route_layout)
    actual = values_and_gradients()
    tolerance = 1e-4 if dtype == jnp.float32 else 2e-2
    for got, want in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        assert np.isfinite(got).all()
        np.testing.assert_allclose(
            np.asarray(got, np.float32), np.asarray(want, np.float32), atol=tolerance, rtol=tolerance
        )
