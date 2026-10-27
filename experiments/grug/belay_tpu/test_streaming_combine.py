# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check route sums and custom pullbacks against direct grouped outputs."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.grug.belay_tpu.local_experts import combine_local
from experiments.grug.belay_tpu.streaming_combine import (
    combine_streaming,
    dispatch_streaming,
    sum_routes,
    sum_routes_blocked,
)
from experiments.grug.belay_tpu.streaming_combine_check import routing_weight_gradient_reference


@pytest.mark.parametrize("implementation", ["unrolled", "scan", "flat"])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("scalar_permutation", ["gather", "sort"])
@pytest.mark.parametrize("weight_dtype", [jnp.float32, jnp.bfloat16])
def test_streaming_values_and_pullbacks(implementation, dtype, empty, scalar_permutation, weight_dtype):
    rng = np.random.default_rng(424)
    tokens, topk, width = 31, 3, 16
    order = jnp.asarray(rng.permutation(tokens * topk), jnp.int32)
    inverse = jnp.argsort(order)
    rows = jnp.asarray(rng.normal(size=(tokens * topk, width)), dtype)
    rows = rows.at[0 if empty else 73 :].set(0)
    weights = jnp.asarray(rng.random((tokens, topk)), weight_dtype).at[:, 0].set(0)
    gradient = jnp.asarray(rng.normal(size=(tokens, width)), dtype)

    def evaluate(combine):
        output, backward = jax.vjp(lambda x, w: combine(x, order, inverse, w), rows, weights)
        return output, backward(gradient)

    expected = evaluate(combine_local)
    actual = evaluate(
        functools.partial(combine_streaming, implementation=implementation, scalar_permutation=scalar_permutation)
    )
    tolerance = 1e-4 if dtype == jnp.float32 else 2e-2
    for got, want in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        assert np.isfinite(got).all()
        np.testing.assert_allclose(
            np.asarray(got, np.float32), np.asarray(want, np.float32), atol=tolerance, rtol=tolerance
        )
    # Dispatch's pullback is the unweighted route sum, including rejected rows.
    expected_sum = np.asarray(rows, np.float32)[np.asarray(inverse)].reshape(tokens, topk, width).sum(axis=1)
    observed = sum_routes(rows, inverse.reshape(tokens, topk), implementation=implementation)
    np.testing.assert_allclose(np.asarray(observed, np.float32), expected_sum, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_dispatch_gradient_sums_duplicate_token_routes(dtype):
    rng = np.random.default_rng(425)
    tokens, topk, width = 31, 3, 16
    order = jnp.asarray(rng.permutation(tokens * topk), jnp.int32)
    inverse = jnp.argsort(order)
    x = jnp.asarray(rng.normal(size=(tokens, width)), dtype)
    gradient = jnp.asarray(rng.normal(size=(tokens * topk, width)), dtype)
    output, backward = jax.vjp(lambda value: dispatch_streaming(value, order, inverse, topk), x)
    token_ids = np.asarray(order) // topk
    np.testing.assert_array_equal(np.asarray(output), np.asarray(x)[token_ids])
    expected = np.zeros(x.shape, np.float32)
    np.add.at(expected, token_ids, np.asarray(gradient, np.float32))
    tolerance = 1e-4 if dtype == jnp.float32 else 2e-2
    np.testing.assert_allclose(np.asarray(backward(gradient)[0], np.float32), expected, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("width", [16, 256])
@pytest.mark.parametrize("empty", [False, True])
def test_routing_weight_reference_matches_autodiff(width, empty):
    rng = np.random.default_rng(436)
    tokens, topk = 17, 3
    order = rng.permutation(tokens * topk)
    inverse = np.argsort(order)
    rows = rng.normal(size=(tokens * topk, width))
    rows[0 if empty else 37 :] = 0
    upstream = rng.normal(size=(tokens, width))
    actual = routing_weight_gradient_reference(rows, order, inverse, upstream, topk=topk)
    with jax.enable_x64():
        slots = jnp.asarray(rows[inverse].reshape(tokens, topk, width))

        def objective(weights):
            combined = jnp.sum(slots * weights[..., None], axis=1)
            return jnp.sum(combined * jnp.asarray(upstream))

        expected = jax.grad(objective)(jnp.ones((tokens, topk), jnp.float64))
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("backward_tile", [0, 17])
@pytest.mark.parametrize("tile_tokens", [8, 32])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("empty", [False, True])
def test_token_tiles_match_weighted_outputs_and_pullbacks(backward_tile, tile_tokens, dtype, empty):
    rng = np.random.default_rng(437)
    tokens, topk, width = 31, 10, 16
    order = jnp.asarray(rng.permutation(tokens * topk), jnp.int32)
    inverse = jnp.argsort(order)
    rows = jnp.asarray(rng.normal(size=(tokens * topk, width)), dtype)
    rows = rows.at[0 if empty else 279 :].set(0)
    weights = jnp.asarray(rng.random((tokens, topk)), dtype).at[:, 0].set(0)
    gradient = jnp.asarray(rng.normal(size=(tokens, width)), dtype)

    def evaluate(combine):
        output, backward = jax.vjp(lambda x, w: combine(x, order, inverse, w), rows, weights)
        return output, backward(gradient)

    actual = evaluate(functools.partial(combine_streaming, token_tile=tile_tokens, backward_tile=backward_tile))
    expected = evaluate(combine_local)
    tolerance = 1e-4 if dtype == jnp.float32 else 2e-2
    for got, want in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        assert np.isfinite(got).all()
        np.testing.assert_allclose(
            np.asarray(got, np.float32), np.asarray(want, np.float32), atol=tolerance, rtol=tolerance
        )
    expected_sum = np.asarray(rows, np.float32)[np.asarray(inverse)].reshape(tokens, topk, width).sum(axis=1)
    actual_sum = sum_routes_blocked(rows, inverse.reshape(tokens, topk), tile_tokens=tile_tokens)
    np.testing.assert_allclose(np.asarray(actual_sum, np.float32), expected_sum, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("backward_tile", [0, 17, 512])
@pytest.mark.parametrize("implementation", ["unrolled", "scan", "flat", "blocked"])
@pytest.mark.parametrize("empty", [False, True])
def test_fp32_token_output_preserves_bf16_route_gradients(backward_tile, implementation, empty):
    rng = np.random.default_rng(439)
    tokens, topk, width = 31, 10, 256
    order = jnp.asarray(rng.permutation(tokens * topk), jnp.int32)
    inverse = jnp.argsort(order)
    rows = jnp.asarray(rng.normal(size=(tokens * topk, width)), jnp.bfloat16)
    rows = rows.at[0 if empty else 279 :].set(0)
    weights = jnp.asarray(rng.random((tokens, topk)), jnp.bfloat16).at[:, 0].set(0)
    gradient = jnp.asarray(rng.normal(size=(tokens, width)), jnp.float32)

    def reference(x, w):
        slots = x.astype(jnp.float32)[inverse].reshape(tokens, topk, width)
        return (slots * w.astype(jnp.float32)[..., None]).sum(axis=1)

    def candidate(x, w):
        return combine_streaming(
            x,
            order,
            inverse,
            w,
            implementation="scan" if implementation == "blocked" else implementation,
            token_tile=8 if implementation == "blocked" else 0,
            output_dtype=jnp.float32,
            backward_tile=backward_tile,
        )

    expected, expected_backward = jax.vjp(reference, rows, weights)
    actual, actual_backward = jax.vjp(candidate, rows, weights)
    assert actual.dtype == jnp.float32
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)
    for got, want in zip(actual_backward(gradient), expected_backward(gradient), strict=True):
        assert np.isfinite(got).all()
        np.testing.assert_allclose(np.asarray(got, np.float32), np.asarray(want, np.float32), atol=2e-2, rtol=2e-2)
