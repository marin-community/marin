# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Routing values, derivatives and capacity masks; no optimizer or training."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.grug.belay_tpu.gather_layout import packed_gather
from experiments.grug.belay_tpu.local_movement import combine_expert_order
from experiments.grug.belay_tpu.routing_tpu import (
    accepted_assignments_cutoff,
    permute_assignment_weights_sort,
    restore_assignment_scalars_sort,
    unrolled_router_logits,
)


@pytest.mark.parametrize("experts,topk", [(32, 4), (384, 10)])
@pytest.mark.parametrize("case", ["random", "ties", "signed_zero"])
def test_router_selection_and_transpose_preserve_selected_values(experts, topk, case):
    rng = np.random.RandomState(301)
    logits = rng.normal(size=(31, experts)).astype(np.float32)
    if case == "ties":
        logits = np.round(logits)
    elif case == "signed_zero":
        logits[:, ::2], logits[:, 1::2] = 0.0, -0.0
    selected = np.argsort(-logits, axis=-1, kind="stable")[:, :topk].astype(np.int32)
    cotangent = rng.normal(size=selected.shape).astype(np.float32)
    cotangent.flat[::7] = -0.0

    def evaluate(x, indices, dy):
        output, pullback = jax.vjp(lambda value: unrolled_router_logits(value, indices), x)
        return output, pullback(dy)[0]

    output, gradient = jax.jit(evaluate)(logits, selected, cotangent)
    rows = np.arange(logits.shape[0])[:, None]
    expected_gradient = np.zeros_like(logits)
    expected_gradient[rows, selected] = np.where(cotangent == 0, np.float32(0), cotangent)
    np.testing.assert_array_equal(np.asarray(output).view(np.uint32), logits[rows, selected].view(np.uint32))
    np.testing.assert_array_equal(np.asarray(gradient).view(np.uint32), expected_gradient.view(np.uint32))


@pytest.mark.parametrize("case", ["mixed", "all_padding", "concentrated"])
def test_capacity_cutoffs_keep_the_same_expert_prefixes(case):
    rng = np.random.RandomState(302)
    experts = 24
    selected = rng.randint(0, experts + 1, 131, dtype=np.int32)
    if case == "all_padding":
        selected.fill(experts)
    elif case == "concentrated":
        selected[:] = 3
        selected[::7] = experts
    permutation = np.argsort(selected, kind="stable").astype(np.int32)
    groups = np.bincount(selected, minlength=experts + 1)[:experts].astype(np.int32)
    accepted = np.asarray([rng.randint(0, size + 1) for size in groups], np.int32)
    expected = np.zeros(selected.shape, bool)
    for expert, size in enumerate(accepted):
        expected[np.flatnonzero(selected == expert)[:size]] = True
    actual = jax.jit(accepted_assignments_cutoff)(selected, permutation, groups, accepted)
    np.testing.assert_array_equal(actual, expected)


def test_scalar_sorts_preserve_float_payloads_in_both_directions():
    rng = np.random.RandomState(303)
    bits = rng.randint(0, 2**32, 131, dtype=np.uint32)
    values = jax.lax.bitcast_convert_type(jnp.asarray(bits), jnp.float32)
    permutation = rng.permutation(131).astype(np.int32)
    sorted_values = jax.jit(permute_assignment_weights_sort)(values, permutation)
    restored_values = jax.jit(restore_assignment_scalars_sort)(sorted_values, permutation)
    np.testing.assert_array_equal(jax.lax.bitcast_convert_type(sorted_values, jnp.uint32), bits[permutation])
    np.testing.assert_array_equal(jax.lax.bitcast_convert_type(restored_values, jnp.uint32), bits)


@pytest.mark.parametrize("topk", [3, 10])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("case", ["mixed", "empty"])
def test_expert_order_combine_matches_weighted_sum_and_derivatives(topk, dtype, case):
    rng = np.random.default_rng(391)
    tokens, width = 17, 128
    order = rng.permutation(tokens * topk).astype(np.int32)
    inverse = np.argsort(order).astype(np.int32)
    rows = jnp.asarray(rng.normal(size=(tokens * topk, width)), dtype)
    rows = rows.at[-19:].set(0)
    if case == "empty":
        rows = jnp.zeros_like(rows)
    weights = jnp.asarray(rng.random((tokens, topk)), jnp.float32)
    weights = weights.at[:, 0].set(0)
    gradient = jnp.asarray(rng.normal(size=(tokens, width)), dtype)

    def evaluate(x, w, dy):
        output, pullback = jax.vjp(lambda a, b: combine_expert_order(a, order, inverse, b), x, w)
        return output, *pullback(dy)

    actual = jax.jit(evaluate)(rows, weights, gradient)
    host_rows, host_weights, host_gradient = (np.asarray(x, np.float64) for x in (rows, weights, gradient))
    expected_output = np.zeros((tokens, width), np.float64)
    expected_rows = np.zeros_like(host_rows)
    expected_weights = np.zeros_like(host_weights)
    # A scalar route loop provides an oracle independent of the permutation VJP.
    for row, slot in enumerate(order):
        token, route = divmod(int(slot), topk)
        expected_output[token] += host_rows[row] * host_weights[token, route]
        expected_rows[row] = host_gradient[token] * host_weights[token, route]
        expected_weights[token, route] = np.dot(host_gradient[token], host_rows[row])
    expected = (jnp.asarray(expected_output, dtype), jnp.asarray(expected_rows, dtype), expected_weights)
    tolerance = 1e-4 if dtype == jnp.float32 else 2e-2
    for got, want in zip(actual, expected, strict=True):
        np.testing.assert_allclose(
            np.asarray(got, np.float32), np.asarray(want, np.float32), rtol=tolerance, atol=tolerance
        )


@pytest.mark.parametrize("barrier", [False, True])
def test_packed_gather_preserves_bf16_bits_with_repeated_rows(barrier):
    bits = np.random.default_rng(419).integers(0, 65536, size=(17, 256), dtype=np.uint16)
    rows = jax.lax.bitcast_convert_type(jnp.asarray(bits), jnp.bfloat16)
    indices = jnp.asarray([0, 16, 0, 7, 2], jnp.int32)
    result = packed_gather(rows, indices, barrier=barrier)
    actual = np.asarray(jax.lax.bitcast_convert_type(result, jnp.uint16))
    np.testing.assert_array_equal(actual, bits[np.asarray(indices)])
