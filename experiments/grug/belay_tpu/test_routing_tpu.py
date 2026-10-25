# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Routing values, derivatives and capacity masks; no optimizer or training."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

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
