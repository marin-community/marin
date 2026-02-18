# Copyright 2025 The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0

import jax.numpy as jnp

import haliax as hax
from haliax.nn.loss import cross_entropy_loss, maybe_reduce_loss


def test_maybe_reduce_loss_weighted_mean():
    Batch, Pos = hax.make_axes(Batch=2, Pos=3)
    arr = hax.named(jnp.arange(6, dtype=jnp.float32).reshape(2, 3), (Batch, Pos))
    weights = hax.named(jnp.array([[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]], dtype=jnp.float32), (Batch, Pos))

    reduced = maybe_reduce_loss(arr, hax.mean, Pos, where=None, weight=weights)

    expected = jnp.array(
        [
            (0 * 1 + 1 * 0 + 2 * 2) / (1 + 0 + 2),
            (3 * 0 + 4 * 1 + 5 * 1) / (0 + 1 + 1),
        ]
    )
    assert jnp.allclose(reduced.array, expected)


def test_maybe_reduce_loss_weighted_sum():
    Batch, Pos = hax.make_axes(Batch=2, Pos=2)
    arr = hax.named(jnp.array([[1.0, 2.0], [3.0, 4.0]]), (Batch, Pos))
    weights = hax.named(jnp.array([[0.5, 1.5], [1.0, 0.0]], dtype=jnp.float32), (Batch, Pos))

    reduced = maybe_reduce_loss(arr, hax.sum, Pos, where=None, weight=weights)

    expected = jnp.array(
        [
            1.0 * 0.5 + 2.0 * 1.5,
            3.0 * 1.0 + 4.0 * 0.0,
        ]
    )
    assert jnp.allclose(reduced.array, expected)


def test_cross_entropy_loss_respects_weights():
    Batch, Label = hax.make_axes(Batch=3, Label=2)
    logits = hax.named(jnp.array([[3.0, 1.0], [1.0, 3.0], [2.0, 2.0]]), (Batch, Label))
    targets = hax.nn.one_hot(hax.named(jnp.array([0, 1, 0], dtype=jnp.int32), Batch), Label)
    weights = hax.named(jnp.array([1.0, 0.0, 2.0], dtype=jnp.float32), Batch)

    per_example = cross_entropy_loss(logits, Label, targets, reduction=None, reduction_axis=())
    manual = jnp.sum(per_example.array * weights.array) / jnp.sum(weights.array)

    weighted = cross_entropy_loss(
        logits,
        Label,
        targets,
        reduction=hax.mean,
        reduction_axis=None,
        weight=weights,
    )

    assert jnp.allclose(weighted.array, manual)
