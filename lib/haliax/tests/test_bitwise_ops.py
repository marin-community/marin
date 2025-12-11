# Copyright 2025 The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0

import jax.numpy as jnp
import haliax as hax
from haliax import Axis


def test_bitwise_count_invert():
    A = Axis("A", 4)
    x = hax.named(jnp.array([0, 1, 2, 3], dtype=jnp.uint8), (A,))

    inv = hax.bitwise_invert(x)
    assert jnp.all(inv.array == jnp.bitwise_invert(x.array))

    cnt = hax.bitwise_count(x)
    assert jnp.all(cnt.array == jnp.bitwise_count(x.array))


def test_bitwise_shift():
    A = Axis("A", 4)
    x = hax.named(jnp.array([1, 2, 3, 4], dtype=jnp.int32), (A,))
    shift = hax.named(jnp.array([1, 1, 1, 1], dtype=jnp.int32), (A,))

    left = hax.bitwise_left_shift(x, shift)
    assert jnp.all(left.array == jnp.bitwise_left_shift(x.array, shift.array))

    right = hax.bitwise_right_shift(left, shift)
    assert jnp.all(right.array == jnp.bitwise_right_shift(left.array, shift.array))

    right_scalar = hax.bitwise_right_shift(x, 1)
    assert jnp.all(right_scalar.array == jnp.bitwise_right_shift(x.array, 1))


def test_packbits_unpackbits():
    B = Axis("B", 10)
    data = hax.named(jnp.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 0], dtype=jnp.uint8), (B,))

    packed = hax.packbits(data, B)
    expected_packed = jnp.packbits(data.array, axis=0)
    assert jnp.all(packed.array == expected_packed)
    assert packed.axes[0].name == B.name
    assert packed.axes[0].size == 2

    unpacked = hax.unpackbits(packed, "B", count=B.size)
    expected_unpacked = jnp.unpackbits(packed.array, axis=0, count=B.size)
    assert jnp.all(unpacked.array == expected_unpacked)
    assert unpacked.axes[0].size == B.size
