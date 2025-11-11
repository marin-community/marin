# Copyright 2025 The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0


# these test if the rearrange logic works for partial orders
import pytest
from jax import numpy as jnp

import haliax as hax
from haliax import Axis


def test_dot():
    Height = Axis("Height", 2)
    Width = Axis("Width", 3)
    Depth = Axis("Depth", 4)

    m1 = hax.ones((Height, Width, Depth))
    m2 = hax.ones((Depth, Width, Height))

    assert jnp.all(jnp.equal(hax.dot(m1, m2, axis=Height).array, jnp.einsum("ijk,kji->jk", m1.array, m2.array)))
    assert jnp.all(
        jnp.equal(
            hax.dot(m1, m2, axis=(Height, Width)).array,
            jnp.einsum("ijk,kji->k", m1.array, m2.array),
        )
    )
    assert jnp.all(
        jnp.equal(
            hax.dot(m1, m2, axis=(Height, Width, Depth)).array,
            jnp.einsum("ijk,kji->", m1.array, m2.array),
        )
    )

    # reduce to scalar
    assert jnp.all(
        jnp.equal(
            hax.dot(m1, m2, axis=None).array,
            jnp.einsum("ijk,kji->", m1.array, m2.array),
        )
    )


def test_dot_string_selection():
    Height = Axis("Height", 2)
    Width = Axis("Width", 3)
    Depth = Axis("Depth", 4)

    m1 = hax.ones((Height, Width, Depth))
    m2 = hax.ones((Depth, Width, Height))

    assert jnp.all(jnp.equal(hax.dot(m1, m2, axis="Height").array, jnp.einsum("ijk,kji->jk", m1.array, m2.array)))
    assert jnp.all(
        jnp.equal(
            hax.dot(m1, m2, axis=("Height", "Width")).array,
            jnp.einsum("ijk,kji->k", m1.array, m2.array),
        )
    )
    assert jnp.all(
        jnp.equal(
            hax.dot(m1, m2, axis=("Height", "Width", "Depth")).array,
            jnp.einsum("ijk,kji->", m1.array, m2.array),
        )
    )


def test_dot_errors_if_different_sized_axes():
    Height = Axis("Height", 2)
    Width = Axis("Width", 3)
    Depth = Axis("Depth", 4)

    H2 = Axis("Height", 4)

    m1 = hax.ones((Height, Width, Depth))
    m2 = hax.ones((Depth, Width, H2))

    with pytest.raises(ValueError):
        hax.dot(m1, m2, axis="Height")


def test_dot_with_output_axes():
    Height = Axis("Height", 2)
    Width = Axis("Width", 3)
    Depth = Axis("Depth", 4)

    m1 = hax.ones((Height, Width, Depth))
    m2 = hax.ones((Depth, Width, Height))

    assert jnp.all(
        jnp.equal(
            hax.dot(m1, m2, axis=Height, out_axes=(Width, ...)).array,
            jnp.einsum("ijk,kji->jk", m1.array, m2.array),
        )
    )

    assert jnp.all(
        jnp.equal(
            hax.dot(m1, m2, axis=Height, out_axes=(Depth, ...)).array,
            jnp.einsum("ijk,kji->kj", m1.array, m2.array),
        )
    )

    assert jnp.all(
        jnp.equal(
            hax.dot(m1, m2, axis=Height, out_axes=(Depth, Width)).array,
            jnp.einsum("ijk,kji->kj", m1.array, m2.array),
        )
    )

    assert jnp.all(
        jnp.equal(
            hax.dot(m1, m2, axis=(), out_axes=(Depth, Height, ...)).array,
            jnp.einsum("ijk,kji->kij", m1.array, m2.array),
        )
    )

    assert jnp.all(
        jnp.equal(
            hax.dot(m1, m2, axis=(), out_axes=(..., Depth, Height)).array,
            jnp.einsum("ijk,kji->jki", m1.array, m2.array),
        )
    )

    assert jnp.all(
        jnp.equal(
            hax.dot(m1, m2, axis=(), out_axes=(..., Depth, ...)).array,
            jnp.einsum("ijk,kji->ijk", m1.array, m2.array),
        )
    )

    assert jnp.all(
        jnp.equal(
            hax.dot(m1, m2, axis=(), out_axes=(..., Depth, Height, ...)).array,
            jnp.einsum("ijk,kji->kij", m1.array, m2.array),
        )
    )
