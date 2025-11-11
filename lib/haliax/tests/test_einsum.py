# Copyright 2025 The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0


import jax.numpy as jnp
import pytest

import haliax as hax
from haliax import Axis, NamedArray
from haliax._src.einsum import einsum


def test_einsum_basic_positional():
    Height = Axis("Height", 2)
    Width = Axis("Width", 3)
    Depth = Axis("Depth", 4)

    m1 = NamedArray(jnp.ones((Height.size, Width.size, Depth.size)), (Height, Width, Depth))
    m2 = NamedArray(jnp.ones((Depth.size, Width.size, Height.size)), (Depth, Width, Height))

    assert jnp.all(jnp.equal(einsum("i j k,k j i-> j k", m1, m2).array, jnp.einsum("ijk,kji->jk", m1.array, m2.array)))
    assert jnp.all(
        jnp.equal(
            einsum("i j k,k j i->k", m1, m2).array,
            jnp.einsum("ijk,kji->k", m1.array, m2.array),
        )
    )
    assert jnp.all(
        jnp.equal(
            einsum("i j k,k j i->", m1, m2).array,
            jnp.einsum("ijk,kji->", m1.array, m2.array),
        )
    )

    # reduce to scalar
    assert jnp.all(
        jnp.equal(
            einsum("i j k,k j i->", m1, m2).array,
            jnp.einsum("ijk,kji->", m1.array, m2.array),
        )
    )


def test_einsum_positional_aliases():
    Height = Axis("Height", 2)
    Width = Axis("Width", 3)
    Depth = Axis("Depth", 4)

    m1 = NamedArray(jnp.ones((Height.size, Width.size, Depth.size)), (Height, Width, Depth))
    m2 = NamedArray(jnp.ones((Depth.size, Width.size, Height.size)), (Depth, Width, Height))

    assert jnp.all(
        jnp.equal(einsum("i j k,k j i-> j k", m1, m2, i=Height).array, jnp.einsum("ijk,kji->jk", m1.array, m2.array))
    )

    with pytest.raises(ValueError):
        einsum("i j k,k j i-> j k", m1, m2, i=Width)

    with pytest.raises(ValueError):
        einsum("i j k,q j i-> j k", m1, m2, i=Height, q=Height)

    with pytest.raises(ValueError):
        einsum("i j k,k j i-> j k", m1, m2, i=Height, q=Height)


def test_einsum_basic_named():
    Height = Axis("Height", 2)
    Width = Axis("Width", 3)
    Depth = Axis("Depth", 4)

    m1 = hax.ones((Height, Width, Depth))
    m2 = hax.ones((Depth, Width, Height))

    assert jnp.all(
        jnp.equal(
            einsum("{Height Width Depth} -> Height Width", m1, m2).array, jnp.einsum("ijk,kji->ij", m1.array, m2.array)
        )
    )

    assert jnp.all(jnp.equal(einsum(" -> Height Width", m1, m2).array, jnp.einsum("ijk,kji->ij", m1.array, m2.array)))
    assert jnp.all(
        jnp.equal(
            einsum("{Height Width Depth} -> Depth", m1, m2).array,
            jnp.einsum("ijk,kji->k", m1.array, m2.array),
        )
    )
    assert jnp.all(
        jnp.equal(
            einsum("-> Depth", m1, m2).array,
            jnp.einsum("ijk,kji->k", m1.array, m2.array),
        )
    )

    assert jnp.all(
        jnp.equal(
            einsum("{Height Width Depth} -> ", m1, m2).array,
            jnp.einsum("ijk,kji->", m1.array, m2.array),
        )
    )

    # outer product
    assert jnp.all(
        jnp.equal(
            einsum("{} ->", m1, m2).array,
            jnp.einsum("ijk,kji->ijk", m1.array, m2.array),
        )
    )

    assert jnp.all(
        jnp.equal(
            einsum("{Height Depth} -> ", m1, m2).array,
            jnp.einsum("ijk,kji->j", m1.array, m2.array),
        )
    )

    assert jnp.all(
        jnp.equal(
            einsum("{Height Depth} -> ", m1, m2).array,
            jnp.einsum("ijk,kji->j", m1.array, m2.array),
        )
    )


def test_einsum_unordered_ellipses():
    Height = Axis("Height", 2)
    Width = Axis("Width", 3)
    Depth = Axis("Depth", 4)

    m1 = hax.ones((Height, Width, Depth))
    m2 = hax.ones((Depth, Width, Height))

    # collapse all
    assert jnp.all(
        jnp.equal(
            einsum("{...} ->", m1, m2).array,
            jnp.einsum("ijk,kji->", m1.array, m2.array),
        )
    )

    # keep all
    assert jnp.all(
        jnp.equal(
            einsum("{...} -> ...", m1, m2).array,
            jnp.einsum("ijk,kji->ijk", m1.array, m2.array),
        )
    )

    assert jnp.all(
        jnp.equal(
            einsum("{} -> ...", m1, m2).array,
            jnp.einsum("ijk,kji->ijk", m1.array, m2.array),
        )
    )

    assert jnp.all(
        jnp.equal(
            einsum("{Depth} -> Depth ...", m1, m2).array,
            jnp.einsum("ijk,kji->kij", m1.array, m2.array),
        )
    )

    assert jnp.all(
        jnp.equal(
            einsum("{Depth Height} -> Depth ... Height", m1, m2).array,
            jnp.einsum("ijk,kji->kji", m1.array, m2.array),
        )
    )

    assert jnp.all(
        jnp.equal(
            einsum("{Depth} -> ... Depth ...", m1, m2).array,
            jnp.einsum("ijk,kji->ijk", m1.array, m2.array),
        )
    )


def test_einsum_unordered_aliases():
    Height = Axis("Height", 2)
    Width = Axis("Width", 3)
    Depth = Axis("Depth", 4)

    m1 = hax.ones((Height, Width, Depth))
    m2 = hax.ones((Depth, Width, Height))

    assert jnp.all(
        jnp.equal(
            einsum("{h w d} -> h w", m1, m2, h=Height, w=Width, d=Depth).array,
            jnp.einsum("ijk,kji->ij", m1.array, m2.array),
        )
    )

    # test error cases:

    # Missing alias
    with pytest.raises(ValueError, match="Axis d not present"):
        einsum("{h w d} -> h w", m1, m2, h=Height, w=Width)

    # Extra alias
    with pytest.raises(ValueError, match="Axis alias d not used"):
        einsum("{h w} -> h w", m1, m2, h=Height, w=Width, d=Depth)


def test_einsum_ordered_ellipsis():
    Height = Axis("Height", 2)
    Width = Axis("Width", 3)
    Depth = Axis("Depth", 4)

    m1 = hax.ones((Height, Width, Depth))
    m2 = hax.ones((Depth, Width, Height))

    assert jnp.all(
        jnp.equal(
            einsum("h ... d,d ... h-> ...", m1, m2).array,
            jnp.einsum("ijk,kji->j", m1.array, m2.array),
        )
    )

    assert jnp.all(
        jnp.equal(
            einsum("h ... d,d ... h-> ... d", m1, m2).array,
            jnp.einsum("ijk,kji->jk", m1.array, m2.array),
        )
    )


def test_einsum_works_with_same_initial_axis_letter():
    Height = Axis("Height", 2)
    Hidth = Axis("Hidth", 3)
    Depth = Axis("Depth", 4)

    m1 = hax.ones((Height, Hidth, Depth))
    m2 = hax.ones((Depth, Hidth, Height))

    assert jnp.all(
        jnp.equal(
            einsum("h ... d,d ... h-> ...", m1, m2).array,
            jnp.einsum("ijk,kji->j", m1.array, m2.array),
        )
    )

    assert jnp.all(
        jnp.equal(
            einsum("{Height Hidth Depth} -> Hidth", m1, m2).array,
            jnp.einsum("ijk,kji->j", m1.array, m2.array),
        )
    )


def test_einsum_various_errors():
    Height = Axis("Height", 2)
    Hidth = Axis("Hidth", 3)
    Depth = Axis("Depth", 4)

    m1 = hax.ones((Height, Hidth, Depth))
    m2 = hax.ones((Depth, Hidth, Height))

    with pytest.raises(ValueError, match="multiple times"):
        einsum("-> Height Height", m1, m2)

    with pytest.raises(ValueError, match="does not match number of arrays"):
        einsum("x y z, x y z -> x", m1, m2, m1)

    with pytest.raises(ValueError, match="does not match number of arrays"):
        einsum("x y z -> x", m1, m2)

    with pytest.raises(ValueError, match="Mismatched number of axes"):
        einsum("x y z a b -> x", m1)

    with pytest.raises(ValueError, match="Mismatched number of axes"):
        einsum("x y ... a b -> x", m1)

    with pytest.raises(ValueError, match="Mismatched number of axes"):
        einsum("x y -> x", m1)


def test_einsum_examples():
    Batch, Embed, H, W, C = hax.make_axes(batch=32, embed=64, h=16, w=16, c=3)

    # for jax
    im = jnp.zeros((32, 16, 16, 3))
    w2 = jnp.zeros((3, 64))

    # for haliax
    hax_im = hax.zeros((Batch, H, W, C))
    hax_w2 = hax.zeros((C, Embed))

    # Tests:
    # | [`jnp.einsum("bhwc,ce -> bhwe", im, w2)`][jax.numpy.einsum]    | [`hax.einsum("b h w c, c e -> b h w e", im, w2)`][haliax.einsum]   |
    # | [`jnp.einsum("...c,ce -> ...e", im, w2)`][jax.numpy.einsum]    | [`hax.einsum("... c, c e -> ... e", im, w2)`][haliax.einsum]       |
    # | [`jnp.einsum("bhwc,ce -> bhw", im, w2)`][jax.numpy.einsum]     | [`hax.einsum("{c embed} -> embed", im, w2)`][haliax.einsum]        |
    # | [`jnp.einsum("bhwc,ce -> bhw", im, w2)`][jax.numpy.einsum]     | [`hax.einsum("-> batch h w embed", im, w2)`][haliax.einsum]        |

    hax_out = hax.einsum("b h w c, c e -> b h w e", hax_im, hax_w2)
    jnp_out = jnp.einsum("bhwc,ce -> bhwe", im, w2)
    assert jnp.all(jnp.equal(hax_out.array, jnp_out))

    hax_out = hax.einsum("... c, c e -> ... e", hax_im, hax_w2)
    jnp_out = jnp.einsum("...c,ce -> ...e", im, w2)
    assert jnp.all(jnp.equal(hax_out.array, jnp_out))

    hax_out = hax.einsum("{c embed} -> embed", hax_im, hax_w2)
    jnp_out = jnp.einsum("bhwc,ce -> bhwe", im, w2)
    assert jnp.all(jnp.equal(hax_out.array, jnp_out))

    hax_out = hax.einsum("-> batch h w embed", hax_im, hax_w2)
    jnp_out = jnp.einsum("bhwc,ce -> bhwe", im, w2)
    assert jnp.all(jnp.equal(hax_out.array, jnp_out))

    # | [`jnp.einsum("bhwc,ce -> bhwce", im, w2)`][jax.numpy.einsum]   | [`hax.einsum("{...} -> ...", im, w2)`][haliax.einsum]              |
    # | [`jnp.einsum("bhwc,ce -> ", im, w2)`][jax.numpy.einsum]        | [`hax.einsum("{...} -> ", im, w2)`][haliax.einsum]                 |

    hax_out = hax.einsum("{...} -> ...", hax_im, hax_w2)
    jnp_out = jnp.einsum("bhwc,ce -> bhwce", im, w2)
    assert jnp.all(jnp.equal(hax_out.array, jnp_out))

    hax_out = hax.einsum("{...} -> ", hax_im, hax_w2)
    jnp_out = jnp.einsum("bhwc,ce -> ", im, w2)
    assert jnp.all(jnp.equal(hax_out.array, jnp_out))


def test_einsum_output_only_mode():
    # tests "-> out axes"
    Height = Axis("Height", 2)
    Width = Axis("Width", 3)
    Depth = Axis("Depth", 4)

    m1 = hax.ones((Height, Width, Depth))
    m2 = hax.ones((Depth, Width, Height))
    m3 = hax.ones((Height, Depth))

    assert jnp.all(jnp.equal(einsum("-> Height Width", m1, m2).array, jnp.einsum("ijk,kji->ij", m1.array, m2.array)))
    assert jnp.all(jnp.equal(einsum("-> Height", m1).array, jnp.einsum("ijk->i", m1.array)))

    assert jnp.all(jnp.equal(einsum("-> ...", m1, m2).array, jnp.einsum("ijk,kji->ijk", m1.array, m2.array)))
    assert jnp.all(jnp.equal(einsum("-> ... Width", m1, m2).array, jnp.einsum("ijk,kji->ikj", m1.array, m2.array)))
    assert jnp.all(
        jnp.equal(einsum("-> Depth ... Width", m1, m2).array, jnp.einsum("ijk,kji->kij", m1.array, m2.array))
    )

    with pytest.raises(ValueError):
        einsum("-> Q Width", m1)

    with pytest.raises(ValueError, match=".*Unused aliases from kwargs: Q$"):
        einsum("-> Height Width", m1, m2, Q=Axis("Q", 2))

    assert jnp.all(jnp.equal(einsum("-> h w", m1, h=Height, w=Width).array, jnp.einsum("ijk->ij", m1.array)))

    assert jnp.all(
        jnp.equal(einsum("-> h w", m1, m3, h=Height, w=Width).array, jnp.einsum("ijk,ik->ij", m1.array, m3.array))
    )

    with pytest.raises(ValueError, match=".*Size mismatch.*"):
        einsum("-> h w", m1, h=Height.resize(4), w=Width)

    with pytest.raises(ValueError, match=".*not found in any of the input arrays.*"):
        einsum("-> h w", m3, h=Height, w=Width.resize(4))
