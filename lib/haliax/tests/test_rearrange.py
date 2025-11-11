# Copyright 2025 The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0


import pytest
from jax.random import PRNGKey

import haliax as hax
from haliax import Axis
from haliax._src.rearrange import einops_rearrange

# some axes
H, W, C, D, B = hax.make_axes(H=6, W=4, C=3, D=2, B=5)
Q = hax.Axis("Q", B.size * H.size)
E = Axis("E", H.size * W.size * D.size)

z = hax.random.randint(PRNGKey(0), (B, D, H, W, C), 0, 255)
zq = hax.random.randint(PRNGKey(0), (Q, D, W, C), 0, 255)


def test_basic_identity():
    assert einops_rearrange(z, "b d h w c -> b d h w c").axes == (B, D, H, W, C)
    assert (einops_rearrange(z, "b d h w c -> b d h w c").array == z.array).all()


def test_basic_rearrange_transpose():
    assert einops_rearrange(z, "b d h w c -> b h w d c").axes == (B, H, W, D, C)
    # make sure the values are right too
    z_t = z.array.transpose((0, 2, 3, 1, 4))
    assert (einops_rearrange(z, "b d h w c -> b h w d c").array == z_t).all()


def test_basic_rearrange_unordered():
    assert einops_rearrange(z, "{B D H W C} -> B H W D C").axes == (B, H, W, D, C)
    z_t = z.array.transpose((0, 2, 3, 1, 4))
    assert (einops_rearrange(z, "{B D H W C} -> B H W D C").array == z_t).all()

    assert einops_rearrange(z, "{C W H D B} -> B H W D C").axes == (B, H, W, D, C)
    assert (einops_rearrange(z, "{C W H D B} -> B H W D C").array == z_t).all()


def test_rearrange_with_ellipsis():
    assert einops_rearrange(z, "... w c -> ... c w").axes == (B, D, H, C, W)
    assert einops_rearrange(z, "b d ... -> d ... b").axes == (D, H, W, C, B)

    assert einops_rearrange(z, "b ... c -> b c ...").axes == (B, C, D, H, W)
    # make sure the values are right too
    z_t = z.array.transpose((0, 4, 1, 2, 3))
    assert (einops_rearrange(z, "b ... c -> b c ...").array == z_t).all()


def test_rearrange_with_ellipsis_unordered():
    assert einops_rearrange(z, "{W C} -> ... C W").axes == (B, D, H, C, W)
    assert einops_rearrange(z, "{B D} -> D ... B").axes == (D, H, W, C, B)

    assert einops_rearrange(z, "{B C} -> B C ...").axes == (B, C, D, H, W)
    # make sure the values are right too
    z_t = z.array.transpose((0, 4, 1, 2, 3))
    assert (einops_rearrange(z, "{B C} -> B C ...").array == z_t).all()
    assert (einops_rearrange(z, "{qqq C} -> qqq C ...", qqq=B).array == z_t).all()


def test_rearrange_with_flattening():
    assert einops_rearrange(z, "b d h w c -> d (Q: b h) w c").axes == (D, Q, W, C)
    # make sure the values are right too
    z_t = z.array.transpose((1, 0, 2, 3, 4)).reshape((D.size, Q.size, W.size, C.size))
    assert (einops_rearrange(z, "b d h w c -> d (Q: b h) w c").array == z_t).all()

    # test with ellipsis
    assert einops_rearrange(z, "b d h ... c -> d (Q: b h) ... c").axes == (D, Q, W, C)
    # make sure the values are right too
    z_t = z.array.transpose((1, 0, 2, 3, 4)).reshape((D.size, Q.size, W.size, C.size))
    assert (einops_rearrange(z, "b d h ... c -> d (Q: b h) ... c").array == z_t).all()


def test_rearrange_with_flattening_unordered():
    assert einops_rearrange(z, "{B D H} -> D (Q: B H) ...").axes == (D, Q, W, C)
    # make sure the values are right too
    z_t = z.array.transpose((1, 0, 2, 3, 4)).reshape((D.size, Q.size, W.size, C.size))
    assert (einops_rearrange(z, "{B D H} -> D (Q: B H) ...").array == z_t).all()

    # test with ellipsis
    assert einops_rearrange(z, "{B D H ... C} -> D (Q: B H) ... C").axes == (D, Q, W, C)
    # make sure the values are right too
    z_t = z.array.transpose((1, 0, 2, 3, 4)).reshape((D.size, Q.size, W.size, C.size))
    assert (einops_rearrange(z, "{B D H ... C} -> D (Q: B H) ... C").array == z_t).all()


def test_rearrange_with_unflatten():
    assert einops_rearrange(zq, "(Q: B H) d w c -> B d H w c", B=5).axes == (B, D, H, W, C)
    # make sure the values are right too
    z_t = zq.array.reshape((B.size, H.size, D.size, W.size, C.size)).transpose((0, 2, 1, 3, 4))
    assert (einops_rearrange(zq, "(Q: B H) d w c -> B d H w c", B=5).array == z_t).all()

    # test with explicit axis arg
    assert einops_rearrange(zq, "(Q: b H) d w c -> b d H w c", b=B).axes == (B, D, H, W, C)
    # make sure the values are right too
    z_t = zq.array.reshape((B.size, H.size, D.size, W.size, C.size)).transpose((0, 2, 1, 3, 4))
    assert (einops_rearrange(zq, "(Q: b H) d w c -> b d H w c", b=B).array == z_t).all()


def test_rearrange_with_unflatten_unordered():
    assert einops_rearrange(zq, "{ (Q: B H) D } -> B D H ... ", B=5).axes == (B, D, H, W, C)
    # make sure the values are right too
    z_t = zq.array.reshape((B.size, H.size, D.size, W.size, C.size)).transpose((0, 2, 1, 3, 4))
    assert (einops_rearrange(zq, "{ (Q: B H) D } -> B D H ... ", B=5).array == z_t).all()

    # test with explicit axis arg
    assert einops_rearrange(zq, "{ (Q: b H) D } -> b D H ... ", b=B).axes == (B, D, H, W, C)
    # make sure the values are right too
    z_t = zq.array.reshape((B.size, H.size, D.size, W.size, C.size)).transpose((0, 2, 1, 3, 4))
    assert (einops_rearrange(zq, "{ (Q: b H) D } -> b D H ... ", b=B).array == z_t).all()


def test_with_unflatten_flatten():
    Z = Axis("Z", B.size * C.size)
    assert einops_rearrange(zq, "(Q: B H) d w c -> d (Z: B c) w H", H=H).axes == (D, Z, W, H)
    # make sure the values are right too
    z_t = (
        zq.array.reshape((B.size, H.size, D.size, W.size, C.size))
        .transpose((2, 0, 4, 3, 1))
        .reshape((D.size, Z.size, W.size, H.size))
    )
    assert (einops_rearrange(zq, "(Q: B H) d w c -> d (Z: B c) w H", H=H).array == z_t).all()


def test_with_unflatten_flatten_unordered():
    Z = Axis("Z", B.size * C.size)
    assert einops_rearrange(zq, "{ W D C (Q: B H)} -> D (Z: B C) W H", H=H).axes == (D, Z, W, H)
    # make sure the values are right too
    z_t = (
        zq.array.reshape((B.size, H.size, D.size, W.size, C.size))
        .transpose((2, 0, 4, 3, 1))
        .reshape((D.size, Z.size, W.size, H.size))
    )
    assert (einops_rearrange(zq, "{ W D C (Q: B H)} -> D (Z: B C) W H", H=H).array == z_t).all()


def test_rearrange_multiple_ellipses():
    z_out = einops_rearrange(z, "b d h w c  -> d ... c ... h")
    assert z_out.axes == (D, B, C, W, H)

    z_out = einops_rearrange(z, "b d h w c -> d ... (Q: b h) ... w")
    assert z_out.axes == (D, Q, C, W)

    z_out = einops_rearrange(z, "b d h w c -> d ... (BB: b) ... w")
    assert z_out.axes == (D, B.alias("BB"), H, C, W)

    z_out = einops_rearrange(z, "{H W D} -> ... (E: H W D) ...")
    assert z_out.axes == (B, E, C)

    z_out = einops_rearrange(z, "{B H} -> ... (Q: B H) ...")
    assert z_out.axes == (Q, D, W, C)

    z_out = einops_rearrange(z, "{B H W} -> ... (Q: B H) ... W")

    assert z_out.axes == (Q, D, C, W)


def test_semantic_errors():
    with pytest.raises(ValueError, match="Too many axes in lhs"):
        einops_rearrange(z, "b d h w c q -> b d h w c q")

    with pytest.raises(ValueError, match="Not all axes are bound"):
        einops_rearrange(z, "b d h w -> b d h w c q")

    with pytest.raises(ValueError, match="Too many axes in lhs"):
        einops_rearrange(z, "b d h w x y z -> b d h w c q")

    with pytest.raises(ValueError, match="Axis q is not bound on the lhs"):
        einops_rearrange(z, "b d h w c -> b d h w c q")

    with pytest.raises(ValueError, match="Only one ellipsis allowed"):
        einops_rearrange(z, "b d ... h w c ... -> b d h w c ...")

    with pytest.raises(ValueError, match="Capture q is assigned more than once"):
        einops_rearrange(z, "b d c q q -> b d h w c q")

    with pytest.raises(ValueError, match="is used more than once"):
        einops_rearrange(z, "b d c q t -> b d c t q q")

    with pytest.raises(ValueError, match="Capture q is assigned more than once"):
        einops_rearrange(z, "b d c (q q) z -> b d h w c q", q=4)

    with pytest.raises(ValueError, match="Not all intermediate axes are used"):
        einops_rearrange(z, "b d c q z -> b d c z")

    with pytest.raises(ValueError, match="is bound more than once"):
        einops_rearrange(z, "b d c q z ... d e f -> b d c q d e f")

    with pytest.raises(ValueError, match="Axis q is not bound on the lhs"):
        einops_rearrange(z, "b d h w c -> b d h w c q")

    with pytest.raises(ValueError, match="Only one ellipsis allowed"):
        einops_rearrange(z, "b d ... h w c ... -> b d h w c ...")

    with pytest.raises(ValueError, match="Capture q is assigned more than once"):
        einops_rearrange(z, "b d c q q -> b d h w c q")

    with pytest.raises(ValueError, match="Capture q is assigned more than once"):
        einops_rearrange(z, "b d c (q q) z -> b d h w c q", q=4)

    with pytest.raises(ValueError, match="Capture q is assigned more than once"):
        einops_rearrange(z, "b d c q q -> b d h w c q")

    with pytest.raises(ValueError, match="Not all intermediate axes are used"):
        einops_rearrange(z, "b d c q z -> b d c z")

    with pytest.raises(ValueError, match="is bound more than once"):
        einops_rearrange(z, "b d c q z ... d e f -> b d c q d e f")

    with pytest.raises(ValueError, match="not divide"):
        einops_rearrange(z, "(x y) d c q z -> d c q x y z", x=13, y=17)

    with pytest.raises(ValueError, match="not divide"):
        einops_rearrange(z, "(x y) d c q z -> d c q x y z", x=13)

    with pytest.raises(ValueError, match="must have a name"):
        einops_rearrange(z, "b d ... h w c  -> (b d h w c) ...")

    with pytest.raises(ValueError, match="not bound on the lhs"):
        einops_rearrange(z, "b d ... h w c  -> (z: w r) ...")

    with pytest.raises(ValueError, match="are ambiguous"):
        einops_rearrange(z, "(b q) d ... h w c  -> (z: w r) ...")

    with pytest.raises(ValueError, match="more than once"):
        einops_rearrange(z, "{B B} -> Z Z")

    with pytest.raises(ValueError, match="must be bound by name"):
        einops_rearrange(z, "{(q) } -> (Z: Z Z)")

    with pytest.raises(ValueError, match="Could not resolve"):
        einops_rearrange(z, "{(Z: q) } -> (Z: Z Z)")


def test_examples():
    # Cases to support:
    # identity
    # * 'a b c d -> a b c d' or 'a, b, c, d -> a, b, c, d'
    assert einops_rearrange(z, "a b c d e -> a b c d e").axes == (B, D, H, W, C)
    # merge a and b into m
    # * 'a b c d -> (m: a b) c d'
    assert einops_rearrange(z, "a b c d e -> (M: a b) c d e").axes == (Axis("M", B.size * D.size), H, W, C)
    #  > without rearrange or names: x.reshape((a * b, c, d))
    # split a into b and c
    # * '(a: b c) d -> b c d'
    assert einops_rearrange(zq, "(q: b h) d w c -> b h d w c", b=B, h="H").axes == (B, H, D, W, C)
    #  > without rearrange or names: x.reshape((b, c, d))
    # reorder a and b
    # * 'a b ... -> b a ...'
    assert einops_rearrange(z, "a b ... -> b a ...").axes == (D, B, H, W, C)
    assert einops_rearrange(z, "a b ... -> b ... a").axes == (D, H, W, C, B)
    #  > without rearrange or names: x.transpose((1, 0, ...))
    # split into 2 groups, rename to x and y
    # * 'a b c d -> (x: a b) (y: c d)'  # split into two groups, rename to x and y
    assert einops_rearrange(z, "a b c d ... -> (x: a b) (y: c d) ...").axes == (
        Axis("x", B.size * D.size),
        Axis("y", H.size * W.size),
        C,
    )
    #  > without rearrange or names: x.reshape((a * b, c * d))
    # unordered match of a, b, c by name, move to end
    # * `{c b a} -> ... a b c`
    assert einops_rearrange(z, "{B C W} -> ... B C W").axes == (D, H, B, C, W)
    #   > without rearrange or names: x.transpose((2, 1, 0, ...))
    # unordered match of a and d by name, split a into b and c, reorder
    # * `{(a: b c) d} -> ... b d c`
    assert einops_rearrange(zq, "{(Q: b h) C} -> ... b C h", b=B, h=H).axes == (D, W, B, C, H)
    #   > without rearrange or names: x.reshape(..., b, c, d).transpose((..., -3, -1, -2))
    # unordered match of a and d by name, split a into b and c, d into e and f, reorder
    # * `{(a: b c) (d: e f)} -> ... b e c f`
    assert einops_rearrange(zq, "{(Q: b h) (W: e f)} -> ... b e h f", b=B, h=H, e=2).axes == (
        D,
        C,
        B,
        Axis("e", 2),
        H,
        Axis("f", 2),
    )
    # flatten each image into a vector
    # * `{h w c} -> (c: c h w)`
    assert einops_rearrange(z, "{H W C} -> ... (C: C H W)").axes == (B, D, Axis("C", C.size * H.size * W.size))
    # split each image into 4 smaller (top-left, top-right, bottom-left, bottom-right), 128 = 32 * 2 * 2:
    # * `{b (h: h1 h) (w: w1 w)} -> (b: b h1 w1) h w ...`
    sH = Axis("H", H.size // 2)
    sW = Axis("W", W.size // 2)
    # this works right now, would be good to support the below
    r = einops_rearrange(z, "{B (H: h1 h) (W: w1 w)} -> (B: B h1 w1) (H: h) (W: w) ...", h1=2, w1=2)
    assert r.axes == (Axis("B", B.size * 2 * 2), sH, sW, D, C)

    r = einops_rearrange(z, "{B (H: h1 H) (W: w1 W)} -> (B: B h1 w1) H W ...", h1=2, w1=2)
    assert r.axes == (Axis("B", B.size * 2 * 2), sH, sW, D, C)
    # sequence of flattened patches:
    # * `{b (h: h1 h) (w: w1 w) c} -> (b: b h1 w1) (c: c h w) ...`
    r = einops_rearrange(z, "{B (H: h1 h) (W: w1 w) C} -> (B: B h1 w1) ... (C: C h w) ", h1=2, w1=2)
    assert r.axes == (Axis("B", B.size * 2 * 2), D, Axis("C", C.size * sH.size * sW.size))
    # unet attention reordering:
    # positional: (qkv heads c) h w -> qkv heads c (h w)
    # named: { (embed: qkv heads c) h w } -> qkv heads c (pos: h w)
    Embed = Axis("embed", 3 * 4 * C.size)
    attn = hax.random.randint(PRNGKey(0), (Embed, H, W), 0, 255)
    r = einops_rearrange(attn, "{(embed: qkv heads C) H W} -> qkv heads C (pos: H W)", qkv=3, heads=4)
    assert r.axes == (Axis("qkv", 3), Axis("heads", 4), C, (Axis("pos", H.size * W.size)))
    # space to depth
    # * {b (h: h1 h) (w: w1 w) c} -> ... b h w (c: c h1 w1)
    r = einops_rearrange(z, "{B (H: h1 h) (W: w1 w) C} -> ... B (H: h) (W: w) (C: C h1 w1)", h1=2, w1=2)
    assert r.axes == (D, B, sH, sW, Axis("C", C.size * 2 * 2))

    r = einops_rearrange(z, "{B (H: h1 H) (W: w1 W) C} -> B ... H W (C: C h1 w1)", h1=2, w1=2)
    assert r.axes == (B, D, sH, sW, Axis("C", C.size * 2 * 2))

    # image patches specifying H and W
    q = hax.rearrange(z, "{ B (H: h1 H) (W: w1 W)} -> (B: B h1 w1) H W ...", H=2, W=2)
    assert q.axes == (Axis("B", B.size * sH.size * sW.size), H.resize(2), W.resize(2), D, C)

    q = hax.rearrange(z, "{ B (H: h1 H) (W: w1 W) D} -> (B: B h1 w1) (E: H W D)...", H=2, W=2)
    assert q.axes == (Axis("B", B.size * sH.size * sW.size), Axis("E", 2 * 2 * D.size), C)
