# Copyright 2025 The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0


import jax.numpy as jnp

import haliax as hax
from haliax import Axis, dslice

# -----------------------------------------------------------------------------
# Helper for reference via take_along_axis
# -----------------------------------------------------------------------------


def _ref_gather(src, axis, idx):
    ax_num = src.axes.index(axis)
    # broadcast idx to match src without the gathered axis
    other_axes = tuple(ax for ax in src.axes if ax != axis)
    broadcast_axes = other_axes
    for ax in idx.axes:
        if ax not in broadcast_axes:
            broadcast_axes += (ax,)
    idx_b = hax.broadcast_to(idx, broadcast_axes, enforce_no_extra_axes=False)
    if idx_b.array.ndim == src.array.ndim - 1:
        idx_arr = idx_b.array[..., None]
    else:
        idx_arr = idx_b.array
    out = jnp.take_along_axis(src.array, idx_arr, axis=ax_num)
    if idx_b.array.ndim == src.array.ndim - 1:
        out = out.squeeze(ax_num)
    return out


# ---------------------------- 1. single batched selector ----------------------


def test_single_batched_selector():
    B, S, V = Axis("batch", 4), Axis("seq", 3), Axis("vocab", 7)
    x = hax.arange((B, S, V))
    idx = hax.arange((B, S), dtype=jnp.int32) % V.size
    out = x["vocab", idx]
    assert out.axes == (B, S)
    assert jnp.array_equal(out.array, _ref_gather(x, V, idx))


# ---------------------------- 2. selector adds new axis -----------------------


def test_selector_adds_new_axis():
    B, S, V, T = Axis("batch", 2), Axis("seq", 3), Axis("vocab", 5), Axis("step", 4)
    logits = hax.arange((B, S, V))
    idx = hax.arange((B, T), dtype=jnp.int32) % V.size
    out = logits["vocab", idx]
    assert set(out.axes) == {B, S, T}
    ref = jnp.transpose(_ref_gather(logits, V, idx), (0, 2, 1))
    assert jnp.array_equal(out.array, ref)


# ------------------------ 3. two contiguous selector arrays -------------------


def test_two_contiguous_selectors():
    B, X, Y = Axis("batch", 3), Axis("x", 5), Axis("y", 7)
    a = hax.arange((B, X, Y))
    ix = hax.arange((B,), dtype=jnp.int32) % X.size
    iy = hax.arange((B,), dtype=jnp.int32) % Y.size
    out = a["x", ix, "y", iy]
    assert out.axes == (B,)
    ref = a.array[jnp.arange(3), ix.array, iy.array]
    assert jnp.array_equal(out.array, ref)


# ------------------ 4. non-contiguous selectors → axes to front --------------


def test_noncontig_selectors():
    B, X, Z, Y = Axis("batch", 2), Axis("x", 4), Axis("z", 6), Axis("y", 5)
    a = hax.arange((B, X, Z, Y))
    ix = hax.arange((B,), dtype=jnp.int32) % X.size
    iy = hax.arange((B,), dtype=jnp.int32) % Y.size
    out = a["x", ix, "y", iy]
    assert out.axes == (B, Z)
    ref = a.array[jnp.arange(2), ix.array, :, iy.array]
    assert jnp.array_equal(out.array, ref)


# ----------------- 5. integer elimination + selector --------------------------


def test_mixed_int_and_selector():
    B, C, V = Axis("batch", 3), Axis("channel", 2), Axis("vocab", 6)
    x = hax.arange((B, C, V))
    idx = hax.arange((B,), dtype=jnp.int32) % V.size
    out = x["channel", 1, "vocab", idx]
    assert out.axes == (B,)
    ref = x.array[:, 1, :][jnp.arange(3), idx.array]
    assert jnp.array_equal(out.array, ref)


def test_dslice_with_selector():
    B, S, V = Axis("batch", 2), Axis("seq", 5), Axis("vocab", 10)
    x = hax.arange((B, S, V))
    idx = (hax.arange((B, S), dtype=jnp.int32) + 2) % 4
    shard = V.resize(4)
    x_shard = x["vocab", dslice(0, shard)]
    out = x_shard["vocab", idx]
    assert out.axes == (B, S)
    ref = x.array[:, :, :4][jnp.arange(B.size)[:, None], jnp.arange(S.size)[None, :], idx.array]
    assert jnp.array_equal(out.array, ref)


def test_scalar_eliminates_axis():
    B, S, V = Axis("batch", 2), Axis("seq", 3), Axis("vocab", 4)
    x = hax.arange((B, S, V))
    out = x["seq", 1]
    assert out.axes == (B, V)
    assert jnp.array_equal(out.array, x.array[:, 1, :])


# ----------------- 9. plain ndarray selector sugar ----------------------------


def test_plain_ndarray_selector():
    B, V = Axis("batch", 3), Axis("vocab", 5)
    x = hax.arange((B, V))
    idx = jnp.array([0, 2, 4], dtype=jnp.int32)
    out = x["vocab", idx]
    assert out.axes == (B,)
    assert jnp.array_equal(out.array, x.array[jnp.arange(3), idx])


# ----------------- 10. two selectors needing broadcast ------------------------


def test_multiselector_broadcast():
    B, S, V = Axis("batch", 2), Axis("seq", 3), Axis("vocab", 6)
    a = hax.arange((B, S, V))
    idx1 = hax.arange((B, S), dtype=jnp.int32) % V.size
    out = a["vocab", idx1]
    assert out.axes == (B, S)
    assert jnp.array_equal(out.array, _ref_gather(a, V, idx1))


# ----------------- 11. scatter-ADD via .at[…].add -----------------------------


def test_scatter_add():
    B, S, V = Axis("batch", 2), Axis("seq", 3), Axis("vocab", 5)
    x = hax.zeros((B, S, V))
    idx = hax.arange((B, S), dtype=jnp.int32) % V.size
    ones = hax.ones((B, S))
    y = x.at[{V: idx}].add(ones)
    ref = jnp.zeros((2, 3, 5)).at[jnp.arange(2)[:, None], jnp.arange(3)[None, :], idx.array].add(1.0)
    assert jnp.array_equal(y.array, ref)


# ----------------- 12. scatter-SET via .at[…].set -----------------------------


def test_scatter_set():
    B, V = Axis("batch", 2), Axis("vocab", 6)
    x = hax.zeros((B, V))
    idx = hax.named(jnp.array([1, 4]), B)
    val = hax.ones(B) * 9
    y = x.at[{V: idx}].set(val)
    ref = jnp.zeros((2, 6)).at[jnp.arange(2), idx.array].set(9)
    assert jnp.array_equal(y.array, ref)
