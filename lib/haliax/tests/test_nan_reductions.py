# Copyright 2025 The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Callable

import jax.numpy as jnp
import haliax as hax


def _sample_array():
    Height, Width = hax.make_axes(Height=2, Width=3)
    data = jnp.array([[1.0, jnp.nan, 3.0], [jnp.nan, 5.0, 6.0]])
    arr = hax.named(data, (Height, Width))
    return Height, Width, data, arr


def test_amin_alias():
    Height, Width = hax.make_axes(Height=2, Width=3)
    data = jnp.arange(6.0).reshape(2, 3)
    arr = hax.named(data, (Height, Width))
    assert jnp.array_equal(hax.amin(arr).array, jnp.amin(data))
    assert jnp.array_equal(hax.amin(arr, axis=Height).array, jnp.amin(data, axis=0))
    assert hax.amin(arr, axis=Height).axes == (Width,)


def test_nan_reductions():
    Height, Width, data, arr = _sample_array()

    funcs: list[tuple[Callable[..., Any], Callable[..., Any]]] = [
        (hax.nanmin, jnp.nanmin),
        (hax.nanmax, jnp.nanmax),
        (hax.nanmean, jnp.nanmean),
        (hax.nansum, jnp.nansum),
        (hax.nanprod, jnp.nanprod),
        (hax.nanstd, jnp.nanstd),
        (hax.nanvar, jnp.nanvar),
    ]

    for hfunc, jfunc in funcs:
        assert jnp.allclose(hfunc(arr).array, jfunc(data), equal_nan=True)
        assert jnp.allclose(hfunc(arr, axis=Height).array, jfunc(data, axis=0), equal_nan=True)
        assert hfunc(arr, axis=Height).axes == (Width,)

    arg_funcs: list[tuple[Callable[..., Any], Callable[..., Any]]] = [
        (hax.nanargmax, jnp.nanargmax),
        (hax.nanargmin, jnp.nanargmin),
    ]

    for hfunc, jfunc in arg_funcs:
        out = hfunc(arr, axis=Height)
        assert jnp.array_equal(out.array, jfunc(data, axis=0))
        assert out.axes == (Width,)

    axiswise_funcs: list[tuple[Callable[..., Any], Callable[..., Any]]] = [
        (hax.nancumsum, jnp.nancumsum),
        (hax.nancumprod, jnp.nancumprod),
    ]

    for hfunc, jfunc in axiswise_funcs:
        out = hfunc(arr, axis=Height)
        assert jnp.allclose(out.array, jfunc(data, axis=0), equal_nan=True)
        assert out.axes == (Height, Width)
