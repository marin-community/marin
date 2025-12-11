# Copyright 2025 The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Polynomial helpers for :mod:`haliax`.

This module provides NamedArray-aware wrappers around :mod:`jax.numpy`'s
polynomial utilities.
"""

from __future__ import annotations

from typing import Literal, overload

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .axis import Axis, AxisSelector, axis_name
from .core import NamedArray, NamedOrNumeric
from .wrap import unwrap_namedarrays

DEFAULT_POLY_AXIS_NAME = "degree"
"""Default name used for polynomial coefficient axes when none is provided."""


def _poly_axis_from_input(p: NamedArray | ArrayLike, size: int) -> Axis:
    if isinstance(p, NamedArray):
        if p.ndim != 1:
            raise ValueError("Polynomial coefficient arrays must be 1D")
        return p.axes[0].resize(size)
    else:
        return Axis(DEFAULT_POLY_AXIS_NAME, size)


def poly(seq_of_zeros: NamedArray | ArrayLike) -> NamedArray:
    """Named version of [jax.numpy.poly][].

    If ``seq_of_zeros`` is not a [haliax.NamedArray][], the returned coefficient axis
    is named ``degree``.
    """

    (roots,) = unwrap_namedarrays(seq_of_zeros)
    result = jnp.poly(roots)
    axis = _poly_axis_from_input(seq_of_zeros, result.shape[0])
    return NamedArray(result, (axis,))


def polyadd(p1: NamedArray | ArrayLike, p2: NamedArray | ArrayLike) -> NamedArray:
    """Named version of [jax.numpy.polyadd][].

    If neither input is a [haliax.NamedArray][], the coefficient axis is named
    ``degree``; otherwise the axis from the NamedArray input is reused (resized as
    needed).
    """

    a1, a2 = unwrap_namedarrays(p1, p2)
    result = jnp.polyadd(a1, a2)
    axis = _poly_axis_from_input(p1 if isinstance(p1, NamedArray) else p2, result.shape[0])
    return NamedArray(result, (axis,))


def polysub(p1: NamedArray | ArrayLike, p2: NamedArray | ArrayLike) -> NamedArray:
    """Named version of [jax.numpy.polysub][].

    If neither input is a [haliax.NamedArray][], the coefficient axis is named
    ``degree``; otherwise the axis from the NamedArray input is reused (resized as
    needed).
    """

    a1, a2 = unwrap_namedarrays(p1, p2)
    result = jnp.polysub(a1, a2)
    axis = _poly_axis_from_input(p1 if isinstance(p1, NamedArray) else p2, result.shape[0])
    return NamedArray(result, (axis,))


def polymul(p1: NamedArray | ArrayLike, p2: NamedArray | ArrayLike) -> NamedArray:
    """Named version of [jax.numpy.polymul][].

    If neither input is a [haliax.NamedArray][], the coefficient axis is named
    ``degree``; otherwise the axis from the NamedArray input is reused (resized as
    needed).
    """

    a1, a2 = unwrap_namedarrays(p1, p2)
    result = jnp.polymul(a1, a2)
    axis = _poly_axis_from_input(p1 if isinstance(p1, NamedArray) else p2, result.shape[0])
    return NamedArray(result, (axis,))


def polydiv(p1: NamedArray | ArrayLike, p2: NamedArray | ArrayLike) -> tuple[NamedArray, NamedArray]:
    """Named version of [jax.numpy.polydiv][].

    The quotient and remainder reuse the coefficient axis from the NamedArray input
    when available; otherwise their coefficient axes are named ``degree``.
    """

    a1, a2 = unwrap_namedarrays(p1, p2)
    q, r = jnp.polydiv(a1, a2)
    base = p1 if isinstance(p1, NamedArray) else p2
    axis_q = _poly_axis_from_input(base, q.shape[0])
    axis_r = _poly_axis_from_input(base, r.shape[0])
    return NamedArray(q, (axis_q,)), NamedArray(r, (axis_r,))


def polyint(p: NamedArray | ArrayLike, m: int = 1, k: ArrayLike | NamedArray | None = None) -> NamedArray:
    """Named version of [jax.numpy.polyint][].

    If ``p`` is not a [haliax.NamedArray][], the integrated polynomial uses a
    coefficient axis named ``degree``.
    """

    (arr,) = unwrap_namedarrays(p)
    k_arr = None
    if k is not None:
        (k_arr,) = unwrap_namedarrays(k)
    result = jnp.polyint(arr, m=m, k=k_arr)
    axis = _poly_axis_from_input(p, result.shape[0])
    return NamedArray(result, (axis,))


def polyder(p: NamedArray | ArrayLike, m: int = 1) -> NamedArray:
    """Named version of [jax.numpy.polyder][].

    If ``p`` is not a [haliax.NamedArray][], the differentiated polynomial uses a
    coefficient axis named ``degree``.
    """

    (arr,) = unwrap_namedarrays(p)
    result = jnp.polyder(arr, m=m)
    axis = _poly_axis_from_input(p, result.shape[0])
    return NamedArray(result, (axis,))


def polyval(p: NamedArray | ArrayLike, x: NamedOrNumeric) -> NamedOrNumeric:
    """Named version of [jax.numpy.polyval][].

    When ``x`` is a [haliax.NamedArray][], the returned array reuses ``x``'s axes.
    Otherwise a regular :mod:`jax.numpy` array is returned.
    """

    arr_p, arr_x = unwrap_namedarrays(p, x)
    result = jnp.polyval(arr_p, arr_x)
    if isinstance(x, NamedArray):
        return NamedArray(result, x.axes)
    else:
        return result


@overload
def polyfit(
    x: NamedArray | ArrayLike,
    y: NamedArray | ArrayLike,
    deg: int,
    rcond: ArrayLike | None = ...,
    full: Literal[False] = ...,
    w: NamedArray | ArrayLike | None = ...,
    cov: Literal[False] = ...,
) -> NamedArray: ...


@overload
def polyfit(
    x: NamedArray | ArrayLike,
    y: NamedArray | ArrayLike,
    deg: int,
    rcond: ArrayLike | None = ...,
    full: Literal[True] = ...,
    w: NamedArray | ArrayLike | None = ...,
    cov: Literal[False] = ...,
) -> tuple[NamedArray, Array, Array, Array, Array]: ...


@overload
def polyfit(
    x: NamedArray | ArrayLike,
    y: NamedArray | ArrayLike,
    deg: int,
    rcond: ArrayLike | None = ...,
    full: Literal[False] = ...,
    w: NamedArray | ArrayLike | None = ...,
    cov: Literal[True] = ...,
) -> tuple[NamedArray, NamedArray]: ...


@overload
def polyfit(
    x: NamedArray | ArrayLike,
    y: NamedArray | ArrayLike,
    deg: int,
    rcond: ArrayLike | None = ...,
    full: Literal[True] = ...,
    w: NamedArray | ArrayLike | None = ...,
    cov: Literal[True] = ...,
) -> tuple[NamedArray, Array, Array, Array, Array]: ...


def polyfit(
    x: NamedArray | ArrayLike,
    y: NamedArray | ArrayLike,
    deg: int,
    rcond: ArrayLike | None = None,
    full: bool = False,
    w: NamedArray | ArrayLike | None = None,
    cov: bool = False,
) -> NamedArray | tuple:
    """Named version of [jax.numpy.polyfit][].

    If neither ``x`` nor ``y`` is a [haliax.NamedArray][], the fitted coefficients
    use a coefficient axis named ``degree``; otherwise the axis from the NamedArray
    input is reused. When ``cov`` is ``True``, the returned covariance matrix is
    wrapped in a [haliax.NamedArray][] whose row axis matches the coefficient
    axis and whose column axis uses the same name with a ``"_cov"`` suffix.
    """

    x_arr, y_arr = unwrap_namedarrays(x, y)
    rcond_arr = None
    if rcond is not None:
        (rcond_arr,) = unwrap_namedarrays(rcond)
    w_arr = None
    if w is not None:
        (w_arr,) = unwrap_namedarrays(w)
    result = jnp.polyfit(x_arr, y_arr, deg, rcond=rcond_arr, full=full, w=w_arr, cov=cov)

    def wrap_coeffs(coeffs):
        base = x if isinstance(x, NamedArray) else y
        axis = _poly_axis_from_input(base, coeffs.shape[0])
        return NamedArray(coeffs, (axis,))

    if full:
        coeffs = wrap_coeffs(result[0])
        return (coeffs,) + result[1:]
    coeffs = wrap_coeffs(result if not cov else result[0])
    if cov:
        cov_axis = coeffs.axes[0]
        return coeffs, NamedArray(result[1], (cov_axis, cov_axis.alias(f"{cov_axis.name}_cov")))
    return coeffs


def roots(p: NamedArray | ArrayLike) -> NamedArray:
    """Named version of [jax.numpy.roots][].

    If ``p`` is not a [haliax.NamedArray][], the root axis is named ``degree``.
    """

    (arr,) = unwrap_namedarrays(p)
    result = jnp.roots(arr)
    axis = _poly_axis_from_input(p, result.shape[0])
    return NamedArray(result, (axis,))


def trim_zeros(f: NamedArray | ArrayLike, trim: str = "fb") -> NamedArray:
    """Named version of [jax.numpy.trim_zeros][].

    If ``f`` is not a [haliax.NamedArray][], the trimmed coefficient axis is named
    ``degree``.
    """

    (arr,) = unwrap_namedarrays(f)
    result = jnp.trim_zeros(arr, trim=trim)
    axis = _poly_axis_from_input(f, result.shape[0])
    return NamedArray(result, (axis,))


def vander(x: NamedArray, degree: AxisSelector) -> NamedArray:
    """Named version of [jax.numpy.vander][].

    Args:
        x: Input array of shape ``(n,)``.
        degree: Axis for the polynomial degree in the output. If a string is
            provided, an axis with that name and size ``n`` is created.

    Returns:
        Vandermonde matrix with row axis from ``x`` and the provided degree axis.
    """

    if x.ndim != 1:
        raise ValueError("vander only supports 1D input")

    if isinstance(degree, Axis):
        N = degree.size
        deg_axis = degree
    else:
        N = x.axes[0].size
        deg_axis = Axis(axis_name(degree), N)

    result = jnp.vander(x.array, N=N)
    return NamedArray(result, (x.axes[0], deg_axis))


__all__ = [
    "DEFAULT_POLY_AXIS_NAME",
    "poly",
    "polyadd",
    "polysub",
    "polymul",
    "polydiv",
    "polyint",
    "polyder",
    "polyval",
    "polyfit",
    "roots",
    "trim_zeros",
    "vander",
]
