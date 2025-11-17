# Copyright 2025 The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0

import jax.numpy as jnp
from typing import cast, no_type_check
import haliax as hax
from haliax import Axis
from haliax.core import NamedArray


def _assert_coefficients(coeffs: NamedArray, expected: jnp.ndarray, axis: Axis) -> None:
    assert jnp.allclose(coeffs.array, expected)  # type: ignore[union-attr]
    assert coeffs.axes[0] == axis.resize(coeffs.array.shape[0])  # type: ignore[union-attr]


def _assert_covariance_matrix(cov: NamedArray, axis: Axis, expected: jnp.ndarray) -> None:
    assert jnp.allclose(cov.array, expected)  # type: ignore[union-attr]
    assert cov.axes[0] == axis  # type: ignore[union-attr]
    assert cov.axes[1].name == f"{axis.name}_cov"  # type: ignore[union-attr]
    assert cov.axes[1].size == axis.size  # type: ignore[union-attr]


def test_poly():
    R = Axis("R", 3)
    roots = hax.named([1.0, 2.0, 3.0], (R,))
    coeffs = hax.poly(roots)
    assert jnp.allclose(coeffs.array, jnp.poly(roots.array))
    assert coeffs.axes[0] == R.resize(coeffs.array.shape[0])


def test_poly_arithmetic():
    C1 = Axis("C1", 4)
    p = hax.named([1.0, 0.0, -2.0, 1.0], (C1,))
    C2 = Axis("C2", 2)
    q = hax.named([1.0, -1.0], (C2,))

    add = hax.polyadd(p, q)
    assert jnp.allclose(add.array, jnp.polyadd(p.array, q.array))
    assert add.axes[0] == C1.resize(add.array.shape[0])

    sub = hax.polysub(p, q)
    assert jnp.allclose(sub.array, jnp.polysub(p.array, q.array))
    assert sub.axes[0] == C1.resize(sub.array.shape[0])

    mul = hax.polymul(p, q)
    assert jnp.allclose(mul.array, jnp.polymul(p.array, q.array))
    assert mul.axes[0] == C1.resize(mul.array.shape[0])

    div_q, div_r = hax.polydiv(p, q)
    exp_q, exp_r = jnp.polydiv(p.array, q.array)
    assert jnp.allclose(div_q.array, exp_q)
    assert jnp.allclose(div_r.array, exp_r)
    assert div_q.axes[0] == C1.resize(div_q.array.shape[0])
    assert div_r.axes[0] == C1.resize(div_r.array.shape[0])


def test_poly_int_der():
    C = Axis("C", 4)
    p = hax.named([1.0, 0.0, -2.0, 1.0], (C,))

    d = hax.polyder(p)
    i = hax.polyint(p)

    assert jnp.allclose(d.array, jnp.polyder(p.array))
    assert jnp.allclose(i.array, jnp.polyint(p.array))
    assert d.axes[0] == C.resize(d.array.shape[0])
    assert i.axes[0] == C.resize(i.array.shape[0])


@no_type_check
def test_polyval_polyfit():
    C = Axis("C", 4)
    p = hax.named([1.0, 0.0, -2.0, 1.0], (C,))

    X = Axis("X", 5)
    x = hax.arange(X, dtype=jnp.float64)
    y = hax.polyval(p, x)

    pv = cast(NamedArray, hax.polyval(p, x))
    assert jnp.allclose(pv.array, jnp.polyval(p.array, x.array))

    fit = cast(NamedArray, hax.polyfit(x, y, 3))
    assert jnp.allclose(fit.array, p.array, atol=1e-5)
    assert fit.axes[0] == X.resize(fit.array.shape[0])

    full_fit = hax.polyfit(x, y, 3, full=True)
    assert isinstance(full_fit, tuple)
    coeffs_full = full_fit[0]
    assert isinstance(coeffs_full, NamedArray)
    coeffs_full = cast(NamedArray, coeffs_full)
    exp_full = jnp.polyfit(x.array, y.array, 3, full=True)
    _assert_coefficients(coeffs_full, exp_full[0], X)  # type: ignore[union-attr]
    for res, exp in zip(full_fit[1:], exp_full[1:]):
        assert jnp.allclose(res, exp)

    cov_fit = hax.polyfit(x, y, 3, cov=True)
    assert isinstance(cov_fit, tuple)
    coeffs_cov = cov_fit[0]
    cov = cov_fit[1]
    assert isinstance(coeffs_cov, NamedArray)
    assert isinstance(cov, NamedArray)
    coeffs_cov = cast(NamedArray, coeffs_cov)
    cov = cast(NamedArray, cov)
    exp_coeffs, exp_cov = jnp.polyfit(x.array, y.array, 3, cov=True)
    _assert_coefficients(coeffs_cov, exp_coeffs, X)  # type: ignore[union-attr]
    _assert_covariance_matrix(cov, coeffs_cov.axes[0], exp_cov)  # type: ignore[union-attr]


def test_poly_roots_trim_vander():
    C = Axis("C", 4)
    p = hax.named([1.0, 0.0, -2.0, 1.0], (C,))

    r = hax.roots(p)
    assert jnp.allclose(r.array, jnp.roots(p.array))
    assert r.axes[0] == C.resize(r.array.shape[0])

    C2 = Axis("C2", 4)
    p2 = hax.named([0.0, 0.0, 1.0, 2.0], (C2,))
    t = hax.trim_zeros(p2)
    assert jnp.allclose(t.array, jnp.trim_zeros(p2.array))
    assert t.axes[0] == C2.resize(t.array.shape[0])

    X = Axis("X", 3)
    x = hax.named([1.0, 2.0, 3.0], (X,))
    D = Axis("D", 3)
    v = hax.vander(x, D)
    assert jnp.allclose(v.array, jnp.vander(x.array, N=3))
    assert v.axes == (X, D)

    v2 = hax.vander(x, "E")
    assert jnp.allclose(v2.array, jnp.vander(x.array))
    assert v2.axes[0] == X
    assert v2.axes[1] == Axis("E", X.size)
