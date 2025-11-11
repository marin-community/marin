# Copyright 2025 The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0

import jax.numpy as jnp
import jax.numpy.fft as jfft

import haliax as hax
from haliax import Axis


def test_fft_axis_options():
    N = Axis("n", 8)
    x = hax.arange(N, dtype=jnp.float32)

    # string axis
    assert jnp.allclose(hax.fft(x, axis="n").array, jfft.fft(x.array))

    # Axis object with resize
    N2 = Axis("n", 16)
    f = hax.fft(x, axis=N2)
    assert f.axes[0] == N2
    assert jnp.allclose(f.array, jfft.fft(x.array, n=16))

    r = hax.rfft(x)
    assert r.axes[0].size == 5
    assert jnp.allclose(r.array, jfft.rfft(x.array))

    ir = hax.irfft(r)
    assert ir.axes[0].size == 8
    assert jnp.allclose(ir.array, jfft.irfft(jfft.rfft(x.array)))

    h = hax.hfft(r)
    assert h.axes[0].size == 8
    assert jnp.allclose(h.array, jfft.hfft(jfft.rfft(x.array)))

    ih = hax.ihfft(x)
    assert ih.axes[0].size == 5
    assert jnp.allclose(ih.array, jfft.ihfft(x.array))


def test_fft_freq_and_shift():
    N = Axis("n", 8)
    x = hax.arange(N)

    f = hax.fftfreq(N)
    assert f.axes == (N,)
    assert jnp.allclose(f.array, jfft.fftfreq(8))

    rf = hax.rfftfreq(N)
    assert rf.axes[0].size == 5
    assert jnp.allclose(rf.array, jfft.rfftfreq(8))

    shifted = hax.fftshift(x)
    assert jnp.allclose(shifted.array, jfft.fftshift(x.array))
    unshifted = hax.ifftshift(shifted)
    assert jnp.allclose(unshifted.array, x.array)


def test_fft_multi_axis():
    X = Axis("x", 4)
    Y = Axis("y", 6)
    Z = Axis("z", 8)
    arr = hax.arange((X, Y, Z), dtype=jnp.float32)

    f = hax.fft(arr, axis={"y": None, "z": None})
    assert jnp.allclose(f.array, jfft.fftn(arr.array, axes=(1, 2)))

    f_seq = hax.fft(arr, axis=("y", "z"))
    assert jnp.allclose(f_seq.array, jfft.fftn(arr.array, axes=(1, 2)))
    assert f_seq.axes == f.axes

    rf = hax.rfft(arr, axis={"y": None, "z": None})
    assert rf.axes[2].size == 5
    assert jnp.allclose(rf.array, jfft.rfftn(arr.array, axes=(1, 2)))

    irf = hax.irfft(rf, axis={"y": None, "z": Z})
    assert jnp.allclose(irf.array, jfft.irfftn(jfft.rfftn(arr.array, axes=(1, 2)), s=(Y.size, Z.size), axes=(1, 2)))

    # resizing via dict values
    f2 = hax.fft(arr, axis={"y": 4, "z": None})
    assert f2.axes[1].size == 4
    assert jnp.allclose(f2.array, jfft.fftn(arr.array, s=(4, 8), axes=(1, 2)))
