# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Value parity for QuACK's varlen-k grouped weight gradient.

The kernel needs an SM100 GPU and the CUDA 13 GPU extra, so these skip everywhere else. What
they are here to catch is the wiring rather than the arithmetic: a wrong `mode` permutation in
one of the tensor specs, a `ragged_axis` that silently reverts to grouping over rows, or a QuACK
bump that reorders `make_varlen_args`' slots. Each of those returns a plausibly-shaped array of
wrong numbers, which no shape check and no benchmark timing would notice.

The argument checks below need no GPU and run everywhere.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def _sm100_or_skip():
    quack_moe_cute = pytest.importorskip(
        "levanter.grug._moe.quack_moe_cute", reason="QuACK ships only with the CUDA 13 GPU extra"
    )
    if jax.default_backend() != "gpu":
        pytest.skip("QuACK's grouped GEMMs are SM100 kernels")
    if float(jax.devices("gpu")[0].compute_capability) < 10.0:
        pytest.skip("QuACK's grouped GEMMs require SM100")
    return quack_moe_cute


def _reference(lhs: np.ndarray, rhs: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    """Per-group ``lhs.T @ rhs`` in float32, over exactly the rows each group owns."""
    out = np.zeros((len(sizes), lhs.shape[1], rhs.shape[1]), dtype=np.float32)
    start = 0
    for i, size in enumerate(sizes):
        stop = start + int(size)
        out[i] = lhs[start:stop].astype(np.float32).T @ rhs[start:stop].astype(np.float32)
        start = stop
    return out


@pytest.mark.parametrize(
    "sizes",
    [
        pytest.param([64, 96, 32], id="uneven"),
        # Not a multiple of any tile the kernel might use, which is the shape the deleted cuDNN
        # path had to pad away from and this one must handle as it lies.
        pytest.param([37, 91, 5], id="unaligned"),
        # A capacity-clipped expert. `_prefix_cap_counts` produces these on ordinary EP steps, and
        # the kernel has to clear the accumulator rather than leave whatever the last tile wrote.
        pytest.param([128, 0, 64], id="empty-group"),
        pytest.param([0, 0, 192], id="leading-empty-groups"),
    ],
)
def test_grouped_wgrad_matches_a_per_group_reference(sizes):
    quack_moe_cute = _sm100_or_skip()

    sizes = np.array(sizes, dtype=np.int64)
    rows, m, n = 256, 32, 64
    rng = np.random.default_rng(0)
    lhs = rng.standard_normal(size=(rows, m), dtype=np.float32)
    rhs = rng.standard_normal(size=(rows, n), dtype=np.float32)
    # Rows past the last group are live memory the kernel must not read; a NaN here turns an
    # over-read into a NaN output rather than a small drift.
    lhs[int(sizes.sum()) :] = np.nan
    rhs[int(sizes.sum()) :] = np.nan

    lhs_d = jnp.asarray(lhs, dtype=jnp.bfloat16)
    rhs_d = jnp.asarray(rhs, dtype=jnp.bfloat16)
    cu = jnp.concatenate([jnp.zeros((1,), jnp.int32), jnp.cumsum(jnp.asarray(sizes, jnp.int32))])

    got = np.asarray(jax.jit(quack_moe_cute.quack_grouped_wgrad)(lhs_d, rhs_d, cu), dtype=np.float32)
    want = _reference(np.asarray(lhs_d, dtype=np.float32), np.asarray(rhs_d, dtype=np.float32), sizes)

    assert np.isfinite(got).all(), "kernel read past the last group"
    # An empty group must be zero, not whatever the previous tile left behind.
    for i, size in enumerate(sizes):
        if size == 0:
            np.testing.assert_array_equal(got[i], np.zeros_like(got[i]))
    scale = max(np.abs(want).max(), 1e-6)
    np.testing.assert_allclose(got / scale, want / scale, atol=2e-2)


def test_grouped_wgrad_transposes_the_contraction():
    """A kernel grouping over rows instead of the contraction would still return this shape."""
    quack_moe_cute = _sm100_or_skip()

    sizes = np.array([96, 160], dtype=np.int64)
    m, n = 32, 48
    rng = np.random.default_rng(1)
    lhs = rng.standard_normal(size=(int(sizes.sum()), m), dtype=np.float32)
    rhs = rng.standard_normal(size=(int(sizes.sum()), n), dtype=np.float32)
    lhs_d = jnp.asarray(lhs, dtype=jnp.bfloat16)
    rhs_d = jnp.asarray(rhs, dtype=jnp.bfloat16)
    cu = jnp.concatenate([jnp.zeros((1,), jnp.int32), jnp.cumsum(jnp.asarray(sizes, jnp.int32))])

    got = jax.jit(quack_moe_cute.quack_grouped_wgrad)(lhs_d, rhs_d, cu)
    assert got.shape == (len(sizes), m, n)

    # Swapping the operands must transpose each group's result, which a row-grouped kernel or a
    # mismatched pair of tensor `mode` maps would not reproduce.
    swapped = jax.jit(quack_moe_cute.quack_grouped_wgrad)(rhs_d, lhs_d, cu)
    np.testing.assert_allclose(
        np.asarray(got, dtype=np.float32),
        np.transpose(np.asarray(swapped, dtype=np.float32), (0, 2, 1)),
        atol=2e-2,
    )


@pytest.mark.parametrize(
    "shapes,message",
    [
        ({"lhs": (8,), "rhs": (8, 16)}, "rank 2"),
        ({"lhs": (8, 16), "rhs": (9, 16)}, "row counts"),
        ({"lhs": (8, 12), "rhs": (8, 16)}, "divide"),
    ],
)
def test_grouped_wgrad_rejects_bad_shapes(shapes, message):
    """Shape validation is pure Python and needs no GPU."""
    quack_moe_cute = pytest.importorskip(
        "levanter.grug._moe.quack_moe_cute", reason="QuACK ships only with the CUDA 13 GPU extra"
    )
    lhs = jnp.zeros(shapes["lhs"], dtype=jnp.bfloat16)
    rhs = jnp.zeros(shapes["rhs"], dtype=jnp.bfloat16)
    cu = jnp.asarray([0, 8], dtype=jnp.int32)
    with pytest.raises(ValueError, match=message):
        quack_moe_cute.quack_grouped_wgrad(lhs, rhs, cu)


def test_grouped_wgrad_refuses_float32_rather_than_demoting_to_tf32():
    """fp32 operands would be accepted by the kernel and quietly run at TF32 precision."""
    quack_moe_cute = pytest.importorskip(
        "levanter.grug._moe.quack_moe_cute", reason="QuACK ships only with the CUDA 13 GPU extra"
    )
    lhs = jnp.zeros((8, 16), dtype=jnp.float32)
    rhs = jnp.zeros((8, 16), dtype=jnp.float32)
    cu = jnp.asarray([0, 8], dtype=jnp.int32)
    with pytest.raises(ValueError, match="16-bit float"):
        quack_moe_cute.quack_grouped_wgrad(lhs, rhs, cu)
