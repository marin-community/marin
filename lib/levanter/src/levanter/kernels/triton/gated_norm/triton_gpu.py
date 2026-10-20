# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Triton kernels for the restructured backward's two ``[T, D]`` elementwise steps.

Each step also reduces, ``dg`` over rows and ``dx`` over columns, and XLA runs the reduction as a
second fusion that re-reads the inputs. These kernels do each step in one 2-D-tiled sweep and write
per-tile fp32 partial sums, which the caller adds up. The rank-``R`` GEMMs around them stay on XLA:
a ``K = R`` ``tl.dot`` inside these kernels ran at about 150 TFLOP/s on B200 with Triton 3.6, so
fusing the GEMMs in made the op slower than XLA.
"""

import jax
import jax.numpy as jnp

try:
    import jax_triton as jt
    import triton
    import triton.language as tl
except ModuleNotFoundError:
    jt = None
    triton = None
    tl = None

# Best measured on one B200 at the hero shape ([65_536, 6_144] bf16) among 32-256 rows, 64-256
# columns, and 4 or 8 warps.
ROW_BLOCK = 64
COL_BLOCK = 128
NUM_WARPS = 4


def triton_gated_norm_available() -> bool:
    """Whether the Triton packages are installed and this process's default devices are GPUs."""
    return jt is not None and triton is not None and jax.devices()[0].platform == "gpu"


def triton_gated_norm_shapes_supported(tokens: int, hidden: int) -> str | None:
    """Returns None when the kernels can tile a ``[tokens, hidden]`` shard, else a reason."""
    if tokens % ROW_BLOCK or hidden % COL_BLOCK:
        return f"local shape [{tokens}, {hidden}] is not a multiple of the [{ROW_BLOCK}, {COL_BLOCK}] tile"
    return None


if triton is not None:

    @triton.jit
    def _dg_kernel(
        x_ptr,
        dy_ptr,
        g_ptr,
        rstd_ptr,
        w_ptr,
        dg_ptr,
        row_sum_ptr,
        T: tl.constexpr,
        D: tl.constexpr,
        BT: tl.constexpr,
        BD: tl.constexpr,
    ):
        rows = tl.program_id(0) * BT + tl.arange(0, BT)
        cols = tl.program_id(1) * BD + tl.arange(0, BD)
        offsets = rows[:, None] * D + cols[None, :]
        x = tl.load(x_ptr + offsets).to(tl.float32)
        dy = tl.load(dy_ptr + offsets).to(tl.float32)
        gate = tl.sigmoid(tl.load(g_ptr + offsets).to(tl.float32))
        rstd = tl.load(rstd_ptr + rows)
        w = tl.load(w_ptr + cols).to(tl.float32)
        dy_gate_n = dy * gate * (x * rstd[:, None] * w[None, :])
        tl.store(dg_ptr + offsets, (dy_gate_n * (1.0 - gate)).to(dg_ptr.dtype.element_ty))
        tl.store(row_sum_ptr + tl.program_id(1) * T + rows, tl.sum(dy_gate_n, axis=1))

    @triton.jit
    def _dx_kernel(
        x_ptr,
        dy_ptr,
        g_ptr,
        dnb_ptr,
        rstd_ptr,
        c_ptr,
        w_ptr,
        dx_ptr,
        col_sum_ptr,
        D: tl.constexpr,
        BT: tl.constexpr,
        BD: tl.constexpr,
    ):
        rows = tl.program_id(0) * BT + tl.arange(0, BT)
        cols = tl.program_id(1) * BD + tl.arange(0, BD)
        offsets = rows[:, None] * D + cols[None, :]
        x = tl.load(x_ptr + offsets).to(tl.float32)
        dy = tl.load(dy_ptr + offsets).to(tl.float32)
        gate = tl.sigmoid(tl.load(g_ptr + offsets).to(tl.float32))
        dn = dy * gate + tl.load(dnb_ptr + offsets).to(tl.float32)
        rstd = tl.load(rstd_ptr + rows)
        c = tl.load(c_ptr + rows)
        w = tl.load(w_ptr + cols).to(tl.float32)
        xhat = x * rstd[:, None]
        dx = rstd[:, None] * (dn * w[None, :] - xhat * (c[:, None] / D))
        tl.store(dx_ptr + offsets, dx.to(dx_ptr.dtype.element_ty))
        tl.store(col_sum_ptr + tl.program_id(0) * D + cols, tl.sum(dn * xhat, axis=0))


def _launch_kwargs(tokens: int, hidden: int) -> dict:
    reason = triton_gated_norm_shapes_supported(tokens, hidden)
    assert reason is None, reason
    return dict(
        grid=(tokens // ROW_BLOCK, hidden // COL_BLOCK),
        num_warps=NUM_WARPS,
        num_stages=1,
        BT=ROW_BLOCK,
        BD=COL_BLOCK,
    )


def dg_triton(x, dy, g, rstd, rms_weight):
    """``xla.dg_xla`` in one sweep."""
    tokens, hidden = x.shape
    dg, row_sums = jt.triton_call(
        x,
        dy,
        g,
        rstd,
        rms_weight,
        kernel=_dg_kernel,
        out_shape=(
            jax.ShapeDtypeStruct((tokens, hidden), dy.dtype),
            jax.ShapeDtypeStruct((hidden // COL_BLOCK, tokens), jnp.float32),
        ),
        T=tokens,
        D=hidden,
        **_launch_kwargs(tokens, hidden),
    )
    return dg, jnp.sum(row_sums, axis=0)


def dx_triton(x, dy, g, dn_bottleneck, rstd, c, rms_weight):
    """``xla.dx_xla`` in one sweep."""
    tokens, hidden = x.shape
    dx, col_sums = jt.triton_call(
        x,
        dy,
        g,
        dn_bottleneck,
        rstd,
        c,
        rms_weight,
        kernel=_dx_kernel,
        out_shape=(
            jax.ShapeDtypeStruct((tokens, hidden), dy.dtype),
            jax.ShapeDtypeStruct((tokens // ROW_BLOCK, hidden), jnp.float32),
        ),
        D=hidden,
        **_launch_kwargs(tokens, hidden),
    )
    return dx, jnp.sum(col_sums, axis=0)
