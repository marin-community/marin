# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Streaming Triton kernels for the depthwise causal short convolution.

The Pallas kernels (``pallas_gpu.py``) read each ``[BS, BC]`` tile once per tap at a shifted
offset, because Pallas Triton cannot slice or shift a register tile. That is 3.0 HBM passes
forward and 6.7 backward against floors of 2 and 3.

These kernels walk the sequence instead, the way ``Dao-AILab/causal-conv1d`` does. A program
owns one channel block of one sequence chunk and steps through its rows in order, keeping the
previous ``W - 1`` rows (and, backward, the next ``W - 1`` rows of the output cotangent) in
registers as loop-carried values. Each row of every tensor is read once from HBM, plus a halo of
``W - 1`` rows per chunk, so the traffic sits at the floor: forward reads ``x`` and writes ``y``;
backward reads ``x`` and ``dy`` and writes ``dx`` and one fp32 ``dw`` partial per chunk.

Numerics match the Pallas kernels. With ``exact`` set, every multiply and add rounds to the
activation dtype in the reference's order: ascending lags forward, descending lags then tap 0
for ``dx``. The forward and ``dx`` are then bit-identical to ``short_conv_reference``. ``dw``
accumulates in fp32 per chunk; the caller sums the partials.

The kernels handle ``kernel_size == 4`` only, the hero's width.
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from .config import OOB_SEGMENT

try:
    import jax_triton as jt
    import triton
    import triton.language as tl
except ModuleNotFoundError:
    jt = None
    triton = None
    tl = None

KERNEL_SIZE = 4
#: Rows each program walks. The halo re-read is ``(KERNEL_SIZE - 1) / CHUNK`` of a pass.
CHUNK = 64
#: Rows loaded per loop iteration; independent loads in flight per program.
UNROLL = 8
_MAX_CHANNEL_BLOCK = 1024


def triton_short_conv_available() -> bool:
    return jt is not None and triton is not None and jax.default_backend() == "gpu"


def _channel_block(channels: int) -> int | None:
    block = _MAX_CHANNEL_BLOCK
    while block >= 64:
        if channels % block == 0:
            return block
        block //= 2
    return None


def triton_short_conv_shapes_supported(weight_shape: tuple[int, ...], x_shape: tuple[int, ...]) -> str | None:
    """Returns None when the kernels can run these shapes, else a reason."""
    if len(x_shape) != 3 or len(weight_shape) != 2:
        return f"expected weight [W, C] and x [B, S, C], got {weight_shape} and {x_shape}"
    width, weight_channels = weight_shape
    _, seq_len, channels = x_shape
    if width != KERNEL_SIZE:
        return f"kernel_size must be {KERNEL_SIZE}, got {width}"
    if weight_channels != channels:
        return f"weight channel dim {weight_channels} != x channel dim {channels}"
    if seq_len % CHUNK:
        return f"seq_len {seq_len} not divisible by {CHUNK}"
    if _channel_block(channels) is None:
        return f"channels {channels} not divisible by 64"
    return None


if triton is not None and tl is not None:

    @triton.jit
    def _round(value, dtype: tl.constexpr, exact: tl.constexpr):
        if exact:
            return value.to(dtype).to(tl.float32)
        return value

    @triton.jit
    def _row(base_ptr, row, seq_len: tl.constexpr, channels: tl.constexpr, cols):
        """Row ``row`` of a ``[S, C]`` slab at ``cols``, zero outside ``[0, S)``."""
        inside = (row >= 0) & (row < seq_len)
        return tl.load(base_ptr + row.to(tl.int64) * channels + cols, mask=inside, other=0.0)

    @triton.jit
    def _seg(seg_ptr, row, seq_len: tl.constexpr):
        inside = (row >= 0) & (row < seq_len)
        return tl.load(seg_ptr + row, mask=inside, other=-1)

    @triton.jit
    def _short_conv_fwd_kernel(
        x_ptr,
        seg_ptr,
        w_ptr,
        out_ptr,
        seq_len: tl.constexpr,
        channels: tl.constexpr,
        chunk: tl.constexpr,
        block_c: tl.constexpr,
        unroll: tl.constexpr,
        exact: tl.constexpr,
    ):
        dtype = out_ptr.dtype.element_ty
        cols = tl.program_id(0) * block_c + tl.arange(0, block_c)
        start = tl.program_id(1) * chunk
        batch = tl.program_id(2).to(tl.int64)
        x_base = x_ptr + batch * seq_len * channels
        out_base = out_ptr + batch * seq_len * channels
        seg_base = seg_ptr + batch * seq_len
        w0 = tl.load(w_ptr + cols).to(tl.float32)
        w1 = tl.load(w_ptr + channels + cols).to(tl.float32)
        w2 = tl.load(w_ptr + 2 * channels + cols).to(tl.float32)
        w3 = tl.load(w_ptr + 3 * channels + cols).to(tl.float32)
        # x at t-1, t-2, t-3 and their segment ids; outside the sequence reads as a zero row
        # whose segment id matches nothing.
        x1 = _row(x_base, start - 1, seq_len, channels, cols)
        x2 = _row(x_base, start - 2, seq_len, channels, cols)
        x3 = _row(x_base, start - 3, seq_len, channels, cols)
        s1 = _seg(seg_base, start - 1, seq_len)
        s2 = _seg(seg_base, start - 2, seq_len)
        s3 = _seg(seg_base, start - 3, seq_len)
        for offset in range(0, chunk, unroll):
            for u in tl.static_range(unroll):
                t = start + offset + u
                x0 = tl.load(x_base + t.to(tl.int64) * channels + cols)
                s0 = tl.load(seg_base + t)
                # Ascending lags, the reference's order; a tap across a document boundary reads 0.
                acc = _round(w0 * x0.to(tl.float32), dtype, exact)
                k1 = tl.where(s1 == s0, x1, 0.0).to(tl.float32)
                acc = _round(acc + _round(w1 * k1, dtype, exact), dtype, exact)
                k2 = tl.where(s2 == s0, x2, 0.0).to(tl.float32)
                acc = _round(acc + _round(w2 * k2, dtype, exact), dtype, exact)
                k3 = tl.where(s3 == s0, x3, 0.0).to(tl.float32)
                acc = _round(acc + _round(w3 * k3, dtype, exact), dtype, exact)
                tl.store(out_base + t.to(tl.int64) * channels + cols, acc.to(dtype))
                x3, s3 = x2, s2
                x2, s2 = x1, s1
                x1, s1 = x0, s0

    @triton.jit
    def _short_conv_bwd_kernel(
        x_ptr,
        seg_ptr,
        dy_ptr,
        w_ptr,
        dx_ptr,
        dw_ptr,
        seq_len: tl.constexpr,
        channels: tl.constexpr,
        chunk: tl.constexpr,
        block_c: tl.constexpr,
        unroll: tl.constexpr,
        exact: tl.constexpr,
    ):
        dtype = dx_ptr.dtype.element_ty
        cols = tl.program_id(0) * block_c + tl.arange(0, block_c)
        chunk_id = tl.program_id(1)
        start = chunk_id * chunk
        batch = tl.program_id(2).to(tl.int64)
        x_base = x_ptr + batch * seq_len * channels
        dy_base = dy_ptr + batch * seq_len * channels
        dx_base = dx_ptr + batch * seq_len * channels
        seg_base = seg_ptr + batch * seq_len
        w0 = tl.load(w_ptr + cols).to(tl.float32)
        w1 = tl.load(w_ptr + channels + cols).to(tl.float32)
        w2 = tl.load(w_ptr + 2 * channels + cols).to(tl.float32)
        w3 = tl.load(w_ptr + 3 * channels + cols).to(tl.float32)
        # Row r's dx needs dy at r..r+3; its dw terms need x at r..r-3. Carry x behind
        # (xb1..xb3) and dy ahead (d0 = row r, d1, d2), each with its segment ids.
        xb1 = _row(x_base, start - 1, seq_len, channels, cols)
        xb2 = _row(x_base, start - 2, seq_len, channels, cols)
        xb3 = _row(x_base, start - 3, seq_len, channels, cols)
        sb1 = _seg(seg_base, start - 1, seq_len)
        sb2 = _seg(seg_base, start - 2, seq_len)
        sb3 = _seg(seg_base, start - 3, seq_len)
        d0 = _row(dy_base, start, seq_len, channels, cols)
        d1 = _row(dy_base, start + 1, seq_len, channels, cols)
        d2 = _row(dy_base, start + 2, seq_len, channels, cols)
        sd0 = _seg(seg_base, start, seq_len)
        sd1 = _seg(seg_base, start + 1, seq_len)
        sd2 = _seg(seg_base, start + 2, seq_len)
        dw0 = tl.zeros([block_c], dtype=tl.float32)
        dw1 = tl.zeros([block_c], dtype=tl.float32)
        dw2 = tl.zeros([block_c], dtype=tl.float32)
        dw3 = tl.zeros([block_c], dtype=tl.float32)
        for offset in range(0, chunk, unroll):
            for u in tl.static_range(unroll):
                r = start + offset + u
                d3 = _row(dy_base, r + 3, seq_len, channels, cols)
                sd3 = _seg(seg_base, r + 3, seq_len)
                xr = tl.load(x_base + r.to(tl.int64) * channels + cols)
                # dx: descending lags, then tap 0 (the order of the reference's transpose).
                acc = tl.where(sd3 == sd0, _round(w3 * d3.to(tl.float32), dtype, exact), 0.0)
                acc = _round(
                    acc + tl.where(sd2 == sd0, _round(w2 * d2.to(tl.float32), dtype, exact), 0.0), dtype, exact
                )
                acc = _round(
                    acc + tl.where(sd1 == sd0, _round(w1 * d1.to(tl.float32), dtype, exact), 0.0), dtype, exact
                )
                acc = _round(acc + _round(w0 * d0.to(tl.float32), dtype, exact), dtype, exact)
                tl.store(dx_base + r.to(tl.int64) * channels + cols, acc.to(dtype))
                # dw: fp32 sums of dy[r] times the shifted, masked x.
                g = d0.to(tl.float32)
                dw0 += g * xr.to(tl.float32)
                dw1 += g * tl.where(sb1 == sd0, xb1, 0.0).to(tl.float32)
                dw2 += g * tl.where(sb2 == sd0, xb2, 0.0).to(tl.float32)
                dw3 += g * tl.where(sb3 == sd0, xb3, 0.0).to(tl.float32)
                xb3, sb3 = xb2, sb2
                xb2, sb2 = xb1, sb1
                xb1, sb1 = xr, sd0
                d0, sd0 = d1, sd1
                d1, sd1 = d2, sd2
                d2, sd2 = d3, sd3
        partial = dw_ptr + (batch * (seq_len // chunk) + chunk_id) * 4 * channels
        tl.store(partial + cols, dw0)
        tl.store(partial + channels + cols, dw1)
        tl.store(partial + 2 * channels + cols, dw2)
        tl.store(partial + 3 * channels + cols, dw3)

else:
    _short_conv_fwd_kernel = None
    _short_conv_bwd_kernel = None


def _launch_config(x: jax.Array) -> tuple[int, tuple[int, int, int]]:
    batch, seq_len, channels = x.shape
    block_c = _channel_block(channels)
    assert block_c is not None
    return block_c, (channels // block_c, seq_len // CHUNK, batch)


def _num_warps(block_c: int) -> int:
    return 4 if block_c >= 512 else 2


def short_conv_triton_fwd_local(
    weight: Float[Array, "W C"],
    x: Float[Array, "B S C"],
    segment_ids: Int[Array, "B S"],
    *,
    exact_reference_rounding: bool,
) -> Float[Array, "B S C"]:
    """Shard-local forward. Callers must have already entered a ``shard_map``."""
    batch, seq_len, channels = x.shape
    block_c, grid = _launch_config(x)
    return jt.triton_call(
        x,
        segment_ids.astype(jnp.int32),
        weight,
        kernel=_short_conv_fwd_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=grid,
        num_warps=_num_warps(block_c),
        num_stages=1,
        seq_len=seq_len,
        channels=channels,
        chunk=CHUNK,
        block_c=block_c,
        unroll=UNROLL,
        exact=exact_reference_rounding,
    )


def short_conv_triton_bwd_local(
    weight: Float[Array, "W C"],
    x: Float[Array, "B S C"],
    segment_ids: Int[Array, "B S"],
    dy: Float[Array, "B S C"],
    *,
    exact_reference_rounding: bool,
) -> tuple[Float[Array, "B S C"], Float[Array, "P W C"]]:
    """Shard-local backward. Returns ``(dx, dw_partials)`` with ``dw_partials`` ``[B * S / CHUNK, W, C]``."""
    batch, seq_len, channels = x.shape
    block_c, grid = _launch_config(x)
    dx, dw_partials = jt.triton_call(
        x,
        segment_ids.astype(jnp.int32),
        dy,
        weight,
        kernel=_short_conv_bwd_kernel,
        out_shape=(
            jax.ShapeDtypeStruct(x.shape, dy.dtype),
            jax.ShapeDtypeStruct((batch * (seq_len // CHUNK), KERNEL_SIZE, channels), jnp.float32),
        ),
        grid=grid,
        num_warps=_num_warps(block_c),
        num_stages=1,
        seq_len=seq_len,
        channels=channels,
        chunk=CHUNK,
        block_c=block_c,
        unroll=UNROLL,
        exact=exact_reference_rounding,
    )
    return dx, dw_partials


__all__ = [
    "OOB_SEGMENT",
    "short_conv_triton_bwd_local",
    "short_conv_triton_fwd_local",
    "triton_short_conv_available",
    "triton_short_conv_shapes_supported",
]
