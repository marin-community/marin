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

Numerics match the Pallas kernels. With ``exact`` set (bfloat16 only), every multiply and add
rounds to bfloat16 in the reference's order: ascending lags forward, descending lags then tap 0
for ``dx``, with packed bf16 PTX instructions so the compiler cannot fuse them. The forward and
``dx`` are then bit-identical to ``short_conv_reference``. Without it, taps accumulate in fp32.
``dw`` accumulates in fp32 per chunk; the caller sums the partials.

The kernels handle ``kernel_size == 4`` only, the hero's width.
"""

import dataclasses
import math

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


@dataclasses.dataclass(frozen=True)
class TritonShortConvTiles:
    """Launch shape of one direction's kernel.

    Attributes:
      chunk: rows each program walks; the sequence length must divide by it. The halo re-read
        is ``(KERNEL_SIZE - 1) / chunk`` of a pass.
      max_channel_block: channels per program, halved until it divides the channel count.
      num_warps: warps per program.
      num_stages: Triton software-pipelining depth of the row loop.
      rows_per_step: rows loaded together before any is stored, 4 or 8.
    """

    chunk: int
    max_channel_block: int
    num_warps: int
    num_stages: int
    rows_per_step: int

    def channel_block(self, channels: int) -> int | None:
        block = self.max_channel_block
        while block >= 64:
            if channels % block == 0:
                return block
            block //= 2
        return None


# Best measured on GB200 at the hero shapes ([16, 4096, C], C in {6144, 1536}), weighted by calls per
# layer (autoresearch/loop-260930-mfu30/b/sconv_sweep.py). The backward carries twice the ring and four
# fp32 accumulators, so it wants two channels per thread; eight per thread spills its occupancy.
FORWARD_TILES = TritonShortConvTiles(chunk=32, max_channel_block=1024, num_warps=4, num_stages=1, rows_per_step=8)
BACKWARD_TILES = TritonShortConvTiles(chunk=128, max_channel_block=256, num_warps=4, num_stages=1, rows_per_step=8)
#: The local sequence length must be a multiple of this.
SEQUENCE_MULTIPLE = math.lcm(FORWARD_TILES.chunk, BACKWARD_TILES.chunk)


def triton_short_conv_available() -> bool:
    return jt is not None and triton is not None and jax.default_backend() == "gpu"


def triton_short_conv_shapes_supported(
    weight_shape: tuple[int, ...], x_shape: tuple[int, ...], dtype, exact_reference_rounding: bool
) -> str | None:
    """Returns None when the kernels can run these shapes and dtype, else a reason."""
    if exact_reference_rounding and jnp.dtype(dtype) != jnp.dtype(jnp.bfloat16):
        return f"exact reference rounding is implemented for bfloat16 only, got {jnp.dtype(dtype)}"
    if len(x_shape) != 3 or len(weight_shape) != 2:
        return f"expected weight [W, C] and x [B, S, C], got {weight_shape} and {x_shape}"
    width, weight_channels = weight_shape
    _, seq_len, channels = x_shape
    if width != KERNEL_SIZE:
        return f"kernel_size must be {KERNEL_SIZE}, got {width}"
    if weight_channels != channels:
        return f"weight channel dim {weight_channels} != x channel dim {channels}"
    if seq_len % SEQUENCE_MULTIPLE:
        return f"seq_len {seq_len} not divisible by {SEQUENCE_MULTIPLE}"
    if FORWARD_TILES.channel_block(channels) is None or BACKWARD_TILES.channel_block(channels) is None:
        return f"channels {channels} not divisible by 64"
    return None


if triton is not None and tl is not None:

    @triton.jit
    def _mul(a, b, exact: tl.constexpr):
        """``a * b``. Exact: one bf16 rounding per multiply, as the reference rounds each op."""
        if exact:
            # Inline PTX, because LLVM otherwise folds the f32 round trips of the reference's
            # per-op rounding into bf16 arithmetic and then contracts a multiply and the
            # following add into one FMA, which rounds once where the reference rounds twice.
            product = tl.inline_asm_elementwise(
                "mul.rn.bf16x2 $0, $1, $2;", "=r,r,r", [a, b], dtype=tl.bfloat16, is_pure=True, pack=2
            )
        else:
            product = a.to(tl.float32) * b.to(tl.float32)
        return product

    @triton.jit
    def _add(a, b, exact: tl.constexpr):
        """``a + b``. Exact: one bf16 rounding per add. For bf16 operands this equals the
        reference's fp32 add then bf16 rounding: the fp32 sum is exact whenever the rounding
        to bf16 can depend on it."""
        if exact:
            total = tl.inline_asm_elementwise(
                "add.rn.bf16x2 $0, $1, $2;", "=r,r,r", [a, b], dtype=tl.bfloat16, is_pure=True, pack=2
            )
        else:
            total = a + b
        return total

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
    def _conv_row(x0, s0, x1, s1, x2, s2, x3, s3, w0, w1, w2, w3, dtype: tl.constexpr, exact: tl.constexpr):
        # Ascending lags, the reference's order. A tap across a document boundary reads zero
        # before the multiply, as the reference masks the shifted input.
        acc = _mul(w0, x0, exact)
        acc = _add(acc, _mul(w1, tl.where(s1 == s0, x1, 0.0), exact), exact)
        acc = _add(acc, _mul(w2, tl.where(s2 == s0, x2, 0.0), exact), exact)
        acc = _add(acc, _mul(w3, tl.where(s3 == s0, x3, 0.0), exact), exact)
        return acc.to(dtype)

    @triton.jit
    def _load_rows(base, row, channels: tl.constexpr, cols):
        """Rows ``row..row+3`` of a ``[S, C]`` slab, all in range."""
        xa = tl.load(base + row.to(tl.int64) * channels + cols)
        xb = tl.load(base + (row + 1).to(tl.int64) * channels + cols)
        xc = tl.load(base + (row + 2).to(tl.int64) * channels + cols)
        xd = tl.load(base + (row + 3).to(tl.int64) * channels + cols)
        return xa, xb, xc, xd

    @triton.jit
    def _load_rows_masked(base, row, seq_len: tl.constexpr, channels: tl.constexpr, cols):
        """Rows ``row..row+3``, zero past the sequence end."""
        xa = _row(base, row, seq_len, channels, cols)
        xb = _row(base, row + 1, seq_len, channels, cols)
        xc = _row(base, row + 2, seq_len, channels, cols)
        xd = _row(base, row + 3, seq_len, channels, cols)
        return xa, xb, xc, xd

    @triton.jit
    def _load_segs(seg_base, row, seq_len: tl.constexpr):
        return (
            _seg(seg_base, row, seq_len),
            _seg(seg_base, row + 1, seq_len),
            _seg(seg_base, row + 2, seq_len),
            _seg(seg_base, row + 3, seq_len),
        )

    @triton.jit
    def _fwd_rows(
        out_base, t, channels: tl.constexpr, cols, xa, sa, xb, sb, xc, sc, xd, sd, x1, s1, x2, s2, x3, s3,
        w0, w1, w2, w3, dtype: tl.constexpr, exact: tl.constexpr,
    ):  # fmt: skip
        """Stores output rows ``t..t+3``; returns the ring (x and segment ids at t+3, t+2, t+1)."""
        ya = _conv_row(xa, sa, x1, s1, x2, s2, x3, s3, w0, w1, w2, w3, dtype, exact)
        yb = _conv_row(xb, sb, xa, sa, x1, s1, x2, s2, w0, w1, w2, w3, dtype, exact)
        yc = _conv_row(xc, sc, xb, sb, xa, sa, x1, s1, w0, w1, w2, w3, dtype, exact)
        yd = _conv_row(xd, sd, xc, sc, xb, sb, xa, sa, w0, w1, w2, w3, dtype, exact)
        tl.store(out_base + t.to(tl.int64) * channels + cols, ya)
        tl.store(out_base + (t + 1).to(tl.int64) * channels + cols, yb)
        tl.store(out_base + (t + 2).to(tl.int64) * channels + cols, yc)
        tl.store(out_base + (t + 3).to(tl.int64) * channels + cols, yd)
        return xd, sd, xc, sc, xb, sb

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
        rows_per_step: tl.constexpr,
        exact: tl.constexpr,
    ):
        dtype = out_ptr.dtype.element_ty
        cols = tl.program_id(0) * block_c + tl.arange(0, block_c)
        start = tl.program_id(1) * chunk
        batch = tl.program_id(2).to(tl.int64)
        x_base = x_ptr + batch * seq_len * channels
        out_base = out_ptr + batch * seq_len * channels
        seg_base = seg_ptr + batch * seq_len
        w0 = tl.load(w_ptr + cols)
        w1 = tl.load(w_ptr + channels + cols)
        w2 = tl.load(w_ptr + 2 * channels + cols)
        w3 = tl.load(w_ptr + 3 * channels + cols)
        # x at t-1, t-2, t-3 and their segment ids; outside the sequence reads as a zero row
        # whose segment id matches nothing.
        x1 = _row(x_base, start - 1, seq_len, channels, cols)
        x2 = _row(x_base, start - 2, seq_len, channels, cols)
        x3 = _row(x_base, start - 3, seq_len, channels, cols)
        s1 = _seg(seg_base, start - 1, seq_len)
        s2 = _seg(seg_base, start - 2, seq_len)
        s3 = _seg(seg_base, start - 3, seq_len)
        for offset in range(0, chunk, rows_per_step):
            t = start + offset
            # Every row of the step is loaded before any store, so the loads are in flight together.
            xa, xb, xc, xd = _load_rows(x_base, t, channels, cols)
            sa, sb, sc, sd = _load_segs(seg_base, t, seq_len)
            if rows_per_step == 8:
                xe, xf, xg, xh = _load_rows(x_base, t + 4, channels, cols)
                se, sf, sg, sh = _load_segs(seg_base, t + 4, seq_len)
            x1, s1, x2, s2, x3, s3 = _fwd_rows(
                out_base, t, channels, cols, xa, sa, xb, sb, xc, sc, xd, sd, x1, s1, x2, s2, x3, s3,
                w0, w1, w2, w3, dtype, exact,
            )  # fmt: skip
            if rows_per_step == 8:
                x1, s1, x2, s2, x3, s3 = _fwd_rows(
                    out_base, t + 4, channels, cols, xe, se, xf, sf, xg, sg, xh, sh, x1, s1, x2, s2, x3, s3,
                    w0, w1, w2, w3, dtype, exact,
                )  # fmt: skip

    @triton.jit
    def _dx_row(d0, s0, d1, s1, d2, s2, d3, s3, w0, w1, w2, w3, dtype: tl.constexpr, exact: tl.constexpr):
        """``dx[u]`` from ``dy[u..u+3]``: descending lags, then tap 0 (the reference transpose's order).

        The transpose masks after the multiply, the mirror of the forward.
        """
        acc = tl.where(s3 == s0, _mul(w3, d3, exact), 0.0)
        acc = _add(acc, tl.where(s2 == s0, _mul(w2, d2, exact), 0.0), exact)
        acc = _add(acc, tl.where(s1 == s0, _mul(w1, d1, exact), 0.0), exact)
        acc = _add(acc, _mul(w0, d0, exact), exact)
        return acc.to(dtype)

    @triton.jit
    def _masked(x_prev, seg_prev, seg_cur):
        return tl.where(seg_prev == seg_cur, x_prev, 0.0).to(tl.float32)

    @triton.jit
    def _bwd_rows(
        dx_base, r, channels: tl.constexpr, cols, e0, se0, e1, se1, e2, se2, e3, se3, xa, xb, xc, xd,
        xb1, sb1, xb2, sb2, xb3, sb3, d0, sd0, d1, sd1, d2, sd2, dw0, dw1, dw2, dw3,
        w0, w1, w2, w3, dtype: tl.constexpr, exact: tl.constexpr,
    ):  # fmt: skip
        """Stores dx rows ``r..r+3`` and accumulates their dw terms.

        Takes x at ``r..r+3`` and dy at ``r+3..r+6``; the ring carries x at ``r-1..r-3`` and dy at
        ``r..r+2`` with segment ids. Returns the ring advanced by four rows and the accumulators.
        """
        # Segment ids of rows r..r+3 are sd0, sd1, sd2, se0.
        gxa = _dx_row(d0, sd0, d1, sd1, d2, sd2, e0, se0, w0, w1, w2, w3, dtype, exact)
        gxb = _dx_row(d1, sd1, d2, sd2, e0, se0, e1, se1, w0, w1, w2, w3, dtype, exact)
        gxc = _dx_row(d2, sd2, e0, se0, e1, se1, e2, se2, w0, w1, w2, w3, dtype, exact)
        gxd = _dx_row(e0, se0, e1, se1, e2, se2, e3, se3, w0, w1, w2, w3, dtype, exact)
        tl.store(dx_base + r.to(tl.int64) * channels + cols, gxa)
        tl.store(dx_base + (r + 1).to(tl.int64) * channels + cols, gxb)
        tl.store(dx_base + (r + 2).to(tl.int64) * channels + cols, gxc)
        tl.store(dx_base + (r + 3).to(tl.int64) * channels + cols, gxd)
        # dw: fp32 sums of dy[u] times the shifted, masked x, rows u = r..r+3 in order.
        g = d0.to(tl.float32)
        dw0 += g * xa.to(tl.float32)
        dw1 += g * _masked(xb1, sb1, sd0)
        dw2 += g * _masked(xb2, sb2, sd0)
        dw3 += g * _masked(xb3, sb3, sd0)
        g = d1.to(tl.float32)
        dw0 += g * xb.to(tl.float32)
        dw1 += g * _masked(xa, sd0, sd1)
        dw2 += g * _masked(xb1, sb1, sd1)
        dw3 += g * _masked(xb2, sb2, sd1)
        g = d2.to(tl.float32)
        dw0 += g * xc.to(tl.float32)
        dw1 += g * _masked(xb, sd1, sd2)
        dw2 += g * _masked(xa, sd0, sd2)
        dw3 += g * _masked(xb1, sb1, sd2)
        g = e0.to(tl.float32)
        dw0 += g * xd.to(tl.float32)
        dw1 += g * _masked(xc, sd2, se0)
        dw2 += g * _masked(xb, sd1, se0)
        dw3 += g * _masked(xa, sd0, se0)
        return xd, se0, xc, sd2, xb, sd1, e1, se1, e2, se2, e3, se3, dw0, dw1, dw2, dw3

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
        rows_per_step: tl.constexpr,
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
        w0 = tl.load(w_ptr + cols)
        w1 = tl.load(w_ptr + channels + cols)
        w2 = tl.load(w_ptr + 2 * channels + cols)
        w3 = tl.load(w_ptr + 3 * channels + cols)
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
        for offset in range(0, chunk, rows_per_step):
            r = start + offset
            # Rows r..r+3 need x at r..r+3 and dy at r+3..r+6; every row of the step is loaded
            # before any store.
            e0, e1, e2, e3 = _load_rows_masked(dy_base, r + 3, seq_len, channels, cols)
            se0, se1, se2, se3 = _load_segs(seg_base, r + 3, seq_len)
            xa, xb, xc, xd = _load_rows(x_base, r, channels, cols)
            if rows_per_step == 8:
                e4, e5, e6, e7 = _load_rows_masked(dy_base, r + 7, seq_len, channels, cols)
                se4, se5, se6, se7 = _load_segs(seg_base, r + 7, seq_len)
                xe, xf, xg, xh = _load_rows(x_base, r + 4, channels, cols)
            xb1, sb1, xb2, sb2, xb3, sb3, d0, sd0, d1, sd1, d2, sd2, dw0, dw1, dw2, dw3 = _bwd_rows(
                dx_base, r, channels, cols, e0, se0, e1, se1, e2, se2, e3, se3, xa, xb, xc, xd,
                xb1, sb1, xb2, sb2, xb3, sb3, d0, sd0, d1, sd1, d2, sd2, dw0, dw1, dw2, dw3,
                w0, w1, w2, w3, dtype, exact,
            )  # fmt: skip
            if rows_per_step == 8:
                xb1, sb1, xb2, sb2, xb3, sb3, d0, sd0, d1, sd1, d2, sd2, dw0, dw1, dw2, dw3 = _bwd_rows(
                    dx_base, r + 4, channels, cols, e4, se4, e5, se5, e6, se6, e7, se7, xe, xf, xg, xh,
                    xb1, sb1, xb2, sb2, xb3, sb3, d0, sd0, d1, sd1, d2, sd2, dw0, dw1, dw2, dw3,
                    w0, w1, w2, w3, dtype, exact,
                )  # fmt: skip
        partial = dw_ptr + (batch * (seq_len // chunk) + chunk_id) * 4 * channels
        tl.store(partial + cols, dw0)
        tl.store(partial + channels + cols, dw1)
        tl.store(partial + 2 * channels + cols, dw2)
        tl.store(partial + 3 * channels + cols, dw3)

else:
    _short_conv_fwd_kernel = None
    _short_conv_bwd_kernel = None


def _launch_kwargs(x: jax.Array, tiles: TritonShortConvTiles, exact: bool) -> dict:
    batch, seq_len, channels = x.shape
    block_c = tiles.channel_block(channels)
    assert block_c is not None and seq_len % tiles.chunk == 0, (x.shape, tiles)
    assert tiles.rows_per_step in (4, 8) and tiles.chunk % tiles.rows_per_step == 0, tiles
    return dict(
        grid=(channels // block_c, seq_len // tiles.chunk, batch),
        num_warps=tiles.num_warps,
        num_stages=tiles.num_stages,
        seq_len=seq_len,
        channels=channels,
        chunk=tiles.chunk,
        block_c=block_c,
        rows_per_step=tiles.rows_per_step,
        exact=exact,
    )


def short_conv_triton_fwd_local(
    weight: Float[Array, "W C"],
    x: Float[Array, "B S C"],
    segment_ids: Int[Array, "B S"],
    *,
    exact_reference_rounding: bool,
    tiles: TritonShortConvTiles = FORWARD_TILES,
) -> Float[Array, "B S C"]:
    """Shard-local forward. Callers must have already entered a ``shard_map``."""
    return jt.triton_call(
        x,
        segment_ids.astype(jnp.int32),
        weight,
        kernel=_short_conv_fwd_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        **_launch_kwargs(x, tiles, exact_reference_rounding),
    )


def short_conv_triton_bwd_local(
    weight: Float[Array, "W C"],
    x: Float[Array, "B S C"],
    segment_ids: Int[Array, "B S"],
    dy: Float[Array, "B S C"],
    *,
    exact_reference_rounding: bool,
    tiles: TritonShortConvTiles = BACKWARD_TILES,
) -> tuple[Float[Array, "B S C"], Float[Array, "P W C"]]:
    """Shard-local backward. Returns ``(dx, dw_partials)`` with ``dw_partials`` ``[B * S / chunk, W, C]``."""
    batch, seq_len, channels = x.shape
    dx, dw_partials = jt.triton_call(
        x,
        segment_ids.astype(jnp.int32),
        dy,
        weight,
        kernel=_short_conv_bwd_kernel,
        out_shape=(
            jax.ShapeDtypeStruct(x.shape, dy.dtype),
            jax.ShapeDtypeStruct((batch * (seq_len // tiles.chunk), KERNEL_SIZE, channels), jnp.float32),
        ),
        **_launch_kwargs(x, tiles, exact_reference_rounding),
    )
    return dx, dw_partials


__all__ = [
    "BACKWARD_TILES",
    "FORWARD_TILES",
    "OOB_SEGMENT",
    "SEQUENCE_MULTIPLE",
    "TritonShortConvTiles",
    "short_conv_triton_bwd_local",
    "short_conv_triton_fwd_local",
    "triton_short_conv_available",
    "triton_short_conv_shapes_supported",
]
