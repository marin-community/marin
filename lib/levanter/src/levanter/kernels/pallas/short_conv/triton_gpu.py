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
backward reads ``x`` and ``dy`` and writes ``dx`` and one fp32 ``dw`` partial per chunk. A step
loads ``rows_per_step`` rows of every input before it stores any, so those loads are in flight
together.

With every tap masked, the backward ran close to the instruction-issue limit. A step whose rows,
halo included, all lie in one document now skips the segment masks: every mask is true there, so
the result is the same, and the step issues about 40% fewer instructions. ``dw`` accumulates with
explicit fused multiply-adds.

Numerics match the Pallas kernels. With ``exact`` set (bfloat16 only), every multiply and add
rounds to bfloat16 in the reference's order: ascending lags forward, descending lags then tap 0
for ``dx``. The forward and ``dx`` are then bit-identical to ``short_conv_reference``. On SM90 and
newer the bf16 ops are packed PTX (``mul.rn.bf16x2``, ``add.rn.bf16x2``); older GPUs, which lack
them, take an fp32 op and a bf16 cast, which rounds the same way. Without ``exact``, taps
accumulate in fp32. ``dw`` accumulates in fp32 per chunk; the caller sums the partials.

The kernels handle ``kernel_size == 4`` only, the hero's width.
"""

import dataclasses
import functools
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

# PTX's packed bf16 add and multiply (``add.rn.bf16x2``, ``mul.rn.bf16x2``) need SM90 or newer.
_PACKED_BF16_MIN_COMPUTE_CAPABILITY = 9.0


@dataclasses.dataclass(frozen=True)
class TritonShortConvTiles:
    """Launch shape of one direction's kernel.

    Attributes:
      chunk: rows each program walks; the sequence length must divide by it. The halo re-read
        is ``(KERNEL_SIZE - 1) / chunk`` of a pass.
      max_channel_block: channels per program, halved until it divides the channel count.
      num_warps: warps per program.
      rows_per_step: rows loaded together before any is stored; divides ``chunk``.
    """

    chunk: int
    max_channel_block: int
    num_warps: int
    rows_per_step: int

    def channel_block(self, channels: int) -> int | None:
        block = self.max_channel_block
        while block >= 64:
            if channels % block == 0:
                return block
            block //= 2
        return None


# Best measured on one GB200 at the hero shapes ([16, 4096, C] bf16, 9 documents per sequence), kernel
# microseconds at C = 6144 / 1536:
#
#   direction  chunk, channels, warps, rows   us
#   forward    16, 512, 4, 8                  226.3 / 59.0
#   forward    16, 256, 2, 8                  226.8 / 59.2
#   forward    32, 256, 2, 16                 230.6 / 59.3
#   forward    8, 512, 4, 8                   239.9 / 61.1
#   backward   128, 512, 4, 4                 384.1 / 104.7
#   backward   128, 512, 8, 8                 382.7 / 104.5  (401.2 / 112.9 with 129 documents)
#   backward   128, 256, 4, 8                 400.4 / 108.2
#   backward   128, 512, 4, 2                 457.7 / 134.5
#
# Both directions take four channels per thread. A backward row holds x, dy and their fp32 dw terms,
# so the backward loads four rows per step where the forward loads eight; eight rows cut its
# occupancy and take 403 us at C=6144.
FORWARD_TILES = TritonShortConvTiles(chunk=16, max_channel_block=512, num_warps=4, rows_per_step=8)
BACKWARD_TILES = TritonShortConvTiles(chunk=128, max_channel_block=512, num_warps=4, rows_per_step=4)
#: The local sequence length must be a multiple of this.
SEQUENCE_MULTIPLE = math.lcm(FORWARD_TILES.chunk, BACKWARD_TILES.chunk)


def triton_short_conv_available() -> bool:
    return jt is not None and triton is not None and jax.default_backend() == "gpu"


@functools.cache
def _packed_bf16_arithmetic() -> bool:
    """Whether this process's GPU runs packed bf16 adds and multiplies."""
    device = jax.devices()[0]
    return device.platform == "gpu" and float(device.compute_capability) >= _PACKED_BF16_MIN_COMPUTE_CAPABILITY


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
    _HALO = tl.constexpr(KERNEL_SIZE - 1)
    _WIDTH = tl.constexpr(KERNEL_SIZE)
    _OOB = tl.constexpr(OOB_SEGMENT)

    @triton.jit
    def _mul(a, b, exact: tl.constexpr, packed: tl.constexpr):
        """``a * b``. Exact: one bf16 rounding per multiply, as the reference rounds each op."""
        if exact and packed:
            # Two elements per instruction, rounded as the fp32 multiply and bf16 cast below round.
            product = tl.inline_asm_elementwise(
                "mul.rn.bf16x2 $0, $1, $2;", "=r,r,r", [a, b], dtype=tl.bfloat16, is_pure=True, pack=2
            )
        elif exact:
            product = (a.to(tl.float32) * b.to(tl.float32)).to(tl.bfloat16)
        else:
            product = a.to(tl.float32) * b.to(tl.float32)
        return product

    @triton.jit
    def _add(a, b, exact: tl.constexpr, packed: tl.constexpr):
        """``a + b``. Exact: one bf16 rounding per add. For bf16 operands this equals the
        reference's fp32 add then bf16 rounding: the fp32 sum is exact whenever the rounding
        to bf16 can depend on it."""
        if exact and packed:
            total = tl.inline_asm_elementwise(
                "add.rn.bf16x2 $0, $1, $2;", "=r,r,r", [a, b], dtype=tl.bfloat16, is_pure=True, pack=2
            )
        elif exact:
            total = (a.to(tl.float32) + b.to(tl.float32)).to(tl.bfloat16)
        else:
            total = a + b
        return total

    @triton.jit
    def _keep(value, seg_other, seg_cur, masked: tl.constexpr):
        """``value``, zeroed when its tap crosses a document boundary."""
        if masked:
            value = tl.where(seg_other == seg_cur, value, 0.0)
        return value

    @triton.jit
    def _one_segment(segs, count: tl.constexpr):
        """Whether the first ``count`` segment ids are all equal, so no tap among them crosses a document."""
        same = segs[0] == segs[1]
        for i in tl.static_range(2, count):
            same = same & (segs[i] == segs[0])
        return same

    @triton.jit
    def _row(base_ptr, row, seq_len: tl.constexpr, channels: tl.constexpr, cols):
        """Row ``row`` of a ``[S, C]`` slab at ``cols``, zero outside ``[0, S)``."""
        inside = (row >= 0) & (row < seq_len)
        return tl.load(base_ptr + row.to(tl.int64) * channels + cols, mask=inside, other=0.0)

    @triton.jit
    def _seg(seg_ptr, row, seq_len: tl.constexpr):
        inside = (row >= 0) & (row < seq_len)
        return tl.load(seg_ptr + row, mask=inside, other=_OOB)

    @triton.jit
    def _conv_row(
        xs, segs, cur: tl.constexpr, weights, masked: tl.constexpr, exact: tl.constexpr, packed: tl.constexpr
    ):
        """The output row at ``xs[cur]`` from ``xs[cur - lag]``, the input row ``lag`` behind it.

        Ascending lags, the reference's order. A tap across a document boundary reads zero
        before the multiply, as the reference masks the shifted input.
        """
        acc = _mul(weights[0], xs[cur], exact, packed)
        for lag in tl.static_range(1, _WIDTH):
            tap = _keep(xs[cur - lag], segs[cur - lag], segs[cur], masked)
            acc = _add(acc, _mul(weights[lag], tap, exact, packed), exact, packed)
        return acc

    @triton.jit
    def _fwd_rows(
        out_base, t, channels: tl.constexpr, cols, xs, segs, weights, rows: tl.constexpr, masked: tl.constexpr,
        dtype: tl.constexpr, exact: tl.constexpr, packed: tl.constexpr,
    ):  # fmt: skip
        """Stores output rows ``t..t+rows-1``. ``xs[i]`` and ``segs[i]`` are the input at row ``t - 3 + i``."""
        for j in tl.static_range(rows):
            y = _conv_row(xs, segs, j + _HALO, weights, masked, exact, packed)
            tl.store(out_base + (t + j).to(tl.int64) * channels + cols, y.to(dtype))

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
        packed: tl.constexpr,
    ):
        dtype = out_ptr.dtype.element_ty
        cols = tl.program_id(0) * block_c + tl.arange(0, block_c)
        start = tl.program_id(1) * chunk
        batch = tl.program_id(2).to(tl.int64)
        x_base = x_ptr + batch * seq_len * channels
        out_base = out_ptr + batch * seq_len * channels
        seg_base = seg_ptr + batch * seq_len
        weights = (
            tl.load(w_ptr + cols),
            tl.load(w_ptr + channels + cols),
            tl.load(w_ptr + 2 * channels + cols),
            tl.load(w_ptr + 3 * channels + cols),
        )
        # x at t-3, t-2, t-1 and their segment ids, oldest first; outside the sequence reads as a
        # zero row whose segment id matches nothing.
        history = (
            _row(x_base, start - 3, seq_len, channels, cols),
            _row(x_base, start - 2, seq_len, channels, cols),
            _row(x_base, start - 1, seq_len, channels, cols),
        )
        history_segs = (
            _seg(seg_base, start - 3, seq_len),
            _seg(seg_base, start - 2, seq_len),
            _seg(seg_base, start - 1, seq_len),
        )
        for offset in range(0, chunk, rows_per_step):
            t = start + offset
            # xs[i] and segs[i] are row t - 3 + i. Every row of the step is loaded before any store.
            xs = history
            segs = history_segs
            for j in tl.static_range(rows_per_step):
                xs = xs + (tl.load(x_base + (t + j).to(tl.int64) * channels + cols),)
                segs = segs + (tl.load(seg_base + t + j),)
            if _one_segment(segs, rows_per_step + _HALO):
                _fwd_rows(out_base, t, channels, cols, xs, segs, weights, rows_per_step, False, dtype, exact, packed)
            else:
                _fwd_rows(out_base, t, channels, cols, xs, segs, weights, rows_per_step, True, dtype, exact, packed)
            history = (xs[rows_per_step], xs[rows_per_step + 1], xs[rows_per_step + 2])
            history_segs = (segs[rows_per_step], segs[rows_per_step + 1], segs[rows_per_step + 2])

    @triton.jit
    def _dx_row(dys, segs, j: tl.constexpr, weights, masked: tl.constexpr, exact: tl.constexpr, packed: tl.constexpr):
        """``dx`` at row ``u``, where ``dys[j + lag]`` is dy at ``u + lag`` and ``segs[j + 3 + lag]`` its
        segment id.

        Descending lags, then tap 0: the reference transpose's order. The transpose masks after
        the multiply, the mirror of the forward.
        """
        cur: tl.constexpr = j + _HALO
        acc = _keep(_mul(weights[_HALO], dys[j + _HALO], exact, packed), segs[cur + _HALO], segs[cur], masked)
        for lag in tl.static_range(_HALO - 1, 0, -1):
            term = _keep(_mul(weights[lag], dys[j + lag], exact, packed), segs[cur + lag], segs[cur], masked)
            acc = _add(acc, term, exact, packed)
        return _add(acc, _mul(weights[0], dys[j], exact, packed), exact, packed)

    @triton.jit
    def _dw_add(dw, dy, xs, segs, cur: tl.constexpr, masked: tl.constexpr):
        """``dw[lag] += dy * x[u - lag]`` in fp32 for the row ``u`` at ``xs[cur]``."""
        g = dy.to(tl.float32)
        out = (tl.fma(g, xs[cur].to(tl.float32), dw[0]),)
        for lag in tl.static_range(1, _WIDTH):
            shifted = _keep(xs[cur - lag].to(tl.float32), segs[cur - lag], segs[cur], masked)
            out = out + (tl.fma(g, shifted, dw[lag]),)
        return out

    @triton.jit
    def _bwd_rows(
        dx_base, r, channels: tl.constexpr, cols, xs, dys, segs, dw, weights, rows: tl.constexpr,
        masked: tl.constexpr, dtype: tl.constexpr, exact: tl.constexpr, packed: tl.constexpr,
    ):  # fmt: skip
        """Stores dx rows ``r..r+rows-1`` and returns ``dw`` with their terms added.

        ``xs[i]`` is x at row ``r - 3 + i``, ``dys[i]`` dy at row ``r + i`` and ``segs[i]`` the
        segment id of row ``r - 3 + i``.
        """
        for j in tl.static_range(rows):
            dx = _dx_row(dys, segs, j, weights, masked, exact, packed)
            tl.store(dx_base + (r + j).to(tl.int64) * channels + cols, dx.to(dtype))
        # dw: fp32 sums of dy[u] times the shifted, masked x, rows u = r..r+rows-1 in order.
        for j in tl.static_range(rows):
            dw = _dw_add(dw, dys[j], xs, segs, j + _HALO, masked)
        return dw

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
        packed: tl.constexpr,
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
        weights = (
            tl.load(w_ptr + cols),
            tl.load(w_ptr + channels + cols),
            tl.load(w_ptr + 2 * channels + cols),
            tl.load(w_ptr + 3 * channels + cols),
        )
        # Row r's dx needs dy at r..r+3; its dw terms need x at r..r-3. Carry x behind (rows
        # r-3..r-1) and dy ahead (rows r..r+2), each with its segment ids.
        behind = (
            _row(x_base, start - 3, seq_len, channels, cols),
            _row(x_base, start - 2, seq_len, channels, cols),
            _row(x_base, start - 1, seq_len, channels, cols),
        )
        ahead = (
            _row(dy_base, start, seq_len, channels, cols),
            _row(dy_base, start + 1, seq_len, channels, cols),
            _row(dy_base, start + 2, seq_len, channels, cols),
        )
        behind_segs = (
            _seg(seg_base, start - 3, seq_len),
            _seg(seg_base, start - 2, seq_len),
            _seg(seg_base, start - 1, seq_len),
        )
        ahead_segs = (
            _seg(seg_base, start, seq_len),
            _seg(seg_base, start + 1, seq_len),
            _seg(seg_base, start + 2, seq_len),
        )
        zero = tl.zeros([block_c], dtype=tl.float32)
        dw = (zero, zero, zero, zero)
        for offset in range(0, chunk, rows_per_step):
            r = start + offset
            # Rows r..r+rows-1 need x at r-3..r+rows-1 and dy at r..r+rows+2, past the end of the
            # sequence in its last step. Every row of the step is loaded before any store.
            xs = behind
            dys = ahead
            segs = behind_segs + ahead_segs
            for j in tl.static_range(rows_per_step):
                dys = dys + (_row(dy_base, r + _HALO + j, seq_len, channels, cols),)
                segs = segs + (_seg(seg_base, r + _HALO + j, seq_len),)
                xs = xs + (tl.load(x_base + (r + j).to(tl.int64) * channels + cols),)
            if _one_segment(segs, rows_per_step + 2 * _HALO):
                dw = _bwd_rows(
                    dx_base, r, channels, cols, xs, dys, segs, dw, weights, rows_per_step, False, dtype, exact, packed
                )
            else:
                dw = _bwd_rows(
                    dx_base, r, channels, cols, xs, dys, segs, dw, weights, rows_per_step, True, dtype, exact, packed
                )
            behind = (xs[rows_per_step], xs[rows_per_step + 1], xs[rows_per_step + 2])
            behind_segs = (segs[rows_per_step], segs[rows_per_step + 1], segs[rows_per_step + 2])
            ahead = (dys[rows_per_step], dys[rows_per_step + 1], dys[rows_per_step + 2])
            ahead_segs = (segs[rows_per_step + 3], segs[rows_per_step + 4], segs[rows_per_step + 5])
        partial = dw_ptr + (batch * (seq_len // chunk) + chunk_id) * _WIDTH * channels
        for lag in tl.static_range(_WIDTH):
            tl.store(partial + lag * channels + cols, dw[lag])

else:
    _short_conv_fwd_kernel = None
    _short_conv_bwd_kernel = None


def _launch_kwargs(x: jax.Array, tiles: TritonShortConvTiles, exact: bool) -> dict:
    batch, seq_len, channels = x.shape
    block_c = tiles.channel_block(channels)
    assert block_c is not None and seq_len % tiles.chunk == 0, (x.shape, tiles)
    assert tiles.chunk % tiles.rows_per_step == 0, tiles
    return dict(
        grid=(channels // block_c, seq_len // tiles.chunk, batch),
        num_warps=tiles.num_warps,
        # Triton's pipeliner leaves these loads alone: 1 and 2 stages compile to the same program.
        num_stages=1,
        # Every fused multiply-add is explicit (tl.fma for dw). Left to itself, LLVM folds the bf16
        # fallback's fp32 round trips into bf16 arithmetic and contracts a multiply and the following
        # add into one bf16 FMA, which rounds once where the reference rounds twice.
        enable_fp_fusion=False,
        seq_len=seq_len,
        channels=channels,
        chunk=tiles.chunk,
        block_c=block_c,
        rows_per_step=tiles.rows_per_step,
        exact=exact,
        packed=exact and _packed_bf16_arithmetic(),
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
