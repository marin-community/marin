# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Fused depthwise causal short convolution (SConv) as a Pallas GPU kernel.

Why this exists
---------------
The op is depthwise with ``kernel_size=4``: 8 FLOP per element against a 4-byte floor of
traffic, an arithmetic intensity of **2.0 FLOP/byte** against a GB200 ridge of ~312. It is
bandwidth-bound by ~156x, so the *only* lever is how many times the tensor crosses HBM.
The floor is 2 passes forward (read x, write y) and 3 passes backward (read x, read dy,
write dx).

The pad-and-shift reference (``reference.py``) does not sit at that floor, but **not for
the reason you would guess, and not for the reason CPU HLO suggests.** Compiled for CPU,
its VJP materialises four full-size fp32 copies of ``dy * mask * shift(x)`` and reduces
each with a separate ``reduce-window``, needing 4.84 GB of scratch at the hero per-layer
shape. **None of that happens on GPU.** XLA:GPU fuses those reductions into their
producers: the compiled GPU HLO has one fusion and two full-size buffers in the forward,
five fusions and three full-size buffers in the VJP, and 13 MB of temp. Do not repeat the
CPU-HLO story; it is a backend artifact.

The real cost is **repeated traversal**, measured on one GB200 at the hero per-layer shape
against a read-one-write-one stream calibrated on the same tensor:

=========================  ===============  ===============
                           forward          backward
=========================  ===============  ===============
floor                      2.0 passes       3.0 passes
pad-and-shift reference    4.5 passes       14.3 passes
this kernel                2.0 passes        3.6 passes
=========================  ===============  ===============

Each tap of the reference is a separate offset read of the whole tensor that the cache does
not fully absorb, and its VJP additionally launches five fusions that each re-read their
inputs. This kernel streams instead, the way ``Dao-AILab/causal-conv1d`` does: a program owns
one channel block of one sequence block and walks its rows in order, carrying the previous
``W - 1`` rows of ``x`` (backward: also the next ``W - 1`` rows of ``dy``) and their segment
ids in registers as ``fori_loop`` values, so each row crosses HBM once plus a halo of
``W - 1`` rows per block. Pallas Triton cannot slice a register tile, so the kernel never
builds a multi-row tile: every load and store is one ``[c_block]`` row, ``rows_per_step`` of
them loaded together before any is stored. At C=6144 the forward takes 0.230 ms and the
backward 0.404 ms; the previous version of this kernel, which re-read a shifted
``[s_block, c_block]`` window per tap (3.7 and 9.2 passes), took 0.420 ms and 1.024 ms.

The backward runs close to the instruction-issue limit, so its code matters as much as its
traffic. The bf16 multiplies and adds are packed PTX (``_bf16x2``); ``dw`` accumulates with an
explicit fused multiply-add, which XLA's Triton pipeline otherwise never forms; program ids
are clamped to the grid so the int32 offsets provably do not overflow and each row's offset
folds into a load immediate (``_bounded``); and a step whose rows all sit in one document
skips the segment masks (``_branch_on_segments``).

``dw`` never touches HBM as a full-size tensor. It is accumulated in fp32 registers per
program and emitted as a ``[batch * num_s_blocks, W, C]`` partial that a cheap outer
``sum(0)`` folds down -- the deterministic reduction that FLA's Triton conv uses, rather
than ``atomicAdd``. It costs ``2 * W * 4 / (s_block_size * itemsize)`` of a pass (12.5% at the
default backward block) and is bit-reproducible run to run.

Backend
-------
**Pallas Triton, not Mosaic GPU.** The repo's kernel skill prefers Mosaic for new GPU
kernels; measured on GB200/SM100 with JAX 0.11, Mosaic's layout inference fails on this
kernel and on a trivial ``o = w * x`` body alike ("Layout inference failed to find a
solution"). Triton lowers both. Revisit when Mosaic's layout inference improves.

Halo handling
-------------
The ``W - 1`` rows before a sequence (and, backward, after it) are read with masked
single-row loads that return a zero row and ``OOB_SEGMENT`` (``_edge_row``). A masked load
of a multi-row window would be *wrong* under Pallas's own interpreter, whose clamp silently
misaligns the window, but a single row is kept or discarded whole, so the CPU interpreter
and the GPU agree.

Numerics
--------
``exact_reference_rounding=True`` (the default) reproduces the reference's rounding
*operation for operation*: every multiply and every accumulate rounds to the activation
dtype, in the reference's association order. That order is not cosmetic -- the forward is
left-nested over ascending lags, and JAX transposes it by walking the jaxpr in reverse, so
``dx`` accumulates over *descending* lags. Matching both makes the forward and ``dx``
bit-identical to the reference; under sequence sharding the boundary tokens' ``dx`` is the
sum of two separately rounded partials (see ``short_conv``), so only the forward stays
bitwise there. On the GPU the bf16 ops are ``mul.rn.bf16x2`` and ``add.rn.bf16x2``, which
round exactly as the reference's f32 op followed by a bf16 cast; the interpreter, and other
dtypes, take that f32 round trip literally. ``dw`` is a reduction over 65,536 tokens whose
association order XLA does not define, so it agrees to fp32 reassociation error and is
validated against a float64 oracle instead. Setting the flag to ``False`` keeps a single
fp32 accumulator across taps: strictly more accurate, no longer bit-comparable.
"""

import contextlib
import functools
import math

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jaxtyping import Array, Float, Int

from levanter.kernels.pallas.cost_estimate_utils import with_io_bytes_accessed

from .config import OOB_SEGMENT, ShortConvBlockSizes, ShortConvTiles
from .reference import short_conv_reference

try:  # pragma: no cover - import guard, exercised only by environment
    from jax.experimental.pallas import triton as pltriton

    _HAS_PALLAS_TRITON = True
except (ImportError, ModuleNotFoundError):  # pragma: no cover
    pltriton = None  # type: ignore[assignment]
    _HAS_PALLAS_TRITON = False

_FORCE_INTERPRET = False


@contextlib.contextmanager
def interpret_mode():
    """Run the kernels through Pallas's reference interpreter.

    This is the only way these kernels get CPU coverage: a Pallas GPU kernel cannot execute
    on CPU, but the interpreter executes the *kernel body* -- grid, block specs, the row
    loop and its carried window, edge masking and all -- with plain XLA ops. It exercises
    the real algorithm, not a paraphrase. It says nothing about whether the kernel *lowers*
    on a given GPU architecture; that needs a GPU.
    """
    global _FORCE_INTERPRET
    previous = _FORCE_INTERPRET
    _FORCE_INTERPRET = True
    try:
        yield
    finally:
        _FORCE_INTERPRET = previous


def pallas_short_conv_available() -> bool:
    """True when the Pallas Triton backend imported and we are on a GPU or interpreting.

    The kernels' masked edge loads are Pallas Triton primitives even under the interpreter.
    """
    return _HAS_PALLAS_TRITON and (_FORCE_INTERPRET or jax.default_backend() == "gpu")


def _is_pow2(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _channel_block(channels: int, c_block_size: int) -> int:
    """The widest power-of-two channel block no wider than ``c_block_size`` that divides ``channels``."""
    block = c_block_size
    while channels % block:
        block //= 2
    return block


def _tiles_supported(name: str, seq_len: int, tiles: ShortConvTiles) -> str | None:
    bs, rows = tiles.s_block_size, tiles.rows_per_step
    if seq_len % bs:
        return f"seq_len {seq_len} not divisible by {name} s_block_size {bs}"
    if not _is_pow2(bs):
        return f"{name} s_block_size {bs} must be a power of 2"
    # Pallas Triton requires power-of-2 tile shapes; every read in the kernel is one row.
    if not _is_pow2(tiles.c_block_size):
        return f"{name} c_block_size {tiles.c_block_size} must be a power of 2 (Pallas Triton tile constraint)"
    if not _is_pow2(rows) or rows > bs:
        return f"{name} rows_per_step {rows} must be a power of 2 no larger than s_block_size {bs}"
    return None


def short_conv_shapes_supported(
    weight_shape: tuple[int, ...],
    x_shape: tuple[int, ...],
    block_sizes: ShortConvBlockSizes,
) -> str | None:
    """Returns None when the kernels can run these shapes, else a human-readable reason."""
    if len(x_shape) != 3 or len(weight_shape) != 2:
        return f"expected weight [W, C] and x [B, S, C], got {weight_shape} and {x_shape}"
    width, weight_channels = weight_shape
    _, seq_len, channels = x_shape
    if weight_channels != channels:
        return f"weight channel dim {weight_channels} != x channel dim {channels}"
    if width < 1:
        return f"kernel_size must be >= 1, got {width}"
    return _tiles_supported("forward", seq_len, block_sizes.forward) or _tiles_supported(
        "backward", seq_len, block_sizes.backward
    )


def _bf16x2(op: str, a: jax.Array, b: jax.Array) -> jax.Array:
    """``a op b`` on bfloat16 as packed PTX, two elements per instruction. It rounds exactly
    as the f32 op and bf16 cast of ``_mul_round``/``_add_round``, without the conversions."""
    [out] = pltriton.elementwise_inline_asm(
        f"{op}.rn.bf16x2 $0, $1, $2;",
        args=[a, b],
        constraints="=r,r,r",
        pack=2,
        result_shape_dtypes=[jax.ShapeDtypeStruct(a.shape, jnp.bfloat16)],
    )
    return out


def _fma(a: jax.Array, b: jax.Array, c: jax.Array) -> jax.Array:
    """``a * b + c`` in fp32, fused on the GPU, where XLA's Triton pipeline never contracts a
    multiply and an add."""
    if _FORCE_INTERPRET:
        return a * b + c
    [out] = pltriton.elementwise_inline_asm(
        "fma.rn.f32 $0, $1, $2, $3;",
        args=[a, b, c],
        constraints="=f,f,f,f",
        pack=1,
        result_shape_dtypes=[jax.ShapeDtypeStruct(a.shape, jnp.float32)],
    )
    return out


# PTX's packed bf16 add and multiply (``add.rn.bf16x2``, ``mul.rn.bf16x2``) need SM90 or newer.
_PACKED_BF16_MIN_COMPUTE_CAPABILITY = 9.0


@functools.cache
def _packed_bf16_arithmetic() -> bool:
    """Whether this process's GPU runs packed bf16 adds and multiplies."""
    device = jax.devices()[0]
    return device.platform == "gpu" and float(device.compute_capability) >= _PACKED_BF16_MIN_COMPUTE_CAPABILITY


def _use_asm(dtype, exact: bool) -> bool:
    """Whether to round bf16 with packed PTX. Older GPUs and the interpreter take the fp32 op and
    bf16 cast, which rounds the same way."""
    return exact and not _FORCE_INTERPRET and jnp.dtype(dtype) == jnp.dtype(jnp.bfloat16) and _packed_bf16_arithmetic()


def _mul_round(weight_row: jax.Array, row: jax.Array, dtype, exact: bool) -> jax.Array:
    """``weight_row * row`` with the reference's rounding."""
    if _use_asm(dtype, exact):
        return _bf16x2("mul", weight_row, row)
    product = weight_row.astype(jnp.float32) * row.astype(jnp.float32)
    return product.astype(dtype) if exact else product


def _add_round(acc, term, dtype, exact: bool) -> jax.Array:
    if acc is None:
        return term
    if _use_asm(dtype, exact):
        return _bf16x2("add", acc, term)
    total = acc.astype(jnp.float32) + term.astype(jnp.float32)
    return total.astype(dtype) if exact else total


def _keep(seg_shifted: jax.Array, seg_cur: jax.Array, row: jax.Array, masked: bool) -> jax.Array:
    if not masked:
        return row
    return jnp.where(seg_shifted == seg_cur, row, jnp.zeros_like(row))


def _one_segment(segs) -> jax.Array:
    """Whether all of ``segs`` are equal, so no tap of the rows they cover crosses a document."""
    first, *rest = segs
    same = jnp.bool_(True)
    for seg in rest:
        same = same & (seg == first)
    return same


def _edge_row(vals_ref, segs_ref, row, seq_len: int):
    """Row ``row`` of a whole-sequence view and its segment id; a zero row with ``OOB_SEGMENT``
    outside ``[0, seq_len)``.

    One row is masked as a whole, so the interpreter's clamped read of an out-of-range row is
    discarded rather than misaligned, and the interpreter and the GPU agree.
    """
    inside = (row >= 0) & (row < seq_len)
    # An array index, not a literal 0: the interpreter's masked-load rule needs a shape on it.
    first = jnp.zeros((), jnp.int32)
    vals = pltriton.load(vals_ref.at[first, row, :], mask=inside, other=0)
    return vals, pltriton.load(segs_ref.at[first, row], mask=inside, other=OOB_SEGMENT)


def _bounded(index: jax.Array, size: int) -> jax.Array:
    """``index``, which the grid already keeps below ``size``, clamped so the compiler knows it.

    Pallas computes element offsets in int32. Without a bound on the program ids, LLVM cannot
    rule out overflow, so it materializes a 64-bit address for every row a program reads;
    bounded, it folds each row's offset into the load's immediate.
    """
    return jnp.minimum(index, size - 1)


def _chunk_start(chunk: int):
    return _bounded(pl.program_id(1), pl.num_programs(1)) * chunk


def _branch_on_segments(segs, compute, *operands):
    """``compute(masked, *operands)``, skipping the masks when every row of the step is in one document.

    The unmasked branch computes exactly what the masked one would, since every mask is
    true there; it only saves the selects, about a third of the backward's instructions.
    """
    return jax.lax.cond(
        _one_segment(segs), functools.partial(compute, False), functools.partial(compute, True), *operands
    )


# --------------------------------------------------------------------------------------
# Forward
# --------------------------------------------------------------------------------------


def _conv_row(window, weights, dtype, exact: bool, masked: bool) -> jax.Array:
    """One output row from ``window[lag]``, the ``(x, segment id)`` of row ``t - lag``."""
    x_cur, seg_cur = window[0]
    # Ascending lags, left-nested, rounding after every op: the reference's exact order.
    acc = _mul_round(weights[0], x_cur, dtype, exact)
    for lag in range(1, len(window)):
        shifted, seg_shifted = window[lag]
        term = _mul_round(weights[lag], _keep(seg_shifted, seg_cur, shifted, masked), dtype, exact)
        acc = _add_round(acc, term, dtype, exact)
    return acc.astype(dtype)


def _fwd_kernel(x_ref, seg_ref, w_ref, out_ref, *, kernel_size: int, rows: int, exact: bool):
    seq_len, chunk = x_ref.shape[1], out_ref.shape[1]
    start = _chunk_start(chunk)
    weights = [w_ref[lag] for lag in range(kernel_size)]
    # ring[k] is row t-1-k of x with its segment id.
    ring = [_edge_row(x_ref, seg_ref, start - 1 - k, seq_len) for k in range(kernel_size - 1)]

    def step(i, ring):
        local = i * rows
        loaded = [(x_ref[0, start + local + j, :], seg_ref[0, start + local + j]) for j in range(rows)]
        segs = [seg for _, seg in (*ring, *loaded)]
        windows = []
        for row in loaded:
            windows.append([row, *ring])
            ring = windows[-1][: kernel_size - 1]

        def store(masked):
            for j, window in enumerate(windows):
                out_ref[0, local + j, :] = _conv_row(window, weights, out_ref.dtype, exact, masked)

        _branch_on_segments(segs, store)
        return ring

    jax.lax.fori_loop(0, chunk // rows, step, ring)


# --------------------------------------------------------------------------------------
# Backward
# --------------------------------------------------------------------------------------


def _dx_row(window, weights, dtype, exact: bool, masked: bool) -> jax.Array:
    """``dx[t] = sum_lag w[lag] * [seg[t] == seg[t+lag]] * dy[t+lag]`` from ``window[lag]``, the
    ``(dy, segment id)`` of row ``t + lag``.

    Descending lags then tap 0, because that is the order JAX's transpose of the forward
    produces and matching it is what makes ``dx`` bit-identical rather than merely close.
    """
    dy_cur, seg_cur = window[0]
    acc = None
    for lag in range(len(window) - 1, 0, -1):
        dy_ahead, seg_ahead = window[lag]
        term = _mul_round(weights[lag], dy_ahead, dtype, exact)
        acc = _add_round(acc, _keep(seg_ahead, seg_cur, term, masked), dtype, exact)
    return _add_round(acc, _mul_round(weights[0], dy_cur, dtype, exact), dtype, exact).astype(dtype)


def _dw_accumulate(dw, dy_cur: jax.Array, window, masked: bool):
    """``dw[lag] += dy[t] * [seg[t-lag] == seg[t]] * x[t-lag]`` in fp32, from ``window[lag]``, the
    ``(x, segment id)`` of row ``t - lag``."""
    seg_cur = window[0][1]
    dy_f32 = dy_cur.astype(jnp.float32)
    out = []
    for lag, (acc, (shifted, seg_shifted)) in enumerate(zip(dw, window)):
        if lag:
            shifted = _keep(seg_shifted, seg_cur, shifted, masked)
        out.append(_fma(dy_f32, shifted.astype(jnp.float32), acc))
    return out


def _bwd_kernel(x_ref, seg_ref, dy_ref, w_ref, dx_ref, dw_partial_ref, *, kernel_size: int, rows: int, exact: bool):
    seq_len, chunk = x_ref.shape[1], dx_ref.shape[1]
    start = _chunk_start(chunk)
    halo = kernel_size - 1
    weights = [w_ref[lag] for lag in range(kernel_size)]
    # behind[k] is row t-1-k of x and ahead[k] row t+k of dy, each with its segment id.
    behind = [_edge_row(x_ref, seg_ref, start - 1 - k, seq_len) for k in range(halo)]
    ahead = [_edge_row(dy_ref, seg_ref, start + k, seq_len) for k in range(halo)]
    dw = [jnp.zeros(dw_partial_ref.shape[2:], jnp.float32) for _ in range(kernel_size)]

    def step(i, carry):
        behind, ahead, dw = carry
        local = i * rows
        # dy runs `halo` rows ahead of x, past the end of the sequence in the last chunk.
        dy_rows = [_edge_row(dy_ref, seg_ref, start + local + halo + j, seq_len) for j in range(rows)]
        x_rows = [x_ref[0, start + local + j, :] for j in range(rows)]
        segs = [seg for _, seg in (*behind, *ahead, *dy_rows)]
        dy_windows, x_windows = [], []
        for j in range(rows):
            dy_windows.append([*ahead, dy_rows[j]])
            x_windows.append([(x_rows[j], dy_windows[-1][0][1]), *behind])
            ahead, behind = dy_windows[-1][1:], x_windows[-1][:halo]

        def accumulate(masked, dw):
            for j in range(rows):
                dx_ref[0, local + j, :] = _dx_row(dy_windows[j], weights, dx_ref.dtype, exact, masked)
                dw = _dw_accumulate(dw, dy_windows[j][0][0], x_windows[j], masked)
            return dw

        return behind, ahead, _branch_on_segments(segs, accumulate, dw)

    _, _, dw = jax.lax.fori_loop(0, chunk // rows, step, (behind, ahead, dw))
    for lag in range(kernel_size):
        dw_partial_ref[0, lag, :] = dw[lag]


# --------------------------------------------------------------------------------------
# Wrappers
# --------------------------------------------------------------------------------------


def _compiler_params(tiles: ShortConvTiles):
    if pltriton is None or _FORCE_INTERPRET:  # pragma: no cover
        return None
    return pltriton.CompilerParams(num_warps=tiles.num_warps, num_stages=tiles.num_stages)


def _cost_estimate(body, primals, kernel_inputs_specs, kernel_outputs_specs):
    return with_io_bytes_accessed(
        pl.estimate_cost(body, *primals),
        kernel_inputs_specs=kernel_inputs_specs,
        kernel_outputs_specs=kernel_outputs_specs,
    )


def _grid_and_maps(batch: int, num_s: int, num_c: int):
    """The grid, channel blocks fastest so neighbouring programs share DRAM rows, and an
    adapter that hands index maps bounded ``(b, si, ci)``."""

    def index_map(f):
        return lambda ci, si, b: f(_bounded(b, batch), _bounded(si, num_s), _bounded(ci, num_c))

    return (num_c, num_s, batch), index_map


def short_conv_pallas_fwd_local(
    weight: Float[Array, "W C"],
    x: Float[Array, "B S C"],
    segment_ids: Int[Array, "B S"],
    *,
    block_sizes: ShortConvBlockSizes,
    exact_reference_rounding: bool,
) -> Float[Array, "B S C"]:
    """Shard-local fused forward. Callers must have already entered a ``shard_map``."""
    batch, seq_len, channels = x.shape
    width = weight.shape[0]
    tiles = block_sizes.forward
    bs, bc = tiles.s_block_size, _channel_block(channels, tiles.c_block_size)
    num_s, num_c = seq_len // bs, channels // bc
    grid, im = _grid_and_maps(batch, num_s, num_c)

    whole = im(lambda b, si, ci: (b, 0, ci))  # whole-sequence pointer window
    whole_1d = im(lambda b, si, ci: (b, 0))
    out_shape = jax.ShapeDtypeStruct((batch, seq_len, channels), x.dtype)

    call = pl.pallas_call(
        functools.partial(_fwd_kernel, kernel_size=width, rows=tiles.rows_per_step, exact=exact_reference_rounding),
        out_shape=out_shape,
        grid=grid,
        in_specs=[
            pl.BlockSpec((1, seq_len, bc), whole),
            pl.BlockSpec((1, seq_len), whole_1d),
            pl.BlockSpec((width, bc), im(lambda b, si, ci: (0, ci))),
        ],
        out_specs=pl.BlockSpec((1, bs, bc), im(lambda b, si, ci: (b, si, ci))),
        compiler_params=_compiler_params(tiles),
        interpret=_FORCE_INTERPRET,
        cost_estimate=_cost_estimate(
            short_conv_reference,
            (weight, x, segment_ids),
            kernel_inputs_specs=(weight, x, segment_ids),
            kernel_outputs_specs=(out_shape,),
        ),
        name="short_conv_fwd",
    )
    return call(x, segment_ids, weight)


def short_conv_pallas_bwd_local(
    weight: Float[Array, "W C"],
    x: Float[Array, "B S C"],
    segment_ids: Int[Array, "B S"],
    dy: Float[Array, "B S C"],
    *,
    block_sizes: ShortConvBlockSizes,
    exact_reference_rounding: bool,
) -> tuple[Float[Array, "B S C"], Float[Array, "P W C"]]:
    """Shard-local fused backward. Returns ``(dx, dw_partials)``.

    ``dw_partials`` is ``[batch * num_s_blocks, W, C]`` fp32; the caller sums axis 0.
    Keeping that reduction outside the kernel is what makes ``dw`` deterministic, and
    because the leading axis stays batch-major it lets XLA fold the cross-shard reduction
    into the gradient collective it was going to issue anyway.
    """
    batch, seq_len, channels = x.shape
    width = weight.shape[0]
    tiles = block_sizes.backward
    bs, bc = tiles.s_block_size, _channel_block(channels, tiles.c_block_size)
    num_s, num_c = seq_len // bs, channels // bc
    grid, im = _grid_and_maps(batch, num_s, num_c)

    whole = im(lambda b, si, ci: (b, 0, ci))
    whole_1d = im(lambda b, si, ci: (b, 0))
    dx_shape = jax.ShapeDtypeStruct((batch, seq_len, channels), dy.dtype)
    dw_shape = jax.ShapeDtypeStruct((batch * num_s, width, channels), jnp.float32)

    def _vjp_body(w, x_, seg):
        _, vjp = jax.vjp(lambda a, b: short_conv_reference(a, b, seg), w, x_)
        return vjp(x_)

    call = pl.pallas_call(
        functools.partial(_bwd_kernel, kernel_size=width, rows=tiles.rows_per_step, exact=exact_reference_rounding),
        out_shape=[dx_shape, dw_shape],
        grid=grid,
        in_specs=[
            pl.BlockSpec((1, seq_len, bc), whole),
            pl.BlockSpec((1, seq_len), whole_1d),
            pl.BlockSpec((1, seq_len, bc), whole),
            pl.BlockSpec((width, bc), im(lambda b, si, ci: (0, ci))),
        ],
        out_specs=[
            pl.BlockSpec((1, bs, bc), im(lambda b, si, ci: (b, si, ci))),
            pl.BlockSpec((1, width, bc), im(lambda b, si, ci: (b * num_s + si, 0, ci))),
        ],
        compiler_params=_compiler_params(tiles),
        interpret=_FORCE_INTERPRET,
        cost_estimate=_cost_estimate(
            _vjp_body,
            (weight, x, segment_ids),
            kernel_inputs_specs=(weight, x, segment_ids, dy),
            kernel_outputs_specs=(dx_shape, dw_shape),
        ),
        name="short_conv_bwd",
    )
    dx, dw_partials = call(x, segment_ids, dy, weight)
    return dx, dw_partials


def expected_bytes_moved(x_shape: tuple[int, ...], itemsize: int, width: int, block_sizes) -> dict[str, float]:
    """Traffic model for the fused kernels, in bytes. Feeds the benchmark's GB/s column.

    Forward: read ``x``, write ``y``, plus the ``W - 1`` halo rows each sequence block re-reads.
    Backward: read ``x``, read ``dy``, write ``dx``, plus both halos, plus the ``dw`` partials
    written by the kernel and read by the outer ``sum(0)``.
    """
    elements = math.prod(x_shape)
    tensor = elements * itemsize
    fwd_bs, bwd_bs = block_sizes.forward.s_block_size, block_sizes.backward.s_block_size
    dw_partial = 2.0 * width * 4 / (bwd_bs * itemsize)
    return {
        "forward": tensor * (2.0 + (width - 1) / fwd_bs),
        "backward": tensor * (3.0 + 2.0 * (width - 1) / bwd_bs + dw_partial),
    }
