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
this kernel                3.0 passes        6.7 passes
=========================  ===============  ===============

Each tap is a separate offset read of the whole tensor that the cache does not fully
absorb, and the reference's VJP additionally launches five fusions that each re-read their
inputs. The kernel collapses those to one launch per direction, so the forward drops from
0.811 ms to 0.547 ms and the backward from 2.558 ms to 1.200 ms at C=6144 -- 1.5x and 2.1x.
The residual gap to the floor is the same offset-read problem: Pallas Triton cannot slice
a register tile (no ``lax.slice`` lowering) and requires power-of-2 tile shapes, so the
kernel still issues ``W`` overlapping loads per tile rather than loading once and shifting
in registers the way ``Dao-AILab/causal-conv1d`` does with an SMEM ring carry.

Closing that last gap needs a register/SMEM carry, which needs CUDA or a Mosaic backend
that lowers here. Neither is available today; see "Backend" below.

``dw`` never touches HBM as a full-size tensor. It is accumulated in fp32 registers per
program and emitted as a ``[batch * num_s_blocks, W, C]`` partial that a cheap outer
``sum(0)`` folds down -- the deterministic reduction that FLA's Triton conv uses, rather
than ``atomicAdd``. It costs ``2 * W * 4 / (s_block_size * itemsize)`` of a pass (3% at the
default tile) and is bit-reproducible run to run.

Backend
-------
**Pallas Triton, not Mosaic GPU.** The repo's kernel skill prefers Mosaic for new GPU
kernels; measured on GB200/SM100 with JAX 0.11, Mosaic's layout inference fails on this
kernel and on a trivial ``o = w * x`` body alike ("Layout inference failed to find a
solution"). Triton lowers both. Revisit when Mosaic's layout inference improves.

Triton imposes one hard constraint that shapes the whole design: **every intermediate tile
must have a power-of-2 shape.** Arbitrary *offsets* are fine; only *sizes* are
constrained. That rules out the obvious halo construction -- concatenating the ``lag``
rows carried over from the previous block onto this block's first ``BS-lag`` rows -- since
``BS-lag`` is never a power of two. Confirmed on hardware: it fails with "Encountered an
array of shape (127, 128)".

Halo handling
-------------
So every shifted tile is a single ``pl.ds(start, BS)`` read at an arbitrary offset out of a
whole-sequence window. That is exact and power-of-2 for every block except the ones at the
ends of the sequence, where ``start`` would run off the array. Rather than rely on an
out-of-bounds read being masked -- which happens to work on Triton but is *wrong* under
Pallas's own interpreter, where the clamp silently misaligns the tile and would make the
CPU tests disagree with the GPU -- the kernel takes a small pre-padded **head block**
(``[B, BS+W-1, C]``: ``W-1`` zero rows then the first ``BS`` rows of ``x``) and, in the
backward, a matching **tail block** for the anti-causal direction. ``pl.when`` routes the
edge programs to them. Building both touches ``~2*BS/S`` of the tensor, under 7% of a
pass, and every read in the kernel is then unconditionally in bounds.

Numerics
--------
``exact_reference_rounding=True`` (the default) reproduces the reference's rounding
*operation for operation*: every multiply and every accumulate rounds to the activation
dtype, in the reference's association order. That order is not cosmetic -- the forward is
left-nested over ascending lags, and JAX transposes it by walking the jaxpr in reverse, so
``dx`` accumulates over *descending* lags. Matching both makes the forward and ``dx``
bit-identical to the reference. ``dw`` is a reduction over 65,536 tokens whose association
order XLA does not define, so it agrees to fp32 reassociation error and is validated
against a float64 oracle instead. Setting the flag to ``False`` keeps a single fp32
accumulator across taps: strictly more accurate, no longer bit-comparable.
"""

import contextlib
import functools
import math

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jaxtyping import Array, Float, Int

from levanter.kernels.pallas.cost_estimate_utils import with_io_bytes_accessed

from .config import ShortConvBlockSizes
from .reference import short_conv_reference

try:  # pragma: no cover - import guard, exercised only by environment
    from jax.experimental.pallas import triton as pltriton

    _HAS_PALLAS_TRITON = True
except (ImportError, ModuleNotFoundError):  # pragma: no cover
    pltriton = None  # type: ignore[assignment]
    _HAS_PALLAS_TRITON = False

#: Sentinel written into the shifted segment ids for positions outside the sequence.
#: `jnp.pad(segment_ids, ..., constant_values=-1)` in the reference; matching it exactly is
#: what makes the first `kernel_size - 1` positions of every sequence agree.
_OOB_SEGMENT = -1

_FORCE_INTERPRET = False


@contextlib.contextmanager
def interpret_mode():
    """Run the kernels through Pallas's reference interpreter.

    This is the only way these kernels get CPU coverage: a Pallas GPU kernel cannot execute
    on CPU, but the interpreter executes the *kernel body* -- grid, block specs, head/tail
    routing, register accumulation and all -- with plain XLA ops. It exercises the real
    algorithm, not a paraphrase. It says nothing about whether the kernel *lowers* on a
    given GPU architecture; that needs a GPU.
    """
    global _FORCE_INTERPRET
    previous = _FORCE_INTERPRET
    _FORCE_INTERPRET = True
    try:
        yield
    finally:
        _FORCE_INTERPRET = previous


def pallas_short_conv_available() -> bool:
    """True when the Pallas Triton backend imported and we are on a GPU."""
    if _FORCE_INTERPRET:
        return True
    return _HAS_PALLAS_TRITON and jax.default_backend() == "gpu"


def _is_pow2(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def short_conv_shapes_supported(
    weight_shape: tuple[int, ...],
    x_shape: tuple[int, ...],
    block_sizes: ShortConvBlockSizes,
) -> str | None:
    """Returns None when the kernel can run these shapes, else a human-readable reason."""
    if len(x_shape) != 3 or len(weight_shape) != 2:
        return f"expected weight [W, C] and x [B, S, C], got {weight_shape} and {x_shape}"
    width, weight_channels = weight_shape
    _, seq_len, channels = x_shape
    if weight_channels != channels:
        return f"weight channel dim {weight_channels} != x channel dim {channels}"
    if width < 1:
        return f"kernel_size must be >= 1, got {width}"
    bs, bc = block_sizes.s_block_size, block_sizes.c_block_size
    if seq_len % bs:
        return f"seq_len {seq_len} not divisible by s_block_size {bs}"
    if channels % bc:
        return f"channels {channels} not divisible by c_block_size {bc}"
    # Pallas Triton requires power-of-2 tile shapes; every read in the kernel is [bs, bc].
    if not _is_pow2(bs):
        return f"s_block_size {bs} must be a power of 2 (Pallas Triton tile constraint)"
    if not _is_pow2(bc):
        return f"c_block_size {bc} must be a power of 2 (Pallas Triton tile constraint)"
    if bs < width - 1:
        return f"s_block_size {bs} must be >= kernel_size - 1 = {width - 1}"
    return None


def _mul_round(weight_row: jax.Array, tile: jax.Array, dtype, exact: bool) -> jax.Array:
    """``weight_row[None, :] * tile`` with the reference's rounding."""
    product = weight_row[None, :].astype(jnp.float32) * tile.astype(jnp.float32)
    return product.astype(dtype) if exact else product


def _add_round(acc, term, dtype, exact: bool) -> jax.Array:
    if acc is None:
        return term
    total = acc.astype(jnp.float32) + term.astype(jnp.float32)
    return total.astype(dtype) if exact else total


def _keep(seg_shifted: jax.Array, seg_cur: jax.Array, tile: jax.Array) -> jax.Array:
    return jnp.where((seg_shifted == seg_cur)[:, None], tile, jnp.zeros_like(tile))


# --------------------------------------------------------------------------------------
# Forward
# --------------------------------------------------------------------------------------


def _fwd_body(
    vals_ref, segs_ref, w_ref, out_ref, base: int | jax.Array, block_seq: int, kernel_size: int, exact: bool
) -> None:
    """One block's forward, reading taps at ``base - lag`` out of ``vals_ref``.

    ``base`` is the row in ``vals_ref`` corresponding to this block's first output row, so
    the general path passes ``si * BS`` into the whole-sequence view and the first-block
    path passes ``W - 1`` into the pre-padded head view. Both then read identically.
    """
    seg_cur = segs_ref[0, pl.ds(base, block_seq)]
    tile = vals_ref[0, pl.ds(base, block_seq), :]
    dtype = tile.dtype
    # Ascending lags, left-nested, rounding after every op: the reference's exact order.
    acc = _mul_round(w_ref[0], tile, dtype, exact)
    for lag in range(1, kernel_size):
        shifted = vals_ref[0, pl.ds(base - lag, block_seq), :]
        seg_shifted = segs_ref[0, pl.ds(base - lag, block_seq)]
        shifted = _keep(seg_shifted, seg_cur, shifted)
        acc = _add_round(acc, _mul_round(w_ref[lag], shifted, dtype, exact), dtype, exact)
    out_ref[0] = acc.astype(dtype)


def _fwd_kernel(x_ref, xh_ref, seg_ref, segh_ref, w_ref, out_ref, *, kernel_size: int, exact: bool):
    block_seq = out_ref.shape[1]
    si = pl.program_id(1)

    @pl.when(si == 0)
    def _first():
        _fwd_body(xh_ref, segh_ref, w_ref, out_ref, kernel_size - 1, block_seq, kernel_size, exact)

    @pl.when(si != 0)
    def _general():
        _fwd_body(x_ref, seg_ref, w_ref, out_ref, si * block_seq, block_seq, kernel_size, exact)


# --------------------------------------------------------------------------------------
# Backward
# --------------------------------------------------------------------------------------


def _dx_body(dy_ref, segs_ref, w_ref, dx_ref, base, block_seq: int, kernel_size: int, exact: bool) -> None:
    """``dx[t] = sum_lag w[lag] * [seg[t] == seg[t+lag]] * dy[t+lag]``.

    Descending lags then tap 0, because that is the order JAX's transpose of the forward
    produces and matching it is what makes ``dx`` bit-identical rather than merely close.
    """
    seg_cur = segs_ref[0, pl.ds(base, block_seq)]
    dy = dy_ref[0, pl.ds(base, block_seq), :]
    dtype = dy.dtype
    acc = None
    for lag in range(kernel_size - 1, 0, -1):
        dy_ahead = dy_ref[0, pl.ds(base + lag, block_seq), :]
        seg_ahead = segs_ref[0, pl.ds(base + lag, block_seq)]
        term = _mul_round(w_ref[lag], dy_ahead, dtype, exact)
        acc = _add_round(acc, _keep(seg_ahead, seg_cur, term), dtype, exact)
    dx_ref[0] = _add_round(acc, _mul_round(w_ref[0], dy, dtype, exact), dtype, exact).astype(dtype)


def _dw_body(x_ref, segs_ref, dy, dw_ref, base, block_seq: int, kernel_size: int) -> None:
    """``dw[lag] = sum_t dy[t] * [seg[t-lag] == seg[t]] * x[t-lag]``, fp32, in registers.

    The shifted-``x`` construction is the forward's, so the mask semantics are shared by
    construction rather than by a comment asking you to keep them in sync.
    """
    seg_cur = segs_ref[0, pl.ds(base, block_seq)]
    dy_f32 = dy.astype(jnp.float32)
    for lag in range(kernel_size):
        shifted = x_ref[0, pl.ds(base - lag, block_seq), :]
        if lag:
            shifted = _keep(segs_ref[0, pl.ds(base - lag, block_seq)], seg_cur, shifted)
        partial = jnp.sum(dy_f32 * shifted.astype(jnp.float32), axis=0)
        # Store per tap rather than stacking: Triton's `stack` lowering takes exactly two
        # operands, and a W-way stack would build non-power-of-2 intermediates anyway.
        dw_ref[0, pl.ds(lag, 1), :] = partial[None, :]


def _bwd_kernel(
    x_ref,
    xh_ref,
    seg_ref,
    segh_ref,
    dy_ref,
    dyt_ref,
    segt_ref,
    w_ref,
    dx_ref,
    dw_partial_ref,
    *,
    kernel_size: int,
    exact: bool,
):
    block_seq = dx_ref.shape[1]
    si = pl.program_id(1)
    last = pl.num_programs(1) - 1

    # dx reads dy *ahead* of this block, so only the final block needs the tail view.
    @pl.when(si != last)
    def _dx_general():
        _dx_body(dy_ref, seg_ref, w_ref, dx_ref, si * block_seq, block_seq, kernel_size, exact)

    @pl.when(si == last)
    def _dx_last():
        _dx_body(dyt_ref, segt_ref, w_ref, dx_ref, 0, block_seq, kernel_size, exact)

    # dw reads x *behind* this block, so only the first block needs the head view.
    @pl.when(si != 0)
    def _dw_general():
        _dw_body(
            x_ref,
            seg_ref,
            dy_ref[0, pl.ds(si * block_seq, block_seq), :],
            dw_partial_ref,
            si * block_seq,
            block_seq,
            kernel_size,
        )

    @pl.when(si == 0)
    def _dw_first():
        _dw_body(
            xh_ref,
            segh_ref,
            dy_ref[0, pl.ds(0, block_seq), :],
            dw_partial_ref,
            kernel_size - 1,
            block_seq,
            kernel_size,
        )


# --------------------------------------------------------------------------------------
# Edge views, wrappers
# --------------------------------------------------------------------------------------


def _head_views(x, segment_ids, block_seq: int, width: int):
    """``[B, BS+W-1, C]`` / ``[B, BS+W-1]``: ``W-1`` pad rows then the first ``BS`` rows.

    Exactly what ``jnp.pad(x, ((0,0),(W-1,0),(0,0)))`` would give for those rows, so the
    first-block path is the reference's semantics with no special casing inside the kernel.
    """
    pad_vals = jnp.zeros((x.shape[0], width - 1, x.shape[2]), x.dtype)
    pad_segs = jnp.full((segment_ids.shape[0], width - 1), _OOB_SEGMENT, segment_ids.dtype)
    return (
        jnp.concatenate([pad_vals, x[:, :block_seq, :]], axis=1),
        jnp.concatenate([pad_segs, segment_ids[:, :block_seq]], axis=1),
    )


def _tail_views(dy, segment_ids, block_seq: int, width: int):
    """``[B, BS+W-1, C]`` / ``[B, BS+W-1]``: the last ``BS`` rows then ``W-1`` pad rows.

    The anti-causal mirror of ``_head_views``. Positions past the end contribute nothing to
    ``dx``, so the pad value is zero and its segment id can never match.
    """
    pad_vals = jnp.zeros((dy.shape[0], width - 1, dy.shape[2]), dy.dtype)
    pad_segs = jnp.full((segment_ids.shape[0], width - 1), _OOB_SEGMENT, segment_ids.dtype)
    return (
        jnp.concatenate([dy[:, -block_seq:, :], pad_vals], axis=1),
        jnp.concatenate([segment_ids[:, -block_seq:], pad_segs], axis=1),
    )


def _compiler_params(block_sizes: ShortConvBlockSizes):
    if pltriton is None or _FORCE_INTERPRET:  # pragma: no cover
        return None
    return pltriton.CompilerParams(num_warps=block_sizes.num_warps, num_stages=block_sizes.num_stages)


def _cost_estimate(body, primals, kernel_inputs_specs, kernel_outputs_specs):
    return with_io_bytes_accessed(
        pl.estimate_cost(body, *primals),
        kernel_inputs_specs=kernel_inputs_specs,
        kernel_outputs_specs=kernel_outputs_specs,
    )


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
    bs, bc = block_sizes.s_block_size, block_sizes.c_block_size
    num_s, num_c = seq_len // bs, channels // bc
    x_head, seg_head = _head_views(x, segment_ids, bs, width)

    whole = lambda b, si, ci: (b, 0, ci)  # noqa: E731 - whole-sequence pointer window
    whole_1d = lambda b, si, ci: (b, 0)  # noqa: E731
    out_shape = jax.ShapeDtypeStruct((batch, seq_len, channels), x.dtype)

    call = pl.pallas_call(
        functools.partial(_fwd_kernel, kernel_size=width, exact=exact_reference_rounding),
        out_shape=out_shape,
        grid=(batch, num_s, num_c),
        in_specs=[
            pl.BlockSpec((1, seq_len, bc), whole),
            pl.BlockSpec((1, bs + width - 1, bc), whole),
            pl.BlockSpec((1, seq_len), whole_1d),
            pl.BlockSpec((1, bs + width - 1), whole_1d),
            pl.BlockSpec((width, bc), lambda b, si, ci: (0, ci)),
        ],
        out_specs=pl.BlockSpec((1, bs, bc), lambda b, si, ci: (b, si, ci)),
        compiler_params=_compiler_params(block_sizes),
        interpret=_FORCE_INTERPRET,
        cost_estimate=_cost_estimate(
            short_conv_reference,
            (weight, x, segment_ids),
            # The head view is under 4% of `x` and every window aliases the same buffers, so
            # modelling traffic as one read of x plus one write of the output is honest.
            kernel_inputs_specs=(weight, x, segment_ids),
            kernel_outputs_specs=(out_shape,),
        ),
        name="short_conv_fwd",
    )
    return call(x, x_head, segment_ids, seg_head, weight)


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
    bs, bc = block_sizes.s_block_size, block_sizes.c_block_size
    num_s, num_c = seq_len // bs, channels // bc
    x_head, seg_head = _head_views(x, segment_ids, bs, width)
    dy_tail, seg_tail = _tail_views(dy, segment_ids, bs, width)

    whole = lambda b, si, ci: (b, 0, ci)  # noqa: E731
    whole_1d = lambda b, si, ci: (b, 0)  # noqa: E731
    dx_shape = jax.ShapeDtypeStruct((batch, seq_len, channels), dy.dtype)
    dw_shape = jax.ShapeDtypeStruct((batch * num_s, width, channels), jnp.float32)

    def _vjp_body(w, x_, seg):
        _, vjp = jax.vjp(lambda a, b: short_conv_reference(a, b, seg), w, x_)
        return vjp(x_)

    call = pl.pallas_call(
        functools.partial(_bwd_kernel, kernel_size=width, exact=exact_reference_rounding),
        out_shape=[dx_shape, dw_shape],
        grid=(batch, num_s, num_c),
        in_specs=[
            pl.BlockSpec((1, seq_len, bc), whole),
            pl.BlockSpec((1, bs + width - 1, bc), whole),
            pl.BlockSpec((1, seq_len), whole_1d),
            pl.BlockSpec((1, bs + width - 1), whole_1d),
            pl.BlockSpec((1, seq_len, bc), whole),
            pl.BlockSpec((1, bs + width - 1, bc), whole),
            pl.BlockSpec((1, bs + width - 1), whole_1d),
            pl.BlockSpec((width, bc), lambda b, si, ci: (0, ci)),
        ],
        out_specs=[
            pl.BlockSpec((1, bs, bc), lambda b, si, ci: (b, si, ci)),
            pl.BlockSpec((1, width, bc), lambda b, si, ci: (b * num_s + si, 0, ci)),
        ],
        compiler_params=_compiler_params(block_sizes),
        interpret=_FORCE_INTERPRET,
        cost_estimate=_cost_estimate(
            _vjp_body,
            (weight, x, segment_ids),
            kernel_inputs_specs=(weight, x, segment_ids, dy),
            kernel_outputs_specs=(dx_shape, dw_shape),
        ),
        name="short_conv_bwd",
    )
    dx, dw_partials = call(x, x_head, segment_ids, seg_head, dy, dy_tail, seg_tail, weight)
    return dx, dw_partials


def expected_bytes_moved(x_shape: tuple[int, ...], itemsize: int, width: int, block_sizes) -> dict[str, float]:
    """Traffic model for the fused kernels, in bytes. Feeds the benchmark's GB/s column.

    Forward: read ``x``, write ``y``, plus the head view (built and read once).
    Backward: read ``x``, read ``dy``, write ``dx``, plus head and tail views, plus the
    ``dw`` partials written by the kernel and read by the outer ``sum(0)``.
    """
    elements = math.prod(x_shape)
    tensor = elements * itemsize
    seq_len = x_shape[1]
    bs = block_sizes.s_block_size
    edge = 2.0 * (bs + width - 1) / seq_len  # build (read+write) one BS-row edge view
    dw_partial = 2.0 * elements * width * 4 / (bs * itemsize) / tensor
    return {
        "forward": tensor * (2.0 + edge),
        "backward": tensor * (3.0 + 2.0 * edge + dw_partial),
    }
