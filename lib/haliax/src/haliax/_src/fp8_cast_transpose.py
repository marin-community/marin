# Copyright The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Fused FP8 cast-transpose (+ amax) for the ragged FP8 path.

FP8 ``wgmma`` needs the contracting dim contiguous for both operands, so several
operands are needed in *two* layouts (natural and transposed). Naively that is a
quantize pass plus a separate (slow, strided) FP8 transpose, and the amax pass
for delayed scaling on top -- three reads of the operand.

These Pallas Mosaic-GPU kernels do it in **one read**: load the bf16 tile, emit
a per-tile amax partial (reduced to the per-tensor amax by XLA, which avoids
serializing every tile on one global ``atomic_max`` location), quantize with the
delayed-scaling scale, and store *both* the natural and the transposed FP8 tile.
Hopper has no byte-granularity transpose instruction, so the transpose happens
in registers *before* the FP8 cast: a ``WGMMA -> WGMMA_TRANSPOSED`` layout cast
on the f32 tile (a few register shuffles), quantize in the transposed layout,
and store through a transposed SMEM view -- both FP8 stores are then contiguous.
``in_q_ct`` wraps this in the same ``OverwriteWithGradient`` custom-VJP
threading as ``in_q`` so the per-tensor scale / amax-history state still flows
through ``apply_updates``.
"""

import functools

import jax
import jax.numpy as jnp
from jax import custom_vjp
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

from .fp8 import get_fp8_max, roll_amax_history, scale_from_history

# The register-level transpose works on WGMMA layouts, whose warpgroup tile is
# 64 wide on both axes, so tiles (and therefore dims) must be multiples of 64.
_CT_BLOCKS = (128, 64)


def _block_size(dim: int) -> int:
    block = next((b for b in _CT_BLOCKS if dim % b == 0), None)
    if block is None:
        raise ValueError(f"dim={dim} must be a multiple of 64 for the cast-transpose tile sizes {_CT_BLOCKS}")
    return block


def _smem_transforms(minor: int, dtype) -> tuple:
    """Tiling+swizzle for an SMEM tile with ``minor`` contiguous elements of ``dtype``.

    Mosaic refuses register<->SMEM transfers for which it cannot synthesize a
    bank-conflict-free plan; a swizzled tiled layout always has one.
    """
    swizzle = plgpu.find_swizzle(minor * jnp.dtype(dtype).itemsize * 8)
    swizzle_elems = swizzle // jnp.dtype(dtype).itemsize
    return (plgpu.TilingTransform((8, swizzle_elems)), plgpu.SwizzleTransform(swizzle))


# ---------------------------------------------------------------------------
# Cast-transpose + amax kernels: 2D (activations, gradients) and 3D (weights).
# ---------------------------------------------------------------------------


def _ct_kernel(a_ref, inv_ref, nat_ref, t_ref, amax_ref, *, q_dtype, dtype_max, grid_ndim):
    """Fused cast-transpose+amax kernel body.

    Loads the bf16 tile once (WGMMA register layout), emits the tile's amax
    partial, quantizes, and stores both the natural and the transposed FP8
    tile.  The transpose is a register-level ``WGMMA -> WGMMA_TRANSPOSED``
    layout cast done at f32 (Mosaic only supports the shuffle-based transpose
    at 16/32 bit), so both FP8 stores are contiguous; quantization happens
    after the cast with the same f32 -> FP8 rounding as the natural tile, so
    the two layouts stay bit-identical.  The amax partials go straight to the
    unblocked GMEM output (one scalar per tile) and are reduced to the
    per-tensor amax outside the kernel; a single global ``atomic_max`` location
    would serialize every tile of the grid on one cacheline.  The same body is
    shared by the 2D and 3D dispatch functions (``grid_ndim`` distinguishes
    them); Pallas presents the inner ``(bm, bn)`` tile uniformly regardless of
    whether the call grids over an outer expert dimension.
    """
    a = plgpu.load(a_ref, (), layout=plgpu.Layout.WGMMA).astype(jnp.float32)
    tile_idx = tuple(pl.program_id(d) for d in range(grid_ndim))
    # Two chained single-axis reductions: Mosaic tiled layouts have no multi-axis reduce.
    amax_ref[tile_idx] = jnp.max(jnp.max(jnp.abs(a), axis=1))
    q = jnp.clip(a * inv_ref[0], -dtype_max, dtype_max)
    nat_ref[...] = q.astype(q_dtype)
    q_t = plgpu.layout_cast(q, plgpu.Layout.WGMMA_TRANSPOSED)
    plgpu.transpose_ref(t_ref, (1, 0))[...] = q_t.astype(q_dtype)


def _ct_body_cost(q_dtype, dtype_max, x_struct):
    """Flops/transcendentals estimate of the fused amax+quantize body (bytes are computed separately)."""

    def _body(x_, inv_):
        a_f32 = x_.astype(jnp.float32)
        _ = jnp.max(jnp.abs(a_f32))  # unused value: keeps the amax reduction in the cost estimate
        return jnp.clip(a_f32 * inv_[0], -dtype_max, dtype_max).astype(q_dtype)

    inv_struct = jax.ShapeDtypeStruct((1,), jnp.float32)
    return pl.estimate_cost(_body, x_struct, inv_struct)


@functools.lru_cache(maxsize=16)
def _make_ct_call_2d(m: int, k: int, q_dtype, x_dtype, t_extra: int = 0):
    """Build and cache the cast-transpose+amax pallas_call for 2-D inputs.

    ``t_extra`` widens the transposed output to ``[k, m + t_extra]``; the extra
    columns are left unwritten (the wgrad boundary appendix fills them, see
    ``write_boundary_appendix``).
    """
    # A 64-row tile on the (token) M axis measures better bandwidth than 128
    # at the d2560 shapes; K keeps the widest tile.
    bm = 64 if m % 64 == 0 else _block_size(m)
    bk = _block_size(k)
    dtype_max = float(get_fp8_max(q_dtype, jnp.float32))

    grid_m, grid_k = m // bm, k // bk
    body_cost = _ct_body_cost(q_dtype, dtype_max, jax.ShapeDtypeStruct((m, k), x_dtype))
    input_bytes = m * k * jnp.dtype(x_dtype).itemsize + jnp.dtype(jnp.float32).itemsize
    # Two output stores: natural [m,k] + transposed [k,m] = 2*m*k FP8 bytes, plus amax partials.
    output_bytes = 2 * m * k * jnp.dtype(q_dtype).itemsize + grid_m * grid_k * jnp.dtype(jnp.float32).itemsize
    cost = pl.CostEstimate(
        flops=body_cost.flops,
        transcendentals=body_cost.transcendentals,
        bytes_accessed=input_bytes + output_bytes,
    )

    return pl.pallas_call(
        functools.partial(_ct_kernel, q_dtype=q_dtype, dtype_max=dtype_max, grid_ndim=2),
        out_shape=[
            jax.ShapeDtypeStruct((m, k), q_dtype),
            jax.ShapeDtypeStruct((k, m + t_extra), q_dtype),
            jax.ShapeDtypeStruct((grid_m, grid_k), jnp.float32),
        ],
        in_specs=[
            plgpu.BlockSpec((bm, bk), lambda i, j: (i, j), transforms=_smem_transforms(bk, x_dtype)),
            pl.BlockSpec(memory_space=plgpu.GMEM),
        ],
        out_specs=[
            plgpu.BlockSpec((bm, bk), lambda i, j: (i, j), transforms=_smem_transforms(bk, q_dtype)),
            plgpu.BlockSpec((bk, bm), lambda i, j: (j, i), transforms=_smem_transforms(bm, q_dtype)),
            pl.BlockSpec(memory_space=plgpu.GMEM),
        ],
        grid=(grid_m, grid_k),
        compiler_params=plgpu.CompilerParams(reduction_scratch_bytes=6144),
        cost_estimate=cost,
    )


def cast_transpose_amax_2d(x, inv_scale, q_dtype, t_extra=0):
    """``x[M,K]`` bf16 → (fp8 ``[M,K]``, fp8 ``[K,M+t_extra]``, amax ``[1]``) in one read.

    Loads the bf16 tile once, emits per-tile amax partials (reduced by XLA),
    quantizes, and stores both the natural and the transposed FP8 tiles.

    Args:
        x: ``[M, K]`` bfloat16 input.
        inv_scale: ``[1]`` float32 inverse scale (``1 / new_scale``).
        q_dtype: target FP8 dtype (``jnp.float8_e4m3fn`` or ``jnp.float8_e5m2``).
        t_extra: extra (unwritten) columns appended to the transposed output for
            the wgrad boundary appendix (``write_boundary_appendix``).

    Returns:
        ``(q_natural, q_transposed, amax)``: natural ``[M, K]``, transposed
        ``[K, M + t_extra]``, and per-tensor amax ``[1]``.
    """
    m, k = x.shape
    q_nat, q_t, amax_parts = _make_ct_call_2d(m, k, q_dtype, x.dtype, t_extra)(x, inv_scale)
    return q_nat, q_t, jnp.max(amax_parts).reshape(1)


@functools.lru_cache(maxsize=16)
def _make_ct_call_3d(e: int, k: int, n: int, q_dtype, x_dtype):
    """Build and cache the cast-transpose+amax pallas_call for 3-D weight inputs.

    Grids over the expert dimension ``E``; each grid cell processes one
    ``(bm, bn)`` tile of the ``[K, N]`` slice.  The transposed out-spec
    ``(None, bn, bm)`` indexed ``(e, j, i)`` writes the ``[K, N]`` block into
    the ``[N, K]`` transposed layout via the register-level layout cast in the
    kernel body.
    """
    bm, bn = _block_size(k), _block_size(n)
    dtype_max = float(get_fp8_max(q_dtype, jnp.float32))

    grid_k, grid_n = k // bm, n // bn
    body_cost = _ct_body_cost(q_dtype, dtype_max, jax.ShapeDtypeStruct((e, k, n), x_dtype))
    input_bytes = e * k * n * jnp.dtype(x_dtype).itemsize + jnp.dtype(jnp.float32).itemsize
    # Two output stores: natural [e,k,n] + transposed [e,n,k] = 2*e*k*n FP8 bytes, plus amax partials.
    output_bytes = 2 * e * k * n * jnp.dtype(q_dtype).itemsize + e * grid_k * grid_n * jnp.dtype(jnp.float32).itemsize
    cost = pl.CostEstimate(
        flops=body_cost.flops,
        transcendentals=body_cost.transcendentals,
        bytes_accessed=input_bytes + output_bytes,
    )

    return pl.pallas_call(
        functools.partial(_ct_kernel, q_dtype=q_dtype, dtype_max=dtype_max, grid_ndim=3),
        out_shape=[
            jax.ShapeDtypeStruct((e, k, n), q_dtype),
            jax.ShapeDtypeStruct((e, n, k), q_dtype),
            jax.ShapeDtypeStruct((e, grid_k, grid_n), jnp.float32),
        ],
        in_specs=[
            plgpu.BlockSpec((None, bm, bn), lambda e, i, j: (e, i, j), transforms=_smem_transforms(bn, x_dtype)),
            pl.BlockSpec(memory_space=plgpu.GMEM),
        ],
        out_specs=[
            plgpu.BlockSpec((None, bm, bn), lambda e, i, j: (e, i, j), transforms=_smem_transforms(bn, q_dtype)),
            plgpu.BlockSpec((None, bn, bm), lambda e, i, j: (e, j, i), transforms=_smem_transforms(bm, q_dtype)),
            pl.BlockSpec(memory_space=plgpu.GMEM),
        ],
        grid=(e, k // bm, n // bn),
        compiler_params=plgpu.CompilerParams(reduction_scratch_bytes=6144),
        cost_estimate=cost,
    )


def cast_transpose_amax_3d(x, inv_scale, q_dtype):
    """``x[E,K,N]`` bf16 → (fp8 ``[E,K,N]``, fp8 ``[E,N,K]``, amax ``[1]``) in one read.

    Like ``cast_transpose_amax_2d`` but grids over the expert dimension ``E``.

    Args:
        x: ``[E, K, N]`` bfloat16 weight tensor.
        inv_scale: ``[1]`` float32 inverse scale (``1 / new_scale``).
        q_dtype: target FP8 dtype.

    Returns:
        ``(q_natural, q_transposed, amax)``: natural ``[E, K, N]``, transposed
        ``[E, N, K]``, and per-tensor amax ``[1]``.
    """
    e, k, n = x.shape
    q_nat, q_t, amax_parts = _make_ct_call_3d(e, k, n, q_dtype, x.dtype)(x, inv_scale)
    return q_nat, q_t, jnp.max(amax_parts).reshape(1)


# ---------------------------------------------------------------------------
# Delayed-scaling helper.
# ---------------------------------------------------------------------------


def _next_scale_and_inv(q_dtype, scale, amax_history):
    """Delayed-scaling scale for this step (shared TE recipe from ``_src/fp8``) + its reciprocal."""
    new_scale = scale_from_history(q_dtype, scale, amax_history)
    inv_scale = (1.0 / new_scale).astype(jnp.float32).reshape(1)
    return new_scale, inv_scale


# ---------------------------------------------------------------------------
# Cast-transpose analog of in_q: produces both layouts + threads delayed-scaling state.
# ---------------------------------------------------------------------------


def cast_transpose_delayed(q_dtype, mode, t_extra, inp, scale, amax_history):
    """Cast-transpose ``inp`` with delayed scaling.

    Returns ``(q_natural, q_transposed, new_scale, new_history)``.  The amax is
    folded into the kernel (per-tile partials reduced by XLA), so
    ``new_history`` reflects the current step's observed magnitude without a
    separate read of the input.

    Args:
        q_dtype: target FP8 dtype.
        mode: ``"2d"`` for activations ``[M, K]`` or ``"3d"`` for weights ``[E, K, N]``.
        t_extra: extra transposed-output columns for the wgrad boundary appendix
            (2-D mode only; pass 0 when unused).
        inp: bf16 tensor to quantize.
        scale: current delayed-scaling scale ``[1]``.
        amax_history: rolling amax history.
    """
    new_scale, inv_scale = _next_scale_and_inv(q_dtype, scale, amax_history)
    if mode == "3d":
        if t_extra:
            raise ValueError("t_extra is only supported for 2-D cast-transpose (wgrad appendix)")
        q_nat, q_t, cur_amax = cast_transpose_amax_3d(inp, inv_scale, q_dtype)
    else:
        q_nat, q_t, cur_amax = cast_transpose_amax_2d(inp, inv_scale, q_dtype, t_extra)
    new_history = roll_amax_history(cur_amax[0], amax_history)
    return q_nat, q_t, new_scale, new_history


@functools.partial(custom_vjp, nondiff_argnums=(0, 1, 2))
def in_q_ct(q_dtype, mode, t_extra, inp, scale, amax_history):
    """Cast-transpose analog of ``in_q``: returns ``(q_natural, q_transposed, new_scale)``.

    Produces both FP8 layouts (natural and transposed) in a single bf16 read
    and threads the delayed-scaling ``OverwriteWithGradient`` state (``scale``
    and ``amax_history`` are overwritten by the backward via the returned
    cotangents, just like ``in_q``).

    Args:
        q_dtype: target FP8 dtype (non-differentiable static arg).
        mode: ``"2d"`` or ``"3d"`` (non-differentiable static arg).
        t_extra: extra transposed-output columns for the wgrad boundary appendix
            (non-differentiable static arg; 2-D mode only).
        inp: bf16 tensor to quantize.
        scale: current delayed-scaling scale ``[1]``.
        amax_history: rolling amax history.

    Returns:
        ``(q_natural, q_transposed, new_scale)``.
    """
    q_nat, q_t, new_scale, _ = cast_transpose_delayed(q_dtype, mode, t_extra, inp, scale, amax_history)
    return q_nat, q_t, new_scale


def _in_q_ct_fwd(q_dtype, mode, t_extra, inp, scale, amax_history):
    q_nat, q_t, new_scale, new_history = cast_transpose_delayed(q_dtype, mode, t_extra, inp, scale, amax_history)
    return (q_nat, q_t, new_scale), (new_scale, new_history)


def _in_q_ct_bwd(q_dtype, mode, t_extra, res, _cts):
    new_scale, new_history = res
    # No grad to inp; pass updated scale/history through as cotangents so
    # OverwriteWithGradient overwrites the persisted delayed-scaling state.
    return None, new_scale, new_history


in_q_ct.defvjp(_in_q_ct_fwd, _in_q_ct_bwd)


# ---------------------------------------------------------------------------
# Wgrad boundary appendix: pre-masked group-boundary token blocks.
# ---------------------------------------------------------------------------

# Token-block width of the wgrad pipeline (== the wgrad kernel's block_k) and of
# the appendix slots: each group owns two 128-token slots (first / last block).
WGRAD_TOKEN_BLOCK = 128


def _boundary_appendix_kernel(start_ref, end_ref, q_ref, out_ref, smem_ref, barrier_ref, *, block_rows, t_real):
    """Write group ``g``'s pre-masked first/last token blocks into its appendix slots.

    Grid ``(G, K // block_rows)``.  Reads the boundary blocks from the main
    region ``[:, :t_real]`` and stores masked copies at columns
    ``t_real + g*2*B + {0, B}`` (``B = WGRAD_TOKEN_BLOCK``).  The universal keep
    predicate ``start <= token < end`` makes single-block groups (first == last)
    correct in either slot, and empty groups write all-zero blocks that the
    wgrad never reads.
    """
    b = WGRAD_TOKEN_BLOCK
    g = pl.program_id(0)
    i = pl.program_id(1)
    start = start_ref[g]
    end = end_ref[g]
    rows = pl.ds(i * block_rows, block_rows)
    first_blk = start // b * b
    last_blk = jnp.maximum(end - 1, 0) // b * b
    blks = (first_blk, last_blk)
    # Stage through SMEM: TMA both boundary blocks in, mask in registers, TMA out.
    for slot in (0, 1):
        plgpu.copy_gmem_to_smem(q_ref.at[rows, pl.ds(blks[slot], b)], smem_ref.at[slot], barrier_ref.at[slot])
    cols = plgpu.broadcasted_iota(jnp.int32, (block_rows, b), 1, layout=plgpu.Layout.WGMMA)
    for slot in (0, 1):
        plgpu.barrier_wait(barrier_ref.at[slot])
        src = plgpu.load(smem_ref.at[slot], (), layout=plgpu.Layout.WGMMA)
        keep = (blks[slot] + cols >= start) & (blks[slot] + cols < end)
        val = jnp.where(keep, src.astype(jnp.float16), jnp.float16(0)).astype(src.dtype)
        smem_ref.at[slot][...] = val
        plgpu.commit_smem()
        plgpu.copy_smem_to_gmem(smem_ref.at[slot], out_ref.at[rows, pl.ds(t_real + g * 2 * b + slot * b, b)])
    plgpu.wait_smem_to_gmem(0, wait_read_only=True)


def write_boundary_appendix(q_t, group_sizes):
    """Fill the wgrad boundary appendix of a widened transposed operand, in place.

    ``q_t`` is ``[K, T + G*256]`` (from ``cast_transpose_amax_2d(..., t_extra=G*256)``):
    the main region ``[:, :T]`` holds the token-contiguous FP8 operand and the
    appendix holds, per group, its first and last ``WGRAD_TOKEN_BLOCK``-token
    blocks with out-of-group tokens zeroed.  The wgrad kernel reads boundary
    pipeline steps from the appendix instead of masking in SMEM
    (``mgpu_dwgrad``), which keeps its pipeline body branch-free.

    The buffer is aliased through the kernel (no copy of the main region).

    Args:
        q_t: ``[K, T + G*2*WGRAD_TOKEN_BLOCK]`` FP8, token dim contiguous.
        group_sizes: ``[G]`` int32 token count per group (``sum == T``).

    Returns:
        The same-shape array with the appendix written.
    """
    g = group_sizes.shape[0]
    k_dim, t_wide = q_t.shape
    b = WGRAD_TOKEN_BLOCK
    t_real = t_wide - g * 2 * b
    if t_real <= 0 or t_real % b != 0:
        raise ValueError(f"q_t width {t_wide} does not contain a {b}-aligned main region for {g} groups")
    if k_dim % b != 0:
        raise ValueError(f"K={k_dim} must be a multiple of {b} for the boundary-appendix writer")
    ends = jnp.cumsum(group_sizes.astype(jnp.int32))
    starts = ends - group_sizes.astype(jnp.int32)
    return pl.pallas_call(
        functools.partial(_boundary_appendix_kernel, block_rows=b, t_real=t_real),
        out_shape=jax.ShapeDtypeStruct(q_t.shape, q_t.dtype),
        grid=(g, k_dim // b),
        in_specs=[
            pl.BlockSpec(memory_space=plgpu.GMEM),
            pl.BlockSpec(memory_space=plgpu.GMEM),
            pl.BlockSpec(memory_space=plgpu.GMEM),
        ],
        out_specs=pl.BlockSpec(memory_space=plgpu.GMEM),
        scratch_shapes=[
            plgpu.SMEM((2, b, b), q_t.dtype, transforms=_smem_transforms(b, q_t.dtype)),
            plgpu.Barrier(num_barriers=2),
        ],
        input_output_aliases={2: 0},
        compiler_params=plgpu.CompilerParams(),
    )(starts, ends, q_t)
