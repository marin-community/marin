# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Fused RMSNorm + GatedNorm forward as two Pallas Triton GPU kernels.

Why this exists
---------------
``y = rms_norm(x) * w``, ``out = y * sigmoid(silu(y @ W_down) @ W_up)`` with a rank-128 gate.
XLA runs it as five kernels per call: the norm fusion (read ``x``, write ``y``), a cuBLAS GEMM
reading ``y``, a cuBLAS GEMM writing the full-width gate logits, a sigmoid pass (read and write
them again: XLA's Triton GEMM fusion takes the sigmoid, the autotuner falls back to cuBLAS, and
the leftover epilogue stays a separate kernel), and the final multiply (read ``y`` and the gate,
write ``out``). That is about nine full-width passes. The rank-128 GEMMs sit between the
row reduction and the gating multiply, so XLA cannot fuse across them.

These kernels take four passes: the statistics kernel reads ``x`` once and produces the row
scale and the rank-128 projection; the output kernel reads ``x`` again, recomputes ``y``,
computes the gate from the projection on tensor cores, and writes ``out`` and the gate (the
backward needs the gate, and writing it costs one pass where recomputing it would cost a
GEMM and two passes).

Numerics
--------
The row scale commutes with the down projection: ``y @ W_down = rstd * ((x * w) @ W_down)``
up to rounding, so the statistics kernel accumulates the sum of squares and the projection in
one sweep and applies ``rstd`` afterwards. The reference rounds ``x * rstd * w`` to the
activation dtype before the GEMM; this kernel rounds ``x * w``. Everything downstream (the
SiLU, the gate logits, the sigmoid, ``y`` and the product) rounds to the activation dtype at
the same points as XLA's elementwise lowering, which computes each bf16 op in f32 and rounds.
Outputs agree with the reference to bf16 rounding, not bitwise.

Backend
-------
Pallas Triton, like the short-conv kernel next door: Mosaic GPU's layout inference fails on
GB200 with JAX 0.11. Triton requires power-of-2 tile shapes; the hidden size only needs to be
a multiple of the tile sizes.
"""

import contextlib
import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jaxtyping import Array, Float

from levanter.kernels.pallas.cost_estimate_utils import with_io_bytes_accessed

from .config import GatedRmsNormBlockSizes
from .reference import gated_rms_norm_reference

try:  # pragma: no cover - import guard, exercised only by environment
    from jax.experimental.pallas import triton as pltriton

    _HAS_PALLAS_TRITON = True
except (ImportError, ModuleNotFoundError):  # pragma: no cover
    pltriton = None  # type: ignore[assignment]
    _HAS_PALLAS_TRITON = False

_FORCE_INTERPRET = False


@contextlib.contextmanager
def interpret_mode():
    """Run the kernels through Pallas's interpreter, which executes the kernel bodies on CPU."""
    global _FORCE_INTERPRET
    previous = _FORCE_INTERPRET
    _FORCE_INTERPRET = True
    try:
        yield
    finally:
        _FORCE_INTERPRET = previous


def pallas_gated_rms_norm_available() -> bool:
    """True when the Pallas Triton backend imported and we are on a GPU."""
    if _FORCE_INTERPRET:
        return True
    return _HAS_PALLAS_TRITON and jax.default_backend() == "gpu"


def _is_pow2(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def gated_rms_norm_shapes_supported(
    x_shape: tuple[int, ...], w_down_shape: tuple[int, ...], block_sizes: GatedRmsNormBlockSizes
) -> str | None:
    """Returns None when the kernels can run these shapes, else a human-readable reason.

    The token count need not divide ``t_block_size``: the wrapper pads rows.
    """
    if len(x_shape) != 2 or len(w_down_shape) != 2:
        return f"expected x [T, D] and w_down [D, R], got {x_shape} and {w_down_shape}"
    hidden, rank = w_down_shape
    if x_shape[1] != hidden:
        return f"w_down input dim {hidden} != hidden size {x_shape[1]}"
    for name in ("t_block_size", "stats_d_block_size", "out_d_block_size"):
        value = getattr(block_sizes, name)
        if not _is_pow2(value):
            return f"{name} {value} must be a power of 2 (Pallas Triton tile constraint)"
    for name in ("stats_d_block_size", "out_d_block_size"):
        if hidden % getattr(block_sizes, name):
            return f"hidden size {hidden} not divisible by {name} {getattr(block_sizes, name)}"
    if not _is_pow2(rank) or rank < 16:
        return f"gate rank {rank} must be a power of 2 and >= 16 (tensor-core tile)"
    if block_sizes.t_block_size < 16:
        return f"t_block_size {block_sizes.t_block_size} must be >= 16 (tensor-core tile)"
    return None


def _round(value: jax.Array, dtype) -> jax.Array:
    """Round an f32 value to ``dtype`` and back, as XLA does after every low-precision op."""
    return value.astype(dtype).astype(jnp.float32)


def _logistic(value: jax.Array, dtype) -> jax.Array:
    """XLA's lowering of ``logistic`` in ``dtype``: ``1 / (1 + exp(-v))``, rounding each op."""
    exp = _round(jnp.exp(-value), dtype)
    return _round(1.0 / _round(1.0 + exp, dtype), dtype)


def _stats_kernel(x_ref, w_ref, w_down_ref, h_ref, rstd_ref, *, d_block: int, eps: float, dtype):
    t_block, hidden = x_ref.shape
    rank = w_down_ref.shape[1]

    def body(step, carry):
        acc, sumsq = carry
        span = pl.ds(step * d_block, d_block)
        x = x_ref[:, span].astype(jnp.float32)
        w = w_ref[span].astype(jnp.float32)
        sumsq = sumsq + jnp.sum(x * x, axis=1)
        xw = (x * w[None, :]).astype(dtype)
        acc = acc + pl.dot(xw, w_down_ref[span, :].astype(dtype))
        return acc, sumsq

    init = (jnp.zeros((t_block, rank), jnp.float32), jnp.zeros((t_block,), jnp.float32))
    acc, sumsq = jax.lax.fori_loop(0, hidden // d_block, body, init)
    rstd = jax.lax.rsqrt(sumsq * (1.0 / hidden) + eps)
    rstd_ref[...] = rstd
    h_ref[...] = (acc * rstd[:, None]).astype(h_ref.dtype)


def _output_kernel(x_ref, w_ref, rstd_ref, h_ref, w_up_ref, out_ref, gate_ref, *, dtype):
    h = h_ref[...].astype(jnp.float32)
    silu = _round(h * _logistic(h, dtype), dtype)
    logits = _round(pl.dot(silu.astype(dtype), w_up_ref[...].astype(dtype)), dtype)
    gate = _logistic(logits, dtype)
    x = x_ref[...].astype(jnp.float32)
    y = _round(x * rstd_ref[...][:, None] * w_ref[...].astype(jnp.float32)[None, :], dtype)
    out_ref[...] = (y * gate).astype(out_ref.dtype)
    gate_ref[...] = gate.astype(gate_ref.dtype)


def _compiler_params(num_warps: int, num_stages: int):
    if pltriton is None or _FORCE_INTERPRET:  # pragma: no cover
        return None
    return pltriton.CompilerParams(num_warps=num_warps, num_stages=num_stages)


def gated_rms_norm_pallas_fwd_local(
    x: Float[Array, "T D"],
    norm_weight: Float[Array, " D"],
    w_down: Float[Array, "D R"],
    w_up: Float[Array, "R D"],
    *,
    eps: float,
    block_sizes: GatedRmsNormBlockSizes,
) -> tuple[Float[Array, "T D"], Float[Array, "T D"], Float[Array, "T R"], Float[Array, " T"]]:
    """Shard-local fused forward. Returns ``(out, gate, gate_hidden_pre_silu, rstd)``.

    ``T`` must be a multiple of ``t_block_size``; the public wrapper pads it.
    """
    tokens, hidden = x.shape
    rank = w_down.shape[1]
    bt, bd = block_sizes.t_block_size, block_sizes.out_d_block_size
    dtype = x.dtype

    h_shape = jax.ShapeDtypeStruct((tokens, rank), dtype)
    rstd_shape = jax.ShapeDtypeStruct((tokens,), jnp.float32)
    stats = pl.pallas_call(
        functools.partial(_stats_kernel, d_block=block_sizes.stats_d_block_size, eps=eps, dtype=dtype),
        out_shape=[h_shape, rstd_shape],
        grid=(tokens // bt,),
        in_specs=[
            pl.BlockSpec((bt, hidden), lambda i: (i, 0)),
            pl.BlockSpec((hidden,), lambda i: (0,)),
            pl.BlockSpec((hidden, rank), lambda i: (0, 0)),
        ],
        out_specs=[
            pl.BlockSpec((bt, rank), lambda i: (i, 0)),
            pl.BlockSpec((bt,), lambda i: (i,)),
        ],
        compiler_params=_compiler_params(block_sizes.stats_num_warps, block_sizes.stats_num_stages),
        interpret=_FORCE_INTERPRET,
        cost_estimate=with_io_bytes_accessed(
            pl.estimate_cost(lambda a, b: (a @ b), x, w_down),
            kernel_inputs_specs=(x, norm_weight, w_down),
            kernel_outputs_specs=(h_shape, rstd_shape),
        ),
        name="gated_rms_norm_stats",
    )
    h, rstd = stats(x, norm_weight, w_down)

    out_shape = jax.ShapeDtypeStruct((tokens, hidden), dtype)
    gate_shape = jax.ShapeDtypeStruct((tokens, hidden), dtype)
    output = pl.pallas_call(
        functools.partial(_output_kernel, dtype=dtype),
        out_shape=[out_shape, gate_shape],
        grid=(tokens // bt, hidden // bd),
        in_specs=[
            pl.BlockSpec((bt, bd), lambda i, j: (i, j)),
            pl.BlockSpec((bd,), lambda i, j: (j,)),
            pl.BlockSpec((bt,), lambda i, j: (i,)),
            pl.BlockSpec((bt, rank), lambda i, j: (i, 0)),
            pl.BlockSpec((rank, bd), lambda i, j: (0, j)),
        ],
        out_specs=[
            pl.BlockSpec((bt, bd), lambda i, j: (i, j)),
            pl.BlockSpec((bt, bd), lambda i, j: (i, j)),
        ],
        compiler_params=_compiler_params(block_sizes.out_num_warps, block_sizes.out_num_stages),
        interpret=_FORCE_INTERPRET,
        cost_estimate=with_io_bytes_accessed(
            pl.estimate_cost(
                functools.partial(gated_rms_norm_reference, eps=eps), x, norm_weight, w_down, w_up
            ),
            kernel_inputs_specs=(x, norm_weight, rstd_shape, h_shape, w_up),
            kernel_outputs_specs=(out_shape, gate_shape),
        ),
        name="gated_rms_norm_output",
    )
    out, gate = output(x, norm_weight, rstd, h, w_up)
    return out, gate, h, rstd


def expected_bytes_moved(tokens: int, hidden: int, itemsize: int) -> dict[str, float]:
    """HBM traffic of the fused forward: read ``x`` twice, write ``out`` and the gate."""
    tensor = tokens * hidden * itemsize
    return {"forward": 4.0 * tensor}
