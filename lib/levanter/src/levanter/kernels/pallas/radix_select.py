# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Select each row's k-th largest float32 value by counting radix-key prefixes.

The JAX reference and Pallas-Triton kernel use ordered float bit patterns. The GPU kernel runs all counting passes in
one program per row.
"""

import functools
import math

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu
from jaxtyping import Array, Float, Int

from levanter.kernels.pallas.cost_estimate_utils import with_io_bytes_accessed

_SIGN = 0x80000000
# The float whose key is 0, the smallest (a negative NaN): padding with it never changes the k-th largest key.
_SMALLEST_KEY_FLOAT_BITS = 0xFFFFFFFF
# Tuned on MI355X at [256, 32768], with and without an RCCL all-to-all running beside the kernel. A row of up to this
# many columns is read once into registers (32 per thread) and every pass counts from there; longer rows are read once
# per pass in tiles of this many columns. With the row in registers the time is in the compares, so one-bit digits win:
# 32 passes of one compare per element. A program uses 60 VGPRs (32 hold the row) and 64 bytes of LDS on MI355X, so two
# fit on a compute unit and they fit beside a collective's workgroups.
DEFAULT_BLOCK = 32768
DEFAULT_NUM_WARPS = 16
DEFAULT_RADIX_BITS = 1


def ordered_key(x: Float[Array, "..."]) -> jax.Array:
    """float32 -> uint32 that orders like ``x`` in IEEE 754 totalOrder (-NaN < -inf < ... < -0 < +0 < ... < +NaN)."""
    bits = jax.lax.bitcast_convert_type(x.astype(jnp.float32), jnp.uint32)
    flip = jnp.where(bits >= jnp.uint32(_SIGN), jnp.uint32(0xFFFFFFFF), jnp.uint32(_SIGN))
    return bits ^ flip


def from_ordered_key(key: jax.Array) -> jax.Array:
    """Inverse of :func:`ordered_key`."""
    flip = jnp.where(key >= jnp.uint32(_SIGN), jnp.uint32(_SIGN), jnp.uint32(0xFFFFFFFF))
    return jax.lax.bitcast_convert_type(key ^ flip, jnp.float32)


def _check_args(x: jax.Array, radix_bits: int) -> None:
    if x.ndim != 2:
        raise ValueError(f"x must be rank-2 [rows, n], got shape={x.shape}")
    if radix_bits not in (1, 2, 4):
        raise ValueError(f"radix_bits must be 1, 2 or 4, got {radix_bits}")


def radix_select_reference(
    x: Float[Array, "R N"], k: Int[Array, ""] | Int[Array, "R"] | int, *, radix_bits: int = DEFAULT_RADIX_BITS
) -> Float[Array, "R"]:
    """The ``k``-th largest value (1-based, counting repeats) of each row of ``x``, as float32, in plain JAX.

    Equal, bit for bit, to ``jax.lax.top_k(x, k_max)[0][:, k - 1]`` whenever the sort orders the values as totalOrder
    does: for every finite and infinite value, but where -0.0 and +0.0 compare equal the sort may return either zero.
    ``k`` may be traced, a scalar or one per row, and must lie in ``[1, N]``.
    """
    _check_args(x, radix_bits)
    rows = x.shape[0]
    key = ordered_key(x)
    k = jnp.broadcast_to(jnp.asarray(k, jnp.int32), (rows,))
    digits = jnp.arange(1, 1 << radix_bits, dtype=jnp.uint32)
    prefix = jnp.zeros((rows,), jnp.uint32)
    for shift in range(32 - radix_bits, -1, -radix_bits):
        # candidates[r, d]: the prefix with digit d + 1 at this position; their counts fall as d grows, so the
        # digit to keep is the number of candidates whose count still reaches k.
        candidates = prefix[:, None] | (digits[None, :] << shift)
        counts = jnp.sum(key[:, None, :] >= candidates[:, :, None], axis=-1, dtype=jnp.int32)
        digit = jnp.sum(counts >= k[:, None], axis=-1, dtype=jnp.uint32)
        prefix = prefix | (digit << shift)
    return from_ordered_key(prefix)


def _radix_select_kernel(x_ref, k_ref, out_ref, *, columns: int, block: int, radix_bits: int):
    row = pl.program_id(0)
    k = k_ref[...]
    tiles = columns // block
    # A row of one tile is read once and stays in registers for every pass.
    whole_row = ordered_key(x_ref[row, :]) if tiles == 1 else None

    def one_pass(p, prefix):
        shift = (32 - radix_bits - p * radix_bits).astype(jnp.uint32)
        # Digit 0 is the prefix itself, whose count always reaches k: count only the other candidates.
        candidates = [prefix | (jnp.uint32(d) << shift) for d in range(1, 1 << radix_bits)]
        if whole_row is not None:
            counts = [jnp.sum((whole_row >= cand).astype(jnp.int32)) for cand in candidates]
        else:
            # Each lane keeps its own counts over the tiles, so the threads sum them only once per pass.
            def count_block(j, counts):
                key = ordered_key(x_ref[row, pl.ds(j * block, block)])
                return tuple(c + (key >= cand).astype(jnp.int32) for c, cand in zip(counts, candidates, strict=True))

            zeros = tuple(jnp.zeros((block,), jnp.int32) for _ in candidates)
            counts = [jnp.sum(c) for c in jax.lax.fori_loop(0, tiles, count_block, zeros)]
        digit = sum((c >= k).astype(jnp.uint32) for c in counts)
        return prefix | (digit << shift)

    prefix = jax.lax.fori_loop(0, 32 // radix_bits, one_pass, jnp.uint32(0))
    out_ref[...] = from_ordered_key(prefix)


@functools.lru_cache(maxsize=None)
def _radix_select_call(
    rows: int, columns: int, block: int, radix_bits: int, num_warps: int, interpret: bool, manual_axis_type
):
    x_spec = jax.ShapeDtypeStruct((rows, columns), jnp.float32)
    k_spec = jax.ShapeDtypeStruct((rows,), jnp.int32)
    # Inside a shard_map the output varies over the same mesh axes as the input.
    out_spec = jax.ShapeDtypeStruct((rows,), jnp.float32, manual_axis_type=manual_axis_type)
    cost = with_io_bytes_accessed(
        pl.estimate_cost(functools.partial(radix_select_reference, radix_bits=radix_bits), x_spec, k_spec),
        kernel_inputs_specs=(x_spec, k_spec),
        kernel_outputs_specs=out_spec,
    )
    return pl.pallas_call(
        functools.partial(_radix_select_kernel, columns=columns, block=block, radix_bits=radix_bits),
        out_shape=out_spec,
        in_specs=[pl.no_block_spec, pl.BlockSpec((None,), lambda r: (r,))],
        out_specs=pl.BlockSpec((None,), lambda r: (r,)),
        grid=(rows,),
        compiler_params=plgpu.CompilerParams(num_warps=num_warps, num_stages=1),
        cost_estimate=cost,
        interpret=interpret,
    )


def radix_select(
    x: Float[Array, "R N"],
    k: Int[Array, ""] | Int[Array, "R"] | int,
    *,
    radix_bits: int = DEFAULT_RADIX_BITS,
    block: int = DEFAULT_BLOCK,
    num_warps: int = DEFAULT_NUM_WARPS,
    interpret: bool = False,
) -> Float[Array, "R"]:
    """:func:`radix_select_reference` as one Pallas-Triton kernel, one program per row.

    Same result, bit for bit. Each program reads its row in ``block``-column tiles (at most the row length rounded up
    to a power of two; the row is padded to a whole number of tiles): a row of one tile once, kept in registers for
    every pass, a longer row once per pass. ``x`` gets no gradient. The call runs on whatever ``x`` each device holds:
    call it inside a ``shard_map`` when ``x`` is sharded.

    The kernel is written for the Pallas Triton backend and raises on TPU. On CPU this is
    :func:`radix_select_reference` (the Pallas interpreter cannot run inside a ``shard_map`` that checks varying axes),
    unless ``interpret`` runs the kernel in the interpreter.
    """
    _check_args(x, radix_bits)
    backend = jax.default_backend()
    if backend == "tpu" and not interpret:
        raise ValueError("radix_select is a Pallas-Triton kernel and does not run on TPU; use radix_select_reference")
    if backend == "cpu" and not interpret:
        return radix_select_reference(x, k, radix_bits=radix_bits)
    rows, columns = x.shape
    block = min(block, 1 << math.ceil(math.log2(columns)))
    padded = -(-columns // block) * block
    x = jax.lax.stop_gradient(x.astype(jnp.float32))
    if padded != columns:
        fill = jax.lax.bitcast_convert_type(jnp.uint32(_SMALLEST_KEY_FLOAT_BITS), jnp.float32)
        x = jnp.pad(x, ((0, 0), (0, padded - columns)), constant_values=fill)
    k = jnp.broadcast_to(jnp.asarray(k, jnp.int32), (rows,))
    call = _radix_select_call(rows, padded, block, radix_bits, num_warps, interpret, jax.typeof(x).manual_axis_type)
    return call(x, k)
