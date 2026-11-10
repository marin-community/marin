# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Pallas-Triton row gathers for the Grug MoE exchange's backward passes.

XLA runs the ``ring_filtered`` backward's row gathers as loop fusions at about 2.5-2.9 TB/s on MI355X: the dispatch
backward's gather-sum (each received row sums its at most ``K`` dispatch rows' cotangents) and the sender combine
backward's masked gather (each send row takes its token's cotangent). The kernel here reads whole rows with indexed
loads instead and writes each output tile once.

It reproduces the XLA formulations bit for bit: :func:`gather_sum_rows` adds the ``K`` rows in float32 in slot order
and rounds once, like ``ep_ring._gather_sum_slots`` after XLA drops its ``0 +``, and :func:`gather_valid_rows` selects rows
without arithmetic, like ``ep_ring._take_valid_rows``.

The kernel is Pallas-Triton: it runs on GPUs, and on CPU in the Pallas interpreter (for tests); other backends raise.
``ring_filtered`` calls it on shard-local arrays inside its ``shard_map``, so the wrappers take local arrays only.
"""

import functools
import math
from typing import Literal

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu
from jaxtyping import Array, Bool, Float, Int

MoeGatherSum = Literal["xla", "pallas"]
"""How ``ring_filtered`` runs its backward row gathers: XLA fusions ("xla") or this module's kernel ("pallas")."""

# Output rows and hidden columns per program, and warps: the best of a sweep at [119904, 2560] bf16 with K=4 on
# MI355X (0.277 ms against 0.486 ms for XLA's fusion).
_BLOCK_ROWS = 32
_MAX_BLOCK_COLS = 512
_MIN_BLOCK_COLS = 16
_NUM_WARPS = 4


def _block_cols(hidden: int) -> int | None:
    """The largest power of two up to ``_MAX_BLOCK_COLS`` that divides ``hidden``, or None below the minimum."""
    cols = math.gcd(hidden, _MAX_BLOCK_COLS)
    cols = 1 << (cols.bit_length() - 1)
    return cols if cols >= _MIN_BLOCK_COLS else None


def resolve_interpret(interpret: bool | None) -> bool:
    """``interpret=None``: compile with Triton on GPU and use the Pallas interpreter on CPU; other backends raise."""
    if interpret is not None:
        return interpret
    backend = jax.default_backend()
    if backend not in ("gpu", "cpu"):
        raise ValueError(f"the Pallas-Triton MoE kernels need a GPU (or the CPU interpreter), got backend {backend!r}")
    return backend == "cpu"


def _gather_rows_kernel(src_ref, slots_ref, out_ref, *, block_rows: int, block_cols: int, accumulate: bool):
    num_src = src_ref.shape[0]
    num_out = out_ref.shape[0]
    row0 = pl.program_id(0) * block_rows
    col0 = pl.program_id(1) * block_cols
    out_rows = row0 + jnp.arange(block_rows)
    in_bounds = out_rows < num_out
    cols = col0 + jnp.arange(block_cols)
    acc = None
    for k in range(slots_ref.shape[1]):
        # A scalar array index: the interpreter's discharge rule rejects a Python int next to a slice.
        slot = plgpu.load(slots_ref.at[pl.ds(row0, block_rows), jnp.int32(k)], mask=in_bounds, other=num_src)
        keep = in_bounds & (slot >= 0) & (slot < num_src)
        src_rows = jnp.clip(slot, 0, num_src - 1)
        # Indexing both dimensions with arrays (rather than rows with an array and columns with a slice) lowers to
        # the same pointer tile and also runs in the interpreter.
        rows = plgpu.load(src_ref.at[src_rows[:, None], cols[None, :]], mask=keep[:, None], other=0)
        if not accumulate:
            acc = rows
        elif acc is None:
            # XLA simplifies the reference's `zeros + rows` to `rows`, which keeps a -0.0; so does starting here.
            acc = rows.astype(jnp.float32)
        else:
            acc = acc + rows.astype(jnp.float32)
    plgpu.store(
        out_ref.at[pl.ds(row0, block_rows), pl.ds(col0, block_cols)],
        acc.astype(out_ref.dtype),
        mask=in_bounds[:, None],
    )


def _gather_rows(
    src: Float[Array, "P H"], slots: Int[Array, "T K"], *, accumulate: bool, interpret: bool | None
) -> Float[Array, "T H"]:
    num_src, hidden = src.shape
    num_out, topk = slots.shape
    block_cols = _block_cols(hidden)
    if block_cols is None:
        raise ValueError(f"the Pallas gather needs a hidden width with a power-of-two divisor >= 16, got {hidden}")
    if not accumulate and topk != 1:
        raise ValueError(f"a gather without accumulation takes one slot per row, got {topk}")
    interpret = resolve_interpret(interpret)
    itemsize = jnp.dtype(src.dtype).itemsize
    return pl.pallas_call(
        functools.partial(_gather_rows_kernel, block_rows=_BLOCK_ROWS, block_cols=block_cols, accumulate=accumulate),
        grid=(pl.cdiv(num_out, _BLOCK_ROWS), hidden // block_cols),
        in_specs=[pl.no_block_spec, pl.no_block_spec],
        out_specs=pl.no_block_spec,
        out_shape=jax.ShapeDtypeStruct((num_out, hidden), src.dtype),
        compiler_params=plgpu.CompilerParams(num_warps=_NUM_WARPS, num_stages=1),
        cost_estimate=pl.CostEstimate(
            flops=num_out * topk * hidden if accumulate else 0,
            transcendentals=0,
            bytes_accessed=(topk + 1) * num_out * hidden * itemsize + num_out * topk * 4,
        ),
        interpret=interpret,
        name="grug_gather_sum" if accumulate else "grug_gather_rows",
    )(src, slots.astype(jnp.int32))


def gather_sum_rows(
    rows: Float[Array, "P H"], slots: Int[Array, "T K"], *, interpret: bool | None = None
) -> Float[Array, "T H"]:
    """``out[t] = sum_k rows[slots[t, k]]``, summed in float32 in ``k`` order and rounded once.

    Slots outside ``[0, P)`` add nothing. Equals ``ep_ring._gather_sum_slots`` bit for bit for non-negative slots
    (``jnp.take`` wraps negative ones). See :func:`resolve_interpret` for ``interpret``.
    """
    return _gather_rows(rows, slots, accumulate=True, interpret=interpret)


def gather_valid_rows(
    x: Float[Array, "T H"], index: Int[Array, "P"], valid: Bool[Array, "P"], *, interpret: bool | None = None
) -> Float[Array, "P H"]:
    """``where(valid, x[index], 0)`` row by row, without arithmetic: ``ep_ring._take_valid_rows`` bit for bit.

    ``index`` must lie in ``[0, T)`` where ``valid``. See :func:`resolve_interpret` for ``interpret``.
    """
    slots = jnp.where(valid, index.astype(jnp.int32), x.shape[0])[:, None]
    return _gather_rows(x, slots, accumulate=False, interpret=interpret)
