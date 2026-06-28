# Copyright The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Guards that lock the genuine-mixed-fp8 invariant for the ragged-dot FP8 path.

Two levels, mirroring the two ways the recipe can silently regress:

* :func:`assert_fp8_contraction` — a trace-time check that each grouped GEMM feeds the
  expected f8 dtype pair. Catches an operand degrading to bf16 (a dequantize/QDQ fallback)
  or the output gradient collapsing from E5M2 to E4M3. Runs at trace time, so it costs
  nothing at runtime.
* :func:`fp8_ragged_lowered_text` + :func:`lowering_contains_fp8` — the lowered StableHLO of
  a call and a predicate over it, for asserting in tests that the genuine f8 operands survive
  lowering rather than being pre-dequantized to a bf16 GEMM.

The invariant is recorded in the project memory ``grug-fp8-ragged-mixed-required``: the
backward must contract E5M2 output-grad against the E4M3 weight/activation, never the
same-dtype E5M2×E5M2 or all-E4M3 shortcut, and the forward stays E4M3×E4M3.
"""

from enum import StrEnum

import jax
import jax.numpy as jnp

_E4M3 = jnp.dtype(jnp.float8_e4m3fn)
_E5M2 = jnp.dtype(jnp.float8_e5m2)


class Fp8Contraction(StrEnum):
    """Which grouped GEMM of the FP8 ragged dot is being checked."""

    FORWARD = "forward"  # E4M3 act × E4M3 weight
    DLHS = "dlhs"  # output-grad × E4M3 weight  → input grad
    DRHS = "drhs"  # E4M3 act × output-grad     → weight grad


def _expected_pair(
    contraction: Fp8Contraction, grad_dtype: jnp.dtype
) -> tuple[jnp.dtype, jnp.dtype]:
    """The (lhs, rhs) dtype pair a genuine-fp8 grouped GEMM must contract for ``contraction``.

    The forward is always all-E4M3; the two backward dots contract the output gradient
    (``grad_dtype`` — E5M2 in the hybrid recipe) against the E4M3 weight/activation.
    """
    if contraction == Fp8Contraction.FORWARD:
        return (_E4M3, _E4M3)
    if contraction == Fp8Contraction.DLHS:
        return (grad_dtype, _E4M3)
    if contraction == Fp8Contraction.DRHS:
        return (_E4M3, grad_dtype)
    raise ValueError(f"unknown contraction {contraction!r}")


def assert_fp8_contraction(
    lhs: jax.Array,
    rhs: jax.Array,
    *,
    contraction: Fp8Contraction,
    grad_dtype,
) -> None:
    """Assert a genuine-fp8 grouped GEMM contracts the dtype pair its recipe requires.

    A mismatch means the mixed-fp8 recipe has silently regressed — an operand fell back to
    bf16 (a QDQ/``__triton_gemm`` dequant), or the output gradient collapsed from E5M2 to
    E4M3. Either would defeat the point of the f8 backward, so this fails loudly at trace
    time (``grug-fp8-ragged-mixed-required``).
    """
    expected = _expected_pair(contraction, jnp.dtype(grad_dtype))
    actual = (lhs.dtype, rhs.dtype)
    if actual != expected:
        raise AssertionError(
            f"fp8 ragged {contraction} expected genuine f8 operands {expected}, got {actual}; "
            "a bf16/QDQ fallback or an all-E4M3 collapse would break the mixed-fp8 recipe "
            "(grug-fp8-ragged-mixed-required)"
        )


def fp8_ragged_lowered_text(fn, *args, **kwargs) -> str:
    """StableHLO text of ``fn`` lowered at ``args`` — for asserting in tests that the genuine
    f8 operands survive lowering (no dequantize-to-bf16 fallback before the GEMM)."""
    return jax.jit(fn).lower(*args, **kwargs).as_text()


def lowering_contains_fp8(text: str) -> bool:
    """True if the lowered text references an f8 type, i.e. operands reach the GEMM still
    quantized rather than pre-dequantized to bf16.

    Case-insensitive so it matches both the StableHLO spelling (``f8E4M3FN``/``f8E5M2``) and the
    lowercase PTX form the Mosaic-GPU wgmma path emits (``f8e4m3``/``f8e5m2``).
    """
    lowered = text.lower()
    return "f8e4m3" in lowered or "f8e5m2" in lowered
