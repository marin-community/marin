# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Host rounding for independent numerical oracles."""

import numpy as np

_BFLOAT16_SIGNIFICAND_BITS = 8
_BFLOAT16_MIN_NORMAL_EXPONENT = -126
_BFLOAT16_OVERFLOW_MIDPOINT = float.fromhex("0x1.ffp127")


def round_to_bfloat16(values: np.ndarray) -> np.ndarray:
    """Return nearest-even BF16 values in FP64, without intermediate FP32 rounding.

    NumPy's BF16 cast can round through FP32, losing which side of a BF16
    midpoint an FP64 oracle lies on. Quantizing the FP64 significand directly
    also handles BF16 subnormals, signed zero and overflow ties.
    """
    values = np.asarray(values, np.float64)
    finite = np.isfinite(values) & (np.abs(values) < _BFLOAT16_OVERFLOW_MIDPOINT)
    safe = np.where(finite, values, 0.0)
    _, exponent = np.frexp(safe)
    quantum_exponent = np.maximum(
        exponent - _BFLOAT16_SIGNIFICAND_BITS,
        _BFLOAT16_MIN_NORMAL_EXPONENT - (_BFLOAT16_SIGNIFICAND_BITS - 1),
    )
    rounded = np.ldexp(np.rint(np.ldexp(safe, -quantum_exponent)), quantum_exponent)
    result = np.where(finite, rounded, np.copysign(np.inf, values))
    return np.where(np.isnan(values), values, result)
