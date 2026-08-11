# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""BF16 distance metrics with explicit dtype and shape contracts."""

from typing import Any

import ml_dtypes
import numpy as np

_BFLOAT16_DTYPE = np.dtype(ml_dtypes.bfloat16)


def bfloat16_ulp_distance(left: Any, right: Any) -> np.ndarray:
    """Return signed-order BF16 ULP distances for equal-shaped arrays.

    ``right`` is rounded to BF16 before comparison. NaNs are rejected because
    payload ordering is not a numerical distance. Infinities remain adjacent
    to the corresponding maximum finite values in the BF16 total order.
    """
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    if left_array.dtype != _BFLOAT16_DTYPE:
        raise TypeError("left ULP operand must have BF16 dtype")
    if left_array.shape != right_array.shape:
        raise ValueError("ULP operands must have identical shapes")
    right_bfloat16 = right_array.astype(_BFLOAT16_DTYPE)
    if np.isnan(left_array).any() or np.isnan(right_bfloat16).any():
        raise ValueError("BF16 ULP distance is undefined for NaN values")

    left_bits = left_array.view(np.uint16).astype(np.int32)
    right_bits = right_bfloat16.view(np.uint16).astype(np.int32)
    left_ordered = np.where(left_bits & 0x8000, 0x8000 - (left_bits & 0x7FFF), 0x8000 + left_bits)
    right_ordered = np.where(right_bits & 0x8000, 0x8000 - (right_bits & 0x7FFF), 0x8000 + right_bits)
    return np.abs(left_ordered - right_ordered)
