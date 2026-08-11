# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import ml_dtypes
import numpy as np
import pytest

from tile_lifetime.bfloat16_metrics import bfloat16_ulp_distance

BFLOAT16 = ml_dtypes.bfloat16


def _from_bits(*bits: int) -> np.ndarray:
    return np.asarray(bits, dtype=np.uint16).view(BFLOAT16)


def test_every_finite_bfloat16_encoding_is_monotone_in_signed_ulp_order() -> None:
    values = np.arange(1 << 16, dtype=np.uint16).view(BFLOAT16)
    finite = values[np.isfinite(values)]
    ordered = finite[np.argsort(finite.astype(np.float32), kind="stable")]

    adjacent = bfloat16_ulp_distance(ordered[:-1], ordered[1:])

    assert set(np.unique(adjacent)) == {0, 1}
    assert np.count_nonzero(adjacent == 0) == 1  # -0 and +0 share one numerical rank.


def test_bfloat16_ulp_distance_covers_zero_subnormal_normal_and_infinity_boundaries() -> None:
    assert bfloat16_ulp_distance(_from_bits(0x8000), _from_bits(0x0000)).item() == 0
    assert bfloat16_ulp_distance(_from_bits(0x0000), _from_bits(0x0001)).item() == 1
    assert bfloat16_ulp_distance(_from_bits(0x8001), _from_bits(0x0001)).item() == 2
    assert bfloat16_ulp_distance(_from_bits(0x007F), _from_bits(0x0080)).item() == 1
    assert bfloat16_ulp_distance(_from_bits(0x807F), _from_bits(0x8080)).item() == 1
    assert bfloat16_ulp_distance(_from_bits(0x7F7F), _from_bits(0x7F80)).item() == 1
    assert bfloat16_ulp_distance(_from_bits(0xFF7F), _from_bits(0xFF80)).item() == 1
    assert bfloat16_ulp_distance(_from_bits(0xBF80), _from_bits(0x3F80)).item() == 32512


def test_bfloat16_ulp_distance_rounds_reference_ties_to_bfloat16() -> None:
    left = np.asarray([1.0, 1.015625], dtype=BFLOAT16)
    reference = np.asarray([1.00390625, 1.01171875], dtype=np.float64)

    assert bfloat16_ulp_distance(left, reference).tolist() == [0, 0]


def test_bfloat16_ulp_distance_rejects_wrong_dtype_shape_and_nan_payloads() -> None:
    with pytest.raises(TypeError, match="BF16 dtype"):
        bfloat16_ulp_distance(np.zeros(2, dtype=np.float32), np.zeros(2, dtype=np.float32))
    with pytest.raises(ValueError, match="identical shapes"):
        bfloat16_ulp_distance(np.zeros(2, dtype=BFLOAT16), np.zeros((1, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="undefined for NaN"):
        bfloat16_ulp_distance(_from_bits(0x7FC1), _from_bits(0x3F80))
