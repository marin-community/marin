# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from levanter.testing.precision import round_to_bfloat16


def test_bfloat16_rounding_at_every_finite_midpoint():
    # BF16 endpoints embed exactly in FP32. Construct midpoint expectations
    # from adjacent bit patterns, independently of frexp/significand rounding.
    bits = np.arange(0x7F7F, dtype=np.uint32)
    lower = (bits << 16).view(np.float32).astype(np.float64)
    upper = ((bits + 1) << 16).view(np.float32).astype(np.float64)
    midpoint = (lower + upper) / 2
    inputs = np.stack((np.nextafter(midpoint, -np.inf), midpoint, np.nextafter(midpoint, np.inf)))
    expected = np.stack((lower, np.where(bits % 2 == 0, lower, upper), upper))
    # Includes powers of two, where the spacing changes, and all subnormals.
    np.testing.assert_array_equal(round_to_bfloat16(inputs), expected)
    np.testing.assert_array_equal(round_to_bfloat16(-inputs), -expected)


def test_bfloat16_rounding_zero_overflow_and_nonfinite():
    overflow_midpoint = float.fromhex("0x1.ffp127")
    maximum = float.fromhex("0x1.fep127")
    inputs = np.array(
        [0.0, np.nextafter(0.0, 1.0), np.nextafter(overflow_midpoint, 0.0), overflow_midpoint, np.inf, np.nan]
    )
    expected = np.array([0.0, 0.0, maximum, np.inf, np.inf, np.nan])
    np.testing.assert_array_equal(round_to_bfloat16(inputs), expected)
    np.testing.assert_array_equal(round_to_bfloat16(-inputs), -expected)
    assert np.signbit(round_to_bfloat16(np.array([-0.0, -np.nextafter(0.0, 1.0)]))).all()
