# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from decimal import Decimal, localcontext

import ml_dtypes
import numpy as np

from experiments.benchmarks.diagnose_sigmoid_precision import error_summary, sigmoid_reference


def test_sigmoid_reference_resolves_observed_rounding_and_subnormal_outputs():
    inputs = np.asarray([-93, -92.5, -89, -87.5, -1, -0.0194091796875, 0, 1, 6.25, 93], dtype=np.float32)
    with localcontext() as context:
        context.prec = 80
        expected = np.asarray([float(1 / (1 + (-Decimal(float(x))).exp())) for x in inputs])
    reference = sigmoid_reference(inputs)
    np.testing.assert_array_equal(reference.astype(ml_dtypes.bfloat16), expected.astype(ml_dtypes.bfloat16))
    # Actual H100 default-BF16 sigmoid at the captured input, compared with the independent reference.
    captured = np.asarray([0.4921875], dtype=np.float32)
    summary = error_summary(inputs[5:6], captured, reference[5:6])
    assert summary["regions"]["all_finite_inputs"]["rounded_reference_mismatches"] == 1
    assert summary["largest_bf16_step_examples"][0]["rounded_bf16_reference"] == 0.49609375
    assert summary["largest_bf16_step_examples"][0]["bf16_steps"] == 2
