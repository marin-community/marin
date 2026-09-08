# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from scipy.special import expit

from experiments.post_training.math_eval.learning_curve import fit_sigmoid


def test_known_sigmoid_curve_recovers_intercept_and_update_scaled_slope():
    updates = [0, 50, 100, 150, 200, 250, 300]
    result = fit_sigmoid(updates, [float(expit(-2 + 1.5 * u / 200)) for u in updates])
    assert result["status"] == "fit"
    assert result["A"] == pytest.approx(-2, abs=1e-6)
    assert result["B"] == pytest.approx(1.5, abs=1e-6)
    assert result["rmse"] < 1e-7


@pytest.mark.parametrize(
    "updates,rates,status",
    [
        ([0, 50, 100], [0.1, 0.2, 0.3], "requires_200_update_span"),
        ([0, 100, 200], [0.1, 0.2, 0.3], "requires_five_checkpoints"),
        ([0, 50, 100, 150, 200], [0, 0, 0, 1, 1], "boundary_only_rates_not_identifiable"),
    ],
)
def test_sigmoid_insufficient_data_is_reported_without_invented_coefficients(updates, rates, status):
    result = fit_sigmoid(updates, rates)
    assert result["status"] == status and "A" not in result
