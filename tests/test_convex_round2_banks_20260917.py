# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A convex average must align policy coordinates and preserve missing labels."""

import numpy as np
import pandas as pd
import pytest

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    validate_convex_round2_banks_20260917 as banks,
)


def test_average_aligns_coordinates_before_selecting_policy():
    frame = pd.DataFrame(
        {
            "model": ["a", "a", "b", "b"],
            "coordinate_id": ["x", "y", "y", "x"],
            "measured": [1.0, np.nan, np.nan, 1.0],
            "prediction": [0.2, 0.8, 0.6, 0.4],
            "predicted_delta": [0.0, 0.6, 0.2, 0.0],
        }
    )
    result = banks.equal_average(frame, ("a", "b"), ["coordinate_id"]).set_index("coordinate_id")
    np.testing.assert_allclose(result.prediction, [0.3, 0.7])
    np.testing.assert_allclose(result.predicted_delta, [0.0, 0.4])
    assert result.prediction.idxmin() == "x"
    assert np.isnan(result.loc["y", "measured"])


def test_average_rejects_different_candidate_support():
    frame = pd.DataFrame(
        {
            "model": ["a", "a", "b"],
            "coordinate_id": ["x", "y", "x"],
            "measured": [1.0, 2.0, 1.0],
            "prediction": [0.2, 0.8, 0.4],
        }
    )
    with pytest.raises(ValueError, match="identical prediction support"):
        banks.equal_average(frame, ("a", "b"), ["coordinate_id"])
