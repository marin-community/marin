# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pytest

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    analyze_mariner_policy_banks_20260917 as analysis,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    validate_mariner_policy_banks_20260917 as validation,
)


def test_selection_distinguishes_best_of_k_from_average_of_k():
    bank = pd.DataFrame(
        {
            "coordinate_id": ["a", "b", "c", "d", "e", "f"],
            "measured": [1.2, 1.5, 1.4, 1.3, 1.0, 1.1],
            "prediction": [0.9, 1.0, 1.1, 1.2, 1.3, 1.4],
        }
    )
    result = validation.selection_metrics(bank.sample(frac=1, random_state=10))
    assert result["selected_id"] == "a"
    assert result["regret_at_1"] == pytest.approx(0.2)
    assert result["regret_at_5"] == 0
    assert result["selected_optimism"] == pytest.approx(0.3)
    assert result["selected_rank"] == 3


def test_shared_source_coordinate_is_in_each_cohort_once():
    bank = pd.DataFrame(
        {
            "sources": [
                "validation::a;archive::b",
                "validation::a",
                "validation::a",
                "archive::b",
                "archive::b",
                "conditional_epoch_dose_response",
            ]
        }
    )
    masks = dict(validation.source_masks(bank))
    assert masks["all"].sum() == 6
    np.testing.assert_array_equal(masks["validation_cohort"], [True, True, True, False, False, False])
    np.testing.assert_array_equal(masks["archive::b"], [True, False, False, True, True, False])


def test_identical_predictions_tie_for_every_bootstrap_menu():
    frames = []
    for model in validation.MODEL_IDS:
        frames.append(
            pd.DataFrame(
                {
                    "model": model,
                    "coordinate_id": ["a", "b", "c", "d", "e"],
                    "sources": ["validation::a"] * 5,
                    "measured": [1.4, 1.1, 1.6, 1.3, 1.2],
                    "prediction": [1.2, 1.0, 1.3, 1.1, 1.5],
                }
            )
        )
    rows = validation.paired_bootstrap(pd.concat(frames, ignore_index=True))
    assert len(rows) == 4
    for row in rows:
        assert row["tie_fraction"] == 1
        assert row["regret_delta_mean"] == 0
        assert row["rmse_delta_mean"] == 0


@pytest.mark.parametrize("sign", [-1, 1])
def test_uneven_epoch_jensen_check_uses_interpolation_weights(sign, tmp_path):
    epochs = np.array([0.0, 1.0, 3.0])
    frame = pd.DataFrame(
        {
            "model": [validation.MODEL_IDS[0]] * 3,
            "target": ["uncheatable"] * 3,
            "focal_domain": ["example"] * 3,
            "run_name": ["a", "b", "c"],
            "target_simulated_epochs": epochs,
            "weight::example": epochs / 3,
            "measured": sign * epochs**2,
            "prediction": sign * epochs**2,
            "in_fit_panel": [False] * 3,
            "observed_complete": [True] * 3,
            "affine_max_abs_error": [0.0] * 3,
        }
    )
    result = analysis.convexity_audit(frame, tmp_path)
    assert len(result) == 1
    assert result.measured_convexity_violation.iloc[0] == pytest.approx(-2 * sign)
    assert result.raw_secant_decrease.iloc[0] == pytest.approx(-3 * sign)
    assert result.all_heldout.iloc[0]
