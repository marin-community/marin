# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiments.domain_phase_mix.exploratory.two_phase_many import audit_two_phase_round2_20260921 as audit


def test_raw_menu_retains_simplex_boundaries_and_physical_exposure_geometry() -> None:
    panel = audit.BenchPanel(
        name="synthetic",
        buckets=("nemotron_full", "starcoder"),
        phase0=np.asarray([[0.8, 0.2]]),
        phase1=np.asarray([[0.2, 0.8]]),
        alpha=0.8,
        c0=np.asarray([0.8, 8.0]),
        c1=np.asarray([0.2, 2.0]),
        targets={},
        rows=pd.DataFrame({"tied": [False]}),
        input_hashes={},
        metadata={},
    )
    menu = audit.raw_menu(panel, 5)
    np.testing.assert_array_equal(menu.phase0[[0, 4, 20, 24]], [[1, 0], [1, 0], [0, 1], [0, 1]])
    np.testing.assert_array_equal(menu.phase1[[0, 4, 20, 24]], [[1, 0], [0, 1], [1, 0], [0, 1]])
    np.testing.assert_array_equal(np.flatnonzero(menu.tied), [0, 6, 12, 18, 24])
    tied = audit.previous_audit.tied_queries(menu)
    np.testing.assert_allclose(menu.epochs, tied.epochs, rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose(menu.phase0_epochs[-1], [0, 8])
    np.testing.assert_allclose(menu.phase1_epochs[-1], [0, 2])


def test_raw_menu_reports_true_value_regret_and_invented_null_gain() -> None:
    frame = pd.DataFrame(
        {
            "truth": [1.0, 1.0, 1.0],
            "prediction": [1.0, 0.9, 1.0],
            "tied_truth": [1.0, 1.0, 1.0],
            "tied_prediction": [1.0, 1.0, 1.0],
            "tied": [True, False, True],
            "phase0_code": [0.0, 0.0, 1.0],
            "phase1_code": [0.0, 1.0, 1.0],
        }
    )
    result = audit.menu_metrics(frame)
    assert result["true_grid_optimum_gain_over_tied"] == 0
    assert result["raw_policy_regret"] == 0
    np.testing.assert_allclose(result["predicted_grid_optimum_gain_over_tied"], 0.1)
    np.testing.assert_allclose(result["selected_optimism"], 0.1)
    np.testing.assert_allclose(result["matched_gain_rmse"], 0.1 / np.sqrt(3))
    frame["truth"] = [1.0, 1.2, 1.0]
    result = audit.menu_metrics(frame)
    np.testing.assert_allclose(result["raw_policy_regret"], 0.2)
    np.testing.assert_allclose(result["true_gain_of_selected_policies"], -0.2)
