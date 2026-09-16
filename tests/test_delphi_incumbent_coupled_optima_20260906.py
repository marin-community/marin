# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from scipy.special import rel_entr

from experiments.domain_phase_mix.exploratory.two_phase_many import optimize_delphi_incumbent_coupling_20260906 as optima


def test_cross_objective_gains_keep_each_models_preferred_policy_orientation():
    first = np.array([0.8, 0.2])
    second = np.array([0.3, 0.7])
    natural = np.array([0.6, 0.4])
    result = optima.compare_policies(
        lambda query: query[:, 1], lambda query: 3 * query[:, 0], first, second, natural, 0.02
    )
    penalty_difference = 0.02 * (rel_entr(first, natural).sum() - rel_entr(second, natural).sum())
    assert result["between_model_tv"] == pytest.approx(0.5)
    assert result["kappa1_own_objective_gain"] == pytest.approx(1.5 + penalty_difference)
    assert result["kappa0_own_objective_gain"] == pytest.approx(0.5 - penalty_difference)
    assert result["kappa0_bpb_at_kappa1"] == pytest.approx(0.7)
    assert result["kappa1_bpb_at_kappa0"] == pytest.approx(2.4)


def test_restart_selection_excludes_failed_minimum_and_measures_flat_policy_spread():
    rows = [
        ({"objective": value, "success": success, "start": index}, np.array([weight, 1 - weight]))
        for index, (value, success, weight) in enumerate(
            [(1.0, True, 0.2), (1.0 + 5e-7, True, 0.4), (1.0 + 2e-5, True, 0.8), (0.99, False, 0.9)]
        )
    ]
    selected, weights, spread = optima.choose_restart(rows)
    assert selected["start"] == 0
    np.testing.assert_allclose(weights, [0.2, 0.8], atol=1e-14)
    assert spread["near_optimal_restarts"] == 2
    assert spread["near_optimal_max_tv_to_selected"] == pytest.approx(0.2)
    assert spread["converged_max_tv_to_selected"] == pytest.approx(0.6)
    assert spread["converged_objective_spread"] == pytest.approx(2e-5)
