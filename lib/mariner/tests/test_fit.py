# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from mariner.fit import final_inner_folds, fit_objective


def test_fit_objective_recovers_generating_model_on_fresh_mixtures(synthetic_problem):
    fit = fit_objective(synthetic_problem.swarm, synthetic_problem.objective, synthetic_problem.anchors)
    fresh = np.random.default_rng(1).dirichlet(np.ones(6), size=40)
    error = fit.predict_tasks(fresh) - synthetic_problem.truth.predict_tasks(fresh)
    assert np.sqrt(np.mean(error**2, axis=0)).max() < 0.002  # below the injected noise


def test_final_inner_folds_never_validate_calibration_rows(synthetic_problem):
    swarm = synthetic_problem.swarm
    folds = final_inner_folds(swarm)
    assert all(0 in train and len(validation) > 0 for train, validation in folds)
    validated = np.concatenate([validation for _, validation in folds])
    # Every run but the calibration row 0 is validated exactly once.
    np.testing.assert_array_equal(np.sort(validated), np.arange(1, len(swarm.runs)))
