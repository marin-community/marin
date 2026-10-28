# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import json

import numpy as np
from mariner.fit import fit_objective
from mariner.model import ObjectiveFit, design_matrix

FIT_KEYS = {"objective", "buckets", "epochs_per_unit_weight", "task_weights", "tasks", "inner_fold_labels"}
TASK_KEYS = {
    "task",
    "rho",
    "kappa",
    "tau",
    "gamma",
    "nu",
    "phi",
    "flat_profile",
    "inner_cv_rmse",
    "intercept",
    "coefficients",
}


def test_objective_fit_json_round_trip_predicts_identically(synthetic_problem):
    swarm = synthetic_problem.swarm
    fit = fit_objective(swarm, synthetic_problem.objective, synthetic_problem.anchors)
    payload = json.loads(json.dumps(fit.to_json()))
    loaded = ObjectiveFit.from_json(payload)
    np.testing.assert_array_equal(loaded.predict_tasks(swarm.weights), fit.predict_tasks(swarm.weights))
    assert loaded.to_json() == payload
    # The fit file is a public format: its field names, each run's inner fold in run order and the calibration
    # label -1.
    assert set(payload) == FIT_KEYS
    assert all(set(task) == TASK_KEYS for task in payload["tasks"])
    assert [run for run, _ in payload["inner_fold_labels"]] == list(swarm.runs)
    assert payload["inner_fold_labels"][0] == [swarm.runs[0], -1]


def test_pool_fractions_divide_exposures_as_in_the_swarm(synthetic_problem):
    swarm, truth = synthetic_problem.swarm, synthetic_problem.truth
    fractions = np.tile(0.5 ** np.arange(1, len(swarm.buckets) + 1), (len(swarm.runs), 1))  # 1/2, 1/4, ... per bucket
    exposures = dataclasses.replace(swarm, pool_fractions=fractions).exposures
    expected = np.column_stack([task.head.predict(design_matrix(exposures, task.shape)) for task in truth.tasks])
    # The two paths sum each prediction's terms in different orders (measured 4.4e-16 apart).
    np.testing.assert_allclose(truth.predict_tasks(swarm.weights, fractions), expected, rtol=1e-13, atol=0)
