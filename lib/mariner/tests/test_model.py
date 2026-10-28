# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import json

import numpy as np
from mariner.fit import fit_objective
from mariner.model import ObjectiveFit

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
    swarm, objective, anchors, _ = synthetic_problem
    fit = fit_objective(swarm, objective, anchors)
    payload = json.loads(json.dumps(fit.to_json()))
    loaded = ObjectiveFit.from_json(payload)
    np.testing.assert_array_equal(loaded.predict_tasks(swarm.weights), fit.predict_tasks(swarm.weights))
    assert loaded.inner_fold_labels == fit.inner_fold_labels
    # The fit file is a public format: its field names and the calibration label -1.
    assert set(payload) == FIT_KEYS
    assert all(set(task) == TASK_KEYS for task in payload["tasks"])
    assert payload["inner_fold_labels"][swarm.runs[0]] == -1


def test_pool_fraction_divides_exposure(synthetic_problem):
    swarm, _, _, truth = synthetic_problem
    halved = truth.predict_tasks(swarm.weights, pool_fractions=np.full_like(swarm.weights, 0.5))
    doubled = dataclasses.replace(truth, epochs_per_unit_weight=2 * truth.epochs_per_unit_weight)
    np.testing.assert_array_equal(halved, doubled.predict_tasks(swarm.weights))
