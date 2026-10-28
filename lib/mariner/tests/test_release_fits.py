# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pytest
from mariner.data import (
    MAIN_SWARM,
    TRAINED_MARINER,
    objective_from_tasks,
    read_table,
    release_anchors,
    release_heldout_bank,
    release_swarm,
    release_tasks,
    released_task_parameters,
    released_task_predictions,
    trained_counts,
)
from mariner.fit import final_inner_folds, fit_objective, fit_task
from mariner.optimize import propose

# Agreed tolerances. The released Qwen3 fit used epoch multipliers a relative 5.0e-7 above the dataset's exact
# D_tgt / P_i, which is the whole gap; the archived swarms were refitted on the dataset's own multipliers.
PREDICTION_TOLERANCE = 1e-6
MAIN_SWARM_TOLERANCES = {"nu": 1e-5, "phi": 2.5e-6, "inner_cv_rmse": 2e-8}
ARCHIVED_TOLERANCE = 1e-9
DISCRETE_FIELDS = ("rho", "kappa", "tau", "gamma")
HELDOUT_EXPOSURES = {"uncheatable": 408, "obe_easy": 247}
ARCHIVED_TASKS = ("uncheatable/ao3_english", "uncheatable/github_python", "mmlu_humanities", "minerva_math_algebra")


@pytest.mark.data_integration
@pytest.mark.timeout(600)
@pytest.mark.parametrize("objective", ["uncheatable", "obe_easy"])
def test_main_swarm_refit_reproduces_release(snapshot, objective):
    swarm = release_swarm(snapshot, MAIN_SWARM, objective)
    target = objective_from_tasks(release_tasks(snapshot), objective)
    fit = fit_objective(swarm, target, release_anchors(snapshot, MAIN_SWARM, objective))
    tasks = list(target.tasks)

    in_sample = released_task_predictions(snapshot, "fits_swarm_predictions", objective)
    assert len(in_sample) == len(swarm.runs)
    released_in_sample = in_sample.loc[list(swarm.runs), tasks].to_numpy()
    in_sample_gap = np.abs(fit.predict_tasks(swarm.weights, swarm.pool_fractions) - released_in_sample)
    assert in_sample_gap.max() <= PREDICTION_TOLERANCE

    bank = release_heldout_bank(snapshot, objective)
    heldout = released_task_predictions(snapshot, "fits_heldout_predictions", objective)
    exposure = read_table(snapshot, "heldout").set_index("observation_id")["exposure_id"]
    by_exposure = heldout.groupby(exposure.loc[heldout.index].to_numpy())
    assert (by_exposure.nunique() == 1).to_numpy().all()  # the release predicts each exposure once
    assert by_exposure.ngroups == len(bank.exposure_ids) == HELDOUT_EXPOSURES[objective]
    released_heldout = by_exposure.first().loc[list(bank.exposure_ids), tasks].to_numpy()
    heldout_gap = np.abs(fit.predict_tasks(bank.weights, bank.pool_fractions) - released_heldout)
    assert heldout_gap.max() <= PREDICTION_TOLERANCE

    fitted = pd.DataFrame([task.to_json() for task in fit.tasks]).set_index("task")
    parameters = released_task_parameters(snapshot, MAIN_SWARM, objective).loc[fitted.index]
    for field in DISCRETE_FIELDS:
        np.testing.assert_array_equal(fitted[field].to_numpy(), parameters[field].to_numpy())
    for field, tolerance in MAIN_SWARM_TOLERANCES.items():
        assert np.abs(fitted[field] - parameters[field]).max() <= tolerance

    proposal = propose(fit, swarm.weights)
    np.testing.assert_array_equal(proposal.counts, trained_counts(snapshot, TRAINED_MARINER[objective]))


@pytest.mark.data_integration
@pytest.mark.timeout(300)
@pytest.mark.parametrize("swarm", ["llama_160m_1p2b", "llama_200m_6b"])
def test_archived_swarm_refit_matches_released_parameters(snapshot, swarm):
    objectives = release_tasks(snapshot).set_index("task_id")["objective"]
    flat_profiles = set()
    for task in ARCHIVED_TASKS:
        objective = objectives[task]
        fit_rows = release_swarm(snapshot, swarm, objective)
        anchor = release_anchors(snapshot, swarm, objective)[task]
        fit = fit_task(fit_rows, task, anchor, final_inner_folds(fit_rows)).to_json()
        parameters = released_task_parameters(snapshot, swarm, objective).loc[task]
        assert [fit[field] for field in DISCRETE_FIELDS] == [parameters[field] for field in DISCRETE_FIELDS]
        for field in ("nu", "phi", "inner_cv_rmse"):
            assert abs(fit[field] - parameters[field]) <= ARCHIVED_TOLERANCE
        flat_profiles.add(fit["flat_profile"])
    assert flat_profiles == {True, False}  # the tasks take both the default nu and the searched one
