# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["joblib", "numpy", "pandas", "scipy"]
# ///

"""Extend the independent artifact audit to raw and softplus convex links."""

import dataclasses
import json

import numpy as np

from experiments.domain_phase_mix.exploratory.two_phase_many import audit_mariner_harm_screen_20260917 as audit
from experiments.domain_phase_mix.exploratory.two_phase_many import benchmark_single_phase_observatory_20260902 as bench
from experiments.domain_phase_mix.exploratory.two_phase_many import mariner_convex_round2_models_20260917 as candidates
from experiments.domain_phase_mix.exploratory.two_phase_many import screen_mariner_convex_round2_20260917 as runner

CANONICAL_GRID_CHECK = audit.check_selected_grid
CANONICAL_REPLAY = audit.replay_selected


def check_grid(task: bench.FitTask, panel: bench.BenchPanel, saved: dict) -> None:
    features, model = audit.component_model(task, panel)
    if not isinstance(model, candidates.LinkedConvexModel):
        CANONICAL_GRID_CHECK(task, panel, saved)
        return
    grid = model.base
    shapes = grid.candidate_shapes(features)
    table = saved["cv_table"]
    audit.require(table.shape == (len(shapes), len(grid.ridge_grid)), "Custom link CV table dimensions differ")
    i, j = np.unravel_index(int(np.argmin(table)), table.shape)
    audit.require(json.loads(str(saved["shape_json"])) == dict(shapes[i]), "Wrong custom-link selected shape")
    audit.assert_close(float(saved["ridge"]), grid.ridge_grid[j], "Wrong custom-link selected ridge")
    diagnostics = json.loads(str(saved["diagnostics_json"]))
    audit.assert_close(float(table[i, j]), diagnostics["selection_inner_cv_rmse"], "Wrong selected grid score")


def replay(task: bench.FitTask, panel: bench.BenchPanel, split, saved: dict) -> dict:
    features, model = audit.component_model(task, panel)
    if not isinstance(model, candidates.LinkedConvexModel):
        return CANONICAL_REPLAY(task, panel, split, saved)
    shape = json.loads(str(saved["shape_json"]))
    diagnostics = json.loads(str(saved["diagnostics_json"]))
    response = panel.group(task.target).outcomes[:, task.component_index]
    matrix = model.base.design(features, shape)
    ridge = float(saved["ridge"])
    scores = []
    for kappa in (model.base.head.floor_kappa_prior, float(diagnostics["kappa"])):
        squared_error, count = 0.0, 0
        for train, validation in split.inner:
            head = model.selected_head(features, response, train, shape, ridge, kappa)
            linear = head.linear.intercept + matrix.values[validation] @ head.linear.coefficients
            prediction = (
                np.maximum(linear, 0.0) if model.link == "raw" else head.floor + head.scale * np.logaddexp(0.0, linear)
            )
            squared_error += float(np.sum((prediction - response[validation]) ** 2))
            count += len(validation)
        scores.append(float(np.sqrt(squared_error / count)))
    audit.assert_close(scores[0], diagnostics["selection_inner_cv_rmse"], "Replayed custom-link grid CV")
    audit.assert_close(scores[1], diagnostics["inner_cv_rmse"], "Replayed custom-link final CV")
    fitted = runner.load_fitted(saved)
    audit.assert_close(
        model.predict(fitted, features, split.train), saved["train_prediction"], "Portable custom train head"
    )
    audit.assert_close(model.predict(fitted, features, split.test), saved["prediction"], "Portable custom test head")
    return dataclasses.asdict(task) | {
        "selected_grid_cv_rmse": scores[0],
        "final_cv_rmse": scores[1],
        "heads_refitted": 2 * len(split.inner),
        "max_prediction_difference": 0.0,
    }


def main() -> None:
    # The existing audit owns exact input/row coverage and independent metric
    # reconstruction. These adapters add the two explicitly registered link types.
    audit.runner.DEFAULT_OUTPUT = runner.DEFAULT_OUTPUT
    audit.runner.register_candidates = runner.register_candidates
    audit.check_selected_grid = check_grid
    audit.replay_selected = replay
    audit.main()


if __name__ == "__main__":
    main()
