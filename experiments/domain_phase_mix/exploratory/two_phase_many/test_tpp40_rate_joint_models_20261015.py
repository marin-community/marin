# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import numpy as np
import pytest

from experiments.domain_phase_mix.exploratory.two_phase_many import tpp40_joint_estimation_20261014 as frozen
from experiments.domain_phase_mix.exploratory.two_phase_many import tpp40_rate_joint_models_20261015 as models

SHAPE = {"rate": 0.25, "power": 0.7, "threshold": 3.0}
ALPHA = 0.8


def geometry() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20261015)
    total = np.exp(rng.uniform(-2, 3, (36, 3)))
    fraction = rng.uniform(0.05, 0.95, total.shape)
    return np.tile(total * fraction, (3, 1)), np.tile(total * (1 - fraction), (3, 1)), np.repeat(np.arange(3), 36)


@pytest.mark.parametrize("parameters", ({"kappa": 128.0, "nu": 2.0}, {"kappa": 0.0, "nu": 0.0}))
def test_combined_law_tied_and_null_restriction(parameters: dict[str, float]) -> None:
    early, late, _ = geometry()
    total = early + late
    matrix, _ = models.design_and_offset(
        ALPHA * total, (1 - ALPHA) * total, ALPHA, SHAPE, "bounded_benefit_rate", parameters
    )
    np.testing.assert_array_equal(matrix, frozen.mariner.design_matrix(ALPHA * total + (1 - ALPHA) * total, SHAPE))
    null, _ = models.design_and_offset(early, late, ALPHA, SHAPE, "bounded_benefit_rate", {"kappa": 0.0, "nu": 0.0})
    np.testing.assert_array_equal(null, frozen.mariner.design_matrix(total, SHAPE))


def test_refined_controls_are_rescored_and_refit_instead_of_borrowing_heads() -> None:
    early, late, labels = geometry()
    parameters = {"kappa": 0.5, "nu": 0.0}
    matrix, _ = models.design_and_offset(early, late, ALPHA, SHAPE, "bounded_benefit_odds", parameters)
    head = frozen.mariner.Head(1.0, np.array([0.5, 0.8, 0.6, 0.03, 0.02, 0.01]), 0.6)
    observed = head.predict(matrix)
    gamma = float((np.median(observed) - 0.6) / (np.median(observed) - observed.min()))
    control = models.JointFit(
        frozen.JointFit(
            "bounded_benefit_odds",
            ALPHA,
            SHAPE,
            parameters,
            0.0,
            gamma,
            None,
            0.0,
            head,
            (),
            {"selected_from_failed_refinement": False},
        )
    )
    grid = models.TestGrid(({"kappa": 0.0, "nu": 0.0}, parameters, {"kappa": 0.5, "nu": 1.0}), (0.0,), (1.0, 2.5, 6.0))
    base = models.fit_joint(
        early, late, observed, labels, ALPHA, "bounded_benefit_rate", shapes=(SHAPE,), test_grid=grid
    )
    fit = models.retain_controls(base, (control,), early, late, observed, labels)
    poisoned = models.JointFit(
        replace(control.selected, inner_rmse=-100.0, head=frozen.mariner.Head(10.0, np.ones(6), -9.0))
    )
    replay = models.retain_controls(base, (poisoned,), early, late, observed, labels)
    assert fit.inner_rmse < 1e-12
    assert fit.inner_rmse <= base.inner_rmse
    np.testing.assert_allclose(fit.predict(early, late), observed, atol=2e-12, rtol=2e-12)
    np.testing.assert_array_equal(replay.predict(early, late), fit.predict(early, late))
    assert replay.inner_rmse == fit.inner_rmse
    np.testing.assert_allclose(
        fit.search_space["retained_control_candidates"][0]["inner_predictions"], observed, atol=2e-12, rtol=2e-12
    )
    restored = models.JointFit.from_json(json.loads(json.dumps(fit.to_json(), allow_nan=False)))
    np.testing.assert_array_equal(restored.predict(early, late), fit.predict(early, late))


def process_replay(fit: models.JointFit, early: np.ndarray, late: np.ndarray) -> np.ndarray:
    return models.JointFit.from_json(fit.to_json()).predict(early, late)


def test_isolated_backend_preserves_frozen_fit_and_spawn_replay() -> None:
    early, late, labels = geometry()
    head = frozen.mariner.Head(1.0, np.array([0.5, 0.8, 0.6, 0.03, 0.02, 0.01]), 0.6)
    observed = head.predict(frozen.mariner.design_matrix(early + late, SHAPE))
    native_grid = frozen.TestGrid(({},), (0.01,), (1.0, 6.0))
    before = frozen.fit_joint(
        early, late, observed, labels, ALPHA, "aggregate_mariner", shapes=(SHAPE,), test_grid=native_grid
    )
    fit = models.fit_joint(
        early,
        late,
        observed,
        labels,
        ALPHA,
        "aggregate_mariner",
        shapes=(SHAPE,),
        test_grid=models.TestGrid(({"kappa": 0.0, "nu": 0.0},), (0.01,), (1.0, 6.0)),
    )
    after = frozen.fit_joint(
        early, late, observed, labels, ALPHA, "aggregate_mariner", shapes=(SHAPE,), test_grid=native_grid
    )
    np.testing.assert_array_equal(before.predict(early, late), after.predict(early, late))
    np.testing.assert_array_equal(fit.predict(early, late), before.predict(early, late))
    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) as executor:
        replay = executor.submit(process_replay, fit, early, late).result(timeout=30)
    np.testing.assert_array_equal(replay, fit.predict(early, late))
