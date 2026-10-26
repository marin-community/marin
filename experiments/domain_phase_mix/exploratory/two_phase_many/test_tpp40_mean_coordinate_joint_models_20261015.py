# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Toy-data control retention, training-local floor and process replay contracts."""

import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import numpy as np
import pytest

from experiments.domain_phase_mix.exploratory.two_phase_many import tpp40_joint_estimation_20261014 as frozen
from experiments.domain_phase_mix.exploratory.two_phase_many import tpp40_mean_coordinate_joint_models_20261015 as models

SHAPE = {"rate": 0.25, "power": 0.7, "threshold": 3.0}
ALPHA = 0.8


def geometry() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20261015)
    total = np.exp(rng.uniform(-2, 3, (48, 3)))
    fraction = rng.uniform(0.05, 0.95, total.shape)
    return total * fraction, total * (1 - fraction), np.arange(len(total)) % 3


@pytest.mark.parametrize("family", models.TEMPORAL_FAMILIES)
def test_complete_control_is_freshly_refit_with_local_floors_and_failure_history(family: str) -> None:
    early, late, labels = geometry()
    matrix = frozen.mariner.design_matrix(early + late, SHAPE)
    truth = frozen.mariner.Head(1.0, np.array([0.5, 0.8, 0.6, 0.03, 0.02, 0.01]), 0.6)
    observed = truth.predict(matrix)
    aggregate = models.fit_joint(
        early,
        late,
        observed,
        labels,
        ALPHA,
        "aggregate_mariner",
        shapes=(SHAPE,),
        test_grid=models.TestGrid(({},), (0.0,), (1.0, 6.0)),
    )
    poisoned = models.JointFit(
        replace(
            aggregate.selected,
            inner_rmse=-100.0,
            head=frozen.mariner.Head(10.0, np.ones(6), -9.0),
            search_space={**aggregate.search_space, "selected_from_failed_refinement": True},
        )
    )
    fit = models.fit_joint(
        early,
        late,
        observed,
        labels,
        ALPHA,
        family,
        controls=(poisoned,),
        shapes=(SHAPE,),
        test_grid=models.TestGrid(models.parameter_grid(family), (0.0,), (1.0, 6.0)),
    )
    retained = fit.search_space["retained_control_candidates"][0]
    expected = np.empty(len(observed))
    floors = []
    for train, held in frozen.temporal.inner_splits(labels):
        head = frozen.fit_fixed_head(
            matrix[train], np.zeros(len(train)), observed[train], aggregate.ridge, aggregate.kappa, None
        )
        floors.append(head.floor)
        expected[held] = head.predict(matrix[held])
    assert np.ptp(floors) > 1e-4
    np.testing.assert_allclose(retained["inner_predictions"], expected, atol=1e-13, rtol=1e-13)
    np.testing.assert_array_equal(retained["head"]["coefficients"], aggregate.head.coefficients)
    assert retained["head"]["floor"] == aggregate.head.floor
    assert retained["fresh_inner_rmse"] == aggregate.inner_rmse
    assert fit.inner_rmse <= aggregate.inner_rmse
    assert retained["parameters"] == {"lambda": 0.0}
    assert retained["source_selected_from_failed_refinement"]
    source = fit.search_space["complete_aggregate_source"]
    assert source["parameters"] == {}
    assert source["selection_profile"] == poisoned.to_json()["selection_profile"]
    assert source["search_space"]["refinement_results"] == aggregate.search_space["refinement_results"]
    assert fit.selection_profile[-1]["stage"] == "retained_control_final_union"
    assert fit.selection_profile[-1]["parameter_index"] == 0


def process_replay(fit: models.JointFit, early: np.ndarray, late: np.ndarray) -> np.ndarray:
    return models.JointFit.from_json(json.loads(json.dumps(fit.to_json(), allow_nan=False))).predict(early, late)


@pytest.mark.parametrize("family", models.TEMPORAL_FAMILIES)
def test_native_parity_exact_tied_and_nonnull_spawn_replay(family: str) -> None:
    early, late, labels = geometry()
    matrix, _ = models.design_and_offset(early, late, ALPHA, SHAPE, family, {"lambda": 0.5})
    head = frozen.mariner.Head(1.0, np.array([0.5, 0.8, 0.6, 0.03, 0.02, 0.01]), 0.6)
    observed = head.predict(matrix)
    native_grid = frozen.TestGrid(({},), (0.01,), (1.0, 6.0))
    before = frozen.fit_joint(
        early, late, observed, labels, ALPHA, "aggregate_mariner", shapes=(SHAPE,), test_grid=native_grid
    )
    aggregate = models.fit_joint(
        early, late, observed, labels, ALPHA, "aggregate_mariner", shapes=(SHAPE,), test_grid=native_grid
    )
    fit = models.fit_joint(
        early,
        late,
        observed,
        labels,
        ALPHA,
        family,
        controls=(aggregate,),
        shapes=(SHAPE,),
        test_grid=models.TestGrid(models.parameter_grid(family), (0.01,), (1.0, 6.0)),
    )
    after = frozen.fit_joint(
        early, late, observed, labels, ALPHA, "aggregate_mariner", shapes=(SHAPE,), test_grid=native_grid
    )
    np.testing.assert_array_equal(before.predict(early, late), after.predict(early, late))
    np.testing.assert_array_equal(aggregate.predict(early, late), before.predict(early, late))
    assert fit.inner_rmse <= aggregate.inner_rmse
    query0, query1 = early[:8] * 0.7, late[:8] * 1.3
    nonnull = models.JointFit(replace(fit.selected, parameters={"lambda": 0.5}))
    expected_matrix, _ = models.design_and_offset(
        query0, query1, nonnull.alpha, nonnull.shape, nonnull.model, nonnull.parameters
    )
    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) as executor:
        replay = executor.submit(process_replay, nonnull, query0, query1).result(timeout=30)
    np.testing.assert_array_equal(replay, nonnull.head.predict(expected_matrix))
    total = query0 + query1
    tied0, tied1 = ALPHA * total, (1 - ALPHA) * total
    np.testing.assert_array_equal(
        nonnull.predict(tied0, tied1),
        nonnull.head.predict(frozen.mariner.design_matrix(tied0 + tied1, nonnull.shape)),
    )
