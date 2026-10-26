# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import replace

import numpy as np
import pytest

from experiments.domain_phase_mix.exploratory.two_phase_many import tpp40_joint_estimation_20261014 as joint

SHAPE = {"rate": 0.25, "power": 0.7, "threshold": 3.0}
OTHER_SHAPE = {"rate": 0.5, "power": 0.7, "threshold": 4.0}
ALPHA = 0.8
COEFFICIENTS = np.array([0.5, 0.8, 0.6, 0.03, 0.02, 0.01])


def geometry(seed: int, rows: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    total = np.exp(rng.uniform(-2.0, 4.0, (rows, 3)))
    fraction = rng.uniform(0.15, 0.95, total.shape)
    return fraction * total, (1 - fraction) * total


def truth(early: np.ndarray, late: np.ndarray, model: str) -> np.ndarray:
    total = early + late
    benefit = 1 - np.exp(-((0.25 * total) ** 0.7))
    offset = np.zeros(len(early))
    if model == "bounded_benefit_odds":
        earlier = 1 - np.exp(-((0.25 * early) ** 0.7))
        diagonal = 1 - np.exp(-((0.25 * ALPHA * total) ** 0.7))
        odds = benefit / (1 - benefit) * np.exp(4 * (diagonal - earlier))
        benefit = odds / (1 + odds)
    if model == "contrast_magnitude":
        offset = -0.1 * np.mean((early / total - ALPHA) ** 2, axis=1) / (ALPHA * (1 - ALPHA))
    harm = np.logaddexp(0, np.log1p(total) - 3) ** 2
    return 0.6 + np.exp(1 - benefit @ COEFFICIENTS[:3] + harm @ COEFFICIENTS[3:] + offset)


def test_candidate_holdout_predictions_do_not_use_held_out_labels() -> None:
    early, late = geometry(815, 90)
    observed = truth(early, late, "aggregate_mariner")
    labels = np.arange(len(observed)) % 3
    matrix, offset = joint.design_and_offset(early, late, ALPHA, SHAPE, "aggregate_mariner", {})
    original = joint.candidate_oof(matrix, offset, observed, labels, 0.01, 2.5)
    poisoned = observed.copy()
    poisoned[labels == 0] += np.linspace(0.4, 1.0, np.sum(labels == 0))
    changed = joint.candidate_oof(matrix, offset, poisoned, labels, 0.01, 2.5)
    np.testing.assert_array_equal(original[labels == 0], changed[labels == 0])
    assert np.max(abs(original[labels != 0] - changed[labels != 0])) > 0.01


@pytest.mark.parametrize("model", joint.MODEL_NAMES)
def test_known_floor_search_recovers_noiseless_law_and_serialized_predictions(model: str) -> None:
    early, late = geometry(827, 90)
    observed = truth(early, late, model)
    parameters = {
        "aggregate_mariner": ({},),
        "bounded_benefit_odds": ({"kappa": 0.0}, {"kappa": 4.0}),
        "contrast_magnitude": ({"theta": 0.0}, {"theta": -0.1}),
    }[model]
    fit = joint.fit_known_floor(
        early,
        late,
        observed,
        np.arange(len(observed)) % 3,
        ALPHA,
        model,
        floor=0.6,
        shapes=(OTHER_SHAPE, SHAPE),
        test_grid=joint.TestGrid(parameters, (0.01, 0.0), (2.5,)),
    )
    query0, query1 = geometry(119, 31)
    expected = truth(query0, query1, model)
    np.testing.assert_allclose(fit.predict(query0, query1), expected, rtol=2e-12, atol=2e-12)
    total = query0 + query1
    np.testing.assert_allclose(
        fit.predict(ALPHA * total, (1 - ALPHA) * total),
        truth(ALPHA * total, (1 - ALPHA) * total, model),
        rtol=2e-12,
        atol=2e-12,
    )
    replay = joint.JointFit.from_json(json.loads(json.dumps(fit.to_json(), allow_nan=False)))
    np.testing.assert_array_equal(fit.predict(query0, query1), replay.predict(query0, query1))


@pytest.mark.parametrize("model", ("bounded_benefit_odds", "contrast_magnitude"))
def test_joint_search_retains_complete_null_and_exact_static_restriction(model: str) -> None:
    # Repeated geometry makes the median/minimum map identical in every fold.
    # The supplied null has an off-grid gamma giving the exact generating floor;
    # dropping it would leave only approximate scalar-search candidates.
    base0, base1 = geometry(412, 30)
    early, late = np.tile(base0, (3, 1)), np.tile(base1, (3, 1))
    labels = np.repeat(np.arange(3), len(base0))
    observed = truth(early, late, "aggregate_mariner")
    null = joint.fit_known_floor(
        early,
        late,
        observed,
        labels,
        ALPHA,
        "aggregate_mariner",
        floor=0.6,
        shapes=(SHAPE,),
        test_grid=joint.TestGrid(({},), (0.0,), (2.5,)),
    )
    gamma = float((np.median(observed) - 0.6) / (np.median(observed) - np.min(observed)))
    null = replace(null, kappa=gamma, known_floor=None)
    settings = ({"kappa": 0.0}, {"kappa": 4.0}) if model == "bounded_benefit_odds" else ({"theta": 0.0}, {"theta": -0.1})
    fit = joint.fit_joint(
        early,
        late,
        observed,
        labels,
        ALPHA,
        model,
        shapes=(SHAPE,),
        test_grid=joint.TestGrid(settings, (0.0,), (1.5, 2.5, 4.0, 6.0)),
        matched_null_fit=null,
    )
    np.testing.assert_allclose(fit.predict(early, late), observed, rtol=2e-12, atol=2e-12)
    assert fit.inner_rmse < 1e-12
    assert not any(fit.parameters.values())
    query0, query1 = geometry(277, 17)
    total = query0 + query1
    static = joint.mariner.design_matrix(ALPHA * total + (1 - ALPHA) * total, fit.shape)
    np.testing.assert_array_equal(fit.predict(ALPHA * total, (1 - ALPHA) * total), fit.head.predict(static))
    null_matrix = joint.mariner.design_matrix(query0 + query1, fit.shape)
    np.testing.assert_array_equal(fit.predict(query0, query1), fit.head.predict(null_matrix))
    selected = fit.selection_profile[fit.search_space["selected_trial_index"]]
    assert selected["stage"] == "retained_matched_null"
