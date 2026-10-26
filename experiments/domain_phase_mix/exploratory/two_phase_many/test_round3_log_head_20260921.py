# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiments.domain_phase_mix.exploratory.two_phase_many import round3_log_head_20260921 as log_head


@pytest.fixture
def head_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260922)
    aggregate = rng.uniform(-1, 1, (41, 3))
    phase = rng.uniform(-0.5, 0.5, (41, 2))
    # The first aggregate effect points outside the allowed cone, so its
    # coefficient must hit the bound in at least one oracle comparison.
    response = 0.3 + np.exp(-0.4 + aggregate @ np.array([-0.2, 0.3, 0.1]) + phase @ np.array([-0.6, 0.4]))
    return aggregate, phase, response


@pytest.mark.parametrize("ridge", [0.0, 0.03, 2.0])
def test_log_head_matches_exhaustive_active_set_oracle(head_data, ridge) -> None:
    aggregate, phase, response = head_data
    aggregate_penalty = np.array([0.5, 2.0, 0.9])
    phase_penalty = np.array([1.4, 0.8])
    spec = log_head.mariner.FloorSpec(np.nan, 0.0, 2.5)
    actual = log_head.fit_log_head(aggregate, phase, response, ridge, spec, aggregate_penalty, phase_penalty)
    floor = log_head.mariner.floor_value(response, spec)
    target = np.log(np.maximum(response - floor, log_head.mariner.DEFICIT_FLOOR))
    matrix = np.column_stack([np.ones(len(response)), aggregate, phase])
    penalty = np.diag(np.r_[0.0, aggregate_penalty, phase_penalty])
    # An exhaustive active-set oracle uses only unconstrained least squares,
    # with the intercept kept explicitly. It shares neither BVLS nor centering.
    full = np.vstack([matrix, np.sqrt(ridge) * penalty])
    rhs = np.r_[target, np.zeros(matrix.shape[1])]
    candidates = []
    for active in itertools.product([False, True], repeat=aggregate.shape[1]):
        included = np.r_[True, active, np.ones(phase.shape[1], dtype=bool)]
        coefficients = np.zeros(matrix.shape[1])
        coefficients[included] = np.linalg.lstsq(full[:, included], rhs, rcond=None)[0]
        if np.all(coefficients[1 : 1 + aggregate.shape[1]] >= -1e-12):
            candidates.append((float(np.sum((full @ coefficients - rhs) ** 2)), coefficients))
    _, expected_coefficients = min(candidates, key=lambda item: item[0])
    expected = floor + np.exp(matrix @ expected_coefficients)
    np.testing.assert_allclose(actual.predict(aggregate, phase), expected, rtol=2e-11, atol=2e-12)
    assert np.min(actual.aggregate_coefficients) >= 0


@pytest.mark.parametrize("ridge", [0.0, 0.1])
def test_log_head_aggregate_only_matches_native_mariner(head_data, ridge) -> None:
    aggregate, _, response = head_data
    empty_phase = np.empty((len(response), 0))
    spec = log_head.mariner.FloorSpec(np.nan, 0.0, 2.5)
    actual = log_head.fit_log_head(aggregate, empty_phase, response, ridge, spec, np.ones(3), np.zeros(0))
    native = log_head.mariner.fit_head(aggregate, response, ridge, spec)
    query = np.vstack([aggregate, np.array([[-3.0, 1.0, 2.0], [1.5, -2.0, 0.2]])])
    np.testing.assert_allclose(
        actual.predict(query, np.empty((len(query), 0))), native.predict(query), rtol=2e-11, atol=2e-12
    )


def test_log_head_recovers_both_phase_signs_on_factorial_design() -> None:
    phase = np.array(list(itertools.product([-1.0, 0.0, 1.0], repeat=2)))
    response = np.exp(0.2 + phase @ np.array([-0.5, 0.35]))
    aggregate = np.zeros((len(response), 0))
    # This explicit anchor and kappa yield zero under the native gap formula.
    spec = log_head.mariner.FloorSpec(float(2 * response.min()), 0.0, 2.0)
    head = log_head.fit_log_head(aggregate, phase, response, 0.0, spec, np.zeros(0), np.ones(2))
    np.testing.assert_allclose(head.phase_coefficients, [-0.5, 0.35], rtol=2e-11, atol=2e-12)
    np.testing.assert_allclose(head.predict(aggregate, phase), response, rtol=2e-11, atol=2e-12)


@pytest.mark.parametrize("ridge", [0.0, 0.03])
def test_log_head_preserves_rank_deficient_contrast_relabeling_and_units(head_data, ridge) -> None:
    aggregate, two_phase, response = head_data
    phase = np.column_stack([two_phase, -two_phase.sum(axis=1)])
    aggregate_penalty = np.array([2.0, 0.7, 1.4])
    phase_penalty = np.array([1.0, 1.3, 2.0])
    spec = log_head.mariner.FloorSpec(np.nan, 0.0, 2.5)
    original = log_head.fit_log_head(aggregate, phase, response, ridge, spec, aggregate_penalty, phase_penalty)
    aggregate_order = np.array([2, 0, 1])
    phase_order = np.array([1, 2, 0])
    aggregate_scale = np.array([0.1, 7.0, 2.0])
    phase_scale = np.array([-5.0, 0.2, 3.0])
    changed = log_head.fit_log_head(
        aggregate[:, aggregate_order] * aggregate_scale,
        phase[:, phase_order] * phase_scale,
        response,
        ridge,
        spec,
        aggregate_penalty[aggregate_order] * aggregate_scale,
        phase_penalty[phase_order] * np.abs(phase_scale),
    )
    rng = np.random.default_rng(75)
    query_aggregate = rng.uniform(-2, 2, (11, 3))
    query_two = rng.uniform(-2, 2, (11, 2))
    query_phase = np.column_stack([query_two, -query_two.sum(axis=1)])
    np.testing.assert_allclose(
        changed.predict(
            query_aggregate[:, aggregate_order] * aggregate_scale, query_phase[:, phase_order] * phase_scale
        ),
        original.predict(query_aggregate, query_phase),
        rtol=2e-11,
        atol=2e-12,
    )


def test_log_head_tied_queries_reproduce_same_native_head_with_negative_floor() -> None:
    aggregate = np.array([[-2.0, 0.5], [0.1, 3.0], [100.0, 0.0]])
    head = log_head.Head(-0.2, np.array([0.4, 0.2]), np.array([-3.0, 4.0]), -1.1)
    native = log_head.mariner.Head(head.intercept, head.aggregate_coefficients, head.floor)
    np.testing.assert_array_equal(head.predict(aggregate, np.zeros((3, 2))), native.predict(aggregate))
