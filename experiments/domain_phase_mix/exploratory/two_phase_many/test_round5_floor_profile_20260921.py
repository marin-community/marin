# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import round5_floor_profile_20260921 as profile


@pytest.fixture
def curve() -> tuple[np.ndarray, np.ndarray]:
    coordinate = np.linspace(0, 2, 43)
    matrix = np.column_stack([-coordinate, coordinate**2])
    response = 0.2 + np.exp(matrix @ [2.0, 0.15])
    return matrix, response


@pytest.mark.parametrize("ridge", [0.0, 0.01, 1.0])
def test_profile_matches_independent_dense_native_search(curve, ridge) -> None:
    matrix, noiseless = curve
    response = noiseless + np.random.default_rng(20260927).normal(0, 0.003, len(noiseless))
    fitted = profile.fit_profile_head(matrix, response, ridge)
    # Native standalone uses QR-reduced NNLS rather than the profiled BVLS
    # solver. A dense fixed mesh is independent of the adaptive scalar search.
    scores = []
    for gamma in np.linspace(1, 6, 2001):
        native = profile.mariner.fit_head(matrix, response, ridge, profile.mariner.FloorSpec(np.nan, 0, gamma))
        scores.append(float(np.sum((native.predict(matrix) - response) ** 2)))
    assert fitted.training_sse <= min(scores) + 1e-10
    native = profile.mariner.fit_head(matrix, response, ridge, profile.mariner.FloorSpec(np.nan, 0, fitted.gamma))
    np.testing.assert_allclose(
        fitted.head.predict(matrix, np.empty((len(response), 0))), native.predict(matrix), rtol=1e-10, atol=1e-11
    )
    assert np.all(fitted.head.aggregate_coefficients >= 0)


@pytest.mark.parametrize("shift", [0.0, -0.22])
def test_profile_recovers_identified_positive_and_negative_floors(curve, shift) -> None:
    matrix, response = curve
    response = response + shift
    fitted = profile.fit_profile_head(matrix, response, 0.0)
    assert abs(fitted.head.floor - (0.2 + shift)) < 1e-6
    np.testing.assert_allclose(fitted.head.aggregate_coefficients, [2, 0.15], rtol=1e-5, atol=1e-6)
    coordinate = np.linspace(-0.2, 3, 73)
    query = np.column_stack([-coordinate, coordinate**2])
    expected = 0.2 + shift + np.exp(query @ [2.0, 0.15])
    np.testing.assert_allclose(fitted.head.predict(query, np.empty((len(query), 0))), expected, rtol=2e-6, atol=1e-6)
    if shift < 0:
        assert fitted.lower_floor < fitted.head.floor < 0


def test_profile_preserves_response_units_and_origin(curve) -> None:
    matrix, response = curve
    response = response + np.random.default_rng(20260926).normal(0, 0.002, len(response))
    original = profile.fit_profile_head(matrix, response, 0.01)
    scale, shift = 3.7, -0.8
    changed = profile.fit_profile_head(matrix, scale * response + shift, 0.01)
    empty = np.empty((len(response), 0))
    np.testing.assert_allclose(
        changed.head.predict(matrix, empty),
        scale * original.head.predict(matrix, empty) + shift,
        rtol=1e-6,
        atol=1e-7,
    )
    np.testing.assert_allclose(changed.training_sse, scale**2 * original.training_sse, rtol=1e-8, atol=1e-11)


def test_constant_response_reports_unidentified_floor_and_native_gap_fallback() -> None:
    matrix = np.column_stack([np.linspace(-1, 1, 13), np.zeros(13)])
    response = np.full(13, 1.125)
    fitted = profile.fit_profile_head(matrix, response, 0.1)
    query = np.array([[-10.0, 0], [3.0, 0], [10.0, 0]])
    np.testing.assert_allclose(fitted.head.predict(query, np.empty((3, 0))), 1.125, rtol=0, atol=1e-14)
    assert fitted.lower_floor < fitted.upper_floor < response.min()
    saved = json.loads(json.dumps(fitted.to_json(), allow_nan=False))
    assert saved["floor_identification"] == "not_identified_constant_response"
    assert saved["constant_response"] is True
    assert fitted.gamma == 1.0


def test_profile_retains_native_clamp_boundary_and_replayable_search_receipt(curve) -> None:
    matrix, response = curve
    fitted = profile.fit_profile_head(matrix, response, 0.0)
    saved = json.loads(json.dumps(fitted.to_json(), allow_nan=False))
    coarse = [trial for trial in saved["trace"] if trial["stage"] == "coarse"]
    assert [trial["gamma"] for trial in coarse] == np.linspace(1, 6, 21).tolist()
    assert coarse[0]["deficit_clamp_count"] == 1
    # Recompute both native boundary heads without using the profile's cached
    # scores; the boundaries remain eligible even when refinement is interior.
    for gamma, trial in [(1.0, coarse[0]), (6.0, coarse[-1])]:
        native = profile.mariner.fit_head(matrix, response, 0, profile.mariner.FloorSpec(np.nan, 0, gamma))
        score = float(np.sum((native.predict(matrix) - response) ** 2))
        np.testing.assert_allclose(trial["training_sse"], score, rtol=1e-10, atol=1e-11)
    assert all(np.isfinite(trial["training_sse"]) for trial in saved["trace"])
    assert saved["conditional_solves"] == len({trial["gamma"] for trial in saved["trace"]})
