# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import round6_floor_cv_20260921 as floor_cv


@pytest.fixture
def curve() -> tuple[np.ndarray, np.ndarray, tuple[tuple[np.ndarray, np.ndarray], ...]]:
    coordinate = np.linspace(0, 2, 43)
    matrix = np.column_stack([-coordinate, coordinate**2])
    response = 0.2 + np.exp(matrix @ np.array([2.0, 0.15]))
    labels = np.arange(len(response)) % 3
    folds = tuple((np.flatnonzero(labels != fold), np.flatnonzero(labels == fold)) for fold in range(3))
    return matrix, response, folds


@pytest.mark.parametrize("floor", [-0.8, -0.02, 0.2])
@pytest.mark.parametrize("ridge", [0.0, 0.01, 1.0])
def test_fixed_floor_head_matches_native_qr_nnls(curve, floor, ridge) -> None:
    matrix, response, _ = curve
    actual = floor_cv.fixed_floor_head(matrix, response, ridge, floor)
    # The standalone solver uses QR-reduced NNLS; the new adapter preserves
    # the existing BVLS solver. Compare predictions through independent solves.
    native = floor_cv.mariner.fit_head(matrix, response, ridge, floor_cv.mariner.FloorSpec(floor, 0.0, 0.0))
    coordinate = np.linspace(-0.2, 3, 73)
    query = np.column_stack([-coordinate, coordinate**2])
    np.testing.assert_allclose(
        actual.predict(query, np.empty((len(query), 0))), native.predict(query), rtol=2e-11, atol=2e-12
    )
    assert np.float64(actual.floor).tobytes() == np.float64(floor).tobytes()


def test_fixed_floor_head_retains_native_equality_clamp_but_rejects_above_minimum(curve) -> None:
    matrix, response, _ = curve
    floor = float(response.min())
    actual = floor_cv.fixed_floor_head(matrix, response, 0.01, floor)
    native = floor_cv.mariner.fit_head(matrix, response, 0.01, floor_cv.mariner.FloorSpec(floor, 0.0, 0.0))
    np.testing.assert_allclose(
        actual.predict(matrix, np.empty((len(matrix), 0))), native.predict(matrix), rtol=2e-11, atol=2e-12
    )
    with pytest.raises(ValueError):
        floor_cv.fixed_floor_head(matrix, response, 0.01, float(np.nextafter(floor, np.inf)))


@pytest.mark.parametrize("shift", [0.0, -0.22])
def test_cv_recovers_off_grid_floor_and_transports_it_to_outer_refit(curve, shift) -> None:
    matrix, response, folds = curve
    response = response + shift
    result = floor_cv.search_floor_cv(matrix, response, 0.0, folds)
    assert abs(result.floor - (0.2 + shift)) <= 1e-5
    assert result.inner_rmse <= 1e-6
    squared_error = 0.0
    count = 0
    for training, validation in folds:
        head = floor_cv.fixed_floor_head(matrix[training], response[training], 0.0, result.floor)
        prediction = head.predict(matrix[validation], np.empty((len(validation), 0)))
        squared_error += float(np.sum((prediction - response[validation]) ** 2))
        count += len(validation)
        assert np.float64(head.floor).tobytes() == np.float64(result.floor).tobytes()
    np.testing.assert_allclose(result.inner_rmse, np.sqrt(squared_error / count), rtol=1e-9, atol=1e-12)
    head = floor_cv.fixed_floor_head(matrix, response, 0.0, result.floor)
    coordinate = np.linspace(-0.2, 3, 73)
    query = np.column_stack([-coordinate, coordinate**2])
    expected = 0.2 + shift + np.exp(query @ np.array([2.0, 0.15]))
    np.testing.assert_allclose(head.predict(query, np.empty((len(query), 0))), expected, rtol=2e-6, atol=1e-6)
    assert np.float64(head.floor).tobytes() == np.float64(result.floor).tobytes()
    # An infeasible coarse neighbor must not discard the interval containing
    # the true floor. The finite/infinite transition is part of this case.
    saved = json.loads(json.dumps(result.to_json(), allow_nan=False))
    assert any(trial["inner_rmse"] is None for trial in saved["trace"])


def test_cv_admissibility_uses_each_actual_training_fold() -> None:
    matrix = -np.linspace(0.0, 1.0, 6)[:, None]
    response = 1.0 + np.exp(matrix[:, 0])
    response[0] = 0.05
    first, second = np.arange(3), np.arange(3, 6)
    folds = ((second, first), (first, second))
    result = floor_cv.search_floor_cv(matrix, response, 0.1, folds)
    saved = result.to_json()
    trial = next(trial for trial in saved["trace"] if trial["floor"] == 0.125)
    assert trial["inner_rmse"] is None
    assert trial["folds"][0]["admissible"] is True
    assert trial["folds"][1]["admissible"] is False
    assert trial["folds"][0]["training_minimum"] == float(response[second].min())
    assert trial["folds"][1]["training_minimum"] == float(response[first].min())


def test_fixed_floor_head_preserves_affine_response_changes(curve) -> None:
    matrix, response, _ = curve
    scale, shift, floor = 3.7, -0.8, 0.2
    original = floor_cv.fixed_floor_head(matrix, response, 0.01, floor)
    changed = floor_cv.fixed_floor_head(matrix, scale * response + shift, 0.01, scale * floor + shift)
    query = np.vstack([matrix, np.array([[-3.0, 9.0], [0.2, 0.04]])])
    empty = np.empty((len(query), 0))
    np.testing.assert_allclose(
        changed.predict(query, empty), scale * original.predict(query, empty) + shift, rtol=2e-11, atol=2e-12
    )


def test_fixed_floor_head_constant_outcomes_keep_explicit_negative_floor() -> None:
    matrix = np.column_stack([np.linspace(-1, 1, 13), np.zeros(13)])
    response = np.full(13, 1.125)
    floor = -1.1
    head = floor_cv.fixed_floor_head(matrix, response, 0.1, floor)
    query = np.array([[-10.0, 0.0], [3.0, 0.0], [10.0, 0.0]])
    np.testing.assert_allclose(head.predict(query, np.empty((3, 0))), response[0], rtol=0, atol=1e-14)
    assert np.float64(head.floor).tobytes() == np.float64(floor).tobytes()
