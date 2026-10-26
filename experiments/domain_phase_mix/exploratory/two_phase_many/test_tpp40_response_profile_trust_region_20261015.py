# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Independent residual derivatives and finite-incumbent numerical contracts."""

import numpy as np
import pytest

from experiments.domain_phase_mix.exploratory.two_phase_many import tpp40_response_profile_trust_region_20261015 as model


@pytest.mark.parametrize("floor", [0.4, None])
def test_profile_jacobian_nonzero_residual_matches_finite_differences(floor):
    t = np.linspace(0.05, 2.0, 37)
    matrix = np.column_stack((-t, t**2))
    amplitudes = np.array([0.6, 0.11])
    observed = 0.4 + 1.3 * np.exp(matrix @ np.array([0.7, 0.13])) + 0.03 * np.sin(9 * t)
    record = model.prior.evaluate(matrix, observed, amplitudes, floor)
    residual, jacobian, diagnostics = model.residual_jacobian(matrix, observed, record, floor)
    assert diagnostics["jacobian_regular"]
    assert jacobian is not None
    assert np.linalg.norm(residual) > 1

    def direct_residual(a):
        phi = np.exp(matrix @ a)
        if floor is None:
            z = np.column_stack((np.ones(len(t)), phi))
            coefficients = np.linalg.lstsq(z, observed, rcond=None)[0]
            assert np.all(coefficients > 0)
            prediction = z @ coefficients
        else:
            prediction = floor + phi * (phi @ (observed - floor)) / (phi @ phi)
        return (prediction - observed) / (0.003 * np.sqrt(len(t)))

    step = 1e-5
    finite_difference = np.column_stack(
        [
            (direct_residual(amplitudes + step * unit) - direct_residual(amplitudes - step * unit)) / (2 * step)
            for unit in np.eye(2)
        ]
    )
    np.testing.assert_allclose(residual, direct_residual(amplitudes), atol=1e-11, rtol=0)
    np.testing.assert_allclose(jacobian, finite_difference, atol=2e-6, rtol=2e-7)
    np.testing.assert_allclose(2 * jacobian.T @ residual, record["gradient"], atol=1e-8, rtol=1e-10)
    shifted = matrix + np.array([18.0, -7.0])
    shifted_record = model.prior.evaluate(shifted, observed, amplitudes, floor)
    shifted_residual, shifted_jacobian, _ = model.residual_jacobian(shifted, observed, shifted_record, floor)
    np.testing.assert_allclose(shifted_residual, residual, atol=1e-10, rtol=0)
    np.testing.assert_allclose(shifted_jacobian, jacobian, atol=1e-9, rtol=0)


def test_floor_zero_face_and_known_floor_constant_column():
    t = np.linspace(0, 2, 29)
    matrix = -t[:, None]
    amplitudes = np.array([0.7])
    observed = 1.5 * np.exp(-0.7 * t) - 0.1
    record = model.prior.evaluate(matrix, observed, amplitudes, None)
    residual, jacobian, diagnostics = model.residual_jacobian(matrix, observed, record, None)
    assert record["floor"] == 0 and diagnostics["jacobian_regular"]
    assert jacobian is not None
    assert diagnostics["active_linear_columns"] == [1]
    step = 1e-5
    residuals = []
    for a in (amplitudes + step, amplitudes - step):
        phi = np.exp(matrix @ a)
        residuals.append((phi * (phi @ observed) / (phi @ phi) - observed) / (0.003 * np.sqrt(len(t))))
    np.testing.assert_allclose(jacobian[:, 0], (residuals[0] - residuals[1]) / (2 * step), atol=1e-6, rtol=1e-7)
    assert np.linalg.norm(residual) > 0
    constant = np.ones((len(t), 1))
    known = model.prior.evaluate(constant, observed, amplitudes, 0.2)
    _, known_jacobian, known_status = model.residual_jacobian(constant, observed, known, 0.2)
    assert known_status["jacobian_regular"]
    np.testing.assert_array_equal(known_jacobian, np.zeros_like(constant))


def test_face_transition_is_retained_without_false_convergence():
    matrix = np.zeros((9, 1))
    observed = np.ones(9)
    starts = [{"amplitudes": [0.4]}]
    fit = model.fit_response(matrix, observed, None, starts)
    assert fit["selected"]["raw_mean_squared_bpb"] == 0
    assert fit["unresolved"] and not fit["selected_from_successful_solve"]
    assert not fit["selected"]["jacobian_regular"]
    assert fit["solves"][0]["actual_evaluation_calls"] == 1


def test_strict_floor_only_face_and_tied_maximum_have_correct_derivatives():
    t = np.linspace(0, 2, 17)
    matrix = -t[:, None]
    observed = 1 + t
    record = model.prior.evaluate(matrix, observed, np.array([0.7]), None)
    _, jacobian, status = model.residual_jacobian(matrix, observed, record, None)
    assert record["scale"] == 0 and record["floor"] > 0
    assert status["jacobian_regular"] and status["zero_scale_closure"]
    np.testing.assert_array_equal(jacobian, np.zeros_like(matrix))

    matrix = np.array([[-1, 0], [0, -1], [-2, -1], [-1, -2], [-3, -1], [-1, -3]], float)
    amplitudes = np.ones(2)
    observed = 0.4 + 1.2 * np.exp(matrix @ np.array([0.7, 0.9])) + np.arange(6) * 0.001
    record = model.prior.evaluate(matrix, observed, amplitudes, 0.4)
    _, jacobian, status = model.residual_jacobian(matrix, observed, record, 0.4)
    assert status["jacobian_regular"]
    step = 1e-5
    differences = []
    for unit in np.eye(2):
        residuals = []
        for sign in (1, -1):
            phi = np.exp(matrix @ (amplitudes + sign * step * unit))
            prediction = 0.4 + phi * (phi @ (observed - 0.4)) / (phi @ phi)
            residuals.append((prediction - observed) / (0.003 * np.sqrt(6)))
        differences.append((residuals[0] - residuals[1]) / (2 * step))
    np.testing.assert_allclose(jacobian, np.column_stack(differences), atol=1e-6, rtol=1e-7)


@pytest.mark.parametrize("floor", [0.6, None])
def test_trf_recovers_clean_response_and_preserves_capped_incumbent(floor):
    t = np.linspace(0.05, 3, 41)
    matrix = np.column_stack((-t, t**2))
    observed = 0.6 + 1.4 * np.exp(matrix @ np.array([0.8, 0.06]))
    starts = [{"amplitudes": [0.4, 0.03]}, {"amplitudes": [1.0, 0.1]}]
    fitted = model.fit_response(matrix, observed, floor, starts)
    predicted = model.prior.predict(matrix, fitted["selected"])
    np.testing.assert_allclose(predicted, observed, atol=2e-8, rtol=0)
    assert all(item["actual_evaluation_calls"] <= 5000 for item in fitted["solves"])
    capped = model.fit_response(matrix, observed, floor, starts, evaluation_cap=1)
    assert capped["selected"] is not None and capped["unresolved"]
    assert all(item["actual_evaluation_calls"] == 1 for item in capped["solves"])
    assert all(not item["success"] for item in capped["solves"])
    expected = min(
        model.prior.evaluate(matrix, observed, np.asarray(item["amplitudes"]), floor)["scaled_objective"]
        for item in starts
    )
    assert capped["selected"]["scaled_objective"] == expected
