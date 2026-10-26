# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Invented-data numerical checks; no frozen scientific arrays are opened."""

import numpy as np
import pytest

from experiments.domain_phase_mix.exploratory.two_phase_many import diagnose_tpp40_noise_leverage_20261015 as diagnostic


def test_scaled_svd_preserves_full_rank_response_under_reparameterization():
    rng = np.random.default_rng(41)
    train = np.column_stack((np.ones(30), rng.normal(size=(30, 3))))
    query = np.column_stack((np.ones(6), rng.normal(size=(6, 3))))
    expected = query @ np.linalg.pinv(train)
    for scales in (np.ones(4), np.array([1e-8, 1e5, 0.7, 1e-3])):
        actual, info = diagnostic.response_map(train * scales, query * scales)
        np.testing.assert_allclose(actual, expected, atol=2e-15, rtol=2e-14)
        assert info["identified"].all()


def test_unidentified_gradient_and_exact_identity_are_distinguished():
    train = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    _, info = diagnostic.response_map(train, np.array([[1.0, 1.0], [1.0, -1.0], [0.0, 0.0]]))
    assert info["rank"] == 1
    assert info["identified"].tolist() == [True, False, True]
    assert info["information_gap"][-1] == 0


def test_heteroscedastic_log_noise_and_paired_gain_covariance():
    train = np.column_stack((np.ones(8), np.linspace(-1, 1, 8)))
    query = np.array([[1.0, -0.2], [1.0, 0.6], [1.0, -0.2], [1.0, 0.4]])
    deficit = np.exp(train @ [0.0, 0.8])
    query_deficit = np.exp(query @ [0.0, 0.8])
    noise = np.linspace(-0.002, 0.002, 8)
    values, _ = diagnostic.noise_response(train, query, deficit, query_deficit, noise, np.ones(2, bool), 2, False)
    h = query @ np.linalg.pinv(train)
    bpb_map = query_deficit[:, None] * h
    np.testing.assert_allclose(values["first_order_bpb"][:4], bpb_map @ (noise / deficit), atol=2e-17)
    np.testing.assert_allclose(
        values["log_estimator_sd"][:4], np.linalg.norm(diagnostic.SIGMA * bpb_map / deficit, axis=1), atol=2e-17
    )
    assert values["log_estimator_sd"][4] == 0
    expected = np.linalg.norm(diagnostic.SIGMA * (bpb_map[3] - bpb_map[1]) / deficit)
    np.testing.assert_allclose(values["log_estimator_sd"][5], expected, atol=2e-17)


def test_jensen_and_exponential_curvature_match_second_order_noise_average():
    train = np.column_stack((np.ones(9), np.linspace(-1, 1, 9)))
    query = np.array([[1.0, -1.5], [1.0, 1.8]])
    deficit, query_deficit = np.exp(train @ [0.0, 0.4]), np.exp(query @ [0.0, 0.4])
    values, _ = diagnostic.noise_response(train, query, deficit, query_deficit, np.zeros(9), np.ones(2, bool), 1, False)
    h = query @ np.linalg.pinv(train)
    # Independent symmetric one-coordinate Hessian stencil for E[f(noise)]-f(0).
    eps = 1e-4
    expected = np.zeros(2)
    for index in range(9):
        perturbation = np.zeros(9)
        perturbation[index] = eps
        plus = query_deficit * np.expm1(h @ np.log1p(perturbation / deficit))
        minus = query_deficit * np.expm1(h @ np.log1p(-perturbation / deficit))
        expected += 0.5 * diagnostic.SIGMA**2 * (plus + minus) / eps**2
    np.testing.assert_allclose(values["expected_second_order_bpb_bias"][:2], expected, atol=2e-12, rtol=2e-6)


def test_null_gain_information_is_exact_and_active_face_is_conditional():
    train = np.column_stack((np.ones(10), np.linspace(0, 1, 10)))
    query = np.array([[1.0, 0.2], [1.0, 0.2 + 1e-15]])
    for active in (np.ones(2, bool), np.array([True, False])):
        values, details = diagnostic.noise_response(train, query, np.ones(10), np.ones(2), np.zeros(10), active, 1, True)
        for name in ("log_estimator_sd", "fisher_sd", "log_information_gap", "fisher_information_gap"):
            assert values[name][-1] == 0
        assert values["log_identified"][-1] and values["fisher_identified"][-1]
        assert details["active_indices"] == np.flatnonzero(active).tolist()


def test_misspec_cosine_residualizes_both_vectors():
    nuisance = np.column_stack((np.ones(20), np.linspace(-1, 1, 20)))
    vector = np.linspace(-1, 1, 20) ** 2
    result = diagnostic.residual_cosine(nuisance, vector + nuisance @ [4.0, -2.0], -3 * vector + nuisance @ [-7.0, 5.0])
    np.testing.assert_allclose(result["cosine"], -1.0, atol=2e-15)
    np.testing.assert_allclose(result["residual_tangent_norm"], 3 * result["residual_misspec_norm"], atol=2e-14)


def test_saved_active_face_retains_omitted_clean_coefficient_bias():
    train = np.column_stack((np.ones(12), np.linspace(-1, 1, 12)))
    query = np.array([[1.0, -1.5], [1.0, 1.5]])
    clean = np.exp(train @ [0.3, 0.4])
    truth = np.exp(query @ [0.3, 0.4])
    observed = np.exp(train @ [0.2, -0.1])
    # A nonnegative-slope fit to this decreasing target has slope zero and free mean intercept.
    expected = np.full(2, np.exp(np.log(observed).mean())) - truth
    values, details = diagnostic.noise_response(
        train, query, clean, truth, observed - clean, np.array([True, False]), 1, False
    )
    np.testing.assert_allclose(values["exact_face_error_bpb"][:2], expected, atol=4e-15)
    assert np.max(abs(values["face_clean_approximation_bias_bpb"][:2])) > 0.5
    assert details["fisher_design"]["columns"] == 2
    assert details["log_design"]["columns"] == 1


def test_admitted_noise_response_does_not_silently_clamp_deficits():
    with pytest.raises(ValueError, match="no deficit clamp: 1 rows"):
        diagnostic.noise_response(
            np.ones((3, 1)),
            np.ones((2, 1)),
            np.ones(3),
            np.ones(2),
            np.array([-1.0, 0.0, 0.0]),
            np.ones(1, bool),
            1,
            True,
        )
