# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Finite Gaussian-pair and endpoint-versus-gain contracts on invented data."""

import math

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad

from experiments.domain_phase_mix.exploratory.two_phase_many import diagnose_tpp40_gaussian_pairs_20261015 as diagnostic


def normal_density(x: float, mean: float, sigma: float) -> float:
    return math.exp(-0.5 * ((x - mean) / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))


def test_gaussian_pair_matches_overlap_integral_and_rotation() -> None:
    # For two univariate normals, equal-prior Bayes error is half their overlap.
    sigma, displacement = 0.7, 1.3
    # Split at the density crossing, so the reference does not integrate a cusp.
    left, _ = quad(lambda x: normal_density(x, displacement, sigma), -np.inf, displacement / 2, epsabs=1e-12)
    right, _ = quad(lambda x: normal_density(x, 0, sigma), displacement / 2, np.inf, epsabs=1e-12)
    overlap = left + right
    one = diagnostic.gaussian_pair(np.array([displacement]), np.array([0.0]), sigma)
    rotated = diagnostic.gaussian_pair(np.array([displacement / math.sqrt(2)] * 2), np.zeros(2), sigma)
    assert one["bayes_error"] == pytest.approx(overlap / 2, abs=1e-12)
    assert rotated["distance"] == pytest.approx(one["distance"], abs=1e-14)
    assert rotated["bayes_error"] == pytest.approx(one["bayes_error"], abs=1e-14)


def test_identical_training_means_can_hide_endpoint_difference_without_phase_gain() -> None:
    pair = diagnostic.gaussian_pair(np.array([1.0, 2.0]), np.array([1.0, 2.0]), 0.003)
    frame = pd.DataFrame(
        {
            "role": ["asymmetric", "matched_tied", "actual_tied"],
            "source_index": [7, 7, 8],
            "row_id": ["a", "a", "b"],
            "truth": [1.0, 1.0, 2.0],
            "predicted": [5.0, 5.0, 6.0],
        }
    )
    endpoints, gains = diagnostic.functional_rows(frame, pair, True)
    assert pair["bayes_error"] == 0.5
    np.testing.assert_allclose(endpoints.pair_mse_lower_bound, 4.0, atol=0, rtol=0)
    np.testing.assert_allclose(gains.signed_separation, 0.0, atol=0, rtol=0)
    np.testing.assert_allclose(gains.pair_mse_lower_bound, 0.0, atol=0, rtol=0)


def test_gain_sign_and_misspecified_world_exclusion() -> None:
    frame = pd.DataFrame(
        {
            "role": ["asymmetric", "matched_tied"],
            "source_index": [1, 1],
            "row_id": ["a", "a"],
            "truth": [0.8, 1.0],
            "predicted": [0.9, 0.95],
        }
    )
    pair = diagnostic.gaussian_pair(np.ones(2), np.ones(2), 0.003)
    endpoint, gain = diagnostic.functional_rows(frame, pair, False)
    assert gain.signed_separation.iloc[0] == pytest.approx(-0.15, abs=1e-15)
    assert endpoint.pair_mse_lower_bound.isna().all()
    assert gain.pair_mse_lower_bound.isna().all()
    member = diagnostic.membership("heterogeneous_null", "equal_group", None, 0.6, 0)
    assert not member["common_family_member"] and not member["analytic_pair_eligible"]


def test_raw_clipping_and_positive_static_embedding_remain_explicit() -> None:
    predicted, count = diagnostic.head_prediction(
        np.array([[40.0], [0.0]]), {"floor": 0.6, "intercept": 0.0, "coefficients": [1.0]}
    )
    np.testing.assert_allclose(predicted, [0.6 + math.exp(30), 1.6], atol=0, rtol=1e-15)
    member = diagnostic.membership("pool_positive", "aggregate_mariner", None, 0.6, count)
    assert count == 1
    assert member["common_family_member"] and member["aggregate_embedded_as_zero_mu"]
    assert not member["analytic_pair_eligible"]
    assert member["common_clock"] == "global_pool"
    assert not member["evaluated_unclipped_domain"]


def test_reordered_gain_counterparts_are_rejected() -> None:
    frame = pd.DataFrame(
        {
            "role": ["asymmetric", "asymmetric", "matched_tied", "matched_tied"],
            "source_index": [1, 2, 2, 1],
            "row_id": ["a", "b", "b", "a"],
            "truth": [1.0] * 4,
            "predicted": [2.0] * 4,
        }
    )
    pair = diagnostic.gaussian_pair(np.ones(2), np.ones(2), 0.003)
    with pytest.raises(ValueError, match="pair the same aggregate"):
        diagnostic.functional_rows(frame, pair, True)


def test_gaussian_tail_retains_logarithmic_bound_after_probability_underflow() -> None:
    pair = diagnostic.gaussian_pair(np.array([100.0]), np.zeros(1), 1.0)
    frame = pd.DataFrame(
        {
            "role": ["asymmetric", "matched_tied"],
            "source_index": [1, 1],
            "row_id": ["a", "a"],
            "truth": [1.0, 1.0],
            "predicted": [2.0, 3.0],
        }
    )
    endpoints, gains = diagnostic.functional_rows(frame, pair, True)
    assert pair["bayes_error"] == 0.0 and math.isfinite(pair["log_bayes_error"])
    assert endpoints.bound_underflowed.all() and gains.bound_underflowed.all()
    assert np.isfinite(endpoints.pair_mse_log_lower_bound).all()


def test_query_exponent_underflow_excludes_analytic_bound_and_preserves_mean() -> None:
    selected = {"scale": 2.0, "floor": 0.7, "amplitudes": [1.0], "normalization": 0.0}
    predicted, underflow = diagnostic.response_prediction(np.array([[0.0], [-1000.0]]), selected)
    np.testing.assert_allclose(predicted, [2.7, 0.7], atol=0, rtol=0)
    member = diagnostic.membership(
        "native_null", "aggregate_mariner", diagnostic.ARMS[2], 0.7, 0, "response_profile", underflow
    )
    assert underflow == 1 and member["common_family_member"] and not member["analytic_pair_eligible"]
    unchanged = diagnostic.membership("pool_positive", "global_pool", None, 0.65, 0, "unchanged_j2")
    assert unchanged["analytic_pair_eligible"] and unchanged["common_floor_domain"] == "nonnegative"


def test_failed_nonwinning_refinement_is_preserved_without_excluding_finite_function() -> None:
    flags = {
        "selected_from_failed_refinement": False,
        "failed_refinements": 1,
        "query_clipping_count": 0,
        "training_deficit_clip_count": 0,
        "inner_training_deficit_clip_count": 0,
        "inner_prediction_clip_count": 0,
    }
    unresolved, preserved = diagnostic.original_numerical_status("unchanged_j2", flags)
    member = diagnostic.membership("pool_positive", "global_pool", None, 0.65, 0, "unchanged_j2")
    assert unresolved and preserved == flags
    assert member["analytic_pair_eligible"]
