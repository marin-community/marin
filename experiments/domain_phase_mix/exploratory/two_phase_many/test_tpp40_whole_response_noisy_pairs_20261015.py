# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Invented means test failure retention and inherited fixed-pair arithmetic."""

import math

import numpy as np
import pandas as pd
import pytest

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    diagnose_tpp40_whole_response_noisy_pairs_20261015 as d,
)


def frame(predicted):
    return pd.DataFrame(
        {
            "role": ["asymmetric", "matched_tied", "actual_tied"],
            "source_index": [1, 1, 2],
            "row_id": ["a", "a", "b"],
            "truth": [1.0, 1.0, 2.0],
            "predicted": predicted,
        }
    )


def head(pair, eligible=True):
    return {
        "pair_id": "toy",
        "source": "toy",
        "source_fit_path": "TP046/toy/fit.json",
        "source_fit_sha256": "toy-fit-hash",
        "world": "native_null",
        "noise": "toy",
        "outer_fold": 3,
        "candidate": "aggregate_mariner",
        "analytic_pair_eligible": eligible,
        "selected_fit_unresolved": True,
        "original_numerical_status": "unresolved",
        "within_class_exclusion": None,
        **pair,
    }


def test_response_overflow_keeps_finite_rows_and_log_response():
    values, logs, statuses, underflow = d.response_rows(
        np.array([[0.0], [1000.0], [-1000.0]]), {"floor": 0.6, "scale": 1.0, "normalization": 0.0, "amplitudes": [1.0]}
    )
    assert values[0] == 1.6 and math.isinf(values[1]) and values[2] == 0.6
    np.testing.assert_array_equal(logs, [0, 1000, -1000])
    assert statuses == ["finite", "nonfinite_response", "finite"] and underflow == 1
    pair = d.pair_summary(np.ones(2), np.ones(2))
    endpoints, gains = d.functional_tables(frame([1.6, np.inf, 0.6]), pair, False)
    assert len(endpoints) == 3 and len(gains) == 1
    assert gains[0]["signed_separation"] is None and gains[0]["bound_status"] == "nonfinite_functional"
    result, intervals = d.menu_diagnostic(
        head(pair, False), frame([1.6, np.inf, 0.6]), "held_asymmetric_with_counterfactual_ties"
    )
    assert result["status"] == "unavailable_nonfinite" and not intervals
    result, _ = d.menu_diagnostic(head(pair, False), frame([1.6, np.inf, 0.6]), "held_joint")
    assert result["status"] == "finite" and result["within_class_menu_regret_bound"] is None


def test_scalar_and_gain_bounds_match_frozen_helper_and_preserve_misspecification():
    pair = d.pair_summary(np.array([1.003, 1.0]), np.ones(2))
    original_end, original_gain = d.pairs.functional_rows(frame([2.0, 3.0, 4.0]), pair, True)
    endpoints, gains = d.functional_tables(frame([2.0, 3.0, 4.0]), pair, True)
    np.testing.assert_allclose(
        [r["pair_mse_lower_bound"] for r in endpoints], original_end.pair_mse_lower_bound, atol=0, rtol=2e-15
    )
    assert gains[0]["pair_mse_lower_bound"] == pytest.approx(original_gain.pair_mse_lower_bound.iloc[0], rel=2e-15)
    endpoints, gains = d.functional_tables(frame([2.0, 3.0, 4.0]), pair, False)
    assert all(r["pair_mse_lower_bound"] is None for r in endpoints + gains)
    assert gains[0]["signed_separation"] == 1.0


def test_exact_zero_and_probability_underflow_remain_distinct():
    pair = d.pair_summary(np.array([0.3]), np.zeros(1))
    result = d.scalar_bound(1.0, pair, True)
    assert result["pair_mse_lower_bound"] == 0 and math.isfinite(result["pair_mse_log_lower_bound"])
    assert result["bound_status"] == "underflow"
    zero = d.scalar_bound(0.0, pair, True)
    assert zero["pair_mse_lower_bound"] == 0 and zero["bound_status"] == "exact_zero_separation"


def test_no_information_menu_has_half_compromise_cost_and_keeps_original_flags():
    pair = d.pair_summary(np.ones(2), np.ones(2))
    result, _ = d.menu_diagnostic(head(pair), frame([2.0, 2.0, 1.0]), "held_joint")
    assert result["finite_pair_bayes_regret"] == pytest.approx(0.5, abs=1e-15)
    assert result["selected_fit_unresolved"]
    assert result["truth_regret_of_first_alternative_best"] == 1.0


def test_missing_head_and_nonfinite_training_never_produce_finite_pair_risk():
    values, _, statuses, _ = d.response_rows(np.ones((3, 1)), None)
    pair = d.pair_summary(values, np.ones(3))
    assert pair["pair_status"] == "unavailable" and statuses == ["missing_head"] * 3
    result, _ = d.menu_diagnostic(head(pair), frame([1.0, 1.0, 2.0]), "held_joint")
    assert result["status"] == "unavailable_nonfinite"
    assert d.json_finite({"value": np.inf, "log": -np.inf}) == {"value": None, "log": None}


def test_reordered_gain_rows_are_not_silently_paired():
    data = pd.concat(
        [frame([1.0, 2.0, 3.0]), frame([1.0, 2.0, 3.0]).assign(source_index=[4, 4, 5], row_id=["c", "c", "d"])]
    )
    data = data.iloc[[0, 3, 4, 1, 2, 5]]
    with pytest.raises(ValueError, match="counterpart identities"):
        d.functional_tables(data, d.pair_summary(np.ones(2), np.ones(2)), True)


def test_resume_rejects_a_valid_cell_copied_from_another_context(tmp_path):
    path = tmp_path / "cell.json"
    payload = {"head": {"world": "native_null", "noise": "seed1"}}
    frozen = {"registration": "toy"}
    d.write(path, {"identity": frozen, "payload": payload, "payload_sha256": d.canonical(payload)})
    assert d.cached_payload(path, frozen, payload["head"]) == payload
    with pytest.raises(ValueError, match="another context"):
        d.cached_payload(path, frozen, {"world": "native_null", "noise": "seed2"})


def test_menu_lineage_points_to_current_heads_and_original_fit_for_every_status():
    pair = d.pair_summary(np.ones(2), np.ones(2))
    selected = head(pair)
    for predicted in ([2.0, 2.0, 1.0], [2.0, 2.0, np.inf]):
        result, _ = d.menu_diagnostic(selected, frame(predicted), "held_joint")
        assert result["prerequisite_heads_path"] == str(d.OUTPUT / "heads.csv")
        assert result["source_fit_path"] == selected["source_fit_path"]
        assert result["source_fit_sha256"] == selected["source_fit_sha256"]


def test_realized_likelihood_matches_direct_gaussian_density_difference():
    truth = np.array([1.0, 2.0, 3.0])
    candidate = truth + np.array([0.001, -0.004, 0.002])
    observed = truth + np.array([0.003, 0.001, -0.002])
    result = d.realized_likelihood(candidate, truth, observed)
    expected = (sum((observed - candidate) ** 2) - sum((observed - truth) ** 2)) / (2 * d.pairs.SIGMA**2)
    assert result["realized_log_likelihood_ratio_truth_over_candidate"] == pytest.approx(expected, abs=2e-14)
    adaptive = d.realized_likelihood(observed, truth, observed)
    assert adaptive["realized_log_likelihood_ratio_truth_over_candidate"] < 0
    assert adaptive["realized_likelihood_status"] == "finite_adaptive_descriptive"
    for extreme in (np.array([np.inf, 0.0, 0.0]), np.array([1e308, 0.0, 0.0])):
        failed = d.realized_likelihood(extreme, truth, observed)
        assert failed["realized_log_likelihood_ratio_truth_over_candidate"] is None
        assert failed["realized_likelihood_status"].startswith("unavailable_")
