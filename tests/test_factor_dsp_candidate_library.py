# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pytest

from experiments.domain_phase_mix.exploratory.two_phase_many.build_factor_dsp_candidate_library import (
    EffectiveExposureDSPModel,
    average_phase_tv,
    candidate_diagnostics,
    discover_regularized_endpoints,
    generate_sobol_logit_candidates,
    kl_to_proportional,
    predict_y_factor,
    softmax_phase_logits,
    top_domain_delta_table,
)


def test_predict_y_factor_uses_effective_exposure_penalty_form():
    model = EffectiveExposureDSPModel(
        domain_names=("alpha", "beta"),
        rho=np.array([0.5, 1.0]),
        tau=np.array([0.25, 0.5]),
        gamma=np.array([2.0, 0.5]),
        intercept=0.3,
        benefit_coef=np.array([1.5, 0.25]),
        penalty_coef=np.array([0.2, 0.4]),
        c0=np.array([4.0, 2.0]),
        c1=np.array([3.0, 5.0]),
        metrics={"oof_rmse": 0.1},
    )
    weights = np.array([[[0.25, 0.75], [0.5, 0.5]]])

    z = np.array([[0.25 * 4.0 + 2.0 * 0.5 * 3.0, 0.75 * 2.0 + 0.5 * 0.5 * 5.0]])
    signal = 1.0 - np.exp(-np.array([0.5, 1.0]) * z)
    penalty = np.logaddexp(0.0, np.log1p(z) - np.array([0.25, 0.5])) ** 2
    loss = 0.3 - signal @ np.array([1.5, 0.25]) + penalty @ np.array([0.2, 0.4])

    assert predict_y_factor(model, weights) == pytest.approx(-loss)


def test_generate_sobol_logit_candidates_is_deterministic_and_simplex():
    proportional = np.array([0.1, 0.2, 0.7])

    first = generate_sobol_logit_candidates(
        proportional=proportional,
        num_candidates=8,
        max_alpha=0.35,
        seed=17,
    )
    second = generate_sobol_logit_candidates(
        proportional=proportional,
        num_candidates=8,
        max_alpha=0.35,
        seed=17,
    )

    assert first.shape == (8, 2, 3)
    assert np.allclose(first, second)
    assert np.all(first > 0.0)
    assert np.allclose(first.sum(axis=2), 1.0)


def test_average_phase_tv_averages_over_phases():
    candidates = np.array(
        [
            [[0.8, 0.2], [0.5, 0.5]],
            [[0.5, 0.5], [0.0, 1.0]],
        ]
    )
    proportional = np.array([0.5, 0.5])

    assert average_phase_tv(candidates, proportional) == pytest.approx([0.15, 0.25])


def test_candidate_diagnostics_reports_nearest_observed_and_support():
    candidate_ids = np.array(["candidate_a", "candidate_b"])
    candidate_sources = np.array(["sobol", "sobol"])
    candidates = np.array(
        [
            [[0.8, 0.2], [0.5, 0.5]],
            [[0.5, 0.5], [0.0, 1.0]],
        ]
    )
    observed_ids = np.array(["near", "far"])
    observed = np.array(
        [
            [[0.7, 0.3], [0.5, 0.5]],
            [[0.0, 1.0], [0.0, 1.0]],
        ]
    )
    proportional = np.array([0.5, 0.5])
    predicted_y = np.array([1.0, 0.25])

    diagnostics = candidate_diagnostics(
        candidate_ids=candidate_ids,
        candidate_sources=candidate_sources,
        weights=candidates,
        proportional=proportional,
        predicted_y_factor=predicted_y,
        proportional_predicted_y_factor=0.4,
        observed_ids=observed_ids,
        observed_weights=observed,
        oof_rmse=0.1,
        lcb_z=1.0,
    )

    assert diagnostics.loc[0, "nearest_observed_run"] == "near"
    assert diagnostics.loc[0, "nearest_observed_tv"] == pytest.approx(0.05)
    assert diagnostics.loc[0, "predicted_y_factor_gain_vs_proportional"] == pytest.approx(0.6)
    assert diagnostics.loc[0, "predicted_y_factor_gain_lcb"] == pytest.approx(0.5)
    assert diagnostics.loc[1, "min_phase_support_gt_1e3"] == 1


def test_top_domain_delta_table_keeps_largest_positive_and_negative_deltas():
    summary = pd.DataFrame(
        {
            "candidate_id": ["a", "b", "c"],
            "predicted_y_factor_gain_lcb": [0.5, 0.1, 1.0],
        }
    )
    weights = np.array(
        [
            [[0.4, 0.1, 0.5], [0.2, 0.3, 0.5]],
            [[0.2, 0.2, 0.6], [0.3, 0.2, 0.5]],
            [[0.1, 0.7, 0.2], [0.5, 0.2, 0.3]],
        ]
    )

    table = top_domain_delta_table(
        summary=summary,
        weights=weights,
        proportional=np.array([0.2, 0.3, 0.5]),
        domain_names=("alpha", "beta", "gamma"),
        top_candidate_count=2,
        top_domain_count=1,
    )

    records = table.sort_values(["candidate_id", "phase", "delta_rank"]).to_dict("records")
    assert {record["candidate_id"] for record in records} == {"a", "c"}
    assert any(
        record["candidate_id"] == "c" and record["domain"] == "beta" and record["direction"] == "positive"
        for record in records
    )
    assert any(
        record["candidate_id"] == "a" and record["domain"] == "beta" and record["direction"] == "negative"
        for record in records
    )


def test_softmax_phase_logits_returns_valid_two_phase_weights():
    logits = np.array([[2.0, 0.0, -1.0], [-1.0, 0.5, 1.0]])

    weights = softmax_phase_logits(logits)

    assert weights.shape == (2, 3)
    assert np.all(weights > 0.0)
    assert np.allclose(weights.sum(axis=1), 1.0)
    assert weights[0, 0] > weights[0, 1] > weights[0, 2]


def test_kl_to_proportional_is_zero_at_proportional_and_positive_for_tilts():
    proportional = np.array([0.25, 0.75])
    proportional_weights = np.stack([proportional, proportional])
    tilted_weights = np.array([[0.5, 0.5], [0.1, 0.9]])

    assert kl_to_proportional(proportional_weights, proportional) == pytest.approx(0.0)
    assert kl_to_proportional(tilted_weights, proportional) > 0.0


def test_discover_regularized_endpoints_recovers_simple_model_direction():
    model = EffectiveExposureDSPModel(
        domain_names=("good", "bad"),
        rho=np.array([2.0, 0.0]),
        tau=np.array([10.0, 10.0]),
        gamma=np.array([1.0, 1.0]),
        intercept=0.0,
        benefit_coef=np.array([1.0, 0.0]),
        penalty_coef=np.array([0.0, 0.0]),
        c0=np.array([1.0, 1.0]),
        c1=np.array([1.0, 1.0]),
        metrics={"oof_rmse": 0.0},
    )
    proportional = np.array([0.5, 0.5])
    start_weights = np.array(
        [
            [[0.5, 0.5], [0.5, 0.5]],
            [[0.8, 0.2], [0.8, 0.2]],
        ]
    )

    summary, endpoints = discover_regularized_endpoints(
        model=model,
        proportional=proportional,
        start_weights=start_weights,
        kl_penalties=(0.0, 10.0),
        max_weight_penalty=0.0,
        max_weight_target=1.0,
        random_start_count=0,
        seed=7,
        maxiter=80,
    )

    unpenalized_idx = int(summary.loc[summary["kl_penalty"].eq(0.0), "endpoint_index"].iloc[0])
    regularized_idx = int(summary.loc[summary["kl_penalty"].eq(10.0), "endpoint_index"].iloc[0])
    assert endpoints[unpenalized_idx, :, 0].min() > 0.95
    assert endpoints[regularized_idx, :, 0].mean() < endpoints[unpenalized_idx, :, 0].mean()
    assert summary.loc[summary["kl_penalty"].eq(0.0), "optimization_success"].iloc[0]
