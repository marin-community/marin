# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import math

import pytest

from experiments.post_training.math_eval.calibration import (
    assumed_mde,
    crossed_mde_sensitivity,
    crossed_variance_components,
    greedy_repeat_variances,
    stochastic_variances,
)


def test_calibration_separates_question_noise_from_empirical_seed_spread():
    panels = {k: {17: {"a": [0] * k, "b": [1] * k}, 29: {"a": [1] * k, "b": [1] * k}} for k in (1, 4, 8)}
    result = stochastic_variances(panels)
    assert result[8]["per_seed_question_mean_variance"] == {17: 0.5, 29: 0}
    assert result[8]["per_seed_within_question_variance"] == {17: 0, 29: 0}
    assert result[8]["aggregate_training_seed_sample_variance"] == 0.125
    assert result[8]["seed_degrees_of_freedom"] == 1
    assert result[1]["per_seed_within_question_variance"] == {17: None, 29: None}
    assert not result[8]["future_seed_variance_certified"]


def test_calibration_mixed_question_has_within_question_noise():
    panels = {k: {seed: {"a": ([0, 1] * 4)[:k], "b": [1] * k} for seed in (17, 29)} for k in (1, 4, 8)}
    assert stochastic_variances(panels)[8]["per_seed_within_question_variance"][17] == pytest.approx(1 / 7)
    panels[4][29]["foreign"] = panels[4][29].pop("a")
    with pytest.raises(ValueError, match="membership"):
        stochastic_variances(panels)


def test_greedy_outcome_change_bites_even_when_most_passes_agree():
    panel = {17: {"a": [0, 0, 0, 0, 1], "b": [1] * 5}, 29: {"a": [0] * 5, "b": [1] * 5}}
    result = greedy_repeat_variances(panel)
    assert result[17]["questions_with_changed_outcome"] == 1
    assert result[17]["mean_pairwise_outcome_disagreement"] == 0.2
    assert result[29]["pass_mean_sample_variance"] == 0
    assert not result[29]["zero_population_noise_certified"]


def test_mde_correlation_family_and_seed_floor_sensitivity():
    config = dict(
        question_variances=(0.25, 0.25), run_variances=(0.01, 0.01), questions=256, training_seeds=2, family_size=1
    )
    zero = assumed_mde(**config, rho_question=0, rho_run=0)
    worst = assumed_mde(**config, rho_question=-1, rho_run=-1)
    assert worst["mde"] == pytest.approx(zero["mde"] * math.sqrt(2))
    many = assumed_mde(**(config | {"questions": 1000000}), rho_question=0, rho_run=0)
    assert zero["mde"] > many["mde"] > zero["infinite_question_mde_floor"] > 0
    family = assumed_mde(**(config | {"family_size": 10}), rho_question=0, rho_run=0)
    assert family["mde"] > zero["mde"] and family["planning_alpha"] == 0.005
    assert not zero["power_certified"]
    with pytest.raises(ValueError, match="Zero"):
        assumed_mde(**(config | {"question_variances": (0, 0), "run_variances": (0, 0)}), rho_question=0, rho_run=0)


def test_crossed_anova_hand_computed_binary_panel_preserves_negative_moments():
    panel = {17: {"a": [0, 0], "b": [0, 1]}, 29: {"a": [1, 0], "b": [1, 1]}}
    result = crossed_variance_components(panel, 2)
    assert result["sums_of_squares"] == {"seed": 0.5, "question": 0.5, "interaction": 0, "draw": 1}
    assert result["raw_moments"] == {"seed": 0.125, "question": 0.125, "interaction": -0.125, "draw": 0.25}
    assert result["planning_nonnegative_moments"]["interaction"] == 0
    assert result["degrees_of_freedom"] == {"seed": 1, "question": 1, "interaction": 1, "draw": 4}
    assert not result["truncation_is_conservative"]
    single = crossed_variance_components(
        {seed: {q: values[:1] for q, values in rows.items()} for seed, rows in panel.items()}, 1
    )
    assert single["raw_moments"]["draw"] is None and single["raw_moments"]["interaction"] is None
    assert single["interaction_plus_draw_when_k1"] is not None


def test_crossed_mde_persists_all_covariances_and_correct_design_scaling():
    pairs = {"seed": (0.01, 0.01), "question": (0.04, 0.04), "interaction": (0.02, 0.02), "draw": (0.1, 0.1)}
    args = dict(component_pairs=pairs, questions=100, training_seeds=2, samples=8, family_size=4)
    result = crossed_mde_sensitivity(**args)
    assert len(result["all_component_scenarios"]) == 256
    assert len(result["shared_rho_scenarios"]) == 4
    assert result["planning_alpha"] == 0.0125
    base = result["primary_rho0_assumption"]
    assert base["variance_of_mean_components"] == {
        "seed": 0.01,
        "question": 0.0008,
        "interaction": 0.0002,
        "draw": 0.000125,
    }
    assert result["shared_rho_scenarios"][0]["mde"] == pytest.approx(base["mde"] * math.sqrt(2))
    doubled = crossed_mde_sensitivity(**(args | {"samples": 16}))["primary_rho0_assumption"]
    assert doubled["variance_of_mean_components"]["draw"] == base["variance_of_mean_components"]["draw"] / 2
    assert doubled["variance_of_mean_components"]["seed"] == base["variance_of_mean_components"]["seed"]
    zeros = crossed_mde_sensitivity(**(args | {"component_pairs": {key: (0, 0) for key in pairs}}))
    assert all(row["mde"] is None and not row["power_certified"] for row in zeros["all_component_scenarios"])
    with pytest.raises(ValueError, match="identified"):
        crossed_mde_sensitivity(**(args | {"component_pairs": pairs | {"draw": (None, None)}}))
