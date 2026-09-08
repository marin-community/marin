# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import math

import pytest

from experiments.post_training.math_eval.calibration import assumed_mde, greedy_repeat_variances, stochastic_variances


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
