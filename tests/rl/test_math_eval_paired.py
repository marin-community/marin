# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from experiments.post_training.math_eval.paired import (
    SeedInference,
    bootstrap,
    bootstrap_records,
    family_bootstrap,
    holm_adjust,
    required_questions,
)


def test_joint_seed_bootstrap_retains_seed_spread_when_question_means_cancel():
    reference = {17: {"a": [0], "b": [0]}, 29: {"a": [1], "b": [1]}}
    candidate = {17: {"a": [1], "b": [1]}, 29: {"a": [0], "b": [0]}}
    fixed = bootstrap(reference, candidate, seed=17, repetitions=1000, inference=SeedInference.FIXED)
    joint = bootstrap(reference, candidate, seed=17, repetitions=1000, inference=SeedInference.POPULATION)
    assert fixed.interval == (0, 0)
    assert joint.delta == 0
    assert joint.interval == (-1, 1)


def test_repeated_responses_do_not_become_independent_questions():
    reference = {17: {"a": [0, 1], "b": [0, 1]}}
    candidate = {17: {"a": [1, 1], "b": [1, 1]}}
    result = bootstrap(reference, candidate, seed=1, repetitions=100, inference=SeedInference.FIXED)
    repeated = {17: {question: values * 8 for question, values in candidate[17].items()}}
    assert bootstrap(reference, repeated, seed=1, repetitions=100, inference=SeedInference.FIXED) == result
    assert result.delta == 0.5
    assert result.questions == 2
    assert result.interval == (0.5, 0.5)


def test_paired_bootstrap_rejects_mismatched_question_sets():
    with pytest.raises(ValueError, match="identical question hashes"):
        bootstrap(
            {17: {"a": [0], "b": [0]}},
            {17: {"a": [0], "c": [0]}},
            seed=1,
            repetitions=100,
            inference=SeedInference.FIXED,
        )


def test_holm_stepdown_matches_four_hypothesis_worked_example():
    assert holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.005}) == pytest.approx(
        {"a": 0.03, "b": 0.06, "c": 0.06, "d": 0.02}
    )


def test_miller_power_example_requires_about_969_questions():
    assert required_questions(delta=0.03, difference_variance=1 / 9, alpha=0.05, power=0.8) == 969


@pytest.mark.parametrize("variance", [0, float("nan"), float("inf"), -1])
def test_degenerate_variance_does_not_certify_power(variance):
    with pytest.raises(ValueError, match="positive variance"):
        required_questions(delta=0.02, difference_variance=variance, alpha=0.05, power=0.8)


def test_record_tables_default_to_completion_and_keep_repeats_clustered():
    reference = {
        17: [
            {"prompt_sha256": q, "row_ordinal": i, "score_contract_completed": 0, "score_contract": 1}
            for i, q in enumerate(["a", "a", "b", "b"])
        ]
    }
    candidate = {17: [dict(row, score_contract_completed=1) for row in reference[17]]}
    result = bootstrap_records(reference, candidate, seed=17, repetitions=100, inference=SeedInference.FIXED)
    assert result.questions == 2 and result.delta == 1 and result.interval == (1, 1)


def test_family_resamples_common_seed_question_indices_and_labels_adjustments():
    arms = {
        "reference": {17: {"a": [0], "b": [0]}, 29: {"a": [0], "b": [0]}},
        "small": {17: {"a": [0], "b": [0.25]}, 29: {"a": [0.25], "b": [0.5]}},
    }
    arms["large"] = {s: {q: [2 * v[0]] for q, v in rows.items()} for s, rows in arms["small"].items()}
    result = family_bootstrap(arms, reference="reference", seed=17, repetitions=2000, inference=SeedInference.POPULATION)
    small = result["results"]["small"]
    large = result["results"]["large"]
    assert large["delta"] == 2 * small["delta"]
    assert large["simultaneous_interval"] == pytest.approx([2 * v for v in small["simultaneous_interval"]])
    assert result["joint_covariance"]["large"]["small"] == pytest.approx(
        2 * result["joint_covariance"]["small"]["small"], abs=1e-12
    )
    assert large["p_value"] == small["p_value"]
    assert large["holm_adjusted_p_value"] >= large["p_value"]
    assert result["interval_method"].startswith("Bonferroni")


def test_family_population_inference_retains_seed_noise_with_constant_questions():
    arms = {
        "ref": {17: {"a": [0], "b": [0]}, 29: {"a": [0], "b": [0]}},
        "candidate": {17: {"a": [0], "b": [0]}, 29: {"a": [1], "b": [1]}},
    }
    result = family_bootstrap(arms, reference="ref", seed=17, repetitions=1000, inference=SeedInference.POPULATION)
    assert result["results"]["candidate"]["simultaneous_interval"] == (0, 1)
    assert result["joint_covariance"]["candidate"]["candidate"] > 0
