# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from tasktrove_verify.adapters.harness_native import exact_match, likelihood_choice
from tasktrove_verify.grade import InvalidTask
from tasktrove_verify.spec import EmptyOutputPolicy


@pytest.mark.parametrize(
    "candidate,references,options,reward",
    [
        ("42 ", ["42"], {}, 0),
        ("İ", ["i\u0307"], {"ignore_case": True}, 0),
        ("İ", ["i", "long"], {"ignore_case": True}, 1),
        ("ß", ["ss"], {"ignore_case": True}, 0),
        ("x1!", ["x"], {"ignore_numbers": True, "ignore_punctuation": True}, 1),
        ("ab", ["a", "ab"], {}, 1),
        ("abc", ["c"], {"regexes_to_ignore": ["a", "b"]}, 1),
    ],
)
def test_native_exact_preserves_source_normalization(candidate, references, options, reward):
    assert exact_match(candidate, references, **options).reward == reward


@pytest.mark.parametrize(
    "normalization,gold,reward", [("raw", [0], 1), ("characters", [1], 1), ("bytes", [0], 1), ("raw", [1], 0)]
)
def test_likelihood_normalization_unicode_and_first_tie(normalization, gold, reward):
    assert likelihood_choice(["é", "aa"], [-2, -2], gold, normalization).reward == reward


def test_unknown_options_fail_closed():
    with pytest.raises(InvalidTask, match="unsupported"):
        exact_match("x", ["x"], casefold=True)
    with pytest.raises(InvalidTask, match="unsupported"):
        likelihood_choice(["x"], [1], [0], "tokens")


@pytest.mark.parametrize("likelihoods", [[float("nan"), 0], [float("inf"), 0]])
def test_nonfinite_likelihood_does_not_select_correct_default(likelihoods):
    with pytest.raises(InvalidTask, match="finite"):
        likelihood_choice(["correct", "wrong"], likelihoods, [0])


@pytest.mark.parametrize("targets", [[], [-1], [2], [True]])
def test_invalid_gold_does_not_award_a_selected_default(targets):
    with pytest.raises(InvalidTask):
        likelihood_choice(["correct", "wrong"], [0, -1], targets)


def test_empty_target_string_is_a_valid_literal_reference():
    assert exact_match("", [""]).reward == 1
    assert exact_match("attempt", [""]).reward == 0


@pytest.mark.parametrize("candidate", ["", "!!!"])
def test_explicit_empty_policy_rejects_normalized_empty_without_hiding_invalid_reference(candidate):
    assert exact_match(candidate, ["!!!"], ignore_punctuation=True).reward == 1
    assert exact_match(candidate, ["!!!"], ignore_punctuation=True, empty_output=EmptyOutputPolicy.ZERO).reward == 0
    with pytest.raises(InvalidTask, match="references"):
        exact_match(candidate, ["!!!", None], ignore_punctuation=True, empty_output=EmptyOutputPolicy.ZERO)
    with pytest.raises(InvalidTask, match="policy"):
        exact_match(candidate, ["!!!"], ignore_punctuation=True, empty_output="unknown")
