# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2026 The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""AGIEval's two winners may differ and either may have multiple accepted golds."""

from types import SimpleNamespace

import pytest
from verifyit.adapters.harness_agieval import agieval_metrics, validate_agieval_task
from verifyit.grade import InvalidTask


def test_raw_and_character_winners_preserve_alternative_gold():
    assert agieval_metrics(["é", "long option", "其他"], [1, 2], [(-1.0, False), (-2.0, True), (-3.0, False)]) == {
        "acc": 0.0,
        "acc_norm": 1.0,
    }


def test_ties_choose_first_option_for_both_metrics():
    assert agieval_metrics(["a", "b"], [1], [(-2.0, False), (-2.0, True)]) == {"acc": 0.0, "acc_norm": 0.0}


def test_more_than_alphabet_options_and_extreme_finite_likelihoods():
    responses = [(-1000.0, False)] * 30
    responses[29] = (-999.0, True)
    assert agieval_metrics(["x"] * 30, [29], responses) == {"acc": 1.0, "acc_norm": 1.0}


@pytest.mark.parametrize(
    "choices,gold,responses",
    [
        (["a", ""], [0], [(-1.0, True), (-2.0, False)]),
        (["a"], [True], [(-1.0, True)]),
        (["a"], [1], [(-1.0, True)]),
        (["a"], [], [(-1.0, True)]),
        (["a"], [0], [(float("nan"), True)]),
        (["a"], [0], [(float("-inf"), True)]),
        (["a"], [0], [(True, True)]),
        (["a"], [0], [(0.1, True)]),
        (["a"], [0], [(-1.0, 1)]),
        (["a"], [0], [(-1.0,)]),
        (["a"], [0], [(10**1000, True)]),
    ],
)
def test_malformed_reference_or_runtime_evidence_is_unscored(choices, gold, responses):
    with pytest.raises(InvalidTask):
        agieval_metrics(choices, gold, responses)


@pytest.mark.parametrize(
    "task",
    [SimpleNamespace(), SimpleNamespace(config=None), SimpleNamespace(config=SimpleNamespace(process_results=None))],
)
def test_unrelated_tasks_without_a_custom_scorer_are_ignored(task):
    assert validate_agieval_task(task) is False
