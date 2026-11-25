# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from experiments.post_training.abstention_sft.knowledge import Label, held_out, knowledge_label


@pytest.mark.parametrize(
    "correct,label",
    [(8, Label.ANSWER), (6, Label.ANSWER), (5, Label.DROP), (2, Label.DROP), (1, Label.ABSTAIN), (0, Label.ABSTAIN)],
)
def test_correct_count_sets_the_label(correct, label):
    assert knowledge_label(correct) is label


def test_correct_count_outside_the_sample_range_is_rejected():
    with pytest.raises(AssertionError):
        knowledge_label(9)


def test_held_out_is_stable_and_near_ten_percent():
    ids = [f"law-wiki-{index:05d}" for index in range(10_000)]
    assert [held_out(request_id) for request_id in ids] == [held_out(request_id) for request_id in ids]
    assert 900 <= sum(held_out(request_id) for request_id in ids) <= 1100
