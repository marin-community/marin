# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from experiments.belay.submit import asks_for_ladder


@pytest.mark.parametrize(
    "text",
    [
        "Given the speedrun win of lm_head rows as MLA value embeddings, run its 4 rung scaling ladder.",
        "Take X0221 (CPLM Plus) up the ladder with the settings of its speedrun win.",
        "Launch the scaling ladder for X0235.",
    ],
)
def test_a_follow_up_that_asks_for_the_ladder_in_words_is_read_as_one(text: str) -> None:
    assert asks_for_ladder(text)


@pytest.mark.parametrize(
    "text",
    [
        "Add a narrow exact-attention branch; the gain should hold on the scaling ladder.",
        "Scaling rule: heads = d_model / 128, so it carries across the ladder rungs.",
    ],
)
def test_a_rationale_that_only_mentions_the_ladder_is_not_a_request(text: str) -> None:
    assert not asks_for_ladder(text)
