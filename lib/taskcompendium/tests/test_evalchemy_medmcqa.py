# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from verifyit.grade import Status

from taskcompendium.importers.evalchemy.scorers.choices import grade_medmcqa_cot


@pytest.mark.parametrize(
    "response,reward,extracted",
    [
        ("Scurvy follows vitamin C deficiency.\nFinal Answer: B", 1, "B"),
        ("Final Answer: **B**", 1, "B"),
        ("Final Answer:\nB", 1, "B"),
        ("Final Answer: A", 0, "A"),
        # The first marker decides, as lm-eval's take_first filter does.
        ("Final Answer: A\nOn reflection, Final Answer: B", 0, "A"),
        ("final answer: b", 0, None),
        ("The answer is B.", 0, None),
        ("Final Answer: (B)", 0, None),
        ("", 0, None),
    ],
)
def test_medmcqa_cot_extraction(response, reward, extracted):
    result = grade_medmcqa_cot("B", response)
    assert result.status == Status.SCORED
    assert result.reward == reward
    assert result.detail["extracted"] == extracted
