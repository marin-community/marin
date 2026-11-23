# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from verifyit.grade import Status

from taskcompendium.importers.evalchemy.scorers.arc_agi import grade_arc_agi

EXPECTED = [[1, 2], [3, 4]]


@pytest.mark.parametrize(
    "response,reward",
    [
        ("The answer is \\boxed{[[1, 2], [3, 4]]}", 1),
        ("```json\n[[1, 2], [3, 4]]\n```", 1),
        # The first boxed grid wins over a later bare grid, as in the reference parser.
        ("\\boxed{[[1, 2], [3, 4]]} then [[0, 0], [0, 0]]", 1),
        # Without a box, the last bracketed value is the answer.
        ("first [[1, 2], [3, 4]] then [[0, 0], [0, 0]]", 0),
        ("[[1, 2], [3, 5]]", 0),
        ("[[1, 2]]", 0),
        ("[[1.0, 2], [3, 4]]", 0),
        ('[[1, 2], [3, 4]] {"note": 1}', 0),
        ("[[1, 2], [3, 4]", 0),
        # Non-finite numbers are not grids, so the verdict stays valid JSON.
        ("[[NaN, 2], [3, 4]]", 0),
        ("\\boxed{[[1e309, 2], [3, 4]]}", 0),
        ("no grid here", 0),
        ("", 0),
    ],
)
def test_grade_arc_agi_extracts_like_reference_and_requires_exact_grid(response, reward):
    result = grade_arc_agi(EXPECTED, 0.5, response)
    assert result.status == Status.SCORED
    assert result.reward == reward
    assert result.detail["metric_weights"] == {"accuracy": 0.5}
