# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Record boundaries and multiplicities in captured shell output."""

import pytest
from verifyit.modes.grade_nl2bash import score_capture


@pytest.mark.parametrize(
    "actual,expected,reward",
    [
        ("./dir2/b.txt\0./dir1/a.txt\0", "./dir1/a.txt\0./dir2/b.txt\0", 1),
        ("/workspace/a.txt\n./b.txt\n", "./b.txt\0./a.txt\0", 1),
        ("a.txt\0", "a.txt\0a.txt\0", 0),
        ("a.txt\npermission denied\n", "a.txt\n", 0),
        ("a.txt\nextra record\n", "a.txt\n", 1),
        ("\n", "", 1),
        ("unexpected\n", "", 0),
    ],
)
def test_capture_scores_records_with_ordering_duplicates_and_errors(actual, expected, reward):
    assert score_capture(actual, expected)[0] == reward
