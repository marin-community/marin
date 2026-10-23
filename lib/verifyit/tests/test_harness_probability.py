# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0


import pytest
from verifyit.adapters.harness_probability import probability_mass
from verifyit.grade import InvalidTask


@pytest.mark.parametrize(
    "labels,responses",
    [
        ([], []),
        ([0], [(-1, False)]),
        ([True], [(-1, False)]),
        ([2], [(-1, False)]),
        ([1, 0], [(-1, False)]),
        ([1], [(float("nan"), False)]),
        ([1], [(float("-inf"), False)]),
        ([1], [(1, False)]),
        ([1], [(-1, 1)]),
        ([1], [(-1,)]),
        ([1], [(10**1000, False)]),
    ],
)
def test_malformed_evidence_cannot_score(labels, responses):
    with pytest.raises(InvalidTask):
        probability_mass(labels, responses)
