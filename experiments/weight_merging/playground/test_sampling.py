# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from experiments.weight_merging.playground.sampling import independent_requests, require_diverse_probe


def test_independent_seeds_survive_chunking_and_resume():
    requests = independent_requests(["a", "b", "c"], [0, 17, 200], 8)
    resumed = independent_requests(["b", "c"], [17, 200], 8)
    assert requests[8:] == resumed
    assert len({row.seed for row in requests}) == 24
    assert [(row.prompt_index, row.sample_index) for row in requests] == [
        (index, sample) for index in (0, 17, 200) for sample in range(8)
    ]


def test_probe_rejects_cloned_samples_but_accepts_varied_samples():
    with pytest.raises(ValueError, match="Sampling diversity failed"):
        require_diverse_probe([["same"] * 8 for _ in range(20)], 8)
    proof = require_diverse_probe([[str(i) for i in range(8)] for _ in range(20)], 8)
    assert proof["mean_distinct"] == 8
    with pytest.raises(ValueError, match="Incomplete"):
        require_diverse_probe([["short"]], 8)
