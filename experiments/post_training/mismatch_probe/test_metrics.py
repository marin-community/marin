# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import math

import pytest

from experiments.post_training.mismatch_probe.metrics import comparison_metrics, prompt_cluster_bootstrap


def test_known_ratios_raw_and_capped_ess():
    target = [[math.log(1), math.log(2)], [math.log(4), math.log(1)]]
    reference = [[0.0, 0.0], [0.0, 0.0]]
    result = comparison_metrics(target, reference, [[True, True], [True, False]], tis_cap=2.0)
    assert result["tokens"] == 3
    assert result["abs_min"] == 0.0
    assert result["abs_p50"] == pytest.approx(math.log(2))
    assert result["abs_p75"] == pytest.approx((math.log(2) + math.log(4)) / 2)
    assert result["abs_max"] == pytest.approx(math.log(4))
    assert result["share_beyond_2x"] == pytest.approx(1 / 3)
    assert result["k3"] == pytest.approx(sum(ratio - 1 - math.log(ratio) for ratio in (1, 2, 4)) / 3)
    assert result["token_ess_fraction_raw"] == pytest.approx(49 / (3 * 21))
    assert result["token_ess_fraction_capped"] == pytest.approx(25 / (3 * 9))
    assert result["sequence_ess_fraction_raw"] == pytest.approx(36 / (2 * 20))


def test_bootstrap_carries_all_samples_of_each_prompt():
    prompt_ids = ["a", "a", "b"]

    def calculate(indices):
        return {"answer_count": len(indices), "a_count": sum(prompt_ids[index] == "a" for index in indices)}

    result = prompt_cluster_bootstrap(prompt_ids, calculate, seed=7, draws=60)
    assert result.point["answer_count"] == 3
    assert all(draw["a_count"] in {0, 2, 4} for draw in result.draws)
    assert "a_count" in result.intervals
