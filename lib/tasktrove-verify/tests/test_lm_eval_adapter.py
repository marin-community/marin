# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Metric bridge tests with a task implementing a weighted perplexity contract."""

import math
from collections.abc import Sequence
from typing import Any

import pytest
from tasktrove_verify.adapters.lm_eval import aggregate_task, score_task
from tasktrove_verify.grade import Status, write_reward


class WeightedTask:
    def process_results(self, doc: dict, results: Sequence[Any]) -> dict:
        return {"word_perplexity": (results[0], doc["words"]), "acc": doc["correct"]}

    def aggregation(self) -> dict:
        return {"word_perplexity": weighted_perplexity, "acc": lambda values: sum(values) / len(values)}


def weighted_perplexity(values):
    return math.exp(-sum(value for value, _ in values) / sum(weight for _, weight in values))


def test_task_bridge_preserves_weighted_metrics_and_scalar_reward(tmp_path):
    task = WeightedTask()
    samples = [
        score_task(task, {"words": 1, "correct": 1}, [-math.log(2)]),
        score_task(task, {"words": 3, "correct": 0}, [-3 * math.log(4)]),
    ]
    result = aggregate_task(task, [sample.metrics for sample in samples], "acc")
    assert result.metrics["word_perplexity"] == pytest.approx(128**0.25)
    assert result.metrics["acc"] == 0.5
    assert result.verdict is not None
    write_reward(tmp_path, result.verdict)
    assert (tmp_path / "reward.txt").read_text().strip() == "0.5"
    invalid = aggregate_task(task, [sample.metrics for sample in samples], "word_perplexity")
    assert invalid.metrics == result.metrics
    assert invalid.verdict is not None and invalid.verdict.status == Status.INVALID_TASK
    write_reward(tmp_path, invalid.verdict)
    assert not (tmp_path / "reward.txt").exists()


def test_task_bridge_retains_non_scalar_metric_instead_of_scoring_zero():
    result = score_task(WeightedTask(), {"words": 2, "correct": 0}, [-6.0], "word_perplexity")
    assert result.metrics == {"word_perplexity": (-6.0, 2), "acc": 0}
    assert result.verdict is not None and result.verdict.status == Status.INVALID_TASK
