# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import json
import math

import numpy as np
import pytest

pytest.importorskip("lm_eval")

from lm_eval import evaluator
from lm_eval.api.model import LM
from lm_eval.tasks import TaskManager

from levanter.eval_harness_metrics import task_config_with_smooth_metrics


class FixedResponses(LM):
    def loglikelihood(self, requests):
        return [
            (-request.doc.get("nll", math.log(2 if request.doc["gold"] == 0 else 8)), False) for request in requests
        ]

    def loglikelihood_rolling(self, requests):
        raise AssertionError("Expected multiple choice")

    def generate_until(self, requests):
        raise AssertionError("Expected multiple choice")


def test_mmlu_group_preserves_smooth_scores_and_unweighted_subject_average(tmp_path):
    # Unequal subject sizes distinguish the historic subject-macro BPB from document-weighting.
    for name, gold, count in (("subject_a", 0, 1), ("subject_b", 1, 3)):
        data = tmp_path / f"{name}.jsonl"
        data.write_text("\n".join(json.dumps({"gold": gold}) for _ in range(count)))
        config = {
            "task": name,
            "dataset_path": "json",
            "dataset_kwargs": {"data_files": {"test": str(data)}},
            "test_split": "test",
            "output_type": "multiple_choice",
            "doc_to_text": "Answer:",
            "doc_to_choice": ["a", "b"],
            "doc_to_target": "{{gold}}",
            "num_fewshot": 0,
        }
        (tmp_path / f"{name}.yaml").write_text(json.dumps(config))
    (tmp_path / "mmlu.yaml").write_text(json.dumps({"group": "mmlu", "task": ["subject_a", "subject_b"]}))
    manager = TaskManager(include_defaults=False, include_path=str(tmp_path))
    tasks = manager.load_config(task_config_with_smooth_metrics({"task": "mmlu"}))
    result = evaluator.evaluate(FixedResponses(), tasks, bootstrap_iters=0, log_samples=True)
    assert result["results"]["subject_a"]["bpb,none"] == pytest.approx(1)
    assert result["results"]["subject_b"]["bpb,none"] == pytest.approx(3)
    assert result["groups"]["mmlu"]["bpb,none"] == pytest.approx(2)
    assert result["groups"]["mmlu"]["acc,none"] == pytest.approx(0.25)
    for samples in result["samples"].values():
        for sample in samples:
            assert sample["choice_logprob"] == pytest.approx(-math.log(2))
            assert sample["choice_prob_norm"] == pytest.approx(0.5)
            assert sample["choice_logprob_norm"] == pytest.approx(-math.log(2))


def test_bpb_aggregation_retains_historical_accumulation(tmp_path):
    nlls = np.linspace(1.1, 4.9, 200)
    data = tmp_path / "responses.jsonl"
    data.write_text("\n".join(json.dumps({"gold": 0, "nll": float(nll)}) for nll in nlls))
    config = task_config_with_smooth_metrics(
        {
            "task": "accumulation",
            "dataset_path": "json",
            "dataset_kwargs": {"data_files": {"test": str(data)}},
            "test_split": "test",
            "output_type": "multiple_choice",
            "doc_to_text": "Answer:",
            "doc_to_choice": ["a", "b"],
            "doc_to_target": "{{gold}}",
            "num_fewshot": 0,
            "metric_list": [{"metric": "bpb", "aggregation": "mean", "higher_is_better": False}],
        }
    )
    tasks = TaskManager(include_defaults=False).load_config(config)
    result = evaluator.evaluate(FixedResponses(), tasks, bootstrap_iters=0, log_samples=True)
    # The former harness returned NumPy BPBs into sum(arr) / len(arr).
    expected = sum(nlls * (1 / math.log(2))) / len(nlls)
    assert result["results"]["accumulation"]["bpb,none"] == expected
