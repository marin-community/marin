"""Regression coverage for TruthfulQA MC2 probability normalization."""

import inspect
import math
from pathlib import Path

import pytest
from lm_eval.tasks import TaskManager
from lm_eval.tasks._yaml_loader import load_yaml

from eval.lm_eval_tasks.truthfulqa.utils import process_results_mc2


def test_mc2_probability_mass_is_stable_for_large_negative_loglikelihoods():
    doc = {"mc2_targets": {"labels": [1, 0, 0]}}
    results = [(-1000.0, False), (-1001.0, False), (-1002.0, False)]

    score = process_results_mc2(doc, results)["acc"]

    expected = 1.0 / (1.0 + math.exp(-1.0) + math.exp(-2.0))
    assert math.isfinite(score)
    assert score == pytest.approx(expected)


def test_mc2_probability_mass_uses_labels_instead_of_choice_order():
    doc = {"mc2_targets": {"labels": [0, 1, 0, 1]}}
    results = [(0.0, False), (-1.0, False), (-2.0, False), (-3.0, False)]

    score = process_results_mc2(doc, results)["acc"]

    denominator = sum(math.exp(value) for value in (0.0, -1.0, -2.0, -3.0))
    expected = (math.exp(-1.0) + math.exp(-3.0)) / denominator
    assert score == pytest.approx(expected)


def test_truthfulqa_mc2_override_uses_stable_scorer():
    task_root = Path(__file__).parents[2] / "eval" / "lm_eval_tasks"
    task_manager = TaskManager(include_path=[str(task_root)])
    entry = task_manager.task_index["truthfulqa_mc2"]

    assert entry.yaml_path == task_root / "truthfulqa" / "truthfulqa_mc2.yaml"
    config = load_yaml(entry.yaml_path, resolve_func=True)
    scorer_path = Path(inspect.getsourcefile(config["process_results"]))
    assert scorer_path.resolve() == (task_root / "truthfulqa" / "utils.py").resolve()
