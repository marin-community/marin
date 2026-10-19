"""Regression coverage for the packaged CruxEval evaluator."""

import importlib


def test_cruxeval_evaluator_imports_from_installed_package():
    evaluator = importlib.import_module("eval.chat_benchmarks.CruxEval.evaluation")

    assert callable(evaluator.evaluate_generations)
