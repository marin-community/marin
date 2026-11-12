"""ZeroEval source routing and canonical answer interpretation."""

import json

import pytest
from datasets import Dataset

from eval.chat_benchmarks.zeroeval.eval_instruct import ZeroEvalBenchmark
from eval.contracts.eval_spec import SourceRef
from eval.sources.common import SourceOptions
from tests.contracts.source_harness import FixedEndpoint, evaluate_source


def _source(monkeypatch, task, target):
    source = ZeroEvalBenchmark(tasks=[task])
    monkeypatch.setattr(
        "eval.chat_benchmarks.zeroeval.src.task_configs.load_dataset",
        lambda *args, **kwargs: Dataset.from_list([{
            "id": "zero", "question": "Choose an answer", "problem": "Choose an answer",
            "choices": ["first", "second", "third", "fourth"], "answer": target,
        }]),
    )
    return source, SourceOptions("zeroeval", SourceRef("fixture"))


@pytest.mark.parametrize("answer,target,expected", [
    ("1/4+1/4", "0.5", 100.0),
    ("1/2+1", "1", 0.0),
    ("-(1/4 + 1/4)", "-0.5", 100.0),
    ("2**3/4", "2", 100.0),
    ("1/2", "0.5", 100.0),
    ("$1,200 miles", "1200", 100.0),
])
def test_zeroeval_numeric_answers_preserve_native_arithmetic(monkeypatch, answer, target, expected):
    response = json.dumps({"answer": answer, "reasoning": "computed"})
    source, options = _source(monkeypatch, "math-l5", target)

    metrics, samples = evaluate_source(source, options, FixedEndpoint(response))

    assert metrics["accuracy"] == expected
    assert samples[0]["accuracy"] == expected


@pytest.mark.parametrize("answer,target,expected", [("B", "B", 100.0), ("B", "A", 0.0), ("1/2+1", "1", 0.0)])
def test_zeroeval_mmlu_redux_keeps_literal_grading(monkeypatch, answer, target, expected):
    response = json.dumps({"answer": answer, "reasoning": "selected"})
    source, options = _source(monkeypatch, "mmlu-redux", target)

    metrics, samples = evaluate_source(source, options, FixedEndpoint(response))

    assert metrics["accuracy"] == expected
    assert samples[0]["accuracy"] == expected


def test_zeroeval_arithmetic_does_not_execute_python(monkeypatch, tmp_path):
    marker = tmp_path / "executed"
    answer = f"__import__('pathlib').Path({str(marker)!r}).write_text('executed')/2"
    source, options = _source(monkeypatch, "math-l5", "0.5")

    metrics, _ = evaluate_source(source, options, FixedEndpoint(json.dumps({"answer": answer})))

    assert metrics["accuracy"] == 0.0
    assert not marker.exists()


@pytest.mark.parametrize("answer,target", [
    ("2**1001/2", "2"),
    ("2/" + "1" * 4096, "2"),
    ("2/1" + "+1" * 64, "2"),
], ids=["exponent", "length", "nodes"])
def test_zeroeval_rejected_arithmetic_cannot_match_its_first_operand(monkeypatch, answer, target):
    source, options = _source(monkeypatch, "math-l5", target)

    metrics, samples = evaluate_source(source, options, FixedEndpoint(json.dumps({"answer": answer})))

    assert metrics["accuracy"] == 0.0
    assert samples[0]["accuracy"] == 0.0


@pytest.mark.parametrize("solution,answered", [({}, True), ([], True), ("", True), (None, False), ("missing", False)])
def test_zeroeval_grid_distinguishes_empty_solution_from_unanswered(monkeypatch, solution, answered):
    truth = {"header": ["House", "Name"], "rows": [["1", "Alice"], ["2", "Bob"]]}
    rows = [{"id": name, "puzzle": "Assign names", "size": "2*2", "solution": truth} for name in ("probe", "anchor")]
    monkeypatch.setattr("eval.chat_benchmarks.zeroeval.src.task_configs.load_dataset", lambda *args, **kwargs: Dataset.from_list(rows))
    monkeypatch.setattr("eval.chat_benchmarks.zeroeval.eval_instruct.load_dataset", lambda *args, **kwargs: Dataset.from_list(rows))
    probe = {"reasoning": "abc"}
    if solution != "missing":
        probe["solution"] = solution
    responses = {
        "probe": json.dumps(probe),
        "anchor": json.dumps({"reasoning": "abcde", "solution": {"House 1": {"Name": "Alice"}, "House 2": {"Name": "Bob"}}}),
    }
    source = ZeroEvalBenchmark(tasks=["zebra-grid"])

    metrics, samples = evaluate_source(
        source, SourceOptions("zeroeval", SourceRef("fixture")),
        FixedEndpoint(lambda request: responses[request.metadata["source_id"]]),
    )

    assert metrics["accuracy"] == 50.0
    assert metrics["cell_acc"] == 50.0
    assert metrics["no_answer"] == (0.0 if answered else 50.0)
    assert metrics["reason_lens"] == (4.0 if answered else 5.0)
    assert sum(sample["no_answer"] for sample in samples) == (0 if answered else 100)
