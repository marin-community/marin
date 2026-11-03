"""Regression tests for GPQA-Diamond answer placement."""

from eval.chat_benchmarks.GPQADiamond.eval_instruct import GPQADiamondBenchmark
from eval.contracts.eval_spec import SourceRef
from eval.sources.common import SourceOptions


def _question(index: int) -> dict[str, str]:
    return {
        "Question": f"Question {index}",
        "Correct Answer": "correct",
        "Incorrect Answer 1": "wrong one",
        "Incorrect Answer 2": "wrong two",
        "Incorrect Answer 3": "wrong three",
    }


def test_source_assigns_reproducibly_varied_answer_positions(monkeypatch):
    benchmark = GPQADiamondBenchmark()
    questions = [_question(index) for index in range(32)]
    monkeypatch.setattr(benchmark, "load_questions", lambda: questions)
    options = SourceOptions("GPQADiamond", SourceRef("Idavidrein/gpqa"))

    first = [(spec.target, spec.requests[0].prompt[0]["content"]) for spec in benchmark.iter_specs(options)]
    second = [(spec.target, spec.requests[0].prompt[0]["content"]) for spec in benchmark.iter_specs(options)]

    assert {answer for answer, _ in first} == {"A", "B", "C", "D"}
    assert first == second
