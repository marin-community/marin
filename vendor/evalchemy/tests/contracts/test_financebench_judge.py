"""FinanceBench source and answer-equivalence judge parity."""

import json

import pytest

from eval.completion_response import FailedGeneration
from eval.chat_benchmarks.FinanceBench.eval_instruct import FinanceBenchBenchmark, PROMPT
from eval.contracts.eval_spec import SourceRef
from eval.sources.common import SourceOptions
from tests.contracts.source_harness import FixedEndpoint, evaluate_source


def test_financebench_preserves_document_prompt_and_equivalence_judge_plan(tmp_path):
    row = {"id": "filing-7", "question": "What was revenue?", "answer": "$10", "evidence_text": "Revenue was $10."}
    data_file = tmp_path / "financebench.jsonl"
    data_file.write_text(json.dumps(row) + "\n")
    source = FinanceBenchBenchmark(data_file=str(data_file))
    candidate = FixedEndpoint("$10")
    judge = FixedEndpoint("correct")

    metrics, samples = evaluate_source(
        source,
        SourceOptions("FinanceBench", SourceRef(str(data_file))),
        candidate,
        judges={"FinanceBench": judge},
    )

    assert metrics["accuracy"] == 1.0
    assert samples[0]["judge_label"] == "correct"
    assert candidate.requests[0].prompt == [{"role": "user", "content": PROMPT.format(**row)}]
    assert 'Reference answer(s):\n["$10"]' in judge.requests[0].prompt


@pytest.mark.parametrize("available", [0, 1])
def test_financebench_excludes_unavailable_judges_from_accuracy(tmp_path, available):
    rows = [
        {"id": "filing-1", "question": "What was revenue?", "answer": "$10", "evidence_text": "Revenue was $10."},
        {"id": "filing-2", "question": "What was income?", "answer": "$5", "evidence_text": "Income was $5."},
    ]
    data_file = tmp_path / "financebench.jsonl"
    data_file.write_text("".join(json.dumps(row) + "\n" for row in rows))
    judge = FixedEndpoint(lambda request: "correct" if len(judge.requests) <= available else FailedGeneration("model_transport"))

    metrics, _samples = evaluate_source(
        FinanceBenchBenchmark(data_file=str(data_file)),
        SourceOptions("FinanceBench", SourceRef(str(data_file))),
        FixedEndpoint("candidate"),
        judges={"FinanceBench": judge},
    )

    assert metrics["accuracy"] == (1.0 if available else None)
    assert metrics["num_judged"] == available
    assert metrics["num_judge_failed"] == 2 - available
    assert metrics["judge_coverage"] == pytest.approx(available / 2)
