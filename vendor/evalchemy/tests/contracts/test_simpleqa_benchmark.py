"""SimpleQA source data and public execution-boundary behavior."""

import pytest

from eval.chat_benchmarks.SimpleQA.eval_instruct import DATASET_SIZE, SimpleQABenchmark
from eval.chat_benchmarks.SimpleQAMini.eval_instruct import MINI_DATASET_SIZE, SimpleQAMiniBenchmark
from eval.chat_benchmarks.SimpleQAVerified.eval_instruct import (
    DATASET_SIZE as VERIFIED_DATASET_SIZE,
    SimpleQAVerifiedBenchmark,
)
from eval.contracts.eval_spec import SourceRef
from eval.graders.simpleqa import SIMPLEQA_VERIFIED_GRADER_TEMPLATE
from eval.sources.common import SourceOptions
from tests.contracts.source_harness import FixedEndpoint, evaluate_source


@pytest.mark.data_integration
def test_simpleqa_loads_the_canonical_full_dataset():
    questions = SimpleQABenchmark().load_questions()

    assert len(questions) == DATASET_SIZE
    assert questions[0]["source_index"] == 0
    assert questions[0]["question"] == "Who received the IEEE Frank Rosenblatt Award in 2010?"
    assert questions[0]["answer"] == "Michio Sugeno"


@pytest.mark.data_integration
def test_simpleqa_mini_uses_the_reference_seeded_subset():
    questions = SimpleQAMiniBenchmark().load_questions()

    assert len(questions) == MINI_DATASET_SIZE
    assert [question["source_index"] for question in questions[:10]] == [3155, 3445, 331, 2121, 4188, 3980, 3317, 2484, 3904, 2933]
    assert len({question["source_index"] for question in questions}) == MINI_DATASET_SIZE


@pytest.mark.data_integration
def test_simpleqa_verified_loads_the_corrected_publisher_revision():
    questions = SimpleQAVerifiedBenchmark().load_questions()

    assert len(questions) == VERIFIED_DATASET_SIZE
    # The publisher corrected this question on 2026-03-09; older mirrors still ask for the latitude.
    lilongwe = next(question for question in questions if question["metadata"]["original_index"] == 55)
    assert lilongwe["question"] == "What is the longitude of Lilongwe in decimal format?"


def test_simpleqa_verified_grades_questions_with_the_verified_autorater():
    source = SimpleQAVerifiedBenchmark()
    rows = source.load_questions()
    candidate = FixedEndpoint("candidate answer")
    judge = FixedEndpoint(lambda request: "A" if rows[0]["question"] in request.prompt else "C")

    metrics, samples = evaluate_source(
        source,
        SourceOptions("SimpleQAVerified", source.SOURCE),
        candidate,
        judges={"SimpleQAVerified": judge},
    )

    assert sorted(request.prompt[0]["content"] for request in candidate.requests) == sorted(
        row["question"] for row in rows
    )
    assert sorted(request.prompt for request in judge.requests) == sorted(
        SIMPLEQA_VERIFIED_GRADER_TEMPLATE.format(
            question=row["question"], target=row["answer"], predicted_answer="candidate answer"
        )
        for row in rows
    )
    assert (metrics["correct_rate"], metrics["incorrect_rate"], metrics["not_attempted_rate"]) == (0.5, 0.0, 0.5)
    assert {sample["doc"]["metadata"]["original_index"]: sample["judge_label"] for sample in samples} == {
        5: "correct",
        8: "not_attempted",
    }


@pytest.mark.parametrize("benchmark_class", [SimpleQABenchmark, SimpleQAMiniBenchmark])
def test_simpleqa_uses_candidate_and_judge_endpoints_for_the_limited_source(benchmark_class):
    source = benchmark_class(debug=True)
    candidate = FixedEndpoint("candidate answer")
    judge = FixedEndpoint("A")
    task = source.__class__.__name__

    metrics, samples = evaluate_source(
        source,
        SourceOptions(task, source.SOURCE, limit=1),
        candidate,
        judges={task: judge},
    )

    assert metrics["accuracy"] == 1.0
    assert samples[0]["judge_label"] == "correct"
    assert candidate.requests[0].prompt == [{"role": "user", "content": samples[0]["doc"]["question"]}]
    assert "Predicted answer: candidate answer" in judge.requests[0].prompt


def test_simpleqa_csv_retains_metadata_and_seeded_source_identity():
    full = SimpleQABenchmark().load_questions()
    mini = SimpleQAMiniBenchmark().load_questions()

    assert full[0] == {"source_index": 0, "question": "question 0", "answer": "answer 0", "metadata": {"topic": "fixture"}}
    assert [row["source_index"] for row in mini[:10]] == [3155, 3445, 331, 2121, 4188, 3980, 3317, 2484, 3904, 2933]
    assert all(row == full[row["source_index"]] for row in mini)
    assert len({row["source_index"] for row in mini}) == MINI_DATASET_SIZE


def test_simpleqa_reasoning_wrapped_label_is_an_unavailable_judgment(tmp_path):
    data = tmp_path / "simpleqa.csv"
    data.write_text("problem,answer,metadata\nCapital of France?,Paris,{}\nCapital of Spain?,Madrid,{}\n")
    judge = FixedEndpoint(lambda _request: "A" if len(judge.requests) == 1 else "<think>reasoning</think>\nA")

    metrics, samples = evaluate_source(
        SimpleQABenchmark(data_file=str(data)), SourceOptions("SimpleQA", SourceRef("fixture")),
        FixedEndpoint("candidate"), judges={"SimpleQA": judge},
    )

    assert metrics["accuracy"] == 1.0
    assert metrics["num_judged"] == 1
    assert metrics["num_judge_failed"] == 1
    assert metrics["judge_coverage"] == 0.5
    failed = next(sample for sample in samples if sample["judge_failed"])
    assert failed["failure_category"] == "grader_infrastructure"
    assert "accuracy" not in failed
