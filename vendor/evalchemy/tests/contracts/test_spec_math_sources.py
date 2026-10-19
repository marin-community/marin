"""Offline source fixtures for math and instruction benchmark item protocols."""

import json
from pathlib import Path

import pytest

from eval.chat_benchmarks.AIME24.eval_instruct import AIME24Benchmark
from eval.chat_benchmarks.AIME25.eval_instruct import AIME25Benchmark
from eval.chat_benchmarks.AIW.eval_instruct import AIWBenchmark
from eval.chat_benchmarks.AMC23.eval_instruct import AMC23Benchmark
from eval.chat_benchmarks.HLE.eval_instruct import HLESubsetBenchmark
from eval.chat_benchmarks.HMMT.eval_instruct import HMMTBenchmark
from eval.chat_benchmarks.HLE.testing_utils import get_multiple_choice_answer as extract_hle_answer
from eval.chat_benchmarks.IFEval.eval_instruct import IFEvalBenchmark
from eval.chat_benchmarks.JEEBench.eval_instruct import JEEBenchBenchmark
from eval.contracts.eval_spec import SourceRef
from eval.sources.common import SourceOptions
from tests.contracts.source_harness import FixedEndpoint, evaluate_source


def _options(task):
    return SourceOptions(task, SourceRef("fixture"), seed=7)


def test_aiw_selects_target_ids_before_expanding_trials(tmp_path):
    data = tmp_path / "aiw.json"
    data.write_text(json.dumps([
        {"id": 1, "prompt": "skip", "right_answer": 0},
        {"id": 577, "prompt": "keep", "right_answer": 42},
    ]))

    specs = list(AIWBenchmark(data_file=data, n_trials=2).iter_specs(_options("AIW")))

    assert [spec.source_id for spec in specs] == [577, 577]
    assert [spec.trial for spec in specs] == [0, 1]
    assert specs[0].target == "42"
    assert specs[0].requests[0].generation["temperature"] == 1.0
    assert [spec.requests[0].generation["seed"] for spec in specs] == [7, 8]


def test_aiw_reports_final_trial_statistics_and_retains_every_scored_trial(tmp_path):
    data = tmp_path / "aiw.json"
    data.write_text(json.dumps([
        {"id": 577, "prompt": "first", "right_answer": 42},
        {"id": 580, "prompt": "second", "right_answer": 42},
    ]))
    answers = {577: (42, 42, 0), 580: (0, 42, 42)}
    endpoint = FixedEndpoint(lambda request: f"Answer: {answers[request.metadata['source_id']][request.metadata['trial']]}")

    metrics, samples = evaluate_source(AIWBenchmark(data_file=data, n_trials=3), _options("AIW"), endpoint)

    # The original source overwrites model_answer on each trial, then reports
    # statistics over that final answer. Earlier successes must not enter its
    # headline score, although every generated trial remains an auditable sample.
    assert metrics == {
        "num_trials": 3, "num_examples": 2,
        "per_id_stats": {"577": {"accuracy": 0.0, "variance": 0.0}, "580": {"accuracy": 1.0, "variance": 0.0}},
        "overall_accuracy": 0.5, "overall_variance": 0.25,
    }
    assert len(samples) == 6
    assert sum(sample["accuracy"] for sample in samples) == 4


def test_aime24_keeps_legacy_per_trial_sampling_seeds(tmp_path):
    data = tmp_path / "aime24.jsonl"
    data.write_text(json.dumps({"id": "aime", "problem": "Find 1", "expected_answer": "1"}) + "\n")

    source = AIME24Benchmark(data_file=data)
    specs = list(source.iter_specs(_options("AIME24")))

    assert len(specs) == 10
    assert [spec.requests[0].generation["seed"] for spec in specs] == list(range(7, 17))
    assert specs[0].requests[0].generation["until"]
    assert specs[0].grader.reference == {"comparator": "math_answers_equivalent"}

    source.n_repeat = 2
    endpoint = FixedEndpoint(lambda request: "\\boxed{" + str(1 - request.metadata["trial"]) + "}")
    metrics, samples = evaluate_source(source, _options("AIME24"), endpoint)

    assert len(endpoint.requests) == len(samples) == 2
    assert sorted(sample["sample_repeat"] for sample in samples) == [0, 1]
    assert sorted(request.generation["seed"] for request in endpoint.requests) == [7, 8]
    assert metrics["accuracy_avg"] == 0.5


def test_math_source_comparator_protocols_are_serialized(tmp_path):
    aime25_data = tmp_path / "aime25.jsonl"
    aime25_data.write_text(json.dumps({"id": "aime", "problem": "Find 1", "answer": "1"}) + "\n")
    amc_data = tmp_path / "amc.jsonl"
    amc_data.write_text(json.dumps({"id": "amc", "question": "Find 1", "answer": "1"}) + "\n")

    aime25 = next(AIME25Benchmark(data_file=str(aime25_data)).iter_specs(_options("AIME25")))
    amc = next(AMC23Benchmark(data_file=str(amc_data)).iter_specs(_options("AMC23")))

    assert aime25.grader.reference == {"comparator": "math_answers_equivalent"}
    assert amc.grader.reference == {"extractor": "hendrycks_math", "comparator": "hendrycks_math"}


def test_hle_and_hmmt_preserve_prompt_and_repeat_protocols(monkeypatch):
    hle = HLESubsetBenchmark()
    monkeypatch.setattr(hle, "load_questions", lambda: [{"id": "h", "answer": "B", "answer_type": "multipleChoice", "image": "", "question": "Choose B"}])
    hle_specs = list(hle.iter_specs(_options("HLE")))
    assert [spec.trial for spec in hle_specs] == [0, 1, 2]
    assert hle_specs[0].requests[0].prompt == [{"role": "user", "content": "Your response should be in the following format:\nAnswer: {your chosen multiple choice letter}. Include only the letter, no other text.\nChoose B"}]

    hmmt = HMMTBenchmark()
    monkeypatch.setattr(hmmt, "load_questions", lambda: [{"id": "m", "problem": "Find x", "answer": "3"}])
    hmmt_specs = list(hmmt.iter_specs(_options("HMMT")))
    assert len(hmmt_specs) == 10
    assert hmmt_specs[0].requests[0].prompt[0]["content"].endswith("\\boxed{}.\nAnswer:")


@pytest.mark.parametrize("response,target,expected", [
    ("Answer: E", "E", 1.0),
    ("Answer: B. Choice A is wrong", "B", 1.0),
    (r"Answer: \boxed{F}", "F", 1.0),
    ("Answer: A", "B", 0.0),
])
def test_hle_grades_its_native_answer_extraction(monkeypatch, response, target, expected):
    source = HLESubsetBenchmark()
    source.n_repeat = 1
    monkeypatch.setattr(source, "load_questions", lambda: [{
        "id": "hle", "answer": target, "answer_type": "multipleChoice", "image": "", "question": "Choose a letter",
    }])

    metrics, samples = evaluate_source(source, _options("HLE"), FixedEndpoint(response))

    assert float(extract_hle_answer(response) == target) == expected
    assert metrics["accuracy_avg"] == expected
    assert samples[0]["accuracy"] == expected


@pytest.mark.parametrize("target, expected", [(r"\frac{1311}{2017}", 0.0), (r"1 - \frac{1}{2025!}", 1.0)])
def test_hmmt_scores_large_rational_without_expanding_it_for_sample_audit(monkeypatch, target, expected):
    # The saved H100 response parses successfully, but expanding 2025! to text
    # exceeds Python's integer-string limit after native correctness is known.
    answer = r"1 - \frac{1}{2025!}"
    source = HMMTBenchmark()
    source.n_repeat = 1
    monkeypatch.setattr(source, "load_questions", lambda: [{"id": "large", "problem": "Find the fraction", "answer": target}])

    metrics, samples = evaluate_source(source, _options("HMMT"), FixedEndpoint(rf"\boxed{{{answer}}}"))

    assert metrics["accuracy_avg"] == expected
    assert len(samples) == 1
    assert samples[0]["accuracy"] == expected


def test_hmmt_keeps_native_integer_list_and_unboxed_scores(monkeypatch):
    answers = {
        "integer": ("2", r"\boxed{2}"),
        "list": ("1,2", r"\boxed{2,1}"),
        "unboxed": ("17", "The answer is 17"),
        "expression": ("8", r"\boxed{2^3}"),
    }
    source = HMMTBenchmark()
    source.n_repeat = 1
    monkeypatch.setattr(source, "load_questions", lambda: [
        {"id": name, "problem": name, "answer": target} for name, (target, _) in answers.items()
    ])
    endpoint = FixedEndpoint(lambda request: answers[request.metadata["source_id"]][1])

    metrics, samples = evaluate_source(source, _options("HMMT"), endpoint)

    assert metrics["accuracy_avg"] == 1.0
    assert len(samples) == 4
    assert all(sample["accuracy"] == 1.0 for sample in samples)


def test_jeebench_and_ifeval_specs_keep_source_selection(tmp_path, monkeypatch):
    jee = JEEBenchBenchmark()
    monkeypatch.setattr(jee, "load_questions", lambda: [{"description": "JEE Adv 2016 Paper 1", "index": 3, "type": "Integer", "question": "\n\nFind 2.", "gold": "2", "subject": "math"}])
    jee_specs = list(jee.iter_specs(_options("JEEBench")))
    assert len(jee_specs) == 3
    assert jee_specs[0].grader.name == "jee"
    assert "Problem: Find 2." in jee_specs[0].requests[0].prompt[0]["content"]

    data = tmp_path / "input_data.jsonl"
    rows = [{"key": index, "prompt": f"prompt {index}", "instruction_id_list": ["length_constraints:number_words"], "kwargs": [{}]} for index in range(12)]
    data.write_text("".join(json.dumps(row) + "\n" for row in rows))
    ifeval_specs = list(IFEvalBenchmark(data_dir=str(tmp_path), start_idx=10, end_idx=12).iter_specs(_options("IFEval")))
    assert [spec.source_id for spec in ifeval_specs] == [10, 11]
    assert ifeval_specs[0].grader.reference == {"protocol": "ifeval"}


def test_jeebench_scores_each_paper_when_local_question_indices_repeat(monkeypatch):
    # Four original rows from the pinned dataset revision recorded in the fixture.
    # Question 1 appears in different papers and, across years, different subjects.
    metadata = json.loads((Path(__file__).parent / "fixtures/jeebench-identities.json").read_text())
    rows = metadata["rows"]
    source = JEEBenchBenchmark()
    monkeypatch.setattr(source, "load_questions", lambda: rows)
    specs = list(source.iter_specs(_options("JEEBench")))
    answers = {spec.source_id: spec.target for spec in specs}

    for row, spec in zip(rows, specs[::3], strict=True):
        assert spec.doc == row
        assert spec.target == row["gold"]
        assert row["question"].replace("\n\n", "\n").strip() in spec.requests[0].prompt[0]["content"]
    endpoint = FixedEndpoint(lambda request: "\\boxed{" + answers[request.metadata["source_id"]] + "}")
    metrics, samples = evaluate_source(source, _options("JEEBench"), endpoint)

    assert metrics["accuracy_avg"] == 1.0
    assert len(samples) == 12
    assert len({sample["sample_id"] for sample in samples}) == 4
    assert len({(sample["sample_id"], sample["sample_repeat"]) for sample in samples}) == 12
