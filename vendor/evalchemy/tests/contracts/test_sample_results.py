# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Contract coverage for records emitted by EvalSpec sources."""

from __future__ import annotations

import json
from typing import Any

import pytest
from finestore.eval import ARCHIVE_SAMPLES_TABLE, sample_from_archive_row
from finestore.reader import ReadView

from eval.chat_benchmarks.FinanceBench.eval_instruct import FinanceBenchBenchmark
from eval.chat_benchmarks.HumanEvalPlus.eval_instruct import HumanEvalPlusBenchmark
from eval.chat_benchmarks.MTBench import eval_instruct as mtbench_source
from eval.chat_benchmarks.MixEval.eval_instruct import MixEvalBenchmark
from eval.chat_benchmarks.SimpleQA.eval_instruct import SimpleQABenchmark
from eval.chat_benchmarks.WildBench import eval_instruct as wildbench_source
from eval.contracts.eval_spec import ReducerSpec, SourceRef
from eval.contracts.finestore_output import FineStoreOutput
from eval.contracts.sample_results import (
    SAMPLE_METRICS_FIELD,
    SampleMetricsError,
    record_sample_metrics,
    sample_metric_fields,
    validate_sample_metrics,
)
from eval.sample_logging import canonicalize_samples
from eval.sources.common import SourceOptions, text_specs
from tests.contracts.source_harness import FixedEndpoint, evaluate_source


def _persist(tmp_path, task_name, records):
    tmp_path.mkdir()
    root = str(tmp_path / "archive")
    with FineStoreOutput(root, "run", [task_name]) as output:
        for record in records:
            output.write_sample(task_name, record)
        output.finish_task()
    table = ReadView(root).scan(ARCHIVE_SAMPLES_TABLE)
    assert table is not None
    return [sample_from_archive_row(row) for row in table.to_pylist(maps_as_pydicts="strict")]


def test_humanevalplus_persists_scored_samples_in_the_generation_language(tmp_path):
    problems = [
        {"task_id": f"python/{index}", "prompt": "def answer():\n", "test": f"assert answer() == {expected}"}
        for index, expected in enumerate((42, 43))
    ]
    (tmp_path / "humanevalplus-python.jsonl").write_text("".join(json.dumps(problem) + "\n" for problem in problems))
    source = HumanEvalPlusBenchmark(data_dir=str(tmp_path), timeout=1.0)

    metrics, records = evaluate_source(
        source,
        SourceOptions("HumanEvalPlus", source.SOURCE),
        FixedEndpoint("```python\ndef answer():\n    return 42\n```"),
    )
    persisted = _persist(tmp_path / "archive-output", "HumanEvalPlus", records)

    assert {record["sample_namespace"] for record in records} == {"python"}
    assert {record["source_id"]: record["accuracy"] for record in records} == {"python/0": 1.0, "python/1": 0.0}
    assert {sample.doc_id: sample.metrics["accuracy"] for sample in persisted} == {
        record["sample_id"]: record["accuracy"] for record in records
    }
    assert metrics["python_pass@1"] == metrics["pass@1"] == 0.5


def test_judge_sources_emit_metrics_and_persist_records(tmp_path, monkeypatch):
    """Exercise judge-backed sources through candidate, judge, and sample persistence."""
    finance_row = {
        "id": "filing-7",
        "question": "What was revenue?",
        "answer": "$10",
        "evidence_text": "Revenue was $10.",
    }
    finance_file = tmp_path / "finance.jsonl"
    finance_file.write_text(json.dumps(finance_row) + "\n")
    simpleqa_file = tmp_path / "simpleqa.csv"
    simpleqa_file.write_text("problem,answer,metadata\nCapital of France?,Paris,{}\n")
    wildbench_row = {
        "session_id": "session-1",
        "conversation_input": [{"role": "user", "content": "question"}],
        "checklist": ["be right"],
        "primary_tag": "Math",
        "secondary_tags": [],
    }
    monkeypatch.setattr(wildbench_source, "load_dataset", lambda *_args, **_kwargs: [wildbench_row])
    cases = [
        (
            "FinanceBench",
            FinanceBenchBenchmark(data_file=str(finance_file)),
            FixedEndpoint("$10"),
            FixedEndpoint("correct"),
            {"accuracy"},
        ),
        (
            "SimpleQA",
            SimpleQABenchmark(data_file=str(simpleqa_file)),
            FixedEndpoint("Paris"),
            FixedEndpoint("A"),
            {"accuracy", "incorrect", "not_attempted", "judge_failed"},
        ),
        (
            "MixEval",
            MixEvalBenchmark(debug=True),
            FixedEndpoint("candidate"),
            FixedEndpoint(
                lambda request: "[[1.0]]" if "correctness" in request.prompt[-1]["content"].lower() else "[[C]]"
            ),
            {"accuracy"},
        ),
        (
            "WildBench",
            wildbench_source.WildBenchBenchmark(),
            FixedEndpoint("answer"),
            FixedEndpoint('{"score": "7"}'),
            {"score", "adjusted_score", "avg_output_length"},
        ),
    ]
    for task_name, source, candidate, judge, expected_metrics in cases:
        _metrics, records = evaluate_source(
            source, SourceOptions(task_name, SourceRef("fixture"), limit=1), candidate, judges={task_name: judge}
        )
        assert records
        assert all(expected_metrics <= set(record[SAMPLE_METRICS_FIELD]) for record in records)
        assert all(isinstance(record[name], (int, float)) for record in records for name in expected_metrics)
        persisted = _persist(tmp_path / task_name, task_name, records)
        assert len(persisted) == len(records)
        assert {sample.doc_id for sample in persisted} == {record["sample_id"] for record in records}


def test_mtbench_runtime_keeps_turn_identity_and_judge_scores(monkeypatch):
    question = {"question_id": 9, "category": "writing", "turns": ["first", "second"]}
    templates = {
        "single-v1": {"system_prompt": "judge", "prompt_template": "{question} {answer}"},
        "single-v1-multi-turn": {
            "system_prompt": "judge",
            "prompt_template": "{question_1} {answer_1} {question_2} {answer_2}",
        },
    }
    monkeypatch.setattr(mtbench_source, "load_questions", lambda *_args: [question])
    monkeypatch.setattr(mtbench_source, "load_judge_prompts", lambda *_args: templates)
    monkeypatch.setattr(mtbench_source, "load_model_answers", lambda *_args: {})
    metrics, records = evaluate_source(
        mtbench_source.MTBenchBenchmark(),
        SourceOptions("MTBench", SourceRef("fixture")),
        FixedEndpoint("candidate"),
        judges={"MTBench": FixedEndpoint("[[7]]")},
    )
    assert metrics["Average"] == 7.0
    assert records[0]["source_id"] == 9
    assert set(records[0]["metrics"]) == {"Turn 1", "Turn 2", "Average", "judge_failed"}
    assert len(records[0]["resps"]) == 2


@pytest.mark.parametrize("judge_model", ["gpt-4o-mini-2024-07-18", "gpt-4-turbo"])
def test_mtbench_uses_canonical_reference_answers_independent_of_judge(monkeypatch, judge_model):
    source = mtbench_source.MTBenchBenchmark(annotator_model=judge_model)
    questions = mtbench_source.load_questions(source.question_file, None, None)
    question = next(row for row in questions if row["category"] == "math")
    references = mtbench_source.load_model_answers(source.ref_answer_dir)
    expected = references["gpt-4"][question["question_id"]]["choices"][0]["turns"]
    monkeypatch.setattr(mtbench_source, "load_questions", lambda *_args: [question])

    [spec] = source.iter_specs(SourceOptions("MTBench", SourceRef("fixture")))

    assert spec.grader.reference["answers"] == expected


def test_repeated_eval_specs_keep_one_record_per_trial():
    class RepeatedSource:
        REDUCER = ReducerSpec()
        n_repeat = 3

        def iter_specs(self, options):
            return text_specs(
                [{"id": "question", "prompt": "Choose A", "answer": "A"}],
                options,
                prompt=lambda row: row["prompt"],
                target=lambda row: row["answer"],
                grader="exact_match",
                generation_kwargs={"do_sample": True},
                repeats=self.n_repeat,
                seed_repeats=True,
            )

    metrics, records = evaluate_source(
        RepeatedSource(),
        SourceOptions("Repeated", SourceRef("fixture")),
        FixedEndpoint(lambda request: "B" if request.generation["seed"] == 1 else "A"),
    )
    assert [record["sample_repeat"] for record in records] == [0, 1, 2]
    assert [record["accuracy"] for record in records] == [1.0, 0.0, 1.0]
    assert metrics["accuracy_avg"] == pytest.approx(2 / 3)


@pytest.mark.parametrize("task_name", ["ifeval", "ifeval_ca", "ifeval_es", "leaderboard_ifeval"])
def test_ifeval_native_instruction_metrics_preserve_flattened_accuracy(task_name):
    samples = [
        {
            "metrics": [
                "prompt_level_strict_acc",
                "prompt_level_loose_acc",
                "inst_level_strict_acc",
                "inst_level_loose_acc",
            ],
            "prompt_level_strict_acc": False,
            "prompt_level_loose_acc": True,
            "inst_level_strict_acc": [True, False, False],
            "inst_level_loose_acc": [True, True, True],
        },
        {
            "metrics": [
                "prompt_level_strict_acc",
                "prompt_level_loose_acc",
                "inst_level_strict_acc",
                "inst_level_loose_acc",
            ],
            "prompt_level_strict_acc": True,
            "prompt_level_loose_acc": True,
            "inst_level_strict_acc": [True],
            "inst_level_loose_acc": [True],
        },
    ]
    records = canonicalize_samples(task_name, samples)
    assert [record["inst_level_strict_acc"] for record in records] == [1 / 3, 1.0]
    assert [record["inst_level_loose_acc"] for record in records] == [1.0, 1.0]
    assert [record["strict_instruction_pass"] for record in records] == [[True, False, False], [True]]
    assert [record["loose_instruction_pass"] for record in records] == [[True, True, True], [True]]
    assert (
        sum(map(sum, (record["strict_instruction_pass"] for record in records)))
        / sum(len(record["strict_instruction_pass"]) for record in records)
        == 0.5
    )


def test_other_tasks_reject_list_valued_metrics():
    with pytest.raises(SampleMetricsError, match="must be a finite number"):
        canonicalize_samples("other_task", [{"metrics": ["accuracy"], "accuracy": [True, False]}])


def test_a_sample_set_missing_its_metrics_is_rejected_at_the_serialization_boundary():
    with pytest.raises(SampleMetricsError, match="persists no per-sample metrics"):
        canonicalize_samples("MATH500", [{"doc_id": 0, "resps": [["4"]]}])


def test_a_recorded_metric_may_not_shadow_a_reserved_record_field():
    with pytest.raises(SampleMetricsError, match="shadows a reserved sample-record field"):
        record_sample_metrics({}, target=1.0)


def test_recorded_booleans_and_partial_credit_reach_the_record_as_floats():
    example: dict[str, Any] = {}
    record_sample_metrics(example, accuracy=True, prefix_hit=0.25)
    fields = sample_metric_fields(example)
    assert set(fields[SAMPLE_METRICS_FIELD]) == {"accuracy", "prefix_hit"}
    assert (fields["accuracy"], fields["prefix_hit"]) == (1.0, 0.25)


def test_a_listed_metric_without_a_value_is_rejected():
    with pytest.raises(SampleMetricsError, match="carries no value"):
        validate_sample_metrics("MATH500", [{"metrics": ["accuracy"]}])
