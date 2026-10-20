"""Regression coverage for runtime samples written to FineStore."""

from __future__ import annotations

import json

import pytest
from finestore.eval import ARCHIVE_SAMPLES_TABLE, sample_from_archive_row
from finestore.reader import ReadView

from eval.chat_benchmarks.FinanceBench.eval_instruct import FinanceBenchBenchmark
from eval.contracts.eval_spec import ReducerSpec, SourceRef
from eval.contracts.finestore_output import FineStoreOutput
from eval.sources.common import SourceOptions, text_specs
from tests.contracts.source_harness import FixedEndpoint, evaluate_source


class ExactMatchSource:
    REDUCER = ReducerSpec()

    def __init__(self, task_name, document):
        self.task_name = task_name
        self.document = document

    def iter_specs(self, options):
        return text_specs(
            [self.document],
            options,
            prompt=lambda row: row["prompt"],
            target=lambda row: row["answer"],
            grader="exact_match",
            generation_kwargs={},
        )


def _archive(tmp_path, task_name, records):
    root = str(tmp_path / "archive")
    with FineStoreOutput(root, task_name, [task_name]) as output:
        for record in records:
            output.write_sample(task_name, record)
        output.finish_task()
    table = ReadView(root).scan(ARCHIVE_SAMPLES_TABLE)
    assert table is not None
    return [sample_from_archive_row(row) for row in table.to_pylist(maps_as_pydicts="strict")]


@pytest.mark.parametrize("task_name", ["MATH500", "GPQADiamond", "HumanEvalPlus", "MBPPPlus", "IFEval"])
def test_runtime_samples_write_canonical_finestore_records(tmp_path, task_name):
    document = {"id": task_name, "prompt": "What is 1 + 1?", "answer": "2"}
    _metrics, records = evaluate_source(
        ExactMatchSource(task_name, document),
        SourceOptions(task_name, SourceRef("fixture")),
        FixedEndpoint("2"),
    )
    samples = _archive(tmp_path, task_name, records)
    assert len(samples) == 1
    assert samples[0].task == task_name
    assert samples[0].metrics == {"accuracy": 1.0}
    assert json.loads(samples[0].doc) == document


def test_judge_runtime_keeps_candidate_output_out_of_the_persisted_document(tmp_path):
    row = {"id": "filing-7", "question": "What was revenue?", "answer": "$10", "evidence_text": "Revenue was $10."}
    data_file = tmp_path / "finance.jsonl"
    data_file.write_text(json.dumps(row) + "\n")
    _metrics, records = evaluate_source(
        FinanceBenchBenchmark(data_file=str(data_file)),
        SourceOptions("FinanceBench", SourceRef(str(data_file))),
        FixedEndpoint("$10"),
        judges={"FinanceBench": FixedEndpoint("correct")},
    )
    samples = _archive(tmp_path, "FinanceBench", records)
    assert samples[0].metrics["accuracy"] == 1.0
    assert json.loads(samples[0].doc) == row
    assert records[0]["filtered_resps"] == ["$10"]
