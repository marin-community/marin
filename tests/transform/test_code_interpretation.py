# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import gzip
import json
from pathlib import Path

from marin.transform.evaluation.code_interpretation import (
    CODE_INTERPRETATION_NUM_FEWSHOT,
    CODE_INTERPRETATION_TASKS_BY_KEY,
    CODE_INTERPRETATION_TEMPLATES,
    DEFAULT_CODE_INTERPRETATION_OUTPUT_FILENAME,
    CodeInterpretationStagingConfig,
    code_interpretation_record,
    stage_code_interpretation_source,
)


def _read_records(output_path: Path) -> list[dict]:
    with gzip.open(output_path / DEFAULT_CODE_INTERPRETATION_OUTPUT_FILENAME, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_repl_expression_record_scores_only_output_line() -> None:
    record = code_interpretation_record("expression_strings_collections", "python_repl", 0)

    assert record["input"].endswith(">>> len('prompt') * 7\n")
    assert record["target"] == "42"
    assert record["target"] not in record["input"]
    assert record["input"].count(">>> ") >= CODE_INTERPRETATION_NUM_FEWSHOT + 1
    assert record["provenance"]["task_family"] == "expression_only"
    assert record["provenance"]["semantic_target"] == "42"


def test_doctest_function_definition_record_completes_to_expected_output() -> None:
    record = code_interpretation_record("function_definition_calls", "python_doctest", 0)
    rendered = record["input"] + record["target"]

    assert (
        ">>> def edge_repeat(text):\n...     return text[0] + text[-1] * 2\n>>> edge_repeat('marin')\n"
        in record["input"]
    )
    assert rendered.endswith(">>> edge_repeat('marin')\nmnn")
    assert "mnn" not in record["input"]
    assert record["provenance"]["task_family"] == "function_definition"
    assert record["provenance"]["heldout_code"].startswith("def edge_repeat")


def test_function_definition_class_record_preserves_semantic_target_and_provenance() -> None:
    record = code_interpretation_record("function_definition_calls", "python_repl", 2)

    assert "class PairBox:" in record["input"]
    assert "PairBox('north', 'east').flipped().upper()" in record["input"]
    assert record["target"] == "EAST-NORTH"
    assert record["target"] not in record["input"]
    assert record["provenance"]["support_example_ids"] == [
        "fn-support-01",
        "fn-support-02",
        "fn-support-03",
        "fn-support-04",
        "fn-support-05",
    ]
    assert record["provenance"]["semantic_target"] == "EAST-NORTH"


def test_every_slice_renders_heldout_query_unfinished() -> None:
    for task in CODE_INTERPRETATION_TASKS_BY_KEY.values():
        for template in CODE_INTERPRETATION_TEMPLATES:
            for heldout_index, heldout in enumerate(task.heldout_examples):
                record = code_interpretation_record(task.key, template.key, heldout_index)
                unfinished_query = template.renderer(heldout, False)
                finished_query = template.renderer(heldout, True)

                assert record["input"].endswith(unfinished_query)
                assert not record["input"].endswith(finished_query)
                assert record["target"] == finished_query[len(unfinished_query) :]
                assert record["target"] not in record["input"]
                assert (record["input"] + record["target"]).endswith(finished_query)
                assert record["provenance"]["semantic_target"] == heldout.target


def test_stage_code_interpretation_source_writes_valid_target_only_jsonl(tmp_path: Path) -> None:
    result = stage_code_interpretation_source(
        CodeInterpretationStagingConfig(
            output_path=str(tmp_path),
            task_key="function_definition_calls",
            template_key="python_repl",
        )
    )

    task = CODE_INTERPRETATION_TASKS_BY_KEY["function_definition_calls"]
    records = _read_records(tmp_path)
    assert result["record_count"] == len(task.heldout_examples)
    assert len(records) == len(task.heldout_examples)
    for record, heldout in zip(records, task.heldout_examples, strict=True):
        assert set(record) == {"id", "input", "target", "source", "provenance"}
        assert record["input"].endswith("\n")
        assert record["target"] == heldout.target
        assert heldout.target not in record["input"]
        assert record["provenance"]["task_family"] == "function_definition"
        assert record["provenance"]["num_fewshot"] == CODE_INTERPRETATION_NUM_FEWSHOT
