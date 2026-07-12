# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import gzip
import json
from pathlib import Path

from marin.transform.evaluation.prompt_format_sensitivity import (
    DEFAULT_PROMPT_FORMAT_OUTPUT_FILENAME,
    PROMPT_FORMAT_NUM_FEWSHOT,
    PROMPT_FORMAT_TASKS_BY_KEY,
    PROMPT_FORMAT_TEMPLATES,
    PromptFormatSensitivityStagingConfig,
    prompt_format_record,
    stage_prompt_format_sensitivity_source,
)


def _read_records(output_path: Path) -> list[dict]:
    with gzip.open(output_path / DEFAULT_PROMPT_FORMAT_OUTPUT_FILENAME, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_prompt_format_record_keeps_heldout_target_out_of_input() -> None:
    record = prompt_format_record("mcqa_science", "qa", 0)

    assert record["target"] == "The root takes in water from soil."
    assert record["input"].endswith("A: ")
    assert record["target"] not in record["input"]
    assert record["input"].count("\nA: ") == PROMPT_FORMAT_NUM_FEWSHOT + 1
    assert record["provenance"]["support_example_ids"] == [
        "mcqa-support-01",
        "mcqa-support-02",
        "mcqa-support-03",
        "mcqa-support-04",
        "mcqa-support-05",
    ]


def test_prompt_format_templates_preserve_identical_targets_for_same_task_item() -> None:
    template_keys = ["plain_arrow", "json_object", "yaml", "csv", "python_doctest", "shell_transcript"]
    records = [prompt_format_record("short_factual_qa", template_key, 1) for template_key in template_keys]
    semantic_target = "Leonardo da Vinci painted the Mona Lisa."

    assert len({record["input"] for record in records}) == len(template_keys)
    for record in records:
        assert semantic_target in record["target"]
        assert semantic_target not in record["input"]
        assert record["provenance"]["semantic_target"] == semantic_target
        assert record["provenance"]["num_fewshot"] == 5


def test_prompt_format_markdown_table_renders_supports_and_unfinished_query() -> None:
    record = prompt_format_record("string_transformation", "markdown_table", 0)

    assert "| Input | Output |" not in record["input"]
    assert "| region=west; id=42; color=blue | WEST-0042-BLUE" in record["input"]
    assert "| region=west; id=108; color=orange | " in record["input"]
    assert "WEST-0108-ORANGE" not in record["input"]
    assert record["target"] == "WEST-0108-ORANGE"


def test_prompt_format_jsonl_target_completes_valid_heldout_record() -> None:
    record = prompt_format_record("mcqa_science", "jsonl", 2)
    rendered = record["input"] + record["target"]

    jsonl_records = [json.loads(line) for line in rendered.splitlines() if line.startswith("{")]

    assert len(jsonl_records) == PROMPT_FORMAT_NUM_FEWSHOT + 1
    assert jsonl_records[-1] == {
        "input": "Question: Which material is a good electrical conductor?\nA. Rubber\nB. Copper\nC. Glass\nD. Wood",
        "output": "Copper is a good electrical conductor.",
    }


def test_prompt_format_record_extraction_uses_template_neutral_field_list() -> None:
    record = prompt_format_record("record_extraction", "qa", 0)

    assert '"order_id"' not in record["input"]
    assert '"order_id"' not in record["target"]
    assert record["provenance"]["semantic_target"] == (
        "order_id=2880; owner=Jules; city=Tallinn; date=2026-06-03; items=sensors; quantity=9"
    )
    assert record["input"].endswith("A: ")
    assert record["target"] == record["provenance"]["semantic_target"]


def test_prompt_format_every_slice_renders_heldout_query_unfinished() -> None:
    for task in PROMPT_FORMAT_TASKS_BY_KEY.values():
        for template in PROMPT_FORMAT_TEMPLATES:
            for heldout_index, heldout in enumerate(task.heldout_examples):
                record = prompt_format_record(task.key, template.key, heldout_index)
                unfinished_query = template.renderer(heldout, False)
                finished_query = template.renderer(heldout, True)

                assert record["input"].endswith(unfinished_query)
                assert not record["input"].endswith(finished_query)
                assert record["target"] == finished_query[len(unfinished_query) :]
                assert (record["input"] + record["target"]).endswith(finished_query)
                assert record["provenance"]["semantic_target"] == heldout.target


def test_stage_prompt_format_sensitivity_source_writes_target_only_records(tmp_path: Path) -> None:
    result = stage_prompt_format_sensitivity_source(
        PromptFormatSensitivityStagingConfig(
            output_path=str(tmp_path),
            task_key="record_extraction",
            template_key="xml",
        )
    )

    task = PROMPT_FORMAT_TASKS_BY_KEY["record_extraction"]
    records = _read_records(tmp_path)
    assert result["record_count"] == len(task.heldout_examples)
    for record, heldout in zip(records, task.heldout_examples, strict=True):
        assert record["input"].endswith("<output>")
        assert record["target"].endswith("</output></example>")
        assert heldout.target not in record["input"]
        assert record["provenance"]["semantic_target"] == heldout.target
        assert record["provenance"]["template_key"] == "xml"
        assert record["provenance"]["num_fewshot"] == PROMPT_FORMAT_NUM_FEWSHOT
