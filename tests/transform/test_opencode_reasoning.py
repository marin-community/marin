# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import gzip
import json

import pytest
from marin.transform.conversation import opencode_reasoning as ocr2
from marin.transform.conversation.opencode_reasoning import (
    OCR2DropReason,
    OCR2HydratedPrompt,
    OCR2HydrationTask,
    OCR2TransformConfig,
    collect_ocr2_question_keys_from_rows,
    extract_apps_prompt,
    extract_code_contests_prompt,
    extract_codeforces_prompt,
    extract_taco_prompt,
    load_ocr2_hydration_cache,
    ocr2_source_key,
    ocr2_source_load_kwargs,
    parse_ocr2_source_key,
    process_ocr2_hydration_task,
    transform_ocr2_row,
)

OCR2_ROW = {
    "id": "ocr2-row-1",
    "question": "-",
    "r1_generation": "<think>Use brute force.</think>\n```python\nprint(42)\n```",
    "judgement": "right",
    "pass_rate": "0.9",
    "dataset": "taco",
    "split": "train",
    "index": 7,
    "question_id": "taco-question-7",
    "source": "TACO",
    "difficulty": "easy",
    "license": "apache-2.0",
}


def _transform_config(**overrides) -> OCR2TransformConfig:
    values = {
        "output_path": "/tmp/ocr2-output",
        "hydration_path": "/tmp/ocr2-hydration",
        "split": "python",
    }
    values.update(overrides)
    return OCR2TransformConfig(**values)


def _hydration_cache(prompt: str = "Return the answer to life.") -> dict[str, OCR2HydratedPrompt]:
    key = ocr2_source_key("taco", "train", 7)
    return {
        key: OCR2HydratedPrompt(
            key=key,
            prompt=prompt,
            source_dataset_id="BAAI/TACO",
            source_revision="d593ed0a2becbbc952230bb89be09189bf1056dc",
            source_license="apache-2.0",
        )
    }


def _write_hydration_records(root, records):
    data_dir = root / "data" / "taco"
    data_dir.mkdir(parents=True)
    path = data_dir / "train.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as writer:
        for record in records:
            writer.write(json.dumps(record) + "\n")
    return path


def test_ocr2_source_load_kwargs_use_file_backed_loaders_for_script_sources():
    taco_task = OCR2HydrationTask(
        ocr2_dataset_value="taco",
        source_dataset_id="BAAI/TACO",
        source_revision="d593ed0a2becbbc952230bb89be09189bf1056dc",
        source_split="train",
        source_license="apache-2.0",
        requested_indexes=("0",),
        question_ids_by_index={},
        output_path="/tmp/out",
    )
    apps_task = OCR2HydrationTask(
        ocr2_dataset_value="apps",
        source_dataset_id="codeparrot/apps",
        source_revision="21e74ddf8de1a21436da12e3e653065c5213e9d1",
        source_split="test",
        source_license="mit",
        requested_indexes=("0",),
        question_ids_by_index={},
        output_path="/tmp/out",
    )

    taco_kwargs = ocr2_source_load_kwargs(taco_task)
    apps_kwargs = ocr2_source_load_kwargs(apps_task)

    assert taco_kwargs == {
        "path": "arrow",
        "data_files": "hf://datasets/BAAI/TACO@d593ed0a2becbbc952230bb89be09189bf1056dc/train/data-*.arrow",
        "split": "train",
        "streaming": True,
    }
    assert apps_kwargs == {
        "path": "json",
        "data_files": "hf://datasets/codeparrot/apps@21e74ddf8de1a21436da12e3e653065c5213e9d1/test.jsonl",
        "split": "train",
        "streaming": True,
    }


def test_ocr2_source_key_preserves_dataset_slashes_and_normalizes_index():
    key = ocr2_source_key("open-r1/codeforces", "train", 13)

    assert key == "open-r1/codeforces/train/13"
    assert ocr2_source_key("open-r1/codeforces", "train", "13") == key
    assert parse_ocr2_source_key(key) == ("open-r1/codeforces", "train", "13")


def test_collect_ocr2_question_keys_from_rows_deduplicates_and_counts_unexpected_sources():
    rows = [
        {"dataset": "taco", "split": "train", "index": 1, "question_id": "q1"},
        {"dataset": "taco", "split": "train", "index": "1", "question_id": "q1-again"},
        {"dataset": "unexpected", "split": "train", "index": 2},
    ]

    summary = collect_ocr2_question_keys_from_rows(rows, ocr2_split="python")

    assert summary.keys == {"taco/train/1", "unexpected/train/2"}
    assert summary.rows_by_ocr2_split == {"python": 3}
    assert summary.duplicate_key_occurrences == 1
    assert summary.question_ids_by_key["taco/train/1"] == "q1"
    assert summary.unexpected_datasets == {"unexpected": 1}


def test_source_extractors_return_prompt_text():
    assert extract_taco_prompt({"question": "TACO prompt"}) == "TACO prompt"
    assert extract_apps_prompt({"question": "APPS prompt"}) == "APPS prompt"
    assert extract_code_contests_prompt({"description": "Contest prompt"}) == "Contest prompt"
    assert extract_code_contests_prompt({"description": ""}) is None

    codeforces_prompt = extract_codeforces_prompt(
        {
            "description": "Solve A+B.",
            "input_format": "Two integers a and b.",
            "output_format": "Their sum.",
            "examples": [{"input": "1 2", "output": "3"}],
            "note": "Use 64-bit integers.",
        }
    )

    assert codeforces_prompt == "\n\n".join(
        [
            "Solve A+B.",
            "Input\nTwo integers a and b.",
            "Output\nTheir sum.",
            "Examples\nExample 1 Input\n1 2\n\nExample 1 Output\n3",
            "Note\nUse 64-bit integers.",
        ]
    )


def test_process_ocr2_hydration_task_writes_prompt_cache_and_missing_metrics(tmp_path, monkeypatch):
    def source_rows(_task):
        return [
            {"question": "Hydrated TACO prompt"},
            {"question": "Unused prompt"},
        ]

    monkeypatch.setattr(ocr2, "_source_rows", source_rows)
    task = OCR2HydrationTask(
        ocr2_dataset_value="taco",
        source_dataset_id="BAAI/TACO",
        source_revision="d593ed0a2becbbc952230bb89be09189bf1056dc",
        source_split="train",
        source_license="apache-2.0",
        requested_indexes=("0", "2"),
        question_ids_by_index={"0": "question-0", "2": "question-2"},
        output_path=str(tmp_path),
    )

    metrics = process_ocr2_hydration_task(task)
    cache = load_ocr2_hydration_cache(str(tmp_path))

    assert cache["taco/train/0"].prompt == "Hydrated TACO prompt"
    assert metrics["requested_keys"] == 2
    assert metrics["hydrated_keys"] == 1
    assert metrics["missing_key_values"] == ["taco/train/2"]


def test_load_ocr2_hydration_cache_rejects_conflicting_duplicate_prompts(tmp_path):
    key = "taco/train/7"
    base_record = {
        "key": key,
        "prompt": "First prompt",
        "source_dataset_id": "BAAI/TACO",
        "source_revision": "d593ed0a2becbbc952230bb89be09189bf1056dc",
        "source_license": "apache-2.0",
    }
    _write_hydration_records(
        tmp_path,
        [
            base_record,
            {**base_record, "prompt": "Different prompt"},
        ],
    )

    with pytest.raises(ValueError, match="Conflicting OCR2 hydration records"):
        load_ocr2_hydration_cache(str(tmp_path))


def test_transform_ocr2_row_uses_hydrated_prompt_and_rewrites_think_tags():
    result = transform_ocr2_row(OCR2_ROW, _hydration_cache(), _transform_config())

    assert result.drop_reason is None
    assert result.record is not None
    assert result.record["source"] == "nvidia/OpenCodeReasoning-2"
    assert result.record["messages"] == [
        {
            "role": "user",
            "content": "Return the answer to life.",
            "name": None,
            "tool_calls": None,
            "tool_call_id": None,
        },
        {
            "role": "assistant",
            "content": "<|start_think|>Use brute force.<|end_think|>\n```python\nprint(42)\n```",
            "name": None,
            "tool_calls": None,
            "tool_call_id": None,
        },
    ]
    assert result.record["metadata"] == {
        "id": "ocr2-row-1",
        "question_id": "taco-question-7",
        "dataset": "taco",
        "split": "train",
        "index": 7,
        "source": "TACO",
        "difficulty": "easy",
        "license": "apache-2.0",
        "judgement": "right",
        "pass_rate": "0.9",
        "hydrated_prompt_key": "taco/train/7",
        "hydration_source_dataset_id": "BAAI/TACO",
        "hydration_source_revision": "d593ed0a2becbbc952230bb89be09189bf1056dc",
        "hydration_source_license": "apache-2.0",
    }


def test_transform_ocr2_row_filters_wrong_judgement_before_hydration_lookup():
    row = {**OCR2_ROW, "judgement": "wrong"}

    result = transform_ocr2_row(row, {}, _transform_config())

    assert result.record is None
    assert result.drop_reason == OCR2DropReason.FILTERED_JUDGEMENT


def test_transform_ocr2_row_missing_hydration_fails_without_using_placeholder_question():
    with pytest.raises(ValueError, match="missing hydrated prompt"):
        transform_ocr2_row(OCR2_ROW, {}, _transform_config())

    result = transform_ocr2_row(OCR2_ROW, {}, _transform_config(require_hydration=False))

    assert result.record is None
    assert result.drop_reason == OCR2DropReason.MISSING_HYDRATION


def test_transform_ocr2_row_rejects_placeholder_hydrated_prompt():
    result = transform_ocr2_row(OCR2_ROW, _hydration_cache("-"), _transform_config())

    assert result.record is None
    assert result.drop_reason == OCR2DropReason.PLACEHOLDER_PROMPT


def test_transform_ocr2_row_filters_pass_rate_only_when_configured():
    result = transform_ocr2_row(OCR2_ROW, _hydration_cache(), _transform_config(min_pass_rate=0.95))

    assert result.record is None
    assert result.drop_reason == OCR2DropReason.FILTERED_PASS_RATE
