# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from marin.datakit.download.stackmathqa import (
    FULL_QALIST_CONFIG,
    FULL_QALIST_GLOB,
    HF_DATASET_ID,
    HF_REVISION,
    NORMALIZED_STEP_NAME,
    PROCESSED_STEP_NAME,
    RAW_STEP_NAME,
    row_to_doc,
    stackmathqa_full_qalist_normalize_steps,
    transform,
)


def _valid_row(**overrides) -> dict:
    row = {
        "Q": "How can you prove that $\\sqrt{2}$ is irrational?",
        "A_list": [
            "Assume $\\sqrt{2} = p / q$ in lowest terms and derive a parity contradiction.",
            "A geometric proof also works by infinite descent.",
        ],
        "meta": {
            "language": "en",
            "url": "https://math.stackexchange.com/questions/2",
            "timestamp": "2023-03-29T00:00:00",
            "source": "stackexchange",
            "question_score": "42",
            "answer_count": 2,
        },
    }
    row.update(overrides)
    return row


def test_row_to_doc_renders_question_with_all_valid_answers():
    [doc] = row_to_doc(_valid_row())

    assert doc["source"] == HF_DATASET_ID
    assert doc["stackmathqa_config"] == FULL_QALIST_CONFIG
    assert doc["url"] == "https://math.stackexchange.com/questions/2"
    assert doc["language"] == "en"
    assert doc["timestamp"] == "2023-03-29T00:00:00"
    assert doc["stackmathqa_source"] == "stackexchange"
    assert doc["question_score"] == 42
    assert doc["question_score_raw"] == "42"
    assert doc["answer_count"] == 2
    assert doc["num_answers_rendered"] == 2
    assert len(doc["id"]) == 64
    assert len(doc["question_hash"]) == 64
    assert doc["text"] == (
        "<question>\n"
        "How can you prove that $\\sqrt{2}$ is irrational?\n"
        "</question>\n\n"
        '<answer index="0">\n'
        "Assume $\\sqrt{2} = p / q$ in lowest terms and derive a parity contradiction.\n"
        "</answer>\n\n"
        '<answer index="1">\n'
        "A geometric proof also works by infinite descent.\n"
        "</answer>"
    )


@pytest.mark.parametrize("question", ["", "   ", None, ["not a question"]])
def test_row_to_doc_filters_bad_questions(question):
    assert row_to_doc(_valid_row(Q=question)) == []


@pytest.mark.parametrize(
    "overrides",
    [
        {"A_list": None},
        {"A_list": "answer"},
        {"A_list": []},
        {"A_list": ["", "   ", None, 3]},
    ],
)
def test_row_to_doc_filters_missing_or_empty_answer_lists(overrides):
    assert row_to_doc(_valid_row(**overrides)) == []


def test_row_to_doc_keeps_only_non_empty_string_answers():
    [doc] = row_to_doc(_valid_row(A_list=[" First answer. ", "", None, 7, "Second answer."]))

    assert doc["num_answers_rendered"] == 2
    assert '<answer index="0">\nFirst answer.\n</answer>' in doc["text"]
    assert '<answer index="1">\nSecond answer.\n</answer>' in doc["text"]
    assert "None" not in doc["text"]


def test_row_to_doc_preserves_unparseable_question_score_as_raw_text():
    [doc] = row_to_doc(_valid_row(meta={**_valid_row()["meta"], "question_score": "unknown"}))

    assert doc["question_score"] is None
    assert doc["question_score_raw"] == "unknown"


def test_stackmathqa_full_qalist_normalize_steps_use_pinned_revision_and_stable_names():
    processed, normalized = stackmathqa_full_qalist_normalize_steps()
    download = processed.deps[0]

    assert download.name == RAW_STEP_NAME
    assert download.hash_attrs["hf_dataset_id"] == HF_DATASET_ID
    assert download.hash_attrs["revision"] == HF_REVISION
    assert download.hash_attrs["hf_urls_glob"] == [FULL_QALIST_GLOB]
    assert processed.name == PROCESSED_STEP_NAME
    assert processed.deps == [download]
    assert normalized.name == NORMALIZED_STEP_NAME
    assert normalized.deps == [processed]


def test_transform_reads_jsonl_and_writes_valid_docs(tmp_path: Path):
    raw_dir = tmp_path / "raw" / "preprocessed" / "stackexchange-math"
    raw_dir.mkdir(parents=True)
    rows = [
        _valid_row(),
        _valid_row(Q=""),
    ]
    (raw_dir / "math.stackexchange.com.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    output_dir = tmp_path / "processed"
    transform(str(tmp_path / "raw"), str(output_dir))

    written_rows = [row for path in output_dir.rglob("*.parquet") for row in pq.read_table(path).to_pylist()]
    assert len(written_rows) == 1
    assert written_rows[0]["source"] == HF_DATASET_ID
    assert written_rows[0]["stackmathqa_config"] == FULL_QALIST_CONFIG
    assert written_rows[0]["url"] == "https://math.stackexchange.com/questions/2"
    assert written_rows[0]["answer_count"] == 2
    assert written_rows[0]["num_answers_rendered"] == 2
    assert "<question>\nHow can you prove that $\\sqrt{2}$ is irrational?\n</question>" in written_rows[0]["text"]
    assert '<answer index="0">\nAssume $\\sqrt{2} = p / q$' in written_rows[0]["text"]
