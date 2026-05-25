# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from marin.datakit.download.mathnet import (
    MathNetLanguagePolicy,
    MathNetSolutionPolicy,
    MathNetTextSftConfig,
    row_to_text_sft_records,
    transform_text_sft,
)

EXPECTED_SOURCE = "ShadenA/MathNet"
EXPECTED_HF_REVISION = "ae12e35eef0fc52bbbef270d6ef0f5b002252eb9"


def _base_config(
    *,
    language_policy: MathNetLanguagePolicy = MathNetLanguagePolicy.ENGLISH_OR_UNKNOWN,
    solution_policy: MathNetSolutionPolicy = MathNetSolutionPolicy.FIRST,
    excluded_ids_path: str = "",
) -> MathNetTextSftConfig:
    return MathNetTextSftConfig(
        raw_input_path="/raw",
        output_path="/processed",
        language_policy=language_policy,
        solution_policy=solution_policy,
        excluded_ids_path=excluded_ids_path,
    )


def _base_row(**overrides):
    row = {
        "id": "abc123",
        "problem_markdown": "Prove that $a+b=b+a$.",
        "solutions_markdown": ["Addition is commutative."],
        "images": [],
        "country": "United States",
        "competition": "Example Olympiad",
        "topics_flat": ["Algebra"],
        "language": "English",
        "problem_type": "proof only",
        "final_answer": None,
    }
    row.update(overrides)
    return row


def test_text_only_row_becomes_two_message_record():
    records = row_to_text_sft_records(_base_row(), _base_config())

    assert len(records) == 1
    record = records[0]
    assert record["id"] == f"mathnet:{EXPECTED_HF_REVISION}:abc123:solution-0"
    assert record["source"] == EXPECTED_SOURCE
    assert record["messages"] == [
        {
            "role": "user",
            "content": "Prove that $a+b=b+a$.",
            "name": None,
            "tool_calls": None,
            "tool_call_id": None,
        },
        {
            "role": "assistant",
            "content": "Addition is commutative.",
            "name": None,
            "tool_calls": None,
            "tool_call_id": None,
        },
    ]
    assert record["metadata"]["mathnet_id"] == "abc123"
    assert record["metadata"]["country"] == "United States"
    assert record["metadata"]["competition"] == "Example Olympiad"
    assert record["metadata"]["topics_flat"] == ["Algebra"]
    assert record["metadata"]["language"] == "English"
    assert record["metadata"]["problem_type"] == "proof only"
    assert record["metadata"]["final_answer"] is None
    assert record["metadata"]["solution_index"] == 0
    assert record["metadata"]["solution_count"] == 1
    assert record["metadata"]["language_policy"] == "english_or_unknown"
    assert record["metadata"]["solution_policy"] == "first"
    assert record["metadata"]["hf_revision"] == EXPECTED_HF_REVISION
    assert record["metadata"]["license"] == "cc-by-4.0"


def test_first_solution_policy_uses_first_non_empty_solution():
    row = _base_row(solutions_markdown=["", "First usable solution.", "Second usable solution."])

    records = row_to_text_sft_records(row, _base_config())

    assert len(records) == 1
    assert records[0]["id"] == f"mathnet:{EXPECTED_HF_REVISION}:abc123:solution-1"
    assert records[0]["messages"][1]["content"] == "First usable solution."
    assert records[0]["metadata"]["solution_count"] == 2


def test_all_solution_policy_expands_each_non_empty_solution():
    row = _base_row(solutions_markdown=["First solution.", "", "Second solution."])

    records = row_to_text_sft_records(row, _base_config(solution_policy=MathNetSolutionPolicy.ALL))

    assert [record["id"] for record in records] == [
        f"mathnet:{EXPECTED_HF_REVISION}:abc123:solution-0",
        f"mathnet:{EXPECTED_HF_REVISION}:abc123:solution-2",
    ]
    assert [record["messages"][1]["content"] for record in records] == ["First solution.", "Second solution."]


@pytest.mark.parametrize(
    "row",
    [
        _base_row(images=[{"path": "diagram.png"}]),
        _base_row(problem_markdown="Use this figure.\n![](attached_image_1.png)"),
        _base_row(solutions_markdown=["Refer to the figure.\n![](attached_image_1.png)"]),
    ],
)
def test_image_dependent_rows_are_skipped(row):
    assert row_to_text_sft_records(row, _base_config()) == []


def test_first_solution_policy_ignores_unselected_image_dependent_solution():
    row = _base_row(solutions_markdown=["Text-only solution.", "Figure solution.\n![](attached_image_1.png)"])

    records = row_to_text_sft_records(row, _base_config())

    assert len(records) == 1
    assert records[0]["messages"][1]["content"] == "Text-only solution."


def test_all_solution_policy_skips_only_image_dependent_solution_variants():
    row = _base_row(solutions_markdown=["Text-only solution.", "Figure solution.\n![](attached_image_1.png)"])

    records = row_to_text_sft_records(row, _base_config(solution_policy=MathNetSolutionPolicy.ALL))

    assert len(records) == 1
    assert records[0]["id"] == f"mathnet:{EXPECTED_HF_REVISION}:abc123:solution-0"
    assert records[0]["messages"][1]["content"] == "Text-only solution."


@pytest.mark.parametrize(
    "row",
    [
        _base_row(id=""),
        _base_row(problem_markdown=""),
        _base_row(solutions_markdown=[]),
        _base_row(solutions_markdown=["", "   "]),
    ],
)
def test_empty_id_problem_or_solution_rows_are_skipped(row):
    assert row_to_text_sft_records(row, _base_config()) == []


@pytest.mark.parametrize("language", [None, "", "en", "English"])
def test_english_or_unknown_language_policy_keeps_primary_languages(language):
    records = row_to_text_sft_records(_base_row(language=language), _base_config())

    assert len(records) == 1
    assert records[0]["metadata"]["language"] == language


def test_english_or_unknown_language_policy_skips_non_english_rows():
    row = _base_row(language="Spanish")

    assert row_to_text_sft_records(row, _base_config()) == []


def test_all_languages_policy_keeps_non_english_rows():
    row = _base_row(language="Spanish")

    records = row_to_text_sft_records(row, _base_config(language_policy=MathNetLanguagePolicy.ALL_LANGUAGES))

    assert len(records) == 1
    assert records[0]["metadata"]["language_policy"] == "all_languages"


def test_transform_text_sft_filters_and_writes_chat_jsonl(tmp_path: Path, read_all_jsonl_gz):
    raw_dir = tmp_path / "raw"
    raw_shard = raw_dir / "data" / "all" / "train-00000-of-00001.parquet"
    raw_shard.parent.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(
            [
                _base_row(id="kept", problem_markdown="Solve $x+1=2$.", solutions_markdown=["Subtract 1 to get $x=1$."]),
                _base_row(id="heldout"),
                _base_row(id="needs_image", problem_markdown="Use attached_image_1.png to solve this."),
                _base_row(id="spanish", language="Spanish"),
                _base_row(id="empty_solution", solutions_markdown=["", "   "]),
            ]
        ),
        raw_shard,
    )
    excluded_ids = tmp_path / "heldout.txt"
    excluded_ids.write_text("heldout\n\nunused-id\n", encoding="utf-8")

    output_dir = tmp_path / "processed"
    transform_text_sft(
        MathNetTextSftConfig(
            raw_input_path=str(raw_dir),
            output_path=str(output_dir),
            excluded_ids_path=str(excluded_ids),
        )
    )

    records = read_all_jsonl_gz(output_dir)
    assert len(records) == 1
    assert records[0]["id"] == f"mathnet:{EXPECTED_HF_REVISION}:kept:solution-0"
    assert records[0]["messages"] == [
        {
            "role": "user",
            "content": "Solve $x+1=2$.",
            "name": None,
            "tool_calls": None,
            "tool_call_id": None,
        },
        {
            "role": "assistant",
            "content": "Subtract 1 to get $x=1$.",
            "name": None,
            "tool_calls": None,
            "tool_call_id": None,
        },
    ]
    assert records[0]["metadata"]["mathnet_id"] == "kept"
    assert records[0]["metadata"]["excluded_ids_path"] == str(excluded_ids)
