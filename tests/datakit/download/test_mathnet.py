# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from marin.datakit.download.mathnet import (
    HF_REVISION,
    HF_URLS_GLOB,
    MathNetLanguagePolicy,
    MathNetSolutionPolicy,
    MathNetTextSftConfig,
    _load_excluded_ids,
    download_mathnet_raw_step,
    mathnet_text_sft_primary_step,
    row_to_text_sft_records,
    transform_text_sft,
)


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
    assert record["id"] == f"mathnet:{HF_REVISION}:abc123:solution-0"
    assert record["source"] == "ShadenA/MathNet"
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
    assert record["metadata"]["hf_revision"] == HF_REVISION
    assert record["metadata"]["license"] == "cc-by-4.0"


def test_first_solution_policy_uses_first_non_empty_solution():
    row = _base_row(solutions_markdown=["", "First usable solution.", "Second usable solution."])

    records = row_to_text_sft_records(row, _base_config())

    assert len(records) == 1
    assert records[0]["id"] == f"mathnet:{HF_REVISION}:abc123:solution-1"
    assert records[0]["messages"][1]["content"] == "First usable solution."
    assert records[0]["metadata"]["solution_count"] == 2


def test_all_solution_policy_expands_each_non_empty_solution():
    row = _base_row(solutions_markdown=["First solution.", "", "Second solution."])

    records = row_to_text_sft_records(row, _base_config(solution_policy=MathNetSolutionPolicy.ALL))

    assert [record["id"] for record in records] == [
        f"mathnet:{HF_REVISION}:abc123:solution-0",
        f"mathnet:{HF_REVISION}:abc123:solution-2",
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
    assert records[0]["id"] == f"mathnet:{HF_REVISION}:abc123:solution-0"
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


def test_excluded_ids_are_skipped():
    row = _base_row(id="heldout")

    assert row_to_text_sft_records(row, _base_config(), {"heldout"}) == []


def test_load_excluded_ids_reads_one_id_per_line(tmp_path: Path):
    excluded_path = tmp_path / "heldout.txt"
    excluded_path.write_text("abc123\n\nxyz789\n", encoding="utf-8")

    assert _load_excluded_ids(str(excluded_path)) == {"abc123", "xyz789"}


def test_download_step_restricts_to_default_all_config_files():
    step = download_mathnet_raw_step()

    assert step.hash_attrs["hf_dataset_id"] == "ShadenA/MathNet"
    assert step.hash_attrs["revision"] == HF_REVISION
    assert step.hash_attrs["hf_urls_glob"] == HF_URLS_GLOB


def test_processed_step_records_policy_in_hash_attrs():
    step = mathnet_text_sft_primary_step(
        language_policy=MathNetLanguagePolicy.ALL_LANGUAGES,
        solution_policy=MathNetSolutionPolicy.ALL,
        excluded_ids_path="gs://example/heldout.txt",
    )

    assert step.name == "processed/mathnet-v0/text-sft-primary"
    assert step.hash_attrs["hf_dataset_id"] == "ShadenA/MathNet"
    assert step.hash_attrs["hf_revision"] == HF_REVISION
    assert step.hash_attrs["hf_urls_glob"] == HF_URLS_GLOB
    assert step.hash_attrs["language_policy"] == "all_languages"
    assert step.hash_attrs["solution_policy"] == "all"
    assert step.hash_attrs["excluded_ids_path"] == "gs://example/heldout.txt"


def test_transform_text_sft_writes_chat_jsonl(tmp_path: Path, read_all_jsonl_gz):
    raw_dir = tmp_path / "raw"
    raw_shard = raw_dir / "data" / "all" / "train-00000-of-00001.parquet"
    raw_shard.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([_base_row()]), raw_shard)

    output_dir = tmp_path / "processed"
    transform_text_sft(
        MathNetTextSftConfig(
            raw_input_path=str(raw_dir),
            output_path=str(output_dir),
        )
    )

    records = read_all_jsonl_gz(output_dir)
    assert len(records) == 1
    assert records[0]["id"] == f"mathnet:{HF_REVISION}:abc123:solution-0"
    assert records[0]["messages"][0]["role"] == "user"
    assert records[0]["messages"][1]["role"] == "assistant"
