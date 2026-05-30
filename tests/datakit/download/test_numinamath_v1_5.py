# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from marin.datakit.download.numinamath_v1_5 import (
    HF_DATASET_ID,
    HF_REVISION,
    TRAIN_PARQUET_GLOB,
    numinamath_v1_5_normalize_steps,
    row_to_doc,
    transform,
)
from marin.datakit.sources import all_sources


def _valid_row(**overrides) -> dict:
    row = {
        "problem": "Find $x$ if $x + 2 = 5$.",
        "solution": "Subtracting 2 from both sides gives $x = 3$.",
        "answer": "3",
        "problem_type": "Algebra",
        "question_type": "math-word-problem",
        "problem_is_valid": "Yes",
        "solution_is_valid": "Yes",
        "source": "olympiads",
        "synthetic": False,
    }
    row.update(overrides)
    return row


def test_row_to_doc_renders_valid_problem_solution_pair():
    [doc] = row_to_doc(_valid_row(synthetic=True))

    assert doc["source"] == HF_DATASET_ID
    assert doc["numina_source"] == "olympiads"
    assert doc["answer"] == "3"
    assert doc["problem_type"] == "Algebra"
    assert doc["question_type"] == "math-word-problem"
    assert doc["synthetic"] is True
    assert len(doc["id"]) == 64
    assert len(doc["problem_hash"]) == 64
    assert doc["id"] != doc["problem_hash"]
    assert doc["text"] == (
        "<user>\n"
        "Find $x$ if $x + 2 = 5$.\n"
        "</user>\n\n"
        "<assistant>\n"
        "Subtracting 2 from both sides gives $x = 3$.\n"
        "</assistant>"
    )


def test_problem_hash_is_stable_across_solution_variants():
    first = row_to_doc(_valid_row(solution="Solution A."))[0]
    second = row_to_doc(_valid_row(solution="Solution B."))[0]

    assert first["problem_hash"] == second["problem_hash"]
    assert first["id"] != second["id"]


@pytest.mark.parametrize(
    "overrides",
    [
        {"problem": ""},
        {"problem": "   "},
        {"problem": None},
        {"solution": ""},
        {"solution": "   "},
        {"solution": None},
        {"problem_is_valid": "Incomplete"},
        {"problem_is_valid": "Not a problem"},
        {"solution_is_valid": "Incomplete"},
        {"solution_is_valid": "Problem not solved"},
        {"solution_is_valid": "Not matched with problem"},
    ],
)
def test_row_to_doc_drops_invalid_or_empty_rows(overrides):
    assert row_to_doc(_valid_row(**overrides)) == []


def test_row_to_doc_preserves_optional_metadata_as_empty_strings():
    [doc] = row_to_doc(
        _valid_row(
            answer=None,
            problem_type=None,
            question_type=None,
            source=None,
            synthetic="false",
        )
    )

    assert doc["answer"] == ""
    assert doc["problem_type"] == ""
    assert doc["question_type"] == ""
    assert doc["numina_source"] == ""
    assert doc["synthetic"] is False


def test_numinamath_v1_5_normalize_steps_use_train_split_and_stable_names():
    processed, normalized = numinamath_v1_5_normalize_steps()
    download = processed.deps[0]

    assert download.name == "raw/numinamath-1.5"
    assert download.hash_attrs["hf_dataset_id"] == HF_DATASET_ID
    assert download.hash_attrs["revision"] == HF_REVISION
    assert download.hash_attrs["hf_urls_glob"] == [TRAIN_PARQUET_GLOB]
    assert processed.name == "processed/numinamath-1.5"
    assert processed.deps == [download]
    assert normalized.name == "normalized/numinamath-1.5"
    assert normalized.deps == [processed]


def test_numinamath_v1_5_is_registered_as_datakit_source():
    source = all_sources()["numinamath-1.5"]

    assert source.rough_token_count_b == 0.40
    assert source.normalize_steps[0].name == "processed/numinamath-1.5"
    assert source.normalized.name == "normalized/numinamath-1.5"


def test_transform_reads_parquet_and_writes_valid_docs(tmp_path: Path):
    raw_dir = tmp_path / "raw" / "data"
    raw_dir.mkdir(parents=True)
    table = pa.Table.from_pylist(
        [
            _valid_row(synthetic=True),
            _valid_row(problem="bad", problem_is_valid="Incomplete"),
        ]
    )
    pq.write_table(table, raw_dir / "train-00000-of-00001.parquet")

    output_dir = tmp_path / "processed"
    transform(str(tmp_path / "raw"), str(output_dir))

    rows = [row for path in output_dir.rglob("*.parquet") for row in pq.read_table(path).to_pylist()]
    assert len(rows) == 1
    assert rows[0]["source"] == HF_DATASET_ID
    assert rows[0]["numina_source"] == "olympiads"
    assert rows[0]["synthetic"] is True
    assert "<user>\nFind $x$ if $x + 2 = 5$.\n</user>" in rows[0]["text"]
    assert "<assistant>\nSubtracting 2 from both sides gives $x = 3$.\n</assistant>" in rows[0]["text"]
