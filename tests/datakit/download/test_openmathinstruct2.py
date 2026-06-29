# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from marin.datakit.download.openmathinstruct2 import (
    HF_DATASET_ID,
    HF_REVISION,
    download_openmathinstruct2_step,
    row_to_doc,
    transform,
)


def _valid_row(**overrides) -> dict:
    row = {
        "problem": "Solve for $x$: $x + 2 = 5$.",
        "generated_solution": "Subtracting 2 from both sides gives $x = 3$.",
        "expected_answer": "3",
        "problem_source": "augmented_math",
    }
    row.update(overrides)
    return row


def test_row_to_doc_renders_problem_solution_pair():
    expected_text = (
        "<user>\n"
        "Solve for $x$: $x + 2 = 5$.\n"
        "</user>\n\n"
        "<assistant>\n"
        "Subtracting 2 from both sides gives $x = 3$.\n"
        "</assistant>"
    )

    [doc] = row_to_doc(_valid_row())

    assert doc == {
        "id": hashlib.sha256(expected_text.encode("utf-8")).hexdigest(),
        "problem_hash": hashlib.sha256(b"Solve for $x$: $x + 2 = 5$.").hexdigest(),
        "text": expected_text,
        "source": HF_DATASET_ID,
        "problem_source": "augmented_math",
        "expected_answer": "3",
        "synthetic": True,
        "benchmark_adjacent": True,
        "hf_revision": HF_REVISION,
        "split": "train",
    }


def test_problem_hash_is_stable_across_solution_variants():
    first = row_to_doc(_valid_row(generated_solution="Solution A."))[0]
    second = row_to_doc(_valid_row(generated_solution="Solution B."))[0]

    assert first["problem_hash"] == second["problem_hash"]
    assert first["id"] != second["id"]


@pytest.mark.parametrize(
    "problem_source",
    ["augmented_gsm8k", "augmented_math", "gsm8k", "math"],
)
def test_row_to_doc_accepts_expected_problem_sources(problem_source):
    [doc] = row_to_doc(_valid_row(problem_source=problem_source))

    assert doc["problem_source"] == problem_source


@pytest.mark.parametrize(
    "overrides",
    [
        {"problem": ""},
        {"problem": "   "},
        {"problem": None},
        {"generated_solution": ""},
        {"generated_solution": "   "},
        {"generated_solution": None},
        {"problem_source": ""},
        {"problem_source": None},
        {"problem_source": "other"},
    ],
)
def test_row_to_doc_drops_invalid_or_empty_rows(overrides):
    assert row_to_doc(_valid_row(**overrides)) == []


def test_row_to_doc_preserves_empty_expected_answer():
    [doc] = row_to_doc(_valid_row(expected_answer=None))

    assert doc["expected_answer"] == ""


def test_download_step_uses_full_train_split():
    processed = download_openmathinstruct2_step()
    [download] = processed.deps

    assert download.hash_attrs["hf_dataset_id"] == HF_DATASET_ID
    assert download.hash_attrs["revision"] == HF_REVISION
    assert download.hash_attrs["hf_urls_glob"] == ["data/train-*.parquet"]
    assert processed.hash_attrs["split"] == "train"


def test_transform_reads_parquet_and_writes_valid_docs(tmp_path: Path):
    raw_dir = tmp_path / "raw" / "data"
    raw_dir.mkdir(parents=True)
    table = pa.Table.from_pylist(
        [
            _valid_row(),
            _valid_row(problem_source="other"),
        ]
    )
    pq.write_table(table, raw_dir / "train-00000-of-00001.parquet")

    output_dir = tmp_path / "processed"
    transform(str(tmp_path / "raw"), str(output_dir))

    rows = [row for path in output_dir.rglob("*.parquet") for row in pq.read_table(path).to_pylist()]
    assert rows == row_to_doc(_valid_row())
