# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from marin.datakit.download.numinamath_tir import (
    HF_DATASET_ID,
    HF_REVISION,
    TRAIN_PARQUET_GLOB,
    numinamath_tir_normalize_steps,
    row_to_doc,
    transform,
)
from marin.datakit.sources import all_sources


def _valid_row(**overrides) -> dict:
    row = {
        "problem": "Solve $x + 1 = 3$.",
        "solution": "We can verify with code.\n```python\nprint(3 - 1)\n```\n```output\n2\n```\nThus x = 2.",
        "messages": [
            {"role": "user", "content": " Solve $x + 1 = 3$. "},
            {
                "role": "assistant",
                "content": " We can verify with code.\n```python\nprint(3 - 1)\n```\n```output\n2\n```\nThus x = 2. ",
            },
        ],
    }
    row.update(overrides)
    return row


def test_row_to_doc_renders_messages_as_tagged_transcript():
    expected_text = (
        "<user>\n"
        "Solve $x + 1 = 3$.\n"
        "</user>\n\n"
        "<assistant>\n"
        "We can verify with code.\n"
        "```python\n"
        "print(3 - 1)\n"
        "```\n"
        "```output\n"
        "2\n"
        "```\n"
        "Thus x = 2.\n"
        "</assistant>"
    )

    [doc] = row_to_doc(_valid_row())

    assert doc == {
        "id": hashlib.sha256(expected_text.encode("utf-8")).hexdigest(),
        "text": expected_text,
        "source": HF_DATASET_ID,
    }


@pytest.mark.parametrize(
    "row",
    [
        {},
        {"messages": "not-a-list"},
        {"messages": []},
        {"messages": [{"role": "critic", "content": "Nope."}]},
        {"messages": [{"role": "user", "content": ""}]},
        {"messages": [{"role": "user", "content": None}]},
        {"messages": [{"role": "user"}]},
        {"messages": [{"content": "Missing role."}]},
        {"messages": [{"role": "user", "content": "Hi"}, "bad-message"]},
    ],
)
def test_row_to_doc_drops_invalid_rows(row):
    assert row_to_doc(row) == []


def test_numinamath_tir_normalize_steps_use_train_split_and_stable_names():
    processed, normalized = numinamath_tir_normalize_steps()
    download = processed.deps[0]

    assert download.name == "raw/numinamath-tir"
    assert download.hash_attrs["hf_dataset_id"] == HF_DATASET_ID
    assert download.hash_attrs["revision"] == HF_REVISION
    assert download.hash_attrs["hf_urls_glob"] == [TRAIN_PARQUET_GLOB]
    assert processed.name == "processed/numinamath-tir"
    assert processed.deps == [download]
    assert normalized.name == "normalized/numinamath-tir"
    assert normalized.deps == [processed]


def test_numinamath_tir_is_registered_as_datakit_source():
    source = all_sources()["numinamath-tir"]

    assert source.rough_token_count_b == 0.08
    assert source.normalize_steps[0].name == "processed/numinamath-tir"
    assert source.normalized.name == "normalized/numinamath-tir"


def test_transform_reads_parquet_and_writes_valid_docs(tmp_path: Path):
    raw_dir = tmp_path / "raw" / "data"
    raw_dir.mkdir(parents=True)
    table = pa.Table.from_pylist(
        [
            _valid_row(),
            _valid_row(messages=[{"role": "critic", "content": "Nope."}]),
        ]
    )
    pq.write_table(table, raw_dir / "train-00000-of-00001.parquet")

    output_dir = tmp_path / "processed"
    transform(str(tmp_path / "raw"), str(output_dir))

    rows = [row for path in output_dir.rglob("*.parquet") for row in pq.read_table(path).to_pylist()]
    assert len(rows) == 1
    assert rows[0]["source"] == HF_DATASET_ID
    assert rows[0]["text"].startswith("<user>\nSolve $x + 1 = 3$.\n</user>")
    assert "```python\nprint(3 - 1)\n```" in rows[0]["text"]
    assert rows[0]["id"] == hashlib.sha256(rows[0]["text"].encode("utf-8")).hexdigest()
