# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from marin.datakit.download.numinamath_tir import (
    HF_DATASET_ID,
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
        {"messages": "not-a-list"},
        {"messages": []},
        {"messages": [{"role": "critic", "content": "Nope."}]},
        {"messages": [{"role": "user", "content": ""}]},
        {"messages": [{"role": "user", "content": None}]},
        {"messages": [{"role": "user", "content": "Hi"}, "bad-message"]},
    ],
    ids=[
        "messages-not-list",
        "empty-transcript",
        "unsupported-role",
        "empty-content",
        "non-string-content",
        "malformed-message",
    ],
)
def test_row_to_doc_drops_invalid_rows(row):
    assert row_to_doc(row) == []


def test_numinamath_tir_is_materializable_from_datakit_registry():
    source = all_sources()["numinamath-tir"]
    processed, normalized = source.normalize_steps

    assert source.normalized is normalized
    assert processed.deps
    assert normalized.deps == [processed]


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
    assert rows == row_to_doc(_valid_row())
