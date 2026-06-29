# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from marin.datakit.download.olympiad_books_open_source import (
    olympiad_books_open_source_normalize_steps,
    row_to_doc,
    transform,
)


def _valid_row(**overrides) -> dict:
    row = {
        "text": "Let $n$ be an integer. If $n$ is even, then $n^2$ is even.",
        "book": "Open Logic",
        "book_key": "openlogic",
        "subject": "logic",
        "level": "intro",
        "part": "Proofs",
        "chapter": "Direct proofs",
        "section": "Parity",
        "source_file": "proofs/parity.md",
        "chunk_id": 7,
        "tokens_est": 17,
    }
    row.update(overrides)
    return row


def test_row_to_doc_preserves_textbook_metadata():
    [doc] = row_to_doc(_valid_row())

    assert doc["source"] == "yoonholee/olympiad-books-open-source"
    assert doc["source_revision"] == "016c098"
    assert doc["text"] == "Let $n$ be an integer. If $n$ is even, then $n^2$ is even."
    assert doc["book"] == "Open Logic"
    assert doc["book_key"] == "openlogic"
    assert doc["book_license"] == "CC BY 4.0"
    assert doc["license_view"] == "cc-by-sa-reviewed"
    assert doc["subject"] == "logic"
    assert doc["level"] == "intro"
    assert doc["part"] == "Proofs"
    assert doc["chapter"] == "Direct proofs"
    assert doc["section"] == "Parity"
    assert doc["source_file"] == "proofs/parity.md"
    assert doc["chunk_id"] == 7
    assert doc["tokens_est"] == 17
    assert len(doc["id"]) == 64
    assert len(doc["text_hash"]) == 64


@pytest.mark.parametrize(
    "overrides",
    [
        {"text": ""},
        {"text": "   "},
        {"text": None},
    ],
)
def test_row_to_doc_drops_empty_text(overrides):
    assert row_to_doc(_valid_row(**overrides)) == []


def test_row_to_doc_drops_unknown_book_key():
    assert row_to_doc(_valid_row(book_key="unknown-book")) == []


@pytest.mark.parametrize(
    "book_key",
    [
        "mathematical-reasoning",
        "exploring-combinatorial-math",
        "aata",
        "bogart",
        "fcla",
        "ent",
        "ra",
    ],
)
def test_row_to_doc_drops_books_outside_reviewed_license_view(book_key):
    assert row_to_doc(_valid_row(book_key=book_key)) == []


def test_row_to_doc_collapses_repeated_half_text():
    [doc] = row_to_doc(
        _valid_row(
            text=("A group homomorphism preserves multiplication.\n\n" "A group homomorphism preserves multiplication.")
        )
    )

    assert doc["text"] == "A group homomorphism preserves multiplication."


def test_olympiad_books_open_source_normalize_steps_use_pinned_train_file():
    processed, normalized = olympiad_books_open_source_normalize_steps()
    download = processed.deps[0]

    assert download.name == "raw/olympiad-books-open-source"
    assert download.override_output_path == "raw/olympiad-books-open-source-016c098"
    assert download.hash_attrs["hf_dataset_id"] == "yoonholee/olympiad-books-open-source"
    assert download.hash_attrs["revision"] == "016c098"
    assert download.hash_attrs["hf_urls_glob"] == ["data/train-00000-of-00001.parquet"]
    assert processed.name == "processed/olympiad-books-open-source/cc-by-sa-reviewed"
    assert processed.deps == [download]
    assert processed.hash_attrs["license_view"] == "cc-by-sa-reviewed"
    assert processed.hash_attrs["drop_exact_duplicates"] is True
    assert processed.hash_attrs["collapse_repeated_half_rows"] is True
    assert normalized.name == "normalized/olympiad-books-open-source/cc-by-sa-reviewed"
    assert normalized.deps == [processed]


def test_transform_writes_cleaned_reviewed_license_docs(tmp_path: Path):
    raw_dir = tmp_path / "raw" / "data"
    raw_dir.mkdir(parents=True)
    table = pa.Table.from_pylist(
        [
            _valid_row(text="Unique proof text.", source_file="a.md", chunk_id=0),
            _valid_row(text="Unique proof text.", source_file="a.md", chunk_id=1),
            _valid_row(
                text="Repeated lemma.\n\nRepeated lemma.",
                source_file="b.md",
                chunk_id=0,
            ),
            _valid_row(
                text="Excluded license text.",
                book_key="aata",
                source_file="c.md",
                chunk_id=0,
            ),
            _valid_row(text="   ", source_file="d.md", chunk_id=0),
        ]
    )
    pq.write_table(table, raw_dir / "train-00000-of-00001.parquet")

    output_dir = tmp_path / "processed"
    transform(str(tmp_path / "raw"), str(output_dir))

    rows = [row for path in output_dir.rglob("*.parquet") for row in pq.read_table(path).to_pylist()]
    assert [row["text"] for row in rows] == ["Unique proof text.", "Repeated lemma."]
    assert {row["license_view"] for row in rows} == {"cc-by-sa-reviewed"}
    assert {row["book_key"] for row in rows} == {"openlogic"}
    assert {row["source"] for row in rows} == {"yoonholee/olympiad-books-open-source"}
