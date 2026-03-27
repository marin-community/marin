# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for datakit normalize step."""

import gzip
import json
from pathlib import Path

import dupekit
import pyarrow.parquet as pq
import pytest
from fray.v1.job import create_job_ctx, fray_default_job_ctx

from marin.datakit.normalize import generate_id, normalize_to_parquet


@pytest.fixture(autouse=True)
def flow_backend_ctx():
    """Set up sync backend for all normalize tests."""
    with fray_default_job_ctx(create_job_ctx("sync")):
        yield


@pytest.fixture
def write_jsonl_gz():
    """Write records as gzipped JSONL."""

    def _write(path: Path, records: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record))
                f.write("\n")

    return _write


@pytest.fixture
def write_parquet():
    """Write records as Parquet."""

    def _write(path: Path, records: list[dict]) -> None:
        import pyarrow as pa

        path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pylist(records)
        pq.write_table(table, str(path))

    return _write


def _read_all_parquet(output_dir: Path) -> list[dict]:
    """Read all parquet files from output_dir and return records sorted by id."""
    records = []
    for pf in sorted(output_dir.glob("**/*.parquet")):
        table = pq.read_table(str(pf))
        records.extend(table.to_pylist())
    return records


def test_normalize_jsonl_gz(tmp_path: Path, write_jsonl_gz):
    """JSONL.gz input produces normalized parquet with id, text, and source_id."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"

    records = [
        {"doc_id": "abc", "text": "Hello world", "lang": "en"},
        {"doc_id": "def", "text": "Goodbye world", "lang": "fr"},
    ]
    write_jsonl_gz(input_dir / "data.jsonl.gz", records)

    normalize_to_parquet(input_path=str(input_dir), output_path=str(output_dir))

    results = _read_all_parquet(output_dir)
    assert len(results) == 2

    # Check required columns exist
    for r in results:
        assert "id" in r
        assert "text" in r
        assert "source_id" in r
        assert "lang" in r

    # Check id is deterministic xxh3_128
    texts = {r["text"] for r in results}
    assert texts == {"Hello world", "Goodbye world"}

    for r in results:
        assert r["id"] == generate_id(r["text"])

    # Check source_id preserved
    source_ids = {r["source_id"] for r in results}
    assert source_ids == {"abc", "def"}


def test_normalize_parquet_input(tmp_path: Path, write_parquet):
    """Parquet input produces normalized parquet output."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"

    records = [
        {"uid": "r1", "text": "First document"},
        {"uid": "r2", "text": "Second document"},
    ]
    write_parquet(input_dir / "data.parquet", records)

    normalize_to_parquet(input_path=str(input_dir), output_path=str(output_dir))

    results = _read_all_parquet(output_dir)
    assert len(results) == 2
    for r in results:
        assert r["id"] == generate_id(r["text"])
        assert r["source_id"] in ("r1", "r2")


def test_deterministic_ids():
    """Same text always produces the same id via xxh3_128."""
    text = "The quick brown fox"
    id1 = generate_id(text)
    id2 = generate_id(text)
    assert id1 == id2
    assert len(id1) == 32  # 128 bits = 32 hex chars

    # Consistent with dupekit directly
    raw = dupekit.hash_xxh3_128(text.encode("utf-8"))
    assert id1 == format(raw, "032x")


def test_different_texts_different_ids():
    """Different texts produce different ids."""
    assert generate_id("hello") != generate_id("world")


def test_custom_text_field(tmp_path: Path, write_jsonl_gz):
    """Custom text_field extracts text from the specified column."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"

    records = [
        {"body": "Document body here", "title": "A Title"},
    ]
    write_jsonl_gz(input_dir / "data.jsonl.gz", records)

    normalize_to_parquet(
        input_path=str(input_dir),
        output_path=str(output_dir),
        text_field="body",
    )

    results = _read_all_parquet(output_dir)
    assert len(results) == 1
    assert results[0]["text"] == "Document body here"
    assert results[0]["id"] == generate_id("Document body here")
    # Original "body" key should not remain (it's been renamed to "text")
    assert "body" not in results[0]
    # Other columns preserved
    assert results[0]["title"] == "A Title"


def test_explicit_id_field(tmp_path: Path, write_jsonl_gz):
    """Explicit id_field overrides auto-detection."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"

    records = [
        {"my_custom_id": "custom-1", "doc_id": "should-not-use", "text": "Some text"},
    ]
    write_jsonl_gz(input_dir / "data.jsonl.gz", records)

    normalize_to_parquet(
        input_path=str(input_dir),
        output_path=str(output_dir),
        id_field="my_custom_id",
    )

    results = _read_all_parquet(output_dir)
    assert len(results) == 1
    assert results[0]["source_id"] == "custom-1"
    # doc_id should remain as a regular column (not treated as source_id)
    assert results[0]["doc_id"] == "should-not-use"


def test_missing_text_raises(tmp_path: Path, write_jsonl_gz):
    """Records with missing text field raise ValueError."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"

    records = [{"doc_id": "abc", "other": "no text here"}]
    write_jsonl_gz(input_dir / "data.jsonl.gz", records)

    with pytest.raises(Exception, match="text"):
        normalize_to_parquet(input_path=str(input_dir), output_path=str(output_dir))


def test_empty_text_raises(tmp_path: Path, write_jsonl_gz):
    """Records with empty text raise ValueError."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"

    records = [{"text": "   "}]
    write_jsonl_gz(input_dir / "data.jsonl.gz", records)

    with pytest.raises(Exception, match="text"):
        normalize_to_parquet(input_path=str(input_dir), output_path=str(output_dir))


def test_sorted_by_id(tmp_path: Path, write_jsonl_gz):
    """Output partitions are sorted by id within each partition."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"

    # Write enough records that they end up in the same shard
    records = [{"text": f"Document number {i}"} for i in range(20)]
    write_jsonl_gz(input_dir / "data.jsonl.gz", records)

    normalize_to_parquet(input_path=str(input_dir), output_path=str(output_dir))

    for pf in output_dir.glob("*.parquet"):
        table = pq.read_table(str(pf))
        ids = table.column("id").to_pylist()
        assert ids == sorted(ids), f"Partition {pf.name} is not sorted by id"


def test_extra_columns_preserved(tmp_path: Path, write_jsonl_gz):
    """Arbitrary extra columns from raw data are preserved."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"

    records = [
        {"text": "Hello", "lang": "en", "score": 0.9, "metadata": {"source": "web"}},
    ]
    write_jsonl_gz(input_dir / "data.jsonl.gz", records)

    normalize_to_parquet(input_path=str(input_dir), output_path=str(output_dir))

    results = _read_all_parquet(output_dir)
    assert len(results) == 1
    assert results[0]["lang"] == "en"
    assert results[0]["score"] == 0.9
    assert results[0]["metadata"] == {"source": "web"}


def test_directory_structure_preserved(tmp_path: Path, write_jsonl_gz):
    """Subdirectory structure from input is preserved in output."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"

    write_jsonl_gz(input_dir / "subset_a" / "data.jsonl.gz", [{"text": "A doc"}])
    write_jsonl_gz(input_dir / "subset_b" / "data.jsonl.gz", [{"text": "B doc"}])

    normalize_to_parquet(input_path=str(input_dir), output_path=str(output_dir))

    # Both subdirectories should have parquet output
    a_files = list((output_dir / "subset_a").glob("*.parquet"))
    b_files = list((output_dir / "subset_b").glob("*.parquet"))
    assert len(a_files) >= 1
    assert len(b_files) >= 1

    a_records = _read_all_parquet(output_dir / "subset_a")
    b_records = _read_all_parquet(output_dir / "subset_b")
    assert len(a_records) == 1
    assert a_records[0]["text"] == "A doc"
    assert len(b_records) == 1
    assert b_records[0]["text"] == "B doc"


def test_exact_dedup(tmp_path: Path, write_jsonl_gz):
    """Exact duplicate documents (same text) are deduplicated."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"

    records = [
        {"text": "Duplicate text", "source": "file1"},
        {"text": "Duplicate text", "source": "file2"},
        {"text": "Unique text", "source": "file3"},
    ]
    write_jsonl_gz(input_dir / "data.jsonl.gz", records)

    normalize_to_parquet(input_path=str(input_dir), output_path=str(output_dir))

    results = _read_all_parquet(output_dir)
    assert len(results) == 2
    texts = {r["text"] for r in results}
    assert texts == {"Duplicate text", "Unique text"}


def test_deterministic_output(tmp_path: Path, write_jsonl_gz):
    """Running twice on the same input produces identical output."""
    input_dir = tmp_path / "input"
    output_a = tmp_path / "output_a"
    output_b = tmp_path / "output_b"

    records = [{"text": f"Document {i}", "idx": i} for i in range(10)]
    write_jsonl_gz(input_dir / "data.jsonl.gz", records)

    normalize_to_parquet(input_path=str(input_dir), output_path=str(output_a))
    normalize_to_parquet(input_path=str(input_dir), output_path=str(output_b))

    files_a = sorted(output_a.glob("*.parquet"))
    files_b = sorted(output_b.glob("*.parquet"))
    assert len(files_a) == len(files_b)

    for fa, fb in zip(files_a, files_b, strict=True):
        assert fa.name == fb.name
        table_a = pq.read_table(str(fa))
        table_b = pq.read_table(str(fb))
        assert table_a.equals(table_b), f"Mismatch in {fa.name}"


def test_skip_existing(tmp_path: Path, write_jsonl_gz):
    """Second run with skip_existing does not rewrite files."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"

    records = [{"text": "Hello world"}]
    write_jsonl_gz(input_dir / "data.jsonl.gz", records)

    normalize_to_parquet(input_path=str(input_dir), output_path=str(output_dir))

    parquet_files = list(output_dir.glob("*.parquet"))
    assert len(parquet_files) == 1
    mtime_first = parquet_files[0].stat().st_mtime

    # Run again — files should be skipped
    normalize_to_parquet(input_path=str(input_dir), output_path=str(output_dir))

    parquet_files = list(output_dir.glob("*.parquet"))
    assert len(parquet_files) == 1
    mtime_second = parquet_files[0].stat().st_mtime
    assert mtime_first == mtime_second


def test_partition_naming(tmp_path: Path, write_jsonl_gz):
    """Output files follow part-XXXXX-of-YYYYY.parquet naming."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"

    records = [{"text": f"Doc {i}"} for i in range(5)]
    write_jsonl_gz(input_dir / "data.jsonl.gz", records)

    normalize_to_parquet(input_path=str(input_dir), output_path=str(output_dir))

    parquet_files = list(output_dir.glob("*.parquet"))
    assert len(parquet_files) >= 1
    for pf in parquet_files:
        assert pf.name.startswith("part-"), f"Unexpected filename: {pf.name}"
        assert "-of-" in pf.name, f"Missing -of- in filename: {pf.name}"


def test_no_files_raises(tmp_path: Path):
    """Empty input directory raises FileNotFoundError."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "output"

    with pytest.raises(FileNotFoundError):
        normalize_to_parquet(input_path=str(input_dir), output_path=str(output_dir))


def test_text_field_default_is_text(tmp_path: Path, write_jsonl_gz):
    """Default text_field='text' works without explicit specification."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"

    records = [{"text": "Default field"}]
    write_jsonl_gz(input_dir / "data.jsonl.gz", records)

    normalize_to_parquet(input_path=str(input_dir), output_path=str(output_dir))

    results = _read_all_parquet(output_dir)
    assert len(results) == 1
    assert results[0]["text"] == "Default field"
    # When text_field is "text", the original column IS the text column — no rename
    assert "text" in results[0]
