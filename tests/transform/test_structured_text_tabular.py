# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Byte-preservation tests for the CSV/TSV staging step.

The invariant under test is that raw input bytes survive into the emitted
``text`` field unchanged, aside from optional header replication and
deterministic chunking at line boundaries.
"""

import gzip
import json
from pathlib import Path

import pytest

from marin.transform.structured_text.tabular import (
    TabularStagingConfig,
    chunk_lines_by_bytes,
    serialize_csv_document,
    stage_tabular_source,
)

# ---------------------------------------------------------------------------
# Pure-function tests
# ---------------------------------------------------------------------------


def test_serialize_csv_document_preserves_bytes_verbatim():
    header = "col_a,col_b,col_c\n"
    body = [
        "1,0.00001,hello\n",
        "2,,world\n",  # empty cell — missing-value marker must survive
        '3,"spaces  in quotes","with, comma"\n',
    ]
    text = serialize_csv_document(header, body)
    assert text == header + "".join(body)


def test_serialize_csv_document_no_header():
    body = ["a\tb\tc\n", "1\t2\t3\n"]
    assert serialize_csv_document(None, body) == "a\tb\tc\n1\t2\t3\n"


def test_serialize_csv_document_preserves_precision():
    # This is the key correctness property — the ingestion layer must not
    # round-trip numerics through float. We feed in source-formatted floats
    # with varying precision and check the output byte-matches.
    header = "x,y,z\n"
    body = ["0.1,0.2,0.3\n", "3.14159265358979323846,1.4142135623730951,1e-30\n"]
    text = serialize_csv_document(header, body)
    assert "3.14159265358979323846" in text
    assert "1e-30" in text
    assert "0.1,0.2,0.3" in text


def test_chunk_lines_by_bytes_respects_cap():
    lines = [f"row_{i}\n" for i in range(100)]
    # Each line is 6 or 7 bytes; 60-byte cap should produce several chunks.
    chunks = list(chunk_lines_by_bytes(lines, max_bytes_per_chunk=60))
    total_lines = sum(len(c) for c in chunks)
    assert total_lines == 100
    for chunk in chunks:
        chunk_bytes = sum(len(line.encode("utf-8")) for line in chunk)
        # A chunk can exceed the cap only if it's a single line that on its
        # own was over-budget. Our synthetic lines are well under 60 bytes.
        assert chunk_bytes <= 60 or len(chunk) == 1


def test_chunk_lines_by_bytes_single_line_exceeds_cap_becomes_own_chunk():
    # Cap smaller than an individual line — the line still must be emitted
    # intact (no mid-row splitting).
    lines = ["a,b,c\n", "x" * 100 + "\n", "d,e,f\n"]
    chunks = list(chunk_lines_by_bytes(lines, max_bytes_per_chunk=10))
    assert len(chunks) == 3
    assert chunks[0] == ["a,b,c\n"]
    assert chunks[1] == ["x" * 100 + "\n"]
    assert chunks[2] == ["d,e,f\n"]


def test_chunk_lines_by_bytes_reserves_header_budget():
    header = "h1,h2,h3\n"  # 9 bytes
    lines = ["1,2,3\n"] * 10  # 6 bytes each
    chunks = list(chunk_lines_by_bytes(lines, max_bytes_per_chunk=20, header_line=header))
    # Budget after header = 11 bytes → each chunk holds ~1 line.
    for chunk in chunks:
        chunk_with_header_bytes = len(header.encode("utf-8")) + sum(len(line.encode("utf-8")) for line in chunk)
        # Either we fit under the cap, or the chunk is a single line (unsplittable).
        assert chunk_with_header_bytes <= 20 or len(chunk) == 1


def test_chunk_lines_by_bytes_rejects_nonpositive_cap():
    with pytest.raises(ValueError):
        list(chunk_lines_by_bytes(["a\n"], max_bytes_per_chunk=0))


def test_chunk_lines_by_bytes_rejects_header_larger_than_cap():
    with pytest.raises(ValueError):
        list(chunk_lines_by_bytes(["a\n"], max_bytes_per_chunk=5, header_line="a" * 20 + "\n"))


# ---------------------------------------------------------------------------
# End-to-end staging tests against the local filesystem
# ---------------------------------------------------------------------------


def _write_csv(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _read_staged_records(output_path: Path, filename: str = "staged.jsonl.gz") -> list[dict]:
    with gzip.open(output_path / filename, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_stage_tabular_preserves_delimiters_and_missing_values(tmp_path):
    # Input exercises: empty cells, exotic unicode, varying numeric precision,
    # quoted commas, tabs, and NA markers.
    input_dir = tmp_path / "raw"
    output_dir = tmp_path / "staged"

    # source bytes survive the ingestion pipeline unchanged.
    content = (
        "id,name,value,unit\n"
        "1,alpha,0.00001,m\n"
        "2,beta,,s\n"  # empty cell = missing-value marker
        "3,γ,NA,kg\n"  # noqa: RUF001 -- testing byte preservation of non-ASCII
        '4,"δ, d","1.4142135623730951",m/s\n'
    )
    _write_csv(input_dir / "a.csv", content)

    cfg = TabularStagingConfig(
        input_path=str(input_dir),
        output_path=str(output_dir),
        source_label="test:csv",
        max_bytes_per_document=1024,
    )
    result = stage_tabular_source(cfg)
    assert result["record_count"] == 1

    records = _read_staged_records(output_dir)
    assert len(records) == 1
    record = records[0]
    # Whole-file byte equality (with header preserved at the top of the chunk).
    assert record["text"] == content
    assert record["source"] == "test:csv"
    assert record["provenance"]["header_preserved"] is True
    # Critical tokens must appear verbatim
    assert "0.00001" in record["text"]
    assert "1.4142135623730951" in record["text"]
    assert ",,s\n" in record["text"]  # empty-cell delimiter survived
    assert "γ" in record["text"]  # noqa: RUF001 -- see fixture above
    assert '"δ, d"' in record["text"]


def test_stage_tabular_preserves_repeated_header_like_rows(tmp_path):
    input_dir = tmp_path / "raw"
    output_dir = tmp_path / "staged"

    content = "id,value\n1,alpha\nid,value\n2,beta\n"
    _write_csv(input_dir / "repeated.csv", content)

    cfg = TabularStagingConfig(
        input_path=str(input_dir),
        output_path=str(output_dir),
        source_label="test:repeated",
        max_bytes_per_document=1024,
    )
    result = stage_tabular_source(cfg)

    assert result["record_count"] == 1
    records = _read_staged_records(output_dir)
    assert records[0]["text"] == content


def test_stage_tabular_rejects_non_utf8_input(tmp_path):
    input_dir = tmp_path / "raw"
    output_dir = tmp_path / "staged"
    input_dir.mkdir()
    (input_dir / "bad.csv").write_bytes(b"id,value\n1,\xff\n")

    cfg = TabularStagingConfig(
        input_path=str(input_dir),
        output_path=str(output_dir),
        source_label="test:bad",
    )
    with pytest.raises(UnicodeDecodeError):
        stage_tabular_source(cfg)


def test_stage_tabular_chunks_large_file_and_replicates_header(tmp_path):
    input_dir = tmp_path / "raw"
    output_dir = tmp_path / "staged"

    header = "col_a,col_b\n"
    body = "".join(f"{i},{i*0.5}\n" for i in range(200))
    _write_csv(input_dir / "big.csv", header + body)

    cfg = TabularStagingConfig(
        input_path=str(input_dir),
        output_path=str(output_dir),
        source_label="test:big",
        max_bytes_per_document=200,  # small to force chunking
    )
    result = stage_tabular_source(cfg)
    assert result["record_count"] > 1

    records = _read_staged_records(output_dir)
    # Every chunk must start with the original header
    for record in records:
        assert record["text"].startswith(header)
        assert record["provenance"]["header_preserved"] is True

    # Concatenating chunks (minus replicated header after the first) must
    # byte-reproduce the original file.
    reconstructed = records[0]["text"] + "".join(r["text"][len(header) :] for r in records[1:])
    assert reconstructed == header + body


def test_stage_tabular_respects_max_bytes_per_source(tmp_path):
    input_dir = tmp_path / "raw"
    output_dir = tmp_path / "staged"

    header = "a,b\n"
    # Each file is ~500 bytes; cap at 800 bytes should cut off around
    # file 1 or mid-file 2.
    for filename_index in range(5):
        body = "".join(f"{i},{i}\n" for i in range(100))
        _write_csv(input_dir / f"{filename_index:03d}.csv", header + body)

    cfg = TabularStagingConfig(
        input_path=str(input_dir),
        output_path=str(output_dir),
        source_label="test:cap",
        max_bytes_per_source=800,
        max_bytes_per_document=400,
    )
    result = stage_tabular_source(cfg)

    # Ensure we stopped under (or near) the cap — the staging step may
    # overshoot by at most one document because it emits whole documents.
    assert result["bytes_written"] <= 800 + 400
    assert result["record_count"] >= 1


def test_stage_tabular_errors_on_empty_input(tmp_path):
    input_dir = tmp_path / "raw"
    output_dir = tmp_path / "staged"
    input_dir.mkdir()

    cfg = TabularStagingConfig(
        input_path=str(input_dir),
        output_path=str(output_dir),
        source_label="test:empty",
    )
    with pytest.raises(ValueError, match="No source files"):
        stage_tabular_source(cfg)


def test_stage_tabular_filters_by_extension(tmp_path):
    input_dir = tmp_path / "raw"
    output_dir = tmp_path / "staged"

    _write_csv(input_dir / "keep.csv", "a\n1\n")
    _write_csv(input_dir / "skip.json", "{}")  # wrong extension

    cfg = TabularStagingConfig(
        input_path=str(input_dir),
        output_path=str(output_dir),
        source_label="test:ext",
        file_extensions=(".csv",),
    )
    result = stage_tabular_source(cfg)
    records = _read_staged_records(output_dir)
    assert result["record_count"] == 1
    assert len(records) == 1
    # The ``{}`` content from skip.json must not appear anywhere.
    assert "{}" not in records[0]["text"]
