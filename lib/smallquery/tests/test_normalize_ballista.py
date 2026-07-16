# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for datakit normalize step using DataFusion-Ballista implementation."""

import gzip
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from smallquery.datakit.normalize import generate_id, normalize_to_parquet


@pytest.fixture
def write_jsonl_gz():
    def _write(path: Path, records: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record))
                f.write("\n")

    return _write


def _read_all_parquet(output_dir: Path) -> list[dict]:
    records = []
    main_dir = output_dir / "outputs" / "main"
    for pf in sorted(main_dir.glob("*.parquet")):
        records.extend(pq.read_table(str(pf)).to_pylist())
    return records


def test_normalize_happy_path(tmp_path: Path, write_jsonl_gz):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"

    records = [
        {"id": "abc", "text": "Hello world", "lang": "en", "score": 0.9},
        {"id": "def", "text": "Goodbye world", "lang": "fr", "score": 0.7},
    ]
    write_jsonl_gz(input_dir / "data.jsonl.gz", records)

    normalize_to_parquet(input_path=str(input_dir), output_path=str(output_dir))

    results = _read_all_parquet(output_dir)
    assert len(results) == 2
    by_source = {r["source_id"]: r for r in results}
    assert by_source.keys() == {"abc", "def"}
    assert by_source["abc"]["text"] == "Hello world"
    assert by_source["abc"]["id"] == generate_id("Hello world")
    assert by_source["abc"]["lang"] == "en"
    assert by_source["abc"]["score"] == 0.9


def test_custom_text_field(tmp_path: Path, write_jsonl_gz):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"

    records = [{"body": "Document body here"}]
    write_jsonl_gz(input_dir / "data.jsonl.gz", records)

    normalize_to_parquet(
        input_path=str(input_dir),
        output_path=str(output_dir),
        text_field="body",
    )

    results = _read_all_parquet(output_dir)
    assert results[0]["text"] == "Document body here"
    assert "body" not in results[0]


def test_binary_text_field_decoded_as_utf8(tmp_path: Path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    pq.write_table(
        pa.table({"sha1_git": ["abc"], "content": [b"hello \xff"]}),
        input_dir / "data.parquet",
    )

    normalize_to_parquet(
        input_path=str(input_dir),
        output_path=str(output_dir),
        text_field="content",
        id_field="sha1_git",
        bare=True,
    )

    assert _read_all_parquet(output_dir) == [
        {"id": generate_id("hello \ufffd"), "text": "hello \ufffd", "source_id": "abc"}
    ]


def test_custom_id_field(tmp_path: Path, write_jsonl_gz):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"

    records = [{"my_custom_id": "custom-1", "text": "Some text"}]
    write_jsonl_gz(input_dir / "data.jsonl.gz", records)

    normalize_to_parquet(
        input_path=str(input_dir),
        output_path=str(output_dir),
        id_field="my_custom_id",
    )

    results = _read_all_parquet(output_dir)
    assert results[0]["source_id"] == "custom-1"
    assert "my_custom_id" not in results[0]


def test_bare_mode_strips_extra_columns(tmp_path: Path, write_jsonl_gz):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"

    records = [
        {"id": "a", "text": "row a", "meta": {"max_stars_count": 3}, "lang": "en"},
        {"id": "b", "text": "row b", "meta": None, "lang": "fr"},
    ]
    write_jsonl_gz(input_dir / "data.jsonl.gz", records)

    normalize_to_parquet(
        input_path=str(input_dir),
        output_path=str(output_dir),
        bare=True,
    )

    results = _read_all_parquet(output_dir)
    assert len(results) == 2
    for r in results:
        assert set(r.keys()) == {"id", "text", "source_id"}
    assert {r["source_id"] for r in results} == {"a", "b"}
    assert {r["text"] for r in results} == {"row a", "row b"}


def test_missing_id_field_silently_skipped(tmp_path: Path, write_jsonl_gz):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"

    write_jsonl_gz(input_dir / "data.jsonl.gz", [{"text": "No id field here"}])

    normalize_to_parquet(input_path=str(input_dir), output_path=str(output_dir))

    results = _read_all_parquet(output_dir)
    assert "source_id" not in results[0]


@pytest.mark.parametrize(
    "record",
    [
        {"other": "no text here"},
        {"text": "   "},
        {"text": "\xa0\xa0\xa0\n\n\xa0\xa0\xa0"},
        {"text": ""},
        {"text": None},
    ],
    ids=["missing", "whitespace", "nbsp", "empty", "none"],
)
def test_missing_or_empty_text_filtered(tmp_path: Path, write_jsonl_gz, record):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"

    write_jsonl_gz(input_dir / "data.jsonl.gz", [{"text": "valid"}, record])

    result = normalize_to_parquet(input_path=str(input_dir), output_path=str(output_dir))

    results = _read_all_parquet(output_dir)
    assert len(results) == 1
    assert results[0]["text"] == "valid"

    assert result.counters.get("normalize/empty_text_filtered", 0) >= 1


def test_all_records_empty_text_raises(tmp_path: Path, write_jsonl_gz):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"

    write_jsonl_gz(input_dir / "data.jsonl.gz", [{"text": "   "}, {"text": ""}, {"text": None}])

    with pytest.raises(ValueError, match=r"All 3 records were filtered out.*wrong column"):
        normalize_to_parquet(input_path=str(input_dir), output_path=str(output_dir))


def test_subdirectories_merged_into_single_output(tmp_path: Path, write_jsonl_gz):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"

    write_jsonl_gz(input_dir / "subset_a" / "data.jsonl.gz", [{"text": "A doc"}])
    write_jsonl_gz(input_dir / "subset_b" / "data.jsonl.gz", [{"text": "B doc"}])

    result = normalize_to_parquet(input_path=str(input_dir), output_path=str(output_dir))

    assert result.main_output_dir == str(output_dir / "outputs" / "main")
    assert {r["text"] for r in _read_all_parquet(output_dir)} == {"A doc", "B doc"}


def test_exact_dedup(tmp_path: Path, write_jsonl_gz):
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
    assert {r["text"] for r in results} == {"Duplicate text", "Unique text"}
    assert len(results) == 2


def test_whitespace_compaction(tmp_path: Path, write_jsonl_gz):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"

    records = [
        {"id": "normal", "text": "Hello world"},
        {"id": "pathological", "text": "before" + " " * 500 + "after"},
        {"id": "also_normal", "text": "short  spaces  are  fine"},
    ]
    write_jsonl_gz(input_dir / "data.jsonl.gz", records)

    normalize_to_parquet(
        input_path=str(input_dir),
        output_path=str(output_dir),
        max_whitespace_run_chars=100,
    )

    results = _read_all_parquet(output_dir)
    assert len(results) == 3
    by_source = {r["source_id"]: r for r in results}
    assert by_source["pathological"]["text"] == "before" + " " * 100 + "after"
    assert by_source["pathological"]["id"] == generate_id("before" + " " * 100 + "after")
    assert by_source["normal"]["text"] == "Hello world"


def test_no_input_files_raises(tmp_path: Path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "output"

    with pytest.raises(FileNotFoundError):
        normalize_to_parquet(input_path=str(input_dir), output_path=str(output_dir))
