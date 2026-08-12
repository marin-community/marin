# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for datakit normalize step."""

import gzip
import json
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fray.current_client import set_current_client
from fray.local_backend import LocalClient
from marin.datakit import normalize as normalize_mod
from marin.datakit.normalize import NORMALIZED_DATA_VERSION, NormalizedData, generate_id, normalize_to_parquet
from marin.execution.artifact import ArtifactRecord, read_artifact, write_artifact, write_record


@pytest.fixture(autouse=True)
def flow_backend_ctx():
    with set_current_client(LocalClient()):
        yield


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
    """Read every main-branch Parquet file under *output_dir*.

    Normalize writes a single ``outputs/main/`` (and ``outputs/dups/``) branch
    per run; tests want just the main output.
    """
    records = []
    for pf in sorted((output_dir / "outputs" / "main").glob("*.parquet")):
        records.extend(pq.read_table(str(pf)).to_pylist())
    return records


def test_normalized_data_artifact_stores_relative_output_dirs(tmp_path: Path, monkeypatch):
    marin_root = tmp_path / "marin"
    monkeypatch.setenv("MARIN_PREFIX", str(marin_root))
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()

    write_artifact(
        NormalizedData(
            main_output_dir=str(marin_root / "datakit/source/outputs/main"),
            dup_output_dir=str(marin_root / "datakit/source/outputs/dups"),
            counters={"records": 3},
        ),
        str(artifact_dir),
    )

    record = json.loads((artifact_dir / ".artifact.json").read_text())
    assert record["result"] == {
        "version": NORMALIZED_DATA_VERSION,
        "main_output_dir": "datakit/source/outputs/main",
        "dup_output_dir": "datakit/source/outputs/dups",
        "counters": {"records": 3},
    }

    loaded = read_artifact(str(artifact_dir), NormalizedData)
    assert loaded.main_output_dir == str(marin_root / "datakit/source/outputs/main")
    assert loaded.dup_output_dir == str(marin_root / "datakit/source/outputs/dups")


def test_normalized_data_loads_v1_paths_from_another_region(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", "gs://marin-us-central1")
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    write_record(
        ArtifactRecord(
            output_path=str(artifact_dir),
            result={
                "version": "v1",
                "main_output_dir": "s3://marin-us-east-02a/marin/datakit/source/outputs/main",
                "dup_output_dir": "s3://marin-us-east-02a/marin/datakit/source/outputs/dups",
                "counters": {},
            },
        )
    )

    loaded = read_artifact(str(artifact_dir), NormalizedData)
    assert loaded.version == NORMALIZED_DATA_VERSION
    assert loaded.main_output_dir == "s3://marin-us-east-02a/marin/datakit/source/outputs/main"
    assert loaded.dup_output_dir == "s3://marin-us-east-02a/marin/datakit/source/outputs/dups"


def test_normalized_data_artifact_preserves_output_dirs_outside_active_prefix(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", "gs://marin-us-central1")
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    write_artifact(
        NormalizedData(
            main_output_dir="s3://unregistered-bucket/data/outputs/main",
            dup_output_dir="s3://unregistered-bucket/data/outputs/dups",
            counters={},
        ),
        str(artifact_dir),
    )

    loaded = read_artifact(str(artifact_dir), NormalizedData)
    assert loaded.main_output_dir == "s3://unregistered-bucket/data/outputs/main"
    assert loaded.dup_output_dir == "s3://unregistered-bucket/data/outputs/dups"


def test_normalize_happy_path(tmp_path: Path, write_jsonl_gz):
    """Produces id (generated), text, source_id (from id_field), and preserves extra columns."""
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
    """text_field override renames the source column to 'text'."""
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
    """Binary Parquet content becomes text rather than a Python bytes representation."""
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
    """id_field override extracts source_id from the chosen column."""
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
    """bare=True drops every column that isn't id, text, or source_id.

    Motivating case: sources whose extra columns vary across shards (e.g.
    starcoderdata's 87 language subdirs each ship a different set of
    GitHub-meta columns, or proof-pile-2's nested ``meta`` dict with
    optional-typed fields). A uniform schema is the only safe option,
    so dump everything but id/text/source_id at the record level before
    the writer sees it.
    """
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


def test_drop_fields_removes_selected_columns_and_preserves_metadata(tmp_path: Path, write_jsonl_gz):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"

    records = [
        {
            "identifier": "a",
            "text": "row a",
            "__index_level_0__": 0,
            "open_type": "Open Science",
        },
        {
            "identifier": "b",
            "text": "row b",
            "open_type": "Open Culture",
        },
    ]
    write_jsonl_gz(input_dir / "data.jsonl.gz", records)

    normalize_to_parquet(
        input_path=str(input_dir),
        output_path=str(output_dir),
        id_field="identifier",
        drop_fields=("__index_level_0__",),
    )

    results = _read_all_parquet(output_dir)
    assert {record["source_id"] for record in results} == {"a", "b"}
    assert {record["open_type"] for record in results} == {"Open Science", "Open Culture"}
    assert all("__index_level_0__" not in record for record in results)


def test_missing_id_field_silently_skipped(tmp_path: Path, write_jsonl_gz):
    """When id_field is absent from records, source_id is omitted (not an error)."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"

    write_jsonl_gz(input_dir / "data.jsonl.gz", [{"text": "No id field here"}])

    normalize_to_parquet(input_path=str(input_dir), output_path=str(output_dir))

    results = _read_all_parquet(output_dir)
    assert "source_id" not in results[0]


@pytest.mark.parametrize(
    "record",
    [
        {"other": "no text here"},  # missing text field
        {"text": "   "},  # whitespace-only text
        {"text": "\xa0\xa0\xa0\n\n\xa0\xa0\xa0"},  # non-breaking spaces + newlines
        {"text": ""},  # empty string
        {"text": None},  # explicit None
    ],
    ids=["missing", "whitespace", "nbsp", "empty", "none"],
)
def test_missing_or_empty_text_filtered(tmp_path: Path, write_jsonl_gz, record):
    """Records with missing or blank text are silently filtered out."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"

    write_jsonl_gz(input_dir / "data.jsonl.gz", [{"text": "valid"}, record])

    result = normalize_to_parquet(input_path=str(input_dir), output_path=str(output_dir))

    results = _read_all_parquet(output_dir)
    assert len(results) == 1
    assert results[0]["text"] == "valid"

    assert result.counters.get("normalize/empty_text_filtered", 0) >= 1


def test_all_records_empty_text_raises(tmp_path: Path, write_jsonl_gz):
    """Pipeline fails when every record has missing/empty text."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"

    write_jsonl_gz(input_dir / "data.jsonl.gz", [{"text": "   "}, {"text": ""}, {"text": None}])

    with pytest.raises(ValueError, match=r"All 3 records were filtered out.*wrong column"):
        normalize_to_parquet(input_path=str(input_dir), output_path=str(output_dir))


def test_subdirectories_merged_into_single_output(tmp_path: Path, write_jsonl_gz):
    """Files discovered across input subdirectories are merged into one flat output."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"

    write_jsonl_gz(input_dir / "subset_a" / "data.jsonl.gz", [{"text": "A doc"}])
    write_jsonl_gz(input_dir / "subset_b" / "data.jsonl.gz", [{"text": "B doc"}])

    result = normalize_to_parquet(input_path=str(input_dir), output_path=str(output_dir))

    assert result.main_output_dir == str(output_dir / "outputs" / "main")
    assert {r["text"] for r in _read_all_parquet(output_dir)} == {"A doc", "B doc"}


def test_exact_dedup(tmp_path: Path, write_jsonl_gz):
    """Records with identical text are deduplicated by content hash."""
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
    """Long whitespace runs are compacted, not dropped. Content is preserved."""
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
    # All three records survive — the pathological one is compacted, not dropped
    assert len(results) == 3
    by_source = {r["source_id"]: r for r in results}
    assert by_source["pathological"]["text"] == "before" + " " * 100 + "after"
    # id is recomputed from the compacted text
    assert by_source["pathological"]["id"] == generate_id("before" + " " * 100 + "after")
    # Normal docs are untouched
    assert by_source["normal"]["text"] == "Hello world"


def test_no_input_files_raises(tmp_path: Path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "output"

    with pytest.raises(FileNotFoundError):
        normalize_to_parquet(input_path=str(input_dir), output_path=str(output_dir))


def test_iter_input_batches_widens_null_column_across_batches(tmp_path: Path, monkeypatch):
    """Later batches may widen Null→Utf8 under vertical_relaxed; rows are kept."""
    monkeypatch.setattr(normalize_mod, "_INPUT_BATCH_ROWS", 2)
    path = tmp_path / "data.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"id": "1", "text": None}),
                json.dumps({"id": "2", "text": None}),
                json.dumps({"id": "3", "text": "hello"}),
            ]
        )
        + "\n"
    )
    frames = list(normalize_mod._iter_input_batches(str(path)))
    assert len(frames) == 2
    assert frames[1].schema["text"] == pl.String
    assert frames[1]["text"].to_list() == ["hello"]


def test_iter_input_batches_rejects_new_column_after_first_batch(tmp_path: Path, monkeypatch):
    """A column that appears only after the first batch is a hard error."""
    monkeypatch.setattr(normalize_mod, "_INPUT_BATCH_ROWS", 2)
    path = tmp_path / "data.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"id": "1"}),
                json.dumps({"id": "2"}),
                json.dumps({"id": "3", "text": "late"}),
            ]
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="Row structure changed after 2 rows"):
        list(normalize_mod._iter_input_batches(str(path)))


def test_align_dataframe_vertical_relaxed_widens_null_to_string():
    first = pl.DataFrame({"a": [1], "b": [None]}, infer_schema_length=None)
    second = pl.DataFrame({"a": [2], "b": ["x"]}, infer_schema_length=None)
    aligned, schema = normalize_mod._align_dataframe_vertical_relaxed(second, first.schema)
    assert schema["b"] == pl.String
    assert aligned.schema == schema


def test_align_dataframe_vertical_relaxed_rejects_new_column():
    first = pl.DataFrame({"a": [1]})
    second = pl.DataFrame({"a": [2], "b": [3]})
    with pytest.raises(ValueError, match="column set mismatch"):
        normalize_mod._align_dataframe_vertical_relaxed(second, first.schema)
