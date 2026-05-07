# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for writers module."""

import os
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import vortex
from zephyr.writers import (
    atomic_rename,
    infer_arrow_schema,
    unique_temp_path,
    write_levanter_cache,
    write_parquet_file,
    write_vortex_file,
)


def test_unique_temp_path_produces_distinct_paths():
    """Each call to unique_temp_path returns a different path."""
    paths = {unique_temp_path("/some/output.txt") for _ in range(10)}
    assert len(paths) == 10
    for p in paths:
        assert p.startswith("/some/output.txt.tmp.")


def test_atomic_rename_uses_unique_temp_paths(tmp_path):
    """Concurrent atomic_rename calls use distinct temp paths (UUID collision avoidance)."""
    output = str(tmp_path / "out.txt")
    observed_temps = []

    for _ in range(5):
        with atomic_rename(output) as temp_path:
            observed_temps.append(temp_path)
            Path(temp_path).write_text("data")

    assert len(set(observed_temps)) == 5, "Each call should produce a unique temp path"
    for tp in observed_temps:
        assert ".tmp." in tp


def test_atomic_rename_cleans_up_on_error(tmp_path):
    """Temp file is removed when the context raises an exception."""
    output = str(tmp_path / "out.txt")

    with pytest.raises(RuntimeError, match="boom"):
        with atomic_rename(output) as temp_path:
            Path(temp_path).write_text("bad")
            raise RuntimeError("boom")

    assert not Path(temp_path).exists()
    assert not Path(output).exists()


def _make_levanter_records(n: int) -> list[dict[str, list[int]]]:
    return [{"input_ids": [i, i + 100], "attention_mask": [1, 1]} for i in range(n)]


def _require_levanter():
    cache_mod = pytest.importorskip("levanter.store.cache")
    tree_store_mod = pytest.importorskip("levanter.store.tree_store")
    return cache_mod.CacheMetadata, cache_mod.SerialCacheWriter, tree_store_mod.TreeStore


def test_write_vortex_file_basic():
    """Test basic vortex file writing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = str(Path(tmpdir) / "test.vortex")
        records = [
            {"id": 1, "name": "Alice", "age": 30},
            {"id": 2, "name": "Bob", "age": 25},
            {"id": 3, "name": "Charlie", "age": 35},
        ]

        result = write_vortex_file(records, output_path)

        assert result["path"] == output_path
        assert result["count"] == 3
        assert Path(output_path).exists()

        # Verify we can read it back
        vf = vortex.open(output_path)
        reader = vf.to_arrow()
        table = reader.read_all()

        assert len(table) == 3
        assert table.column("name").to_pylist() == ["Alice", "Bob", "Charlie"]


def test_write_vortex_file_empty():
    """Test writing an empty vortex file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = str(Path(tmpdir) / "empty.vortex")
        records = []

        result = write_vortex_file(records, output_path)

        assert result["path"] == output_path
        assert result["count"] == 0
        assert Path(output_path).exists()

        # Verify we can read it back
        vf = vortex.open(output_path)
        reader = vf.to_arrow()
        table = reader.read_all()
        assert len(table) == 0


def test_write_vortex_file_single_record():
    """Test writing a single record."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = str(Path(tmpdir) / "single.vortex")
        records = [{"id": 1, "name": "Alice"}]

        result = write_vortex_file(records, output_path)

        assert result["path"] == output_path
        assert result["count"] == 1

        vf = vortex.open(output_path)
        reader = vf.to_arrow()
        table = reader.read_all()
        assert len(table) == 1
        assert table.column("name").to_pylist() == ["Alice"]


def test_write_parquet_file_basic():
    """Test basic parquet file writing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = str(Path(tmpdir) / "test.parquet")
        records = [
            {"id": 1, "name": "Alice", "age": 30},
            {"id": 2, "name": "Bob", "age": 25},
        ]

        result = write_parquet_file(records, output_path)

        assert result["path"] == output_path
        assert result["count"] == 2
        assert Path(output_path).exists()

        # Verify we can read it back
        table = pq.read_table(output_path)
        assert len(table) == 2


def test_write_parquet_file_widens_null_to_concrete_type():
    """First batch pins a field as null; a later batch with a concrete type widens cleanly.

    This is the stackv2 failure mode: the first ``_MICRO_BATCH_SIZE`` (=8)
    records all had ``None`` for a field, pinning it to ``pa.null()`` —
    later records with real values would fail without schema widening.
    Behavior must: (a) succeed, (b) land the widened schema on disk, (c)
    preserve all values from both batches.
    """
    records = [{"x": None}] * 8 + [{"x": "hello"}]
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = str(Path(tmpdir) / "test.parquet")
        result = write_parquet_file(records, output_path)
        assert result["count"] == 9

        table = pq.read_table(output_path)
        assert len(table) == 9
        assert pa.types.is_string(table.schema.field("x").type)
        xs = table.column("x").to_pylist()
        assert xs[:8] == [None] * 8
        assert xs[8] == "hello"


def test_write_parquet_file_captures_fields_appearing_in_later_batches():
    """A field absent from the first batch but present later must not be silently dropped."""
    records = [{"x": "a"}] * 8 + [{"x": "b", "z": 42}]
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = str(Path(tmpdir) / "test.parquet")
        result = write_parquet_file(records, output_path)
        assert result["count"] == 9

        table = pq.read_table(output_path)
        assert "z" in table.schema.names, "field `z` must survive to disk, not be dropped"
        assert table.column("z").to_pylist() == [None] * 8 + [42]


def test_write_parquet_file_raises_on_incompatible_type_conflict():
    """Genuine type conflicts (e.g. int vs string) must still raise a clear error."""
    records = [{"x": i} for i in range(8)] + [{"x": "stringy"}]
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = str(Path(tmpdir) / "test.parquet")
        with pytest.raises((pa.ArrowInvalid, pa.ArrowTypeError)) as excinfo:
            write_parquet_file(records, output_path)
    msg = str(excinfo.value)
    assert "int" in msg.lower() or "int64" in msg.lower()
    assert "string" in msg.lower()


def test_write_parquet_file_empty():
    """Test writing an empty parquet file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = str(Path(tmpdir) / "empty.parquet")
        records = []

        result = write_parquet_file(records, output_path)

        assert result["path"] == output_path
        assert result["count"] == 0
        assert Path(output_path).exists()

        table = pq.read_table(output_path)
        assert len(table) == 0


def test_write_levanter_cache_end_to_end():
    """Write records and verify they can be read back."""
    _, _, TreeStore = _require_levanter()

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = str(Path(tmpdir) / "cache")
        records = _make_levanter_records(8)

        result = write_levanter_cache(iter(records), output_path, metadata={})

        assert result["path"] == output_path
        assert result["count"] == len(records)
        assert Path(output_path, ".success").exists()

        store = TreeStore.open(records[0], output_path, mode="r", cache_metadata=False)
        assert len(store) == len(records)
        assert store[0]["input_ids"].tolist() == records[0]["input_ids"]
        assert store[len(records) - 1]["input_ids"].tolist() == records[len(records) - 1]["input_ids"]


def test_atomic_rename_s3_directory_preserves_layout(tmp_path):
    """S3 atomic_rename must not add extra nesting for directory outputs.

    When the yielded path is used as a directory (e.g. by write_levanter_cache),
    fs.put must place contents directly at the destination — not under an extra
    ``output/`` subdirectory.  fsspec nests when the source has no trailing
    slash and the destination already exists, so atomic_rename must account for
    that.
    """
    from unittest.mock import patch

    from fsspec.implementations.local import LocalFileSystem

    dest = tmp_path / "dest"
    dest.mkdir()  # pre-create so fsspec considers it "existing"
    local_fs = LocalFileSystem()

    with patch("zephyr.writers.url_to_fs", return_value=(local_fs, str(dest))):
        with atomic_rename("s3://bucket/dest") as local_path:
            os.makedirs(local_path)
            (Path(local_path) / "shard_0.bin").write_bytes(b"data0")
            (Path(local_path) / "shard_1.bin").write_bytes(b"data1")

    assert (dest / "shard_0.bin").exists(), "shard_0.bin should be directly under dest"
    assert (dest / "shard_1.bin").exists(), "shard_1.bin should be directly under dest"
    assert not (dest / "output").exists(), "should not have extra 'output' nesting"


def test_atomic_rename_s3_single_file(tmp_path):
    """S3 atomic_rename works correctly for single-file outputs."""
    from unittest.mock import patch

    from fsspec.implementations.local import LocalFileSystem

    dest = tmp_path / "output.jsonl"
    local_fs = LocalFileSystem()

    with patch("zephyr.writers.url_to_fs", return_value=(local_fs, str(dest))):
        with atomic_rename("s3://bucket/output.jsonl") as local_path:
            Path(local_path).write_text("line1\nline2\n")

    assert dest.exists()
    assert dest.read_text() == "line1\nline2\n"


def test_infer_arrow_schema_basic():
    """Test schema inference with basic Python types."""
    records = [{"id": 1, "name": "Alice", "score": 95.5, "active": True}]
    schema = infer_arrow_schema(records)
    assert schema.field("id").type == pa.int64()
    assert schema.field("score").type == pa.float64()
    assert schema.field("active").type == pa.bool_()
    assert len(schema) == 4


def test_infer_arrow_schema_none_in_first_row():
    """Schema inference resolves None from non-None values in later rows."""
    records = [
        {"id": 1, "name": "Alice", "score": None},
        {"id": 2, "name": "Bob", "score": 95.5},
    ]
    schema = infer_arrow_schema(records)
    assert schema.field("score").type == pa.float64()


def test_infer_arrow_schema_all_none():
    """When all values for a field are None, the type is null."""
    records = [
        {"id": 1, "value": None},
        {"id": 2, "value": None},
    ]
    schema = infer_arrow_schema(records)
    assert schema.field("value").type == pa.null()


def test_infer_arrow_schema_nested_dict():
    """Schema inference handles nested dicts."""
    records = [{"id": 1, "meta": {"key": "val", "count": 3}}]
    schema = infer_arrow_schema(records)
    meta_type = schema.field("meta").type
    assert isinstance(meta_type, pa.StructType)
    assert meta_type.get_field_index("key") >= 0
    assert meta_type.get_field_index("count") >= 0


def test_infer_arrow_schema_mixed_types_fails():
    """Schema inference fails when a column has incompatible types (float then string)."""
    records = [
        {"id": 1, "foo": None},
        {"id": 2, "foo": 1.5},
        {"id": 3, "foo": 2.5},
        {"id": 4, "foo": "bar"},
    ]
    with pytest.raises(pa.lib.ArrowInvalid):
        infer_arrow_schema(records)
