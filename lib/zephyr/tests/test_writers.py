# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for writers module."""

import tempfile
from pathlib import Path

import pyarrow.parquet as pq
import pytest
import vortex

import pyarrow as pa

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
