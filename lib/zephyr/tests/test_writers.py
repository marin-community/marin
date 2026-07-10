# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for writers module."""

import tempfile
import uuid
from pathlib import Path

import fsspec
import fsspec.config
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import vortex
from pyarrow import fs as pa_fs
from zephyr.writers import (
    _pyarrow_filesystem,
    _s3_filesystem_kwargs,
    infer_arrow_schema,
    write_parquet_file,
    write_vortex_file,
)


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


def test_write_parquet_file_unaddressable_protocol_falls_back_to_fsspec():
    """Protocols pyarrow cannot address still round-trip via the fsspec handle."""
    bucket = f"zephyr-writers-{uuid.uuid4().hex}"
    output_path = f"memory://{bucket}/out.parquet"
    records = [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]

    result = write_parquet_file(iter(records), output_path)

    assert result["count"] == 2
    with fsspec.filesystem("memory").open(f"/{bucket}/out.parquet", "rb") as f:
        table = pq.read_table(f)
    assert table.to_pylist() == records


def test_pyarrow_filesystem_selection():
    """Local paths get a native LocalFileSystem; unknown protocols return None."""
    fs, path = _pyarrow_filesystem("/tmp/out.parquet")
    assert isinstance(fs, pa_fs.LocalFileSystem)
    assert path == "/tmp/out.parquet"

    fs, path = _pyarrow_filesystem("file:///tmp/out.parquet")
    assert isinstance(fs, pa_fs.LocalFileSystem)
    assert path == "/tmp/out.parquet"

    assert _pyarrow_filesystem("memory://bucket/out.parquet") is None


def test_s3_filesystem_kwargs_from_fsspec_conf(monkeypatch):
    """The iris-exported FSSPEC_S3 block maps onto native S3FileSystem kwargs.

    CoreWeave object storage rejects path-style requests with HTTP 400, so
    the virtual addressing style configured for s3fs must translate to
    ``force_virtual_addressing`` on the native filesystem.
    """
    monkeypatch.setitem(
        fsspec.config.conf,
        "s3",
        {
            "endpoint_url": "https://object.example.coreweave.com",
            "client_kwargs": {"region_name": "auto"},
            "config_kwargs": {"s3": {"addressing_style": "virtual"}},
        },
    )
    kwargs = _s3_filesystem_kwargs()
    assert kwargs["endpoint_override"] == "https://object.example.coreweave.com"
    assert kwargs["region"] == "auto"
    assert kwargs["force_virtual_addressing"] is True
    assert kwargs["connect_timeout"] > 0
    assert kwargs["request_timeout"] > 0

    monkeypatch.setitem(fsspec.config.conf, "s3", {})
    monkeypatch.delenv("AWS_ENDPOINT_URL", raising=False)
    monkeypatch.delenv("AWS_REGION", raising=False)
    assert "endpoint_override" not in _s3_filesystem_kwargs()
    assert "force_virtual_addressing" not in _s3_filesystem_kwargs()


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
