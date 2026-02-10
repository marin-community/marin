# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for writers module."""

import tempfile
from pathlib import Path

import pyarrow.parquet as pq
import vortex

from zephyr.writers import write_parquet_file, write_vortex_file


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
