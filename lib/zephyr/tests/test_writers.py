# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for writers module."""

import json
import logging
import tempfile
from pathlib import Path

import fsspec
import pyarrow.parquet as pq
import pytest
import vortex

from zephyr.writers import write_levanter_cache, write_parquet_file, write_vortex_file


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


def test_write_levanter_cache_resumes_from_partial_tmp_end_to_end():
    """A rerun should resume from an interrupted .tmp cache directory."""
    CacheMetadata, SerialCacheWriter, TreeStore = _require_levanter()

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = str(Path(tmpdir) / "cache")
        tmp_output_path = f"{output_path}.tmp"
        records = _make_levanter_records(8)
        expected_token_count = sum(len(record["input_ids"]) for record in records)

        with SerialCacheWriter(
            tmp_output_path, records[0], shard_name=output_path, metadata=CacheMetadata({}), mode="w"
        ) as writer:
            writer.write_batch(records[:3])

        result = write_levanter_cache(iter(records), output_path, metadata={})

        assert result["path"] == output_path
        assert result["count"] == len(records)
        assert result["token_count"] == expected_token_count
        assert Path(output_path, ".success").exists()

        stats = json.loads(Path(output_path, ".stats.json").read_text())
        assert stats["count"] == len(records)
        assert stats["token_count"] == expected_token_count

        store = TreeStore.open(records[0], output_path, mode="r", cache_metadata=False)
        assert len(store) == len(records)
        assert store[0]["input_ids"].tolist() == records[0]["input_ids"]
        assert store[len(records) - 1]["input_ids"].tolist() == records[len(records) - 1]["input_ids"]


def test_write_levanter_cache_ignores_stale_tmp_when_output_exists():
    """A stale tmp directory must not override an already-published output."""
    CacheMetadata, SerialCacheWriter, TreeStore = _require_levanter()

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = str(Path(tmpdir) / "cache")
        old_records = _make_levanter_records(6)
        new_records = _make_levanter_records(3)

        write_levanter_cache(iter(old_records), output_path, metadata={})

        tmp_output_path = f"{output_path}.tmp"
        with SerialCacheWriter(
            tmp_output_path, old_records[0], shard_name=output_path, metadata=CacheMetadata({}), mode="w"
        ) as writer:
            writer.write_batch(old_records[:5])

        result = write_levanter_cache(iter(new_records), output_path, metadata={})
        assert result["count"] == len(new_records)

        store = TreeStore.open(new_records[0], output_path, mode="r", cache_metadata=False)
        assert len(store) == len(new_records)
        assert store[len(new_records) - 1]["input_ids"].tolist() == new_records[len(new_records) - 1]["input_ids"]


def test_write_levanter_cache_fails_if_partial_tmp_exceeds_input():
    """If tmp data is ahead of the input stream, fail instead of publishing stale data."""
    CacheMetadata, SerialCacheWriter, _ = _require_levanter()

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = str(Path(tmpdir) / "cache")
        stale_records = _make_levanter_records(5)
        new_records = _make_levanter_records(2)

        tmp_output_path = f"{output_path}.tmp"
        with SerialCacheWriter(
            tmp_output_path, stale_records[0], shard_name=output_path, metadata=CacheMetadata({}), mode="w"
        ) as writer:
            writer.write_batch(stale_records)

        with pytest.raises(ValueError, match="Temporary cache"):
            write_levanter_cache(iter(new_records), output_path, metadata={})


def test_write_levanter_cache_restores_previous_output_if_publish_fails(monkeypatch):
    """If final publish fails, the previously published output should be restored."""
    _, _, TreeStore = _require_levanter()

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = str(Path(tmpdir) / "cache")
        original_records = _make_levanter_records(5)
        replacement_records = _make_levanter_records(2)

        write_levanter_cache(iter(original_records), output_path, metadata={})

        original_mv = fsspec.implementations.local.LocalFileSystem.mv
        target_tmp_path = f"{output_path}.tmp"
        failed_once = False

        def flaky_mv(self, path1, path2, **kwargs):
            nonlocal failed_once
            if path1 == target_tmp_path and path2 == output_path and not failed_once:
                failed_once = True
                raise RuntimeError("simulated publish failure")
            return original_mv(self, path1, path2, **kwargs)

        monkeypatch.setattr(fsspec.implementations.local.LocalFileSystem, "mv", flaky_mv)

        with pytest.raises(RuntimeError, match="simulated publish failure"):
            write_levanter_cache(iter(replacement_records), output_path, metadata={})

        restored_store = TreeStore.open(original_records[0], output_path, mode="r", cache_metadata=False)
        assert len(restored_store) == len(original_records)
