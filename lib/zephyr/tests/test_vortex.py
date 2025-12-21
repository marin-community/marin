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

"""Tests for vortex file format support."""

import pytest

from fray.job import create_job_ctx
from zephyr import Backend, Dataset
from zephyr.expr import col
from zephyr.readers import InputFileSpec, load_vortex
from zephyr.writers import write_vortex_file


@pytest.fixture
def vortex_file(tmp_path):
    """Create a test vortex file with sample data."""
    records = [{"id": i, "name": f"item_{i}", "score": i * 10} for i in range(100)]
    path = tmp_path / "test.vortex"
    write_vortex_file(records, str(path))
    return path


@pytest.fixture(
    params=[
        pytest.param("sync", id="sync"),
        pytest.param("threadpool", id="thread"),
    ]
)
def sync_backend(request):
    """Backend fixture for sync and threadpool backends."""
    return create_job_ctx(request.param, max_workers=2)


class TestVortexReader:
    """Tests for load_vortex() function."""

    def test_load_vortex_basic(self, vortex_file):
        """Test basic vortex file reading."""
        records = list(load_vortex(str(vortex_file)))
        assert len(records) == 100
        assert records[0]["id"] == 0
        assert records[0]["name"] == "item_0"
        assert records[0]["score"] == 0

    def test_load_vortex_column_projection(self, vortex_file):
        """Test column selection (projection)."""
        spec = InputFileSpec(path=str(vortex_file), columns=["id", "name"])
        records = list(load_vortex(spec))
        assert len(records) == 100
        assert set(records[0].keys()) == {"id", "name"}

    def test_load_vortex_empty_file(self, tmp_path):
        """Test loading an empty vortex file."""
        empty_path = tmp_path / "empty.vortex"
        write_vortex_file([], str(empty_path))

        records = list(load_vortex(str(empty_path)))
        assert records == []


class TestVortexWriter:
    """Tests for write_vortex_file() function."""

    def test_write_vortex_basic(self, tmp_path):
        """Test basic vortex file writing."""
        records = [{"id": i, "value": i * 2} for i in range(10)]
        output_path = tmp_path / "output.vortex"

        result = write_vortex_file(records, str(output_path))

        assert result["count"] == 10
        assert output_path.exists()

        # Verify roundtrip
        loaded = list(load_vortex(str(output_path)))
        assert loaded == records

    def test_write_vortex_empty(self, tmp_path):
        """Test writing empty dataset."""
        output_path = tmp_path / "empty.vortex"
        result = write_vortex_file([], str(output_path))

        assert result["count"] == 0
        assert output_path.exists()

    def test_write_vortex_single_record(self, tmp_path):
        """Test writing single record."""
        records = [{"key": "value", "number": 42}]
        output_path = tmp_path / "single.vortex"

        result = write_vortex_file(records, str(output_path))
        assert result["count"] == 1

        loaded = list(load_vortex(str(output_path)))
        assert loaded == records


class TestVortexPipeline:
    """Tests for vortex in Dataset pipelines."""

    def test_read_write_pipeline(self, sync_backend, vortex_file, tmp_path):
        """Test read -> filter -> write pipeline with vortex."""
        output_pattern = str(tmp_path / "output-{shard:05d}.vortex")

        ds = (
            Dataset.from_files(str(vortex_file))
            .load_vortex()
            .filter(lambda r: r["score"] > 500)
            .write_vortex(output_pattern)
        )

        results = list(Backend.execute(ds, context=sync_backend))
        assert len(results) == 1

        # Verify output
        loaded = list(load_vortex(results[0]))
        assert len(loaded) == 49  # scores 510, 520, ..., 990
        assert all(r["score"] > 500 for r in loaded)

    def test_load_file_auto_detects_vortex(self, sync_backend, vortex_file, tmp_path):
        """Test that load_file() auto-detects vortex format."""
        output_pattern = str(tmp_path / "output-{shard:05d}.jsonl.gz")

        ds = Dataset.from_files(str(vortex_file)).load_file().filter(lambda r: r["id"] < 10).write_jsonl(output_pattern)

        results = list(Backend.execute(ds, context=sync_backend))
        assert len(results) == 1

    def test_vortex_to_parquet_conversion(self, sync_backend, vortex_file, tmp_path):
        """Test converting vortex to parquet."""
        output_pattern = str(tmp_path / "output-{shard:05d}.parquet")

        ds = Dataset.from_files(str(vortex_file)).load_vortex().write_parquet(output_pattern)

        results = list(Backend.execute(ds, context=sync_backend))
        assert len(results) == 1

        # Verify parquet output
        from zephyr.readers import load_parquet

        loaded = list(load_parquet(results[0]))
        assert len(loaded) == 100

    def test_parquet_to_vortex_conversion(self, sync_backend, tmp_path):
        """Test converting parquet to vortex."""
        # Create parquet file
        from zephyr.writers import write_parquet_file

        records = [{"a": i, "b": f"val_{i}"} for i in range(50)]
        parquet_path = tmp_path / "input.parquet"
        write_parquet_file(records, str(parquet_path))

        output_pattern = str(tmp_path / "output-{shard:05d}.vortex")

        ds = Dataset.from_files(str(parquet_path)).load_parquet().write_vortex(output_pattern)

        results = list(Backend.execute(ds, context=sync_backend))
        assert len(results) == 1

        # Verify vortex output
        loaded = list(load_vortex(results[0]))
        assert loaded == records


class TestVortexFilterPushdown:
    """Tests for filter pushdown to vortex reader."""

    def test_expression_filter_pushdown(self, sync_backend, vortex_file):
        """Test filter pushdown with expression."""
        ds = Dataset.from_files(str(vortex_file)).load_vortex().filter(col("score") > 500)

        results = list(Backend.execute(ds, context=sync_backend))
        assert len(results) == 49  # scores 510, 520, ..., 990
        assert all(r["score"] > 500 for r in results)

    def test_column_select_pushdown(self, sync_backend, vortex_file):
        """Test column selection pushdown."""
        ds = Dataset.from_files(str(vortex_file)).load_vortex().select("id", "score")

        results = list(Backend.execute(ds, context=sync_backend))
        assert len(results) == 100
        assert set(results[0].keys()) == {"id", "score"}
