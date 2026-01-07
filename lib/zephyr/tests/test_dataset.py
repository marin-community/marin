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

"""Tests for Dataset API."""

import json
import time
from functools import partial
from pathlib import Path

import pytest
from fray.job import create_job_ctx
from zephyr import Backend, Dataset, load_file, load_parquet
from zephyr._test_helpers import SampleDataclass
from zephyr.dataset import FilterOp, MapOp, WindowOp
from zephyr.writers import write_parquet_file

from .conftest import CallCounter


def create_vortex_file(tmp_path):
    """Create a test vortex file with sample data for pushdown tests.

    Creates 100 records with id (0-99), name, and score (id * 10).
    """
    from zephyr.writers import write_vortex_file

    records = [{"id": i, "name": f"item_{i}", "score": i * 10} for i in range(100)]
    path = tmp_path / "test.vortex"
    write_vortex_file(records, str(path))
    return path


@pytest.fixture
def sample_data():
    """Sample data for testing."""
    return list(range(1, 11))  # [1, 2, 3, ..., 10]


def test_from_list(sample_data, backend):
    """Test creating dataset from list."""
    ds = Dataset.from_list(sample_data)
    assert list(Backend.execute(ds, context=backend)) == sample_data


def test_from_iterable(backend):
    """Test creating dataset from iterable."""
    ds = Dataset.from_iterable(range(5))
    assert list(Backend.execute(ds, context=backend)) == [0, 1, 2, 3, 4]


def test_filter(sample_data, backend):
    """Test filtering dataset."""
    ds = Dataset.from_list(sample_data).filter(lambda x: x % 2 == 0)
    assert list(Backend.execute(ds, context=backend)) == [2, 4, 6, 8, 10]


def test_take_per_shard(backend):
    ds = Dataset.from_list([list(range(10))]).flat_map(lambda x: x).take_per_shard(5)
    result = list(Backend.execute(ds, context=backend))
    assert result == [0, 1, 2, 3, 4]

    ds = Dataset.from_list([list(range(10))]).flat_map(lambda x: x).take_per_shard(0)
    result = list(Backend.execute(ds, context=backend))
    assert result == []

    # Create 3 shards with 5 items each
    ds = (
        Dataset.from_list([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]])
        .flat_map(lambda x: x)
        .take_per_shard(2)
    )

    result = sorted(list(Backend.execute(ds, context=backend)))
    # Each of 3 shards contributes 2 items = 6 total
    # Shard 0: [0, 1], Shard 1: [5, 6], Shard 2: [10, 11]
    assert result == [0, 1, 5, 6, 10, 11]


def test_take_with_filter_and_map(backend):
    """Test take fuses with other operations."""
    ds = (
        Dataset.from_list([list(range(20))])
        .flat_map(lambda x: x)
        .filter(lambda x: x % 2 == 0)  # [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
        .take_per_shard(5)  # [0, 2, 4, 6, 8]
        .map(lambda x: x * 2)  # [0, 4, 8, 12, 16]
    )
    result = list(Backend.execute(ds, context=backend))
    assert result == [0, 4, 8, 12, 16]


def test_window(backend):
    """Test window operation (same as batch)."""
    ds = Dataset.from_list([[1, 2, 3, 4, 5]]).flat_map(lambda x: x).window(2)
    windows = list(Backend.execute(ds, context=backend))
    assert windows == [[1, 2], [3, 4], [5]]


def test_window_by_size_based(backend):
    """Test window_by with size-based windowing."""
    data = [
        {"id": 1, "size": 5_000_000_000},  # 5GB
        {"id": 2, "size": 6_000_000_000},  # 6GB - triggers new window
        {"id": 3, "size": 3_000_000_000},  # 3GB
        {"id": 4, "size": 8_000_000_000},  # 8GB - triggers new window
    ]

    # Use flat_map to ensure all items are in a single shard
    ds = (
        Dataset.from_list([data])
        .flat_map(lambda x: x)
        .window_by(
            folder_fn=lambda total_size, item: (total_size + item["size"] < 10_000_000_000, total_size + item["size"]),
            initial_state=0,
        )
    )

    windows = list(Backend.execute(ds, context=backend))
    # Window 1: [id=1 (5GB)] - total 5GB
    # Window 2: [id=2 (6GB), id=3 (3GB)] - total 9GB
    # Window 3: [id=4 (8GB)] - total 8GB
    assert len(windows) == 3
    assert len(windows[0]) == 1
    assert windows[0][0]["id"] == 1
    assert len(windows[1]) == 2
    assert windows[1][0]["id"] == 2
    assert windows[1][1]["id"] == 3
    assert len(windows[2]) == 1
    assert windows[2][0]["id"] == 4


def test_window_by_count_based(backend):
    """Test window_by with custom count logic."""
    data = list(range(1, 11))

    # Window by sum < 10
    # Use flat_map to ensure all items are in a single shard
    ds = (
        Dataset.from_list([data])
        .flat_map(lambda x: x)
        .window_by(
            folder_fn=lambda total, item: (total + item < 10, total + item),
            initial_state=0,
        )
    )

    windows = list(Backend.execute(ds, context=backend))
    # Window 1: [1, 2, 3] - sum = 6
    # Window 2: [4, 5] - sum = 9
    # Window 3: [6] - sum = 6
    # Window 4: [7] - sum = 7
    # Window 5: [8] - sum = 8
    # Window 6: [9] - sum = 9
    # Window 7: [10] - sum = 10
    assert len(windows) >= 5


def test_map(backend):
    """Test map operation with all backends."""
    ds = Dataset.from_list([1, 2, 3, 4, 5]).map(lambda x: x * 2)
    result = list(Backend.execute(ds, context=backend))
    assert sorted(result) == [2, 4, 6, 8, 10]


def test_bounded_parallelism_ray():
    """Test that Ray backend respects max_parallelism."""

    def slow_fn(x):
        time.sleep(0.2)
        return x * 2

    # With max_parallelism=2, processing 6 items should take ~3*0.2 = 0.6s
    ctx = create_job_ctx("ray")
    ds = Dataset.from_list([1, 2, 3, 4, 5, 6]).map(slow_fn)

    start = time.time()
    result = list(Backend.execute(ds, ctx, max_parallelism=2))
    elapsed = time.time() - start

    assert sorted(result) == [2, 4, 6, 8, 10, 12]
    # Should take at least 0.6s due to bounded parallelism
    assert elapsed >= 0.5, f"Elapsed time {elapsed} suggests no bounding"


def test_chaining_operations(backend):
    """Test chaining multiple operations.

    Use flat_map to create multi-item shards that work across all backends.
    """
    ds = (
        Dataset.from_list([list(range(1, 11))])
        .flat_map(lambda x: x)
        .filter(lambda x: x % 2 == 0)  # [2, 4, 6, 8, 10]
        .map(lambda x: x * 2)  # [4, 8, 12, 16, 20]
        .window(2)  # [[4, 8], [12, 16], [20]]
    )

    result = list(Backend.execute(ds, context=backend))

    # Flatten and sort for comparison since some backends may reorder
    flattened = [item for batch in result for item in batch]
    assert sorted(flattened) == [4, 8, 12, 16, 20]
    assert len(result) == 3
    assert len(result[0]) == 2
    assert len(result[1]) == 2
    assert len(result[2]) == 1


def test_lazy_evaluation():
    """Test that operations are lazy until backend executes."""
    call_count = 0

    def counting_fn(x):
        nonlocal call_count
        call_count += 1
        return x * 2

    # Create dataset with map - should not execute yet
    ds = Dataset.from_list([1, 2, 3]).map(counting_fn)
    assert call_count == 0

    # Now execute - should call function
    ctx = create_job_ctx("sync")
    result = list(Backend.execute(ds, context=ctx))
    assert result == [2, 4, 6]
    assert call_count == 3


def test_empty_dataset(backend):
    """Test operations on empty dataset."""
    ds = Dataset.from_list([])
    assert list(Backend.execute(ds, context=backend)) == []
    assert list(Backend.execute(ds.filter(lambda x: True), backend)) == []
    assert list(Backend.execute(ds.map(lambda x: x * 2), backend)) == []
    assert list(Backend.execute(ds.window(10), backend)) == []


def test_reshard(backend):
    """Test reshard operation redistributes data across target number of shards."""
    # Test 1: Start with 1 item (1 shard), expand to many items, then reshard
    ds = (
        Dataset.from_list([list(range(50))])  # 1 shard with 50 records
        .flat_map(lambda x: x)  # Expand to 50 items
        .reshard(5)  # Redistribute to 5 shards
        .map(lambda x: x * 2)
    )
    result = sorted(Backend.execute(ds, context=backend))
    assert result == [x * 2 for x in range(50)]

    # Test 2: Start with many items, reshard to fewer
    ds = Dataset.from_list(range(50)).reshard(5).map(lambda x: x + 100)  # 50 shards  # Consolidate to 5 shards
    result = sorted(Backend.execute(ds, context=backend))
    assert result == [x + 100 for x in range(50)]

    # Test 3: Reshard preserves order when materializing all shards
    ds = Dataset.from_list(range(10)).reshard(3)
    result = list(Backend.execute(ds, context=backend))
    assert sorted(result) == list(range(10))


def test_reshard_noop(backend):
    """Test reshard with None is a noop, and non-positive values raise ValueError"""

    def yield_1(it):
        yield from [1]

    ds = Dataset.from_list(range(10)).reshard(None).map_shard(yield_1)
    assert sum(list(Backend.execute(ds, context=backend))) == 10

    ds = Dataset.from_list(range(10)).reshard(2).map_shard(yield_1)
    assert sum(list(Backend.execute(ds, context=backend))) == 2

    with pytest.raises(ValueError, match="num_shards must be positive"):
        Dataset.from_list(range(10)).reshard(-5)

    with pytest.raises(ValueError, match="num_shards must be positive"):
        Dataset.from_list(range(10)).reshard(0)


def test_complex_pipeline(backend):
    """Test a more complex data processing pipeline."""
    ds = (
        Dataset.from_list(range(1, 21))
        .filter(lambda x: x > 5)  # [6, 7, 8, ..., 20]
        .map(lambda x: {"value": x, "squared": x * x, "label": "even" if x % 2 == 0 else "odd"})
        .filter(lambda item: item["label"] == "even")  # Even items only
    )

    result = list(Backend.execute(ds, context=backend))
    assert len(result) == 8  # 6, 8, 10, 12, 14, 16, 18, 20
    assert all(item["label"] == "even" for item in result)
    assert all(item["value"] % 2 == 0 for item in result)


def test_operations_are_dataclasses():
    """Test that operations are stored as inspectable dataclasses."""
    ds = Dataset.from_list([1, 2, 3]).map(lambda x: x * 2).filter(lambda x: x > 2).window(2)

    # Should have 3 operations
    assert len(ds.operations) == 3

    # Check operation types
    assert isinstance(ds.operations[0], MapOp)
    assert isinstance(ds.operations[1], FilterOp)
    assert isinstance(ds.operations[2], WindowOp)
    # WindowOp has folder_fn and initial_state
    assert callable(ds.operations[2].folder_fn)
    assert ds.operations[2].initial_state == 0


def test_from_files_basic(tmp_path):
    """Test basic file globbing."""
    # Create test input files
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "file1.txt").write_text("data1")
    (input_dir / "file2.txt").write_text("data2")
    (input_dir / "file3.txt").write_text("data3")

    # Create dataset
    ds = Dataset.from_files(f"{input_dir}/*.txt")
    files = list(ds.source)  # Access source directly without backend execution

    assert len(files) == 3
    assert all(isinstance(f, str) for f in files)

    # Check that all input files are from input_dir
    assert all(str(input_dir) in f for f in files)
    assert all(f.endswith(".txt") for f in files)


def test_from_files_nested(tmp_path):
    """Test from_files with nested directories."""
    # Create nested structure
    input_dir = tmp_path / "input"
    (input_dir / "subdir1").mkdir(parents=True)
    (input_dir / "subdir2").mkdir(parents=True)

    (input_dir / "file1.txt").write_text("data1")
    (input_dir / "subdir1" / "file2.txt").write_text("data2")
    (input_dir / "subdir2" / "file3.txt").write_text("data3")

    # Use ** pattern to match nested files
    ds = Dataset.from_files(f"{input_dir}/**/*.txt")
    files = list(ds.source)

    assert len(files) == 3

    # Check that nested structure is in file paths
    assert any("subdir1" in path for path in files)
    assert any("subdir2" in path for path in files)


def test_from_files_empty_glob_ok(tmp_path):
    """Test from_files with empty_glob_ok=True."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    # No error when empty_glob_ok=True
    ds = Dataset.from_files(f"{input_dir}/*.txt", empty_glob_ok=True)
    files = list(ds.source)
    assert len(files) == 0


def test_from_files_empty_glob_error(tmp_path):
    """Test from_files raises error when no files match and empty_glob_ok=False."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    # Should raise FileNotFoundError
    with pytest.raises(FileNotFoundError, match="No files found"):
        Dataset.from_files(f"{input_dir}/*.txt", empty_glob_ok=False)


def test_from_files_with_map(tmp_path, backend):
    """Test from_files integrated with map and write operations."""
    # Create test input files
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    for i in range(3):
        (input_dir / f"file{i}.txt").write_text(f"data{i}")

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    def process_file(input_path):
        """Read file and return transformed record."""
        with open(input_path) as f:
            content = f.read()
        # Return a single record with uppercased content
        return {"content": content.upper(), "path": input_path}

    # Create dataset, process files, and write output
    ds = (
        Dataset.from_files(f"{input_dir}/*.txt")
        .map(process_file)
        .write_jsonl(str(output_dir / "output-{shard:05d}.jsonl"))
    )

    result = list(Backend.execute(ds, context=backend))

    # Verify output files were created
    assert len(result) == 3
    assert all(Path(p).exists() for p in result)
    assert all("output-" in p for p in result)


def test_write_and_read_parquet(tmp_path, backend):
    """Test writing and reading parquet files."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # Create dataset with structured data
    sample_data = [
        {"id": 1, "name": "Alice", "score": 95.5, "active": True},
        {"id": 2, "name": "Bob", "score": 87.3, "active": False},
        {"id": 3, "name": "Charlie", "score": 92.1, "active": True},
        {"id": 4, "name": "Diana", "score": 88.9, "active": True},
        {"id": 5, "name": "Eve", "score": 91.2, "active": False},
    ]

    # Write to parquet
    ds = Dataset.from_list(sample_data).write_parquet(str(output_dir / "data-{shard:05d}.parquet"))

    output_files = list(Backend.execute(ds, context=backend))

    # Verify output files were created
    assert len(output_files) > 0
    assert all(Path(p).exists() for p in output_files)
    assert all(p.endswith(".parquet") for p in output_files)

    # Read back from parquet
    ds_read = Dataset.from_files(f"{output_dir}/*.parquet").flat_map(load_parquet)

    records = list(Backend.execute(ds_read, context=backend))

    # Verify data integrity
    assert len(records) == len(sample_data)

    # Sort both lists by id for comparison
    records_sorted = sorted(records, key=lambda x: x["id"])
    sample_sorted = sorted(sample_data, key=lambda x: x["id"])

    for record, expected in zip(records_sorted, sample_sorted, strict=True):
        assert record["id"] == expected["id"]
        assert record["name"] == expected["name"]
        assert abs(record["score"] - expected["score"]) < 0.01
        assert record["active"] == expected["active"]


def test_write_and_read_parquet_nested(tmp_path, backend):
    """Test writing and reading parquet files with nested structures."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # Create dataset with nested data
    sample_data = [
        {"id": 1, "metadata": {"created": "2024-01-01", "version": 1}},
        {"id": 2, "metadata": {"created": "2024-01-02", "version": 2}},
        {"id": 3, "metadata": {"created": "2024-01-03", "version": 1}},
    ]

    # Write to parquet
    ds = Dataset.from_list(sample_data).write_parquet(str(output_dir / "nested-{shard:05d}.parquet"))

    output_files = list(Backend.execute(ds, context=backend))

    # Verify output files were created
    assert len(output_files) > 0
    assert all(Path(p).exists() for p in output_files)

    # Read back from parquet
    ds_read = Dataset.from_files(f"{output_dir}/*.parquet").flat_map(load_parquet)

    records = list(Backend.execute(ds_read, context=backend))

    # Verify data integrity
    assert len(records) == len(sample_data)

    # Sort both lists by id for comparison
    records_sorted = sorted(records, key=lambda x: x["id"])
    sample_sorted = sorted(sample_data, key=lambda x: x["id"])

    for record, expected in zip(records_sorted, sample_sorted, strict=True):
        assert record["id"] == expected["id"]
        assert record["metadata"]["created"] == expected["metadata"]["created"]
        assert record["metadata"]["version"] == expected["metadata"]["version"]


@pytest.mark.parametrize("output_format", ["jsonl", "parquet"])
def test_write_dataclass(tmp_path, backend, output_format: str):
    """Test writing and reading jsonl files with dataclass instances."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # Create dataset with dataclass instances
    sample_data = [
        SampleDataclass("alpha", 100),
        SampleDataclass("beta", 200),
        SampleDataclass("gamma", 300),
    ]

    if output_format == "jsonl":
        ds = Dataset.from_list(sample_data).write_jsonl(str(output_dir / "dataclass-{shard:05d}.jsonl"))
    else:
        ds = Dataset.from_list(sample_data).write_parquet(str(output_dir / "dataclass-{shard:05d}.parquet"))

    output_files = list(Backend.execute(ds, context=backend))

    # Verify output files were created
    assert len(output_files) > 0

    ds_read = Dataset.from_files(f"{output_dir}/*.{output_format}").flat_map(load_file)

    records = list(Backend.execute(ds_read, context=backend))

    # Verify data integrity
    assert len(records) == len(sample_data)

    # Sort both lists by name for comparison
    records_sorted = sorted(records, key=lambda x: x["name"])
    sample_sorted = sorted(sample_data, key=lambda x: x.name)

    for record, expected in zip(records_sorted, sample_sorted, strict=True):
        assert record["name"] == expected.name
        assert record["value"] == expected.value


def test_load_file_parquet(tmp_path, backend):
    """Test load_file with .parquet files."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # Create test Parquet file
    sample_data = [
        {"id": 1, "name": "Alice", "score": 95.5},
        {"id": 2, "name": "Bob", "score": 87.3},
        {"id": 3, "name": "Charlie", "score": 92.1},
    ]

    # Write to parquet with shard pattern
    ds = Dataset.from_list(sample_data).write_parquet(str(output_dir / "test-{shard:05d}.parquet"))
    _ = list(Backend.execute(ds, context=backend))

    # Load using load_file
    ds_read = Dataset.from_files(f"{output_dir}/*.parquet").flat_map(load_file)
    records = list(Backend.execute(ds_read, context=backend))

    # Verify data
    assert len(records) == len(sample_data)
    records_sorted = sorted(records, key=lambda x: x["id"])
    sample_sorted = sorted(sample_data, key=lambda x: x["id"])

    for record, expected in zip(records_sorted, sample_sorted, strict=True):
        assert record["id"] == expected["id"]
        assert record["name"] == expected["name"]
        assert abs(record["score"] - expected["score"]) < 0.01


def test_load_file_mixed_directory(tmp_path, backend):
    """Test load_file with mixed JSONL and Parquet files."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    # Create JSONL file
    jsonl_data = [
        {"id": 1, "source": "jsonl"},
        {"id": 2, "source": "jsonl"},
    ]

    jsonl_file = input_dir / "data.jsonl"
    with open(jsonl_file, "w") as f:
        for record in jsonl_data:
            f.write(json.dumps(record) + "\n")

    # Create Parquet file with shard pattern
    parquet_data = [
        {"id": 3, "source": "parquet"},
        {"id": 4, "source": "parquet"},
    ]

    ds = Dataset.from_list(parquet_data).write_parquet(str(input_dir / "data-{shard:05d}.parquet"))
    list(Backend.execute(ds, context=backend))

    # Load all files using load_file
    ds_read = Dataset.from_files(f"{input_dir}/*").flat_map(load_file)
    records = list(Backend.execute(ds_read, context=backend))

    # Verify we got data from both files
    assert len(records) == 4
    jsonl_records = [r for r in records if r["source"] == "jsonl"]
    parquet_records = [r for r in records if r["source"] == "parquet"]
    assert len(jsonl_records) == 2
    assert len(parquet_records) == 2


def test_load_file_unsupported_extension(tmp_path, backend):
    """Test load_file raises ValueError for unsupported file extensions."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    # Create file with unsupported extension
    txt_file = input_dir / "test.txt"
    txt_file.write_text("some text")

    # Try to load using load_file
    ds = Dataset.from_files(str(input_dir), "*.txt").flat_map(load_file)

    # Should raise ValueError during execution
    with pytest.raises(ValueError, match="Unsupported"):
        list(Backend.execute(ds, context=backend))


def test_write_without_shard_pattern_multiple_shards(tmp_path, backend):
    """Test that writing multiple shards without {shard} pattern raises ValueError."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # Create dataset with 3 items (will create multiple shards)
    sample_data = [{"id": 1}, {"id": 2}, {"id": 3}]

    # Try to write without {shard} in pattern
    ds = Dataset.from_list(sample_data).write_jsonl(str(output_dir / "output.jsonl"))

    # Should raise ValueError during execution when total > 1
    with pytest.raises(ValueError, match="Output pattern must"):
        list(Backend.execute(ds, context=backend))


def test_write_without_shard_pattern_single_shard(tmp_path, backend):
    """Test that writing single shard without {shard} pattern is allowed."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # Create dataset with 1 item (single shard)
    sample_data = [{"id": 1}]

    # Write without {shard} in pattern - should work for single shard
    ds = Dataset.from_list(sample_data).write_jsonl(str(output_dir / "output.jsonl"))

    # Should succeed since there's only one shard
    output_files = list(Backend.execute(ds, context=backend))
    assert len(output_files) == 1
    assert Path(output_files[0]).exists()


def test_reduce_sum(backend):
    """Test basic sum reduction."""
    ds = Dataset.from_list(range(100)).reduce(sum)
    results = list(Backend.execute(ds, context=backend))
    assert len(results) == 1
    assert results[0] == sum(range(100))


def test_reduce_count(backend):
    """Test count reduction."""
    ds = Dataset.from_list([{"id": i} for i in range(50)]).reduce(
        local_reducer=lambda items: sum(1 for _ in items),
        global_reducer=sum,
    )
    results = list(Backend.execute(ds, context=backend))
    assert len(results) == 1
    assert results[0] == 50


def test_reduce_complex_aggregation(backend):
    """Test custom aggregation with stats."""

    def local_stats(items):
        items_list = list(items)
        if not items_list:
            return {"sum": 0, "count": 0, "min": float("inf"), "max": float("-inf")}
        return {
            "sum": sum(items_list),
            "count": len(items_list),
            "min": min(items_list),
            "max": max(items_list),
        }

    def global_stats(shard_stats):
        stats_list = list(shard_stats)
        return {
            "sum": sum(s["sum"] for s in stats_list),
            "count": sum(s["count"] for s in stats_list),
            "min": min(s["min"] for s in stats_list),
            "max": max(s["max"] for s in stats_list),
        }

    ds = Dataset.from_list(range(1, 101)).reduce(
        local_reducer=local_stats,
        global_reducer=global_stats,
    )

    results = list(Backend.execute(ds, context=backend))
    assert len(results) == 1
    assert results[0]["sum"] == sum(range(1, 101))
    assert results[0]["count"] == 100
    assert results[0]["min"] == 1
    assert results[0]["max"] == 100


def test_reduce_empty(backend):
    """Test reduce on empty dataset."""
    ds = Dataset.from_list([]).reduce(sum)
    results = list(Backend.execute(ds, context=backend))
    assert len(results) == 0


def test_reduce_with_pipeline(backend):
    """Test reduce integrated with other operations."""
    ds = Dataset.from_list(range(1, 21)).filter(lambda x: x % 2 == 0).map(lambda x: x * 2).reduce(sum)

    results = list(Backend.execute(ds, context=backend))
    assert len(results) == 1
    expected = sum(x * 2 for x in range(1, 21) if x % 2 == 0)
    assert results[0] == expected


def test_count_basic(backend):
    """Test basic count operation."""
    ds = Dataset.from_list(range(100)).count()
    results = list(Backend.execute(ds, context=backend))
    assert len(results) == 1
    assert results[0] == 100


def test_count_empty(backend):
    """Test count on empty dataset."""
    ds = Dataset.from_list([]).count()
    results = list(Backend.execute(ds, context=backend))
    assert len(results) == 0


def test_count_with_filter(backend):
    """Test count with filter operation."""
    ds = Dataset.from_list(range(100)).filter(lambda x: x % 2 == 0).count()
    results = list(Backend.execute(ds, context=backend))
    assert len(results) == 1
    assert results[0] == 50


def test_sorted_merge_join_inner_basic(backend):
    """Test basic inner sorted merge join."""
    # Create pre-sorted, co-partitioned datasets via group_by
    left = Dataset.from_list(
        [{"id": 1, "text": "hello"}, {"id": 2, "text": "world"}, {"id": 3, "text": "foo"}]
    ).group_by(key=lambda x: x["id"], reducer=lambda k, items: next(iter(items)), num_output_shards=5)
    right = Dataset.from_list([{"id": 1, "score": 0.9}, {"id": 2, "score": 0.3}]).group_by(
        key=lambda x: x["id"], reducer=lambda k, items: next(iter(items)), num_output_shards=5
    )

    joined = left.sorted_merge_join(right, left_key=lambda x: x["id"], right_key=lambda x: x["id"], how="inner")

    results = sorted(list(Backend.execute(joined, context=backend)), key=lambda x: x["id"])
    assert len(results) == 2
    assert results[0] == {"id": 1, "text": "hello", "score": 0.9}
    assert results[1] == {"id": 2, "text": "world", "score": 0.3}


def test_sorted_merge_join_left(backend):
    """Test left sorted merge join with missing right items."""
    # Create pre-sorted, co-partitioned datasets
    left = Dataset.from_list([{"id": 1, "text": "hello"}, {"id": 2, "text": "world"}]).group_by(
        key=lambda x: x["id"], reducer=lambda k, items: next(iter(items)), num_output_shards=5
    )
    right = Dataset.from_list([{"id": 1, "score": 0.9}]).group_by(
        key=lambda x: x["id"], reducer=lambda k, items: next(iter(items)), num_output_shards=5
    )

    joined = left.sorted_merge_join(
        right,
        left_key=lambda x: x["id"],
        right_key=lambda x: x["id"],
        combiner=lambda left, right: {**left, "score": right["score"] if right else 0.0},
        how="left",
    )

    results = sorted(list(Backend.execute(joined, context=backend)), key=lambda x: x["id"])
    assert len(results) == 2
    assert results[0] == {"id": 1, "text": "hello", "score": 0.9}
    assert results[1] == {"id": 2, "text": "world", "score": 0.0}


def test_sorted_merge_join_duplicate_keys(backend):
    """Test sorted merge join with duplicate keys (cartesian product)."""
    # Create datasets with duplicate keys
    left = Dataset.from_list([{"id": 1, "text": "a"}, {"id": 1, "text": "b"}]).group_by(
        key=lambda x: x["id"], reducer=lambda k, items: list(items), num_output_shards=5
    )
    right = Dataset.from_list([{"id": 1, "score": 0.9}, {"id": 1, "score": 0.3}]).group_by(
        key=lambda x: x["id"], reducer=lambda k, items: list(items), num_output_shards=5
    )

    # Flatten the grouped items
    left = left.flat_map(lambda x: x)
    right = right.flat_map(lambda x: x)

    joined = left.sorted_merge_join(right, left_key=lambda x: x["id"], right_key=lambda x: x["id"], how="inner")

    results = sorted(list(Backend.execute(joined, context=backend)), key=lambda x: (x["text"], x["score"]))
    assert len(results) == 4
    assert results[0] == {"id": 1, "text": "a", "score": 0.3}
    assert results[1] == {"id": 1, "text": "a", "score": 0.9}
    assert results[2] == {"id": 1, "text": "b", "score": 0.3}
    assert results[3] == {"id": 1, "text": "b", "score": 0.9}


def test_sorted_merge_join_after_group_by(backend):
    """Test realistic pipeline: group_by followed by sorted_merge_join."""
    # Simulate a typical use case: group documents and attributes, then join
    docs = Dataset.from_list(
        [
            {"id": 1, "text": "hello", "version": 1},
            {"id": 1, "text": "hello updated", "version": 2},
            {"id": 2, "text": "world", "version": 1},
            {"id": 3, "text": "foo", "version": 1},
        ]
    ).group_by(
        key=lambda x: x["id"],
        reducer=lambda k, items: max(items, key=lambda x: x["version"]),  # Keep latest version
        num_output_shards=10,
    )

    attrs = Dataset.from_list(
        [
            {"id": 1, "quality": 0.9},
            {"id": 2, "quality": 0.3},
            {"id": 3, "quality": 0.8},
        ]
    ).group_by(key=lambda x: x["id"], reducer=lambda k, items: next(iter(items)), num_output_shards=10)

    joined = docs.sorted_merge_join(attrs, left_key=lambda x: x["id"], right_key=lambda x: x["id"], how="inner")

    results = sorted(list(Backend.execute(joined, context=backend)), key=lambda x: x["id"])
    assert len(results) == 3
    assert results[0] == {"id": 1, "text": "hello updated", "version": 2, "quality": 0.9}
    assert results[1] == {"id": 2, "text": "world", "version": 1, "quality": 0.3}
    assert results[2] == {"id": 3, "text": "foo", "version": 1, "quality": 0.8}


def test_sorted_merge_join_shard_mismatch(backend):
    """Test that shard count mismatch raises error."""
    # Create datasets with different shard counts
    left = Dataset.from_list([{"id": 1, "text": "hello"}]).group_by(
        key=lambda x: x["id"], reducer=lambda k, items: next(iter(items)), num_output_shards=5
    )

    right = Dataset.from_list([{"id": 1, "score": 0.9}]).group_by(
        key=lambda x: x["id"],
        reducer=lambda k, items: next(iter(items)),
        num_output_shards=10,  # Different shard count!
    )

    joined = left.sorted_merge_join(right, left_key=lambda x: x["id"], right_key=lambda x: x["id"], how="inner")

    with pytest.raises(ValueError, match="Sorted merge join requires equal shard counts"):
        list(Backend.execute(joined, context=backend))


def test_sorted_merge_join_empty_datasets(backend):
    """Test sorted merge join with empty datasets.

    Note: Empty datasets with group_by create 0 shards, so shard counts won't match.
    This test verifies that mismatched shard counts raise an error.
    """
    # Empty left dataset - will create 0 shards
    left = Dataset.from_list([]).group_by(
        key=lambda x: x["id"], reducer=lambda k, items: next(iter(items)), num_output_shards=5
    )
    right = Dataset.from_list([{"id": 1, "score": 0.9}]).group_by(
        key=lambda x: x["id"], reducer=lambda k, items: next(iter(items)), num_output_shards=5
    )

    joined = left.sorted_merge_join(right, left_key=lambda x: x["id"], right_key=lambda x: x["id"], how="inner")

    # Empty dataset creates 0 shards, non-empty creates N shards - this is a mismatch
    with pytest.raises(ValueError, match="Sorted merge join requires equal shard counts"):
        list(Backend.execute(joined, context=backend))

    # Empty right dataset - will create 0 shards
    left = Dataset.from_list([{"id": 1, "text": "hello"}]).group_by(
        key=lambda x: x["id"], reducer=lambda k, items: next(iter(items)), num_output_shards=5
    )
    right = Dataset.from_list([]).group_by(
        key=lambda x: x["id"], reducer=lambda k, items: next(iter(items)), num_output_shards=5
    )

    joined = left.sorted_merge_join(right, left_key=lambda x: x["id"], right_key=lambda x: x["id"], how="inner")

    # Empty dataset creates 0 shards, non-empty creates N shards - this is a mismatch
    with pytest.raises(ValueError, match="Sorted merge join requires equal shard counts"):
        list(Backend.execute(joined, context=backend))


def test_map_shard_stateful_deduplication(backend):
    """Test map_shard for stateful within-shard deduplication."""

    def deduplicate_shard(items):
        seen = set()
        for item in items:
            key = item["id"]
            if key not in seen:
                seen.add(key)
                yield item

    data = [
        {"id": 1, "val": "a"},
        {"id": 2, "val": "b"},
        {"id": 1, "val": "c"},  # Duplicate
        {"id": 3, "val": "d"},
        {"id": 2, "val": "e"},  # Duplicate
    ]

    ds = Dataset.from_list([data]).flat_map(lambda x: x).map_shard(deduplicate_shard)
    result = sorted(list(Backend.execute(ds, context=backend)), key=lambda x: x["id"])

    # Should keep first occurrence of each id
    assert len(result) == 3
    assert result[0] == {"id": 1, "val": "a"}
    assert result[1] == {"id": 2, "val": "b"}
    assert result[2] == {"id": 3, "val": "d"}


def test_map_shard_empty_result(backend):
    """Test map_shard that filters everything out."""

    def filter_all(items):
        for _ in items:
            pass  # Consume but don't yield
        return iter([])  # Return empty iterator

    ds = Dataset.from_list([list(range(1, 6))]).flat_map(lambda x: x).map_shard(filter_all)
    result = list(Backend.execute(ds, context=backend))
    assert result == []


def test_map_shard_error_propagation(backend):
    """Test that exceptions in map_shard functions propagate correctly."""

    def failing_generator(items):
        for item in items:
            if item == 3:
                raise ValueError("Test error")
            yield item

    ds = Dataset.from_list([list(range(1, 6))]).flat_map(lambda x: x).map_shard(failing_generator)

    with pytest.raises(ValueError, match="Test error"):
        list(Backend.execute(ds, context=backend))


@pytest.fixture
def sample_input_files(tmp_path):
    """Create standard sample input files for skip_existing tests."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    for i in range(3):
        with open(input_dir / f"input-{i}.jsonl", "w") as f:
            f.write(f'{{"id": {i}}}\n')
    return input_dir


def test_skip_existing_clean_run(tmp_path, sample_input_files):
    """Test skip_existing with no existing files - all shards process."""
    ctx = create_job_ctx("sync")
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    counter = CallCounter()
    ds = (
        Dataset.from_files(f"{sample_input_files}/*.jsonl")
        .flat_map(lambda x: counter.counting_flat_map(x))
        .map(lambda x: counter.counting_map(x))
        .write_jsonl(str(output_dir / "output-{shard:05d}.jsonl"), skip_existing=True)
    )

    result = list(Backend.execute(ds, context=ctx))
    assert len(result) == 3
    assert all(Path(p).exists() for p in result)
    assert counter.flat_map_count == 3  # All files loaded
    assert counter.map_count == 3  # All items mapped
    assert sorted(counter.processed_ids) == [0, 1, 2]  # All shards ran


def test_skip_existing_one_file_exists(tmp_path, sample_input_files):
    """Test skip_existing with one output file existing - only that shard skips."""
    ctx = create_job_ctx("sync")
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # Manually create one output file (shard 1)
    with open(output_dir / "output-00001.jsonl", "w") as f:
        f.write('{"id": 1, "processed": true}\n')

    counter = CallCounter()
    ds = (
        Dataset.from_files(f"{sample_input_files}/*.jsonl")
        .flat_map(lambda x: counter.counting_flat_map(x))
        .map(lambda x: counter.counting_map(x))
        .write_jsonl(str(output_dir / "output-{shard:05d}.jsonl"), skip_existing=True)
    )

    result = list(Backend.execute(ds, context=ctx))
    assert len(result) == 3
    assert all(Path(p).exists() for p in result)
    assert counter.flat_map_count == 2  # Only 2 files loaded (shard 1 skipped)
    assert counter.map_count == 2  # Only 2 items mapped
    assert sorted(counter.processed_ids) == [0, 2]  # Only shards 0 and 2 ran


def test_skip_existing_all_files_exist(tmp_path, sample_input_files):
    """Test skip_existing with all output files existing - all shards skip."""
    ctx = create_job_ctx("sync")
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    counter = CallCounter()
    ds = (
        Dataset.from_files(f"{sample_input_files}/*.jsonl")
        .flat_map(lambda x: counter.counting_flat_map(x))
        .map(lambda x: counter.counting_map(x))
        .write_jsonl(str(output_dir / "output-{shard:05d}.jsonl"), skip_existing=True)
    )

    # First run: create all output files
    result = list(Backend.execute(ds, context=ctx))
    assert len(result) == 3
    assert counter.flat_map_count == 3
    assert counter.map_count == 3
    assert sorted(counter.processed_ids) == [0, 1, 2]  # All shards ran

    # Second run: all files exist, nothing should process
    counter.reset()
    ds = (
        Dataset.from_files(f"{sample_input_files}/*.jsonl")
        .flat_map(counter.counting_flat_map)
        .map(counter.counting_map)
        .write_jsonl(str(output_dir / "output-{shard:05d}.jsonl"), skip_existing=True)
    )

    result = list(Backend.execute(ds, context=ctx))
    assert len(result) == 3
    assert counter.flat_map_count == 0  # Nothing loaded
    assert counter.map_count == 0  # Nothing mapped
    assert counter.processed_ids == []  # No shards ran


def test_repr_handles_partials():
    """__repr__ should unwrap functools.partial"""
    assert repr(MapOp(partial(int, base=2))) == "MapOp(fn=int)"

    def my_base(n: str, base: int = 10) -> int:
        return int(n, base)

    op = MapOp(partial(my_base, base=2))
    assert repr(op) == "MapOp(fn=test_repr_handles_partials.<locals>.my_base)"


def test_repr_handles_lambdas():
    """Ensure anonymous lambdas work correctly."""
    op = FilterOp(lambda x: x > 0)
    assert repr(op) == "FilterOp(predicate=test_repr_handles_lambdas.<locals>.<lambda>)"


def test_filter_with_expression(backend):
    """Test filter with expression on in-memory data."""
    from zephyr import col

    ds = Dataset.from_list(
        [
            {"name": "alice", "score": 80},
            {"name": "bob", "score": 60},
            {"name": "charlie", "score": 90},
        ]
    ).filter(col("score") > 70)

    results = list(Backend.execute(ds, context=backend))
    assert len(results) == 2
    assert all(r["score"] > 70 for r in results)


def test_filter_expression_equality(backend):
    """Test filter with equality expression."""
    from zephyr import col

    ds = Dataset.from_list(
        [
            {"category": "A", "value": 1},
            {"category": "B", "value": 2},
            {"category": "A", "value": 3},
        ]
    ).filter(col("category") == "A")

    results = list(Backend.execute(ds, context=backend))
    assert len(results) == 2
    assert all(r["category"] == "A" for r in results)


def test_filter_expression_logical_and(backend):
    """Test filter with logical AND expression."""
    from zephyr import col

    ds = Dataset.from_list(
        [
            {"a": 1, "b": 2},
            {"a": -1, "b": 3},
            {"a": 2, "b": -1},
            {"a": -1, "b": -1},
        ]
    ).filter((col("a") > 0) & (col("b") > 0))

    results = list(Backend.execute(ds, context=backend))
    assert len(results) == 1
    assert results[0] == {"a": 1, "b": 2}


def test_filter_expression_logical_or(backend):
    """Test filter with logical OR expression."""
    from zephyr import col

    ds = Dataset.from_list(
        [
            {"a": 1, "b": 2},
            {"a": -1, "b": 3},
            {"a": 2, "b": -1},
            {"a": -1, "b": -1},
        ]
    ).filter((col("a") > 0) | (col("b") > 0))

    results = list(Backend.execute(ds, context=backend))
    assert len(results) == 3


def test_filter_nested_field(backend):
    """Test filter with nested field access."""
    from zephyr import col

    ds = Dataset.from_list(
        [
            {"id": 1, "meta": {"score": 0.9}},
            {"id": 2, "meta": {"score": 0.3}},
            {"id": 3, "meta": {"score": 0.7}},
        ]
    ).filter(col("meta")["score"] > 0.5)

    results = list(Backend.execute(ds, context=backend))
    assert len(results) == 2
    assert all(r["meta"]["score"] > 0.5 for r in results)


def test_select_columns(backend):
    """Test column projection with select."""
    ds = Dataset.from_list(
        [
            {"id": 1, "name": "alice", "score": 80, "extra": "x"},
            {"id": 2, "name": "bob", "score": 60, "extra": "y"},
        ]
    ).select("id", "name")

    results = list(Backend.execute(ds, context=backend))
    assert len(results) == 2
    assert results[0] == {"id": 1, "name": "alice"}
    assert results[1] == {"id": 2, "name": "bob"}


def test_select_partial_columns(backend):
    """Test select with columns that don't exist in all records."""
    ds = Dataset.from_list(
        [
            {"id": 1, "name": "alice"},
            {"id": 2, "name": "bob", "score": 60},
        ]
    ).select("id", "score")

    results = list(Backend.execute(ds, context=backend))
    assert len(results) == 2
    assert results[0] == {"id": 1}
    assert results[1] == {"id": 2, "score": 60}


def test_filter_and_select_combined(backend):
    """Test combined filter and select."""
    from zephyr import col

    ds = (
        Dataset.from_list(
            [
                {"id": 1, "name": "alice", "score": 80},
                {"id": 2, "name": "bob", "score": 60},
                {"id": 3, "name": "charlie", "score": 90},
            ]
        )
        .filter(col("score") > 70)
        .select("id", "name")
    )

    results = list(Backend.execute(ds, context=backend))
    assert len(results) == 2
    assert all(set(r.keys()) == {"id", "name"} for r in results)


@pytest.mark.parametrize("output_format", ["parquet", "jsonl"])
def test_filter_and_select(tmp_path, backend, output_format: str):
    """Test combined filter and select on parquet."""
    from zephyr import col

    ds = Dataset.from_list(
        [
            {"id": 1, "name": "alice", "score": 80, "extra": "x"},
            {"id": 2, "name": "bob", "score": 60, "extra": "y"},
            {"id": 3, "name": "charlie", "score": 90, "extra": "z"},
        ]
    ).reshard(1)

    output_path = tmp_path / ("data." + output_format)

    if output_format == "parquet":
        ds = ds.write_parquet(str(output_path))
    elif output_format == "jsonl":
        ds = ds.write_jsonl(str(output_path))

    Backend.execute(ds, context=backend)

    ds = Dataset.from_files(str(output_path)).load_file().filter(col("score") > 70).select("id", "name")

    results = list(Backend.execute(ds, context=backend))
    assert len(results) == 2
    assert all(set(r.keys()) == {"id", "name"} for r in results)
    names = {r["name"] for r in results}
    assert names == {"alice", "charlie"}


def test_filter_expression_repr():
    """Test FilterOp repr with expression."""
    from zephyr import col
    from zephyr.dataset import FilterOp

    expr = col("score") > 50
    op = FilterOp(predicate=expr.evaluate, expr=expr)
    assert "FilterOp(expr=" in repr(op)
    assert "col('score')" in repr(op)


def test_mixed_filter_expression_and_lambda(backend):
    """Test combining expression filter with lambda filter."""
    from zephyr import col

    ds = (
        Dataset.from_list(
            [
                {"a": 1, "b": "x"},
                {"a": 2, "b": "y"},
                {"a": 3, "b": "x"},
                {"a": 4, "b": "y"},
            ]
        )
        .filter(col("a") > 1)
        .filter(lambda r: r["b"] == "x")
    )

    results = list(Backend.execute(ds, context=backend))
    assert len(results) == 1
    assert results[0] == {"a": 3, "b": "x"}


def test_input_file_spec_with_columns_and_row_range(tmp_path):
    """Test InputFileSpec with both columns and row_range."""
    from zephyr.readers import InputFileSpec, load_parquet

    data = [{"id": i, "name": f"item_{i}", "value": i * 10} for i in range(50)]
    input_path = tmp_path / "data.parquet"
    write_parquet_file(data, str(input_path))

    spec = InputFileSpec(
        path=str(input_path),
        format="parquet",
        columns=["id", "value"],
        row_start=5,
        row_end=10,
    )

    records = list(load_parquet(spec))

    assert len(records) == 5
    assert set(records[0].keys()) == {"id", "value"}
    assert records[0]["id"] == 5
    assert records[-1]["id"] == 9


def test_vortex_load_file_auto_detects_format(backend, tmp_path):
    """Test that load_file() auto-detects vortex format."""
    from zephyr.writers import write_vortex_file

    # Create input vortex file
    records = [{"id": i, "name": f"item_{i}"} for i in range(50)]
    input_path = tmp_path / "input.vortex"
    write_vortex_file(records, str(input_path))

    output_pattern = str(tmp_path / "output-{shard:05d}.jsonl.gz")

    ds = (
        Dataset.from_files(str(input_path))
        .load_file()  # Should auto-detect vortex
        .filter(lambda r: r["id"] < 10)
        .write_jsonl(output_pattern)
    )

    results = list(Backend.execute(ds, context=backend))
    assert len(results) == 1


def test_expression_filter_pushdown(backend, tmp_path):
    """Test filter pushdown with expression.

    Verifies that vortex format supports predicate pushdown,
    filtering at the I/O layer instead of in Python.
    """
    from zephyr.expr import col

    vortex_file = create_vortex_file(tmp_path)
    ds = Dataset.from_files(str(vortex_file)).load_vortex().filter(col("score") > 500)

    results = list(Backend.execute(ds, context=backend))
    assert len(results) == 49  # scores 510, 520, ..., 990
    assert all(r["score"] > 500 for r in results)


def test_column_select_pushdown(backend, tmp_path):
    """Test column selection pushdown.

    Verifies that vortex format supports projection pushdown,
    loading only requested columns.
    """
    vortex_file = create_vortex_file(tmp_path)
    ds = Dataset.from_files(str(vortex_file)).load_vortex().select("id", "score")

    results = list(Backend.execute(ds, context=backend))
    assert len(results) == 100
    assert set(results[0].keys()) == {"id", "score"}
