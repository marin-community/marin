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

import time
from pathlib import Path

import pytest
from zephyr import Dataset, create_backend


@pytest.fixture(autouse=True)
def ensure_ray(ray_cluster):
    """Ensure Ray is initialized for all tests."""
    pass


@pytest.fixture(
    params=[
        pytest.param(create_backend("sync"), id="sync"),
        pytest.param(create_backend("threadpool", max_parallelism=2), id="thread"),
        pytest.param(create_backend("ray", max_parallelism=2), id="ray"),
    ]
)
def backend(request):
    """Parametrized fixture providing all backend types."""
    return request.param


def test_from_list(sample_data, backend):
    """Test creating dataset from list."""
    ds = Dataset.from_list(sample_data)
    assert list(backend.execute(ds)) == sample_data


def test_from_iterable(backend):
    """Test creating dataset from iterable."""
    ds = Dataset.from_iterable(range(5))
    assert list(backend.execute(ds)) == [0, 1, 2, 3, 4]


def test_filter(sample_data, backend):
    """Test filtering dataset."""
    ds = Dataset.from_list(sample_data).filter(lambda x: x % 2 == 0)
    assert list(backend.execute(ds)) == [2, 4, 6, 8, 10]


def test_batch(backend):
    """Test batching dataset."""
    ds = Dataset.from_list([[1, 2, 3, 4, 5]]).flat_map(lambda x: x).batch(2)
    batches = list(backend.execute(ds))
    assert batches == [[1, 2], [3, 4], [5]]


def test_batch_exact_size(backend):
    """Test batching when size divides evenly."""
    ds = Dataset.from_list([[1, 2, 3, 4, 5, 6]]).flat_map(lambda x: x).batch(3)
    batches = list(backend.execute(ds))
    assert batches == [[1, 2, 3], [4, 5, 6]]


def double(x):
    """Simple function for map tests."""
    return x * 2


def slow_double(x):
    """Slow function to test parallelism."""
    time.sleep(0.1)
    return x * 2


def test_map(backend):
    """Test map operation with all backends."""
    ds = Dataset.from_list([1, 2, 3, 4, 5]).map(lambda x: x * 2)
    result = list(backend.execute(ds))
    assert sorted(result) == [2, 4, 6, 8, 10]


def test_bounded_parallelism_ray():
    """Test that Ray backend respects max_parallelism."""

    def slow_fn(x):
        time.sleep(0.2)
        return x * 2

    # With max_parallelism=2, processing 6 items should take ~3*0.2 = 0.6s
    backend = create_backend("ray", max_parallelism=2)
    ds = Dataset.from_list([1, 2, 3, 4, 5, 6]).map(slow_fn)

    start = time.time()
    result = list(backend.execute(ds))
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
        .batch(2)  # [[4, 8], [12, 16], [20]]
    )

    result = list(backend.execute(ds))

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
    backend = create_backend("sync")
    result = list(backend.execute(ds))
    assert call_count == 3
    assert result == [2, 4, 6]


def test_empty_dataset(backend):
    """Test operations on empty dataset."""
    ds = Dataset.from_list([])
    assert list(backend.execute(ds)) == []
    assert list(backend.execute(ds.filter(lambda x: True))) == []
    assert list(backend.execute(ds.map(lambda x: x * 2))) == []
    assert list(backend.execute(ds.batch(10))) == []


def test_reshard(backend):
    """Test reshard operation redistributes data across target number of shards."""
    # Test 1: Start with 1 item (1 shard), expand to many items, then reshard
    ds = (
        Dataset.from_list([list(range(50))])  # 1 shard with 50 records
        .flat_map(lambda x: x)  # Expand to 50 items
        .reshard(5)  # Redistribute to 5 shards
        .map(lambda x: x * 2)
    )
    result = sorted(backend.execute(ds))
    assert result == [x * 2 for x in range(50)]

    # Test 2: Start with many items, reshard to fewer
    ds = Dataset.from_list(range(50)).reshard(5).map(lambda x: x + 100)  # 50 shards  # Consolidate to 5 shards
    result = sorted(backend.execute(ds))
    assert result == [x + 100 for x in range(50)]

    # Test 3: Reshard preserves order when materializing all shards
    ds = Dataset.from_list(range(10)).reshard(3)
    result = list(backend.execute(ds))
    assert sorted(result) == list(range(10))


def process_item(x):
    """Simulate some processing."""
    return {"value": x, "squared": x * x, "label": "even" if x % 2 == 0 else "odd"}


def test_complex_pipeline(backend):
    """Test a more complex data processing pipeline."""
    ds = (
        Dataset.from_list(range(1, 21))
        .filter(lambda x: x > 5)  # [6, 7, 8, ..., 20]
        .map(lambda x: {"value": x, "squared": x * x, "label": "even" if x % 2 == 0 else "odd"})
        .filter(lambda item: item["label"] == "even")  # Even items only
    )

    result = list(backend.execute(ds))
    assert len(result) == 8  # 6, 8, 10, 12, 14, 16, 18, 20
    assert all(item["label"] == "even" for item in result)
    assert all(item["value"] % 2 == 0 for item in result)


def test_operations_are_dataclasses():
    """Test that operations are stored as inspectable dataclasses."""
    ds = Dataset.from_list([1, 2, 3]).map(lambda x: x * 2).filter(lambda x: x > 2).batch(2)

    # Should have 3 operations
    assert len(ds.operations) == 3

    # Check operation types
    from zephyr.dataset import BatchOp, FilterOp, MapOp

    assert isinstance(ds.operations[0], MapOp)
    assert isinstance(ds.operations[1], FilterOp)
    assert isinstance(ds.operations[2], BatchOp)
    assert ds.operations[2].batch_size == 2


def test_from_files_basic(tmp_path):
    """Test basic file globbing."""
    # Create test input files
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "file1.txt").write_text("data1")
    (input_dir / "file2.txt").write_text("data2")
    (input_dir / "file3.txt").write_text("data3")

    # Create dataset
    ds = Dataset.from_files(str(input_dir), "*.txt")
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
    ds = Dataset.from_files(str(input_dir), "**/*.txt")
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
    ds = Dataset.from_files(str(input_dir), "*.txt", empty_glob_ok=True)
    files = list(ds.source)
    assert len(files) == 0


def test_from_files_empty_glob_error(tmp_path):
    """Test from_files raises error when no files match and empty_glob_ok=False."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    # Should raise FileNotFoundError
    with pytest.raises(FileNotFoundError, match="No files found"):
        Dataset.from_files(str(input_dir), "*.txt", empty_glob_ok=False)


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
        Dataset.from_files(str(input_dir), "*.txt")
        .map(process_file)
        .write_jsonl(str(output_dir / "output-{shard:05d}.jsonl"))
    )

    result = list(backend.execute(ds))

    # Verify output files were created
    assert len(result) == 3
    assert all(Path(p).exists() for p in result)
    assert all("output-" in p for p in result)


def test_write_and_read_parquet(tmp_path, backend):
    """Test writing and reading parquet files."""
    from zephyr import load_parquet

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

    output_files = list(backend.execute(ds))

    # Verify output files were created
    assert len(output_files) > 0
    assert all(Path(p).exists() for p in output_files)
    assert all(p.endswith(".parquet") for p in output_files)

    # Read back from parquet
    ds_read = Dataset.from_files(str(output_dir), "*.parquet").flat_map(load_parquet)

    records = list(backend.execute(ds_read))

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
    from zephyr import load_parquet

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

    output_files = list(backend.execute(ds))

    # Verify output files were created
    assert len(output_files) > 0
    assert all(Path(p).exists() for p in output_files)

    # Read back from parquet
    ds_read = Dataset.from_files(str(output_dir), "*.parquet").flat_map(load_parquet)

    records = list(backend.execute(ds_read))

    # Verify data integrity
    assert len(records) == len(sample_data)

    # Sort both lists by id for comparison
    records_sorted = sorted(records, key=lambda x: x["id"])
    sample_sorted = sorted(sample_data, key=lambda x: x["id"])

    for record, expected in zip(records_sorted, sample_sorted, strict=True):
        assert record["id"] == expected["id"]
        assert record["metadata"]["created"] == expected["metadata"]["created"]
        assert record["metadata"]["version"] == expected["metadata"]["version"]


def test_load_file_parquet(tmp_path, backend):
    """Test load_file with .parquet files."""
    from zephyr import load_file

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
    _ = list(backend.execute(ds))

    # Load using load_file
    ds_read = Dataset.from_files(str(output_dir), "*.parquet").flat_map(load_file)
    records = list(backend.execute(ds_read))

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
    from zephyr import load_file

    input_dir = tmp_path / "input"
    input_dir.mkdir()

    # Create JSONL file
    import json

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
    list(backend.execute(ds))

    # Load all files using load_file
    ds_read = Dataset.from_files(str(input_dir), "*").flat_map(load_file)
    records = list(backend.execute(ds_read))

    # Verify we got data from both files
    assert len(records) == 4
    jsonl_records = [r for r in records if r["source"] == "jsonl"]
    parquet_records = [r for r in records if r["source"] == "parquet"]
    assert len(jsonl_records) == 2
    assert len(parquet_records) == 2


def test_load_file_unsupported_extension(tmp_path, backend):
    """Test load_file raises ValueError for unsupported file extensions."""
    from zephyr import load_file

    input_dir = tmp_path / "input"
    input_dir.mkdir()

    # Create file with unsupported extension
    txt_file = input_dir / "test.txt"
    txt_file.write_text("some text")

    # Try to load using load_file
    ds = Dataset.from_files(str(input_dir), "*.txt").flat_map(load_file)

    # Should raise ValueError during execution
    with pytest.raises(ValueError, match="Unsupported"):
        list(backend.execute(ds))


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
        list(backend.execute(ds))


def test_write_without_shard_pattern_single_shard(tmp_path, backend):
    """Test that writing single shard without {shard} pattern is allowed."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # Create dataset with 1 item (single shard)
    sample_data = [{"id": 1}]

    # Write without {shard} in pattern - should work for single shard
    ds = Dataset.from_list(sample_data).write_jsonl(str(output_dir / "output.jsonl"))

    # Should succeed since there's only one shard
    output_files = list(backend.execute(ds))
    assert len(output_files) == 1
    assert Path(output_files[0]).exists()


def test_reduce_sum(backend):
    """Test basic sum reduction."""
    ds = Dataset.from_list(range(100)).reduce(sum)
    results = list(backend.execute(ds))
    assert len(results) == 1
    assert results[0] == sum(range(100))


def test_reduce_count(backend):
    """Test count reduction."""
    ds = Dataset.from_list([{"id": i} for i in range(50)]).reduce(
        local_reducer=lambda items: sum(1 for _ in items),
        global_reducer=sum,
    )
    results = list(backend.execute(ds))
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

    results = list(backend.execute(ds))
    assert len(results) == 1
    assert results[0]["sum"] == sum(range(1, 101))
    assert results[0]["count"] == 100
    assert results[0]["min"] == 1
    assert results[0]["max"] == 100


def test_reduce_empty(backend):
    """Test reduce on empty dataset."""
    ds = Dataset.from_list([]).reduce(sum)
    results = list(backend.execute(ds))
    assert len(results) == 0


def test_reduce_with_pipeline(backend):
    """Test reduce integrated with other operations."""
    ds = Dataset.from_list(range(1, 21)).filter(lambda x: x % 2 == 0).map(lambda x: x * 2).reduce(sum)

    results = list(backend.execute(ds))
    assert len(results) == 1
    expected = sum(x * 2 for x in range(1, 21) if x % 2 == 0)
    assert results[0] == expected


def test_join_inner_basic(backend):
    """Test basic inner join."""
    left = Dataset.from_list([{"id": 1, "text": "hello"}, {"id": 2, "text": "world"}, {"id": 3, "text": "foo"}])
    right = Dataset.from_list(
        [
            {"id": 1, "score": 0.9},
            {"id": 2, "score": 0.3},
        ]
    )

    joined = left.join(right, left_key=lambda x: x["id"], right_key=lambda x: x["id"], how="inner")

    results = sorted(list(backend.execute(joined)), key=lambda x: x["id"])
    assert len(results) == 2
    assert results[0] == {"id": 1, "text": "hello", "score": 0.9}
    assert results[1] == {"id": 2, "text": "world", "score": 0.3}


def test_join_left(backend):
    """Test left join."""
    left = Dataset.from_list(
        [
            {"id": 1, "text": "hello"},
            {"id": 2, "text": "world"},
        ]
    )
    right = Dataset.from_list(
        [
            {"id": 1, "score": 0.9},
        ]
    )

    joined = left.join(
        right,
        left_key=lambda x: x["id"],
        right_key=lambda x: x["id"],
        combiner=lambda left, right: {**left, "score": right["score"] if right else 0.0},
        how="left",
    )

    results = sorted(list(backend.execute(joined)), key=lambda x: x["id"])
    assert len(results) == 2
    assert results[0] == {"id": 1, "text": "hello", "score": 0.9}
    assert results[1] == {"id": 2, "text": "world", "score": 0.0}


def test_join_right(backend):
    """Test right join."""
    left = Dataset.from_list(
        [
            {"id": 1, "text": "hello"},
        ]
    )
    right = Dataset.from_list(
        [
            {"id": 1, "score": 0.9},
            {"id": 2, "score": 0.3},
        ]
    )

    joined = left.join(
        right,
        left_key=lambda x: x["id"],
        right_key=lambda x: x["id"],
        combiner=lambda left, right: {**right, "text": left["text"] if left else ""},
        how="right",
    )

    results = sorted(list(backend.execute(joined)), key=lambda x: x["id"])
    assert len(results) == 2
    assert results[0] == {"id": 1, "score": 0.9, "text": "hello"}
    assert results[1] == {"id": 2, "score": 0.3, "text": ""}


def test_join_outer(backend):
    """Test outer join."""
    left = Dataset.from_list(
        [
            {"id": 1, "text": "hello"},
            {"id": 3, "text": "baz"},
        ]
    )
    right = Dataset.from_list(
        [
            {"id": 1, "score": 0.9},
            {"id": 2, "score": 0.3},
        ]
    )

    def combiner(left, right):
        result = {}
        if left:
            result.update(left)
        if right:
            result.update(right)
        if not left:
            result["text"] = ""
        if not right:
            result["score"] = 0.0
        return result

    joined = left.join(right, left_key=lambda x: x["id"], right_key=lambda x: x["id"], combiner=combiner, how="outer")

    results = sorted(list(backend.execute(joined)), key=lambda x: x["id"])
    assert len(results) == 3
    assert results[0] == {"id": 1, "text": "hello", "score": 0.9}
    assert results[1] == {"id": 2, "score": 0.3, "text": ""}
    assert results[2] == {"id": 3, "text": "baz", "score": 0.0}


def test_join_empty_left(backend):
    """Test join with empty left dataset."""
    left = Dataset.from_list([])
    right = Dataset.from_list([{"id": 1, "score": 0.9}])

    joined = left.join(right, left_key=lambda x: x["id"], right_key=lambda x: x["id"], how="inner")

    results = list(backend.execute(joined))
    assert len(results) == 0


def test_join_empty_right(backend):
    """Test join with empty right dataset."""
    left = Dataset.from_list([{"id": 1, "text": "hello"}])
    right = Dataset.from_list([])

    joined = left.join(right, left_key=lambda x: x["id"], right_key=lambda x: x["id"], how="inner")

    results = list(backend.execute(joined))
    assert len(results) == 0


def test_join_duplicate_keys(backend):
    """Test join with duplicate keys (cartesian product)."""
    left = Dataset.from_list(
        [
            {"id": 1, "text": "a"},
            {"id": 1, "text": "b"},
        ]
    )
    right = Dataset.from_list(
        [
            {"id": 1, "score": 0.9},
            {"id": 1, "score": 0.3},
        ]
    )

    joined = left.join(right, left_key=lambda x: x["id"], right_key=lambda x: x["id"], how="inner")

    results = sorted(list(backend.execute(joined)), key=lambda x: (x["text"], x["score"]))
    assert len(results) == 4
    assert results[0] == {"id": 1, "text": "a", "score": 0.3}
    assert results[1] == {"id": 1, "text": "a", "score": 0.9}
    assert results[2] == {"id": 1, "text": "b", "score": 0.3}
    assert results[3] == {"id": 1, "text": "b", "score": 0.9}


def test_join_with_filter(backend):
    """Test join combined with filter (consolidate use case)."""
    docs = Dataset.from_list(
        [
            {"id": 1, "text": "hello"},
            {"id": 2, "text": "world"},
            {"id": 3, "text": "foo"},
        ]
    )

    attrs = Dataset.from_list(
        [
            {"id": 1, "quality": 0.9},
            {"id": 2, "quality": 0.3},
            {"id": 3, "quality": 0.8},
        ]
    )

    filtered_attrs = attrs.filter(lambda x: x["quality"] > 0.5)

    result = docs.join(filtered_attrs, left_key=lambda x: x["id"], right_key=lambda x: x["id"], how="inner")

    results = sorted(list(backend.execute(result)), key=lambda x: x["id"])
    assert len(results) == 2
    assert results[0]["id"] == 1
    assert results[1]["id"] == 3
