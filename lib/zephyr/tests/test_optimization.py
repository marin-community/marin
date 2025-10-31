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

"""Tests for operation fusion optimization."""

import tempfile

from zephyr import Dataset, create_backend
from zephyr.dataset import BatchOp, FilterOp, FusedMapOp, MapOp, ReshardOp, WriteJsonlOp


def test_optimize_single_operation():
    """Single operation should not be wrapped in FusedMapOp."""
    backend = create_backend("sync")
    operations = [MapOp(lambda x: x * 2)]
    optimized = backend._optimize_operations(operations)

    assert len(optimized) == 1
    assert isinstance(optimized[0], MapOp)


def test_optimize_consecutive_maps():
    """Consecutive maps should be fused."""
    backend = create_backend("sync")
    operations = [
        MapOp(lambda x: x * 2),
        MapOp(lambda x: x + 1),
        MapOp(lambda x: x * 3),
    ]
    optimized = backend._optimize_operations(operations)

    assert len(optimized) == 1
    assert isinstance(optimized[0], FusedMapOp)
    assert len(optimized[0].operations) == 3


def test_optimize_map_filter_map():
    """Map, filter, map should be fused."""
    backend = create_backend("sync")
    operations = [
        MapOp(lambda x: x * 2),
        FilterOp(lambda x: x > 5),
        MapOp(lambda x: x + 1),
    ]
    optimized = backend._optimize_operations(operations)

    assert len(optimized) == 1
    assert isinstance(optimized[0], FusedMapOp)
    assert len(optimized[0].operations) == 3


def test_optimize_with_reshard_breaks_fusion():
    """Reshard operation should break fusion."""
    backend = create_backend("sync")
    operations = [
        MapOp(lambda x: x * 2),
        MapOp(lambda x: x + 1),
        ReshardOp(num_shards=10),
        MapOp(lambda x: x * 3),
    ]
    optimized = backend._optimize_operations(operations)

    # Should have: FusedMapOp, ReshardOp, MapOp
    assert len(optimized) == 3
    assert isinstance(optimized[0], FusedMapOp)
    assert isinstance(optimized[1], ReshardOp)
    assert isinstance(optimized[2], MapOp)


def test_optimize_write_in_fusion():
    """Write operation should be included in fusion."""
    backend = create_backend("sync")
    operations = [
        MapOp(lambda x: x * 2),
        MapOp(lambda x: x + 1),
        WriteJsonlOp("output.jsonl"),
    ]
    optimized = backend._optimize_operations(operations)

    # Should have: FusedMapOp containing all three operations
    assert len(optimized) == 1
    assert isinstance(optimized[0], FusedMapOp)
    assert len(optimized[0].operations) == 3


def test_optimize_batch_in_fusion():
    """Batch operation should be included in fusion."""
    backend = create_backend("sync")
    operations = [
        MapOp(lambda x: x * 2),
        FilterOp(lambda x: x > 5),
        BatchOp(batch_size=10),
    ]
    optimized = backend._optimize_operations(operations)

    # Should have: FusedMapOp containing all three operations
    assert len(optimized) == 1
    assert isinstance(optimized[0], FusedMapOp)
    assert len(optimized[0].operations) == 3


def test_fused_execution_correctness():
    """Test that fused execution produces correct results."""
    backend = create_backend("sync")
    ds = (
        Dataset.from_list([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        .map(lambda x: x * 2)  # [2, 4, 6, ..., 20]
        .filter(lambda x: x > 10)  # [12, 14, 16, 18, 20]
        .map(lambda x: x + 100)  # [112, 114, 116, 118, 120]
    )

    result = sorted(backend.execute(ds))
    assert result == [112, 114, 116, 118, 120]


def test_fused_execution_with_flat_map():
    """Test fusion with flat_map operations."""
    backend = create_backend("sync")
    ds = (
        Dataset.from_list([[1, 2], [3, 4], [5, 6]])
        .flat_map(lambda x: x)  # [1, 2, 3, 4, 5, 6]
        .map(lambda x: x * 2)  # [2, 4, 6, 8, 10, 12]
        .filter(lambda x: x % 4 == 0)  # [4, 8, 12]
    )

    result = sorted(backend.execute(ds))
    assert result == [4, 8, 12]


def test_empty_filter_in_fusion():
    """Test that fusion handles filters that eliminate all items."""
    backend = create_backend("sync")
    ds = Dataset.from_list([1, 2, 3]).map(lambda x: x * 2).filter(lambda x: x > 1000).map(lambda x: x + 1)

    result = list(backend.execute(ds))
    assert result == []


def test_fused_execution_with_batch():
    """Test fusion with batch operations.

    Note: Batching happens per-shard. Since each input item becomes its own shard,
    and filtering may reduce items per shard, batches may not span across shards.
    """
    backend = create_backend("sync")
    # Use a flat_map to create multiple items in a single shard
    ds = (
        Dataset.from_list([[1, 2, 3, 4, 5, 6]])
        .flat_map(lambda x: x)  # Unfold into individual items within the shard
        .map(lambda x: x * 2)  # [2, 4, 6, 8, 10, 12]
        .filter(lambda x: x > 4)  # [6, 8, 10, 12]
        .batch(2)  # [[6, 8], [10, 12]]
    )

    result = list(backend.execute(ds))
    assert result == [[6, 8], [10, 12]]


def test_fused_execution_with_write_jsonl():
    """Test fusion with write_jsonl operation.

    Note: Each input item becomes its own shard and writes its own file.
    Filtered-out items still create files (but empty).
    """
    backend = create_backend("sync")

    with tempfile.TemporaryDirectory() as tmpdir:
        output_pattern = f"{tmpdir}/output-{{shard:05d}}.jsonl"

        ds = (
            Dataset.from_list([{"value": 1}, {"value": 2}, {"value": 3}])
            .map(lambda x: {**x, "doubled": x["value"] * 2})
            .filter(lambda x: x["doubled"] > 2)
            .write_jsonl(output_pattern)
        )

        output_files = list(backend.execute(ds))

        # Each input item creates a shard, so we get 3 files
        assert len(output_files) == 3

        # Verify file contents
        import json

        non_empty_count = 0
        for output_file in output_files:
            with open(output_file) as f:
                content = f.read().strip()
                if content:
                    records = [json.loads(line) for line in content.split("\n")]
                    assert len(records) == 1
                    assert records[0]["doubled"] > 2
                    non_empty_count += 1

        # Only 2 out of 3 files should have content (value=1 was filtered out)
        assert non_empty_count == 2


def test_flat_map_with_generator():
    """Test that flat_map works with generator functions."""
    backend = create_backend("sync")

    def yield_range(x):
        yield from range(x)

    ds = Dataset.from_list([1, 2, 3]).flat_map(yield_range).map(lambda x: x * 10)

    result = sorted(backend.execute(ds))
    # from_list([1, 2, 3])
    # flat_map(yield_range) -> [0, 0, 1, 0, 1, 2]
    # map(x * 10) -> [0, 0, 10, 0, 10, 20]
    assert result == [0, 0, 0, 10, 10, 20]


def test_complex_pipeline_fusion():
    """Test complex pipeline with multiple operation types."""
    backend = create_backend("sync")

    ds = (
        Dataset.from_list([[1, 2, 3], [4, 5], [6, 7, 8, 9]])
        .flat_map(lambda x: x)  # [1, 2, 3, 4, 5, 6, 7, 8, 9]
        .map(lambda x: x * 2)  # [2, 4, 6, 8, 10, 12, 14, 16, 18]
        .filter(lambda x: x % 4 == 0)  # [4, 8, 12, 16]
        .map(lambda x: x + 1)  # [5, 9, 13, 17]
    )

    result = sorted(backend.execute(ds))
    assert result == [5, 9, 13, 17]


def test_no_fusion_with_all_reshards():
    """Test that ReshardOp always breaks fusion."""
    backend = create_backend("sync")
    operations = [
        ReshardOp(num_shards=5),
        ReshardOp(num_shards=10),
        ReshardOp(num_shards=3),
    ]
    optimized = backend._optimize_operations(operations)

    # All operations should remain separate
    assert len(optimized) == 3
    assert all(isinstance(op, ReshardOp) for op in optimized)


def test_empty_operations_list():
    """Test that empty operations list is handled."""
    backend = create_backend("sync")
    operations = []
    optimized = backend._optimize_operations(operations)

    assert optimized == []
