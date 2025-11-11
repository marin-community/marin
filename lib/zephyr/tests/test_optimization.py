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

from zephyr import Dataset, create_backend
from zephyr.dataset import FilterOp, FusedMapOp, MapOp, ReshardOp, TakeOp


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


def test_optimize_with_take():
    """Take should be fused with map and filter."""
    backend = create_backend("sync")
    operations = [
        MapOp(lambda x: x * 2),
        TakeOp(10),
        FilterOp(lambda x: x > 5),
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

    # Should have: FusedMapOp, ReshardOp, FusedMapOp
    assert len(optimized) == 3
    assert isinstance(optimized[0], FusedMapOp)
    assert isinstance(optimized[1], ReshardOp)
    assert isinstance(optimized[2], FusedMapOp)


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
