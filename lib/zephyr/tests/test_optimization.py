# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for operation fusion optimization via compute_plan."""

from zephyr import Dataset, compute_plan
from zephyr.dataset import FilterOp, MapOp, ReshardOp, TakePerShardOp
from zephyr.plan import Map, PhysicalStage, Reshard


def test_optimize_consecutive_maps():
    """Consecutive maps should be fused into a single stage."""
    ds = Dataset(
        source=[1, 2, 3],
        operations=[
            MapOp(lambda x: x * 2),
            MapOp(lambda x: x + 1),
            MapOp(lambda x: x * 3),
        ],
    )
    plan = compute_plan(ds)

    assert len(plan.stages) == 1
    assert len(plan.stages[0].operations) == 1
    fused_op = plan.stages[0].operations[0]
    assert isinstance(fused_op, Map)


def test_optimize_map_filter_map():
    """Map, filter, map should be fused into a single stage."""
    ds = Dataset(
        source=[1, 2, 3],
        operations=[
            MapOp(lambda x: x * 2),
            FilterOp(lambda x: x > 5),
            MapOp(lambda x: x + 1),
        ],
    )
    plan = compute_plan(ds)

    assert len(plan.stages) == 1
    fused_op = plan.stages[0].operations[0]
    assert isinstance(fused_op, Map)


def test_optimize_with_take():
    """Take should be fused with map and filter."""
    ds = Dataset(
        source=[1, 2, 3],
        operations=[
            MapOp(lambda x: x * 2),
            TakePerShardOp(10),
            FilterOp(lambda x: x > 5),
        ],
    )
    plan = compute_plan(ds)

    assert len(plan.stages) == 1
    fused_op = plan.stages[0].operations[0]
    assert isinstance(fused_op, Map)


def test_optimize_with_reshard_breaks_fusion():
    """Reshard operation should break fusion into separate stages."""
    ds = Dataset(
        source=[1, 2, 3],
        operations=[
            MapOp(lambda x: x * 2),
            MapOp(lambda x: x + 1),
            ReshardOp(num_shards=10),
            MapOp(lambda x: x * 3),
        ],
    )
    plan = compute_plan(ds)

    # Should have: Map stage, Reshard stage, Map stage
    assert len(plan.stages) == 3
    assert isinstance(plan.stages[0].operations[0], Map)
    assert isinstance(plan.stages[1].operations[0], Reshard)
    assert isinstance(plan.stages[2].operations[0], Map)


def test_fused_execution_with_batch():
    """Test fusion with batch operations.

    Note: Batching happens per-shard. Since each input item becomes its own shard,
    and filtering may reduce items per shard, batches may not span across shards.
    """
    from zephyr.execution import ZephyrContext

    # Use a flat_map to create multiple items in a single shard
    ds = (
        Dataset.from_list([[1, 2, 3, 4, 5, 6]])
        .flat_map(lambda x: x)  # Unfold into individual items within the shard
        .map(lambda x: x * 2)  # [2, 4, 6, 8, 10, 12]
        .filter(lambda x: x > 4)  # [6, 8, 10, 12]
        .window(2)  # [[6, 8], [10, 12]]
    )

    with ZephyrContext(name="test_fusion") as ctx:
        result = list(ctx.execute(ds))
    assert result == [[6, 8], [10, 12]]


def test_stage_name():
    """PhysicalStage.stage_name() generates descriptive names from operations."""
    ds = Dataset(
        source=[1, 2, 3],
        operations=[
            MapOp(lambda x: x * 2),
            FilterOp(lambda x: x > 5),
        ],
    )
    plan = compute_plan(ds)

    assert len(plan.stages) == 1
    assert plan.stages[0].stage_name() == "Map"


def test_stage_name_truncation():
    """PhysicalStage.stage_name() truncates long names."""
    stage = PhysicalStage(operations=[Map(fn=lambda x: x) for _ in range(20)])
    name = stage.stage_name(max_length=20)
    assert len(name) <= 20
    assert name.endswith("...")
