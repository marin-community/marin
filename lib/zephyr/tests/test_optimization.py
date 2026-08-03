# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for operation fusion optimization via compute_plan."""

import itertools

import pyarrow as pa
import pyarrow.parquet as pq
from zephyr.dataset import Dataset, FilterOp, MapOp, ReshardOp, TakePerShardOp
from zephyr.execution import ZephyrContext
from zephyr.expr import col
from zephyr.plan import Map, PhysicalStage, Reshard, StageType, compute_plan


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

    # Use a flat_map to create multiple items in a single shard
    ds = (
        Dataset.from_list([[1, 2, 3, 4, 5, 6]])
        .flat_map(lambda x: x)  # Unfold into individual items within the shard
        .map(lambda x: x * 2)  # [2, 4, 6, 8, 10, 12]
        .filter(lambda x: x > 4)  # [6, 8, 10, 12]
        .window(2)  # [[6, 8], [10, 12]]
    )

    ctx = ZephyrContext(name="test_fusion")
    result = ctx.execute(ds).results
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
    stage = PhysicalStage(operations=[Map(fn=lambda x: x) for _ in range(20)], stage_type=StageType.MAP_WORKER)
    name = stage.stage_name(max_length=20)
    assert len(name) <= 20
    assert name.endswith("...")


def test_lambda_filter_blocks_select_pushdown(tmp_path):
    """A lambda filter prevents SelectOp pushdown — otherwise the projection
    would drop columns the lambda reads, KeyError-ing the user code."""
    path = str(tmp_path / "data.parquet")
    pq.write_table(
        pa.Table.from_pylist([{"a": 1, "b": 10, "c": 100}, {"a": 2, "b": 20, "c": 200}]),
        path,
    )

    # Lambda reads column "c" but later select("a", "b") would drop it.
    ds = Dataset.from_files(path).load_parquet().filter(lambda r: r["c"] > 150).select("a", "b")
    results = ZephyrContext(name="test").execute(ds).results
    assert results == [{"a": 2, "b": 20}]

    # Sanity: an Expr filter (introspectable) does still allow select pushdown
    # because referenced columns are added back at read time.
    ds_expr = Dataset.from_files(path).load_parquet().filter(col("c") > 150).select("a", "b")
    assert ZephyrContext(name="test").execute(ds_expr).results == [{"a": 2, "b": 20}]


def test_parquet_splits_stay_paired_with_their_input_file(tmp_path):
    """approx_shard_bytes splits each parquet file into contiguous row spans, and
    every span stays attached to the file it was computed from — the planner reads
    the footers concurrently, so the spans must be re-paired in input order."""
    row_counts = [6, 24, 12, 40]
    paths = []
    for file_idx, row_count in enumerate(row_counts):
        path = str(tmp_path / f"data-{file_idx}.parquet")
        records = [{"id": file_idx * 100 + row} for row in range(row_count)]
        pq.write_table(pa.Table.from_pylist(records), path, row_group_size=2)
        paths.append(path)

    row_group_bytes = pq.ParquetFile(paths[0]).metadata.row_group(0).total_byte_size
    ds = Dataset.from_list(paths).load_parquet(approx_shard_bytes=row_group_bytes * 2 + 1)
    plan = compute_plan(ds)

    spans_by_path: dict[str, list[tuple[int, int]]] = {}
    for expected_shard_idx, item in enumerate(plan.source_items):
        assert item.shard_idx == expected_shard_idx
        spans_by_path.setdefault(item.data.path, []).append((item.data.row_start, item.data.row_end))

    assert list(spans_by_path) == paths
    for path, row_count in zip(paths, row_counts, strict=True):
        spans = spans_by_path[path]
        assert len(spans) > 1
        assert spans[0][0] == 0
        # The last span ends at this file's own row count.
        assert spans[-1][1] == row_count
        for (_, end), (start, _) in itertools.pairwise(spans):
            assert end == start
