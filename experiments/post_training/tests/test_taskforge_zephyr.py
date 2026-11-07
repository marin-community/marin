# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable, Iterator, Mapping

from zephyr.dataset import MapShardOp, ReshardOp, WriteOp

from experiments.post_training.taskforge_zephyr import taskforge_pipeline


def pass_rows(rows: Iterable[Mapping[str, object]], shard_info: object) -> Iterator[Mapping[str, object]]:
    del shard_info
    yield from rows


def test_literal_output_collapses_mapped_rows_to_one_write_shard():
    pipeline, shards = taskforge_pipeline(
        ({"id": value} for value in range(3)),
        run_shard=pass_rows,
        output="s3://bucket/results.jsonl",
        logical_shards=128,
    )

    assert shards == 3
    assert [type(operation) for operation in pipeline.operations] == [ReshardOp, MapShardOp, ReshardOp, WriteOp]
    assert pipeline.operations[0].num_shards == 3
    assert pipeline.operations[2].num_shards == 1


def test_sharded_output_keeps_map_fanout():
    pipeline, shards = taskforge_pipeline(
        ({"id": value} for value in range(3)),
        run_shard=pass_rows,
        output="s3://bucket/results-{shard}.jsonl",
        logical_shards=2,
    )

    assert shards == 2
    assert [type(operation) for operation in pipeline.operations] == [ReshardOp, MapShardOp, WriteOp]
