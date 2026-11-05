# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared Zephyr graph binding for Taskforge batch campaigns."""

from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass

from fray.types import ResourceConfig
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset

TaskforgeRow = Mapping[str, object]
ShardRunner = Callable[[Iterable[TaskforgeRow], object], Iterator[TaskforgeRow]]


@dataclass(frozen=True)
class TaskforgeZephyrConfig:
    """Resources and fan-out for one outer Taskforge lifecycle map."""

    name: str
    max_workers: int = 32
    logical_shards: int = 128
    region: str | None = "us-central1"
    preemptible: bool = True

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must be nonempty")
        if self.max_workers < 1 or self.logical_shards < 1:
            raise ValueError("worker and shard counts must be positive")


def taskforge_pipeline(
    rows: Iterable[TaskforgeRow],
    *,
    run_shard: ShardRunner,
    output: str,
    logical_shards: int,
) -> tuple[Dataset, int]:
    """Build the single-map lifecycle graph and return it with its effective shard count."""
    materialized = list(rows)
    if not materialized:
        raise ValueError("at least one task lifecycle row is required")
    if logical_shards < 1:
        raise ValueError("logical_shards must be positive")
    shards = min(logical_shards, len(materialized))
    dataset = Dataset.from_list(materialized)
    if shards > 1:
        dataset = dataset.reshard(shards)
    pipeline = dataset.map_shard(run_shard)
    if "{shard}" not in output:
        pipeline = pipeline.reshard(1)
    return pipeline.write_jsonl(output), shards


def run_taskforge_zephyr(
    rows: Iterable[TaskforgeRow],
    *,
    run_shard: ShardRunner,
    output: str,
    config: TaskforgeZephyrConfig,
) -> object:
    """Run one complete task lifecycle per row through a single outer ``map_shard``.

    A literal output path is collapsed to one shard before writing. An output containing
    ``{shard}`` retains the mapped shard fan-out.
    """
    pipeline, shards = taskforge_pipeline(
        rows,
        run_shard=run_shard,
        output=output,
        logical_shards=config.logical_shards,
    )
    regions = [config.region] if config.region else None
    with ZephyrContext(
        max_workers=min(config.max_workers, shards),
        resources=ResourceConfig(
            cpu=1,
            ram="4g",
            disk="16g",
            regions=regions,
            preemptible=config.preemptible,
        ),
        coordinator_resources=ResourceConfig(
            cpu=0.1,
            ram="2g",
            disk="8g",
            regions=regions,
            preemptible=config.preemptible,
        ),
        name=config.name,
        heartbeat_timeout=900,
    ) as context:
        return context.execute(pipeline, name=config.name)
