# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Global exact deduplication of normalized Datakit sources by record ID."""

import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from fray.types import ResourceConfig
from marin.datakit.normalize import NormalizedData
from pydantic import BaseModel
from rigging.filesystem import StoragePath, prefix_join, url_to_fs
from zephyr import counters
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext
from zephyr.writers import write_parquet_file

COUNTER_PREFIX = "global_exact_dedup"
_DUPLICATE_SCHEMA = pa.schema(
    [
        pa.field("id", pa.string(), nullable=False),
        pa.field("row_index", pa.int64(), nullable=False),
    ]
)


class GlobalExactDedupData(BaseModel):
    """Output sources and counters from global exact deduplication."""

    version: str = "v1"
    sources: dict[str, NormalizedData]
    counters: dict[str, int | float]


@dataclass(frozen=True)
class _SourceShard:
    source: str
    source_rank: int
    shard_index: int
    input_path: str
    output_path: str
    duplicate_ids_path: str


def _source_counter(source: str, name: str) -> str:
    return f"{COUNTER_PREFIX}/source/{source}/{name}"


def _input_shards(
    sources: dict[str, NormalizedData],
    output_path: str,
) -> tuple[list[_SourceShard], dict[str, NormalizedData]]:
    shards: list[_SourceShard] = []
    outputs: dict[str, NormalizedData] = {}

    for source_rank, source in enumerate(sorted(sources)):
        normalized = sources[source]
        input_root = StoragePath(normalized.main_output_dir)
        input_paths = sorted(
            StoragePath(f"{normalized.main_output_dir.rstrip('/')}/**/*.parquet").glob(),
            key=str,
        )
        if not input_paths:
            raise FileNotFoundError(f"No Parquet files found under {normalized.main_output_dir}")

        source_root = prefix_join(output_path, f"sources/{source_rank:05d}")
        main_output_dir = prefix_join(source_root, "outputs/main")
        outputs[source] = NormalizedData(
            main_output_dir=main_output_dir,
            dup_output_dir=prefix_join(source_root, "outputs/dups"),
            counters={},
        )

        output_names: set[str] = set()
        for input_path in input_paths:
            relative_path = input_path.relative_to(input_root)
            if relative_path in output_names:
                raise ValueError(f"Duplicate Parquet path {relative_path!r} for source {source!r}")
            output_names.add(relative_path)

            shard_index = len(shards)
            shards.append(
                _SourceShard(
                    source=source,
                    source_rank=source_rank,
                    shard_index=shard_index,
                    input_path=str(input_path),
                    output_path=prefix_join(main_output_dir, relative_path),
                    duplicate_ids_path=prefix_join(output_path, f"duplicate_ids/{shard_index:08d}.parquet"),
                )
            )

    return shards, outputs


def _read_record_ids(shard: _SourceShard) -> Iterator[dict[str, int | str]]:
    row_index = 0
    with StoragePath(shard.input_path).open("rb") as input_file:
        parquet = pq.ParquetFile(input_file)
        if "id" not in parquet.schema_arrow.names:
            raise ValueError(f"Parquet file has no id column: {shard.input_path}")
        for batch in parquet.iter_batches(columns=["id"]):
            for record_id in batch.column("id").to_pylist():
                if not isinstance(record_id, str):
                    raise ValueError(f"Record ID is not a string in {shard.input_path}: {record_id!r}")
                counters.pipeline.update_counter(f"{COUNTER_PREFIX}/records_in", 1)
                counters.pipeline.update_counter(_source_counter(shard.source, "records_in"), 1)
                yield {
                    "id": record_id,
                    "source": shard.source,
                    "source_rank": shard.source_rank,
                    "shard_index": shard.shard_index,
                    "row_index": row_index,
                }
                row_index += 1


def _select_duplicates(_record_id: str, records: Iterator[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    canonical = next(records)
    counters.pipeline.update_counter(f"{COUNTER_PREFIX}/records_out", 1)
    counters.pipeline.update_counter(_source_counter(canonical["source"], "records_out"), 1)

    for duplicate in records:
        counters.pipeline.update_counter(f"{COUNTER_PREFIX}/duplicate_records", 1)
        counters.pipeline.update_counter(_source_counter(duplicate["source"], "duplicate_records"), 1)
        yield duplicate


def _duplicate_id_writer(
    shard_paths: dict[int, str],
) -> Callable[[int, Iterator[dict[str, Any]]], dict[str, int | str]]:
    def write_duplicate_ids(shard_index: int, records: Iterator[dict[str, Any]]) -> dict[str, int | str]:
        return write_parquet_file(
            ({"id": record["id"], "row_index": record["row_index"]} for record in records),
            output_path=shard_paths[shard_index],
            schema=_DUPLICATE_SCHEMA,
        )

    return write_duplicate_ids


def _copy_file(source: str, destination: str) -> None:
    source_fs, source_path = url_to_fs(source)
    destination_fs, destination_path = url_to_fs(destination)
    if source_fs.protocol != destination_fs.protocol:
        raise ValueError(f"Cannot copy between filesystems: {source!r} to {destination!r}")
    destination_fs.makedirs(os.path.dirname(destination_path), exist_ok=True)
    destination_fs.copy(source_path, destination_path)


def _duplicate_rows(path: str) -> set[int]:
    if not StoragePath(path).exists():
        return set()
    with StoragePath(path).open("rb") as duplicate_file:
        return set(pq.read_table(duplicate_file, columns=["row_index"]).column("row_index").to_pylist())


def _filter_shard(shard: _SourceShard) -> dict[str, int | str]:
    duplicate_rows = _duplicate_rows(shard.duplicate_ids_path)
    with StoragePath(shard.input_path).open("rb") as input_file:
        parquet = pq.ParquetFile(input_file)
        records_in = parquet.metadata.num_rows
        schema = parquet.schema_arrow

        if not duplicate_rows:
            _copy_file(shard.input_path, shard.output_path)
            return {
                "source": shard.source,
                "path": shard.output_path,
                "records_in": records_in,
                "records_out": records_in,
            }

        def unique_records() -> Iterator[dict[str, Any]]:
            row_index = 0
            for batch in parquet.iter_batches():
                for record in batch.to_pylist():
                    if row_index not in duplicate_rows:
                        yield record
                    row_index += 1

        result = write_parquet_file(unique_records(), output_path=shard.output_path, schema=schema)

    return {
        "source": shard.source,
        "path": shard.output_path,
        "records_in": records_in,
        "records_out": result["count"],
    }


def global_exact_deduplicate(
    *,
    sources: dict[str, NormalizedData],
    output_path: str,
    worker_resources: ResourceConfig,
    max_workers: int,
) -> GlobalExactDedupData:
    """Keep one record for each record ID across all normalized sources.

    The lexicographically first source keeps a shared record ID. The function
    keeps each source schema and shard layout.
    """
    if not sources:
        raise ValueError("Global exact deduplication requires at least one source")

    shards, output_sources = _input_shards(sources, output_path)
    shard_paths = {shard.shard_index: shard.duplicate_ids_path for shard in shards}

    dedup_pipeline = (
        Dataset.from_list(shards)
        .flat_map(_read_record_ids)
        .group_by(
            key=lambda record: record["id"],
            reducer=_select_duplicates,
            sort_by=lambda record: (record["source_rank"], record["shard_index"], record["row_index"]),
        )
        .group_by(
            key=lambda record: record["shard_index"],
            reducer=_duplicate_id_writer(shard_paths),
            sort_by=lambda record: record["id"],
        )
    )
    dedup_context = ZephyrContext(
        name="datakit-global-exact-dedup",
        resources=worker_resources,
        max_workers=max_workers,
    )
    dedup_outcome = dedup_context.execute(dedup_pipeline)

    filter_context = ZephyrContext(
        name="datakit-global-exact-dedup-filter",
        resources=worker_resources,
        max_workers=max_workers,
    )
    filter_outcome = filter_context.execute(Dataset.from_list(shards).map(_filter_shard))

    stage_counters: dict[str, int | float] = {}
    for name, value in dedup_outcome.counters.items():
        if name.startswith(f"{COUNTER_PREFIX}/"):
            stage_counters[name] = stage_counters.get(name, 0) + value

    filter_results = filter_outcome.results
    records_in = sum(int(result["records_in"]) for result in filter_results)
    records_out = sum(int(result["records_out"]) for result in filter_results)
    if records_in != stage_counters.get(f"{COUNTER_PREFIX}/records_in", 0):
        raise ValueError(f"Input record count changed between deduplication passes: {records_in}")
    if records_out != stage_counters.get(f"{COUNTER_PREFIX}/records_out", 0):
        raise ValueError(f"Output record count changed between deduplication passes: {records_out}")

    counter_dict = dict(stage_counters)
    for source, normalized in output_sources.items():
        normalized.counters.update(
            {
                name: value
                for name, value in counter_dict.items()
                if name.startswith(f"{COUNTER_PREFIX}/source/{source}/")
            }
        )

    return GlobalExactDedupData(sources=output_sources, counters=counter_dict)
