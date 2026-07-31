# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Global exact deduplication of normalized Datakit sources by record ID.

The global shuffle carries record IDs and row positions only. A second pass
uses the row positions to filter each source shard without changing its schema.
The two passes read rows in Parquet order, so their row positions are equal.

``Dataset.deduplicate`` cannot retain the source and shard position that the
second pass needs.
"""

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from fray.types import ResourceConfig
from marin.datakit.normalize import NormalizedData
from pydantic import BaseModel
from rigging.filesystem import StoragePath, atomic_rename, prefix_join, url_to_fs
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext
from zephyr.writers import parquet_sink, write_parquet_file

COUNTER_PREFIX = "global_exact_dedup"
_DEDUP_SUCCESS_FILE = "_DEDUP_SUCCESS"
_FILTER_BATCH_ROWS = 256
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
    shard_index: int
    input_path: str
    output_path: str
    duplicate_ids_path: str


def global_exact_dedup_source_path(output_path: str, source_rank: int) -> str:
    """Get the output root for one source rank."""
    return prefix_join(output_path, f"sources/{source_rank:05d}")


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

        source_root = global_exact_dedup_source_path(output_path, source_rank)
        main_output_dir = prefix_join(source_root, "outputs/main")
        outputs[source] = NormalizedData(
            main_output_dir=main_output_dir,
            dup_output_dir=normalized.dup_output_dir,
            counters={},
        )

        for input_path in input_paths:
            relative_path = input_path.relative_to(input_root)
            shard_index = len(shards)
            shards.append(
                _SourceShard(
                    source=source,
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
        id_type = parquet.schema_arrow.field("id").type
        if not (pa.types.is_string(id_type) or pa.types.is_large_string(id_type)):
            raise ValueError(f"Record ID column is not a string in {shard.input_path}: {id_type}")
        for batch in parquet.iter_batches(columns=["id"]):
            for record_id in batch.column("id").to_pylist():
                yield {
                    "id": record_id,
                    "shard_index": shard.shard_index,
                    "row_index": row_index,
                }
                row_index += 1


def _select_duplicates(_record_id: str, records: Iterator[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    next(records)
    yield from records


def _duplicate_id_writer(
    output_path: str,
) -> Callable[[int, Iterator[dict[str, Any]]], dict[str, int | str]]:
    def write_duplicate_ids(shard_index: int, records: Iterator[dict[str, Any]]) -> dict[str, int | str]:
        return write_parquet_file(
            ({"id": record["id"], "row_index": record["row_index"]} for record in records),
            output_path=prefix_join(output_path, f"duplicate_ids/{shard_index:08d}.parquet"),
            schema=_DUPLICATE_SCHEMA,
        )

    return write_duplicate_ids


def _copy_file(source: str, destination: str) -> None:
    source_storage = StoragePath(source)
    destination_storage = StoragePath(destination)
    if (source_storage.scheme, source_storage.netloc) != (destination_storage.scheme, destination_storage.netloc):
        raise ValueError(f"Cannot copy between storage roots: {source!r} to {destination!r}")

    source_fs, source_path = url_to_fs(source)
    destination_storage.parent.mkdirs()
    with atomic_rename(destination) as temp_path:
        temp_fs, resolved_temp_path = url_to_fs(temp_path)
        if source_fs.protocol != temp_fs.protocol:
            raise ValueError(f"Cannot copy between filesystems: {source!r} to {destination!r}")
        temp_fs.copy(source_path, resolved_temp_path)


def _duplicate_rows(path: str) -> set[int]:
    if not StoragePath(path).exists():
        return set()
    with StoragePath(path).open("rb") as duplicate_file:
        return set(pq.read_table(duplicate_file, columns=["row_index"]).column("row_index").to_pylist())


def _filter_shard(shard: _SourceShard) -> dict[str, int | str]:
    duplicate_rows = _duplicate_rows(shard.duplicate_ids_path)
    if StoragePath(shard.output_path).exists():
        with StoragePath(shard.input_path).open("rb") as input_file:
            records_in = pq.ParquetFile(input_file).metadata.num_rows
        with StoragePath(shard.output_path).open("rb") as output_file:
            records_out = pq.ParquetFile(output_file).metadata.num_rows
        return {
            "source": shard.source,
            "path": shard.output_path,
            "records_in": records_in,
            "records_out": records_out,
        }

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

        records_out = 0
        row_index = 0
        StoragePath(shard.output_path).parent.mkdirs()
        with atomic_rename(shard.output_path) as temp_path:
            with parquet_sink(temp_path) as (where_fd, native_fs):
                with pq.ParquetWriter(where_fd, schema, filesystem=native_fs) as writer:
                    for batch in parquet.iter_batches(batch_size=_FILTER_BATCH_ROWS):
                        keep = pa.array(
                            (index not in duplicate_rows for index in range(row_index, row_index + batch.num_rows)),
                            type=pa.bool_(),
                        )
                        filtered = batch.filter(keep)
                        writer.write_batch(filtered)
                        records_out += filtered.num_rows
                        row_index += batch.num_rows

    return {
        "source": shard.source,
        "path": shard.output_path,
        "records_in": records_in,
        "records_out": records_out,
    }


def _write_success_file(path: str) -> None:
    StoragePath(path).parent.mkdirs()
    with atomic_rename(path) as temp_path:
        with StoragePath(temp_path).open("wb") as success_file:
            success_file.write(b"")


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
    context = ZephyrContext(
        name="datakit-global-exact-dedup",
        resources=worker_resources,
        max_workers=max_workers,
    )

    dedup_success_path = prefix_join(output_path, f"duplicate_ids/{_DEDUP_SUCCESS_FILE}")
    if not StoragePath(dedup_success_path).exists():
        dedup_pipeline = (
            Dataset.from_list(shards)
            .flat_map(_read_record_ids)
            .group_by(
                key=lambda record: record["id"],
                reducer=_select_duplicates,
                sort_by=lambda record: (record["shard_index"], record["row_index"]),
            )
            .group_by(
                key=lambda record: record["shard_index"],
                reducer=_duplicate_id_writer(output_path),
                sort_by=lambda record: record["id"],
            )
        )
        context.execute(dedup_pipeline)
        _write_success_file(dedup_success_path)

    filter_outcome = context.execute(Dataset.from_list(shards).map(_filter_shard))

    filter_results = filter_outcome.results
    records_in = sum(int(result["records_in"]) for result in filter_results)
    records_out = sum(int(result["records_out"]) for result in filter_results)
    source_ranks = {source: rank for rank, source in enumerate(sorted(sources))}
    stage_counters: dict[str, int | float] = {
        f"{COUNTER_PREFIX}/records_in": records_in,
        f"{COUNTER_PREFIX}/records_out": records_out,
        f"{COUNTER_PREFIX}/duplicate_records": records_in - records_out,
    }
    for source, normalized in output_sources.items():
        source_results = [result for result in filter_results if result["source"] == source]
        source_records_in = sum(int(result["records_in"]) for result in source_results)
        source_records_out = sum(int(result["records_out"]) for result in source_results)
        source_counters = {
            f"{COUNTER_PREFIX}/records_in": source_records_in,
            f"{COUNTER_PREFIX}/records_out": source_records_out,
            f"{COUNTER_PREFIX}/duplicate_records": source_records_in - source_records_out,
        }
        rank = source_ranks[source]
        stage_counters[f"{COUNTER_PREFIX}/source/{rank}/records_in"] = source_records_in
        stage_counters[f"{COUNTER_PREFIX}/source/{rank}/records_out"] = source_records_out
        stage_counters[f"{COUNTER_PREFIX}/source/{rank}/duplicate_records"] = source_records_in - source_records_out
        output_sources[source] = normalized.model_copy(update={"counters": source_counters})

    return GlobalExactDedupData(sources=output_sources, counters=stage_counters)
