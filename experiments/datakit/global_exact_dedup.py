# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Global record-ID duplicate attributes for normalized Datakit sources.

The Zephyr pipeline shuffles record IDs across all sources. It keeps the first
source, shard, and row occurrence. It writes sparse duplicate markers back to
the source shards that contain duplicates.

Normalized record IDs are content hashes, so equal IDs represent exact text
duplicates. This pass also covers records for which MinHash emits no fuzzy
attributes, such as text that is too short for the configured n-gram size.

Output rows have this schema::

    {
      id: str,
      dup_doc: bool,
    }

Output files keep the matching normalized shard names. A missing file means
that its source shard has no exact duplicates.
"""

from collections.abc import Iterator
from typing import TypedDict

import pyarrow as pa
import pyarrow.parquet as pq
from fray.types import ResourceConfig
from marin.datakit.copartitioned import (
    CopartitionedShard,
    CopartitionedSource,
    build_copartitioned_shards,
    write_copartitioned_source_manifest,
)
from marin.datakit.normalize import NormalizedData
from marin.datakit.source_key import DatakitArtifactPath, datakit_source_key
from pydantic import BaseModel
from rigging.filesystem import StoragePath
from zephyr import counters
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext
from zephyr.worker_context import zephyr_worker_ctx
from zephyr.writers import write_parquet_file

COUNTER_PREFIX = "global_exact_dedup"
_SHARED_ENTRIES_KEY = "global_exact_dedup_entries"
GLOBAL_EXACT_DEDUP_DATA_VERSION = 3
_ATTR_SCHEMA = pa.schema(
    [
        pa.field("id", pa.string(), nullable=False),
        pa.field("dup_doc", pa.bool_(), nullable=False),
    ]
)


class ExactDupsPerSource(BaseModel):
    """Sparse exact-duplicate attribute files for one normalized source."""

    attr_dir: DatakitArtifactPath


class GlobalExactDedupData(BaseModel):
    """Co-partitioned exact-duplicate attributes for normalized sources.

    ``sources`` maps each input's prefix-relative source key to its attribute
    directory.
    """

    version: str = f"v{GLOBAL_EXACT_DEDUP_DATA_VERSION}"
    sources: dict[str, ExactDupsPerSource]
    counters: dict[str, int | float]


class _ExactRecord(TypedDict):
    id: str
    file_idx: int
    row_index: int


def _build_shard_index(
    sources: dict[str, NormalizedData],
    output_path: str,
) -> tuple[list[CopartitionedShard], dict[str, ExactDupsPerSource]]:
    source_entries = [
        (source_name, normalized, datakit_source_key(normalized.main_output_dir))
        for source_name, normalized in sources.items()
    ]
    source_keys: set[str] = set()
    for _source_name, _normalized, source_key in source_entries:
        if source_key in source_keys:
            raise ValueError(f"Multiple sources use source_key={source_key!r}")
        source_keys.add(source_key)

    ordered_sources = [
        CopartitionedSource(source_key=source_key, input_dir=normalized.main_output_dir)
        for _source_name, normalized, source_key in sorted(source_entries, key=lambda entry: entry[0])
    ]
    entries, attr_dirs = build_copartitioned_shards(
        sources=ordered_sources,
        output_path=output_path,
    )
    outputs = {source_key: ExactDupsPerSource(attr_dir=attr_dir) for source_key, attr_dir in attr_dirs.items()}
    return entries, outputs


def _read_record_ids(entry: CopartitionedShard) -> Iterator[_ExactRecord]:
    input_path = entry.input_path
    row_index = 0
    with StoragePath(input_path).open("rb") as input_file:
        parquet = pq.ParquetFile(input_file)
        if "id" not in parquet.schema_arrow.names:
            raise ValueError(f"Parquet file has no id column: {input_path}")
        id_type = parquet.schema_arrow.field("id").type
        if not (pa.types.is_string(id_type) or pa.types.is_large_string(id_type)):
            raise ValueError(f"Record ID column is not a string in {input_path}: {id_type}")
        for batch in parquet.iter_batches(columns=["id"]):
            for record_id in batch.column("id").to_pylist():
                yield {
                    "id": record_id,
                    "file_idx": entry.file_idx,
                    "row_index": row_index,
                }
                row_index += 1
    counters.pipeline.update_counter(f"{COUNTER_PREFIX}/records_in", row_index)


def _select_duplicates(_key: str, records: Iterator[_ExactRecord]) -> Iterator[_ExactRecord]:
    next(records)
    yield from records


def _write_shard(file_idx: int, records: Iterator[_ExactRecord]) -> None:
    entries: list[CopartitionedShard] = zephyr_worker_ctx().get_shared(_SHARED_ENTRIES_KEY)
    entry = entries[file_idx]
    duplicate_records = 0

    def duplicate_rows() -> Iterator[dict[str, str | bool]]:
        nonlocal duplicate_records
        for record in records:
            duplicate_records += 1
            yield {"id": record["id"], "dup_doc": True}

    write_parquet_file(duplicate_rows(), output_path=entry.output_path, schema=_ATTR_SCHEMA)
    counters.pipeline.update_counter(f"{COUNTER_PREFIX}/duplicate_records", duplicate_records)


def global_exact_deduplicate(
    *,
    sources: dict[str, NormalizedData],
    output_path: str,
    worker_resources: ResourceConfig,
    max_workers: int,
    zephyr_context: ZephyrContext | None = None,
) -> GlobalExactDedupData:
    """Mark duplicate record IDs across all normalized sources.

    Source names set canonical priority. The first shard and row set priority
    within a source.
    """
    if not sources:
        raise ValueError("Global exact deduplication requires at least one source")

    entries, outputs = _build_shard_index(sources, output_path)
    context = zephyr_context or ZephyrContext(
        name="datakit-global-exact-dedup", resources=worker_resources, max_workers=max_workers
    )
    context.put(_SHARED_ENTRIES_KEY, entries)
    shuffle_shards = min(max_workers, len(entries))
    pipeline = (
        Dataset.from_list(entries)
        .flat_map(_read_record_ids)
        .group_by(
            key=lambda record: record["id"],
            reducer=_select_duplicates,
            sort_by=lambda record: (record["file_idx"], record["row_index"]),
            # Bound the cross-source ID shuffle by worker capacity. The next
            # group_by inherits this output shard count.
            num_output_shards=shuffle_shards,
        )
        .group_by(
            key=lambda record: record["file_idx"],
            reducer=_write_shard,
            sort_by=lambda record: record["id"],
        )
    )
    outcome = context.execute(
        pipeline,
        map_task_resources=worker_resources,
    )
    write_copartitioned_source_manifest(
        output_path=output_path,
        attr_dirs={source_key: source.attr_dir for source_key, source in outputs.items()},
    )
    return GlobalExactDedupData(sources=outputs, counters=dict(outcome.counters))
