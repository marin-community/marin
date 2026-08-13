# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Repack fuzzy candidate rows for a changed normalized shard layout."""

import os
from collections.abc import Iterator

import pyarrow as pa
from fray.types import ResourceConfig
from rigging.filesystem import StoragePath, prefix_join
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext

from marin.datakit import partition_filename
from marin.datakit.copartitioned import write_copartitioned_source_manifest
from marin.datakit.normalize import NormalizedData
from marin.datakit.source_key import datakit_source_key
from marin.processing.classification.deduplication.fuzzy_dups import FuzzyDupsAttrData, FuzzyDupsPerSource

_CANDIDATE_COLUMNS = ["id", "dup_cluster_id", "is_cluster_canonical"]
_CANDIDATE_SCHEMA = pa.schema(
    [
        pa.field("id", pa.string(), nullable=False),
        pa.field("dup_cluster_id", pa.string(), nullable=False),
        pa.field("is_cluster_canonical", pa.bool_(), nullable=False),
    ]
)


def _parquet_paths(directory: str) -> list[str]:
    return sorted(str(path) for path in StoragePath(prefix_join(directory, "*.parquet")).glob())


def _normalized_shard_basenames(normalized: NormalizedData) -> list[str]:
    paths = _parquet_paths(normalized.main_output_dir)
    if not paths:
        raise FileNotFoundError(f"No Parquet files found under {normalized.main_output_dir}")

    basenames = [os.path.basename(path) for path in paths]
    expected = [partition_filename(shard, len(paths)) for shard in range(len(paths))]
    if basenames != expected:
        raise ValueError(
            f"Normalized shard names do not use the Datakit partition layout. Expected {expected}, got {basenames}"
        )
    return basenames


def _select_consistent_candidate(doc_id: str, rows: Iterator[dict]) -> dict:
    first = next(rows)
    candidate = {
        "id": doc_id,
        "dup_cluster_id": first["dup_cluster_id"],
        "is_cluster_canonical": first["is_cluster_canonical"],
    }
    for row in rows:
        if row["dup_cluster_id"] != candidate["dup_cluster_id"] or (
            row["is_cluster_canonical"] != candidate["is_cluster_canonical"]
        ):
            raise ValueError(f"Candidate rows disagree for id={doc_id!r}")
    return candidate


def repack_fuzzy_dups_source(
    *,
    candidates: FuzzyDupsAttrData,
    legacy_source_key: str,
    normalized: NormalizedData,
    output_path: str,
    max_workers: int = 64,
    worker_resources: ResourceConfig | None = None,
    coordinator_resources: ResourceConfig | None = None,
    map_task_resources: ResourceConfig | None = None,
    reduce_task_resources: ResourceConfig | None = None,
    zephyr_context: ZephyrContext | None = None,
) -> FuzzyDupsAttrData:
    """Move one candidate source to the current normalized shard layout.

    Normalization and this function use the same deterministic ``id`` shard
    rule. Thus, this function does not read the normalized text.
    """
    legacy_source = candidates.sources.get(legacy_source_key)
    if legacy_source is None:
        raise KeyError(f"Fuzzy candidate data has no source_key={legacy_source_key!r}")

    source_key = datakit_source_key(normalized.main_output_dir)
    if source_key == legacy_source_key:
        raise ValueError("The source key did not change")
    if source_key in candidates.sources:
        raise ValueError(f"Fuzzy candidate data already has source_key={source_key!r}")

    candidate_paths = _parquet_paths(legacy_source.attr_dir)
    if not candidate_paths:
        raise FileNotFoundError(f"No Parquet files found under {legacy_source.attr_dir}")
    normalized_basenames = _normalized_shard_basenames(normalized)
    attr_dir = prefix_join(output_path, "outputs/repacked_source")

    def output_path_for_shard(shard: int, total: int) -> str:
        if total != len(normalized_basenames):
            raise ValueError(f"Expected {len(normalized_basenames)} output shards, got {total}")
        return prefix_join(attr_dir, normalized_basenames[shard])

    pipeline = (
        Dataset.from_list(candidate_paths)
        .load_parquet(columns=_CANDIDATE_COLUMNS)
        .group_by(
            key=lambda row: row["id"],
            reducer=_select_consistent_candidate,
            num_output_shards=len(normalized_basenames),
        )
        .write_parquet(output_path_for_shard, schema=_CANDIDATE_SCHEMA)
    )

    context_args: dict = {"name": "repack-fuzzy-dups", "max_workers": max_workers}
    if worker_resources is not None:
        context_args["resources"] = worker_resources
    if coordinator_resources is not None:
        context_args["coordinator_resources"] = coordinator_resources
    context = zephyr_context or ZephyrContext(**context_args)
    outcome = context.execute(
        pipeline,
        verbose=True,
        map_task_resources=map_task_resources or worker_resources,
        reduce_task_resources=reduce_task_resources or worker_resources,
    )

    sources = dict(candidates.sources)
    del sources[legacy_source_key]
    sources[source_key] = FuzzyDupsPerSource(attr_dir=attr_dir)
    write_copartitioned_source_manifest(
        output_path=output_path,
        attr_dirs={key: source.attr_dir for key, source in sources.items()},
    )
    counters = dict(candidates.counters)
    counters.update({f"repack/{key}": value for key, value in outcome.counters.items()})
    return FuzzyDupsAttrData(params=candidates.params, sources=sources, counters=counters)
