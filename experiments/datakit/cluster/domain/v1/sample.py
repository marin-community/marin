# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Sample Harrier embeddings for centroid training."""

import hashlib
import json
import logging
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from fray.types import ResourceConfig
from rigging.filesystem import StoragePath, open_url, prefix_join
from zephyr import counters
from zephyr.dataset import Dataset, ShardInfo
from zephyr.execution import ZephyrContext
from zephyr.readers import InputFileSpec, load_file
from zephyr.runners import InlineRunner

from experiments.datakit.embeddings.harrier.pipeline import HARRIER_DIM

logger = logging.getLogger(__name__)

_SAMPLE_SCHEMA = pa.schema(
    [
        pa.field("source", pa.string()),
        pa.field("embedding", pa.list_(pa.int8(), HARRIER_DIM)),
    ]
)


@dataclass(frozen=True)
class SourceSample:
    source: str
    paths: tuple[str, ...]
    rows: tuple[int, ...]
    quota: int


def largest_remainder_quotas(weights: tuple[int, ...], target: int) -> tuple[int, ...]:
    total = sum(weights)
    if target == 0:
        return (0,) * len(weights)
    if total == 0:
        raise ValueError("cannot allocate a positive quota across zero available rows")
    numerators = [target * weight for weight in weights]
    quotas = [numerator // total for numerator in numerators]
    leftover = target - sum(quotas)
    order = sorted(range(len(weights)), key=lambda index: (-(numerators[index] % total), index))
    for index in order[:leftover]:
        quotas[index] += 1
    return tuple(quotas)


def proportional_quotas(
    counts: dict[str, int],
    target: int,
    small_source_max_rows: int,
    small_source_quota: int,
) -> dict[str, int]:
    if target < len(counts) * small_source_quota:
        raise ValueError("target is too small for the source floor")
    if target > sum(counts.values()):
        raise ValueError("target exceeds the available documents")

    small = {source for source, rows in counts.items() if rows <= small_source_max_rows}
    quotas = {source: small_source_quota for source in small}
    large_sources = tuple(sorted(source for source in counts if source not in small))
    remaining = target - sum(quotas.values())
    large_quotas = largest_remainder_quotas(tuple(counts[source] for source in large_sources), remaining)
    quotas.update(dict(zip(large_sources, large_quotas, strict=True)))
    if any(quota > counts[source] for source, quota in quotas.items()):
        raise ValueError("a source quota exceeds its document count")
    return quotas


def _row_count(path: str) -> int:
    with StoragePath(path).open("rb") as file:
        return pq.ParquetFile(file).metadata.num_rows


def _source_samples(
    embedding_paths: dict[str, str],
    target_rows: int,
    small_source_max_rows: int,
    small_source_quota: int,
    load_parallelism: int,
) -> tuple[SourceSample, ...]:
    source_paths = {
        source: tuple(sorted(str(shard) for shard in StoragePath(prefix_join(path, "*.parquet")).glob()))
        for source, path in sorted(embedding_paths.items())
    }
    if any(not paths for paths in source_paths.values()):
        missing = [source for source, paths in source_paths.items() if not paths]
        raise ValueError(f"sources have no embedding shards: {missing}")

    all_paths = [path for paths in source_paths.values() for path in paths]
    with ThreadPoolExecutor(max_workers=load_parallelism) as pool:
        all_rows = list(pool.map(_row_count, all_paths))
    path_rows = dict(zip(all_paths, all_rows, strict=True))
    source_rows = {source: sum(path_rows[path] for path in paths) for source, paths in source_paths.items()}
    empty_sources = [source for source, rows in source_rows.items() if rows == 0]
    if empty_sources:
        logger.warning("Skipping %d zero-row sources: %s", len(empty_sources), empty_sources)
        source_paths = {source: paths for source, paths in source_paths.items() if source_rows[source]}
        source_rows = {source: rows for source, rows in source_rows.items() if rows}
    quotas = proportional_quotas(source_rows, target_rows, small_source_max_rows, small_source_quota)
    return tuple(
        SourceSample(
            source=source,
            paths=paths,
            rows=tuple(path_rows[path] for path in paths),
            quota=quotas[source],
        )
        for source, paths in sorted(source_paths.items())
    )


def _sample_shard(
    batches: Iterator[list[dict[str, Any]]],
    _shard: ShardInfo,
    *,
    source: str,
    path: str,
    rows: int,
    quota: int,
    seed: int,
) -> Iterator[dict[str, Any]]:
    digest = hashlib.sha256(f"{seed}:{source}:{path}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    selected = np.sort(rng.choice(rows, size=quota, replace=False))
    offset = 0
    written = 0
    for batch in batches:
        begin = int(np.searchsorted(selected, offset, side="left"))
        end = int(np.searchsorted(selected, offset + len(batch), side="left"))
        for index in selected[begin:end] - offset:
            yield {"source": source, "embedding": batch[int(index)]["embedding"]}
            written += 1
        offset += len(batch)
    if offset != rows or written != quota:
        raise ValueError(f"sampled {written}/{quota} rows after reading {offset}/{rows} from {path}")
    counters.pipeline.update_counter("harrier_cluster/sample_rows", written)


def _sample_source(
    sample: SourceSample,
    output_path: str,
    seed: int,
    worker_resources: ResourceConfig,
    coordinator_resources: ResourceConfig,
    max_workers: int,
) -> None:
    quotas = largest_remainder_quotas(sample.rows, sample.quota)
    selected = [
        (path, rows, quota) for path, rows, quota in zip(sample.paths, sample.rows, quotas, strict=True) if quota
    ]
    source_dir = prefix_join(output_path, sample.source.replace("/", "-"))
    output_paths = tuple(prefix_join(source_dir, f"sample-{index:06d}.parquet") for index in range(len(selected)))
    dataset = (
        Dataset.from_list([InputFileSpec(path=path, columns=["embedding"]) for path, _, _ in selected])
        .flat_map(load_file)
        .window(4_096)
        .map_shard(
            lambda batches, shard, parts=tuple(selected), source=sample.source: _sample_shard(
                batches,
                shard,
                source=source,
                path=parts[shard.shard_idx][0],
                rows=parts[shard.shard_idx][1],
                quota=parts[shard.shard_idx][2],
                seed=seed,
            )
        )
        .write_parquet(lambda shard_idx, _total: output_paths[shard_idx], schema=_SAMPLE_SCHEMA, skip_existing=True)
    )
    ZephyrContext(
        resources=worker_resources,
        coordinator_resources=coordinator_resources,
        max_workers=min(max_workers, len(selected)),
        name=f"sample-harrier-{hashlib.sha256(sample.source.encode()).hexdigest()[:12]}",
        stage_runner_factory=InlineRunner,
    ).execute(dataset, verbose=True)


def sample_centroid_inputs(
    output_path: str,
    embedding_paths: dict[str, str],
    target_rows: int,
    small_source_max_rows: int,
    small_source_quota: int,
    seed: int,
    worker_resources: ResourceConfig,
    coordinator_resources: ResourceConfig,
    max_workers: int,
    parallel_sources: int,
    load_parallelism: int,
) -> None:
    samples = _source_samples(
        embedding_paths,
        target_rows,
        small_source_max_rows,
        small_source_quota,
        load_parallelism,
    )
    with ThreadPoolExecutor(max_workers=parallel_sources) as pool:
        futures = {
            pool.submit(
                _sample_source,
                sample,
                output_path,
                seed,
                worker_resources,
                coordinator_resources,
                max_workers,
            ): sample.source
            for sample in samples
        }
        for future in as_completed(futures):
            future.result()
            logger.info("Sampled %s", futures[future])

    with open_url(prefix_join(output_path, "sample_stats.json"), "w") as file:
        json.dump(
            {
                "target_rows": target_rows,
                "source_count": len(samples),
                "source_rows": {sample.source: sum(sample.rows) for sample in samples},
                "source_quotas": {sample.source: sample.quota for sample in samples},
                "embedding_paths": embedding_paths,
                "small_source_max_rows": small_source_max_rows,
                "small_source_quota": small_source_quota,
                "seed": seed,
            },
            file,
            indent=2,
            sort_keys=True,
        )
