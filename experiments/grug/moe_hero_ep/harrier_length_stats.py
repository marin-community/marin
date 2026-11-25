# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read Harrier document lengths and publish small per-cell statistics."""

import argparse
import asyncio
from dataclasses import dataclass

import numpy as np
from fray.types import ResourceConfig
from levanter.data.text.length_bucket import (
    DOCUMENT_CHUNK_ROWS,
    LENGTH_STATS_FILE,
    BucketLengthStats,
    DocumentLengthStats,
    document_length_stats,
)
from levanter.store.cache import TreeCache
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, StepContext, lower
from marin.execution.remote import remote
from marin.execution.step_runner import StepRunner
from rigging.filesystem.cluster_config import marin_temp_bucket
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath, prefix_join
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset

from experiments.grug.moe_hero_ep.harrier_mix_2026_08_18 import _SPEC, HARRIER_MIX_2026_08_18_STORE

LENGTH_THRESHOLDS = (65_536, 131_072, 262_144)
_MAX_WORKERS = 32


def _cell_cache_dir(store: str, cell: str) -> str:
    return prefix_join(store, f"cluster={int(cell[1:3])}/quality={int(cell[4])}")


@dataclass(frozen=True)
class LengthStatsConfig:
    output_path: str
    store: str
    cells: tuple[str, ...]
    thresholds: tuple[int, ...]
    chunk_rows: int


def write_length_stats(
    output_path: str,
    store: str,
    cells: tuple[str, ...],
    *,
    thresholds: tuple[int, ...],
    chunk_rows: int,
    zephyr_context: ZephyrContext,
) -> BucketLengthStats:
    """Write one JSON metadata file for the existing Harrier caches; no tokens are copied."""
    configure_coreweave_s3()

    def count(cell: str) -> tuple[str, DocumentLengthStats]:
        configure_coreweave_s3()
        cache = TreeCache.load(_cell_cache_dir(store, cell), {"input_ids": np.empty(0, dtype=np.int32)})
        return cell, asyncio.run(document_length_stats(cache, thresholds=thresholds, chunk_rows=chunk_rows))

    pipeline = Dataset.from_list(list(cells)).map(count)
    cell_stats = dict(zephyr_context.execute(pipeline, verbose=True).results)
    stats = BucketLengthStats(chunk_rows=chunk_rows, buckets=cell_stats)
    StoragePath(output_path).mkdirs()
    StoragePath(prefix_join(output_path, LENGTH_STATS_FILE)).write_text(stats.model_dump_json(indent=2))
    return stats


def _run_length_stats(config: LengthStatsConfig) -> Artifact:
    context = ZephyrContext(
        resources=ResourceConfig(cpu=1, ram="2g"),
        coordinator_resources=ResourceConfig(cpu=1, ram="4g", preemptible=False),
        max_workers=min(_MAX_WORKERS, len(config.cells)),
        chunk_storage_prefix=marin_temp_bucket(ttl_days=1, prefix="zephyr", source_prefix=config.output_path),
        name="harrier-length-stats",
    )
    write_length_stats(
        config.output_path,
        config.store,
        config.cells,
        thresholds=config.thresholds,
        chunk_rows=config.chunk_rows,
        zephyr_context=context,
    )
    return Artifact(path=config.output_path)


def harrier_length_statistics() -> ArtifactStep[Artifact]:
    """Bind the Harrier offset scan into the training artifact graph."""

    def build_config(ctx: StepContext) -> LengthStatsConfig:
        return LengthStatsConfig(
            output_path=ctx.output_path,
            store=ctx.artifact_path(HARRIER_MIX_2026_08_18_STORE),
            cells=tuple(cell for cell, _ in _SPEC.available_tokens),
            thresholds=LENGTH_THRESHOLDS,
            chunk_rows=DOCUMENT_CHUNK_ROWS,
        )

    return ArtifactStep(
        name="grug/moe-hero/harrier-length-stats",
        version="2026.11.05",
        artifact_type=Artifact,
        run=remote(
            _run_length_stats,
            resources=ResourceConfig(cpu=1, ram="4g", disk="8g"),
        ),
        build_config=build_config,
        deps=(HARRIER_MIX_2026_08_18_STORE,),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    StepRunner().run([lower(harrier_length_statistics())], dry_run=args.dry_run)


if __name__ == "__main__":
    main()
