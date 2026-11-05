# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read Harrier document lengths and publish small per-cell statistics."""

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor

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
from marin.execution.remote import remote
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath, prefix_join

from experiments.grug.moe_hero_ep.harrier_mix_2026_08_18 import _SPEC

LENGTH_THRESHOLDS = (65_536, 131_072, 262_144)
_MAX_WORKERS = 16


def _cell_cache_dir(store: str, cell: str) -> str:
    return prefix_join(store, f"cluster={int(cell[1:3])}/quality={int(cell[4])}")


def write_length_stats(output_path: str, store: str, cells: tuple[str, ...]) -> BucketLengthStats:
    """Write one JSON metadata file for the existing Harrier caches; no tokens are copied."""
    configure_coreweave_s3()

    def count(cell: str) -> tuple[str, DocumentLengthStats]:
        cache = TreeCache.load(_cell_cache_dir(store, cell), {"input_ids": np.empty(0, dtype=np.int32)})
        return cell, asyncio.run(
            document_length_stats(cache, thresholds=LENGTH_THRESHOLDS, chunk_rows=DOCUMENT_CHUNK_ROWS)
        )

    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
        cell_stats = dict(pool.map(count, cells))
    stats = BucketLengthStats(chunk_rows=DOCUMENT_CHUNK_ROWS, buckets=cell_stats)
    StoragePath(output_path).mkdirs()
    StoragePath(prefix_join(output_path, LENGTH_STATS_FILE)).write_text(stats.model_dump_json(indent=2))
    return stats


def length_stats_step(store: str | None = None) -> StepSpec:
    """Create the read-only Harrier length scan and its small metadata artifact."""
    cells = tuple(cell for cell, _ in _SPEC.available_tokens)
    store = store or _SPEC.candidate_store_uri
    return StepSpec(
        name="grug/moe-hero/harrier-length-stats",
        hash_attrs={"store": store, "thresholds": LENGTH_THRESHOLDS, "chunk_rows": DOCUMENT_CHUNK_ROWS, "cells": cells},
        fn=remote(
            lambda output_path: write_length_stats(output_path, store, cells),
            resources=ResourceConfig(cpu=8, ram="16g", disk="8g"),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    StepRunner().run([length_stats_step()], dry_run=args.dry_run)


if __name__ == "__main__":
    main()
