# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Datakit nemotron ferry: weekly reference-DAG run on the Nemotron-CC high split.

The first step confirms the ``quality=high`` subtree of the Nemotron-CC dump is
already staged at ``NEMOTRON_RAW_PATH`` and refuses to initiate a Common Crawl
download. Normalize and every reference stage after it write one-day TTL outputs;
see :mod:`experiments.ferries.datakit_reference_ferry`.
"""

import logging

from fray.types import ResourceConfig
from marin.datakit.normalize import normalize_step
from marin.execution.step_spec import StepSpec
from marin.processing.classification.deduplication.cluster_text import ClusterTextParams
from rigging.filesystem.cluster_config import check_path_in_region, region_from_metadata
from rigging.filesystem.factory import url_to_fs
from rigging.filesystem.storage_path import prefix_join
from rigging.log_setup import configure_logging

from experiments.datakit.reference_pipeline import (
    ClusterConfig,
    FuzzyClusterConfig,
    PipelineScale,
    PoolConfig,
    StoreConfig,
)
from experiments.ferries.datakit_reference_ferry import ferry_output_prefix, run_reference_ferry

logger = logging.getLogger(__name__)

FERRY_NAME = "datakit-nemotron-smoke"
SOURCE_NAME = "nemotron-cc-high"

# Canonical, region-pinned location of the staged Nemotron-CC raw dump. The
# dump was populated by a one-off download into marin-eu-west4; the ferry only
# reads from it and will fail-fast if it isn't there.
NEMOTRON_RAW_PATH = "gs://marin-eu-west4/raw/nemotro-cc-eeb783"
NEMOTRON_DATA_SUBDIR = "contrib/Nemotron/Nemotron-CC/data-jsonl"
NEMOTRON_QUALITY_DIR = "quality=high"

# The yaml sets FERRY_TEST_MAX_FILES=1000 to cap the input shard count
# (quality=high has ~2,755 shards / ~960 GB). ~1,380 normalized shards. Every Marin
# VM has a 100 GB disk, so small disk requests set how many workers share a host;
# Zephyr stages pull shards, so a partly scheduled pool is slower, not stuck.
TIER3_SCALE = PipelineScale(
    cluster=ClusterConfig(k_train=64, k_views=(8, 16), cluster_view=8),
    pool=PoolConfig(
        n_workers=512,
        worker=ResourceConfig(cpu=2, ram="16g", disk="16g"),
        task=ResourceConfig(cpu=2, ram="16g", disk="16g"),
    ),
    fuzzy=FuzzyClusterConfig(
        text=ClusterTextParams(output_shards=1024),
        worker=ResourceConfig(cpu=16, ram="128g", disk="64g"),
        map_task=ResourceConfig(cpu=1, ram="8g", disk="8g"),
        reduce_task=ResourceConfig(cpu=1, ram="8g", disk="8g"),
        max_workers=32,
    ),
    store=StoreConfig(task_count=256, worker=ResourceConfig(cpu=2, ram="16g", disk="32g")),
    dedup_max_parallelism=1024,
    train_centroids_resources=ResourceConfig.with_cpu(cpu=4, ram="16g"),
)


def _verify_nemotron_quality_present(output_path: str) -> None:
    """Confirm the quality split is staged at ``output_path``; never downloads.

    Invoked by StepRunner only on a cache miss. Raises with a clear message so
    that an accidental cache eviction can never trigger a multi-TB Common Crawl
    re-download.
    """
    quality_dir = f"{output_path}/{NEMOTRON_DATA_SUBDIR}/{NEMOTRON_QUALITY_DIR}"
    fs, _ = url_to_fs(quality_dir)
    if not fs.exists(quality_dir):
        raise RuntimeError(
            f"Nemotron-CC {NEMOTRON_QUALITY_DIR} not found at {quality_dir}. "
            "The nemotron ferry refuses to download Common Crawl — stage the raw dump externally first."
        )
    sample = fs.glob(f"{quality_dir}/**/*.jsonl.*", maxdepth=4)
    if not sample:
        raise RuntimeError(f"Nemotron-CC {NEMOTRON_QUALITY_DIR} at {quality_dir} contains no .jsonl.* files.")
    logger.info("Nemotron-CC %s confirmed at %s (e.g. %s)", NEMOTRON_QUALITY_DIR, quality_dir, sample[0])


def build_sources(output_prefix: str) -> dict[str, StepSpec]:
    # Verify-only raw step. Uses an absolute override so it points at the
    # pre-staged dump regardless of MARIN_PREFIX.
    download = StepSpec(
        name=f"{FERRY_NAME}/download",
        fn=_verify_nemotron_quality_present,
        override_output_path=NEMOTRON_RAW_PATH,
    )
    # Sizes mirror validate_normalize_phase1.py, which ran successfully on
    # nemotron_v1 in eu-west4. FERRY_TEST_MAX_FILES is read at execution time
    # by `_discover_files`.
    normalized = normalize_step(
        name=f"{FERRY_NAME}/normalize",
        download=download,
        text_field="text",
        id_field="id",
        relative_input_path=f"{NEMOTRON_DATA_SUBDIR}/{NEMOTRON_QUALITY_DIR}",
        worker_resources=ResourceConfig(cpu=2, ram="16g", disk="5g"),
        max_workers=512,
        override_output_path=prefix_join(output_prefix, "normalize"),
    )
    return {SOURCE_NAME: normalized}


def main() -> None:
    configure_logging()
    # Guard against accidental cross-region reads of the multi-TB raw dump.
    region = region_from_metadata()
    if region:
        check_path_in_region("nemotron_raw", NEMOTRON_RAW_PATH, region)

    output_prefix = ferry_output_prefix(FERRY_NAME)
    run_reference_ferry(
        ferry_name=FERRY_NAME,
        sources=build_sources(output_prefix),
        scale=TIER3_SCALE,
        output_prefix=output_prefix,
    )


if __name__ == "__main__":
    main()
