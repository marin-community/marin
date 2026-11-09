# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Datakit nemotron ferry: weekly reference-DAG run on the Nemotron-CC high-quality actual split.

The input is the normalized Nemotron-CC v1 ``hq_actual`` split in eu-west4. The first
step confirms that the split is at ``NEMOTRON_HQ_ACTUAL_PATH`` and never rebuilds it.
Normalize runs again on those Parquet shards. Normalize and every reference stage
after it write one-day TTL outputs; see :mod:`experiments.ferries.datakit_reference_ferry`.
"""

import logging
import os

from fray.types import ResourceConfig
from marin.datakit.normalize import normalize_step
from marin.execution.step_spec import StepSpec
from marin.processing.classification.deduplication.cluster_text import ClusterTextParams
from rigging.filesystem.cluster_config import check_path_in_region, region_from_metadata
from rigging.filesystem.storage_path import StoragePath, prefix_join
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

# Region-pinned output of ``normalize_nemotron_v1_step(split="hq_actual")``: 3,741
# Parquet files, 704M documents. The ferry only reads it and fails fast if it is absent.
NEMOTRON_HQ_ACTUAL_PATH = "gs://marin-eu-west4/normalized/nemotron_v1/hq_actual_d6502d2c"
NEMOTRON_HQ_ACTUAL_DATA_SUBDIR = "outputs/main"

# The yaml sets FERRY_TEST_MAX_FILES=1600: about 301M documents in 655 GB. Normalize
# hash-partitions by document ID, so each file is a uniform sample of the split. The
# raw-dump runs normalized 301M documents from 371 GB of JSONL into 1,382 shards. At
# 448 MiB per partition, the Parquet input also gives about 1,390 shards of 216K
# documents.
NORMALIZE_TARGET_PARTITION_BYTES = 448 * 1024 * 1024

# Every Marin VM has a 100 GB disk, so small disk requests set how many workers share
# a host; Zephyr stages pull shards, so a partly scheduled pool is slower, not stuck.
TIER3_SCALE = PipelineScale(
    cluster=ClusterConfig(k_train=64, k_views=(8, 16), cluster_view=8),
    pool=PoolConfig(
        n_workers=512,
        worker=ResourceConfig(cpu=2, ram="16g", disk="16g"),
        task=ResourceConfig(cpu=2, ram="16g", disk="16g"),
    ),
    fuzzy=FuzzyClusterConfig(
        text=ClusterTextParams(output_shards=1024),
        # 8 tasks per worker. At 16 tasks, cluster text peaked at 66 GiB and verify at
        # 26 GiB. A 16-CPU/128g worker fits once per v5e host, and verify waited 3h46m.
        worker=ResourceConfig(cpu=8, ram="64g", disk="32g"),
        map_task=ResourceConfig(cpu=1, ram="8g", disk="8g"),
        reduce_task=ResourceConfig(cpu=1, ram="8g", disk="8g"),
        max_workers=64,
    ),
    store=StoreConfig(task_count=256, worker=ResourceConfig(cpu=2, ram="16g", disk="32g")),
    dedup_max_parallelism=1024,
    train_centroids_resources=ResourceConfig.with_cpu(cpu=4, ram="16g"),
)


def _verify_nemotron_hq_actual_present(output_path: str) -> None:
    """Confirm the normalized split is at ``output_path``; never rebuilds it.

    StepRunner calls this only on a cache miss, which means the split's
    ``.executor_status`` is gone. The ferry then fails instead of rebuilding a
    multi-TB split.
    """
    data_dir = StoragePath(output_path) / NEMOTRON_HQ_ACTUAL_DATA_SUBDIR
    if not data_dir.exists():
        raise RuntimeError(
            f"Normalized Nemotron-CC hq_actual not found at {data_dir}. "
            "The nemotron ferry does not rebuild its input; restore the split or point the ferry at another copy."
        )
    sample = (data_dir / "*.parquet").glob()
    if not sample:
        raise RuntimeError(f"Normalized Nemotron-CC hq_actual at {data_dir} contains no .parquet files.")
    logger.info("Normalized Nemotron-CC hq_actual confirmed at %s (e.g. %s)", data_dir, sample[0])


def build_sources(output_prefix: str) -> dict[str, StepSpec]:
    # Verify-only input step. The absolute override points at the normalized split
    # regardless of MARIN_PREFIX.
    download = StepSpec(
        name=f"{FERRY_NAME}/download",
        fn=_verify_nemotron_hq_actual_present,
        override_output_path=NEMOTRON_HQ_ACTUAL_PATH,
    )
    # The input already has a content-hash ``id`` and keeps the WARC record ID in
    # ``source_id``. Reading ``source_id`` as the ID field gives the same output
    # schema as normalizing the raw dump. FERRY_TEST_MAX_FILES is read at execution
    # time by ``_discover_files``.
    normalized = normalize_step(
        name=f"{FERRY_NAME}/normalize",
        download=download,
        text_field="text",
        id_field="source_id",
        relative_input_path=NEMOTRON_HQ_ACTUAL_DATA_SUBDIR,
        file_extensions=(".parquet",),
        target_partition_bytes=NORMALIZE_TARGET_PARTITION_BYTES,
        worker_resources=ResourceConfig(cpu=2, ram="16g", disk="5g"),
        max_workers=512,
        override_output_path=prefix_join(output_prefix, "normalize"),
    )
    return {SOURCE_NAME: normalized}


def main() -> None:
    configure_logging()
    # Guard against accidental cross-region reads of the multi-TB input.
    region = region_from_metadata()
    if region:
        check_path_in_region("nemotron_hq_actual", NEMOTRON_HQ_ACTUAL_PATH, region)

    output_prefix = ferry_output_prefix(FERRY_NAME, os.environ["SMOKE_RUN_ID"])
    run_reference_ferry(
        ferry_name=FERRY_NAME,
        sources=build_sources(output_prefix),
        scale=TIER3_SCALE,
        output_prefix=output_prefix,
        status_path=os.environ.get("FERRY_STATUS_PATH"),
    )


if __name__ == "__main__":
    main()
