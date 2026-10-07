# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Datakit tier-1 ferry: the reference Datakit DAG on FineWeb-Edu ``sample/10BT``.

Downloads and normalizes the subset, then runs every reference stage through the
final clustered store. See :mod:`experiments.ferries.datakit_reference_ferry`.
"""

import os

from fray.types import ResourceConfig
from marin.datakit.download.huggingface_zephyr import download_hf_step
from marin.datakit.normalize import normalize_step
from marin.execution.step_spec import StepSpec
from marin.processing.classification.deduplication.cluster_text import ClusterTextParams
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

FERRY_NAME = "datakit-smoke"
SOURCE_NAME = "fineweb-edu-10bt"

# FineWeb-Edu sample/10BT: 14 parquet shards, ~9.7M rows, ~106 normalized shards.
TIER1_SCALE = PipelineScale(
    cluster=ClusterConfig(k_train=64, k_views=(8, 16), cluster_view=8),
    pool=PoolConfig(
        n_workers=128,
        worker=ResourceConfig(cpu=2, ram="16g", disk="16g"),
        task=ResourceConfig(cpu=2, ram="16g", disk="16g"),
    ),
    fuzzy=FuzzyClusterConfig(
        text=ClusterTextParams(output_shards=256),
        worker=ResourceConfig(cpu=8, ram="64g", disk="64g"),
        map_task=ResourceConfig(cpu=1, ram="8g", disk="8g"),
        reduce_task=ResourceConfig(cpu=1, ram="8g", disk="8g"),
        max_workers=16,
    ),
    store=StoreConfig(task_count=None, worker=ResourceConfig(cpu=4, ram="32g", disk="64g")),
    dedup_max_parallelism=128,
    train_centroids_resources=ResourceConfig.with_cpu(cpu=4, ram="16g"),
)


def build_sources(output_prefix: str) -> dict[str, StepSpec]:
    # Filtered download: restrict to sample/10BT so we don't pull the whole repo (TBs).
    downloaded = download_hf_step(
        f"{FERRY_NAME}/download",
        hf_dataset_id="HuggingFaceFW/fineweb-edu",
        revision="87f0914",
        hf_urls_glob=["sample/10BT/*.parquet"],
        zephyr_max_parallelism=14,  # fineweb-edu sample/10BT has 14 parquet shards
        override_output_path=prefix_join(output_prefix, "download"),
    )
    # Normalize peaked at ~10 GB mem, 17 GB disk on 10BT; bump disk from default 10g.
    normalized = normalize_step(
        name=f"{FERRY_NAME}/normalize",
        download=downloaded,
        relative_input_path="sample/10BT",
        worker_resources=ResourceConfig(cpu=2, ram="16g", disk="20g"),
        override_output_path=prefix_join(output_prefix, "normalize"),
    )
    return {SOURCE_NAME: normalized}


def main() -> None:
    configure_logging()
    output_prefix = ferry_output_prefix(FERRY_NAME, os.environ["SMOKE_RUN_ID"])
    run_reference_ferry(
        ferry_name=FERRY_NAME,
        sources=build_sources(output_prefix),
        scale=TIER1_SCALE,
        output_prefix=output_prefix,
        status_path=os.environ.get("FERRY_STATUS_PATH"),
    )


if __name__ == "__main__":
    main()
