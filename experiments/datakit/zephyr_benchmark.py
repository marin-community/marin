# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the S3-backed Zephyr stages of the Datakit reference pipeline.

The benchmark starts from an existing normalized Datakit sample, so its measured
pipeline does not include Hugging Face corpus download time. It runs global exact
deduplication, per-source tokenization and MinHash, then cross-source fuzzy dedup.

Example::

    python -m experiments.datakit.zephyr_benchmark \
        --sample-prefix s3://marin-us-east-02a/marin/datakit/sample_100b_8ae7a94f \
        --sources all --run-tag zephyr-100b-v1 \
        --coordinator-cpu 8 --coordinator-ram 32g --coordinator-disk 16g \
        --pool-workers 60 --pool-cpu 16 --pool-ram 160g --pool-disk 32g \
        --task-cpu 1 --task-ram 10g --task-disk 2g \
        --chunk-storage-prefix s3://bucket/tmp/ttl=7d/zephyr-100b-v1/chunks \
        --last-stage fuzzy \
        --max-concurrent 4 --dedup-max-parallelism 4096
"""

import argparse
import logging
from dataclasses import replace
from enum import StrEnum

from fray.types import ResourceConfig
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
from rigging.log_setup import configure_logging
from zephyr.execution import ZephyrContext
from zephyr.runners import SubprocessRunner

from experiments.datakit.reference_pipeline import (
    SMOKE_SCALE,
    PoolConfig,
    ZephyrDatakitSteps,
    sample_sources,
    zephyr_datakit_steps,
)


class LastStage(StrEnum):
    """Last stage included in a benchmark run."""

    EXACT = "exact"
    TOKENIZE = "tokenize"
    MINHASH = "minhash"
    FUZZY = "fuzzy"


def _steps_through(steps: ZephyrDatakitSteps, last_stage: LastStage) -> list[StepSpec]:
    selected = [steps.exact_dedup]
    if last_stage in {LastStage.TOKENIZE, LastStage.MINHASH, LastStage.FUZZY}:
        selected.extend(steps.tokenize.values())
    if last_stage in {LastStage.MINHASH, LastStage.FUZZY}:
        selected.extend(steps.minhash.values())
    if last_stage is LastStage.FUZZY:
        selected.append(steps.fuzzy_dedup)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-prefix", required=True)
    parser.add_argument("--sources", required=True, help="Comma-separated source names or 'all'.")
    parser.add_argument("--run-tag", required=True, help="Fresh identity tag that forces uncached benchmark stages.")
    parser.add_argument("--coordinator-cpu", required=True, type=float)
    parser.add_argument("--coordinator-ram", required=True)
    parser.add_argument("--coordinator-disk", required=True)
    parser.add_argument("--pool-workers", required=True, type=int)
    parser.add_argument("--pool-cpu", required=True, type=float)
    parser.add_argument("--pool-ram", required=True)
    parser.add_argument("--pool-disk", required=True)
    parser.add_argument("--task-cpu", required=True, type=float)
    parser.add_argument("--task-ram", required=True)
    parser.add_argument("--task-disk", required=True)
    parser.add_argument("--chunk-storage-prefix", required=True)
    parser.add_argument("--last-stage", required=True, type=LastStage, choices=list(LastStage))
    parser.add_argument("--max-concurrent", required=True, type=int)
    parser.add_argument("--dedup-max-parallelism", required=True, type=int)
    args = parser.parse_args()

    configure_logging(logging.INFO)
    selected_sources = None if args.sources == "all" else [name.strip() for name in args.sources.split(",")]
    sources = sample_sources(args.sample_prefix, selected_sources, args.run_tag)
    coordinator = ResourceConfig(
        cpu=args.coordinator_cpu,
        ram=args.coordinator_ram,
        disk=args.coordinator_disk,
        preemptible=False,
    )
    pool_worker = ResourceConfig(cpu=args.pool_cpu, ram=args.pool_ram, disk=args.pool_disk)
    task = ResourceConfig(cpu=args.task_cpu, ram=args.task_ram, disk=args.task_disk)
    scale = replace(
        SMOKE_SCALE,
        pool=PoolConfig(n_workers=args.pool_workers, worker=task),
        dedup_max_parallelism=args.dedup_max_parallelism,
    )
    with ZephyrContext(
        name="datakit-zephyr-benchmark",
        resources=pool_worker,
        coordinator_resources=coordinator,
        max_concurrent_pipelines=args.max_concurrent,
        max_workers=args.pool_workers,
        chunk_storage_prefix=args.chunk_storage_prefix,
        stage_runner_factory=SubprocessRunner,
    ) as zephyr_context:
        steps = zephyr_datakit_steps(sources, scale, zephyr_context)
        StepRunner().run(_steps_through(steps, args.last_stage), max_concurrent=args.max_concurrent)


if __name__ == "__main__":
    main()
