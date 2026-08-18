# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the S3-backed Zephyr stages of the Datakit reference pipeline.

The benchmark starts from an existing normalized Datakit sample, so its measured
pipeline does not include Hugging Face corpus download time. It runs global exact
deduplication, per-source tokenization and MinHash, then cross-source fuzzy dedup.
Every generated stage output is routed under a seven-day temporary prefix keyed by
the required run tag.

Example::

    python -m experiments.datakit.zephyr_benchmark \
        --sample-prefix s3://marin-us-east-02a/marin/datakit/sample_100b_8ae7a94f \
        --sources all --run-tag zephyr-100b-v1 \
        --pool-workers 60 --pool-cpu 16 --pool-ram 160g --pool-disk 32g \
        --first-stage exact --last-stage fuzzy \
        --max-concurrent 4 --dedup-max-parallelism 4096
"""

import argparse
import logging
from dataclasses import replace
from enum import StrEnum

from fray.types import ResourceConfig
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
from rigging.filesystem.cluster_config import marin_temp_bucket
from rigging.filesystem.storage_path import prefix_join
from rigging.log_setup import configure_logging

from experiments.datakit.reference_pipeline import (
    SMOKE_SCALE,
    PoolConfig,
    ZephyrDatakitSteps,
    sample_sources,
    zephyr_datakit_steps,
)

BENCHMARK_OUTPUT_TTL_DAYS = 7
BENCHMARK_OUTPUT_PREFIX = "zephyr-benchmark"


class BenchmarkStage(StrEnum):
    """Stage boundary for a benchmark run."""

    EXACT = "exact"
    TOKENIZE = "tokenize"
    MINHASH = "minhash"
    FUZZY = "fuzzy"


_STAGE_ORDER = tuple(BenchmarkStage)


def _steps_between(
    steps: ZephyrDatakitSteps,
    first_stage: BenchmarkStage,
    last_stage: BenchmarkStage,
) -> list[StepSpec]:
    first_index = _STAGE_ORDER.index(first_stage)
    last_index = _STAGE_ORDER.index(last_stage)
    if first_index > last_index:
        raise ValueError(f"first stage {first_stage} follows last stage {last_stage}")

    stage_steps = {
        BenchmarkStage.EXACT: [steps.exact_dedup],
        BenchmarkStage.TOKENIZE: list(steps.tokenize.values()),
        BenchmarkStage.MINHASH: list(steps.minhash.values()),
        BenchmarkStage.FUZZY: [steps.fuzzy_dedup],
    }
    return [step for stage in _STAGE_ORDER[first_index : last_index + 1] for step in stage_steps[stage]]


def _route_outputs(steps: ZephyrDatakitSteps, output_prefix: str) -> ZephyrDatakitSteps:
    tokenize = {name: replace(step, output_path_prefix=output_prefix) for name, step in steps.tokenize.items()}
    minhash = {name: replace(step, output_path_prefix=output_prefix) for name, step in steps.minhash.items()}
    return ZephyrDatakitSteps(
        exact_dedup=replace(steps.exact_dedup, output_path_prefix=output_prefix),
        tokenize=tokenize,
        minhash=minhash,
        fuzzy_dedup=replace(steps.fuzzy_dedup, output_path_prefix=output_prefix, deps=list(minhash.values())),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-prefix", required=True)
    parser.add_argument("--sources", required=True, help="Comma-separated source names or 'all'.")
    parser.add_argument("--run-tag", required=True, help="Fresh identity tag that forces uncached benchmark stages.")
    parser.add_argument("--pool-workers", required=True, type=int)
    parser.add_argument("--pool-cpu", required=True, type=float)
    parser.add_argument("--pool-ram", required=True)
    parser.add_argument("--pool-disk", required=True)
    parser.add_argument("--first-stage", required=True, type=BenchmarkStage, choices=list(BenchmarkStage))
    parser.add_argument("--last-stage", required=True, type=BenchmarkStage, choices=list(BenchmarkStage))
    parser.add_argument("--max-concurrent", required=True, type=int)
    parser.add_argument("--dedup-max-parallelism", required=True, type=int)
    args = parser.parse_args()

    configure_logging(logging.INFO)
    selected_sources = None if args.sources == "all" else [name.strip() for name in args.sources.split(",")]
    sources = sample_sources(args.sample_prefix, selected_sources, args.run_tag)
    worker = ResourceConfig(cpu=args.pool_cpu, ram=args.pool_ram, disk=args.pool_disk)
    scale = replace(
        SMOKE_SCALE,
        pool=PoolConfig(n_workers=args.pool_workers, worker=worker),
        dedup_max_parallelism=args.dedup_max_parallelism,
    )
    output_prefix = marin_temp_bucket(
        ttl_days=BENCHMARK_OUTPUT_TTL_DAYS,
        prefix=prefix_join(prefix_join(BENCHMARK_OUTPUT_PREFIX, args.run_tag), "outputs"),
        source_prefix=args.sample_prefix,
    )
    steps = _route_outputs(zephyr_datakit_steps(sources, scale), output_prefix)
    StepRunner().run(
        _steps_between(steps, args.first_stage, args.last_stage),
        max_concurrent=args.max_concurrent,
    )


if __name__ == "__main__":
    main()
