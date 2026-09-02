# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Create a Zephyr benchmark sample and its reusable MinHash inputs.

``--mode copy`` copies normalized Parquet shards from any existing sample to a
new sample root, rewriting ``NormalizedData`` artifacts for that destination.
``--mode regenerate`` downloads the registered source data, normalizes it, and
samples the requested token count (100B by default) into a new sample root.
Both modes then compute the permanent MinHash artifacts used by shuffle-only
benchmarks. ``--mode minhash`` backfills those artifacts into an existing
sample without copying or regenerating normalized data. See
``experiments/datakit/README.md`` for the required region-local Iris commands
and cost caveats.
"""

import argparse
import logging
from dataclasses import dataclass, replace
from enum import StrEnum

from fray.types import ResourceConfig
from marin.datakit.normalize import NormalizedData
from marin.datakit.sources import all_sources
from marin.execution.artifact import read_artifact
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
from rigging.filesystem.cluster_config import data_config, use_data_config
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath, prefix_join
from rigging.log_setup import configure_logging
from zephyr.context import ZephyrContext

from experiments.datakit.reference_pipeline import (
    SMOKE_SCALE,
    PipelineScale,
    ZephyrDatakitSteps,
    datakit_zephyr_context,
    fuzzy_dedup_step,
    sample_sources,
    zephyr_datakit_steps,
)
from experiments.datakit.testbed.sampler import proportional_sample_fractions, sample_normalized_shards

DEFAULT_MAX_CONCURRENT = 4
MATERIALIZE_STEP_PREFIX = "datakit/benchmark_sample"
DEFAULT_TARGET_TOTAL_TOKENS_B = 100.0
BENCHMARK_SAMPLE_INPUTS_DIR = "_benchmark_inputs"
GCP_BENCHMARK_SAMPLE_PREFIX = "gs://marin-eu-west4/datakit/sample_100b_8ae7a94f"
COREWEAVE_BENCHMARK_SAMPLE_PREFIX = "s3://marin-us-east-02a/marin/datakit/sample_100b_8ae7a94f"

logger = logging.getLogger(__name__)


class SampleMode(StrEnum):
    """Work performed by the benchmark sample materializer."""

    COPY = "copy"
    REGENERATE = "regenerate"
    MINHASH = "minhash"


@dataclass(frozen=True)
class BenchmarkSampleFuzzySteps:
    """Sample-owned MinHash steps and the fuzzy step that consumes them."""

    minhash: dict[str, StepSpec]
    fuzzy_dedup: StepSpec


def benchmark_sample_inputs_prefix(sample_prefix: str) -> str:
    return prefix_join(sample_prefix, BENCHMARK_SAMPLE_INPUTS_DIR)


def benchmark_datakit_steps(
    sources: dict[str, StepSpec],
    scale: PipelineScale,
    zephyr_context: ZephyrContext,
    output_path_prefix: str,
) -> ZephyrDatakitSteps:
    """Build Datakit stages with all generated outputs under one benchmark prefix."""
    steps = zephyr_datakit_steps(sources, scale, zephyr_context)
    tokenize = {name: replace(step, output_path_prefix=output_path_prefix) for name, step in steps.tokenize.items()}
    minhash = {name: replace(step, output_path_prefix=output_path_prefix) for name, step in steps.minhash.items()}
    fuzzy_dedup = fuzzy_dedup_step(list(minhash.values()), scale, zephyr_context)
    return ZephyrDatakitSteps(
        exact_dedup=replace(steps.exact_dedup, output_path_prefix=output_path_prefix),
        tokenize=tokenize,
        minhash=minhash,
        fuzzy_dedup=replace(fuzzy_dedup, output_path_prefix=output_path_prefix),
    )


def benchmark_sample_fuzzy_steps(
    sample_prefix: str,
    sources: dict[str, StepSpec],
    scale: PipelineScale,
    zephyr_context: ZephyrContext,
) -> BenchmarkSampleFuzzySteps:
    """Build canonical MinHash inputs and their fuzzy-dedup consumer."""
    steps = benchmark_datakit_steps(
        sources,
        scale,
        zephyr_context,
        output_path_prefix=benchmark_sample_inputs_prefix(sample_prefix),
    )
    return BenchmarkSampleFuzzySteps(minhash=steps.minhash, fuzzy_dedup=steps.fuzzy_dedup)


def _sample_main_output_step(
    name: str,
    source: StepSpec,
    destination_prefix: str,
    sample_fraction: float,
) -> StepSpec:
    """Write a normalized source's sampled main output under ``destination_prefix``."""
    source_path = source.output_path

    def sample(output_path: str) -> NormalizedData:
        source_data = read_artifact(source_path, NormalizedData).model_copy(
            update={"main_output_dir": prefix_join(source_path, "outputs/main")}
        )
        sampled = sample_normalized_shards(
            source=source_data,
            output_path=output_path,
            sample_fraction=sample_fraction,
        )
        return sampled.model_copy(update={"dup_output_dir": ""})

    return StepSpec(
        name=f"{MATERIALIZE_STEP_PREFIX}/{name}",
        deps=[source],
        hash_attrs={"source_path": source_path, "sample_fraction": sample_fraction, "version": "v2"},
        fn=sample,
        override_output_path=prefix_join(destination_prefix, name),
    )


def copy_sample_steps(source_prefix: str, destination_prefix: str) -> list[StepSpec]:
    """Build one copy step for every source in an existing benchmark sample."""
    sources = sample_sources(source_prefix)
    if not sources:
        raise ValueError(f"no normalized source artifacts found under {source_prefix}")
    return [
        _sample_main_output_step(name, source, destination_prefix, sample_fraction=1.0)
        for name, source in sorted(sources.items())
    ]


def regenerate_sample_steps(
    source_prefix: str,
    destination_prefix: str,
    target_total_tokens_b: float,
) -> list[StepSpec]:
    """Build the source download, normalization, and sampling steps for a fresh benchmark sample."""
    source_names = set(sample_sources(source_prefix))
    if not source_names:
        raise ValueError(f"no normalized source artifacts found under {source_prefix}")
    registry = all_sources()
    missing = sorted(source_names - set(registry))
    if missing:
        raise ValueError(f"source registry no longer defines {missing}")
    sources = {name: registry[name] for name in source_names}
    fractions = proportional_sample_fractions(tuple(sources.values()), target_total_tokens_b)
    return [
        _sample_main_output_step(name, source.normalized, destination_prefix, fractions[name])
        for name, source in sorted(sources.items())
    ]


def _validate_data_prefix(data_prefix: str, destination_prefix: str) -> None:
    data_root = StoragePath(data_prefix)
    destination = StoragePath(destination_prefix)
    if data_root.scheme not in ("gs", "s3") or not data_root.bucket:
        raise ValueError(f"data prefix must be an object-store root: {data_prefix}")
    if destination.scheme != data_root.scheme or destination.bucket != data_root.bucket:
        raise ValueError(f"destination {destination_prefix} must be under data prefix {data_prefix}")
    destination.relative_to(data_root)


def _verify_source_set(expected_names: set[str], destination_prefix: str) -> None:
    destination_names = set(sample_sources(destination_prefix))
    if destination_names != expected_names:
        raise RuntimeError(
            f"materialized source set differs: missing={sorted(expected_names - destination_names)}, "
            f"unexpected={sorted(destination_names - expected_names)}"
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=list(SampleMode), type=SampleMode, default=SampleMode.COPY)
    parser.add_argument("--source-prefix", default=COREWEAVE_BENCHMARK_SAMPLE_PREFIX)
    parser.add_argument("--destination-prefix", default=GCP_BENCHMARK_SAMPLE_PREFIX)
    parser.add_argument(
        "--data-prefix",
        help="Object-store root for source downloads and normalized artifacts (--mode regenerate only).",
    )
    parser.add_argument("--target-total-tokens-b", type=float, default=DEFAULT_TARGET_TOTAL_TOKENS_B)
    parser.add_argument("--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT)
    parser.add_argument("--minhash-max-concurrent", required=True, type=int)
    parser.add_argument("--pool-workers", required=True, type=int)
    parser.add_argument("--pool-cpu", required=True, type=float)
    parser.add_argument("--pool-ram", required=True)
    parser.add_argument("--pool-disk", required=True)
    parser.add_argument("--map-task-cpu", required=True, type=float)
    parser.add_argument("--map-task-ram", required=True)
    parser.add_argument("--map-task-disk", required=True)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.max_concurrent < 1:
        raise ValueError(f"max concurrent must be positive: {args.max_concurrent}")
    if args.minhash_max_concurrent < 1:
        raise ValueError(f"minhash max concurrent must be positive: {args.minhash_max_concurrent}")
    if args.pool_workers < 1:
        raise ValueError(f"pool workers must be positive: {args.pool_workers}")
    if args.mode is SampleMode.REGENERATE and args.target_total_tokens_b <= 0:
        raise ValueError(f"target total tokens must be positive: {args.target_total_tokens_b}")


def _materialize_sources(args: argparse.Namespace) -> set[str]:
    if args.destination_prefix.startswith("s3://") or (
        args.mode is not SampleMode.MINHASH and args.source_prefix.startswith("s3://")
    ):
        configure_coreweave_s3()

    if args.mode is SampleMode.COPY:
        if args.data_prefix is not None:
            raise ValueError("--data-prefix applies only to --mode regenerate")
        if StoragePath(args.source_prefix) == StoragePath(args.destination_prefix):
            raise ValueError("source and destination prefixes must differ")
        steps = copy_sample_steps(args.source_prefix, args.destination_prefix)
    elif args.mode is SampleMode.REGENERATE:
        if args.data_prefix is None:
            raise ValueError("--data-prefix is required for --mode regenerate")
        _validate_data_prefix(args.data_prefix, args.destination_prefix)
        with use_data_config(replace(data_config(), root=args.data_prefix)):
            steps = regenerate_sample_steps(args.source_prefix, args.destination_prefix, args.target_total_tokens_b)
            StepRunner().run(steps, max_concurrent=args.max_concurrent)
    else:
        if args.data_prefix is not None:
            raise ValueError("--data-prefix applies only to --mode regenerate")
        steps = []
    if args.mode is SampleMode.COPY:
        StepRunner().run(steps, max_concurrent=args.max_concurrent)

    if steps:
        source_names = {step.name.removeprefix(f"{MATERIALIZE_STEP_PREFIX}/") for step in steps}
        _verify_source_set(source_names, args.destination_prefix)
        return source_names

    source_names = set(sample_sources(args.destination_prefix))
    if not source_names:
        raise ValueError(f"no normalized source artifacts found under {args.destination_prefix}")
    return source_names


def _materialize_minhash(args: argparse.Namespace, source_names: set[str]) -> None:
    worker = ResourceConfig(cpu=args.pool_cpu, ram=args.pool_ram, disk=args.pool_disk)
    map_task = ResourceConfig(cpu=args.map_task_cpu, ram=args.map_task_ram, disk=args.map_task_disk)
    scale = replace(
        SMOKE_SCALE,
        pool=replace(
            SMOKE_SCALE.pool,
            n_workers=args.pool_workers,
            worker=worker,
            map_task=map_task,
        ),
    )
    zephyr_context = datakit_zephyr_context(
        "zephyr-benchmark-sample-minhash",
        scale,
        args.minhash_max_concurrent,
    )
    sources = sample_sources(args.destination_prefix, sorted(source_names))
    minhash_steps = benchmark_sample_fuzzy_steps(
        args.destination_prefix,
        sources,
        scale,
        zephyr_context,
    ).minhash
    with zephyr_context:
        StepRunner().run(list(minhash_steps.values()), max_concurrent=args.minhash_max_concurrent)
    logger.info(
        "Created %d normalized benchmark sources and MinHash artifacts at %s with %s",
        len(source_names),
        args.destination_prefix,
        args.mode,
    )


def main() -> None:
    args = _parse_args()
    configure_logging(logging.INFO)
    _validate_args(args)
    source_names = _materialize_sources(args)
    _materialize_minhash(args, source_names)


if __name__ == "__main__":
    main()
